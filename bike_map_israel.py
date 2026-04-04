"""
All-Israel interactive transport mode map – 2008 and 2022 census
Based on Israel CBS Census public-use files

Geometry: 2022 statistical areas (ezorim_statistiim_2022 GDB) used for both years.
          ~85% of 2008 large-city sub-areas match 2022 boundaries; the rest show grey.

Transport modes are aligned across years by name; code numbers differ between censuses.
2008 codes 98 (children 0-14) and 99 (unknown) are excluded as non-commuters.
"""

import json
import os
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import warnings
warnings.filterwarnings("ignore")

# ── Shared geographic column names (same in both census files) ────────────────
SETT_COL = "smlyishuvpuf"
ROVA_COL = "rovaktvtmegurimpuf"
AREA_COL = "tatrovaktvtmegurimpuf"

# ── Per-year census config ────────────────────────────────────────────────────
YEAR_CFG = {
    2008: {
        "file":          "census/census2008bike.csv",
        "transport_col": "emtzaehagaaikarimakomavdmchvpuf",
        "weight_col":    "mishkalpuf",
        "exclude_codes": {98, 99},   # children / unknown
    },
    2022: {
        "file":          "census/census2022bike.csv",
        "transport_col": "emtzaehagaaikarimakomavdpuf",
        "weight_col":    "mishkalpratpuf",
        "exclude_codes": set(),
    },
}

# ── Aligned transport modes (codes differ by year) ────────────────────────────
# slug: short key used in GeoJSON properties  (e.g. "bicycle" → p08_bicycle)
# 2008/2022: transport code for that year, or None if mode didn't exist
MODES = [
    {"slug": "car_driver",  "label": "Private car – driver",     2008: 1,    2022: 1 },
    {"slug": "car_pass",    "label": "Private car – passenger",   2008: 2,    2022: 2 },
    {"slug": "bus",         "label": "Public bus",                2008: 3,    2022: 3 },
    {"slug": "light_rail",  "label": "Light rail / metro",        2008: None, 2022: 4 },
    {"slug": "employer",    "label": "Employer transport",         2008: 4,    2022: 5 },
    {"slug": "train",       "label": "Israel Railways",            2008: 5,    2022: 6 },
    {"slug": "taxi_svc",    "label": "Service taxi",               2008: 6,    2022: 7 },
    {"slug": "taxi_sp",     "label": "Special taxi",               2008: 7,    2022: 8 },
    {"slug": "moto",        "label": "Motorcycle / moped",         2008: 8,    2022: 9 },
    {"slug": "bicycle",     "label": "Bicycle",                   2008: 9,    2022: 10},
    {"slug": "walking",     "label": "Walking",                   2008: 10,   2022: 11},
    {"slug": "truck",       "label": "Truck",                     2008: 11,   2022: 12},
    {"slug": "other",       "label": "Other vehicle",             2008: 12,   2022: 13},
]
SLUGS = [m["slug"] for m in MODES]


# ── 1. Compute mode shares for one year ───────────────────────────────────────
def load_year_stats(year):
    cfg = YEAR_CFG[year]
    print(f"\n[{year}] Loading {cfg['file']} …")
    df = pd.read_csv(cfg["file"])
    df.columns = df.columns.str.lower().str.strip()

    tc = cfg["transport_col"]
    wc = cfg["weight_col"]

    # Commuters: non-null transport, excluding special non-commuter codes
    mask = df[tc].notna() & ~df[tc].isin(cfg["exclude_codes"])
    workers = df[mask].copy()
    print(f"  {len(workers):,} commuters")

    W = workers[wc]
    # Weighted mode dummies per slug
    for m in MODES:
        code = m[year]
        workers[f"wm_{m['slug']}"] = ((workers[tc] == code) * W) if code else 0.0
    workers["w_tot"] = W

    wm_cols  = [f"wm_{s}" for s in SLUGS]
    sum_cols = wm_cols + ["w_tot"]

    def compute(grp_df, grp_cols):
        agg = grp_df.groupby(grp_cols)[sum_cols].sum().reset_index()
        for s in SLUGS:
            agg[f"pct_{s}"] = (agg[f"wm_{s}"] / agg["w_tot"] * 100).round(3)
        return agg

    w_large  = workers[workers[ROVA_COL].notna()]
    w_medium = workers[workers[ROVA_COL].isna() & workers[AREA_COL].notna()]
    w_small  = workers[workers[AREA_COL].isna()]

    stats = {
        "large":  compute(w_large,  [SETT_COL, ROVA_COL, AREA_COL]),
        "medium": compute(w_medium, [SETT_COL, AREA_COL]),
        "small":  compute(w_small,  [SETT_COL]),
    }
    print(f"  Groups: large={len(stats['large'])}, "
          f"medium={len(stats['medium'])}, small={len(stats['small'])}")
    return stats


# ── 2. Load & dissolve shapefile (done once, used for both years) ─────────────
def load_shapes():
    gdb = "ezorim_statistiim_2022.gdb"
    if not os.path.exists(gdb):
        os.symlink(str(pathlib.Path("ezorim_statistiim_2022").resolve()), gdb)
    print("\nLoading shapefile …")
    gdf = gpd.read_file(gdb)

    shp_large  = gdf[gdf["ROVA"].notna()].copy()
    shp_medium = gdf[gdf["ROVA"].isna() & gdf["TAT_ROVA"].notna()].copy()
    shp_small  = gdf[gdf["TAT_ROVA"].isna()].copy()

    diss_large  = shp_large.dissolve( by=["SEMEL_YISHUV","ROVA","TAT_ROVA"], aggfunc="first").reset_index()
    diss_medium = shp_medium.dissolve(by=["SEMEL_YISHUV","TAT_ROVA"],         aggfunc="first").reset_index()
    diss_small  = shp_small.dissolve( by=["SEMEL_YISHUV"],                    aggfunc="first").reset_index()
    print(f"  Dissolved: large={len(diss_large)}, medium={len(diss_medium)}, small={len(diss_small)}")
    return gdf.crs, diss_large, diss_medium, diss_small


# ── 3. Merge one year's stats into the dissolved shapes ──────────────────────
PCT_COLS = [f"pct_{s}" for s in SLUGS]

def to_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def merge_year(diss_large, diss_medium, diss_small, stats, year):
    sl, sm, ss = (d.copy() for d in (diss_large, diss_medium, diss_small))

    to_float(sl, ["SEMEL_YISHUV","ROVA","TAT_ROVA"])
    to_float(sm, ["SEMEL_YISHUV","TAT_ROVA"])
    to_float(ss, ["SEMEL_YISHUV"])
    to_float(stats["large"],  [SETT_COL, ROVA_COL, AREA_COL])
    to_float(stats["medium"], [SETT_COL, AREA_COL])
    to_float(stats["small"],  [SETT_COL])

    ml = sl.merge(stats["large"] [[SETT_COL,ROVA_COL,AREA_COL,"w_tot"]+PCT_COLS],
                  left_on=["SEMEL_YISHUV","ROVA","TAT_ROVA"],
                  right_on=[SETT_COL,ROVA_COL,AREA_COL], how="left")
    ml["geo_level"] = "sub_neighbourhood"

    mm = sm.merge(stats["medium"][[SETT_COL,AREA_COL,"w_tot"]+PCT_COLS],
                  left_on=["SEMEL_YISHUV","TAT_ROVA"],
                  right_on=[SETT_COL,AREA_COL], how="left")
    mm["geo_level"] = "neighbourhood"

    ms = ss.merge(stats["small"] [[SETT_COL,"w_tot"]+PCT_COLS],
                  left_on=["SEMEL_YISHUV"], right_on=[SETT_COL], how="left")
    ms["geo_level"] = "city"

    keep = ["SEMEL_YISHUV","SHEM_YISHUV","SHEM_YISHUV_ENGLISH","ROVA","TAT_ROVA",
            "geo_level","geometry","w_tot"] + PCT_COLS
    combined = pd.concat(
        [d[[c for c in keep if c in d.columns]] for d in (ml, mm, ms)],
        ignore_index=True
    )
    return combined


# ── 4. Build GeoJSON with both years' data ────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 3)
    except Exception:
        return None

def build_geojson(combined_08, combined_22, crs):
    # Reproject and simplify 2022 (canonical geometry)
    geo22 = gpd.GeoDataFrame(combined_22, crs=crs).to_crs(epsg=4326)
    geo22["geometry"] = geo22["geometry"].simplify(0.0003, preserve_topology=True)

    # Build per-area lookup for 2008 stats (match on SEMEL_YISHUV + ROVA + TAT_ROVA)
    def make_key(row):
        r = row.get("ROVA"); t = row.get("TAT_ROVA")
        return (safe_float(row["SEMEL_YISHUV"]),
                safe_float(r) if pd.notna(r) else None,
                safe_float(t) if pd.notna(t) else None)

    lookup08 = {}
    for _, row in combined_08.iterrows():
        k = make_key(row)
        lookup08[k] = row

    print("Building GeoJSON …")
    features = []
    for _, row in geo22.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue

        city_en = str(row.get("SHEM_YISHUV_ENGLISH") or "").strip()
        city_he = str(row.get("SHEM_YISHUV") or "").strip()
        city    = city_en if city_en else city_he

        level = row.get("geo_level", "city")
        rova  = row.get("ROVA"); tat = row.get("TAT_ROVA")
        if level == "sub_neighbourhood" and pd.notna(rova) and pd.notna(tat):
            area = f"Sub-neighbourhood {int(rova)}/{int(tat)}"
        elif level == "neighbourhood" and pd.notna(tat):
            area = f"Neighbourhood {int(tat)}"
        else:
            area = None

        props = {"city": city, "area": area, "level": level}

        # 2022 stats
        props["n22"] = safe_float(row.get("w_tot"))
        for s in SLUGS:
            props[f"p22_{s}"] = safe_float(row.get(f"pct_{s}"))

        # 2008 stats — look up by composite key
        k = make_key(row)
        row08 = lookup08.get(k)
        props["n08"] = safe_float(row08["w_tot"]) if row08 is not None else None
        for s in SLUGS:
            props[f"p08_{s}"] = safe_float(row08[f"pct_{s}"]) if row08 is not None else None

        features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})

    geojson_str = json.dumps({"type": "FeatureCollection", "features": features},
                             ensure_ascii=False)
    print(f"  {len(features)} features, ~{len(geojson_str)//1024} KB")
    return geojson_str


# ── 5. Compute per-year colour caps ──────────────────────────────────────────
def compute_caps(combined, year_tag):
    caps = {}
    for s in SLUGS:
        vals = combined[f"pct_{s}"].dropna()
        caps[s] = float(round(vals.quantile(0.95), 2)) if len(vals) else 1.0
    return caps


# ── MAIN ──────────────────────────────────────────────────────────────────────
stats_08 = load_year_stats(2008)
stats_22 = load_year_stats(2022)

crs, diss_large, diss_medium, diss_small = load_shapes()

print("\nMerging 2008 …")
combined_08 = merge_year(diss_large, diss_medium, diss_small, stats_08, 2008)
print("Merging 2022 …")
combined_22 = merge_year(diss_large, diss_medium, diss_small, stats_22, 2022)

geojson_str = build_geojson(combined_08, combined_22, crs)

caps_08 = compute_caps(combined_08, "08")
caps_22 = compute_caps(combined_22, "22")

caps_js  = json.dumps({"2008": caps_08, "2022": caps_22})
modes_js = json.dumps([{"slug": m["slug"], "label": m["label"],
                         "has2008": m[2008] is not None} for m in MODES],
                      ensure_ascii=False)


# ── 6. Write HTML ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Transport Mode by Area</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #111827; display: flex; height: 100vh; overflow: hidden; }
#map { flex: 1; }
#panel {
  width: 300px; flex-shrink: 0;
  background: #1f2937; color: #d1d5db;
  display: flex; flex-direction: column;
  border-left: 1px solid #374151; overflow: hidden;
}
#panel-header { padding: 12px 14px 10px; border-bottom: 1px solid #374151; }
#panel-header h1 { font-size: 13px; font-weight: 700; color: #f9fafb; line-height: 1.5; }
#year-toggle { display: flex; gap: 6px; margin-top: 8px; }
.yr-btn {
  flex: 1; padding: 5px 0; border-radius: 4px; border: 1px solid #374151;
  background: #111827; color: #9ca3af; cursor: pointer; font-size: 13px;
  font-weight: 600; transition: all 0.12s;
}
.yr-btn:hover { background: #374151; color: #f3f4f6; }
.yr-btn.active { background: #0369a1; border-color: #0284c7; color: #fff; }
#mode-list { flex: 1; overflow-y: auto; padding: 8px; }
#mode-list::-webkit-scrollbar { width: 5px; }
#mode-list::-webkit-scrollbar-track { background: #111827; }
#mode-list::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
.mode-btn {
  display: block; width: 100%; text-align: left;
  padding: 7px 10px; margin: 2px 0;
  border: 1px solid transparent; border-radius: 4px;
  background: transparent; color: #9ca3af;
  cursor: pointer; font-size: 12px; transition: all 0.12s;
}
.mode-btn:hover:not(:disabled) { background: #374151; color: #f3f4f6; border-color: #4b5563; }
.mode-btn.active { background: #1d4ed8; border-color: #2563eb; color: #fff; font-weight: 600; }
.mode-btn:disabled { color: #374151; cursor: default; }
#panel-footer { padding: 10px 14px; border-top: 1px solid #374151; }
#legend-title { font-size: 10px; color: #6b7280; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
#legend-bar { height: 10px; border-radius: 3px; margin: 4px 0;
  background: linear-gradient(to right, #ffffcc, #a1dab4, #41b6c4, #2c7fb8, #253494); }
.legend-ticks { display: flex; justify-content: space-between; font-size: 10px; color: #6b7280; }
#geo-note { margin-top: 8px; font-size: 10px; color: #4b5563; line-height: 1.5; }
.leaflet-tooltip {
  background: #1f2937 !important; border: 1px solid #374151 !important;
  border-radius: 6px !important; color: #d1d5db !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important; padding: 8px 12px !important;
}
.leaflet-tooltip::before { display: none !important; }
</style>
</head>
<body>
<div id="map"></div>
<div id="panel">
  <div id="panel-header">
    <h1>Transport Mode by Area<br>
      <span style="font-weight:400;color:#6b7280">Israel Census</span></h1>
    <div id="year-toggle">
      <button class="yr-btn" data-year="2008">2008</button>
      <button class="yr-btn active" data-year="2022">2022</button>
    </div>
  </div>
  <div id="mode-list"></div>
  <div id="panel-footer">
    <div id="legend-title">Share of commuters</div>
    <div id="legend-bar"></div>
    <div class="legend-ticks"><span>0%</span><span id="leg-cap">–</span></div>
    <div id="geo-note">
      Large cities: sub-neighbourhood level<br>
      Medium cities: neighbourhood level<br>
      Small settlements: city level<br>
      Grey = no data<br>
      <span style="color:#374151">2022 boundaries used for both years</span>
    </div>
  </div>
</div>

<script>
var CAPS  = __CAPS__;
var MODES = __MODES__;
var DATA  = __DATA__;

var currentYear = "2022";
var currentSlug = "bicycle";

var map = L.map('map', { preferCanvas: true }).setView([31.5, 34.9], 8);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com">CARTO</a>',
  subdomains: 'abcd', maxZoom: 19
}).addTo(map);

var RAMP = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'];
function hexToRgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
function getColor(pct, cap) {
  if (pct == null) return '#6b7280';
  var t = Math.min(pct / (cap || 1), 1);
  var n = RAMP.length - 1;
  var i = Math.min(Math.floor(t * n), n - 1);
  var f = t * n - i;
  var a = hexToRgb(RAMP[i]), b = hexToRgb(RAMP[i+1]);
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','+Math.round(a[2]+(b[2]-a[2])*f)+')';
}

function propKey() { return 'p' + currentYear.slice(2) + '_' + currentSlug; }
function nKey()    { return 'n' + currentYear.slice(2); }

function getCap() {
  var c = (CAPS[currentYear] || {})[currentSlug];
  return (c != null && c > 0) ? c : 1;
}

function styleFn(feature) {
  var p   = feature.properties[propKey()];
  var cap = getCap();
  return {
    fillColor:   getColor(p, cap),
    weight:      0.4, color: '#555', opacity: 0.8,
    fillOpacity: (p == null) ? 0.2 : 0.72,
  };
}

var geojsonLayer = L.geoJSON(DATA, {
  style: styleFn,
  onEachFeature: function(feature, layer) {
    var p = feature.properties;
    layer.on({
      mouseover: function(e) {
        e.target.setStyle({ weight: 2, color: '#fff', fillOpacity: 0.92 });
        var pct    = p[propKey()];
        var pctStr = (pct != null) ? pct.toFixed(2) + '%' : 'No data';
        var n      = p[nKey()];
        var nStr   = (n != null) ? Math.round(n).toLocaleString() : '—';
        var mLabel = MODES.find(function(m){ return m.slug === currentSlug; }).label;
        layer.bindTooltip(
          '<div style="min-width:180px;line-height:1.7">' +
          '<b style="font-size:13px">' + (p.city || '—') + '</b>' +
          (p.area ? '<br><span style="font-size:11px;color:#9ca3af">' + p.area + '</span>' : '') +
          '<hr style="border:none;border-top:1px solid #374151;margin:5px 0">' +
          mLabel + ' (' + currentYear + '):<br>' +
          '<b style="font-size:14px">' + pctStr + '</b><br>' +
          '<span style="font-size:10px;color:#9ca3af">Weighted commuters: ' + nStr + '</span>' +
          '</div>',
          { sticky: true }
        ).openTooltip();
      },
      mouseout: function(e) {
        geojsonLayer.resetStyle(e.target);
        e.target.closeTooltip();
      },
    });
  },
}).addTo(map);

function updateMap() {
  geojsonLayer.setStyle(styleFn);
  var cap = getCap();
  document.getElementById('leg-cap').textContent = cap.toFixed(1) + '%';
  document.querySelectorAll('.mode-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.slug === currentSlug);
  });
  document.querySelectorAll('.yr-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.year === currentYear);
  });
  // Disable modes not available in 2008
  document.querySelectorAll('.mode-btn').forEach(function(b) {
    var mode = MODES.find(function(m){ return m.slug === b.dataset.slug; });
    b.disabled = (currentYear === '2008' && !mode.has2008);
    if (b.disabled && b.dataset.slug === currentSlug) {
      currentSlug = 'bicycle';  // fallback
      updateMap();
    }
  });
}

// Year buttons
document.querySelectorAll('.yr-btn').forEach(function(b) {
  b.onclick = function() { currentYear = b.dataset.year; updateMap(); };
});

// Mode buttons
var list = document.getElementById('mode-list');
MODES.forEach(function(m, i) {
  var btn = document.createElement('button');
  btn.className = 'mode-btn' + (m.slug === currentSlug ? ' active' : '');
  btn.dataset.slug = m.slug;
  btn.textContent = (i + 1) + '. ' + m.label;
  btn.onclick = function() {
    if (!btn.disabled) { currentSlug = m.slug; updateMap(); }
  };
  list.appendChild(btn);
});

document.getElementById('leg-cap').textContent = getCap().toFixed(1) + '%';
</script>
</body>
</html>
"""

html_out = (HTML
    .replace("__CAPS__",  caps_js)
    .replace("__MODES__", modes_js)
    .replace("__DATA__",  geojson_str))

out_path = "transport_mode_israel.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_out)

size_mb = len(html_out) / 1_048_576
print(f"\nSaved → {out_path}  ({size_mb:.1f} MB)")
print("Done ✓")
