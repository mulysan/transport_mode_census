"""
All-Israel interactive transport mode map – multi-year census
Supports single-year view and change-between-years view.
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
        "exclude_codes": {98, 99},
    },
    2022: {
        "file":          "census/census2022bike.csv",
        "transport_col": "emtzaehagaaikarimakomavdpuf",
        "weight_col":    "mishkalpratpuf",
        "exclude_codes": set(),
    },
}
YEARS = sorted(YEAR_CFG.keys())

# ── Aligned transport modes ───────────────────────────────────────────────────
MODES = [
    {"slug": "car_driver",  "label": "Private car - driver",     2008: 1,    2022: 1 },
    {"slug": "car_pass",    "label": "Private car - passenger",  2008: 2,    2022: 2 },
    {"slug": "bus",         "label": "Public bus",               2008: 3,    2022: 3 },
    {"slug": "light_rail",  "label": "Light rail / metro",       2008: None, 2022: 4 },
    {"slug": "employer",    "label": "Employer transport",        2008: 4,    2022: 5 },
    {"slug": "train",       "label": "Israel Railways",           2008: 5,    2022: 6 },
    {"slug": "taxi_svc",    "label": "Service taxi",              2008: 6,    2022: 7 },
    {"slug": "taxi_sp",     "label": "Special taxi",              2008: 7,    2022: 8 },
    {"slug": "moto",        "label": "Motorcycle / moped",        2008: 8,    2022: 9 },
    {"slug": "bicycle",     "label": "Bicycle",                  2008: 9,    2022: 10},
    {"slug": "walking",     "label": "Walking",                  2008: 10,   2022: 11},
    {"slug": "truck",       "label": "Truck",                    2008: 11,   2022: 12},
    {"slug": "other",       "label": "Other vehicle",            2008: 12,   2022: 13},
]
SLUGS = [m["slug"] for m in MODES]


# ── 1. Compute mode shares for one year ───────────────────────────────────────
def load_year_stats(year):
    cfg = YEAR_CFG[year]
    print(f"\n[{year}] Loading {cfg['file']} ...")
    df = pd.read_csv(cfg["file"])
    df.columns = df.columns.str.lower().str.strip()

    tc = cfg["transport_col"]
    wc = cfg["weight_col"]

    mask = df[tc].notna() & ~df[tc].isin(cfg["exclude_codes"])
    workers = df[mask].copy()
    print(f"  {len(workers):,} commuters")

    W = workers[wc]
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


# ── 2. Load & dissolve 2022 shapefile (geometry used for all views) ───────────
def load_shapes():
    gdb = "ezorim_statistiim_2022.gdb"
    if not os.path.exists(gdb):
        os.symlink(str(pathlib.Path("ezorim_statistiim_2022").resolve()), gdb)
    print("\nLoading shapefile ...")
    gdf = gpd.read_file(gdb)

    shp_large  = gdf[gdf["ROVA"].notna()].copy()
    shp_medium = gdf[gdf["ROVA"].isna() & gdf["TAT_ROVA"].notna()].copy()
    shp_small  = gdf[gdf["TAT_ROVA"].isna()].copy()

    diss_large  = shp_large.dissolve( by=["SEMEL_YISHUV","ROVA","TAT_ROVA"], aggfunc="first").reset_index()
    diss_medium = shp_medium.dissolve(by=["SEMEL_YISHUV","TAT_ROVA"],        aggfunc="first").reset_index()
    diss_small  = shp_small.dissolve( by=["SEMEL_YISHUV"],                   aggfunc="first").reset_index()
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


# ── 4. Build GeoJSON features (all years embedded per feature) ────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 3)
    except Exception:
        return None

def build_features(combined_by_year, crs):
    # Use the last year's combined as the canonical geometry base
    canonical_year = YEARS[-1]
    geo_base = gpd.GeoDataFrame(combined_by_year[canonical_year], crs=crs).to_crs(epsg=4326)
    geo_base["geometry"] = geo_base["geometry"].simplify(0.0003, preserve_topology=True)

    # Build lookups for each non-canonical year
    def make_key(row):
        r = row.get("ROVA"); t = row.get("TAT_ROVA")
        return (safe_float(row["SEMEL_YISHUV"]),
                safe_float(r) if pd.notna(r) else None,
                safe_float(t) if pd.notna(t) else None)

    lookups = {}
    for yr in YEARS:
        if yr == canonical_year:
            continue
        lk = {}
        for _, row in combined_by_year[yr].iterrows():
            lk[make_key(row)] = row
        lookups[yr] = lk

    print("Building GeoJSON ...")
    features = []
    for _, row in geo_base.iterrows():
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

        # Canonical year stats
        yy = str(canonical_year)[2:]
        props[f"n{yy}"] = safe_float(row.get("w_tot"))
        for s in SLUGS:
            props[f"p{yy}_{s}"] = safe_float(row.get(f"pct_{s}"))

        # Other years via lookup
        k = make_key(row)
        for yr in YEARS:
            if yr == canonical_year:
                continue
            yy2 = str(yr)[2:]
            row_yr = lookups[yr].get(k)
            props[f"n{yy2}"] = safe_float(row_yr["w_tot"]) if row_yr is not None else None
            for s in SLUGS:
                props[f"p{yy2}_{s}"] = safe_float(row_yr[f"pct_{s}"]) if row_yr is not None else None

        features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})

    print(f"  {len(features)} features")
    return features


# ── 5. Compute colour caps ────────────────────────────────────────────────────
def compute_caps(combined_by_year):
    single_caps = {}
    for yr in YEARS:
        combined = combined_by_year[yr]
        caps = {}
        for s in SLUGS:
            vals = combined[f"pct_{s}"].dropna()
            caps[s] = float(round(vals.quantile(0.95), 2)) if len(vals) else 1.0
        single_caps[str(yr)] = caps

    # Change caps: 95th percentile of |diff| for each ordered pair (min_yr, max_yr)
    change_caps = {}
    for i, y1 in enumerate(YEARS):
        for y2 in YEARS[i+1:]:
            c1 = combined_by_year[y1].copy()
            c2 = combined_by_year[y2].copy()
            # Merge on geo keys
            key_cols = ["SEMEL_YISHUV","ROVA","TAT_ROVA"]
            for df in (c1, c2):
                for col in key_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            merged = c2.merge(c1, on=key_cols, suffixes=("_new","_old"), how="inner")
            pair_key = f"{y1}_{y2}"
            pair_caps = {}
            for s in SLUGS:
                col_new = f"pct_{s}_new"
                col_old = f"pct_{s}_old"
                if col_new in merged.columns and col_old in merged.columns:
                    diffs = (merged[col_new] - merged[col_old]).dropna().abs()
                    pair_caps[s] = float(round(diffs.quantile(0.95), 2)) if len(diffs) else 1.0
                else:
                    pair_caps[s] = 1.0
            change_caps[pair_key] = pair_caps

    return {"single": single_caps, "change": change_caps}


# ── MAIN ──────────────────────────────────────────────────────────────────────
all_stats = {yr: load_year_stats(yr) for yr in YEARS}

crs, diss_large, diss_medium, diss_small = load_shapes()

combined_by_year = {}
for yr in YEARS:
    print(f"\nMerging {yr} ...")
    combined_by_year[yr] = merge_year(diss_large, diss_medium, diss_small, all_stats[yr], yr)

features   = build_features(combined_by_year, crs)
caps_all   = compute_caps(combined_by_year)

geojson_str = json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False)
print(f"  GeoJSON ~{len(geojson_str)//1024} KB")

caps_js  = json.dumps(caps_all)
years_js = json.dumps(YEARS)
modes_js = json.dumps([
    {"slug": m["slug"], "label": m["label"],
     "years": [yr for yr in YEARS if m[yr] is not None]}
    for m in MODES
], ensure_ascii=False)



# ── 6. HTML template ──────────────────────────────────────────────────────────
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
#view-toggle { display: flex; gap: 4px; margin-top: 8px; flex-wrap: wrap; }
.view-btn {
  flex: 1; padding: 5px 6px; border-radius: 4px; border: 1px solid #374151;
  background: #111827; color: #9ca3af; cursor: pointer; font-size: 12px;
  font-weight: 600; transition: all 0.12s; white-space: nowrap;
}
.view-btn:hover { background: #374151; color: #f3f4f6; }
.view-btn.active { background: #0369a1; border-color: #0284c7; color: #fff; }
.view-btn.active-change { background: #7c3aed; border-color: #8b5cf6; color: #fff; }
#change-selectors {
  display: none; margin-top: 8px; padding: 8px 10px;
  background: #111827; border-radius: 6px; border: 1px solid #374151;
}
#change-selectors.visible { display: block; }
#change-selectors label { font-size: 11px; color: #6b7280; display: block; margin-bottom: 2px; }
#change-selectors select {
  width: 100%; padding: 4px 6px; border-radius: 4px;
  border: 1px solid #374151; background: #1f2937; color: #d1d5db;
  font-size: 12px; margin-bottom: 6px; cursor: pointer;
}
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
#legend-bar { height: 10px; border-radius: 3px; margin: 4px 0; }
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
    <div id="view-toggle"></div>
    <div id="change-selectors">
      <label>From year</label>
      <select id="sel-from"></select>
      <label>To year</label>
      <select id="sel-to"></select>
    </div>
  </div>
  <div id="mode-list"></div>
  <div id="panel-footer">
    <div id="legend-title">Share of commuters</div>
    <div id="legend-bar"></div>
    <div class="legend-ticks">
      <span id="leg-low">0%</span>
      <span id="leg-mid" style="display:none">0</span>
      <span id="leg-high">-</span>
    </div>
    <div id="geo-note">
      Large cities: sub-neighbourhood level<br>
      Medium cities: neighbourhood level<br>
      Small settlements: city level<br>
      Grey = no data
    </div>
  </div>
</div>

<script>
var CAPS  = __CAPS__;
var MODES = __MODES__;
var YEARS = __YEARS__;
var DATA  = __DATA__;

// ── State ─────────────────────────────────────────────────────────────────────
var viewMode    = "single";          // "single" | "change"
var currentYear = String(YEARS[YEARS.length - 1]);
var fromYear    = String(YEARS[0]);
var toYear      = String(YEARS[YEARS.length - 1]);
var currentSlug = "bicycle";

// ── Map ───────────────────────────────────────────────────────────────────────
var map = L.map('map', { preferCanvas: true }).setView([31.5, 34.9], 8);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com">CARTO</a>',
  subdomains: 'abcd', maxZoom: 19
}).addTo(map);

// ── Colour functions ──────────────────────────────────────────────────────────
function hexToRgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
function lerp(a, b, f) {
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','+Math.round(a[2]+(b[2]-a[2])*f)+')';
}
function rampColor(ramp, t) {
  var n = ramp.length - 1;
  var i = Math.min(Math.floor(t * n), n - 1);
  return lerp(hexToRgb(ramp[i]), hexToRgb(ramp[i+1]), t * n - i);
}

var RAMP_SEQ = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'];
// RdBu diverging: index 0 = strong red (big decrease), 4 = neutral, 8 = strong blue (big increase)
var RAMP_DIV = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4'];

function getColorSingle(pct, cap) {
  if (pct == null) return '#6b7280';
  return rampColor(RAMP_SEQ, Math.min(pct / (cap || 1), 1));
}
function getColorChange(diff, cap) {
  if (diff == null) return '#6b7280';
  // Map [-cap, +cap] → [0, 1]; negative=red(low), positive=blue(high)
  var t = (Math.max(-cap, Math.min(cap, diff)) / (cap || 1) + 1) / 2;
  return rampColor(RAMP_DIV, t);
}

// ── Value/cap helpers ─────────────────────────────────────────────────────────
function getValue(props) {
  if (viewMode === "single") {
    return props["p" + currentYear.slice(-2) + "_" + currentSlug];
  }
  var pTo   = props["p" + toYear.slice(-2)   + "_" + currentSlug];
  var pFrom = props["p" + fromYear.slice(-2) + "_" + currentSlug];
  if (pTo == null || pFrom == null) return null;
  return pTo - pFrom;
}

function getCap() {
  if (viewMode === "single") {
    return ((CAPS.single || {})[currentYear] || {})[currentSlug] || 1;
  }
  var y1 = Math.min(+fromYear, +toYear);
  var y2 = Math.max(+fromYear, +toYear);
  return ((CAPS.change || {})[y1 + "_" + y2] || {})[currentSlug] || 1;
}

// ── Map layer ─────────────────────────────────────────────────────────────────
function styleFn(feature) {
  var val = getValue(feature.properties);
  var cap = getCap();
  var color = (viewMode === "single") ? getColorSingle(val, cap) : getColorChange(val, cap);
  return {
    fillColor: color,
    weight: 0.4, color: '#555', opacity: 0.8,
    fillOpacity: (val == null) ? 0.2 : 0.72,
  };
}

var geojsonLayer = L.geoJSON(DATA, {
  style: styleFn,
  onEachFeature: function(feature, layer) {
    var p = feature.properties;
    layer.on({
      mouseover: function(e) {
        e.target.setStyle({ weight: 2, color: '#fff', fillOpacity: 0.92 });
        var val    = getValue(p);
        var cap    = getCap();
        var mLabel = (MODES.find(function(m){ return m.slug === currentSlug; }) || {}).label || currentSlug;
        var valStr, nStr, yearLabel;

        if (viewMode === "single") {
          valStr    = (val != null) ? val.toFixed(2) + "%" : "No data";
          var nKey  = "n" + currentYear.slice(-2);
          var n     = p[nKey];
          nStr      = (n != null) ? Math.round(n).toLocaleString() : "—";
          yearLabel = currentYear;
        } else {
          var sign  = (val != null && val >= 0) ? "+" : "";
          valStr    = (val != null) ? sign + val.toFixed(2) + " pp" : "No data";
          var nFrom = p["n" + fromYear.slice(-2)];
          var nTo   = p["n" + toYear.slice(-2)];
          nStr      = (nFrom != null ? Math.round(nFrom).toLocaleString() : "—")
                    + " → "
                    + (nTo   != null ? Math.round(nTo).toLocaleString()   : "—");
          yearLabel = fromYear + " → " + toYear;
        }

        layer.bindTooltip(
          '<div style="min-width:180px;line-height:1.7">' +
          '<b style="font-size:13px">' + (p.city || "—") + "</b>" +
          (p.area ? '<br><span style="font-size:11px;color:#9ca3af">' + p.area + "</span>" : "") +
          '<hr style="border:none;border-top:1px solid #374151;margin:5px 0">' +
          mLabel + " (" + yearLabel + "):<br>" +
          '<b style="font-size:14px">' + valStr + "</b><br>" +
          '<span style="font-size:10px;color:#9ca3af">Weighted commuters: ' + nStr + "</span>" +
          "</div>",
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

// ── Legend update ─────────────────────────────────────────────────────────────
function updateLegend() {
  var cap = getCap();
  var bar = document.getElementById("legend-bar");
  var low = document.getElementById("leg-low");
  var mid = document.getElementById("leg-mid");
  var high = document.getElementById("leg-high");
  var title = document.getElementById("legend-title");

  if (viewMode === "single") {
    bar.style.background = "linear-gradient(to right, " + RAMP_SEQ.join(",") + ")";
    low.textContent  = "0%";
    mid.style.display = "none";
    high.textContent = cap.toFixed(1) + "%";
    title.textContent = "Share of commuters";
  } else {
    bar.style.background = "linear-gradient(to right, " + RAMP_DIV.join(",") + ")";
    low.textContent  = "-" + cap.toFixed(1) + " pp";
    mid.style.display = "inline";
    mid.textContent  = "0";
    high.textContent = "+" + cap.toFixed(1) + " pp";
    title.textContent = "Change (percentage points)";
  }
}

// ── Mode buttons ──────────────────────────────────────────────────────────────
var list = document.getElementById("mode-list");
MODES.forEach(function(m, i) {
  var btn = document.createElement("button");
  btn.className = "mode-btn" + (m.slug === currentSlug ? " active" : "");
  btn.dataset.slug = m.slug;
  btn.textContent = (i + 1) + ". " + m.label;
  btn.onclick = function() {
    if (!btn.disabled) { currentSlug = m.slug; updateAll(); }
  };
  list.appendChild(btn);
});

function updateModeButtons() {
  document.querySelectorAll(".mode-btn").forEach(function(b) {
    var mode = MODES.find(function(m){ return m.slug === b.dataset.slug; });
    b.classList.toggle("active", b.dataset.slug === currentSlug);
    if (viewMode === "single") {
      b.disabled = (mode.years.indexOf(+currentYear) === -1);
    } else {
      // In change mode: disable if mode missing from either year
      b.disabled = (mode.years.indexOf(+fromYear) === -1 || mode.years.indexOf(+toYear) === -1);
    }
    // If currently selected mode is now disabled, fall back to bicycle
    if (b.disabled && b.dataset.slug === currentSlug) {
      currentSlug = "bicycle";
    }
  });
}

// ── Year / change selectors ───────────────────────────────────────────────────
var viewToggle   = document.getElementById("view-toggle");
var changeSelectors = document.getElementById("change-selectors");
var selFrom      = document.getElementById("sel-from");
var selTo        = document.getElementById("sel-to");

// Build view buttons: one per year + one "Change" button
YEARS.forEach(function(yr) {
  var btn = document.createElement("button");
  btn.className = "view-btn" + (String(yr) === currentYear && viewMode === "single" ? " active" : "");
  btn.dataset.year = String(yr);
  btn.textContent  = String(yr);
  btn.onclick = function() {
    viewMode    = "single";
    currentYear = btn.dataset.year;
    updateAll();
  };
  viewToggle.appendChild(btn);
});

var changeBtn = document.createElement("button");
changeBtn.className = "view-btn" + (viewMode === "change" ? " active-change" : "");
changeBtn.textContent = "Change";
changeBtn.onclick = function() {
  viewMode = "change";
  updateAll();
};
viewToggle.appendChild(changeBtn);

// Populate From/To dropdowns
YEARS.forEach(function(yr) {
  var o1 = document.createElement("option");
  o1.value = o1.textContent = String(yr);
  if (String(yr) === fromYear) o1.selected = true;
  selFrom.appendChild(o1);

  var o2 = document.createElement("option");
  o2.value = o2.textContent = String(yr);
  if (String(yr) === toYear) o2.selected = true;
  selTo.appendChild(o2);
});

selFrom.onchange = function() { fromYear = selFrom.value; updateAll(); };
selTo.onchange   = function() { toYear   = selTo.value;   updateAll(); };

// ── Master update ─────────────────────────────────────────────────────────────
function updateAll() {
  // Sync view-toggle button states
  document.querySelectorAll(".view-btn").forEach(function(b) {
    b.className = "view-btn";
    if (b.dataset.year && viewMode === "single" && b.dataset.year === currentYear) b.classList.add("active");
    if (!b.dataset.year && viewMode === "change") b.classList.add("active-change");
  });
  // Show/hide change selectors
  changeSelectors.classList.toggle("visible", viewMode === "change");

  updateModeButtons();
  geojsonLayer.setStyle(styleFn);
  updateLegend();
}

updateAll();
</script>
</body>
</html>
"""

html_out = (HTML
    .replace("__CAPS__",  caps_js)
    .replace("__MODES__", modes_js)
    .replace("__YEARS__", years_js)
    .replace("__DATA__",  geojson_str))

out_path = "transport_mode_israel.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_out)

size_mb = len(html_out) / 1_048_576
print(f"\nSaved -> {out_path}  ({size_mb:.1f} MB)")
print("Done")
