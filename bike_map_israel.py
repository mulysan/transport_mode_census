"""
All-Israel interactive transport mode map
Based on Israel Census 2022 national PUF (census/census2022bike.csv)

Geographic hierarchy (3 tiers):
  Large cities  (19 settlements): sub-neighbourhood level  (SEMEL_YISH + ROVA + TAT_ROVA)
  Medium cities (35 settlements): district level            (SEMEL_YISH + TAT_ROVA, no ROVA)
  Small towns   (remaining)     : city level                (SEMEL_YISH only)

Transport codes confirmed from CBS codebook / STATA:
  1=Car driver, 2=Car passenger, 3=Public bus, 4=Light rail/metro,
  5=Employer transport, 6=Israel Railways, 7=Service taxi, 8=Special taxi,
  9=Motorcycle, 10=Bicycle, 11=Walking, 12=Truck, 13=Other
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import warnings
warnings.filterwarnings("ignore")

# ── Column aliases ─────────────────────────────────────────────────────────────
TRANSPORT_COL = "emtzaehagaaikarimakomavdpuf"
ROVA_COL      = "rovaktvtmegurimpuf"
AREA_COL      = "tatrovaktvtmegurimpuf"
SETT_COL      = "smlyishuvpuf"
WEIGHT_COL    = "mishkalpratpuf"

TRANSPORT_MODES = {
    1:  "Private car – driver",
    2:  "Private car – passenger",
    3:  "Public bus",
    4:  "Light rail / metro",
    5:  "Employer transport",
    6:  "Israel Railways",
    7:  "Service taxi (sherut)",
    8:  "Special taxi",
    9:  "Motorcycle / moped",
    10: "Bicycle",
    11: "Walking",
    12: "Truck",
    13: "Other vehicle",
}

# ── 1. Load census ─────────────────────────────────────────────────────────────
print("Loading census data…")
df = pd.read_csv("census/census2022bike.csv", encoding="utf-8-sig")
df.columns = df.columns.str.lower().str.strip()
print(f"  {len(df):,} rows total")

# Keep only commuters (non-missing transport code) — matches STATA "replace . if missing"
workers = df[df[TRANSPORT_COL].notna()].copy()
print(f"  {len(workers):,} commuters with recorded mode")

# Weighted mode dummies  (sum later → weighted proportion = STATA collapse [aweight])
W = workers[WEIGHT_COL]
for m in range(1, 14):
    workers[f"wm_{m}"] = (workers[TRANSPORT_COL] == m) * W
workers["w_tot"] = W

sum_cols = [f"wm_{m}" for m in range(1, 14)] + ["w_tot"]

# Split into 3 geographic tiers
w_large  = workers[workers[ROVA_COL].notna()]                                    # has ROVA
w_medium = workers[workers[ROVA_COL].isna() & workers[AREA_COL].notna()]         # TAT_ROVA only
w_small  = workers[workers[AREA_COL].isna()]                                     # neither
print(f"  Commuter split → large:{len(w_large):,}  medium:{len(w_medium):,}  small:{len(w_small):,}")


# ── 2. Compute weighted mode shares per tier ───────────────────────────────────
def compute_pcts(grp_df, grp_cols):
    agg = grp_df.groupby(grp_cols)[sum_cols].sum().reset_index()
    for m in range(1, 14):
        agg[f"pct_{m}"] = (agg[f"wm_{m}"] / agg["w_tot"] * 100).round(3)
    return agg

print("Computing mode shares…")
stats_large  = compute_pcts(w_large,  [SETT_COL, ROVA_COL, AREA_COL])
stats_medium = compute_pcts(w_medium, [SETT_COL, AREA_COL])
stats_small  = compute_pcts(w_small,  [SETT_COL])
print(f"  Stat groups → large:{len(stats_large)}  medium:{len(stats_medium)}  small:{len(stats_small)}")


# ── 3. Load & dissolve shapefile ───────────────────────────────────────────────
print("Loading shapefile (GDB)…")
import os, pathlib
gdb_path = "ezorim_statistiim_2022.gdb"
if not os.path.exists(gdb_path):
    os.symlink(str(pathlib.Path("ezorim_statistiim_2022").resolve()), gdb_path)
gdf = gpd.read_file(gdb_path)

shp_large  = gdf[gdf["ROVA"].notna()].copy()
shp_medium = gdf[gdf["ROVA"].isna() & gdf["TAT_ROVA"].notna()].copy()
shp_small  = gdf[gdf["TAT_ROVA"].isna()].copy()

print("Dissolving shapefile tiers…")
diss_large  = shp_large.dissolve( by=["SEMEL_YISHUV", "ROVA", "TAT_ROVA"], aggfunc="first").reset_index()
diss_medium = shp_medium.dissolve(by=["SEMEL_YISHUV", "TAT_ROVA"],          aggfunc="first").reset_index()
diss_small  = shp_small.dissolve( by=["SEMEL_YISHUV"],                       aggfunc="first").reset_index()
print(f"  Dissolved areas → large:{len(diss_large)}  medium:{len(diss_medium)}  small:{len(diss_small)}")


# ── 4. Merge stats into dissolved shapes ───────────────────────────────────────
pct_cols = [f"pct_{m}" for m in range(1, 14)]

def to_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Large: (SEMEL_YISHUV, ROVA, TAT_ROVA)
diss_large  = to_float(diss_large,  ["SEMEL_YISHUV", "ROVA", "TAT_ROVA"])
stats_large = to_float(stats_large, [SETT_COL, ROVA_COL, AREA_COL])
m_large = diss_large.merge(
    stats_large[[SETT_COL, ROVA_COL, AREA_COL, "w_tot"] + pct_cols],
    left_on=["SEMEL_YISHUV", "ROVA", "TAT_ROVA"],
    right_on=[SETT_COL, ROVA_COL, AREA_COL],
    how="left",
)
m_large["geo_level"] = "sub_neighbourhood"

# Medium: (SEMEL_YISHUV, TAT_ROVA)
diss_medium  = to_float(diss_medium,  ["SEMEL_YISHUV", "TAT_ROVA"])
stats_medium = to_float(stats_medium, [SETT_COL, AREA_COL])
m_medium = diss_medium.merge(
    stats_medium[[SETT_COL, AREA_COL, "w_tot"] + pct_cols],
    left_on=["SEMEL_YISHUV", "TAT_ROVA"],
    right_on=[SETT_COL, AREA_COL],
    how="left",
)
m_medium["geo_level"] = "district"

# Small: (SEMEL_YISHUV)
diss_small  = to_float(diss_small,  ["SEMEL_YISHUV"])
stats_small = to_float(stats_small, [SETT_COL])
m_small = diss_small.merge(
    stats_small[[SETT_COL, "w_tot"] + pct_cols],
    left_on=["SEMEL_YISHUV"],
    right_on=[SETT_COL],
    how="left",
)
m_small["geo_level"] = "city"

# Combine all tiers
keep_cols = (
    ["SEMEL_YISHUV", "SHEM_YISHUV", "SHEM_YISHUV_ENGLISH", "ROVA", "TAT_ROVA",
     "geo_level", "geometry", "w_tot"]
    + pct_cols
)

def slim(d):
    return d[[c for c in keep_cols if c in d.columns]].copy()

combined = pd.concat([slim(m_large), slim(m_medium), slim(m_small)], ignore_index=True)
combined = gpd.GeoDataFrame(combined, crs=gdf.crs).to_crs(epsg=4326)

# Simplify geometry to reduce HTML file size
combined["geometry"] = combined["geometry"].simplify(0.0003, preserve_topology=True)
print(f"  Combined areas: {len(combined)}")


# ── 5. Build GeoJSON features ──────────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 3)
    except Exception:
        return None

print("Building GeoJSON…")
features = []
for _, row in combined.iterrows():
    geom = row["geometry"]
    if geom is None or geom.is_empty:
        continue

    # City name: prefer English, keep Hebrew if no English available
    city_en = str(row.get("SHEM_YISHUV_ENGLISH") or "").strip()
    city_he = str(row.get("SHEM_YISHUV") or "").strip()
    city    = city_en if city_en else city_he   # English first; Hebrew fallback

    level = row.get("geo_level", "city")
    rova  = row.get("ROVA")
    tat   = row.get("TAT_ROVA")

    # Area qualifier (English, numeric) shown below city name in tooltip
    if level == "sub_neighbourhood" and pd.notna(rova) and pd.notna(tat):
        area = f"Sub-neighbourhood {int(rova)}/{int(tat)}"
    elif level == "district" and pd.notna(tat):
        area = f"Neighbourhood {int(tat)}"
    else:
        area = None

    props = {
        "city":  city,
        "area":  area,
        "level": level,
        "n":     safe_float(row.get("w_tot")),
    }
    for m in range(1, 14):
        props[f"p{m}"] = safe_float(row.get(f"pct_{m}"))

    features.append({
        "type": "Feature",
        "properties": props,
        "geometry": mapping(geom),
    })

geojson_str = json.dumps(
    {"type": "FeatureCollection", "features": features},
    ensure_ascii=False,
)
print(f"  GeoJSON: {len(features)} features, ~{len(geojson_str)//1024} KB")


# ── 6. Colour caps (95th percentile per mode) ──────────────────────────────────
caps = {}
for m in range(1, 14):
    vals = combined[f"pct_{m}"].dropna()
    caps[m] = float(round(vals.quantile(0.95), 2)) if len(vals) else 1.0

caps_js  = json.dumps({str(k): v for k, v in caps.items()})
modes_js = json.dumps({str(k): v for k, v in TRANSPORT_MODES.items()}, ensure_ascii=False)


# ── 7. Write HTML ──────────────────────────────────────────────────────────────
print("Writing HTML…")

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Israel Census 2022 – Transport Mode Map</title>
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
#panel-header { padding: 14px 14px 10px; border-bottom: 1px solid #374151; }
#panel-header h1 { font-size: 13px; font-weight: 700; color: #f9fafb; line-height: 1.5; }
#panel-header p  { font-size: 10px; color: #6b7280; margin-top: 3px; }
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
.mode-btn:hover { background: #374151; color: #f3f4f6; border-color: #4b5563; }
.mode-btn.active {
  background: #1d4ed8; border-color: #2563eb;
  color: #fff; font-weight: 600;
}
#panel-footer { padding: 10px 14px; border-top: 1px solid #374151; }
#legend-title { font-size: 10px; color: #6b7280; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.06em; }
#legend-bar {
  height: 10px; border-radius: 3px; margin: 4px 0;
  background: linear-gradient(to right, #ffffcc, #a1dab4, #41b6c4, #2c7fb8, #253494);
}
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
    <h1>Israel Census 2022<br>Transport Mode to Work</h1>
    <p>Share of commuters (%) by area</p>
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
      Grey = no data
    </div>
  </div>
</div>

<script>
var CAPS  = __CAPS__;
var MODES = __MODES__;
var DATA  = __DATA__;

var currentMode = 10;

var map = L.map('map', { preferCanvas: true }).setView([31.5, 34.9], 8);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com">CARTO</a>',
  subdomains: 'abcd', maxZoom: 19
}).addTo(map);

// 5-stop sequential colour ramp (YlGnBu)
var RAMP = ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'];

function hexToRgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
function getColor(pct, cap) {
  if (pct == null) return '#6b7280';
  var t = Math.min(pct / (cap || 1), 1);
  var n = RAMP.length - 1;
  var i = Math.min(Math.floor(t * n), n - 1);
  var f = t * n - i;
  var a = hexToRgb(RAMP[i]), b = hexToRgb(RAMP[i + 1]);
  return 'rgb(' +
    Math.round(a[0] + (b[0]-a[0])*f) + ',' +
    Math.round(a[1] + (b[1]-a[1])*f) + ',' +
    Math.round(a[2] + (b[2]-a[2])*f) + ')';
}

function styleFn(feature) {
  var p   = feature.properties['p' + currentMode];
  var cap = CAPS[String(currentMode)];
  return {
    fillColor:   getColor(p, cap),
    weight:      0.4,
    color:       '#555',
    opacity:     0.8,
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
        var pct    = p['p' + currentMode];
        var pctStr = (pct != null) ? pct.toFixed(2) + '%' : 'No data';
        var nStr   = (p.n != null) ? Math.round(p.n).toLocaleString() : '—';
        layer.bindTooltip(
          '<div style="min-width:180px;line-height:1.7">' +
          '<b style="font-size:13px">' + (p.city || '—') + '</b>' +
          (p.area ? '<br><span style="font-size:11px;color:#9ca3af">' + p.area + '</span>' : '') +
          '<hr style="border:none;border-top:1px solid #374151;margin:5px 0">' +
          MODES[String(currentMode)] + ':<br>' +
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
  var cap = CAPS[String(currentMode)];
  document.getElementById('leg-cap').textContent = cap.toFixed(1) + '%';
  document.querySelectorAll('.mode-btn').forEach(function(b) {
    b.classList.toggle('active', +b.dataset.mode === currentMode);
  });
}

// Build mode buttons
var list = document.getElementById('mode-list');
Object.keys(MODES).map(Number).sort(function(a,b){return a-b;}).forEach(function(m) {
  var btn = document.createElement('button');
  btn.className = 'mode-btn' + (m === currentMode ? ' active' : '');
  btn.dataset.mode = m;
  btn.textContent = m + '. ' + MODES[m];
  btn.onclick = function() { currentMode = m; updateMap(); };
  list.appendChild(btn);
});

// Initialise legend cap
document.getElementById('leg-cap').textContent = CAPS[String(currentMode)].toFixed(1) + '%';
</script>
</body>
</html>
"""

html_out = (
    HTML
    .replace("__CAPS__",  caps_js)
    .replace("__MODES__", modes_js)
    .replace("__DATA__",  geojson_str)
)

out_path = "transport_mode_israel.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_out)

size_mb = len(html_out) / 1_048_576
print(f"Saved → {out_path}  ({size_mb:.1f} MB)")
print("Done ✓")
