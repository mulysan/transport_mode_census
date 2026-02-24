"""
All-Israel interactive transport mode choropleth map
Israel Census 2022 – National public use file (census2022bike.csv)

Generates a single self-contained HTML with a dropdown to choose
any of the 13 transport modes (codes 1–13).
"""

import json
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Transport mode labels ──────────────────────────────────────────────────────
MODE_LABELS = {
    1:  ("כנהג – במכונית פרטית/מסחרית",        "Private car / van – driver"),
    2:  ("כנוסע – במכונית פרטית/מסחרית",       "Private car / van – passenger"),
    3:  ("אוטובוס ציבורי",                        "Public bus"),
    4:  ("רכבת קלה / מטרונית וכד׳",             "Light rail / metro"),
    5:  ("הסעה מאורגנת ע״י מקום העבודה",         "Employer-organised transport"),
    6:  ("רכבת ישראל",                            "Israel Railways"),
    7:  ("מונית שירות",                           "Sherut taxi"),
    8:  ("מונית ספיישל",                          "Special taxi"),
    9:  ("כלי רכב דו-גלגלי מנועי (אופנוע וכד׳)", "Motorised 2-wheel vehicle (motorcycle etc.)"),
    10: ("רכיבה על אופניים",                      "Bicycle"),
    11: ("הליכה ברגל",                            "Walking"),
    12: ("משאית",                                 "Truck"),
    13: ("כלי רכב אחר",                          "Other vehicle"),
}

# Mode-specific sequential colour palettes (light → dark)
MODE_COLORS = {
    1:  ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94701", "#7f2704"],  # Oranges
    2:  ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94701", "#7f2704"],  # Oranges (passenger)
    3:  ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"],  # Blues
    4:  ["#f7f4f9", "#d4b9da", "#df65b0", "#ce1256", "#67001f"],  # Pinks
    5:  ["#fff7fb", "#d0d1e6", "#74a9cf", "#0570b0", "#034e7b"],  # Steel blues
    6:  ["#f7fcfd", "#b2e2e2", "#41ae76", "#006d2c", "#00441b"],  # Greens
    7:  ["#ffffd9", "#c7e9b4", "#41b6c4", "#1d91c0", "#0c2c84"],  # Yellow-Blues
    8:  ["#ffffcc", "#c7e9b4", "#7fcdbb", "#1d91c0", "#0c2c84"],  # Yellow-Blues2
    9:  ["#fff7f3", "#fcc5c0", "#f768a1", "#ae017e", "#49006a"],  # Pinks/Purple
    10: ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],  # Greens (bike)
    11: ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],  # Purples
    12: ["#fffff4", "#d9f0a3", "#78c679", "#238443", "#004529"],  # YlGn
    13: ["#fff5eb", "#fdd0a2", "#fdae6b", "#e6550d", "#a63603"],  # Oranges2
}

TRANSPORT_COL = "emtzaehagaaikarimakomavdpuf"
YISH_COL      = "smlyishuvpuf"
ROVA_COL      = "rovaktvtmegurimpuf"
AREA_COL      = "tatrovaktvtmegurimpuf"
WEIGHT_COL    = "mishkalpratpuf"

# ── 1. Load census data ────────────────────────────────────────────────────────
print("Loading census data …")
df = pd.read_csv("census/census2022bike.csv", encoding="utf-8-sig")
df.columns = df.columns.str.lower().str.strip()
print(f"  Total rows: {len(df):,}")

# Keep only rows where transport mode is recorded
workers = df[df[TRANSPORT_COL].notna()].copy()
print(f"  With transport mode recorded: {len(workers):,}")

# Fill NaN geographic sub-codes with 0 so small (single-tat-rova) settlements
# are included in the groupby rather than silently dropped
workers[ROVA_COL] = workers[ROVA_COL].fillna(0)
workers[AREA_COL] = workers[AREA_COL].fillna(0)
print(f"  Unique (settlement, rova, tat_rova) groups after filling NaN→0: "
      f"{workers.groupby([YISH_COL, ROVA_COL, AREA_COL]).ngroups}")

# ── 2. Compute weighted share for all 13 modes in one pass ────────────────────
print("Computing weighted shares for all 13 modes …")
GROUP_COLS = [YISH_COL, ROVA_COL, AREA_COL]

for c in range(1, 14):
    workers[f"_is_{c}"] = (workers[TRANSPORT_COL] == c).astype(float)

def _agg_all(grp):
    w = grp[WEIGHT_COL]
    total_w = w.sum()
    result = {}
    for c in range(1, 14):
        target_w = (grp[f"_is_{c}"] * w).sum()
        result[f"pct_{c}"] = round(target_w / total_w * 100, 4) if total_w > 0 else None
    return pd.Series(result)

print("  Running groupby (may take ~30s for 2M rows) …")
stats = workers.groupby(GROUP_COLS, sort=False).apply(_agg_all).reset_index()
stats.rename(columns={YISH_COL: "SEMEL_YISH_k", ROVA_COL: "ROVA_k", AREA_COL: "TAT_ROVA_k"}, inplace=True)
for col in ["SEMEL_YISH_k", "ROVA_k", "TAT_ROVA_k"]:
    stats[col] = pd.to_numeric(stats[col], errors="coerce")
print(f"  Stats: {len(stats):,} (SEMEL_YISH, ROVA, TAT_ROVA) groups")

# ── 3. Load & dissolve shapefile to (SEMEL_YISH, ROVA, TAT_ROVA) ─────────────
print("Loading shapefile …")
gdf = gpd.read_file("statistical_areas_2022/statistical_areas_2022.shp")
print(f"  Total statistical area polygons: {len(gdf):,}")

# Fill NaN ROVA/TAT_ROVA with 0 so small single-area settlements are kept
gdf["ROVA"]     = pd.to_numeric(gdf["ROVA"],     errors="coerce").fillna(0)
gdf["TAT_ROVA"] = pd.to_numeric(gdf["TAT_ROVA"], errors="coerce").fillna(0)

gdf_diss = gdf.dissolve(by=["SEMEL_YISH", "ROVA", "TAT_ROVA"], aggfunc="first").reset_index()
print(f"  After dissolve: {len(gdf_diss):,} (SEMEL_YISH, ROVA, TAT_ROVA) polygons")

# Simplify geometry (tolerance in metres – EPSG:2039)
gdf_diss["geometry"] = gdf_diss["geometry"].simplify(50, preserve_topology=True)

# Convert to WGS84
gdf_wgs = gdf_diss.to_crs(epsg=4326)

# ── 4. Merge stats + geometry ─────────────────────────────────────────────────
print("Merging stats with geometry …")
gdf_wgs["SEMEL_YISH_k"] = pd.to_numeric(gdf_wgs["SEMEL_YISH"], errors="coerce")
gdf_wgs["ROVA_k"]       = pd.to_numeric(gdf_wgs["ROVA"],       errors="coerce")
gdf_wgs["TAT_ROVA_k"]   = pd.to_numeric(gdf_wgs["TAT_ROVA"],   errors="coerce")

merged = gdf_wgs.merge(
    stats,
    on=["SEMEL_YISH_k", "ROVA_k", "TAT_ROVA_k"],
    how="left",
)
print(f"  Merged: {len(merged):,} polygons "
      f"({merged['pct_10'].notna().sum()} with bicycle data)")

# ── 5. Prepare GeoJSON ────────────────────────────────────────────────────────
print("Preparing GeoJSON …")
pct_cols = [f"pct_{c}" for c in range(1, 14)]
keep = ["geometry", "SEMEL_YISH", "ROVA", "TAT_ROVA"] + pct_cols
geo = merged[keep].copy()

# Replace NaN with None for clean JSON nulls
for col in pct_cols:
    geo[col] = geo[col].where(geo[col].notna(), other=None)

geojson_str = geo.to_json(na="null")

# ── 6. Per-mode 95th-percentile colour caps ───────────────────────────────────
caps = {}
for c in range(1, 14):
    vals = merged[f"pct_{c}"].dropna()
    caps[c] = round(float(vals.quantile(0.95)), 2) if len(vals) > 0 else 10.0

# ── 7. Build option list for dropdown ────────────────────────────────────────
option_lines = []
for c in range(1, 14):
    he, en = MODE_LABELS[c]
    selected = ' selected' if c == 10 else ''
    option_lines.append(f'      <option value="{c}"{selected}>{c} · {en} — {he}</option>')
options_html = "\n".join(option_lines)

# ── 8. Serialise JS objects ───────────────────────────────────────────────────
labels_js   = json.dumps({str(c): {"he": MODE_LABELS[c][0], "en": MODE_LABELS[c][1]} for c in range(1, 14)}, ensure_ascii=False)
caps_js     = json.dumps({str(c): caps[c] for c in range(1, 14)})
colors_js   = json.dumps({str(c): MODE_COLORS[c] for c in range(1, 14)})

# ── 9. Write HTML (template-based to avoid f-string / JSON brace conflicts) ───
print("Writing HTML …")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Israel Census 2022 – Transport Mode by Tat-Rova</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/chroma-js/2.4.2/chroma.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #map { width: 100vw; height: 100vh; }

    #controls {
      position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
      z-index: 1000; background: rgba(255,255,255,0.97); border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25); padding: 14px 18px;
      min-width: 380px; max-width: 90vw;
    }
    #controls h2 {
      font-size: 13px; color: #444; margin-bottom: 10px;
      text-align: center; font-weight: 600; letter-spacing: 0.02em;
    }
    #mode-select {
      width: 100%; padding: 8px 10px; font-size: 13px;
      border: 1px solid #ccc; border-radius: 6px;
      cursor: pointer; background: #fff; color: #222;
    }
    #controls .sub {
      font-size: 11px; color: #888; text-align: center; margin-top: 6px;
    }

    #legend {
      position: absolute; bottom: 28px; right: 12px; z-index: 1000;
      background: rgba(255,255,255,0.96); border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.18); padding: 12px 14px;
      min-width: 190px;
    }
    #legend h4 { font-size: 12px; color: #333; margin-bottom: 8px; font-weight: 600; }
    #grad-bar { height: 14px; border-radius: 3px; }
    .leg-labels {
      display: flex; justify-content: space-between;
      font-size: 11px; color: #666; margin-top: 4px;
    }
    .leg-note { font-size: 10px; color: #aaa; margin-top: 6px; }

    #tip {
      position: absolute; bottom: 28px; left: 12px; z-index: 1000;
      background: rgba(255,255,255,0.96); border-radius: 8px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.18); padding: 10px 14px;
      font-size: 12px; color: #333; display: none; max-width: 260px;
      pointer-events: none;
    }
  </style>
</head>
<body>
  <div id="map"></div>

  <div id="controls">
    <h2>Israel Census 2022 · Transport Mode to Work</h2>
    <select id="mode-select">
__OPTIONS__
    </select>
    <div class="sub">Weighted share of commuters (%) · by tat-rova sub-neighbourhood</div>
  </div>

  <div id="legend">
    <h4 id="leg-title">Bicycle share (%)</h4>
    <div id="grad-bar"></div>
    <div class="leg-labels"><span>0 %</span><span id="leg-max"></span></div>
    <div class="leg-note">Grey = no data / not in sample</div>
  </div>

  <div id="tip"></div>

  <script>
    const GEOJSON  = __GEOJSON__;
    const LABELS   = __LABELS__;
    const CAPS     = __CAPS__;
    const PALETTES = __COLORS__;

    const map = L.map('map', { zoomControl: true })
      .setView([31.5, 34.9], 8);

    L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
      { attribution: '&copy; <a href="https://carto.com/">CARTO</a>', subdomains: 'abcd', maxZoom: 19 }
    ).addTo(map);

    let layer = null;
    let currentMode = '10';

    function getColor(mode, pct) {
      if (pct == null || isNaN(pct)) return '#cccccc';
      const cap    = CAPS[mode] || 10;
      const colors = PALETTES[mode] || PALETTES['1'];
      return chroma.scale(colors).domain([0, cap])(Math.min(pct, cap)).hex();
    }

    function styleFn(feature) {
      const pct = feature.properties['pct_' + currentMode];
      return {
        fillColor: getColor(currentMode, pct),
        color: '#666', weight: 0.4, fillOpacity: 0.78,
      };
    }

    function onEach(feature, lyr) {
      lyr.on({
        mouseover: function(e) {
          e.target.setStyle({ weight: 2, color: '#111', fillOpacity: 0.95 });
          const p   = feature.properties;
          const pct = p['pct_' + currentMode];
          const lbl = LABELS[currentMode];
          const tip = document.getElementById('tip');
          tip.style.display = 'block';
          tip.innerHTML =
            '<b>Settlement ' + p.SEMEL_YISH +
            ' · Rova ' + p.ROVA +
            ' · Tat-rova ' + p.TAT_ROVA + '</b><br>' +
            lbl.en + '<br>' +
            '<span dir="rtl">' + lbl.he + '</span><br>' +
            '<b>' + (pct != null && !isNaN(pct) ? pct.toFixed(2) + ' %' : 'No data') + '</b>';
        },
        mouseout: function(e) {
          layer.resetStyle(e.target);
          document.getElementById('tip').style.display = 'none';
        },
        click: function(e) { map.fitBounds(e.target.getBounds()); },
      });
    }

    function updateMode(mode) {
      currentMode = String(mode);
      const lbl  = LABELS[currentMode];
      const cap  = CAPS[currentMode] || 10;
      const pal  = PALETTES[currentMode] || PALETTES['1'];

      // Legend
      document.getElementById('leg-title').textContent = lbl.en + ' (%)';
      document.getElementById('leg-max').textContent = cap.toFixed(1) + ' %';
      document.getElementById('grad-bar').style.background =
        'linear-gradient(to right, ' + pal[0] + ', ' + pal[2] + ', ' + pal[4] + ')';

      // Recolour
      if (layer) layer.setStyle(styleFn);
    }

    // Build layer once
    layer = L.geoJSON(GEOJSON, { style: styleFn, onEachFeature: onEach }).addTo(map);

    // Initialise legend
    updateMode(10);

    document.getElementById('mode-select').addEventListener('change', function() {
      updateMode(this.value);
    });
  </script>
</body>
</html>
"""

html = (HTML_TEMPLATE
        .replace("__OPTIONS__",  options_html)
        .replace("__GEOJSON__",  geojson_str)
        .replace("__LABELS__",   labels_js)
        .replace("__CAPS__",     caps_js)
        .replace("__COLORS__",   colors_js))

out_path = "transport_mode_israel.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

import os
size_mb = os.path.getsize(out_path) / 1_048_576
print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")
print("\nDone ✓")
