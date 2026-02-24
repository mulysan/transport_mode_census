"""
Bike usage share by statistical area (tat-rova) in Jerusalem
Based on Israel Census 2022 – Jerusalem public use file
Transport code 8 = bicycle, 9 = electric bicycle / scooter
"""

import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load census data ───────────────────────────────────────────────────────
print("Loading census data …")
df = pd.read_csv("census/census2022jer.csv", encoding="utf-8-sig")
df.columns = df.columns.str.lower().str.strip()

TRANSPORT_COL = "emtzaehagaaikarimakomavdpuf"   # main transport mode to work
AREA_COL      = "tatrovaktvtmegurimpuf"          # statistical area (tat-rova)
WEIGHT_COL    = "mishkalpratpuf"                 # person weight

BIKE_CODES = {8, 9}   # 8 = bicycle, 9 = electric bicycle / scooter

# Keep only people with a recorded transport mode (= commuters)
workers = df[df[TRANSPORT_COL].notna()].copy()
workers["is_bike"] = workers[TRANSPORT_COL].isin(BIKE_CODES)

print(f"  Total commuters: {len(workers):,}")
print(f"  Bike commuters (codes 8+9): {workers['is_bike'].sum():,}  "
      f"({workers['is_bike'].mean()*100:.1f}% city-wide)")

# ── 2. Weighted bike share by statistical area ────────────────────────────────
def weighted_stats(grp):
    w       = grp[WEIGHT_COL]
    bike_w  = (grp["is_bike"] * w).sum()
    total_w = w.sum()
    return pd.Series({
        "bike_weighted":  bike_w,
        "total_weighted": total_w,
        "bike_pct":       bike_w / total_w * 100 if total_w > 0 else np.nan,
        "n_commuters":    len(grp),
    })

stats = workers.groupby(AREA_COL).apply(weighted_stats).reset_index()
stats.rename(columns={AREA_COL: "TAT_ROVA_key"}, inplace=True)
print(f"\n  Statistical areas with commuters: {len(stats)}")
print(f"  Bike share range: {stats['bike_pct'].min():.1f}% – "
      f"{stats['bike_pct'].max():.1f}%")

# ── 3. Load & dissolve shapefile ──────────────────────────────────────────────
print("\nLoading statistical areas shapefile …")
gdf_all = gpd.read_file("statistical_areas_2022/statistical_areas_2022.shp")

# Filter Jerusalem (SEMEL_YISH == 3000)
jer = gdf_all[gdf_all["SEMEL_YISH"] == 3000].copy()

# Dissolve sub-polygons by TAT_ROVA to get one shape per statistical area
jer_dissolved = jer.dissolve(by="TAT_ROVA", aggfunc="first").reset_index()
print(f"  Jerusalem tat-rova areas: {len(jer_dissolved)}")

# ── 4. Merge ──────────────────────────────────────────────────────────────────
jer_dissolved["TAT_ROVA_key"] = jer_dissolved["TAT_ROVA"].astype(float)
stats["TAT_ROVA_key"] = stats["TAT_ROVA_key"].astype(float)

merged = jer_dissolved.merge(stats, on="TAT_ROVA_key", how="left")
print(f"  Areas with bike data: {merged['bike_pct'].notna().sum()} / {len(merged)}")

# Convert to WGS84 for web mapping
merged_wgs = merged.to_crs(epsg=4326)

# ── 5. Interactive Folium map ─────────────────────────────────────────────────
print("\nBuilding interactive map …")

center = [31.785, 35.215]   # Jerusalem city center
m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

# Colour scale
max_pct  = stats["bike_pct"].quantile(0.95)   # cap at 95th percentile for colour
colormap = folium.LinearColormap(
    colors=["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"],
    vmin=0,
    vmax=max_pct,
    caption="Bike commuters share (%) — bicycle + e-bike",
)
colormap.add_to(m)

def style_fn(feature):
    pct = feature["properties"].get("bike_pct")
    if pct is None or np.isnan(pct):
        return {
            "fillColor": "#cccccc",
            "color": "#666666",
            "weight": 0.5,
            "fillOpacity": 0.5,
        }
    color = colormap(min(pct, max_pct))
    return {
        "fillColor": color,
        "color": "#444444",
        "weight": 0.6,
        "fillOpacity": 0.75,
    }

def highlight_fn(feature):
    return {"weight": 2, "color": "#222222", "fillOpacity": 0.9}

# Build GeoJSON tooltip fields
merged_wgs["bike_pct_label"] = merged_wgs["bike_pct"].apply(
    lambda x: f"{x:.1f}%" if pd.notna(x) else "No data"
)
merged_wgs["n_commuters_label"] = merged_wgs["n_commuters"].apply(
    lambda x: f"{int(x):,}" if pd.notna(x) else "—"
)

folium.GeoJson(
    merged_wgs,
    style_function=style_fn,
    highlight_function=highlight_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["TAT_ROVA_key", "bike_pct_label", "n_commuters_label"],
        aliases=["Statistical area:", "Bike share:", "Commuters in sample:"],
        localize=True,
        sticky=True,
        labels=True,
    ),
).add_to(m)

folium.LayerControl().add_to(m)
m.save("bike_usage_jerusalem.html")
print("  Saved → bike_usage_jerusalem.html")

# ── 6. Static PNG map ─────────────────────────────────────────────────────────
print("Building static map …")

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Background: all Jerusalem areas in light grey
jer_dissolved_wgs = jer_dissolved.to_crs(epsg=4326)
jer_dissolved_wgs.plot(ax=ax, color="#e0e0e0", edgecolor="#aaaaaa", linewidth=0.3)

# Choropleth layer
cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=max_pct)

merged_wgs_plot = merged_wgs.copy()
merged_wgs_plot["color"] = merged_wgs_plot["bike_pct"].apply(
    lambda x: mcolors.to_hex(cmap(norm(x))) if pd.notna(x) else "#cccccc"
)

# Areas with data
has_data = merged_wgs_plot[merged_wgs_plot["bike_pct"].notna()]
no_data  = merged_wgs_plot[merged_wgs_plot["bike_pct"].isna()]

has_data.plot(
    ax=ax, column="bike_pct", cmap=cmap, norm=norm,
    edgecolor="#555555", linewidth=0.4, legend=False,
)
if not no_data.empty:
    no_data.plot(
        ax=ax, color="#cccccc", edgecolor="#aaaaaa", linewidth=0.3,
    )

# Colour-bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Bike commuters (%)\nbicycle + e-bike", fontsize=11)

# Legend for no-data areas
no_data_patch = mpatches.Patch(color="#cccccc", label="No data / no commuters")
ax.legend(handles=[no_data_patch], loc="lower left", fontsize=9)

ax.set_title(
    "Share of bike commuters by statistical area\nJerusalem – Census 2022",
    fontsize=15, fontweight="bold", pad=16,
)
ax.axis("off")
plt.tight_layout()
fig.savefig("bike_usage_jerusalem.png", dpi=150, bbox_inches="tight")
print("  Saved → bike_usage_jerusalem.png")

# ── 7. Summary table ─────────────────────────────────────────────────────────
print("\nTop statistical areas by bike share:")
top = (
    stats.sort_values("bike_pct", ascending=False)
    .head(10)[["TAT_ROVA_key", "bike_pct", "n_commuters"]]
    .reset_index(drop=True)
)
top.columns = ["Statistical area", "Bike share (%)", "Commuters in sample"]
top["Bike share (%)"] = top["Bike share (%)"].round(1)
print(top.to_string(index=False))

print("\nDone ✓")
