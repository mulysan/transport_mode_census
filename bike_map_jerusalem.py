"""
Bike usage share by statistical area (tat-rova) in Jerusalem
Based on Israel Census 2022 – Jerusalem public use file
Transport code 10 = bicycle, 9 = electric bicycle / scooter
(Code 10 confirmed from STATA: gen bike = (emtzaehagaaikarimakomavdpuf == 10))

Generates separate maps for:
  • bicycle only (code 10)
  • e-bike / scooter only (code 9)
  • both combined (codes 9 + 10)
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

# ── 1. Load census data once ──────────────────────────────────────────────────
print("Loading census data …")
df = pd.read_csv("census/census2022jer.csv", encoding="utf-8-sig")
df.columns = df.columns.str.lower().str.strip()

TRANSPORT_COL = "emtzaehagaaikarimakomavdpuf"   # main transport mode to work
ROVA_COL      = "rovaktvtmegurimpuf"             # neighbourhood (rova)
AREA_COL      = "tatrovaktvtmegurimpuf"          # sub-neighbourhood (tat-rova)
WEIGHT_COL    = "mishkalpratpuf"                 # person weight

# All commuters (transport code recorded)
# The census PUF geographic detail stops at tat-rova level – no STAT_2022 field.
workers = df[df[TRANSPORT_COL].notna()].copy()
print(f"  Total commuters: {len(workers):,}")

# ── 2. Load & dissolve shapefile once ─────────────────────────────────────────
print("Loading shapefile …")
gdf_all = gpd.read_file("statistical_areas_2022/statistical_areas_2022.shp")
jer = gdf_all[gdf_all["SEMEL_YISH"] == 3000].copy()
# Dissolve the fine STAT_2022 polygons up to (ROVA, TAT_ROVA) level –
# the finest geography available in the census PUF.
jer_dissolved = jer.dissolve(by=["ROVA", "TAT_ROVA"], aggfunc="first").reset_index()
jer_dissolved["ROVA_key"]     = jer_dissolved["ROVA"].astype(float)
jer_dissolved["TAT_ROVA_key"] = jer_dissolved["TAT_ROVA"].astype(float)
jer_dissolved_wgs = jer_dissolved.to_crs(epsg=4326)
print(f"  Jerusalem tat-rova areas: {len(jer_dissolved)}")


# ── 3. Helper: compute weighted share for a given set of mode codes ───────────
def compute_stats(mode_codes):
    w_df = workers.copy()
    w_df["is_target"] = w_df[TRANSPORT_COL].isin(mode_codes)

    def _agg(grp):
        w       = grp[WEIGHT_COL]
        target_w = (grp["is_target"] * w).sum()
        total_w  = w.sum()
        return pd.Series({
            "target_weighted": target_w,
            "total_weighted":  total_w,
            "pct":             target_w / total_w * 100 if total_w > 0 else np.nan,
            "n_commuters":     len(grp),
        })

    stats = w_df.groupby([ROVA_COL, AREA_COL]).apply(_agg).reset_index()
    stats.rename(columns={ROVA_COL: "ROVA_key", AREA_COL: "TAT_ROVA_key"}, inplace=True)
    stats["ROVA_key"]     = stats["ROVA_key"].astype(float)
    stats["TAT_ROVA_key"] = stats["TAT_ROVA_key"].astype(float)
    return stats


# ── 4. Helper: build and save one pair of maps ────────────────────────────────
def make_maps(stats, slug, title, folium_colors, mpl_cmap):
    """
    stats        : DataFrame from compute_stats()
    slug         : filename stem, e.g. "bicycle"
    title        : human-readable title for plot
    folium_colors: list of hex colours for folium.LinearColormap
    mpl_cmap     : matplotlib colormap name string
    """
    print(f"\n── {title} ──")
    pct_min = stats["pct"].min()
    pct_max = stats["pct"].max()
    cap     = stats["pct"].quantile(0.95)   # colour cap
    print(f"  Share range: {pct_min:.2f}% – {pct_max:.2f}%  (colour cap {cap:.2f}%)")

    # Merge geometry + stats on composite key (ROVA, TAT_ROVA)
    merged = jer_dissolved.merge(stats, on=["ROVA_key", "TAT_ROVA_key"], how="left")
    merged_wgs = merged.to_crs(epsg=4326)

    # Label columns for tooltip
    merged_wgs["area_label"] = merged_wgs.apply(
        lambda r: f"{int(r['ROVA_key'])} / {int(r['TAT_ROVA_key'])}", axis=1
    )
    merged_wgs["pct_label"] = merged_wgs["pct"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "No data"
    )
    merged_wgs["n_label"] = merged_wgs["n_commuters"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "—"
    )

    # ── Interactive Folium map ─────────────────────────────────────────────────
    center = [31.785, 35.215]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

    colormap = folium.LinearColormap(
        colors=folium_colors,
        vmin=0,
        vmax=cap,
        caption=f"{title} – share of commuters (%)",
    )
    colormap.add_to(m)

    def style_fn(feature):
        pct = feature["properties"].get("pct")
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            return {"fillColor": "#cccccc", "color": "#666", "weight": 0.5, "fillOpacity": 0.5}
        return {
            "fillColor": colormap(min(pct, cap)),
            "color": "#444",
            "weight": 0.6,
            "fillOpacity": 0.75,
        }

    def highlight_fn(feature):
        return {"weight": 2, "color": "#111", "fillOpacity": 0.9}

    folium.GeoJson(
        merged_wgs,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["area_label", "pct_label", "n_label"],
            aliases=["Rova / Tat-rova:", f"{title} share:", "Commuters in sample:"],
            localize=True,
            sticky=True,
            labels=True,
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    html_path = f"bike_usage_jerusalem_{slug}.html"
    m.save(html_path)
    print(f"  Saved → {html_path}")

    # ── Static PNG map ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))
    jer_dissolved_wgs.plot(ax=ax, color="#e8e8e8", edgecolor="#aaa", linewidth=0.3)

    cmap = plt.get_cmap(mpl_cmap)
    norm = mcolors.Normalize(vmin=0, vmax=cap)

    has_data = merged_wgs[merged_wgs["pct"].notna()]
    no_data  = merged_wgs[merged_wgs["pct"].isna()]

    has_data.plot(
        ax=ax, column="pct", cmap=cmap, norm=norm,
        edgecolor="#555", linewidth=0.4, legend=False,
    )
    if not no_data.empty:
        no_data.plot(ax=ax, color="#cccccc", edgecolor="#aaa", linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"{title}\nshare of commuters (%)", fontsize=11)

    if not no_data.empty:
        nd_patch = mpatches.Patch(color="#cccccc", label="No data / no commuters")
        ax.legend(handles=[nd_patch], loc="lower left", fontsize=9)

    ax.set_title(
        f"{title}\nby statistical area – Jerusalem Census 2022",
        fontsize=15, fontweight="bold", pad=16,
    )
    ax.axis("off")
    plt.tight_layout()
    png_path = f"bike_usage_jerusalem_{slug}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {png_path}")


# ── 5. Generate all three maps ────────────────────────────────────────────────

VARIANTS = [
    {
        "codes":         {10},
        "slug":          "bicycle",
        "title":         "Bicycle commuters",
        "folium_colors": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
        "mpl_cmap":      "Greens",
    },
    {
        "codes":         {9},
        "slug":          "ebike",
        "title":         "E-bike / scooter commuters",
        "folium_colors": ["#fcfbfd", "#dadaeb", "#9e9ac8", "#6a51a3", "#3f007d"],
        "mpl_cmap":      "Purples",
    },
    {
        "codes":         {9, 10},
        "slug":          "combined",
        "title":         "Bicycle + e-bike / scooter commuters",
        "folium_colors": ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"],
        "mpl_cmap":      "Blues",
    },
]

for v in VARIANTS:
    stats = compute_stats(v["codes"])
    make_maps(stats, v["slug"], v["title"], v["folium_colors"], v["mpl_cmap"])

print("\nAll maps done ✓")
