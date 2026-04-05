"""
All-Israel interactive transport mode map – multi-year census

Geometry strategy:
  - 2008 single-year view:  2008 shapefile (100% coverage of 2008 sub-areas)
  - 2022 single-year view:  2022 GDB
  - Change view:            2008 geometry + area-weighted 2022 stats via spatial overlay
    (for each 2008 polygon, the 2022 value is the area-weighted average of all
     overlapping 2022 polygons within the same settlement)
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

METRIC_CRS = "EPSG:2039"

# ── Shared census column names ────────────────────────────────────────────────
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

# ── Transport modes ───────────────────────────────────────────────────────────
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
SLUGS    = [m["slug"] for m in MODES]
PCT_COLS = [f"pct_{s}" for s in SLUGS]


# ── 1. Census statistics ──────────────────────────────────────────────────────
def load_year_stats(year):
    cfg = YEAR_CFG[year]
    print(f"\n[{year}] Loading {cfg['file']} ...")
    df = pd.read_csv(cfg["file"])
    df.columns = df.columns.str.lower().str.strip()

    tc, wc = cfg["transport_col"], cfg["weight_col"]
    workers = df[df[tc].notna() & ~df[tc].isin(cfg["exclude_codes"])].copy()
    print(f"  {len(workers):,} commuters")

    W = workers[wc]
    for m in MODES:
        code = m[year]
        workers[f"wm_{m['slug']}"] = ((workers[tc] == code) * W) if code else 0.0
    workers["w_tot"] = W

    sum_cols = [f"wm_{s}" for s in SLUGS] + ["w_tot"]

    def compute(grp_df, grp_cols):
        agg = grp_df.groupby(grp_cols)[sum_cols].sum().reset_index()
        for s in SLUGS:
            agg[f"pct_{s}"] = (agg[f"wm_{s}"] / agg["w_tot"] * 100).round(3)
        return agg

    stats = {
        "large":  compute(workers[workers[ROVA_COL].notna()], [SETT_COL, ROVA_COL, AREA_COL]),
        "medium": compute(workers[workers[ROVA_COL].isna() & workers[AREA_COL].notna()], [SETT_COL, AREA_COL]),
        "small":  compute(workers[workers[AREA_COL].isna()], [SETT_COL]),
    }
    print(f"  Groups: large={len(stats['large'])}, medium={len(stats['medium'])}, small={len(stats['small'])}")
    return stats


# ── 2. Geometries ─────────────────────────────────────────────────────────────
def to_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_shapes_2022():
    gdb = "ezorim_statistiim_2022.gdb"
    if not os.path.exists(gdb):
        os.symlink(str(pathlib.Path("ezorim_statistiim_2022").resolve()), gdb)
    print("\nLoading 2022 GDB ...")
    gdf = gpd.read_file(gdb)

    diss_large  = gdf[gdf["ROVA"].notna()].dissolve(
                      by=["SEMEL_YISHUV","ROVA","TAT_ROVA"], aggfunc="first").reset_index()
    diss_medium = gdf[gdf["ROVA"].isna() & gdf["TAT_ROVA"].notna()].dissolve(
                      by=["SEMEL_YISHUV","TAT_ROVA"], aggfunc="first").reset_index()
    diss_small  = gdf[gdf["TAT_ROVA"].isna()].dissolve(
                      by=["SEMEL_YISHUV"], aggfunc="first").reset_index()
    print(f"  Dissolved: large={len(diss_large)}, medium={len(diss_medium)}, small={len(diss_small)}")

    city_names = {}
    for _, row in gdf.iterrows():
        sn = row["SEMEL_YISHUV"]
        if pd.notna(sn) and int(sn) not in city_names:
            city_names[int(sn)] = (
                str(row.get("SHEM_YISHUV_ENGLISH") or "").strip(),
                str(row.get("SHEM_YISHUV") or "").strip(),
            )
    return gdf.crs, diss_large, diss_medium, diss_small, city_names

def load_shapes_2008():
    print("Loading 2008 shapefile ...")
    shp = gpd.read_file("TatRova_for_census2008/TatRova_for_census2008.shp")
    shp = shp.rename(columns={"SEMEL_YISH": "SEMEL_YISHUV", "TATROVA": "TAT_ROVA"})
    # The 2008 shp is functionally EPSG:2039 but lacks the authority code
    shp = shp.set_crs(METRIC_CRS, allow_override=True)
    for c in ["SEMEL_YISHUV","ROVA","TAT_ROVA"]:
        shp[c] = pd.to_numeric(shp[c], errors="coerce")
    diss = shp.dissolve(by=["SEMEL_YISHUV","ROVA","TAT_ROVA"], aggfunc="first").reset_index()
    print(f"  Dissolved: {len(diss)} large-city polygons")
    return diss


# ── 3. Merge census stats onto shapes ────────────────────────────────────────
def merge_stats_on_shapes(diss_large, diss_medium, diss_small, stats, crs_large, crs_med_small):
    """
    Merge census stats onto three tiers of shapes.
    crs_large, crs_med_small: CRS objects for the respective tiers (may differ for 2008).
    Returns a list of (gdf_with_stats, geo_level) tuples.
    """
    def _merge(diss, on_cols, stat_df, stat_on_cols, level, crs):
        d = diss.copy()
        to_float(d, on_cols)
        to_float(stat_df, stat_on_cols)
        merged = d.merge(
            stat_df[stat_on_cols + ["w_tot"] + PCT_COLS],
            left_on=on_cols, right_on=stat_on_cols, how="left"
        )
        merged["geo_level"] = level
        return gpd.GeoDataFrame(merged, crs=crs)

    large_gdf  = _merge(diss_large,  ["SEMEL_YISHUV","ROVA","TAT_ROVA"],
                         stats["large"],  [SETT_COL, ROVA_COL, AREA_COL],
                         "sub_neighbourhood", crs_large)
    medium_gdf = _merge(diss_medium, ["SEMEL_YISHUV","TAT_ROVA"],
                         stats["medium"], [SETT_COL, AREA_COL],
                         "neighbourhood", crs_med_small)
    small_gdf  = _merge(diss_small,  ["SEMEL_YISHUV"],
                         stats["small"],  [SETT_COL],
                         "city", crs_med_small)
    return large_gdf, medium_gdf, small_gdf


# ── 4. Spatial overlay: project 2022 stats onto 2008 polygons ─────────────────
def project_2022_onto_2008(gdf_08_large, gdf_22_all):
    """
    For each 2008 large-city polygon, compute area-weighted average of 2022
    transport shares from overlapping 2022 polygons within the same settlement.

    gdf_22_all: all 2022 tiers combined (large + medium + small) in METRIC_CRS.
    Some settlements were 'large' in 2008 but 'medium' in 2022, so we need all tiers.
    Both inputs must already be in METRIC_CRS.
    Returns DataFrame with columns [SEMEL_YISHUV, ROVA, TAT_ROVA, n22, pct22_{slug}...]
    """
    to_float(gdf_08_large, ["SEMEL_YISHUV","ROVA","TAT_ROVA"])
    to_float(gdf_22_all,   ["SEMEL_YISHUV"])

    settlements_08 = set(gdf_08_large["SEMEL_YISHUV"].dropna().astype(int))
    settlements_22 = set(gdf_22_all["SEMEL_YISHUV"].dropna().astype(int))
    common = settlements_08 & settlements_22
    print(f"  Spatial overlay: {len(common)} settlements, {len(gdf_08_large)} 2008 polygons ...")

    pct22_cols = [f"pct22_{s}" for s in SLUGS]
    results = []

    for sett in sorted(common):
        g08_s = gdf_08_large[gdf_08_large["SEMEL_YISHUV"] == sett][
            ["SEMEL_YISHUV","ROVA","TAT_ROVA","geometry"]].copy().reset_index(drop=True)
        g22_s = gdf_22_all[gdf_22_all["SEMEL_YISHUV"] == sett][
            ["geometry","w_tot"] + PCT_COLS].copy().reset_index(drop=True)

        if len(g22_s) == 0:
            continue

        try:
            inter = gpd.overlay(g08_s, g22_s, how="intersection", keep_geom_type=False)
        except Exception:
            continue

        if len(inter) == 0:
            continue

        inter["_area"] = inter.geometry.area

        for (r08, t08), grp in inter.groupby(["ROVA","TAT_ROVA"]):
            total_area = grp["_area"].sum()
            if total_area == 0:
                continue
            w = grp["_area"] / total_area
            row = {"SEMEL_YISHUV": sett, "ROVA": r08, "TAT_ROVA": t08,
                   "n22": float((grp["w_tot"] * w).sum())}
            for s in SLUGS:
                col = f"pct_{s}"
                row[f"pct22_{s}"] = float((grp[col] * w).sum()) if col in grp.columns else None
            results.append(row)

    df = pd.DataFrame(results) if results else pd.DataFrame(
        columns=["SEMEL_YISHUV","ROVA","TAT_ROVA","n22"] + pct22_cols)
    print(f"  Overlay done: {len(df)} 2008 polygons got 2022 stats")
    return df


# ── 5. GeoJSON helpers ────────────────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 3)
    except Exception:
        return None

def gdf_to_features(gdf, year_tag, city_names=None, extra_cols=None):
    """
    Convert a GeoDataFrame to GeoJSON features.
    year_tag:   "08" or "22" — determines property prefix p{tag}_{slug} and n{tag}.
    extra_cols: dict of {prop_name: col_name} for additional properties.
    """
    geo = gdf.copy()
    if geo.crs and geo.crs.to_epsg() != 4326:
        geo = geo.to_crs(epsg=4326)
    geo["geometry"] = geo["geometry"].simplify(0.0003, preserve_topology=True)

    features = []
    for _, row in geo.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue

        city_en = str(row.get("SHEM_YISHUV_ENGLISH") or "").strip()
        city_he = str(row.get("SHEM_YISHUV") or "").strip()
        if not city_en and city_names:
            sn = safe_float(row.get("SEMEL_YISHUV"))
            if sn is not None:
                city_en, city_he = city_names.get(int(sn), ("", ""))
        city = city_en if city_en else city_he

        level = row.get("geo_level", "city")
        rova  = row.get("ROVA"); tat = row.get("TAT_ROVA")
        if level == "sub_neighbourhood" and pd.notna(rova) and pd.notna(tat):
            area = f"Sub-neighbourhood {int(rova)}/{int(tat)}"
        elif level == "neighbourhood" and pd.notna(tat):
            area = f"Neighbourhood {int(tat)}"
        else:
            area = None

        props = {"city": city, "area": area, "level": level,
                 f"n{year_tag}": safe_float(row.get("w_tot"))}
        for s in SLUGS:
            props[f"p{year_tag}_{s}"] = safe_float(row.get(f"pct_{s}"))

        if extra_cols:
            for prop, col in extra_cols.items():
                props[prop] = safe_float(row.get(col))

        features.append({"type": "Feature", "properties": props, "geometry": mapping(geom)})
    return features


# ── 6. Colour caps ────────────────────────────────────────────────────────────
def compute_caps(stats_by_year, change_features_08):
    single_caps = {}
    for yr in YEARS:
        st = stats_by_year[yr]
        # Use all tiers to compute caps
        all_vals = {}
        for tier_df in st.values():
            for s in SLUGS:
                col = f"pct_{s}"
                if col in tier_df.columns:
                    all_vals.setdefault(s, []).extend(tier_df[col].dropna().tolist())
        caps = {}
        for s in SLUGS:
            vals = all_vals.get(s, [])
            caps[s] = float(round(np.percentile(vals, 95), 2)) if vals else 1.0
        single_caps[str(yr)] = caps

    # Change caps from 2008 features (which have p08_* and p22_* both)
    change_caps = {}
    for i, y1 in enumerate(YEARS):
        for y2 in YEARS[i+1:]:
            pair_key = f"{y1}_{y2}"
            t1, t2 = str(y1)[2:], str(y2)[2:]
            diffs_by_slug = {s: [] for s in SLUGS}
            for f in change_features_08:
                p = f["properties"]
                for s in SLUGS:
                    v1 = p.get(f"p{t1}_{s}")
                    v2 = p.get(f"p{t2}_{s}")
                    if v1 is not None and v2 is not None:
                        diffs_by_slug[s].append(abs(v2 - v1))
            pair_caps = {}
            for s in SLUGS:
                d = diffs_by_slug[s]
                pair_caps[s] = float(round(np.percentile(d, 95), 2)) if d else 1.0
            change_caps[pair_key] = pair_caps

    return {"single": single_caps, "change": change_caps}


# ── MAIN ──────────────────────────────────────────────────────────────────────
stats_08 = load_year_stats(2008)
stats_22 = load_year_stats(2022)

crs22, diss22_large, diss22_medium, diss22_small, city_names = load_shapes_2022()
diss08_large = load_shapes_2008()

# ── 2022 single-year view ─────────────────────────────────────────────────────
print("\nBuilding 2022 layer ...")
g22_large, g22_medium, g22_small = merge_stats_on_shapes(
    diss22_large, diss22_medium, diss22_small, stats_22, crs22, crs22)
features_22 = (
    gdf_to_features(g22_large,  "22", city_names) +
    gdf_to_features(g22_medium, "22", city_names) +
    gdf_to_features(g22_small,  "22", city_names)
)
print(f"  {len(features_22)} features")

# ── 2008 single-year + change view ────────────────────────────────────────────
print("\nBuilding 2008 layer (2008 shp for large, 2022 GDB for medium/small) ...")
# Large: merge 2008 stats onto 2008 shp polygons
g08_large = diss08_large.copy()
to_float(g08_large, ["SEMEL_YISHUV","ROVA","TAT_ROVA"])
to_float(stats_08["large"], [SETT_COL, ROVA_COL, AREA_COL])
g08_large = g08_large.merge(
    stats_08["large"][[SETT_COL, ROVA_COL, AREA_COL, "w_tot"] + PCT_COLS],
    left_on=["SEMEL_YISHUV","ROVA","TAT_ROVA"],
    right_on=[SETT_COL, ROVA_COL, AREA_COL], how="left"
)
g08_large["geo_level"] = "sub_neighbourhood"
g08_large = gpd.GeoDataFrame(g08_large, crs=METRIC_CRS)

# Medium/small: use 2022 GDB geometry with 2008 stats
_, g08_medium, g08_small = merge_stats_on_shapes(
    diss22_large, diss22_medium, diss22_small, stats_08, crs22, crs22)
# (g08_medium and g08_small have 2022 geometry but 2008 stats)

# Spatial overlay: project 2022 stats onto 2008 polygons
# Use all 2022 tiers: some settlements were 'large' in 2008 but are 'medium' in 2022
print("\nSpatial overlay: projecting 2022 stats onto 2008 polygons ...")
g22_all_metric = pd.concat([
    g22_large .to_crs(METRIC_CRS)[["SEMEL_YISHUV","geometry","w_tot"] + PCT_COLS],
    g22_medium.to_crs(METRIC_CRS)[["SEMEL_YISHUV","geometry","w_tot"] + PCT_COLS],
    g22_small .to_crs(METRIC_CRS)[["SEMEL_YISHUV","geometry","w_tot"] + PCT_COLS],
], ignore_index=True)
g22_all_metric = gpd.GeoDataFrame(g22_all_metric, crs=METRIC_CRS)
g08_large_metric = g08_large.copy()   # already in METRIC_CRS
p22_on_08 = project_2022_onto_2008(g08_large_metric, g22_all_metric)

# Merge p22 data back onto g08_large
to_float(p22_on_08, ["SEMEL_YISHUV","ROVA","TAT_ROVA"])
g08_large = g08_large.merge(p22_on_08, on=["SEMEL_YISHUV","ROVA","TAT_ROVA"], how="left")

# For medium/small change view: key-match 2022 stats to 2022-geometry 2008 layer
# g08_medium/small already have pct_* (2008 stats); add pct22_* via key lookup
to_float(g08_medium, ["SEMEL_YISHUV","TAT_ROVA"])
to_float(g08_small,  ["SEMEL_YISHUV"])
to_float(stats_22["medium"], [SETT_COL, AREA_COL])
to_float(stats_22["small"],  [SETT_COL])

g08_medium = g08_medium.merge(
    stats_22["medium"][[SETT_COL, AREA_COL, "w_tot"] + PCT_COLS].rename(
        columns={c: c.replace("pct_","pct22_") for c in PCT_COLS} | {"w_tot": "n22", SETT_COL: "_sett22", AREA_COL: "_area22"}),
    left_on=["SEMEL_YISHUV","TAT_ROVA"], right_on=["_sett22","_area22"], how="left"
).drop(columns=["_sett22","_area22"], errors="ignore")

g08_small = g08_small.merge(
    stats_22["small"][[SETT_COL, "w_tot"] + PCT_COLS].rename(
        columns={c: c.replace("pct_","pct22_") for c in PCT_COLS} | {"w_tot": "n22", SETT_COL: "_sett22"}),
    left_on=["SEMEL_YISHUV"], right_on=["_sett22"], how="left"
).drop(columns=["_sett22"], errors="ignore")

# Build features for 2008 layer (has both p08_* and p22_* for change view)
print("Building 2008 GeoJSON ...")
extra_22 = {f"p22_{s}": f"pct22_{s}" for s in SLUGS}
extra_22["n22"] = "n22"

features_08 = (
    gdf_to_features(g08_large,  "08", city_names, extra_22) +
    gdf_to_features(g08_medium, "08", city_names, extra_22) +
    gdf_to_features(g08_small,  "08", city_names, extra_22)
)
print(f"  {len(features_08)} features")

# Caps
caps = compute_caps({2008: stats_08, 2022: stats_22}, features_08)

geojson_08 = json.dumps({"type": "FeatureCollection", "features": features_08}, ensure_ascii=False)
geojson_22 = json.dumps({"type": "FeatureCollection", "features": features_22}, ensure_ascii=False)
print(f"\nDATA_08: ~{len(geojson_08)//1024} KB,  DATA_22: ~{len(geojson_22)//1024} KB")

caps_js  = json.dumps(caps)
years_js = json.dumps(YEARS)
modes_js = json.dumps([
    {"slug": m["slug"], "label": m["label"],
     "years": [yr for yr in YEARS if m[yr] is not None]}
    for m in MODES
], ensure_ascii=False)



# ── HTML template ─────────────────────────────────────────────────────────────
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
#view-toggle { display: flex; gap: 4px; margin-top: 8px; }
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
#leg-mid { display: none; }
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
      <span id="leg-mid">0</span>
      <span id="leg-high">-</span>
    </div>
    <div id="geo-note">
      Large cities: sub-neighbourhood level<br>
      Medium cities: neighbourhood level<br>
      Small settlements: city level<br>
      Grey = no data<br>
      <span id="change-note" style="display:none;color:#7c3aed">
        Change view uses 2008 boundaries.<br>
        2022 values are area-weighted.
      </span>
    </div>
  </div>
</div>
<script>
var CAPS    = __CAPS__;
var MODES   = __MODES__;
var YEARS   = __YEARS__;
var DATA_08 = __DATA_08__;
var DATA_22 = __DATA_22__;

// ── State ─────────────────────────────────────────────────────────────────────
var viewMode    = "single";
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

// ── Colour helpers ─────────────────────────────────────────────────────────
function hexToRgb(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}
function rampColor(ramp, t) {
  var n = ramp.length - 1;
  var i = Math.min(Math.floor(t * n), n - 1);
  var f = t * n - i;
  var a = hexToRgb(ramp[i]), b = hexToRgb(ramp[i+1]);
  return 'rgb('+Math.round(a[0]+(b[0]-a[0])*f)+','+Math.round(a[1]+(b[1]-a[1])*f)+','+Math.round(a[2]+(b[2]-a[2])*f)+')';
}
var RAMP_SEQ = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'];
var RAMP_DIV = ['#d73027','#f46d43','#fdae61','#fee090','#ffffbf','#e0f3f8','#abd9e9','#74add1','#4575b4'];

function getColorSingle(pct, cap) {
  if (pct == null) return '#6b7280';
  return rampColor(RAMP_SEQ, Math.min(pct / (cap || 1), 1));
}
function getColorChange(diff, cap) {
  if (diff == null) return '#6b7280';
  var t = (Math.max(-cap, Math.min(cap, diff)) / (cap || 1) + 1) / 2;
  return rampColor(RAMP_DIV, t);
}

// ── Value / cap ───────────────────────────────────────────────────────────────
function getValue(props) {
  if (viewMode === "single") {
    return props["p" + currentYear.slice(-2) + "_" + currentSlug];
  }
  // change: DATA_08 has p08_* (exact) and p22_* (area-weighted)
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

// ── Layer management ──────────────────────────────────────────────────────────
var activeLayer = null;

function styleFn(feature) {
  var val = getValue(feature.properties);
  var cap = getCap();
  var color = (viewMode === "single")
    ? getColorSingle(val, cap)
    : getColorChange(val, cap);
  return {
    fillColor: color, weight: 0.4, color: '#555', opacity: 0.8,
    fillOpacity: (val == null) ? 0.2 : 0.72,
  };
}

function makeOnEach(feature, layer) {
  var p = feature.properties;
  layer.on({
    mouseover: function(e) {
      e.target.setStyle({ weight: 2, color: '#fff', fillOpacity: 0.92 });
      var val    = getValue(p);
      var mLabel = (MODES.find(function(m){ return m.slug === currentSlug; }) || {}).label || currentSlug;
      var valStr, nStr, yearLabel;

      if (viewMode === "single") {
        valStr    = (val != null) ? val.toFixed(2) + "%" : "No data";
        var n     = p["n" + currentYear.slice(-2)];
        nStr      = (n != null) ? Math.round(n).toLocaleString() : "\u2014";
        yearLabel = currentYear;
      } else {
        var sign  = (val != null && val >= 0) ? "+" : "";
        valStr    = (val != null) ? sign + val.toFixed(2) + " pp" : "No data";
        var nFrom = p["n" + fromYear.slice(-2)];
        var nTo   = p["n" + toYear.slice(-2)];
        nStr      = (nFrom != null ? Math.round(nFrom).toLocaleString() : "\u2014")
                  + " \u2192 "
                  + (nTo   != null ? Math.round(nTo).toLocaleString()   : "\u2014");
        yearLabel = fromYear + " \u2192 " + toYear;
      }

      layer.bindTooltip(
        '<div style="min-width:180px;line-height:1.7">' +
        '<b style="font-size:13px">' + (p.city || "\u2014") + "</b>" +
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
      if (activeLayer) activeLayer.resetStyle(e.target);
      e.target.closeTooltip();
    },
  });
}

function getActiveData() {
  // Change view and 2008 single-year both use DATA_08 (2008 geometry)
  if (viewMode === "change" || currentYear === "2008") return DATA_08;
  return DATA_22;
}

function rebuildLayer() {
  if (activeLayer) { activeLayer.remove(); activeLayer = null; }
  activeLayer = L.geoJSON(getActiveData(), {
    style: styleFn,
    onEachFeature: makeOnEach,
  }).addTo(map);
}

// ── Legend ────────────────────────────────────────────────────────────────────
function updateLegend() {
  var cap   = getCap();
  var bar   = document.getElementById("legend-bar");
  var low   = document.getElementById("leg-low");
  var mid   = document.getElementById("leg-mid");
  var high  = document.getElementById("leg-high");
  var title = document.getElementById("legend-title");
  var note  = document.getElementById("change-note");

  if (viewMode === "single") {
    bar.style.background = "linear-gradient(to right," + RAMP_SEQ.join(",") + ")";
    low.textContent   = "0%";
    mid.style.display = "none";
    high.textContent  = cap.toFixed(1) + "%";
    title.textContent = "Share of commuters";
    note.style.display = "none";
  } else {
    bar.style.background = "linear-gradient(to right," + RAMP_DIV.join(",") + ")";
    low.textContent   = "-" + cap.toFixed(1) + " pp";
    mid.style.display = "inline";
    mid.textContent   = "0";
    high.textContent  = "+" + cap.toFixed(1) + " pp";
    title.textContent = "Change (percentage points)";
    note.style.display = "block";
  }
}

// ── Mode buttons ──────────────────────────────────────────────────────────────
var modeList = document.getElementById("mode-list");
MODES.forEach(function(m, i) {
  var btn = document.createElement("button");
  btn.className    = "mode-btn" + (m.slug === currentSlug ? " active" : "");
  btn.dataset.slug = m.slug;
  btn.textContent  = (i + 1) + ". " + m.label;
  btn.onclick = function() {
    if (!btn.disabled) { currentSlug = m.slug; updateAll(false); }
  };
  modeList.appendChild(btn);
});

function updateModeButtons() {
  document.querySelectorAll(".mode-btn").forEach(function(b) {
    var mode = MODES.find(function(m){ return m.slug === b.dataset.slug; });
    b.classList.toggle("active", b.dataset.slug === currentSlug);
    if (viewMode === "single") {
      b.disabled = (mode.years.indexOf(+currentYear) === -1);
    } else {
      b.disabled = (mode.years.indexOf(+fromYear) === -1 || mode.years.indexOf(+toYear) === -1);
    }
    if (b.disabled && b.dataset.slug === currentSlug) currentSlug = "bicycle";
  });
}

// ── View toggle ───────────────────────────────────────────────────────────────
var viewToggle = document.getElementById("view-toggle");
var changeSel  = document.getElementById("change-selectors");
var selFrom    = document.getElementById("sel-from");
var selTo      = document.getElementById("sel-to");

YEARS.forEach(function(yr) {
  var btn = document.createElement("button");
  btn.className    = "view-btn";
  btn.dataset.year = String(yr);
  btn.textContent  = String(yr);
  btn.onclick = function() { viewMode = "single"; currentYear = btn.dataset.year; updateAll(true); };
  viewToggle.appendChild(btn);
});

var changeBtn = document.createElement("button");
changeBtn.className   = "view-btn";
changeBtn.textContent = "Change";
changeBtn.onclick = function() { viewMode = "change"; updateAll(true); };
viewToggle.appendChild(changeBtn);

YEARS.forEach(function(yr) {
  var o1 = new Option(yr, yr); if (String(yr) === fromYear) o1.selected = true;
  selFrom.appendChild(o1);
  var o2 = new Option(yr, yr); if (String(yr) === toYear) o2.selected = true;
  selTo.appendChild(o2);
});
selFrom.onchange = function() { fromYear = selFrom.value; updateAll(false); };
selTo.onchange   = function() { toYear   = selTo.value;   updateAll(false); };

// ── Master update ─────────────────────────────────────────────────────────────
function updateAll(rebuild) {
  document.querySelectorAll(".view-btn").forEach(function(b) {
    b.className = "view-btn";
    if (b.dataset.year && viewMode === "single" && b.dataset.year === currentYear)
      b.classList.add("active");
    if (!b.dataset.year && viewMode === "change")
      b.classList.add("active-change");
  });
  changeSel.classList.toggle("visible", viewMode === "change");
  updateModeButtons();
  updateLegend();
  if (rebuild) {
    rebuildLayer();
  } else if (activeLayer) {
    activeLayer.setStyle(styleFn);
  }
}

rebuildLayer();
updateAll(false);
</script>
</body>
</html>
"""

html_out = (HTML
    .replace("__CAPS__",    caps_js)
    .replace("__MODES__",   modes_js)
    .replace("__YEARS__",   years_js)
    .replace("__DATA_08__", geojson_08)
    .replace("__DATA_22__", geojson_22))

out_path = "transport_mode_israel.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html_out)

size_mb = len(html_out) / 1_048_576
print(f"\nSaved -> {out_path}  ({size_mb:.1f} MB)")
print("Done")
