# Transport Mode by Area – Israel Census 2008 & 2022

Interactive choropleth map showing the share of commuters using each transport mode by geographic area, based on Israel CBS census public-use files for 2008 and 2022.

## Output

**`transport_mode_israel.html`** – self-contained interactive map (no server required, open in any browser).

- 13 transport modes selectable from the side panel
- Single-year view (2008 or 2022) and change view (2008→2022, in percentage-point difference)
- Three aggregation levels: Sub-borough / Borough / City, toggled on the fly
- Hover tooltips with city name, area, share %, and weighted commuter count
- CSV export at city / borough / sub-borough level; in change mode exports pp differences
- In-map methodology panel (click **Methodology & data sources** at the bottom of the panel)

## Geographic levels

| Display name | CBS code | Coverage |
|---|---|---|
| Sub-borough | ROVA + TAT_ROVA | 19 large cities (Jerusalem, Tel Aviv, Haifa …) |
| Borough | TAT_ROVA | ~35 medium cities (Netivot, Dimona, Eilat …) |
| City | SEMEL_YISHUV | All other settlements |

Grey areas have no match in the census PUF (East Jerusalem Arab neighbourhoods and a small number of outer sub-areas excluded from the 2008 census).

## Data sources

| File | Description |
|---|---|
| `census/census2008bike.csv` | Israel CBS 2008 Census PUF – transport-relevant columns (~385 k commuters) |
| `census/census2022bike.csv` | Israel CBS 2022 Census PUF – transport-relevant columns (~761 k commuters) |
| `ezorim_statistiim_2022/` | CBS 2022 statistical areas geodatabase (ESRI File GDB, EPSG:2039) |
| `TatRova_for_census2008/` | CBS 2008 sub-area shapefile – 325 polygons, large cities only (functionally EPSG:2039) |

## Geometry strategy

- **2022 single-year view** – CBS 2022 GDB geometry with 2022 census values.
- **2008 single-year view** – 2008 shapefile for large-city sub-areas; 2022 GDB for medium/small cities with 2008 census values.
- **Change view (2008→2022)** – 2008 boundaries throughout. Each 2008 polygon's 2022 value is derived by area-weighted spatial overlay against all overlapping 2022 polygons within the same settlement (325/325 matched).

## CSV exports

Six files in `csv_exports/`:

```
transport_{year}_city.csv        — city level, one row per settlement
transport_{year}_borough.csv     — borough level (ROVA), large cities only
transport_{year}_subborough.csv  — sub-borough level (ROVA+TAT_ROVA / TAT_ROVA / settlement)
```

In-browser export (panel → Export CSV) mirrors these levels and outputs pp-difference columns when change mode is active, with filename `transport_{level}_2008_to_2022.csv`.

## Regenerating the map

```bash
pip install geopandas pandas numpy shapely
python bike_map_israel.py
```

Writes `transport_mode_israel.html` and six CSV files to the working directory.

## Transport mode codes

| Mode | 2008 code | 2022 code |
|---|:---:|:---:|
| Private car – driver | 1 | 1 |
| Private car – passenger | 2 | 2 |
| Public bus | 3 | 3 |
| Light rail / metro | — | 4 |
| Employer transport | 4 | 5 |
| Israel Railways | 5 | 6 |
| Service taxi | 6 | 7 |
| Special taxi | 7 | 8 |
| Motorcycle / moped | 8 | 9 |
| Bicycle | 9 | 10 |
| Walking | 10 | 11 |
| Truck | 11 | 12 |
| Other vehicle | 12 | 13 |

Light rail is absent from the 2008 PUF and is excluded from change-mode CSV exports.
