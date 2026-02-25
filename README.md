# Transport Mode by Area – Israel Census 2022

Interactive choropleth map showing the share of commuters using each transport mode, by geographic area, based on the Israel CBS 2022 Census public-use file.

## Output

**`transport_mode_israel.html`** – self-contained interactive map (no server required, open in any browser).

- 13 transport modes selectable from the side panel (car, bus, bicycle, walking, etc.)
- Colour scale updates instantly on mode change; hover for city name, area, share %, and commuter count
- 1,790 geographic areas across Israel

## Geographic levels

| Level | Cities | Detail |
|---|---|---|
| Sub-neighbourhood | 19 large cities (Jerusalem, Tel Aviv, Haifa …) | ROVA + TAT_ROVA |
| Neighbourhood | ~35 medium cities (Netivot, Dimona, Eilat …) | TAT_ROVA only |
| City | All other settlements | Settlement level |

Areas not covered by the census public-use file are shown in grey.

## Data sources

| File | Description |
|---|---|
| `census/census2022bike.csv` | Israel CBS 2022 Census PUF – national, transport-relevant columns only (2,048,457 persons) |
| `ezorim_statistiim_2022/` | CBS 2022 statistical areas geodatabase (ESRI File GDB, EPSG:2039) |

## Regenerating the map

```bash
pip install geopandas pandas numpy shapely
python bike_map_israel.py
```

The script writes `transport_mode_israel.html` to the working directory.

## Transport mode codes

| Code | Mode |
|---|---|
| 1 | Private car – driver |
| 2 | Private car – passenger |
| 3 | Public bus |
| 4 | Light rail / metro |
| 5 | Employer transport |
| 6 | Israel Railways |
| 7 | Service taxi |
| 8 | Special taxi |
| 9 | Motorcycle / moped |
| 10 | Bicycle |
| 11 | Walking |
| 12 | Truck |
| 13 | Other vehicle |
