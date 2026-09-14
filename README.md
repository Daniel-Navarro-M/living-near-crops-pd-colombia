# Living Near Crops and the Severity of Parkinson's Disease in Southwestern Colombia

Replication files for the GEO-REF / LARGE-PD Colombia residential-proxy analysis.

## Contents

| File | Role |
|---|---|
| `living_near_crops_pd_colombia.tex` / `.pdf` | Manuscript |
| `references.bib` | Bibliography |
| `living_near_crops_analysis.py` | Builds every table in `tables/` and the figure in `figures/` from the survey export and the two derived CSVs below |
| `living_near_crops_georef_distances.py` | Distance from each geocoded life-period residence to the nearest cultivated CORINE polygon (2005 map for periods up to age 60, 2018 for 61+) → `georef_distances.csv` |
| `living_near_crops_dane_clase.py` | DANE MGN 2024 class (cabecera / centro poblado / rural disperso) of each geocoded current residence, by point-in-polygon query of the DANE map service → `dane_clase_residencia.csv` |
| `analytic_sample.csv` | Analytic sample written by the analysis script (346 patients, derived variables only) |
| `tables/`, `figures/` | Outputs read by the manuscript |

## Inputs not included

The survey export (`20260603121727_GEOREF1916_DATA_2026-06-02_1223.csv`), the geocoded coordinates (`coordenadas.csv`) and the CORINE land-cover GeoJSON files are held by the study group and are not distributed here. Paths are set at the top of each script.

## Reproduce

```
python living_near_crops_georef_distances.py   # needs geopandas; writes georef_distances.csv
python living_near_crops_dane_clase.py         # needs requests and network access; writes dane_clase_residencia.csv
python living_near_crops_analysis.py           # needs pandas, statsmodels, scipy, matplotlib
latexmk -pdf living_near_crops_pd_colombia.tex
```

The two derived CSVs are included, so the analysis script and the manuscript can be rebuilt without the raw inputs.
