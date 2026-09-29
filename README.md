# Rurality and Cumulative Preventive-Care Deficits in U.S. Counties

County-level analytic dataset combining CDC PLACES (2025 release) preventive-care estimates with USDA ERS rurality and socioeconomic data. One row per county or county-equivalent; merged on five-digit FIPS code.

**File:** `county_preventive_rurality_2025.csv` — 3,143 counties × 190 variables

## Sources

| Dataset | Publisher | Version | Variables used |
|---|---|---|---|
| PLACES: County Data (GIS Friendly Format), 2025 release | CDC | 2025 release | All 167 columns (base file) |
| 2023 Rural-Urban Continuum Codes | USDA ERS | Released Jan 22, 2024 | `RUCC_2023`, description, 2020 population |
| Poverty estimates for the U.S., States, and counties, 2023 | USDA ERS | Updated Jan 31, 2025 | `PCTPOVALL_2023`, `MEDHHINC_2023` |
| Educational attainment for adults 25+, 1970–2023 | USDA ERS | Updated Jan 31, 2025 | % bachelor's or higher and % no HS diploma, 2019–23 |
| Unemployment and median household income, 2000–23 | USDA ERS | Updated Jan 31, 2025 | `Unemployment_rate_2023`, `Median_Household_Income_2022` |

PLACES: https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2025-releas/i46a-9kgh
USDA ERS: https://www.ers.usda.gov/data-products/county-level-data-sets

## Reproducing

1. Download the five source files as CSV into `raw/` (file names are matched by pattern — see the top of `build_dataset.py`).
2. Run `python build_dataset.py` (requires pandas and numpy).
3. Run `python analysis.py` (also requires statsmodels and scipy) to reproduce the descriptive tables and logistic regression models in `results/`.
4. Run `python sensitivity.py` for sensitivity analyses: state-clustered standard errors, a three-measure score without cholesterol screening (retains KY and PA), alternative deficit cutoffs, ordinal logistic regression on the 0–4 score, and rural-urban gradient models.

PLACES is the base file; each USDA file is left-joined on FIPS, so every PLACES county is kept.

## Variables added to the PLACES columns

| Variable | Description |
|---|---|
| `FIPS` | Five-digit county FIPS code (string, leading zeros kept; renamed from PLACES `CountyFIPS`) |
| `RUCC_2023` | 2023 Rural-Urban Continuum Code (1–9) |
| `RUCC_2023_Description` | ERS text description of the RUCC category |
| `RUCC_Population_2020` | 2020 Census county population from the RUCC file |
| `PCTPOVALL_2023` | % of all persons below the poverty threshold, 2023 |
| `MEDHHINC_2023` | Median household income, 2023 (poverty file) |
| `PCT_BACHELORS_PLUS_2019_23` | % of adults 25+ with a bachelor's degree or higher, 2019–23 ACS |
| `PCT_NO_HS_DIPLOMA_2019_23` | % of adults 25+ without a high school diploma, 2019–23 ACS |
| `Unemployment_rate_2023` | Annual average unemployment rate, 2023 |
| `Median_Household_Income_2022` | Median household income, 2022 (unemployment file) |
| `Census_Region` | Northeast / Midwest / South / West, from `StateAbbr` (DC = South) |
| `Metro_Status` | Metropolitan (RUCC 1–3) or Nonmetropolitan (RUCC 4–9) |
| `Nonmetro` | 1 = nonmetropolitan, 0 = metropolitan |
| `Nonmetro_Urban_Size` | Nonmetro only: urban pop ≥20,000 (RUCC 4–5), 5,000–19,999 (6–7), <5,000 (8–9) |
| `Nonmetro_Metro_Adjacent` | Nonmetro only: Adjacent (RUCC 4, 6, 8) or Nonadjacent (5, 7, 9) |
| `Deficit_CHOLSCREEN` | 1 if `CHOLSCREEN_AdjPrev` ≤ 81.5 (national county 25th percentile) |
| `Deficit_COLON_SCREEN` | 1 if `COLON_SCREEN_AdjPrev` ≤ 54.7 |
| `Deficit_MAMMOUSE` | 1 if `MAMMOUSE_AdjPrev` ≤ 70.6 |
| `Deficit_DENTAL` | 1 if `DENTAL_AdjPrev` ≤ 53.1 |
| `Deficit_Score` | Sum of the four deficit flags (0–4); missing unless all four measures present |
| `Deficit_Any_1plus` | 1 if score ≥ 1 |
| `Deficit_2plus` | 1 if score ≥ 2 |
| `High_Deficit_Burden_3plus` | **Primary outcome:** 1 if score ≥ 3 |
| `Deficit_All_4` | 1 if score = 4 |

Percentile thresholds are computed from the data at build time (pandas linear interpolation) and reproduce the values above.

## Missing data

- **Cholesterol screening:** not estimated in PLACES 2025 for Kentucky (120 counties) or Pennsylvania (67), so `Deficit_Score` and outcomes are missing there → 2,956 counties with a score.
- **Unemployment 2023:** missing for Connecticut's 9 planning regions in the ERS file.
- **Poverty 2023:** missing for Kalawao County, HI (15005).
- Counties with complete data for the fully adjusted model (outcome, rurality, poverty, education, unemployment, region): **2,946**.

## Notes

These are aggregate, model-based county estimates; associations are ecological and should not be used to infer individual-level behavior.
