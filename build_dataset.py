"""
Build the county-level analytic dataset for the rurality / preventive-care deficit study.

Inputs (place in ./raw/):
  PLACES__County_Data_GIS_Friendly_Format_2025_release.csv   (CDC PLACES, 2025 release)
  Ruralurbancontinuumcodes2023.csv                            (USDA ERS RUCC 2023)
  Poverty2023.csv                                             (USDA ERS poverty estimates, 2023)
  Education2023.csv                                           (USDA ERS educational attainment, 1970-2023)
  Unemployment2023.csv                                        (USDA ERS unemployment & median HH income, 2000-23)

Output:
  county_preventive_rurality_2025.csv
"""
import glob
import os
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "county_preventive_rurality_2025.csv")


def find(pattern):
    hits = glob.glob(os.path.join(RAW, pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected one file matching {pattern} in {RAW}, found {hits}")
    return hits[0]


def fips5(s):
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)


def usda_wide(path, fips_col, attrs, rename):
    """Pivot a long-format USDA ERS file (FIPS, ..., Attribute, Value) to one row per county."""
    d = pd.read_csv(path, dtype=str, encoding="latin1")
    d = d[d["Attribute"].isin(attrs)].copy()
    d["FIPS"] = fips5(d[fips_col])
    w = d.pivot_table(index="FIPS", columns="Attribute", values="Value", aggfunc="first").reset_index()
    w = w.rename(columns=rename)
    for c in w.columns:
        if c != "FIPS" and c not in ("RUCC_2023_Description",):
            w[c] = pd.to_numeric(w[c], errors="coerce")
    return w


# ---- 1. CDC PLACES (base file) ----------------------------------------------
places = pd.read_csv(find("PLACES*County*2025*.csv"), dtype={"CountyFIPS": str})
places["CountyFIPS"] = fips5(places["CountyFIPS"])
places = places.rename(columns={"CountyFIPS": "FIPS"})

# ---- 2. USDA ERS files --------------------------------------------------------
rucc = usda_wide(
    find("Ruralurbancontinuumcodes2023*.csv"), "FIPS",
    ["RUCC_2023", "Description", "Population_2020"],
    {"Description": "RUCC_2023_Description", "Population_2020": "RUCC_Population_2020"},
)
pov = usda_wide(
    find("Poverty2023*.csv"), "FIPS_Code",
    ["PCTPOVALL_2023", "MEDHHINC_2023"], {},
)
edu = usda_wide(
    find("Education2023*.csv"), "FIPS Code",
    ["Percent of adults with a bachelor's degree or higher, 2019-23",
     "Percent of adults who are not high school graduates, 2019-23"],
    {"Percent of adults with a bachelor's degree or higher, 2019-23": "PCT_BACHELORS_PLUS_2019_23",
     "Percent of adults who are not high school graduates, 2019-23": "PCT_NO_HS_DIPLOMA_2019_23"},
)
unemp = usda_wide(
    find("Unemployment2023*.csv"), "FIPS_Code",
    ["Unemployment_rate_2023", "Median_Household_Income_2022"], {},
)

# ---- 3. Sequential left joins onto PLACES ---------------------------------------
df = places
for part in (rucc, pov, edu, unemp):
    df = df.merge(part, on="FIPS", how="left", validate="one_to_one")

# ---- 4. Derived variables ---------------------------------------------------------
REGION = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN",
              "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
state_region = {s: r for r, ss in REGION.items() for s in ss}
df["Census_Region"] = df["StateAbbr"].map(state_region)

r = df["RUCC_2023"]
df["Metro_Status"] = np.select([r.between(1, 3), r.between(4, 9)], ["Metropolitan", "Nonmetropolitan"], default=None)
df["Nonmetro"] = np.where(r.isna(), np.nan, (r >= 4).astype(float))
df["Nonmetro_Urban_Size"] = np.select(
    [r.isin([4, 5]), r.isin([6, 7]), r.isin([8, 9])],
    ["Urban pop >=20,000", "Urban pop 5,000-19,999", "Urban pop <5,000"], default=None)
df["Nonmetro_Metro_Adjacent"] = np.select(
    [r.isin([4, 6, 8]), r.isin([5, 7, 9])], ["Adjacent", "Nonadjacent"], default=None)

MEASURES = {"CHOLSCREEN": "CHOLSCREEN_AdjPrev", "COLON_SCREEN": "COLON_SCREEN_AdjPrev",
            "MAMMOUSE": "MAMMOUSE_AdjPrev", "DENTAL": "DENTAL_AdjPrev"}
thresholds = {}
for short, col in MEASURES.items():
    q25 = df[col].quantile(0.25)
    thresholds[short] = q25
    df[f"Deficit_{short}"] = np.where(df[col].isna(), np.nan, (df[col] <= q25).astype(float))

deficit_cols = [f"Deficit_{s}" for s in MEASURES]
complete = df[deficit_cols].notna().all(axis=1)
df["Deficit_Score"] = np.where(complete, df[deficit_cols].sum(axis=1), np.nan)
for k, name in [(1, "Deficit_Any_1plus"), (2, "Deficit_2plus"), (3, "High_Deficit_Burden_3plus"), (4, "Deficit_All_4")]:
    df[name] = np.where(complete, (df["Deficit_Score"] >= k).astype(float), np.nan)

df.to_csv(OUT, index=False)

print(f"Wrote {OUT}: {df.shape[0]} counties x {df.shape[1]} variables")
print("25th-percentile thresholds:", {k: round(v, 2) for k, v in thresholds.items()})
print("Unmatched after merge:", {c: int(df[c].isna().sum()) for c in
      ["RUCC_2023", "PCTPOVALL_2023", "PCT_BACHELORS_PLUS_2019_23", "Unemployment_rate_2023", "Census_Region"]})
print("Deficit score computable:", int(complete.sum()))
