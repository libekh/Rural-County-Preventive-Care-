"""
Sensitivity and extension analyses for the rurality / preventive-care deficit study.

Input:  county_preventive_rurality_2025.csv  (built by build_dataset.py)
Output: results/sensitivity_*.csv

  S1. State-clustered standard errors (Models 1-3)
  S2. Three-measure deficit score excluding cholesterol screening (retains KY and PA)
  S3. Alternative deficit cutoffs (20th / 33rd percentile) and burden thresholds (>=2 of 4, 4 of 4)
  S4. Ordinal logistic regression on the full 0-4 deficit score
  S5. Rural-urban gradient: RUCC trend, nine-category RUCC, urban size and metro adjacency
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

base = pd.read_csv(os.path.join(HERE, "county_preventive_rurality_2025.csv"), dtype={"FIPS": str})
base["Census_Region"] = pd.Categorical(base["Census_Region"], categories=["Northeast", "Midwest", "South", "West"])

ALL4 = ["CHOLSCREEN_AdjPrev", "COLON_SCREEN_AdjPrev", "MAMMOUSE_AdjPrev", "DENTAL_AdjPrev"]
NO_CHOL = ["COLON_SCREEN_AdjPrev", "MAMMOUSE_AdjPrev", "DENTAL_AdjPrev"]
SES = "PCTPOVALL_2023 + PCT_BACHELORS_PLUS_2019_23 + Unemployment_rate_2023"
COVARS = ["PCTPOVALL_2023", "PCT_BACHELORS_PLUS_2019_23", "Unemployment_rate_2023", "Census_Region"]


def add_outcome(df, measures, pct, k, name="Y"):
    """Deficit = at or below the national `pct` percentile; outcome = deficits in >= k measures."""
    d = df.copy()
    flags = []
    for m in measures:
        q = d[m].quantile(pct)
        d[f"_def_{m}"] = np.where(d[m].isna(), np.nan, (d[m] <= q).astype(float))
        flags.append(f"_def_{m}")
    complete = d[flags].notna().all(axis=1)
    d["_score"] = np.where(complete, d[flags].sum(axis=1), np.nan)
    d[name] = np.where(complete, (d["_score"] >= k).astype(float), np.nan)
    return d


def fit(d, rhs, term="Nonmetro", cluster=False, y="Y"):
    cols = [y, "Nonmetro", "StateAbbr"] + [c for c in COVARS if c in rhs]
    if "RUCC" in rhs:
        cols.append("RUCC_2023")
    for c in ("Nonmetro_Urban_Size", "Nonmetro_Metro_Adjacent"):
        if c in rhs:
            cols.append(c)
    d = d.dropna(subset=list(dict.fromkeys(cols)))
    m = smf.logit(f"{y} ~ {rhs}", data=d)
    kw = dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["StateAbbr"])[0]}) if cluster else {}
    r = m.fit(disp=0, maxiter=200, **kw)
    ci = r.conf_int()
    ev = d[y].sum()
    by_region = d.groupby("Census_Region", observed=True)[y].sum()
    zero_regions = [str(g) for g, v in by_region.items() if v == 0] if "Census_Region" in rhs else []
    return {"N": int(r.nobs), "Events": int(ev), "OR": np.exp(r.params[term]),
            "CI_low": np.exp(ci.loc[term, 0]), "CI_high": np.exp(ci.loc[term, 1]), "p": r.pvalues[term],
            "Zero_event_regions": ", ".join(zero_regions)}, r


RHS = {"Model 1": "Nonmetro", "Model 2": f"Nonmetro + {SES}", "Model 3": f"Nonmetro + {SES} + C(Census_Region)"}
rows = []

# ---- Primary definition, conventional vs state-clustered SEs (S1) --------------------
prim = add_outcome(base, ALL4, 0.25, 3)
for mname, rhs in RHS.items():
    for cl in (False, True):
        res, _ = fit(prim, rhs, cluster=cl)
        rows.append({"Analysis": "S1 Primary (4 measures, 25th pct, >=3)",
                     "SE": "State-clustered" if cl else "Conventional", "Model": mname, **res})
# Model 3 excluding the zero-event Northeast (separation check)
res, _ = fit(prim[prim.Census_Region != "Northeast"].assign(
    Census_Region=lambda x: x.Census_Region.cat.remove_unused_categories()), RHS["Model 3"], cluster=True)
rows.append({"Analysis": "S1 Primary, Northeast excluded", "SE": "State-clustered", "Model": "Model 3", **res})

# ---- S2: three-measure score without cholesterol (keeps KY and PA) ---------------------
for k, lab in [(2, ">=2 of 3"), (3, "3 of 3")]:
    d = add_outcome(base, NO_CHOL, 0.25, k)
    for mname, rhs in RHS.items():
        res, _ = fit(d, rhs, cluster=True)
        rows.append({"Analysis": f"S2 3 measures, no cholesterol, 25th pct, {lab}",
                     "SE": "State-clustered", "Model": mname, **res})

# ---- S3: alternative percentile cutoffs and burden thresholds ------------------------
for pct, k in [(0.20, 3), (1 / 3, 3), (0.25, 2), (0.25, 4)]:
    d = add_outcome(base, ALL4, pct, k)
    for mname in ("Model 1", "Model 3"):
        res, _ = fit(d, RHS[mname], cluster=True)
        rows.append({"Analysis": f"S3 4 measures, {round(pct * 100)}th pct, >={k} of 4",
                     "SE": "State-clustered", "Model": mname, **res})

sens = pd.DataFrame(rows)
sens.to_csv(os.path.join(OUT, "sensitivity_models.csv"), index=False, float_format="%.4f")

# ---- S4: ordinal logistic regression on the 0-4 score ---------------------------------
ord_rows = []
d = prim.dropna(subset=["_score", "Nonmetro", "PCTPOVALL_2023", "PCT_BACHELORS_PLUS_2019_23",
                        "Unemployment_rate_2023", "Census_Region"]).copy()
X = pd.get_dummies(d[["Nonmetro", "PCTPOVALL_2023", "PCT_BACHELORS_PLUS_2019_23", "Unemployment_rate_2023",
                      "Census_Region"]], columns=["Census_Region"], drop_first=True, dtype=float)
specs = {"Unadjusted": ["Nonmetro"], "Adjusted (SES + region)": list(X.columns)}
for lab, cols in specs.items():
    om = OrderedModel(d["_score"].astype(int), X[cols], distr="logit")
    r = om.fit(method="bfgs", disp=0, maxiter=2000)
    ci = r.conf_int()
    ord_rows.append({"Model": lab, "N": int(r.nobs), "OR": np.exp(r.params["Nonmetro"]),
                     "CI_low": np.exp(ci.loc["Nonmetro", 0]), "CI_high": np.exp(ci.loc["Nonmetro", 1]),
                     "p": r.pvalues["Nonmetro"]})
# Brant-style check of proportional odds: compare nonmetro ORs across cumulative splits
for k in (1, 2, 3, 4):
    dd = d.assign(Y=(d["_score"] >= k).astype(float))
    res, _ = fit(dd, RHS["Model 3"], cluster=True)
    ord_rows.append({"Model": f"Binary split: score >= {k} (adjusted, clustered)", **{
        x: res[x] for x in ("N", "OR", "CI_low", "CI_high", "p")}})
ordinal = pd.DataFrame(ord_rows)
ordinal.to_csv(os.path.join(OUT, "sensitivity_ordinal.csv"), index=False, float_format="%.4f")

# ---- S5: rural-urban gradient -----------------------------------------------------------
grad_rows = []
for adj, extra in [("Unadjusted", ""), ("Adjusted (SES + region)", f" + {SES} + C(Census_Region)")]:
    # linear trend across RUCC 1-9
    dd = prim.dropna(subset=["RUCC_2023"])
    cols = ["Y", "RUCC_2023", "StateAbbr"] + (COVARS if extra else [])
    dd = dd.dropna(subset=cols)
    r = smf.logit(f"Y ~ RUCC_2023{extra}", data=dd).fit(
        disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["StateAbbr"])[0]})
    ci = r.conf_int()
    grad_rows.append({"Model": adj, "Term": "RUCC linear trend (per 1-code increase)", "N": int(r.nobs),
                      "OR": np.exp(r.params["RUCC_2023"]), "CI_low": np.exp(ci.loc["RUCC_2023", 0]),
                      "CI_high": np.exp(ci.loc["RUCC_2023", 1]), "p": r.pvalues["RUCC_2023"]})
    # nine categories, RUCC 1 reference
    r = smf.logit(f"Y ~ C(RUCC_2023){extra}", data=dd).fit(
        disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["StateAbbr"])[0]})
    ci = r.conf_int()
    for t in [t for t in r.params.index if t.startswith("C(RUCC_2023)")]:
        code = t.split("[T.")[1].rstrip("]").replace(".0", "")
        grad_rows.append({"Model": adj, "Term": f"RUCC {code} vs 1", "N": int(r.nobs), "OR": np.exp(r.params[t]),
                          "CI_low": np.exp(ci.loc[t, 0]), "CI_high": np.exp(ci.loc[t, 1]), "p": r.pvalues[t]})
    # among nonmetro counties: urban size and adjacency, mutually adjusted
    nm = dd[dd.Nonmetro == 1].copy()
    nm["Size"] = pd.Categorical(nm["Nonmetro_Urban_Size"],
                                categories=["Urban pop >=20,000", "Urban pop 5,000-19,999", "Urban pop <5,000"])
    nm["Nonadjacent"] = (nm["Nonmetro_Metro_Adjacent"] == "Nonadjacent").astype(float)
    r = smf.logit(f"Y ~ C(Size) + Nonadjacent{extra}", data=nm).fit(
        disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": pd.factorize(nm["StateAbbr"])[0]})
    ci = r.conf_int()
    for t, lab in [("C(Size)[T.Urban pop 5,000-19,999]", "Nonmetro: urban 5,000-19,999 vs >=20,000"),
                   ("C(Size)[T.Urban pop <5,000]", "Nonmetro: urban <5,000 vs >=20,000"),
                   ("Nonadjacent", "Nonmetro: nonadjacent vs adjacent to metro")]:
        grad_rows.append({"Model": adj, "Term": lab, "N": int(r.nobs), "OR": np.exp(r.params[t]),
                          "CI_low": np.exp(ci.loc[t, 0]), "CI_high": np.exp(ci.loc[t, 1]), "p": r.pvalues[t]})
gradient = pd.DataFrame(grad_rows)
gradient.to_csv(os.path.join(OUT, "sensitivity_gradient.csv"), index=False, float_format="%.4f")

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
for name, t in [("MODELS", sens), ("ORDINAL", ordinal), ("GRADIENT", gradient)]:
    print(f"\n==== {name}")
    print(t.round(3).to_string(index=False))
