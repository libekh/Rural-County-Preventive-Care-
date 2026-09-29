"""
Analysis: rurality and cumulative preventive-care deficit burden in U.S. counties.

Input:  county_preventive_rurality_2025.csv  (built by build_dataset.py)
Output: results/ folder with descriptive tables and logistic regression results.
"""
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(HERE, "county_preventive_rurality_2025.csv"), dtype={"FIPS": str})

MEASURES = ["CHOLSCREEN_AdjPrev", "COLON_SCREEN_AdjPrev", "MAMMOUSE_AdjPrev", "DENTAL_AdjPrev"]
COVARS = ["PCTPOVALL_2023", "PCT_BACHELORS_PLUS_2019_23", "Unemployment_rate_2023", "MEDHHINC_2023"]
OUTCOMES = ["Deficit_Any_1plus", "Deficit_2plus", "High_Deficit_Burden_3plus", "Deficit_All_4"]


def fmt_mean_sd(s):
    s = s.dropna()
    return f"{s.mean():.1f} ({s.std():.1f})"


def fmt_n_pct(s):
    s = s.dropna()
    return f"{int(s.sum())} ({100 * s.mean():.1f}%)"


# ---- Table 1: county characteristics by metro status ---------------------------
groups = {"All counties": df,
          "Metropolitan": df[df.Metro_Status == "Metropolitan"],
          "Nonmetropolitan": df[df.Metro_Status == "Nonmetropolitan"]}
rows = [("Counties, n", {g: str(len(d)) for g, d in groups.items()})]
for v in COVARS + MEASURES:
    rows.append((f"{v}, mean (SD)", {g: fmt_mean_sd(d[v]) for g, d in groups.items()}))
rows.append(("Counties with deficit score, n", {g: str(int(d.Deficit_Score.notna().sum())) for g, d in groups.items()}))
rows.append(("Deficit score, mean (SD)", {g: fmt_mean_sd(d.Deficit_Score) for g, d in groups.items()}))
for v in OUTCOMES:
    rows.append((f"{v}, n (%)", {g: fmt_n_pct(d[v]) for g, d in groups.items()}))
t1 = pd.DataFrame({k: v for k, v in rows}).T
t1.index.name = "Characteristic"
t1.to_csv(os.path.join(OUT, "table1_by_metro_status.csv"))

# ---- Table 2: deficit outcomes across the nine RUCC categories ------------------
scored = df[df.Deficit_Score.notna()]
t2 = scored.groupby("RUCC_2023").agg(
    Counties=("FIPS", "size"),
    Mean_deficit_score=("Deficit_Score", "mean"),
    Pct_1plus=("Deficit_Any_1plus", "mean"),
    Pct_2plus=("Deficit_2plus", "mean"),
    Pct_3plus_high_burden=("High_Deficit_Burden_3plus", "mean"),
    Pct_all_4=("Deficit_All_4", "mean"),
)
for c in [c for c in t2.columns if c.startswith("Pct")]:
    t2[c] = (100 * t2[c]).round(1)
t2["Mean_deficit_score"] = t2["Mean_deficit_score"].round(2)
desc = df.drop_duplicates("RUCC_2023").set_index("RUCC_2023")["RUCC_2023_Description"]
t2.insert(0, "Description", desc.reindex(t2.index))
t2.index = t2.index.astype(int)
t2.to_csv(os.path.join(OUT, "table2_by_rucc.csv"))

# Deficit-score distribution by metro status
dist = pd.crosstab(scored.Metro_Status, scored.Deficit_Score.astype(int), margins=True)
dist.to_csv(os.path.join(OUT, "deficit_score_distribution.csv"))

# ---- Table 3: logistic regression, sequential models -------------------------------
df["Census_Region"] = pd.Categorical(df["Census_Region"], categories=["Northeast", "Midwest", "South", "West"])
models = {
    "Model 1 (unadjusted)": ("High_Deficit_Burden_3plus ~ Nonmetro", []),
    "Model 2 (+ SES)": ("High_Deficit_Burden_3plus ~ Nonmetro + PCTPOVALL_2023 + PCT_BACHELORS_PLUS_2019_23"
                        " + Unemployment_rate_2023", []),
    "Model 3 (+ region)": ("High_Deficit_Burden_3plus ~ Nonmetro + PCTPOVALL_2023 + PCT_BACHELORS_PLUS_2019_23"
                           " + Unemployment_rate_2023 + C(Census_Region)", []),
}
needed = {
    "Model 1 (unadjusted)": ["High_Deficit_Burden_3plus", "Nonmetro"],
    "Model 2 (+ SES)": ["High_Deficit_Burden_3plus", "Nonmetro", "PCTPOVALL_2023",
                        "PCT_BACHELORS_PLUS_2019_23", "Unemployment_rate_2023"],
    "Model 3 (+ region)": ["High_Deficit_Burden_3plus", "Nonmetro", "PCTPOVALL_2023",
                           "PCT_BACHELORS_PLUS_2019_23", "Unemployment_rate_2023", "Census_Region"],
}
res_rows = []
for name, (formula, _) in models.items():
    d = df.dropna(subset=needed[name])
    fit = smf.logit(formula, data=d).fit(disp=0)
    ci = fit.conf_int()
    for term in fit.params.index:
        if term == "Intercept" or "Census_Region" in term:
            continue  # ML region terms are not estimable (separation); see Firth model below
        res_rows.append({
            "Model": name, "N": int(fit.nobs), "Events": int(d.High_Deficit_Burden_3plus.sum()),
            "Term": term, "OR": np.exp(fit.params[term]),
            "CI_low": np.exp(ci.loc[term, 0]), "CI_high": np.exp(ci.loc[term, 1]),
            "p": fit.pvalues[term],
        })
    with open(os.path.join(OUT, f"{name.split(' (')[0].replace(' ', '_').lower()}_summary.txt"), "w") as f:
        f.write(str(fit.summary()))

# ---- Model 3, Firth penalized logistic regression -----------------------------------
# No Northeast county meets the high-burden definition in the analytic sample (quasi-complete
# separation), so standard ML cannot estimate the region contrasts. Firth's bias-reduced
# likelihood yields finite estimates; CIs are penalized profile-likelihood intervals.
def firth_logit(X, y, max_iter=200, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-X @ beta))
        W = p * (1 - p)
        XtWX = X.T @ (X * W[:, None])
        inv = np.linalg.inv(XtWX)
        h = W * np.einsum("ij,jk,ik->i", X, inv, X)
        U = X.T @ (y - p + h * (0.5 - p))
        step = inv @ U
        # step-halving on the penalized log-likelihood
        ll0 = pen_ll(X, y, beta)
        t = 1.0
        while pen_ll(X, y, beta + t * step) < ll0 - 1e-12 and t > 1e-8:
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < tol:
            break
    return beta


def pen_ll(X, y, beta):
    eta = X @ beta
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    ll = np.sum(y * eta - np.logaddexp(0, eta))
    sign, logdet = np.linalg.slogdet(X.T @ (X * W[:, None]))
    return ll + 0.5 * logdet


def firth_constrained(X, y, j, value):
    """Maximize penalized likelihood with beta_j fixed (for profile CIs)."""
    keep = [k for k in range(X.shape[1]) if k != j]
    offset = X[:, j] * value
    b = np.zeros(len(keep))
    for _ in range(200):
        full = np.insert(b, j, value)
        p = 1 / (1 + np.exp(-(X @ full)))
        W = p * (1 - p)
        inv_full = np.linalg.inv(X.T @ (X * W[:, None]))
        h = W * np.einsum("ij,jk,ik->i", X, inv_full, X)
        Xk = X[:, keep]
        U = Xk.T @ (y - p + h * (0.5 - p))
        step = np.linalg.solve(Xk.T @ (Xk * W[:, None]), U)
        ll0 = pen_ll(X, y, full)
        t = 1.0
        while pen_ll(X, y, np.insert(b + t * step, j, value)) < ll0 - 1e-12 and t > 1e-8:
            t /= 2
        b = b + t * step
        if np.max(np.abs(t * step)) < 1e-9:
            break
    return pen_ll(X, y, np.insert(b, j, value))


def profile_ci(X, y, beta, j, crit=3.841458820694124):
    from scipy.optimize import brentq
    lmax = pen_ll(X, y, beta)
    p = 1 / (1 + np.exp(-X @ beta))
    se = np.sqrt(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))[j, j])
    f = lambda v: 2 * (lmax - firth_constrained(X, y, j, v)) - crit
    out = []
    for direction in (-1, 1):
        mult = 1.5
        v = beta[j] + direction * mult * se
        while f(v) < 0 and mult < 40:
            mult *= 1.5
            v = beta[j] + direction * mult * se
        out.append(brentq(f, min(beta[j], v), max(beta[j], v), xtol=1e-7))
    return out


from scipy.stats import chi2
d3 = df.dropna(subset=needed["Model 3 (+ region)"]).copy()
X = pd.get_dummies(d3[["Nonmetro", "PCTPOVALL_2023", "PCT_BACHELORS_PLUS_2019_23",
                       "Unemployment_rate_2023", "Census_Region"]], columns=["Census_Region"],
                   drop_first=True, dtype=float)
X.insert(0, "Intercept", 1.0)
names = list(X.columns)
Xa, ya = X.to_numpy(), d3["High_Deficit_Burden_3plus"].to_numpy()
bf = firth_logit(Xa, ya)
lmax = pen_ll(Xa, ya, bf)
for j, term in enumerate(names):
    if term == "Intercept":
        continue
    lo, hi = profile_ci(Xa, ya, bf, j)
    p = chi2.sf(2 * (lmax - firth_constrained(Xa, ya, j, 0.0)), 1)
    res_rows.append({"Model": "Model 3 (+ region), Firth", "N": len(d3), "Events": int(ya.sum()),
                     "Term": term.replace("Census_Region_", "Region: "), "OR": np.exp(bf[j]),
                     "CI_low": np.exp(lo), "CI_high": np.exp(hi), "p": p})

t3 = pd.DataFrame(res_rows)
t3.to_csv(os.path.join(OUT, "table3_logistic_models.csv"), index=False, float_format="%.4f")

pd.set_option("display.width", 200)
print(t1.to_string(), "\n")
print(t2.to_string(), "\n")
print(dist.to_string(), "\n")
print(t3.round(3).to_string(index=False))
