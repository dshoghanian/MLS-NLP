"""
Fixed Effects Regression — Phase 1a & 1b
==========================================

Three model families, each shown with and without fixed effects:

  Model A — PPG/Points (OLS, next-season):
    A1  Naive OLS (replicates old approach, new data)
    A2  + Club FE + Year FE
    A3  + xpoints control (external quality normalization)

  Model B — Playoff qualification (Logit, same season):
    B1  Naive logit
    B2  + Club FE (conditional logit via entity dummies)

  Model C — Financial outcomes (OLS, same season):
    C1  Valuation ~ narrative + controls
    C2  Revenue ~ narrative + controls
    C3  Sponsor deal value ~ narrative + controls

Outputs (data/analysis/shared/):
  - fe_regression_report.md          — full results narrative
  - fe_results_points.csv            — Model A coefficient tables
  - fe_results_playoff.csv           — Model B coefficient tables
  - fe_results_financial.csv         — Model C coefficient tables
  - fe_model_comparison.csv          — side-by-side fit stats

Plots (data/analysis/shared/plots/):
  - fe_coef_comparison.png           — A1 vs A2 coefficient comparison
  - fe_playoff_marginal_effects.png  — Model B marginal effects
  - fe_financial_coefs.png           — Model C coefficients
  - fe_within_r2.png                 — within-R² decomposition
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHARED_DIR  = ROOT / "data" / "analysis" / "shared"
CHARTS_DIR  = SHARED_DIR / "plots"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "primary":  "#1a3a5c",
    "accent":   "#e84393",
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
    "bg":       "#f8f9fa",
}
plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["bg"],
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

STARS = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    df = pd.read_csv(SHARED_DIR / "mls_analysis_panel.csv")

    # Next-season points (T+1 target for Model A)
    df = df.sort_values(["entity", "year"]).reset_index(drop=True)
    df["next_points"] = df.groupby("entity")["points"].shift(-1)

    # Points-per-game
    df["ppg"] = df["points"] / df["wins"].add(df["losses"]).add(df["draws"]).replace(0, np.nan)
    df["next_ppg"] = df.groupby("entity")["ppg"].shift(-1)

    # Payroll in millions (easier coefficients)
    df["payroll_m"] = df["total_payroll"] / 1_000_000

    # Valuation log (stabilises variance for financial models)
    df["log_valuation"] = np.log(df["valuation_usd_m"].replace(0, np.nan))
    df["log_revenue"]   = np.log(df["revenue_usd_m"].replace(0, np.nan))
    df["log_payroll"]   = np.log(df["total_payroll"].replace(0, np.nan))
    df["log_attendance"]= np.log(df["avg_attendance"].replace(0, np.nan))

    # Conference dummy
    df["eastern"] = (df["conference"] == "Eastern").astype(int)

    # Club and year dummies (for FE via OLS)
    club_dummies = pd.get_dummies(df["entity"], prefix="club", drop_first=True)
    year_dummies = pd.get_dummies(df["year"],   prefix="yr",   drop_first=True)
    df = pd.concat([df, club_dummies, year_dummies], axis=1)

    print(f"Panel loaded: {len(df)} rows, {df['entity'].nunique()} clubs, "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------

def ols_fit(df: pd.DataFrame, y_col: str, x_cols: list[str],
            label: str) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    sub = df[x_cols + [y_col]].dropna()
    X   = sm.add_constant(sub[x_cols].astype(float))
    y   = sub[y_col].astype(float)
    res = sm.OLS(y, X).fit(cov_type="HC1")  # heteroskedasticity-robust SEs

    coef_df = pd.DataFrame({
        "model":   label,
        "feature": res.params.index,
        "coef":    res.params.values,
        "se":      res.bse.values,
        "t":       res.tvalues.values,
        "p":       res.pvalues.values,
        "ci_lo":   res.conf_int()[0].values,
        "ci_hi":   res.conf_int()[1].values,
        "stars":   [STARS(p) for p in res.pvalues.values],
    })

    print(f"\n{'='*60}")
    print(f"  {label}  |  N={int(res.nobs)}  R²={res.rsquared:.3f}  Adj.R²={res.rsquared_adj:.3f}"
          f"  F={res.fvalue:.2f} (p={res.f_pvalue:.4f})")
    print(f"{'='*60}")
    # Print only non-FE rows to keep console tidy
    show = coef_df[~coef_df["feature"].str.startswith(("club_", "yr_"))]
    for _, r in show.iterrows():
        sig = f"  {r['stars']}" if r["stars"] else ""
        print(f"  {r['feature']:<35} {r['coef']:>8.4f}  (p={r['p']:.3f}){sig}")
    return res, coef_df


def logit_fit(df: pd.DataFrame, y_col: str, x_cols: list[str],
              label: str) -> tuple[object, pd.DataFrame]:
    sub = df[x_cols + [y_col]].dropna()
    X   = sm.add_constant(sub[x_cols].astype(float))
    y   = sub[y_col].astype(float)
    res = sm.Logit(y, X).fit(disp=False)

    # Marginal effects at mean
    me  = res.get_margeff()

    coef_df = pd.DataFrame({
        "model":   label,
        "feature": res.params.index,
        "coef":    res.params.values,
        "se":      res.bse.values,
        "z":       res.tvalues.values,
        "p":       res.pvalues.values,
        "stars":   [STARS(p) for p in res.pvalues.values],
        "marginal_effect": [np.nan] + list(me.margeff),
    })

    print(f"\n{'='*60}")
    print(f"  {label}  |  N={int(res.nobs)}  Pseudo-R²={res.prsquared:.3f}  "
          f"LLR p={res.llr_pvalue:.4f}")
    print(f"{'='*60}")
    show = coef_df[~coef_df["feature"].str.startswith(("club_", "yr_"))]
    for _, r in show.iterrows():
        sig  = f"  {r['stars']}" if r["stars"] else ""
        me_v = f"  [ME={r['marginal_effect']:.4f}]" if not np.isnan(r["marginal_effect"]) else ""
        print(f"  {r['feature']:<35} {r['coef']:>8.4f}  (p={r['p']:.3f}){sig}{me_v}")
    return res, coef_df


# ---------------------------------------------------------------------------
# Model A — Points prediction (next season)
# ---------------------------------------------------------------------------

def run_model_a(df: pd.DataFrame) -> list[pd.DataFrame]:
    club_fe = [c for c in df.columns if c.startswith("club_")]
    year_fe = [c for c in df.columns if c.startswith("yr_")]

    base_narrative = [
        "press_strength_z",       # EPL-equivalent normalized press prominence
        "press_cooc_pagerank",    # cooccurrence network centrality
        "press_net_momentum",     # net quarterly momentum (rising - falling windows)
        "press_avg_sentiment",    # VADER press sentiment
        "reddit_avg_pagerank",    # fan-side centrality
        "reddit_avg_sentiment",   # fan-side sentiment
        "made_playoffs",          # lagged performance indicator
    ]

    # A1: Naive OLS — no FE, no controls
    _, a1 = ols_fit(df.dropna(subset=["next_points"]), "next_points",
                    base_narrative, "A1: Naive OLS")

    # A2: Club FE + Year FE
    _, a2 = ols_fit(df.dropna(subset=["next_points"]), "next_points",
                    base_narrative + club_fe + year_fe,
                    "A2: Club FE + Year FE")

    # A3: FE + xpoints (external quality normalization) + payroll (financial control)
    # Drop 2020 salary nulls for this model
    df_a3 = df.dropna(subset=["next_points", "xpoints", "payroll_m"])
    _, a3 = ols_fit(df_a3, "next_points",
                    base_narrative + ["xpoints", "payroll_m", "eastern"] + club_fe + year_fe,
                    "A3: FE + xPoints + Payroll")

    return [a1, a2, a3]


# ---------------------------------------------------------------------------
# Model B — Playoff qualification (same season)
# ---------------------------------------------------------------------------

def run_model_b(df: pd.DataFrame) -> list[pd.DataFrame]:
    club_fe = [c for c in df.columns if c.startswith("club_")]
    year_fe = [c for c in df.columns if c.startswith("yr_")]

    base_narrative = [
        "press_strength_z",
        "press_cooc_pagerank",
        "press_net_momentum",
        "press_avg_sentiment",
        "reddit_avg_pagerank",
        "reddit_avg_sentiment",
    ]

    # B1: Naive logit
    _, b1 = logit_fit(df, "made_playoffs", base_narrative, "B1: Naive Logit")

    # B2: Logit + year FE (club FE drops perfectly-predicted clubs)
    _, b2 = logit_fit(df, "made_playoffs",
                      base_narrative + year_fe,
                      "B2: Logit + Year FE")

    # B3: Logit + xpoints control (separates narrative from underlying quality)
    _, b3 = logit_fit(df.dropna(subset=["xpoints"]), "made_playoffs",
                      base_narrative + ["xpoints", "eastern"] + year_fe,
                      "B3: Logit + xPoints + Year FE")

    return [b1, b2, b3]


# ---------------------------------------------------------------------------
# Model C — Financial outcomes (same season)
# ---------------------------------------------------------------------------

def run_model_c(df: pd.DataFrame) -> list[pd.DataFrame]:
    year_fe = [c for c in df.columns if c.startswith("yr_")]

    narrative_fin = [
        "press_strength_z",
        "press_to_reddit_ratio",
        "sentiment_gap",          # press sentiment minus reddit (institutional vs fan)
        "press_net_momentum",
    ]
    perf_controls = ["points", "eastern"]
    fin_controls  = ["log_attendance"]

    results = []

    # C1: Valuation ~ narrative + performance + attendance
    df_c1 = df.dropna(subset=["log_valuation", "log_attendance"])
    _, c1 = ols_fit(df_c1, "log_valuation",
                    narrative_fin + perf_controls + fin_controls + year_fe,
                    "C1: log(Valuation)")
    results.append(c1)

    # C2: Revenue
    df_c2 = df.dropna(subset=["log_revenue", "log_attendance"])
    _, c2 = ols_fit(df_c2, "log_revenue",
                    narrative_fin + perf_controls + fin_controls + year_fe,
                    "C2: log(Revenue)")
    results.append(c2)

    # C3: Sponsor deal value (only clubs with sponsor data)
    df_c3 = df.dropna(subset=["deal_value_usd_m_est", "log_attendance"])
    _, c3 = ols_fit(df_c3, "deal_value_usd_m_est",
                    narrative_fin + perf_controls + fin_controls + year_fe,
                    "C3: Sponsor Deal Value ($M)")
    results.append(c3)

    # C4: Payroll ~ narrative (does narrative prominence drive investment decisions?)
    df_c4 = df.dropna(subset=["log_payroll", "log_attendance"])
    _, c4 = ols_fit(df_c4, "log_payroll",
                    narrative_fin + perf_controls + fin_controls + year_fe,
                    "C4: log(Payroll)")
    results.append(c4)

    return results


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_coef_comparison(a_results: list[pd.DataFrame]):
    """Side-by-side coefficient plot: A1 (naive) vs A2 (with FE)."""
    key_features = [
        "press_strength_z", "press_cooc_pagerank", "press_net_momentum",
        "press_avg_sentiment", "reddit_avg_pagerank", "reddit_avg_sentiment",
        "made_playoffs",
    ]
    labels = {
        "press_strength_z":     "Press Strength (z-score)",
        "press_cooc_pagerank":  "Press PageRank (cooc.)",
        "press_net_momentum":   "Press Net Momentum",
        "press_avg_sentiment":  "Press Sentiment",
        "reddit_avg_pagerank":  "Reddit PageRank",
        "reddit_avg_sentiment": "Reddit Sentiment",
        "made_playoffs":        "Made Playoffs (T)",
        "xpoints":              "Expected Points",
        "payroll_m":            "Payroll ($M)",
        "eastern":              "Eastern Conference",
    }

    models_to_plot = [("A1: Naive OLS", STYLE["neutral"]),
                      ("A2: Club FE + Year FE", STYLE["primary"]),
                      ("A3: FE + xPoints + Payroll", STYLE["accent"])]
    n_models = len(models_to_plot)

    # Collect the rows we want
    all_rows = pd.concat(a_results, ignore_index=True)
    plot_features = [f for f in key_features + ["xpoints", "payroll_m"]
                     if f in all_rows["feature"].values]

    y_pos    = np.arange(len(plot_features))
    bar_h    = 0.25
    offsets  = np.linspace(-(n_models - 1) * bar_h / 2,
                            (n_models - 1) * bar_h / 2, n_models)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor(STYLE["bg"])

    for (mlabel, color), offset in zip(models_to_plot, offsets):
        sub = all_rows[all_rows["model"] == mlabel].set_index("feature")
        for i, feat in enumerate(plot_features):
            if feat not in sub.index:
                continue
            row = sub.loc[feat]
            bar = ax.barh(y_pos[i] + offset, row["coef"], height=bar_h,
                          color=color, alpha=0.82, label=mlabel
                          if i == 0 else "_nolegend_")
            ax.errorbar(row["coef"], y_pos[i] + offset,
                        xerr=1.96 * row["se"],
                        fmt="none", color="black", capsize=3, lw=1)
            if row["stars"]:
                ax.text(row["coef"] + (0.15 if row["coef"] >= 0 else -0.15),
                        y_pos[i] + offset, row["stars"],
                        va="center", ha="left" if row["coef"] >= 0 else "right",
                        fontsize=8, fontweight="bold")

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels.get(f, f) for f in plot_features], fontsize=10)
    ax.set_xlabel("Coefficient (effect on next-season points)", fontsize=11)
    ax.set_title("Fixed Effects vs Naive OLS: Narrative Predictors of Next-Season Points\n"
                 "(* p<0.05  ** p<0.01  *** p<0.001)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "fe_coef_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved fe_coef_comparison.png")


def chart_playoff_marginal_effects(b_results: list[pd.DataFrame]):
    labels = {
        "press_strength_z":     "Press Strength (z-score)",
        "press_cooc_pagerank":  "Press PageRank",
        "press_net_momentum":   "Press Net Momentum",
        "press_avg_sentiment":  "Press Sentiment",
        "reddit_avg_pagerank":  "Reddit PageRank",
        "reddit_avg_sentiment": "Reddit Sentiment",
        "xpoints":              "Expected Points",
        "eastern":              "Eastern Conference",
    }
    key_features = list(labels.keys())

    all_rows = pd.concat(b_results, ignore_index=True)
    # Use B3 (most controlled) for the main chart
    b3 = all_rows[all_rows["model"] == "B3: Logit + xPoints + Year FE"].copy()
    b3 = b3[b3["feature"].isin(key_features)].copy()
    b3 = b3.sort_values("marginal_effect")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor(STYLE["bg"])
    colors = [STYLE["positive"] if v > 0 else STYLE["negative"]
              for v in b3["marginal_effect"]]
    ax.barh(b3["feature"].map(labels).fillna(b3["feature"]),
            b3["marginal_effect"], color=colors, alpha=0.85, height=0.6)
    ax.errorbar(b3["marginal_effect"],
                range(len(b3)),
                xerr=1.96 * b3["se"],
                fmt="none", color="black", capsize=4, lw=1.2)
    for i, (_, row) in enumerate(b3.iterrows()):
        if row["stars"]:
            offset = 0.005 if row["marginal_effect"] >= 0 else -0.005
            ax.text(row["marginal_effect"] + offset, i, row["stars"],
                    va="center", ha="left" if row["marginal_effect"] >= 0 else "right",
                    fontsize=10, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Marginal Effect on P(Playoffs)", fontsize=11)
    ax.set_title("Logit Marginal Effects: What Drives Playoff Qualification?\n"
                 "Model B3 (Logit + xPoints + Year FE)\n(* p<0.05  ** p<0.01  *** p<0.001)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "fe_playoff_marginal_effects.png", dpi=150)
    plt.close(fig)
    print("Saved fe_playoff_marginal_effects.png")


def chart_financial_coefs(c_results: list[pd.DataFrame]):
    labels = {
        "press_strength_z":    "Press Strength (z-score)",
        "press_to_reddit_ratio": "Press/Reddit Ratio",
        "sentiment_gap":       "Sentiment Gap (Press−Reddit)",
        "press_net_momentum":  "Press Net Momentum",
        "points":              "Points",
        "eastern":             "Eastern Conference",
        "log_attendance":      "log(Attendance)",
    }
    key_features = list(labels.keys())
    model_labels = ["C1: log(Valuation)", "C2: log(Revenue)",
                    "C3: Sponsor Deal Value ($M)", "C4: log(Payroll)"]
    colors = [STYLE["primary"], STYLE["accent"], STYLE["positive"], STYLE["neutral"]]

    all_rows = pd.concat(c_results, ignore_index=True)
    narrative_only = all_rows[all_rows["feature"].isin(
        ["press_strength_z", "press_to_reddit_ratio", "sentiment_gap", "press_net_momentum"]
    )].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(STYLE["bg"])

    feat_list = ["press_strength_z", "press_to_reddit_ratio",
                 "sentiment_gap", "press_net_momentum"]
    y_pos  = np.arange(len(feat_list))
    bar_h  = 0.2
    n      = len(model_labels)
    offs   = np.linspace(-(n - 1) * bar_h / 2, (n - 1) * bar_h / 2, n)

    for (mlabel, color), offset in zip(zip(model_labels, colors), offs):
        sub = narrative_only[narrative_only["model"] == mlabel].set_index("feature")
        for i, feat in enumerate(feat_list):
            if feat not in sub.index:
                continue
            row = sub.loc[feat]
            ax.barh(y_pos[i] + offset, row["coef"], height=bar_h,
                    color=color, alpha=0.82,
                    label=mlabel if i == 0 else "_nolegend_")
            ax.errorbar(row["coef"], y_pos[i] + offset,
                        xerr=1.96 * row["se"],
                        fmt="none", color="black", capsize=3, lw=1)
            if row["stars"]:
                ax.text(row["coef"] + (0.03 if row["coef"] >= 0 else -0.03),
                        y_pos[i] + offset, row["stars"],
                        va="center", ha="left" if row["coef"] >= 0 else "right",
                        fontsize=8, fontweight="bold")

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels.get(f, f) for f in feat_list], fontsize=10)
    ax.set_xlabel("Coefficient", fontsize=11)
    ax.set_title("Narrative Centrality → Financial Outcomes\n"
                 "Narrative predictors only shown  (* p<0.05  ** p<0.01  *** p<0.001)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "fe_financial_coefs.png", dpi=150)
    plt.close(fig)
    print("Saved fe_financial_coefs.png")


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(a_res, b_res, c_res, models_meta: list[dict]) -> pd.DataFrame:
    rows = []
    for m in models_meta:
        rows.append({
            "model":      m["label"],
            "family":     m["family"],
            "n_obs":      m["n"],
            "r2":         m.get("r2", np.nan),
            "adj_r2":     m.get("adj_r2", np.nan),
            "pseudo_r2":  m.get("pseudo_r2", np.nan),
            "f_or_lr":    m.get("f_or_lr", np.nan),
            "p_val":      m.get("p_val", np.nan),
            "fe_club":    m.get("fe_club", False),
            "fe_year":    m.get("fe_year", False),
            "xpoints":    m.get("xpoints", False),
            "payroll":    m.get("payroll", False),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_fe_report(a_res, b_res, c_res):
    lines = [
        "# Fixed Effects Regression — Phase 1a & 1b",
        "",
        "## Overview",
        "",
        "Three model families test whether narrative centrality explains club outcomes "
        "after controlling for club-level heterogeneity (club fixed effects) and "
        "league-wide shocks (year fixed effects).",
        "",
        "**Core narrative predictors across all models:**",
        "- `press_strength_z` — within-year z-score of press co-occurrence edge weight "
        "(EPL-equivalent `press_strength_z`)",
        "- `press_cooc_pagerank` — PageRank in the club co-occurrence network",
        "- `press_net_momentum` — net quarterly rising/falling momentum signal",
        "- `reddit_avg_pagerank` — fan-side network centrality",
        "- Sentiment (press and Reddit separately)",
        "",
        "All OLS models use HC1 heteroskedasticity-robust standard errors.",
        "",
        "---",
        "",
        "## Model A — Next-Season Points",
        "",
        "**A1** replicates the original OLS with updated regressors (no fixed effects).",
        "**A2** adds club and year fixed effects, absorbing stable team-level differences "
        "(market size, stadium capacity, ownership quality) and league-wide year shocks "
        "(COVID 2020, Messi 2023).",
        "**A3** further adds `xpoints` (expected points from shot quality) as an external "
        "factor normalisation and `payroll` as a financial control.",
        "",
        "**Key question:** Does `press_strength_z` or `press_cooc_pagerank` remain "
        "significant after absorbing club and year heterogeneity?",
        "",
    ]

    # A model tables
    all_a = pd.concat(a_res, ignore_index=True)
    key_a = ["press_strength_z", "press_cooc_pagerank", "press_net_momentum",
             "press_avg_sentiment", "reddit_avg_pagerank", "reddit_avg_sentiment",
             "made_playoffs", "xpoints", "payroll_m", "eastern", "const"]
    label_map = {
        "press_strength_z":     "Press Strength (z-score)",
        "press_cooc_pagerank":  "Press PageRank (cooc.)",
        "press_net_momentum":   "Press Net Momentum",
        "press_avg_sentiment":  "Press Sentiment",
        "reddit_avg_pagerank":  "Reddit PageRank",
        "reddit_avg_sentiment": "Reddit Sentiment",
        "made_playoffs":        "Made Playoffs (T)",
        "xpoints":              "Expected Points",
        "payroll_m":            "Payroll ($M)",
        "eastern":              "Eastern Conference",
        "const":                "Intercept",
    }
    col_models_a = ["A1: Naive OLS", "A2: Club FE + Year FE",
                    "A3: FE + xPoints + Payroll"]

    lines += ["### Coefficient Table — Model A (Dep. var: Next-Season Points)", ""]
    header = "| Feature |" + "".join(f" {m} |" for m in col_models_a)
    sep    = "|---|" + "---|" * len(col_models_a)
    lines += [header, sep]

    for feat in key_a:
        row_vals = []
        for m in col_models_a:
            sub = all_a[(all_a["model"] == m) & (all_a["feature"] == feat)]
            if sub.empty:
                row_vals.append(" — ")
            else:
                r = sub.iloc[0]
                row_vals.append(f" {r['coef']:.3f}{r['stars']} (p={r['p']:.3f})")
        lines.append(f"| {label_map.get(feat, feat)} |" +
                     "".join(f"{v} |" for v in row_vals))

    lines += [
        "",
        "---",
        "",
        "## Model B — Playoff Qualification (Logit)",
        "",
        "Dependent variable: `made_playoffs` (binary).",
        "**B3** (most controlled) includes year FE and `xpoints` to separate "
        "narrative prominence from underlying squad quality.",
        "",
        "Marginal effects show the change in *probability* of making playoffs "
        "for a one-unit increase in each predictor.",
        "",
    ]

    # B table
    all_b = pd.concat(b_res, ignore_index=True)
    key_b = ["press_strength_z", "press_cooc_pagerank", "press_net_momentum",
             "press_avg_sentiment", "reddit_avg_pagerank", "reddit_avg_sentiment",
             "xpoints", "eastern", "const"]
    col_models_b = ["B1: Naive Logit", "B2: Logit + Year FE",
                    "B3: Logit + xPoints + Year FE"]

    lines += ["### Coefficient Table — Model B (Dep. var: Made Playoffs)", ""]
    header = "| Feature |" + "".join(f" {m} (ME) |" for m in col_models_b)
    sep    = "|---|" + "---|" * len(col_models_b)
    lines += [header, sep]

    for feat in key_b:
        row_vals = []
        for m in col_models_b:
            sub = all_b[(all_b["model"] == m) & (all_b["feature"] == feat)]
            if sub.empty:
                row_vals.append(" — ")
            else:
                r = sub.iloc[0]
                me_str = f" / ME={r['marginal_effect']:.3f}" if not np.isnan(r.get("marginal_effect", np.nan)) else ""
                row_vals.append(f" {r['coef']:.3f}{r['stars']}{me_str}")
        lines.append(f"| {label_map.get(feat, feat)} |" +
                     "".join(f"{v} |" for v in row_vals))

    lines += [
        "",
        "---",
        "",
        "## Model C — Financial Outcomes",
        "",
        "Tests whether narrative centrality independently drives commercial outcomes "
        "after controlling for on-field performance and market size.",
        "",
        "Dependent variables (all log-transformed except C3):",
        "- **C1** log(Valuation)",
        "- **C2** log(Revenue)",
        "- **C3** Sponsor deal value ($M)",
        "- **C4** log(Payroll)",
        "",
        "**Key predictor:** `press_strength_z` — if significant in C1/C2 after "
        "controlling for points and attendance, narrative has *independent* commercial value.",
        "",
    ]

    all_c = pd.concat(c_res, ignore_index=True)
    key_c = ["press_strength_z", "press_to_reddit_ratio", "sentiment_gap",
             "press_net_momentum", "points", "eastern", "log_attendance", "const"]
    col_models_c = ["C1: log(Valuation)", "C2: log(Revenue)",
                    "C3: Sponsor Deal Value ($M)", "C4: log(Payroll)"]

    lines += ["### Coefficient Table — Model C (Financial Outcomes)", ""]
    header = "| Feature |" + "".join(f" {m} |" for m in col_models_c)
    sep    = "|---|" + "---|" * len(col_models_c)
    lines += [header, sep]

    for feat in key_c:
        row_vals = []
        for m in col_models_c:
            sub = all_c[(all_c["model"] == m) & (all_c["feature"] == feat)]
            if sub.empty:
                row_vals.append(" — ")
            else:
                r = sub.iloc[0]
                row_vals.append(f" {r['coef']:.3f}{r['stars']} (p={r['p']:.3f})")
        lines.append(f"| {label_map.get(feat, feat)} |" +
                     "".join(f"{v} |" for v in row_vals))

    lines += [
        "",
        "---",
        "",
        "## Figures",
        "- `fe_coef_comparison.png` — A1 vs A2 vs A3 coefficient comparison",
        "- `fe_playoff_marginal_effects.png` — Model B3 marginal effects on P(playoff)",
        "- `fe_financial_coefs.png` — Model C narrative predictors across financial outcomes",
        "",
    ]

    out = SHARED_DIR / "fe_regression_report.md"
    out.write_text("\n".join(lines))
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------

def save_csvs(a_res, b_res, c_res):
    pd.concat(a_res, ignore_index=True).to_csv(
        SHARED_DIR / "fe_results_points.csv", index=False)
    pd.concat(b_res, ignore_index=True).to_csv(
        SHARED_DIR / "fe_results_playoff.csv", index=False)
    pd.concat(c_res, ignore_index=True).to_csv(
        SHARED_DIR / "fe_results_financial.csv", index=False)
    print("Saved fe_results_*.csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Fixed Effects Regression ===\n")
    df = load_panel()

    print("\n--- Model A: Next-Season Points ---")
    a_res = run_model_a(df)

    print("\n--- Model B: Playoff Qualification (Logit) ---")
    b_res = run_model_b(df)

    print("\n--- Model C: Financial Outcomes ---")
    c_res = run_model_c(df)

    print("\n--- Charts ---")
    chart_coef_comparison(a_res)
    chart_playoff_marginal_effects(b_res)
    chart_financial_coefs(c_res)

    save_csvs(a_res, b_res, c_res)
    write_fe_report(a_res, b_res, c_res)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
