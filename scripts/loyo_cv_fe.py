"""
Leave-One-Year-Out CV for Fixed Effects Models
================================================
Robustness check for Models A (points), B (playoffs), C (financial).
For FE models we retrain WITH club/year dummies each fold.

Outputs (data/analysis/shared/):
  - fe_loyo_points.csv
  - fe_loyo_playoff.csv
  - fe_loyo_financial.csv
  - fe_loyo_report.md

Plots (data/analysis/shared/plots/):
  - fe_loyo_points.png
  - fe_loyo_playoff.png
  - fe_loyo_financial.png
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHARED_DIR = ROOT / "data" / "analysis" / "shared"
CHARTS_DIR = SHARED_DIR / "plots"

STYLE = {"primary": "#1a3a5c", "accent": "#e84393",
         "positive": "#27ae60", "negative": "#e74c3c", "bg": "#f8f9fa"}
plt.rcParams.update({"figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["bg"],
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(SHARED_DIR / "mls_analysis_panel.csv")
    df = df.sort_values(["entity", "year"]).reset_index(drop=True)
    df["next_points"]    = df.groupby("entity")["points"].shift(-1)
    df["payroll_m"]      = df["total_payroll"] / 1_000_000
    df["log_valuation"]  = np.log(df["valuation_usd_m"].replace(0, np.nan))
    df["log_revenue"]    = np.log(df["revenue_usd_m"].replace(0, np.nan))
    df["log_attendance"] = np.log(df["avg_attendance"].replace(0, np.nan))
    df["eastern"]        = (df["conference"] == "Eastern").astype(int)
    return df


# ── generic LOYO helper ────────────────────────────────────────────────────

def loyo_ols(df: pd.DataFrame, y_col: str, narrative_cols: list[str],
             extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Train with club+year dummies, leaving one year out at a time."""
    extra_cols = extra_cols or []
    years  = sorted(df["year"].dropna().unique())
    rows   = []

    for held in years:
        train = df[df["year"] != held].copy()
        test  = df[df["year"] == held].copy()

        # Build dummies from training set only
        club_d = pd.get_dummies(train["entity"], prefix="club", drop_first=True)
        year_d = pd.get_dummies(train["year"],   prefix="yr",   drop_first=True)
        train_X_full = pd.concat([train[narrative_cols + extra_cols], club_d, year_d], axis=1)

        # Test: align to training columns
        test_club_d = pd.get_dummies(test["entity"], prefix="club")
        test_year_d = pd.get_dummies(test["year"],   prefix="yr")
        test_X_raw  = pd.concat([test[narrative_cols + extra_cols],
                                  test_club_d, test_year_d], axis=1)
        test_X_full = test_X_raw.reindex(columns=train_X_full.columns, fill_value=0)

        sub_train = pd.concat([train_X_full, train[[y_col]]], axis=1).dropna()
        sub_test  = pd.concat([test_X_full,  test[[y_col]]], axis=1).dropna()
        if len(sub_train) < 10 or len(sub_test) == 0:
            continue

        X_tr = sm.add_constant(sub_train[train_X_full.columns].astype(float))
        y_tr = sub_train[y_col].astype(float)
        X_te = sm.add_constant(sub_test[train_X_full.columns].astype(float))
        y_te = sub_test[y_col].astype(float)

        # align constant
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

        res    = sm.OLS(y_tr, X_tr).fit()
        y_pred = res.predict(X_te)

        r2   = r2_score(y_te, y_pred)
        mae  = mean_absolute_error(y_te, y_pred)
        rmse = float(np.sqrt(np.mean((y_te.values - y_pred.values) ** 2)))
        rows.append({"held_out_year": held, "n_test": len(y_te),
                     "r2": round(r2, 4), "mae": round(mae, 3), "rmse": round(rmse, 3)})

    return pd.DataFrame(rows)


def loyo_logit(df: pd.DataFrame, y_col: str, narrative_cols: list[str],
               extra_cols: list[str] | None = None) -> pd.DataFrame:
    """LOYO for logit — returns accuracy and AUC."""
    extra_cols = extra_cols or []
    years  = sorted(df["year"].dropna().unique())
    rows   = []

    for held in years:
        train = df[df["year"] != held].copy()
        test  = df[df["year"] == held].copy()

        year_d_tr = pd.get_dummies(train["year"], prefix="yr", drop_first=True)
        train_X   = pd.concat([train[narrative_cols + extra_cols], year_d_tr], axis=1)

        test_year_d = pd.get_dummies(test["year"], prefix="yr")
        test_X_raw  = pd.concat([test[narrative_cols + extra_cols], test_year_d], axis=1)
        test_X      = test_X_raw.reindex(columns=train_X.columns, fill_value=0)

        sub_tr = pd.concat([train_X, train[[y_col]]], axis=1).dropna()
        sub_te = pd.concat([test_X,  test[[y_col]]],  axis=1).dropna()
        if len(sub_tr) < 10 or len(sub_te) == 0:
            continue

        X_tr = sm.add_constant(sub_tr[train_X.columns].astype(float))
        y_tr = sub_tr[y_col].astype(float)
        X_te = sm.add_constant(sub_te[train_X.columns].astype(float))
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
        y_te = sub_te[y_col].astype(float)

        try:
            mdl    = sm.Logit(y_tr, X_tr).fit(disp=False, maxiter=200)
            probs  = mdl.predict(X_te)
            preds  = (probs >= 0.5).astype(int)
            acc    = accuracy_score(y_te, preds)
            auc    = roc_auc_score(y_te, probs) if y_te.nunique() > 1 else np.nan
            rows.append({"held_out_year": held, "n_test": len(y_te),
                         "accuracy": round(acc, 4), "auc": round(auc, 4)})
        except Exception:
            pass

    return pd.DataFrame(rows)


# ── charts ─────────────────────────────────────────────────────────────────

def chart_loyo(cv_df: pd.DataFrame, metric: str, title: str, fname: str,
               color: str = "#1a3a5c", zero_line: bool = True):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(STYLE["bg"])
    years = cv_df["held_out_year"].astype(str)
    ax.bar(years, cv_df[metric], color=color, alpha=0.82, edgecolor="white")
    mean_val = cv_df[metric].mean()
    ax.axhline(mean_val, color=STYLE["accent"], lw=2, ls="--",
               label=f"Mean={mean_val:.3f}")
    if zero_line:
        ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Held-Out Year", fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("=== LOYO CV for Fixed Effects Models ===\n")
    df = load_panel()

    narrative_base = [
        "press_strength_z", "press_cooc_pagerank", "press_net_momentum",
        "press_avg_sentiment", "reddit_avg_pagerank", "reddit_avg_sentiment",
    ]

    # ── Model A: next-season points ──────────────────────────────────────
    print("--- Model A: Next-Season Points ---")
    df_a = df.dropna(subset=["next_points"] + narrative_base)
    cv_a = loyo_ols(df_a, "next_points", narrative_base, ["made_playoffs"])
    cv_a.to_csv(SHARED_DIR / "fe_loyo_points.csv", index=False)
    print(cv_a.to_string(index=False))
    print(f"  Mean R²={cv_a['r2'].mean():.3f}  MAE={cv_a['mae'].mean():.2f}pts\n")
    chart_loyo(cv_a, "r2", "LOYO CV — Next-Season Points (FE Model)\nR² per Held-Out Year",
               "fe_loyo_points.png")

    # ── Model B: playoff qualification ──────────────────────────────────
    print("--- Model B: Playoff Qualification ---")
    df_b = df.dropna(subset=["made_playoffs"] + narrative_base + ["xpoints"])
    cv_b = loyo_logit(df_b, "made_playoffs", narrative_base, ["xpoints", "eastern"])
    cv_b.to_csv(SHARED_DIR / "fe_loyo_playoff.csv", index=False)
    print(cv_b.to_string(index=False))
    if not cv_b.empty:
        print(f"  Mean Accuracy={cv_b['accuracy'].mean():.3f}  AUC={cv_b['auc'].mean():.3f}\n")
    chart_loyo(cv_b, "auc", "LOYO CV — Playoff Qualification (Logit + Year FE)\nAUC per Held-Out Year",
               "fe_loyo_playoff.png", color=STYLE["accent"], zero_line=False)

    # ── Model C: financial outcomes ──────────────────────────────────────
    print("--- Model C: Financial Outcomes (Valuation) ---")
    fin_narrative = ["press_strength_z", "press_to_reddit_ratio",
                     "sentiment_gap", "press_net_momentum"]
    df_c = df.dropna(subset=["log_valuation", "log_attendance"] + fin_narrative)
    cv_c = loyo_ols(df_c, "log_valuation", fin_narrative,
                    ["points", "eastern", "log_attendance"])
    cv_c.to_csv(SHARED_DIR / "fe_loyo_financial.csv", index=False)
    print(cv_c.to_string(index=False))
    print(f"  Mean R²={cv_c['r2'].mean():.3f}  MAE={cv_c['mae'].mean():.3f}\n")
    chart_loyo(cv_c, "r2", "LOYO CV — log(Valuation) Model\nR² per Held-Out Year",
               "fe_loyo_financial.png", color=STYLE["positive"])

    # ── Markdown summary ────────────────────────────────────────────────
    lines = [
        "# LOYO Cross-Validation — Fixed Effects Models", "",
        "## Model A: Next-Season Points", "",
        "| Held-Out Year | N | R² | MAE | RMSE |",
        "|---|---|---|---|---|",
    ]
    for _, r in cv_a.iterrows():
        lines.append(f"| {int(r.held_out_year)} | {int(r.n_test)} | {r.r2:.3f} | {r.mae:.2f} | {r.rmse:.2f} |")
    lines += [f"| **Mean** | — | **{cv_a.r2.mean():.3f}** | **{cv_a.mae.mean():.2f}** | **{cv_a.rmse.mean():.2f}** |", ""]

    lines += ["## Model B: Playoff Qualification", "",
              "| Held-Out Year | N | Accuracy | AUC |",
              "|---|---|---|---|"]
    for _, r in cv_b.iterrows():
        lines.append(f"| {int(r.held_out_year)} | {int(r.n_test)} | {r.accuracy:.3f} | {r.auc:.3f} |")
    if not cv_b.empty:
        lines += [f"| **Mean** | — | **{cv_b.accuracy.mean():.3f}** | **{cv_b.auc.mean():.3f}** |", ""]

    lines += ["## Model C: log(Valuation)", "",
              "| Held-Out Year | N | R² | MAE | RMSE |",
              "|---|---|---|---|---|"]
    for _, r in cv_c.iterrows():
        lines.append(f"| {int(r.held_out_year)} | {int(r.n_test)} | {r.r2:.3f} | {r.mae:.3f} | {r.rmse:.3f} |")
    lines += [f"| **Mean** | — | **{cv_c.r2.mean():.3f}** | **{cv_c.mae.mean():.3f}** | **{cv_c.rmse.mean():.3f}** |"]

    (SHARED_DIR / "fe_loyo_report.md").write_text("\n".join(lines))
    print("Saved fe_loyo_report.md")


if __name__ == "__main__":
    main()
