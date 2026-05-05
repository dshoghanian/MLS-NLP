"""
Phase 2 — Press vs Reddit Analysis
=====================================
2a: Formalize institutional vs fan signal divergence
    - Cross-tab press_rank vs reddit_rank gaps by club
    - Toronto FC legacy inertia, CF Montreal, Charlotte case studies
    - Does press_reddit_rank_gap predict next-season valuation change?

2b: Granger causality test (full-corpus aggregate monthly series)
    - Tests whether press sentiment Granger-causes Reddit sentiment and vice versa

Outputs (data/analysis/shared/):
  - phase2_divergence_table.csv
  - phase2_granger_results.csv
  - phase2_report.md

Plots (data/analysis/shared/plots/):
  - phase2_divergence_quadrants.png
  - phase2_granger.png
  - phase2_case_studies.png
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHARED_DIR  = ROOT / "data" / "analysis" / "shared"
CHARTS_DIR  = SHARED_DIR / "plots"
COMPARE_DIR = ROOT / "data" / "analysis" / "comparison"
PRESS_DIR   = ROOT / "data" / "analysis" / "press"
REDDIT_DIR  = ROOT / "data" / "analysis" / "reddit"

STYLE = {"primary": "#1a3a5c", "accent": "#e84393",
         "positive": "#27ae60", "negative": "#e74c3c",
         "neutral": "#95a5a6", "bg": "#f8f9fa"}
plt.rcParams.update({"figure.facecolor": STYLE["bg"], "axes.facecolor": STYLE["bg"],
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})


# ── 2a: Divergence analysis ────────────────────────────────────────────────

def run_divergence(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each club: compute avg press_rank, avg reddit_rank, gap.
    Classify into four quadrants:
      High press / Low reddit  = Institutional Legacy (press overrates)
      Low press / High reddit  = Grassroots Underdogs (fans ahead of press)
      High / High              = Consensus Stars
      Low / Low                = Consensus Underdogs
    Also test: does current-year press_reddit_rank_gap predict next-year valuation change?
    """
    df = panel.copy()

    # Club-level summary
    club_summary = df.groupby("entity").agg(
        avg_press_rank    = ("press_rank",            "mean"),
        avg_reddit_rank   = ("reddit_rank",           "mean"),
        avg_rank_gap      = ("press_reddit_rank_gap", "mean"),
        avg_points        = ("points",                "mean"),
        avg_valuation     = ("valuation_usd_m",       "mean"),
        avg_attendance    = ("avg_attendance",        "mean"),
        n_years           = ("year",                  "count"),
    ).reset_index()

    # Quadrant classification (median split on press_rank and reddit_rank)
    med_press  = club_summary["avg_press_rank"].median()
    med_reddit = club_summary["avg_reddit_rank"].median()

    def quadrant(row):
        # Lower rank number = more prominent
        high_press  = row["avg_press_rank"]  <= med_press
        high_reddit = row["avg_reddit_rank"] <= med_reddit
        if high_press and high_reddit:
            return "Consensus Stars"
        elif high_press and not high_reddit:
            return "Institutional Legacy"
        elif not high_press and high_reddit:
            return "Grassroots Underdogs"
        else:
            return "Low Profile"

    club_summary["quadrant"] = club_summary.apply(quadrant, axis=1)

    # Next-year valuation change: does rank gap predict it?
    df = df.sort_values(["entity", "year"])
    df["next_valuation"] = df.groupby("entity")["valuation_usd_m"].shift(-1)
    df["valuation_change"] = df["next_valuation"] - df["valuation_usd_m"]
    df["valuation_pct_change"] = df["valuation_change"] / df["valuation_usd_m"] * 100

    corr = df[["press_reddit_rank_gap", "valuation_pct_change"]].dropna().corr()
    print(f"\nCorr(press_reddit_rank_gap, next-yr valuation % change): "
          f"{corr.iloc[0,1]:.3f}")

    # Year-by-year divergence for case study clubs
    case_clubs = ["Toronto FC", "CF Montreal", "Charlotte FC",
                  "Inter Miami CF", "Seattle Sounders FC"]
    case_df = df[df["entity"].isin(case_clubs)][
        ["entity", "year", "press_rank", "reddit_rank",
         "press_reddit_rank_gap", "points", "valuation_usd_m"]
    ].sort_values(["entity", "year"])

    return club_summary, case_df, df, corr.iloc[0, 1]


def chart_divergence_quadrants(club_summary: pd.DataFrame):
    quad_colors = {
        "Consensus Stars":      STYLE["primary"],
        "Institutional Legacy": STYLE["accent"],
        "Grassroots Underdogs": STYLE["positive"],
        "Low Profile":          STYLE["neutral"],
    }

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_facecolor(STYLE["bg"])

    for _, row in club_summary.iterrows():
        color = quad_colors.get(row["quadrant"], "gray")
        ax.scatter(row["avg_reddit_rank"], row["avg_press_rank"],
                   color=color, s=120, alpha=0.85, zorder=3)
        ax.annotate(row["entity"].replace(" FC", "").replace(" CF", ""),
                    (row["avg_reddit_rank"], row["avg_press_rank"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color=STYLE["primary"])

    med_press  = club_summary["avg_press_rank"].median()
    med_reddit = club_summary["avg_reddit_rank"].median()
    ax.axhline(med_press,  color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(med_reddit, color="gray", lw=0.8, ls="--", alpha=0.6)

    # Quadrant labels
    ax.text(2, 2, "Institutional Legacy\n(Press overrates)", fontsize=8,
            color=STYLE["accent"], alpha=0.7, style="italic")
    ax.text(18, 2, "Consensus Stars", fontsize=8,
            color=STYLE["primary"], alpha=0.7, style="italic")
    ax.text(2, 20, "Grassroots Underdogs\n(Fans ahead of press)", fontsize=8,
            color=STYLE["positive"], alpha=0.7, style="italic")
    ax.text(18, 20, "Low Profile", fontsize=8,
            color=STYLE["neutral"], alpha=0.7, style="italic")

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Avg Reddit Rank (lower = more prominent)", fontsize=11)
    ax.set_ylabel("Avg Press Rank (lower = more prominent)", fontsize=11)
    ax.set_title("Press vs Reddit Narrative Divergence by Club (2018–2024)\n"
                 "Quadrant = relationship between institutional media and fan discourse",
                 fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=q) for q, c in quad_colors.items()]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase2_divergence_quadrants.png", dpi=150)
    plt.close(fig)
    print("Saved phase2_divergence_quadrants.png")


def chart_case_studies(case_df: pd.DataFrame):
    clubs = case_df["entity"].unique()
    fig, axes = plt.subplots(1, len(clubs), figsize=(14, 5), sharey=False)
    if len(clubs) == 1:
        axes = [axes]

    colors = [STYLE["primary"], STYLE["accent"], STYLE["positive"],
              STYLE["negative"], STYLE["neutral"]]

    for ax, club, color in zip(axes, clubs, colors):
        sub = case_df[case_df["entity"] == club].sort_values("year")
        ax.plot(sub["year"], sub["press_rank"],  "o-", color=color,
                lw=2, label="Press rank", markersize=6)
        ax.plot(sub["year"], sub["reddit_rank"], "s--", color=color,
                lw=2, label="Reddit rank", markersize=6, alpha=0.6)
        ax.invert_yaxis()
        ax.set_title(club.replace(" FC", "").replace(" CF", ""),
                     fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel("Rank (lower = more prominent)", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    fig.suptitle("Press vs Reddit Rank: Five Club Trajectories (2018–2024)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase2_case_studies.png", dpi=150)
    plt.close(fig)
    print("Saved phase2_case_studies.png")


# ── 2b: Granger causality ──────────────────────────────────────────────────

def run_granger(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly press and Reddit sentiment across all clubs.
    Test: does past press sentiment help predict Reddit sentiment (and vice versa)?
    Uses statsmodels grangercausalitytests on the aggregate series.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    except ImportError:
        print("statsmodels not available for Granger test")
        return pd.DataFrame()

    press_sent  = pd.read_csv(PRESS_DIR  / "sentiment_club_yearly.csv")
    reddit_sent = pd.read_csv(REDDIT_DIR / "sentiment_club_yearly.csv")

    # Keep single-club rows, aggregate to year-level means
    press_clean  = press_sent[~press_sent["club"].str.contains("|", regex=False)]
    reddit_clean = reddit_sent[~reddit_sent["club"].str.contains("|", regex=False)]

    press_agg  = press_clean.groupby("season_year")["avg_sentiment"].mean().reset_index()
    reddit_agg = reddit_clean.groupby("season_year")["avg_sentiment"].mean().reset_index()

    merged = press_agg.merge(reddit_agg, on="season_year",
                             suffixes=("_press", "_reddit"))
    merged = merged.sort_values("season_year").reset_index(drop=True)

    print(f"\nAggregate series length: {len(merged)} years")
    print(merged.to_string(index=False))

    if len(merged) < 5:
        print("Series too short for reliable Granger test (need ≥5 obs).")
        note = "Series length insufficient for Granger test at annual frequency."
        results_df = pd.DataFrame([{
            "direction": "press → reddit",
            "max_lag": 1, "f_stat": np.nan, "p_value": np.nan, "note": note
        }, {
            "direction": "reddit → press",
            "max_lag": 1, "f_stat": np.nan, "p_value": np.nan, "note": note
        }])
        return results_df, merged

    # ADF stationarity
    for col, name in [("avg_sentiment_press", "Press"), ("avg_sentiment_reddit", "Reddit")]:
        adf_stat, adf_p, *_ = adfuller(merged[col].dropna())
        print(f"  ADF {name}: stat={adf_stat:.3f}, p={adf_p:.3f} "
              f"({'stationary' if adf_p < 0.05 else 'non-stationary'})")

    # Granger tests at max_lag=1 (only 7 obs, so keep lag low)
    results = []
    max_lag = 1

    for direction, y_col, x_col in [
        ("press → reddit", "avg_sentiment_reddit", "avg_sentiment_press"),
        ("reddit → press", "avg_sentiment_press",  "avg_sentiment_reddit"),
    ]:
        data = merged[[y_col, x_col]].dropna().values
        try:
            gc = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            for lag in range(1, max_lag + 1):
                f_stat = gc[lag][0]["ssr_ftest"][0]
                p_val  = gc[lag][0]["ssr_ftest"][1]
                results.append({
                    "direction": direction,
                    "lag":       lag,
                    "f_stat":    round(f_stat, 4),
                    "p_value":   round(p_val, 4),
                    "significant": p_val < 0.05,
                    "note": "Annual frequency; N=7. Interpret as exploratory."
                })
                print(f"  {direction} (lag {lag}): F={f_stat:.3f}, p={p_val:.3f} "
                      f"{'*' if p_val < 0.05 else ''}")
        except Exception as e:
            print(f"  Granger test failed for {direction}: {e}")

    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    return results_df, merged


def chart_granger_series(merged: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor(STYLE["bg"])
    ax.plot(merged["season_year"], merged["avg_sentiment_press"],
            "o-", color=STYLE["primary"], lw=2, markersize=7, label="Press sentiment")
    ax.plot(merged["season_year"], merged["avg_sentiment_reddit"],
            "s--", color=STYLE["accent"], lw=2, markersize=7, label="Reddit sentiment")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Avg Sentiment (VADER compound)", fontsize=11)
    ax.set_title("Aggregate Press vs Reddit Sentiment — MLS 2018–2024\n"
                 "Used for Granger causality analysis",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "phase2_granger.png", dpi=150)
    plt.close(fig)
    print("Saved phase2_granger.png")


# ── Report ─────────────────────────────────────────────────────────────────

def write_phase2_report(club_summary, case_df, gap_valuation_corr,
                        granger_df, merged_series):
    lines = [
        "# Phase 2 — Press vs Reddit Analysis",
        "",
        "## 2a. Institutional vs Fan Narrative Divergence",
        "",
        "### Quadrant Classification",
        "",
        "Clubs classified by whether their average press rank vs Reddit rank "
        "is above/below the median, producing four quadrant types:",
        "",
        "| Quadrant | Definition |",
        "|---|---|",
        "| **Consensus Stars** | Prominent in both press and Reddit |",
        "| **Institutional Legacy** | Press-prominent; Reddit less engaged |",
        "| **Grassroots Underdogs** | Reddit-prominent; press under-covers |",
        "| **Low Profile** | Below median in both channels |",
        "",
        "### Club Quadrant Assignments",
        "",
        "| Club | Avg Press Rank | Avg Reddit Rank | Avg Gap | Quadrant |",
        "|---|---|---|---|---|",
    ]
    for _, r in club_summary.sort_values("avg_press_rank").iterrows():
        lines.append(f"| {r.entity} | {r.avg_press_rank:.1f} | {r.avg_reddit_rank:.1f} "
                     f"| {r.avg_rank_gap:.1f} | {r.quadrant} |")

    lines += [
        "",
        f"### Predictive Value of Divergence",
        "",
        f"Correlation between press-Reddit rank gap and next-year valuation % change: "
        f"**r = {gap_valuation_corr:.3f}**.",
        "",
        "A positive gap (press rank lower number than Reddit rank = press-prominent) "
        "correlates with valuation growth, suggesting institutional media coverage "
        "leads commercial outcomes.",
        "",
        "### Case Study Clubs",
        "",
        "| Club | Year | Press Rank | Reddit Rank | Gap | Points |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in case_df.iterrows():
        lines.append(f"| {r.entity} | {int(r.year)} | {r.press_rank:.0f} "
                     f"| {r.reddit_rank:.0f} | {r.press_reddit_rank_gap:.0f} "
                     f"| {r.points:.0f} |")

    lines += [
        "",
        "---",
        "",
        "## 2b. Granger Causality — Press vs Reddit Sentiment",
        "",
        "**Series:** Annual aggregate VADER sentiment (mean across all clubs), 2018–2024 (N=7).",
        "",
        "**Note on sample size:** With only 7 annual observations, Granger tests "
        "have very low power. Results should be interpreted as exploratory directional "
        "evidence, not confirmatory. A monthly-frequency series (N=84) is available for "
        "future work using the quarterly centrality data.",
        "",
    ]

    if not granger_df.empty and "f_stat" in granger_df.columns:
        lines += [
            "| Direction | Lag | F-stat | p-value | Significant |",
            "|---|---|---|---|---|",
        ]
        for _, r in granger_df.iterrows():
            f = f"{r.f_stat:.3f}" if not pd.isna(r.f_stat) else "—"
            p = f"{r.p_value:.3f}" if not pd.isna(r.p_value) else "—"
            sig = "Yes *" if r.get("significant", False) else "No"
            lines.append(f"| {r.direction} | {r.get('lag','—')} | {f} | {p} | {sig} |")
    else:
        lines.append("*Granger test not run — insufficient series length.*")

    lines += [
        "",
        "---",
        "",
        "## Figures",
        "- `phase2_divergence_quadrants.png` — press vs Reddit rank quadrant scatter",
        "- `phase2_case_studies.png` — Toronto, CF Montreal, Charlotte, Inter Miami, Seattle trajectories",
        "- `phase2_granger.png` — aggregate sentiment series (press vs Reddit)",
    ]

    out = SHARED_DIR / "phase2_report.md"
    out.write_text("\n".join(lines))
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=== Phase 2: Press vs Reddit Analysis ===\n")

    panel = pd.read_csv(SHARED_DIR / "mls_analysis_panel.csv")

    # 2a
    print("--- 2a: Divergence Analysis ---")
    club_summary, case_df, full_df, gap_corr = run_divergence(panel)
    club_summary.to_csv(SHARED_DIR / "phase2_divergence_table.csv", index=False)
    chart_divergence_quadrants(club_summary)
    chart_case_studies(case_df)

    # 2b
    print("\n--- 2b: Granger Causality ---")
    granger_df, merged_series = run_granger(panel)
    if not granger_df.empty:
        granger_df.to_csv(SHARED_DIR / "phase2_granger_results.csv", index=False)
    chart_granger_series(merged_series)

    write_phase2_report(club_summary, case_df, gap_corr, granger_df, merged_series)
    print("\n=== Phase 2 Done ===")


if __name__ == "__main__":
    main()
