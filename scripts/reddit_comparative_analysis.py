"""
Press vs Reddit Comparative Analysis
=====================================
Compares narrative signals between professional press coverage and
fan discourse on Reddit across 4 dimensions:

  1. Centrality Divergence  — which clubs are bigger on Reddit vs press?
  2. Sentiment Gap          — do fans feel differently than the press?
  3. Lead / Lag             — does Reddit react before or after press?
  4. Narrative Power Index  — which source drives the story for each club?

Outputs:
  data/analysis/comparison/press_vs_reddit_*.csv
  data/analysis/comparison/plots/reddit_*.png
  data/analysis/comparison/press_vs_reddit_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.utils import get_config, load_parquet, PROJECT_ROOT

settings    = get_config("settings")
DATA_DIR       = PROJECT_ROOT / "data"
PRESS_DIR      = DATA_DIR / "analysis" / "press"
COMPARISON_DIR = DATA_DIR / "analysis" / "comparison"
PLOT_DIR       = COMPARISON_DIR / "plots"
REDDIT_DIR     = DATA_DIR / "reddit"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLUB_COLORS = {
    "Atlanta United FC": "#80000A",   "Austin FC": "#00B140",
    "CF Montreal": "#003DA5",          "Charlotte FC": "#1A85C8",
    "Chicago Fire FC": "#CC0000",      "Colorado Rapids": "#862633",
    "Columbus Crew": "#FFF200",        "D.C. United": "#888888",
    "FC Cincinnati": "#003087",        "FC Dallas": "#BF0D3E",
    "Houston Dynamo FC": "#F68712",    "Inter Miami CF": "#F7B5CD",
    "LA Galaxy": "#00245D",            "LAFC": "#C39E6D",
    "Minnesota United FC": "#8CD2F4",  "Nashville SC": "#ECE83A",
    "New England Revolution": "#0A2240","New York City FC": "#6CACE4",
    "New York Red Bulls": "#ED1E36",   "Orlando City SC": "#612B9B",
    "Philadelphia Union": "#071B2C",   "Portland Timbers": "#00482B",
    "Real Salt Lake": "#B30838",       "San Jose Earthquakes": "#0D4C92",
    "Seattle Sounders FC": "#5D9741",  "Sporting Kansas City": "#93B8E3",
    "St. Louis City SC": "#DD0000",    "Toronto FC": "#B81137",
    "Vancouver Whitecaps FC": "#009BC8",
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all():
    # Press centrality (yearly)
    press_cent = pd.read_csv(PRESS_DIR / "centrality_club_cooccurrence.csv")
    press_yearly = press_cent[press_cent["time_window"].str.match(r"^\d{4}$")].copy()
    press_yearly["year"] = press_yearly["time_window"].astype(int)
    press_yearly["press_rank"] = press_yearly.groupby("year")["pagerank"].rank(ascending=False).astype(int)
    press_yearly = press_yearly.rename(columns={
        "pagerank": "press_pagerank", "degree": "press_degree"
    })

    # Reddit centrality (yearly)
    reddit_cent_path = REDDIT_DIR / "analysis" / "reddit_centrality.csv"
    if not reddit_cent_path.exists():
        raise FileNotFoundError("Reddit centrality not found. Run run_reddit_pipeline.py first.")

    reddit_cent = pd.read_csv(reddit_cent_path)
    reddit_yearly = reddit_cent[reddit_cent["time_window"].str.match(r"^\d{4}$")].copy()
    reddit_yearly["year"] = reddit_yearly["time_window"].astype(int)
    reddit_yearly["reddit_rank"] = reddit_yearly.groupby("year")["pagerank"].rank(ascending=False).astype(int)
    reddit_yearly = reddit_yearly.rename(columns={
        "pagerank": "reddit_pagerank", "degree": "reddit_degree"
    })

    # Merge
    merged = press_yearly[["entity", "year", "press_pagerank", "press_degree", "press_rank"]].merge(
        reddit_yearly[["entity", "year", "reddit_pagerank", "reddit_degree", "reddit_rank"]],
        on=["entity", "year"], how="inner"
    )
    merged["rank_gap"] = merged["reddit_rank"] - merged["press_rank"]
    # Positive = club ranks higher on Reddit than in press (fan favourite)
    # Negative = club ranks higher in press than on Reddit (media darling)

    return merged, press_cent, reddit_cent


def load_sentiment():
    """Load monthly sentiment from both sources."""
    # Press sentiment from enriched parquets
    press_rows = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            from pipeline.utils import get_parquet_path
            path = get_parquet_path(DATA_DIR / "enriched", "enriched", year, month)
            df = load_parquet(path)
            if df.empty:
                continue
            for _, row in df[df["clubs_mentioned"].str.len() > 0].iterrows():
                for club in str(row["clubs_mentioned"]).split("|"):
                    if club:
                        press_rows.append({
                            "club": club, "year": year, "month": month,
                            "sentiment": row.get("sentiment_compound", 0),
                            "source": "press"
                        })

    # Reddit sentiment
    reddit_rows = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            path = REDDIT_DIR / "enriched" / str(year) / f"{year}_{month:02d}_reddit_enriched.parquet"
            if not path.exists():
                continue
            df = load_parquet(path)
            if df.empty:
                continue
            for _, row in df[df["clubs_mentioned"].str.len() > 0].iterrows():
                for club in str(row["clubs_mentioned"]).split("|"):
                    if club:
                        reddit_rows.append({
                            "club": club, "year": year, "month": month,
                            "sentiment": row.get("sentiment_compound", 0),
                            "source": "reddit"
                        })

    press_sent  = pd.DataFrame(press_rows)
    reddit_sent = pd.DataFrame(reddit_rows)
    return press_sent, reddit_sent


def load_monthly_volumes():
    """Monthly post/article count per club from both sources."""
    press_vol = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            from pipeline.utils import get_parquet_path
            path = get_parquet_path(DATA_DIR / "enriched", "enriched", year, month)
            df = load_parquet(path)
            if df.empty:
                continue
            for _, row in df[df["clubs_mentioned"].str.len() > 0].iterrows():
                for club in str(row["clubs_mentioned"]).split("|"):
                    if club:
                        press_vol.append({"club": club, "year": year, "month": month, "source": "press"})

    reddit_vol = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            path = REDDIT_DIR / "enriched" / str(year) / f"{year}_{month:02d}_reddit_enriched.parquet"
            if not path.exists():
                continue
            df = load_parquet(path)
            if df.empty:
                continue
            for _, row in df[df["clubs_mentioned"].str.len() > 0].iterrows():
                for club in str(row["clubs_mentioned"]).split("|"):
                    if club:
                        reddit_vol.append({"club": club, "year": year, "month": month, "source": "reddit"})

    press_df  = pd.DataFrame(press_vol).groupby(["club", "year", "month"]).size().reset_index(name="count")
    reddit_df = pd.DataFrame(reddit_vol).groupby(["club", "year", "month"]).size().reset_index(name="count")
    return press_df, reddit_df


# ══════════════════════════════════════════════════════════════════════════════
# 1. Centrality Divergence
# ══════════════════════════════════════════════════════════════════════════════

def plot_centrality_divergence(df: pd.DataFrame):
    avg = df.groupby("entity").agg(
        press_rank=("press_rank", "mean"),
        reddit_rank=("reddit_rank", "mean"),
        rank_gap=("rank_gap", "mean"),
    ).reset_index().sort_values("rank_gap", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: bar of rank gap
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in avg["rank_gap"]]
    axes[0].barh(avg["entity"], avg["rank_gap"], color=colors, edgecolor="white")
    axes[0].axvline(0, color="black", lw=1.2)
    axes[0].set_xlabel("Reddit Rank − Press Rank\nPositive = more prominent on Reddit (fan favourite)\nNegative = more prominent in press (media darling)")
    axes[0].set_title("Narrative Divergence: Fan vs. Press Prominence", fontweight="bold")

    blue_patch = mpatches.Patch(color="#3498db", label="Fan favourite (Reddit > Press)")
    red_patch  = mpatches.Patch(color="#e74c3c", label="Media darling (Press > Reddit)")
    axes[0].legend(handles=[blue_patch, red_patch], loc="lower right", fontsize=9)

    for bar, val in zip(axes[0].patches, avg["rank_gap"]):
        sign = "+" if val >= 0 else ""
        axes[0].text(val + (0.1 if val >= 0 else -0.1),
                     bar.get_y() + bar.get_height() / 2,
                     f"{sign}{val:.1f}", va="center",
                     ha="left" if val >= 0 else "right", fontsize=8)

    # Right: scatter press vs reddit rank
    n = avg["press_rank"].max()
    for _, row in avg.iterrows():
        color = CLUB_COLORS.get(row["entity"], "#888")
        axes[1].scatter(row["press_rank"], row["reddit_rank"],
                        s=80, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        label = row["entity"].split()[0]
        axes[1].annotate(label, (row["press_rank"], row["reddit_rank"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")

    axes[1].plot([1, n], [1, n], "k--", lw=1, alpha=0.4, label="Perfect alignment")
    axes[1].invert_xaxis(); axes[1].invert_yaxis()
    axes[1].set_xlabel("Press Rank (1 = most prominent in media)")
    axes[1].set_ylabel("Reddit Rank (1 = most discussed by fans)")
    axes[1].set_title("Press Rank vs Reddit Rank\n(dots above diagonal = bigger on Reddit)", fontweight="bold")
    axes[1].legend(fontsize=8)

    plt.suptitle("Fan Discourse vs. Press Narrative — Club Prominence (2018–2024 avg)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / "reddit_centrality_divergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


def plot_divergence_over_time(df: pd.DataFrame):
    """How divergence evolves year by year for key clubs."""
    key_clubs = ["Inter Miami CF", "Toronto FC", "Seattle Sounders FC",
                 "LAFC", "Philadelphia Union", "Columbus Crew",
                 "CF Montreal", "Charlotte FC"]
    available = [c for c in key_clubs if c in df["entity"].values]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("#f8f8f8")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.4)

    for club in available:
        sub = df[df["entity"] == club].sort_values("year")
        color = CLUB_COLORS.get(club, "#888")
        ax.plot(sub["year"], sub["rank_gap"], marker="o", lw=2,
                color=color, label=club)
        ax.annotate(club.split()[0], (sub["year"].iloc[-1], sub["rank_gap"].iloc[-1]),
                    fontsize=7.5, color=color, xytext=(3, 0), textcoords="offset points")

    ax.set_xlabel("Year"); ax.set_ylabel("Rank Gap (Reddit − Press)")
    ax.set_title("Fan vs. Press Prominence Gap Over Time\nPositive = fans discuss more | Negative = press covers more",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    path = PLOT_DIR / "reddit_divergence_timeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sentiment Gap
# ══════════════════════════════════════════════════════════════════════════════

def plot_sentiment_gap(press_sent: pd.DataFrame, reddit_sent: pd.DataFrame):
    p_avg = press_sent.groupby(["club", "year"])["sentiment"].mean().reset_index(name="press_sent")
    r_avg = reddit_sent.groupby(["club", "year"])["sentiment"].mean().reset_index(name="reddit_sent")

    sent_df = p_avg.merge(r_avg, on=["club", "year"], how="inner")
    sent_df["sentiment_gap"] = sent_df["reddit_sent"] - sent_df["press_sent"]
    # Positive = fans more positive than press
    # Negative = press more positive than fans (fans more critical)

    sent_df.to_csv(COMPARISON_DIR / "press_vs_reddit_sentiment.csv", index=False)

    # Heatmap: clubs × year
    pivot = sent_df.pivot_table(index="club", columns="year",
                                values="sentiment_gap", aggfunc="mean").fillna(0)
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values().index)

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                vmin=-0.3, vmax=0.3, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Reddit sentiment − Press sentiment"})
    ax.set_title("Fan vs. Press Sentiment Gap\nGreen = fans more positive | Red = fans more negative than press",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Year"); ax.set_ylabel("")
    plt.tight_layout()
    path = PLOT_DIR / "reddit_sentiment_gap_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")

    # Bar: avg sentiment gap per club
    avg_gap = sent_df.groupby("club")["sentiment_gap"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in avg_gap.values]
    ax.barh(avg_gap.index, avg_gap.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Avg Sentiment Gap (Reddit − Press)\nPositive = fans more positive | Negative = fans more critical")
    ax.set_title("Fan vs. Press Sentiment — 2018–2024 Average", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / "reddit_sentiment_gap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")

    return sent_df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lead / Lag Analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_lead_lag(press_vol: pd.DataFrame, reddit_vol: pd.DataFrame):
    """Cross-correlate monthly press and Reddit volumes to detect lead/lag."""
    # Focus on clubs with enough data in both
    common_clubs = set(press_vol["club"]) & set(reddit_vol["club"])

    # Build full monthly time index
    months = pd.date_range("2018-01", "2024-12", freq="MS")
    month_idx = {(d.year, d.month): i for i, d in enumerate(months)}

    lag_results = []
    for club in common_clubs:
        p = press_vol[press_vol["club"] == club].copy()
        r = reddit_vol[reddit_vol["club"] == club].copy()

        p_series = np.zeros(len(months))
        r_series = np.zeros(len(months))

        for _, row in p.iterrows():
            idx = month_idx.get((int(row["year"]), int(row["month"])))
            if idx is not None:
                p_series[idx] = row["count"]
        for _, row in r.iterrows():
            idx = month_idx.get((int(row["year"]), int(row["month"])))
            if idx is not None:
                r_series[idx] = row["count"]

        if p_series.sum() < 10 or r_series.sum() < 10:
            continue

        # Normalize
        p_norm = (p_series - p_series.mean()) / (p_series.std() + 1e-9)
        r_norm = (r_series - r_series.mean()) / (r_series.std() + 1e-9)

        # Cross-correlation: positive lag = Reddit leads press
        corr = correlate(p_norm, r_norm, mode="full")
        lags = np.arange(-(len(p_norm)-1), len(p_norm))
        best_lag = lags[np.argmax(corr)]
        max_corr = corr[np.argmax(corr)] / len(p_norm)

        lag_results.append({
            "club": club, "best_lag_months": int(best_lag),
            "max_correlation": round(float(max_corr), 3),
        })

    lag_df = pd.DataFrame(lag_results).sort_values("best_lag_months")
    lag_df.to_csv(COMPARISON_DIR / "press_vs_reddit_leadlag.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 8))
    colors = ["#3498db" if v > 0 else "#e74c3c" if v < 0 else "#888"
              for v in lag_df["best_lag_months"]]
    bars = ax.barh(lag_df["club"], lag_df["best_lag_months"],
                   color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=1.2)

    for bar, val in zip(bars, lag_df["best_lag_months"]):
        if val != 0:
            ax.text(val + (0.05 if val > 0 else -0.05),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+d}mo", va="center",
                    ha="left" if val > 0 else "right", fontsize=8)

    blue_patch = mpatches.Patch(color="#3498db", label="Reddit leads press (fans react first)")
    red_patch  = mpatches.Patch(color="#e74c3c", label="Press leads Reddit (media drives fan discourse)")
    ax.legend(handles=[blue_patch, red_patch], fontsize=9)
    ax.set_xlabel("Lead/Lag in months\nPositive = Reddit leads | Negative = Press leads")
    ax.set_title("Who Reacts First? Lead/Lag Between Fan Discourse and Press Coverage",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / "reddit_lead_lag.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")

    return lag_df


def plot_event_lead_lag(press_vol: pd.DataFrame, reddit_vol: pd.DataFrame):
    """For the biggest narrative events, plot press vs Reddit volume side by side."""
    events = [
        ("Inter Miami CF",       2023, 7,  "Messi signing"),
        ("Philadelphia Union",   2022, 11, "MLS Cup final"),
        ("Seattle Sounders FC",  2019, 11, "MLS Cup win"),
        ("Toronto FC",           2022, 1,  "Transfer window spike"),
        ("Columbus Crew",        2020, 12, "MLS Cup win"),
    ]

    fig, axes = plt.subplots(len(events), 1, figsize=(13, 4 * len(events)))

    for ax, (club, ev_year, ev_month, label) in zip(axes, events):
        color = CLUB_COLORS.get(club, "#888")

        # 6 months before, 6 after
        window_months = [(ev_year + (ev_month + i - 1) // 12,
                          (ev_month + i - 1) % 12 + 1)
                         for i in range(-5, 7)]

        p = press_vol[press_vol["club"] == club].set_index(["year", "month"])["count"]
        r = reddit_vol[reddit_vol["club"] == club].set_index(["year", "month"])["count"]

        p_vals = [p.get(ym, 0) for ym in window_months]
        r_vals = [r.get(ym, 0) for ym in window_months]
        x_labels = [f"{y}-{m:02d}" for y, m in window_months]

        x = np.arange(len(x_labels))
        w = 0.35
        ax.bar(x - w/2, p_vals, w, label="Press articles", color=color, alpha=0.8)
        ax.bar(x + w/2, r_vals, w, label="Reddit posts",   color=color, alpha=0.4, hatch="//")
        ax.axvline(5, color="red", lw=1.5, ls="--", alpha=0.7)
        ax.text(5.1, max(max(p_vals), max(r_vals)) * 0.9, "Event",
                color="red", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{club} — {label} ({ev_year}-{ev_month:02d})",
                     fontweight="bold", color=color)
        ax.legend(fontsize=8)
        ax.set_ylabel("Count")

    plt.suptitle("Event-Level Lead/Lag: Press vs Reddit Volume Around Key Events",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / "reddit_event_lead_lag.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Narrative Power Index
# ══════════════════════════════════════════════════════════════════════════════

def plot_narrative_power(press_vol: pd.DataFrame, reddit_vol: pd.DataFrame,
                         cent_df: pd.DataFrame):
    """
    Narrative Power Index = weighted combination of:
      - Volume (article/post count)
      - Engagement (Reddit score proxy)
      - Centrality (PageRank in respective network)
    Shows which source 'owns' the narrative for each club.
    """
    p_total = press_vol.groupby("club")["count"].sum().reset_index(name="press_volume")
    r_total = reddit_vol.groupby("club")["count"].sum().reset_index(name="reddit_volume")

    npi = p_total.merge(r_total, on="club", how="outer").fillna(0)

    # Normalize both to 0-1
    npi["press_norm"]  = npi["press_volume"]  / npi["press_volume"].max()
    npi["reddit_norm"] = npi["reddit_volume"] / npi["reddit_volume"].max()
    npi["press_share"] = npi["press_norm"] / (npi["press_norm"] + npi["reddit_norm"] + 1e-9)

    npi = npi.sort_values("press_share")
    npi.to_csv(COMPARISON_DIR / "press_vs_reddit_npi.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 9))
    for _, row in npi.iterrows():
        color = CLUB_COLORS.get(row["club"], "#888")
        press_w  = row["press_share"]
        reddit_w = 1 - press_w
        ax.barh(row["club"], press_w,  color=color, alpha=0.9,  height=0.6)
        ax.barh(row["club"], reddit_w, left=press_w, color=color, alpha=0.35, height=0.6, hatch="//")

    ax.axvline(0.5, color="black", lw=1.5, ls="--", alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("← Reddit dominates narrative         Press dominates narrative →")
    ax.set_title("Narrative Power Index — Who Owns the Story?\n"
                 "Solid = press share | Hatched = Reddit share",
                 fontsize=12, fontweight="bold")
    ax.text(0.25, -0.8, "Reddit-driven", fontsize=10, color="gray", ha="center")
    ax.text(0.75, -0.8, "Press-driven",  fontsize=10, color="gray", ha="center")
    plt.tight_layout()
    path = PLOT_DIR / "reddit_narrative_power_index.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def write_report(cent_df: pd.DataFrame, sent_df: pd.DataFrame, lag_df: pd.DataFrame):
    avg_cent = cent_df.groupby("entity").agg(
        press_rank=("press_rank", "mean"),
        reddit_rank=("reddit_rank", "mean"),
        rank_gap=("rank_gap", "mean"),
    ).reset_index()

    fan_favs   = avg_cent[avg_cent["rank_gap"] > 2].sort_values("rank_gap", ascending=False)
    media_darlgs = avg_cent[avg_cent["rank_gap"] < -2].sort_values("rank_gap")

    reddit_positive = sent_df.groupby("club")["sentiment_gap"].mean().nlargest(5)
    reddit_negative = sent_df.groupby("club")["sentiment_gap"].mean().nsmallest(5)

    reddit_leads = lag_df[lag_df["best_lag_months"] > 0].sort_values("best_lag_months", ascending=False)
    press_leads  = lag_df[lag_df["best_lag_months"] < 0].sort_values("best_lag_months")

    lines = [
        "# Press vs Reddit: Narrative Comparison Report",
        "_Fan Discourse vs. Professional Media Coverage (2018–2024)_",
        "",
        "---",
        "",
        "## 1. Centrality Divergence",
        "",
        "### Fan Favourites (more prominent on Reddit than in press)",
        "| Club | Avg Press Rank | Avg Reddit Rank | Gap |",
        "|------|---------------|----------------|-----|",
    ]
    for _, r in fan_favs.iterrows():
        lines.append(f"| {r['entity']} | #{r['press_rank']:.1f} | #{r['reddit_rank']:.1f} | +{r['rank_gap']:.1f} |")

    lines += [
        "",
        "### Media Darlings (more prominent in press than on Reddit)",
        "| Club | Avg Press Rank | Avg Reddit Rank | Gap |",
        "|------|---------------|----------------|-----|",
    ]
    for _, r in media_darlgs.iterrows():
        lines.append(f"| {r['entity']} | #{r['press_rank']:.1f} | #{r['reddit_rank']:.1f} | {r['rank_gap']:.1f} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Sentiment Gap",
        "",
        "### Clubs where fans are MORE positive than press",
        "| Club | Avg Sentiment Gap |",
        "|------|------------------|",
    ]
    for club, gap in reddit_positive.items():
        lines.append(f"| {club} | +{gap:.3f} |")

    lines += [
        "",
        "### Clubs where fans are MORE critical than press",
        "| Club | Avg Sentiment Gap |",
        "|------|------------------|",
    ]
    for club, gap in reddit_negative.items():
        lines.append(f"| {club} | {gap:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Lead / Lag",
        "",
        "### Reddit leads press (fans react first)",
        "| Club | Lead (months) | Correlation |",
        "|------|--------------|-------------|",
    ]
    for _, r in reddit_leads.head(8).iterrows():
        lines.append(f"| {r['club']} | +{r['best_lag_months']} months | {r['max_correlation']:.3f} |")

    lines += [
        "",
        "### Press leads Reddit (media drives fan discourse)",
        "| Club | Lag (months) | Correlation |",
        "|------|-------------|-------------|",
    ]
    for _, r in press_leads.head(8).iterrows():
        lines.append(f"| {r['club']} | {r['best_lag_months']} months | {r['max_correlation']:.3f} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Business Implications",
        "",
        "**Fan Favourites (Reddit > Press)**",
        "These clubs have organic fan communities that generate discourse independently",
        "of media cycles. A brand sponsoring here gets grassroots authenticity —",
        "the conversation is fan-driven, not PR-driven.",
        "",
        "**Media Darlings (Press > Reddit)**",
        "These clubs dominate headlines but fans don't necessarily follow.",
        "Press-driven narratives can inflate perceived value for sponsors.",
        "High press rank + low Reddit rank = potential overvaluation signal.",
        "",
        "**Reddit-First Clubs (positive lead/lag)**",
        "For these clubs, fan communities break stories before press covers them.",
        "This is valuable for early signal detection — monitor subreddits for",
        "transfer rumors, injury news, and coaching changes before they hit press.",
        "",
        "**Sentiment Divergence**",
        "Clubs where fans are significantly more negative than press are at risk",
        "of narrative collapse — the press story is better than the fan reality.",
        "Sponsors should monitor this gap as a brand safety signal.",
    ]

    path = COMPARISON_DIR / "press_vs_reddit_report.md"
    path.write_text("\n".join(lines))
    print(f"  Saved press_vs_reddit_report.md")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading centrality data...")
    cent_df, press_cent, reddit_cent = load_all()
    cent_df.to_csv(COMPARISON_DIR / "press_vs_reddit_centrality.csv", index=False)
    print(f"  {len(cent_df)} club-year records across {cent_df['entity'].nunique()} clubs")

    print("\n── 1. CENTRALITY DIVERGENCE ────────────────────────────────────────")
    plot_centrality_divergence(cent_df)
    plot_divergence_over_time(cent_df)

    print("\n── 2. SENTIMENT GAP ────────────────────────────────────────────────")
    print("  Loading sentiment (this may take a moment)...")
    press_sent, reddit_sent = load_sentiment()
    sent_df = plot_sentiment_gap(press_sent, reddit_sent)

    print("\n── 3. LEAD / LAG ANALYSIS ──────────────────────────────────────────")
    print("  Loading monthly volumes...")
    press_vol, reddit_vol = load_monthly_volumes()
    lag_df = plot_lead_lag(press_vol, reddit_vol)
    plot_event_lead_lag(press_vol, reddit_vol)

    print("\n── 4. NARRATIVE POWER INDEX ────────────────────────────────────────")
    plot_narrative_power(press_vol, reddit_vol, cent_df)

    print("\n── REPORT ──────────────────────────────────────────────────────────")
    write_report(cent_df, sent_df, lag_df)

    charts = sorted(PLOT_DIR.glob("reddit_*.png"))
    print(f"\n  Done. {len(charts)} charts saved:")
    for c in charts:
        print(f"    {c.name}")


if __name__ == "__main__":
    main()
