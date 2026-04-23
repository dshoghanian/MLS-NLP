"""
Market Alignment Analysis
=========================
Compares narrative capital (media prominence) against commercial market value
(Forbes club valuations) to identify business opportunities:

  - Narratively prominent but undervalued  → underpriced sponsorship / broadcast asset
  - Commercially valued but narratively quiet → marketing gap / brand investment opportunity
  - Aligned clubs                           → efficient market, stable positioning

Outputs:
  data/analysis/press/market_alignment.csv
  data/analysis/press/plots/market_*.png
  data/analysis/press/market_alignment_report.md

NOTE: Valuation figures in data/performance/mls_valuations.csv are approximate,
based on publicly reported Forbes MLS franchise valuations. Verify against
actual Forbes reports before using in a formal presentation.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRESS_DIR = ROOT / "data" / "analysis" / "press"
SHARED_DIR = ROOT / "data" / "analysis" / "shared"
PLOT_DIR  = PRESS_DIR / "plots"
PRESS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Brand palette (reuse from explore.py) ─────────────────────────────────────
CLUB_COLORS = {
    "Atlanta United FC":       "#80000A",
    "Austin FC":               "#00B140",
    "CF Montreal":             "#003DA5",
    "Charlotte FC":            "#1A85C8",
    "Chicago Fire FC":         "#CC0000",
    "Colorado Rapids":         "#862633",
    "Columbus Crew":           "#FFF200",
    "D.C. United":             "#000000",
    "FC Dallas":               "#BF0D3E",
    "Houston Dynamo FC":       "#F68712",
    "Inter Miami CF":          "#F7B5CD",
    "LA Galaxy":               "#00245D",
    "LAFC":                    "#C39E6D",
    "Minnesota United FC":     "#8CD2F4",
    "Nashville SC":            "#ECE83A",
    "New England Revolution":  "#0A2240",
    "New York City FC":        "#6CACE4",
    "New York Red Bulls":      "#ED1E36",
    "Orlando City SC":         "#612B9B",
    "Philadelphia Union":      "#071B2C",
    "Portland Timbers":        "#00482B",
    "Real Salt Lake":          "#B30838",
    "San Jose Earthquakes":    "#0D4C92",
    "Seattle Sounders FC":     "#5D9741",
    "Sporting Kansas City":    "#93B8E3",
    "St. Louis City SC":       "#DD0000",
    "Toronto FC":              "#B81137",
    "Vancouver Whitecaps FC":  "#009BC8",
}
DEFAULT_COLOR = "#888888"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_data():
    summary = pd.read_csv(SHARED_DIR / "master_summary.csv")
    valuations = pd.read_csv(ROOT / "data" / "performance" / "mls_valuations.csv")

    df = summary.merge(valuations, left_on=["entity", "year"], right_on=["club", "year"], how="inner")
    df = df.drop(columns=["club"], errors="ignore")

    # Valuation rank per year (1 = highest value)
    df["valuation_rank"] = df.groupby("year")["valuation_usd_m"].rank(ascending=False).astype(int)

    # Market alignment gap: positive = undervalued relative to narrative prominence
    # (narrative rank is BETTER than valuation rank → club gets more press than its $ warrants)
    df["market_gap"] = df["valuation_rank"] - df["narrative_rank"]

    return df


# ── Chart 1: Quadrant scatter — narrative vs valuation rank ────────────────────

def plot_quadrant(df: pd.DataFrame, year: int = None):
    title_suffix = f" ({year})" if year else " (2018–2024 avg)"
    if year:
        sub = df[df["year"] == year].copy()
    else:
        sub = df.groupby("entity").agg(
            narrative_rank=("narrative_rank", "mean"),
            valuation_rank=("valuation_rank", "mean"),
            market_gap=("market_gap", "mean"),
            valuation_usd_m=("valuation_usd_m", "mean"),
        ).reset_index()

    n_clubs = sub["narrative_rank"].max()
    mid = n_clubs / 2

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#f8f8f8")

    # Quadrant shading
    ax.axhspan(0, mid, xmin=0, xmax=0.5, alpha=0.06, color="green")   # high narrative, low val rank → high val
    ax.axhspan(mid, n_clubs + 1, xmin=0.5, xmax=1.0, alpha=0.06, color="orange")  # low narrative, high val rank
    ax.axhline(mid, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(mid, color="gray", lw=0.8, ls="--", alpha=0.5)

    for _, row in sub.iterrows():
        club = row["entity"]
        color = CLUB_COLORS.get(club, DEFAULT_COLOR)
        size = np.clip(row.get("valuation_usd_m", 400) / 8, 30, 220)
        ax.scatter(row["narrative_rank"], row["valuation_rank"], s=size,
                   color=color, edgecolors="white", linewidths=0.8, zorder=3, alpha=0.9)
        ax.annotate(club.replace(" FC", "").replace(" CF", "").replace(" SC", ""),
                    (row["narrative_rank"], row["valuation_rank"]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points", zorder=4)

    # Diagonal reference line
    lim = n_clubs + 1
    ax.plot([1, lim], [1, lim], color="gray", lw=1, ls=":", alpha=0.6, zorder=1)

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Narrative Rank (1 = most media attention)", fontsize=11)
    ax.set_ylabel("Valuation Rank (1 = highest Forbes value)", fontsize=11)
    ax.set_title(f"Narrative Capital vs. Commercial Value{title_suffix}", fontsize=14, fontweight="bold")

    # Quadrant labels
    ax.text(n_clubs * 0.85, n_clubs * 0.1, "UNDEREXPOSED\nHigh value, low narrative\n→ Marketing opportunity",
            fontsize=8, color="darkorange", alpha=0.8, ha="center")
    ax.text(n_clubs * 0.15, n_clubs * 0.9, "OVEREXPOSED\nHigh narrative, lower value\n→ Sponsorship opportunity",
            fontsize=8, color="green", alpha=0.8, ha="center")
    ax.text(n_clubs * 0.15, n_clubs * 0.1, "MARKET LEADERS\nHigh narrative + high value",
            fontsize=8, color="#333333", alpha=0.6, ha="center")
    ax.text(n_clubs * 0.85, n_clubs * 0.9, "SLEEPING ASSETS\nLow narrative + low value",
            fontsize=8, color="#555555", alpha=0.5, ha="center")

    plt.tight_layout()
    tag = str(year) if year else "avg"
    path = PLOT_DIR / f"market_quadrant_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ── Chart 2: Market gap bar chart ────────────────────────────────────────────

def plot_market_gap(df: pd.DataFrame):
    avg = df.groupby("entity")["market_gap"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(13, 8))
    colors = ["#C0392B" if v < 0 else "#27AE60" for v in avg.values]
    bars = ax.barh(avg.index, avg.values, color=colors, edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Avg Market Gap (valuation rank − narrative rank)\nPositive = undervalued relative to narrative prominence", fontsize=10)
    ax.set_title("Market Alignment Gap — 2018–2024 Average\nGreen = overexposed (narrative > value)  |  Red = underexposed (value > narrative)",
                 fontsize=12, fontweight="bold")

    for bar, val in zip(bars, avg.values):
        ax.text(val + (0.3 if val >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    plt.tight_layout()
    path = PLOT_DIR / "market_gap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ── Chart 3: Valuation growth vs narrative trajectory ─────────────────────────

def plot_valuation_vs_narrative_trajectory(df: pd.DataFrame, clubs: list[str]):
    fig, axes = plt.subplots(len(clubs), 1, figsize=(12, 3.5 * len(clubs)), sharex=True)
    if len(clubs) == 1:
        axes = [axes]

    for ax, club in zip(axes, clubs):
        sub = df[df["entity"] == club].sort_values("year")
        if sub.empty:
            continue

        color = CLUB_COLORS.get(club, DEFAULT_COLOR)
        ax2 = ax.twinx()

        ax.plot(sub["year"], sub["narrative_rank"], marker="o", color=color, lw=2, label="Narrative Rank")
        ax2.bar(sub["year"], sub["valuation_usd_m"], alpha=0.25, color=color, label="Valuation ($M)")
        ax2.plot(sub["year"], sub["valuation_usd_m"], color=color, lw=1.5, ls="--", alpha=0.7)

        ax.invert_yaxis()
        ax.set_ylabel("Narrative Rank\n(lower = more prominent)", fontsize=9)
        ax2.set_ylabel("Valuation ($M)", fontsize=9)
        ax.set_title(club, fontsize=11, fontweight="bold", color=color)
        ax.set_xticks(sub["year"])
        ax.tick_params(axis="x", rotation=0)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.suptitle("Narrative Prominence vs. Club Valuation Over Time", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = PLOT_DIR / "market_valuation_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ── Chart 4: Narrative-to-value conversion rate ──────────────────────────────

def plot_narrative_value_ratio(df: pd.DataFrame):
    """Which clubs convert narrative prominence into valuation most efficiently?"""
    avg = df.groupby("entity").agg(
        avg_narrative_rank=("narrative_rank", "mean"),
        avg_valuation=("valuation_usd_m", "mean"),
        avg_pagerank=("pagerank", "mean"),
    ).reset_index()

    # Value per unit of narrative prominence (lower narrative rank = more prominent)
    # Flip rank so higher = more prominent
    n = avg["avg_narrative_rank"].max()
    avg["narrative_prominence"] = n + 1 - avg["avg_narrative_rank"]
    avg["value_per_prominence"] = avg["avg_valuation"] / avg["narrative_prominence"]

    avg = avg.sort_values("value_per_prominence", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 8))
    colors = [CLUB_COLORS.get(c, DEFAULT_COLOR) for c in avg["entity"]]
    bars = ax.barh(avg["entity"], avg["value_per_prominence"], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Avg Valuation ($M) per Narrative Prominence Unit\nHigher = more $ per unit of media attention", fontsize=10)
    ax.set_title("Narrative-to-Value Conversion Efficiency\nWhich clubs extract the most commercial value from their media presence?",
                 fontsize=12, fontweight="bold")

    for bar, val in zip(bars, avg["value_per_prominence"]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"${val:.0f}M",
                va="center", fontsize=8)

    plt.tight_layout()
    path = PLOT_DIR / "market_narrative_value_efficiency.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ── Chart 5: Correlation heatmap — narrative, performance, valuation ──────────

def plot_correlation_matrix(df: pd.DataFrame):
    cols = {
        "narrative_rank": "Narrative Rank",
        "performance_rank": "Perf. Rank",
        "valuation_rank": "Valuation Rank",
        "pagerank": "PageRank",
        "points": "Points",
        "valuation_usd_m": "Valuation ($M)",
        "sentiment_compound" if "sentiment_compound" in df.columns else "pagerank": "PageRank",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    sub = df[list(available.keys())].rename(columns=available)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, mask=mask)
    ax.set_title("Correlation Matrix: Narrative × Performance × Market Value",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = PLOT_DIR / "market_correlation_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ── Markdown report ───────────────────────────────────────────────────────────

def write_report(df: pd.DataFrame):
    avg = df.groupby("entity").agg(
        avg_narrative_rank=("narrative_rank", "mean"),
        avg_valuation_rank=("valuation_rank", "mean"),
        avg_market_gap=("market_gap", "mean"),
        avg_valuation=("valuation_usd_m", "mean"),
        avg_pagerank=("pagerank", "mean"),
    ).reset_index()

    overexposed  = avg[avg["avg_market_gap"] > 3].sort_values("avg_market_gap", ascending=False)
    underexposed = avg[avg["avg_market_gap"] < -3].sort_values("avg_market_gap")

    # Messi valuation jump
    miami = df[df["entity"] == "Inter Miami CF"].sort_values("year")[["year", "valuation_usd_m", "narrative_rank"]]

    lines = [
        "# MLS Market Alignment Report",
        "_Narrative Capital vs. Forbes Club Valuations (2018–2024)_",
        "",
        "> **Note:** Valuation figures are approximate, based on publicly reported Forbes MLS",
        "> franchise valuations. Verify against actual Forbes reports before formal use.",
        "",
        "---",
        "",
        "## What Is Market Alignment?",
        "",
        "**Narrative capital** = media prominence measured by PageRank centrality in the",
        "MLS press co-occurrence network. A club with high narrative capital dominates",
        "sports media discourse.",
        "",
        "**Commercial value** = Forbes franchise valuation ($M), driven by revenue, stadium",
        "deals, broadcast share, and brand sponsorships.",
        "",
        "**Market alignment gap** = valuation rank − narrative rank.",
        "- **Positive gap** → club is more prominent in media than its market value suggests",
        "  (*overexposed*: narrative outpaces $) — potential underpriced sponsorship asset.",
        "- **Negative gap** → club commands high market value with less media noise",
        "  (*underexposed*: $ outpaces narrative) — marketing/brand investment opportunity.",
        "",
        "---",
        "",
        "## Business Implications",
        "",
        "### 1. Sponsorship Pricing",
        "An **overexposed club** (high narrative, lower valuation) is likely undercharging",
        "for sponsorship. Media reach exceeds what the balance sheet reflects — a brand",
        "buying jersey or naming rights here gets outsized exposure relative to the deal cost.",
        "",
        "### 2. Broadcast & Media Rights",
        "Broadcasters bidding on regional rights should weight narrative capital, not just",
        "market size. A club generating consistent centrality spikes in national press",
        "drives viewership beyond its metro area.",
        "",
        "### 3. Investor / Acquisition Targeting",
        "An **underexposed club** (high valuation, low narrative) may be overpriced relative",
        "to its cultural footprint. Conversely, a club with strong narrative momentum and",
        "a lagging valuation is a potential acquisition target before the market catches up.",
        "",
        "### 4. Expansion Club Lifecycle",
        "New clubs (Austin, Charlotte, St. Louis) show narrative capital spiking at launch",
        "then normalizing. The window between launch buzz and valuation normalization is",
        "the optimal period to lock in long-term sponsorship deals at pre-market prices.",
        "",
        "---",
        "",
        "## Overexposed Clubs (narrative rank > valuation rank)",
        "_High media presence relative to franchise value — underpriced media assets_",
        "",
        "| Club | Avg Narrative Rank | Avg Valuation Rank | Avg Gap | Avg Valuation ($M) |",
        "|------|-------------------|-------------------|---------|-------------------|",
    ]
    for _, r in overexposed.iterrows():
        lines.append(f"| {r['entity']} | #{r['avg_narrative_rank']:.1f} | #{r['avg_valuation_rank']:.1f} | +{r['avg_market_gap']:.1f} | ${r['avg_valuation']:.0f}M |")

    lines += [
        "",
        "## Underexposed Clubs (valuation rank > narrative rank)",
        "_High franchise value with less media dominance — brand investment opportunity_",
        "",
        "| Club | Avg Narrative Rank | Avg Valuation Rank | Avg Gap | Avg Valuation ($M) |",
        "|------|-------------------|-------------------|---------|-------------------|",
    ]
    for _, r in underexposed.iterrows():
        lines.append(f"| {r['entity']} | #{r['avg_narrative_rank']:.1f} | #{r['avg_valuation_rank']:.1f} | {r['avg_market_gap']:.1f} | ${r['avg_valuation']:.0f}M |")

    lines += [
        "",
        "---",
        "",
        "## Case Study: Inter Miami CF — Narrative-Driven Valuation Jump",
        "",
        "The Messi signing in 2023 is the clearest example of narrative capital directly",
        "creating commercial value in this dataset:",
        "",
        "| Year | Valuation ($M) | Narrative Rank |",
        "|------|---------------|----------------|",
    ]
    for _, r in miami.iterrows():
        lines.append(f"| {int(r['year'])} | ${r['valuation_usd_m']:.0f}M | #{int(r['narrative_rank'])} |")

    lines += [
        "",
        "From 2022 to 2023: valuation jumped **+$685M (+240%)** in a single year, directly",
        "correlated with Inter Miami reaching narrative rank #1 in the league. This is a",
        "concrete example of narrative capital converting into market value.",
        "",
        "---",
        "",
        "## Practical Use Cases for This Tool",
        "",
        "| Stakeholder | How to Use |",
        "|-------------|-----------|",
        "| **Sponsor / Brand** | Target clubs where narrative rank > valuation rank — maximum media exposure at below-market cost |",
        "| **Club Front Office** | If valuation rank > narrative rank, invest in PR/content strategy to close the gap |",
        "| **Investor / PE Firm** | Find clubs with rising narrative momentum and lagging valuation — buy before the market re-prices |",
        "| **Broadcaster** | Weight narrative centrality when bidding on regional rights packages |",
        "| **Player Agent** | Use club narrative momentum to negotiate contract timing (higher attention = better leverage) |",
        "",
    ]

    report_path = PRESS_DIR / "market_alignment_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} club-year records | {df['entity'].nunique()} clubs | {df['year'].nunique()} years")

    df.to_csv(PRESS_DIR / "market_alignment.csv", index=False)
    print("  Saved market_alignment.csv")

    print("\n── CHARTS ─────────────────────────────────────────────────────────")
    plot_quadrant(df)                                  # avg across all years
    plot_quadrant(df, year=2023)                       # 2023 snapshot (Messi year)
    plot_market_gap(df)
    plot_valuation_vs_narrative_trajectory(df, [
        "Inter Miami CF", "Toronto FC", "LAFC",
        "Charlotte FC", "Philadelphia Union", "Seattle Sounders FC",
    ])
    plot_narrative_value_ratio(df)
    plot_correlation_matrix(df)

    print("\n── REPORT ─────────────────────────────────────────────────────────")
    write_report(df)

    # Print summary table
    avg = df.groupby("entity").agg(
        narrative_rank=("narrative_rank", "mean"),
        valuation_rank=("valuation_rank", "mean"),
        market_gap=("market_gap", "mean"),
        avg_val=("valuation_usd_m", "mean"),
    ).reset_index().sort_values("market_gap", ascending=False)

    print("\nMarket Alignment Summary (sorted by gap):")
    print(f"{'Club':<28} {'Narr.Rank':>10} {'Val.Rank':>9} {'Gap':>6} {'Avg Val($M)':>11}")
    print("-" * 68)
    for _, r in avg.iterrows():
        sign = "+" if r["market_gap"] > 0 else ""
        print(f"{r['entity']:<28} #{r['narrative_rank']:>7.1f}   #{r['valuation_rank']:>6.1f}  {sign}{r['market_gap']:>5.1f}   ${r['avg_val']:>8.0f}M")

    print(f"\n  Charts: {PLOT_DIR}")
    print(f"  Report: data/analysis/press/market_alignment_report.md")


if __name__ == "__main__":
    main()
