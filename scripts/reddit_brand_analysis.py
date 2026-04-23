"""
Reddit Brand Analysis & Press vs Reddit Brand Comparison
=========================================================
Scans Reddit fan posts for the same brand keywords used in press analysis,
then compares press earned media vs fan organic brand mentions side by side.

Key question:
  "Which sponsors have press coverage but no fan resonance — and which
   brands do fans talk about that the press ignores?"

This gap = the difference between paid/earned media and authentic fan engagement.

Outputs (data/analysis/reddit/ and data/analysis/comparison/:
  - brand_reddit_earned_media.csv
  - brand_press_vs_reddit.csv

Charts (data/analysis/reddit/plots/ and comparison/plots/:
  - brand_reddit_volume.png          — Reddit brand mention volume
  - brand_reddit_sentiment.png       — Reddit brand sentiment
  - brand_press_vs_reddit_volume.png — side-by-side press vs Reddit volume
  - brand_press_vs_reddit_sentiment.png — sentiment divergence
  - brand_resonance_matrix.png       — press prominence vs fan resonance scatter
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.utils import get_logger, PROJECT_ROOT

logger = get_logger("reddit_brand_analysis")

PRESS_DIR      = ROOT / "data" / "analysis" / "press"
REDDIT_DIR     = ROOT / "data" / "analysis" / "reddit"
COMPARISON_DIR = ROOT / "data" / "analysis" / "comparison"
REDDIT_PLOTS   = REDDIT_DIR / "plots"
COMPARISON_PLOTS = COMPARISON_DIR / "plots"
REDDIT_RAW     = ROOT / "data" / "reddit" / "raw"
REDDIT_DIR.mkdir(parents=True, exist_ok=True)
REDDIT_PLOTS.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_PLOTS.mkdir(parents=True, exist_ok=True)

STYLE = {
    "press":    "#1a3a5c",
    "reddit":   "#FF4500",   # Reddit orange
    "positive": "#27ae60",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
    "bg":       "#f8f9fa",
}

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["bg"],
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

# ── Same brand keywords as press analysis ─────────────────────────────────────
BRAND_KEYWORDS = {
    "Adidas":            [r"\badidas\b"],
    "Nike":              [r"\bnike\b"],
    "Puma":              [r"\bpuma\b"],
    "Herbalife":         [r"\bherbalife\b"],
    "Red Bull":          [r"\bred bull\b"],
    "Etihad Airways":    [r"\betihad\b"],
    "BMO":               [r"\bbmo\b", r"\bbank of montreal\b"],
    "Subaru":            [r"\bsubaru\b"],
    "Motorola":          [r"\bmotorola\b"],
    "Audi":              [r"\baudi\b"],
    "Bank of America":   [r"\bbank of america\b", r"\bbofa\b"],
    "Alaska Airlines":   [r"\balaska airlines\b"],
    "Yeti":              [r"\byeti\b"],
    "PayPal":            [r"\bpaypal\b"],
    "BetMGM":            [r"\bbetmgm\b"],
    "Acura":             [r"\bacura\b"],
    "Geodis":            [r"\bgeodis\b"],
    "Leidos":            [r"\bleidos\b"],
    "MasterCard":        [r"\bmastercard\b", r"\bmaster card\b"],
    "Zulily":            [r"\bzulily\b"],
    "Blue Cross Blue Shield": [r"\bblue cross\b", r"\bbcbs\b"],
    "Guaranteed Rate":   [r"\bguaranteed rate\b"],
    "Frontier Airlines": [r"\bfrontier airlines\b"],
    "Apple":             [r"\bapple tv\b", r"\bapple mls\b", r"\bmls season pass\b"],
    "AT&T":              [r"\bat&t\b", r"\bat and t\b"],
    "Heineken":          [r"\bheineken\b"],
    "Target":            [r"\btarget\b"],
    "Wells Fargo":       [r"\bwells fargo\b"],
    "Allstate":          [r"\ballstate\b"],
    "Gatorade":          [r"\bgatorade\b"],
    "EA Sports":         [r"\bea sports\b", r"\bfifa\b", r"\bfc 24\b"],
}


# ── Scan Reddit posts for brand mentions ──────────────────────────────────────

def scan_reddit_brands() -> pd.DataFrame:
    logger.info("Scanning Reddit posts for brand mentions ...")

    compiled = {
        brand: [re.compile(p, re.IGNORECASE) for p in patterns]
        for brand, patterns in BRAND_KEYWORDS.items()
    }

    records = []
    total_posts = 0

    for year in range(2018, 2025):
        for month in range(1, 13):
            raw_path = REDDIT_RAW / str(year) / f"{year}_{month:02d}_reddit.parquet"
            if not raw_path.exists():
                continue

            df = pd.read_parquet(str(raw_path))
            if df.empty:
                continue
            total_posts += len(df)

            for _, row in df.iterrows():
                full_text = f"{row.get('title', '')} {row.get('text', '')}".lower()
                clubs = [c for c in str(row.get("primary_club", "")).split("|") if c]
                # Also pick up clubs from subreddit mapping
                if not clubs and row.get("primary_club"):
                    clubs = [str(row["primary_club"])]

                sentiment = 0.0  # populated from enriched below
                pub_date  = str(row.get("published_date", f"{year}-{month:02d}-01"))

                for brand, patterns in compiled.items():
                    if any(p.search(full_text) for p in patterns):
                        records.append({
                            "brand":        brand,
                            "club":         clubs[0] if clubs else None,
                            "year":         int(row.get("collection_year", year)),
                            "month":        int(row.get("collection_month", month)),
                            "subreddit":    str(row.get("subreddit", "")),
                            "post_type":    str(row.get("post_type", "")),
                            "score":        int(row.get("score", 0)),
                            "published_date": pub_date,
                            "article_id":   str(row["article_id"]),
                        })

    logger.info(f"  Scanned {total_posts:,} Reddit posts")
    df_out = pd.DataFrame(records).drop_duplicates(subset=["brand", "article_id", "club"])

    # Merge sentiment + event_type from enriched files
    df_out = _attach_enriched(df_out)

    # Match press brand_earned_media.csv column order, with Reddit-specific cols appended
    col_order = ["brand", "club", "year", "month", "sentiment", "event_type",
                 "published_date", "article_id", "subreddit", "post_type", "score"]
    df_out = df_out[[c for c in col_order if c in df_out.columns]]

    df_out.to_csv(REDDIT_DIR / "brand_reddit_earned_media.csv", index=False)
    logger.info(f"  Found {len(df_out):,} brand-post-club co-occurrences")
    logger.info("  Saved brand_reddit_earned_media.csv")
    return df_out


def _attach_enriched(df: pd.DataFrame) -> pd.DataFrame:
    """Join Reddit enriched sentiment + event_type onto brand records by article_id."""
    REDDIT_ENRICHED = ROOT / "data" / "reddit" / "enriched"
    enr_frames = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            path = REDDIT_ENRICHED / str(year) / f"{year}_{month:02d}_reddit_enriched.parquet"
            if path.exists():
                enr = pd.read_parquet(str(path))[
                    ["article_id", "sentiment_compound", "primary_event_type"]
                ]
                enr_frames.append(enr)
    if not enr_frames:
        df["sentiment"]   = 0.0
        df["event_type"]  = ""
        return df
    enr_all = pd.concat(enr_frames, ignore_index=True).drop_duplicates("article_id")
    df = df.merge(enr_all, on="article_id", how="left")
    df["sentiment"]  = df["sentiment_compound"].fillna(0.0)
    df["event_type"] = df["primary_event_type"].fillna("")
    df = df.drop(columns=["sentiment_compound", "primary_event_type"], errors="ignore")
    return df


# ── Build press vs Reddit comparison table ────────────────────────────────────

def build_comparison(df_reddit: pd.DataFrame) -> pd.DataFrame:
    press_path = PRESS_DIR / "brand_earned_media.csv"
    if not press_path.exists():
        logger.warning("brand_earned_media.csv not found — run brand_analysis.py first")
        return pd.DataFrame()

    df_press = pd.read_csv(press_path)

    def summarize(df, source_label):
        return (
            df.drop_duplicates(subset=["brand", "article_id"])
            .groupby("brand")
            .agg(
                mentions   = ("article_id", "count"),
                avg_sentiment = ("sentiment", "mean"),
            )
            .reset_index()
            .rename(columns={
                "mentions":      f"{source_label}_mentions",
                "avg_sentiment": f"{source_label}_sentiment",
            })
        )

    press_summary  = summarize(df_press,  "press")
    reddit_summary = summarize(df_reddit, "reddit")

    comp = press_summary.merge(reddit_summary, on="brand", how="outer").fillna(0)
    comp["total_mentions"]    = comp["press_mentions"] + comp["reddit_mentions"]
    comp["reddit_share"]      = comp["reddit_mentions"] / (comp["total_mentions"] + 1e-9)
    comp["sentiment_gap"]     = comp["press_sentiment"] - comp["reddit_sentiment"]
    comp["resonance_score"]   = (comp["press_mentions"] * comp["reddit_mentions"]) ** 0.5

    # Category labels
    def resonance_label(row):
        if row["press_mentions"] >= 10 and row["reddit_mentions"] >= 5:
            return "High reach, high resonance"
        elif row["press_mentions"] >= 10 and row["reddit_mentions"] < 5:
            return "Press-only (low fan resonance)"
        elif row["press_mentions"] < 10 and row["reddit_mentions"] >= 5:
            return "Fan-organic (low press coverage)"
        else:
            return "Low presence"

    comp["category"] = comp.apply(resonance_label, axis=1)
    comp = comp.sort_values("total_mentions", ascending=False)
    comp.to_csv(COMPARISON_DIR / "brand_press_vs_reddit.csv", index=False)
    logger.info(f"  Saved brand_press_vs_reddit.csv ({len(comp)} brands)")
    return comp


# ── Charts ─────────────────────────────────────────────────────────────────────

def chart_reddit_volume(df: pd.DataFrame):
    vol = (df.drop_duplicates(subset=["brand", "article_id"])
           .groupby("brand").size()
           .sort_values(ascending=False)
           .head(20))

    if vol.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.Oranges(np.linspace(0.35, 0.9, len(vol)))[::-1]
    bars = ax.barh(vol.index[::-1], vol.values[::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, vol.values[::-1]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8)

    ax.set_xlabel("Unique Reddit Posts Mentioning Brand", fontsize=11)
    ax.set_title("Brand Organic Fan Mentions — Reddit MLS Discourse (2018–2024)\n"
                 "Top 20 brands by unique post count",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REDDIT_PLOTS / "brand_reddit_volume.png", dpi=150)
    plt.close(fig)
    logger.info("Saved brand_reddit_volume.png")


def chart_reddit_sentiment(df: pd.DataFrame):
    brand_stats = (
        df.drop_duplicates(subset=["brand", "article_id"])
        .groupby("brand")
        .agg(avg_sentiment=("sentiment", "mean"), mentions=("article_id", "count"))
        .reset_index()
        .query("mentions >= 3")
        .sort_values("avg_sentiment", ascending=False)
    )
    if brand_stats.empty:
        return

    colors = ["#27ae60" if v >= 0.2 else "#e74c3c" if v < 0 else "#f39c12"
              for v in brand_stats["avg_sentiment"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(brand_stats["brand"], brand_stats["avg_sentiment"],
                   color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    for bar, row in zip(bars, brand_stats.itertuples()):
        ax.text(row.avg_sentiment + 0.005 if row.avg_sentiment >= 0 else row.avg_sentiment - 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{row.avg_sentiment:.2f}  (n={row.mentions})",
                va="center", ha="left" if row.avg_sentiment >= 0 else "right", fontsize=8)

    ax.invert_yaxis()
    ax.set_xlabel("Avg VADER Sentiment in Reddit Posts", fontsize=10)
    ax.set_title("Brand Sentiment in Fan Discourse\n"
                 "How do Reddit fans feel about each brand?",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(REDDIT_PLOTS / "brand_reddit_sentiment.png", dpi=150)
    plt.close(fig)
    logger.info("Saved brand_reddit_sentiment.png")


def chart_press_vs_reddit_volume(comp: pd.DataFrame):
    if comp.empty:
        return
    top = comp.nlargest(18, "total_mentions").sort_values("total_mentions")

    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(top))
    height = 0.38

    ax.barh(y + height / 2, top["press_mentions"],  height=height,
            color=STYLE["press"],  alpha=0.85, label="Press articles",  edgecolor="white")
    ax.barh(y - height / 2, top["reddit_mentions"], height=height,
            color=STYLE["reddit"], alpha=0.85, label="Reddit posts", edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(top["brand"], fontsize=10)
    ax.set_xlabel("Unique document mentions", fontsize=11)
    ax.set_title("Press vs Reddit Brand Mentions — Which Channel Drives Each Brand?\n"
                 "Top 18 brands by total combined mentions (2018–2024)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOTS / "brand_press_vs_reddit_volume.png", dpi=150)
    plt.close(fig)
    logger.info("Saved brand_press_vs_reddit_volume.png")


def chart_press_vs_reddit_sentiment(comp: pd.DataFrame):
    """Dot plot: press sentiment vs Reddit sentiment per brand."""
    if comp.empty:
        return
    top = comp[comp["total_mentions"] >= 5].nlargest(20, "total_mentions")

    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(top))

    ax.scatter(top["press_sentiment"],  y, s=80, color=STYLE["press"],
               zorder=3, label="Press sentiment")
    ax.scatter(top["reddit_sentiment"], y, s=80, color=STYLE["reddit"],
               zorder=3, label="Reddit sentiment", marker="D")

    for i, (_, row) in enumerate(top.iterrows()):
        ax.plot([row["press_sentiment"], row["reddit_sentiment"]], [i, i],
                color="gray", lw=1, alpha=0.5, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(top["brand"], fontsize=9)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Avg VADER Sentiment Score", fontsize=11)
    ax.set_title("Press vs Fan Sentiment per Brand\n"
                 "Circles = Press | Diamonds = Reddit | Gap = disconnect",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOTS / "brand_press_vs_reddit_sentiment.png", dpi=150)
    plt.close(fig)
    logger.info("Saved brand_press_vs_reddit_sentiment.png")


def chart_resonance_matrix(comp: pd.DataFrame):
    """Scatter: press mentions (X) vs Reddit mentions (Y) — resonance quadrants."""
    if comp.empty:
        return
    plot = comp[comp["total_mentions"] >= 3].copy()

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor(STYLE["bg"])

    cat_colors = {
        "High reach, high resonance":   "#27ae60",
        "Press-only (low fan resonance)": STYLE["press"],
        "Fan-organic (low press coverage)": STYLE["reddit"],
        "Low presence":                 STYLE["neutral"],
    }

    for cat, sub in plot.groupby("category"):
        ax.scatter(sub["press_mentions"], sub["reddit_mentions"],
                   color=cat_colors.get(cat, "gray"), s=90, alpha=0.8,
                   label=cat, edgecolors="white", lw=0.5)

    for _, row in plot.iterrows():
        ax.annotate(row["brand"],
                    (row["press_mentions"], row["reddit_mentions"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, alpha=0.85)

    # Quadrant lines
    med_x = plot["press_mentions"].median()
    med_y = plot["reddit_mentions"].median()
    ax.axvline(med_x, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(med_y, color="gray", lw=0.8, ls="--", alpha=0.5)

    ax.text(med_x * 0.02, plot["reddit_mentions"].max() * 0.95,
            "Fan-organic\n(low press)", fontsize=8, color=STYLE["reddit"], alpha=0.8)
    ax.text(plot["press_mentions"].max() * 0.55, med_y * 0.1,
            "Press-only\n(low fan resonance)", fontsize=8, color=STYLE["press"], alpha=0.8)
    ax.text(plot["press_mentions"].max() * 0.55, plot["reddit_mentions"].max() * 0.9,
            "High reach\n+ resonance", fontsize=8, color="#27ae60", alpha=0.9)

    ax.set_xlabel("Press Article Mentions", fontsize=11)
    ax.set_ylabel("Reddit Post Mentions", fontsize=11)
    ax.set_title("Brand Resonance Matrix: Press Coverage vs Fan Engagement\n"
                 "Top-right = sponsored AND loved by fans",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOTS / "brand_resonance_matrix.png", dpi=150)
    plt.close(fig)
    logger.info("Saved brand_resonance_matrix.png")


# ── Markdown report ────────────────────────────────────────────────────────────

def write_report(df_reddit: pd.DataFrame, comp: pd.DataFrame):
    reddit_summary = (
        df_reddit.drop_duplicates(subset=["brand", "article_id"])
        .groupby("brand")
        .agg(mentions=("article_id", "count"), avg_sentiment=("sentiment", "mean"))
        .reset_index()
        .sort_values("mentions", ascending=False)
    )

    high_resonance = comp[comp["category"] == "High reach, high resonance"].sort_values("resonance_score", ascending=False)
    press_only     = comp[comp["category"] == "Press-only (low fan resonance)"].sort_values("press_mentions", ascending=False)
    fan_organic    = comp[comp["category"] == "Fan-organic (low press coverage)"].sort_values("reddit_mentions", ascending=False)

    lines = [
        "# Reddit Brand Analysis: Fan Organic Mentions vs Press Earned Media",
        "_MLS Discourse 2018–2024 — Brand Resonance Comparison_",
        "",
        "## What This Measures",
        "The same brand keywords used in press analysis were scanned across all",
        f"{df_reddit['article_id'].nunique():,} Reddit fan posts. This captures **organic fan brand resonance** —",
        "how often fans mention brands unprompted, vs press earned media (journalist coverage).",
        "",
        "The gap between the two reveals:",
        "- **Press-only brands**: paid-for coverage that doesn't reach fan communities",
        "- **Fan-organic brands**: authentic fan engagement brands may not be leveraging",
        "- **High resonance brands**: appearing in both channels → maximum marketing impact",
        "",
        "---",
        "",
        "## Reddit Brand Mention Rankings",
        "",
        "| Brand | Reddit Posts | Avg Fan Sentiment |",
        "|-------|-------------|------------------|",
    ]

    for _, r in reddit_summary.head(20).iterrows():
        sent_label = "positive" if r["avg_sentiment"] >= 0.2 else "negative" if r["avg_sentiment"] < 0 else "neutral"
        lines.append(f"| {r['brand']} | {int(r['mentions'])} | {r['avg_sentiment']:.2f} ({sent_label}) |")

    if not comp.empty:
        lines += [
            "",
            "---",
            "",
            "## Press vs Reddit Brand Comparison",
            "",
            "| Brand | Press Mentions | Reddit Mentions | Reddit Share | Sentiment Gap | Category |",
            "|-------|---------------|----------------|-------------|--------------|----------|",
        ]
        for _, r in comp.head(25).iterrows():
            lines.append(
                f"| {r['brand']} | {int(r['press_mentions'])} | {int(r['reddit_mentions'])} "
                f"| {r['reddit_share']:.0%} | {r['sentiment_gap']:+.2f} | {r['category']} |"
            )

        lines += ["", "---", ""]

        if not high_resonance.empty:
            lines += [
                "## High Reach + High Resonance Brands",
                "These brands appear frequently in *both* press and fan discourse — the most",
                "commercially valuable position for a sponsor.",
                "",
            ]
            for _, r in high_resonance.head(8).iterrows():
                lines.append(f"- **{r['brand']}** — {int(r['press_mentions'])} press | {int(r['reddit_mentions'])} Reddit")
            lines.append("")

        if not press_only.empty:
            lines += [
                "## Press-Only Brands (Low Fan Resonance)",
                "High press mentions but fans rarely discuss them organically.",
                "These sponsors are paying for media coverage but not generating authentic fan engagement.",
                "",
            ]
            for _, r in press_only.head(8).iterrows():
                lines.append(f"- **{r['brand']}** — {int(r['press_mentions'])} press | {int(r['reddit_mentions'])} Reddit")
            lines.append("")

        if not fan_organic.empty:
            lines += [
                "## Fan-Organic Brands (Low Press Coverage)",
                "Fans discuss these brands often, but press coverage is low.",
                "Opportunity: brands with authentic fan equity that aren't paying for press placement.",
                "",
            ]
            for _, r in fan_organic.head(8).iterrows():
                lines.append(f"- **{r['brand']}** — {int(r['press_mentions'])} press | {int(r['reddit_mentions'])} Reddit")
            lines.append("")

    lines += [
        "---",
        "",
        "## Figures",
        "- `brand_reddit_volume.png` — Reddit brand mention volume (top 20)",
        "- `brand_reddit_sentiment.png` — How fans feel about each brand",
        "- `brand_press_vs_reddit_volume.png` — Press vs Reddit side-by-side volume",
        "- `brand_press_vs_reddit_sentiment.png` — Sentiment divergence between channels",
        "- `brand_resonance_matrix.png` — Quadrant: press coverage × fan engagement",
        "",
        "---",
        "_Generated by scripts/reddit_brand_analysis.py_",
    ]

    path = REDDIT_DIR / "reddit_brand_report.md"
    path.write_text("\n".join(lines))
    logger.info(f"Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== Reddit Brand Analysis ===")
    df_reddit = scan_reddit_brands()
    comp      = build_comparison(df_reddit)

    chart_reddit_volume(df_reddit)
    chart_reddit_sentiment(df_reddit)
    chart_press_vs_reddit_volume(comp)
    chart_press_vs_reddit_sentiment(comp)
    chart_resonance_matrix(comp)
    write_report(df_reddit, comp)

    logger.info("Done. Outputs:")
    logger.info("  data/analysis/reddit/brand_reddit_earned_media.csv")
    logger.info("  data/analysis/comparison/brand_press_vs_reddit.csv")
    logger.info("  data/analysis/reddit/reddit_brand_report.md")
    logger.info("  charts → data/analysis/reddit/plots/ + data/analysis/comparison/plots/")


if __name__ == "__main__":
    main()
