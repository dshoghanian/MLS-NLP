"""
Topic Modeling (LDA)
====================
Runs Latent Dirichlet Allocation on the combined press + Reddit corpus
to reveal *what* MLS discourse is about — not just *how much*.

Topics discovered are labelled by their top words and tracked over time
to show how discourse themes shift across seasons and clubs.

Outputs (data/analysis/shared/):
  - topic_words.csv          — top N words per topic with weights
  - topic_model_results.csv  — per-document topic distributions
  - topic_yearly.csv         — topic prevalence averaged by year
  - topic_club.csv           — dominant topic per club per year

Charts (data/analysis/shared/plots/):
  - topic_words_heatmap.png      — top words per topic
  - topic_yearly_trends.png      — how topics rise/fall 2018–2024
  - topic_club_heatmap.png       — which topics dominate each club
  - topic_press_vs_reddit.png    — topic mix comparison across corpora

Usage:
  python scripts/topic_modeling.py
  python scripts/topic_modeling.py --n-topics 8   # tune number of topics
  python scripts/topic_modeling.py --source press  # press only
  python scripts/topic_modeling.py --source reddit
"""

from __future__ import annotations

import argparse
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

from pipeline.utils import get_logger, PROJECT_ROOT, load_parquet, get_parquet_path

logger = get_logger("topic_modeling")

ANALYSIS_DIR = ROOT / "data" / "analysis"
SHARED_DIR   = ANALYSIS_DIR / "shared"
CHARTS_DIR   = SHARED_DIR / "plots"
SHARED_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

STYLE = {
    "press":   "#1a3a5c",
    "reddit":  "#FF4500",
    "bg":      "#f8f9fa",
    "accent":  "#e84393",
}

plt.rcParams.update({
    "figure.facecolor": STYLE["bg"],
    "axes.facecolor":   STYLE["bg"],
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "DejaVu Sans",
})

# ── Stop words ────────────────────────────────────────────────────────────────
STOP_WORDS = {
    # Standard English
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","shall","can",
    "this","that","these","those","it","its","i","we","he","she","they",
    "their","his","her","our","your","my","said","says","say","from","by",
    "as","not","also","about","after","before","up","out","more","so","then",
    "than","just","into","through","during","against","between","under","over",
    "while","since","until","very","here","there","when","where","who","which",
    "how","what","all","one","two","three","first","last","new","old","big",
    "small","good","get","got","going","make","off","don","think","gonna",
    "let","back","like","some","him","her","time","year","them","you","us",
    "im","ve","ll","re","didn","isn","wasn","weren","shouldn","wouldn","couldn",
    "doesn","nt","its","amp","gt","lt","quot",
    # Day / month abbreviations (date artifacts in press headlines)
    "mon","tue","wed","thu","fri","sat","sun",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
    # Reddit-specific meta words
    "thread","comments","post","reddit","sub","subreddit","crosspost",
    "edit","update","tldr","imo","iirc","afaik","eli5","fwiw","tbh","ngl",
    "meme","lol","lmao","wtf","omg","smh","gg","wp","upvote","downvote",
    "karma","removed","deleted","mod","moderator","bot","automod",
    # Spam / off-topic noise found in corpus
    "anavar","muscle","steroid","bodybuilding","fitness","gym","workout",
    # Too-specific player names that dominate single topics
    "ojeda","rbny","zlatan",
    # Generic conversational fillers not caught by first pass
    "most","even","only","now","well","right","still","need","know",
    "day","night","today","much","many","same","own","too","left","want",
    "look","see","long","way","ever","every","something","nothing","lot",
    # MLS / soccer generic terms (present in nearly every document)
    "mls","major","league","soccer","club","team","season","game","match",
    "www","http","https","com","html","via","re","fc","sc","cf","united",
    "city","sporting","real","inter","football","play","player","players",
    "teams","games","season","seasons","week","weeks","tonight","named",
    # Club names / cities (too universal to discriminate topics)
    "atlanta","austin","charlotte","chicago","dallas","houston","miami",
    "angeles","galaxy","lafc","minnesota","nashville","england","revolution",
    "york","orlando","philadelphia","colorado","rapids","salt","lake","jose",
    "seattle","sounders","kansas","louis","toronto","portland","timbers",
    "vancouver","whitecaps","montreal","crew","columbus","dynamo","cincinnati",
    # Broadcast / media noise
    "watch","live","espn","apple","tv","stream","broadcast","free","per",
    "sports","highlights","goal","goals","score","scored",
    # ── Other North American sports leagues ────────────────────────────────────
    # These appear when MLS articles are published on general sports outlets
    # alongside NFL/NBA/NHL/MLB content; they carry no MLS-specific signal
    "nba","nhl","nfl","mlb","nwsl","ncaa","wnba",
    # Sport-specific jargon that does not apply to soccer
    "basketball","baseball","hockey",
    "bowl","arena","fantasy",            # Super Bowl / arena sports / fantasy sports
    "touchdown","quarterback","rebound","dunk","puck","inning","homerun",
    # ── European club names ────────────────────────────────────────────────────
    # Appear in transfer coverage because MLS imports/exports players from EPL etc.
    # Blocking forces LDA to capture MLS-specific transfer language rather than
    # producing a generic "global soccer transfer" topic
    "chelsea","arsenal","manchester","liverpool","tottenham","everton","leicester",
    "barcelona","madrid","juventus","bayern","psg","dortmund","ajax","porto",
    "premier",                           # "Premier League" — cross-league term
    "bundesliga","laliga","serie",
    # ── Generic noise surviving the first stopword pass ────────────────────────
    "other","because","any","really","people","best","great","love",
    "available","business","group","local","including","google",
    "hills","theatre","links","recap","tifo","angel",
}

# ── Load corpus ───────────────────────────────────────────────────────────────

def load_press_corpus() -> pd.DataFrame:
    settings = __import__("pipeline.utils", fromlist=["get_config"]).get_config("settings")
    data_dir = PROJECT_ROOT / settings["pipeline"]["data_dir"]
    frames = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            raw_path = get_parquet_path(data_dir / "press" / "raw",      "raw",      year, month)
            enr_path = get_parquet_path(data_dir / "press" / "enriched", "enriched", year, month)
            raw = load_parquet(raw_path)
            enr = load_parquet(enr_path)
            if raw.empty or enr.empty:
                continue
            merged = raw[["article_id", "title", "text"]].merge(
                enr[["article_id", "clubs_mentioned", "published_date"]],
                on="article_id", how="inner"
            )
            merged["year"]   = year
            merged["month"]  = month
            merged["source"] = "press"
            merged["full_text"] = (merged["title"].fillna("") + " " +
                                   merged["text"].fillna("")).str.strip()
            frames.append(merged[["article_id","year","month","source",
                                   "clubs_mentioned","full_text"]])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info(f"Press corpus: {len(df):,} articles")
    return df


def load_reddit_corpus() -> pd.DataFrame:
    REDDIT_RAW = ROOT / "data" / "reddit" / "raw"
    frames = []
    for year in range(2018, 2025):
        for month in range(1, 13):
            path = REDDIT_RAW / str(year) / f"{year}_{month:02d}_reddit.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(str(path))
            if df.empty:
                continue
            df["year"]      = year
            df["month"]     = month
            df["source"]    = "reddit"
            df["full_text"] = (df["title"].fillna("") + " " +
                               df["text"].fillna("")).str.strip()
            df["clubs_mentioned"] = df["primary_club"].fillna("")
            frames.append(df[["article_id","year","month","source",
                               "clubs_mentioned","full_text"]])
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info(f"Reddit corpus: {len(df):,} posts")
    return df


# ── Text preprocessing ────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    text  = text.lower()
    text  = re.sub(r"http\S+", "", text)
    text  = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) > 2]


def build_vocabulary(docs: list[list[str]], min_df: int = 5, max_df_ratio: float = 0.85
                     ) -> tuple[dict, list]:
    from collections import Counter
    word_doc_count: Counter = Counter()
    for doc in docs:
        for w in set(doc):
            word_doc_count[w] += 1

    n = len(docs)
    vocab_words = [
        w for w, c in word_doc_count.items()
        if min_df <= c <= max_df_ratio * n
    ]
    vocab = {w: i for i, w in enumerate(sorted(vocab_words))}
    return vocab, vocab_words


def docs_to_bow(docs: list[list[str]], vocab: dict) -> list[list[tuple[int, int]]]:
    from collections import Counter
    bow = []
    for doc in docs:
        counts = Counter(vocab[w] for w in doc if w in vocab)
        bow.append(list(counts.items()))
    return bow


# ── LDA ───────────────────────────────────────────────────────────────────────

def run_lda(corpus: pd.DataFrame, n_topics: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from gensim import corpora, models
        USE_GENSIM = True
    except ImportError:
        USE_GENSIM = False

    logger.info(f"Tokenizing {len(corpus):,} documents ...")
    corpus["tokens"] = corpus["full_text"].apply(tokenize)
    corpus = corpus[corpus["tokens"].apply(len) >= 5].copy()
    logger.info(f"After filtering short docs: {len(corpus):,}")

    if USE_GENSIM:
        dictionary = corpora.Dictionary(corpus["tokens"])
        dictionary.filter_extremes(no_below=5, no_above=0.85)
        bow_corpus  = [dictionary.doc2bow(doc) for doc in corpus["tokens"]]

        logger.info(f"Running LDA: {n_topics} topics, vocab size {len(dictionary):,} ...")
        lda = models.LdaModel(
            bow_corpus,
            num_topics=n_topics,
            id2word=dictionary,
            passes=10,
            alpha="auto",
            random_state=42,
        )

        # Top words per topic
        topic_words_rows = []
        for tid in range(n_topics):
            for word, weight in lda.show_topic(tid, topn=20):
                topic_words_rows.append({"topic_id": tid, "word": word, "weight": weight})
        topic_words_df = pd.DataFrame(topic_words_rows)

        # Per-document topic distributions
        doc_topics = []
        for i, bow in enumerate(bow_corpus):
            dist = dict(lda.get_document_topics(bow, minimum_probability=0))
            row  = {f"topic_{tid}": dist.get(tid, 0.0) for tid in range(n_topics)}
            row["dominant_topic"] = max(dist, key=dist.get) if dist else 0
            doc_topics.append(row)

        topics_meta = corpus[["article_id","year","month","source","clubs_mentioned"]].copy()
        topics_meta = topics_meta.reset_index(drop=True)
        doc_topics_df = pd.DataFrame(doc_topics)
        result_df = pd.concat([topics_meta, doc_topics_df], axis=1)

    else:
        # Fallback: sklearn NMF (no gensim required)
        logger.warning("gensim not found — using sklearn NMF as LDA substitute")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF

        corpus["text_joined"] = corpus["tokens"].apply(" ".join)
        vec = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.85)
        X   = vec.fit_transform(corpus["text_joined"])
        feature_names = vec.get_feature_names_out()

        nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
        W   = nmf.fit_transform(X)
        H   = nmf.components_

        topic_words_rows = []
        for tid in range(n_topics):
            top_idx = H[tid].argsort()[-20:][::-1]
            for rank, idx in enumerate(top_idx):
                topic_words_rows.append({
                    "topic_id": tid,
                    "word":     feature_names[idx],
                    "weight":   float(H[tid, idx]),
                })
        topic_words_df = pd.DataFrame(topic_words_rows)

        # Normalize W rows to get topic proportions
        W_norm = W / (W.sum(axis=1, keepdims=True) + 1e-9)

        topics_meta = corpus[["article_id","year","month","source","clubs_mentioned"]].copy()
        topics_meta = topics_meta.reset_index(drop=True)
        doc_topics_df = pd.DataFrame(
            {f"topic_{i}": W_norm[:, i] for i in range(n_topics)}
        )
        doc_topics_df["dominant_topic"] = W_norm.argmax(axis=1)
        result_df = pd.concat([topics_meta, doc_topics_df], axis=1)

    return topic_words_df, result_df


# ── Auto-label topics from top words ─────────────────────────────────────────

TOPIC_LABELS = {
    frozenset(["transfer","signing","contract","deal","loan","fee","market",
               "window","rumors","move"]): "Transfers & Signings",
    frozenset(["goal","score","win","loss","draw","points","result","final",
               "penalty","extra","time"]): "Match Results",
    frozenset(["injury","injured","return","surgery","fitness","health",
               "week","knee","hamstring","medical"]): "Injuries & Fitness",
    frozenset(["coach","manager","head","fired","appointed","staff","staff",
               "tactical","formation","lineup"]): "Coaching & Tactics",
    frozenset(["playoff","cup","final","championship","mls","conference",
               "supporter","shield","knockout"]): "Playoffs & Championships",
    frozenset(["fan","support","fans","ticket","stadium","home","away",
               "crowd","atmosphere","sold","out"]): "Fan Culture & Attendance",
    frozenset(["value","salary","wage","budget","spend","ownership","invest",
               "franchise","revenue","worth"]): "Finance & Ownership",
}


def assign_label(top_words: list[str]) -> str:
    word_set = set(top_words)
    best_label, best_score = "General MLS News", 0
    for keyword_set, label in TOPIC_LABELS.items():
        score = len(keyword_set & word_set)
        if score > best_score:
            best_score, best_label = score, label
    return best_label


def ensure_unique_labels(labels: dict[int, str]) -> dict[int, str]:
    """Append (2), (3) etc. to duplicate label names so columns are always unique."""
    seen: dict[str, int] = {}
    result = {}
    for tid, label in sorted(labels.items()):
        if label in seen:
            seen[label] += 1
            result[tid] = f"{label} ({seen[label]})"
        else:
            seen[label] = 1
            result[tid] = label
    return result


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_topic_words(topic_words_df: pd.DataFrame, labels: dict[int, str]):
    n_topics = topic_words_df["topic_id"].nunique()
    top10    = (topic_words_df.groupby("topic_id")
                .apply(lambda g: g.nlargest(10, "weight"))
                .reset_index(drop=True))

    pivot = top10.pivot_table(index="word", columns="topic_id",
                              values="weight", aggfunc="sum").fillna(0)

    fig, ax = plt.subplots(figsize=(max(10, n_topics * 1.5), 9))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_topics))
    ax.set_xticklabels([f"T{i}\n{labels.get(i,'')[:18]}" for i in range(n_topics)],
                       fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="Word weight in topic")
    ax.set_title("Topic Model: Top Words per Topic (LDA/NMF)\n"
                 "Darker = word more characteristic of that topic",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "topic_words_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved topic_words_heatmap.png")


def chart_yearly_trends(result_df: pd.DataFrame, labels: dict[int, str]):
    topic_cols = [c for c in result_df.columns if c.startswith("topic_")]
    yearly     = result_df.groupby("year")[topic_cols].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.get_cmap("tab10", len(topic_cols))

    for i, col in enumerate(topic_cols):
        tid   = int(col.split("_")[1])
        label = labels.get(tid, f"Topic {tid}")
        ax.plot(yearly.index, yearly[col], marker="o", ms=5, lw=2,
                color=cmap(i), label=label)

    ax.set_xlabel("Season")
    ax.set_ylabel("Avg topic proportion")
    ax.set_title("Topic Trends Over Time (2018–2024)\nHow discourse themes shift by season",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2, bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "topic_yearly_trends.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved topic_yearly_trends.png")


def chart_club_topics(result_df: pd.DataFrame, labels: dict[int, str]):
    topic_cols = [c for c in result_df.columns if c.startswith("topic_")]

    # Expand clubs_mentioned (pipe-separated) and keep rows with a club
    exploded = result_df.copy()
    exploded["club"] = exploded["clubs_mentioned"].str.split("|")
    exploded = exploded.explode("club")
    exploded = exploded[exploded["club"].str.strip() != ""]

    club_topic = exploded.groupby("club")[topic_cols].mean()
    # Keep top 18 clubs by document count
    club_counts = exploded.groupby("club").size()
    top_clubs   = club_counts.nlargest(18).index
    club_topic  = club_topic.loc[club_topic.index.isin(top_clubs)]

    if club_topic.empty:
        logger.warning("No club-level topic data — skipping club heatmap")
        return

    club_topic.columns = [labels.get(int(c.split("_")[1]), c) for c in club_topic.columns]

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(club_topic, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.3, ax=ax)
    ax.set_title("Dominant Discourse Topics per Club\n"
                 "Avg topic proportion across all documents mentioning each club",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Club")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "topic_club_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved topic_club_heatmap.png")


def chart_press_vs_reddit(result_df: pd.DataFrame, labels: dict[int, str]):
    if "source" not in result_df.columns:
        return
    topic_cols = [c for c in result_df.columns if c.startswith("topic_")]
    by_source  = result_df.groupby("source")[topic_cols].mean()

    if len(by_source) < 2:
        return

    x     = np.arange(len(topic_cols))
    width = 0.38
    topic_labels = [labels.get(int(c.split("_")[1]), f"T{c.split('_')[1]}") for c in topic_cols]

    fig, ax = plt.subplots(figsize=(13, 6))
    if "press" in by_source.index:
        ax.bar(x - width/2, by_source.loc["press", topic_cols],
               width, color=STYLE["press"],  alpha=0.85, label="Press",  edgecolor="white")
    if "reddit" in by_source.index:
        ax.bar(x + width/2, by_source.loc["reddit", topic_cols],
               width, color=STYLE["reddit"], alpha=0.85, label="Reddit", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(topic_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Avg topic proportion")
    ax.set_title("Topic Mix: Press vs Reddit Fan Discourse\n"
                 "Which topics dominate each channel?",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "topic_press_vs_reddit.png", dpi=150)
    plt.close(fig)
    logger.info("Saved topic_press_vs_reddit.png")


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_outputs(topic_words_df: pd.DataFrame, result_df: pd.DataFrame,
                 labels: dict[int, str]):
    topic_words_df["topic_label"] = topic_words_df["topic_id"].map(labels)
    topic_words_df.to_csv(SHARED_DIR / "topic_words.csv", index=False)

    result_df.to_csv(SHARED_DIR / "topic_model_results.csv", index=False)

    topic_cols = [c for c in result_df.columns if c.startswith("topic_")]
    yearly = (result_df.groupby(["year","source"])[topic_cols]
              .mean().reset_index())
    yearly.columns = (list(yearly.columns[:2]) +
                      [labels.get(int(c.split("_")[1]), c) for c in topic_cols])
    yearly.to_csv(SHARED_DIR / "topic_yearly.csv", index=False)

    # Dominant topic per club per year
    exploded = result_df.copy()
    exploded["club"] = exploded["clubs_mentioned"].str.split("|")
    exploded = exploded.explode("club")
    exploded = exploded[exploded["club"].str.strip() != ""]
    if not exploded.empty:
        club_year = (exploded.groupby(["club","year","source"])["dominant_topic"]
                     .agg(lambda x: x.mode()[0] if len(x) > 0 else 0)
                     .reset_index())
        club_year["topic_label"] = club_year["dominant_topic"].map(labels)
        club_year.to_csv(SHARED_DIR / "topic_club.csv", index=False)

    logger.info("Saved topic_words.csv, topic_model_results.csv, topic_yearly.csv, topic_club.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LDA topic modeling on MLS corpus")
    parser.add_argument("--n-topics", type=int, default=7,
                        help="Number of LDA topics (default: 7)")
    parser.add_argument("--source", choices=["press", "reddit", "both"], default="both",
                        help="Which corpus to model (default: both)")
    args = parser.parse_args()

    logger.info(f"=== Topic Modeling (n_topics={args.n_topics}, source={args.source}) ===")

    frames = []
    if args.source in ("press", "both"):
        frames.append(load_press_corpus())
    if args.source in ("reddit", "both"):
        frames.append(load_reddit_corpus())

    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus[corpus["full_text"].str.len() >= 30].copy()
    logger.info(f"Combined corpus: {len(corpus):,} documents")

    topic_words_df, result_df = run_lda(corpus, n_topics=args.n_topics)

    # Auto-label topics
    labels: dict[int, str] = {}
    for tid in topic_words_df["topic_id"].unique():
        top_words = (topic_words_df[topic_words_df["topic_id"] == tid]
                     .nlargest(10, "weight")["word"].tolist())
        labels[tid] = assign_label(top_words)
        logger.info(f"  Topic {tid} [{labels[tid]}]: {', '.join(top_words[:7])}")

    labels = ensure_unique_labels(labels)
    save_outputs(topic_words_df, result_df, labels)
    chart_topic_words(topic_words_df, labels)
    chart_yearly_trends(result_df, labels)
    chart_club_topics(result_df, labels)
    chart_press_vs_reddit(result_df, labels)

    logger.info("Done.")
    logger.info("Outputs: topic_words.csv, topic_model_results.csv, "
                "topic_yearly.csv, topic_club.csv")
    logger.info("Charts:  topic_words_heatmap.png, topic_yearly_trends.png, "
                "topic_club_heatmap.png, topic_press_vs_reddit.png")


if __name__ == "__main__":
    main()
