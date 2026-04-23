"""
Reddit Pipeline Runner
======================
Runs all stages of the Reddit analysis pipeline:

  Step 1 — Ingest:   Extract zip → normalized monthly parquets in data/reddit/raw/
  Step 2 — Enrich:   NLP enrichment  → data/reddit/enriched/
  Step 3 — Networks: Co-occurrence graphs → data/reddit/networks/
  Step 4 — Analyze:  Centrality + momentum → data/reddit/analysis/

Usage:
    python scripts/run_reddit_pipeline.py
    python scripts/run_reddit_pipeline.py --steps ingest enrich   # partial run
    python scripts/run_reddit_pipeline.py --force                 # ignore checkpoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.utils import get_config, get_logger, get_parquet_path, load_parquet, save_parquet, PROJECT_ROOT
from pipeline.enricher import EnrichmentPipeline
from pipeline.network_builder import NetworkBuilderPipeline
from pipeline.analyzer import AnalysisPipeline

logger = get_logger("reddit_pipeline")

REDDIT_RAW      = PROJECT_ROOT / "data" / "reddit" / "raw"
REDDIT_ENRICHED = PROJECT_ROOT / "data" / "reddit" / "enriched"
REDDIT_NETWORKS = PROJECT_ROOT / "data" / "reddit" / "networks"
REDDIT_ANALYSIS = PROJECT_ROOT / "data" / "reddit" / "analysis"


# ── Subclasses that redirect paths to data/reddit/ ────────────────────────────

class RedditEnrichmentPipeline(EnrichmentPipeline):
    STAGE = "reddit_enricher"

    def __init__(self, force: bool = False):
        super().__init__(skip_completed=not force)
        self.raw_dir      = REDDIT_RAW
        self.enriched_dir = REDDIT_ENRICHED
        REDDIT_ENRICHED.mkdir(parents=True, exist_ok=True)

    def _enrich_month(self, year: int, month: int, force: bool = False) -> int:
        from pipeline.utils import get_parquet_path, load_parquet, save_parquet
        from pipeline.models import EnrichedArticle

        key = f"reddit_{year}_{month:02d}"
        if not force and self.skip_completed and self.db.is_complete(self.STAGE, key):
            return 0

        # Reddit raw files are named YYYY_MM_reddit.parquet
        raw_path = self.raw_dir / str(year) / f"{year}_{month:02d}_reddit.parquet"
        if not raw_path.exists():
            return 0

        raw_df = load_parquet(raw_path)
        if raw_df.empty:
            return 0

        self.db.set_status(self.STAGE, key, "in_progress")
        try:
            enriched = []
            for _, row in raw_df.iterrows():
                title = str(row.get("title", ""))
                text  = str(row.get("text", ""))
                full  = f"{title} {text}".strip()

                extracted  = self.matcher.extract(title, text)
                clubs      = extracted.get("clubs", [])
                players    = extracted.get("players", [])
                coaches    = extracted.get("coaches", [])
                executives = extracted.get("executives", [])

                event_types = self.classifier.classify(title, text)
                primary_event = event_types[0] if event_types else ""

                sent_result = self.sentiment.analyze(full)
                sentiment_compound = sent_result.get("compound", 0)
                sentiment_label    = self.sentiment.label(sentiment_compound)
                pos = sent_result.get("pos", 0)
                neg = sent_result.get("neg", 0)
                neu = sent_result.get("neu", 0)

                temporal_ctx  = self.temporal.label(str(row.get("published_date", "")))
                season_phase  = temporal_ctx.get("season_phase", "")
                transfer_window = temporal_ctx.get("transfer_window", "none")

                enriched.append({
                    "article_id":          row.get("article_id", ""),
                    "url":                 "",
                    "domain":              "reddit.com",
                    "source":              str(row.get("source", "")),
                    "subreddit":           str(row.get("subreddit", "")),
                    "post_type":           str(row.get("post_type", "submission")),
                    "score":               int(row.get("score", 0)),
                    "title":               title,
                    "published_date":      str(row.get("published_date", "")),
                    "published_datetime":  str(row.get("published_datetime", "")),
                    "collection_year":     int(row.get("collection_year", year)),
                    "collection_month":    int(row.get("collection_month", month)),
                    "text_length":         int(row.get("text_length", 0)),
                    "clubs_mentioned":     "|".join(clubs),
                    "players_mentioned":   "|".join(players),
                    "coaches_mentioned":   "|".join(coaches),
                    "executives_mentioned":"|".join(executives),
                    "event_types":         "|".join(event_types),
                    "primary_event_type":  primary_event,
                    "sentiment_compound":  sentiment_compound,
                    "sentiment_label":     sentiment_label,
                    "sentiment_pos":       pos,
                    "sentiment_neg":       neg,
                    "sentiment_neu":       neu,
                    "club_mention_count":  len(clubs),
                    "entity_density":      len(clubs) + len(players) + len(coaches),
                    "season_phase":        season_phase,
                    "transfer_window":     transfer_window,
                    "season_year":         year,
                    "primary_club":        str(row.get("primary_club", "")),
                })

            if not enriched:
                self.db.set_status(self.STAGE, key, "complete")
                return 0

            import pandas as pd
            out_dir = REDDIT_ENRICHED / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{year}_{month:02d}_reddit_enriched.parquet"
            pd.DataFrame(enriched).to_parquet(str(out_path), index=False)
            self.db.set_status(self.STAGE, key, "complete")
            return len(enriched)

        except Exception as e:
            self.db.set_status(self.STAGE, key, "failed", str(e))
            logger.error(f"Reddit enrichment failed {year}-{month:02d}: {e}")
            return 0


class RedditNetworkPipeline(NetworkBuilderPipeline):
    STAGE = "reddit_network"

    def __init__(self, force: bool = False):
        super().__init__(skip_completed=not force)
        self.enriched_dir = REDDIT_ENRICHED
        self.networks_dir = REDDIT_NETWORKS
        REDDIT_NETWORKS.mkdir(parents=True, exist_ok=True)

    def _load_months(self, year: int, months: list[int]):
        import pandas as pd
        frames = []
        for month in months:
            path = REDDIT_ENRICHED / str(year) / f"{year}_{month:02d}_reddit_enriched.parquet"
            if path.exists():
                df = load_parquet(path)
                if not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


class RedditAnalysisPipeline(AnalysisPipeline):
    def __init__(self):
        super().__init__()
        self.networks_dir = REDDIT_NETWORKS
        self.output_dir   = REDDIT_ANALYSIS
        REDDIT_ANALYSIS.mkdir(parents=True, exist_ok=True)


# ── Stage runners ─────────────────────────────────────────────────────────────

def step_ingest():
    from pipeline.reddit_ingestor import RedditIngestor
    logger.info("=== STEP 1: Reddit Ingestion ===")
    ingestor = RedditIngestor()
    total = ingestor.run()
    logger.info(f"Ingestion complete: {total:,} posts saved.")


def step_enrich(force: bool = False):
    logger.info("=== STEP 2: Reddit Enrichment ===")
    pipeline = RedditEnrichmentPipeline(force=force)
    total = 0
    for year in range(2018, 2025):
        for month in range(1, 13):
            n = pipeline._enrich_month(year, month, force=force)
            if n:
                logger.info(f"  {year}-{month:02d}: {n:,} posts enriched")
            total += n
    logger.info(f"Enrichment complete: {total:,} posts total.")


def step_networks(force: bool = False):
    logger.info("=== STEP 3: Reddit Network Construction ===")
    pipeline = RedditNetworkPipeline(force=force)
    pipeline.run(years=list(range(2018, 2025)))
    logger.info("Network construction complete.")


def step_analyze():
    logger.info("=== STEP 4: Reddit Centrality Analysis ===")
    import json
    import networkx as nx
    import pandas as pd
    from pipeline.analyzer import CentralityCalculator, MomentumTracker
    from pipeline.network_builder import load_graph

    REDDIT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    measures = get_config("settings")["analysis"]["centrality_measures"]
    calc     = CentralityCalculator(measures=measures)
    tracker  = MomentumTracker(window_size=3)
    all_records = []

    for window_dir in sorted(REDDIT_NETWORKS.iterdir()):
        if not window_dir.is_dir():
            continue
        label = window_dir.name
        json_path = window_dir / f"{label}_club_cooccurrence.json"
        if not json_path.exists():
            continue
        G = load_graph(json_path)
        if G.number_of_nodes() == 0:
            continue
        df_window = calc.compute(G, label, "club_cooccurrence")
        if not df_window.empty:
            all_records.append(df_window)

    if all_records:
        df = pd.concat(all_records, ignore_index=True)
        df.to_csv(REDDIT_ANALYSIS / "reddit_centrality.csv", index=False)
        logger.info(f"Saved reddit_centrality.csv ({len(df)} rows)")

        mom = tracker.compute(df)
        mom.to_csv(REDDIT_ANALYSIS / "reddit_momentum.csv", index=False)
        logger.info(f"Saved reddit_momentum.csv ({len(mom)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run the Reddit analysis pipeline")
    parser.add_argument("--steps", nargs="+",
                        choices=["ingest", "enrich", "networks", "analyze"],
                        default=["ingest", "enrich", "networks", "analyze"],
                        help="Which steps to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore checkpoints and reprocess everything")
    args = parser.parse_args()

    if "ingest"   in args.steps: step_ingest()
    if "enrich"   in args.steps: step_enrich(force=args.force)
    if "networks" in args.steps: step_networks(force=args.force)
    if "analyze"  in args.steps: step_analyze()

    logger.info("Reddit pipeline complete.")


if __name__ == "__main__":
    main()
