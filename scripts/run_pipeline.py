#!/usr/bin/env python
"""
Full end-to-end pipeline runner.

Runs all four stages in sequence:
  1. Collection  → data/press/raw/
  2. Enrichment  → data/press/enriched/
  3. Networks    → data/press/networks/
  4. Analysis    → data/analysis/

Usage examples:

  # Full run, all years
  python scripts/run_pipeline.py

  # Specific years
  python scripts/run_pipeline.py --years 2021 2022

  # Skip collection (start from enrichment)
  python scripts/run_pipeline.py --start-from enrichment

  # Run only analysis on already-built networks
  python scripts/run_pipeline.py --start-from analysis

  # Force re-run all stages
  python scripts/run_pipeline.py --force

  # Print overall status
  python scripts/run_pipeline.py --status
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.collector      import CollectionPipeline
from pipeline.enricher       import EnrichmentPipeline
from pipeline.network_builder import NetworkBuilderPipeline
from pipeline.analyzer       import AnalysisPipeline
from pipeline.utils          import get_logger

logger = get_logger("run_pipeline")

STAGES = ["collection", "enrichment", "networks", "analysis"]


def main():
    parser = argparse.ArgumentParser(description="MLS end-to-end pipeline")
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--start-from", choices=STAGES, default="collection",
                        help="Start from this stage (default: collection)")
    parser.add_argument("--stop-after", choices=STAGES, default="analysis",
                        help="Stop after this stage (default: analysis)")
    parser.add_argument("--force",  action="store_true",
                        help="Force re-run even if already complete")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip full-text extraction during collection")
    parser.add_argument("--windows", nargs="+",
                        choices=["yearly", "quarterly", "monthly"],
                        default=["yearly", "quarterly"])
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    start_idx = STAGES.index(args.start_from)
    stop_idx  = STAGES.index(args.stop_after)

    active_stages = STAGES[start_idx: stop_idx + 1]
    logger.info(f"Pipeline stages to run: {' → '.join(active_stages)}")
    logger.info(f"Target years: {args.years or '2018–2024'}")

    t0 = time.time()

    # ── Stage 1: Collection ──────────────────────────────────────────
    if "collection" in active_stages:
        logger.info("=" * 60)
        logger.info("STAGE 1: COLLECTION")
        logger.info("=" * 60)
        col = CollectionPipeline(
            start_year=2018,
            end_year=2024,
            extract_full_text=not args.no_text,
            skip_completed=not args.force,
        )
        if args.status:
            print(col.status_report().to_string(index=False))
        else:
            total = col.run(years=args.years)
            logger.info(f"Collection done: {total:,} new articles in "
                        f"{time.time() - t0:.1f}s")

    # ── Stage 2: Enrichment ──────────────────────────────────────────
    if "enrichment" in active_stages:
        logger.info("=" * 60)
        logger.info("STAGE 2: NLP ENRICHMENT")
        logger.info("=" * 60)
        enr = EnrichmentPipeline(skip_completed=not args.force)
        if args.status:
            print(enr.status_report().to_string(index=False))
        else:
            total = enr.run(years=args.years)
            logger.info(f"Enrichment done: {total:,} articles in "
                        f"{time.time() - t0:.1f}s")

    # ── Stage 3: Network construction ────────────────────────────────
    if "networks" in active_stages:
        logger.info("=" * 60)
        logger.info("STAGE 3: NETWORK CONSTRUCTION")
        logger.info("=" * 60)
        net = NetworkBuilderPipeline(skip_completed=not args.force)
        net.time_windows = args.windows
        if not args.status:
            net.run(years=args.years)
            logger.info(f"Network construction done in {time.time() - t0:.1f}s")

    # ── Stage 4: Analysis ────────────────────────────────────────────
    if "analysis" in active_stages:
        logger.info("=" * 60)
        logger.info("STAGE 4: CENTRALITY & NARRATIVE ANALYSIS")
        logger.info("=" * 60)
        ana = AnalysisPipeline()
        if not args.status:
            ana.run_all()
            logger.info(f"Analysis done in {time.time() - t0:.1f}s")

    elapsed = time.time() - t0
    logger.info(f"\nPipeline finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
