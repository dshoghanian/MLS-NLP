#!/usr/bin/env python
"""
NLP enrichment script.

Usage examples:

  # Enrich all years
  python scripts/run_enrichment.py

  # Enrich specific years
  python scripts/run_enrichment.py --years 2021 2022

  # Force re-enrich
  python scripts/run_enrichment.py --years 2021 --force

  # Print status
  python scripts/run_enrichment.py --status
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.enricher import EnrichmentPipeline
from pipeline.utils import get_logger

logger = get_logger("run_enrichment")


def main():
    parser = argparse.ArgumentParser(description="MLS NLP enrichment pipeline")
    parser.add_argument("--years", type=int, nargs="+",
                        help="Years to enrich (default: all)")
    parser.add_argument("--year",  type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    pipeline = EnrichmentPipeline(skip_completed=not args.force)

    if args.status:
        report = pipeline.status_report()
        print(report.to_string(index=False))
        complete = (report["status"] == "complete").sum()
        total    = len(report)
        count    = report["enriched_count"].sum()
        print(f"\nProgress: {complete}/{total} months enriched | {count:,} articles")
        return

    if args.year and args.month:
        logger.info(f"Enriching {args.year}-{args.month:02d}")
        n = pipeline.run_month(args.year, args.month, force=args.force)
        logger.info(f"Done: {n} articles enriched")
    else:
        years = args.years or None
        logger.info(f"Starting enrichment for years: {years or 'all'}")
        total = pipeline.run(years=years)
        logger.info(f"Enrichment complete. Total: {total:,} articles")


if __name__ == "__main__":
    main()
