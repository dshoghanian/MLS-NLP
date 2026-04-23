#!/usr/bin/env python
"""
Article collection script.

Usage examples:

  # Collect all years (2018–2024)
  python scripts/run_collection.py

  # Collect specific years
  python scripts/run_collection.py --years 2021 2022

  # Collect a single month
  python scripts/run_collection.py --year 2021 --month 7

  # Force re-collect even if already marked complete
  python scripts/run_collection.py --years 2021 --force

  # Disable full-text extraction (faster, metadata only)
  python scripts/run_collection.py --no-text

  # Print status report without collecting
  python scripts/run_collection.py --status
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.collector import CollectionPipeline
from pipeline.utils import get_logger

logger = get_logger("run_collection")


def main():
    parser = argparse.ArgumentParser(description="MLS article collection pipeline")
    parser.add_argument("--years", type=int, nargs="+",
                        help="Years to collect (default: 2018–2024)")
    parser.add_argument("--year", type=int, help="Single year (use with --month)")
    parser.add_argument("--month", type=int, help="Single month (1–12)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if already marked complete")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip full-text extraction")
    parser.add_argument("--status", action="store_true",
                        help="Print status report and exit")
    parser.add_argument("--retry-failed", action="store_true", default=True,
                        help="Retry previously failed months (default: True)")
    args = parser.parse_args()

    pipeline = CollectionPipeline(
        start_year=2018,
        end_year=2024,
        extract_full_text=not args.no_text,
        skip_completed=not args.force,
    )

    if args.status:
        report = pipeline.status_report()
        print(report.to_string(index=False))
        complete = (report["status"] == "complete").sum()
        total    = len(report)
        count    = report["article_count"].sum()
        print(f"\nProgress: {complete}/{total} months complete | {count:,} articles collected")
        return

    if args.year and args.month:
        logger.info(f"Collecting {args.year}-{args.month:02d}")
        n = pipeline.run_month(args.year, args.month, force=args.force)
        logger.info(f"Done: {n} new articles")
    else:
        years = args.years or None
        logger.info(f"Starting collection for years: {years or '2018–2024'}")
        total = pipeline.run(years=years, retry_failed=args.retry_failed)
        logger.info(f"Collection complete. Total new articles: {total:,}")


if __name__ == "__main__":
    main()
