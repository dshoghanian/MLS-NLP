#!/usr/bin/env python
"""
Centrality and narrative analysis script.

Usage examples:

  # Full analysis on club co-occurrence networks
  python scripts/run_analysis.py

  # Also analyze club-entity networks
  python scripts/run_analysis.py --all-networks

  # Show top 10 clubs by PageRank for a given window
  python scripts/run_analysis.py --top --window 2022 --n 10

  # Show narrative momentum for a specific year
  python scripts/run_analysis.py --momentum --year 2022

  # Misalignment report (narrative vs performance)
  python scripts/run_analysis.py --misalignment --perf data/performance.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.analyzer import AnalysisPipeline
from pipeline.utils import get_logger

logger = get_logger("run_analysis")


def main():
    parser = argparse.ArgumentParser(description="MLS centrality and narrative analysis")
    parser.add_argument("--all-networks",  action="store_true",
                        help="Analyze all network types (default: club_cooccurrence only)")
    parser.add_argument("--network",       default="club_cooccurrence",
                        help="Network type to analyze")
    parser.add_argument("--top",           action="store_true",
                        help="Print top-N clubs for a time window")
    parser.add_argument("--window",        type=str,
                        help="Time window label, e.g. '2022' or '2022_Q3'")
    parser.add_argument("--n",             type=int, default=10)
    parser.add_argument("--metric",        default="pagerank",
                        help="Metric for --top ranking")
    parser.add_argument("--momentum",      action="store_true",
                        help="Print momentum summary")
    parser.add_argument("--year",          type=int)
    parser.add_argument("--misalignment",  action="store_true")
    parser.add_argument("--perf",          type=str,
                        help="Path to performance CSV for misalignment report")
    args = parser.parse_args()

    pipeline = AnalysisPipeline()

    # Default: run full analysis
    if args.all_networks:
        results = pipeline.run_all()
    elif not (args.top or args.momentum or args.misalignment):
        results = pipeline.run(network_type=args.network)
        if not results.empty:
            logger.info(f"Analysis complete: {len(results)} centrality records")

    # Top-N clubs
    if args.top:
        if not args.window:
            print("--window required with --top (e.g. --window 2022)")
            sys.exit(1)
        df = pipeline.top_clubs(
            window=args.window,
            network_type=args.network,
            n=args.n,
            metric=args.metric,
        )
        if df.empty:
            print(f"No data for window={args.window}. Run analysis first.")
        else:
            print(f"\nTop {args.n} clubs — {args.window} — {args.metric}")
            print(df.to_string(index=False))

    # Momentum summary
    if args.momentum:
        import pandas as pd
        from pathlib import Path as P
        from pipeline.utils import PROJECT_ROOT
        path = PROJECT_ROOT / "data" / "analysis" / f"centrality_{args.network}.csv"
        if not path.exists():
            print("No centrality data. Run analysis first.")
            sys.exit(1)
        df = pd.read_csv(path)
        if args.year:
            df = df[df["time_window"].str.startswith(str(args.year))]
        if "entity_type" in df.columns:
            df = df[df["entity_type"] == "club"]
        summary = (
            df.groupby(["entity", "momentum_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        print(f"\nNarrative Momentum Summary{' — ' + str(args.year) if args.year else ''}")
        print(summary.to_string(index=False))

    # Misalignment report
    if args.misalignment:
        df = pipeline.misalignment_report(
            performance_csv=args.perf,
            year=args.year,
        )
        print("\nNarrative vs. Performance Misalignment")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
