#!/usr/bin/env python
"""
Network construction script.

Usage examples:

  # Build all networks (yearly + quarterly by default)
  python scripts/run_networks.py

  # Build for specific years
  python scripts/run_networks.py --years 2021 2022

  # Build only yearly networks
  python scripts/run_networks.py --windows yearly

  # Force rebuild
  python scripts/run_networks.py --force

  # List built graphs
  python scripts/run_networks.py --list
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.network_builder import NetworkBuilderPipeline
from pipeline.utils import get_logger, get_config

logger = get_logger("run_networks")


def main():
    parser = argparse.ArgumentParser(description="MLS network construction pipeline")
    parser.add_argument("--years",   type=int, nargs="+")
    parser.add_argument("--windows", nargs="+",
                        choices=["yearly", "quarterly", "monthly"],
                        help="Time window granularities to build")
    parser.add_argument("--force",  action="store_true")
    parser.add_argument("--list",   action="store_true",
                        help="List all built graphs and exit")
    args = parser.parse_args()

    pipeline = NetworkBuilderPipeline(skip_completed=not args.force)

    if args.list:
        graphs = pipeline.list_graphs()
        if not graphs:
            print("No graphs built yet.")
        for g in graphs:
            print(f"  {g['window']:12s}  {g['network_type']:25s}  "
                  f"{g['node_count']:4d} nodes  {g['edge_count']:5d} edges  "
                  f"({g['article_count']} articles)")
        return

    # Override time windows if specified
    if args.windows:
        pipeline.time_windows = args.windows

    years = args.years or None
    logger.info(f"Building networks for years={years or 'all'}, "
                f"windows={pipeline.time_windows}")
    pipeline.run(years=years)
    logger.info("Network construction complete.")


if __name__ == "__main__":
    main()
