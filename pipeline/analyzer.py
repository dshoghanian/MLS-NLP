"""
Analysis stage: centrality metrics and narrative momentum tracking.

Computes graph-theoretic centrality measures for each time window and
derives narrative momentum — the rate of change in a club's or entity's
centrality position relative to prior windows.

Outputs:
  - data/analysis/centrality_<window_type>.csv
  - data/analysis/momentum_<window_type>.csv
  - data/analysis/narrative_summary_<year>.csv

Centrality measures:
  degree         — how many distinct entities a club co-occurs with
  eigenvector    — importance weighted by importance of neighbors
  pagerank       — Google-style importance (directed or undirected)
  betweenness    — how often a club sits on the shortest path between others
  closeness      — inverse average distance to all other nodes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
import numpy as np

from .utils import (
    CheckpointDB,
    get_config,
    get_logger,
    save_parquet,
    PROJECT_ROOT,
)
from .network_builder import load_graph

logger = get_logger("analyzer")


# ---------------------------------------------------------------------------
# Centrality calculator
# ---------------------------------------------------------------------------

class CentralityCalculator:
    """
    Computes multiple centrality measures on a NetworkX graph.
    Falls back gracefully when a measure cannot converge.
    """

    def __init__(self, measures: list[str]):
        self.measures = measures

    def compute(self, G: nx.Graph, window: str, network_type: str) -> pd.DataFrame:
        """Return a DataFrame with one row per node, all centrality measures."""
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        results: dict[str, dict] = {node: {"entity": node} for node in G.nodes()}

        # Add entity_type from node attributes
        for node in G.nodes():
            attrs = G.nodes[node]
            results[node]["entity_type"] = attrs.get("entity_type", "unknown")

        # Degree
        if "degree" in self.measures:
            deg = nx.degree_centrality(G)
            raw_deg = dict(G.degree())
            weight_deg = dict(G.degree(weight="weight"))
            for node in G.nodes():
                results[node]["degree_centrality"] = round(deg.get(node, 0.0), 6)
                results[node]["degree"] = raw_deg.get(node, 0)
                results[node]["total_weight"] = weight_deg.get(node, 0)

        # Eigenvector
        if "eigenvector" in self.measures:
            try:
                eig = nx.eigenvector_centrality_numpy(G, weight="weight")
                for node in G.nodes():
                    results[node]["eigenvector_centrality"] = round(eig.get(node, 0.0), 6)
            except Exception as e:
                logger.debug(f"Eigenvector centrality failed for {window}: {e}")
                for node in G.nodes():
                    results[node]["eigenvector_centrality"] = 0.0

        # PageRank
        if "pagerank" in self.measures:
            try:
                pr = nx.pagerank(G, weight="weight", alpha=0.85)
                for node in G.nodes():
                    results[node]["pagerank"] = round(pr.get(node, 0.0), 6)
            except Exception as e:
                logger.debug(f"PageRank failed for {window}: {e}")
                for node in G.nodes():
                    results[node]["pagerank"] = 0.0

        # Betweenness
        if "betweenness" in self.measures:
            try:
                bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
                for node in G.nodes():
                    results[node]["betweenness_centrality"] = round(bet.get(node, 0.0), 6)
            except Exception:
                for node in G.nodes():
                    results[node]["betweenness_centrality"] = 0.0

        # Closeness
        if "closeness" in self.measures:
            try:
                clo = nx.closeness_centrality(G)
                for node in G.nodes():
                    results[node]["closeness_centrality"] = round(clo.get(node, 0.0), 6)
            except Exception:
                for node in G.nodes():
                    results[node]["closeness_centrality"] = 0.0

        df = pd.DataFrame(list(results.values()))
        df["time_window"]   = window
        df["network_type"]  = network_type
        df["momentum_delta"] = 0.0
        df["momentum_label"] = "stable"
        return df


# ---------------------------------------------------------------------------
# Narrative momentum tracker
# ---------------------------------------------------------------------------

class MomentumTracker:
    """
    Computes narrative momentum as the rolling change in PageRank
    across consecutive time windows.

    momentum_delta = current_pagerank - prior_pagerank (smoothed)
    momentum_label = rising | stable | falling
    """

    def __init__(self, window_size: int = 3):
        self.window_size = window_size

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: centrality DataFrame with columns [entity, time_window, pagerank, ...]
        Returns df with momentum_delta and momentum_label filled in.
        """
        if "pagerank" not in df.columns or "time_window" not in df.columns:
            return df

        df = df.copy()
        df["momentum_delta"] = 0.0
        df["momentum_label"] = "stable"

        # Sort windows chronologically
        windows = sorted(df["time_window"].unique())
        if len(windows) < 2:
            return df

        for i, window in enumerate(windows):
            if i < 1:
                continue
            prior_windows = windows[max(0, i - self.window_size): i]
            mask_current  = df["time_window"] == window
            current_nodes = df[mask_current]

            for _, row in current_nodes.iterrows():
                entity = row["entity"]
                # Average pagerank over prior windows
                prior_data = df[
                    (df["entity"] == entity) &
                    (df["time_window"].isin(prior_windows))
                ]["pagerank"]
                if prior_data.empty:
                    continue
                prior_mean = prior_data.mean()
                delta = round(row["pagerank"] - prior_mean, 6)
                df.loc[(df["time_window"] == window) & (df["entity"] == entity),
                       "momentum_delta"] = delta

        # Label momentum
        threshold = 0.002
        df.loc[df["momentum_delta"] >  threshold, "momentum_label"] = "rising"
        df.loc[df["momentum_delta"] < -threshold, "momentum_label"] = "falling"

        return df


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

class AnalysisPipeline:
    """
    Loads all built graphs, computes centrality metrics, derives narrative
    momentum, and writes output CSV/parquet files.
    """

    STAGE = "analyzer"

    def __init__(self):
        settings = get_config("settings")
        sources  = get_config("sources")

        data_dir  = PROJECT_ROOT / settings["pipeline"]["data_dir"]
        state_dir = PROJECT_ROOT / settings["pipeline"]["state_dir"]

        self.networks_dir  = data_dir / "press" / "networks"
        self.analysis_dir  = data_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        analysis_cfg = settings.get("analysis", {})
        measures     = analysis_cfg.get("centrality_measures",
                           ["degree", "eigenvector", "pagerank", "betweenness"])
        momentum_win = analysis_cfg.get("momentum_window", 3)
        self.output_fmt  = analysis_cfg.get("output_format", "csv")

        self.start_year = sources["collection"]["start_year"]
        self.end_year   = sources["collection"]["end_year"]

        self.db         = CheckpointDB(str(state_dir / "pipeline_state.db"))
        self.calculator = CentralityCalculator(measures)
        self.momentum   = MomentumTracker(momentum_win)

    def run(self, network_type: str = "club_cooccurrence"):
        """Compute centrality and momentum for all time windows of a given network type."""
        all_frames = []

        # Discover all built graphs matching the network_type
        graph_paths = sorted(self.networks_dir.glob(f"**/*_{network_type}.json"))
        if not graph_paths:
            logger.warning(f"No graphs found for network_type='{network_type}'")
            return pd.DataFrame()

        for path in graph_paths:
            window = path.parent.name  # directory name is the time window label
            logger.info(f"Computing centrality for {window} ({network_type})")
            try:
                G = load_graph(path)
                df = self.calculator.compute(G, window, network_type)
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                logger.error(f"Analysis failed for {window}: {e}")

        if not all_frames:
            logger.warning("No centrality data computed.")
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)

        # Compute momentum
        combined = self.momentum.compute(combined)

        # Save
        out_stem = f"centrality_{network_type}"
        self._save(combined, out_stem)

        # Also save narrative summary (clubs only, yearly windows)
        summary = self._build_narrative_summary(combined, network_type)
        if not summary.empty:
            self._save(summary, f"narrative_summary_{network_type}")

        logger.info(f"Analysis complete. {len(combined)} centrality records.")
        return combined

    def run_all(self):
        """Run analysis for all network types."""
        results = {}
        for net_type in ["club_cooccurrence", "club_entity"]:
            results[net_type] = self.run(network_type=net_type)
        return results

    def _build_narrative_summary(
        self,
        df: pd.DataFrame,
        network_type: str,
    ) -> pd.DataFrame:
        """
        Build a summary table for clubs only, with one row per (club, time_window).
        Includes narrative signal: overhyped = high centrality + (external) low performance.
        """
        clubs_only = df[df["entity_type"] == "club"].copy() if "entity_type" in df.columns else df.copy()
        if clubs_only.empty:
            return pd.DataFrame()

        # Rank clubs by pagerank within each window
        clubs_only["pr_rank"] = clubs_only.groupby("time_window")["pagerank"].rank(
            ascending=False, method="min"
        )

        # Momentum summary per year (aggregate quarterly/monthly into yearly view)
        clubs_only["year"] = clubs_only["time_window"].str[:4]

        yearly = (
            clubs_only.groupby(["entity", "year"])
            .agg(
                avg_pagerank=("pagerank", "mean"),
                max_pagerank=("pagerank", "max"),
                avg_degree_centrality=("degree_centrality", "mean"),
                avg_momentum=("momentum_delta", "mean"),
                windows_rising=(
                    "momentum_label",
                    lambda x: (x == "rising").sum()
                ),
                windows_falling=(
                    "momentum_label",
                    lambda x: (x == "falling").sum()
                ),
            )
            .reset_index()
        )
        yearly["net_momentum_signal"] = yearly["windows_rising"] - yearly["windows_falling"]
        yearly["narrative_trend"] = yearly["net_momentum_signal"].apply(
            lambda x: "rising" if x > 1 else ("falling" if x < -1 else "stable")
        )
        yearly["network_type"] = network_type
        return yearly

    def top_clubs(
        self,
        window: str,
        network_type: str = "club_cooccurrence",
        n: int = 10,
        metric: str = "pagerank",
    ) -> pd.DataFrame:
        """
        Return the top-N clubs by a given metric for a time window.
        Loads from the saved centrality file.
        """
        path = self.analysis_dir / f"centrality_{network_type}.csv"
        if not path.exists():
            logger.warning("No centrality data found. Run analysis first.")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df = df[(df["time_window"] == window) & (df.get("entity_type", "club") == "club")]
        if metric not in df.columns:
            metric = "pagerank"
        return df.nlargest(n, metric)[
            ["entity", "time_window", metric, "degree", "total_weight", "momentum_label"]
        ].reset_index(drop=True)

    def misalignment_report(
        self,
        performance_csv: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Compare narrative centrality with actual on-field performance.

        performance_csv: path to a CSV with columns [club, year, points] (or similar).
                         If not provided, only narrative data is returned.
        Returns a DataFrame with narrative rank, optional performance rank, and gap.
        """
        centrality_path = self.analysis_dir / "centrality_club_cooccurrence.csv"
        if not centrality_path.exists():
            logger.warning("No centrality data. Run analysis first.")
            return pd.DataFrame()

        df = pd.read_csv(centrality_path)
        if year:
            df = df[df["time_window"] == str(year)]

        clubs = df[df.get("entity_type", pd.Series(["club"] * len(df))) == "club"].copy()
        clubs["narrative_rank"] = clubs.groupby("time_window")["pagerank"].rank(
            ascending=False, method="min"
        )

        if not performance_csv:
            return clubs[["entity", "time_window", "pagerank", "narrative_rank",
                          "momentum_label"]].reset_index(drop=True)

        # Merge with performance data
        perf = pd.read_csv(performance_csv)
        merged = clubs.merge(perf, left_on=["entity", "time_window"],
                             right_on=["club", "year"], how="left")
        if "points" in merged.columns:
            merged["performance_rank"] = merged.groupby("time_window")["points"].rank(
                ascending=False, method="min"
            )
            merged["narrative_vs_performance_gap"] = (
                merged["performance_rank"] - merged["narrative_rank"]
            )
        return merged.reset_index(drop=True)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame, stem: str):
        if self.output_fmt == "parquet":
            path = self.analysis_dir / f"{stem}.parquet"
            df.to_parquet(str(path), index=False)
        else:
            path = self.analysis_dir / f"{stem}.csv"
            df.to_csv(str(path), index=False)
        logger.info(f"Saved analysis → {path}")
