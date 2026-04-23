"""
Network construction stage.

Transforms enriched article data into weighted co-occurrence graphs.

Network types:
  - club_cooccurrence:  undirected graph where nodes are clubs and
                        edges reflect joint mention in the same article.
  - club_entity:        bipartite graph connecting clubs to the players,
                        coaches, and executives mentioned alongside them.

Time windows:
  - yearly:    one graph per year (2018–2024)
  - quarterly: one graph per quarter (e.g. "2021_Q3")
  - monthly:   one graph per month  (e.g. "2021_07")

Output:
  Each network is saved as two files:
    - <window>_<type>_edges.parquet   — edge list with weights
    - <window>_<type>_graph.json      — NetworkX node-link JSON
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterator, Optional

import networkx as nx
import pandas as pd

from .models import NetworkEdge
from .utils import (
    CheckpointDB,
    get_config,
    get_logger,
    load_all_parquet,
    load_parquet,
    save_parquet,
    year_month_pairs,
    quarter_label,
    PROJECT_ROOT,
)

logger = get_logger("network_builder")


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

class ClubCooccurrenceBuilder:
    """
    Builds a weighted undirected graph where:
      - Nodes = canonical club names
      - Edge weight = number of articles mentioning both clubs
    """

    def __init__(self, min_edge_weight: int = 2):
        self.min_edge_weight = min_edge_weight

    def build(self, df: pd.DataFrame) -> nx.Graph:
        """
        df must have a 'clubs_mentioned' column (pipe-delimited club names).
        """
        G = nx.Graph()
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)
        article_edge_support: dict[tuple[str, str], set[str]] = defaultdict(set)

        for _, row in df.iterrows():
            clubs = [c for c in str(row.get("clubs_mentioned", "")).split("|") if c]
            if len(clubs) < 2:
                continue
            art_id = str(row.get("article_id", ""))
            # All pairs of clubs co-mentioned
            for a, b in combinations(sorted(set(clubs)), 2):
                key = (a, b)
                edge_counts[key] += 1
                if art_id:
                    article_edge_support[key].add(art_id)

        for (a, b), weight in edge_counts.items():
            if weight < self.min_edge_weight:
                continue
            G.add_edge(
                a, b,
                weight=weight,
                article_count=len(article_edge_support[(a, b)]),
            )

        # Node attributes
        for node in G.nodes():
            G.nodes[node]["entity_type"] = "club"

        return G


class ClubEntityBuilder:
    """
    Builds a bipartite graph:
      - Club nodes (type='club')
      - Person nodes (type='player' | 'coach' | 'executive')
      - Edges = co-mention in the same article
    """

    def __init__(self, min_edge_weight: int = 1):
        self.min_edge_weight = min_edge_weight

    def build(self, df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        edge_counts: dict[tuple[str, str], int] = defaultdict(int)

        role_cols = {
            "players_mentioned":    "player",
            "coaches_mentioned":    "coach",
            "executives_mentioned": "executive",
        }

        for _, row in df.iterrows():
            clubs = [c for c in str(row.get("clubs_mentioned", "")).split("|") if c]
            if not clubs:
                continue
            for col, role in role_cols.items():
                persons = [p for p in str(row.get(col, "")).split("|") if p]
                for club in clubs:
                    for person in persons:
                        edge_counts[(club, person)] += 1
                        if club not in G:
                            G.add_node(club, entity_type="club")
                        if person not in G:
                            G.add_node(person, entity_type=role)

        for (a, b), weight in edge_counts.items():
            if weight < self.min_edge_weight:
                continue
            G.add_edge(a, b, weight=weight)

        return G


# ---------------------------------------------------------------------------
# Graph I/O helpers
# ---------------------------------------------------------------------------

def save_graph(G: nx.Graph, directory: Path, filename_stem: str):
    """Save a NetworkX graph as both JSON (node-link) and an edge-list parquet."""
    directory.mkdir(parents=True, exist_ok=True)

    # Node-link JSON
    json_path = directory / f"{filename_stem}.json"
    data = nx.node_link_data(G)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Edge list parquet
    edges = []
    for u, v, attrs in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "weight": attrs.get("weight", 1),
            "article_count": attrs.get("article_count", 0),
        })
    if edges:
        parquet_path = directory / f"{filename_stem}_edges.parquet"
        pd.DataFrame(edges).to_parquet(str(parquet_path), index=False)

    logger.info(f"Saved graph {filename_stem}: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges → {directory}")


def load_graph(path: str | Path) -> nx.Graph:
    """Load a NetworkX graph from a JSON node-link file."""
    path = Path(path)
    if not path.exists():
        return nx.Graph()
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


# ---------------------------------------------------------------------------
# Main network builder pipeline
# ---------------------------------------------------------------------------

class NetworkBuilderPipeline:
    """
    Reads enriched parquet files and builds co-occurrence networks
    at multiple time granularities.
    """

    STAGE = "network_builder"

    def __init__(self, skip_completed: bool = True):
        settings = get_config("settings")
        sources  = get_config("sources")

        data_dir  = PROJECT_ROOT / settings["pipeline"]["data_dir"]
        state_dir = PROJECT_ROOT / settings["pipeline"]["state_dir"]

        self.enriched_dir = data_dir / "press" / "enriched"
        self.networks_dir = data_dir / "press" / "networks"
        self.skip_completed = skip_completed

        net_cfg = settings.get("networks", {})
        self.min_edge_weight    = net_cfg.get("min_edge_weight", 2)
        self.time_windows       = net_cfg.get("time_windows", ["yearly", "quarterly"])
        self.build_cooccurrence = net_cfg.get("build_club_cooccurrence", True)
        self.build_entity       = net_cfg.get("build_club_entity", True)
        self.excluded_domains   = set(net_cfg.get("excluded_domains", []))
        self.max_clubs_per_article = net_cfg.get("max_clubs_per_article", 0)

        self.start_year = sources["collection"]["start_year"]
        self.end_year   = sources["collection"]["end_year"]

        self.db            = CheckpointDB(str(state_dir / "pipeline_state.db"))
        self.cooccurrence  = ClubCooccurrenceBuilder(self.min_edge_weight)
        self.club_entity   = ClubEntityBuilder(min_edge_weight=1)

    def run(self, years: Optional[list[int]] = None):
        """Build all networks for the specified years."""
        target_years = years or list(range(self.start_year, self.end_year + 1))

        for tw in self.time_windows:
            if tw == "yearly":
                for year in target_years:
                    self._build_for_window(str(year), self._load_year(year))

            elif tw == "quarterly":
                for year in target_years:
                    for q in range(1, 5):
                        months = range((q - 1) * 3 + 1, (q - 1) * 3 + 4)
                        label  = f"{year}_Q{q}"
                        df     = self._load_months(year, list(months))
                        self._build_for_window(label, df)

            elif tw == "monthly":
                for year in target_years:
                    for month in range(1, 13):
                        label = f"{year}_{month:02d}"
                        df    = self._load_months(year, [month])
                        self._build_for_window(label, df)

        logger.info("Network construction complete.")

    def run_window(self, window_label: str, df: pd.DataFrame, force: bool = False):
        """Build networks for a single time window from a pre-loaded DataFrame."""
        self._build_for_window(window_label, df, force=force)

    def _filter_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove aggregator articles before building networks."""
        original = len(df)
        if self.excluded_domains and "domain" in df.columns:
            df = df[~df["domain"].isin(self.excluded_domains)]
        if self.max_clubs_per_article > 0 and "club_mention_count" in df.columns:
            df = df[df["club_mention_count"] <= self.max_clubs_per_article]
        removed = original - len(df)
        if removed:
            logger.debug(f"Domain/club filter removed {removed} aggregator articles ({original} → {len(df)})")
        return df

    def _build_for_window(self, label: str, df: pd.DataFrame, force: bool = False):
        if df.empty:
            return

        key = label
        if not force and self.skip_completed and self.db.is_complete(self.STAGE, key):
            logger.debug(f"Skipping completed network window: {label}")
            return

        self.db.set_status(self.STAGE, key, "in_progress")

        try:
            df = self._filter_df(df)
            if df.empty:
                logger.info(f"No articles remain after filtering for {label}")
                self.db.set_status(self.STAGE, key, "complete")
                return

            window_dir = self.networks_dir / label

            if self.build_cooccurrence:
                G = self.cooccurrence.build(df)
                if G.number_of_edges() > 0:
                    save_graph(G, window_dir, f"{label}_club_cooccurrence")
                    self._attach_metadata(G, df, label, "club_cooccurrence")
                else:
                    logger.info(f"No co-occurrence edges for {label}")

            if self.build_entity:
                G2 = self.club_entity.build(df)
                if G2.number_of_edges() > 0:
                    save_graph(G2, window_dir, f"{label}_club_entity")
                else:
                    logger.info(f"No club-entity edges for {label}")

            self.db.set_status(self.STAGE, key, "complete")

        except Exception as e:
            self.db.set_status(self.STAGE, key, "failed", str(e))
            logger.error(f"Network build failed for {label}: {e}")

    def _attach_metadata(self, G: nx.Graph, df: pd.DataFrame, label: str, net_type: str):
        """Save a metadata JSON alongside the graph (article count, date range, etc.)."""
        meta = {
            "window": label,
            "network_type": net_type,
            "article_count": len(df),
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "built_at": datetime.utcnow().isoformat(),
        }
        meta_path = self.networks_dir / label / f"{label}_{net_type}_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_year(self, year: int) -> pd.DataFrame:
        return self._load_months(year, list(range(1, 13)))

    def _load_months(self, year: int, months: list[int]) -> pd.DataFrame:
        frames = []
        for month in months:
            from .utils import get_parquet_path
            path = get_parquet_path(self.enriched_dir, "enriched", year, month)
            df   = load_parquet(path)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def list_graphs(self) -> list[dict]:
        """Return a list of all built graph files with metadata."""
        graphs = []
        for meta_path in sorted(self.networks_dir.glob("**/*_meta.json")):
            with open(meta_path) as f:
                graphs.append(json.load(f))
        return graphs
