"""
Data models for the MLS NLP pipeline.

All models are plain dataclasses with to_dict() / from_dict() helpers
so they serialize cleanly to/from pandas DataFrames and Parquet files.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Raw article (output of collection stage)
# ---------------------------------------------------------------------------

@dataclass
class Article:
    """A single collected news article."""

    # Identifiers
    article_id: str = ""          # SHA256 content hash (primary key)
    url: str = ""
    url_hash: str = ""            # SHA256 of normalized URL

    # Source metadata
    source: str = ""              # GDELT | rss:<feed_name>
    domain: str = ""
    feed_name: str = ""

    # Content
    title: str = ""
    text: str = ""                # full text if extracted, else snippet
    snippet: str = ""             # short excerpt from API

    # Dates (ISO 8601 strings for parquet compatibility)
    published_date: str = ""      # e.g. "2021-07-15"
    published_datetime: str = ""  # e.g. "2021-07-15T14:30:00"
    collected_at: str = ""        # when we retrieved it

    # Collection context
    collection_year: int = 0
    collection_month: int = 0
    query_used: str = ""          # GDELT query that found this article

    # Quality flags
    has_full_text: bool = False
    text_length: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Article":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @staticmethod
    def make_url_hash(url: str) -> str:
        normalized = url.lower().strip().rstrip("/")
        # strip common tracking params
        normalized = re.sub(r'[?&](utm_[^&]*|ref=[^&]*|source=[^&]*)', '', normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def make_content_hash(title: str, text: str) -> str:
        combined = (title + " " + text[:500]).lower().strip()
        combined = re.sub(r'\s+', ' ', combined)
        return hashlib.sha256(combined.encode()).hexdigest()[:20]

    def populate_ids(self) -> "Article":
        """Compute and set article_id and url_hash from content."""
        self.url_hash = self.make_url_hash(self.url)
        self.article_id = self.make_content_hash(self.title, self.text or self.snippet)
        self.text_length = len(self.text)
        return self


# ---------------------------------------------------------------------------
# Enriched article (output of NLP stage)
# ---------------------------------------------------------------------------

@dataclass
class EnrichedArticle:
    """An article augmented with NLP-derived structured information."""

    # Carry-forward from Article
    article_id: str = ""
    url: str = ""
    domain: str = ""
    title: str = ""
    published_date: str = ""
    published_datetime: str = ""
    collection_year: int = 0
    collection_month: int = 0
    source: str = ""
    text_length: int = 0

    # NLP outputs — stored as pipe-delimited strings for parquet compatibility
    clubs_mentioned: str = ""        # "Atlanta United FC|LAFC|..."
    players_mentioned: str = ""      # "Xherdan Shaqiri|..."
    coaches_mentioned: str = ""
    executives_mentioned: str = ""
    other_entities: str = ""

    # Event classification
    event_types: str = ""            # "transfer_signing|rumor"
    primary_event_type: str = ""

    # Sentiment
    sentiment_compound: float = 0.0  # VADER compound: -1 to +1
    sentiment_label: str = ""        # positive / neutral / negative
    sentiment_pos: float = 0.0
    sentiment_neg: float = 0.0
    sentiment_neu: float = 0.0

    # Importance signal
    club_mention_count: int = 0      # total club mentions in article
    entity_density: float = 0.0     # entities per 100 words

    # Temporal context
    season_phase: str = ""           # preseason / early_season / mid_season / late_season / playoffs
    transfer_window: str = ""        # winter / summer / none
    season_year: int = 0

    # Enrichment metadata
    enriched_at: str = ""
    enrichment_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EnrichedArticle":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def clubs_list(self) -> list[str]:
        return [c for c in self.clubs_mentioned.split("|") if c]

    def players_list(self) -> list[str]:
        return [p for p in self.players_mentioned.split("|") if p]

    def event_types_list(self) -> list[str]:
        return [e for e in self.event_types.split("|") if e]


# ---------------------------------------------------------------------------
# Network edge (output of network builder stage)
# ---------------------------------------------------------------------------

@dataclass
class NetworkEdge:
    """A weighted edge between two entities in a co-occurrence network."""

    source: str = ""
    target: str = ""
    weight: int = 0
    network_type: str = ""    # club_cooccurrence | club_player | club_entity
    time_window: str = ""     # e.g. "2021" | "2021_Q3" | "2021_07"
    article_count: int = 0   # number of distinct articles supporting this edge

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Centrality record (output of analysis stage)
# ---------------------------------------------------------------------------

@dataclass
class CentralityRecord:
    """Centrality metrics for a single node in a single time window."""

    entity: str = ""
    entity_type: str = ""     # club | player | coach | other
    time_window: str = ""
    network_type: str = ""

    degree_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    pagerank: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    degree: int = 0           # raw degree (number of neighbors)
    total_weight: int = 0     # sum of edge weights

    # Narrative momentum (filled in by analyzer)
    momentum_delta: float = 0.0   # change in pagerank vs previous window
    momentum_label: str = ""      # rising | stable | falling

    def to_dict(self) -> dict:
        return asdict(self)
