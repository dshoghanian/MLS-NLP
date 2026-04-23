"""
NLP enrichment stage.

For each raw article, this stage produces an EnrichedArticle containing:
  - Clubs, players, coaches, executives, and other entities mentioned
  - Primary and secondary event types
  - VADER sentiment scores
  - Temporal context (season phase, transfer window)
  - Entity density and importance signals

Architecture:
  - EntityMatcher   — combines spaCy NER with custom MLS entity lookup
  - EventClassifier — keyword-based event type detection
  - SentimentAnalyzer — VADER sentiment scoring
  - TemporalContexter — season/window labeling from article date
  - EnrichmentPipeline — orchestrates enrichment batch by batch
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

import pandas as pd

from .models import Article, EnrichedArticle
from .utils import (
    CheckpointDB,
    get_config,
    get_logger,
    get_parquet_path,
    iter_parquet_files,
    load_parquet,
    save_parquet,
    year_month_pairs,
    PROJECT_ROOT,
)

logger = get_logger("enricher")


# ---------------------------------------------------------------------------
# Entity Matcher
# ---------------------------------------------------------------------------

class EntityMatcher:
    """
    Matches MLS entities in article text using:
      1. Custom dictionary lookup (exact + alias matching)
      2. spaCy NER for general persons and organizations
    """

    def __init__(self, entities_cfg: dict, spacy_model: str = "en_core_web_sm"):
        self.clubs_cfg    = entities_cfg.get("clubs", {})
        self.league_ents  = set(entities_cfg.get("league_entities", []))
        self._build_club_index()
        self._nlp = self._load_spacy(spacy_model)

    def _build_club_index(self):
        """Build a flat alias→canonical_name lookup."""
        self._alias_map: dict[str, str] = {}
        self._abbreviations: dict[str, str] = {}

        for canonical, info in self.clubs_cfg.items():
            # Map canonical name itself
            self._alias_map[canonical.lower()] = canonical
            # Map all aliases
            for alias in info.get("aliases", []):
                self._alias_map[alias.lower()] = canonical
            # Map abbreviation
            abbr = info.get("abbreviation", "")
            if abbr:
                self._abbreviations[abbr.upper()] = canonical

        # Compiled regex for fast multi-pattern matching
        # Longest patterns first to avoid partial matches
        patterns = sorted(self._alias_map.keys(), key=len, reverse=True)
        escaped  = [re.escape(p) for p in patterns]
        self._club_re = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )
        # Abbreviation pattern (uppercase only to reduce false positives)
        abbr_patterns = sorted(self._abbreviations.keys(), key=len, reverse=True)
        if abbr_patterns:
            self._abbr_re = re.compile(
                r'\b(' + '|'.join(re.escape(a) for a in abbr_patterns) + r')\b'
            )
        else:
            self._abbr_re = None

    @staticmethod
    def _load_spacy(model: str):
        try:
            import spacy
            return spacy.load(model, disable=["parser", "lemmatizer"])
        except Exception as e:
            logger.warning(f"Could not load spaCy model '{model}': {e}. NER will use dict only.")
            return None

    def extract(self, title: str, text: str) -> dict[str, list[str]]:
        """
        Returns a dict with keys:
          clubs, players, coaches, executives, other_orgs
        """
        full_text = f"{title} {text}"
        clubs     = self._match_clubs(full_text)
        persons, orgs = self._spacy_ner(full_text)

        # Remove any clubs that spaCy mis-tagged as PERSON
        club_names_lower = set(c.lower() for c in self._alias_map.values())
        club_names_lower.update(c.lower() for c in self._alias_map.keys())
        persons = [p for p in persons if p.lower() not in club_names_lower]

        # Heuristic role classification from nearby context
        players    = []
        coaches    = []
        executives = []
        other_orgs = list(orgs - set(clubs))

        # Role keywords that immediately precede or follow the name (within 3 tokens)
        _COACH_KW = re.compile(
            r'\b(head coach|assistant coach|technical director|sporting director|interim coach|manager)\b',
            re.IGNORECASE,
        )
        _EXEC_KW = re.compile(
            r'\b(president|CEO|chief executive|general manager|GM|owner|chairman|'
            r'investor|executive vice president|commissioner|co-owner|sporting director)\b',
            re.IGNORECASE,
        )
        # Common verbs that indicate a PERSON entity is a false positive (title-verb parse)
        _VERB_STARTS = re.compile(
            r'^(signs|named|hired|fired|sacked|joins|acquires|scores|beats|wins|loses|re-signs)\s',
            re.IGNORECASE,
        )

        for person in persons:
            # Drop false positives: spaCy sometimes grabs "Verb Name" as a person
            if _VERB_STARTS.match(person):
                person = re.sub(r'^\S+\s+', '', person).strip()
            if not person or len(person.split()) < 1:
                continue

            idx = full_text.lower().find(person.lower())
            if idx == -1:
                players.append(person)
                continue

            # Very tight window: 25 chars before, 35 chars after
            before = full_text[max(0, idx - 25): idx]
            after  = full_text[idx + len(person): idx + len(person) + 35]
            window = before + after

            if _COACH_KW.search(window):
                coaches.append(person)
            elif _EXEC_KW.search(window):
                executives.append(person)
            else:
                players.append(person)  # default: assume player

        return {
            "clubs":      clubs,
            "players":    list(dict.fromkeys(players)),    # deduplicated, order preserved
            "coaches":    list(dict.fromkeys(coaches)),
            "executives": list(dict.fromkeys(executives)),
            "other_orgs": other_orgs,
        }

    def _match_clubs(self, text: str) -> list[str]:
        """Return deduplicated list of canonical club names found in text."""
        found: dict[str, int] = defaultdict(int)

        # Alias regex match
        for m in self._club_re.finditer(text):
            canonical = self._alias_map.get(m.group(0).lower())
            if canonical:
                found[canonical] += 1

        # Abbreviation match (conservative — requires standalone token)
        if self._abbr_re:
            for m in self._abbr_re.finditer(text):
                canonical = self._abbreviations.get(m.group(0).upper())
                if canonical:
                    found[canonical] += 1

        # Return sorted by frequency (most mentioned first)
        return [club for club, _ in sorted(found.items(), key=lambda x: -x[1])]

    def _spacy_ner(self, text: str) -> tuple[list[str], set[str]]:
        """Return (persons, organizations) from spaCy NER."""
        if self._nlp is None:
            return [], set()
        try:
            # Truncate very long texts to avoid OOM
            doc = self._nlp(text[:50_000])
        except Exception:
            return [], set()

        persons = []
        orgs    = set()
        for ent in doc.ents:
            val = ent.text.strip()
            if not val or len(val) < 2:
                continue
            if ent.label_ == "PERSON":
                # Filter out common false positives (single words, numbers, etc.)
                if len(val.split()) >= 2:
                    persons.append(val)
            elif ent.label_ in ("ORG",):
                # Only keep org if not already in club list
                if val.lower() not in self._alias_map:
                    orgs.add(val)
        return persons, orgs


# ---------------------------------------------------------------------------
# Event Classifier
# ---------------------------------------------------------------------------

class EventClassifier:
    """
    Classifies article event type(s) using keyword matching against the
    event_types config. Returns a ranked list of matching event types.
    """

    def __init__(self, event_types_cfg: dict):
        self._patterns: dict[str, re.Pattern] = {}
        for event_type, cfg in event_types_cfg.items():
            keywords = cfg.get("keywords", [])
            if not keywords:
                continue
            pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            self._patterns[event_type] = re.compile(pattern, re.IGNORECASE)

    def classify(self, title: str, text: str) -> list[str]:
        """Return list of matching event types, ordered by keyword hit count."""
        combined = f"{title} {title} {text}"  # title weighted 2x
        scores: dict[str, int] = {}
        for event_type, pattern in self._patterns.items():
            matches = pattern.findall(combined)
            if matches:
                scores[event_type] = len(matches)
        return [et for et, _ in sorted(scores.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """VADER-based sentiment analysis."""

    def __init__(self):
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._analyzer = SentimentIntensityAnalyzer()
            self._available = True
        except ImportError:
            logger.warning("vaderSentiment not available. Sentiment will be skipped.")
            self._available = False

    def analyze(self, text: str) -> dict[str, float]:
        """Return compound, pos, neg, neu scores."""
        if not self._available or not text:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        try:
            scores = self._analyzer.polarity_scores(text[:5000])
            return {
                "compound": round(scores["compound"], 4),
                "pos":      round(scores["pos"], 4),
                "neg":      round(scores["neg"], 4),
                "neu":      round(scores["neu"], 4),
            }
        except Exception:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    @staticmethod
    def label(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"


# ---------------------------------------------------------------------------
# Temporal Contexter
# ---------------------------------------------------------------------------

class TemporalContexter:
    """Assigns season phase and transfer window labels to an article date."""

    # MLS season phase by month (approximate)
    _PHASE = {
        1:  "preseason",
        2:  "preseason",
        3:  "early_season",
        4:  "early_season",
        5:  "early_season",
        6:  "mid_season",
        7:  "mid_season",
        8:  "late_season",
        9:  "late_season",
        10: "late_season",
        11: "playoffs",
        12: "playoffs",
    }

    _TRANSFER_WINDOW = {
        1:  "winter",
        2:  "winter",
        7:  "summer",
        8:  "summer",
    }

    def label(self, published_date: str) -> dict[str, str | int]:
        if not published_date:
            return {"season_phase": "", "transfer_window": "none", "season_year": 0}
        try:
            dt    = datetime.fromisoformat(published_date)
            month = dt.month
            return {
                "season_phase":    self._PHASE.get(month, "regular_season"),
                "transfer_window": self._TRANSFER_WINDOW.get(month, "none"),
                "season_year":     dt.year,
            }
        except ValueError:
            return {"season_phase": "", "transfer_window": "none", "season_year": 0}


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

class EnrichmentPipeline:
    """
    Reads raw article parquet files, enriches them with NLP,
    and writes enriched parquet files.

    Checkpointing: one checkpoint key per (year, month).
    """

    STAGE = "enricher"

    def __init__(self, skip_completed: bool = True, spacy_batch_size: int = 32):
        settings = get_config("settings")
        entities = get_config("entities")

        data_dir  = PROJECT_ROOT / settings["pipeline"]["data_dir"]
        state_dir = PROJECT_ROOT / settings["pipeline"]["state_dir"]

        self.raw_dir      = data_dir / "press" / "raw"
        self.enriched_dir = data_dir / "press" / "enriched"
        self.skip_completed = skip_completed
        self.batch_size     = spacy_batch_size

        enrich_cfg = settings.get("enrichment", {})
        spacy_model = enrich_cfg.get("spacy_model", "en_core_web_sm")

        self.db         = CheckpointDB(str(state_dir / "pipeline_state.db"))
        self.matcher    = EntityMatcher(entities, spacy_model=spacy_model)
        self.classifier = EventClassifier(entities.get("event_types", {}))
        self.sentiment  = SentimentAnalyzer()
        self.temporal   = TemporalContexter()

    def run(self, years: Optional[list[int]] = None):
        """Enrich all raw data for the specified years."""
        start_year = get_config("sources")["collection"]["start_year"]
        end_year   = get_config("sources")["collection"]["end_year"]
        target_years = years or list(range(start_year, end_year + 1))
        total = 0

        for year in target_years:
            for month in range(1, 13):
                n = self._enrich_month(year, month)
                if n:
                    logger.info(f"Enriched {year}-{month:02d}: {n} articles")
                total += n

        logger.info(f"Enrichment complete. Total: {total} articles enriched.")
        return total

    def run_month(self, year: int, month: int, force: bool = False):
        return self._enrich_month(year, month, force=force)

    def _enrich_month(self, year: int, month: int, force: bool = False) -> int:
        key = f"{year}_{month:02d}"

        if not force and self.skip_completed and self.db.is_complete(self.STAGE, key):
            return 0

        raw_path = get_parquet_path(self.raw_dir, "raw", year, month)
        raw_df   = load_parquet(raw_path)
        if raw_df.empty:
            return 0

        self.db.set_status(self.STAGE, key, "in_progress")

        enriched_records = []
        try:
            for _, row in raw_df.iterrows():
                try:
                    record = self._enrich_article(row.to_dict())
                    enriched_records.append(record)
                except Exception as e:
                    logger.debug(f"Article enrichment failed ({row.get('article_id', '?')}): {e}")

            if enriched_records:
                out_path = get_parquet_path(self.enriched_dir, "enriched", year, month)
                df = pd.DataFrame([r.to_dict() for r in enriched_records])
                save_parquet(df, out_path)

            self.db.set_status(self.STAGE, key, "complete")
        except Exception as e:
            self.db.set_status(self.STAGE, key, "failed", str(e))
            logger.error(f"Enrichment failed for {year}-{month:02d}: {e}")

        return len(enriched_records)

    def _enrich_article(self, row: dict) -> EnrichedArticle:
        title   = str(row.get("title", "") or "")
        text    = str(row.get("text", "") or row.get("snippet", "") or "")
        pub_date = str(row.get("published_date", "") or "")

        # Entity extraction
        entities = self.matcher.extract(title, text)

        # Event classification
        event_types = self.classifier.classify(title, text)
        primary_event = event_types[0] if event_types else ""

        # Sentiment on title + first 500 chars of text
        sent_input = f"{title}. {text[:500]}"
        sentiment  = self.sentiment.analyze(sent_input)

        # Temporal context
        temporal = self.temporal.label(pub_date)

        # Importance signals
        club_count = len(entities["clubs"])
        word_count = max(len(text.split()), 1)
        all_ents   = (entities["clubs"] + entities["players"] +
                      entities["coaches"] + entities["executives"])
        density    = round(len(all_ents) / word_count * 100, 4)

        return EnrichedArticle(
            article_id=str(row.get("article_id", "")),
            url=str(row.get("url", "")),
            domain=str(row.get("domain", "")),
            title=title,
            published_date=pub_date,
            published_datetime=str(row.get("published_datetime", "") or ""),
            collection_year=int(row.get("collection_year", 0) or 0),
            collection_month=int(row.get("collection_month", 0) or 0),
            source=str(row.get("source", "")),
            text_length=int(row.get("text_length", 0) or 0),

            clubs_mentioned="|".join(entities["clubs"]),
            players_mentioned="|".join(entities["players"]),
            coaches_mentioned="|".join(entities["coaches"]),
            executives_mentioned="|".join(entities["executives"]),
            other_entities="|".join(entities["other_orgs"]),

            event_types="|".join(event_types),
            primary_event_type=primary_event,

            sentiment_compound=sentiment["compound"],
            sentiment_label=SentimentAnalyzer.label(sentiment["compound"]),
            sentiment_pos=sentiment["pos"],
            sentiment_neg=sentiment["neg"],
            sentiment_neu=sentiment["neu"],

            club_mention_count=club_count,
            entity_density=density,

            season_phase=temporal["season_phase"],
            transfer_window=temporal["transfer_window"],
            season_year=int(temporal["season_year"]),

            enriched_at=datetime.utcnow().isoformat(),
        )

    def status_report(self) -> pd.DataFrame:
        start_year = get_config("sources")["collection"]["start_year"]
        end_year   = get_config("sources")["collection"]["end_year"]
        rows = []
        for year, month in year_month_pairs(start_year, end_year):
            key    = f"{year}_{month:02d}"
            status = self.db.get_status(self.STAGE, key) or "pending"
            path   = get_parquet_path(self.enriched_dir, "enriched", year, month)
            df     = load_parquet(path)
            count  = len(df) if not df.empty else 0
            rows.append({"year": year, "month": month, "status": status, "enriched_count": count})
        return pd.DataFrame(rows)
