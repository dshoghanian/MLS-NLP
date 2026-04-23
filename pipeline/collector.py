"""
Article collection stage.

Sources:
  1. GDELT Article Search API — free historical news archive
  2. RSS feeds — ongoing and supplemental coverage

Architecture:
  - CollectionPipeline orchestrates collection month-by-month
  - Each (year, month, query) combination is a checkpoint key
  - Completed months are skipped on re-runs
  - Failed months are retried up to max_retries times
  - Deduplication is tracked in the checkpoint DB
"""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from typing import Iterator, Optional
from urllib.parse import urlparse

import feedparser
import pandas as pd
import requests

from .models import Article
from .utils import (
    CheckpointDB,
    RateLimiter,
    get_config,
    get_logger,
    get_parquet_path,
    load_parquet,
    month_date_range,
    save_parquet,
    year_month_pairs,
    PROJECT_ROOT,
)

logger = get_logger("collector")


# ---------------------------------------------------------------------------
# GDELT collector
# ---------------------------------------------------------------------------

class GDELTCollector:
    """
    Queries the GDELT Article Search API for MLS-related news articles.

    GDELT API docs: https://blog.gdeltproject.org/gdelt-2-0-our-updates-three-global-knowledge-graphs/
    Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rate_limiter = RateLimiter(min_interval=cfg.get("request_delay", 1.2))
        self.max_retries = cfg.get("max_retries", 3)
        self.retry_delay = cfg.get("retry_delay", 5)
        self.excluded_domains = set(cfg.get("excluded_domains", []))
        self.language = cfg.get("language", "eng")

    def fetch_month(
        self,
        year: int,
        month: int,
        query: str,
        query_label: str = "",
    ) -> list[Article]:
        """Fetch articles for a single month/query combination."""
        start_dt, end_dt = month_date_range(year, month)
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": self.cfg.get("max_records", 250),
            "format": "json",
            "startdatetime": start_dt,
            "enddatetime": end_dt,
        }
        # Language filter — GDELT uses sourcelang: param in the query string
        full_query = f"{query} sourcelang:{self.language}"
        params["query"] = full_query

        articles = []
        for attempt in range(1, self.max_retries + 1):
            try:
                self.rate_limiter.wait()
                resp = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30,
                    headers={"User-Agent": "MLS-NLP-Research/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
                articles = self._parse_response(data, year, month, query_label or query)
                logger.debug(f"GDELT {year}-{month:02d} '{query_label}': {len(articles)} articles")
                break
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = self.retry_delay * attempt
                    logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt})")
                    time.sleep(wait)
                else:
                    logger.warning(f"HTTP {resp.status_code} for query '{query}': {e}")
                    break
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt} for '{query}' {year}-{month:02d}")
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error for '{query}' {year}-{month:02d}: {e}")
                time.sleep(self.retry_delay)

        return articles

    def _parse_response(
        self,
        data: dict,
        year: int,
        month: int,
        query_label: str,
    ) -> list[Article]:
        articles = []
        items = data.get("articles", [])
        for item in items:
            url = item.get("url", "")
            if not url:
                continue
            domain = urlparse(url).netloc.lstrip("www.")
            if domain in self.excluded_domains:
                continue

            # Parse date — GDELT returns "YYYYMMDDTHHmmssZ"
            raw_date = item.get("seendate", "")
            published_dt, published_date = self._parse_gdelt_date(raw_date)

            # Filter to target month (GDELT sometimes bleeds into adjacent months)
            if published_date:
                try:
                    dt = datetime.fromisoformat(published_date)
                    if dt.year != year or dt.month != month:
                        continue
                except ValueError:
                    pass

            snippet = item.get("title", "") + " " + item.get("socialimage", "")
            art = Article(
                url=url,
                domain=domain,
                source="gdelt",
                feed_name=f"gdelt:{query_label}",
                title=item.get("title", "").strip(),
                snippet=item.get("title", "").strip(),
                published_date=published_date,
                published_datetime=published_dt,
                collected_at=datetime.utcnow().isoformat(),
                collection_year=year,
                collection_month=month,
                query_used=query_label,
                has_full_text=False,
                text="",
            )
            art.populate_ids()
            articles.append(art)
        return articles

    @staticmethod
    def _parse_gdelt_date(raw: str) -> tuple[str, str]:
        """Parse GDELT date format '20210715T143000Z' → (datetime_str, date_str)."""
        if not raw:
            return "", ""
        try:
            # Format: YYYYMMDDTHHmmssZ
            raw = raw.replace("Z", "").replace("T", "")
            if len(raw) >= 8:
                dt = datetime.strptime(raw[:14] if len(raw) >= 14 else raw[:8] + "000000",
                                       "%Y%m%d%H%M%S")
                return dt.isoformat(), dt.date().isoformat()
        except ValueError:
            pass
        return "", ""


# ---------------------------------------------------------------------------
# RSS collector
# ---------------------------------------------------------------------------

class RSSCollector:
    """Collects articles from RSS/Atom feeds."""

    def __init__(self, feeds: list[dict]):
        self.feeds = feeds

    def fetch_feed(self, feed_cfg: dict) -> list[Article]:
        """Parse a single RSS feed and return Article objects."""
        articles = []
        try:
            d = feedparser.parse(
                feed_cfg["url"],
                request_headers={"User-Agent": "MLS-NLP-Research/1.0"},
            )
            for entry in d.entries:
                url = entry.get("link", "")
                if not url:
                    continue

                domain = urlparse(url).netloc.lstrip("www.")

                # Date parsing
                published_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if published_struct:
                    dt = datetime(*published_struct[:6], tzinfo=timezone.utc)
                    published_dt = dt.isoformat()
                    published_date = dt.date().isoformat()
                    year, month = dt.year, dt.month
                else:
                    published_dt = published_date = ""
                    year = month = 0

                # Text extraction — prefer summary over title
                snippet = (
                    entry.get("summary", "")
                    or entry.get("description", "")
                    or entry.get("title", "")
                )
                # Strip HTML tags from snippet
                import re
                snippet = re.sub(r"<[^>]+>", " ", snippet).strip()
                snippet = re.sub(r"\s+", " ", snippet)[:500]

                art = Article(
                    url=url,
                    domain=domain,
                    source=f"rss:{feed_cfg['name']}",
                    feed_name=feed_cfg["name"],
                    title=entry.get("title", "").strip(),
                    snippet=snippet,
                    published_date=published_date,
                    published_datetime=published_dt,
                    collected_at=datetime.utcnow().isoformat(),
                    collection_year=year,
                    collection_month=month,
                    query_used="rss",
                    has_full_text=False,
                    text="",
                )
                art.populate_ids()
                articles.append(art)
        except Exception as e:
            logger.warning(f"RSS fetch failed for {feed_cfg['name']}: {e}")
        return articles


# ---------------------------------------------------------------------------
# Full-text extractor
# ---------------------------------------------------------------------------

class TextExtractor:
    """
    Attempts to extract full article text from a URL.

    Uses trafilatura as primary, newspaper3k as fallback.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._trafilatura_available = self._check_trafilatura()
        self._newspaper_available = self._check_newspaper()

    @staticmethod
    def _check_trafilatura() -> bool:
        try:
            import trafilatura  # noqa
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_newspaper() -> bool:
        try:
            import newspaper  # noqa
            return True
        except ImportError:
            return False

    def extract(self, url: str) -> Optional[str]:
        """Return extracted full text, or None on failure."""
        text = None
        if self._trafilatura_available:
            text = self._extract_trafilatura(url)
        if not text and self._newspaper_available:
            text = self._extract_newspaper(url)
        return text

    def _extract_trafilatura(self, url: str) -> Optional[str]:
        try:
            import trafilatura
            # Use requests directly so we control the timeout, then pass HTML to trafilatura
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 (compatible; MLS-NLP-Research/1.0)"},
                allow_redirects=True,
            )
            resp.raise_for_status()
            text = trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return text.strip() if text else None
        except Exception as e:
            logger.debug(f"trafilatura failed for {url}: {e}")
        return None

    def _extract_newspaper(self, url: str) -> Optional[str]:
        try:
            import newspaper
            article = newspaper.Article(url, request_timeout=self.timeout)
            article.download()
            article.parse()
            if article.text and len(article.text) > 100:
                return article.text.strip()
        except Exception as e:
            logger.debug(f"newspaper failed for {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main collection pipeline
# ---------------------------------------------------------------------------

class CollectionPipeline:
    """
    Orchestrates month-by-month article collection from GDELT and RSS.

    Checkpointing:
      Key format: "{year}_{month:02d}_{source}"
      e.g. "2021_07_gdelt_mls-soccer" or "2021_07_rss_mlsofficial"

    On re-run: completed keys are skipped; failed keys are retried.
    """

    STAGE = "collector"

    def __init__(
        self,
        start_year: int = 2018,
        end_year: int = 2024,
        extract_full_text: bool = True,
        skip_completed: bool = True,
        max_articles_per_month: int = 0,
    ):
        settings = get_config("settings")
        sources  = get_config("sources")

        self.start_year = start_year
        self.end_year = end_year
        self.extract_full_text = extract_full_text
        self.skip_completed = skip_completed
        self.max_articles_per_month = max_articles_per_month

        data_dir  = PROJECT_ROOT / settings["pipeline"]["data_dir"]
        state_dir = PROJECT_ROOT / settings["pipeline"]["state_dir"]

        self.raw_dir   = data_dir / "press" / "raw"
        self.db        = CheckpointDB(str(state_dir / "pipeline_state.db"))
        self.gdelt     = GDELTCollector(sources["gdelt"])
        self.rss       = RSSCollector(sources.get("rss_feeds", []))
        self.extractor = TextExtractor(
            timeout=settings.get("collection", {}).get("extraction_timeout", 15)
        )
        self.min_text_length = settings.get("collection", {}).get("min_text_length", 100)
        self.gdelt_queries   = sources["gdelt"].get("queries", [])

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, years: Optional[list[int]] = None, retry_failed: bool = True):
        """
        Run full collection for specified years (or all years if None).
        """
        target_years = years or list(range(self.start_year, self.end_year + 1))
        total_new = 0

        for year in target_years:
            for month in range(1, 13):
                new = self._collect_month(year, month, retry_failed=retry_failed)
                total_new += new
                logger.info(f"  {year}-{month:02d}: +{new} new articles (total so far: {total_new})")

        logger.info(f"Collection complete. Total new articles: {total_new}")
        return total_new

    def run_month(self, year: int, month: int, force: bool = False):
        """Run collection for a single month."""
        return self._collect_month(year, month, retry_failed=True, force=force)

    # ------------------------------------------------------------------
    # Core month collection logic
    # ------------------------------------------------------------------

    def _collect_month(
        self,
        year: int,
        month: int,
        retry_failed: bool = True,
        force: bool = False,
    ) -> int:
        """
        Collect all articles for a (year, month) window.
        Returns number of new articles written.
        """
        month_key = f"{year}_{month:02d}"

        if not force and self.skip_completed and self.db.is_complete(self.STAGE, month_key):
            logger.debug(f"Skipping completed month {year}-{month:02d}")
            return 0

        self.db.set_status(self.STAGE, month_key, "in_progress")

        all_articles: list[Article] = []

        # 1. GDELT queries
        for qcfg in self.gdelt_queries:
            query     = qcfg["query"]
            query_key = f"{month_key}_gdelt_{query[:30].replace(' ', '_')}"
            if not force and self.skip_completed and self.db.is_complete(self.STAGE, query_key):
                continue
            try:
                articles = self.gdelt.fetch_month(year, month, query, qcfg.get("description", query))
                all_articles.extend(articles)
                self.db.set_status(self.STAGE, query_key, "complete")
            except Exception as e:
                self.db.set_status(self.STAGE, query_key, "failed", str(e))
                logger.error(f"GDELT query '{query}' failed for {year}-{month:02d}: {e}")

        # 2. RSS feeds (only useful for recent years; still included for completeness)
        for feed in self.rss.feeds:
            feed_key = f"{month_key}_rss_{feed['name']}"
            if not force and self.skip_completed and self.db.is_complete(self.STAGE, feed_key):
                continue
            try:
                rss_articles = self.rss.fetch_feed(feed)
                # Filter to this month
                filtered = [
                    a for a in rss_articles
                    if a.collection_year == year and a.collection_month == month
                ]
                all_articles.extend(filtered)
                self.db.set_status(self.STAGE, feed_key, "complete")
            except Exception as e:
                self.db.set_status(self.STAGE, feed_key, "failed", str(e))
                logger.warning(f"RSS feed '{feed['name']}' failed: {e}")

        # 3. Deduplicate
        new_articles = self._deduplicate(all_articles)
        logger.info(f"{year}-{month:02d}: {len(all_articles)} raw → {len(new_articles)} unique")

        # 4. Optional full-text extraction
        if self.extract_full_text and new_articles:
            new_articles = self._enrich_with_text(new_articles)

        # 5. Filter by minimum text length
        if self.min_text_length > 0:
            new_articles = [
                a for a in new_articles
                if len(a.text or a.snippet or "") >= self.min_text_length
            ]

        # 6. Cap per-month volume if configured
        if self.max_articles_per_month > 0:
            new_articles = new_articles[: self.max_articles_per_month]

        # 7. Save
        if new_articles:
            self._save_articles(new_articles, year, month)

        # 8. Mark seen in dedup DB
        for a in new_articles:
            self.db.mark_seen(a.article_id, a.url_hash)

        self.db.set_status(self.STAGE, month_key, "complete")
        return len(new_articles)

    def _deduplicate(self, articles: list[Article]) -> list[Article]:
        """Remove articles already seen or duplicates within this batch."""
        seen_ids:    set[str] = set()
        seen_urls:   set[str] = set()
        unique: list[Article] = []

        for a in articles:
            if not a.article_id:
                a.populate_ids()
            # Check global DB
            if self.db.is_seen(article_id=a.article_id, url_hash=a.url_hash):
                continue
            # Check within-batch
            if a.article_id in seen_ids or a.url_hash in seen_urls:
                continue
            seen_ids.add(a.article_id)
            seen_urls.add(a.url_hash)
            unique.append(a)
        return unique

    def _enrich_with_text(self, articles: list[Article]) -> list[Article]:
        """Attempt full-text extraction for each article."""
        from tqdm import tqdm

        results = []
        for art in tqdm(articles, desc="Extracting text", unit="article", leave=False):
            try:
                text = self.extractor.extract(art.url)
                if text and len(text) >= self.min_text_length:
                    art.text = text
                    art.has_full_text = True
                    art.text_length = len(text)
                else:
                    # Fall back to snippet
                    art.text = art.snippet
                    art.has_full_text = False
                    art.text_length = len(art.text)
            except Exception as e:
                logger.debug(f"Text extraction failed for {art.url}: {e}")
                art.text = art.snippet
            results.append(art)
        return results

    def _save_articles(self, articles: list[Article], year: int, month: int):
        """Save articles to parquet, appending to any existing file."""
        path = get_parquet_path(self.raw_dir, "raw", year, month)
        # Reload existing to check for new duplicates (edge case on re-runs)
        existing = load_parquet(path)
        if not existing.empty and "article_id" in existing.columns:
            existing_ids = set(existing["article_id"].dropna())
            articles = [a for a in articles if a.article_id not in existing_ids]
        if not articles:
            return
        df = pd.DataFrame([a.to_dict() for a in articles])
        save_parquet(df, path, append=not existing.empty)
        logger.info(f"Saved {len(articles)} articles → {path}")

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def status_report(self) -> pd.DataFrame:
        """Return a DataFrame summarising collection status per month."""
        pairs = year_month_pairs(self.start_year, self.end_year)
        rows = []
        for year, month in pairs:
            key   = f"{year}_{month:02d}"
            status = self.db.get_status(self.STAGE, key) or "pending"
            path  = get_parquet_path(self.raw_dir, "raw", year, month)
            df    = load_parquet(path)
            count = len(df) if not df.empty else 0
            rows.append({"year": year, "month": month, "status": status, "article_count": count})
        return pd.DataFrame(rows)
