"""
Reddit Ingestion Stage
======================
Reads club and league subreddit .zst files from the zip archive,
normalizes posts + high-score comments into the same parquet schema
used by the press pipeline, and saves monthly files to:

  data/reddit/raw/YYYY/YYYY_MM.parquet

Submissions:  min_score >= 1,  min_text >= 20 chars
Comments:     min_score >= 5,  min_text >= 20 chars  (reduces noise)
Date range:   2018-01-01 – 2024-12-31
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import zstandard

from .utils import get_logger, PROJECT_ROOT, save_parquet

logger = get_logger("reddit_ingestor")

ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "reddit_data.zip"
OUT_DIR  = PROJECT_ROOT / "data" / "reddit" / "raw"

START_YEAR = 2018
END_YEAR   = 2024

# Score-based sampling — keep only top N posts per subreddit per month.
# Targets ~2-3× the press corpus (~7,236 articles) for fair comparison.
#   Club subreddits  : top 6 / month  → 28 × 6 × 84 ≈ 14,112
#   Multi-club subs  : top 15 / month → 2 × 15 × 84  ≈  2,520
#   Total target     : ~16,600 posts  ≈ 2.3× press
MULTI_CLUB_SUBS    = {"MLS", "ussoccer"}
TOP_N_CLUB         = 6    # per subreddit per month
TOP_N_MULTI        = 15   # per subreddit per month (MLS, ussoccer)

# Map zip folder prefix → canonical club name (None = multi-club, skip brand mapping)
SUBREDDIT_CLUB_MAP: dict[str, Optional[str]] = {
    "AtlantaUnited":          "Atlanta United FC",
    "AustinFC":               "Austin FC",
    "CharlotteFootballClub":  "Charlotte FC",
    "chicagofire":            "Chicago Fire FC",
    "DCUnited":               "D.C. United",
    "dynamo":                 "Houston Dynamo FC",
    "FCCincinnati":           "FC Cincinnati",
    "fcdallas":               "FC Dallas",
    "InterMiami":             "Inter Miami CF",
    "LAFC":                   "LAFC",
    "LAGalaxy":               "LA Galaxy",
    "minnesotaunited":        "Minnesota United FC",
    "MLS":                    None,
    "NashvilleSC":            "Nashville SC",
    "newenglandrevolution":   "New England Revolution",
    "NYCFC":                  "New York City FC",
    "OCLions":                "Orlando City SC",
    "PhillyUnion":            "Philadelphia Union",
    "Rapids":                 "Colorado Rapids",
    "rbny":                   "New York Red Bulls",
    "ReAlSaltLake":           "Real Salt Lake",
    "SJEarthquakes":          "San Jose Earthquakes",
    "SoundersFC":             "Seattle Sounders FC",
    "SportingKC":             "Sporting Kansas City",
    "stlouiscitysc":          "St. Louis City SC",
    "tfc":                    "Toronto FC",
    "TheMassive":             "Columbus Crew",
    "timbers":                "Portland Timbers",
    "ussoccer":               None,
    "whitecapsfc":            "Vancouver Whitecaps FC",
    "FantasyMLS":             None,
    # YogaPants is unrelated — skip
}


def _article_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _stream_zst(zf: zipfile.ZipFile, inner_path: str) -> Iterator[dict]:
    """Stream JSON lines from a .zst file inside the zip."""
    with zf.open(inner_path) as raw:
        dctx   = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(raw)
        text   = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for line in text:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_ts(val) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(int(val), tz=timezone.utc)
    except Exception:
        return None


def _normalize_submission(obj: dict, subreddit_key: str) -> Optional[dict]:
    dt = _parse_ts(obj.get("created_utc"))
    if not dt or not (START_YEAR <= dt.year <= END_YEAR):
        return None

    score = int(obj.get("score", 0))
    if score < 1:
        return None

    title    = str(obj.get("title", "")).strip()
    selftext = str(obj.get("selftext", "")).strip()
    if selftext in ("[deleted]", "[removed]", ""):
        selftext = ""

    text = f"{title} {selftext}".strip()
    if len(text) < 20:
        return None

    canonical_club = SUBREDDIT_CLUB_MAP.get(subreddit_key)

    return {
        "article_id":       _article_id(f"reddit_{obj.get('id','')}_{text[:100]}"),
        "source":           f"r/{obj.get('subreddit', subreddit_key)}",
        "domain":           "reddit.com",
        "subreddit":        obj.get("subreddit", subreddit_key),
        "subreddit_key":    subreddit_key,
        "post_type":        "submission",
        "reddit_id":        str(obj.get("id", "")),
        "title":            title,
        "text":             text,
        "score":            score,
        "num_comments":     int(obj.get("num_comments", 0)),
        "published_date":   dt.strftime("%Y-%m-%d"),
        "published_datetime": dt.isoformat(),
        "collection_year":  dt.year,
        "collection_month": dt.month,
        "has_full_text":    True,
        "text_length":      len(text),
        "primary_club":     canonical_club or "",
    }


def _normalize_comment(obj: dict, subreddit_key: str) -> Optional[dict]:
    dt = _parse_ts(obj.get("created_utc"))
    if not dt or not (START_YEAR <= dt.year <= END_YEAR):
        return None

    score = int(obj.get("score", 0))
    if score < 5:
        return None

    body = str(obj.get("body", "")).strip()
    if body in ("[deleted]", "[removed]", "") or len(body) < 20:
        return None

    canonical_club = SUBREDDIT_CLUB_MAP.get(subreddit_key)

    return {
        "article_id":       _article_id(f"reddit_{obj.get('id','')}_{body[:100]}"),
        "source":           f"r/{obj.get('subreddit', subreddit_key)}",
        "domain":           "reddit.com",
        "subreddit":        obj.get("subreddit", subreddit_key),
        "subreddit_key":    subreddit_key,
        "post_type":        "comment",
        "reddit_id":        str(obj.get("id", "")),
        "title":            "",
        "text":             body,
        "score":            score,
        "num_comments":     0,
        "published_date":   dt.strftime("%Y-%m-%d"),
        "published_datetime": dt.isoformat(),
        "collection_year":  dt.year,
        "collection_month": dt.month,
        "has_full_text":    True,
        "text_length":      len(body),
        "primary_club":     canonical_club or "",
    }


class RedditIngestor:
    """
    Extracts and normalizes Reddit data from the zip archive.
    Saves monthly parquet files matching the press pipeline schema.
    """

    def __init__(self):
        if not ZIP_PATH.exists():
            raise FileNotFoundError(f"Reddit zip not found: {ZIP_PATH}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self):
        logger.info(f"Opening {ZIP_PATH}")

        # Buffer: (subreddit_key, year, month) → list of candidate records
        # We collect ALL passing records per subreddit/month, then keep top N by score.
        # This avoids loading everything into memory at once.
        from collections import defaultdict

        # sub_month_buf[(sub_key, year, month)] = [(score, rec), ...]
        sub_month_buf: dict[tuple, list] = defaultdict(list)
        total_read = 0

        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            inner_files = [n for n in zf.namelist() if n.endswith(".zst")]
            logger.info(f"  {len(inner_files)} .zst files in archive")

            for inner_path in sorted(inner_files):
                fname = Path(inner_path).stem
                if fname.endswith("_submissions"):
                    subreddit_key = fname[:-len("_submissions")]
                    post_type     = "submission"
                elif fname.endswith("_comments"):
                    subreddit_key = fname[:-len("_comments")]
                    post_type     = "comment"
                else:
                    continue

                if subreddit_key not in SUBREDDIT_CLUB_MAP:
                    continue
                if subreddit_key == "YogaPants":
                    continue

                top_n = TOP_N_MULTI if subreddit_key in MULTI_CLUB_SUBS else TOP_N_CLUB
                logger.info(f"  {inner_path}  (top {top_n}/month) ...")
                file_read = 0

                for obj in _stream_zst(zf, inner_path):
                    total_read += 1
                    file_read  += 1

                    if post_type == "submission":
                        rec = _normalize_submission(obj, subreddit_key)
                    else:
                        rec = _normalize_comment(obj, subreddit_key)

                    if rec is None:
                        continue

                    key = (subreddit_key, rec["collection_year"], rec["collection_month"])
                    sub_month_buf[key].append((rec["score"], rec))

                logger.info(f"    {file_read:,} read")

        # Keep top N by score per (subreddit, year, month)
        logger.info("Applying top-N score sampling ...")
        buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
        seen_ids: set[str] = set()
        total_kept = 0

        for (sub_key, year, month), candidates in sub_month_buf.items():
            top_n = TOP_N_MULTI if sub_key in MULTI_CLUB_SUBS else TOP_N_CLUB
            # Sort descending by score, keep top N
            candidates.sort(key=lambda x: x[0], reverse=True)
            for _, rec in candidates[:top_n]:
                if rec["article_id"] in seen_ids:
                    continue
                seen_ids.add(rec["article_id"])
                buckets[(year, month)].append(rec)
                total_kept += 1

        # Save monthly parquets
        logger.info(f"Saving {len(buckets)} monthly parquet files ...")
        for (year, month), records in sorted(buckets.items()):
            year_dir = OUT_DIR / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            path = year_dir / f"{year}_{month:02d}_reddit.parquet"
            pd.DataFrame(records).to_parquet(str(path), index=False)

        logger.info(f"Done. {total_read:,} read → {total_kept:,} kept "
                    f"(top {TOP_N_CLUB}/month club subs, top {TOP_N_MULTI}/month multi-club subs) "
                    f"across {len(buckets)} months.")
        return total_kept
