"""
Shared utilities: logging, checkpointing, file I/O, and deduplication.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Return a logger that writes to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{name}_{datetime.now():%Y%m%d}.log"
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_config_cache: dict = {}

def load_config(config_path: str) -> dict:
    """Load and cache a YAML config file."""
    if config_path not in _config_cache:
        with open(config_path, "r") as f:
            _config_cache[config_path] = yaml.safe_load(f)
    return _config_cache[config_path]


def find_project_root() -> Path:
    """Walk up from this file to find the project root (contains config/)."""
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "config").exists():
            return p
        p = p.parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()


def get_config(name: str) -> dict:
    """Load config/<name>.yaml relative to project root."""
    path = PROJECT_ROOT / "config" / f"{name}.yaml"
    return load_config(str(path))


# ---------------------------------------------------------------------------
# Checkpoint / state DB
# ---------------------------------------------------------------------------

class CheckpointDB:
    """
    SQLite-backed checkpoint store for tracking pipeline run state.

    Table: checkpoints
      stage       TEXT   — collector | enricher | network_builder | analyzer
      key         TEXT   — e.g. "2021_07_gdelt" or "2021_07"
      status      TEXT   — pending | in_progress | complete | failed
      attempts    INT
      last_error  TEXT
      updated_at  TEXT
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    stage       TEXT NOT NULL,
                    key         TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    last_error  TEXT,
                    updated_at  TEXT,
                    PRIMARY KEY (stage, key)
                )
            """)
            # Article deduplication table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_articles (
                    article_id  TEXT PRIMARY KEY,
                    url_hash    TEXT,
                    first_seen  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_url_hash ON seen_articles(url_hash)")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_status(self, stage: str, key: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM checkpoints WHERE stage=? AND key=?", (stage, key)
            ).fetchone()
        return row["status"] if row else None

    def set_status(self, stage: str, key: str, status: str, error: str = ""):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO checkpoints (stage, key, status, attempts, last_error, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT(stage, key) DO UPDATE SET
                    status=excluded.status,
                    attempts=attempts+1,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
            """, (stage, key, status, error, now))

    def is_complete(self, stage: str, key: str) -> bool:
        return self.get_status(stage, key) == "complete"

    def get_failed_keys(self, stage: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key FROM checkpoints WHERE stage=? AND status='failed'", (stage,)
            ).fetchall()
        return [r["key"] for r in rows]

    def get_all_keys(self, stage: str) -> dict[str, str]:
        """Return {key: status} for all keys in a stage."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT key, status FROM checkpoints WHERE stage=?", (stage,)
            ).fetchall()
        return {r["key"]: r["status"] for r in rows}

    # --- Deduplication helpers ---

    def is_seen(self, article_id: str = "", url_hash: str = "") -> bool:
        with self._conn() as conn:
            if article_id:
                row = conn.execute(
                    "SELECT 1 FROM seen_articles WHERE article_id=?", (article_id,)
                ).fetchone()
                if row:
                    return True
            if url_hash:
                row = conn.execute(
                    "SELECT 1 FROM seen_articles WHERE url_hash=?", (url_hash,)
                ).fetchone()
                if row:
                    return True
        return False

    def mark_seen(self, article_id: str, url_hash: str):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO seen_articles (article_id, url_hash, first_seen)
                VALUES (?, ?, ?)
            """, (article_id, url_hash, now))

    def seen_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM seen_articles").fetchone()[0]


# ---------------------------------------------------------------------------
# Parquet I/O helpers
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: str | Path, append: bool = False):
    """Save a DataFrame to a Parquet file, optionally appending."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        existing = pd.read_parquet(path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(str(path), index=False, engine="pyarrow", compression="snappy")


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file, returning empty DataFrame if not found."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(str(path), engine="pyarrow")


def iter_parquet_files(directory: str | Path, pattern: str = "**/*.parquet"):
    """Yield all parquet files matching a glob pattern."""
    for p in sorted(Path(directory).glob(pattern)):
        yield p


def load_all_parquet(directory: str | Path, pattern: str = "**/*.parquet") -> pd.DataFrame:
    """Load and concatenate all parquet files in a directory."""
    frames = []
    for p in iter_parquet_files(directory, pattern):
        df = load_parquet(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Rate limiting helper
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._last_call = 0.0

    def wait(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def year_month_pairs(start_year: int, end_year: int) -> list[tuple[int, int]]:
    """Generate (year, month) tuples from start_year-01 through end_year-12."""
    pairs = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            pairs.append((y, m))
    return pairs


def month_date_range(year: int, month: int) -> tuple[str, str]:
    """Return (start_datetime, end_datetime) strings for GDELT API (YYYYMMDDHHmmss)."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}{month:02d}01000000"
    end   = f"{year}{month:02d}{last_day}235959"
    return start, end


def quarter_label(month: int) -> str:
    return f"Q{(month - 1) // 3 + 1}"


def get_parquet_path(data_dir: str | Path, stage: str, year: int, month: int) -> Path:
    """Canonical path for per-month parquet files."""
    return Path(data_dir) / stage / str(year) / f"{year}_{month:02d}.parquet"
