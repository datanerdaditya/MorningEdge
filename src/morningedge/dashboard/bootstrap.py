"""Bootstrap data source for the dashboard.

When deployed to Streamlit Cloud, the dashboard runs in a fresh container
each session — there's no persistent local database. We pull the latest
DuckDB from the GitHub ``data`` branch on cold start.

When running locally (with .env DUCKDB_PATH pointing to a local file),
this is a no-op.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from loguru import logger


# Public URL to the daily-updated DuckDB on the ``data`` branch.
REMOTE_DB_URL = (
    "https://raw.githubusercontent.com/datanerdaditya/MorningEdge/"
    "data/data/morningedge.duckdb"
)

# How long to keep a downloaded copy before re-fetching.
CACHE_TTL = timedelta(hours=1)


def ensure_db_available(local_path: Path) -> Path:
    """Make sure a DuckDB file exists at ``local_path``.

    Behaviour:
    - If the file exists and is fresh (< CACHE_TTL old), do nothing.
    - If the file is stale or missing, download from REMOTE_DB_URL.
    - If we're running locally (USE_LOCAL_DB env var = "1"), never download.
    """
    # Local-dev escape hatch
    if os.environ.get("USE_LOCAL_DB", "").strip() == "1":
        if local_path.exists():
            logger.info(f"Using local DB at {local_path} (USE_LOCAL_DB=1)")
        else:
            logger.warning(f"USE_LOCAL_DB=1 but {local_path} does not exist")
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache check
    if local_path.exists():
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            local_path.stat().st_mtime, tz=timezone.utc
        )
        if age < CACHE_TTL:
            logger.info(f"Using cached DB at {local_path} (age {age})")
            return local_path

    # Download
    logger.info(f"Downloading latest DB from {REMOTE_DB_URL}")
    try:
        with httpx.stream("GET", REMOTE_DB_URL, timeout=60.0, follow_redirects=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=64 * 1024):
                    f.write(chunk)
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded DB: {size_mb:.1f}MB -> {local_path}")
    except Exception as e:
        if local_path.exists():
            logger.warning(f"Download failed ({e}); using stale cached DB")
        else:
            logger.error(f"Download failed and no cache exists: {e}")
            raise

    return local_path