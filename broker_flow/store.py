"""每層輸出寫成 Parquet（第 0 節第 4 條）。路徑：data/broker_flow/<代號>/<層>.parquet"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from paths import DATA_DIR

log = logging.getLogger("broker_scraper")
OUT = DATA_DIR / "broker_flow"

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except ImportError:  # SPEC-GAP：環境沒有 pyarrow 時（例如沒有網路的測試機）退回 pickle，正式環境 requirements 已含 pyarrow
    HAS_PARQUET = False


def path(sid: str, layer: str) -> Path:
    return OUT / sid / f"{layer}.{'parquet' if HAS_PARQUET else 'pkl'}"


def write(df: pd.DataFrame, sid: str, layer: str) -> Path:
    p = path(sid, layer)
    p.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PARQUET:
        df.to_parquet(p, index=False)
    else:
        df.to_pickle(p)
    return p


def read(sid: str, layer: str) -> pd.DataFrame:
    p = path(sid, layer)
    return pd.read_parquet(p) if HAS_PARQUET else pd.read_pickle(p)
