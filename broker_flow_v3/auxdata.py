"""
§1 FinMind 輔助資料（逐檔查詢，按股票快取在 FM_DIR/aux/<代號>/<資料集>.json；FM_DIR 在雲端是 $RUNNER_TEMP，不進公開快取）
  先找私人倉庫 Release「aux」下載下來的全市場年檔（FM_DIR/aux_year/<資料集>-<YYYY>.parquet），沒有才用 API 逐檔查
  token：環境變數 FINMIND_TOKEN（不印出）；沒有 token、查不到 → 空表（呼叫端依 SPEC-GAP-v3 退回 v2 做法）
"""
from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd

log = logging.getLogger("broker_scraper")
API = "https://api.finmindtrade.com/api/v4/data"


def _root() -> Path:
    from paths import DATA_DIR
    return Path(os.environ.get("FM_DIR") or (DATA_DIR / "_finmind"))


def _token() -> str:
    return os.environ.get("FINMIND_TOKEN", "").strip()


def _api(params: dict) -> list[dict]:
    import requests
    tok = _token()
    if not tok:
        return []
    for k in range(4):
        try:
            r = requests.get(API, params=params, headers={"Authorization": f"Bearer {tok}"}, timeout=120)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            j = r.json()
            if j.get("status") not in (200, None):
                log.warning("FinMind %s：%s", params.get("dataset"), str(j.get("msg"))[:120])
                return []
            return j.get("data") or []
        except Exception as e:  # noqa: BLE001
            if k == 3:
                log.warning("FinMind %s 查詢失敗：%s", params.get("dataset"), e)
                return []
            time.sleep(5 * (k + 1))
    return []


def _year_files(ds: str) -> list[Path]:
    return sorted((_root() / "aux_year").glob(f"{ds}-*.parquet"))


@lru_cache(maxsize=None)
def _year_table(ds: str) -> pd.DataFrame | None:
    fs = _year_files(ds)
    if not fs:
        return None
    try:
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    except Exception as e:  # noqa: BLE001
        log.warning("aux 年檔 %s 讀取失敗：%s", ds, e)
        return None


@lru_cache(maxsize=None)
def stock(ds: str, sid: str, start: str = "2019-01-01") -> pd.DataFrame:
    """某資料集、某檔股票（data_id）的全部資料"""
    yt = _year_table(ds)
    if yt is not None and "stock_id" in yt:
        return yt[yt["stock_id"].astype(str) == sid].reset_index(drop=True)
    f = _root() / "aux" / sid / f"{ds}.json"
    if f.exists() and time.time() - f.stat().st_mtime < 12 * 3600:
        return pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
    rows = _api({"dataset": ds, "data_id": sid, "start_date": start, "end_date": time.strftime("%Y-%m-%d")})
    if rows:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def static(ds: str) -> pd.DataFrame:
    f = _root() / "aux" / f"{ds}.json"
    yt = _root() / "aux_year" / f"{ds}.parquet"
    if yt.exists():
        return pd.read_parquet(yt)
    if f.exists() and time.time() - f.stat().st_mtime < 24 * 3600:
        return pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
    p = {"dataset": ds}
    if ds == "TaiwanStockTradingDate":
        p.update(start_date="1990-01-01", end_date=time.strftime("%Y-12-31"))
    rows = _api(p)
    if rows:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return pd.DataFrame(rows)


def trading_calendar() -> list[str] | None:
    """§3.1 官方交易日曆；拿不到 → None（退回「個股有價格的日子」，SPEC-GAP）"""
    df = static("TaiwanStockTradingDate")
    if df is None or not len(df) or "date" not in df:
        return None
    return sorted(df["date"].astype(str))


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(float("nan"), index=df.index)
