"""
§2 切分：開發集 D、調參集 T、跨股票保留 H、時間保留（T_cut 之後）
  H：個股依成交值十分位分層，各層以固定種子抽 20%（SPEC-GAP-v3：已知關係企業集團清單不足 10 組，先只用個股分層抽樣）
  清單寫進 config/holdout_universe.yaml（決定性：同一股票池與種子永遠得到同一份）；所有輸出寫入前呼叫 assert_writable
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml

from .params import P

log = logging.getLogger("broker_scraper")
HOLD_FILE = Path(__file__).resolve().parent / "config" / "holdout_universe.yaml"


class HoldoutError(RuntimeError):
    pass


def _pool() -> list[tuple[str, float]]:
    """股票池（有價格的股票）與 T_cut 前 60 日的日均成交值"""
    from paths import DATA_DIR
    out = []
    cut = str(P.governance.t_cut)
    for f in sorted((DATA_DIR / "longhist").glob("*/_prices.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8")).get("data") or {}
        except Exception:  # noqa: BLE001
            continue
        rows = [d[k] for k in sorted(d) if k < cut][-int(P.universe.lookback):]
        if rows:
            out.append((f.parent.name, float(np.mean([float(r.get("close") or 0) * float(r.get("vol") or 0) for r in rows]))))
    return out


def holdout() -> dict:
    if HOLD_FILE.exists():
        return yaml.safe_load(HOLD_FILE.read_text(encoding="utf-8")) or {}
    g = P.governance
    dev = set(g.dev_stocks)
    pool = [(s, v) for s, v in _pool() if s not in dev]
    if not pool:
        return {"stocks": [], "groups": []}
    rng = np.random.default_rng(int(g.holdout_seed))
    vals = np.array([v for _, v in pool])
    dec = np.minimum((np.argsort(np.argsort(vals)) * 10 // len(vals)), 9)
    H = []
    for k in range(10):
        ids = sorted(pool[i][0] for i in np.flatnonzero(dec == k))
        m = int(round(len(ids) * float(g.holdout_stock_frac)))
        if m:
            H += sorted(rng.choice(ids, m, replace=False).tolist())
    doc = {"t_cut": str(g.t_cut), "seed": int(g.holdout_seed), "pool_size": len(pool), "stocks": sorted(H), "groups": [],
           "note": "SPEC-GAP-v3：已知關係企業集團清單不足 10 組，H 只用個股分層抽樣"}
    try:
        HOLD_FILE.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    log.info("v3 保留池：%d 檔中抽 %d 檔", len(pool), len(H))
    return doc


def is_holdout(stock: str, date: str | None = None) -> bool:
    if stock in set(holdout().get("stocks") or []):
        return True
    return False


def assert_writable(stock: str) -> None:
    """§2.1：H 股票不得寫進 Firestore、網頁、log 摘要"""
    if is_holdout(stock):
        raise HoldoutError(f"{stock} 是保留池股票，不得輸出")


def split_of(stock: str, date: str) -> str:
    g = P.governance
    if stock in set(g.dev_stocks):
        return "dev"
    if is_holdout(stock):
        return "holdout_xs"
    return "holdout_time" if date > str(g.t_cut) else "tune"
