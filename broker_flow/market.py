"""
全市場層：讀 longhist 裡所有股票，算各檔的活躍日、z_net、行為分類，給維度 C 與假集團用
「全市場」＝本專案 longhist 有建檔的股票（SPEC-GAP：不是證交所全部上市櫃股票）
記憶體：每檔只留 z_net（float32）、活躍日（bool）、每日成交量與收盤，以及窗口結束日的分類
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np

from . import features
from .data_layer import LONG_DIR, build_panel
from .params import P

log = logging.getLogger("broker_scraper")


def market_calendar() -> np.ndarray:
    """所有建檔股票的交易日聯集（SPEC-GAP：沒有官方交易日曆，用這個代替）"""
    days = set()
    for f in LONG_DIR.glob("*/_prices.json"):
        try:
            days |= set((json.loads(f.read_text(encoding="utf-8")).get("data") or {}).keys())
        except Exception:  # noqa: BLE001
            continue
    return np.array(sorted(days))


def window_end_dates(cal: np.ndarray) -> list[str]:
    """6.0：窗口 120 日、步長 20，固定格點（從分析起點起算），再加上最後一個交易日
    固定格點 → 過去的窗口不會因為多了新資料而改變，可以存起來重用"""
    W, step = int(P.detection.window_days), int(P.detection.window_step)
    first = int(P.accounting.warmup_days) + W - 1
    ends = list(range(first, len(cal), step))
    if ends and ends[-1] != len(cal) - 1:
        ends.append(len(cal) - 1)
    return [str(cal[i]) for i in ends]


@dataclass
class Stock:
    sid: str
    dates: np.ndarray
    keys: list[str]
    names: list[str]
    btype: list[str]
    z: np.ndarray            # (分點, 日) float32
    active: np.ndarray       # (分點, 日) bool
    gross: np.ndarray        # (分點, 日) float32 成交量（股）
    close: np.ndarray
    volume: np.ndarray
    cls: dict = field(default_factory=dict)   # 窗口結束日 → 分類 list

    def end_index(self, end: str) -> int:
        return int(np.searchsorted(self.dates, end, side="right")) - 1


def load_stock(sid: str, ends: list[str]) -> Stock | None:
    try:
        pn = build_panel(sid, write=False)
    except Exception:  # noqa: BLE001
        return None
    if len(pn.dates) < int(P.accounting.warmup_days) + int(P.detection.window_days) or not pn.branches:
        return None
    _, active, z = features.scale_and_active(pn)
    st = Stock(sid, pn.dates, pn.branches, pn.names, pn.btype, z.astype(np.float32), active,
               (pn.B + pn.S).astype(np.float32), pn.close, pn.volume)
    for d in ends:
        e = st.end_index(d)
        if e >= int(P.detection.window_days) - 1:
            st.cls[d] = features.classify(pn, active, e)
    return st


def load_market(ends: list[str], only: list[str] | None = None) -> dict[str, Stock]:
    out = {}
    sids = only or sorted(p.parent.name for p in LONG_DIR.glob("*/_prices.json"))
    for sid in sids:
        st = load_stock(sid, ends)
        if st:
            out[sid] = st
    log.info("全市場：%d 檔、%d 個分點代碼", len(out), len({k for s in out.values() for k in s.keys}))
    return out
