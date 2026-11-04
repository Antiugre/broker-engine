"""
§3.1 資料層（v3）
  ・主資料：FinMind SecIdAgg（2021-06-30 起）；分點一律以 securities_trader_id 對應（§1.3）
  ・交易日曆：TaiwanStockTradingDate（官方）；個股在日曆上沒有有效價格（close > 0）的日子標 suspended（§3.1、§1.7）
  ・價格：TaiwanStockPrice 開高低收量（振幅用）、TaiwanStockPriceAdj 還原收盤（報酬率用）；拿不到 → longhist _prices.json（SPEC-GAP-v3）
  ・每日不平衡率 imb = |Σb − Σs| / max(Σb, Σs)（給維度 B 排除 > 2% 的日子）
SPEC-GAP-v3（今晚上線版）
  ・2021-06-30 之前的富邦資料不併入（v3 窗口從 FinMind 起算）
  ・鉅額交易分點明細（2026-04-28 起）尚未併入會計層；data_quality 對帳尚未產出
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from paths import DATA_DIR

from . import auxdata as aux
from .params import P

log = logging.getLogger("broker_scraper")
LOT = 1000
LONG_DIR = DATA_DIR / "longhist"
from flowlab.core import INSTITUTIONAL_HINTS  # noqa: E402


def branch_type(name: str) -> str:
    n = name or ""
    if "自營" in n:
        return "proprietary"
    if any(h in n for h in INSTITUTIONAL_HINTS):
        return "foreign"
    if any(h in n for h in P.behavior.gov_bank_names) and ("總公司" in n or n.rstrip().endswith(("證券", "金"))):
        return "gov_bank"
    return "brokerage"


@dataclass
class Panel:
    sid: str
    dates: np.ndarray
    branches: list[str]          # securities_trader_id
    names: list[str]
    btype: list[str]
    B: np.ndarray                # (分點, 日) 股
    S: np.ndarray
    BP: np.ndarray               # 分點當日買均價（沒成交 NaN）
    SP: np.ndarray
    close: np.ndarray            # 收盤（未還原；會計用）
    vwap: np.ndarray
    volume: np.ndarray           # 股
    high: np.ndarray
    low: np.ndarray
    adj_close: np.ndarray        # 還原收盤（報酬率用）
    suspended: np.ndarray        # bool
    imb: np.ndarray              # 每日不平衡率
    meta: dict = field(default_factory=dict)

    @property
    def ret(self) -> np.ndarray:
        a = self.adj_close
        r = np.r_[0.0, a[1:] / np.where(a[:-1] > 0, a[:-1], np.nan) - 1]
        return np.nan_to_num(r)

    @property
    def amp(self) -> np.ndarray:
        prev = np.r_[np.nan, self.close[:-1]]
        a = (self.high - self.low) / np.where(prev > 0, prev, np.nan)
        return np.nan_to_num(a)


def _prices_longhist(sid: str) -> pd.DataFrame:
    f = LONG_DIR / sid / "_prices.json"
    if not f.exists():
        return pd.DataFrame(columns=["date", "close", "vwap", "volume"])
    prices = json.loads(f.read_text(encoding="utf-8")).get("data") or {}
    px = pd.DataFrame.from_dict(prices, orient="index")
    px.index.name = "date"
    px = px.reset_index().sort_values("date")
    return pd.DataFrame({"date": px["date"].astype(str), "close": px["close"].astype(float),
                         "vwap": px.get("vwap", px["close"]).astype(float).fillna(px["close"].astype(float)),
                         "volume": px["vol"].astype(float) * LOT})


def prices(sid: str) -> pd.DataFrame:
    """date, open, high, low, close, vwap, volume（股）, adj_close；close > 0 才是有效日"""
    lh = _prices_longhist(sid)
    tp = aux.stock("TaiwanStockPrice", sid)
    if len(tp) and "close" in tp:
        df = pd.DataFrame({"date": tp["date"].astype(str), "open": aux.num(tp, "open"), "high": aux.num(tp, "max"),
                           "low": aux.num(tp, "min"), "close": aux.num(tp, "close"),
                           "volume": aux.num(tp, "Trading_Volume"), "money": aux.num(tp, "Trading_money")})
        df = df[df["close"] > 0]
        df["vwap"] = np.where(df["volume"] > 0, df["money"] / df["volume"].where(df["volume"] > 0), df["close"])
        src = "finmind"
    else:
        df = lh.assign(open=np.nan, high=np.nan, low=np.nan)
        src = "longhist"
    adj = aux.stock("TaiwanStockPriceAdj", sid)
    if len(adj) and "close" in adj:
        a = pd.DataFrame({"date": adj["date"].astype(str), "adj_close": aux.num(adj, "close")})
        df = df.merge(a[a["adj_close"] > 0], on="date", how="left")
    else:
        df["adj_close"] = np.nan
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df.attrs["source"] = src
    return df


def build_panel(sid: str, end: str | None = None) -> Panel:
    import fm_store
    if not fm_store.enabled():
        raise RuntimeError("v3 需要 FinMind 資料（FM_DIR）")
    fm = fm_store.stock_table(sid)
    if not len(fm):
        raise RuntimeError(f"{sid}：FinMind 沒有資料")
    px = prices(sid)
    start = str(P.data.finmind_start)
    cal = aux.trading_calendar()
    last = max(str(fm["date"].max()), str(px["date"].max()) if len(px) else "")
    if end:
        last = min(last, end)
    if cal:
        dates = np.array([d for d in cal if start <= d <= last])
    else:
        dates = np.array(sorted(d for d in px["date"] if start <= d <= last))
    pos = {d: i for i, d in enumerate(dates)}
    T = len(dates)
    pxi = px.set_index("date").reindex(dates)
    suspended = ~(pxi["close"].to_numpy(float) > 0)
    close = pxi["close"].ffill().bfill().to_numpy(float)
    names = fm.drop_duplicates("securities_trader_id", keep="last").set_index("securities_trader_id")["securities_trader"]
    ids = [str(b) for b in names.index]
    bix = {b: i for i, b in enumerate(ids)}
    n = len(ids)
    B = np.zeros((n, T)); S = np.zeros((n, T))
    BP = np.full((n, T), np.nan); SP = np.full((n, T), np.nan)
    f = fm[fm["date"].astype(str).isin(pos)]
    ii = f["securities_trader_id"].astype(str).map(bix).to_numpy(int)
    jj = f["date"].astype(str).map(pos).to_numpy(int)
    bv, sv = f["buy_volume"].to_numpy(float), f["sell_volume"].to_numpy(float)
    np.add.at(B, (ii, jj), bv)
    np.add.at(S, (ii, jj), sv)
    BP[ii, jj] = np.where(bv > 0, f["buy_price"].to_numpy(float), np.nan)
    SP[ii, jj] = np.where(sv > 0, f["sell_price"].to_numpy(float), np.nan)
    tb, ts = B.sum(0), S.sum(0)
    imb = np.abs(tb - ts) / np.maximum(np.maximum(tb, ts), 1)
    vol = pxi["volume"].fillna(0).to_numpy(float)
    vwap = pxi["vwap"].fillna(pxi["close"]).ffill().bfill().to_numpy(float)
    nm = [str(names[b]) for b in names.index]
    pn = Panel(sid, dates, ids, nm, [branch_type(x) for x in nm], B, S, BP, SP, close, vwap, vol,
               pxi["high"].to_numpy(float), pxi["low"].to_numpy(float),
               pxi["adj_close"].ffill().bfill().to_numpy(float), suspended, imb,
               {"source": "finmind", "price_source": px.attrs.get("source")})
    log.info("v3 panel %s：%d 日（停牌 %d）、%d 個分點、價格來源 %s", sid, T, int(suspended.sum()), n, pn.meta["price_source"])
    return pn


def fubon_keys(pn: Panel) -> list[str]:
    """網頁點選用：securities_trader_id → 富邦 longhist key（找不到就用原代碼）"""
    try:
        import fm_store
        from history import history_key
        return [history_key({"b_raw": fm_store.fubon_row(b, n)["b_raw"]}) for b, n in zip(pn.branches, pn.names)]
    except Exception:  # noqa: BLE001
        return list(pn.branches)
