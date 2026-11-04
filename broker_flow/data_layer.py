"""
第 2、3 節 資料層：adapter（本專案的 longhist）→ 標準資料表；交易日曆；hold-out 防護

本資料源的觀測邊界（比規格書第 1 節更窄，見 SPEC_GAPS.md）
  - 只有「日期 × 分點」的買進、賣出張數，沒有成交價位 → 第 3.3 守恆檢查、維度 B、Wasserstein 無法計算
  - 個股只有 VWAP、收盤、成交量，沒有開高低 → 6.1 的「振幅」因子、9.4 事件定義無法計算
  - 只有曾進入前 15 名或全分點反查補進來的分點，不是全部分點
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from paths import DATA_DIR

from . import store
from .params import P

LOT = 1000   # 1 張 = 1,000 股；內部一律用股
LONG_DIR = DATA_DIR / "longhist"

# 外資、自營判斷（與 flowlab/core.py INSTITUTIONAL_HINTS 相同來源）
from flowlab.core import INSTITUTIONAL_HINTS  # noqa: E402


class HoldoutError(RuntimeError):
    pass


def branch_type(name: str) -> str:
    n = name or ""
    if "自營" in n:
        return "proprietary"
    if any(h in n for h in INSTITUTIONAL_HINTS):
        return "foreign"
    return "brokerage"


def adapter_longhist(sid: str) -> dict[str, pd.DataFrame]:
    """longhist/<代號>/ → branch_daily、daily_prices、branch_master（第 2 節 schema 的子集）"""
    d = LONG_DIR / sid
    prices = json.loads((d / "_prices.json").read_text(encoding="utf-8")).get("data") or {}
    px = pd.DataFrame.from_dict(prices, orient="index")
    px.index.name = "date"
    px = px.reset_index().sort_values("date")
    daily = pd.DataFrame({"date": px["date"], "stock_id": sid,
                          "close": px["close"].astype(float), "vwap": px.get("vwap", px["close"]).astype(float).fillna(px["close"]),
                          "volume": px["vol"].astype(float) * LOT})
    rows, master = [], []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        c = json.loads(f.read_text(encoding="utf-8"))
        name = c.get("broker") or f.stem
        master.append({"branch_id": f.stem, "name": name, "branch_type": branch_type(name)})
        for r in c.get("rows") or []:
            rows.append((r["date"], f.stem, float(r.get("buy") or 0) * LOT, float(r.get("sell") or 0) * LOT))
    bd = pd.DataFrame(rows, columns=["date", "branch_id", "buy_shares", "sell_shares"])
    bd = bd.groupby(["date", "branch_id"], as_index=False).sum()
    bd.insert(1, "stock_id", sid)
    return {"branch_daily": bd, "daily_prices": daily, "branch_master": pd.DataFrame(master)}


def adapter_finmind(sid: str) -> dict[str, pd.DataFrame] | None:
    """FinMind（Sponsor Pro）全部分點的每日買賣股數與分點買／賣均價；價格仍用 longhist/_prices.json。
    FinMind 不能用（沒有資料、太舊）→ None，呼叫端改用 adapter_longhist（富邦，保留作備用）"""
    import datetime as dt

    import fm_store
    from history import history_key
    if not fm_store.enabled() or not (LONG_DIR / sid / "_prices.json").exists():
        return None
    df = fm_store.stock_table(sid)
    last = fm_store.last_date()
    if not len(df) or not last:
        return None
    base = adapter_longhist_prices(sid)
    if (dt.date.fromisoformat(str(base["date"].iloc[-1])) - dt.date.fromisoformat(last)).days > fm_store.STALE_DAYS:
        return None
    names = df.drop_duplicates("securities_trader_id", keep="last").set_index("securities_trader_id")["securities_trader"]
    key = {bid: history_key({"b_raw": fm_store.fubon_row(str(bid), str(n))["b_raw"]}) for bid, n in names.items()}
    master = pd.DataFrame({"branch_id": [key[b] for b in names.index],
                           "name": [fm_store.fubon_row(str(b), str(n))["broker"] for b, n in names.items()]})
    master["branch_type"] = master["name"].map(branch_type)
    master = master.drop_duplicates("branch_id")
    bd = pd.DataFrame({"date": df["date"], "stock_id": sid, "branch_id": df["securities_trader_id"].map(key),
                       "buy_shares": df["buy_volume"].astype(float), "sell_shares": df["sell_volume"].astype(float),
                       "buy_price": df["buy_price"].astype(float), "sell_price": df["sell_price"].astype(float)})
    bd = bd.groupby(["date", "stock_id", "branch_id"], as_index=False).agg(
        buy_shares=("buy_shares", "sum"), sell_shares=("sell_shares", "sum"), buy_price=("buy_price", "max"), sell_price=("sell_price", "max"))
    return {"branch_daily": bd, "daily_prices": base, "branch_master": master.reset_index(drop=True), "source": "finmind"}


def adapter_longhist_prices(sid: str) -> pd.DataFrame:
    prices = json.loads((LONG_DIR / sid / "_prices.json").read_text(encoding="utf-8")).get("data") or {}
    px = pd.DataFrame.from_dict(prices, orient="index")
    px.index.name = "date"
    px = px.reset_index().sort_values("date")
    return pd.DataFrame({"date": px["date"], "stock_id": sid,
                         "close": px["close"].astype(float), "vwap": px.get("vwap", px["close"]).astype(float).fillna(px["close"]),
                         "volume": px["vol"].astype(float) * LOT})


@dataclass
class Panel:
    """一檔股票的分點 × 交易日矩陣（股）。dates 為交易日曆（以個股有成交價格的日子為準）"""
    sid: str
    dates: np.ndarray            # str 'YYYY-MM-DD'
    branches: list[str]
    names: list[str]
    btype: list[str]
    B: np.ndarray                # (分點, 日) 買進股數
    S: np.ndarray                # (分點, 日) 賣出股數
    close: np.ndarray
    vwap: np.ndarray
    volume: np.ndarray
    holdout_start: int           # 第一個 hold-out 交易日的索引
    meta: dict = field(default_factory=dict)

    def guard(self, end: int, allow_holdout: bool = False) -> None:
        """第 0 節第 3 條：用到 hold-out 期間（索引 ≥ holdout_start）的資料必須明確傳 allow_holdout=True"""
        if end >= self.holdout_start and not allow_holdout:
            raise HoldoutError(f"索引 {end}（{self.dates[min(end, len(self.dates) - 1)]}）落在 hold-out 期間"
                               f"（{self.dates[self.holdout_start]} 起），需要 allow_holdout=True")


def holdout_start_index(dates: np.ndarray) -> int:
    last = pd.Timestamp(str(dates[-1]))
    cut = (last - pd.DateOffset(months=int(P.validation.holdout_months))).strftime("%Y-%m-%d")
    return int(np.searchsorted(dates, cut, side="right"))


def build_panel(sid: str, write: bool = True) -> Panel:
    t = None
    try:
        t = adapter_finmind(sid)          # FinMind 優先（完整分點＋分點買賣均價）
    except Exception as ex:  # noqa: BLE001
        import logging
        logging.getLogger("broker_scraper").warning("主力集團 %s：FinMind 讀取失敗（%s），改用富邦 longhist", sid, ex)
    source = "finmind" if t else "fubon"
    if t is None:
        t = adapter_longhist(sid)         # 富邦（保留作備用）
    t.pop("source", None)
    if write:
        for k, v in t.items():
            store.write(v, sid, k)
    daily = t["daily_prices"]
    dates = daily["date"].to_numpy(str)
    # 交易日曆：個股有價格的日子（SPEC-GAP：沒有官方日曆表；停牌日不會出現，T+2 映射本版用不到）
    pos = {d: i for i, d in enumerate(dates)}
    m = t["branch_master"]
    bix = {b: i for i, b in enumerate(m["branch_id"])}
    B = np.zeros((len(m), len(dates)))
    S = np.zeros_like(B)
    bd = t["branch_daily"]
    bd = bd[bd["date"].isin(pos)]
    ii = bd["branch_id"].map(bix).to_numpy(int)
    jj = bd["date"].map(pos).to_numpy(int)
    B[ii, jj] = bd["buy_shares"].to_numpy()
    S[ii, jj] = bd["sell_shares"].to_numpy()
    meta = {"source": source}
    if "buy_price" in bd:                 # 分點當日買／賣均價（FinMind）；沒有成交的格子是 NaN
        BP = np.full(B.shape, np.nan)
        SP = np.full(B.shape, np.nan)
        BP[ii, jj] = np.where(bd["buy_shares"].to_numpy() > 0, bd["buy_price"].to_numpy(), np.nan)
        SP[ii, jj] = np.where(bd["sell_shares"].to_numpy() > 0, bd["sell_price"].to_numpy(), np.nan)
        meta.update(buy_price=BP, sell_price=SP)
    return Panel(sid, dates, list(m["branch_id"]), list(m["name"]), list(m["branch_type"]), B, S,
                 daily["close"].to_numpy(float), daily["vwap"].to_numpy(float), daily["volume"].to_numpy(float),
                 holdout_start_index(dates), meta)
