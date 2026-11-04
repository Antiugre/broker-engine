"""
mainforce.py — 本土主力（分點估）：把每個本土券商分點當成一個大戶，用 5 年分點明細推算它目前還抱著多少張
====================================================================================================

  庫存（不偷看未來）  c_t = Σ 淨買超（張）；h_t = c_t − min_{s≤t} c_s
                      → 以「到當天為止的累計最低點」當 0，算出至少還抱著的張數（起點之前的部位看不到）
  平均成本            平均成本法：買進時依當日成交均價加權；賣出時按比例扣減（成本不變）；庫存歸 0 時重設
  本土主力            排除外資券商與自營（名稱判斷，同 flowlab.INSTITUTIONAL_HINTS）、排除隔日沖型分點
                      （近 1 年大買後隔日賣出 ≥ 70% 的次數過多）、排除散戶混合型大分點（周轉率過高），
                      庫存 ≥ MIN_LOTS 張才計入合計
  地緣                與公司同郵遞區號／同區／同縣市（flowlab.cluster.GeoMatcher）

  輸出 data/force/<日>/<代號>.json → Firestore force/{代號}（Ultra 限定）
  {
    stock_id, asof, min_lots, n_branches（有明細的本土分點數）,
    total: {dates: [週五…], lots: [合計張數…]},                  ← 每週一筆（約 5 年），前端換算佔發行股數
    top:   [{key, broker, city, district, geo, lots, cost, since, last_add, tag}],   ← 最新一天庫存最多的 15 個
    weekly: {dates: [...], rows: {key: [張數…]}},                  ← 這些分點每週庫存（前端「資料日」用）
    meta:  {key: {broker, city, district, geo}}
  }

  python mainforce.py 2026-10-30 [代號…]
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

import numpy as np
import pandas as pd

import quant

log = logging.getLogger("broker_scraper")

OUT_DIR = quant.DATA_DIR / "force"
MIN_LOTS = 200          # 庫存至少幾張才算「主力」計入合計
TOP_N = 15              # 排行與每週序列保留幾個分點
WEEKLY_KEEP = 40        # 每週序列多保留幾個（資料日往前看時，排行可能換人）
FLIP_SHARE = 0.10       # 隔日沖判斷：當日淨買 ≥ 當日成交量 10%…
FLIP_SELL = 0.70        # …隔日賣出 ≥ 70%
FLIP_RATIO = 0.5        # 近 1 年這類大買中，隔日沖比例過半 → 視為隔日沖分點
MIXED_TURNOVER = 12     # 近 1 年買＋賣張數 ÷ 目前庫存 > 12 → 散戶混合型大分點（淨額只是客戶互相抵銷的結果），不計入


def _institutional(name: str) -> bool:
    from flowlab.core import INSTITUTIONAL_HINTS
    return any(h in (name or "") for h in INSTITUTIONAL_HINTS)


def branch_sources(sid: str, d):
    """每個分點的（key, 名稱, 每日 DataFrame[date, buy, sell(, buy_price)]）：FinMind（Sponsor Pro）優先，
    不能用時讀 longhist（富邦，保留作備用）"""
    try:
        import fm_store
        fm = fm_store.branch_series(sid)
    except Exception as ex:  # noqa: BLE001
        log.warning("[%s] FinMind 分點讀取失敗（%s），改讀 longhist", sid, ex)
        fm = None
    if fm:
        return fm
    out = []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        cache = quant._read(f) or {}
        if not cache.get("rows"):
            continue
        df = pd.DataFrame(cache["rows"])
        if not df.empty:
            out.append((f.stem, cache.get("broker") or "", df))
    return out


def _holdings(net: np.ndarray, vwap: np.ndarray):
    """回傳 (庫存, 平均成本, 最近一次歸零後的起始索引)；全部向量長度相同"""
    n = len(net)
    hold = np.zeros(n)
    cost = np.full(n, np.nan)
    since = np.zeros(n, dtype=int)
    h, c, s = 0.0, 0.0, 0
    for i in range(n):
        x = net[i]
        if x > 0:
            px = vwap[i] if np.isfinite(vwap[i]) else c
            c = (c * h + px * x) / (h + x) if h + x > 0 else px
            if h <= 0:
                s = i
            h += x
        elif x < 0:
            h = max(0.0, h + x)          # 賣超超過庫存 → 歸零（等同以新的累計最低點當 0）
            if h == 0:
                c = 0.0
        hold[i], cost[i], since[i] = h, (c if h > 0 else np.nan), s
    return hold, cost, since


def _is_flipper(net: np.ndarray, vol: np.ndarray, last_n: int = 250) -> bool:
    net, vol = net[-last_n:], vol[-last_n:]
    big = np.where((vol > 0) & (net >= FLIP_SHARE * vol))[0]
    big = big[big < len(net) - 1]
    if len(big) < 3:
        return False
    flips = sum(1 for i in big if -net[i + 1] >= FLIP_SELL * net[i])
    return flips / len(big) > FLIP_RATIO


def compute(sid: str, day: str) -> dict | None:
    from flowlab.cluster import GeoMatcher
    from flowlab.sources import company_directory, zip_table

    d = quant.LONG_DIR / sid
    if not d.exists():
        return None
    prices = quant._read(d / "_prices.json") or {}
    px = pd.DataFrame.from_dict(prices.get("data") or {}, orient="index")
    if px.empty:
        return None
    px.index = pd.to_datetime(px.index)
    px = px[px.index <= pd.Timestamp(day)].sort_index()
    for c in ("vwap", "close", "vol"):
        if c not in px:
            px[c] = np.nan
    px["vwap"] = px["vwap"].fillna(px["close"])
    idx = px.index
    vwap = px["vwap"].to_numpy(float)
    vol = px["vol"].to_numpy(float)

    try:
        from broker_location import Locator, load_directory
        directory = load_directory(quiet=True)
        loc = Locator(directory) if directory else None
    except Exception:  # noqa: BLE001
        loc = None
    geo = GeoMatcher(company_directory().get(sid), zip_table())

    series: dict[str, np.ndarray] = {}
    info: dict[str, dict] = {}
    mixed = 0
    for k, name, df in branch_sources(sid, d):
        if _institutional(name):
            continue
        df["date"] = pd.to_datetime(df["date"])
        bp = df.groupby("date")["buy_price"].max().reindex(idx).to_numpy(float) if "buy_price" in df else None
        df = df.groupby("date")[["buy", "sell"]].sum().reindex(idx).fillna(0.0)
        net = (df["buy"] - df["sell"]).to_numpy(float)
        if not np.any(net) or _is_flipper(net, vol):
            continue
        # 成本：FinMind 有分點自己的買進均價就用它，沒有才用個股成交均價
        hold, cost, since = _holdings(net, vwap if bp is None else np.where(np.isfinite(bp), bp, vwap))
        if hold.max() < MIN_LOTS:
            continue
        gross = float((df["buy"] + df["sell"]).to_numpy()[-250:].sum())
        if hold[-1] > 0 and gross / hold[-1] > MIXED_TURNOVER:
            mixed += 1
            continue
        series[k] = hold
        e = loc.lookup(None, name) if loc else None
        m = {"broker": name, "city": e["city"] if e else None, "district": e["district"] if e else None}
        m["geo"] = round(geo.weight(m), 2)
        last_add = int(np.max(np.where(net > 0)[0])) if np.any(net > 0) else None
        info[k] = {**m, "cost": None if np.isnan(cost[-1]) else round(float(cost[-1]), 2),
                   "since": idx[since[-1]].strftime("%Y-%m-%d") if hold[-1] > 0 else None,
                   "last_add": idx[last_add].strftime("%Y-%m-%d") if last_add is not None else None}
    if not series:
        return {"stock_id": sid, "asof": idx[-1].strftime("%Y-%m-%d"), "min_lots": MIN_LOTS, "n_branches": 0,
                "total": {"dates": [], "lots": []}, "top": [], "weekly": {"dates": [], "rows": {}}, "meta": {}}
    H = pd.DataFrame(series, index=idx)
    total = H.where(H >= MIN_LOTS, 0).sum(axis=1)
    wk = H.resample("W-FRI").last().dropna(how="all")
    wk.index = [min(t, idx[-1]) for t in wk.index]          # 最後一週截在最新交易日
    tot_w = total.resample("W-FRI").last().dropna()
    tot_w.index = [min(t, idx[-1]) for t in tot_w.index]
    last = H.iloc[-1].sort_values(ascending=False)
    top_keys = [k for k in last.index if last[k] >= MIN_LOTS][:TOP_N]
    peak = H.max().sort_values(ascending=False)   # 資料日往前看：曾經抱很多的也保留
    keep = list(dict.fromkeys(top_keys + list(peak.index[:WEEKLY_KEEP])))
    iso = lambda t: pd.Timestamp(t).strftime("%Y-%m-%d")  # noqa: E731
    return {
        "stock_id": sid, "asof": iso(idx[-1]), "min_lots": MIN_LOTS, "n_branches": len(series), "n_mixed": mixed,
        "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "total": {"dates": [iso(t) for t in tot_w.index], "lots": [int(round(v)) for v in tot_w.to_numpy()]},
        "top": [{"key": k, **{x: info[k][x] for x in ("broker", "city", "district", "geo", "cost", "since", "last_add")},
                 "lots": int(round(last[k]))} for k in top_keys],
        "weekly": {"dates": [iso(t) for t in wk.index], "rows": {k: [int(round(v)) for v in wk[k].fillna(0).to_numpy()] for k in keep}},
        "meta": {k: {x: info[k][x] for x in ("broker", "city", "district", "geo")} for k in keep},
    }


def run(day: str, stocks: list[str]) -> int:
    t0 = time.time()
    out = OUT_DIR / day
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid in stocks:
        try:
            doc = compute(sid, day)
            if doc is None:
                continue
            (out / f"{sid}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            n += 1
        except Exception as ex:  # noqa: BLE001
            log.warning("[%s] 本土主力計算失敗：%s", sid, ex)
    log.info("本土主力（分點估）：%d 檔（%.0f 秒）", n, time.time() - t0)
    return n


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    day = sys.argv[1]
    stocks = sys.argv[2:] or sorted(p.name for p in quant.LONG_DIR.iterdir() if p.is_dir())
    run(day, stocks)
