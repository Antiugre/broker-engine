"""
每日運算範圍（universe）與全市場日成交
=====================================

順序（越前面越優先；時間不夠時後面的會留到下次）
  1. Ultra 會員的自選股（Firestore users/*，tier == "ultra"，依被選次數排序）
  2. 全市場成交值前 N 名（local_settings.json 的 top_n，預設 100；排除 ETF、權證）

全市場日成交（一次請求拿到所有股票，不必逐檔抓）
  上市：證交所 OpenAPI STOCK_DAY_ALL　　上櫃：櫃買中心 OpenAPI 每日收盤行情
  用途：(a) 挑成交值前 N 名　(b) 直接把當天的成交均價／收盤／張數補進每檔的價格快取
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re

from paths import DATA_DIR, TOP_N

log = logging.getLogger("broker_scraper")
TWSE_ALL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_ALL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"
STOCK_RE = re.compile(r"^[1-9]\d{3}$")          # 一般股票（排除 00 開頭 ETF、權證、特別股）
WATCH_RE = re.compile(r"^\d{4,6}[A-Z]?$")


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _roc_date(s) -> str | None:
    s = re.sub(r"\D", "", str(s or ""))
    if len(s) == 7:   # 1141030
        return f"{int(s[:3]) + 1911}-{s[3:5]}-{s[5:]}"
    if len(s) == 8:   # 20251030
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return None


def _pick(d: dict, *keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def parse_market(twse: list[dict], tpex: list[dict]) -> tuple[str | None, dict[str, dict]]:
    """→ (資料日期, {代號: {name, close, vol（張）, value（元）, vwap}})"""
    out, dates = {}, set()
    for rows, kind in ((twse or [], "twse"), (tpex or [], "tpex")):
        for x in rows:
            code = str(_pick(x, "Code", "SecuritiesCompanyCode", "證券代號") or "").strip()
            if not code:
                continue
            shares = _num(_pick(x, "TradeVolume", "TradingShares", "成交股數"))
            value = _num(_pick(x, "TradeValue", "TransactionAmount", "成交金額"))
            close = _num(_pick(x, "ClosingPrice", "Close", "收盤價"))
            d = _roc_date(_pick(x, "Date", "日期"))
            if d:
                dates.add(d)
            if not shares or not value or not close:
                continue
            out[code] = {"name": _pick(x, "Name", "CompanyName", "證券名稱"), "close": close, "vol": round(shares / 1000),
                         "value": value, "vwap": round(value / shares, 2), "market": kind}
    return (max(dates) if dates else None), out


def fetch_market() -> tuple[str | None, dict[str, dict]]:
    import requests
    got = []
    for url in (TWSE_ALL, TPEX_ALL):
        try:
            got.append(requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json())
        except Exception as ex:  # noqa: BLE001
            log.info("全市場日成交 %s 失敗：%s", url, ex)
            got.append([])
    return parse_market(*got)


def apply_to_price_cache(day: str, market: dict[str, dict], market_date: str | None) -> int:
    """把當天行情直接補進每檔的價格快取（只補已完整建檔、且只差最近幾天的股票）。回傳補了幾檔"""
    import quant
    if market_date != day:
        log.info("全市場日檔日期 %s ≠ %s，不使用（改由各檔增量更新）", market_date, day)
        return 0
    n = 0
    for f in quant.LONG_DIR.glob("*/_prices.json"):
        sid = f.parent.name
        m = market.get(sid)
        if not m:
            continue
        try:
            c = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if c.get("v") != 2 or not c.get("end") or c["end"] >= day:
            continue
        if (dt.date.fromisoformat(day) - dt.date.fromisoformat(c["end"])).days > 5:  # 中間缺太多天 → 交給增量更新
            continue
        c["data"][day] = {"vwap": m["vwap"], "close": m["close"], "vol": m["vol"]}
        c["end"] = day
        f.write_text(json.dumps(c, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        n += 1
    return n


def ultra_watchlists(db) -> list[str]:
    """Ultra 會員自選股（依被選次數排序）；同時寫一份匯總到 data/universe/ultra_watch.json"""
    counts: dict[str, int] = {}
    try:
        for snap in db.collection("users").where("tier", "==", "ultra").stream():
            for sid in (snap.to_dict() or {}).get("watchlist") or []:
                sid = str(sid).strip().upper()
                if WATCH_RE.match(sid):
                    counts[sid] = counts.get(sid, 0) + 1
    except Exception as ex:  # noqa: BLE001
        log.warning("讀取 Ultra 會員自選股失敗：%s", ex)
        return []
    ranked = sorted(counts, key=lambda s: (-counts[s], s))
    p = DATA_DIR / "universe" / "ultra_watch.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"updated": dt.datetime.now().isoformat(timespec="seconds"), "counts": counts},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    return ranked


def top_by_value(market: dict[str, dict], n: int) -> list[str]:
    return [s for s, _ in sorted(((s, m) for s, m in market.items() if STOCK_RE.match(s)), key=lambda x: -x[1]["value"])][:n]


def build(day: str, db=None, market: dict | None = None, top_n: int = TOP_N) -> dict:
    """→ {"stocks": [...依優先順序], "sources": {代號: "ultra"|"top"}}，並存 data/universe/<day>.json"""
    ultra = ultra_watchlists(db) if db is not None else []
    top = top_by_value(market or {}, top_n) if market else []
    path = DATA_DIR / "universe" / f"{day}.json"
    if not top:  # 抓不到全市場資料 → 沿用最近一次的前 N 名
        prev = sorted((DATA_DIR / "universe").glob("????-??-??.json"))
        if prev:
            old = json.loads(prev[-1].read_text(encoding="utf-8"))
            top = [s for s, src in old.get("sources", {}).items() if src == "top"]
    order, src = [], {}
    for kind, lst in (("ultra", ultra), ("top", top)):
        for s in lst:
            if s not in src:
                src[s] = kind
                order.append(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"day": day, "stocks": order, "sources": src}, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("今日運算範圍：%d 檔（Ultra 自選 %d、成交值前 %d 名 %d）", len(order),
             sum(v == "ultra" for v in src.values()), top_n, sum(v == "top" for v in src.values()))
    return {"stocks": order, "sources": src}
