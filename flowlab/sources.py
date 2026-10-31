"""
flowlab.sources — Extract：從本機快取組出 StockContext

  分點宇宙   近 60 個有資料日的買超／賣超前 15 名 ∪ 近 20 日區間合計的買／賣前 15 名
             ∪ 全分點反查（sweep.py）找到、已建好明細的額外分點
  分點明細   quant.load_branch_history（約 5 年快取，每天只補最近幾天）
  個股行情   quant.load_prices（成交均價、收盤價、成交張數）
  公司地址   證交所／櫃買 OpenAPI 公司基本資料（本機快取 7 天）→ 地緣比對
  郵遞區號   data/zip3.csv（選用；欄位 city,district,zip）
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import time

import pandas as pd

import quant

from .core import StockContext

log = logging.getLogger("broker_scraper")
COMPANY_FILE = quant.DATA_DIR / "company_info.json"
ZIP_FILE = quant.DATA_DIR / "zip3.csv"
COMPANY_URLS = ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L",       # 上市
                "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O")    # 上櫃


def universe(fetcher, sid: str, day: str, lookback_days: int = 60, range_days: int = 30) -> dict[str, dict]:
    from history import history_key

    out: dict[str, dict] = {}
    days = sorted(p.name for p in quant.DATA_DIR.glob("????-??-??") if p.is_dir() and p.name <= day)[-lookback_days:]
    for d in reversed(days):
        f = quant.DATA_DIR / d / f"{sid}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        for side in ("buy_side", "sell_side"):
            for row in rec.get(side, []):
                if row.get("b_raw") and row.get("broker_id"):
                    out.setdefault(history_key(row), {**row, "stock_name": rec.get("stock_name")})
    try:  # 前 15 名以外：全分點反查找到、已建好 5 年明細的分點
        import sweep
        for k, r in sweep.engine_extras(day, [sid]).get(sid, {}).items():
            out.setdefault(k, r)
    except Exception as ex:  # noqa: BLE001
        log.warning("[%s] 額外分點讀取失敗：%s", sid, ex)
    if fetcher is not None:
        try:
            rec = quant.range_record(fetcher, sid, day, range_days)
            for side in ("buy_side", "sell_side"):
                for row in rec.get(side, []):
                    if row.get("b_raw") and row.get("broker_id"):
                        out.setdefault(history_key(row), {**row, "stock_name": rec.get("stock_name"), "_hot": True})
        except Exception as ex:  # noqa: BLE001
            log.warning("[%s] 區間榜失敗：%s", sid, ex)
    return out


def company_directory(max_age_days: int = 7) -> dict[str, dict]:
    """{代號: {address, zip, city, district}}"""
    from .cluster import GeoMatcher

    try:
        cached = json.loads(COMPANY_FILE.read_text(encoding="utf-8"))
        if time.time() - cached.get("t", 0) < max_age_days * 86400:
            return cached["data"]
    except Exception:  # noqa: BLE001
        cached = None
    import requests
    data: dict[str, dict] = {}
    for url in COMPANY_URLS:
        try:
            for x in requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json():
                code = x.get("公司代號") or x.get("SecuritiesCompanyCode") or x.get("CompanyCode")
                addr = x.get("住址") or x.get("Address") or x.get("CompanyAddress")
                if code and addr:
                    data[str(code).strip()] = {"address": addr, **GeoMatcher.parse(addr)}
        except Exception as ex:  # noqa: BLE001
            log.info("公司基本資料 %s 失敗：%s", url, ex)
    if data:
        COMPANY_FILE.parent.mkdir(parents=True, exist_ok=True)
        COMPANY_FILE.write_text(json.dumps({"t": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
        return data
    return (cached or {}).get("data", {})


def zip_table() -> dict:
    if not ZIP_FILE.exists():
        return {}
    with ZIP_FILE.open(encoding="utf-8") as f:
        return {(r["city"], r["district"]): r["zip"][:3] for r in csv.DictReader(f) if r.get("zip")}


def build_context(sid: str, day: str, branches: dict[str, dict], histories: dict[str, list[dict]],
                  prices: dict[str, dict], company: dict | None = None, zips: dict | None = None,
                  params: dict | None = None, keep_days: int = 1400) -> StockContext:
    """純資料組裝（不連網）：把分點明細與行情對齊成「交易日 × 分點」矩陣"""
    px = pd.DataFrame.from_dict(prices, orient="index")
    px.index = pd.to_datetime(px.index)
    px = px[px.index <= pd.Timestamp(day)].sort_index()
    for c in ("vwap", "close", "vol"):
        if c not in px:
            px[c] = float("nan")
    px["vwap"] = px["vwap"].fillna(px["close"])
    px = px.dropna(subset=["close"]).tail(keep_days)
    buy, sell = {}, {}
    for k, rows in histories.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.groupby("date")[["buy", "sell"]].sum()
        buy[k], sell[k] = df["buy"], df["sell"]
    B = pd.DataFrame(buy).reindex(px.index).fillna(0.0)
    S = pd.DataFrame(sell).reindex(px.index).fillna(0.0)
    name = next((b.get("stock_name") for b in branches.values() if b.get("stock_name")), None)
    meta = {k: {x: b.get(x) for x in ("broker", "branch_id", "broker_id", "b_raw", "city", "district", "address")}
            for k, b in branches.items()}
    return StockContext(stock_id=sid, stock_name=name, asof=px.index[-1].strftime("%Y-%m-%d") if len(px) else day,
                        px=px[["vwap", "close", "vol"]].astype(float), buy=B.astype(float), sell=S.astype(float),
                        meta=meta, company=company, zip_table=zips, params=params or {})


def extract_context(fetcher, sid: str, day: str, params: dict | None = None, max_branches: int = 60) -> StockContext | None:
    end = dt.date.fromisoformat(day)
    branches = universe(fetcher, sid, day)
    if not branches:
        return None
    branches = dict(list(branches.items())[:max_branches])
    prices = quant.load_prices(sid, end)
    histories = {}
    for k, row in branches.items():
        try:
            histories[k] = quant.load_branch_history(fetcher, sid, row, end,
                                                     max_age=quant.HOT_MAX_AGE if row.get("_hot") else None)
        except Exception as ex:  # noqa: BLE001
            log.warning("[%s] %s 明細失敗：%s", sid, row.get("broker"), ex)
    comp = company_directory().get(sid)
    return build_context(sid, day, branches, histories, prices, comp, zip_table(), params)
