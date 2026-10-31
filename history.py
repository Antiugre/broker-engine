"""
分點近半年進出歷史 + 成本估算
--------------------------------------------------
1. 富邦「單一券商歷史明細」：每日 買進 / 賣出 / 買賣超 張數
   https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco0/zco0.djhtm?A=<代號>&BHID=<總公司>&b=<分點>&C=1&D=<起>&E=<迄>&ver=V3
2. 每日成交均價（成交金額 ÷ 成交股數）
   上市：https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY?date=YYYYMM01&stockNo=<代號>&response=json
   上櫃：https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock?code=<代號>&date=YYYY/MM/01&response=json
   都取不到時，改用富邦頁面內嵌的每日收盤價。

公開資料沒有「分點實際成交價」，因此成本皆為估算：
  （區間 = 近半年；另提供近一個月的 avg_buy_cost_1m / avg_sell_cost_1m）
  avg_buy_cost  區間買進均價 = Σ(買進張數 × 當日均價) ÷ Σ買進張數        ← 主要購買成本
  avg_sell_cost 區間賣出均價 = Σ(賣出張數 × 當日均價) ÷ Σ賣出張數
  hold_cost     持有成本：以區間起點持股 0 張起算，買超日以當日均價加權攤入、
                賣超日減少張數（成本不變，最低減到 0 張）；部位歸零後，下次買超重新起算。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("broker_scraper")

HIST_URL = "https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco0/zco0.djhtm"
TWSE_DAY = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TPEX_DAY = "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock"
HISTORY_DAYS = 183  # 近半年（日曆天）
RECENT_DAYS = 31    # 「近一個月」均價


def _num(s: str) -> float | None:
    s = (s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _ymd(d: dt.date) -> str:
    return f"{d.year}-{d.month}-{d.day}"


def history_url(stock_id: str, bhid: str, b_raw: str, start: dt.date, end: dt.date) -> str:
    return f"{HIST_URL}?A={stock_id}&BHID={bhid}&b={b_raw}&C=1&D={_ymd(start)}&E={_ymd(end)}&ver=V3"


def history_key(row: dict) -> str:
    """歷史檔名：用富邦原始分點參數（十六進位編碼可區分大小寫，Windows 檔名不分大小寫）"""
    key = row.get("b_raw") or row.get("branch_id") or row.get("broker") or "unknown"
    return re.sub(r"[^\w\-]", "_", key)


def parse_history(html: str, end: dt.date) -> tuple[list[dict], dict[str, float]]:
    """回傳 (每日列, {日期: 收盤價})"""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 5:
            continue
        d = tds[0].get_text(strip=True)
        if not re.fullmatch(r"\d{4}/\d{2}/\d{2}", d):
            continue
        buy, sell = _num(tds[1].get_text()), _num(tds[2].get_text())
        net = _num(tds[4].get_text())
        rows.append({"date": d.replace("/", "-"), "buy": int(buy or 0), "sell": int(sell or 0),
                     "net": int(net if net is not None else (buy or 0) - (sell or 0))})
    rows.sort(key=lambda r: r["date"])

    closes: dict[str, float] = {}
    m = re.search(r"'Volumn'\s*:\s*'([^']*)'", html)
    if m:
        parts = m.group(1).split(" ")
        if len(parts) >= 2:
            for md, px in zip(parts[0].split(","), parts[1].split(",")):
                if len(md) != 4 or not px:
                    continue
                mm, dd = int(md[:2]), int(md[2:])
                year = end.year if (mm, dd) <= (end.month, end.day) else end.year - 1
                try:
                    closes[dt.date(year, mm, dd).isoformat()] = float(px)
                except ValueError:
                    pass
    return rows, closes


# ──────────────────────────── 每日均價 ────────────────────────────
def _months(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _roc_to_iso(s: str) -> str | None:
    m = re.match(r"(\d{2,3})/(\d{1,2})/(\d{1,2})", s.strip())
    if not m:
        return None
    return dt.date(int(m[1]) + 1911, int(m[2]), int(m[3])).isoformat()


def fetch_daily(stock_id: str, start: dt.date, end: dt.date, session: requests.Session | None = None
                ) -> tuple[dict[str, float], dict[str, float], str]:
    """回傳 ({日期: 成交均價}, {日期: 收盤價}, 來源)。先試上市，再試上櫃。"""
    s = session or requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36",
               "Accept": "application/json"}
    for market in ("twse", "tpex"):
        vwap: dict[str, float] = {}
        close: dict[str, float] = {}
        try:
            for k, (y, m) in enumerate(_months(start, end)):
                if k:
                    time.sleep(0.8)  # 證交所/櫃買有頻率限制
                if market == "twse":
                    r = s.get(TWSE_DAY, params={"date": f"{y}{m:02d}01", "stockNo": stock_id, "response": "json"},
                              headers=headers, timeout=20)
                    j = r.json()
                    if j.get("stat") != "OK":
                        break
                    data = j.get("data") or []
                else:
                    r = s.get(TPEX_DAY, params={"code": stock_id, "date": f"{y}/{m:02d}/01", "response": "json"},
                              headers=headers, timeout=20)
                    j = r.json()
                    tables = j.get("tables") or []
                    data = tables[0].get("data") if tables else []
                # 欄位：日期, 成交股數, 成交金額, 開盤, 最高, 最低, 收盤, …（上市、上櫃相同順序）
                for row in data or []:
                    d = _roc_to_iso(row[0])
                    if not d:
                        continue
                    shares, amount = _num(row[1]), _num(row[2])
                    if shares and amount:
                        vwap[d] = round(amount / shares, 2)  # 上市：元/股；上櫃：仟元/仟股，比值相同
                    c = _num(row[6]) if len(row) > 6 else None
                    if c:
                        close[d] = c
        except Exception as ex:  # noqa: BLE001
            log.debug("%s 日成交資訊 %s 失敗：%s", stock_id, market, ex)
            vwap, close = {}, {}
        if vwap or close:
            return vwap, close, market
    return {}, {}, "none"


def fetch_vwap(stock_id, start, end, session=None):  # 舊介面相容
    vwap, _close, src = fetch_daily(stock_id, start, end, session)
    return vwap, src


# ──────────────────────────── 成本計算 ────────────────────────────
def compute_costs(rows: list[dict], prices: dict[str, float]) -> dict:
    pos, cost = 0.0, None
    buy_amt = buy_qty = sell_amt = sell_qty = 0.0
    for r in rows:
        p = prices.get(r["date"])
        r["price"] = p
        if p is not None:
            buy_amt += r["buy"] * p
            buy_qty += r["buy"]
            sell_amt += r["sell"] * p
            sell_qty += r["sell"]
        n = r["net"]
        if n > 0 and p is not None:
            cost = p if pos <= 0 or cost is None else (pos * cost + n * p) / (pos + n)
            pos += n
        elif n < 0:
            pos = max(0.0, pos + n)  # 推估持股最低為 0：賣超只減少已累積的部位
            if pos == 0:
                cost = None
        r["hold_lots"] = int(round(pos))
        r["hold_cost"] = round(cost, 2) if cost is not None and pos > 0 else None
    def _avg(sub, key):
        q = sum(r[key] for r in sub if r.get("price") is not None)
        a = sum(r[key] * r["price"] for r in sub if r.get("price") is not None)
        return round(a / q, 2) if q else None

    recent_from = ""
    if rows:
        recent_from = (dt.date.fromisoformat(rows[-1]["date"]) - dt.timedelta(days=RECENT_DAYS)).isoformat()
    recent = [r for r in rows if r["date"] > recent_from]
    return {
        "avg_buy_cost_1m": _avg(recent, "buy"),
        "avg_sell_cost_1m": _avg(recent, "sell"),
        "net_1m": int(sum(r["net"] for r in recent)),
        "avg_buy_cost": round(buy_amt / buy_qty, 2) if buy_qty else None,
        "avg_sell_cost": round(sell_amt / sell_qty, 2) if sell_qty else None,
        "total_buy": int(sum(r["buy"] for r in rows)),
        "total_sell": int(sum(r["sell"] for r in rows)),
        "total_net": int(sum(r["net"] for r in rows)),
        "hold_lots": rows[-1]["hold_lots"] if rows else 0,
        "hold_cost": rows[-1]["hold_cost"] if rows else None,
    }


def enrich_with_history(rec: dict, fetcher, data_dir: Path, end: dt.date, progress=None,
                        delay: tuple[float, float] = (1.0, 2.5)) -> None:
    """為 rec 內每個分點抓近半年歷史、算成本，寫入 data/history/<代號>/<分點>.json"""
    sid = rec["stock_id"]
    start = end - dt.timedelta(days=HISTORY_DAYS)
    rows_all = [r for side in ("buy_side", "sell_side") for r in rec.get(side, [])]
    hist_dir = data_dir / "history" / sid
    hist_dir.mkdir(parents=True, exist_ok=True)

    # 價格與分點明細都改用 quant 的長期快取（約 5 年、每天只增量更新），不再每天重抓半年
    import quant
    try:
        px = quant.load_prices(sid, end)
        vwap = {d: v["vwap"] for d, v in px.items() if v.get("vwap")}
        daily_close = {d: v["close"] for d, v in px.items() if v.get("close")}
        vsrc = "twse"
    except Exception as ex:  # noqa: BLE001
        log.info("[%s] 長期價格快取失敗（%s），改用證交所／櫃買逐月", sid, ex)
        vwap, daily_close, vsrc = fetch_daily(sid, start, end)
    closes: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for i, row in enumerate(rows_all, 1):
        if progress:
            progress(f"{sid} {rec.get('stock_name') or ''}：{row['broker']} 近半年明細", i, len(rows_all))
        key = history_key(row)
        if key in seen:  # 同一分點同時出現在買超/賣超（區間合計時可能發生）
            row.update({k: seen[key].get(k) for k in ("avg_buy_cost", "avg_sell_cost", "avg_buy_cost_1m", "avg_sell_cost_1m",
                                                     "hold_lots", "hold_cost", "history")})
            continue
        if not row.get("b_raw") or not row.get("broker_id"):
            continue
        url = history_url(sid, row["broker_id"], row["b_raw"], start, end)
        try:
            # 單一券商歷史頁晚上才會補上當天；用當日主頁的買賣張數補齊最後一天
            today_row = ([{"date": end.isoformat(), "buy": int(row.get("buy") or 0), "sell": int(row.get("sell") or 0)}]
                         if row.get("buy") is not None else [])
            longrows = quant.load_branch_history(fetcher, sid, row, end, extra_rows=today_row)
            days = [{"date": r["date"], "buy": int(r["buy"]), "sell": int(r["sell"]), "net": int(r["buy"]) - int(r["sell"])}
                    for r in longrows if start.isoformat() <= r["date"] <= end.isoformat()]
            all_close = {**closes, **daily_close}
            summary = compute_costs(days, {**all_close, **vwap})  # 成本用成交均價，缺的日子用收盤價補
            for d in days:
                d["close"] = all_close.get(d["date"])
            price_src = {"twse": "證交所成交均價", "tpex": "櫃買中心成交均價"}.get(vsrc, "收盤價")
            out = {
                "stock_id": sid, "stock_name": rec.get("stock_name"), "broker": row["broker"],
                "branch_id": row.get("branch_id"), "city": row.get("city"), "district": row.get("district"),
                "start": start.isoformat(), "end": end.isoformat(), "price_source": price_src,
                "updated": dt.datetime.now().isoformat(timespec="seconds"), "url": url,
                **summary, "rows": days,
            }
            (hist_dir / f"{key}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            row.update(avg_buy_cost=summary["avg_buy_cost"], avg_sell_cost=summary["avg_sell_cost"],
                       avg_buy_cost_1m=summary["avg_buy_cost_1m"], avg_sell_cost_1m=summary["avg_sell_cost_1m"],
                       hold_lots=summary["hold_lots"], hold_cost=summary["hold_cost"],
                       history=f"history/{sid}/{key}.json")
            seen[key] = row
        except Exception as ex:  # noqa: BLE001
            log.warning("[%s] %s 歷史抓取失敗：%s", sid, row["broker"], ex)
    rec["history_range"] = {"start": start.isoformat(), "end": end.isoformat(), "price_source": vsrc}


def rebuild_costs(data_dir: Path, progress=None) -> int:
    """用已下載的歷史檔重算成本（改了計算方式時不必重新下載）；缺收盤價的會補抓。回傳更新的分點數"""
    hist_root = data_dir / "history"
    if not hist_root.exists():
        return 0
    # 舊版歷史檔沒有收盤價：每檔股票補抓一次日成交資訊
    fill: dict[str, dict[str, float]] = {}
    stock_dirs = [d for d in hist_root.iterdir() if d.is_dir()]
    for i, sdir in enumerate(stock_dirs, 1):
        files = list(sdir.glob("*.json"))
        try:
            sample = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        except Exception:  # noqa: BLE001
            continue
        if not any(r.get("close") is None for h in sample for r in h.get("rows", [])):
            continue
        dates = [r["date"] for h in sample for r in h.get("rows", [])]
        if not dates:
            continue
        if progress:
            progress(f"補抓 {sdir.name} 收盤價（{i}/{len(stock_dirs)}）")
        _v, close, _src = fetch_daily(sdir.name, dt.date.fromisoformat(min(dates)), dt.date.fromisoformat(max(dates)))
        fill[sdir.name] = close
    cache: dict[str, dict] = {}
    n = 0
    for f in hist_root.glob("*/*.json"):
        try:
            h = json.loads(f.read_text(encoding="utf-8"))
            close = fill.get(f.parent.name)
            if close:
                for r in h.get("rows", []):
                    if r.get("close") is None:
                        r["close"] = close.get(r["date"])
            prices = {r["date"]: r["price"] for r in h.get("rows", []) if r.get("price") is not None}
            h.update(compute_costs(h.get("rows", []), prices))
            f.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")
            cache[f"history/{f.parent.name}/{f.name}"] = h
            n += 1
        except Exception as ex:  # noqa: BLE001
            log.warning("重算 %s 失敗：%s", f, ex)
    for f in data_dir.glob("????-??-??/*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            changed = False
            for rec in (data if isinstance(data, list) else [data]):
                for side in ("buy_side", "sell_side"):
                    for row in rec.get(side, []):
                        h = cache.get(row.get("history") or "")
                        if h:
                            for k in ("avg_buy_cost", "avg_sell_cost", "avg_buy_cost_1m", "avg_sell_cost_1m", "hold_lots", "hold_cost"):
                                row[k] = h.get(k)
                            changed = True
            if changed:
                f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as ex:  # noqa: BLE001
            log.warning("更新 %s 失敗：%s", f, ex)
    return n
