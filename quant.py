"""
特定分點勝率演算與跟單訊號（Ultra 功能的運算核心）
==================================================

模組一　交易配對與績效
  fifo_match()      先進先出（FIFO）配對：每日「買超」建立一批庫存（張數、當日成交均價），
                    每日「賣超」從最早的庫存開始扣；每次扣抵 = 一筆完整交易（Trade）。
  performance()     勝率、總已實現損益、平均每筆報酬率、平均持有天數、交易風格

模組二　未實現損益與跟單訊號
  unrealized()      尚未配對（未出場）的庫存：張數、平均成本、未實現損益
  detect_signal()   高勝率（> 70%、平均報酬 > 0、交易數夠多）的分點，
                    近 3 個交易日大額買超且尚未出場 → 跟單訊號

資料流（pipeline.py 每天收盤後呼叫 run()）
  候選分點 = 追蹤股票近 3 個交易日「買超／賣超前 15 名」出現過的分點
  每個（股票, 分點）抓富邦單一分點歷史（最長約 5 年，本機快取、每天只補最近幾天）
  價格 = 每日成交均價（FinMind 一次取得長區間；失敗用證交所／櫃買逐月）
  結果寫到 data/quant/<日期>.json，由 firebase_upload 打包成 quant/<日期>、signals/<日期>

重要假設（公開資料的限制）
  ・富邦只提供每日買賣張數，沒有分點的實際成交價 → 以當日成交均價（VWAP）估算
  ・預設以「每日淨買賣」配對（net）：同一天買進又賣出的部分視為當沖，不列入波段交易；
    可改用 gross（先以當日賣出扣舊庫存，再把當日買進入庫）
  ・資料只回溯約 5 年：區間開始前就持有的部位無從得知，配不到庫存的賣出列為 unmatched，不計入勝率
  ・含交易成本：手續費 0.1425%（可設折扣）×2、證交稅 0.3%（ETF 等可另設）
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("broker_scraper")

ROOT = Path(__file__).resolve().parent
from paths import DATA_DIR  # noqa: E402  （可指向兩台電腦共用的同步資料夾）
LONG_DIR = DATA_DIR / "longhist"
QUANT_DIR = DATA_DIR / "quant"

LONG_DAYS = 1825          # 富邦保留約 5 年
REFRESH_OVERLAP = 10      # 增量更新時往回重抓的日曆天（富邦晚上才補當天，避免漏）
LOT = 1000                # 1 張 = 1000 股

# 分析範圍（使用者在前端選擇；每個範圍都在本機預先算好、各自一份 Firestore 文件）
HORIZONS = {"6m": 183, "1y": 365, "2y": 730, "3y": 1095, "5y": LONG_DAYS}   # 日曆天
DEFAULT_HORIZON = "5y"


def clip_rows(rows: list[dict], day: str, horizon: str) -> list[dict]:
    """只保留分析範圍內的明細（範圍開始前就有的部位視為未知 → 配不到庫存的賣出列為 unmatched）"""
    start = (dt.date.fromisoformat(day) - dt.timedelta(days=HORIZONS[horizon])).isoformat()
    return [r for r in rows if r["date"] >= start]


# ══════════════════════════════ 參數 ══════════════════════════════
@dataclass(frozen=True)
class Params:
    fee_rate: float = 0.001425       # 券商手續費（買、賣各一次）
    fee_discount: float = 1.0        # 手續費折扣（例：0.28 = 2.8 折）
    tax_rate: float = 0.003          # 證交稅（賣出）
    match_mode: str = "net"          # "net"：每日淨買賣配對；"gross"：買、賣分開配對
    min_trades: int = 5              # 交易數太少不給勝率標籤（避免 1 戰 1 勝 = 100%）
    win_rate_min: float = 0.70       # 高勝率門檻（> 70%）
    signal_days: int = 3             # 「近 3 日」
    big_buy_min_lots: int = 30       # 大額買超：近 3 日淨買超至少這麼多張……
    big_buy_mult: float = 3.0        # ……且 ≥ 該分點平日買進張數中位數 × 3
    hold_ratio_min: float = 0.5      # 「未出場」：近 3 日買進的庫存至少還留一半

    @property
    def buy_cost(self) -> float:      # 買進每元的成本倍數
        return 1 + self.fee_rate * self.fee_discount

    @property
    def sell_net(self) -> float:      # 賣出每元實拿倍數
        return 1 - self.fee_rate * self.fee_discount - self.tax_rate


DEFAULT = Params()


# ══════════════════════════════ 模組一：FIFO 配對 ══════════════════════════════
@dataclass
class Lot:
    date: str
    qty: float          # 張
    price: float        # 每股成交均價（估）


@dataclass
class Trade:
    open_date: str
    close_date: str
    qty: float
    buy_price: float
    sell_price: float
    pnl: float          # 已實現損益（元，含手續費與稅）
    ret: float          # 報酬率（小數，含成本）
    hold_days: int      # 持有日曆天

    @property
    def win(self) -> bool:
        return self.pnl > 0


@dataclass
class MatchResult:
    trades: list[Trade] = field(default_factory=list)
    open_lots: list[Lot] = field(default_factory=list)
    unmatched_sell: float = 0.0       # 配不到庫存的賣出張數（區間前就持有、或融券）
    intraday: float = 0.0             # 同日買賣抵銷的張數（net 模式）
    missing_price_days: int = 0


def _days_between(a: str, b: str) -> int:
    return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


def _close_fifo(lots: list[Lot], qty: float, date: str, price: float, p: Params, out: MatchResult) -> None:
    """賣出 qty 張：從最早的庫存開始扣，每次扣抵產生一筆 Trade"""
    while qty > 1e-9 and lots:
        lot = lots[0]
        q = min(qty, lot.qty)
        cost = lot.price * p.buy_cost
        proceeds = price * p.sell_net
        out.trades.append(Trade(
            open_date=lot.date, close_date=date, qty=q, buy_price=lot.price, sell_price=price,
            pnl=round(q * LOT * (proceeds - cost), 2), ret=(proceeds - cost) / cost,
            hold_days=_days_between(lot.date, date)))
        lot.qty -= q
        qty -= q
        if lot.qty <= 1e-9:
            lots.pop(0)
    if qty > 1e-9:
        out.unmatched_sell += qty


def fifo_match(rows: list[dict], prices: dict[str, float], p: Params = DEFAULT) -> MatchResult:
    """
    rows   [{date, buy, sell}]（張），會依日期排序
    prices {date: 每股價格}（成交均價，缺的日子用前一個已知價格）
    """
    out = MatchResult()
    lots: list[Lot] = []
    last_price = None
    for r in sorted(rows, key=lambda x: x["date"]):
        d = r["date"]
        price = prices.get(d)
        if price is None:
            out.missing_price_days += 1
            price = last_price
        if price is None:          # 最前面幾天沒有任何價格：無法估值，略過
            continue
        last_price = price
        buy, sell = float(r.get("buy") or 0), float(r.get("sell") or 0)
        if p.match_mode == "gross":
            if sell:
                _close_fifo(lots, sell, d, price, p, out)
            if buy:
                lots.append(Lot(d, buy, price))
        else:
            net = buy - sell
            out.intraday += min(buy, sell)
            if net > 0:
                lots.append(Lot(d, net, price))
            elif net < 0:
                _close_fifo(lots, -net, d, price, p, out)
    out.open_lots = lots
    return out


def style_of(avg_hold_days: float | None) -> str | None:
    if avg_hold_days is None:
        return None
    if avg_hold_days <= 3:
        return "隔日沖"
    if avg_hold_days <= 20:
        return "短波段"
    if avg_hold_days <= 60:
        return "波段"
    return "長線"


def performance(trades: list[Trade]) -> dict:
    """勝率、總已實現損益、平均每筆報酬率、平均持有天數（筆數平均與張數加權各一）"""
    n = len(trades)
    if not n:
        return {"trades": 0, "wins": 0, "win_rate": None, "win_rate_lots": None, "realized_pnl": 0.0,
                "avg_ret": None, "avg_ret_w": None, "avg_hold_days": None, "avg_hold_days_w": None,
                "best_ret": None, "worst_ret": None, "profit_factor": None, "style": None, "lots_closed": 0.0}
    wins = sum(t.win for t in trades)
    lots = sum(t.qty for t in trades)
    gain = sum(t.pnl for t in trades if t.pnl > 0)
    loss = -sum(t.pnl for t in trades if t.pnl < 0)
    avg_hold_w = sum(t.hold_days * t.qty for t in trades) / lots
    return {
        "trades": n,
        "wins": wins,
        "win_rate": wins / n,                                          # 以筆數計
        "win_rate_lots": sum(t.qty for t in trades if t.win) / lots,   # 以張數計
        "realized_pnl": round(sum(t.pnl for t in trades), 0),
        "avg_ret": sum(t.ret for t in trades) / n,                     # 平均每筆報酬率
        "avg_ret_w": sum(t.ret * t.qty for t in trades) / lots,        # 張數加權報酬率
        "avg_hold_days": sum(t.hold_days for t in trades) / n,
        "avg_hold_days_w": avg_hold_w,
        "best_ret": max(t.ret for t in trades),
        "worst_ret": min(t.ret for t in trades),
        "profit_factor": (gain / loss) if loss else None,
        "style": style_of(avg_hold_w),
        "lots_closed": lots,
    }


# ══════════════════════════════ 模組二：未實現與訊號 ══════════════════════════════
def unrealized(open_lots: list[Lot], last_price: float | None, p: Params = DEFAULT) -> dict:
    lots = sum(l.qty for l in open_lots)
    if lots <= 1e-9:
        return {"open_lots": 0, "open_cost": None, "open_since": None, "last_price": last_price,
                "unrealized_pnl": None, "unrealized_ret": None}
    avg = sum(l.qty * l.price for l in open_lots) / lots
    out = {"open_lots": round(lots), "open_cost": round(avg, 2), "open_since": open_lots[0].date,
           "last_price": last_price, "unrealized_pnl": None, "unrealized_ret": None}
    if last_price:
        cost, proceeds = avg * p.buy_cost, last_price * p.sell_net
        out["unrealized_pnl"] = round(lots * LOT * (proceeds - cost), 0)
        out["unrealized_ret"] = (proceeds - cost) / cost
    return out


def detect_signal(perf: dict, rows: list[dict], open_lots: list[Lot], recent_dates: list[str],
                  p: Params = DEFAULT) -> dict | None:
    """
    條件（全部成立才是訊號）
      1. 高勝率：交易數 ≥ min_trades、勝率 > 70%、平均每筆報酬率 > 0
      2. 大額買超：近 3 個交易日淨買超 ≥ max(30 張, 平日買進中位數 × 3)
      3. 未出場：近 3 日買進而尚未被 FIFO 賣出的庫存 ≥ 近 3 日淨買超的一半
    回傳訊號內容或 None
    """
    if (perf["trades"] < p.min_trades or perf["win_rate"] is None or perf["win_rate"] <= p.win_rate_min
            or (perf["avg_ret"] or 0) <= 0):
        return None
    recent = set(recent_dates[-p.signal_days:])
    by_date = {r["date"]: r for r in rows}
    net3 = sum((by_date[d].get("buy") or 0) - (by_date[d].get("sell") or 0) for d in recent if d in by_date)
    active_buys = [r["buy"] for r in rows if (r.get("buy") or 0) > 0]
    typical = statistics.median(active_buys) if active_buys else 0
    threshold = max(p.big_buy_min_lots, p.big_buy_mult * typical)
    if net3 < threshold:
        return None
    held_recent = sum(l.qty for l in open_lots if l.date in recent)
    if held_recent < p.hold_ratio_min * net3:
        return None
    return {"net3": round(net3), "threshold": round(threshold), "held_recent": round(held_recent),
            "first_buy": min((l.date for l in open_lots if l.date in recent), default=None)}


SERIES_DAYS = 180   # 網頁上「近 N 日買超」可自訂的上限（交易日）


def recent_series(rows: list[dict], open_lots: list[Lot], dates: list[str]) -> list[list]:
    """近 SERIES_DAYS 個交易日、有進出或仍有庫存的日子：[[日期, 淨買超張, 該日買進仍未出場張], …]
    網頁依使用者輸入的天數 N 加總最後 N 天，重算「近 N 日大額買超且未出場」訊號"""
    window = set(dates[-SERIES_DAYS:])
    by_date = {r["date"]: r for r in rows if r["date"] in window}
    held: dict[str, float] = {}
    for l in open_lots:
        if l.date in window:
            held[l.date] = held.get(l.date, 0) + l.qty
    out = []
    for d in sorted(set(by_date) | set(held)):
        r = by_date.get(d, {})
        net = (r.get("buy") or 0) - (r.get("sell") or 0)
        if net or held.get(d):
            out.append([d, round(net), round(held.get(d, 0))])
    return out


def evaluate(rows: list[dict], prices: dict[str, float], recent_dates: list[str], last_price: float | None,
             p: Params = DEFAULT, series_dates: list[str] | None = None) -> dict:
    """單一（股票, 分點）的完整評估"""
    m = fifo_match(rows, prices, p)
    perf = performance(m.trades)
    unr = unrealized(m.open_lots, last_price, p)
    sig = detect_signal(perf, rows, m.open_lots, recent_dates, p)
    high = (perf["trades"] >= p.min_trades and perf["win_rate"] is not None
            and perf["win_rate"] > p.win_rate_min and (perf["avg_ret"] or 0) > 0)
    extra = {}
    if high and series_dates:  # 只替高勝率分點附上近 180 日序列（網頁自訂天數用），控制檔案大小
        active_buys = [r["buy"] for r in rows if (r.get("buy") or 0) > 0]
        extra = {"series": recent_series(rows, m.open_lots, series_dates),
                 "typical": round(statistics.median(active_buys), 2) if active_buys else 0}
    return {**perf, **unr, **extra, "high": high, "signal": sig,
            "unmatched_sell": round(m.unmatched_sell), "intraday": round(m.intraday),
            "first_date": rows[0]["date"] if rows else None, "last_date": rows[-1]["date"] if rows else None,
            "recent_trades": [asdict(t) for t in m.trades[-12:]]}


# ══════════════════════════════ 長期資料（本機快取）══════════════════════════════
def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _chunks(start: dt.date, end: dt.date, days: int = 365):
    e = end
    while e >= start:
        s = max(start, e - dt.timedelta(days=days))
        yield s, e
        e = s - dt.timedelta(days=1)


# 增量更新策略：0 = 每次都向富邦補到今天（舊行為）；pipeline 設為 7 → 沒上榜的分點約每週補一次
#   未補的那幾天，若分點有上每日排行（買超／賣超前 15 名），直接用排行頁上的買賣張數（provisional）
MAX_AGE_DAYS = 0
HOT_MAX_AGE = 3      # 20 日區間榜上的分點（可能是緩步建倉）較常更新
DEADLINE: float | None = None   # pipeline --max-hours：超過就不再向富邦抓（已快取的照常計算）


HEARTBEAT = None   # pipeline 設定：長時間運算中定期回報「還在跑」（讓備援機知道不要接手）


def check_deadline():
    if HEARTBEAT is not None:
        HEARTBEAT()
    if DEADLINE is not None and time.time() > DEADLINE:
        raise TimeoutError("已達本次執行時間上限，剩下的下次再補")
_day_cache: dict[tuple[str, str], dict[str, dict]] = {}


def _listed(sid: str, day: str) -> dict[str, dict]:
    """某日每日排行頁上的分點買賣張數 {分點key: {date, buy, sell}}（讀本機 data/<日>/<代號>.json，有快取）"""
    from history import history_key

    k = (sid, day)
    if k not in _day_cache:
        out = {}
        try:  # 全分點反查（sweep.py）掃到的當日張數；個股頁有的以個股頁為準（下面覆蓋）
            import sweep
            for k, r in sweep.day_rows(sid, day).items():
                out[k] = {"date": day, "buy": int(r["buy"] or 0), "sell": int(r["sell"] or 0)}
        except Exception:  # noqa: BLE001
            pass
        try:
            rec = json.loads((DATA_DIR / day / f"{sid}.json").read_text(encoding="utf-8"))
            for side in ("buy_side", "sell_side"):
                for r in rec.get(side, []):
                    if r.get("buy") is not None:
                        out[history_key(r)] = {"date": day, "buy": int(r.get("buy") or 0), "sell": int(r.get("sell") or 0)}
        except Exception:  # noqa: BLE001
            pass
        _day_cache[k] = out
    return _day_cache[k]


def _stagger(key: str, max_age: int) -> int:
    """把每週補資料分散到不同天（避免回補完的那天每週同時爆量）"""
    import zlib
    if max_age <= 3:
        return max_age
    return 3 + zlib.crc32(key.encode()) % (max_age - 2)


def load_branch_history(fetcher, sid: str, row: dict, end: dt.date, today: dt.date | None = None,
                        max_age: int | None = None, extra_rows: list[dict] | None = None) -> list[dict]:
    """
    單一（股票, 分點）約 5 年的每日買賣張數
      第一次：分年抓（約 5 次請求）
      之後：快取距今 ≤ max_age 天就不連網，只補上每日排行頁已有的張數；超過才向富邦補一次（含重疊天數）
    """
    from history import history_key, history_url, parse_history

    today = today or end
    key = history_key(row)
    path = LONG_DIR / sid / f"{key}.json"
    cache = _read(path) or {}
    floor = today - dt.timedelta(days=LONG_DAYS)
    rows = {r["date"]: r for r in cache.get("rows", []) if r["date"] >= floor.isoformat()}
    full = bool(cache.get("end")) and cache.get("start", "9") <= (floor + dt.timedelta(days=7)).isoformat()
    max_age = MAX_AGE_DAYS if max_age is None else max_age
    if full:
        synced = dt.date.fromisoformat(cache["end"])
        if (end - synced).days <= _stagger(key, max_age):
            prov = {}
            for p in DATA_DIR.glob("????-??-??"):
                if synced.isoformat() < p.name <= end.isoformat():
                    r = _listed(sid, p.name).get(key)
                    if r:
                        prov[r["date"]] = r
            for r in extra_rows or []:
                if r["date"] > synced.isoformat():
                    prov[r["date"]] = {"date": r["date"], "buy": r["buy"], "sell": r["sell"]}
            return sorted({**rows, **prov}.values(), key=lambda r: r["date"])
        spans = [(synced - dt.timedelta(days=REFRESH_OVERLAP), end)]
    else:
        spans = list(_chunks(floor, end))
    check_deadline()
    for s_, e in spans:
        if s_ > e:
            continue
        html = fetcher.get(history_url(sid, row["broker_id"], row["b_raw"], s_, e))
        days, _ = parse_history(html, e)
        for d in days:
            rows[d["date"]] = {"date": d["date"], "buy": d["buy"], "sell": d["sell"]}
    out = sorted(rows.values(), key=lambda r: r["date"])
    _write(path, {"stock_id": sid, "broker": row.get("broker"), "start": floor.isoformat(),
                  "end": end.isoformat(), "rows": out})
    for r in extra_rows or []:  # 富邦歷史頁晚上才補當天 → 用排行頁的張數補齊（不寫進快取）
        if r["date"] not in {x["date"] for x in out[-3:]}:
            out.append({"date": r["date"], "buy": r["buy"], "sell": r["sell"]})
    return out


def fetch_finmind(sid: str, start: dt.date, end: dt.date) -> dict[str, dict]:
    import requests
    headers = {"User-Agent": "Mozilla/5.0"}
    if os.environ.get("FINMIND_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['FINMIND_TOKEN']}"
    r = requests.get("https://api.finmindtrade.com/api/v4/data", timeout=30, headers=headers,
                     params={"dataset": "TaiwanStockPrice", "data_id": sid,
                             "start_date": start.isoformat(), "end_date": end.isoformat()})
    j = r.json()
    if r.status_code != 200 or j.get("status") != 200:
        raise RuntimeError(j.get("msg") or f"HTTP {r.status_code}")
    out = {}
    for x in j.get("data") or []:
        if x.get("close", 0) > 0 and x.get("Trading_Volume"):
            out[x["date"]] = {"vwap": round(x["Trading_money"] / x["Trading_Volume"], 2), "close": x["close"],
                              "vol": round(x["Trading_Volume"] / LOT)}  # 成交張數（給動態門檻、緩步建倉用）
    return out


def load_prices(sid: str, end: dt.date, today: dt.date | None = None) -> dict[str, dict]:
    """{date: {vwap, close}} 約 5 年；第一次用 FinMind 一次取得，之後只補最近一個月"""
    from history import fetch_daily

    today = today or end
    path = LONG_DIR / sid / "_prices.json"
    cache = _read(path) or {}
    data: dict[str, dict] = cache.get("data", {})
    floor = today - dt.timedelta(days=LONG_DAYS)
    have_full = cache.get("v") == 2 and cache.get("start", "9") <= (floor + dt.timedelta(days=7)).isoformat()
    if have_full and cache.get("end", "") >= end.isoformat():  # 已由全市場日檔補到今天 → 不連網
        return data
    start = (dt.date.fromisoformat(cache["end"]) - dt.timedelta(days=REFRESH_OVERLAP)) if have_full and cache.get("end") else floor
    check_deadline()
    try:
        data.update(fetch_finmind(sid, start, end))
    except Exception as ex:  # noqa: BLE001
        log.info("%s FinMind 失敗（%s），改用證交所／櫃買逐月", sid, ex)
        vwap, close, _src = fetch_daily(sid, start, end)
        for d in set(vwap) | set(close):
            data[d] = {"vwap": vwap.get(d), "close": close.get(d)}
    data = {d: v for d, v in data.items() if d >= floor.isoformat()}
    _write(path, {"v": 2, "stock_id": sid, "start": floor.isoformat() if have_full or start == floor else cache.get("start"),
                  "end": end.isoformat(), "data": data})
    return data


def range_record(fetcher, sid: str, day: str, days: int = 30) -> dict:
    """近 days 個日曆天的「區間合計」買超／賣超前 15 名（Expectancy 與 Order Flow 共用，每天每檔只抓一次）"""
    from scraper import BASE_URL, parse_page

    path = DATA_DIR / "range" / day / f"{sid}.json"
    hit = _read(path)
    if hit is not None:
        return hit
    check_deadline()
    end = dt.date.fromisoformat(day)
    start = end - dt.timedelta(days=days)
    rec = parse_page(fetcher.get(f"{BASE_URL}?a={sid}&e={start.year}-{start.month}-{start.day}&f={end.year}-{end.month}-{end.day}"), sid)
    _write(path, rec)
    return rec


# ══════════════════════════════ 每日批次 ══════════════════════════════
def candidates(day: str, stocks: list[str], n_days: int = 3) -> dict[str, dict[str, dict]]:
    """{代號: {分點key: row}}：近 n 個有資料的日子裡，買超／賣超前 15 名出現過的分點"""
    from history import history_key

    days = sorted(p.name for p in DATA_DIR.glob("????-??-??") if p.is_dir() and p.name <= day)[-n_days:]
    out: dict[str, dict[str, dict]] = {}
    for d in reversed(days):  # 新的優先（地區等欄位以最新為準）
        for f in (DATA_DIR / d).glob("*.json"):
            if f.name.startswith("_"):
                continue
            rec = _read(f) or {}
            sid = rec.get("stock_id")
            if not sid or (stocks and sid not in stocks):
                continue
            for side in ("buy_side", "sell_side"):
                for row in rec.get(side, []):
                    if row.get("b_raw") and row.get("broker_id"):
                        out.setdefault(sid, {}).setdefault(history_key(row), {**row, "stock_name": rec.get("stock_name")})
    # 前 15 名以外：全分點反查找到、且已建好 5 年明細的分點（sweep.py）
    try:
        import sweep
        names = {s: next(iter(v.values())).get("stock_name") for s, v in out.items() if v}
        for sid, rows in sweep.engine_extras(day, stocks or list(out), n_days).items():
            for k, r in rows.items():
                out.setdefault(sid, {}).setdefault(k, {**r, "stock_name": names.get(sid)})
    except Exception as ex:  # noqa: BLE001
        log.warning("額外分點讀取失敗：%s", ex)
    return out


def compact(stats: dict) -> dict:
    """上傳用的精簡欄位（百分比 → 小數四位）"""
    r4 = lambda x: None if x is None else round(x, 4)  # noqa: E731
    r1 = lambda x: None if x is None else round(x, 1)  # noqa: E731
    return {
        "n": stats["trades"], "w": stats["wins"], "wr": r4(stats["win_rate"]), "wrl": r4(stats["win_rate_lots"]),
        "pnl": stats["realized_pnl"], "ar": r4(stats["avg_ret"]), "arw": r4(stats["avg_ret_w"]),
        "hd": r1(stats["avg_hold_days"]), "hdw": r1(stats["avg_hold_days_w"]), "pf": r4(stats["profit_factor"]),
        "style": stats["style"], "high": stats["high"],
        "ol": stats["open_lots"], "oc": stats["open_cost"], "os": stats["open_since"],
        "lp": stats["last_price"], "upnl": stats["unrealized_pnl"], "ur": r4(stats["unrealized_ret"]),
        "um": stats["unmatched_sell"], "from": stats["first_date"],
        **({"rs": stats["series"], "typ": stats["typical"]} if stats.get("series") is not None else {}),
    }


def run(day: str, stocks: list[str], fetcher=None, p: Params = DEFAULT, progress=None) -> dict:
    """每日收盤後執行：回傳並存檔 data/quant/<day>.json"""
    own = fetcher is None
    if own:
        from scraper import RequestsFetcher
        fetcher = RequestsFetcher(1.0, 2.5)
    end = dt.date.fromisoformat(day)
    cands = candidates(day, stocks, p.signal_days)
    result = {"date": day, "params": asdict(p), "horizons": {h: {"stocks": {}, "signals": []} for h in HORIZONS},
              "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    t0 = time.time()
    cal: list[str] = []   # 近 SERIES_DAYS 個交易日（取資料最完整的一檔）
    try:
        for sid, branches in cands.items():
            try:
                prices_raw = load_prices(sid, end)
            except Exception as ex:  # noqa: BLE001
                log.warning("[%s] 價格取得失敗，略過勝率計算：%s", sid, ex)
                continue
            dates = sorted(d for d in prices_raw if d <= day)
            prices = {d: (v.get("vwap") or v.get("close")) for d, v in prices_raw.items() if v.get("vwap") or v.get("close")}
            last_price = next((prices_raw[d].get("close") for d in reversed(dates) if prices_raw[d].get("close")), None)
            recent = dates[-p.signal_days:]
            if len(dates) > len(cal):
                cal = dates[-SERIES_DAYS:]
            per_stock = {h: {} for h in HORIZONS}
            for i, (key, row) in enumerate(branches.items(), 1):
                if progress:
                    progress(f"{sid} 分點勝率 {row.get('broker')}", i, len(branches))
                try:
                    rows = load_branch_history(fetcher, sid, row, end)
                except Exception as ex:  # noqa: BLE001
                    log.warning("[%s] %s 長期歷史失敗：%s", sid, row.get("broker"), ex)
                    continue
                for h in HORIZONS:  # 同一份明細，依各分析範圍分別配對與評分
                    sub = clip_rows(rows, day, h)
                    if not sub:
                        continue
                    st = evaluate(sub, prices, recent, last_price, p, series_dates=dates)
                    per_stock[h][key] = {"broker": row.get("broker"), "branch_id": row.get("branch_id"), **compact(st)}
                    if st.get("series") is not None:  # 網頁自行組訊號卡片時需要的欄位
                        per_stock[h][key].update({"bi": row.get("broker_id"), "br": row.get("b_raw"), "sn": row.get("stock_name"),
                                                  "ct": row.get("city"), "ds": row.get("district")})
                    if st["signal"]:
                        result["horizons"][h]["signals"].append({
                            "stock_id": sid, "stock_name": row.get("stock_name"), "key": key,
                            "broker": row.get("broker"), "broker_id": row.get("broker_id"), "b_raw": row.get("b_raw"),
                            "branch_id": row.get("branch_id"), "city": row.get("city"), "district": row.get("district"),
                            **{k: v for k, v in compact(st).items() if k not in ("rs", "typ")}, **st["signal"],
                        })
            for h in HORIZONS:
                result["horizons"][h]["stocks"][sid] = per_stock[h]
    finally:
        if own:
            fetcher.close()
    for h in HORIZONS:
        result["horizons"][h]["rdates"] = cal
        result["horizons"][h]["signals"].sort(key=lambda s: (-(s["wr"] or 0), -(s["ar"] or 0)))
    result.update(result["horizons"][DEFAULT_HORIZON])  # 頂層 stocks / signals = 預設範圍（相容舊用法）
    _write(QUANT_DIR / f"{day}.json", result)
    log.info("分點勝率：%d 檔、%d 個分點；各範圍訊號 %s（%.0f 秒）", len(result["stocks"]),
             sum(len(v) for v in result["stocks"].values()),
             {h: len(v["signals"]) for h, v in result["horizons"].items()}, time.time() - t0)
    return result


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="分點勝率與跟單訊號")
    ap.add_argument("--date", required=True)
    ap.add_argument("--stocks", nargs="*")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import scraper
    r = run(a.date, a.stocks or scraper.load_watchlist())
    print(json.dumps(r["signals"], ensure_ascii=False, indent=1))
    sys.exit(0)
