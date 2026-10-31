"""
全分點反查（補上個股頁前 15 名以外的分點）
=========================================

富邦「個股券商分點」頁只列買超／賣超各前 15 名，這是資料來源的上限。
另一個方向的頁面「券商分點進出排行」（zgb0）列出某個分點當天買超／賣超各前 50 檔股票。
每天把全部約 900 個分點掃一輪、反過來彙整，就能知道每檔股票還有哪些分點在進出。

  抓得到：前 15 名以外、但在該分點自己的進出裡排得上前 50 的股票
  抓不到：在該分點裡排不上前 50 的零星小量（對分析影響很小）

輸出  data/<日>/_sweep.json（檔名開頭底線，個股檔的讀取程式會自動略過）
  {
    "day": "2026-10-30",
    "branches": {"9268": ["凱基-台北", "9200", "9268", "9268"], ...},   # key → [名稱, BHID, b_raw, 分點代碼]
    "done": ["9268", ...],                                               # 已掃完的分點（可中斷續跑）
    "failed": ["...", ...],
    "stocks": {"2330": [["9268", 買進張數, 賣出張數], ...], ...},
    "names": {"2330": "台積電", ...},
    "finished": true
  }

接到三套 Ultra 引擎的方式
  1. select_extras()：每檔挑「不在前 15 名、但淨買賣夠大」的分點（預設最多 15 個）
  2. warm()：替這些分點建立約 5 年的明細快取（時間不夠就留到下次，越跑越完整）
  3. quant.candidates() / flowlab 分點宇宙：有快取的額外分點一併評分
  4. quant._listed()：掃描到的當日張數當作「暫定明細」，已建檔的分點不必每天重抓
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from pathlib import Path

from paths import DATA_DIR

log = logging.getLogger("broker_scraper")

FUBON = "https://fubon-ebrokerdj.fbs.com.tw"
LIST_URL = f"{FUBON}/z/js/zbrokerjs.djjs"
LIST_CACHE = DATA_DIR / "fubon_branches.json"
LIST_DAYS = 7

EXTRA_PER_STOCK = 15      # 每檔最多再加幾個分點（前 15 名以外）
EXTRA_MIN_LOTS = 20       # 淨買賣至少幾張才算（或達當日成交量 0.3%）
EXTRA_MIN_SHARE = 0.003
SAVE_EVERY = 40           # 每掃幾個分點存一次檔（中斷可續跑）
MAX_FAIL_STREAK = 25      # 連續失敗太多次 → 判斷被擋，停止本輪

DEADLINE: float | None = None   # pipeline 設定：超過就停，下次續跑


# ──────────────────────────── 分點清單 ────────────────────────────
def _key(b_raw: str) -> str:
    from history import history_key
    return history_key({"b_raw": b_raw})


def parse_branch_list(js: str) -> list[dict]:
    """g_BrokerList = '總公司代號,名稱!分點b,分點名稱!…;下一家…'（第一項是總公司本身）"""
    from scraper import decode_branch_code

    m = re.search(r"g_BrokerList\s*=\s*'([^']*)'", js)
    if not m:
        raise ValueError("券商清單格式改變")
    out, seen = [], set()
    for grp in m.group(1).split(";"):
        items = [x.split(",") for x in grp.split("!")]
        items = [x for x in items if len(x) >= 2 and x[0] and x[1]]
        if not items:
            continue
        bhid = items[0][0].strip()
        for b, name in items:
            b = b.strip()
            if (bhid, b) in seen:
                continue
            seen.add((bhid, b))
            out.append({"broker": re.sub(r"\s+", "", name), "broker_id": bhid, "b_raw": b,
                        "branch_id": decode_branch_code(b) or b})
    return out


def branch_list(fetcher, refresh: bool = False) -> list[dict]:
    try:
        c = json.loads(LIST_CACHE.read_text(encoding="utf-8"))
        if not refresh and time.time() - c.get("t", 0) < LIST_DAYS * 86400 and c.get("branches"):
            return c["branches"]
    except Exception:  # noqa: BLE001
        c = None
    try:
        lst = parse_branch_list(fetcher.get(LIST_URL))
        LIST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LIST_CACHE.write_text(json.dumps({"t": time.time(), "branches": lst}, ensure_ascii=False), encoding="utf-8")
        return lst
    except Exception as ex:  # noqa: BLE001
        if c and c.get("branches"):
            log.warning("券商清單更新失敗（%s），沿用舊清單", ex)
            return c["branches"]
        raise


# ──────────────────────────── 解析 zgb0 ────────────────────────────
_TD = re.compile(r"<td[^>]*>([\s\S]*?)</td>", re.I)


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).replace("&nbsp;", " ").strip()


def _num(s: str) -> int | None:
    t = _strip(s).replace(",", "")
    return int(t) if re.fullmatch(r"-?\d+", t) else None


def stock_cell(td: str) -> tuple[str, str] | None:
    g = re.search(r"GenLink2stk\(\s*'([A-Z]{0,2})([0-9A-Z]{4,6})'\s*,\s*'([^']*)'", td, re.I)
    if g:
        return g.group(2).upper(), g.group(3).strip()
    text = _strip(td)
    h = re.search(r"Link2Stk\(\s*'([0-9A-Z]{4,6})'", td, re.I) or re.search(r"[?&](?:a|stkid|id)=([0-9]{4}[0-9A-Z]{0,2})\b", td, re.I)
    if h:
        sid = h.group(1).upper()
        return sid, text.replace(sid, "", 1).strip()
    m = re.match(r"^([0-9]{4}[0-9A-Z]{0,2}?)(?=[^0-9A-Z]|$)\s*(.*)$", text) or re.match(r"^([0-9]{4,6}[A-Z]?)\s*(.*)$", text)
    return (m.group(1), m.group(2).strip()) if m else None


def parse_zgb(html: str) -> tuple[str | None, list[tuple[str, str, int, int]]]:
    """→ (資料日期, [(代號, 名稱, 買進張數, 賣出張數)])；買超、賣超兩區合併（同一檔只留一次）"""
    dm = re.search(r"資料日期[：:]\s*(\d{8})", html)
    day = f"{dm[1][:4]}-{dm[1][4:6]}-{dm[1][6:]}" if dm else None
    out: dict[str, tuple[str, str, int, int]] = {}
    for seg in re.split(r"<tr[\s>]", html, flags=re.I)[1:]:
        tds = _TD.findall(seg.split("</tr>")[0] if "</tr>" in seg else seg)
        for i in range(0, len(tds) - 3, 4):
            st = stock_cell(tds[i])
            b, s, n = _num(tds[i + 1]), _num(tds[i + 2]), _num(tds[i + 3])
            if not st or b is None or s is None or n is None or b - s != n:
                continue
            out.setdefault(st[0], (st[0], st[1], b, s))
    return day, list(out.values())


def zgb_url(bhid: str, b_raw: str, day: str) -> str:
    from urllib.parse import quote
    y, m, d = (int(x) for x in day.split("-"))
    return f"{FUBON}/z/zg/zgb/zgb0.djhtm?a={quote(bhid)}&b={quote(b_raw)}&c=E&e={y}-{m}-{d}&f={y}-{m}-{d}"


# ──────────────────────────── 掃描 ────────────────────────────
def sweep_path(day: str) -> Path:
    return DATA_DIR / day / "_sweep.json"


def load(day: str) -> dict | None:
    try:
        return json.loads(sweep_path(day).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _save(doc: dict) -> None:
    p = sweep_path(doc["day"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(p)


def run(day: str, fetcher=None, until: float | None = None, limit: int | None = None) -> dict:
    """掃全部分點（可中斷續跑）。回傳 {done, total, failed, stocks, finished}"""
    own = fetcher is None
    if own:
        from scraper import RequestsFetcher
        fetcher = RequestsFetcher(1.0, 2.5)
    until = until or DEADLINE
    t0 = time.time()
    try:
        branches = branch_list(fetcher)
        doc = load(day) or {"day": day, "branches": {}, "done": [], "failed": [], "stocks": {}, "names": {}}
        done = set(doc["done"])
        stocks: dict[str, dict[str, list]] = {sid: {r[0]: r for r in rows} for sid, rows in doc["stocks"].items()}
        todo = [b for b in branches if _key(b["b_raw"]) not in done]
        if limit:
            todo = todo[:limit]
        streak = n = 0
        stopped = None
        for b in todo:
            if until is not None and time.time() > until:
                stopped = "時間上限"
                break
            k = _key(b["b_raw"])
            try:
                pday, rows = parse_zgb(fetcher.get(zgb_url(b["broker_id"], b["b_raw"], day)))
                streak = 0
            except Exception as ex:  # noqa: BLE001
                streak += 1
                if k not in doc["failed"]:
                    doc["failed"].append(k)
                if streak >= MAX_FAIL_STREAK:
                    stopped = f"連續 {streak} 次失敗（{ex}），可能被暫時封鎖"
                    break
                continue
            doc["branches"][k] = [b["broker"], b["broker_id"], b["b_raw"], b["branch_id"]]
            if pday is None or pday == day:   # 該分點當天沒交易時頁面可能是空的或別的日期
                for sid, name, buy, sell in rows:
                    stocks.setdefault(sid, {})[k] = [k, buy, sell]
                    doc["names"].setdefault(sid, name)
            done.add(k)
            if k in doc["failed"]:
                doc["failed"].remove(k)
            n += 1
            if n % SAVE_EVERY == 0:
                doc["done"], doc["stocks"] = sorted(done), {s: list(v.values()) for s, v in stocks.items()}
                _save(doc)
                log.info("全分點反查 %s：%d / %d 個分點", day, len(done), len(branches))
        doc["done"], doc["stocks"] = sorted(done), {s: list(v.values()) for s, v in stocks.items()}
        doc["finished"] = len(done) >= len(branches)
        _save(doc)
        log.info("全分點反查 %s：本次 %d 個、累計 %d / %d 個分點、涵蓋 %d 檔股票%s（%.0f 秒）", day, n, len(done),
                 len(branches), len(stocks), f"；{stopped}，下次續跑" if stopped else "", time.time() - t0)
        return {"done": len(done), "total": len(branches), "failed": len(doc["failed"]),
                "stocks": len(stocks), "finished": doc["finished"]}
    finally:
        if own:
            fetcher.close()


# ──────────────────────────── 提供給引擎 ────────────────────────────
_mem: dict[str, dict | None] = {}


def _doc(day: str) -> dict | None:
    if day not in _mem:
        _mem[day] = load(day)
    return _mem[day]


def clear_cache():
    _mem.clear()


def day_rows(sid: str, day: str) -> dict[str, dict]:
    """某檔某日所有掃描到的分點 {key: row}（row 欄位與個股頁的分點列相同，source="sweep"）"""
    doc = _doc(day)
    if not doc:
        return {}
    out = {}
    for k, buy, sell in doc["stocks"].get(sid, []):
        info = doc["branches"].get(k)
        if not info:
            continue
        out[k] = {"broker": info[0], "broker_id": info[1], "b_raw": info[2], "branch_id": info[3],
                  "buy": buy, "sell": sell, "net": buy - sell, "source": "sweep"}
    return out


def _top_keys(sid: str, day: str) -> set[str]:
    from history import history_key
    try:
        rec = json.loads((DATA_DIR / day / f"{sid}.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    return {history_key(r) for side in ("buy_side", "sell_side") for r in rec.get(side, [])}


def _day_volume(sid: str, day: str) -> float | None:
    try:
        import quant
        c = json.loads((quant.LONG_DIR / sid / "_prices.json").read_text(encoding="utf-8"))
        return (c.get("data") or {}).get(day, {}).get("vol")
    except Exception:  # noqa: BLE001
        return None


def select_extras(day: str, stocks: list[str], n_days: int = 3, per_stock: int = EXTRA_PER_STOCK,
                  min_lots: int = EXTRA_MIN_LOTS, min_share: float = EXTRA_MIN_SHARE) -> dict[str, dict[str, dict]]:
    """{代號: {key: row}}：近 n_days 個掃描日中，不在個股頁前 15 名、但淨買賣夠大的分點（依 |淨額| 排序）"""
    days = sorted(p.parent.name for p in DATA_DIR.glob("????-??-??/_sweep.json") if p.parent.name <= day)[-n_days:]
    out: dict[str, dict[str, dict]] = {}
    for sid in stocks:
        best: dict[str, dict] = {}
        for d in days:
            top = _top_keys(sid, d)
            vol = _day_volume(sid, d)
            floor = max(min_lots, (vol or 0) * min_share)
            for k, r in day_rows(sid, d).items():
                if k in top or abs(r["net"]) < floor:
                    continue
                if k not in best or abs(r["net"]) > abs(best[k]["net"]):
                    best[k] = {**r, "_extra": True, "_day": d}
        if best:
            out[sid] = dict(sorted(best.items(), key=lambda kv: -abs(kv[1]["net"]))[:per_stock])
    return out


def has_history(sid: str, key: str) -> bool:
    import quant
    return (quant.LONG_DIR / sid / f"{key}.json").exists()


def warm(day: str, stocks: list[str], fetcher=None, until: float | None = None) -> dict:
    """替額外分點建立約 5 年明細快取（大的先建；時間到就停，下次續建）"""
    import quant

    own = fetcher is None
    if own:
        from scraper import RequestsFetcher
        fetcher = RequestsFetcher(1.0, 2.5)
    end = dt.date.fromisoformat(day)
    queue = [(sid, k, r) for sid, rows in select_extras(day, stocks).items()
             for k, r in rows.items() if not has_history(sid, k)]
    queue.sort(key=lambda x: -abs(x[2]["net"]))
    built = failed = 0
    t0 = time.time()
    try:
        for sid, k, r in queue:
            if until is not None and time.time() > until:
                break
            try:
                quant.load_branch_history(fetcher, sid, r, end)
                built += 1
            except TimeoutError:
                break
            except Exception as ex:  # noqa: BLE001
                failed += 1
                log.warning("[%s] 額外分點 %s 明細失敗：%s", sid, r.get("broker"), ex)
    finally:
        if own:
            fetcher.close()
    log.info("額外分點明細：待建 %d、本次建好 %d、失敗 %d（%.0f 秒）%s", len(queue), built, failed, time.time() - t0,
             "；其餘下次續建" if built + failed < len(queue) else "")
    return {"queued": len(queue), "built": built, "failed": failed}


def engine_extras(day: str, stocks: list[str], n_days: int = 3) -> dict[str, dict[str, dict]]:
    """給三套引擎用：已有 5 年明細快取的額外分點"""
    return {sid: {k: r for k, r in rows.items() if has_history(sid, k)}
            for sid, rows in select_extras(day, stocks, n_days).items()}
