#!/usr/bin/env python3
"""
台股盤後「券商分點買賣超」爬蟲
--------------------------------------------------
資料來源：富邦 e-broker 個股「券商分點-進出明細」頁（主力進出）
  https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco.djhtm?a=<代號>&e=<起日>&f=<迄日>
  每檔股票提供「買超前 15 名」與「賣超前 15 名」分點（單位：張）。

輸出：
  data/<YYYY-MM-DD>/<代號>.json   單一個股
  data/<YYYY-MM-DD>/_all.json     當日所有個股彙整
  data/index.json                 給前端用的索引（日期 → 個股清單）

用法：
  python scraper.py                       # 抓最近一次運算範圍（Ultra 自選＋成交值前 N 名）的最近交易日
  python scraper.py 2330 2317 6269        # 指定個股
  python scraper.py --date 2026-10-28     # 指定日期
  python scraper.py --engine playwright   # requests 被擋時改用瀏覽器模式

每個分點會附上 city（縣市）、district（鄉鎮市區）、address，
並在 by_region 依縣市/鄉鎮彙總淨買賣（地址來源見 broker_location.py）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fubon-ebrokerdj.fbs.com.tw/z/zc/zco/zco.djhtm"
ROOT = Path(__file__).resolve().parent
from paths import DATA_DIR  # noqa: E402  （可指向兩台電腦共用的同步資料夾）
DEADLINE: float | None = None  # pipeline --max-hours

# 輪替的瀏覽器 User-Agent，降低被判定為機器人的機率
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
]

log = logging.getLogger("broker_scraper")


# ──────────────────────────── 抓取層 ────────────────────────────
# 同一套嘉實（MoneyDJ）系統的兩個公開鏡像：網址路徑、分點代碼、資料完全相同（2026-11-03 以 2601 比對過）
#   每個網站各自維持原本的間隔（不對任何一站加速）；每次請求挑「最快能送出＋平常回應最快」的那一站，
#   一站失敗時改用另一站。環境變數 BB_MIRRORS=fubon 可只用富邦。
FUBON_HOST = "https://fubon-ebrokerdj.fbs.com.tw"
MIRRORS = {"fubon": FUBON_HOST, "emega": "https://moneydj.emega.com.tw"}


def _mirror_hosts() -> list[str]:
    import os
    names = [x.strip() for x in (os.environ.get("BB_MIRRORS") or "fubon,emega").split(",") if x.strip()]
    return [MIRRORS[n] for n in names if n in MIRRORS] or [FUBON_HOST]


class _Host:
    def __init__(self, base: str):
        self.base = base
        self.session = requests.Session()
        retry = Retry(
            total=2,
            backoff_factor=2,  # 2s, 4s 指數退避（另一站可接手，不必在同一站等太久）
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.last = 0.0          # 上一次請求結束的時間
        self.lat = 1.0           # 平均回應秒數（指數平均）
        self.fails = 0           # 連續失敗次數（失敗越多越少用）
        self.n = 0


class RequestsFetcher:
    """requests + 重試 + 隨機延遲 + 偽裝 headers；富邦／兆豐兩個鏡像輪流使用（每站各自限速）"""

    def __init__(self, min_delay: float, max_delay: float, prefer: str | None = None):
        """prefer = 固定先用這一站（分區同時下載：每個工作各負責一站，那站連續失敗才改用另一站）"""
        self.min_delay, self.max_delay = min_delay, max_delay
        self.hosts = [_Host(h) for h in _mirror_hosts()]
        self.prefer = prefer
        self.session = self.hosts[0].session   # 相容舊程式
        self.req_n, self.req_wait, self.req_time = 0, 0.0, 0.0   # 統計：請求次數、刻意等待秒數、等回應秒數

    def _headers(self, base: str = FUBON_HOST) -> dict:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.6,en;q=0.4",
            "Referer": base + "/",
            "Connection": "keep-alive",
        }

    def _pick(self, gap: float) -> list[_Host]:
        """依「還要等多久＋平均回應時間＋失敗懲罰」排序"""
        now = time.time()
        cost = lambda h: max(0.0, h.last + gap - now) + h.lat + 5 * min(h.fails, 6)  # noqa: E731
        return sorted(self.hosts, key=cost)

    def get(self, url: str, delay=None, only: str | None = None) -> str:
        """only = 指定只用某一站（例如另一站資料還沒更新時，改向富邦確認）"""
        lo, hi = delay or (self.min_delay, self.max_delay)
        gap = random.uniform(lo, hi)
        mirrored = url.startswith(FUBON_HOST)
        order = self._pick(gap) if mirrored else [self.hosts[0]]
        if self.prefer and mirrored and not only:
            first = [h for h in order if h.base == self.prefer and h.fails < 3]
            order = first + [h for h in order if h not in first]
        if only and mirrored:
            order = [h for h in self.hosts if h.base == only] or order
        err = None
        for h in order:
            wait = h.last + gap - time.time()
            if wait > 0:
                time.sleep(wait)
                self.req_wait += wait
            u = h.base + url[len(FUBON_HOST):] if mirrored else url
            t0 = time.time()
            try:
                r = h.session.get(u, headers=self._headers(h.base if mirrored else FUBON_HOST), timeout=20)
                r.raise_for_status()
            except Exception as ex:  # noqa: BLE001 — 換另一站
                h.last, h.fails, err = time.time(), h.fails + 1, ex
                self.req_time += h.last - t0
                log.info("%s 失敗（%s），改用其他鏡像", h.base.split("//")[1], type(ex).__name__)
                continue
            h.last, h.fails, h.n = time.time(), 0, h.n + 1
            h.lat = 0.8 * h.lat + 0.2 * (h.last - t0)
            self.req_n += 1
            self.req_time += h.last - t0
            # 該站為 Big5 編碼；cp950 是 Big5 的超集，可正確解出更多字
            return r.content.decode("cp950", errors="replace")
        raise err

    def stats(self) -> str:
        return "、".join(f"{h.base.split('//')[1].split('.')[0]} {h.n} 次（平均 {h.lat:.1f} 秒）" for h in self.hosts)

    def close(self):
        if sum(h.n for h in self.hosts) >= 20:
            log.info("鏡像使用：%s", self.stats())
        for h in self.hosts:
            h.session.close()


class PlaywrightFetcher:
    """以無頭 Chromium 取頁，適用於 requests 被擋或需要執行 JS 的情況"""

    def __init__(self, min_delay: float, max_delay: float, headless: bool = True):
        from playwright.sync_api import sync_playwright  # 延遲匯入，未安裝也不影響 requests 模式

        self.min_delay, self.max_delay = min_delay, max_delay
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._ctx = self._browser.new_context(
            user_agent=random.choice(USER_AGENTS), locale="zh-TW", viewport={"width": 1366, "height": 900}
        )
        self._page = self._ctx.new_page()

    def get(self, url: str, delay=None) -> str:
        time.sleep(random.uniform(*(delay or (self.min_delay, self.max_delay))))
        self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        self._page.wait_for_selector("table", timeout=15_000)
        return self._page.content()

    def close(self):
        self._ctx.close()
        self._browser.close()
        self._pw.stop()


# ──────────────────────────── 解析 / 清洗層 ────────────────────────────
def to_int(text: str) -> int | None:
    t = (text or "").replace(",", "").replace("\xa0", "").strip()
    if not t or t in {"-", "--"}:
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def to_float(text: str) -> float | None:
    t = (text or "").replace(",", "").replace("%", "").replace("\xa0", "").strip()
    try:
        return float(t) if t else None
    except ValueError:
        return None


def decode_branch_code(code: str) -> str:
    """分點代碼有時以 UTF-16BE 十六進位編碼（如 0039004100380039 → 9A89），這裡還原成可讀代碼。"""
    if code and len(code) >= 8 and len(code) % 4 == 0 and re.fullmatch(r"[0-9a-fA-F]+", code):
        try:
            return bytes.fromhex(code).decode("utf-16-be")
        except (ValueError, UnicodeDecodeError):
            pass
    return code


def parse_broker_cell(td) -> dict:
    name = re.sub(r"\s+", "", td.get_text(strip=True))
    a = td.find("a")
    branch_id = broker_id = b_raw = None
    if a and a.get("href"):
        q = parse_qs(urlparse(a["href"]).query)
        b_raw = q.get("b", [None])[0]
        branch_id = decode_branch_code(b_raw or "") or None
        broker_id = q.get("BHID", [None])[0]
    return {"broker": name, "broker_id": broker_id, "branch_id": branch_id, "b_raw": b_raw}


def parse_page(html: str, stock_id: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    # 股票名稱、資料日期
    m = re.search(r"(\S+?)\(" + re.escape(stock_id) + r"\)\s*券商分點", text)
    stock_name = m.group(1) if m else None
    m = re.search(r"最後更新日[:：]\s*(\d{4})/(\d{1,2})/(\d{1,2})", text)
    data_date = dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat() if m else None

    buy_side, sell_side = [], []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td", recursive=False)
        if len(tds) != 10:
            continue
        # 只處理「第一格是券商分點連結」的資料列，避開表頭與版面用表格
        if not (tds[0].find("a", href=re.compile("zco0")) or tds[5].find("a", href=re.compile("zco0"))):
            continue
        vals = [td.get_text(strip=True) for td in tds]

        if tds[0].get_text(strip=True):
            b = parse_broker_cell(tds[0])
            b.update(buy=to_int(vals[1]), sell=to_int(vals[2]), net=to_int(vals[3]), pct=to_float(vals[4]))
            buy_side.append(b)
        if tds[5].get_text(strip=True):
            s = parse_broker_cell(tds[5])
            net = to_int(vals[8])
            # 頁面上賣超張數不帶負號，統一改為負數方便計算
            s.update(buy=to_int(vals[6]), sell=to_int(vals[7]), net=-abs(net) if net is not None else None, pct=to_float(vals[9]))
            sell_side.append(s)

    for i, r in enumerate(buy_side, 1):
        r["rank"] = i
    for i, r in enumerate(sell_side, 1):
        r["rank"] = i

    def grab(label, cast):
        mm = re.search(label + r"\s*([\d,\.]+)", text)
        return cast(mm.group(1)) if mm else None

    summary = {
        "total_net_buy": grab("合計買超張數", to_int),
        "total_net_sell": grab("合計賣超張數", to_int),
        "avg_buy_cost": grab("平均買超成本", to_float),
        "avg_sell_cost": grab("平均賣超成本", to_float),
    }
    return {
        "stock_id": stock_id,
        "stock_name": stock_name,
        "date": data_date,
        "unit": "張",
        "source": "fubon-ebrokerdj",
        "buy_side": buy_side,
        "sell_side": sell_side,
        "summary": summary,
    }


def validate(record: dict) -> list[str]:
    """基本資料品質檢查：回傳警告訊息清單"""
    warns = []
    if not record["buy_side"] and not record["sell_side"]:
        warns.append("沒有解析到任何分點資料（可能非交易日、代號錯誤或頁面改版）")
    for side in ("buy_side", "sell_side"):
        for r in record[side]:
            if None not in (r["buy"], r["sell"], r["net"]) and abs(r["buy"] - r["sell"]) != abs(r["net"]):
                warns.append(f"{side} {r['broker']} 買賣張數與淨額不符")
    return warns


# ──────────────────────────── 流程 ────────────────────────────
def last_trading_day(today: dt.date | None = None) -> dt.date:
    """粗略推估最近交易日：18:00 前視為前一日（資料通常傍晚更新），並跳過週末。
    國定假日無法判斷，抓到的頁面日期會以「最後更新日」為準。"""
    now = dt.datetime.now()
    d = today or (now.date() if now.hour >= 18 else now.date() - dt.timedelta(days=1))
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def build_url(stock_id: str, date: dt.date | None) -> str:
    if date is None:
        return f"{BASE_URL}?a={stock_id}"
    ds = f"{date.year}-{date.month}-{date.day}"
    return f"{BASE_URL}?a={stock_id}&e={ds}&f={ds}"


def load_watchlist() -> list[str]:
    """手動執行時的預設股票：最近一次每日運算範圍（Ultra 自選＋成交值前 N 名；已不使用 stocks.txt）"""
    files = sorted((DATA_DIR / "universe").glob("????-??-??.json"))
    if files:
        try:
            return json.loads(files[-1].read_text(encoding="utf-8")).get("stocks") or ["2330"]
        except Exception:  # noqa: BLE001
            pass
    return ["2330"]


def update_index():
    index = {}
    for day_dir in sorted(DATA_DIR.glob("????-??-??"), reverse=True):
        stocks = []
        for f in sorted(day_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                stocks.append({"stock_id": d["stock_id"], "stock_name": d.get("stock_name"), "file": f"{day_dir.name}/{f.name}"})
            except (json.JSONDecodeError, KeyError):
                continue
        if stocks:
            index[day_dir.name] = stocks
    (DATA_DIR / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def run_job(stock_ids: list[str], target: dt.date | None = None, *, engine: str = "requests",
            min_delay: float = 2.0, max_delay: float = 5.0, headful: bool = False,
            with_location: bool = True, with_history: bool = True, skip_existing: bool = False,
            progress=None) -> dict:
    """抓取一批個股。progress(stage, message, done, total) 用於回報進度。回傳 {ok: [...], failed: [...]}"""
    lock = threading.RLock()     # 兩個下載工作同時進行：寫索引檔、回報進度時排隊
    def report(stage, msg, done=0, total=0):
        if progress:
            with lock:
                progress(stage, msg, done, total)

    report("prepare", "載入券商所在地目錄…")
    locator = None
    if with_location:
        from broker_location import Locator, load_directory
        directory = load_directory(quiet=True)
        locator = Locator(directory) if directory else None

    results, failed, skipped = [], [], []

    def one(i: int, sid: str, fetcher) -> None:
        t_start = time.time()
        n0, w0, r0 = (getattr(fetcher, "req_n", 0), getattr(fetcher, "req_wait", 0.0), getattr(fetcher, "req_time", 0.0))
        if skip_existing and target:
            f = DATA_DIR / target.isoformat() / f"{sid}.json"
            if f.exists():
                try:
                    old = json.loads(f.read_text(encoding="utf-8"))
                    hr = old.get("history_range") or {}
                    from history import HISTORY_DAYS
                    need_start = (target - dt.timedelta(days=HISTORY_DAYS - 5)).isoformat()
                    if not with_history or (hr.get("start") and hr["start"] <= need_start):  # 舊資料區間不足半年就重抓
                        skipped.append(sid)
                        report("stock", f"{sid} 已有 {target} 資料，略過", i, len(stock_ids))
                        return
                except json.JSONDecodeError:
                    pass
        url = build_url(sid, target)
        report("stock", f"({i}/{len(stock_ids)}) 下載 {sid} 分點買賣超…", i - 1, len(stock_ids))
        try:
            html = fetcher.get(url)
            rec = parse_page(html, sid)
            if target and not rec["buy_side"] and not rec["sell_side"] and isinstance(fetcher, RequestsFetcher) and len(fetcher.hosts) > 1:
                rec = parse_page(fetcher.get(url, only=FUBON_HOST), sid)   # 鏡像可能較晚更新：向富邦再確認一次
            if target and not rec["buy_side"] and not rec["sell_side"]:
                # 富邦「指定日期」查詢要到晚上才會有當天資料，但首頁（近一日）傍晚就更新；
                # 改抓首頁，若最後更新日正是目標日就採用
                url2 = build_url(sid, None)
                rec2 = parse_page(fetcher.get(url2, only=FUBON_HOST) if isinstance(fetcher, RequestsFetcher) else fetcher.get(url2), sid)
                if rec2.get("date") == target.isoformat():
                    rec, url = rec2, url2
                    log.info("[%s] 指定日期查詢尚無資料，改用首頁最新資料（%s）", sid, rec2["date"])
            rec["query_date"] = target.isoformat() if target else None
            rec["fetched_at"] = dt.datetime.now().isoformat(timespec="seconds")
            rec["url"] = url
            if not rec["buy_side"] and not rec["sell_side"]:
                failed.append(sid)
                log.warning("[%s] 沒有解析到分點資料（代號錯誤或非交易日）", sid)
                return
            if locator:
                locator.enrich_record(rec)
            for w in validate(rec):
                log.warning("[%s] %s", sid, w)
            day = rec["date"] or rec["query_date"] or dt.date.today().isoformat()
            if target and rec["date"] and rec["date"] != target.isoformat():
                log.warning("[%s] 查詢 %s 但頁面日期為 %s（可能是假日），以頁面日期存檔", sid, target, rec["date"])
            if with_history:
                from history import enrich_with_history
                end = dt.date.fromisoformat(day)
                enrich_with_history(rec, fetcher, DATA_DIR, end,
                                    progress=lambda m, d, t, _i=i: report(
                                        "history", f"({_i}/{len(stock_ids)}) {m}", d, t))
            out = DATA_DIR / day / f"{sid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            results.append(rec)
            log.info("(%d/%d) %s %s ✓ 買超 %d 家 / 賣超 %d 家 → %s%s", i, len(stock_ids), sid, rec["stock_name"] or "",
                     len(rec["buy_side"]), len(rec["sell_side"]), out.relative_to(ROOT), cost(fetcher, n0, w0, r0, t_start))
            with lock:
                update_index()
        except Exception as e:  # noqa: BLE001 - 單檔失敗不中斷整批
            failed.append(sid)
            log.error("(%d/%d) %s 失敗：%s", i, len(stock_ids), sid, e)

    def cost(fetcher, n0, w0, r0, t_start) -> str:
        if not hasattr(fetcher, "req_n"):
            return ""
        return (f"（{time.time() - t_start:.0f} 秒：請求 {fetcher.req_n - n0} 次、間隔等待 {fetcher.req_wait - w0:.0f} 秒、"
                f"等回應 {fetcher.req_time - r0:.0f} 秒）")

    hosts = _mirror_hosts() if engine == "requests" else []
    stop = threading.Event()

    def worker(items, fetcher):
        try:
            while not stop.is_set():
                with lock:
                    if not items:
                        return
                    i, sid = items.pop(0)
                if DEADLINE is not None and time.time() > DEADLINE:
                    with lock:
                        left = len(items) + 1
                    log.info("已達本次執行時間上限：剩下 %d 檔下次再抓", left)
                    stop.set()
                    return
                try:
                    one(i, sid, fetcher)
                except Exception as e:  # noqa: BLE001 - 單檔失敗不中斷整批
                    failed.append(sid)
                    log.error("(%d/%d) %s 失敗：%s", i, len(stock_ids), sid, e)
        finally:
            fetcher.close()

    items = list(enumerate(stock_ids, 1))
    if len(hosts) > 1 and len(stock_ids) > 1:
        # 分區同時下載：每一站一個工作，從同一份清單輪流領股票（快的那站自然多做）；某站連續失敗就改用另一站
        log.info("同時下載：%s", "、".join(h.split("//")[1] for h in hosts))
        ths = [threading.Thread(target=worker, args=(items, RequestsFetcher(min_delay, max_delay, prefer=h)), daemon=True) for h in hosts]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    else:
        worker(items, PlaywrightFetcher(min_delay, max_delay, headless=not headful) if engine == "playwright"
               else RequestsFetcher(min_delay, max_delay))

    by_day: dict[str, list] = {}
    for r in results:
        by_day.setdefault(r["date"] or r["query_date"], []).append(r)
    for day, recs in by_day.items():
        all_path = DATA_DIR / day / "_all.json"
        existing = {}
        if all_path.exists():
            try:
                existing = {r["stock_id"]: r for r in json.loads(all_path.read_text(encoding="utf-8"))}
            except json.JSONDecodeError:
                pass
        existing.update({r["stock_id"]: r for r in recs})
        all_path.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    update_index()
    log.info("完成：成功 %d 檔，略過 %d 檔，失敗 %d 檔 %s", len(results), len(skipped), len(failed), failed or "")
    return {"ok": [r["stock_id"] for r in results], "skipped": skipped, "failed": failed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="台股券商分點買賣超爬蟲")
    ap.add_argument("stocks", nargs="*", help="股票代號（省略則用最近一次的每日運算範圍）")
    ap.add_argument("--date", help="資料日期 YYYY-MM-DD（預設：最近交易日）")
    ap.add_argument("--latest", action="store_true", help="不帶日期參數，直接抓網站上最新一日")
    ap.add_argument("--engine", choices=["requests", "playwright"], default="requests")
    ap.add_argument("--min-delay", type=float, default=2.0, help="每次請求最短間隔秒數")
    ap.add_argument("--max-delay", type=float, default=5.0, help="每次請求最長間隔秒數")
    ap.add_argument("--headful", action="store_true", help="Playwright 顯示瀏覽器視窗（除錯用）")
    ap.add_argument("--no-location", action="store_true", help="不附加分點所在地")
    ap.add_argument("--no-history", action="store_true", help="不抓分點近一月歷史與成本")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stock_ids = args.stocks or load_watchlist()
    target = None if args.latest else (dt.date.fromisoformat(args.date) if args.date else last_trading_day())
    log.info("目標日期：%s，共 %d 檔", target or "網站最新", len(stock_ids))
    res = run_job(stock_ids, target, engine=args.engine, min_delay=args.min_delay, max_delay=args.max_delay,
                  headful=args.headful, with_location=not args.no_location, with_history=not args.no_history,
                  progress=lambda st, m, d, t: log.info(m) if st != "history" else None)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
