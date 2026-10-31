#!/usr/bin/env python3
"""
純資料腳本：下載 → 清洗 → 上傳 Firebase
====================================================
與本機網頁（app.py）完全分離，沒有伺服器、不開瀏覽器，適合交給 Windows 工作排程器。

  爬蟲 / 清洗      scraper.py、history.py、broker_location.py   （原本就與畫面無關，直接沿用）
  上傳             firebase_upload.py
  本機網頁（選用）  app.py + index.html                           （可繼續在自己電腦上用，不影響本腳本）

用法：
  python pipeline.py                     # 抓「今天」（17:00 前則為前一交易日）並上傳
  python pipeline.py --wait-ready        # 排程用：富邦資料還沒更新就每 20 分鐘再試，最多 4 次
  python pipeline.py --dry-run           # 只下載與打包，不寫入 Firestore（輸出到 out/firestore/ 檢查）
  python pipeline.py --upload-only       # 不下載，只把 data/ 內最新一天重新上傳
  python pipeline.py --date 2026-10-28   # 指定日期

結束代碼：0 成功（或非交易日、無需更新）、1 失敗
日誌：logs/pipeline-YYYYMMDD.log
"""
from __future__ import annotations

import argparse
import json
import datetime as dt
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import scraper  # noqa: E402

log = logging.getLogger("broker_scraper")
READY_HOUR = 17  # 17:00 以後視為「今天的盤後資料應該要出來了」


def setup_logging(verbose: bool = False) -> Path:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"pipeline-{dt.date.today():%Y%m%d}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    handlers = [logging.FileHandler(path, encoding="utf-8")]
    if sys.stdout:  # 用 pythonw.exe（排程、無視窗）執行時沒有 stdout
        handlers.append(logging.StreamHandler(sys.stdout))
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)
    # 只保留最近 30 天日誌
    for old in sorted(log_dir.glob("pipeline-*.log"))[:-30]:
        old.unlink(missing_ok=True)
    return path


def target_day(now: dt.datetime | None = None) -> dt.date:
    now = now or dt.datetime.now()
    d = now.date() if now.hour >= READY_HOUR else now.date() - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def site_latest_date(stock_id: str) -> str | None:
    """看富邦頁面的「最後更新日」，判斷今天的資料出來了沒"""
    f = scraper.RequestsFetcher(0.5, 1.0)
    try:
        return scraper.parse_page(f.get(scraper.build_url(stock_id, None)), stock_id).get("date")
    finally:
        f.close()


def wait_until_ready(target: dt.date, stock_id: str, tries: int, interval_min: int) -> bool:
    for i in range(1, tries + 1):
        try:
            latest = site_latest_date(stock_id)
        except Exception as ex:  # noqa: BLE001
            latest = None
            log.warning("檢查資料是否更新失敗：%s", ex)
        if latest and latest >= target.isoformat():
            log.info("資料已更新（網站最後更新日 %s）", latest)
            return True
        if i < tries:
            log.info("網站最後更新日 %s，尚未到 %s；%d 分鐘後再試（%d/%d）", latest, target, interval_min, i, tries)
            time.sleep(interval_min * 60)
    log.info("等不到 %s 的資料（可能是休市日），本次結束", target)
    return False


def latest_local_date() -> str | None:
    days = sorted(p.name for p in scraper.DATA_DIR.glob("????-??-??") if p.is_dir())
    return days[-1] if days else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="下載 → 清洗 → 上傳 Firebase")
    ap.add_argument("--date", help="YYYY-MM-DD（預設：今天；17:00 前為前一交易日）")
    ap.add_argument("--stocks", nargs="*", help="只算這些股票（預設：Ultra 會員自選＋成交值前 N 名）")
    ap.add_argument("--wait-ready", action="store_true", help="資料未更新時每 20 分鐘重試，最多 4 次")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--interval", type=int, default=20, help="重試間隔（分鐘）")
    ap.add_argument("--dry-run", action="store_true", help="不寫入 Firestore，打包結果輸出到 out/firestore/")
    ap.add_argument("--upload-only", action="store_true", help="略過下載，只上傳 data/ 中的資料")
    ap.add_argument("--no-upload", action="store_true", help="只下載清洗，不上傳")
    ap.add_argument("--force", action="store_true", help="重新下載已存在的資料，並強制重寫 Firestore")
    ap.add_argument("--no-quant", action="store_true", help="略過分點勝率與跟單訊號計算")
    ap.add_argument("--top", type=int, help="成交值前幾名（預設 local_settings.json 的 top_n，100）")
    ap.add_argument("--failover", action="store_true",
                    help="備援模式（Nunu）：主機今天已上傳就直接結束；local_settings.json 的 role=backup 也會開啟")
    ap.add_argument("--max-hours", type=float, help="最多跑幾小時（建檔時分批用；超時就停，下次接著跑）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    log_path = setup_logging(args.verbose)
    target = dt.date.fromisoformat(args.date) if args.date else target_day()
    import paths
    import quant
    quant.MAX_AGE_DAYS = 7                     # 沒上榜的分點約每週向富邦補一次；上榜的直接用排行頁張數
    if args.max_hours:
        quant.DEADLINE = time.time() + args.max_hours * 3600
        scraper.DEADLINE = quant.DEADLINE

    db = None
    if not args.dry_run:  # --no-upload（建檔）也要讀 Ultra 會員自選，只是不寫入
        try:
            import firebase_upload
            db = firebase_upload.get_client()
        except Exception as ex:  # noqa: BLE001
            log.warning("無法連線 Firestore（%s）：略過 Ultra 自選與備援檢查", ex)

    # 0) 備援（Nunu）：主機（MandM）今天已完成就不重複跑
    if (args.failover or paths.ROLE == "backup") and not args.no_upload and db is not None:
        meta = (db.collection("meta").document("latest").get().to_dict() or {})
        if meta.get("pc_done") == target.isoformat():
            log.info("主機 %s 已完成 %s 的運算與上傳，備援機不執行", meta.get("pc_host"), target)
            return 0
        log.info("主機今天尚未完成（最後 %s），備援機接手", meta.get("pc_done"))

    # 0.5) 兩台電腦共用資料夾：另一台正在跑就不重疊執行（鎖檔 6 小時內有效）
    import socket
    lock = paths.DATA_DIR / "_running.json"
    me = socket.gethostname()
    try:
        cur = json.loads(lock.read_text(encoding="utf-8"))
        if cur.get("host") != me and time.time() - cur.get("t", 0) < 6 * 3600:
            log.info("%s 正在執行（%s 開始），本機不重疊執行", cur.get("host"), cur.get("started"))
            return 0
    except Exception:  # noqa: BLE001
        pass
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"host": me, "t": time.time(), "started": dt.datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    try:
        return _run(args, target, db, paths, log_path)
    finally:
        lock.unlink(missing_ok=True)


def _run(args, target, db, paths, log_path) -> int:
    # 1) 今天要算哪些股票：Ultra 會員自選 → 成交值前 N 名
    if args.stocks:
        stocks = args.stocks
    else:
        import universe
        mdate, market = universe.fetch_market()
        n = universe.apply_to_price_cache(target.isoformat(), market, mdate)
        if n:
            log.info("全市場日檔補進 %d 檔的價格快取（不必逐檔抓）", n)
        stocks = universe.build(target.isoformat(), db, market,
                                args.top if args.top is not None else paths.TOP_N)["stocks"]
    if not stocks:
        log.error("今天沒有要計算的股票（讀不到 Ultra 自選，也拿不到全市場成交值排行）")
        return 1
    log.info("===== pipeline 開始：目標 %s，%d 檔，資料夾 %s，日誌 %s =====", target, len(stocks), paths.DATA_DIR, log_path)

    try:
        # 1) 下載與清洗
        if not args.upload_only:
            if args.wait_ready and not wait_until_ready(target, "2330", args.tries, args.interval):  # 用台積電判斷資料是否已更新
                return 0
            res = scraper.run_job(stocks, target, min_delay=1.5, max_delay=3.5, skip_existing=not args.force,
                                  progress=lambda st, m, d, t: log.info(m) if st == "stock" else None)
            if not res["ok"] and not res["skipped"]:
                log.error("全部下載失敗：%s", res["failed"])
                return 1

        # 2) 上傳（以實際存檔的日期為準；休市日時富邦會回傳前一個交易日）
        day = target.isoformat() if (scraper.DATA_DIR / target.isoformat()).exists() else latest_local_date()
        if not day:
            log.error("data/ 中沒有資料可上傳")
            return 1
        # 1.5) 分點勝率與跟單訊號（失敗不影響主資料上傳）
        if not args.no_quant:
            try:
                import quant
                quant.run(day, stocks)
            except Exception:  # noqa: BLE001
                log.exception("分點勝率計算失敗（主資料照常上傳）")
            try:
                import quant_edge
                quant_edge.run(day, stocks)
            except Exception:  # noqa: BLE001
                log.exception("Expectancy 計算失敗（主資料照常上傳）")
            try:
                import flowlab
                flowlab.run(day, stocks)
            except Exception:  # noqa: BLE001
                log.exception("Order Flow 計算失敗（主資料照常上傳）")

        if args.no_upload:
            log.info("--no-upload：略過上傳")
            return 0

        import firebase_upload
        log.info("打包 %s 並%s…", day, "試跑" if args.dry_run else "上傳 Firestore")
        r = firebase_upload.upload(day, stocks, dry_run=args.dry_run, force=args.force,
                                   out_dir=ROOT / "out" / "firestore" if args.dry_run else None)
        log.info("===== 完成：寫入 %d、略過 %d =====", len(r.get("written") or r.get("would_write", [])), len(r["skipped"]))
        return 0
    except Exception:  # noqa: BLE001
        log.exception("pipeline 失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
