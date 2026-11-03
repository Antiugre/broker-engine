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
SWEEP_SLOT = 45 * 60   # --max-hours 時保留給全分點反查與額外分點建檔的秒數
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
        return scraper.parse_page(f.get(scraper.build_url(stock_id, None), only=scraper.FUBON_HOST), stock_id).get("date")
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


BEAT_FRESH = 45 * 60   # 「正在執行」回報多久內有效（秒）


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
    ap.add_argument("--no-sweep", action="store_true", help="略過全分點反查（前 15 名以外的分點）")
    ap.add_argument("--max-hours", type=float, help="最多跑幾小時（建檔時分批用；超時就停，下次接著跑）")
    ap.add_argument("--only-cached", action="store_true",
                    help="只算已建好 5 年資料的股票（雲端備援用：不在雲端做耗時的初次建檔）")
    ap.add_argument("--stage", choices=STAGES, default="all",
                    help="雲端分段接力（每段 < 6 小時）：download＝下載＋全分點反查＋Hit Ratio；edge＝Expectancy；"
                         "flow＝Order Flow＋本土主力＋中小型吃貨＋整批上傳；all＝全部（本機）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    if args.stage in ("edge", "flow"):
        st = read_stage()
        if st and not args.date:
            args.date = st["day"]          # 接力：沿用第一段的日期與股票清單（跨過午夜也不會變）
        if st and not args.stocks:
            args.stocks = st["stocks"]

    log_path = setup_logging(args.verbose)
    target = dt.date.fromisoformat(args.date) if args.date else target_day()
    import paths
    import quant
    quant.MAX_AGE_DAYS = 7                     # 沒上榜的分點約每週向富邦補一次；上榜的直接用排行頁張數
    if args.max_hours:
        quant.DEADLINE = time.time() + args.max_hours * 3600
        # 前 15 名下載先停，留一段時間給「全分點反查」（約 30～40 分鐘）與額外分點建檔
        scraper.DEADLINE = quant.DEADLINE - (SWEEP_SLOT if args.max_hours >= 2 and not args.no_sweep else 0)

    db = None
    if not args.dry_run:  # --no-upload（建檔）也要讀 Ultra 會員自選，只是不寫入
        try:
            import firebase_upload
            db = firebase_upload.get_client()
        except Exception as ex:  # noqa: BLE001
            log.warning("無法連線 Firestore（%s）：略過 Ultra 自選與備援檢查", ex)

    import socket
    me = socket.gethostname()
    # 0) 備援（雲端／另一台）：主機（MandM）今天已完成、或正在跑（45 分鐘內有回報），就不重複跑
    if (args.failover or paths.ROLE == "backup") and not args.no_upload and db is not None:
        meta = (db.collection("meta").document("latest").get().to_dict() or {})
        if meta.get("pc_done") == target.isoformat():
            log.info("主機 %s 已完成 %s 的運算與上傳，備援機不執行", meta.get("pc_host"), target)
            return 0
        run = (db.collection("meta").document("running").get().to_dict() or {})
        if run.get("host") != me and time.time() - float(run.get("t") or 0) < BEAT_FRESH:
            log.info("主機 %s 正在執行（%s 開始，最後回報 %s），備援機不執行", run.get("host"), run.get("started"), run.get("at"))
            return 0
        log.info("主機今天尚未完成（最後 %s），備援機接手", meta.get("pc_done"))

    # 0.5) 不重疊執行：任何一台（含本機的另一個排程）45 分鐘內有回報就視為正在跑
    lock = paths.DATA_DIR / "_running.json"
    try:
        cur = json.loads(lock.read_text(encoding="utf-8"))
        if time.time() - cur.get("t", 0) < BEAT_FRESH:
            log.info("%s 正在執行（%s 開始），不重疊執行", cur.get("host"), cur.get("started"))
            return 0
    except Exception:  # noqa: BLE001
        pass
    started = dt.datetime.now().isoformat(timespec="seconds")
    last = [0.0]

    def beat(stage: str | None = None, force: bool = False) -> None:
        """「還在跑」回報：本機鎖檔＋Firestore meta/running（最多每分鐘一次）"""
        now = time.time()
        if not force and now - last[0] < 60:
            return
        last[0] = now
        info = {"host": me, "t": now, "started": started, "day": target.isoformat(),
                "at": dt.datetime.now().isoformat(timespec="seconds"), **({"stage": stage} if stage else {})}
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        if db is not None and not args.no_upload and not args.dry_run:
            try:
                db.collection("meta").document("running").set(info)
            except Exception:  # noqa: BLE001
                pass

    import quant
    quant.HEARTBEAT = beat
    beat("開始" if args.stage == "all" else f"開始（{args.stage}）", force=True)
    try:
        return _run(args, target, db, paths, log_path, beat)
    finally:
        quant.HEARTBEAT = None
        lock.unlink(missing_ok=True)
        # 分段接力的前兩段：保留 meta/running（下一段幾分鐘內接手），避免備援機在空檔誤以為沒人在跑
        if db is not None and not args.no_upload and not args.dry_run and args.stage in ("all", "flow"):
            try:
                cur = db.collection("meta").document("running").get().to_dict() or {}
                if cur.get("host") == me and cur.get("started") == started:
                    db.collection("meta").document("running").delete()
            except Exception:  # noqa: BLE001
                pass


STAGES = ("all", "download", "edge", "flow")


def stage_file():
    import paths
    return paths.DATA_DIR / "_stage.json"


def read_stage() -> dict | None:
    try:
        return json.loads(stage_file().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _run(args, target, db, paths, log_path, beat=lambda *a, **k: None) -> int:
    stage = getattr(args, "stage", "all")
    do = {  # 哪一段做哪些事
        "download": stage in ("all", "download"), "hit": stage in ("all", "download"),
        "edge": stage in ("all", "edge"), "flow": stage in ("all", "flow"), "final": stage in ("all", "flow"),
    }
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
        if args.only_cached:
            import quant
            have = {p.parent.name for p in quant.LONG_DIR.glob("*/_prices.json")}
            log.info("只算已建檔的股票：%d → %d 檔", len(stocks), len([s for s in stocks if s in have]))
            stocks = [s for s in stocks if s in have]
        try:   # 主力集團偵測（測試版）的測試股票也要每天下載更新
            from broker_flow.params import P as _bf
            stocks += [s for s in _bf.run.test_stocks if s not in stocks]
        except Exception:  # noqa: BLE001
            log.exception("讀取主力集團測試股票失敗（略過）")
    if not stocks:
        log.error("今天沒有要計算的股票（讀不到 Ultra 自選，也拿不到全市場成交值排行）")
        return 1
    log.info("===== pipeline 開始%s：目標 %s，%d 檔，資料夾 %s，日誌 %s =====", "" if stage == "all" else f"（{stage}）",
             target, len(stocks), paths.DATA_DIR, log_path)

    try:
        # 1) 下載與清洗
        if not args.upload_only and do["download"]:
            if args.wait_ready and not wait_until_ready(target, "2330", args.tries, args.interval):  # 用台積電判斷資料是否已更新
                return 0
            res = scraper.run_job(stocks, target, min_delay=1.5, max_delay=3.5, skip_existing=not args.force,
                                  progress=lambda st, m, d, t: (log.info(m), beat(m)) if st == "stock" else None)
            if not res["ok"] and not res["skipped"]:
                log.error("全部下載失敗：%s", res["failed"])
                return 1

        # 2) 上傳（以實際存檔的日期為準；休市日時富邦會回傳前一個交易日）
        day = target.isoformat() if (scraper.DATA_DIR / target.isoformat()).exists() else latest_local_date()
        if not day:
            log.error("data/ 中沒有資料可上傳")
            return 1
        if stage == "download":   # 給下一段接力：日期與股票清單
            stage_file().write_text(json.dumps({"day": day, "target": target.isoformat(), "stocks": stocks}, ensure_ascii=False), encoding="utf-8")
        elif stage in ("edge", "flow") and not (scraper.DATA_DIR / day).exists():
            log.error("找不到 %s 的下載資料，這一段不執行", day)
            return 1
        # 1.2) 全分點反查：補上前 15 名以外的分點，並替其中淨買賣較大的建立 5 年明細（時間不夠就下次續跑）
        if not args.upload_only and not args.no_sweep and do["download"]:
            try:
                import quant
                import sweep
                end = quant.DEADLINE
                beat("全分點反查", force=True)
                sweep.run(day, until=end - 10 * 60 if end else None)
                sweep.warm(day, stocks, until=end)
            except Exception:  # noqa: BLE001
                log.exception("全分點反查失敗（不影響其他步驟）")
        # 1.5) 分點勝率與跟單訊號（失敗不影響主資料上傳）
        #      每算完一套就先上傳：Hit Ratio 只要幾秒，不必等 Expectancy／Order Flow（要上富邦補資料，可能很久）
        def stage_upload(name: str) -> None:
            if args.no_upload or args.dry_run:
                return
            try:
                import firebase_upload
                r = firebase_upload.upload(day, stocks)
                log.info("%s 算完先上傳：寫入 %d 份", name, len(r["written"]))
            except Exception:  # noqa: BLE001
                log.exception("%s 先行上傳失敗（最後會再整批上傳一次）", name)

        if not args.no_quant:
            if do["hit"]:
                try:
                    import quant
                    beat("Hit Ratio", force=True)
                    quant.run(day, stocks)
                    stage_upload("Hit Ratio")
                except Exception:  # noqa: BLE001
                    log.exception("分點勝率計算失敗（主資料照常上傳）")
            if do["edge"]:
                try:
                    import quant_edge
                    beat("Expectancy", force=True)
                    quant_edge.run(day, stocks)
                    stage_upload("Expectancy")
                except Exception:  # noqa: BLE001
                    log.exception("Expectancy 計算失敗（主資料照常上傳）")
            if not do["flow"]:
                log.info("===== 這一段（%s）完成，交給下一段 =====", stage)
                return 0
            try:
                import flowlab
                beat("Order Flow", force=True)
                flowlab.run(day, stocks)
            except Exception:  # noqa: BLE001
                log.exception("Order Flow 計算失敗（主資料照常上傳）")
            try:
                import mainforce
                beat("本土主力", force=True)
                mainforce.run(day, stocks)
            except Exception:  # noqa: BLE001
                log.exception("本土主力計算失敗（主資料照常上傳）")
            try:
                import smallcap
                beat("中小型吃貨", force=True)
                smallcap.run(day, stocks)
            except Exception:  # noqa: BLE001
                log.exception("中小型吃貨計算失敗（主資料照常上傳）")
            try:
                import quant
                from broker_flow import run as bf_run
                from broker_flow.params import P as _bf
                for sid in _bf.run.test_stocks:
                    if (quant.LONG_DIR / sid / "_prices.json").exists():
                        beat(f"主力集團 {sid}", force=True)
                        bf_run.run(sid)
            except Exception:  # noqa: BLE001
                log.exception("主力集團偵測（測試版）失敗（主資料照常上傳）")

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
