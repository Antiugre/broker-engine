"""
計算並上傳「主力集團偵測」（測試版，管理員限定）：全市場維度 C → 測試股票 → 集團（含假集團對照）
  只讀 data/longhist，不連富邦；上傳只寫 syndicate/{代號或集團} 與 syndicate/_index
  python tools/syndicate_now.py [代號 ...] [--max-hours 5.3] [--no-upload]
  第一次執行要做全部歷史窗口的維度 C 與假集團（雲端約數小時）；結果存在 data/broker_flow/_cache，之後每天只算新窗口
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from broker_flow import run as bf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stocks", nargs="*")
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("broker_scraper")
    docs = bf.run_all(a.stocks or None, max_hours=a.max_hours)
    if not docs:
        log.error("沒有算出結果（資料不足）")
        return 1
    if a.no_upload:
        log.info("--no-upload：%d 份結果只存本機", len(docs))
        return 0
    from firebase_upload import _commit, get_client
    items = [(f"syndicate/{d['stock_id']}", d) for d in docs]
    items.append(("syndicate/_index", {"stock_id": "_index", "stocks": [
        {"id": d["stock_id"], "name": d.get("stock_name") or "", "asof": d["asof"], "kind": d.get("kind", "stock")} for d in docs]}))
    _commit(get_client(), items, label="1/1")
    log.info("完成：%s 已上傳", "、".join(d["stock_id"] for d in docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
