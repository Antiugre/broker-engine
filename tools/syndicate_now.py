"""
立即計算並上傳「主力集團偵測」（測試版，管理員限定）
  只讀本機 data/longhist，不連富邦；上傳只寫 syndicate/{代號} 與 syndicate/_index
  python tools/syndicate_now.py [代號 ...]（預設＝broker_flow/config/params.yaml 的 test_stocks）
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from broker_flow import run as bf  # noqa: E402
from broker_flow.params import P  # noqa: E402
from firebase_upload import _commit, get_client  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    stocks = sys.argv[1:] or list(P.run.test_stocks)
    docs = []
    for sid in stocks:
        d = bf.run(sid)
        docs.append((f"syndicate/{sid}", d))
    docs.append(("syndicate/_index", {"stock_id": "_index", "stocks": [{"id": d["stock_id"], "name": d.get("stock_name") or "", "asof": d["asof"]} for _, d in docs]}))
    _commit(get_client(), docs, label="1/1")
    logging.getLogger("broker_scraper").info("完成：%s 已上傳", "、".join(stocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
