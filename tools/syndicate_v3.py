"""
主力集團偵測 v3（開發集：2601、5907、2601G）→ 上傳 syndicate/{代號}@v3、syndicate/_index@v3（管理員限定，與 v2 並存）
  python tools/syndicate_v3.py [代號 ...] [--no-upload] [--no-group]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stocks", nargs="*")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--no-group", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("broker_scraper")
    from broker_flow_v3 import run, split

    def upload(docs):
        if a.no_upload:
            return
        for d in docs:
            if d.get("kind") == "stock":
                split.assert_writable(d["stock_id"])
        from firebase_upload import _commit, get_client
        items = [(f"syndicate/{docs[-1]['stock_id']}@v3", docs[-1])]
        items.append(("syndicate/_index@v3", {"stock_id": "_index", "version": 3, "stocks": [
            {"id": d["stock_id"], "name": d.get("stock_name") or "", "asof": d["asof"], "kind": d.get("kind", "stock")} for d in docs]}))
        _commit(get_client(), items, label="v3")
        log.info("v3 已上傳：%s", docs[-1]["stock_id"])
    docs = run.run_all(a.stocks or None, not a.no_group, on_doc=upload)
    if not docs:
        log.error("v3 沒有算出結果")
        return 1
    log.info("v3 完成：%s", "、".join(d["stock_id"] for d in docs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
