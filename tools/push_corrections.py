"""
人工更正 → Firestore syndicate/_corrections（網站即時查詢用；管理員限定集合）
  python tools/push_corrections.py      （需要 BB_CORRECTIONS 與 FIREBASE_CREDENTIALS）
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    import fm_store
    from firebase_upload import _commit, get_client
    items, ver = fm_store._corrections_raw()
    _commit(get_client(), [("syndicate/_corrections", {"version": ver, "items": items})], label="corrections")
    logging.getLogger("broker_scraper").info("人工更正已同步：%d 筆、版本 %s", len(items), ver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
