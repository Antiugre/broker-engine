import argparse
import json
import logging

from . import run

ap = argparse.ArgumentParser(description="Order Flow：分點行為標籤、群聚、成本防守區、異常背離")
ap.add_argument("--date", required=True)
ap.add_argument("--stocks", nargs="*")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
import scraper  # noqa: E402

print(json.dumps(run(a.date, a.stocks or scraper.load_watchlist()), ensure_ascii=False, indent=1))
