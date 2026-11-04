"""
主力集團：把全市場維度 C 的歷史窗口分給多台雲端機器同時算（每台算一部分窗口），算好的窗口結果存檔給每晚的計算直接使用
  python tools/syndicate_shard.py --shard 0 --of 16 [--max-hours 5.3]

  ・方法完全不變：每個窗口的計算與單機版相同（window_c），只是分工；窗口結果檔 dim_c_<結束日>.pkl 與單機版同格式
  ・秩選擇（dim_c_rank_roles.json）是決定性的（固定亂數種子），每台各自算出相同結果
  ・新的窗口先算；時間到就停（沒算完的留給每晚的計算接著補）
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("broker_scraper")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--max-hours", type=float, default=5.3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from broker_flow.dim_c import window_c
    from broker_flow.run import load_market
    t0 = time.time()
    mk = load_market()
    if mk is None:
        log.warning("全市場資料不足，不計算")
        return 0
    ends = [e for e in sorted(mk.ends, reverse=True) if mk.fit_for(e) is not None]
    mine = ends[a.shard::a.of]
    log.info("分工 %d/%d：%d 個窗口 %s", a.shard, a.of, len(mine), mine)
    for e in mine:
        if time.time() - t0 > a.max_hours * 3600:
            log.warning("時間到，%s 之後的窗口留給每晚的計算", e)
            break
        s = time.time()
        window_c(mk, e, [])
        log.info("窗口 %s 完成（%.0f 秒）", e, time.time() - s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
