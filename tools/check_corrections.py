"""
人工更正驗收（工作流程每一段開始前執行；失敗 → 該段停止，不照算）
  逐筆讀私人倉庫的 corrections.json，從資料層（fm_store）讀回那一天、那個分點的數字，必須完全等於更正值：
    ・FinMind 路徑：fm_store.stock_table（Hit Ratio、Expectancy、Order Flow、本土主力、中小型吃貨、主力集團 v3 都經過這裡）
    ・富邦備用路徑：fm_store.apply_rows（quant.load_branch_history 與 longhist 讀取都經過這裡）
  更正的實際數字只在私人倉庫，公開倉庫的程式不寫死任何人的交易
  python tools/check_corrections.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("broker_scraper")
    import fm_store
    items, ver = fm_store._corrections_raw()          # 缺檔或格式錯誤 → 例外（失敗）
    bad = 0
    for x in items:
        sid, d, name = str(x["stock_id"]), str(x["date"]), x.get("securities_trader", "")
        want = {k: float(x[k]) for k in ("buy_volume", "sell_volume", "buy_price", "sell_price") if x.get(k) is not None}
        # 富邦備用路徑
        rows = fm_store.apply_rows(sid, name, x.get("securities_trader_id") or "", [])
        r = next((r for r in rows if r["date"] == d), None)
        got = {} if r is None else {"buy_volume": r.get("buy", 0) * fm_store.LOT, "sell_volume": r.get("sell", 0) * fm_store.LOT,
                                    "buy_price": r.get("buy_price"), "sell_price": r.get("sell_price")}
        diff = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) is None or abs(float(got[k]) - v) > 1e-9}
        if diff:
            bad += 1
            log.error("✗ 富邦備用路徑 %s %s %s：%s", sid, d, name, diff)
        # FinMind 路徑
        if fm_store.enabled():
            df = fm_store.stock_table(sid)
            m = (df["date"].astype(str) == d) & (df["securities_trader"].astype(str).map(fm_store._norm_name) == fm_store._norm_name(name))
            bid = fm_store._corr_bid(sid, x, df)
            if bid:
                m = m | ((df["date"].astype(str) == d) & (df["securities_trader_id"].astype(str) == bid))
            g = df[m]
            if len(g) != 1:
                bad += 1
                log.error("✗ FinMind %s %s %s：找到 %d 筆", sid, d, name, len(g))
            else:
                row = g.iloc[0]
                diff = {k: (v, float(row[k])) for k, v in want.items() if abs(float(row[k]) - v) > 1e-9}
                if diff:
                    bad += 1
                    log.error("✗ FinMind %s %s %s：%s", sid, d, name, diff)
        if not bad:
            log.info("✓ %s %s %s 與更正一致", sid, d, name)
    if bad:
        log.error("人工更正驗收失敗：%d 筆不一致（版本 %s）", bad, ver)
        return 1
    log.info("人工更正驗收通過：%d 筆（版本 %s）", len(items), ver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
