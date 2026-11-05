"""
單一分點擇時偵測 v1.1.2 → 上傳 syndicate/{代號}@timing、syndicate/_index@timing（管理員限定）
  python tools/timing_v1.py [代號 ...] [--no-upload] [--processes N] [--reps B]
  預設股票：有 longhist/_prices.json 的股票（同 broker_flow_v3/run.py）；保留池 H 不計算、不上傳
  SPEC-GAP-timing：Firestore 路徑暫放 syndicate/ 底下（沿用既有「只有管理員可讀」的 syndicate 規則），規則更新後再搬
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
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--reps", type=int, default=None, help="EI bootstrap 次數（預設 params.yaml 的 999）")
    ap.add_argument("--validate", action="store_true", help="算完後跑 §9 真實資料驗證（安慰劑、合成注入、案例），結果寫 syndicate/_validation@timing")
    ap.add_argument("--val-hosts", type=int, default=40)
    ap.add_argument("--val-reps", type=int, default=5)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("broker_scraper")
    from broker_flow_v3 import split
    from broker_timing import run

    def upload(docs):
        if a.no_upload:
            return
        for d in docs:
            split.assert_writable(d["stock_id"])                              # 保留池 H 永遠不寫
        import fm_store
        from firebase_upload import _commit, get_client
        d = docs[-1]
        if not fm_store.upload_ok(d["stock_id"]):                             # 人工更正安全閘
            log.warning("擇時 %s 這次沒有套用到人工更正，不上傳", d["stock_id"])
            return
        if st := fm_store.stamp(d["stock_id"]):
            d["corrections"] = st
        import json
        items = [(f"syndicate/{d['stock_id']}@timing", json.loads(run.dumps(d)))]          # numpy 型別 → 純 Python
        items.append(("syndicate/_index@timing", {"stock_id": "_index", "kind": "timing", "version": d["version"], "stocks": [
            {"id": x["stock_id"], "name": x.get("stock_name") or "", "asof": x["asof"], "split": x.get("split"),
             "types": (x.get("summary") or {}).get("types")} for x in docs]}))
        # Ultra 測試版（2026-11-05 擁有者決定；Ultra 帳號都是擁有者本人的測試帳號）：同一份結果另寫到 edge/（Ultra 可讀）。
        #   驗證結果 _validation 仍只寫 syndicate/（管理員限定）
        items += [(k.replace("syndicate/", "edge/", 1), v) for k, v in items]
        _commit(get_client(), items, label="timing")
        log.info("擇時 已上傳：%s", d["stock_id"])

    docs = run.run_all(a.stocks or None, processes=a.processes, on_doc=upload, B=a.reps)
    if not docs:
        log.error("擇時 沒有算出結果")
        return 1
    log.info("擇時 完成：%d 檔", len(docs))
    if a.validate:
        from broker_timing import validate_run
        v = validate_run.run_validation(run.LAST_RESULTS, processes=a.processes, hosts=a.val_hosts, reps=a.val_reps)
        import json
        (run.out_dir() / "validation.json").write_text(run.dumps(v), encoding="utf-8")
        if not a.no_upload:
            from firebase_upload import _commit, get_client
            _commit(get_client(), [("syndicate/_validation@timing", {"stock_id": "_validation", **json.loads(run.dumps(v))})],
                    label="timing-validation")
            log.info("擇時驗證 已上傳（管理員限定）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
