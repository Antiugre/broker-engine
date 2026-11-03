"""
一檔股票跑完全部層 → data/broker_flow/<代號>/*.parquet ＋ 網頁用摘要 data/broker_flow/<代號>/summary.json

  python -m broker_flow.run 2601 [--quick]
  即時監看要顯示最近（hold-out 期間）的窗口 → 明確傳 allow_holdout=True（見 SPEC_GAPS.md G-HOLDOUT）；
  驗證（7.4、9.3）只用 hold-out 之前的窗口。
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from . import accounting, store
from .data_layer import build_panel
from .detect import Context, groups_for, naive_group, pairs_table, run_window, window_ends
from .params import P
from .validation import calibration, synthetic

log = logging.getLogger("broker_scraper")
SHOW_WINDOWS = 12      # 網頁顯示最近幾個窗口的明細（其餘只給計數）

SPEC_GAPS = [
    ("資料粒度", "本專案只有「日期 × 分點」買賣張數，沒有成交價位 → 3.3 守恆檢查、維度 B（冷門價位共現）、5.3 Wasserstein 無法計算"),
    ("跨股維度", "維度 C（CP-APR、Fraudar）以「分點 × 股票 × 週」為對象，單檔測試無法計算；依 8.2 第 4 條移除，p_meta = p_A"),
    ("分點範圍", "longhist 只含曾進前 15 名或全分點反查補進的分點，不是全部分點"),
    ("開高低價", "沒有開盤、最高、最低價 → 6.1 少了「振幅」因子；9.4 事件定義（下影線、紅 K）無法判斷"),
    ("分點成交均價", "沒有分點的買進／賣出 VWAP，會計層以個股當日 VWAP 代替"),
    ("公司行動", "沒有除權息、減資資料表，4.2 的成本調整未做"),
    ("集保、判決書", "9.1 集保級距、9.2 判決書驗證所需資料尚未取得"),
    ("hold-out", "網頁的即時監看需要顯示最近 12 個月（hold-out）窗口，執行時明確傳 allow_holdout=True；驗證統計只用 hold-out 之前"),
]


def _r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def _name(sid: str) -> str:
    from paths import DATA_DIR
    for d in sorted(DATA_DIR.glob("????-??-??"), reverse=True)[:10]:
        f = d / f"{sid}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8")).get("stock_name") or ""
            except Exception:  # noqa: BLE001
                return ""
    return ""


def run(sid: str, quick: bool = False, seed: int = 0) -> dict:
    t0 = time.time()
    pn = build_panel(sid)
    cx = Context(pn)
    try:
        from smallcap import _branch_ids
        ids = _branch_ids()
    except Exception:  # noqa: BLE001
        ids = {}
    store.write(accounting.to_table(pn, cx.L), sid, "ledger")
    ends = window_ends(pn)
    wins = [run_window(cx, e, allow_holdout=True) for e in ends]
    pt = pairs_table(pn, wins)
    store.write(pt, sid, "pairs")
    q = float(P.fusion.fdr_q)
    windows = []
    for k, w in enumerate(wins):
        e = w["end"]
        sig = pt[(pt["window_id"] == k) & (pt["q_meta"] <= q)] if len(pt) else pt
        groups = groups_for(cx, w, sig) if len(sig) else []
        naive = naive_group(cx, w)
        item = {"id": k, "start": str(pn.dates[w["start"]]), "end": str(pn.dates[e]),
                "holdout": bool(e >= pn.holdout_start), "candidates": len(w["cand"]),
                "pairs": int((pt["window_id"] == k).sum()) if len(pt) else 0, "significant": int(len(sig)),
                "groups": len(groups), "close": _r(pn.close[e], 2)}
        if k >= len(wins) - SHOW_WINDOWS:
            def mem(i):
                return {"key": pn.branches[i], "name": pn.names[i], **ids.get(pn.branches[i], {}), "type": pn.btype[i], "cls": w["cls"][i],
                        "inv": round(float(cx.L["inventory"][i, e]) / 1000), "cost": _r(cx.L["avg_cost"][i, e], 2),
                        "net20": round(float((pn.B[i, e - 19:e + 1] - pn.S[i, e - 19:e + 1]).sum()) / 1000),
                        "censored": bool(cx.L["left_censored"][i, e])}
            item["group_list"] = [{"members": [mem(i) for i in g["members"]], "C_G": _r(g["C_G"], 2),
                                   "inv": round(g["inventory_G"] / 1000), "net20": round(g["net_20"] / 1000),
                                   "net_window": round(g["net_window"] / 1000), "foreign_share": _r(g["foreign_share"], 2)}
                                  for g in groups]
            item["naive"] = {"members": [mem(i) for i in naive["members"]], "C_G": _r(naive["C_G"], 2),
                             "inv": round(naive["inventory_G"] / 1000), "net20": round(naive["net_20"] / 1000)}
            top = sig.nsmallest(15, "p_meta") if len(sig) else sig
            item["top_pairs"] = [{"a": pn.names[int(r.ia)], "b": pn.names[int(r.ib)], "obs": int(r.obs),
                                  "p_act": float(f"{r.p_act:.3g}"), "p_mag": None if not np.isfinite(r.p_mag) else float(f"{r.p_mag:.3g}"),
                                  "p_A": float(f"{r.p_A:.3g}"), "q": float(f"{r.q_meta:.3g}")} for r in top.itertuples()]
            from collections import Counter
            item["cls_count"] = dict(Counter(w["cls"]))
        windows.append(item)
    log.info("主力集團 %s：%d 個窗口，%d 組配對（%.0f 秒）", sid, len(wins), len(pt), time.time() - t0)
    cal = calibration(cx, ends, seed=seed, max_windows=3 if quick else 8)
    syn = synthetic(cx, ends, seed=seed, reps=3 if quick else None,
                    Ks=[2, 3] if quick else None, rhos=[1.0, 0.5] if quick else None)
    doc = {"stock_id": sid, "stock_name": _name(sid), "asof": str(pn.dates[-1]), "params_version": P.version, "quick": quick,
           "data": {"start": str(pn.dates[0]), "end": str(pn.dates[-1]), "days": len(pn.dates), "branches": len(pn.branches),
                    "holdout_start": str(pn.dates[pn.holdout_start]),
                    "analysis_start": str(pn.dates[int(P.accounting.warmup_days)])},
           "windows": windows, "calibration": cal, "synthetic": syn,
           "spec_gaps": [{"item": a, "note": b} for a, b in SPEC_GAPS],
           "seconds": round(time.time() - t0)}
    p = store.OUT / sid / "summary.json"
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    log.info("主力集團 %s 完成（含驗證）：%.0f 秒", sid, time.time() - t0)
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stocks", nargs="+")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for s in a.stocks:
        run(s, quick=a.quick)
