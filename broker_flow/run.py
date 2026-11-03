"""
跑完全部層 → data/broker_flow/<代號>/*.parquet ＋ 網頁用摘要 data/broker_flow/<代號>/summary.json

  python -m broker_flow.run            測試股票＋集團（params.yaml 的 run.test_stocks、groups）
  python -m broker_flow.run 2601 --quick
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
from .data_layer import LONG_DIR, build_panel
from .detect import Context, attach_c, groups_for, naive_group, pairs_table, run_window, window_ends
from .params import P
from .validation import calibration, synthetic

log = logging.getLogger("broker_scraper")
SHOW_WINDOWS = 12      # 網頁顯示最近幾個窗口的明細（其餘只給計數）
MIN_BRANCHES = 20      # 分點少於這個數不產出（SPEC-GAP：規格未定義，保守起見）
MIN_MARKET = 30        # 全市場少於這麼多檔就不算維度 C

SPEC_GAPS = [
    ("資料粒度", "本專案只有「日期 × 分點」買賣張數，沒有成交價位 → 3.3 守恆檢查、維度 B（冷門價位共現）、5.3 Wasserstein 無法計算"),
    ("全市場", "維度 C 用 longhist 有建檔的全部股票當「全市場」（不是證交所全部上市櫃）；過去窗口的結果算過就存檔，以當時的市場資料為準"),
    ("分點範圍", "longhist 只含曾進前 15 名或全分點反查補進的分點，不是全部分點"),
    ("開高低價", "沒有開盤、最高、最低價 → 6.1 少了「振幅」因子；9.4 事件定義（下影線、紅 K）無法判斷"),
    ("分點成交均價", "沒有分點的買進／賣出 VWAP，會計層以個股當日 VWAP 代替"),
    ("公司行動", "沒有除權息、減資資料表，4.2 的成本調整未做"),
    ("集保、判決書", "9.1 集保級距、9.2 判決書驗證所需資料尚未取得"),
    ("排除外資", "exclude_foreign 預設關閉；開啟時外資不當維度 A 候選，維度 C 只把外資配對拿掉（張量與圖照常含外資）"),
    ("維度 B2", "壓價轉移的事件日第 4 個條件（價位重疊）需要分點價位明細或分點均價，本資料源沒有 → B2 不計算；集團承接比例（5.6）只用條件 1、2，照常報告"),
    ("集團損益", "沒有分點的買進／賣出均價，以個股當日 VWAP、用各成員自己的買（賣）量加權代替（虧損腿的 1% 容差用這個比）"),
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


def _ids():
    try:
        from smallcap import _branch_ids
        return _branch_ids()
    except Exception:  # noqa: BLE001
        return {}


def load_market():
    """全市場（維度 C、假集團用）；建檔股票太少時回傳 None"""
    from . import market
    from .dim_c import Market
    cal = market.market_calendar()
    ends = market.window_end_dates(cal)
    st = market.load_market(ends)
    if len(st) < MIN_MARKET:
        log.warning("全市場只有 %d 檔，維度 C 不計算", len(st))
        return None
    return Market(st, cal, ends)


def run(sid: str, quick: bool = False, seed: int = 0, mk=None, ctx: dict | None = None) -> dict:
    t0 = time.time()
    pn = build_panel(sid)
    cx = (ctx or {}).get(sid) or Context(pn)
    if ctx is not None:
        ctx[sid] = cx
    ids = _ids()
    ends = window_ends(pn, mk.ends if mk else None)
    if not ends or len(pn.branches) < MIN_BRANCHES:   # 資料還沒建完整（例如雲端剛開始追蹤）→ 不產出，免得蓋掉完整的結果
        log.warning("主力集團 %s：資料不足（%d 個交易日、%d 個分點），略過", sid, len(pn.dates), len(pn.branches))
        return {}
    store.write(accounting.to_table(pn, cx.L), sid, "ledger")
    from .dim_c import window_c, window_c_cached
    from .group import deadline_passed
    f3 = lambda x: None if x is None or not np.isfinite(x) else float(f"{x:.3g}")  # noqa: E731
    variants = {}
    for var, xf in (("all", False), ("nofx", True)):      # exclude_foreign = false（預設）／true（消融比較）
        wins = [run_window(cx, e, allow_holdout=True, exclude_foreign=xf) for e in ends]
        if mk is not None:
            for w in reversed(wins):                       # 新的窗口先算；時間到就停（已算的都有存檔，下次接著算）
                end = str(pn.dates[w["end"]])
                if w["res"] is None or mk.fit_for(end) is None or sid not in mk.six:
                    continue
                if deadline_passed():                      # 時間到：只用已存檔的窗口
                    cres = window_c_cached(mk, end)
                    if cres is None:
                        continue
                else:
                    cres = window_c(mk, end, [], seed=seed)
                attach_c(w, pn, cres)
        pt = pairs_table(pn, wins)
        store.write(pt, sid, "pairs" if not xf else "pairs_exclude_foreign")
        variants[var] = (wins, pt)
    wins = variants["all"][0]
    q = float(P.fusion.fdr_q)
    windows = []
    for k, w0 in enumerate(wins):
        e = w0["end"]
        item = {"id": k, "start": str(pn.dates[w0["start"]]), "end": str(pn.dates[e]),
                "holdout": bool(e >= pn.holdout_start), "candidates": len(w0["cand"]),
                "pairs": int((variants["all"][1]["window_id"] == k).sum()) if len(variants["all"][1]) else 0,
                "has_c": bool(w0["res"] is not None and np.isfinite(w0["res"].get("p_C", np.array([np.nan]))).any()),
                "close": _r(pn.close[e], 2)}
        for var, (vw, pt) in variants.items():
            w = vw[k]
            sig = pt[(pt["window_id"] == k) & (pt["q_meta"] <= q)] if len(pt) else pt
            groups = groups_for(cx, w, sig) if len(sig) else []
            v = {"significant": int(len(sig)), "groups": len(groups), "candidates": len(w["cand"]),
                 "opposite": int((sig["dir_A"].eq("opposite") | sig["dir_C"].eq("opposite")).sum()) if len(sig) else 0}
            if k >= len(wins) - SHOW_WINDOWS:
                def mem(i):
                    return {"key": pn.branches[i], "name": pn.names[i], **ids.get(pn.branches[i], {}), "type": pn.btype[i], "cls": w["cls"][i],
                            "inv": round(float(cx.L["inventory"][i, e]) / 1000), "cost": _r(cx.L["avg_cost"][i, e], 2),
                            "net20": round(float((pn.B[i, e - 19:e + 1] - pn.S[i, e - 19:e + 1]).sum()) / 1000),
                            "censored": bool(cx.L["left_censored"][i, e])}
                naive = naive_group(cx, w, var == "nofx")
                v["group_list"] = [{"members": [{**mem(i), "loss_leg": i in g["loss_leg"],
                                                 "absorption": _r(g["absorption"].get(i), 3)} for i in g["members"]],
                                    "C_G": _r(g["C_G"], 2), "inv": round(g["inventory_G"] / 1000), "net20": round(g["net_20"] / 1000),
                                    "net_window": round(g["net_window"] / 1000), "foreign_share": _r(g["foreign_share"], 2),
                                    "pnl": {"realized": round(g["realized"] / 1e4), "unrealized": round(g["unrealized"] / 1e4),
                                            "total": round(g["total"] / 1e4), "incomplete": g["pnl_incomplete"]}}
                                   for g in groups]
                v["naive"] = {"members": [mem(i) for i in naive["members"]], "C_G": _r(naive["C_G"], 2),
                              "inv": round(naive["inventory_G"] / 1000), "net20": round(naive["net_20"] / 1000)}
                top = sig.nsmallest(15, "p_meta") if len(sig) else sig
                v["top_pairs"] = [{"a": pn.names[int(r.ia)], "b": pn.names[int(r.ib)], "obs": int(r.obs),
                                   "p_act": f3(r.p_act), "p_mag": f3(r.p_mag), "p_A": f3(r.p_A), "p_C": f3(r.p_C),
                                   "dir_A": r.dir_A, "dir_C": r.dir_C, "p": f3(r.p_meta), "q": f3(r.q_meta)} for r in top.itertuples()]
            item[var] = v
        windows.append(item)
    pt = variants["all"][1]
    log.info("主力集團 %s：%d 個窗口，%d 組配對（%.0f 秒）", sid, len(wins), len(pt), time.time() - t0)
    cal = calibration(cx, ends, seed=seed, max_windows=3 if quick else 8)
    syn = synthetic(cx, ends, seed=seed, reps=3 if quick else None,
                    Ks=[2, 3] if quick else None, rhos=[1.0, 0.5] if quick else None)
    doc = {"stock_id": sid, "kind": "stock", "stock_name": _name(sid), "asof": str(pn.dates[-1]), "params_version": P.version,
           "exclude_foreign_default": bool(P.exclude_foreign),
           "quick": quick, "market": len(mk.sids) if mk else 0,
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


def run_groups(mk, ctx: dict) -> list[dict]:
    from smallcap import capital_directory
    from .group import run_group
    cap = capital_directory()
    docs = []
    for g in P.groups or []:
        have = [m["sid"] for m in g["members"] if (LONG_DIR / m["sid"] / "_prices.json").exists()]
        if not have:
            log.warning("集團 %s：成員都還沒有分點資料，略過", g["id"])
            continue
        t0 = time.time()
        for s in have:
            if s not in ctx:
                ctx[s] = Context(build_panel(s, write=False))
        res = run_group(g, mk, ctx, cap)
        asof = max(str(ctx[s].pn.dates[-1]) for s in have)
        doc = {"stock_id": g["id"], "kind": "group", "stock_name": g["name"], "asof": asof, "params_version": P.version,
               "have": have, **res, "seconds": round(time.time() - t0)}
        p = store.OUT / g["id"] / "summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        log.info("集團 %s 完成：%.0f 秒", g["id"], time.time() - t0)
        docs.append(doc)
    return docs


def run_all(stocks: list[str] | None = None, quick: bool = False, max_hours: float | None = None) -> list[dict]:
    """每晚：全市場（維度 C）→ 測試股票 → 集團。max_hours：時間到就停（已算的窗口都有存檔，下次接著算）"""
    from . import group
    t0 = time.time()
    # 時間分兩半：前半給逐檔（維度 C 新窗口先算），後半給集團（同一批窗口有存檔可直接用，再往前補算），免得集團一個窗口都沒有
    group.DEADLINE[0] = t0 + max_hours * 3600 * 0.5 if max_hours else None
    mk = load_market()
    ctx: dict = {}
    docs = []
    for sid in stocks or list(P.run.test_stocks):
        if (LONG_DIR / sid / "_prices.json").exists():
            d = run(sid, quick=quick, mk=mk, ctx=ctx)
            if d:
                docs.append(d)
    if mk is not None:
        group.DEADLINE[0] = t0 + max_hours * 3600 if max_hours else None
        docs += run_groups(mk, ctx)
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stocks", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-hours", type=float)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_all(a.stocks or None, quick=a.quick, max_hours=a.max_hours)
