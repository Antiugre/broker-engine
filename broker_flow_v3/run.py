"""
v3 主程式：開發集（2601、5907）逐檔 → 集團模式（2601G）→ 網頁摘要 JSON（data/broker_flow_v3/<代號>/summary.json）
  python -m broker_flow_v3.run [代號 ...] [--no-group]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from . import features, split, store
from .data_layer import LONG_DIR, build_panel, fubon_keys
from .detect import Context, groups_for, naive_group, pairs_frame, run_window, window_ends
from .params import P

log = logging.getLogger("broker_scraper")
SHOW = 12
SPEC_GAPS = [
    ("v3 上線範圍", "維度 C 以分工架構在背景重算；還沒算好的窗口，融合裡維度 C 為缺失（規格 §8.1：缺的分量移除、其餘平分）"),
    ("維度 C 計算量", "依 §7.3「計算時間不夠時觀測與虛無同時降低」：初始值 10 → 5 組、穩定門檻 8 → 4 組"),
    ("維度 C 股票池", "雲端每晚計算範圍內有價格的股票（§2.3 前 800 檔宇宙待輔助資料齊全後切換）"),
    ("資料期間", "只用 FinMind（2021-06-30 起）；之前的富邦資料尚未併入"),
    ("現金股利", "規格 §3.2「C′ = C − D 且收益 += D·I」會重複計算；本實作只調降成本，股利另列參考欄"),
    ("維度 A 虛無", "每次排列抽 ≤ 2,000 個配對計算虛無分數（觀測值算全部配對），池化後的虛無樣本仍有數十萬筆"),
    ("維度 B 配對範圍", "賣方限候選分點（v3 候選已含當沖型；預設版本含外資）；非候選賣方沒有維度 A 配對可融合"),
    ("自我成交", "MAD = 0 時下限取 0.01"),
    ("八大行庫", "以分點名稱比對"),
    ("鉅額交易", "分點鉅額明細（2026-04-28 起）尚未併入會計層"),
    ("保留池", "已知關係企業集團不足 10 組，只用個股分層抽樣"),
    ("假集團", "「候選分點數」「分點集中度 HHI」特徵尚未加入；只算最近 6 個窗口、只算預設版本"),
    ("Firestore", "v3 結果暫存 syndicate/{代號}@v3（同樣只有管理員可讀），規則更新後搬到 syndicate_v3/"),
]


def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def f3(x):
    return None if x is None or not np.isfinite(x) else float(f"{x:.3g}")


def run_stock(sid: str, ctx: dict) -> dict:
    split.assert_writable(sid)
    t0 = time.time()
    pn = build_panel(sid)
    cx = ctx.get(sid) or Context(pn)
    ctx[sid] = cx
    fk = fubon_keys(pn)
    try:
        from smallcap import _branch_ids
        ids = _branch_ids()
    except Exception:  # noqa: BLE001
        ids = {}
    ends = window_ends(pn)
    L = cx.L
    cdir = store.OUT / "_cache" / "dim_c_windows"          # 維度 C 分工計算的結果（tools/dimc_v3.py windows）

    def cres_for(end: str):
        f = cdir / f"dimc_{end}.pkl"
        if not f.exists():
            return None
        try:
            import pickle
            return pickle.loads(f.read_bytes())
        except Exception:  # noqa: BLE001
            return None
    windows = []
    allpairs = []
    for k, (e, prov) in enumerate(ends):
        item = {"id": k, "start": str(pn.dates[e - int(P.detection.window_days) + 1]), "end": str(pn.dates[e]),
                "provisional": prov, "split": split.split_of(sid, str(pn.dates[e])), "close": _r(pn.close[e]), "has_c": False}
        cres = cres_for(str(pn.dates[e]))
        for var, nofx in (("all", False), ("nofx", True)):
            w = run_window(cx, e, nofx, cres)
            r = w["res"]
            if r is None:
                item[var] = {"candidates": len(w["cand"]), "significant": 0, "groups": 0}
                continue
            if not nofx:
                item["has_c"] = bool(np.isfinite(r.get("p_C_same", np.array([np.nan]))).any() or np.isfinite(r.get("p_C_opp", np.array([np.nan]))).any())
                item.update(candidates=len(w["cand"]), pairs=len(r["pairs"]), b_days=w.get("b_days", 0), b_skipped=w.get("b_skipped", 0),
                            has_b=bool(np.isfinite(r["p_B"]).any()))
                allpairs.append(pairs_frame(pn, w))
            v = {"candidates": len(w["cand"]), "sig_same": int(r["sig_same"].sum()), "sig_opp": int(r["sig_opp"].sum())}
            v["significant"] = v["sig_same"] + v["sig_opp"]
            if k >= len(ends) - SHOW:
                cg = groups_for(cx, w)
                e_ = e
                s_ = w["start"]

                def mem(i, g=None):
                    return {"key": fk[i], "tid": pn.branches[i], "name": pn.names[i], **ids.get(fk[i], {}), "type": pn.btype[i], "cls": w["cls"][i],
                            "inv": round(float(L["inventory"][i, e_]) / 1000), "cost": _r(L["avg_cost"][i, e_]),
                            "short": round(float(L["short_inv"][i, e_]) / 1000),
                            "net20": round(float((pn.B[i, e_ - 19:e_ + 1] - pn.S[i, e_ - 19:e_ + 1]).sum()) / 1000),
                            "sc": int(cx.sc[i, s_:e_ + 1].sum()),
                            **({} if g is None else {"loss_leg": i in g["loss_leg"], "absorption": _r(g["absorption"].get(i), 3)})}
                v["cores"] = [{"members": [mem(i, g) for i in g["members"]], "bridges": [mem(i) for i in g["bridges"]],
                               "rivals": [mem(i) for i in g["opp_rivals"]], "density": _r(g["density"], 3),
                               "C_G": _r(g["C_G"]), "inv": round(g["inventory_G"] / 1000), "net20": round(g["net_20"] / 1000),
                               "net_window": round(g["net_window"] / 1000), "foreign_share": _r(g["foreign_share"]),
                               "pnl": {"realized": round(g["realized"] / 1e4), "unrealized": round(g["unrealized"] / 1e4),
                                       "total": round(g["total"] / 1e4), "total_fee30": round(g["total_fee30"] / 1e4),
                                       "uncost": round(g["uncost_value"] / 1e4), "dividend_memo": round(g["dividend_memo"] / 1e4),
                                       "ca_approx": g["ca_approx"]}} for g in cg["cores"]]
                v["pairs2"] = [[mem(a), mem(b)] for a, b in cg["pairs"]][:20]
                v["groups"] = len(cg["cores"])
                nv = naive_group(cx, w, nofx)
                v["naive"] = {"members": [mem(i) for i in nv["members"]], "C_G": _r(nv["C_G"]), "inv": round(nv["inventory_G"] / 1000),
                              "net20": round(nv["net_20"] / 1000)}
                pf = pairs_frame(pn, w)
                pf["best"] = np.fmin(pf["q_same"], pf["q_opp"])
                top = pf[pf["sig_same"] | pf["sig_opp"]].nsmallest(20, "best")
                nm = dict(zip(pn.branches, pn.names))
                v["top_pairs"] = [{"a": nm[x.branch_a], "b": nm[x.branch_b], "obs": int(x.obs), "rho": _r(x.rho, 3),
                                   "p_act": f3(x.p_act), "p_mag_pos": f3(x.p_mag_pos), "p_mag_neg": f3(x.p_mag_neg),
                                   "p_A_same": f3(x.p_A_same), "p_A_opp": f3(x.p_A_opp), "p_B1": f3(x.p_B1), "p_B2": f3(x.p_B2), "p_B": f3(x.p_B),
                                   "p_same": f3(x.p_same), "p_opp": f3(x.p_opp), "q_same": f3(x.q_same), "q_opp": f3(x.q_opp),
                                   "channel": "same" if x.sig_same and (not x.sig_opp or x.q_same <= x.q_opp) else "opp",
                                   "seller": nm.get(x.seller, "") if x.seller else "", "buyer": nm.get(x.buyer, "") if x.buyer else ""}
                                  for x in top.itertuples()]
            else:
                v["groups"] = None
            item[var] = v
        windows.append(item)
        log.info("v3 %s 窗口 %s：候選 %s、同向顯著 %s、反向顯著 %s", sid, item["end"], item.get("candidates"),
                 item["all"].get("sig_same"), item["all"].get("sig_opp"))
    if allpairs:
        import pandas as pd
        store.write(pd.concat(allpairs, ignore_index=True), sid, "pairs")
    doc = {"stock_id": sid, "kind": "stock", "version": 3, "stock_name": _name(sid), "asof": str(pn.dates[-1]), "params_version": P.version,
           "source": "finmind", "price_source": pn.meta.get("price_source"),
           "dim_c_status": "ok" if any(w.get("has_c") for w in windows) else "pending_v3.1",
           "data": {"start": str(pn.dates[0]), "end": str(pn.dates[-1]), "days": len(pn.dates), "branches": len(pn.branches),
                    "analysis_start": str(pn.dates[min(len(pn.dates) - 1, int(P.accounting.warmup_days))]), "t_cut": str(P.governance.t_cut),
                    "suspended_days": int(pn.suspended.sum()), "corp_actions": int(L["n_events"])},
           "windows": windows, "spec_gaps": [{"item": a, "note": b} for a, b in SPEC_GAPS], "seconds": round(time.time() - t0)}
    p = store.OUT / sid / "summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    log.info("v3 %s 完成：%d 個窗口（%.0f 秒）", sid, len(windows), time.time() - t0)
    return doc


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


def run_all(stocks: list[str] | None = None, groups: bool = True, on_doc=None) -> list[dict]:
    from .group import run_group
    ctx: dict = {}
    docs = []
    split.holdout()
    for sid in stocks or list(P.governance.dev_stocks):
        try:
            d = run_stock(sid, ctx)
            if d:
                docs.append(d)
                if on_doc:
                    on_doc(docs)                       # 每算完一檔就先上傳，不必等集團模式
        except Exception as ex:  # noqa: BLE001
            log.exception("v3 %s 失敗：%s", sid, ex)
    if groups:
        pool = sorted(p.parent.name for p in LONG_DIR.glob("*/_prices.json"))
        for g in P.groups or []:
            if not all(s in ctx for s in [m["sid"] for m in g["members"]][:1]):
                continue
            t0 = time.time()
            ends = sorted({str(c.pn.dates[e]) for c in ctx.values() for e, prov in window_ends(c.pn) if not prov})
            try:
                res = run_group(g, {k: v for k, v in ctx.items() if k in {m["sid"] for m in g["members"]}}, pool, ends)
            except Exception as ex:  # noqa: BLE001
                log.exception("v3 集團 %s 失敗：%s", g["id"], ex)
                continue
            asof = max(str(ctx[m["sid"]].pn.dates[-1]) for m in g["members"] if m["sid"] in ctx)
            doc = {"stock_id": g["id"], "kind": "group", "version": 3, "stock_name": g["name"], "asof": asof, "params_version": P.version,
                   "have": [m["sid"] for m in g["members"] if m["sid"] in ctx], **res, "seconds": round(time.time() - t0)}
            p = store.OUT / g["id"] / "summary.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(doc, ensure_ascii=False, default=str), encoding="utf-8")
            docs.append(doc)
            if on_doc:
                on_doc(docs)
    return docs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stocks", nargs="*")
    ap.add_argument("--no-group", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run_all(a.stocks or None, not a.no_group)
