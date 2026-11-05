"""
擇時偵測 v1.1.2 協調流程（逐檔函式可平行化；BH 需要全部股票的 p 值，所以分兩階段）
  階段 A analyze_stock(sid)：面板 → 預先計算（§3.4）→ §6.1 門檻 → 主檢定 E 與原始版 R（§4.2～§4.4）
  BH（§7）：調參集宇宙（排除 D、H）全部通過門檻的「股票 × 分點」為一個族群，E、R 各一個族群
  階段 B finalize_stock(A, q)：只對 q^E ≤ 0.05 或 q^R ≤ 0.05 者做 E+、EI（bootstrap）、分類（§5）、描述欄位（§8）→ 輸出文件
保留池 H 永遠不產生可上傳的文件（split.assert_writable）；開發集 D 照常計算、不進族群、標「開發集，非驗證」
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import VERSION, classify, ei, hac, prep, reaction, report, zigzag
from .params import TP, horizons

log = logging.getLogger("broker_scraper")
DISCLAIMER = "分點是許多客戶的合計，顯著只代表「從這個分點下單的資金整體有此現象」，不能直接說是單一主力，也不構成違法認定。"
DEV_NOTE = "開發集，非驗證"
# SPEC-GAP-timing：以下各項規格未寫明，採最簡單、最保守的做法（詳見 broker_timing/CHANGELOG.md），也寫進輸出文件
SPEC_GAPS = [
    ("BH 族群（開發集）", "開發集 D 不進族群；D 的 q 值改在該股票自己通過門檻的分點之間做 BH（只供除錯顯示）"),
    ("股票宇宙", "§2.3 宇宙以有 longhist/_prices.json 的股票代替（同 v3 run.py）；產業成員也取這個宇宙"),
    ("時間保留", "檢定只用 T_cut（含）以前的資料；描述欄位（持股、損益、最近動作）用到最新一日"),
    ("β 最少筆數", "120 日滾動窗口內有效日報酬 < 60 筆 → β 缺值，該日不進樣本"),
    ("跨日交易量占比", "§6.1 第 4 條：Σ|n^cb| ≥ 0.5% × Σ(成交量 × g)（兩者都換算成最新股數基準）"),
    ("位置分數", "v0 定義不在倉庫：買進 1 − pos、賣出 pos，pos 為 [t−10, t+10] 還原收盤區間位置，|n| 加權"),
    ("來回勝率", "來回 = L 批庫存 0 → 正 → 0 的一段；勝 = 該段已實現 > 0"),
    ("滾動窗口", "沿用全期間反應函數的殘差，不在窗口內重估"),
    ("隔日沖標記", "用 v3 features.classify 在 T_cut 當日（前 120 日）的標籤"),
    ("價位明細", "分點價位明細的平均百分位 F_t(P)（v3 §10.3）尚未產出"),
    ("Firestore", "暫存 syndicate/{代號}@timing（沿用管理員限定的 syndicate 規則）"),
    ("文件大小", "K 線標記只留最近 2 年、滾動窗口只留最近 24 個；仍超過 900 KB 時由排序最後的分點起刪除標記與滾動序列（truncated 註記）"),
]


def _r(x, d=4):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def _f3(x):
    return None if x is None or not np.isfinite(x) else float(f"{x:.3g}")


def out_dir() -> Path:
    from paths import DATA_DIR
    return DATA_DIR / "broker_timing"


# ───────────────────────── 股票層級陣列 ─────────────────────────
def stock_arrays(pn, pr: prep.Prep, events, t_cut: str | None = None) -> dict:
    return arrays_for(pn.dates, pn.suspended, pr, events, t_cut)


def arrays_for(dates: np.ndarray, suspended: np.ndarray, pr: prep.Prep, events, t_cut: str | None = None) -> dict:
    """把預先計算的股票 × 交易日結果對齊到面板日期，並決定檢定期間（≤ T_cut）與樣本（§4.1）。
    g_t 用 events（分點所在股票的公司行動）；§9.1 安慰劑時 pr 換成 j′、events 仍是 j 的（ñ 不變）"""
    t_cut = str(t_cut or _t_cut())
    dates = np.asarray(dates).astype(str)
    idx = pr.align(dates)
    ok = idx >= 0

    def take(a):
        out = np.full((len(idx),) + a.shape[1:], np.nan)
        out[ok] = a[idx[ok]]
        return out
    hz = horizons()
    rex = np.column_stack([take(pr.rex[h]) for h in hz])
    feats = take(pr.feats)
    elig = np.zeros(len(idx), bool)
    elig[ok] = pr.eligible[idx[ok]]
    te = int(np.searchsorted(dates, t_cut, side="right")) - 1     # SPEC-GAP-timing：檢定只用 ≤ T_cut；描述欄位用到最新一日
    T = te + 1
    buf = int(TP.sample_end_buffer)
    tt = np.arange(len(idx))
    sample = elig & (tt <= te - buf) & np.isfinite(rex).all(axis=1) & ~np.asarray(suspended, bool)
    sample[T:] = False
    g, approx = prep.share_basis(dates[:T], events)
    return {"te": te, "T": T, "rex": rex, "ex1": pr.ex1, "pidx": idx, "feats": feats, "eligible": elig, "sample": np.flatnonzero(sample[:T]), "g": g,
            "approx": approx, "industry_factor": pr.industry_factor, "n_peers": pr.n_peers}


def _t_cut() -> str:
    from broker_flow_v3.params import P
    return str(P.governance.t_cut)


# ───────────────────────── 單一分點：門檻與主檢定 ─────────────────────────
def thresholds(n: np.ndarray, active: np.ndarray, sa: dict, close: np.ndarray, volume: np.ndarray) -> dict:
    T = sa["T"]
    g = sa["g"]
    ncb = g * n[:T]
    act = active[:T]
    s = sa["sample"]
    na = int(act.sum())
    vol_cb = float(np.nansum(volume[:T] * g))          # SPEC-GAP-timing：§6.1 第 4 條兩邊都換算成最新股數基準
    share = float(np.abs(ncb).sum() / vol_cb) if vol_cb > 0 else 0.0
    G = hac.geff(act[s]) if len(s) else 0
    out = {"active_days": na, "G_eff": G, "volume_share": share, "cycles": 0, "A": None, "cycles_ca_approx": 0}
    pre = na >= int(TP.min_active_days) and G >= int(TP.min_geff) and share >= float(TP.min_volume_share)
    if na:
        z = zigzag.run(ncb, close[:T], g, act, sa["approx"])
        out.update(cycles=z["cycles"], A=z["A"], cycles_ca_approx=z["cycles_ca_approx"], n_segments=z["n_segments"])
    out["pass"] = {"active_days": na >= int(TP.min_active_days), "G_eff": G >= int(TP.min_geff),
                   "cycles": out["cycles"] >= int(TP.min_cycles), "volume_share": share >= float(TP.min_volume_share)}
    out["eligible"] = bool(pre and out["pass"]["cycles"])
    return out


def branch_test(n: np.ndarray, active: np.ndarray, sa: dict, eplus: bool = False) -> dict:
    """§4.2～§4.4（E）與 §4.5 原始版（R）。n、active 為檢定期間（長度 ≥ T）"""
    T = sa["T"]
    g = sa["g"]
    act = active[:T]
    ntil, med = prep.standardize(g * n[:T], act)
    rf = reaction.fit(ntil, act, sa["feats"][:T], eplus=eplus)
    s = sa["sample"]
    Y = sa["rex"][s]
    tE = main_test(Y, rf["eps"], act, sa)
    out = {"ntil": ntil, "eps": rf["eps"], "median": med, "alpha": rf["alpha"], "n_act": rf["n_act"], "n_dropped": rf["n_dropped"], "E": tE}
    if not eplus:
        out["R"] = main_test(Y, ntil, act, sa)
    return out


def main_test(Y: np.ndarray, x: np.ndarray, act: np.ndarray, sa: dict) -> dict:
    """§4.4 的 p 值。第 2 輪（reverse_hw）：反向迴歸在價格表自己的時間軸上做（r^ex 就是在這條軸上由 e_s 加總），
    面板樣本日 s → 價格表索引 sa["pidx"][s]；其餘方法（fixed_b、t_geff）沿用樣本列上的 NW"""
    s = sa["sample"]
    if str(TP.hac_inference) != "reverse_hw" or sa.get("ex1") is None:
        return hac.test(Y, x[s], act[s], horizons=horizons())
    e = sa["ex1"]
    pi = sa["pidx"][s]
    xt = np.zeros(len(e)); St = np.zeros(len(e), bool); At = np.zeros(len(e), bool)
    xt[pi] = x[s]; St[pi] = True; At[pi] = act[s]
    return hac.reverse_test(xt, St, e, At, horizons=horizons())


def _test_fields(t: dict) -> dict:
    return {"p": _f3(t["p"]), "p_min": _f3(t["p_min"]), "M_eff": _r(t["M_eff"], 3), "df": int(t["df"]), "G_eff": int(t["G_eff"]),
            "h_star": t["h_star"], "theta": [_r(x, 6) for x in t["theta"]], "t": [_r(x, 3) for x in t["t"]]}


# ───────────────────────── 階段 A ─────────────────────────
def analyze_stock(sid: str, *, pn=None, pr: prep.Prep | None = None, events=None, Q=None, industry: prep.Industry | None = None,
                  market_fn: Callable | None = None, split_label: str | None = None, state_dir: Path | None = None,
                  t_cut: str | None = None) -> dict:
    """回傳輕量結果（每個通過門檻的分點的 p^E、p^R 等）；重算 EI 與描述欄位需要的陣列放在 state（inline 或存檔）"""
    t0 = time.time()
    if pn is None:
        from broker_flow_v3.data_layer import build_panel
        pn = build_panel(sid)
    if events is None:
        from broker_flow_v3.accounting import corp_actions
        events = corp_actions(sid)
    if pr is None:
        pr = prep.stock_prep(sid, str(pn.dates[-1]), industry, market_fn, out_dir() / "_cache")
    if split_label is None:
        from broker_flow_v3 import split
        split_label = split.split_of(sid, str(t_cut or _t_cut()))
    sa = stock_arrays(pn, pr, events, t_cut)
    T = sa["T"]
    nets = pn.B - pn.S
    rows, keep = [], []
    st = {}
    diag = []                                                  # 分點查詢（網站「3 分點查詢」）：活躍日 ≥ DIAG_MIN_ACTIVE 的分點各門檻
    for i in range(len(pn.branches)):
        n = nets[i]
        act = (n != 0) & ~pn.suspended
        na = int(act[:T].sum())
        if na < DIAG_MIN_ACTIVE:
            continue
        th = thresholds(n, act, sa, pn.close, pn.volume)
        diag.append([str(pn.branches[i]), str(pn.names[i]), na, int(th["G_eff"]), round(float(th["volume_share"]), 6),
                     int(th["cycles"]), bool(th["eligible"])])
        if not th["eligible"]:
            continue
        bt = branch_test(n, act, sa)
        rows.append({"sid": sid, "i": i, "branch": pn.branches[i], "name": pn.names[i], "type": pn.btype[i], "split": split_label,
                     "pE": float(bt["E"]["p"]), "pR": float(bt["R"]["p"]), "thr": th,
                     "E": _test_fields(bt["E"]), "R": _test_fields(bt["R"]), "alpha": bt["alpha"], "n_act": bt["n_act"]})
        keep.append(i)
        st[i] = {"eps": bt["eps"], "ntil": bt["ntil"], "theta_E": bt["E"]["theta"], "h_star_idx": bt["E"]["h_star_idx"]}
    # 描述欄位需要 v3 會計層（S 批分配依全部分點，所以在完整面板上算一次，只保留通過門檻的分點）
    L = None
    if keep:
        from broker_flow_v3 import accounting
        Lf = accounting.run(pn, events=events, Q=Q)
        L = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in Lf.items()}
    sub = dataclasses.replace(pn, branches=[pn.branches[i] for i in keep], names=[pn.names[i] for i in keep],
                              btype=[pn.btype[i] for i in keep], B=pn.B[keep], S=pn.S[keep], BP=pn.BP[keep], SP=pn.SP[keep]) if keep else None
    state = {"sid": sid, "sub": sub, "keep": keep, "L": L, "sa": sa, "branch": st, "split": split_label,
             "data": {"start": str(pn.dates[0]), "end": str(pn.dates[-1]), "test_end": str(pn.dates[sa["te"]]) if sa["te"] >= 0 else None,
                      "days": len(pn.dates), "branches": len(pn.branches), "sample_days": int(len(sa["sample"])),
                      "price_source": pn.meta.get("price_source"), "corp_actions": int(len(events)) if events is not None else 0,
                      "industry_factor": bool(sa["industry_factor"]), "n_peers": int(sa["n_peers"])}}
    res = {"sid": sid, "split": split_label, "rows": rows, "n_branches": len(pn.branches), "n_eligible": len(rows),
           "seconds": round(time.time() - t0, 1), "diag": diag}
    if state_dir is None:
        res["_state"] = state
    else:
        f = Path(state_dir) / f"state_{sid}.pkl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(pickle.dumps(state))
        res["_state_file"] = str(f)
    log.info("擇時 %s：%d 個分點中 %d 個通過門檻（%.0f 秒）", sid, len(pn.branches), len(rows), time.time() - t0)
    return res


# ───────────────────────── §7 BH 族群 ─────────────────────────
def assign_q(results: list[dict]) -> None:
    """調參集（split == tune）全部通過門檻的「股票 × 分點」為一個族群，E、R 各自 BH。
    D（dev）不進族群：SPEC-GAP-timing，改在該股票內部做 BH，只供除錯顯示。H（holdout_*）永遠不進族群、不給 q。"""
    fam = [r for res in results for r in res["rows"] if r["split"] == "tune"]
    if fam:
        qE = classify.bh([r["pE"] for r in fam])
        qR = classify.bh([r["pR"] for r in fam])
        for r, a, b in zip(fam, qE, qR):
            r["qE"], r["qR"], r["family"] = float(a), float(b), "tune"
    for res in results:
        rows = [r for r in res["rows"] if r["split"] == "dev"]
        if rows:
            qE = classify.bh([r["pE"] for r in rows])
            qR = classify.bh([r["pR"] for r in rows])
            for r, a, b in zip(rows, qE, qR):
                r["qE"], r["qR"], r["family"] = float(a), float(b), f"dev:{res['sid']}"
        for r in res["rows"]:
            if r["split"] not in ("tune", "dev"):
                r["qE"] = r["qR"] = None
                r["family"] = None


def family_members(results: list[dict]) -> list[tuple[str, str]]:
    return [(r["sid"], r["branch"]) for res in results for r in res["rows"] if r.get("family") == "tune"]


# ───────────────────────── 階段 B ─────────────────────────
def evaluate(sid: str, r: dict, n: np.ndarray, act: np.ndarray, sa: dict, bs: dict, B: int | None = None) -> dict:
    """已顯著者：E+（§4.5）、EI 鎖定 h*（§4.5）、分類（§5）"""
    hz = horizons()
    ep = branch_test(n, act, sa, eplus=True)["E"]
    hi = bs["h_star_idx"]
    eir = None
    s = sa["sample"]
    if hi is not None:
        fp, fm = ei.flows(bs["eps"], s, hz[hi])
        eir = ei.run(sa["rex"][s, hi], bs["eps"][s], fp, fm, sid, r["branch"], B=B)
    qE, qR = r.get("qE"), r.get("qR")
    kind = classify.classify(qE if qE is not None else np.nan, qR if qR is not None else np.nan,
                             eir["delta_lo"] if eir else np.nan, eir["delta_hi"] if eir else np.nan,
                             eir["kappa_plus"] if eir else np.nan, bool(eir and eir["unstable"]))
    return {"Eplus": ep, "EI": eir, "kind": kind}


def _load_state(res: dict) -> dict:
    if "_state" in res:
        return res["_state"]
    return pickle.loads(Path(res["_state_file"]).read_bytes())


def finalize_stock(res: dict, stock_name: str = "", asof: str | None = None, writable_check: Callable[[str], None] | None = None,
                   B: int | None = None) -> dict:
    """q 已由 assign_q 填好。只有 q^E ≤ 0.05 或 q^R ≤ 0.05 才做 E+、EI、分類與描述欄位"""
    sid = res["sid"]
    if writable_check is None:
        from broker_flow_v3 import split
        writable_check = split.assert_writable
    writable_check(sid)
    if res["split"] not in ("tune", "dev"):
        raise RuntimeError(f"{sid}：split = {res['split']}，不產生文件")
    st = _load_state(res)
    sa, sub, L = st["sa"], st["sub"], st["L"]
    a = float(TP.fdr_q)
    hz = horizons()
    out_rows = []
    counts = {k: 0 for k in classify.TYPES}
    vcls = None
    for r in res["rows"]:
        qE, qR = r.get("qE"), r.get("qR")
        sig = (qE is not None and qE <= a) or (qR is not None and qR <= a)
        if not sig:
            counts[classify.NONE] += 1
            continue
        i = r["i"]
        k = st["keep"].index(i)
        bs = st["branch"][i]
        n_full = sub.B[k] - sub.S[k]
        T = sa["T"]
        s = sa["sample"]
        ev = evaluate(sid, r, n_full, (n_full != 0) & ~sub.suspended, sa, bs, B=B)
        ep, eir, kind = ev["Eplus"], ev["EI"], ev["kind"]
        counts[kind] += 1
        if kind == classify.NONE:
            continue
        fl = classify.flags(kind, ep["p"], bool(eir and eir["unstable"]), r["thr"]["cycles"])
        if vcls is None:
            from broker_flow_v3 import features
            _, va, _ = features.scale_and_active(dataclasses.replace(sub, B=sub.B[:, :T], S=sub.S[:, :T], suspended=sub.suspended[:T]))
            vcls = features.classify(dataclasses.replace(sub, B=sub.B[:, :T], S=sub.S[:, :T]), va, T - 1)
        if vcls[k] == "overnight":
            fl.append("隔日沖型分點")
        if r["thr"]["cycles_ca_approx"]:
            fl.append(f"有 {r['thr']['cycles_ca_approx']} 個循環跨越現金減資（股數因子為近似值）")
        e_last = len(sub.dates) - 1
        mark_from = _years_before(str(sub.dates[e_last]), MARK_YEARS)
        hz_i = hz.index(20)
        tdates = sub.dates[:T]
        test = {"kind": kind, "label": classify.LABEL[kind], "code": classify.CODE[kind], "flags": fl,
                "pE": _f3(r["pE"]), "qE": _f3(qE), "pR": _f3(r["pR"]), "qR": _f3(qR), "p_Eplus": _f3(ep["p"]),
                "family": r.get("family"),
                "delta": _r(eir["delta"], 6) if eir else None, "delta_ci90": [_r(eir["delta_lo"], 6), _r(eir["delta_hi"], 6)] if eir else None,
                "kappa_plus": _r(eir["kappa_plus"], 6) if eir else None, "kappa_minus": _r(eir["kappa_minus"], 6) if eir else None,
                "bootstrap_discard_frac": _r(eir["discard_frac"], 4) if eir else None, "bootstrap_unstable": bool(eir and eir["unstable"]),
                "E": r["E"], "R": r["R"]}
        rows_s = report.rolling(bs["eps"][s], sa["rex"][s, hz_i], tdates[s])
        desc = {"rho": _r(eir["rho"], 3) if eir else None,
                "rho_ci90": [_r(eir["rho_lo"], 3), _r(eir["rho_hi"], 3)] if eir else None,
                "rho_drop_frac": _r(eir["rho_drop_frac"], 3) if eir else None, "rho_unstable": bool(eir and eir["rho_unstable"]),
                "h_star": r["E"]["h_star"], "cycles": r["thr"]["cycles"], "G_eff": r["thr"]["G_eff"], "active_days": r["thr"]["active_days"],
                "volume_share": _r(r["thr"]["volume_share"], 5), "few_cycles": r["thr"]["cycles"] < int(TP.few_cycles_flag),
                "timing_pnl": round(report.timing_pnl(n_full[:T], sub.vwap[:T], sa["rex"][:T, hz_i])),
                "exec": round(report.exec_component(sub.B[k], sub.S[k], sub.BP[k], sub.SP[k], sub.vwap)),
                "pnl": report.pnl(L, k, e_last, float(sub.close[e_last])),
                "position_score": _r(report.position_score(n_full, sub.adj_close), 3), "position_note": report.NOTE_POSITION,
                "holdings": report.holdings(L, k, e_last, float(sub.close[e_last])),
                "trend": rows_s[-1] if rows_s else None, "rolling": rows_s[-MAX_ROLLING:], "rolling_note": report.NOTE_ROLLING,
                "rolling_total": len(rows_s),
                "last_action": report.last_action(n_full, sub.BP[k], sub.SP[k], sub.dates),
                "marks": [[str(sub.dates[t]), round(float(n_full[t]) / 1000, 3)] for t in np.flatnonzero(n_full != 0)
                          if str(sub.dates[t]) >= mark_from],
                "v3_class": vcls[k]}
        out_rows.append({"branch": r["branch"], "name": r["name"], "branch_type": r["type"], **_web_ids(r["branch"], r["name"]),
                         "test": test, "descriptive": desc})
    out_rows.sort(key=lambda x: (x["test"]["code"], x["test"]["qE"] if x["test"]["qE"] is not None else 1.0))
    from broker_flow_v3.params import P
    doc = {"stock_id": sid, "stock_name": stock_name, "kind": "timing", "version": VERSION, "timing_version": str(TP.timing_version),
           "hac_inference": str(TP.hac_inference),
           "params_version": P.version, "asof": asof or st["data"]["end"], "split": st["split"],
           "split_note": DEV_NOTE if st["split"] == "dev" else None, "t_cut": _t_cut(), "disclaimer": DISCLAIMER,
           "data": st["data"],
           "summary": {"branches": res["n_branches"], "tested": res["n_eligible"], "types": counts},
           "rows": out_rows,
           "sections": {"test": ["kind", "flags", "pE", "qE", "pR", "qR", "delta", "delta_ci90", "kappa_plus", "p_Eplus"],
                        "descriptive": ["rho", "rho_ci90", "h_star", "cycles", "G_eff", "timing_pnl", "exec", "pnl", "position_score",
                                        "holdings", "trend", "last_action"]},
           "spec_gaps": [{"item": x, "note": y} for x, y in SPEC_GAPS],
           "diag": build_diag(res)}
    return cap_size(doc)


DIAG_MIN_ACTIVE = 10


def build_diag(res: dict) -> dict:
    """網站「3 分點查詢」：活躍日 ≥ 10 的分點的 §6.1 各門檻；通過門檻者附 p、q（不論是否顯著）。只供查詢，不是訊號"""
    pq = {str(r["branch"]): r for r in res["rows"]}
    rows = []
    for b, name, na, g, v, c, ok in res.get("diag") or []:
        r = pq.get(b)
        w = _web_ids(b, name)                                  # 系統格式名稱（富邦-二林）與富邦代碼，網站點選查詢用
        rows.append([b, w.get("name", name), na, g, v, c, ok] + ([_f3(r["pE"]), _f3(r.get("qE")), _f3(r["pR"]), _f3(r.get("qR"))] if r else [None] * 4)
                    + [w.get("broker_id"), w.get("b_raw")])
    rows.sort(key=lambda x: (not x[6], -x[2]))
    return {"cols": ["branch", "name", "active_days", "G_eff", "volume_share", "cycles", "eligible", "pE", "qE", "pR", "qR", "broker_id", "b_raw"],
            "min": {"active_days": int(TP.min_active_days), "G_eff": int(TP.min_geff), "volume_share": float(TP.min_volume_share),
                    "cycles": int(TP.min_cycles)}, "rows": rows}


def _web_ids(bid: str, name: str) -> dict:
    """網頁點分點 → K 線標記要用富邦的 broker_id、b_raw（同 broker_flow_v3.data_layer.fubon_keys）；查不到就留給網頁自己推"""
    try:
        import fm_store
        x = fm_store.fubon_row(str(bid), str(name))
        return {"name": x.get("broker") or name, "broker_id": x["broker_id"], "b_raw": x["b_raw"]}   # 名稱用系統格式（富邦-二林）
    except Exception:  # noqa: BLE001
        return {}


MAX_DOC_BYTES = 900_000          # SPEC-GAP-timing：Firestore 單一文件上限 1 MiB，保留餘裕
MAX_ROLLING = 24
MARK_YEARS = 2


def _years_before(d: str, years: int) -> str:
    return f"{int(d[:4]) - years:04d}{d[4:]}"


def cap_size(doc: dict, limit: int = MAX_DOC_BYTES) -> dict:
    """文件 > limit 時，由排序最後（最不重要）的分點起依序刪除 marks、rolling（描述欄位）；檢定欄位一律保留"""
    if len(dumps(doc).encode("utf-8")) <= limit:
        return doc
    doc["truncated"] = []
    for key in ("marks", "rolling"):
        for row in reversed(doc["rows"]):
            if row["descriptive"].get(key):
                row["descriptive"][key] = []
                doc["truncated"].append(f"{row['branch']}:{key}")
                if len(dumps(doc).encode("utf-8")) <= limit:
                    return doc
    # 檢定結果不刪；仍超過就標記，交由上傳端處理（Firestore 會拒收）
    doc["over_limit"] = True
    log.warning("擇時 %s：文件刪除標記與滾動序列後仍超過 %d bytes", doc.get("stock_id"), limit)
    return doc


def dumps(doc: dict) -> str:
    return json.dumps(doc, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ───────────────────────── 全部股票 ─────────────────────────
def universe() -> list[str]:
    from broker_flow_v3.data_layer import LONG_DIR
    return sorted(p.parent.name for p in LONG_DIR.glob("*/_prices.json"))


_WORKER: dict = {}
LAST_RESULTS: list[dict] = []


def _worker_analyze(sid: str) -> dict | None:
    try:
        if "industry" not in _WORKER:
            from . import fixedb
            fixedb.CACHE_DIR = out_dir() / "_cache" / "fixedb"      # spawn 的子行程也要設定（Windows）
            _WORKER["industry"] = prep.Industry(universe())
        return analyze_stock(sid, industry=_WORKER["industry"], state_dir=out_dir() / "_cache")
    except Exception as ex:  # noqa: BLE001
        log.exception("擇時 %s 失敗：%s", sid, ex)
        return None


def run_all(stocks: list[str] | None = None, processes: int = 1, on_doc: Callable[[list[dict]], None] | None = None,
            B: int | None = None) -> list[dict]:
    """§7：先算全部股票的 p（可平行）→ BH → 逐檔產出文件（H 不跑、不寫）"""
    from . import fixedb
    fixedb.CACHE_DIR = out_dir() / "_cache" / "fixedb"           # fixed-b 網格表依 (T, L) 存檔，各行程共用
    from broker_flow_v3 import split
    from broker_flow_v3.run import _name
    split.holdout()
    pool = stocks or universe()
    todo = [s for s in pool if not split.is_holdout(s)]
    log.info("擇時 v%s：%d 檔（另有 %d 檔在保留池，不計算）", VERSION, len(todo), len(pool) - len(todo))
    if processes > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(processes) as ex:
            results = [r for r in ex.map(_worker_analyze, todo) if r]
    else:
        results = [r for r in map(_worker_analyze, todo) if r]
    assign_q(results)
    LAST_RESULTS[:] = results                                    # §9 驗證（tools/timing_validate.py）直接沿用這次的結果
    docs = []
    for res in results:
        try:
            doc = finalize_stock(res, _name(res["sid"]), B=B)
        except Exception as ex:  # noqa: BLE001
            log.exception("擇時 %s 輸出失敗：%s", res["sid"], ex)
            continue
        p = out_dir() / res["sid"] / "summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dumps(doc), encoding="utf-8")
        docs.append(doc)
        if on_doc:
            on_doc(docs)
    return docs
