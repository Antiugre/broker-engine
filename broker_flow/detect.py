"""
偵測窗口（6.0）× 維度 A（6.1）→ 融合（8.2、8.3）→ 集團（8.5）＋ 樸素基準線（9.0）
窗口：全市場交易日曆上的固定格點（market.window_end_dates），各檔對到自己的交易日
維度 A 的結果按 (股票, 窗口) 存檔重用；用到的資料（窗口前 240 日的明細與分點清單）一變就重算
"""
from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd

from . import accounting, dim_a, features
from .data_layer import Panel
from .fusion import bh, cct, components, group_cost
from .params import P
from .store import OUT

CACHE_A = OUT / "_cache" / "dim_a"


def window_ends(pn: Panel, end_dates: list[str] | None = None) -> list[int]:
    """窗口結束日在這檔股票的索引。end_dates＝全市場固定格點；沒給就用這檔自己的日曆做同樣的格點"""
    W, step = int(P.detection.window_days), int(P.detection.window_step)
    first = int(P.accounting.warmup_days) + W - 1
    if end_dates is None:
        ends = list(range(first, len(pn.dates), step))
        if ends and ends[-1] != len(pn.dates) - 1:
            ends.append(len(pn.dates) - 1)
        return ends
    idx = sorted({int(np.searchsorted(pn.dates, d, side="right")) - 1 for d in end_dates})
    return [e for e in idx if e >= first]


class Context:
    """一檔股票的共用中間結果（特徵、當日因子；會計在用到時才算）"""

    def __init__(self, pn: Panel):
        self.pn = pn
        self._L = None
        self.scale, self.active, self.z = features.scale_and_active(pn)
        self.F, self.ret = dim_a.day_factors(pn.close, pn.volume)

    @property
    def L(self):
        if self._L is None:
            self._L = accounting.run(self.pn)
        return self._L


def run_window(cx: Context, e: int, allow_holdout: bool = False, exclude_foreign: bool | None = None) -> dict:
    """exclude_foreign（修正規格第 1 節，預設 false）：外資分點不當候選；市場總量照常計入"""
    pn = cx.pn
    pn.guard(e, allow_holdout)
    if exclude_foreign is None:
        exclude_foreign = bool(P.exclude_foreign)
    W = int(P.detection.window_days)
    s = e - W + 1
    cls = features.classify(pn, cx.active, e)
    A = cx.active[:, s:e + 1]
    cand = [i for i in range(len(pn.branches))
            if cls[i] not in features.EXCLUDED and A[i].sum() >= int(P.detection.min_candidate_active)
            and not (exclude_foreign and pn.btype[i] == "foreign")]
    out = {"end": e, "start": s, "cls": cls, "cand": cand, "exclude_foreign": exclude_foreign}
    if len(cand) < 2:
        out["res"] = None
        return out
    out["res"] = _dim_a_cached(cx, e, cand)
    out["res"]["p_meta"] = out["res"]["p_A"]          # 維度 C 之後由 attach_c 併入
    return out


def _dim_a_cached(cx: Context, e: int, cand: list[int]) -> dict:
    pn = cx.pn
    W = int(P.detection.window_days)
    s = e - W + 1
    lo = max(0, e - W - int(P.behavior.rolling_days) + 1)
    h = hashlib.sha256()
    h.update(("|".join(pn.branches[i] for i in cand)).encode())
    h.update(np.ascontiguousarray(pn.B[:, lo:e + 1]).tobytes()); h.update(np.ascontiguousarray(pn.S[:, lo:e + 1]).tobytes())
    h.update(np.ascontiguousarray(pn.close[lo:e + 1]).tobytes()); h.update(np.ascontiguousarray(pn.volume[lo:e + 1]).tobytes())
    f = CACHE_A / pn.sid / f"{pn.dates[e]}_{h.hexdigest()[:16]}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    res = dim_a.run_window(cx.active[cand, s:e + 1], cx.z[cand, s:e + 1], cx.F[s:e + 1], cx.ret[s:e + 1])
    res.pop("prob", None)
    f.parent.mkdir(parents=True, exist_ok=True)
    for old in f.parent.glob(f"{pn.dates[e]}_*.pkl"):
        old.unlink()
    f.write_bytes(pickle.dumps(res))
    return res


def attach_c(w: dict, pn: Panel, cres: dict | None) -> None:
    """把維度 C 的 p_C 對到維度 A 的配對上，算 p_meta（8.2：A、C 各一票；B 無法計算 → 移除，剩下平分）"""
    r = w["res"]
    if r is None:
        return
    n = len(w["cand"])
    p_c = np.full(len(r["pairs"]), np.nan)
    d_c = np.full(len(r["pairs"]), -1, np.int8)
    pc = (cres or {}).get("pairs", {}).get(pn.sid)
    if pc is not None:
        pos = {k: i for i, k in enumerate(pc["keys"])}
        iu = np.triu_indices(len(pc["keys"]), 1)
        mat = np.full((len(pc["keys"]),) * 2, np.nan)
        dmat = np.full((len(pc["keys"]),) * 2, -1, np.int8)
        mat[iu] = pc["p_C"]
        dmat[iu] = pc.get("dir_C", np.full(len(iu[0]), -1, np.int8))
        for k, (a, b) in enumerate(r["pairs"]):
            ka, kb = pn.branches[w["cand"][a]], pn.branches[w["cand"][b]]
            if ka in pos and kb in pos:
                x, y = sorted((pos[ka], pos[kb]))
                p_c[k] = mat[x, y]
                d_c[k] = dmat[x, y]
    r["p_C"] = p_c
    r["dir_C"] = d_c
    # 8.2：p_meta = CCT(p_A, p_B, p_C) 各 1/3；p_B（B1、B2）沒有價位資料、無法計算 → 移除，剩下平分
    r["p_meta"] = cct([r["p_A"], p_c], [1 / 3, 1 / 3]) if n else r["p_A"]


def pairs_table(pn: Panel, wins: list[dict]) -> pd.DataFrame:
    """8.6 pairs（全局 BH：所有窗口、所有配對一起）"""
    rows = []
    for k, w in enumerate(wins):
        r = w["res"]
        if r is None:
            continue
        c = np.array(w["cand"])
        a, b = c[r["pairs"][:, 0]], c[r["pairs"][:, 1]]
        rows.append(pd.DataFrame({
            "date_end": pn.dates[w["end"]], "stock_id": pn.sid, "window_id": k,
            "ia": a, "ib": b, "branch_a": np.array(pn.branches)[a], "branch_b": np.array(pn.branches)[b],
            "obs": r["obs"], "p_act": r["p_act"], "p_mag": r["p_mag"], "p_A": r["p_A"],
            "dir_A": np.array(["same", "opposite", ""])[r.get("dir_A", np.full(len(a), -1))],
            "p_B1": np.nan, "p_B2": np.nan, "p_B": np.nan, "seller": "", "buyer": "",
            "p_C": r.get("p_C", np.full(len(a), np.nan)),
            "dir_C": np.array(["same", "opposite", ""])[r.get("dir_C", np.full(len(a), -1))],
            "p_meta": r.get("p_meta", r["p_A"])}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(df):
        df["q_meta"] = bh(df["p_meta"])
        df["q_A"] = bh(df["p_A"])
        df["q_B"] = np.nan
        df["q_C"] = bh(df["p_C"])
    return df


def groups_for(cx: Context, w: dict, sig: pd.DataFrame) -> list[dict]:
    pn = cx.pn
    L = cx.L
    e, s = w["end"], w["start"]
    edges = list(zip(sig["ia"].astype(int), sig["ib"].astype(int)))
    vol = float(pn.volume[s:e + 1].sum())
    out = []
    for g in components(len(pn.branches), edges):
        C, inv = group_cost(g, L["inventory"][:, e], L["avg_cost"][:, e], vol)
        net = (pn.B[g, s:e + 1] - pn.S[g, s:e + 1]).sum(axis=0)
        out.append({"members": g, "C_G": C, "inventory_G": inv, "net_window": float(net.sum()),
                    "net_20": float(net[-int(P.validation.naive_last_days):].sum()),
                    "foreign_share": float(np.mean([pn.btype[i] == "foreign" for i in g])),
                    **group_pnl(cx, g, s, e), "absorption": absorption(cx, g, s, e)})
    return out


def group_pnl(cx: Context, g: list[int], s: int, e: int) -> dict:
    """修正四 6.1、6.2：集團損益（只計 left_censored = False 的成員）與虧損腿
    SPEC-GAP：沒有分點的買進／賣出均價 → 以個股當日 VWAP、用各成員自己的買（賣）量加權，算出窗口內買進／賣出均價"""
    pn, L = cx.pn, cx.L
    ok = [i for i in g if not L["left_censored"][i, e]]
    real = {i: float(L["realized"][i, s:e + 1].sum() + L["dt_pnl"][i, s:e + 1].sum()) for i in g}
    unreal = {i: float(L["inventory"][i, e] * (pn.close[e] - L["avg_cost"][i, e])) for i in g}
    rp = sum(real[i] for i in ok)
    up = sum(unreal[i] for i in ok)
    vw = pn.vwap[s:e + 1]

    def avg(x):
        return float((x * vw).sum() / x.sum()) if x.sum() > 0 else None
    buy_v = {i: avg(pn.B[i, s:e + 1]) for i in g}
    sell_v = {i: avg(pn.S[i, s:e + 1]) for i in g}
    tol = float(P.revision.loss_leg_vwap_tol)
    loss = []
    if rp + up >= 0:
        for i in ok:
            if real[i] < 0 and sell_v[i] and any(buy_v[r] and abs(buy_v[r] - sell_v[i]) <= tol * sell_v[i] for r in g if r != i):
                loss.append(i)
    return {"realized": rp, "unrealized": up, "total": rp + up, "pnl_incomplete": len(ok) < len(g), "loss_leg": loss}


def absorption(cx: Context, g: list[int], s: int, e: int) -> dict:
    """修正三 5.6（只報告）：賣方成員 S 的事件日（只要求條件 1 大量賣出、2 價格受壓），集團其他成員淨買量合計 ÷ S 淨賣量
    「過去 120 日活躍日 |淨買賣| 中位數」只用 t 日以前的資料"""
    pn = cx.pn
    rv = P.revision
    out = {}
    net = pn.B - pn.S
    for i in g:
        others = [r for r in g if r != i]
        if not others:
            continue
        num = den = 0.0
        for t in range(max(s, 1), e + 1):
            lo = max(0, t - int(P.behavior.rolling_days))
            hist = net[i, lo:t][cx.active[i, lo:t]]
            if not len(hist):
                continue
            ns = -net[i, t]
            if ns >= float(rv.b2_sell_mult) * np.median(np.abs(hist)) and ns >= float(rv.b2_sell_min_lots) * 1000 \
                    and pn.close[t] <= pn.close[t - 1]:
                num += max(0.0, float(net[others, t].sum()))
                den += ns
        if den > 0:
            out[i] = num / den
    return out


def naive_group(cx: Context, w: dict, nofx: bool = False) -> dict:
    """9.0：窗口最後 20 日累積淨買超前 10 名（排除當沖型、隔日沖型；exclude_foreign 時再排除外資）"""
    pn = cx.pn
    L = cx.L
    e, s = w["end"], w["start"]
    d = int(P.validation.naive_last_days)
    net20 = (pn.B[:, e - d + 1:e + 1] - pn.S[:, e - d + 1:e + 1]).sum(axis=1)
    ok = [i for i in np.argsort(-net20) if w["cls"][i] not in features.EXCLUDED and net20[i] > 0
          and not (nofx and pn.btype[i] == "foreign")][: int(P.validation.naive_top)]
    C, inv = group_cost(ok, L["inventory"][:, e], L["avg_cost"][:, e], float(pn.volume[s:e + 1].sum()))
    return {"members": [int(i) for i in ok], "C_G": C, "inventory_G": inv, "net_20": float(net20[ok].sum())}
