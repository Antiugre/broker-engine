"""
偵測窗口（6.0）× 維度 A（6.1）→ 融合（8.2、8.3）→ 集團（8.5）＋ 樸素基準線（9.0）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import accounting, dim_a, features
from .data_layer import Panel
from .fusion import bh, components, group_cost
from .params import P


def window_ends(pn: Panel) -> list[int]:
    """窗口 120 日、步長 20；分析起點 = 暖身期 180 日之後；最後一個窗口對齊最後一個交易日"""
    W, step = int(P.detection.window_days), int(P.detection.window_step)
    first = int(P.accounting.warmup_days) + W - 1
    return sorted(range(len(pn.dates) - 1, first - 1, -step))


class Context:
    """一檔股票的共用中間結果（會計、特徵、當日因子），只算一次"""

    def __init__(self, pn: Panel):
        self.pn = pn
        self.L = accounting.run(pn)
        self.scale, self.active, self.z = features.scale_and_active(pn)
        self.F, self.ret = dim_a.day_factors(pn.close, pn.volume)


def run_window(cx: Context, e: int, allow_holdout: bool = False) -> dict:
    pn = cx.pn
    pn.guard(e, allow_holdout)
    W = int(P.detection.window_days)
    s = e - W + 1
    cls = features.classify(pn, cx.active, e)
    A = cx.active[:, s:e + 1]
    cand = [i for i in range(len(pn.branches))
            if cls[i] not in features.EXCLUDED and A[i].sum() >= int(P.detection.min_candidate_active)]
    out = {"end": e, "start": s, "cls": cls, "cand": cand}
    if len(cand) < 2:
        out["res"] = None
        return out
    res = dim_a.run_window(A[cand], cx.z[cand, s:e + 1], cx.F[s:e + 1], cx.ret[s:e + 1])
    res["p_meta"] = res["p_A"]                    # 8.2 第 4 條：B、C 無法計算 → 只剩 A
    out["res"] = res
    return out


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
            "p_B": np.nan, "p_CP": np.nan, "p_Fr": np.nan, "p_C": np.nan, "p_meta": r["p_meta"]}))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(df):
        df["q_meta"] = bh(df["p_meta"])
        df["q_A"] = df["q_meta"]
        df["q_B"] = np.nan
        df["q_C"] = np.nan
    return df


def groups_for(cx: Context, w: dict, sig: pd.DataFrame) -> list[dict]:
    pn, L = cx.pn, cx.L
    e, s = w["end"], w["start"]
    edges = list(zip(sig["ia"].astype(int), sig["ib"].astype(int)))
    vol = float(pn.volume[s:e + 1].sum())
    out = []
    for g in components(len(pn.branches), edges):
        C, inv = group_cost(g, L["inventory"][:, e], L["avg_cost"][:, e], vol)
        net = (pn.B[g, s:e + 1] - pn.S[g, s:e + 1]).sum(axis=0)
        out.append({"members": g, "C_G": C, "inventory_G": inv, "net_window": float(net.sum()),
                    "net_20": float(net[-int(P.validation.naive_last_days):].sum()),
                    "foreign_share": float(np.mean([pn.btype[i] != "brokerage" for i in g]))})
    return out


def naive_group(cx: Context, w: dict) -> dict:
    """9.0：窗口最後 20 日累積淨買超前 10 名（排除當沖型、隔日沖型）"""
    pn, L = cx.pn, cx.L
    e, s = w["end"], w["start"]
    d = int(P.validation.naive_last_days)
    net20 = (pn.B[:, e - d + 1:e + 1] - pn.S[:, e - d + 1:e + 1]).sum(axis=1)
    ok = [i for i in np.argsort(-net20) if w["cls"][i] not in features.EXCLUDED and net20[i] > 0][: int(P.validation.naive_top)]
    C, inv = group_cost(ok, L["inventory"][:, e], L["avg_cost"][:, e], float(pn.volume[s:e + 1].sum()))
    return {"members": [int(i) for i in ok], "C_G": C, "inventory_G": inv, "net_20": float(net20[ok].sum())}
