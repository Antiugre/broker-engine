"""
偵測窗口 × 維度 A、B（、C）→ §8 雙通道融合、逐窗口 BH → 核心集團（雙連通分量）＋ 成本、損益、虧損腿、承接比例、樸素基準線
窗口格點同 v2（全市場交易日曆上的固定格點：從分析起點每 20 日一個，再加最後一個交易日；§8.2 最後一個非格點窗口標 provisional）
"""
from __future__ import annotations

import hashlib
import logging
import pickle

import numpy as np
import pandas as pd

from . import accounting, dim_a, dim_b, features
from .fusion import cores, group_cost, window_fdr
from .params import P
from .store import OUT

log = logging.getLogger("broker_scraper")
CACHE = OUT / "_cache"
CODE = 4          # 程式邏輯改變時加 1（快取失效）


def window_ends(pn) -> list[tuple[int, bool]]:
    """[(結束索引, provisional)]"""
    W, step = int(P.detection.window_days), int(P.detection.window_step)
    first = int(P.accounting.warmup_days) + W - 1
    T = len(pn.dates)
    ends = [(e, False) for e in range(first, T, step)]
    if T - 1 >= first and (not ends or ends[-1][0] != T - 1):
        ends.append((T - 1, True))
    return ends


class Context:
    def __init__(self, pn):
        self.pn = pn
        self.scale, self.active, self.z = features.scale_and_active(pn)
        self._L = self._L3 = None
        self._lv = False
        self._sc = None

    @property
    def L(self):
        if self._L is None:
            self._ev = accounting.corp_actions(self.pn.sid)
            self._Q = accounting.short_capacity(self.pn.sid, self.pn.dates)
            self._L = accounting.run(self.pn, events=self._ev, Q=self._Q)
            log.info("v3 會計 %s：公司行動 %d 件、融券＋借券賣出日 %d", self.pn.sid, len(self._ev), int((self._Q > 0).sum()))
        return self._L

    @property
    def L_sens(self):
        if self._L3 is None:
            self.L
            self._L3 = accounting.run(self.pn, fee_discount=float(P.accounting.fee_discount_sensitivity), events=self._ev, Q=self._Q)
        return self._L3

    @property
    def levels(self):
        if self._lv is False:
            try:
                self._lv = dim_b.levels(self.pn)
            except Exception as ex:  # noqa: BLE001
                log.warning("v3 價位明細 %s 讀取失敗（%s）：維度 B 不計算", self.pn.sid, ex)
                self._lv = None
        return self._lv

    @property
    def sc(self):
        if self._sc is None:
            lv = self.levels
            self._sc = features.self_cross(self.pn, lv["X"]) if lv else np.zeros(self.pn.B.shape, bool)
        return self._sc


def _hash(cx: Context, s: int, e: int, cand: list[int]) -> str:
    pn = cx.pn
    lo = max(0, s - int(P.behavior.rolling_days))
    h = hashlib.sha256(f"v{P.version}|code{CODE}|".encode())
    h.update("|".join(pn.branches[i] for i in cand).encode())
    for a in (pn.B[:, lo:e + 1], pn.S[:, lo:e + 1], pn.close[lo:e + 1], pn.volume[lo:e + 1], pn.adj_close[lo:e + 1]):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _cached(kind: str, cx: Context, e: int, hsh: str, fn):
    f = CACHE / kind / cx.pn.sid / f"{cx.pn.dates[e]}_{hsh}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes())
        except Exception:  # noqa: BLE001
            pass
    r = fn()
    f.parent.mkdir(parents=True, exist_ok=True)
    for old in f.parent.glob(f"{cx.pn.dates[e]}_*.pkl"):
        old.unlink()
    f.write_bytes(pickle.dumps(r))
    return r


def candidates(cx: Context, s: int, e: int, cls: list[str], nofx: bool) -> list[int]:
    pn = cx.pn
    na = cx.active[:, s:e + 1].sum(1)
    sc_ev = cx.sc[:, s:e + 1].sum(1)
    mn = int(P.detection.min_candidate_active)
    return [i for i in range(len(pn.branches)) if cls[i] not in features.EXCLUDED
            and (na[i] >= mn or sc_ev[i] >= int(P.behavior.sc_min_events))
            and not (nofx and pn.btype[i] == "foreign")]


def run_window(cx: Context, e: int, nofx: bool = False, cres: dict | None = None) -> dict:
    pn = cx.pn
    W = int(P.detection.window_days)
    s = e - W + 1
    cls = features.classify(pn, cx.active, e)
    cand = candidates(cx, s, e, cls, nofx)
    out = {"end": e, "start": s, "cls": cls, "cand": cand, "nofx": nofx, "res": None}
    if len(cand) < 2:
        return out
    key = (pn.sid, str(pn.dates[e]))
    hsh = _hash(cx, s, e, cand)
    ra = _cached("dim_a", cx, e, hsh, lambda: dim_a.run_window(pn, cand, s, e, cx.active, cx.z, cx.scale, key))
    res = dict(ra)
    out["res"] = res
    lv = cx.levels
    rb = _cached("dim_b", cx, e, hsh, lambda: dim_b.window(pn, cx, out, lv, key)) if lv else None
    nP = len(res["pairs"])
    if rb and "p_B" in rb:
        res.update({k: rb[k] for k in ("p_B1", "p_B2", "p_B", "T_B1", "T_B2", "seller", "buyer")})
    else:
        res.update({"p_B1": np.full(nP, np.nan), "p_B2": np.full(nP, np.nan), "p_B": np.full(nP, np.nan)})
    out["b_days"] = (rb or {}).get("days", 0)
    out["b_skipped"] = (rb or {}).get("skipped_days", 0)
    if cres is not None:
        attach_c(out, pn, cres)
    res.update(window_fdr(res))
    return out


def attach_c(w: dict, pn, cres: dict) -> None:
    """維度 C（§7.4 拆好的兩個通道）對到配對上"""
    r = w["res"]
    pc = (cres or {}).get("pairs", {}).get(pn.sid)
    nP = len(r["pairs"])
    ps, po = np.full(nP, np.nan), np.full(nP, np.nan)
    if pc is not None:
        pos = {k: i for i, k in enumerate(pc["keys"])}
        m = len(pc["keys"])
        iu = np.triu_indices(m, 1)
        Ms, Mo = np.full((m, m), np.nan), np.full((m, m), np.nan)
        Ms[iu], Mo[iu] = pc["p_C_same"], pc["p_C_opp"]
        for k, (a, b) in enumerate(r["pairs"]):
            ka, kb = pn.branches[w["cand"][a]], pn.branches[w["cand"][b]]
            if ka in pos and kb in pos:
                x, y = sorted((pos[ka], pos[kb]))
                ps[k], po[k] = Ms[x, y], Mo[x, y]
    r["p_C_same"], r["p_C_opp"] = ps, po


def sig_edges(w: dict) -> tuple[list, list]:
    r = w["res"]
    c = np.array(w["cand"])
    pr = r["pairs"]
    same = [(int(c[a]), int(c[b])) for (a, b), x in zip(pr, r["sig_same"]) if x]
    opp = [(int(c[a]), int(c[b])) for (a, b), x in zip(pr, r["sig_opp"]) if x]
    return same, opp


def groups_for(cx: Context, w: dict) -> dict:
    pn = cx.pn
    L = cx.L
    e, s = w["end"], w["start"]
    same, opp = sig_edges(w)
    cg = cores(len(pn.branches), same, opp)
    vol = float(pn.volume[s:e + 1].sum())
    opp_set = {(min(a, b), max(a, b)) for a, b in opp}
    for co in cg["cores"]:
        g = co["members"]
        co["C_G"], co["inventory_G"] = group_cost(g, L["inventory"][:, e], L["avg_cost"][:, e], vol)
        net = (pn.B[g, s:e + 1] - pn.S[g, s:e + 1]).sum(axis=0)
        co["net_window"] = float(net.sum())
        co["net_20"] = float(net[-int(P.validation.naive_last_days):].sum())
        co["foreign_share"] = float(np.mean([pn.btype[i] == "foreign" for i in g]))
        co.update(group_pnl(cx, g, s, e, opp_set))
        co["absorption"] = absorption(cx, g, s, e)
    return cg


def group_pnl(cx: Context, g: list[int], s: int, e: int, opp_set: set) -> dict:
    """§8.4：已實現（含當沖）＋未實現；排除 U 批（不再整個排除分點）；虧損腿另需兩成員間有顯著反向邊"""
    pn, L, L3 = cx.pn, cx.L, cx.L_sens
    unreal = accounting.unrealized(L, pn.close[e], e)
    real = {i: float(L["realized"][i, s:e + 1].sum() + L["dt_pnl"][i, s:e + 1].sum()) for i in g}
    rp = sum(real.values())
    up = float(sum(unreal[i] for i in g))
    rp3 = float(sum(L3["realized"][i, s:e + 1].sum() + L3["dt_pnl"][i, s:e + 1].sum() for i in g))
    up3 = float(sum(accounting.unrealized(L3, pn.close[e], e)[i] for i in g))
    unc = float(sum(L["uncost_value"][i, s:e + 1].sum() for i in g))
    div = float(sum(L["div_memo"][i, s:e + 1].sum() for i in g))
    approx = bool(L["ca_approx"][s:e + 1].any())

    def avg(x, px):
        p = np.where(np.isfinite(px), px, pn.vwap[s:e + 1])
        return float((x * p).sum() / x.sum()) if x.sum() > 0 else None
    buy_v = {i: avg(pn.B[i, s:e + 1], pn.BP[i, s:e + 1]) for i in g}
    sell_v = {i: avg(pn.S[i, s:e + 1], pn.SP[i, s:e + 1]) for i in g}
    tol = float(P.revision.loss_leg_vwap_tol)
    loss = []
    if rp + up >= 0:
        for i in g:
            if real[i] < 0 and sell_v[i] and any(buy_v[r] and abs(buy_v[r] - sell_v[i]) <= tol * sell_v[i]
                                                 and (min(i, r), max(i, r)) in opp_set for r in g if r != i):
                loss.append(i)
    return {"realized": rp, "unrealized": up, "total": rp + up, "total_fee30": rp3 + up3, "uncost_value": unc,
            "dividend_memo": div, "ca_approx": approx, "loss_leg": loss}


def absorption(cx: Context, g: list[int], s: int, e: int) -> dict:
    pn = cx.pn
    rv = P.revision
    out = {}
    net = pn.B - pn.S
    for i in g:
        others = [r for r in g if r != i]
        num = den = 0.0
        for t in range(max(s, 1), e + 1):
            lo = max(0, t - int(P.behavior.rolling_days))
            hist = net[i, lo:t][cx.active[i, lo:t]]
            if not len(hist):
                continue
            ns = -net[i, t]
            if ns >= float(rv.b2_sell_mult) * np.median(np.abs(hist)) and ns >= float(rv.b2_sell_min_lots) * 1000 and pn.close[t] <= pn.close[t - 1]:
                num += max(0.0, float(net[others, t].sum()))
                den += ns
        if den > 0:
            out[i] = num / den
    return out


def naive_group(cx: Context, w: dict, nofx: bool) -> dict:
    pn, L = cx.pn, cx.L
    e, s = w["end"], w["start"]
    d = int(P.validation.naive_last_days)
    net20 = (pn.B[:, e - d + 1:e + 1] - pn.S[:, e - d + 1:e + 1]).sum(axis=1)
    ok = [i for i in np.argsort(-net20) if w["cls"][i] not in (features.DAYTRADE, features.OVERNIGHT) and net20[i] > 0
          and not (nofx and pn.btype[i] == "foreign")][: int(P.validation.naive_top)]
    C, inv = group_cost(ok, L["inventory"][:, e], L["avg_cost"][:, e], float(pn.volume[s:e + 1].sum()))
    return {"members": [int(i) for i in ok], "C_G": C, "inventory_G": inv, "net_20": float(net20[ok].sum())}


def pairs_frame(pn, w: dict) -> pd.DataFrame:
    r = w["res"]
    if r is None:
        return pd.DataFrame()
    c = np.array(w["cand"])
    a, b = c[r["pairs"][:, 0]], c[r["pairs"][:, 1]]
    nan = np.full(len(a), np.nan)
    br = np.array(pn.branches)
    return pd.DataFrame({"date_end": pn.dates[w["end"]], "ia": a, "ib": b, "branch_a": br[a], "branch_b": br[b], "obs": r["obs"],
                         "p_act": r["p_act"], "p_mag_pos": r["p_mag_pos"], "p_mag_neg": r["p_mag_neg"], "rho": r["rho"],
                         "p_A_same": r["p_A_same"], "p_A_opp": r["p_A_opp"], "p_B1": r["p_B1"], "p_B2": r["p_B2"], "p_B": r["p_B"],
                         "p_C_same": r.get("p_C_same", nan), "p_C_opp": r.get("p_C_opp", nan),
                         "seller": br[c[r["seller"]]] if "seller" in r else "", "buyer": br[c[r["buyer"]]] if "buyer" in r else "",
                         "p_same": r["p_same"], "p_opp": r["p_opp"], "q_same": r["q_same"], "q_opp": r["q_opp"],
                         "sig_same": r["sig_same"], "sig_opp": r["sig_opp"]})
