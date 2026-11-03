"""
第 7.4 節 校正檢查、第 9.3 節 合成注入（含 C2ST 真實性檢查）
只用 hold-out 之前的窗口（不傳 allow_holdout）

SPEC-GAP
  - 7.4「完全打亂的合成虛無資料」：每個分點各自把窗口內的日子隨機重排（活躍與 z_net 一起移動），
    破壞分點之間的同步，保留各分點自己的活躍率與量的分布
  - 9.3 拆分方式：A 的每個交易日以機率 ρ 由 K 個新代碼平分（共同活躍），否則整天歸給隨機一個新代碼；
    「價位分散」需要價位明細、「偽裝 η」需要其他股票 → 本版只跑 η = 0、全部價位
  - 9.3 合成注入後只在該窗口內做 BH（全局 BH 要重跑所有窗口）
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import dim_a, features
from .data_layer import Panel
from .detect import Context, run_window
from .fusion import bh, components
from .params import P


def calibration(cx: Context, ends: list[int], seed: int = 0, max_windows: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    W = int(P.detection.window_days)
    use = [e for e in ends if e < cx.pn.holdout_start]
    use = use[-max_windows:]
    ps = {"p_act": [], "p_mag": [], "p_A": []}
    for e in use:
        w = run_window(cx, e)
        if not w["cand"] or len(w["cand"]) < 2:
            continue
        s = e - W + 1
        A = cx.active[w["cand"], s:e + 1].copy()
        Z = cx.z[w["cand"], s:e + 1].copy()
        for i in range(len(A)):
            o = rng.permutation(A.shape[1])
            A[i], Z[i] = A[i, o], Z[i, o]
        r = dim_a.run_window(A, Z, cx.F[s:e + 1], cx.ret[s:e + 1])
        for k in ps:
            v = r[k][np.isfinite(r[k])]
            ps[k].append(v)
    rep = {}
    lo, hi = P.validation.calibration.frac01_range
    for k, v in ps.items():
        v = np.concatenate(v) if v else np.array([])
        if not len(v):
            rep[k] = None
            continue
        ks = stats.kstest(v, "uniform").pvalue
        f = float((v < 0.01).mean())
        rep[k] = {"n": int(len(v)), "ks_p": float(ks), "frac_lt_01": f,
                  "pass": bool(ks > float(P.validation.calibration.ks_min_p) and lo <= f <= hi),
                  "hist": np.histogram(v, bins=10, range=(0, 1))[0].tolist()}
    return {"windows": len(use), "dims": rep}


def _subpanel(pn: Panel, lo: int, hi: int) -> Panel:
    return Panel(pn.sid + "~syn", pn.dates[lo:hi + 1], list(pn.branches), list(pn.names), list(pn.btype),
                 pn.B[:, lo:hi + 1].copy(), pn.S[:, lo:hi + 1].copy(), pn.close[lo:hi + 1], pn.vwap[lo:hi + 1],
                 pn.volume[lo:hi + 1], hi - lo + 1)


def inject(pn: Panel, a: int, K: int, rho: float, rng) -> tuple[Panel, list[int]]:
    """把分點 a 的成交重新標記給 K 個新代碼並刪除 a（只是重新標記，總量守恆）"""
    T = pn.B.shape[1]
    nb, ns = np.zeros((K, T)), np.zeros((K, T))
    for t in np.nonzero(pn.B[a] + pn.S[a] > 0)[0]:
        if K == 1 or rng.random() < rho:
            nb[:, t], ns[:, t] = pn.B[a, t] / K, pn.S[a, t] / K
        else:
            k = rng.integers(K)
            nb[k, t], ns[k, t] = pn.B[a, t], pn.S[a, t]
    keep = [i for i in range(len(pn.branches)) if i != a]
    q = Panel(pn.sid, pn.dates, [pn.branches[i] for i in keep] + [f"SYN{k}" for k in range(K)],
              [pn.names[i] for i in keep] + [f"合成{k + 1}" for k in range(K)],
              [pn.btype[i] for i in keep] + ["brokerage"] * K,
              np.vstack([pn.B[keep], nb]), np.vstack([pn.S[keep], ns]), pn.close, pn.vwap, pn.volume, pn.holdout_start)
    assert np.allclose(q.B.sum(axis=0), pn.B.sum(axis=0)) and np.allclose(q.S.sum(axis=0), pn.S.sum(axis=0))   # 守恆
    return q, list(range(len(keep), len(keep) + K))


def _marginal(pn: Panel, active: np.ndarray, idx: list[int], s: int, e: int) -> np.ndarray:
    """C2ST 只用無條件邊際特徵：總成交量、活躍日比例、活躍日平均單日量、買方比例"""
    B, S, A = pn.B[idx, s:e + 1], pn.S[idx, s:e + 1], active[idx, s:e + 1]
    g = B + S
    tot = g.sum(axis=1)
    na = A.sum(axis=1)
    return np.c_[np.log1p(tot), na / A.shape[1], np.log1p(np.where(na > 0, (g * A).sum(axis=1) / np.maximum(na, 1), 0)),
                 np.divide(B.sum(axis=1), tot, out=np.full(len(idx), 0.5), where=tot > 0)]


def c2st(syn: np.ndarray, real: np.ndarray, seed: int = 0) -> float | None:
    if len(syn) < 5 or len(real) < 5:
        return None
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    X = np.vstack([syn, real])
    y = np.r_[np.ones(len(syn)), np.zeros(len(real))]
    folds = min(5, int(min(len(syn), len(real))))
    pr = cross_val_predict(GradientBoostingClassifier(random_state=seed), X, y,
                           cv=StratifiedKFold(folds, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, pr))


def synthetic(cx: Context, ends: list[int], seed: int = 0, reps: int | None = None, Ks=None, rhos=None) -> dict:
    pn = cx.pn
    sp = P.validation.synthetic
    reps = int(reps or sp.reps)
    Ks, rhos = Ks or list(sp.K), rhos or list(sp.rho)
    W, q_thr = int(P.detection.window_days), float(P.fusion.fdr_q)
    rng = np.random.default_rng(seed)
    use = [e for e in ends if e < pn.holdout_start]
    # 候選的「真實大型波段分點」：各窗口中分類為波段型、窗口成交量最大的前幾名（非外資、非自營）
    pool = []
    for e in use:
        cls = features.classify(pn, cx.active, e)
        s = e - W + 1
        g = (pn.B[:, s:e + 1] + pn.S[:, s:e + 1]).sum(axis=1)
        top = [i for i in np.argsort(-g) if cls[i] == features.SWING and pn.btype[i] == "brokerage"][:5]
        pool += [(e, i) for i in top]
    if not pool:
        return {"cells": [], "note": "沒有可用的波段分點"}
    cells = []
    for K in Ks:
        for rho in rhos:
            rec, prec, jac, syn_feats, real_feats = [], [], [], [], []
            for r in range(reps):
                e, a = pool[(r * 7 + K * 3 + int(rho * 10)) % len(pool)]
                lo = max(0, e - W - int(P.behavior.rolling_days) + 1)
                sub = _subpanel(pn, lo, e)
                q, codes = inject(sub, a, K, rho, rng)
                c2 = Context(q)
                w = run_window(c2, len(q.dates) - 1)
                s2 = len(q.dates) - W
                if w["res"] is None:
                    rec.append(0.0); prec.append(0.0); jac.append(0.0)
                    continue
                cand = np.array(w["cand"])
                pr = w["res"]["pairs"]
                ia, ib = cand[pr[:, 0]], cand[pr[:, 1]]
                qv = bh(w["res"]["p_meta"])
                inj = np.isin(ia, codes) & np.isin(ib, codes)
                m = K * (K - 1) // 2
                rec.append(float((inj & (qv <= q_thr)).sum() / m))
                top = np.argsort(w["res"]["p_meta"])[:m]
                prec.append(float(inj[top].mean()))
                sig = qv <= q_thr
                comps = components(len(q.branches), list(zip(ia[sig], ib[sig])))
                best = max((len(set(c) & set(codes)) / len(set(c) | set(codes)) for c in comps), default=0.0)
                jac.append(float(best))
                syn_feats.append(_marginal(q, c2.active, codes, s2, len(q.dates) - 1))
                # 同一十分位數規模的真實分點
                real = [i for i in range(len(q.branches) - K) if w["cls"][i] in (features.SWING, features.UNCLASSIFIED)]
                if real:
                    gt = (q.B[:, s2:] + q.S[:, s2:]).sum(axis=1)
                    dec = np.quantile(gt[real], np.linspace(0, 1, 11))
                    lo_d, hi_d = np.searchsorted(dec, gt[codes].min()) - 1, np.searchsorted(dec, gt[codes].max())
                    lo_v, hi_v = dec[max(0, lo_d)], dec[min(10, hi_d)]
                    real = [i for i in real if lo_v <= gt[i] <= hi_v]
                    if real:
                        real_feats.append(_marginal(q, c2.active, real, s2, len(q.dates) - 1))
            auc = c2st(np.vstack(syn_feats), np.vstack(real_feats), seed) if syn_feats and real_feats else None
            cells.append({"K": K, "rho": rho, "reps": reps, "recall": float(np.mean(rec)), "precision": float(np.mean(prec)),
                          "jaccard": float(np.mean(jac)), "c2st_auc": auc,
                          "accepted": auc is not None and auc <= float(P.validation.c2st_auc_max)})
    return {"cells": cells, "pool": len(pool)}
