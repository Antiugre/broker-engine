"""
第 7.2、7.3 節：同一分層內池化虛無分數 → Phipson–Smyth 經驗 p 值；樣本不足且觀測值在尾端時用 GPD 外推
分層：兩個分點在 (j, w) 內活躍度十分位數組成的無序對 (d_i, d_i')，共 55 層
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .params import P


def deciles(vol: np.ndarray) -> np.ndarray:
    """(j, w) 內各候選分點總成交量的十分位數（0～9）"""
    if len(vol) == 0:
        return np.zeros(0, int)
    r = stats.rankdata(vol, method="average") / len(vol)
    return np.minimum((r * 10).astype(int), 9)


def stratum(da: np.ndarray, db: np.ndarray) -> np.ndarray:
    lo, hi = np.minimum(da, db), np.maximum(da, db)
    return lo * 10 + hi


class Pool:
    """各分層的虛無分數（只收有定義的分數）"""

    def __init__(self):
        self.parts: dict[int, list[np.ndarray]] = {}
        self._sorted: dict[int, np.ndarray] = {}

    def add(self, strata: np.ndarray, scores: np.ndarray) -> None:
        ok = np.isfinite(scores)
        for s in np.unique(strata[ok]):
            self.parts.setdefault(int(s), []).append(scores[ok & (strata == s)].astype(np.float32))
        self._sorted.clear()

    def null(self, s: int) -> np.ndarray:
        if s not in self._sorted:
            v = self.parts.get(s)
            self._sorted[s] = np.sort(np.concatenate(v)) if v else np.zeros(0, np.float32)
        return self._sorted[s]

    def pvalues(self, strata: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, dict]:
        """經驗 p 值（7.2）＋必要時 GPD 外推（7.3）；回傳 p 與 GPD 診斷"""
        g = P.nulls
        out = np.full(len(obs), np.nan)
        diag = {}
        for s in np.unique(strata):
            m = (strata == s) & np.isfinite(obs)
            if not m.any():
                continue
            nul = self.null(int(s))
            N = len(nul)
            if N == 0:
                continue
            ge = N - np.searchsorted(nul, obs[m], side="left")
            p = (1 + ge) / (1 + N)
            if N < float(g.gpd_max_null):
                u = float(np.quantile(nul, float(g.gpd_quantile)))
                tail = obs[m] > u
                exc = nul[nul > u] - u
                if tail.any() and len(exc) >= int(g.gpd_min_exceed):
                    xi, _, sc = stats.genpareto.fit(exc, floc=0)
                    pt = (len(exc) / N) * stats.genpareto.sf(obs[m][tail] - u, xi, 0, sc)
                    pp = p.copy()
                    pp[tail] = np.maximum(pt, 1e-300)
                    p = pp
                    diag[int(s)] = {"n_null": N, "u": u, "xi": float(xi), "scale": float(sc),
                                    "xi_path": _xi_path(nul)}
            out[m] = p
        e = float(P.fusion.p_clip)
        return np.clip(out, e, 1 - e), diag


def _xi_path(nul: np.ndarray) -> list:
    """7.3 門檻穩定性：u 取第 97～99.5 百分位時的形狀參數 ξ"""
    res = []
    for q in (0.97, 0.98, 0.99, 0.995):
        u = float(np.quantile(nul, q))
        exc = nul[nul > u] - u
        if len(exc) >= 10:
            res.append([q, float(stats.genpareto.fit(exc, floc=0)[0])])
    return res
