"""
§5.2、§6.1、§7.6、§8.1 共用 p 值工具
  hash_u          確定性亂數 U ∈ [0,1)：SHA-256(以 | 串接的鍵) 取前 64 位元 ÷ 2⁶⁴
  randomized_p    p = P(X > O) + U·P(X = O)（虛無下恰好連續均勻）
  Pool            分層池化虛無分數 → 隨機化經驗 p、條件 p（§6.1）、GPD 外推（§7.6：≥ 30 個超越值、下限 1/(10·N)）
  cct             Cauchy 組合（不截斷；只做數值保護 [1e-15, 1 − 1e-15]）
  bh              Benjamini–Hochberg q 值
  simes           §9.1
"""
from __future__ import annotations

import functools
import hashlib

import numpy as np
from scipy import stats

from .params import P

EPS = float(P.fusion.p_clip)


def clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def hash_u(*key) -> float:
    h = hashlib.sha256("|".join(str(k) for k in key).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def hash_u_many(prefix: tuple, a: np.ndarray, b: np.ndarray, tag: str) -> np.ndarray:
    """配對 (a, b) 的 U；鍵 = (prefix…, min, max, tag)"""
    return np.array([hash_u(*prefix, min(x, y), max(x, y), tag) for x, y in zip(a, b)])


def randomized_p(p_gt, p_eq, u) -> np.ndarray:
    return np.asarray(p_gt, float) + np.asarray(u, float) * np.asarray(p_eq, float)


def cct(ps, ws=None):
    """CCT(p1..pm) = 1/2 − arctan(Σ w_k tan[(1/2 − p_k)π]) / π；NaN = 缺失（剩下的平分權重）；全缺 → NaN"""
    ps = [np.asarray(p, float) for p in ps]
    shape = functools.reduce(np.broadcast_shapes, [p.shape for p in ps], ())
    Pm = np.stack([np.broadcast_to(p, shape) for p in ps])
    ok = np.isfinite(Pm)
    ws = np.ones(len(ps)) if ws is None else np.asarray(ws, float)
    W = np.where(ok, ws.reshape(-1, *([1] * len(shape))), 0.0)
    tot = W.sum(axis=0)
    W = np.divide(W, tot, out=np.zeros_like(W), where=tot > 0)
    Pc = clip(np.where(ok, Pm, 0.5))
    tanv = np.where(Pc < 1e-8, 1 / (np.pi * Pc), np.where(Pc > 1 - 1e-8, -1 / (np.pi * (1 - Pc)), np.tan((0.5 - Pc) * np.pi)))
    T = (W * tanv).sum(axis=0)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        p = np.where(T > 1e8, 1 / (np.pi * np.where(T > 1e8, T, 1)), 0.5 - np.arctan(T) / np.pi)
    p = np.where(tot > 0, p, np.nan)
    return clip(p) if np.ndim(p) else (float(clip(p)) if np.isfinite(p) else float("nan"))


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    m = len(x)
    if not m:
        return q
    o = np.argsort(x)
    r = np.minimum.accumulate((x[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    qq = np.empty(m)
    qq[o] = np.minimum(r, 1.0)
    q[ok] = qq
    return q


def simes(p) -> float:
    """§9.1：min_r M·p_(r)/r；NaN 不計入 M"""
    x = np.sort(np.asarray(p, float)[np.isfinite(p)])
    if not len(x):
        return float("nan")
    return float(min(1.0, np.min(len(x) * x / np.arange(1, len(x) + 1))))


# ─────────────── 分層池化（原 7.2：十分位配對 55 層）───────────────
def deciles(vol: np.ndarray) -> np.ndarray:
    if len(vol) == 0:
        return np.zeros(0, int)
    r = stats.rankdata(vol, method="average") / len(vol)
    return np.minimum((r * 10).astype(int), 9)


def stratum(da, db):
    da, db = np.asarray(da), np.asarray(db)
    return np.minimum(da, db) * 10 + np.maximum(da, db)


class Pool:
    def __init__(self):
        self.parts: dict[int, list] = {}
        self._s: dict[int, np.ndarray] = {}

    def add(self, strata, scores) -> None:
        strata, scores = np.asarray(strata), np.asarray(scores, float)
        ok = np.isfinite(scores)
        for s in np.unique(strata[ok]):
            self.parts.setdefault(int(s), []).append(scores[ok & (strata == s)].astype(np.float32))
        self._s.clear()

    def null(self, s: int) -> np.ndarray:
        if s not in self._s:
            v = self.parts.get(s)
            self._s[s] = np.sort(np.concatenate(v)).astype(np.float64) if v else np.zeros(0)
        return self._s[s]

    def pvalues(self, strata, obs, u, conditional: bool = False, min_nonzero: int = 0, gpd: bool = True):
        """右尾經驗 p（大＝顯著）。u：每個觀測的確定性亂數（§5.2 同分隨機化）。
        conditional（§6.1）：T = 0 → 缺失；虛無池中 > 0 的分數少於 min_nonzero → 缺失；
                              否則 p = (#{null > t} + U·(#{null = t} + 1)) / (1 + #{null > 0})
        一般：p = (#{null > t} + U·(#{null = t} + 1)) / (1 + N)
        回傳 (p, gpd_flag)"""
        strata, obs, u = np.asarray(strata), np.asarray(obs, float), np.asarray(u, float)
        out = np.full(len(obs), np.nan)
        flag = np.zeros(len(obs), bool)
        g = P.nulls
        for s in np.unique(strata):
            m = (strata == s) & np.isfinite(obs)
            if conditional:
                m &= obs > 0
            if not m.any():
                continue
            nul = self.null(int(s))
            if conditional:
                nul = nul[nul > 0]
                if len(nul) < int(min_nonzero):
                    continue
            N = len(nul)
            if N == 0:
                continue
            t = obs[m]
            gt = N - np.searchsorted(nul, t, side="right")
            eq = np.searchsorted(nul, t, side="right") - np.searchsorted(nul, t, side="left")
            p = (gt + u[m] * (eq + 1)) / (1 + N)
            fl = np.zeros(len(t), bool)
            if gpd and N < float(g.gpd_max_null):
                thr = float(np.quantile(nul, float(g.gpd_quantile)))
                exc = nul[nul > thr] - thr
                tail = t > thr
                if tail.any() and len(exc) >= int(g.gpd_min_exceed):
                    xi, _, sc = stats.genpareto.fit(exc, floc=0)
                    pt = (len(exc) / N) * stats.genpareto.sf(t[tail] - thr, xi, 0, sc)
                    p = p.copy()
                    p[tail] = np.maximum(pt, 1.0 / (10 * N))       # §7.6 下限
                    fl[tail] = True
            out[m] = p
            flag[m] = fl
        return np.where(np.isfinite(out), clip(np.nan_to_num(out, nan=0.5)), np.nan), flag
