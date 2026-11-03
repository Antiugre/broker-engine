"""
第 8 節 融合層：CCT、全局 BH-FDR、顯著配對圖的連通分量 → 集團、集團成本 C_G

本版只有維度 A 能計算（B 需要價位明細、C 需要跨股票資料），依 8.2 第 4 條：
無法計算的維度從合併中移除，剩下的維度平分權重 → p_meta = p_A
"""
from __future__ import annotations

import functools

import numpy as np

from .params import P


def clip(p):
    e = float(P.fusion.p_clip)
    return np.clip(np.asarray(p, float), e, 1 - e)


def cct(ps, ws=None) -> np.ndarray:
    """8.1 Cauchy 組合檢定。ps: list of arrays（NaN = 該維度缺失，剩下的平分權重）"""
    ps = [np.asarray(p, float) for p in ps]
    shape = functools.reduce(np.broadcast_shapes, [p.shape for p in ps], ())   # np.broadcast 最多 64 個參數
    P_ = np.stack([np.broadcast_to(p, shape) for p in ps])
    ok = np.isfinite(P_)
    if ws is None:
        ws = np.ones(len(ps))
    W = np.where(ok, np.asarray(ws, float).reshape(-1, *([1] * len(shape))), 0.0)
    tot = W.sum(axis=0)
    W = np.divide(W, tot, out=np.zeros_like(W), where=tot > 0)
    Pc = clip(np.where(ok, P_, 0.5))
    small = Pc < 1e-8
    tanv = np.where(small, 1 / (np.pi * Pc), np.tan((0.5 - Pc) * np.pi))
    T = (W * tanv).sum(axis=0)
    big = T > 1e8
    with np.errstate(over="ignore", divide="ignore"):
        p = np.where(big, 1 / (np.pi * np.where(big, T, 1)), 0.5 - np.arctan(T) / np.pi)
    p = np.where(tot > 0, p, np.nan)
    return clip(p) if np.ndim(p) else float(clip(p))


def bh(p) -> np.ndarray:
    """Benjamini–Hochberg q 值（NaN 不列入檢定族群）"""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    x = p[ok]
    m = len(x)
    if not m:
        return q
    o = np.argsort(x)
    r = x[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(m)
    qq[o] = np.minimum(r, 1.0)
    q[ok] = qq
    return q


def components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """8.5：連通分量（大小 ≥ 2 才是集團）；不用需要解析度參數的社群偵測"""
    parent = list(range(n))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = f(a), f(b)
        if ra != rb:
            parent[ra] = rb
    g: dict[int, list[int]] = {}
    for i in {x for e in edges for x in e}:
        g.setdefault(f(i), []).append(i)
    return sorted((sorted(v) for v in g.values() if len(v) >= 2), key=lambda v: (-len(v), v))


def group_cost(members, inv_end: np.ndarray, cost_end: np.ndarray, window_volume: float) -> tuple[float | None, float]:
    """8.5 C_G：窗口結束日，庫存 ≥ 窗口總成交量 0.5% 的成員，以庫存加權的移動平均成本"""
    thr = float(P.fusion.group_inventory_share) * window_volume
    m = [i for i in members if inv_end[i] >= thr and inv_end[i] > 0]
    invG = float(sum(inv_end[i] for i in members))
    if not m:
        return None, invG
    w = np.array([inv_end[i] for i in m])
    return float((w * np.array([cost_end[i] for i in m])).sum() / w.sum()), invG
