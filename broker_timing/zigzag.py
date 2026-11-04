"""
§6.2 Zigzag 波段計數（金額門檻）
  I_t = Σ_{τ≤t} n^cb_τ（期初 0）；P̄ = 活躍日 C_t / g_t 的中位數；A = max(5,000,000 / P̄, 5 · median{|n^cb| : 活躍日})
  只走活躍日：
    1. 起點在第一個活躍日之前：hi = lo = 0、方向未定
    2. 方向未定：先更新 hi、lo，再判斷 hi − I ≥ A → 第一段向下；否則 I − lo ≥ A → 第一段向上（極值 = I）
    3. 方向為上：創新高更新極值；I ≤ 極值 − A → 段數 + 1、改為向下、極值 = I
    4. 方向為下：對稱
    5. 完整循環數 = ⌊段數 / 2⌋
"""
from __future__ import annotations

import numpy as np

from .params import TP


def amplitude(ncb: np.ndarray, close: np.ndarray, g: np.ndarray, active: np.ndarray) -> float:
    if not active.any():
        return np.inf
    px = close[active] / g[active]
    px = px[np.isfinite(px) & (px > 0)]
    pbar = float(np.median(px)) if len(px) else np.nan
    a1 = float(TP.zigzag_amount_ntd) / pbar if np.isfinite(pbar) and pbar > 0 else 0.0
    a2 = float(TP.zigzag_median_mult) * float(np.median(np.abs(ncb[active])))
    return max(a1, a2)


def count(ncb: np.ndarray, active: np.ndarray, A: float, approx: np.ndarray | None = None) -> dict:
    """回傳 segments（每段 {start, end, dir}；start/end 為交易日索引，第一段的 start 可能是 −1 = 第一個活躍日之前）、
    n_segments、cycles（⌊段數/2⌋）、cycles_ca_approx（跨越 ca_approx 事件的完整循環數）"""
    ncb = np.asarray(ncb, float)
    active = np.asarray(active, bool)
    I = np.cumsum(np.where(active, ncb, 0.0))
    idx = np.flatnonzero(active)
    segs: list[dict] = []
    if not len(idx) or not np.isfinite(A) or A <= 0:
        return {"segments": segs, "n_segments": 0, "cycles": 0, "A": A, "cycles_ca_approx": 0}
    first = int(idx[0]) - 1
    hi = lo = 0.0
    hi_t = lo_t = first
    d = 0                  # 0 未定、+1 上、−1 下
    ext = 0.0
    ext_t = first
    for t in idx:
        x = I[t]
        if d == 0:
            if x > hi:
                hi, hi_t = x, int(t)
            if x < lo:
                lo, lo_t = x, int(t)
            if hi - x >= A:
                d, ext, ext_t = -1, x, int(t)
                segs.append({"start": hi_t, "end": int(t), "dir": -1})
            elif x - lo >= A:
                d, ext, ext_t = 1, x, int(t)
                segs.append({"start": lo_t, "end": int(t), "dir": 1})
        elif d == 1:
            if x > ext:
                ext, ext_t = x, int(t)
                segs[-1]["end"] = ext_t
            elif x <= ext - A:
                segs.append({"start": ext_t, "end": int(t), "dir": -1})
                d, ext, ext_t = -1, x, int(t)
        else:
            if x < ext:
                ext, ext_t = x, int(t)
                segs[-1]["end"] = ext_t
            elif x >= ext + A:
                segs.append({"start": ext_t, "end": int(t), "dir": 1})
                d, ext, ext_t = 1, x, int(t)
    n = len(segs)
    cyc = n // 2
    ca = 0
    if approx is not None and np.asarray(approx).any():
        ap = np.flatnonzero(approx)
        for k in range(cyc):
            s, e = max(segs[2 * k]["start"], 0), segs[2 * k + 1]["end"]
            if ((ap > s) & (ap <= e)).any():
                ca += 1
    return {"segments": segs, "n_segments": n, "cycles": cyc, "A": A, "cycles_ca_approx": ca}


def run(ncb: np.ndarray, close: np.ndarray, g: np.ndarray, active: np.ndarray, approx: np.ndarray | None = None) -> dict:
    return count(ncb, active, amplitude(ncb, close, g, active), approx)
