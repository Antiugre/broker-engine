"""
§4.5 控制自身後續交易版（EI），鎖定主檢定的最強期間 h*
  F⁺_t = Σ_{τ=t+1}^{t+h*} max(ε̂_τ, 0)、F⁻_t = Σ_{τ=t+1}^{t+h*} min(ε̂_τ, 0)（ε̂ 用第一階段算到 T 的殘差）
  r^ex_{t→t+h*} = a + θ^EI ε̂_t + κ⁺ F⁺_t + κ⁻ F⁻_t + u（最小範數最小平方法 np.linalg.lstsq(X, y, rcond=None)）
  Δ = θ^EI − 0.5 θ^E_{h*}；ρ = θ^EI / θ^E_{h*}（只供顯示）
  移動區塊 bootstrap：整列 (r^ex, ε̂, F⁺, F⁻) 以長度 40 的區塊重抽 B = 999 次；種子 = 穩定雜湊(stock, branch, "EI")
  退化重抽（Σ(ε̂* − mean)² = 0）剔除、不補抽；剔除比例 > 10% → bootstrap 不穩定；分位數只在保留的重抽上計算
  顯示用 ρ 區間：只用 θ^E* > 0 的重抽；剔除比例 > 10% 標不穩定
"""
from __future__ import annotations

import hashlib

import numpy as np

from .params import TP


def seed_for(stock: str, branch: str, tag: str = "EI") -> int:
    """穩定雜湊（不用 Python hash()，跨行程、跨版本一致）"""
    h = hashlib.sha256(f"{stock}\x1f{branch}\x1f{tag}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def flows(eps_full: np.ndarray, sample_idx: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """F⁺、F⁻（第一階段殘差算到 T，所以 t = T − 40 時 [T−39, T] 都可用）"""
    e = np.asarray(eps_full, float)
    cp = np.r_[0.0, np.cumsum(np.maximum(e, 0))]
    cm = np.r_[0.0, np.cumsum(np.minimum(e, 0))]
    T = len(e)
    s = np.asarray(sample_idx, int)
    hi = np.minimum(s + h, T - 1)            # τ ∈ [t+1, t+h]
    return cp[hi + 1] - cp[s + 1], cm[hi + 1] - cm[s + 1]


def slope(y: np.ndarray, x: np.ndarray) -> float:
    xc = x - x.mean()
    sxx = float(xc @ xc)
    return float(xc @ (y - y.mean()) / sxx) if sxx > 0 else np.nan


def ei_fit(y, e, fp, fm) -> np.ndarray:
    X = np.c_[np.ones(len(y)), e, fp, fm]
    return np.linalg.lstsq(X, y, rcond=None)[0]           # (a, θ^EI, κ⁺, κ⁻)


def block_indices(N: int, block: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """移動區塊 bootstrap 的索引 (B, N)：起點均勻取自 [0, N − block]，接起來截到 N"""
    block = min(block, N)
    nb = -(-N // block)
    starts = rng.integers(0, N - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, nb * block)[:, :N]
    return idx


def run(y: np.ndarray, e: np.ndarray, fp: np.ndarray, fm: np.ndarray, stock: str, branch: str,
        B: int | None = None, block: int | None = None) -> dict:
    """y：樣本列的 r^ex_{t→t+h*}；e：樣本列的 ε̂_t；fp、fm：F⁺、F⁻"""
    B = int(TP.ei_bootstrap_reps if B is None else B)
    block = int(TP.ei_bootstrap_block if block is None else block)
    c = float(TP.rho_threshold)
    lvl = float(TP.ei_ci_level)
    qlo, qhi = (1 - lvl) / 2, 1 - (1 - lvl) / 2
    y, e, fp, fm = (np.asarray(a, float) for a in (y, e, fp, fm))
    N = len(y)
    thE = slope(y, e)
    coef = ei_fit(y, e, fp, fm)
    thEI, kp, km = float(coef[1]), float(coef[2]), float(coef[3])
    delta = thEI - c * thE
    rng = np.random.default_rng(seed_for(stock, branch))
    idx = block_indices(N, block, B, rng)
    dE = np.full(B, np.nan)
    dEI = np.full(B, np.nan)
    for b in range(B):
        ii = idx[b]
        eb = e[ii]
        if np.ptp(eb) == 0:            # Σ(ε̂* − mean)² = 0：θ^E* 沒有定義 → 剔除、不補抽
            continue
        yb = y[ii]
        dE[b] = slope(yb, eb)
        dEI[b] = ei_fit(yb, eb, fp[ii], fm[ii])[1]
    keep = np.isfinite(dE) & np.isfinite(dEI)
    disc = 1.0 - keep.mean()
    D = dEI[keep] - c * dE[keep]
    lo, hi = (float(np.quantile(D, qlo)), float(np.quantile(D, qhi))) if len(D) else (np.nan, np.nan)
    unstable = bool(disc > float(TP.ei_max_discard_frac)) or not len(D)
    pos = keep & (dE > 0)
    rho_drop = 1.0 - pos.mean()
    rr = dEI[pos] / dE[pos]
    rlo, rhi = (float(np.quantile(rr, qlo)), float(np.quantile(rr, qhi))) if len(rr) else (np.nan, np.nan)
    return {"theta_E": thE, "theta_EI": thEI, "kappa_plus": kp, "kappa_minus": km, "delta": delta,
            "delta_lo": lo, "delta_hi": hi, "discard_frac": float(disc), "unstable": unstable,
            "rho": thEI / thE if np.isfinite(thE) and thE != 0 else np.nan, "rho_lo": rlo, "rho_hi": rhi,
            "rho_drop_frac": float(rho_drop), "rho_unstable": bool(rho_drop > float(TP.rho_display_max_drop)),
            "B": B, "block": block, "N": N}
