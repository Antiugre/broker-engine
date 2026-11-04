"""
§4.2 反應函數與殘差
  只在活躍日上估計 ñ_t = μ + φᵀ x_t + ε_t
  目標函數 (1/N_act) Σ (ñ − μ − φᵀx)² + λ‖φ‖²，λ = 0.01，μ 不懲罰 → sklearn Ridge(alpha = λ · N_act, fit_intercept=True)
  x_t 在活躍日上標準化（平均 0、標準差 1）
  使用到最後一日 T 的所有活躍日，ε̂ 也算到 T；非活躍日 ε̂ = 0
  E+（§4.5）：x_t 再加入 max(r_t, 0)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .params import TP

N_BASE = 7          # prep.features 的前 7 欄是主檢定特徵，第 8 欄是 E+ 的 max(r_t, 0)


def fit(ntil: np.ndarray, active: np.ndarray, X: np.ndarray, eplus: bool = False, lam: float | None = None) -> dict:
    """ntil：(T,) ñ；active：(T,) bool；X：(T, 8) prep 特徵（已對齊到面板日期）。
    回傳 eps（(T,)，非活躍日 0）、alpha、n_act、coef、intercept、n_dropped（活躍日但特徵缺值 → ε̂ 記 0，SPEC-GAP-timing）"""
    lam = float(TP.ridge_lambda if lam is None else lam)
    cols = X[:, :N_BASE + (1 if eplus else 0)]
    T = len(ntil)
    eps = np.zeros(T)
    use = active & np.isfinite(cols).all(axis=1) & np.isfinite(ntil)
    n_drop = int((active & ~use).sum())
    N = int(use.sum())
    if N < 2:
        return {"eps": eps, "alpha": lam * N, "n_act": N, "coef": None, "intercept": np.nan, "n_dropped": n_drop, "model": None}
    Z = cols[use]
    mu, sd = Z.mean(0), Z.std(0)
    # SPEC-GAP-timing：活躍日上變異為 0 的特徵，標準化後整欄記 0（不影響其餘係數）
    Zs = np.where(sd > 0, (Z - mu) / np.where(sd > 0, sd, 1.0), 0.0)
    alpha = lam * N
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Zs, ntil[use])
    eps[use] = ntil[use] - m.predict(Zs)
    return {"eps": eps, "alpha": alpha, "n_act": N, "coef": m.coef_.copy(), "intercept": float(m.intercept_),
            "n_dropped": n_drop, "model": m, "mu": mu, "sd": sd}
