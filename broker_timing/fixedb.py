"""
§4.4 診斷第 1 輪（§12.2）：Bartlett 核 HAC t 統計量的 fixed-b（Kiefer–Vogelsang）尾機率
  Kiefer & Vogelsang (2005, Econometric Theory 21:1130–1164)：b = M/T 固定時，
    t ⇒ Z / √Q_b，Z ~ N(0,1) 與 Q_b 獨立，Q_b 是 Brownian bridge 的 Bartlett 二次泛函。
  本實作用「有限 T 的同一個二次型」：分數 ψ 在虛無下為 i.i.d. 高斯時，OLS 正規方程使 Σψ̂ = 0（去平均），
    t = Z / √Q，Q = eᵀ P K P e / T，P = I − 11ᵀ/T，K_ij = max(0, 1 − |i−j|/M)，M = L + 1（Ω 的權重 1 − ℓ/(L+1)），
    e ~ N(0, I)。所以 Q = Σ μ_k χ²_1，μ_k = PKP/T 的特徵值；T → ∞、b = M/T 固定時就是 Kiefer–Vogelsang 的 fixed-b 極限。
  尾機率（c ≥ 0）：P(t > c) = ½ P(W > 0)，W = Z² − c² Σ μ_k χ²_k（常態變數的二次型）。
    以 Laplace 反轉 P(W > 0) = (1/π) ∫_0^∞ Re[ M_W(γ + iy) / (γ + iy) ] dy（0 < γ < ½）計算，γ 取鞍點 ŝ（K'_W(ŝ) = 0；ŝ < 0.05 時取 0.05）。
    沿鞍點的垂直線積分沒有 Imhof (1961) 公式在小尾機率的相消問題，相對誤差約 1e-10，完全確定（無亂數）。
    鞍點法參考 Kuonen (1999, Biometrika 86:929–935)；這裡不用 Lugannani–Rice 近似（尾端約高估 8～10%），而是直接數值積分。
  效率：每個 (T, L) 在 c 的網格上算一次 log p，以 PCHIP（保單調）內插；c ≤ 0 用對稱 P(t > c) = 1 − P(t > −c)。
  T > fixedb_max_grid 時以 m = fixedb_max_grid 個格點、M' = M·m/T 近似（b 不變；SPEC-GAP-timing）。
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import integrate, optimize
from scipy.interpolate import PchipInterpolator

from .params import TP

log = logging.getLogger("broker_scraper")
CACHE_DIR: Path | None = None          # run.py 設定；None = 只放記憶體
TABLE_VERSION = str(TP.fixedb_table)


def bandwidth(L: int) -> int:
    """Ω 的權重 1 − ℓ/(L+1) 對應 Bartlett 核的頻寬 M = L + 1（b = M/T ≈ L/T）"""
    return int(L) + 1


@lru_cache(maxsize=64)
def eigs(T: int, M: int) -> np.ndarray:
    m = min(int(T), int(TP.fixedb_max_grid))
    Mm = M * m / T
    i = np.arange(m)
    K = np.maximum(0.0, 1.0 - np.abs(np.subtract.outer(i, i)) / Mm)
    K -= K.mean(axis=0, keepdims=True)
    K -= K.mean(axis=1, keepdims=True)                       # P K P
    mu = np.linalg.eigvalsh(K) / m
    return mu[mu > 1e-13 * mu.max()]


def exact_sf(c: float, mu: np.ndarray) -> float:
    """P(Z/√Q > c)，c > 0；Q = Σ μ_k χ²_1"""
    if c <= 0:
        raise ValueError("c 必須 > 0")
    w = np.r_[1.0, -c * c * mu]
    lo, hi = 1.0 / (2.0 * w.min()), 0.5

    def kp(s):
        return float((w / (1.0 - 2.0 * s * w)).sum())
    eps = (hi - lo) * 1e-13
    s_hat = optimize.brentq(kp, lo + eps, hi - eps, xtol=1e-300, rtol=1e-15, maxiter=1000)
    g = max(s_hat, 0.05)
    k2 = float((2.0 * w * w / (1.0 - 2.0 * g * w) ** 2).sum())
    kg = float(-0.5 * np.log1p(-2.0 * g * w).sum())

    def f(y):
        s = g + 1j * y
        return float((np.exp(-0.5 * np.log(1.0 - 2.0 * s * w).sum() - kg) / s).real)
    sc = 1.0 / np.sqrt(k2)
    edges = np.r_[0.0, sc * np.array([1, 2, 4, 8, 16, 32, 64, 128, 512, 4096]), np.inf]
    v = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        r, _ = integrate.quad(f, a, b, epsabs=0.0, epsrel=1e-10, limit=500)
        v += r
    return float(0.5 * np.exp(kg) * v / np.pi)


def _grid_values(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs = list(np.round(np.arange(0.05, 6.0 + 1e-9, 0.05), 10))
    lp = [np.log(exact_sf(c, mu)) for c in cs]
    c = 6.0
    while lp[-1] > np.log(1e-20) and c < 400:
        c += 0.25 if c < 20 else 1.0
        cs.append(c)
        lp.append(np.log(max(exact_sf(c, mu), 1e-300)))
    return np.r_[0.0, cs], np.r_[np.log(0.5), lp]


@lru_cache(maxsize=64)
def table(T: int, L: int) -> PchipInterpolator:
    M = bandwidth(L)
    f = None
    if CACHE_DIR is not None:
        f = Path(CACHE_DIR) / f"{TABLE_VERSION}_T{T}_M{M}_g{int(TP.fixedb_max_grid)}.npz"
        if f.exists():
            try:
                z = np.load(f)
                return PchipInterpolator(z["c"], z["lp"], extrapolate=True)
            except Exception:  # noqa: BLE001
                pass
    c, lp = _grid_values(eigs(int(T), M))
    if f is not None:
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, c=c, lp=lp)
        except Exception:  # noqa: BLE001
            pass
    return PchipInterpolator(c, lp, extrapolate=True)


def sf(c: float, T: int, L: int | None = None) -> float:
    """fixed-b 單尾機率 P(t > c)；b = (L+1)/T"""
    L = int(TP.hac_lag if L is None else L)
    if not np.isfinite(c):
        return 0.0 if c > 0 else 1.0
    if c < 0:
        return 1.0 - sf(-c, T, L)
    tab = table(int(T), L)
    cmax = float(tab.x[-1])
    if c <= cmax:
        return float(np.exp(tab(c)))
    # 網格外（p < 1e-20）：以最後兩點的 log p 線性外插（只會更小，最後仍截在 p_floor）
    x0, x1 = tab.x[-2], tab.x[-1]
    y0, y1 = tab(x0), tab(x1)
    return float(np.exp(y1 + (y1 - y0) / (x1 - x0) * (c - x1)))
