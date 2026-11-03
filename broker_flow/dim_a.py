"""
6.1 維度 A：單股日頻共現
  A1 活躍共現：logistic（分點固定效果 + 當日因子，L2，C=10）→ 每日活躍機率 → 共同活躍天數的
     Poisson-binomial 精確右尾 p_act（動態規劃）
  A2 幅度相關：每個分點只用活躍日，把 z_net 對（報酬率、log 成交量）做 OLS 取殘差；
     共同活躍日 n ≥ 20 時 Spearman ρ 的單尾 p 值（t 近似，自由度 n − 2）
  p_A = CCT(p_act, p_mag)，各 1/2；p_mag 缺失時 p_A = p_act

SPEC-GAP：當日因子少了「振幅」（沒有最高、最低價）；因子不做標準化（規格未要求）
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .fusion import cct, clip
from .params import P


def day_factors(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """(日, 2)：|報酬率|、log 成交量（第一天報酬率記 0）"""
    ret = np.r_[0.0, close[1:] / close[:-1] - 1]
    return np.c_[np.abs(ret), np.log(np.maximum(volume, 1.0))], ret


def activity_prob(A: np.ndarray, F: np.ndarray) -> np.ndarray:
    """A: (候選, 日) bool；F: (日, k)。回傳 (候選, 日) 的活躍機率"""
    n, T = A.shape
    X = np.zeros((n * T, n + F.shape[1]))
    X[np.arange(n * T), np.repeat(np.arange(n), T)] = 1.0
    X[:, n:] = np.tile(F, (n, 1))
    y = A.reshape(-1).astype(int)
    if y.min() == y.max():
        return np.full((n, T), y.mean())
    m = LogisticRegression(C=float(P.detection.logistic_C), penalty="l2", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m.predict_proba(X)[:, 1].reshape(n, T)


def poisson_binomial_sf(q: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """q: (配對, T) 每日成功機率；obs: (配對,) 觀測次數 → P(X ≥ obs)，動態規劃精確計算"""
    m, T = q.shape
    dist = np.zeros((m, T + 1))
    dist[:, 0] = 1.0
    for t in range(T):
        qt = q[:, t:t + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - qt) + dist[:, :-1] * qt
        dist[:, 0] = dist[:, 0] * (1 - qt[:, 0])
    cum = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]      # cum[:, k] = P(X ≥ k)
    return cum[np.arange(m), np.clip(obs, 0, T).astype(int)]


def residuals(Z: np.ndarray, A: np.ndarray, ret: np.ndarray, F: np.ndarray) -> np.ndarray:
    """活躍日內中心化：每個分點只用自己的活躍日做 OLS，非活躍日為 NaN"""
    E = np.full(Z.shape, np.nan)
    X_all = np.c_[np.ones(len(ret)), ret, F[:, 1]]
    for i in range(Z.shape[0]):
        d = A[i]
        if d.sum() <= X_all.shape[1]:
            continue
        X, y = X_all[d], Z[i, d]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        E[i, d] = y - X @ beta
    return E


def p_magnitude(E: np.ndarray, A: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    nmin = int(P.detection.mag_min_common)
    out = np.full(len(pairs), np.nan)
    for k, (i, j) in enumerate(pairs):
        d = A[i] & A[j]
        n = int(d.sum())
        if n < nmin:
            continue
        rho = stats.spearmanr(E[i, d], E[j, d]).statistic
        if not np.isfinite(rho):
            continue
        rho = min(rho, 1 - 1e-12)
        t = rho * np.sqrt((n - 2) / (1 - rho * rho))
        out[k] = stats.t.sf(t, n - 2)
    return out


def run_window(A: np.ndarray, Z: np.ndarray, F: np.ndarray, ret: np.ndarray) -> dict:
    """一個 (股票, 窗口)：A、Z 為候選分點 × 窗口日；F、ret 為同一段的當日因子。回傳所有配對的 p_act、p_mag、p_A"""
    n = A.shape[0]
    p = activity_prob(A, F)
    ii, jj = np.triu_indices(n, 1)
    pairs = np.c_[ii, jj]
    obs = (A[ii] & A[jj]).sum(axis=1)
    p_act = clip(poisson_binomial_sf(p[ii] * p[jj], obs))
    E = residuals(Z, A, ret, F)
    p_mag = p_magnitude(E, A, pairs)
    p_mag = np.where(np.isfinite(p_mag), clip(np.nan_to_num(p_mag, nan=0.5)), np.nan)
    p_A = cct([p_act, p_mag], [0.5, 0.5])
    return {"pairs": pairs, "obs": obs, "p_act": p_act, "p_mag": p_mag, "p_A": p_A, "prob": p}
