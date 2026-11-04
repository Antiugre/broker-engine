"""
§4.3、§4.4 檢定迴歸、堆疊式 Newey–West、有效區塊數自由度、Cheverud–Nyholt 多期間合併（完全確定，不含亂數）
  r^ex_{t→t+h} = a_h + θ_h x_t + u_{t,h}，H₁: θ_h > 0
  x_c = x − mean(x)；ψ_t = (x_c,t û_{t,h})_h；Γ_ℓ = (1/T) Σ_{t>ℓ} ψ_t ψ_{t−ℓ}ᵀ
  Ω = Γ_0 + Σ_{ℓ=1}^{L} (1 − ℓ/(L+1)) (Γ_ℓ + Γ_ℓᵀ)，L = 40；V = T Ω / (Σ x_c²)²
  G_eff = 樣本從第一日起切成不重疊 20 日區塊（最後不足 20 日也算一塊）中含活躍日的區塊數（§6.1 門檻 G_eff ≥ 8 照舊）
  p_min：hac_inference = fixed_b（診斷第 1 輪，§12.2；預設）→ Bartlett fixed-b 尾機率，b = (L+1)/T（fixedb.py）；
         hac_inference = t_geff（v1.1.2 原文）→ t.sf(T_max, df = G_eff − 1)
  M_eff = K − (1/K) Σ (λ_k − 1)²（K = 4 時即 4 − ¼Σ(λ−1)²），λ 截在 [0, K]、M_eff 截在 [1, K]
  p^E = min(1, M_eff · p_min)，截在 [1e-15, 1]；h* = argmax_h t_h
第 2 輪（CHANGELOG 1.1.2+r2；hac_inference = reverse_hw，預設）：reverse_test()
  反向迴歸（Hodrick 型）：Σ_t x_c[t] r^ex_{t→t+h} = Σ_s w_{s,h} e_s，w_{s,h} = Σ_{t=s−h}^{s−1} x_c[t]
  z_{h,s} = w_{s,h}(e_s − ē)；Ω = Σ z zᵀ + Σ_{ℓ=1}^{5} (1 − ℓ/6)(Γ_ℓ + Γ_ℓᵀ)；t_h = num_h / √Ω_hh；標準常態
  多期間：Hunter–Worsley 上界 p = 4Φ̄(c) − Σ_{MST(R)} Φ̄₂(c, c; ρ)（Owen's T，完全確定）
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import owens_t

from .params import TP


def ols_slopes(Y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """逐欄 OLS：回傳 (theta (K,), a (K,), resid (N, K), Σx_c²)"""
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        K = Y.shape[1]
        return np.full(K, np.nan), np.full(K, np.nan), np.full_like(Y, np.nan), 0.0
    theta = (xc @ (Y - Y.mean(0))) / sxx
    a = Y.mean(0) - theta * x.mean()
    U = Y - a[None, :] - x[:, None] * theta[None, :]
    return theta, a, U, sxx


def newey_west(psi: np.ndarray, L: int) -> np.ndarray:
    """Ω = Γ_0 + Σ_{ℓ=1}^{L} (1 − ℓ/(L+1)) (Γ_ℓ + Γ_ℓᵀ)，Γ_ℓ = (1/T) Σ_{t=ℓ+1}^{T} ψ_t ψ_{t−ℓ}ᵀ"""
    T = psi.shape[0]
    om = psi.T @ psi / T
    for l in range(1, min(L, T - 1) + 1):
        G = psi[l:].T @ psi[:-l] / T
        om = om + (1 - l / (L + 1)) * (G + G.T)
    return om


def geff(active: np.ndarray, block: int | None = None) -> int:
    block = int(TP.block_len_geff if block is None else block)
    a = np.asarray(active, bool)
    if not len(a):
        return 0
    nb = -(-len(a) // block)
    pad = np.zeros(nb * block, bool)
    pad[:len(a)] = a
    return int(pad.reshape(nb, block).any(axis=1).sum())


def nearest_corr(R: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """最近的正定相關矩陣（特徵值截在 eps 以上後重建、再把對角線縮放回 1）"""
    R = (R + R.T) / 2
    w, Q = np.linalg.eigh(R)
    w = np.maximum(w, eps)
    S = (Q * w) @ Q.T
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    np.fill_diagonal(S, 1.0)
    return S


def m_eff(R: np.ndarray) -> float:
    """Cheverud–Nyholt：M_eff = K − (1/K) Σ (λ − 1)²；R 非正定 → 先取最近正定；λ 截在 [0, K]，結果截在 [1, K]"""
    R = np.asarray(R, float)
    K = R.shape[0]
    if not np.isfinite(R).all():
        return float(K)          # SPEC-GAP-timing：相關矩陣無法計算時採最保守的 M_eff = K
    R = (R + R.T) / 2
    lam = np.linalg.eigvalsh(R)
    if lam.min() <= 0:
        lam = np.linalg.eigvalsh(nearest_corr(R))
    lam = np.clip(lam, 0.0, float(K))
    m = K - float(((lam - 1) ** 2).sum()) / K
    return float(np.clip(m, 1.0, float(K)))


def tail(tmax: float, N: int, df: int, L: int | None = None, method: str | None = None) -> float:
    """單變量尾機率 P(t > T_max)"""
    method = str(TP.hac_inference if method is None else method)
    if method == "fixed_b":
        from . import fixedb
        return fixedb.sf(tmax, int(N), L)
    if method == "t_geff":
        return float(stats.t.sf(tmax, df))
    raise ValueError(method)


def combine(t: np.ndarray, N: int, df: int, meff: float, L: int | None = None, method: str | None = None) -> tuple[float, float]:
    """(p_min, p^E = min(1, M_eff · p_min) 截在 [1e-15, 1])。
    t 無法計算 → p = 1；t_geff 方法 df < 1 → p = 1（SPEC-GAP-timing）。fixed_b 不用 df（被 fixed-b 取代）"""
    floor = float(TP.p_floor)
    method = str(TP.hac_inference if method is None else method)
    tt = np.asarray(t, float)
    if not np.isfinite(tt).any() or (method == "t_geff" and df < 1) or N < 3:
        return 1.0, 1.0
    tmax = float(np.nanmax(tt))
    pmin = tail(tmax, N, df, L, method)
    p = min(1.0, meff * pmin)
    return pmin, float(np.clip(p, floor, 1.0))


def test(Y: np.ndarray, x: np.ndarray, active: np.ndarray, L: int | None = None, horizons: list[int] | None = None,
         pvalue: bool = True, method: str | None = None) -> dict:
    """主檢定（§4.3、§4.4）。Y：(N, K) 前瞻超額報酬（樣本列）；x：(N,) ε̂ 或 ñ；active：(N,) 活躍日"""
    L = int(TP.hac_lag if L is None else L)
    if method is None and str(TP.hac_inference) not in ("fixed_b", "t_geff"):
        method = "t_geff"            # 第 2 輪起主檢定用 reverse_test()；這裡只剩原 NW 路徑（滾動窗口描述、對照）
    Y = np.asarray(Y, float)
    x = np.asarray(x, float)
    N, K = Y.shape
    hz = horizons or list(range(K))
    G = geff(active)
    df = G - 1
    out = {"N": N, "G_eff": G, "df": df, "inference": str(TP.hac_inference if method is None else method), "b": (L + 1) / N if N else np.nan, "theta": np.full(K, np.nan), "se": np.full(K, np.nan), "t": np.full(K, np.nan),
           "R": np.full((K, K), np.nan), "M_eff": float(K), "p_min": 1.0, "p": 1.0, "T_max": np.nan, "h_star": None, "h_star_idx": None}
    theta, a, U, sxx = ols_slopes(Y, x)
    if sxx <= 0 or N < 3:
        return out
    xc = x - x.mean()
    psi = xc[:, None] * U
    om = newey_west(psi, L)
    V = N * om / sxx ** 2
    dv = np.diag(V)
    se = np.sqrt(np.where(dv > 0, dv, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = theta / se
        R = V / np.outer(se, se)
    me = m_eff(R)
    pmin, p = combine(t, N, df, me, L, method) if pvalue else (np.nan, np.nan)
    k = int(np.nanargmax(t)) if np.isfinite(t).any() else None
    out.update(theta=theta, se=se, t=t, R=R, M_eff=me, p_min=pmin, p=p, a=a,
               T_max=float(np.nanmax(t)) if k is not None else np.nan, h_star_idx=k, h_star=hz[k] if k is not None else None)
    return out


def single(y: np.ndarray, x: np.ndarray, L: int | None = None) -> tuple[float, float]:
    """單一期間的 θ̂ 與 HAC t（§7 滾動窗口描述用，不輸出 p）"""
    r = test(np.asarray(y, float)[:, None], x, np.asarray(x) != 0, L, pvalue=False)
    return float(r["theta"][0]), float(r["t"][0])


# ───────────────────────── 第 2 輪：反向迴歸 + Hunter–Worsley ─────────────────────────
def bvn_upper(c: float, r: float) -> float:
    """P(Z₁ > c, Z₂ > c)，相關 r（Owen's T，解析計算）"""
    r = float(np.clip(r, -0.999999, 1.0))
    return max(0.0, float(stats.norm.sf(c) - 2 * owens_t(c, np.sqrt((1 - r) / (1 + r)))))


def hunter_worsley(c: float, R: np.ndarray) -> float:
    """P(max_k Z_k > c) 的 Hunter–Worsley 上界：K Φ̄(c) − Σ_{(a,b) ∈ 最大生成樹} Φ̄₂(c, c; R_ab)，截在 [Φ̄(c), 1] 與 [p_floor, 1]"""
    R = np.asarray(R, float)
    K = R.shape[0]
    if not np.isfinite(c):
        return 1.0
    if not np.isfinite(R).all():
        p = K * float(stats.norm.sf(c))              # SPEC-GAP-timing：相關矩陣無法計算 → Bonferroni（最保守）
    else:
        tree, edges = [0], []
        while len(tree) < K:                         # Prim：以相關係數為權重的最大生成樹
            a, b = max(((a, b) for a in tree for b in range(K) if b not in tree), key=lambda ab: R[ab])
            edges.append((a, b)); tree.append(b)
        p = K * float(stats.norm.sf(c)) - sum(bvn_upper(c, R[a, b]) for a, b in edges)
    p = max(p, float(stats.norm.sf(c)))
    return float(np.clip(p, float(TP.p_floor), 1.0))


def reverse_test(x: np.ndarray, sample: np.ndarray, e: np.ndarray, active: np.ndarray,
                 horizons: list[int] | None = None, L: int | None = None) -> dict:
    """主檢定（第 2 輪 §4.4）。全部在時間軸上：x：(N,) ε̂ 或 ñ；sample：(N,) bool 第二階段樣本日；
    e：(≥ N,) 單日超額對數報酬（e[s] 為第 s 日，r^ex_{t→t+h} = Σ_{s=t+1}^{t+h} e[s]）；active：(N,) 活躍日。
    回傳欄位與 test() 相同（另有 num）；df 不再用於 p（保留顯示）、M_eff 只供參考"""
    hz = [int(h) for h in (horizons or TP.horizons)]
    L = int(TP.score_hac_lag if L is None else L)
    x = np.asarray(x, float)
    S = np.asarray(sample, bool)
    e = np.asarray(e, float)
    N, K = len(x), len(hz)
    idx = np.flatnonzero(S)
    G = geff(np.asarray(active, bool)[S]) if len(idx) else 0
    out = {"N": int(len(idx)), "G_eff": G, "df": G - 1, "inference": "reverse_hw", "L": L, "theta": np.full(K, np.nan),
           "se": np.full(K, np.nan), "t": np.full(K, np.nan), "num": np.full(K, np.nan), "R": np.full((K, K), np.nan),
           "M_eff": float(K), "p_min": 1.0, "p": 1.0, "T_max": np.nan, "h_star": None, "h_star_idx": None}
    if len(idx) < 3:
        return out
    xc = np.zeros(N)
    xc[idx] = x[idx] - x[idx].mean()
    sxx = float(xc @ xc)
    if sxx <= 0:
        return out
    hmax = max(hz)
    lo, hi = int(idx[0]) + 1, min(int(idx[-1]) + hmax, len(e) - 1)     # 用得到的日子 s ∈ [min S + 1, max S + h_max]
    ew = e[lo:hi + 1]
    ok = np.isfinite(ew)
    eb = float(ew[ok].mean()) if ok.any() else 0.0
    ec = np.where(ok, ew - eb, 0.0)                  # 樣本日的前瞻期間都有值；缺值只會出現在 w = 0 的日子
    cx = np.r_[0.0, np.cumsum(xc)]
    s = np.arange(lo, hi + 1)
    W = np.array([cx[np.clip(s, 0, N)] - cx[np.clip(s - h, 0, N)] for h in hz])      # w_{s,h} = Σ_{t=s−h}^{s−1} x_c[t]
    Z = W * ec
    num = Z.sum(axis=1)
    om = Z @ Z.T
    for l in range(1, min(L, Z.shape[1] - 1) + 1):
        G_l = Z[:, l:] @ Z[:, :-l].T
        om = om + (1 - l / (L + 1)) * (G_l + G_l.T)
    dv = np.diag(om)
    se = np.sqrt(np.where(dv > 0, dv, np.nan))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = num / se
        R = om / np.outer(se, se)
    theta = num / sxx
    if not np.isfinite(t).any():
        out.update(theta=theta, num=num)
        return out
    k = int(np.nanargmax(t))
    c = float(t[k])
    p = hunter_worsley(c, R)
    out.update(theta=theta, se=se / sxx, t=t, num=num, R=R, M_eff=m_eff(R), p_min=float(np.clip(stats.norm.sf(c), TP.p_floor, 1.0)),
               p=p, T_max=c, h_star_idx=k, h_star=hz[k])
    return out
