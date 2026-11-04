"""
§5 維度 A（v3）
A1  logit q_i,t = α_i + γ_t + δ_iᵀ x_t，x_t = std[|r_adj|, log V, amp]（窗口內標準化）
    α_i 不懲罰（SPEC-GAP-v3：數值上限 |α| ≤ alpha_bound）；γ_t L2 λ_γ = 1；δ_i L2 λ_δ = 1；停牌日不進模型
    O = 共同活躍天數；虛無 Poisson-binomial(q_i q_i')；p_act = P(X > O) + U·P(X = O)（§5.2 確定性隨機化）
A2  每個分點在自己的活躍日上把 z 對 (1, r_adj, log V, amp) 做 OLS 取殘差 e；共同活躍日 n ≥ 20 的配對算 Spearman ρ
    p_mag⁺（ρ > 0）：虛無＝每個分點各自循環平移（殘差、活躍日一起），200 次；
             SPEC-GAP-v3：規格的「k ∈ [10, n−10]」實作為「兩分點的相對位移 ∈ [10, W−10]」——各自位移相同時對齊沒被破壞，
             那個配對在該次排列不列入虛無（否則真正的協同配對會把自己的高相關帶進虛無池）
    p_mag⁻（ρ < 0）：虛無＝每天在同一活躍度十分位內置換候選分點的淨量 n，重算 z、活躍日、殘差與 ρ，200 次
    虛無分數依 55 層池化；經驗 p 以 §5.2 隨機化處理同分
    SPEC-GAP-v3（計算量）：相關係數用「每個分點在自己活躍日上的殘差秩→常態分數」，再算共同活躍日上的 Pearson 相關
                 （矩陣運算，全部配對、全部 200 次排列都算得起；觀測與虛無用同一個統計量，檢定仍然有效）。
                 與共同活躍日上重新排名的 Spearman 只差在排名的範圍（自己的活躍日 vs. 共同活躍日）
輸出 p_A_same = CCT(p_act, p_mag⁺)、p_A_opp = CCT(p_act, p_mag⁻)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .params import P
from .pvals import Pool, cct, clip, deciles, hash_u_many, randomized_p, stratum

NULL_PAIRS = 2000


def factors(pn, s: int, e: int) -> np.ndarray:
    """(日, 3)：|r_adj|、log V、amp，窗口內標準化"""
    x = np.c_[np.abs(pn.ret[s:e + 1]), np.log(np.maximum(pn.volume[s:e + 1], 1.0)), pn.amp[s:e + 1]]
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def activity_prob(A: np.ndarray, X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A (n, T) bool、X (T, k)、valid (T,) 非停牌。回傳 q (n, T)"""
    n, T = A.shape
    k = X.shape[1]
    lg, ld, ab = float(P.detection.lambda_gamma), float(P.detection.lambda_delta), float(P.detection.alpha_bound)
    Y = A.astype(float)
    V = valid.astype(float)[None, :]
    rate = np.clip((Y * V).sum(1) / max(V.sum(), 1), 1e-3, 1 - 1e-3)
    x0 = np.r_[np.log(rate / (1 - rate)), np.zeros(T), np.zeros(n * k)]

    def unpack(th):
        return th[:n], th[n:n + T], th[n + T:].reshape(n, k)

    def f(th):
        a, g, d = unpack(th)
        eta = a[:, None] + g[None, :] + d @ X.T
        q = 1 / (1 + np.exp(-eta))
        ll = (V * (Y * eta - np.logaddexp(0, eta))).sum()
        r = V * (q - Y)
        obj = -ll + 0.5 * lg * (g @ g) + 0.5 * ld * (d * d).sum()
        grad = np.r_[r.sum(1), r.sum(0) + lg * g, (r @ X + ld * d).ravel()]
        return obj, grad
    bounds = [(-ab, ab)] * n + [(None, None)] * (T + n * k)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
    a, g, d = unpack(res.x)
    return 1 / (1 + np.exp(-(a[:, None] + g[None, :] + d @ X.T)))


def poisson_binomial(q: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """q (配對, T) → (P(X > O), P(X = O))"""
    m, T = q.shape
    dist = np.zeros((m, T + 1))
    dist[:, 0] = 1.0
    for t in range(T):
        qt = q[:, t:t + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - qt) + dist[:, :-1] * qt
        dist[:, 0] *= (1 - qt[:, 0])
    o = np.clip(obs, 0, T).astype(int)
    eq = dist[np.arange(m), o]
    gt = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]           # gt[:, k] = P(X ≥ k)
    ge = gt[np.arange(m), o]
    return np.clip(ge - eq, 0, 1), eq


def residuals(Z: np.ndarray, A: np.ndarray, Xr: np.ndarray) -> np.ndarray:
    """每個分點只用自己的活躍日做 OLS（批次解 4×4 正規方程）；非活躍日 NaN"""
    W = A.astype(float)
    XtWX = np.einsum("nt,tk,tl->nkl", W, Xr, Xr)
    XtWy = np.einsum("nt,tk,nt->nk", W, Xr, Z)
    ok = A.sum(1) > Xr.shape[1]
    beta = np.zeros((Z.shape[0], Xr.shape[1]))
    if ok.any():
        beta[ok] = np.linalg.solve(XtWX[ok] + 1e-9 * np.eye(Xr.shape[1]), XtWy[ok][..., None])[..., 0]
    E = Z - beta @ Xr.T
    return np.where(A & ok[:, None], E, np.nan)


def spearman_pairs(E: np.ndarray, A: np.ndarray, ii: np.ndarray, jj: np.ndarray, nmin: int) -> np.ndarray:
    """共同活躍日上的 Spearman ρ（殘差連續、沒有同分）：ρ = 1 − 6Σd² / (c(c² − 1))；c < nmin → NaN"""
    out = np.full(len(ii), np.nan)
    for lo in range(0, len(ii), 20000):
        a, b = ii[lo:lo + 20000], jj[lo:lo + 20000]
        M = A[a] & A[b]
        c = M.sum(1)
        keep = c >= nmin
        if not keep.any():
            continue
        a, b, M, c = a[keep], b[keep], M[keep], c[keep]
        x = np.where(M, E[a], np.inf)
        y = np.where(M, E[b], np.inf)
        rx = np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1, kind="stable")
        ry = np.argsort(np.argsort(y, axis=1, kind="stable"), axis=1, kind="stable")
        d2 = np.where(M, (rx - ry).astype(float) ** 2, 0.0).sum(1)
        cc = c.astype(float)
        idx = np.nonzero(keep)[0] + lo
        out[idx] = 1 - 6 * d2 / (cc * (cc * cc - 1))
    return out


def normal_scores(E: np.ndarray, A: np.ndarray) -> np.ndarray:
    """每個分點在自己的活躍日上：殘差秩 → Φ⁻¹((r − 0.5)/m)；非活躍日 0"""
    from scipy.special import ndtri
    x = np.where(A, E, np.inf)
    r = np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1, kind="stable") + 1
    m = A.sum(1, keepdims=True)
    return np.where(A, ndtri((r - 0.5) / np.maximum(m, 1)), 0.0)


def corr_matrix(E: np.ndarray, A: np.ndarray, nmin: int) -> np.ndarray:
    """共同活躍日上的 Pearson 相關（常態分數），全部配對一次算；共同日 < nmin → NaN"""
    X = normal_scores(np.nan_to_num(E), A)
    Af = A.astype(float)
    cnt = Af @ Af.T
    sx = X @ Af.T                      # sx[i, j] = Σ_{i、j 都活躍} x_i
    sxx = (X * X) @ Af.T
    sxy = X @ X.T
    num = cnt * sxy - sx * sx.T
    den = np.sqrt(np.maximum(cnt * sxx - sx * sx, 0) * np.maximum(cnt * sxx.T - sx.T * sx.T, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((cnt >= nmin) & (den > 0), num / den, np.nan)
    return r


def run_window(pn, cand: list[int], s: int, e: int, active: np.ndarray, z: np.ndarray, scale: np.ndarray, key: tuple) -> dict:
    """一個 (股票, 窗口)。key＝(股票, 窗口結束日)：確定性亂數與排列種子"""
    d = P.detection
    c = np.array(cand)
    n = len(c)
    A = active[c, s:e + 1]
    Z = z[c, s:e + 1]
    valid = ~pn.suspended[s:e + 1]
    X = factors(pn, s, e)
    q = activity_prob(A, X, valid)
    ii, jj = np.triu_indices(n, 1)
    br = np.array(pn.branches)[c]
    obs = (A[ii] & A[jj]).sum(1)
    qq = np.where(valid[None, :], q[ii] * q[jj], 0.0)
    pgt, peq = poisson_binomial(qq, obs)
    p_act = clip(randomized_p(pgt, peq, hash_u_many(key, br[ii], br[jj], "A1")))
    # ── A2 ──
    Xr = np.c_[np.ones(e - s + 1), pn.ret[s:e + 1], np.log(np.maximum(pn.volume[s:e + 1], 1.0)), pn.amp[s:e + 1]]
    nmin = int(d.mag_min_common)
    E = residuals(Z, A, Xr)
    rho = corr_matrix(E, A, nmin)[ii, jj]
    vol = (pn.B[c, s:e + 1] + pn.S[c, s:e + 1]).sum(1)
    dec = deciles(vol)
    strata = stratum(dec[ii], dec[jj])
    seed = int.from_bytes(str(key).encode()[:8].ljust(8, b"\0"), "little") % (2 ** 32)
    rng = np.random.default_rng(seed)
    pos_pool, neg_pool = Pool(), Pool()
    W = e - s + 1
    nperm = int(d.mag_null_perms)
    smin = int(d.mag_shift_min)
    net = (pn.B - pn.S)[c, s:e + 1]
    sc = scale[c, s:e + 1]
    thr = np.maximum(float(d.active_min_shares), float(d.active_scale_mult) * np.nan_to_num(sc, nan=np.inf))
    groups = [np.flatnonzero(dec == k) for k in range(10)]
    defined = np.isfinite(rho)
    if defined.any():
        for _ in range(nperm):
            # p_mag⁺：循環平移
            k = rng.integers(0, W, n)
            idx = (np.arange(W)[None, :] - k[:, None]) % W
            As = np.take_along_axis(A, idx, 1)
            Es = np.take_along_axis(np.nan_to_num(E), idx, 1)
            rel = (k[ii] - k[jj]) % W                    # 兩分點的相對位移必須在 [10, W−10]，否則對齊沒被破壞（不列入虛無）
            r1 = np.where((rel >= smin) & (rel <= W - smin), corr_matrix(Es, As, nmin)[ii, jj], np.nan)
            pos_pool.add(strata, r1)
            # p_mag⁻：每天在同一十分位內置換淨量
            perm = np.tile(np.arange(n), (W, 1))
            for g in groups:
                if len(g) > 1:
                    perm[:, g] = g[rng.random((W, len(g))).argsort(axis=1)]
            npm = np.take_along_axis(net.T, perm, 1).T            # 第 t 天分點 i 拿到 perm[t, i] 的淨量
            with np.errstate(invalid="ignore", divide="ignore"):
                zp = np.where(np.isfinite(sc) & (sc > 0), npm / sc, 0.0)
            Ap = (np.abs(npm) >= thr) & (npm != 0) & valid[None, :]
            Ep = residuals(zp, Ap, Xr)
            r2 = corr_matrix(Ep, Ap, nmin)[ii, jj]
            neg_pool.add(strata, -r2)
    u1 = hash_u_many(key, br[ii], br[jj], "A2+")
    u2 = hash_u_many(key, br[ii], br[jj], "A2-")
    p_pos, _ = pos_pool.pvalues(strata, np.where(defined, rho, np.nan), u1, gpd=True)
    p_neg, _ = neg_pool.pvalues(strata, np.where(defined, -rho, np.nan), u2, gpd=True)
    return {"pairs": np.c_[ii, jj], "obs": obs, "p_act": p_act, "p_mag_pos": p_pos, "p_mag_neg": p_neg, "rho": rho,
            "p_A_same": cct([p_act, p_pos]), "p_A_opp": cct([p_act, p_neg]), "dec": dec}
