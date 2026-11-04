"""
§7 維度 C（v3）：全市場跨股結構，分工計算（每一次擬合都是獨立的雲端工作）
  張量 X：(2 × 分點數) × 股票 × 週；i⁺ 格＝該週正向極端日數、i⁻ 格＝負向極端日數；排除隔日沖型
  v3.1 極端日（取代 |z| ≥ 2；§4.1 改尺度後原門檻未重新校準，屬規格錯誤，修正理由不來自驗證結果）：
    正向：n > 0 且 n ≥ max(Q₀.₉, 0.005·V_j,t, 10,000 股)；負向對稱（|n|）
    Q₀.₉＝分點 i 在股票 j 前 120 個交易日（不含當日）|n| > 0 的日子的 |n| 第 90 百分位數；非零日 < 20 → 不算極端
  v3.1 擬合前先數張量非零格數：超過 max_tensor_nnz（500 萬）就停止並回報，不自行調整門檻
  v3.1 背景項 E 只在非零格上計算；乘法更新分塊計算，不一次建出「非零格數 × 秩」的矩陣
  §7.2 背景分量：X ~ Poisson(E + Σ_r λ_r u_r v_r w_r)，E_a,j,w = X_a,j,· X_·,j,w / X_·,j,·（不參與學習；虛無重擬時由平移後資料重算）
  §7.3 虛無與觀測同一程序（初始值數、穩定性篩選相同）。依 §7.3「計算時間不夠時同時降低」：inits、stable_min 由 params 指定，兩者一起用
  §7.4 S_CP 只計在 (j, w) 兩個角色都有極端日（≥ 1 天）的組合；(++)(−−) → 同向通道，(+−)(−+) → 反向通道；S_Fr 同樣拆兩通道
  §7.5 m_w＝窗口內在最近一次擬合之後的週數比例；出現門檻 Σ_已知週 w_r ≥ 0.05(1 − m_w)；m_w > 0.5 → p_C 缺失
  §7.6 經驗 p 以 §5.2 隨機化處理同分；GPD 至少 30 個超越值；外推下限 1/(10·N)
  §7.7 結果檔名含參數版本、宇宙（股票池）指紋
SPEC-GAP-v3
  ・股票池＝雲端每晚計算範圍內有價格的股票（§2.3 的前 800 檔宇宙待輔助資料齊全後切換）
  ・候選分點不含「自我成交事件 ≥ 3」的補入規則（維度 C 不讀價位明細）
"""
from __future__ import annotations

import hashlib
import heapq
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse

from . import features
from .params import P
from .pvals import Pool, cct, deciles, hash_u, stratum

log = logging.getLogger("broker_scraper")
ROLE_PAIRS = ((0, 0, "same"), (1, 1, "same"), (1, 0, "opp"), (0, 1, "opp"))   # (a 角色, b 角色, 通道)；0 = 買方 +、1 = 賣方 −


def week_index(dates) -> np.ndarray:
    d = np.asarray(dates).astype("datetime64[D]")
    return (d - np.datetime64("1970-01-05")).astype(int) // 7


# ───────────────────────── 全市場資料（只留張量與窗口需要的彙總）─────────────────────────
def overnight_mask(pn, active, t: int) -> np.ndarray:
    """§4.1 隔日沖型（向量化，同 features.classify 的判定）"""
    b = P.behavior
    w = int(b.rolling_days)
    lo = max(0, t - w + 1)
    B, S, A = pn.B[:, lo:t + 1], pn.S[:, lo:t + 1], active[:, lo:t + 1]
    net = B - S
    L = net.shape[1]
    okt = np.array([x not in ("foreign", "proprietary", "gov_bank") for x in pn.btype]) & (A.sum(1) >= int(b.min_active_days))
    pos = np.where(A & (net > 0), net, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(pos, axis=1)
    if L < 4:
        return np.zeros(len(net), bool)
    S3 = np.zeros_like(S)
    for k in (1, 2, 3):
        S3[:, :-k] += S[:, k:]
    ev = (net > 0) & (net >= float(b.overnight_event_mult) * med[:, None])
    ev[:, L - 3:] = False
    with np.errstate(all="ignore"):
        ratio = np.where(ev, S3 / np.where(net > 0, net, 1), np.nan)
        rmed = np.nanmedian(ratio, axis=1)
    return okt & (ev.sum(1) >= int(b.overnight_min_events)) & (np.nan_to_num(rmed) >= float(b.overnight_ratio))


def extreme_days(pn) -> tuple[np.ndarray, np.ndarray]:
    """v3.1 極端日（分點 × 日 bool）：正向、負向"""
    import pandas as pd
    c = P.dim_c
    n = pn.B - pn.S
    a = np.abs(n)
    q = pd.DataFrame(np.where(a > 0, a, np.nan).T).rolling(int(c.ext_lookback), min_periods=int(c.ext_min_nonzero)) \
        .quantile(float(c.ext_quantile)).shift(1).to_numpy().T                     # 不含當日
    floor = np.maximum(float(c.ext_vol_frac) * pn.volume, float(c.ext_min_shares))[None, :]
    thr = np.fmax(q, floor)
    ok = np.isfinite(q) & ~pn.suspended[None, :]
    return ok & (n > 0) & (n >= thr), ok & (n < 0) & (a >= thr)


def build_market(sids: list[str], ends: list[str]) -> dict:
    """每檔：極端日週計數（張量用）＋每個窗口結束日的候選、十分位、極端日數、Fraudar 邊"""
    from .data_layer import build_panel
    W = int(P.detection.window_days)
    fe = float(P.dim_c.fraudar_edge)
    keys: dict[str, int] = {}
    rows, info, ovn = [], {}, {}
    used = []
    for j, sid in enumerate(sids):
        try:
            pn = build_panel(sid)
        except Exception as ex:  # noqa: BLE001
            log.warning("維度 C 全市場：%s 略過（%s）", sid, ex)
            continue
        if len(pn.dates) < int(P.accounting.warmup_days) + W:
            continue
        jj = len(used)
        used.append(sid)
        gi = np.array([keys.setdefault(b, len(keys)) for b in pn.branches])
        _, active, z = features.scale_and_active(pn)
        xp, xn = extreme_days(pn)
        wk = week_index(pn.dates)
        for role, hit in ((0, xp), (1, xn)):
            ii, tt = np.nonzero(hit)
            if len(ii):
                key = np.unique(np.c_[gi[ii] * 2 + role, wk[tt]], axis=0, return_counts=True)
                rows.append(np.c_[key[0][:, 0], np.full(len(key[1]), jj), key[0][:, 1], key[1]])
        for end in ends:
            e = int(np.searchsorted(pn.dates, end, side="right")) - 1
            if e < int(P.accounting.warmup_days) + W - 1 or str(pn.dates[e]) != end:
                continue
            s = e - W + 1
            on = overnight_mask(pn, active, e)
            ovn[(jj, end)] = set(gi[on].tolist())
            na = active[:, s:e + 1].sum(1)
            cand = np.flatnonzero(~on & (na >= int(P.detection.min_candidate_active)))
            if len(cand) < 2:
                continue
            g = (pn.B[cand, s:e + 1] + pn.S[cand, s:e + 1]).sum(1)
            cz = z[:, s:e + 1].sum(1)
            fr = [(int(gi[i]) * 2, jj) for i in np.flatnonzero(~on & (cz >= fe))] + \
                 [(int(gi[i]) * 2 + 1, jj) for i in np.flatnonzero(~on & (cz <= -fe))]
            info[(jj, end)] = {"cand": gi[cand].astype(np.int32), "dec": deciles(g).astype(np.int8),
                               "foreign": np.array([pn.btype[i] == "foreign" for i in cand]),
                               "ext_pos": xp[cand, s:e + 1].sum(1).astype(np.int16), "ext_neg": xn[cand, s:e + 1].sum(1).astype(np.int16),
                               "weeks": np.unique(wk[s:e + 1]).astype(np.int32), "fr_edges": fr}
        log.info("維度 C 全市場 %d/%d：%s（%d 個分點）", j + 1, len(sids), sid, len(pn.branches))
    tab = np.vstack(rows).astype(np.int64) if rows else np.zeros((0, 4), np.int64)
    fp = hashlib.sha256(("|".join(used)).encode()).hexdigest()[:12]
    return {"sids": used, "keys": list(keys), "tab": tab, "info": info, "ovn": ovn, "ends": ends, "universe": fp,
            "params_version": P.version}


# ───────────────────────── CP-APR（含固定背景）─────────────────────────
def tensor(mk: dict, fw: int, shift: np.ndarray | None = None):
    """擬合週 fw 的張量：座標 (角色, 股票, 週 0..span−1)、值、背景 E；shift＝每個分點的循環平移量（兩個角色相同）"""
    span = int(P.dim_c.fit_weeks)
    w0 = fw - span + 1
    t = mk["tab"]
    m = (t[:, 2] >= w0) & (t[:, 2] <= fw)
    t = t[m]
    # 排除擬合日前最近一個窗口結束日分類為隔日沖的分點
    fw_date = str(np.datetime64("1970-01-05") + np.timedelta64(int(fw) * 7 + 4, "D"))
    keep = np.ones(len(t), bool)
    ends = [e for e in mk["ends"] if e <= fw_date]
    if ends:
        last = ends[-1]
        for j in np.unique(t[:, 1]):
            ov = mk["ovn"].get((int(j), last))
            if ov:
                sel = t[:, 1] == j
                keep[sel] = ~np.isin(t[sel, 0] // 2, list(ov))
    t = t[keep]
    wl = t[:, 2] - w0
    if shift is not None:
        wl = (wl + shift[t[:, 0] // 2]) % span
    idx = np.c_[t[:, 0], t[:, 1], wl].astype(np.int64)
    val = t[:, 3].astype(float)
    shape = (2 * len(mk["keys"]), len(mk["sids"]), span)
    return idx, val, shape


def background(idx: np.ndarray, val: np.ndarray, shape: tuple, at: np.ndarray | None = None) -> np.ndarray:
    """E_a,j,w = X_a,j,· X_·,j,w / X_·,j,·（在 at 座標上；預設＝非零格）"""
    at = idx if at is None else at
    aj = sparse.coo_matrix((val, (idx[:, 0], idx[:, 1])), shape=shape[:2]).tocsr()
    jw = np.zeros((shape[1], shape[2]))
    np.add.at(jw, (idx[:, 1], idx[:, 2]), val)
    jt = jw.sum(1)
    a = np.asarray(aj[at[:, 0], at[:, 1]]).ravel()
    return a * jw[at[:, 1], at[:, 2]] / np.where(jt[at[:, 1]] > 0, jt[at[:, 1]], 1)


def cp_apr(idx, val, E, shape, R, rng, mask_idx=None, mask_E=None, max_iter: int = 300, tol: float = 1e-4):
    F = [rng.random((n, R)) + 0.1 for n in shape]
    Sm = None
    if mask_idx is not None and len(mask_idx):
        Sm = [sparse.csr_matrix((np.ones(len(mask_idx)), (mask_idx[:, m], np.arange(len(mask_idx)))), shape=(shape[m], len(mask_idx))) for m in range(3)]
    prev = -np.inf
    ll = prev
    nnz = len(val)
    ch = int(P.dim_c.chunk_rows)
    for _ in range(max_iter):
        for m in range(3):
            a, b = [x for x in range(3) if x != m]
            Phi = np.zeros((shape[m], R))
            for lo in range(0, nnz, ch):                  # 分塊：不建出「非零格數 × 秩」的整個矩陣
                sl = slice(lo, lo + ch)
                ia, ib, im = idx[sl, a], idx[sl, b], idx[sl, m]
                KR = F[a][ia] * F[b][ib]
                model = np.einsum("nr,nr->n", F[m][im], KR)
                w = val[sl] / np.maximum(E[sl] + model, 1e-300)
                Phi += sparse.csr_matrix((np.ones(len(im)), (im, np.arange(len(im)))), shape=(shape[m], len(im))) @ (KR * w[:, None])
            den = np.outer(np.ones(shape[m]), F[a].sum(0) * F[b].sum(0))
            if Sm is not None:
                den = den - Sm[m] @ (F[a][mask_idx[:, a]] * F[b][mask_idx[:, b]])
            F[m] = F[m] * Phi / np.maximum(den, 1e-300)
        model = np.concatenate([np.einsum("nr,nr,nr->n", F[0][idx[lo:lo + ch, 0]], F[1][idx[lo:lo + ch, 1]], F[2][idx[lo:lo + ch, 2]])
                                for lo in range(0, nnz, ch)]) if nnz else np.zeros(0)
        total = float((F[0].sum(0) * F[1].sum(0) * F[2].sum(0)).sum())
        if Sm is not None:
            total -= float(np.einsum("nr,nr,nr->", F[0][mask_idx[:, 0]], F[1][mask_idx[:, 1]], F[2][mask_idx[:, 2]]))
        ll = float((val * np.log(np.maximum(E + model, 1e-300))).sum() - total - E.sum())
        if abs(ll - prev) <= tol * max(1.0, abs(ll)):
            break
        prev = ll
    return F, ll


def normalize(F):
    s = [f.sum(0) for f in F]
    return [f / np.maximum(sv, 1e-300) for f, sv in zip(F, s)]


def fms(a, b) -> np.ndarray:
    out = np.ones((a[0].shape[1], b[0].shape[1]))
    for x, y in zip(a, b):
        xn = x / np.maximum(np.linalg.norm(x, axis=0), 1e-300)
        yn = y / np.maximum(np.linalg.norm(y, axis=0), 1e-300)
        out *= np.abs(xn.T @ yn)
    return out


def stable_fit(idx, val, E, shape, R, seed: int):
    c = P.dim_c
    rng = np.random.default_rng(seed)
    fits = [cp_apr(idx, val, E, shape, R, rng) for _ in range(int(c.inits))]
    best = max(range(len(fits)), key=lambda k: fits[k][1])
    Fb = normalize(fits[best][0])
    cnt = np.zeros(R, int)
    for F, _ in fits:
        cnt += fms(Fb, normalize(F)).max(axis=1) >= float(c.fms_min)
    return [f.astype(np.float32) for f in Fb], np.nonzero(cnt >= int(c.stable_min))[0]


def heldout_ll(idx, val, shape, R, seed) -> float:
    rng = np.random.default_rng(seed)
    nnz = len(val)
    k = max(1, int(0.1 * nnz))
    pick = rng.choice(nnz, k, replace=False)
    keep = np.ones(nnz, bool); keep[pick] = False
    occ = set(map(tuple, idx.tolist()))
    zeros = []
    while len(zeros) < k:
        cc = tuple(int(rng.integers(n)) for n in shape)
        if cc not in occ:
            zeros.append(cc)
    mask = np.vstack([idx[pick], np.array(zeros)])
    mval = np.r_[val[pick], np.zeros(len(zeros))]
    Ek = background(idx[keep], val[keep], shape)
    Em = background(idx[keep], val[keep], shape, at=mask)
    F, _ = cp_apr(idx[keep], val[keep], Ek, shape, R, rng, mask_idx=mask)
    m = Em + np.einsum("nr,nr,nr->n", F[0][mask[:, 0]], F[1][mask[:, 1]], F[2][mask[:, 2]])
    return float((mval * np.log(np.maximum(m, 1e-300)) - m).sum())


def fit_weeks(mk: dict) -> list[int]:
    c = P.dim_c
    wk = sorted({int(w) for w in mk["tab"][:, 2]}) if len(mk["tab"]) else []
    if not wk:
        return []
    allw = list(range(wk[0], wk[-1] + 1))
    span, every = int(c.fit_weeks), int(c.refit_weeks)
    return [allw[k] for k in range(span - 1, len(allw), every)]


def fit_for(mk: dict, end: str) -> int | None:
    w = int(week_index([end])[0])
    ok = [f for f in fit_weeks(mk) if f <= w]
    return ok[-1] if ok else None


def null_shift(mk: dict, fw: int, b: int) -> np.ndarray:
    rng = np.random.default_rng(int(hash_u("C-null", fw, b) * 2 ** 32))
    return rng.integers(0, int(P.dim_c.fit_weeks), len(mk["keys"]))


def fit_task(mk: dict, fw: int, b: int, R: int):
    """b = −1：觀測；b = 0..null_cp−1：虛無（循環平移）"""
    idx, val, shape = tensor(mk, fw, None if b < 0 else null_shift(mk, fw, b))
    if len(val) < R:
        return None
    E = background(idx, val, shape)
    return stable_fit(idx, val, E, shape, R, seed=int(hash_u("C-fit", fw, b) * 2 ** 31))


# ───────────────────────── Fraudar ─────────────────────────
def fraudar_blocks(edges, max_blocks: int):
    E = set(edges)
    blocks = []
    for _ in range(max_blocks):
        if not E:
            break
        adjL: dict = {}
        adjR: dict = {}
        for i, j in E:
            adjL.setdefault(i, []).append(j)
            adjR.setdefault(j, []).append(i)
        wcol = {j: 1.0 / np.log(len(v) + 5) for j, v in adjR.items()}
        pri = {("L", i): sum(wcol[j] for j in v) for i, v in adjL.items()}
        pri.update({("R", j): wcol[j] * len(v) for j, v in adjR.items()})
        alive = set(pri)
        f = sum(wcol[j] for _, j in E)
        heap = [(p, k) for k, p in pri.items()]
        heapq.heapify(heap)
        best, best_k, order = f / len(alive), 0, []
        while alive:
            p, k = heapq.heappop(heap)
            if k not in alive or p != pri[k]:
                continue
            alive.discard(k)
            order.append(k)
            f -= pri[k]
            if k[0] == "L":
                for j in adjL[k[1]]:
                    if ("R", j) in alive:
                        pri[("R", j)] -= wcol[j]; heapq.heappush(heap, (pri[("R", j)], ("R", j)))
            else:
                for i in adjR[k[1]]:
                    if ("L", i) in alive:
                        pri[("L", i)] -= wcol[k[1]]; heapq.heappush(heap, (pri[("L", i)], ("L", i)))
            if alive and f / len(alive) > best:
                best, best_k = f / len(alive), len(order)
        members = order[best_k:]
        Ls = [k[1] for k in members if k[0] == "L"]
        Rs = {k[1] for k in members if k[0] == "R"}
        if not Ls or not Rs:
            break
        tau = {i: (r / (len(Ls) - 1) if len(Ls) > 1 else 1.0) for r, i in enumerate(Ls)}
        blocks.append({"left": tau, "right": Rs})
        E = {(i, j) for i, j in E if not (i in tau and j in Rs)}
    return blocks


def curveball(edges, rng):
    rows: dict = {}
    for i, j in edges:
        rows.setdefault(i, set()).add(j)
    ks = list(rows)
    if len(ks) < 2:
        return list(edges)
    for _ in range(5 * len(ks)):
        a, b = rng.choice(len(ks), 2, replace=False)
        A, B = rows[ks[a]], rows[ks[b]]
        ua, ub = list(A - B), list(B - A)
        if not ua and not ub:
            continue
        pool = ua + ub
        rng.shuffle(pool)
        common = A & B
        rows[ks[a]] = common | set(pool[:len(ua)])
        rows[ks[b]] = common | set(pool[len(ua):])
    return [(i, j) for i, v in rows.items() for j in v]


def s_fraudar(blocks, j: int, cand_g: np.ndarray, conn: set) -> tuple[np.ndarray, np.ndarray]:
    n = len(cand_g)
    iu = np.triu_indices(n, 1)
    same = np.zeros(len(iu[0]))
    opp = np.zeros(len(iu[0]))
    for bl in blocks:
        if j not in bl["right"]:
            continue
        tau = bl["left"]
        t0 = np.array([tau.get(2 * g, np.nan) if 2 * g in conn else np.nan for g in cand_g])
        t1 = np.array([tau.get(2 * g + 1, np.nan) if 2 * g + 1 in conn else np.nan for g in cand_g])
        T = (t0, t1)
        for ra, rb, ch in ROLE_PAIRS:
            v = np.fmin(T[ra][iu[0]], T[rb][iu[1]])
            v = np.where(np.isfinite(T[ra][iu[0]]) & np.isfinite(T[rb][iu[1]]), v, 0.0)
            if ch == "same":
                same = np.maximum(same, v)
            else:
                opp = np.maximum(opp, v)
    return same, opp


# ───────────────────────── 窗口：S_CP、S_Fr → p_C（兩個通道）─────────────────────────
def s_cp(fit, fw: int, mk: dict, j: int, inf: dict):
    """回傳 (same, opp, m_w)；沒有分量出現 → NaN"""
    c = P.dim_c
    n = len(inf["cand"])
    iu = np.triu_indices(n, 1)
    nan = np.full(len(iu[0]), np.nan)
    span = int(c.fit_weeks)
    w0 = fw - span + 1
    weeks = inf["weeks"]
    known = weeks[weeks <= fw]
    m_w = 1 - len(known) / max(len(weeks), 1)
    if fit is None or m_w > float(c.max_missing_week_frac):
        return nan, nan, m_w
    (U, V, Wt), keep = fit
    wloc = known - w0
    wloc = wloc[(wloc >= 0) & (wloc < span)]
    g = inf["cand"].astype(int)
    ep, en = inf["ext_pos"] >= int(c.scp_require_extreme_days), inf["ext_neg"] >= int(c.scp_require_extreme_days)
    best_s, best_o = nan.copy(), nan.copy()
    for k in keep:
        if V[j, k] < float(c.appear_stock) or Wt[wloc, k].sum() < float(c.appear_weeks) * (1 - m_w):
            continue
        up, un = U[2 * g, k].astype(float), U[2 * g + 1, k].astype(float)
        a, b = iu
        ss = np.maximum(np.where(ep[a] & ep[b], up[a] * up[b], 0), np.where(en[a] & en[b], un[a] * un[b], 0))
        oo = np.maximum(np.where(en[a] & ep[b], un[a] * up[b], 0), np.where(ep[a] & en[b], up[a] * un[b], 0))
        best_s = np.fmax(best_s, ss)
        best_o = np.fmax(best_o, oo)
    return best_s, best_o, m_w


def window_c(mk: dict, end: str, fits: dict, sids_detail: set | None = None) -> dict | None:
    """fits＝{b: fit}（b = −1 觀測、0..19 虛無）；回傳每檔股票配對的 p_C_same、p_C_opp"""
    c = P.dim_c
    fw = fit_for(mk, end)
    infos = {j: v for (j, e), v in mk["info"].items() if e == end}
    if not infos:
        return None
    out = {"end": end, "fit_week": fw, "universe": mk["universe"], "params_version": P.version, "pairs": {}}
    obs, pools = {}, {"cp_s": Pool(), "cp_o": Pool(), "fr_s": Pool(), "fr_o": Pool()}
    strata = {}
    for j, inf in infos.items():
        dec = inf["dec"].astype(int)
        iu = np.triu_indices(len(dec), 1)
        strata[j] = stratum(dec[iu[0]], dec[iu[1]])
        s, o, mw = s_cp(fits.get(-1), fw, mk, j, inf) if fw is not None else (np.full(len(iu[0]), np.nan),) * 2 + (1.0,)
        obs[j] = {"cp_s": s, "cp_o": o, "m_w": mw}
        for b in range(int(c.null_cp)):
            if fits.get(b) is None:
                continue
            ns, no, _ = s_cp(fits[b], fw, mk, j, inf)
            pools["cp_s"].add(strata[j], ns)
            pools["cp_o"].add(strata[j], no)
    edges = [e for inf in infos.values() for e in inf["fr_edges"]]
    nb = int(c.fraudar_blocks)

    def score(E):
        blocks = fraudar_blocks(E, nb)
        conn: dict = {}
        for i, jj in E:
            conn.setdefault(jj, set()).add(i)
        return {j: s_fraudar(blocks, j, inf["cand"], conn.get(j, set())) for j, inf in infos.items()}
    fo = score(edges)
    rng = np.random.default_rng(int(hash_u("C-fr", end) * 2 ** 31))
    for _ in range(int(c.null_fr)):
        t = score(curveball(edges, rng))
        for j in infos:
            pools["fr_s"].add(strata[j], t[j][0])
            pools["fr_o"].add(strata[j], t[j][1])
    for j, inf in infos.items():
        sid = mk["sids"][j]
        keys = [mk["keys"][g] for g in inf["cand"]]
        n = len(keys)
        iu = np.triu_indices(n, 1)
        u = np.array([hash_u(sid, end, min(keys[a], keys[b]), max(keys[a], keys[b]), "C") for a, b in zip(*iu)]) if n > 1 else np.zeros(0)
        o = obs[j]
        if o["m_w"] > float(c.max_missing_week_frac):
            pcs = pco = np.full(len(iu[0]), np.nan)
        else:
            pcs, _ = pools["cp_s"].pvalues(strata[j], o["cp_s"], u)
            pco, _ = pools["cp_o"].pvalues(strata[j], o["cp_o"], u)
        pfs, _ = pools["fr_s"].pvalues(strata[j], fo[j][0], u)
        pfo, _ = pools["fr_o"].pvalues(strata[j], fo[j][1], u)
        out["pairs"][sid] = {"keys": keys, "p_C_same": np.asarray(cct([pcs, pfs]), np.float32),
                             "p_C_opp": np.asarray(cct([pco, pfo]), np.float32), "m_w": float(o["m_w"])}
    return out


def choose_rank(mk: dict, workers: int = 2) -> int:
    from concurrent.futures import ProcessPoolExecutor
    c = P.dim_c
    fw = fit_weeks(mk)[0]
    idx, val, shape = tensor(mk, fw)
    cand = [r for r in c.rank_candidates if r <= min(shape)] or [min(shape)]
    with ProcessPoolExecutor(workers) as ex:
        lls = list(ex.map(heldout_ll, [idx] * len(cand), [val] * len(cand), [shape] * len(cand), cand, [0] * len(cand)))
    ll = dict(zip(cand, lls))
    log.info("維度 C v3 秩選擇（含背景分量）：%s", {k: round(v) for k, v in ll.items()})
    return max(ll, key=ll.get)


def save(obj, f: Path) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(obj))


def load(f: Path):
    return pickle.loads(Path(f).read_bytes())


def dump_json(obj, f: Path) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(obj), encoding="utf-8")
