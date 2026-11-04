"""
§6 維度 B（v3）：只進反向通道
  資料：FinMind 價位明細；分點以 securities_trader_id 對應
  §3.1 不平衡：imb = |Σb − Σs| / max(Σb, Σs) > 2% 的日子不用；V_P = max(Σ_i b_i,P, Σ_i s_i,P)
       一致性：分點日 Σ_P b_i,P ≠ SecIdAgg buy_volume（或賣）→ 標 level_mismatch，該分點日不用於維度 B
  B1 冷門價位共現、B2 壓價轉移：定義同 v2，修改：
       B2 條件 1 門檻用淨量中位數（§4.1 活躍日）；條件 2 加「S 的賣出均價 ≤ 當日 VWAP」（§6.2）
  §6.1 條件 p：T = 0 → 缺失；虛無池中 > 0 的分數 < M = 20 → 缺失；否則 p = (#{null > t} + U·(#{null = t} + 1)) / (1 + #{null > 0})
  p_B = CCT(p_B1, p_B2)；兩者都缺 → 缺失
SPEC-GAP-v3
  ・配對範圍＝候選分點（v3 候選已含當沖型；預設版本含外資），非候選分點的賣方沒有維度 A 配對可融合，本版不納入
  ・排列只在候選分點之間進行（同 v2）
"""
from __future__ import annotations

import numpy as np

from . import features
from .params import P
from .pvals import Pool, cct, deciles, hash_u_many, stratum

LOT = 1000


def levels(pn) -> dict | None:
    """{t: (price, 分點索引, buy, sell)}、X（同價位雙向量，分點 × 日）、mismatch（分點 × 日）"""
    try:
        import fm_store
        df = fm_store.tdr(pn.sid)
    except Exception:  # noqa: BLE001
        return None
    if df is None or not len(df):
        return None
    bix = {b: i for i, b in enumerate(pn.branches)}
    pos = {d: i for i, d in enumerate(pn.dates)}
    df = df.assign(bi=df["securities_trader_id"].astype(str).map(bix).fillna(-1).astype(int),
                   ti=df["date"].astype(str).map(pos))
    df = df[df["ti"].notna()]
    df["ti"] = df["ti"].astype(int)
    n, T = pn.B.shape
    X = np.full((n, T), np.nan)
    LB = np.zeros((n, T)); LS = np.zeros((n, T))
    k = df["bi"] >= 0
    g = df[k].assign(mn=np.minimum(df.loc[k, "buy"], df.loc[k, "sell"]))
    agg = g.groupby(["bi", "ti"]).agg(x=("mn", "sum"), b=("buy", "sum"), s=("sell", "sum")).reset_index()
    X[agg["bi"], agg["ti"]] = agg["x"]
    LB[agg["bi"], agg["ti"]] = agg["b"]
    LS[agg["bi"], agg["ti"]] = agg["s"]
    have = np.zeros(T, bool)
    have[df["ti"].unique()] = True
    mism = have[None, :] & ((np.abs(LB - pn.B) > 0) | (np.abs(LS - pn.S) > 0))
    out = {}
    for t, gg in df.groupby("ti", sort=True):
        out[int(t)] = (gg["price"].to_numpy(float), gg["bi"].to_numpy(int), gg["buy"].to_numpy(float), gg["sell"].to_numpy(float))
    X = np.where(have[None, :], np.nan_to_num(X), np.nan)
    return {"days": out, "X": X, "mismatch": mism}


def _day(lv, cand_pos: np.ndarray, n: int, bad: np.ndarray):
    price, bi, b, s = lv
    up, inv = np.unique(price, return_inverse=True)
    K = len(up)
    Vb = np.bincount(inv, weights=b, minlength=K)
    Vs = np.bincount(inv, weights=s, minlength=K)
    V = np.maximum(Vb, Vs)
    Bm = np.zeros((n, K)); Sm = np.zeros((n, K))
    ci = np.where(bi >= 0, cand_pos[np.maximum(bi, 0)], -1)
    m = (ci >= 0) & ~bad[np.maximum(bi, 0)]
    np.add.at(Bm, (ci[m], inv[m]), b[m])
    np.add.at(Sm, (ci[m], inv[m]), s[m])
    return V, Bm, Sm, up


def window(pn, cx, w: dict, lv: dict | None, key: tuple) -> dict | None:
    r = w.get("res")
    if r is None or lv is None:
        return None
    cand = w["cand"]
    n = len(cand)
    s0, e = w["start"], w["end"]
    imax = float(P.data.imbalance_max)
    days = [t for t in range(s0, e + 1) if t in lv["days"]]
    if not days:
        return None
    cb, rv = P.dim_b, P.revision
    cand_pos = np.full(len(pn.branches), -1)
    cand_pos[cand] = np.arange(n)
    net = pn.B - pn.S
    roll = int(P.behavior.rolling_days)
    D1, D2, used, skipped = [], [], [], 0
    cls = w["cls"]
    not_dt = np.array([cls[i] != features.DAYTRADE for i in cand])
    for t in days:
        if pn.imb[t] > imax or pn.suspended[t]:
            skipped += 1
            continue
        V, Bm, Sm, up = _day(lv["days"][t], cand_pos, n, lv["mismatch"][:, t])
        if V.sum() <= 0:
            skipped += 1
            continue
        used.append(t)
        pos = V > 0
        q = np.quantile(V[pos], float(cb.cold_quantile))
        cold = pos & (V <= q) & (V >= float(cb.cold_min_lots) * LOT)
        d1 = np.zeros((n, n), np.float32)
        if cold.any():
            Vc, Sc, Bc = V[cold], Sm[:, cold], Bm[:, cold]
            ps = Sc >= float(cb.participate) * Vc
            pb = Bc >= float(cb.participate) * Vc
            for k in range(len(Vc)):
                part = np.outer(ps[:, k], pb[:, k])
                if not part.any():
                    continue
                M = Sc[:, k][:, None] + Bc[:, k][None, :] - Vc[k]
                d1 += part.astype(np.float32) + (part & (M >= float(cb.frechet_min_lots) * LOT)).astype(np.float32)
        D1.append(d1)
        d2 = np.zeros((n, n), np.float32)
        if t >= 1 and pn.close[t] <= pn.close[t - 1]:
            lo = max(0, t - roll)
            ns = -net[cand, t]
            hist = np.abs(net[cand, lo:t])
            act = cx.active[cand, lo:t]
            med = np.array([np.median(h[a]) if a.any() else np.inf for h, a in zip(hist, act)])
            sp = pn.SP[cand, t]
            below = np.isfinite(sp) & (sp <= pn.vwap[t]) if rv.b2_seller_below_vwap else np.ones(n, bool)
            seller = (ns >= float(rv.b2_sell_mult) * med) & (ns >= float(rv.b2_sell_min_lots) * LOT) & not_dt & below
            if seller.any():
                nb = net[cand, t]
                absorb = nb[None, :] >= float(rv.b2_absorb_ratio) * ns[:, None]
                Mtot = np.maximum(0, Sm[:, None, :] + Bm[None, :, :] - V[None, None, :]).sum(axis=2)
                sv = np.divide((Sm * up).sum(1), Sm.sum(1), out=np.full(n, np.nan), where=Sm.sum(1) > 0)
                bv = np.divide((Bm * up).sum(1), Bm.sum(1), out=np.full(n, np.nan), where=Bm.sum(1) > 0)
                vw = np.abs(sv[:, None] - bv[None, :]) <= float(rv.b2_vwap_tol) * pn.close[t]
                d2 = (seller[:, None] & absorb & ((Mtot >= LOT) | vw)).astype(np.float32)
                np.fill_diagonal(d2, 0)
        D2.append(d2)
    if not used:
        return {"days": 0, "skipped_days": skipped}
    nz = [k for k in range(len(D1)) if D1[k].any() or D2[k].any()]     # 全為 0 的日子排列後仍為 0：不必參與排列
    D1 = np.stack([D1[k] for k in nz]) if nz else np.zeros((1, n, n), np.float32)
    D2 = np.stack([D2[k] for k in nz]) if nz else np.zeros((1, n, n), np.float32)
    dec = r["dec"]
    ii, jj = r["pairs"][:, 0], r["pairs"][:, 1]
    strata = stratum(dec[ii], dec[jj])

    def undirected(T):
        a, b = T[ii, jj], T[jj, ii]
        return np.maximum(a, b), np.where(a >= b, 0, 1)
    T1, _ = undirected(D1.sum(0))
    T2, d2dir = undirected(D2.sum(0))
    rng = np.random.default_rng(int(e))
    pool1, pool2 = Pool(), Pool()
    groups = [np.flatnonzero(dec == k) for k in range(10)]
    for _ in range(int(cb.n_null)):
        perm = np.empty((len(D1), n), int)
        for di in range(len(D1)):
            p = np.arange(n)
            for g in groups:
                if len(g) > 1:
                    p[g] = rng.permutation(g)
            perm[di] = p
        P1 = np.take_along_axis(D1, perm[:, None, :].repeat(n, 1), axis=2).sum(0)
        P2 = np.take_along_axis(D2, perm[:, None, :].repeat(n, 1), axis=2).sum(0)
        pool1.add(strata, undirected(P1)[0])
        pool2.add(strata, undirected(P2)[0])
    br = np.array(pn.branches)[cand]
    M = int(cb.cond_p_min_nonzero_null)
    p1, _ = pool1.pvalues(strata, T1, hash_u_many(key, br[ii], br[jj], "B1"), conditional=True, min_nonzero=M, gpd=False)
    p2, _ = pool2.pvalues(strata, T2, hash_u_many(key, br[ii], br[jj], "B2"), conditional=True, min_nonzero=M, gpd=False)
    a_is_seller = d2dir == 0
    return {"p_B1": p1, "p_B2": p2, "p_B": cct([p1, p2]), "T_B1": T1, "T_B2": T2,
            "seller": np.where(a_is_seller, ii, jj), "buyer": np.where(a_is_seller, jj, ii),
            "days": len(used), "skipped_days": skipped}
