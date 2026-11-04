"""
維度 B：價位微結構（需要分點價位明細：FinMind Sponsor Pro；富邦備用路徑沒有 → 不計算，p_B 缺失、融合時移除）

  3.3 守恆檢查   每個（日、價位）Σ_i b_i,P = Σ_i s_i,P；違反的日子整日不用於維度 B
  B1（主規格 6.2）冷門價位共現
      冷門價位：當日 V_P ≤ 第 20 百分位數且 ≥ 5 張；參與：s_A,P ≥ 0.1 V_P 且 b_B,P ≥ 0.1 V_P
      Fréchet 下界 M_P = max(0, s_A,P + b_B,P − V_P)
      T_AB = Σ 冷門價位 [1(參與) + 1(參與 且 M_P ≥ 1 張)]；T_B1 = max(T_AB, T_BA)
  B2（修正規格 5.1）壓價轉移事件日（有方向 S 賣 → R 買），四個條件同時成立：
      1 S 淨賣 ≥ 過去 120 日（不含當日）活躍日 |淨買賣| 中位數 × 2，且 ≥ 10 張
      2 收盤 ≤ 前一日收盤
      3 R 淨買 ≥ S 淨賣 × 0.1
      4 Σ_P M_P(S→R) ≥ 1 張，或 |R 買進 VWAP − S 賣出 VWAP| ≤ 收盤 × 0.5%
      T_B2 = max(T(S→R), T(R→S))，記錄方向（seller、buyer）
  虛無分布（7.1、修正 5.5）：每日內，買方分點的標籤在同一活躍度十分位數內隨機置換，B = 50；
      每日的貢獻只取決於「賣方那一列 × 買方那一列」→ 置換買方標籤＝把當日貢獻矩陣的欄重新排列
      同一窗口內所有配對、所有排列的虛無分數依分層池化（7.2）；T = 0 → p = 1，不跑排列
  p_B = CCT(p_B1, p_B2)，各 1/2

SPEC-GAP
  - B1「下界計分」只在參與成立時計（未參與的價位不加分）
  - B2 賣方候選限主規格 6.0 的候選分點（配對表以候選分點為範圍）；exclude_foreign = true 時看不到外資賣方
  - 排列只在候選分點之間進行（非候選分點的買量照常計入 V_P）
"""
from __future__ import annotations

import numpy as np

from . import features
from .fusion import cct
from .nulls import Pool, deciles, stratum
from .params import P

LOT = 1000


def levels(pn) -> dict | None:
    """把 FinMind 價位明細轉成每日的（價位, 分點索引, 買, 賣）；沒有資料 → None"""
    try:
        import fm_store
        from history import history_key
    except Exception:  # noqa: BLE001
        return None
    if pn.meta.get("source") != "finmind":
        return None
    df = fm_store.tdr(pn.sid)
    if not len(df):
        return None
    ids = {}
    bix = {b: i for i, b in enumerate(pn.branches)}
    for bid in df["securities_trader_id"].unique():
        k = history_key({"b_raw": fm_store.fubon_row(str(bid), "")["b_raw"]})
        ids[bid] = bix.get(k, -1)
    pos = {d: i for i, d in enumerate(pn.dates)}
    df = df.assign(bi=df["securities_trader_id"].map(ids), ti=df["date"].map(pos))
    df = df[df["ti"].notna()]
    out = {}
    for t, g in df.groupby("ti", sort=True):
        out[int(t)] = (g["price"].to_numpy(float), g["bi"].to_numpy(int), g["buy"].to_numpy(float), g["sell"].to_numpy(float))
    return out


def _day(lv, cand_pos: np.ndarray, n: int):
    """一天：回傳（各價位 V_P, 候選分點的 買矩陣 n×K、賣矩陣 n×K, 守恆是否成立, 價位陣列）"""
    price, bi, b, s = lv
    up, inv = np.unique(price, return_inverse=True)
    K = len(up)
    Vb = np.bincount(inv, weights=b, minlength=K)
    Vs = np.bincount(inv, weights=s, minlength=K)
    ok = bool(np.all(Vb == Vs))
    Bm = np.zeros((n, K))
    Sm = np.zeros((n, K))
    ci = np.where(bi >= 0, cand_pos[np.maximum(bi, 0)], -1)
    m = ci >= 0
    np.add.at(Bm, (ci[m], inv[m]), b[m])
    np.add.at(Sm, (ci[m], inv[m]), s[m])
    return Vb, Bm, Sm, ok, up


def window(cx, w: dict, lv: dict | None) -> dict | None:
    """一個窗口：配對（同維度 A 的 pairs）的 p_B1、p_B2、p_B、seller／buyer。lv＝levels(pn)"""
    r = w.get("res")
    if r is None or lv is None:
        return None
    pn = cx.pn
    cand = w["cand"]
    n = len(cand)
    s0, e = w["start"], w["end"]
    days = [t for t in range(s0, e + 1) if t in lv]
    if not days:
        return None
    cb, rv = P.dim_b, P.revision
    cand_pos = np.full(len(pn.branches), -1)
    cand_pos[cand] = np.arange(n)
    net = pn.B - pn.S
    roll = int(P.behavior.rolling_days)
    D1, D2, used, viol = [], [], [], 0
    for t in days:
        V, Bm, Sm, ok, up = _day(lv[t], cand_pos, n)
        if not ok or V.sum() <= 0:
            viol += 1
            continue
        used.append(t)
        # ── B1 冷門價位 ──
        pos = V > 0
        q = np.quantile(V[pos], float(cb.cold_quantile)) if pos.any() else 0
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
        # ── B2 壓價轉移 ──
        d2 = np.zeros((n, n), np.float32)
        if t >= 1 and pn.close[t] <= pn.close[t - 1]:
            lo = max(0, t - roll)
            ns = -net[cand, t]
            hist = np.abs(net[cand, lo:t])
            act = cx.active[cand, lo:t]
            med = np.array([np.median(h[a]) if a.any() else np.inf for h, a in zip(hist, act)])
            cls = w["cls"]
            seller = (ns >= float(rv.b2_sell_mult) * med) & (ns >= float(rv.b2_sell_min_lots) * LOT) \
                & np.array([cls[i] != features.DAYTRADE for i in cand])
            if seller.any():
                nb = net[cand, t]
                absorb = nb[None, :] >= float(rv.b2_absorb_ratio) * ns[:, None]
                Mtot = np.maximum(0, Sm[:, None, :] + Bm[None, :, :] - V[None, None, :]).sum(axis=2)
                sv = np.divide((Sm * up).sum(1), Sm.sum(1), out=np.full(n, np.nan), where=Sm.sum(1) > 0)
                bv = np.divide((Bm * up).sum(1), Bm.sum(1), out=np.full(n, np.nan), where=Bm.sum(1) > 0)
                close_tol = float(rv.b2_vwap_tol) * pn.close[t]
                vw = np.abs(sv[:, None] - bv[None, :]) <= close_tol
                overlap = (Mtot >= LOT) | vw
                d2 = (seller[:, None] & absorb & overlap).astype(np.float32)
        D2.append(d2)
    if not used:
        return {"violation_days": viol, "days": 0}
    D1, D2 = np.stack(D1), np.stack(D2)
    # 活躍度十分位數（7.1）：(j, w) 內候選分點總成交量
    vol = (pn.B[cand, s0:e + 1] + pn.S[cand, s0:e + 1]).sum(axis=1)
    dec = deciles(vol)
    ii, jj = r["pairs"][:, 0], r["pairs"][:, 1]
    strata = stratum(dec[ii], dec[jj])

    def undirected(T):
        a, b = T[ii, jj], T[jj, ii]
        return np.maximum(a, b), np.where(a >= b, 0, 1)
    T1, _ = undirected(D1.sum(0))
    T2, d2dir = undirected(D2.sum(0))
    rng = np.random.default_rng(int(e))      # 固定種子（同一窗口重算結果相同）
    pool1, pool2 = Pool(), Pool()
    groups = [np.flatnonzero(dec == k) for k in range(10)]
    for _ in range(int(cb.n_null)):
        perm = np.empty((len(used), n), int)
        for di in range(len(used)):
            p = np.arange(n)
            for g in groups:
                if len(g) > 1:
                    p[g] = rng.permutation(g)
            perm[di] = p
        P1 = np.take_along_axis(D1, perm[:, None, :].repeat(n, 1), axis=2).sum(0)
        P2 = np.take_along_axis(D2, perm[:, None, :].repeat(n, 1), axis=2).sum(0)
        pool1.add(strata, undirected(P1)[0])
        pool2.add(strata, undirected(P2)[0])
    p1, _ = pool1.pvalues(strata, T1)
    p2, _ = pool2.pvalues(strata, T2)
    p1 = np.where(T1 > 0, p1, 1.0)
    p2 = np.where(T2 > 0, p2, 1.0)
    cap = 0.99
    pB = cct([np.minimum(p1, cap), np.minimum(p2, cap)], [0.5, 0.5])
    a_is_seller = d2dir == 0
    return {"p_B1": p1, "p_B2": p2, "p_B": pB, "T_B1": T1, "T_B2": T2,
            "seller": np.where(a_is_seller, ii, jj), "buyer": np.where(a_is_seller, jj, ii),
            "days": len(used), "violation_days": viol}
