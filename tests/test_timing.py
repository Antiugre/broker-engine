"""python3 -m pytest tests/test_timing.py -q（單一分點擇時偵測 v1.1.2：規格 §11 步驟 1～6 與步驟 7 的 EXEC = 0 驗收測試；全部用合成資料）"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broker_timing import classify, ei, fixedb, hac, prep, reaction, report, run, validate, zigzag  # noqa: E402
from broker_timing.params import TP  # noqa: E402

LOT = 1000


# ───────────────────────── 合成資料 ─────────────────────────
def _dates(n, start="2020-11-02"):
    return np.array([d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)])


def _market(T, rng, n_peers=6, beta=1.0, idio=0.02):
    rm = rng.normal(0, 0.01, T)
    mkt = 10000 * np.cumprod(1 + rm)
    ri = rm + rng.normal(0, 0.004, T)                      # 產業共同因子
    e = rng.normal(0, idio, T)
    r = beta * rm + 0.5 * (ri - rm) + e
    adj = 50 * np.cumprod(1 + r)
    peers = np.vstack([30 * np.cumprod(1 + ri + rng.normal(0, 0.02, T)) for _ in range(n_peers)]) if n_peers else None
    vol = rng.integers(2_000_000, 6_000_000, T).astype(float)
    return mkt, adj, peers, vol, e


def _prep(T=1500, seed=0, n_peers=6):
    rng = np.random.default_rng(seed)
    d = _dates(T)
    mkt, adj, peers, vol, e = _market(T, rng, n_peers)
    pr = prep.compute(d, adj.copy(), adj, vol, mkt, peers)
    return d, pr, rng


def _panel(dates, nets, close, vol, adj, sid="9999", bp=None, sp=None):
    from broker_flow_v3.data_layer import Panel
    nets = np.atleast_2d(np.asarray(nets, float))
    nb, T = nets.shape
    B = np.where(nets > 0, nets, 0.0)
    S = np.where(nets < 0, -nets, 0.0)
    vw = close.copy()
    BP = np.where(B > 0, vw[None, :] if bp is None else bp, np.nan)
    SP = np.where(S > 0, vw[None, :] if sp is None else sp, np.nan)
    return Panel(sid, np.asarray(dates), [f"b{i}" for i in range(nb)], [f"分點{i}" for i in range(nb)], ["brokerage"] * nb,
                 B, S, BP, SP, close.copy(), vw, vol.copy(), close * 1.01, close * 0.99, adj.copy(), np.zeros(T, bool), np.zeros(T), {})


# ───────────────────────── 步驟 1：前處理（§3） ─────────────────────────
def test_beta_unchanged_by_future_prices():
    d, pr, rng = _prep()
    inp = pr.inputs
    for t0 in (300, 800, 1200):
        adj2 = inp["adj"].copy()
        adj2[t0:] *= np.exp(np.cumsum(rng.normal(0, 0.05, len(adj2) - t0)))
        mk2 = inp["mkt"].copy()
        mk2[t0:] *= 1.3
        pe2 = inp["peer_adj"].copy()
        pe2[:, t0:] *= 0.7
        pr2 = prep.compute(d, adj2, adj2, inp["volume"], mk2, pe2)
        np.testing.assert_array_equal(pr.beta_m[:t0 + 1], pr2.beta_m[:t0 + 1])
        np.testing.assert_array_equal(pr.beta_ind[:t0 + 1], pr2.beta_ind[:t0 + 1])
        assert np.isfinite(pr.beta_m[t0])


def test_beta_recovers_true_value_and_excess_return_definition():
    d, pr, _ = _prep(seed=3)
    t = 1000
    assert abs(pr.beta_m[t] - 1.0) < 0.4
    h = 20
    a = pr.adj
    m = pr.inputs["mkt"]
    pa = pr.inputs["peer_adj"]
    tot = 0.0                                    # 第 2 輪 §3.4：單日超額對數報酬之和，β 用 s−1 的值
    for s in range(t + 1, t + h + 1):
        lr, lm = np.log(a[s] / a[s - 1]), np.log(m[s] / m[s - 1])
        li = np.log1p(np.mean(pa[:, s] / pa[:, s - 1] - 1))
        tot += lr - pr.beta_m[s - 1] * lm - pr.beta_ind[s - 1] * (li - lm)
    assert pr.rex[h][t] == pytest.approx(tot, rel=1e-9)


def test_eligibility_needs_120_prior_valid_days():
    d, pr, _ = _prep(T=400)
    assert not pr.eligible[:120].any()
    assert pr.eligible[120]
    close = pr.inputs["close"].copy()
    close[50:60] = 0                                  # 無公告價的日子不算有效日
    pr2 = prep.compute(d, close, pr.inputs["adj"], pr.inputs["volume"], pr.inputs["mkt"], pr.inputs["peer_adj"])
    assert not pr2.eligible[:130].any() and pr2.eligible[130]
    assert not pr2.eligible[55]


def test_par_value_change_keeps_ncb_continuous():
    """10 元面額改 1 元：股數 × 10、價格 ÷ 10；k = P_before / P_ref = 10"""
    d = _dates(300)
    tev = 150
    events = pd.DataFrame([{"date": d[tev], "kind": "ratio", "value": 10.0, "approx": False}])
    g, approx = prep.share_basis(d, events)
    n = np.where(np.arange(300) < tev, 1000.0, 10000.0)
    ncb = g * n
    assert np.allclose(ncb, 10000.0)
    assert g[-1] == 1.0 and g[tev] == 1.0 and g[tev - 1] == 10.0
    ev2 = pd.DataFrame([{"date": d[100], "kind": "stock", "value": 0.1, "approx": False},
                        {"date": d[200], "kind": "cash", "value": 2.0, "approx": False},
                        {"date": d[250], "kind": "ratio", "value": 0.5, "approx": True}])
    g2, ap2 = prep.share_basis(d, ev2)
    assert g2[50] == pytest.approx(1.1 * 0.5) and g2[150] == pytest.approx(0.5) and g2[260] == 1.0
    assert ap2[250] and ap2.sum() == 1


def test_industry_factor_false_with_fewer_than_5_peers():
    for k, want in ((0, False), (4, False), (5, True)):
        _, pr, _ = _prep(T=400, n_peers=k)
        assert pr.industry_factor is want
        if not want:
            assert np.all(pr.beta_ind == 0)
    # Industry 類別：同產業成員排除自己；少於 5 檔 → peer_matrix 為 None
    info = pd.DataFrame({"stock_id": ["A", "B", "C", "D", "E", "F", "G"], "industry_category": ["x"] * 5 + ["y"] * 2})
    ind = prep.Industry(list("ABCDEFG"), info, adj_fn=lambda s: pd.Series(dtype=float))
    assert ind.peers("A") == ["B", "C", "D", "E"]
    assert ind.peer_matrix("A", _dates(10))[0] is None


def test_standardize_uses_active_median():
    ncb = np.array([0, 2000, -1000, 0, 4000.0])
    act = ncb != 0
    nt, med = prep.standardize(ncb, act)
    assert med == 2000 and np.allclose(nt, [0, 1, -0.5, 0, 2])


# ───────────────────────── 步驟 2：Zigzag（§6.2） ─────────────────────────
def _zz(n_lots, price=10.0):
    n = np.asarray(n_lots, float) * LOT
    act = n != 0
    return zigzag.run(n, np.full(len(n), price), np.ones(len(n)), act)


def test_zigzag_a_one_year_of_four_waves_in_five_years():
    T = 1250
    n = np.zeros(T)
    t = 500
    for _ in range(4):
        n[t:t + 10] = 100; t += 10
        n[t:t + 10] = -100; t += 10
        t += 30
    z = _zz(n)
    assert z["A"] == pytest.approx(500 * LOT)
    assert z["cycles"] == 4


def test_zigzag_b_base_position_buy_1000_sell_700():
    n = []
    for _ in range(5):
        n += [100] * 10 + [-70] * 10
    z = _zz(n)
    assert z["cycles"] == 5


def test_zigzag_c_sell_first():
    n = []
    for _ in range(3):
        n += [-100] * 10 + [100] * 10
    z = _zz(n)
    assert z["cycles"] == 3 and z["segments"][0]["dir"] == -1


def test_zigzag_d_jitter_below_A():
    n = []
    for _ in range(6):
        n += [10] * 49 + [-10] * 49
    z = _zz(n)
    assert z["A"] == pytest.approx(500 * LOT)
    assert z["cycles"] == 0 and z["n_segments"] == 0


def test_zigzag_e_first_day_buy_then_clear():
    n = [1000] + [1, -1] * 20 + [-1000]
    z = _zz(n)
    assert z["A"] == pytest.approx(500 * LOT)
    assert z["n_segments"] == 2 and z["cycles"] == 1
    assert z["segments"][0]["dir"] == 1 and z["segments"][0]["start"] == -1     # 起點在第一個活躍日之前


def test_zigzag_f_small_drop_then_rise_confirms_up():
    """SPEC-GAP-timing（解讀）：I 由 0 跌到 −0.4A，再漲到 +0.8A（從區間低點起漲 1.2A ≥ A）→ 第一段向上"""
    A = 500
    n = [-0.4 * A, 1.2 * A] + [1, -1] * 10
    z = _zz(n)
    assert z["A"] == pytest.approx(A * LOT)
    assert z["n_segments"] == 1 and z["segments"][0]["dir"] == 1


# ───────────────────────── 步驟 3：反應函數（§4.2） ─────────────────────────
def test_reaction_absorbs_buy_after_five_day_drop_and_alpha():
    d, pr, rng = _prep(T=1500, seed=5)
    X = pr.feats
    s5 = X[:, 2]
    n = np.where(np.nan_to_num(s5) < -0.02, np.round(-s5 * 5e6 + rng.normal(0, 5e4, len(s5))), 0.0)
    n[:150] = 0
    act = n != 0
    nt, _ = prep.standardize(n, act)
    rf = reaction.fit(nt, act, X)
    assert rf["n_act"] > 100
    assert rf["alpha"] == pytest.approx(float(TP.ridge_lambda) * rf["n_act"])
    assert rf["model"].alpha == pytest.approx(0.01 * rf["n_act"])
    assert rf["model"].fit_intercept
    e = rf["eps"]
    assert np.all(e[~act] == 0)
    assert abs(e[act].sum()) < 1e-8                     # 截距不懲罰 → 活躍日殘差總和 0
    c = np.corrcoef(e[act], s5[act])[0, 1]
    assert abs(c) < 0.05
    assert abs(np.corrcoef(nt[act], s5[act])[0, 1]) > 0.5
    rf2 = reaction.fit(nt, act, X, eplus=True)
    assert len(rf2["coef"]) == 8 and len(rf["coef"]) == 7


# ───────────────────────── 步驟 4：堆疊 HAC、G_eff、多期間（§4.3、§4.4） ─────────────────────────
HZ = [5, 10, 20, 40]


def _returns(rng, n, kind="iid"):
    """純虛無的單日報酬。iid：常態 2%；garch：GARCH(1,1)、t(4) 厚尾；garch_bounce：再加 MA(1) −0.3（買賣價差反彈）"""
    if kind == "iid":
        return rng.normal(0, 0.02, n)
    w, a, b = 0.02 ** 2 * 0.05, 0.10, 0.85
    z = rng.standard_t(4, n) / np.sqrt(2.0)
    e = np.empty(n)
    s2 = 0.02 ** 2
    for i in range(n):
        e[i] = np.sqrt(s2) * z[i]
        s2 = w + a * e[i] ** 2 + b * s2
    if kind == "garch_bounce":
        e = e + np.r_[0.0, -0.3 * e[:-1]]
    return e


def _null(rng, N=500, sparse=None, common=1.0, kmax=40, ret="iid", react=0.0, with_e=False):
    """純虛無：前瞻超額報酬與分點殘差（給定過去）無關。殘差集中在波段（3～15 日），common = 波段內共同成分的標準差（持續性）。
    e[i] 為第 i 日報酬（長度 N + 40），Y[t, h] = Σ_{i=t+1}^{t+h} e_i；react > 0 時 x 加上 −react·(t−4～t 日報酬和)/0.02（逆勢承接，x 依賴過去報酬）"""
    e = _returns(rng, N + 40, ret)
    cs = np.r_[0, np.cumsum(e)]
    t = np.arange(N)
    Y = np.column_stack([cs[t + 1 + h] - cs[t + 1] for h in HZ])
    act = np.zeros(N, bool)
    x = np.zeros(N)
    k = rng.integers(6, kmax) if sparse is None else sparse
    for _ in range(k):
        s = rng.integers(0, N)
        L = rng.integers(3, 16)
        act[s:s + L] = True
        x[s:s + L] = common * rng.normal() + rng.normal(0, 1, len(x[s:s + L]))
    if react:
        past5 = cs[t + 1] - cs[np.maximum(t - 4, 0)]
        x[act] += react * (-past5[act] / 0.02)
    x[act] -= x[act].mean()
    return (Y, x, act, e) if with_e else (Y, x, act)


def _through_reaction(x, act, e):
    """x 先經本實作的反應函數（§4.2，全期間 Ridge）取殘差；特徵由同一條日報酬算出（第 i 日收盤 = exp(Σ_{j≤i} e_j)）"""
    N = len(x)
    adj = np.exp(np.cumsum(e[:N]))
    r = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
    X = prep.features(adj, r, np.zeros(N), np.full(N, 1e6))
    return reaction.fit(x, act, X)["eps"]


def test_newey_west_matches_bruteforce_sandwich():
    rng = np.random.default_rng(0)
    N, L = 300, 40
    Y, x, act = _null(rng, N=N, sparse=15)
    r = hac.test(Y, x, act, horizons=HZ)
    _, _, U, sxx = hac.ols_slopes(Y, x)
    xc = x - x.mean()
    W = np.maximum(0, 1 - np.abs(np.subtract.outer(np.arange(N), np.arange(N))) / (L + 1))
    psi = xc[:, None] * U
    V = psi.T @ W @ psi / sxx ** 2
    np.testing.assert_allclose(np.sqrt(np.diag(V)), r["se"], rtol=1e-10)
    np.testing.assert_allclose(V / np.outer(r["se"], r["se"]), r["R"], rtol=1e-8, atol=1e-12)


def test_hac_deterministic_and_short_sample():
    rng = np.random.default_rng(1)
    Y, x, act = _null(rng, N=200, sparse=12)
    a = hac.test(Y, x, act, horizons=HZ)
    b = hac.test(Y.copy(), x.copy(), act.copy(), horizons=HZ)
    for k in ("theta", "se", "t", "R"):
        np.testing.assert_array_equal(a[k], b[k])
    assert a["p"] == b["p"] and a["M_eff"] == b["M_eff"]
    assert np.isfinite(a["p"]) and 1e-15 <= a["p"] <= 1 and a["N"] == 200 and np.isfinite(a["t"]).all()
    assert a["h_star"] == HZ[int(np.nanargmax(a["t"]))]


def test_geff_blocks():
    a = np.zeros(205, bool)
    assert hac.geff(a) == 0
    a[[0, 19, 20, 204]] = True
    assert hac.geff(a) == 3                         # 區塊 0、1、最後不足 20 日的區塊 10


def test_p_monotone_in_tmax():
    for method, N in (("fixed_b", 1280), ("fixed_b", 200), ("t_geff", 1280)):
        ps = [hac.combine(np.array([tm, 0, 0, 0]), N, 9, 2.5, method=method)[1] for tm in np.linspace(-3, 12, 300)]
        assert all(b <= a for a, b in zip(ps, ps[1:])), method
        assert ps[-1] >= 1e-15 and ps[0] == 1.0
    fine = [fixedb.sf(c, 1280) for c in np.linspace(0, 9, 2000)]          # 單變量尾機率：細網格嚴格遞減
    assert all(b < a for a, b in zip(fine, fine[1:]))


def test_fixedb_matches_rao_blackwell_mc_and_is_deterministic():
    """fixed-b 尾機率（有限 T 高斯二次型，精確反轉）對照 Rao–Blackwell 蒙地卡羅 E_Q[Φ̄(c√Q)]（測試用固定種子）"""
    from scipy import stats
    T, L = 300, 40
    mu = fixedb.eigs(T, fixedb.bandwidth(L))
    assert 0.5 < mu.sum() <= 1.0 and (mu > 0).all()                     # Σμ = tr(PKP)/T
    rng = np.random.default_rng(0)
    Q = np.concatenate([(rng.standard_normal((20000, len(mu))) ** 2) @ mu for _ in range(10)])
    for c in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        v = stats.norm.sf(c * np.sqrt(Q))
        m, se = v.mean(), v.std() / np.sqrt(len(Q))
        ex = fixedb.exact_sf(c, mu)
        assert abs(ex - m) < 4 * se + 1e-12, (c, ex, m, se)
        assert fixedb.sf(c, T, L) == pytest.approx(ex, rel=2e-3)          # PCHIP 內插誤差
        assert ex > stats.norm.sf(c)                                       # 比常態保守
    assert fixedb.sf(0.0, T, L) == pytest.approx(0.5)
    assert fixedb.sf(-2.0, T, L) == pytest.approx(1 - fixedb.sf(2.0, T, L))
    assert fixedb.exact_sf(5.0, mu) == fixedb.exact_sf(5.0, mu)
    assert 0 < fixedb.sf(30.0, T, L) < 1e-15                               # 深尾仍為正、可截在 1e-15


def test_fixedb_calibrated_for_dense_iid_scores():
    """fixed-b 的設計情境（分數 i.i.d.、b = 41/300）：F̂(α) 通過 §9.1 判準；同資料用常態臨界值則明顯 over-reject"""
    from scipy import stats
    rng = np.random.default_rng(5)
    N, L = 300, 40
    pf, pn = [], []
    for _ in range(6000):
        y = rng.normal(0, 1, N); x = rng.normal(0, 1, N)
        t = hac.test(y[:, None], x, np.ones(N, bool), L=L, pvalue=False)["t"][0]
        pf.append(fixedb.sf(t, N, L)); pn.append(stats.norm.sf(t))
    assert validate.super_uniform(pf)["ok"]
    assert not validate.super_uniform(pn)["ok"]


def test_meff_cheverud_nyholt():
    assert hac.m_eff(np.eye(4)) == pytest.approx(4.0)
    assert hac.m_eff(np.ones((4, 4))) == pytest.approx(1.0)
    prev = None
    vals = []
    for r in np.linspace(0, 0.999, 1000):
        R = np.full((4, 4), r); np.fill_diagonal(R, 1)
        m = hac.m_eff(R)
        vals.append(m)
        assert m == pytest.approx(4 - 3 * r * r, abs=1e-9)
        if prev is not None:
            assert m <= prev + 1e-12 and prev - m < 0.01            # 連續、單調、無跳動
        prev = m
    assert vals[-1] < 1.01
    bad = np.array([[1, .9, .9, -.9], [.9, 1, .9, .9], [.9, .9, 1, .9], [-.9, .9, .9, 1]])   # 非正定 → 最近正定
    assert 1 <= hac.m_eff(bad) <= 4


NULL_SCENARIOS = {                      # §11 步驟 4（第 2 輪起 6 個情境，見 CHANGELOG 1.1.2+r2）
    "iid_common0": dict(common=0.0),
    "iid_common05": dict(common=0.5),
    "garch": dict(common=0.5, ret="garch"),
    "react": dict(common=0.5, ret="garch", react=1.0),
    "react_reaction_fn": dict(common=0.5, ret="garch", react=1.0, resid=True),
    "bounce": dict(common=0.5, ret="garch_bounce"),
}


def null_pvalues(scenario, n_sim=10000, seed=20261104, method=None):
    """純虛無合成資料的 p^E（10,000 組、N = 1,280，只計入活躍日 ≥ 40 且 G_eff ≥ 8）"""
    kw = dict(NULL_SCENARIOS[scenario])
    resid = kw.pop("resid", False)
    rng = np.random.default_rng(seed)
    p, g, na = [], [], []
    for _ in range(n_sim):
        Y, x, act, e = _null(rng, N=1280, kmax=60, with_e=True, **kw)
        if act.sum() < 40 or hac.geff(act) < 8:
            continue
        if resid:
            x = _through_reaction(x, act, e)
        if (method or str(TP.hac_inference)) == "reverse_hw":
            r = hac.reverse_test(x, np.ones(len(x), bool), e, act, horizons=HZ)
        else:
            r = hac.test(Y, x, act, horizons=HZ, method=method)
        p.append(r["p"]); g.append(r["G_eff"]); na.append(int(act.sum()))
    return p, g, na


@pytest.mark.parametrize("scenario", list(NULL_SCENARIOS))
def test_null_calibration_by_geff_strata(scenario):
    """§11 步驟 4：純虛無合成資料，全部、G_eff 三層、活躍日三層的 F̂(α) 都通過 §9.1 判準（6 個情境，各 10,000 組）"""
    p, g, na = null_pvalues(scenario)
    assert len(p) > 5000
    res = validate.stratified(p, g, na)
    bad = {f"{dim}:{k}": {a: round(v["F"], 4) for a, v in s["alphas"].items()}
           for dim in ("G_eff", "active_days") for k, s in res[dim].items() if s["N"] and not s["ok"]}
    assert res["all"]["ok"] and not bad, (res["all"], bad)


def test_reverse_numerator_equals_regression_numerator():
    """反向迴歸的分子 Σ_s w_{s,h} e_s 恆等於 Σ_t x_c[t] r^ex_{t→t+h}（θ̂ 與 OLS 相同）；結果完全確定"""
    rng = np.random.default_rng(3)
    Y, x, act, e = _null(rng, N=600, sparse=20, with_e=True)
    s = np.ones(600, bool)
    s[::7] = False                                   # 樣本不連續（停牌、不合格日）
    a = hac.reverse_test(x, s, e, act, horizons=HZ)
    b = hac.reverse_test(x.copy(), s.copy(), e.copy(), act.copy(), horizons=HZ)
    xc = np.where(s, x - x[s].mean(), 0.0)
    np.testing.assert_allclose(a["num"], xc[s] @ Y[s], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(a["theta"], hac.ols_slopes(Y[s], x[s])[0], rtol=1e-9)
    for k in ("num", "t", "R"):
        np.testing.assert_array_equal(a[k], b[k])
    assert a["p"] == b["p"] and 1e-15 <= a["p"] <= 1 and a["h_star"] == HZ[int(np.argmax(a["t"]))]
    assert a["G_eff"] == hac.geff(act[s])


def test_hunter_worsley_bounds_and_exactness():
    """Hunter–Worsley：Φ̄(c) ≤ p ≤ 4Φ̄(c)；完全相關時等於 Φ̄(c)、獨立時接近 1 − (1 − Φ̄)⁴；Owen's T 的二元尾機率對照 scipy"""
    from scipy import stats
    for c in (1.0, 2.5, 4.0, 6.0):
        one = stats.norm.sf(c)
        assert hac.hunter_worsley(c, np.ones((4, 4))) == pytest.approx(one, rel=1e-6)
        ind = hac.hunter_worsley(c, np.eye(4))
        assert ind == pytest.approx(min(1.0, 4 * one - 3 * one ** 2), rel=1e-6)
        R = np.array([[1, .8, .6, .4], [.8, 1, .8, .6], [.6, .8, 1, .8], [.4, .6, .8, 1.]])
        p = hac.hunter_worsley(c, R)
        assert one <= p <= 4 * one
        if c <= 4:
            exact = 1 - stats.multivariate_normal(np.zeros(4), R).cdf(np.full(4, c))
            assert p >= exact * (1 - 1e-3)                                # 上界
    for r in (-0.5, 0.0, 0.3, 0.9):
        mvn = stats.multivariate_normal([0, 0], [[1, r], [r, 1]])
        c = 1.5
        ref = 1 - 2 * stats.norm.cdf(c) + mvn.cdf([c, c])
        assert hac.bvn_upper(c, r) == pytest.approx(ref, abs=1e-6)
    ps = [hac.hunter_worsley(c, R) for c in np.linspace(-3, 12, 300)]
    assert all(b <= a for a, b in zip(ps, ps[1:])) and ps[-1] >= 1e-15 and ps[0] == 1.0


def test_excess_returns_add_up_and_use_beta_before_s():
    """§3.4（第 2 輪）：r^ex_{t→t+h} = Σ_{s=t+1}^{t+h} e_s；e_s 只用 s−1 以前估的 β"""
    d, pr, _ = _prep(T=900, seed=21)
    e = pr.ex1
    for h in HZ:
        for t in (300, 450, 700):
            if np.isfinite(pr.rex[h][t]):
                assert pr.rex[h][t] == pytest.approx(e[t + 1:t + 1 + h].sum(), rel=1e-12, abs=1e-15)
    s = 500
    lr = np.log(pr.adj[s] / pr.adj[s - 1])
    lm = np.log(1 + pr.rm[s])
    assert e[s] == pytest.approx(lr - pr.beta_m[s - 1] * lm - pr.beta_ind[s - 1] * (pr.lind[s] - lm) if pr.industry_factor else lr - pr.beta_m[s - 1] * lm)


# ───────────────────────── 步驟 5：R、E+、EI 與 bootstrap（§4.5） ─────────────────────────
def _ei_data(kind, rng, N=1500, h=10):
    T = N + 40
    eps = np.zeros(T)
    r = rng.normal(0, 0.01, T + 1)
    if kind == "pump_dump":
        t = 0
        while t < T - 20:
            if rng.random() < 0.08:
                eps[t:t + 5] = rng.uniform(0.5, 1.5, 5)
                eps[t + 5:t + 15] = -rng.uniform(0.5, 1.5, 10)
                t += 15
            else:
                t += 1
        r[:T] += 0.01 * eps                          # 自己的交易推動當日收盤價（衝擊；r[j] 是第 j 日報酬）
    elif kind == "timing":
        on = rng.random(T) < 0.3
        eps[on] = rng.normal(0, 1, on.sum())
        for t in np.flatnonzero(on):                 # 交易後 h 日內的報酬漂移（非自身衝擊）
            r[t + 1:t + 1 + h] += 0.004 * eps[t] / h * 10
    cs = np.r_[0, np.cumsum(r[1:])]
    s = np.arange(N)
    y = cs[s + h] - cs[s]                            # r_{t→t+h}：τ = t+1..t+h
    return eps, s, y


def test_ei_pump_dump_delta_hi_nonpositive():
    rng = np.random.default_rng(7)
    eps, s, y = _ei_data("pump_dump", rng, h=5)          # 這個型態主檢定最強期間 h* = 5（先拉 5 日、後倒 10 日）
    fp, fm = ei.flows(eps, s, 5)
    assert (fp != 0).any() and (fm != 0).any()
    out = ei.run(y, eps[s], fp, fm, "9999", "b_pd", B=999)
    assert out["theta_E"] > 0
    assert out["delta_hi"] <= 0 and out["kappa_plus"] > 0
    assert classify.classify(0.01, 0.01, out["delta_lo"], out["delta_hi"], out["kappa_plus"], out["unstable"]) == classify.PUMP


def test_ei_pure_timing_delta_lo_nonnegative():
    rng = np.random.default_rng(11)
    eps, s, y = _ei_data("timing", rng)
    fp, fm = ei.flows(eps, s, 10)
    out = ei.run(y, eps[s], fp, fm, "9999", "b_t", B=999)
    assert out["theta_E"] > 0 and out["delta_lo"] >= 0
    assert classify.classify(0.01, 0.01, out["delta_lo"], out["delta_hi"], out["kappa_plus"], out["unstable"]) == classify.TIMING


def test_ei_sparse_geff8_no_nan_and_discard_recorded():
    rng = np.random.default_rng(3)
    N = 400
    eps = np.zeros(N + 40)
    for b in rng.choice(N // 20, 8, replace=False):
        eps[b * 20 + rng.integers(0, 20)] = rng.normal(0, 1)
    s = np.arange(N)
    assert hac.geff(eps[s] != 0) == 8
    y = rng.normal(0, 0.05, N)
    fp, fm = ei.flows(eps, s, 20)
    out = ei.run(y, eps[s], fp, fm, "9999", "sparse", B=999)
    assert 0 <= out["discard_frac"] < 1 and "rho_drop_frac" in out
    for k in ("delta", "delta_lo", "delta_hi", "theta_EI", "kappa_plus", "kappa_minus"):
        assert np.isfinite(out[k]), k
    assert out["unstable"] == (out["discard_frac"] > 0.10)
    # 更極端：只有一個活躍日 → 很多重抽全為 0 → 剔除（不補抽）、比例 > 10% → 不穩定 → 只能歸混合型
    e1 = np.zeros(N + 40); e1[200] = 1.0
    fp1, fm1 = ei.flows(e1, s, 20)
    o1 = ei.run(y, e1[s], fp1, fm1, "9999", "one", B=999)
    assert o1["discard_frac"] > 0.10 and o1["unstable"]
    assert np.isfinite(o1["delta_lo"]) and np.isfinite(o1["delta_hi"])
    assert classify.classify(0.01, 0.5, o1["delta_lo"], o1["delta_hi"], o1["kappa_plus"], o1["unstable"]) == classify.MIXED
    assert classify.UNSTABLE_NOTE in classify.flags(classify.MIXED, 0.01, o1["unstable"])


def test_ei_zero_fminus_column_finite():
    rng = np.random.default_rng(4)
    N = 300
    eps = np.abs(rng.normal(0, 1, N + 40)) * (rng.random(N + 40) < 0.3)
    s = np.arange(N)
    fp, fm = ei.flows(eps, s, 10)
    assert np.all(fm == 0)
    y = rng.normal(0, 0.05, N)
    out = ei.run(y, eps[s], fp, fm, "x", "y", B=99)
    assert np.isfinite(out["theta_EI"]) and out["kappa_minus"] == 0.0


def test_ei_flows_use_residuals_up_to_T():
    T = 300
    eps = np.zeros(T)
    eps[T - 1] = 2.5
    eps[T - 41] = -1.0
    s = np.arange(T - 40)                            # 樣本截在 T − 40
    fp, fm = ei.flows(eps, s, 40)
    assert fp[-1] == 2.5 and fm[-1] == 0             # t = T − 40（索引 T−41）的 F 用到 ε̂_T
    fp10, _ = ei.flows(eps, s, 10)
    assert fp10[-1] == 0


def test_ei_seed_stable_and_deterministic():
    assert ei.seed_for("2601", "9A00", "EI") == ei.seed_for("2601", "9A00", "EI")
    assert ei.seed_for("2601", "9A00") != ei.seed_for("2601", "9A01")
    rng = np.random.default_rng(1)
    eps, s, y = _ei_data("timing", rng, N=400)
    fp, fm = ei.flows(eps, s, 10)
    a = ei.run(y, eps[s], fp, fm, "s", "b", B=99)
    b = ei.run(y, eps[s], fp, fm, "s", "b", B=99)
    assert a == b


def test_r_and_eplus_variants_run():
    d, pr, rng = _prep(T=900, seed=9)
    T = len(d)
    n = np.where(rng.random(T) < 0.3, rng.normal(0, 5e4, T).round(), 0.0)
    n[:150] = 0
    pn = _panel(d, n, pr.inputs["close"], pr.inputs["volume"], pr.inputs["adj"])
    sa = run.stock_arrays(pn, pr, None, t_cut=d[-1])
    act = n != 0
    a = run.branch_test(n, act, sa)
    b = run.branch_test(n, act, sa, eplus=True)
    assert 0 < a["E"]["p"] <= 1 and 0 < a["R"]["p"] <= 1 and 0 < b["E"]["p"] <= 1
    assert "R" not in b


# ───────────────────────── 步驟 6：BH 族群與分類（§5、§7） ─────────────────────────
def test_bh_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5, np.nan])
    q = classify.bh(p)
    ref = np.array([0.025, 0.05, 0.05, 0.025, 0.5])     # m = 5（NaN 不計）
    assert np.allclose(q[:5], ref) and np.isnan(q[5])


def test_family_excludes_dev_and_holdout():
    def row(sid, split, p):
        return {"sid": sid, "branch": f"{sid}-b", "split": split, "pE": p, "pR": p}
    results = [{"sid": "2601", "split": "dev", "rows": [row("2601", "dev", 1e-6)]},
               {"sid": "1111", "split": "holdout_xs", "rows": [row("1111", "holdout_xs", 1e-6)]},
               {"sid": "2222", "split": "tune", "rows": [row("2222", "tune", 0.01), row("2222", "tune", 0.2)]},
               {"sid": "3333", "split": "tune", "rows": [row("3333", "tune", 0.03)]}]
    run.assign_q(results)
    fam = run.family_members(results)
    assert {s for s, _ in fam} == {"2222", "3333"} and len(fam) == 3
    assert results[1]["rows"][0]["qE"] is None and results[1]["rows"][0]["family"] is None
    assert results[0]["rows"][0]["family"] == "dev:2601"
    assert results[2]["rows"][0]["qE"] == pytest.approx(0.03)          # m = 3：只有調參集
    calls = []

    def guard(s):
        calls.append(s)
        if s == "1111":
            from broker_flow_v3.split import HoldoutError
            raise HoldoutError(s)
    from broker_flow_v3.split import HoldoutError
    with pytest.raises(HoldoutError):
        run.finalize_stock(results[1], writable_check=guard)


def test_rolling_windows_have_no_p():
    rng = np.random.default_rng(2)
    N = 700
    x = np.where(rng.random(N) < 0.3, rng.normal(0, 1, N), 0.0)
    y = rng.normal(0, 0.05, N)
    rows = report.rolling(x, y, _dates(N))
    assert len(rows) == 1 + (N - 240) // 20
    for r in rows:
        assert set(r) == {"start", "end", "theta20", "t20"}
        assert not any("p" == k or k.startswith("p_") or k.startswith("q") for k in r)


def test_classification_exclusive_and_complete():
    grid_q = [0.001, 0.05, 0.051, 0.5, np.nan]
    grid_d = [-0.1, 0.0, 0.1, np.nan]
    seen = set()
    for qE in grid_q:
        for qR in grid_q:
            for lo in grid_d:
                for hi in grid_d:
                    for kp in (-1.0, 0.0, 1.0, np.nan):
                        for un in (False, True):
                            k = classify.classify(qE, qR, lo, hi, kp, un)
                            sE = np.isfinite(qE) and qE <= 0.05
                            sR = np.isfinite(qR) and qR <= 0.05
                            c1 = sE and not un and np.isfinite(lo) and lo >= 0
                            c2 = sE and not un and not c1 and np.isfinite(hi) and hi <= 0 and np.isfinite(kp) and kp > 0
                            c3 = sE and not c1 and not c2
                            c4 = (not sE) and sR
                            c5 = not (c1 or c2 or c3 or c4)
                            assert sum([c1, c2, c3, c4, c5]) == 1
                            want = [classify.TIMING, classify.PUMP, classify.MIXED, classify.CONTRARIAN, classify.NONE][[c1, c2, c3, c4, c5].index(True)]
                            assert k == want
                            seen.add(k)
    assert seen == set(classify.TYPES)
    assert classify.flags(classify.TIMING, 0.2)[0] == classify.CHASE_NOTE
    assert classify.flags(classify.CONTRARIAN, 0.2) == []


# ───────────────────────── 步驟 7：報告欄位（§8） ─────────────────────────
def test_exec_zero_when_prices_equal_vwap():
    rng = np.random.default_rng(0)
    T = 200
    vw = 10 + rng.random(T)
    b = rng.integers(0, 5, T) * 1000.0
    s = rng.integers(0, 5, T) * 1000.0
    bp = np.where(b > 0, vw, np.nan)
    sp = np.where(s > 0, vw, np.nan)
    assert report.exec_component(b, s, bp, sp, vw) == 0.0
    assert report.exec_component([2000], [1000], [9.0], [11.0], [10.0]) == pytest.approx(1000 * 1.0 + 0)


def test_last_action_pending_and_position_score():
    n = np.zeros(100); n[95] = 201000
    bp = np.full(100, 5.5)
    d = _dates(100)
    la = report.last_action(n, bp, bp, d)
    assert la["status"] == report.PENDING and "買進 201 張 @5.50" in la["text"]
    n2 = np.zeros(100); n2[10] = 1000
    assert report.last_action(n2, bp, bp, d)["status"] is None
    P = np.r_[np.linspace(10, 5, 50), np.linspace(5, 10, 50)]
    n3 = np.zeros(100); n3[49] = 1000; n3[95] = -1000
    assert report.position_score(n3, P) == pytest.approx(1.0)       # 只買在低點；最後 10 天排除


# ───────────────────────── 端到端（合成面板） ─────────────────────────
def test_end_to_end_synthetic_stock():
    rng = np.random.default_rng(42)
    T = 1500
    d = _dates(T)
    mkt, adj, peers, vol, e = _market(T, rng)
    pr = prep.compute(d, adj.copy(), adj, vol, mkt, peers)
    p0 = int(np.searchsorted(d, "2021-06-30"))
    dd = d[p0:]
    Tp = len(dd)
    cs = np.r_[0, np.cumsum(e[p0:])]
    fut = np.r_[cs[21:] - cs[1:-20], np.zeros(20)]
    nets = []
    trade = rng.random(Tp) < 0.35
    sg = np.sign(fut) * np.where(rng.random(Tp) < 0.8, 1, -1)
    nets.append(np.where(trade, sg * 200_000, 0.0))                  # 擇時分點（無衝擊）
    for _ in range(4):
        nets.append(np.where(rng.random(Tp) < 0.3, np.round(rng.normal(0, 1.5e5, Tp), -3), 0.0))
    nets.append(np.where(rng.random(Tp) < 0.01, 1000.0, 0.0))         # 不通過門檻
    nets = np.array(nets)
    pn = _panel(dd, nets, adj[p0:], vol[p0:], adj[p0:])
    res = run.analyze_stock("9999", pn=pn, pr=pr, events=pd.DataFrame(columns=["date", "kind", "value", "approx"]),
                            Q=np.zeros(Tp), split_label="tune", t_cut=dd[-1])
    assert res["n_eligible"] >= 4
    assert all(r["i"] != 5 for r in res["rows"])
    r0 = [r for r in res["rows"] if r["i"] == 0][0]
    assert r0["pE"] < 1e-4
    run.assign_q([res])
    doc = run.finalize_stock(res, "合成", writable_check=lambda s: None, B=199)
    txt = run.dumps(doc)
    back = json.loads(txt)
    assert back["disclaimer"] == run.DISCLAIMER and back["version"] == "1.1.2" and back["split"] == "tune"
    row = [x for x in back["rows"] if x["branch"] == "b0"][0]
    assert row["test"]["kind"] in ("timing", "mixed")
    assert set(row) >= {"test", "descriptive"}
    assert "pE" in row["test"] and "pE" not in row["descriptive"]
    for k in ("timing_pnl", "exec", "pnl", "position_score", "holdings", "trend", "last_action", "rho", "h_star", "cycles", "G_eff"):
        assert k in row["descriptive"], k
    assert row["descriptive"]["exec"] == 0                             # BP = SP = VWAP
    assert row["descriptive"]["timing_pnl"] > 0
    assert "p" not in (row["descriptive"]["trend"] or {})


def test_validation_helpers_smoke():
    stats = {"A": {"industry": "x", "log_mcap": 10, "log_value": 8, "beta_m": 1.0},
             "B": {"industry": "x", "log_mcap": 10, "log_value": 8, "beta_m": 1.0},
             "C": {"industry": "y", "log_mcap": 11, "log_value": 8, "beta_m": 1.1},
             "D": {"industry": "z", "log_mcap": 15, "log_value": 12, "beta_m": 0.2},
             "H": {"industry": "y", "log_mcap": 10, "log_value": 8, "beta_m": 1.0}}
    assert validate.match_placebo("A", stats, exclude={"H"}) == "C"
    rng = np.random.default_rng(0)
    su = validate.super_uniform(rng.random(5000))
    assert su["ok"]
    adj = np.full(200, 10.0) * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    dlt = np.zeros(200); dlt[100] = 1e5
    a2 = validate.impact(adj, np.full(200, 1e6), dlt)
    assert np.allclose(a2[:100], adj[:100]) and np.all(a2[100:] > adj[100:])
    # 注入（少量重複，只確認流程可執行）
    T = 1500
    d = _dates(T)
    mkt, adj, peers, vol, e = _market(T, rng)
    pr = prep.compute(d, adj.copy(), adj, vol, mkt, peers)
    p0 = int(np.searchsorted(d, "2021-06-30"))
    host_n = np.where(rng.random(T - p0) < 0.3, np.round(rng.normal(0, 1e5, T - p0), -3), 0.0)
    host = {"pr": pr, "dates": d[p0:], "suspended": np.zeros(T - p0, bool), "n": host_n, "events": None,
            "close": adj[p0:], "volume": vol[p0:], "sid": "9999", "branch": "host"}
    for g in ("timing", "impact", "pump_dump", "contrarian"):
        out = validate.inject(host, g, 0.05, 0.3, 0.7, reps=2, seed=1, t_cut=d[-1], B=49)
        assert out["reps"] == 2 and "summary" in out
        if g != "timing":
            assert out["summary"]["note"] == "測試無效"


def test_stock_prep_production_path_with_injected_sources(monkeypatch, tmp_path):
    """stock_prep：價格、日曆、產業成員都以替身注入（不碰 FinMind／parquet）；大盤以 market_fn 注入；結果快取"""
    from broker_flow_v3 import auxdata, data_layer
    rng = np.random.default_rng(8)
    T = 600
    d = _dates(T)
    mkt, adj, peers, vol, _ = _market(T, rng)
    px = {s: pd.DataFrame({"date": d, "close": a, "adj_close": a, "volume": vol}) for s, a in
          [("1101", adj)] + [(f"P{k}", peers[k]) for k in range(6)]}
    monkeypatch.setattr(data_layer, "prices", lambda s: px[s])
    monkeypatch.setattr(auxdata, "trading_calendar", lambda: list(d))
    info = pd.DataFrame({"stock_id": list(px), "industry_category": ["水泥"] * 7})
    ind = prep.Industry(list(px), info, adj_fn=lambda s: pd.Series(px[s]["adj_close"].to_numpy(), index=d))
    mf = lambda s: pd.Series(mkt, index=d)  # noqa: E731
    pr = prep.stock_prep("1101", d[-1], ind, mf, tmp_path)
    ref = prep.compute(d, adj, adj, vol, mkt, peers)
    assert pr.industry_factor and pr.n_peers == 6
    np.testing.assert_allclose(pr.rex[20], ref.rex[20], equal_nan=True)
    assert (tmp_path / "prep_1101.pkl").exists()
    pr2 = prep.stock_prep("1101", d[-1], ind, lambda s: (_ for _ in ()).throw(AssertionError("應走快取")), tmp_path)
    np.testing.assert_array_equal(pr2.beta_m, pr.beta_m)
    with pytest.raises(RuntimeError):
        prep.stock_prep("1101", d[-2], ind, lambda s: pd.Series(dtype=float), None)


def test_placebo_stats_builder_and_doc_size_cap():
    d, pr, _ = _prep(T=1500, seed=12)
    st = validate.placebo_stats(["A"], lambda s: pr, lambda s: "水泥", shares_of=lambda s: 1e8, start=d[200], end=d[-1])
    a = st["A"]
    m = (pr.dates >= d[200]) & pr.valid
    assert a["industry"] == "水泥" and a["log_mcap"] == pytest.approx(np.log(1e8 * pr.close[m].mean()))
    assert a["log_value"] == pytest.approx(np.log(np.mean(pr.close[m] * pr.volume[m]) + 1))
    assert abs(a["beta_m"] - 1.0) < 0.2                                   # 合成資料 β^m = 1
    st2 = validate.placebo_stats(["A"], lambda s: pr, lambda s: "x", shares_of=lambda s: None, start=d[200], end=d[-1])
    assert not np.isfinite(st2["A"]["log_mcap"])
    assert validate.match_placebo("A", {**st2, "B": {"industry": "y", "log_mcap": 1, "log_value": 1, "beta_m": 1}}) is None
    big = {"stock_id": "9", "rows": [{"branch": f"b{k}", "test": {"pE": 0.01}, "descriptive": {
        "marks": [["2026-01-01", 1.0]] * 3000, "rolling": [{"end": "x", "theta20": 0.1, "t20": 1.0}] * 300}} for k in range(10)]}
    out = run.cap_size(big, limit=300_000)
    assert len(run.dumps(out).encode()) <= 300_000 and len(out["rows"]) == 10 and out["truncated"]
    assert all("pE" in r["test"] for r in out["rows"])
    assert out["rows"][0]["descriptive"]["marks"]                          # 由最後的分點起刪
