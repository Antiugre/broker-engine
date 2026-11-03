"""python -m pytest tests/test_broker_flow.py -q（主力集團偵測：規格書第 12 節的驗收測試中，本版做得到的部分）"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from broker_flow import accounting, dim_a, features  # noqa: E402
from broker_flow.data_layer import HoldoutError, Panel  # noqa: E402
from broker_flow.fusion import bh, cct, components  # noqa: E402
from broker_flow.validation import inject  # noqa: E402


def _panel(B, S, vwap=None, holdout=None, btype=None):
    n, T = B.shape
    vwap = np.full(T, 10.0) if vwap is None else np.asarray(vwap, float)
    dates = np.array([f"2024-{1 + t // 28:02d}-{1 + t % 28:02d}" for t in range(T)])
    return Panel("9999", dates, [f"b{i}" for i in range(n)], [f"分點{i}" for i in range(n)], btype or ["brokerage"] * n,
                 np.asarray(B, float), np.asarray(S, float), vwap.copy(), vwap, np.full(T, 1e6), holdout or T)


def test_poisson_binomial_matches_brute_force():
    rng = np.random.default_rng(1)
    for T in (3, 7, 12):
        q = rng.random((4, T))
        obs = rng.integers(0, T + 1, 4)
        got = dim_a.poisson_binomial_sf(q, obs)
        for k in range(4):
            tot = 0.0
            for bits in itertools.product([0, 1], repeat=T):
                if sum(bits) >= obs[k]:
                    tot += np.prod([q[k, t] if b else 1 - q[k, t] for t, b in enumerate(bits)])
            assert got[k] == pytest.approx(tot, rel=1e-9, abs=1e-12)


def test_cct_tiny_p_stable_and_missing_dimension():
    p = cct([np.array([1e-20, 1e-9, 0.3]), np.array([0.5, 0.5, 0.3])], [0.5, 0.5])
    assert np.all(np.isfinite(p)) and np.all(p > 0) and p[0] < p[1] < p[2]
    assert p[2] == pytest.approx(0.3, rel=1e-9)                       # 相同 p 值 → CCT 不變
    one = cct([np.array([0.01]), np.array([np.nan])], [0.5, 0.5])
    assert one[0] == pytest.approx(0.01, rel=1e-9)                     # 缺失的維度移除，剩下的平分權重


def test_bh_matches_reference():
    rng = np.random.default_rng(0)
    p = rng.random(200) ** 3
    q = bh(p)
    try:
        from statsmodels.stats.multitest import multipletests
        ref = multipletests(p, method="fdr_bh")[1]
    except ImportError:
        o = np.argsort(p)
        r = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        ref = np.empty_like(p); ref[o] = np.minimum(r, 1)
    assert np.allclose(q, ref)


def test_accounting_hand_cases():
    # 第 0 天買 10 張、第 1 天當沖 5 張、第 2 天賣 4 張、第 3 天賣 20 張（超賣 → uncovered）
    B = np.array([[10000, 5000, 0, 0]])
    S = np.array([[0, 5000, 4000, 20000]])
    pn = _panel(B, S, vwap=[10, 11, 12, 13])
    L = accounting.run(pn)
    assert L["inventory"][0].tolist() == [10000, 10000, 6000, 0]
    assert L["avg_cost"][0, 1] == pytest.approx(10.0) and L["dt"][0, 1] == 5000
    assert L["realized"][0, 2] == pytest.approx(4000 * (12 - 10) - 4000 * 12 * (0.001425 + 0.003))
    assert L["uncovered"][0, 3] == 14000 and L["left_censored"][0, 3] and not L["left_censored"][0, 2]


def test_no_lookahead_features():
    rng = np.random.default_rng(3)
    B = rng.integers(0, 50, (6, 300)) * 1000.0
    S = rng.integers(0, 50, (6, 300)) * 1000.0
    pn = _panel(B, S)
    t = 200
    sc, ac, z = features.scale_and_active(pn)
    cls = features.classify(pn, ac, t)
    B2, S2 = B.copy(), S.copy()
    B2[:, t + 1:] = rng.integers(0, 500, B2[:, t + 1:].shape) * 1000.0
    S2[:, t + 1:] = 0
    pn2 = _panel(B2, S2)
    sc2, ac2, z2 = features.scale_and_active(pn2)
    assert np.array_equal(sc[:, :t + 1], sc2[:, :t + 1], equal_nan=True)
    assert np.array_equal(ac[:, :t + 1], ac2[:, :t + 1]) and np.allclose(z[:, :t + 1], z2[:, :t + 1])
    assert cls == features.classify(pn2, ac2, t)


def test_behavior_classes():
    T = 200
    B = np.zeros((3, T)); S = np.zeros((3, T))
    B[0] = S[0] = 20000                                        # 當沖型：每天買賣相同
    B[1, ::2] = 10000                                         # 隔日沖型：平常小買，每 10 天大買一次、隔天賣光
    for e in range(1, T - 4, 10):
        B[1, e] = 60000; S[1, e + 1] = 60000
    B[2, ::3] = 8000; S[2, 1::7] = 3000                       # 波段型
    pn = _panel(B, S)
    _, ac, _ = features.scale_and_active(pn)
    assert features.classify(pn, ac, T - 1) == [features.DAYTRADE, features.OVERNIGHT, features.SWING]


def test_holdout_guard():
    pn = _panel(np.ones((2, 50)), np.zeros((2, 50)), holdout=40)
    pn.guard(39)
    with pytest.raises(HoldoutError):
        pn.guard(40)
    pn.guard(45, allow_holdout=True)


def test_components_and_injection_conservation():
    assert components(6, [(0, 1), (1, 2), (4, 5)]) == [[0, 1, 2], [4, 5]]
    rng = np.random.default_rng(0)
    B = rng.integers(0, 30, (5, 80)) * 1000.0
    S = rng.integers(0, 30, (5, 80)) * 1000.0
    q, codes = inject(_panel(B, S), 2, 3, 0.5, rng)
    assert len(q.branches) == 7 and "b2" not in q.branches and codes == [4, 5, 6]
    assert np.allclose(q.B.sum(axis=0), B.sum(axis=0)) and np.allclose(q.S.sum(axis=0), S.sum(axis=0))


def test_planted_coactivity_detected():
    """兩個分點在同一批日子一起大買 → p_A 很小；獨立的分點 → 不顯著"""
    rng = np.random.default_rng(5)
    T, n = 120, 8
    A = rng.random((n, T)) < 0.25
    A[1] = A[0]
    Z = np.where(A, rng.normal(1, 1, (n, T)), 0.0)
    Z[1] = np.where(A[1], Z[0] + rng.normal(0, .2, T), 0)
    F, ret = dim_a.day_factors(np.cumprod(1 + rng.normal(0, .02, T)) * 10, np.full(T, 1e6))
    r = dim_a.run_window(A, Z, F, ret)
    k = [i for i, (a, b) in enumerate(r["pairs"]) if (a, b) == (0, 1)][0]
    assert r["p_A"][k] < 1e-6
    q = bh(r["p_A"])
    assert (q <= 0.05).sum() <= 2
