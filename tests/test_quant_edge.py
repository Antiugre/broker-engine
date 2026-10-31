"""python -m pytest tests/test_quant_edge.py -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quant_edge as qe  # noqa: E402
from quant_edge import EdgeParams  # noqa: E402

NOCOST = EdgeParams(fee_rate=0, tax_swing=0, tax_daytrade=0, slip_ticks=0, ghost_cap_days=1e9)


def cal(n, start="2026-01-05"):
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]


def prices(days, px=100.0, vol=10_000):
    return {d: {"vwap": px if not callable(px) else px(i), "close": px if not callable(px) else px(i), "vol": vol} for i, d in enumerate(days)}


def test_tick_size_table():
    s = pd.Series([9.99, 10, 49.95, 50, 99.9, 100, 499.5, 500, 999, 1000])
    assert qe.tick_size(s).tolist() == [0.01, 0.05, 0.05, 0.1, 0.1, 0.5, 0.5, 1, 1, 5]


def test_fifo_vs_lifo_dual_track():
    d = cal(3)
    rows = [{"date": d[0], "buy": 10, "sell": 0}, {"date": d[1], "buy": 10, "sell": 0}, {"date": d[2], "buy": 0, "sell": 10}]
    px = {d[0]: {"vwap": 100, "close": 100, "vol": 1e4}, d[1]: {"vwap": 120, "close": 120, "vol": 1e4}, d[2]: {"vwap": 110, "close": 110, "vol": 1e4}}
    df = qe.build_frame(rows, px, NOCOST)
    f, f_open, _ = qe.match_inventory(df, "fifo", NOCOST)
    l, l_open, _ = qe.match_inventory(df, "lifo", NOCOST)
    assert f.iloc[0]["buy_px"] == 100 and f.iloc[0]["pnl"] > 0      # FIFO：賣掉最早那批 → 賺
    assert l.iloc[0]["buy_px"] == 120 and l.iloc[0]["pnl"] < 0      # LIFO：賣掉最新那批 → 賠（短線主力真實結果）
    assert f_open[0][3] == 120 and l_open[0][3] == 100


def test_daytrade_split_and_tax():
    d = cal(1)
    p = EdgeParams(slip_ticks=0)
    df = qe.build_frame([{"date": d[0], "buy": 30, "sell": 20}], prices(d), p)
    t, rem, _ = qe.match_inventory(df, "fifo", p)
    dt_ = t[t["kind"] == "daytrade"].iloc[0]
    assert dt_["qty"] == 20 and dt_["hold"] == 0
    assert dt_["ret"] == pytest.approx((100 * (1 - p.fee - 0.0015) - 100 * (1 + p.fee)) / (100 * (1 + p.fee)))
    assert rem[0][2] == 10                                          # 淨買 10 張進庫存


def test_slippage_ticks():
    d = cal(2)
    p = EdgeParams(slip_ticks=2, fee_rate=0, tax_swing=0, ghost_cap_days=1e9)
    rows = [{"date": d[0], "buy": 1, "sell": 0}, {"date": d[1], "buy": 0, "sell": 1}]
    t, _, _ = qe.match_inventory(qe.build_frame(rows, prices(d, 100), p), "fifo", p)
    assert t.iloc[0]["buy_px"] == 101 and t.iloc[0]["sell_px"] == 99    # 100 元的升降單位 0.5，2 檔 = 1 元


def test_ghost_inventory_settle_drop_and_cap():
    d = cal(10)
    rows = [{"date": d[0], "buy": 50, "sell": 0}]
    p = EdgeParams(ghost_days=5, ghost_mode="settle", fee_rate=0, tax_swing=0, slip_ticks=0, ghost_cap_days=1e9)
    t, rem, ev = qe.match_inventory(qe.build_frame(rows, prices(d), p), "fifo", p)
    assert ev["ghost"] == 50 and not rem and (t["kind"] == "ghost").all()
    p2 = EdgeParams(ghost_days=5, ghost_mode="drop", ghost_cap_days=1e9)
    t2, rem2, ev2 = qe.match_inventory(qe.build_frame(rows, prices(d), p2), "fifo", p2)
    assert ev2["ghost"] == 50 and t2.empty and not rem2
    p3 = EdgeParams(ghost_cap_days=2)                                 # 20 日均量 100 張 × 2 = 上限 200 張
    rows3 = [{"date": x, "buy": 80, "sell": 0} for x in d]
    _, rem3, ev3 = qe.match_inventory(qe.build_frame(rows3, prices(d, vol=100), p3), "fifo", p3)
    assert sum(q for *_, q, _ in [(a, b, c, e) for a, b, c, e in rem3]) == pytest.approx(200) and ev3["cap"] == pytest.approx(600)


def test_bayesian_shrinkage_beats_small_sample():
    prior = {"p0": 0.5, "alpha": 10, "beta": 10, "win": 0.05, "loss": 0.05}
    small = qe.bayes({"n": 2, "w": 2, "avg_win": 0.30, "avg_loss": None}, prior)
    big = qe.bayes({"n": 60, "w": 45, "avg_win": 0.06, "avg_loss": 0.04}, prior)
    assert small["pw"] == pytest.approx(12 / 22) and big["pw"] == pytest.approx(55 / 80)
    assert small["lo"] < big["lo"]
    assert big["ev"] == pytest.approx(55 / 80 * (45 * 0.06 + 0.5) / 55 - 25 / 80 * (15 * 0.04 + 0.5) / 25)


def test_empirical_prior_uses_pool():
    s = [{"n": 40, "w": 24, "avg_win": 0.06, "avg_loss": 0.04}, {"n": 20, "w": 8, "avg_win": 0.03, "avg_loss": 0.05}]
    pr = qe.empirical_prior(s, EdgeParams(prior_strength=20))
    assert pr["p0"] == pytest.approx(32 / 60) and pr["alpha"] == pytest.approx(20 * 32 / 60)
    assert qe.empirical_prior(s[:1])["p0"] == 0.5                      # 母體 < 50 筆 → 預設先驗


def test_dynamic_threshold_amount_or_volume_share():
    d = cal(5)
    rows = [{"date": x, "buy": 40, "sell": 0} for x in d[-3:]]          # 近 3 日 120 張
    cheap = qe.recent_flow(qe.build_frame(rows, prices(d, 20, vol=2_000)))       # 金額 240 萬，但佔量 2%
    pricey = qe.recent_flow(qe.build_frame(rows, prices(d, 900, vol=50_000)))    # 金額 1.08 億
    thin = qe.recent_flow(qe.build_frame(rows, prices(d, 20, vol=1_000)))        # 佔量 4%
    assert not cheap["big"] and pricey["big"] and thin["big"]
    assert pricey["amt3"] == pytest.approx(120 * 900 * 1000) and thin["vs3"] == pytest.approx(0.04)
    both = qe.recent_flow(qe.build_frame(rows, prices(d, 900, vol=50_000)), EdgeParams(threshold_mode="all"))
    assert not both["big"]


def test_stealth_accumulation_detects_quiet_buyer_and_rejects_spike():
    d = cal(20)
    quiet = [{"date": x, "buy": 60 + (i % 3) * 5, "sell": 5} for i, x in enumerate(d) if i % 5 != 4]
    s = qe.stealth(qe.build_frame(quiet, prices(d, lambda i: 100 + i * 0.3, vol=2_000)))
    assert s["flag"] and s["buy_days"] == 16 and s["day_share"] <= 0.05
    spike = [{"date": d[-1], "buy": 900, "sell": 0}] + [{"date": x, "buy": 3, "sell": 0} for x in d[:-1]]
    s2 = qe.stealth(qe.build_frame(spike, prices(d, vol=2_000)))
    assert not s2["flag"] and not s2["checks"]["day_share"] and not s2["checks"]["top_day"]


def test_analyse_finalize_signal():
    d = cal(300, "2025-01-02")
    rows, px = [], {}
    for i, x in enumerate(d):
        px[x] = {"vwap": 100 + (i % 10), "close": 100 + (i % 10), "vol": 3_000}
    for i in range(0, 280, 10):                                      # 在 100 元買、109 元賣 → 每筆都賺
        rows += [{"date": d[i], "buy": 20, "sell": 0}, {"date": d[i + 9], "buy": 0, "sell": 20}]
    rows += [{"date": x, "buy": 60, "sell": 0} for x in d[-3:]]      # 近 3 日買 180 張（佔量 2%、金額約 1,900 萬）
    a = qe.analyse(rows, px)
    prior = qe.empirical_prior([a["lifo"]])
    f = qe.finalize(a, prior, EdgeParams(min_amount=10_000_000))
    assert f["edge"] and f["signal"] and f["n"] == 28 and f["pw"] == pytest.approx(38 / 48, abs=1e-4) and f["ev"] > 0.04
    assert f["lifo"]["ol"] == 180 and f["style"] == "短波段"
    assert not qe.finalize(a, prior, EdgeParams())["signal"]         # 預設門檻 3,000 萬、3% → 不成立


def test_run_end_to_end(tmp_path, monkeypatch):
    import json
    import quant
    d = cal(300, "2025-01-02")
    px = {x: {"vwap": 100 + (i % 10), "close": 100 + (i % 10), "vol": 3_000} for i, x in enumerate(d)}
    rows = []
    for i in range(0, 280, 10):
        rows += [{"date": d[i], "buy": 20, "sell": 0}, {"date": d[i + 9], "buy": 0, "sell": 20}]
    rows += [{"date": x, "buy": 120, "sell": 0} for x in d[-3:]]
    day = d[-1]
    monkeypatch.setattr(quant, "DATA_DIR", tmp_path)
    monkeypatch.setattr(qe, "EDGE_DIR", tmp_path / "edge")
    (tmp_path / day).mkdir()
    (tmp_path / day / "2330.json").write_text(json.dumps({"stock_id": "2330", "stock_name": "台積電", "buy_side": [
        {"broker": "凱基-台北", "broker_id": "9200", "b_raw": "9268", "branch_id": "9268"}], "sell_side": []}))
    monkeypatch.setattr(qe, "stealth_candidates", lambda *a, **k: {})
    monkeypatch.setattr(quant, "load_prices", lambda sid, end, today=None: px)
    monkeypatch.setattr(quant, "load_branch_history", lambda f, sid, row, end, today=None, **k: rows)

    class F:
        def close(self): pass
    out = qe.run(day, ["2330"], fetcher=F())
    st = out["stocks"]["2330"]["9268"]
    assert st["edge"] and st["signal"] and out["signals"][0]["key"] == "9268"
    assert (tmp_path / "edge" / f"{day}.json").exists()
    json.loads((tmp_path / "edge" / f"{day}.json").read_text())   # 可序列化（numpy 型別已轉換）
