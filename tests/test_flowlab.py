"""python -m pytest tests/test_flowlab.py -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flowlab  # noqa: E402
from flowlab import cluster, core, schema, sources  # noqa: E402

N = 200
DATES = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-05", periods=N)]


def mkprices(close=None, vol=10_000):
    close = close if close is not None else np.full(N, 100.0)
    return {d: {"vwap": float(c), "close": float(c), "vol": vol} for d, c in zip(DATES, close)}


def ctx_of(hist, prices=None, meta=None, company=None, params=None):
    branches = {k: {"broker": (meta or {}).get(k, {}).get("broker", k), "stock_name": "測試",
                    **(meta or {}).get(k, {})} for k in hist}
    return sources.build_context("9999", DATES[-1], branches, hist, prices or mkprices(), company, {}, params or {})


def rows(pairs):
    return [{"date": DATES[i], "buy": b, "sell": s} for i, b, s in pairs]


# ── 模組一 ──
def test_overnight_flipper():
    h = {"flip": rows([(i, 1500, 0) for i in (150, 170, 190)] + [(i + 1, 0, 1400) for i in (150, 170, 190)]),
         "hold": rows([(150, 1500, 0), (151, 0, 100)])}
    doc = flowlab.analyse_context(ctx_of(h))
    t = {b["key"]: [x["type"] for x in b["tags"]] for b in doc["tags"]}
    assert "overnight" in t["flip"] and "overnight" not in t.get("hold", [])


def test_swing_lock_stable_vs_erratic():
    stable = rows([(i, 60 + (i % 3) * 5, 5) for i in range(188, 200) if i % 6 != 0])
    erratic = rows([(i, 400 if i % 4 == 0 else 5, 2) for i in range(188, 200)])
    doc = flowlab.analyse_context(ctx_of({"stable": stable, "erratic": erratic}))
    t = {b["key"]: [x["type"] for x in b["tags"]] for b in doc["tags"]}
    assert "lock" in t["stable"] and "lock" not in t.get("erratic", [])


def test_igniter_quiet_then_burst():
    quiet = rows([(150, 3, 0), (170, 2, 0), (198, 400, 0)])
    busy = rows([(i, 300, 250) for i in range(130, 199)] + [(198, 2000, 0)])
    doc = flowlab.analyse_context(ctx_of({"quiet": quiet, "busy": busy}))
    t = {b["key"]: [x["type"] for x in b["tags"]] for b in doc["tags"]}
    assert "igniter" in t["quiet"] and "igniter" not in t.get("busy", [])


def test_new_tagger_plugs_in_without_touching_pipeline():
    @core.register_tagger
    class BigSeller(core.Tagger):
        key, label, active_days = "dumper", "大賣", 5

        def mask(self, ctx):
            return ctx.net <= -1000
    try:
        doc = flowlab.analyse_context(ctx_of({"x": rows([(198, 0, 1500)])}))
        assert doc["tag_counts"]["dumper"] == 1 and doc["tags"][0]["tags"][0]["label"] == "大賣"
    finally:
        core.TAGGERS.pop("dumper")


# ── 模組二 ──
def test_cluster_coordinated_buyers_before_launch():
    close = np.full(N, 100.0)
    launches = [60, 100, 140, 180]
    for L in launches:
        close[L:] *= 1.12  # 發動
    hist = {"a": [], "b": [], "c": [], "solo": []}
    for L in launches:
        hist["a"] += rows([(L - 3, 400, 0)])
        hist["b"] += rows([(L - 2, 350, 0)])                  # ±1 日內
        hist["c"] += rows([(L - 3, 300, 0)])
    hist["solo"] = rows([(i, 300, 0) for i in (20, 45, 120)])
    meta = {"a": {"city": "新竹市", "district": "東區"}, "b": {"city": "台北市"}}
    doc = flowlab.analyse_context(ctx_of(hist, mkprices(close), meta, company={"city": "新竹市", "district": "東區"}))
    cl = doc["clusters"][0]
    assert {m["key"] for m in cl["members"]} == {"a", "b", "c"} and cl["basis"] == "pre_launch"
    assert cl["co_buys"] >= 8 and cl["cohesion"] == pytest.approx(1.0)
    assert {m["key"]: m["geo"] for m in cl["members"]}["a"] == 1.3 and cl["geo_weight"] > 1


def test_co_buy_matrix_math():
    X = pd.DataFrame({"i": [1, 0, 1, 0, 0], "j": [0, 1, 0, 0, 0], "k": [0, 0, 0, 0, 1]}, dtype=bool)
    Cs, n, J = cluster.co_buy(X, pd.Series([True] * 5), tol=1)
    # i 在第 0、2 日、j 在第 1 日（±1 日內）→ 雙向各至少 1 次；k 在第 4 日與 i 相隔 2 日 → 不算
    assert Cs[0, 1] == 1 and Cs[0, 2] == 0 and n.tolist() == [2, 1, 1]
    assert J[0, 1] == pytest.approx(1 / (2 + 1 - 1))


def test_geo_matcher_zip_district_city():
    g = cluster.GeoMatcher({"zip": "300", "city": "新竹市", "district": "東區"}, {("新竹市", "北區"): "300"})
    assert g.weight({"city": "新竹市", "district": "北區"}) == 1.5      # 查表後同郵遞區號
    assert g.weight({"city": "新竹市", "district": "東區", "zip": "300"}) == 1.5
    assert g.weight({"city": "新竹市", "district": "香山區"}) == 1.1
    assert g.weight({"city": "台北市"}) == 1.0
    assert cluster.GeoMatcher.parse("30078新竹市東區力行二路1號")["zip"] == "300"


# ── 模組三 ──
def test_cost_zone_series_is_plain_json_and_weighted():
    close = np.linspace(100, 120, N)
    stable = rows([(i, 100, 5) for i in range(170, 200) if i % 5 != 0])
    doc = flowlab.analyse_context(ctx_of({"lock": stable}, mkprices(close)))
    z = doc["zones"]
    s = z["series"][-1]
    assert set(s) == {"date", "price_upper", "price_lower", "price_mid", "lots", "signal_type"}
    win = [i for i in range(180, 200) if i % 5 != 0]
    expect = np.average(close[win], weights=[95] * len(win))
    assert s["price_mid"] == pytest.approx(expect, abs=0.01) and s["price_lower"] < s["price_mid"] < s["price_upper"]
    assert s["signal_type"] == "lock" and z["latest"]["position"] in ("above", "inside", "below")
    json.dumps(doc)


# ── 模組四 ──
def test_low_level_absorption_divergence():
    close = np.full(N, 100.0)
    close[175:] = np.linspace(100, 78, N - 175)            # 20 日跌超過 15%
    lock = rows([(i, 30, 2) for i in range(120, 196) if i % 6 != 0] + [(i, 300, 0) for i in range(196, 200)])
    doc = flowlab.analyse_context(ctx_of({"lock": lock}, mkprices(close)))
    assert any(e["type"] == "low_absorb" for e in doc["divergence"]["active"])


def test_high_level_distribution_divergence():
    close = np.full(N, 100.0)
    close[175:] = np.linspace(100, 130, N - 175)
    lock = rows([(i, 40, 2) for i in range(120, 194) if i % 6 != 0] + [(i, 0, 400) for i in range(195, 200)])
    retail = rows([(i, 380, 0) for i in range(195, 200)])
    foreign = rows([(i, 30, 0) for i in range(195, 200)])
    meta = {"foreign": {"broker": "美商高盛"}}
    doc = flowlab.analyse_context(ctx_of({"lock": lock, "retail": retail, "foreign": foreign}, mkprices(close), meta))
    act = [e for e in doc["divergence"]["active"] if e["type"] == "high_distrib"]
    assert act and act[-1]["retail5"] >= 0.6 and act[-1]["lock_net5"] < 0


# ── 模組五 ──
def test_document_schema_and_index():
    doc = flowlab.analyse_context(ctx_of({"x": rows([(i, 60, 5) for i in range(185, 200)])}))
    assert {"schema", "stock_id", "asof", "price", "tag_counts", "tags", "clusters", "zones", "divergence"} <= set(doc)
    assert doc["asof"] == DATES[-1] and "errors" not in doc
    idx = schema.index_entry(doc)
    assert set(idx) == {"name", "tags", "alerts", "clusters", "zone"}
    assert len(json.dumps(doc).encode()) < 1_000_000


def test_horizon_slice_keeps_warmup_but_trims_output():
    close = np.linspace(100, 130, N)
    stable = rows([(i, 100, 5) for i in range(100, 200) if i % 5 != 0])
    full = ctx_of({"lock": stable}, mkprices(close))
    start = DATES[150]
    sub = full.slice(start, warmup=40, horizon="3m")
    assert len(sub.px) == 50 + 40 and sub.start == pd.Timestamp(start)
    doc = flowlab.analyse_context(sub)
    assert doc["horizon"] == "3m" and doc["range_start"] == start
    assert doc["zones"]["series"][0]["date"] >= start
    long = flowlab.analyse_context(full.slice(DATES[0], horizon="all"))
    assert len(long["zones"]["series"]) > len(doc["zones"]["series"])
