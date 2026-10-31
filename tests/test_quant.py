"""python -m pytest tests/test_quant.py -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quant  # noqa: E402
from quant import Params, detect_signal, evaluate, fifo_match, performance, unrealized  # noqa: E402

NOFEE = Params(fee_rate=0, tax_rate=0)


def rows(*items):
    return [{"date": d, "buy": b, "sell": s} for d, b, s in items]


def test_fifo_basic_pairs_oldest_first():
    r = rows(("2026-01-02", 10, 0), ("2026-01-05", 5, 0), ("2026-01-10", 0, 12))
    px = {"2026-01-02": 100, "2026-01-05": 110, "2026-01-10": 120}
    m = fifo_match(r, px, NOFEE)
    assert [(t.open_date, t.qty) for t in m.trades] == [("2026-01-02", 10), ("2026-01-05", 2)]
    assert m.trades[0].pnl == 10 * 1000 * 20 and m.trades[0].hold_days == 8
    assert m.trades[1].ret == pytest.approx(10 / 110)
    assert len(m.open_lots) == 1 and m.open_lots[0].qty == 3 and m.open_lots[0].price == 110


def test_unmatched_sell_and_intraday_net_mode():
    r = rows(("2026-01-02", 0, 7), ("2026-01-03", 20, 15), ("2026-01-06", 0, 5))
    px = {"2026-01-02": 50, "2026-01-03": 51, "2026-01-06": 49}
    m = fifo_match(r, px, NOFEE)
    assert m.unmatched_sell == 7          # 區間前的部位，不計入
    assert m.intraday == 15                # 同日 20 買 15 賣 → 15 張視為當沖
    assert len(m.trades) == 1 and m.trades[0].qty == 5 and m.trades[0].pnl < 0
    assert not m.open_lots


def test_gross_mode_sells_before_buys():
    r = rows(("2026-01-02", 10, 0), ("2026-01-03", 4, 6))
    px = {"2026-01-02": 10, "2026-01-03": 12}
    m = fifo_match(r, px, Params(fee_rate=0, tax_rate=0, match_mode="gross"))
    assert [(t.open_date, t.qty) for t in m.trades] == [("2026-01-02", 6)]
    assert sum(l.qty for l in m.open_lots) == 8


def test_fees_and_tax_turn_flat_trade_into_loss():
    r = rows(("2026-01-02", 1, 0), ("2026-01-03", 0, 1))
    px = {"2026-01-02": 100, "2026-01-03": 100}
    t = fifo_match(r, px).trades[0]
    assert t.pnl == pytest.approx(1000 * (100 * (1 - 0.001425 - 0.003) - 100 * 1.001425))
    assert not t.win


def test_missing_price_uses_previous():
    r = rows(("2026-01-02", 1, 0), ("2026-01-05", 0, 1))
    m = fifo_match(r, {"2026-01-02": 10}, NOFEE)
    assert m.missing_price_days == 1 and m.trades[0].sell_price == 10


def test_performance_metrics():
    r = rows(("2026-01-02", 10, 0), ("2026-01-12", 0, 10), ("2026-02-02", 10, 0), ("2026-02-04", 0, 10))
    px = {"2026-01-02": 100, "2026-01-12": 110, "2026-02-02": 100, "2026-02-04": 95}
    perf = performance(fifo_match(r, px, NOFEE).trades)
    assert perf["trades"] == 2 and perf["wins"] == 1 and perf["win_rate"] == 0.5
    assert perf["realized_pnl"] == 10 * 1000 * 10 - 10 * 1000 * 5
    assert perf["avg_ret"] == pytest.approx((0.10 - 0.05) / 2)
    assert perf["avg_hold_days"] == 6 and perf["style"] == "短波段"
    assert perf["profit_factor"] == pytest.approx(2.0)
    assert performance([])["win_rate"] is None


def test_unrealized():
    lots = fifo_match(rows(("2026-01-02", 10, 0), ("2026-01-05", 30, 0)), {"2026-01-02": 100, "2026-01-05": 120}, NOFEE).open_lots
    u = unrealized(lots, 130, NOFEE)
    assert u["open_lots"] == 40 and u["open_cost"] == 115 and u["open_since"] == "2026-01-02"
    assert u["unrealized_pnl"] == 40 * 1000 * 15
    assert unrealized([], 100)["open_cost"] is None


def _winner_history(extra):
    """6 筆獲利短波段 + 1 筆虧損（勝率 6/7 ≈ 85.7%）+ 最後幾天的資料"""
    items, px, d = [], {}, 0
    import datetime as dt
    base = dt.date(2025, 1, 6)
    for k in range(7):
        b = (base + dt.timedelta(days=d)).isoformat()
        s = (base + dt.timedelta(days=d + 7)).isoformat()
        items += [(b, 20, 0), (s, 0, 20)]
        px[b], px[s] = 100, (90 if k == 3 else 112)
        d += 14
    for date, buy, sell, price in extra:
        items.append((date, buy, sell))
        px[date] = price
    return rows(*items), px


def test_signal_fires_for_big_recent_buy_still_held():
    r, px = _winner_history([("2026-10-27", 80, 0, 100), ("2026-10-28", 60, 10, 101), ("2026-10-29", 30, 0, 102)])
    recent = ["2026-10-27", "2026-10-28", "2026-10-29"]
    st = evaluate(r, px, recent, 102)
    assert st["high"] and st["win_rate"] == pytest.approx(6 / 7) and st["style"] == "短波段"
    assert st["signal"] == {"net3": 160, "threshold": 60, "held_recent": 160, "first_buy": "2026-10-27"}
    assert st["open_lots"] == 160 and st["open_cost"] == pytest.approx((80 * 100 + 50 * 101 + 30 * 102) / 160, abs=0.01)


def test_no_signal_when_sold_or_small_or_low_winrate():
    recent = ["2026-10-27", "2026-10-28", "2026-10-29"]
    r, px = _winner_history([("2026-10-27", 80, 0, 100), ("2026-10-29", 0, 70, 103)])   # 已大部分出場
    assert evaluate(r, px, recent, 103)["signal"] is None
    r, px = _winner_history([("2026-10-29", 25, 0, 100)])                                 # 買超不夠大
    assert evaluate(r, px, recent, 100)["signal"] is None
    r = rows(("2026-01-02", 10, 0), ("2026-01-03", 0, 10), ("2026-10-27", 500, 0))        # 交易數不足
    assert evaluate(r, {"2026-01-02": 1, "2026-01-03": 2, "2026-10-27": 2}, recent, 2)["signal"] is None
    perf = {"trades": 10, "win_rate": 0.9, "avg_ret": -0.01}
    assert detect_signal(perf, [], [], recent) is None                                   # 平均報酬為負


def test_compact_roundtrip_keys():
    r, px = _winner_history([("2026-10-27", 80, 0, 100)])
    c = quant.compact(evaluate(r, px, ["2026-10-27"], 100))
    assert set(c) >= {"n", "wr", "ar", "hd", "style", "high", "ol", "oc", "ur"}
    assert c["wr"] == round(6 / 7, 4)


def test_run_end_to_end(tmp_path, monkeypatch):
    """假的富邦頁面 + 假價格：檢查候選分點、長期快取（第二次只抓增量）、訊號輸出"""
    import datetime as dt
    import json
    monkeypatch.setattr(quant, "DATA_DIR", tmp_path)
    monkeypatch.setattr(quant, "LONG_DIR", tmp_path / "longhist")
    monkeypatch.setattr(quant, "QUANT_DIR", tmp_path / "quant")
    r, px = _winner_history([("2026-10-27", 80, 0, 100), ("2026-10-28", 60, 10, 101), ("2026-10-29", 30, 0, 102)])
    for d in ("2026-10-27", "2026-10-28", "2026-10-29"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "2330.json").write_text(json.dumps({"stock_id": "2330", "stock_name": "台積電", "buy_side": [
            {"broker": "凱基-台北", "broker_id": "9200", "b_raw": "9268", "branch_id": "9268", "city": "台北市"}], "sell_side": []}))

    class Fetcher:
        calls = []

        def get(self, url, delay=None):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(url).query)
            s, e = (dt.date(*map(int, q[k][0].split("-"))).isoformat() for k in ("D", "E"))
            self.calls.append((s, e))
            body = "".join(f"<tr><td>{x['date'].replace('-', '/')}</td><td>{x['buy']}</td><td>{x['sell']}</td><td>0</td><td>{x['buy'] - x['sell']}</td></tr>"
                           for x in r if s <= x["date"] <= e)
            return f"<table>{body}</table>"

        def close(self):
            pass

    monkeypatch.setattr(quant, "load_prices", lambda sid, end, today=None: {d: {"vwap": v, "close": v} for d, v in px.items()})
    f = Fetcher()
    out = quant.run("2026-10-29", ["2330"], fetcher=f)
    assert len(f.calls) >= 5                                   # 第一次分年抓 5 年
    st = out["stocks"]["2330"]["9268"]
    assert st["high"] and st["ol"] == 160
    assert len(out["signals"]) == 1 and out["signals"][0]["broker"] == "凱基-台北" and out["signals"][0]["net3"] == 160
    assert (tmp_path / "quant" / "2026-10-29.json").exists()
    f.calls.clear()
    quant.run("2026-10-29", ["2330"], fetcher=f)
    assert len(f.calls) == 0                                   # 快取已到今天 → 不再連網
    quant.run("2026-10-31", ["2330"], fetcher=f)
    assert len(f.calls) == 1                                   # 隔兩天（MAX_AGE_DAYS=0）→ 補一次，只補最近幾天


def test_horizons_clip_and_separate_results(tmp_path, monkeypatch):
    """同一份明細依分析範圍各自配對：6 個月內沒有完整交易 → 該範圍沒有勝率"""
    assert set(quant.HORIZONS) == {"6m", "1y", "2y", "3y", "5y"}
    rows_ = [{"date": "2024-01-02", "buy": 1, "sell": 0}, {"date": "2026-09-01", "buy": 1, "sell": 0}]
    assert quant.clip_rows(rows_, "2026-10-29", "6m") == rows_[1:]
    assert quant.clip_rows(rows_, "2026-10-29", "5y") == rows_


def test_incremental_policy_uses_listed_rows_until_weekly_refresh(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(quant, "DATA_DIR", tmp_path)
    monkeypatch.setattr(quant, "LONG_DIR", tmp_path / "longhist")
    quant._day_cache.clear()
    row = {"broker": "凱基-台北", "broker_id": "9200", "b_raw": "9268"}
    (tmp_path / "longhist" / "2330").mkdir(parents=True)
    (tmp_path / "longhist" / "2330" / "9268.json").write_text(json.dumps(
        {"start": "2021-10-29", "end": "2026-10-26", "rows": [{"date": "2026-10-26", "buy": 1, "sell": 0}]}))
    (tmp_path / "2026-10-28").mkdir()
    (tmp_path / "2026-10-28" / "2330.json").write_text(json.dumps({"stock_id": "2330", "buy_side": [{**row, "buy": 50, "sell": 3}], "sell_side": []}))

    class F:
        calls = 0

        def get(self, url, delay=None):
            F.calls += 1
            return "<table></table>"
    import datetime as dt
    out = quant.load_branch_history(F(), "2330", row, dt.date(2026, 10, 28), max_age=7,
                                    extra_rows=[{"date": "2026-10-28", "buy": 50, "sell": 3}])
    assert F.calls == 0 and [r["date"] for r in out] == ["2026-10-26", "2026-10-28"] and out[-1]["buy"] == 50
    quant.load_branch_history(F(), "2330", row, dt.date(2026, 11, 5), max_age=7)
    assert F.calls == 1                                        # 超過一週 → 向富邦補一次
