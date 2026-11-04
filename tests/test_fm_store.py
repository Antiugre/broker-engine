"""fm_store（FinMind 分點資料）與 quant 的富邦備用路徑"""
from __future__ import annotations

import datetime as dt
import importlib
import os
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write(dirp: Path, ym: str, rows: list[tuple]) -> None:
    cols = ["stock_id", "securities_trader_id", "securities_trader", "date", "buy_volume", "sell_volume", "buy_price", "sell_price"]
    t = pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    t = t.sort_by([("stock_id", "ascending"), ("securities_trader_id", "ascending"), ("date", "ascending")])
    pq.write_table(t, dirp / f"{ym}.parquet", row_group_size=2)


@pytest.fixture()
def fm(tmp_path, monkeypatch):
    sec = tmp_path / "SecIdAgg"
    sec.mkdir()
    _write(sec, "2026-09", [("2601", "9268", "凱基台北", "2026-09-30", 147000, 23000, 6.5, 6.6),
                            ("2601", "981j", "元大士林", "2026-09-30", 1500, 0, 6.4, 0.0),
                            ("2330", "9268", "凱基台北", "2026-09-30", 5000, 0, 1000.0, 0.0)])
    _write(sec, "2026-10", [("2601", "9268", "凱基台北", "2026-10-02", 0, 12000, 0.0, 6.7),
                            ("2601", "981J", "元大士林二", "2026-10-02", 2000, 0, 6.3, 0.0)])
    monkeypatch.setenv("FM_DIR", str(tmp_path))
    monkeypatch.delenv("BROKER_HISTORY_SOURCE", raising=False)
    import fm_store
    importlib.reload(fm_store)
    return fm_store


def test_history_shares_to_lots_and_case_sensitive_codes(fm):
    h = fm.history("2601", "9268", dt.date(2021, 1, 1), dt.date(2026, 10, 5))
    assert [r["date"] for r in h] == ["2026-09-30", "2026-10-02"]
    assert h[0]["buy"] == 147 and h[0]["sell"] == 23 and h[0]["buy_price"] == 6.5
    assert h[1]["buy_price"] is None and h[1]["sell_price"] == 6.7
    # 富邦 b_raw 十六進位 → 981j（小寫）和 981J 是不同分點
    lower = fm.history("2601", "003900380031006a", dt.date(2021, 1, 1), dt.date(2026, 10, 5))
    assert [r["buy"] for r in lower] == [1.5]


def test_unknown_branch_or_stale_falls_back(fm):
    assert fm.history("2601", "1470", dt.date(2021, 1, 1), dt.date(2026, 10, 5)) is None      # 對不到 → 富邦
    assert fm.history("2601", "9268", dt.date(2021, 1, 1), dt.date(2026, 12, 31)) is None     # 資料太舊 → 富邦
    os.environ["BROKER_HISTORY_SOURCE"] = "fubon"
    try:
        assert fm.history("2601", "9268", dt.date(2021, 1, 1), dt.date(2026, 10, 5)) is None  # 強制走富邦
    finally:
        del os.environ["BROKER_HISTORY_SOURCE"]


def test_range_top(fm):
    r = fm.range_top("2601", "2026-09-01", "2026-10-02")
    assert [x["branch_id"] for x in r["buy_side"]] == ["9268", "981J", "981j"]
    assert r["buy_side"][0]["net"] == 112 and r["buy_side"][0]["broker_id"] == "9200"
    assert r["buy_side"][2]["b_raw"] == "003900380031006a"
    r = fm.range_top("2601", "2026-10-01", "2026-10-02")
    assert r["sell_side"][0]["branch_id"] == "9268" and r["sell_side"][0]["net"] == -12
    assert fm.range_top("2601", "2026-09-01", "2026-10-30") is None   # FinMind 還沒有 10/30 → 富邦


def test_quant_uses_finmind_then_fubon(fm, monkeypatch):
    import quant
    calls = []

    class F:
        def get(self, url):
            calls.append(url)
            raise RuntimeError("不應該連富邦")
    rows = quant.load_branch_history(F(), "2601", {"b_raw": "9268", "broker_id": "9200"}, dt.date(2026, 10, 2))
    assert len(rows) == 2 and not calls
    with pytest.raises(RuntimeError):     # 對不到的分點 → 走原本的富邦路徑（這裡故意讓它失敗以確認有走到）
        quant.load_branch_history(F(), "2601", {"b_raw": "1470", "broker_id": "1470"}, dt.date(2026, 10, 2))
    assert calls
