"""python -m pytest tests/test_universe.py -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import universe  # noqa: E402


def test_parse_market_twse_and_tpex():
    twse = [{"Date": "1151030", "Code": "2330", "Name": "台積電", "TradeVolume": "30,000,000", "TradeValue": "18,300,000,000", "ClosingPrice": "612.00"},
            {"Code": "0050", "Name": "元大台灣50", "TradeVolume": "9,000,000", "TradeValue": "1,800,000,000", "ClosingPrice": "200"}]
    tpex = [{"Date": "1151030", "SecuritiesCompanyCode": "6488", "CompanyName": "環球晶", "TradingShares": "2,000,000", "TransactionAmount": "900,000,000", "Close": "450"}]
    d, m = universe.parse_market(twse, tpex)
    assert d == "2026-10-30" and m["2330"]["vol"] == 30_000 and m["2330"]["vwap"] == 610.0 and m["6488"]["market"] == "tpex"
    assert universe.top_by_value(m, 5) == ["2330", "6488"]              # 0050（ETF）排除


class Snap:
    def __init__(self, d): self.d = d
    def to_dict(self): return self.d


class DB:
    def __init__(self, users): self.users = users
    def collection(self, name): return self
    def where(self, *a): return self
    def stream(self): return [Snap(u) for u in self.users]


def test_build_priority_and_ultra_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(universe, "DATA_DIR", tmp_path)
    db = DB([{"tier": "ultra", "watchlist": ["2454", "3008"]}, {"tier": "ultra", "watchlist": ["3008", "bad!"]}])
    market = {"2330": {"value": 9e9}, "2317": {"value": 5e9}, "3008": {"value": 1e9}}
    u = universe.build("2026-10-30", db, market, top_n=2)
    assert u["stocks"] == ["3008", "2454", "2330", "2317"]               # Ultra 自選（依人數）→ 成交值前 N 名
    assert u["sources"]["3008"] == "ultra" and u["sources"]["2317"] == "top"
    assert json.loads((tmp_path / "universe" / "ultra_watch.json").read_text())["counts"] == {"2454": 1, "3008": 2}


def test_apply_market_to_price_cache(tmp_path, monkeypatch):
    import quant
    monkeypatch.setattr(quant, "LONG_DIR", tmp_path)
    (tmp_path / "2330").mkdir()
    f = tmp_path / "2330" / "_prices.json"
    f.write_text(json.dumps({"v": 2, "start": "2021-10-30", "end": "2026-10-29", "data": {}}))
    n = universe.apply_to_price_cache("2026-10-30", {"2330": {"vwap": 610, "close": 612, "vol": 30000}}, "2026-10-30")
    c = json.loads(f.read_text())
    assert n == 1 and c["end"] == "2026-10-30" and c["data"]["2026-10-30"]["vol"] == 30000
    assert universe.apply_to_price_cache("2026-10-31", {"2330": {}}, "2026-10-30") == 0   # 日期不符不使用
