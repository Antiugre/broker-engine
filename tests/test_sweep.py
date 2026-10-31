"""python -m pytest tests/test_sweep.py -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quant  # noqa: E402
import sweep  # noqa: E402

DAY = "2026-10-30"

# 依 2026-10-30 富邦 zgb0（凱基-台北）實際頁面結構縮減而成：每列 4 格（名稱、買進、賣出、差額），
# 一般股票用 GenLink2stk 腳本、ETF 用 Link2Stk 連結；買超區之後是賣超區（差額為負）
ZGB = """<html><body><div>資料日期：20261030</div><table>
<tr><td class="t4t1" nowrap id="oAddCheckbox"> <a href="javascript:Link2Stk('00687B');">00687B國泰20年美債</a> </td>
<td class="t3n1" nowrap>5,710</td> <td class="t3n1" nowrap>681</td> <td class="t3n1" nowrap>5,029</td> </tr>
<tr> <td class="t4t1" nowrap id="oAddCheckbox"> <SCRIPT LANGUAGE=javascript> <!-- GenLink2stk('AS2303','聯電'); //--> </SCRIPT> </td>
<td class="t3n1" nowrap>13,229</td> <td class="t3n1" nowrap>8,740</td> <td class="t3n1" nowrap>4,489</td> </tr>
<tr><td>賣超</td><td>券商名稱</td><td>買進張數</td><td>賣出張數</td><td>差額</td></tr>
<tr> <td class="t4t1" nowrap id="oAddCheckbox"> <SCRIPT LANGUAGE=javascript> <!-- GenLink2stk('AS2330','台積電'); //--> </SCRIPT> </td>
<td class="t3n1" nowrap>3,205</td> <td class="t3n1" nowrap>39,743</td> <td class="t3n1" nowrap>-36,538</td> </tr>
<tr> <td class="t4t1" nowrap id="oAddCheckbox"> <a href="javascript:Link2Stk('00406A');">00406A主動中信台灣收益</a> </td>
<td class="t3n1" nowrap>37,094</td> <td class="t3n1" nowrap>104,522</td> <td class="t3n1" nowrap>-67,428</td> </tr>
</table></body></html>"""

LIST_JS = "var g_BrokerList = '9200,凱基!9200,凱基!9268,凱基-台北!0039004100380039,凱基-松山;9800,元大!9800,元大!9815,元大-台北';"


def test_parse_zgb_real_structure():
    day, rows = sweep.parse_zgb(ZGB)
    assert day == DAY
    got = {r[0]: r for r in rows}
    assert got["2303"] == ("2303", "聯電", 13229, 8740)
    assert got["2330"][2:] == (3205, 39743)
    assert got["00687B"][1] == "國泰20年美債" and got["00406A"][3] == 104522
    assert "賣超" not in got and len(rows) == 4


def test_parse_branch_list_and_hex_codes():
    lst = sweep.parse_branch_list(LIST_JS)
    keyed = {(b["broker_id"], b["b_raw"]): b for b in lst}
    assert len(lst) == 5                                          # 總公司重複列只留一次
    assert keyed[("9200", "0039004100380039")]["branch_id"] == "9A89"
    assert keyed[("9800", "9815")]["broker"] == "元大-台北"


class FakeFetcher:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def get(self, url, delay=None):
        if url.endswith("zbrokerjs.djjs"):
            return LIST_JS
        self.calls.append(url)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise ConnectionError("blocked")
        if "b=9268" in url:
            return ZGB
        return "<div>資料日期：20261030</div><table></table>"

    def close(self):
        pass


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sweep, "LIST_CACHE", tmp_path / "fubon_branches.json")
    monkeypatch.setattr(quant, "DATA_DIR", tmp_path)
    monkeypatch.setattr(quant, "LONG_DIR", tmp_path / "longhist")
    sweep.clear_cache()
    quant._day_cache.clear()
    yield tmp_path
    sweep.clear_cache()
    quant._day_cache.clear()


def test_run_resume_and_day_rows(data):
    f = FakeFetcher()
    r = sweep.run(DAY, fetcher=f, limit=2)
    assert r["done"] == 2 and not r["finished"]
    r = sweep.run(DAY, fetcher=f)                                 # 續跑：只抓剩下的
    assert r["finished"] and r["done"] == 5 and len(f.calls) == 5
    rows = sweep.day_rows("2330", DAY)
    assert rows["9268"]["broker"] == "凱基-台北" and rows["9268"]["net"] == 3205 - 39743
    assert rows["9268"]["source"] == "sweep"


def test_run_stops_when_blocked(data, monkeypatch):
    monkeypatch.setattr(sweep, "MAX_FAIL_STREAK", 2)
    r = sweep.run(DAY, fetcher=FakeFetcher(fail_after=1))
    assert not r["finished"] and r["done"] == 1 and r["failed"] == 2


def test_select_extras_excludes_top15_and_small(data):
    sweep.run(DAY, fetcher=FakeFetcher())
    (data / DAY / "2303.json").write_text(json.dumps({"stock_id": "2303", "buy_side": [], "sell_side": []}))
    (data / DAY / "2330.json").write_text(json.dumps({"stock_id": "2330", "buy_side": [
        {"broker": "凱基-台北", "broker_id": "9200", "b_raw": "9268"}], "sell_side": []}))
    ex = sweep.select_extras(DAY, ["2330", "2303"])
    assert "2330" not in ex                                       # 已在前 15 名 → 不重複
    assert list(ex["2303"]) == ["9268"] and ex["2303"]["9268"]["_extra"]
    assert sweep.select_extras(DAY, ["2303"], min_lots=10_000) == {}   # 淨額太小 → 不算


def test_engines_see_extras_only_with_history(data):
    sweep.run(DAY, fetcher=FakeFetcher())
    (data / DAY / "2303.json").write_text(json.dumps({"stock_id": "2303", "stock_name": "聯電",
                                                      "buy_side": [], "sell_side": []}))
    assert "2303" not in quant.candidates(DAY, ["2303"])         # 還沒建 5 年明細 → 引擎先不算
    (data / "longhist" / "2303").mkdir(parents=True)
    (data / "longhist" / "2303" / "9268.json").write_text("{}")
    c = quant.candidates(DAY, ["2303"])
    assert c["2303"]["9268"]["broker"] == "凱基-台北"
    assert quant._listed("2303", DAY)["9268"] == {"date": DAY, "buy": 13229, "sell": 8740}   # 暫定明細


def test_warm_builds_biggest_first_and_respects_deadline(data, monkeypatch):
    sweep.run(DAY, fetcher=FakeFetcher())
    for sid in ("2303", "2330"):
        (data / DAY / f"{sid}.json").write_text(json.dumps({"stock_id": sid, "buy_side": [], "sell_side": []}))
    built = []
    monkeypatch.setattr(quant, "load_branch_history", lambda f, sid, row, end, **k: built.append(sid) or [])
    r = sweep.warm(DAY, ["2303", "2330"], fetcher=FakeFetcher())
    assert r["queued"] == 2 and built == ["2330", "2303"]        # |淨額| 36,538 > 4,489
    built.clear()
    assert sweep.warm(DAY, ["2303", "2330"], fetcher=FakeFetcher(), until=0)["built"] == 0 and not built
