"""維度 B（B1 冷門價位共現、B2 壓價轉移）手算小案例"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from broker_flow import dim_b  # noqa: E402
from broker_flow.data_layer import Panel  # noqa: E402

K = 1000


def _panel(T: int, B: np.ndarray, S: np.ndarray, close: np.ndarray) -> Panel:
    n = B.shape[0]
    dates = np.array([f"2026-01-{d + 1:02d}" for d in range(T)])
    return Panel("TEST", dates, [f"b{i}" for i in range(n)], [f"分點{i}" for i in range(n)], ["brokerage"] * n,
                 B, S, close, close.copy(), np.full(T, 1e6), T, {"source": "finmind"})


def test_b1_cold_price_and_frechet_and_conservation():
    # 3 個分點、1 天；價位 10.0 量大（不冷門），10.5 冷門 6 張：分點 0 賣 6、分點 1 買 6 → 參與＋下界 M = 6+6−6 = 6 張 ≥ 1
    T = 1
    B = np.array([[0.0], [6 * K], [94 * K]])
    S = np.array([[6 * K], [0.0], [94 * K]])
    pn = _panel(T, B, S, np.array([10.0]))
    lv = {0: (np.array([10.0, 10.0, 10.5, 10.5]), np.array([2, 2, 0, 1]),
              np.array([94 * K, 0, 0, 6 * K]), np.array([0, 94 * K, 6 * K, 0]))}
    # 冷門門檻：V = [94 張, 6 張]，第 20 百分位數 ≈ 23.6 張 → 只有 10.5 冷門
    V, Bm, Sm, ok, up = dim_b._day(lv[0], np.arange(3), 3)
    assert ok and list(up) == [10.0, 10.5]
    w = {"res": {"pairs": np.array([[0, 1], [0, 2], [1, 2]])}, "cand": [0, 1, 2], "cls": ["swing"] * 3,
         "start": 0, "end": 0}
    cx = SimpleNamespace(pn=pn, active=(B + S) > 0)
    r = dim_b.window(cx, w, lv)
    assert list(r["T_B1"]) == [2.0, 0.0, 0.0]          # (0 賣 → 1 買) 參與 1 分＋下界 1 分
    assert r["p_B1"][1] == 1.0 and r["p_B1"][2] == 1.0
    # 守恆不成立的日子整日排除
    bad = {0: (np.array([10.0]), np.array([0]), np.array([5 * K]), np.array([4 * K]))}
    assert dim_b.window(cx, w, bad)["violation_days"] == 1


def test_b2_event_day_count_and_direction():
    # 2 個分點、3 天。第 0、1 天建立賣方的歷史（|淨買賣| 中位數 10 張）；第 2 天：S（分點 0）淨賣 30 張 ≥ 2×10 且 ≥ 10、
    # 收盤下跌、R（分點 1）淨買 30 張 ≥ 0.1×30、同價位成交 → 事件日 1 天，方向 0 → 1
    T = 3
    B = np.array([[10 * K, 0, 0], [0, 10 * K, 30 * K]], float)
    S = np.array([[0, 10 * K, 30 * K], [10 * K, 0, 0]], float)
    close = np.array([10.0, 10.2, 10.0])
    pn = _panel(T, B, S, close)
    lv = {0: (np.array([10.0, 10.0]), np.array([0, 1]), np.array([10 * K, 0]), np.array([0, 10 * K])),
          1: (np.array([10.2, 10.2]), np.array([1, 0]), np.array([10 * K, 0]), np.array([0, 10 * K])),
          2: (np.array([10.0, 10.0]), np.array([1, 0]), np.array([30 * K, 0]), np.array([0, 30 * K]))}
    w = {"res": {"pairs": np.array([[0, 1]])}, "cand": [0, 1], "cls": ["swing", "swing"], "start": 0, "end": 2}
    cx = SimpleNamespace(pn=pn, active=(B + S) > 0)
    r = dim_b.window(cx, w, lv)
    assert r["T_B2"][0] == 1.0
    assert r["seller"][0] == 0 and r["buyer"][0] == 1
    # 前視檢查：把第 2 天以後的資料拿掉，事件日不受影響（條件只用 t 日以前與當日）
    w2 = dict(w, end=2)
    assert dim_b.window(cx, w2, {k: v for k, v in lv.items() if k <= 2})["T_B2"][0] == 1.0


def test_no_levels_means_missing():
    pn = _panel(1, np.zeros((2, 1)), np.zeros((2, 1)), np.array([10.0]))
    w = {"res": {"pairs": np.array([[0, 1]])}, "cand": [0, 1], "cls": ["swing"] * 2, "start": 0, "end": 0}
    assert dim_b.window(SimpleNamespace(pn=pn, active=np.zeros((2, 1), bool)), w, None) is None
