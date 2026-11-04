"""
§5 分類規則、§7 BH
  1 q^E ≤ 0.05 且 Δ_lo ≥ 0                → 擇時／資訊型
  2 q^E ≤ 0.05 且 Δ_hi ≤ 0 且 κ̂⁺ > 0      → 拉抬型
  3 q^E ≤ 0.05，不符合 1、2               → 混合型（bootstrap 剔除比例 > 10% 也歸此類，標「bootstrap 不穩定」）
  4 q^E > 0.05 且 q^R ≤ 0.05              → 逆勢承接型
  5 其他                                  → 無
  當日追價旗標：類型 1～3 且 p^{E+} > 0.05
"""
from __future__ import annotations

import numpy as np

from .params import TP

TIMING, PUMP, MIXED, CONTRARIAN, NONE = "timing", "pump", "mixed", "contrarian", "none"
TYPES = [TIMING, PUMP, MIXED, CONTRARIAN, NONE]
LABEL = {TIMING: "擇時／資訊型", PUMP: "拉抬型", MIXED: "混合型", CONTRARIAN: "逆勢承接型", NONE: "無"}
CODE = {TIMING: 1, PUMP: 2, MIXED: 3, CONTRARIAN: 4, NONE: 5}
CHASE_NOTE = "顯著性依賴當日上漲時的買進：可能是資訊型，也可能是當日追價碰上短期動能"
UNSTABLE_NOTE = "bootstrap 不穩定"


def bh(p) -> np.ndarray:
    """Benjamini–Hochberg 調整後 q 值（NaN 不列入族群、輸出 NaN）"""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    pv = p[ok]
    o = np.argsort(pv, kind="stable")
    adj = pv[o] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


def classify(qE: float, qR: float, delta_lo: float = np.nan, delta_hi: float = np.nan, kappa_plus: float = np.nan,
             unstable: bool = False) -> str:
    a = float(TP.fdr_q)
    sigE = bool(np.isfinite(qE) and qE <= a)
    sigR = bool(np.isfinite(qR) and qR <= a)
    if sigE:
        if not unstable and np.isfinite(delta_lo) and delta_lo >= 0:
            return TIMING
        if not unstable and np.isfinite(delta_hi) and delta_hi <= 0 and np.isfinite(kappa_plus) and kappa_plus > 0:
            return PUMP
        return MIXED
    if sigR:
        return CONTRARIAN
    return NONE


def flags(kind: str, p_eplus: float | None, unstable: bool = False, cycles: int | None = None) -> list[str]:
    out = []
    if kind in (TIMING, PUMP, MIXED) and p_eplus is not None and np.isfinite(p_eplus) and p_eplus > float(TP.chase_flag_p):
        out.append(CHASE_NOTE)
    if kind in (TIMING, PUMP, MIXED) and unstable:
        out.append(UNSTABLE_NOTE)
    if cycles is not None and cycles < int(TP.few_cycles_flag):
        out.append("樣本偏少")
    return out
