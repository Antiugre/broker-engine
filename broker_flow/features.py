"""
第 5 節 行為特徵層：scale_i、z_net、活躍日（6.0）、分點行為分類遮罩（5.1）、地緣（5.2，報告用）

所有滾動計算只用 t（含）以前的資料（第 0 節第 1 條；tests/test_broker_flow.py 有前視測試）
SPEC-GAP
  - 5.4 scale_i「活躍日總成交量中位數」與 6.0 活躍日定義互相引用：scale_i 的「活躍日」取 b + s > 0 的日子
  - 5.1「活躍日淨買中位數」取活躍日中淨買 > 0 的日子
  - 5.3 Wasserstein 分歧度：沒有價位明細，不計算
"""
from __future__ import annotations

import numpy as np

from .data_layer import Panel
from .params import P

SWING, DAYTRADE, OVERNIGHT, FOREIGN, PROP, UNCLASSIFIED = "swing", "daytrade", "overnight", "foreign", "proprietary", "unclassified"


def scale_and_active(pn: Panel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """scale (分點, 日)、active (bool)、z_net；第 t 天只用 [t−119, t]"""
    w = int(P.behavior.rolling_days)
    g = pn.B + pn.S
    n, T = g.shape
    scale = np.full((n, T), np.nan)
    gz = np.where(g > 0, g, np.nan)
    for t in range(T):
        blk = gz[:, max(0, t - w + 1): t + 1]
        ok = np.isfinite(blk).any(axis=1)
        if ok.any():
            scale[ok, t] = np.nanmedian(blk[ok], axis=1)
    thr = np.maximum(float(P.detection.active_min_shares), float(P.detection.active_scale_mult) * np.nan_to_num(scale, nan=np.inf))
    active = (g >= thr) & (g > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(scale) & (scale > 0), (pn.B - pn.S) / scale, 0.0)
    return scale, active, z


def classify(pn: Panel, active: np.ndarray, t: int) -> list[str]:
    """5.1：第 t 天（含）往前 120 個交易日的行為分類"""
    b = P.behavior
    w = int(b.rolling_days)
    lo = max(0, t - w + 1)
    B, S, A = pn.B[:, lo:t + 1], pn.S[:, lo:t + 1], active[:, lo:t + 1]
    net = B - S
    out = []
    for i in range(B.shape[0]):
        bt = pn.btype[i]
        if bt == "foreign":
            out.append(FOREIGN); continue
        if bt == "proprietary":
            out.append(PROP); continue
        if A[i].sum() < int(b.min_active_days):
            out.append(UNCLASSIFIED); continue
        half = (B[i] + S[i]).sum() / 2
        if half > 0 and np.minimum(B[i], S[i]).sum() / half >= float(b.daytrade_ratio):
            out.append(DAYTRADE); continue
        pos = net[i][A[i] & (net[i] > 0)]
        if len(pos):
            med = np.median(pos)
            L = B.shape[1]
            ev = [e for e in range(L - 3) if net[i, e] >= float(b.overnight_event_mult) * med and net[i, e] > 0]
            if len(ev) >= int(b.overnight_min_events):
                r = [S[i, e + 1:e + 4].sum() / net[i, e] for e in ev]
                if np.median(r) >= float(b.overnight_ratio):
                    out.append(OVERNIGHT); continue
        out.append(SWING)
    return out


EXCLUDED = {DAYTRADE, OVERNIGHT}
