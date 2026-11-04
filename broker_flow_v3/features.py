"""
§4 行為特徵（v3）：留倉淨量 n = b − s
  scale_net_i,t = median{|n_i,τ| : τ ∈ [t−119, t], |n| > 0}；z = n / scale_net
  active = |n| ≥ max(1000, 0.5·scale_net)；停牌日不活躍
  當沖型：照 v2 公式記標籤，不排除；隔日沖型：排除（事件門檻改用淨量中位數）
  分點類型：foreign、proprietary、gov_bank（SPEC-GAP-v3：名稱比對）各自成類
§4.2 自我成交 SC：X = Σ_P min(b_P, s_P)，x = X / ((b+s)/2)；
  SC = [X ≥ 10,000 股 ∧ X / V ≥ 0.01 ∧ (x − 中位數) / (1.4826·MAD) ≥ 3]，基準取 [t−120, t−1]（不含當日）
  MAD 下限 0.01（SPEC-GAP-v3）
"""
from __future__ import annotations

import numpy as np

from .data_layer import Panel
from .params import P

SWING, DAYTRADE, OVERNIGHT, FOREIGN, PROP, GOV, UNCLASSIFIED = "swing", "daytrade", "overnight", "foreign", "proprietary", "gov_bank", "unclassified"
EXCLUDED = {OVERNIGHT}


def scale_and_active(pn: Panel):
    w = int(P.behavior.rolling_days)
    n_ = pn.B - pn.S
    a = np.abs(n_)
    az = np.where(a > 0, a, np.nan)
    nb, T = a.shape
    scale = np.full((nb, T), np.nan)
    for t in range(T):
        blk = az[:, max(0, t - w + 1): t + 1]
        ok = np.isfinite(blk).any(axis=1)
        if ok.any():
            scale[ok, t] = np.nanmedian(blk[ok], axis=1)
    thr = np.maximum(float(P.detection.active_min_shares), float(P.detection.active_scale_mult) * np.nan_to_num(scale, nan=np.inf))
    active = (a >= thr) & (a > 0) & ~pn.suspended[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(scale) & (scale > 0), n_ / scale, 0.0)
    return scale, active, z


def classify(pn: Panel, active: np.ndarray, t: int) -> list[str]:
    """第 t 天（含）往前 120 日；當沖型只是標籤（不排除）"""
    b = P.behavior
    w = int(b.rolling_days)
    lo = max(0, t - w + 1)
    B, S, A = pn.B[:, lo:t + 1], pn.S[:, lo:t + 1], active[:, lo:t + 1]
    net = B - S
    out = []
    for i in range(B.shape[0]):
        bt = pn.btype[i]
        if bt in ("foreign", "proprietary", "gov_bank"):
            out.append(bt); continue
        if A[i].sum() < int(b.min_active_days):
            out.append(UNCLASSIFIED); continue
        pos = net[i][A[i] & (net[i] > 0)]
        if len(pos):
            med = np.median(pos)
            L = B.shape[1]
            ev = [e for e in range(L - 3) if net[i, e] > 0 and net[i, e] >= float(b.overnight_event_mult) * med]
            if len(ev) >= int(b.overnight_min_events):
                r = [S[i, e + 1:e + 4].sum() / net[i, e] for e in ev]
                if np.median(r) >= float(b.overnight_ratio):
                    out.append(OVERNIGHT); continue
        half = (B[i] + S[i]).sum() / 2
        if half > 0 and np.minimum(B[i], S[i]).sum() / half >= float(b.daytrade_ratio):
            out.append(DAYTRADE); continue
        out.append(SWING)
    return out


def self_cross(pn: Panel, X: np.ndarray) -> np.ndarray:
    """§4.2：X (分點, 日) 同價位雙向量（股，NaN＝沒有價位明細）→ SC (bool)"""
    b = P.behavior
    g = (pn.B + pn.S) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(g > 0, X / g, np.nan)
    n, T = x.shape
    sc = np.zeros((n, T), bool)
    w = int(b.rolling_days)
    vol = np.maximum(pn.volume, 1)
    for t in range(1, T):
        hist = x[:, max(0, t - w):t]
        ok = np.isfinite(hist).sum(axis=1) >= 10
        if not ok.any():
            continue
        med = np.nanmedian(np.where(ok[:, None], hist, np.nan), axis=1)
        mad = np.nanmedian(np.abs(hist - med[:, None]), axis=1)
        mad = np.maximum(np.nan_to_num(mad, nan=0.0), float(b.sc_mad_floor))
        zz = (x[:, t] - med) / (1.4826 * mad)
        sc[:, t] = ok & (np.nan_to_num(X[:, t]) >= float(b.sc_min_shares)) & (np.nan_to_num(X[:, t]) / vol[t] >= float(b.sc_min_vol_frac)) \
            & (np.nan_to_num(zz, nan=-np.inf) >= float(b.sc_z))
    return sc
