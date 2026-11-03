"""
第 4 節 會計層：當沖拆分 → 淨量進庫存（移動平均成本）→ 已實現損益

SPEC-GAP（本資料源）
  - 沒有分點的買進／賣出 VWAP：vb、vs 都用個股當日 VWAP 代替 → 當沖損益只剩交易成本
  - 沒有公司行動資料表：4.2 的除權息、減資調整本版不做（成本不扣現金股利）
  - 規格 4.1 的成本公式未把買進手續費併入成本，照字面實作（手續費只在賣出與當沖扣）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_layer import Panel
from .params import P


def run(pn: Panel) -> dict[str, np.ndarray]:
    """回傳 (分點, 日) 矩陣：inventory, avg_cost, realized, dt, dt_pnl, net, uncovered, left_censored"""
    a = P.accounting
    fee, tax, tax_dt = float(a.fee_rate), float(a.tax_sell), float(a.tax_daytrade)
    n, T = pn.B.shape
    inv = np.zeros(n); cost = np.zeros(n); cens = np.zeros(n, bool)
    out = {k: np.zeros((n, T)) for k in ("inventory", "avg_cost", "realized", "dt", "dt_pnl", "net", "uncovered")}
    out["left_censored"] = np.zeros((n, T), bool)
    for t in range(T):
        b, s = pn.B[:, t], pn.S[:, t]
        vb = vs = pn.vwap[t]
        dt = np.minimum(b, s)
        out["dt"][:, t] = dt
        out["dt_pnl"][:, t] = dt * (vs - vb) - (dt * vb * fee + dt * vs * (fee + tax_dt))
        net = b - s
        out["net"][:, t] = net
        buy = net > 0
        new_inv = inv + np.where(buy, net, 0)
        cost = np.where(buy, (inv * cost + np.where(buy, net, 0) * vb) / np.where(new_inv > 0, new_inv, 1), cost)
        inv = new_inv
        sell = net < 0
        q = np.where(sell, -net, 0)
        covered = np.minimum(q, inv)
        unc = q - covered
        out["realized"][:, t] = covered * (vs - cost) - covered * vs * (fee + tax)
        inv = inv - covered
        hit = unc > 0
        inv = np.where(hit, 0, inv)
        cost = np.where(hit | (inv <= 0), 0, cost)
        out["uncovered"][:, t] = unc
        cens |= hit
        out["inventory"][:, t] = inv
        out["avg_cost"][:, t] = cost
        out["left_censored"][:, t] = cens
    return out


def to_table(pn: Panel, L: dict) -> pd.DataFrame:
    """4.5 輸出 ledger（長表；只留有交易或有庫存的格子）"""
    i, t = np.nonzero((pn.B + pn.S > 0) | (L["inventory"] > 0))
    return pd.DataFrame({
        "date": pn.dates[t], "stock_id": pn.sid, "branch_id": np.array(pn.branches)[i],
        "buy": pn.B[i, t], "sell": pn.S[i, t], "vwap_buy": pn.vwap[t], "vwap_sell": pn.vwap[t],
        "dt": L["dt"][i, t], "dt_pnl": L["dt_pnl"][i, t], "net": L["net"][i, t],
        "inventory": L["inventory"][i, t], "avg_cost": L["avg_cost"][i, t], "realized_pnl": L["realized"][i, t],
        "uncovered_sell": L["uncovered"][i, t], "left_censored": L["left_censored"][i, t]})
