"""
flowlab.costs — 分點持股成本（點選分點時在 K 線畫「持股成本」線）

  以完整歷史（約 5 年）逐日計算每個分點的庫存與移動平均成本：
    淨買 n > 0：成本 = (庫存 × 成本 + n × 買價) / (庫存 + n)；庫存 += n
    淨賣 n < 0：庫存 −= |n|（成本不變）；庫存歸零 → 成本歸零，下一次淨買重新起算（since＝這一輪開始建倉的日子）
  買價：FinMind（Sponsor Pro）分點當日真實買進均價；沒有時（富邦備用路徑）用個股成交均價
  資料起點前就有的庫存無法得知：賣超大於庫存時庫存記 0（保守），所以成本只反映資料期間內買進的部分
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import StockContext, iso, r


def holding_costs(ctx: StockContext) -> dict[str, dict]:
    B, S = ctx.buy, ctx.sell
    if B.empty:
        return {}
    vw = ctx.px["vwap"].reindex(B.index).to_numpy(float)
    if ctx.bp is not None:
        BP = ctx.bp.reindex(index=B.index, columns=B.columns).to_numpy(float)
        has = np.isfinite(BP) & (BP > 0)
        P = np.where(has, BP, vw[:, None])
    else:
        has = np.zeros(B.shape, bool)
        P = np.repeat(vw[:, None], B.shape[1], 1)
    net = (B - S).to_numpy(float)
    T, n = net.shape
    inv = np.zeros(n); cost = np.zeros(n); since = np.full(n, -1)
    realq = np.zeros(n); totq = np.zeros(n)
    for t in range(T):
        x = net[t]
        buy = x > 0
        new = inv + np.where(buy, x, 0)
        cost = np.where(buy, (inv * cost + np.where(buy, x, 0) * np.nan_to_num(P[t])) / np.where(new > 0, new, 1), cost)
        since = np.where(buy & (inv <= 0), t, since)
        realq = np.where(buy & (inv <= 0), 0, realq) + np.where(buy & has[t], x, 0)
        totq = np.where(buy & (inv <= 0), 0, totq) + np.where(buy, x, 0)
        inv = np.maximum(0, new + np.where(x < 0, x, 0))
        cost = np.where(inv > 0, cost, 0)
        since = np.where(inv > 0, since, -1)
    out = {}
    for j, k in enumerate(B.columns):
        if inv[j] >= 1 and cost[j] > 0:
            share = realq[j] / totq[j] if totq[j] else 0.0
            out[k] = {"oc": r(cost[j], 2), "ol": int(round(inv[j])), "os": iso(B.index[since[j]]) if since[j] >= 0 else None,
                      "basis": "branch" if share >= 0.5 else "vwap"}
    return out


def attach(doc: dict, costs: dict[str, dict]) -> dict:
    """把持股成本併進行為標籤列與群聚成員（以分點 key 對應）"""
    for b in doc.get("tags") or []:
        c = costs.get(b.get("key"))
        if c:
            b.update(c)
    for cl in doc.get("clusters") or []:
        for m in cl.get("members") or []:
            c = costs.get(m.get("key"))
            if c:
                m.update(c)
    return doc
