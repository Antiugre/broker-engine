"""
flowlab.costs — 分點持股成本（點選分點時在 K 線畫「持股成本」線）

  只用 FinMind（Sponsor Pro）分點當日真實買進均價，不用任何均價估算：
    起算日＝這檔股票第一個有分點真實價格的交易日（FinMind 2021-06-30 起）；之前的富邦資料沒有價格，不列入
    淨買 n > 0：成本 = (庫存 × 成本 + n × 分點真實買價) / (庫存 + n)；庫存 += n
    淨賣 n < 0：庫存 −= |n|（成本不變）；庫存歸零 → 下一次淨買重新起算（since＝這一輪開始建倉的日子）
    結算日＝最後一個有分點真實價格的交易日（FinMind 當天資料晚上才更新；之後幾天的富邦補資料沒有價格，不列入）
    起算日～結算日之間有淨買卻沒有真實買價（例如分點代碼對不上）→ 這個分點不輸出成本（寧缺勿估）
  起算日前就有的庫存無法得知：賣超大於庫存時庫存記 0，所以成本只反映起算日之後買進的部分
"""
from __future__ import annotations

import numpy as np

from .core import StockContext, iso, r


def holding_costs(ctx: StockContext) -> dict[str, dict]:
    if ctx.bp is None or ctx.buy.empty:
        return {}
    B, S = ctx.buy, ctx.sell
    BP = ctx.bp.reindex(index=B.index, columns=B.columns).to_numpy(float)
    has = np.isfinite(BP) & (BP > 0)
    rows = np.flatnonzero(has.any(axis=1))
    if not len(rows):
        return {}
    t0, t1 = rows[0], rows[-1] + 1                     # 第一個／最後一個有分點真實價格的交易日
    net = (B - S).to_numpy(float)[t0:t1]
    P, has = BP[t0:t1], has[t0:t1]
    idx = B.index[t0:t1]
    T, n = net.shape
    inv = np.zeros(n); cost = np.zeros(n); since = np.full(n, -1); bad = np.zeros(n, bool)
    for t in range(T):
        x = net[t]
        buy = x > 0
        bad |= buy & ~has[t]                           # 淨買卻沒有真實買價 → 不估
        new = inv + np.where(buy, x, 0)
        cost = np.where(buy & has[t], (inv * cost + np.where(buy, x, 0) * np.nan_to_num(P[t])) / np.where(new > 0, new, 1), cost)
        since = np.where(buy & (inv <= 0), t, since)
        inv = np.maximum(0, new + np.where(x < 0, x, 0))
        cost = np.where(inv > 0, cost, 0)
        since = np.where(inv > 0, since, -1)
    out = {}
    for j, k in enumerate(B.columns):
        if inv[j] >= 1 and cost[j] > 0 and not bad[j]:
            out[k] = {"oc": r(cost[j], 2), "ol": int(round(inv[j])), "os": iso(idx[since[j]]) if since[j] >= 0 else None,
                      "basis": "branch", "from": iso(idx[0]), "asof": iso(idx[-1])}
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
