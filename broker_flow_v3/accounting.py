"""
§3.2～3.4 會計層（v3）
  當沖 dt = min(b, s)，損益用分點當日買／賣均價（FinMind）
  留倉淨量 n = b − s 分三批（§3.3）：
    L 已知成本多頭批：淨買先回補 S，剩下進 L（移動平均成本）
    S 疑似空頭批：超賣 u = max(0, −n − I^L)；ΔS = u · min(1, Q_t / Σ_k u_k,t)，Q_t = 融券賣出＋借券賣出（個股、股）；成本 = 賣出均價
    U 未知成本批：ΔU = u − ΔS，不進損益；另列「無成本變現金額」Σ ΔU × 賣出均價
  公司行動（§3.2），在生效日 τ 開盤前調整 L、S：
    現金股利 D：C′ = C − D（股利已反映在之後的損益裡；另列 dividend_memo = D·I 只供參考）
      ※ 規格寫「C′ = C − D 且已實現收益 += D·I」會重複計算（除息後賣出損益 P − (C − D) 已含 D），本實作只調成本（CHANGELOG v3 §3.2）
    股票股利 r：I′ = I(1 + r)，C′ = C/(1 + r)
    分割、面額變更、減資：k = P_before / P_ref，I′ = I·k，C′ = C/k；現金減資標 ca_approx
  手續費（§3.4）：fee_rate × fee_discount（預設 1.0；另算 0.3 的敏感度）
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import auxdata as aux
from .data_layer import LOT, Panel
from .params import P

log = logging.getLogger("broker_scraper")


def _pick(df: pd.DataFrame, *subs) -> str | None:
    for c in df.columns:
        lc = c.lower()
        if all(s in lc for s in subs):
            return c
    return None


def _f(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if v is not None and np.isfinite(v) else 0.0


def corp_actions(sid: str) -> pd.DataFrame:
    """事件表：date（生效日）、kind（cash／stock／ratio）、value、approx"""
    ev = []
    dv = aux.stock("TaiwanStockDividend", sid, "2015-01-01")
    if len(dv):
        for _, r in dv.iterrows():
            D = sum(_f(r.get(c)) for c in ("CashEarningsDistribution", "CashStatutorySurplus"))
            rr = sum(_f(r.get(c)) for c in ("StockEarningsDistribution", "StockStatutorySurplus")) / 10.0
            cd, sd = str(r.get("CashExDividendTradingDate") or ""), str(r.get("StockExDividendTradingDate") or "")
            if D > 0 and len(cd) >= 10:
                ev.append((cd[:10], "cash", D, False))
            if rr > 0 and len(sd) >= 10:
                ev.append((sd[:10], "stock", rr, False))
    cr = aux.stock("TaiwanStockCapitalReductionReferencePrice", sid, "2015-01-01")
    if len(cr):
        cb, ca = _pick(cr, "closing"), _pick(cr, "postreduction")
        rs = _pick(cr, "reason")
        for _, r in cr.iterrows():
            b, a = _f(r.get(cb)), _f(r.get(ca))
            if b > 0 and a > 0:
                ev.append((str(r["date"])[:10], "ratio", b / a, "cash" in str(r.get(rs, "")).lower()))
    for ds in ("TaiwanStockSplitPrice", "TaiwanStockParValueChange"):
        x = aux.stock(ds, sid, "2015-01-01")
        if len(x):
            cb = _pick(x, "before")
            ca = _pick(x, "after") or _pick(x, "ref")
            for _, r in x.iterrows():
                b, a = _f(r.get(cb)), _f(r.get(ca))
                if b > 0 and a > 0:
                    ev.append((str(r["date"])[:10], "ratio", b / a, False))
    return pd.DataFrame(ev, columns=["date", "kind", "value", "approx"]).sort_values("date").reset_index(drop=True)


def short_capacity(sid: str, dates: np.ndarray) -> np.ndarray:
    """Q_t（股）＝融券賣出（張 × margin_unit_shares）＋借券賣出（股）；查不到的日子為 0"""
    q = pd.Series(0.0, index=pd.Index(dates))
    m = aux.stock("TaiwanStockMarginPurchaseShortSale", sid, "2019-01-01")
    if len(m) and "ShortSaleSell" in m:
        s = pd.Series(aux.num(m, "ShortSaleSell").to_numpy() * float(P.accounting.margin_unit_shares), index=m["date"].astype(str))
        q = q.add(s.groupby(level=0).sum().reindex(q.index).fillna(0), fill_value=0)
    b = aux.stock("TaiwanDailyShortSaleBalances", sid, "2019-01-01")
    col = _pick(b, "sblshortsales", "shortsales") if len(b) else None
    if col:
        s = pd.Series(aux.num(b, col).to_numpy(), index=b["date"].astype(str))
        q = q.add(s.groupby(level=0).sum().reindex(q.index).fillna(0), fill_value=0)
    return q.to_numpy(float)


def run(pn: Panel, fee_discount: float | None = None, events: pd.DataFrame | None = None, Q: np.ndarray | None = None) -> dict:
    a = P.accounting
    disc = float(a.fee_discount if fee_discount is None else fee_discount)
    fee, tax, tax_dt = float(a.fee_rate) * disc, float(a.tax_sell), float(a.tax_daytrade)
    n, T = pn.B.shape
    if events is None:
        events = corp_actions(pn.sid)
    if Q is None:
        Q = short_capacity(pn.sid, pn.dates) if a.short_lot_enabled else np.zeros(T)
    ev_at: dict[int, list] = {}
    for _, e in events.iterrows():
        t = int(np.searchsorted(pn.dates, e["date"], side="left"))
        if t < T:
            ev_at.setdefault(t, []).append(e)
    IL = np.zeros(n); CL = np.zeros(n); IS = np.zeros(n); CS = np.zeros(n)
    keys = ("inventory", "avg_cost", "short_inv", "short_cost", "realized", "dt", "dt_pnl", "net", "u_shares", "uncost_value", "div_memo")
    out = {k: np.zeros((n, T)) for k in keys}
    approx = np.zeros(T, bool)
    for t in range(T):
        for e in ev_at.get(t, []):            # 開盤前調整
            if e["kind"] == "cash":
                out["div_memo"][:, t] += e["value"] * IL
                CL = np.where(IL > 0, CL - e["value"], CL)
                CS = np.where(IS > 0, CS - e["value"], CS)
            else:
                k = 1 + e["value"] if e["kind"] == "stock" else e["value"]
                IL, IS = IL * k, IS * k
                CL, CS = CL / k, CS / k
                approx[t] |= bool(e["approx"])
        b, s = pn.B[:, t], pn.S[:, t]
        vb = np.where(np.isfinite(pn.BP[:, t]), pn.BP[:, t], pn.vwap[t])
        vs = np.where(np.isfinite(pn.SP[:, t]), pn.SP[:, t], pn.vwap[t])
        dt = np.minimum(b, s)
        out["dt"][:, t] = dt
        out["dt_pnl"][:, t] = dt * (vs - vb) - (dt * vb * fee + dt * vs * (fee + tax_dt))
        net = b - s
        out["net"][:, t] = net
        real = np.zeros(n)
        # 淨買：先回補 S，再進 L
        buy = np.where(net > 0, net, 0.0)
        cov = np.minimum(buy, IS)
        real += cov * (CS - vb) - cov * vb * fee
        IS = IS - cov
        CS = np.where(IS > 0, CS, 0.0)
        add = buy - cov
        newL = IL + add
        CL = np.where(add > 0, (IL * CL + add * vb) / np.where(newL > 0, newL, 1), CL)
        IL = newL
        # 淨賣：先扣 L，超賣分給 S／U
        q = np.where(net < 0, -net, 0.0)
        c = np.minimum(q, IL)
        real += c * (vs - CL) - c * vs * (fee + tax)
        IL = IL - c
        CL = np.where(IL > 0, CL, 0.0)
        u = q - c
        tot_u = u.sum()
        frac = min(1.0, Q[t] / tot_u) if tot_u > 0 else 0.0
        dS = u * frac
        dU = u - dS
        newS = IS + dS
        CS = np.where(dS > 0, (IS * CS + dS * vs) / np.where(newS > 0, newS, 1), CS)
        IS = newS
        real -= dS * vs * (fee + tax)             # 空頭賣出當下的交易成本
        out["u_shares"][:, t] = dU
        out["uncost_value"][:, t] = dU * vs
        out["realized"][:, t] = real
        out["inventory"][:, t] = IL
        out["avg_cost"][:, t] = CL
        out["short_inv"][:, t] = IS
        out["short_cost"][:, t] = CS
    out["ca_approx"] = approx
    out["n_events"] = len(events)
    return out


def unrealized(L: dict, close: float, t: int) -> np.ndarray:
    return L["inventory"][:, t] * (close - L["avg_cost"][:, t]) + L["short_inv"][:, t] * (L["short_cost"][:, t] - close)
