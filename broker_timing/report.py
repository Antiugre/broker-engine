"""
§8 描述欄位（不參與檢定；網頁上與檢定結果分區顯示）
  擇時損益  Σ_t n_t · VWAP_t · r^ex_{t→t+20}（原始股數 × 原始 VWAP × 超額報酬）
  EXEC      Σ (VWAP − BP)(b − d) + Σ (SP − VWAP)(s − d)（原始價、原始股數）
  當沖損益  Σ d (SP − BP) − 費稅：沿用 v3 會計層 dt_pnl（v3 params accounting：fee_rate × fee_discount、tax_daytrade）
  已實現損益、來回勝率：v3 會計層（L 批）；註明「受處置效應影響，偏高」
  總損益    已實現（含當沖，同 v3 group_pnl）＋ 期末未實現（收盤盯市），排除 U 批
  位置分數 T、目前持股與成本、近一年趨勢（最近一個 240 日窗口的 θ̂_20、t_20）、最近一次動作
"""
from __future__ import annotations

import numpy as np

from . import hac
from .params import TP

LOT = 1000
NOTE_REALIZED = "受處置效應影響，偏高"
NOTE_POSITION = "事後描述，使用了交易日之後的價格"
NOTE_ROLLING = "只做描述，不標顯著"
PENDING = "尚無法評估"


def _r(x, d=2):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def exec_component(b, s, bp, sp, vwap) -> float:
    b, s, bp, sp, vwap = (np.asarray(a, float) for a in (b, s, bp, sp, vwap))
    d = np.minimum(b, s)
    xb, xs = b - d, s - d
    tb = np.where(xb > 0, (vwap - np.nan_to_num(bp, nan=0.0)) * xb, 0.0)
    ts = np.where(xs > 0, (np.nan_to_num(sp, nan=0.0) - vwap) * xs, 0.0)
    # 有淨買卻沒有買均價（理論上不會發生）→ 該日不計（SPEC-GAP-timing）
    tb = np.where(xb > 0, np.where(np.isfinite(bp), tb, 0.0), 0.0)
    ts = np.where(xs > 0, np.where(np.isfinite(sp), ts, 0.0), 0.0)
    return float(tb.sum() + ts.sum())


def timing_pnl(n, vwap, rex20) -> float:
    """SPEC-GAP-timing：只加總 r^ex_{t→t+20} 有定義的日子（樣本資格成立、且 t ≤ T − 20）"""
    v = np.asarray(n, float) * np.asarray(vwap, float) * np.asarray(rex20, float)
    return float(np.nansum(v))


def position_score(n, price, k: int | None = None, exclude_last: int | None = None) -> float:
    """SPEC-GAP-timing：v0 的位置分數定義不在本倉庫；本實作採
    pos_t = (P_t − min P[t−k, t+k]) / (max − min)（P 為還原收盤，同區間高低相等時 0.5），
    買進日分數 1 − pos、賣出日分數 pos，以 |n_t| 加權平均；排除最後 k 天（右側窗口不完整）。使用了交易日之後的價格。"""
    k = int(TP.position_k if k is None else k)
    ex = k if exclude_last is None else int(exclude_last)
    n = np.asarray(n, float)
    P = np.asarray(price, float)
    T = len(n)
    num = den = 0.0
    for t in np.flatnonzero(n != 0):
        if t >= T - ex:
            continue
        w = P[max(0, t - k): min(T, t + k + 1)]
        w = w[np.isfinite(w)]
        if not len(w) or not np.isfinite(P[t]):
            continue
        lo, hi = w.min(), w.max()
        pos = (P[t] - lo) / (hi - lo) if hi > lo else 0.5
        sc = 1 - pos if n[t] > 0 else pos
        num += abs(n[t]) * sc
        den += abs(n[t])
    return num / den if den > 0 else np.nan


def round_trips(realized, inventory) -> dict:
    """SPEC-GAP-timing：「來回」= L 批庫存從 0 變正、再回到 0 的一段；勝 = 該段已實現損益 > 0；只算已完成的來回"""
    real = np.asarray(realized, float)
    inv = np.asarray(inventory, float)
    n = w = 0
    open_ = False
    acc = 0.0
    prev = 0.0
    for t in range(len(inv)):
        if not open_ and prev <= 1e-9 and inv[t] > 1e-9:
            open_, acc = True, 0.0
        if open_:
            acc += real[t]
            if inv[t] <= 1e-9:
                n += 1
                w += int(acc > 0)
                open_ = False
        prev = inv[t]
    return {"n": n, "wins": w, "win_rate": (w / n) if n else None}


def rolling(eps_s: np.ndarray, y20_s: np.ndarray, sample_dates: np.ndarray, window: int | None = None, step: int | None = None) -> list[dict]:
    """§7 滾動窗口（只描述）：最後一個窗口結束在樣本最後一日，每 step 日往前；只輸出 θ̂_20 與 t_20，不輸出 p、不做 BH。
    SPEC-GAP-timing：殘差沿用全期間的反應函數（不在窗口內重估）；窗口內樣本不足 window 日時不輸出"""
    W = int(TP.rolling_window if window is None else window)
    st = int(TP.rolling_step if step is None else step)
    N = len(eps_s)
    out = []
    for e in range(N - 1, W - 2, -st):
        s = e - W + 1
        if s < 0:
            break
        x = eps_s[s:e + 1]
        if not (x != 0).any():
            out.append({"start": str(sample_dates[s]), "end": str(sample_dates[e]), "theta20": None, "t20": None})
            continue
        th, t = hac.single(y20_s[s:e + 1], x)
        out.append({"start": str(sample_dates[s]), "end": str(sample_dates[e]), "theta20": _r(th, 6), "t20": _r(t, 3)})
    return out[::-1]


def last_action(n, bp, sp, dates, pending: int | None = None) -> dict | None:
    pending = int(TP.pending_days if pending is None else pending)
    n = np.asarray(n, float)
    nz = np.flatnonzero(n != 0)
    if not len(nz):
        return None
    t = int(nz[-1])
    buy = n[t] > 0
    px = bp[t] if buy else sp[t]
    d = str(dates[t])
    txt = f"{d[5:7]}/{d[8:10]} {'買進' if buy else '賣出'} {abs(n[t]) / LOT:,.0f} 張" + (f" @{px:.2f}" if np.isfinite(px) else "")
    days_ago = len(n) - 1 - t
    return {"date": d, "side": "buy" if buy else "sell", "lots": round(abs(n[t]) / LOT, 3), "price": _r(px),
            "text": txt, "days_ago": days_ago, "status": PENDING if days_ago < pending else None}


def holdings(L: dict, i: int, t: int, close: float) -> dict:
    inv, cost = float(L["inventory"][i, t]), float(L["avg_cost"][i, t])
    sh, sc = float(L["short_inv"][i, t]), float(L["short_cost"][i, t])
    return {"long_lots": round(inv / LOT, 3), "long_cost": _r(cost) if inv > 0 else None,
            "short_lots": round(sh / LOT, 3), "short_cost": _r(sc) if sh > 0 else None,
            "u_lots": round(float(L["u_shares"][i, :t + 1].sum()) / LOT, 3),
            "uncost_value": round(float(L["uncost_value"][i, :t + 1].sum())),
            "unrealized": round(float(inv * (close - cost) + sh * (sc - close)))}


def pnl(L: dict, i: int, t: int, close: float) -> dict:
    """v3 會計層（§3.3）：已實現（不含當沖）、當沖、總損益 = 已實現 + 當沖 + 期末未實現（同 v3 group_pnl，排除 U 批）"""
    real = float(L["realized"][i, :t + 1].sum())
    dt = float(L["dt_pnl"][i, :t + 1].sum())
    unreal = float(L["inventory"][i, t] * (close - L["avg_cost"][i, t]) + L["short_inv"][i, t] * (L["short_cost"][i, t] - close))
    rt = round_trips(L["realized"][i, :t + 1], L["inventory"][i, :t + 1])
    return {"realized": round(real), "daytrade": round(dt), "unrealized": round(unreal), "total": round(real + dt + unreal),
            "round_trips": rt["n"], "win_rate": _r(rt["win_rate"], 3), "realized_note": NOTE_REALIZED,
            "dividend_memo": round(float(L["div_memo"][i, :t + 1].sum()))}
