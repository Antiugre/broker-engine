"""
Expectancy 引擎（Ultra 第二套分點評分；與 quant.py 的 Hit Ratio 引擎並存，使用者在選單切換）
================================================================================

與 Hit Ratio（quant.py：FIFO、勝率 > 70% 且 ≥ 5 筆）的差異
  1. 配對：FIFO / LIFO 雙軌；當沖部位獨立拆出；幽靈庫存清洗（逾期結算或剔除、極值重置）
  2. 摩擦成本：當沖證交稅 0.15%、波段 0.3%；買進 +k 檔、賣出 −k 檔滑價
  3. 統計：Beta-Binomial 貝氏平滑勝率（經驗貝氏先驗），以「每筆期望報酬」排序
  4. 動態門檻：近 3 日買超「金額」或「佔成交量比」，取代固定 30 張
  5. 緩步建倉（Stealth Accumulation）：20 日窗口內連續小量、單向、分散的進貨

資料處理架構（Pandas）
  daily frame（每個 股票×分點 一張，index = 交易日曆）
      buy, sell, net, intraday            ← 富邦單一分點歷史（缺的交易日補 0）
      vwap, close, vol                    ← 個股日成交資料（FinMind / 證交所）
      tick, buy_px, sell_px               ← 向量化計算升降單位與滑價後成交價
      vol3, vol20, amt                    ← rolling 視窗（動態門檻、幽靈上限、建倉濾網）
  配對本身是有狀態的序列運算（庫存 deque），以 numpy 陣列逐日走一次 O(T)；
  其餘特徵全部是向量化 / rolling 運算。
  trades DataFrame → groupby 彙總 → 經驗貝氏先驗（同一批所有分點合併估計）→ 平滑後指標
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import quant

log = logging.getLogger("broker_scraper")
LOT = 1000
EDGE_DIR = quant.DATA_DIR / "edge"


# ══════════════════════════════ 參數 ══════════════════════════════
@dataclass(frozen=True)
class EdgeParams:
    # 1. 配對與幽靈庫存
    track: str = "lifo"                 # 評分用哪一軌（lifo：貼近短線主力最新進出；fifo：會計成本）
    ghost_days: int = 120               # 庫存超過 N 個交易日沒被動用 → 視為匯撥殘留
    ghost_mode: str = "settle"          # settle：以當日價格強制結算（另列，不計勝率）；drop：直接剔除
    ghost_cap_days: float = 5.0         # 庫存 > 個股 20 日均量 × 5 → 極值重置（從最舊的砍）
    # 2. 摩擦成本
    fee_rate: float = 0.001425
    fee_discount: float = 1.0
    tax_swing: float = 0.003
    tax_daytrade: float = 0.0015
    slip_ticks: int = 1                 # 每邊滑價幾檔（1～2）
    # 3. 貝氏平滑
    prior_strength: float = 20.0        # 先驗等效樣本數 m（α + β）
    shrink_k: float = 10.0              # 平均賺／賠幅度向母體收斂的等效筆數
    default_p0: float = 0.5             # 母體樣本不足時的先驗勝率
    ev_min: float = 0.005               # 訊號：每筆期望報酬 > 0.5%
    p_min: float = 0.55                 # 訊號：平滑後勝率 > 55%
    min_trades: int = 3                 # 只防呆（真正的小樣本懲罰交給先驗）
    # 4. 動態門檻（近 signal_days 日淨買超）
    signal_days: int = 3
    min_amount: float = 30_000_000      # 買超金額 ≥ 3,000 萬
    min_vol_share: float = 0.03         # 或 買超張數 ≥ 近 3 日總成交量的 3%
    threshold_mode: str = "any"         # any：任一成立；all：兩者都要
    hold_ratio_min: float = 0.5
    # 5. 緩步建倉
    stealth_window: int = 20
    stealth_min_buy_days: int = 12      # 20 日中至少 12 天淨買
    stealth_max_day_share: float = 0.05 # 任一天淨買 ≤ 當日成交量 5%（不上榜）
    stealth_min_cum_share: float = 0.02 # 20 日累積淨買 ≥ 20 日成交量 2%
    stealth_min_direction: float = 0.6  # 方向一致度 Σnet / Σ(buy+sell)
    stealth_max_cv: float = 1.0         # 買進日的張數變異係數（越小越平均）
    stealth_max_top_day: float = 0.25   # 單日淨買 ≤ 窗口累積的 25%（不是一兩天爆量）
    stealth_max_price_run: float = 0.15 # 窗口內漲幅 ≤ 15%（還沒被拉開）

    @property
    def fee(self) -> float:
        return self.fee_rate * self.fee_discount


DEFAULT = EdgeParams()


# ══════════════════════════════ 資料框 ══════════════════════════════
def tick_size(px: pd.Series) -> pd.Series:
    """台股升降單位（股票）：<10 0.01、<50 0.05、<100 0.1、<500 0.5、<1000 1、≥1000 5"""
    return pd.Series(np.select([px < 10, px < 50, px < 100, px < 500, px < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0), index=px.index)


def build_frame(rows: list[dict], prices: dict[str, dict], p: EdgeParams = DEFAULT) -> pd.DataFrame:
    """一個 股票×分點 的每日資料框（以個股交易日為日曆，分點沒交易的日子補 0）"""
    px = pd.DataFrame.from_dict(prices, orient="index")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    for c in ("vwap", "close", "vol"):
        if c not in px:
            px[c] = np.nan
    px["vwap"] = px["vwap"].fillna(px["close"])
    br = pd.DataFrame(rows or [], columns=["date", "buy", "sell"])
    br["date"] = pd.to_datetime(br["date"])
    br = br.groupby("date")[["buy", "sell"]].sum()
    df = px[["vwap", "close", "vol"]].join(br, how="left")
    df[["buy", "sell"]] = df[["buy", "sell"]].fillna(0).astype(float)
    df = df[df["vwap"].notna()]
    if not br.empty:
        df = df[df.index >= br.index.min()]
    df["net"] = df["buy"] - df["sell"]
    df["intraday"] = np.minimum(df["buy"], df["sell"])       # 同日一買一賣 → 當沖部位
    df["tick"] = tick_size(df["vwap"])
    df["buy_px"] = df["vwap"] + p.slip_ticks * df["tick"]     # 買進吃上 k 檔
    df["sell_px"] = df["vwap"] - p.slip_ticks * df["tick"]    # 賣出讓 k 檔
    df["amt"] = df["net"] * df["vwap"] * LOT                  # 淨買超金額（元）
    df["vol3"] = df["vol"].rolling(p.signal_days, min_periods=1).sum()
    df["vol20"] = df["vol"].rolling(20, min_periods=5).mean()
    return df


# ══════════════════════════════ 1 + 2. 配對（FIFO／LIFO、當沖、幽靈清洗、摩擦成本）══════════════════════════════
def _ret(buy_px: float, sell_px: float, tax: float, p: EdgeParams) -> tuple[float, float]:
    """回傳（每股淨損益, 報酬率）：成本 = 買價 ×（1 + 手續費）；收入 = 賣價 ×（1 − 手續費 − 稅）"""
    cost = buy_px * (1 + p.fee)
    proceeds = sell_px * (1 - p.fee - tax)
    return proceeds - cost, (proceeds - cost) / cost


def match_inventory(df: pd.DataFrame, method: str = "fifo", p: EdgeParams = DEFAULT):
    """
    逐日處理（numpy 陣列，O(T)）：
      1. 當沖：q = min(買, 賣) 以同日買價／賣價配對，證交稅 0.15%，持有 0 天
      2. 波段：淨買 → 新增一批庫存；淨賣 → FIFO（最舊）或 LIFO（最新）扣抵，證交稅 0.3%
      3. 幽靈清洗：庫存年齡 > ghost_days 個交易日 → settle（以當日賣價結算，kind=ghost）或 drop
                   庫存總量 > 20 日均量 × ghost_cap_days → 從最舊的開始砍到上限（kind=cap）
    回傳 (trades DataFrame, 剩餘庫存 list[(i, date, qty, px)], 事件統計 dict)
    """
    dates = df.index.strftime("%Y-%m-%d").to_numpy()
    net, intra = df["net"].to_numpy(), df["intraday"].to_numpy()
    bpx, spx = df["buy_px"].to_numpy(), df["sell_px"].to_numpy()
    vol20 = df["vol20"].to_numpy()
    lots: deque = deque()                     # 每筆 [i, qty, px]
    out, ev = [], {"unmatched": 0.0, "ghost": 0.0, "cap": 0.0}

    def close(i, qty, kind, tax):
        while qty > 1e-9 and lots:
            lot = lots[0] if method == "fifo" or kind != "swing" else lots[-1]
            q = min(qty, lot[1])
            pnl, r = _ret(lot[2], spx[i], tax, p)
            out.append((dates[lot[0]], dates[i], q, lot[2], spx[i], q * LOT * pnl, r, i - lot[0], kind))
            lot[1] -= q
            qty -= q
            if lot[1] <= 1e-9:
                lots.popleft() if lot is lots[0] else lots.pop()
        return qty

    for i in range(len(dates)):
        if intra[i] > 0:  # 當沖：同日配對，不進庫存
            pnl, r = _ret(bpx[i], spx[i], p.tax_daytrade, p)
            out.append((dates[i], dates[i], intra[i], bpx[i], spx[i], intra[i] * LOT * pnl, r, 0, "daytrade"))
        if net[i] > 0:
            lots.append([i, net[i], bpx[i]])
        elif net[i] < 0:
            ev["unmatched"] += close(i, -net[i], "swing", p.tax_swing)
        # 幽靈庫存：逾期
        while lots and i - lots[0][0] > p.ghost_days:
            if p.ghost_mode == "settle":
                ev["ghost"] += lots[0][1]
                q = lots[0][1]
                lots[0][1] = q  # 交給 close 以 FIFO 方式結算這一批
                close(i, q, "ghost", p.tax_swing)
            else:
                ev["ghost"] += lots.popleft()[1]
        # 幽靈庫存：極值（遠超過市場量能，不可能是單一分點的真實部位）
        if not math.isnan(vol20[i]) and vol20[i] > 0:
            cap = vol20[i] * p.ghost_cap_days
            excess = sum(l[1] for l in lots) - cap
            while excess > 1e-9 and lots:
                q = min(excess, lots[0][1])
                lots[0][1] -= q
                excess -= q
                ev["cap"] += q
                if lots[0][1] <= 1e-9:
                    lots.popleft()

    cols = ["open_date", "close_date", "qty", "buy_px", "sell_px", "pnl", "ret", "hold", "kind"]
    trades = pd.DataFrame(out, columns=cols)
    remaining = [(l[0], dates[l[0]], l[1], l[2]) for l in lots if l[1] > 1e-9]
    return trades, remaining, ev


# ══════════════════════════════ 3. 貝氏平滑與期望值 ══════════════════════════════
def raw_summary(trades: pd.DataFrame) -> dict:
    """單一分點、單一軌道的原始彙總（只看波段交易；當沖與幽靈結算另列）"""
    sw = trades[trades["kind"] == "swing"]
    win = sw[sw["pnl"] > 0]
    loss = sw[sw["pnl"] <= 0]
    dt_ = trades[trades["kind"] == "daytrade"]
    return {
        "n": int(len(sw)), "w": int(len(win)),
        "avg_win": float(win["ret"].mean()) if len(win) else None,
        "avg_loss": float(-loss["ret"].mean()) if len(loss) else None,
        "raw_ar": float(sw["ret"].mean()) if len(sw) else None,
        "pnl": float(sw["pnl"].sum()),
        "hold": float(np.average(sw["hold"], weights=sw["qty"])) if len(sw) else None,
        "dn": int(len(dt_)), "dpnl": float(dt_["pnl"].sum()),
        "gpnl": float(trades.loc[trades["kind"] == "ghost", "pnl"].sum()),
    }


def empirical_prior(summaries: list[dict], p: EdgeParams = DEFAULT) -> dict:
    """經驗貝氏：用同一批所有分點的交易合併估計母體勝率與平均賺賠幅度"""
    n = sum(s["n"] for s in summaries)
    w = sum(s["w"] for s in summaries)
    wins = [(s["avg_win"], s["w"]) for s in summaries if s["avg_win"] is not None]
    losses = [(s["avg_loss"], s["n"] - s["w"]) for s in summaries if s["avg_loss"] is not None]
    wavg = lambda xs, d: (sum(a * b for a, b in xs) / sum(b for _, b in xs)) if xs and sum(b for _, b in xs) else d  # noqa: E731
    p0 = w / n if n >= 50 else p.default_p0
    return {"p0": p0, "alpha": p.prior_strength * p0, "beta": p.prior_strength * (1 - p0),
            "win": wavg(wins, 0.05), "loss": wavg(losses, 0.05), "pooled_n": n}


def beta_lower(a: float, b: float, q: float = 0.10) -> float:
    """Beta(a, b) 的下 q 分位數（有 scipy 用精確值，否則常態近似）"""
    try:
        from scipy.stats import beta
        return float(beta.ppf(q, a, b))
    except Exception:  # noqa: BLE001
        mean = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return max(0.0, mean - 1.2816 * sd)


def bayes(s: dict, prior: dict, p: EdgeParams = DEFAULT) -> dict:
    """
    p̃  = (w + α) / (n + α + β)                         平滑後勝率（Beta 後驗平均）
    W̃  = (w·W̄ + k·W₀) / (w + k)                         平均獲利幅度向母體收斂
    L̃  = (l·L̄ + k·L₀) / (l + k)                         平均虧損幅度向母體收斂
    EV = p̃·W̃ − (1 − p̃)·L̃                              每筆期望報酬（已含成本、滑價）
    lo = Beta(w + α, l + β) 的 10% 分位數                   保守勝率（排序次要鍵）
    """
    n, w = s["n"], s["w"]
    l_ = n - w
    a, b = w + prior["alpha"], l_ + prior["beta"]
    pw = a / (a + b)
    k = p.shrink_k
    win = ((s["avg_win"] or 0) * w + prior["win"] * k) / (w + k)
    loss = ((s["avg_loss"] or 0) * l_ + prior["loss"] * k) / (l_ + k)
    return {"pw": pw, "lo": beta_lower(a, b), "ev": pw * win - (1 - pw) * loss, "win": win, "loss": loss}


# ══════════════════════════════ 4. 動態門檻 ══════════════════════════════
def recent_flow(df: pd.DataFrame, p: EdgeParams = DEFAULT) -> dict:
    tail = df.tail(p.signal_days)
    net3 = float(tail["net"].sum())
    amt3 = float(tail["amt"].sum())
    vol3 = float(tail["vol"].sum()) if tail["vol"].notna().any() else None
    share = net3 / vol3 if vol3 else None
    flow = {"net3": net3, "amt3": amt3, "vs3": share, "since": tail.index[0].strftime("%Y-%m-%d") if len(tail) else None}
    return {**flow, "big": is_big(flow, p)}


def is_big(flow: dict, p: EdgeParams = DEFAULT) -> bool:
    """大額買超：金額 ≥ min_amount 或（and）佔近 3 日成交量 ≥ min_vol_share"""
    ok_amt = flow["amt3"] >= p.min_amount
    ok_share = flow["vs3"] is not None and flow["vs3"] >= p.min_vol_share
    passed = (ok_amt or ok_share) if p.threshold_mode == "any" else (ok_amt and ok_share)
    return bool(flow["net3"] > 0 and passed)


# ══════════════════════════════ 5. 緩步建倉 ══════════════════════════════
def stealth(df: pd.DataFrame, p: EdgeParams = DEFAULT) -> dict:
    """
    20 日窗口特徵（全部向量化）
      buy_days     淨買天數                      ≥ stealth_min_buy_days
      max_streak   最長連續淨買天數               （輔助）
      day_share    max(當日淨買 / 當日成交量)      ≤ 5%     → 每天都不上榜
      cum_share    Σ淨買 / Σ成交量                ≥ 2%     → 累積起來有份量
      direction    Σ淨買 / Σ(買+賣)               ≥ 0.6    → 單向、集中在買方
      cv           買進日張數的變異係數             ≤ 1.0    → 平均分散、不是一兩天爆量
      top_day      最大單日淨買 / Σ淨買             ≤ 25%
      price_run    窗口收盤漲幅                     ≤ 15%    → 還沒被拉開
    score = 0～1 的綜合分數（各條件達成度的平均），全部條件成立 → flag
    """
    w = df.tail(p.stealth_window)
    if len(w) < p.stealth_window // 2 or w["vol"].isna().all():
        return {"flag": False, "score": 0.0}
    pos = w["net"] > 0
    buy_days = int(pos.sum())
    streak = int((pos.groupby((~pos).cumsum()).cumsum()).max() or 0)
    day_share = float((w["net"].clip(lower=0) / w["vol"].replace(0, np.nan)).max(skipna=True) or 0)
    cum_net = float(w["net"].sum())
    cum_share = cum_net / float(w["vol"].sum()) if w["vol"].sum() else 0.0
    gross = float((w["buy"] + w["sell"]).sum())
    direction = cum_net / gross if gross else 0.0
    bd = w.loc[pos, "net"]
    cv = float(bd.std(ddof=0) / bd.mean()) if len(bd) > 1 and bd.mean() else 9.9
    top_day = float(bd.max() / cum_net) if cum_net > 0 and len(bd) else 1.0
    price_run = float(w["close"].iloc[-1] / w["close"].iloc[0] - 1) if w["close"].iloc[0] else 0.0
    checks = {
        "buy_days": buy_days >= p.stealth_min_buy_days,
        "day_share": day_share <= p.stealth_max_day_share,
        "cum_share": cum_share >= p.stealth_min_cum_share,
        "direction": direction >= p.stealth_min_direction,
        "cv": cv <= p.stealth_max_cv,
        "top_day": top_day <= p.stealth_max_top_day,
        "price_run": price_run <= p.stealth_max_price_run,
    }
    clip01 = lambda x: max(0.0, min(1.0, x))  # noqa: E731
    score = np.mean([
        clip01(buy_days / p.stealth_min_buy_days), clip01(cum_share / p.stealth_min_cum_share),
        clip01(direction / p.stealth_min_direction), clip01(p.stealth_max_cv / max(cv, 1e-9)),
        1.0 if checks["day_share"] else 0.5, 1.0 if checks["top_day"] else 0.5, 1.0 if checks["price_run"] else 0.5,
    ])
    return {"flag": all(checks.values()), "score": round(float(score), 3), "buy_days": buy_days, "streak": streak,
            "day_share": round(day_share, 4), "cum_net": round(cum_net), "cum_share": round(cum_share, 4),
            "direction": round(direction, 3), "cv": round(cv, 2), "top_day": round(top_day, 3),
            "price_run": round(price_run, 4), "checks": checks}


# ══════════════════════════════ 單一分點評估 ══════════════════════════════
def open_position(remaining, last_price, p: EdgeParams = DEFAULT) -> dict:
    lots = sum(q for _, _, q, _ in remaining)
    if lots <= 1e-9:
        return {"ol": 0, "oc": None, "os": None, "ur": None}
    cost = sum(q * px for _, _, q, px in remaining) / lots
    ur = None
    if last_price:
        ur = (last_price * (1 - p.fee - p.tax_swing) - cost * (1 + p.fee)) / (cost * (1 + p.fee))
    return {"ol": round(lots), "oc": round(cost, 2), "os": min(d for _, d, _, _ in remaining), "ur": ur}


def analyse(rows: list[dict], prices: dict[str, dict], p: EdgeParams = DEFAULT) -> dict:
    """單一 股票×分點：雙軌配對 + 原始彙總（貝氏平滑要等整批先驗算好，見 finalize）"""
    df = build_frame(rows, prices, p)
    if df.empty:
        return {}
    last = float(df["close"].iloc[-1]) if pd.notna(df["close"].iloc[-1]) else None
    recent = set(df.tail(p.signal_days).index.strftime("%Y-%m-%d"))
    out = {"from": df.index[0].strftime("%Y-%m-%d"), "lp": last}
    for track in ("fifo", "lifo"):
        trades, remaining, ev = match_inventory(df, track, p)
        held_recent = sum(q for _, d, q, _ in remaining if d in recent)
        out[track] = {**raw_summary(trades), **open_position(remaining, last, p), "held_recent": held_recent,
                      "ghost": round(ev["ghost"]), "cap": round(ev["cap"]), "um": round(ev["unmatched"])}
    out["flow"] = recent_flow(df, p)
    out["stealth"] = stealth(df, p)
    return out


def finalize(a: dict, prior: dict, p: EdgeParams = DEFAULT) -> dict:
    """套上貝氏平滑、判定訊號，輸出給前端的精簡欄位"""
    t = a[p.track]
    b = bayes(t, prior, p)
    flow, st = a["flow"], a["stealth"]
    edge = t["n"] >= p.min_trades and b["ev"] > p.ev_min and b["pw"] > p.p_min
    big = is_big(flow, p)
    signal = bool(edge and big and t["ol"] > 0 and t["held_recent"] >= p.hold_ratio_min * flow["net3"])
    r4 = lambda x: None if x is None else round(float(x), 4)  # noqa: E731
    return {
        "track": p.track, "n": t["n"], "w": t["w"], "pw": r4(b["pw"]), "lo": r4(b["lo"]), "ev": r4(b["ev"]),
        "raw_wr": r4(t["w"] / t["n"]) if t["n"] else None, "raw_ar": r4(t["raw_ar"]),
        "aw": r4(b["win"]), "al": r4(b["loss"]), "pnl": round(t["pnl"]), "hd": None if t["hold"] is None else round(t["hold"], 1),
        "style": quant.style_of(t["hold"]), "edge": bool(edge), "signal": signal,
        "dn": t["dn"], "dpnl": round(t["dpnl"]),
        "lifo": {k: a["lifo"][k] for k in ("ol", "oc", "os")} | {"ur": r4(a["lifo"]["ur"])},
        "fifo": {k: a["fifo"][k] for k in ("ol", "oc", "os")} | {"ur": r4(a["fifo"]["ur"])},
        "ghost": t["ghost"], "cap": t["cap"], "um": t["um"],
        "net3": round(flow["net3"]), "amt3": round(flow["amt3"]), "vs3": r4(flow["vs3"]), "big": big,
        "stealth": {k: v for k, v in st.items() if k != "checks"}, "lp": a["lp"], "from": a["from"],
    }


# ══════════════════════════════ 每日批次 ══════════════════════════════
def stealth_candidates(fetcher, sid: str, day: str, p: EdgeParams = DEFAULT) -> dict[str, dict]:
    """20 日區間合計的買超前 15 名（緩步建倉的分點常不在單日榜，但會在區間榜）"""
    from history import history_key

    rec = quant.range_record(fetcher, sid, day, int(p.stealth_window * 1.5))
    return {history_key(r): {**r, "stock_name": rec.get("stock_name"), "_hot": True} for r in rec.get("buy_side", [])
            if r.get("b_raw") and r.get("broker_id")}


def run(day: str, stocks: list[str], fetcher=None, p: EdgeParams = DEFAULT, progress=None) -> dict:
    own = fetcher is None
    if own:
        from scraper import RequestsFetcher
        fetcher = RequestsFetcher(1.0, 2.5)
    end = dt.date.fromisoformat(day)
    t0 = time.time()
    cands = quant.candidates(day, stocks, p.signal_days)
    analysed: dict[str, dict[str, dict[str, tuple[dict, dict]]]] = {h: {} for h in quant.HORIZONS}  # 範圍 → 代號 → 分點
    try:
        for sid in stocks:
            branches = dict(cands.get(sid, {}))
            try:
                for k, r in stealth_candidates(fetcher, sid, day, p).items():
                    branches.setdefault(k, r)
            except Exception as ex:  # noqa: BLE001
                log.warning("[%s] 20 日區間榜失敗：%s", sid, ex)
            if not branches:
                continue
            try:
                prices = quant.load_prices(sid, end)
            except Exception as ex:  # noqa: BLE001
                log.warning("[%s] 價格取得失敗：%s", sid, ex)
                continue
            prices = {d: v for d, v in prices.items() if d <= day}
            for i, (key, row) in enumerate(branches.items(), 1):
                if progress:
                    progress(f"{sid} Expectancy {row.get('broker')}", i, len(branches))
                try:
                    rows = quant.load_branch_history(fetcher, sid, row, end,
                                                     max_age=quant.HOT_MAX_AGE if row.get("_hot") else None)
                    for h in quant.HORIZONS:  # 各分析範圍分別配對（幽靈庫存、先驗也各自計算）
                        a = analyse(quant.clip_rows(rows, day, h), prices, p)
                        if a:
                            analysed[h].setdefault(sid, {})[key] = (row, a)
                except Exception as ex:  # noqa: BLE001
                    log.warning("[%s] %s Expectancy 失敗：%s", sid, row.get("broker"), ex)
    finally:
        if own:
            fetcher.close()

    result = {"date": day, "params": asdict(p), "horizons": {},
              "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    for h, by_stock in analysed.items():
        prior = empirical_prior([a[p.track] for per in by_stock.values() for _, a in per.values()], p)
        hr = {"prior": prior, "stocks": {}, "signals": [], "stealth": []}
        for sid, per in by_stock.items():
            hr["stocks"][sid] = {}
            for key, (row, a) in per.items():
                f = finalize(a, prior, p)
                hr["stocks"][sid][key] = {"broker": row.get("broker"), "branch_id": row.get("branch_id"), **f}
                ident = {"stock_id": sid, "stock_name": row.get("stock_name"), "key": key, "broker": row.get("broker"),
                         "broker_id": row.get("broker_id"), "b_raw": row.get("b_raw"), "branch_id": row.get("branch_id"),
                         "city": row.get("city"), "district": row.get("district")}
                if f["signal"]:
                    hr["signals"].append({**ident, **f})
                if f["stealth"].get("flag"):
                    hr["stealth"].append({**ident, **f})
        hr["signals"].sort(key=lambda s: (-(s["ev"] or 0), -(s["lo"] or 0)))      # 期望值為核心排序
        hr["stealth"].sort(key=lambda s: (-s["stealth"]["score"], -(s["ev"] or 0)))
        result["horizons"][h] = hr
    result.update(result["horizons"][quant.DEFAULT_HORIZON])  # 頂層 = 預設範圍
    prior = result["prior"]
    out = EDGE_DIR / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=float), encoding="utf-8")
    log.info("Expectancy：%d 個分點、%d 個訊號、%d 個緩步建倉（先驗 p0=%.3f，%.0f 秒）",
             sum(len(v) for v in result["stocks"].values()), len(result["signals"]), len(result["stealth"]),
             prior["p0"], time.time() - t0)
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Expectancy 引擎：貝氏期望值 + 緩步建倉")
    ap.add_argument("--date", required=True)
    ap.add_argument("--stocks", nargs="*")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import scraper
    r = run(a.date, a.stocks or scraper.load_watchlist())
    print(json.dumps({"signals": r["signals"][:10], "stealth": r["stealth"][:10]}, ensure_ascii=False, indent=1))
