"""
flowlab.zones — 模組三：大戶成本防守區（Cost Support Zone）、模組四：籌碼異常背離

成本防守區（全部 rolling 向量化，每個交易日都有一筆 → 前端直接畫成帶狀區）
  成員 Mᵢ,d = 「近 20 日內曾被標為波段鎖碼」或「與公司同地緣（權重 > 1）」
  g_d      = Σᵢ∈M max(淨買ᵢ,d, 0)                            成員當日建倉張數
  mid_d    = Σ₂₀ Σᵢ gᵢ·Pᵢ / Σ₂₀ g                             近 20 日建倉的加權平均成本
  sd_d     = √(Σ₂₀ Σᵢ gᵢ·Pᵢ² / Σ₂₀ g − mid²)                  成本分散度
             Pᵢ,d＝分點 i 當日真實買進均價（FinMind Sponsor Pro）；只用真實價格，沒有真實價格的建倉不列入（不用均價估算）
             沒有 FinMind 資料（富邦備用路徑、本機查詢）→ 不輸出防守區
  upper    = mid + 0.5·sd，lower = mid − 1.0·sd（半寬至少 1%）  支撐偏下緣
  輸出：[{date, price_upper, price_lower, price_mid, lots, signal_type}]，不含任何圖表套件語法

異常背離
  r20       = 20 日報酬
  N5        = 鎖碼群近 5 日淨買合計；z = (N5 − 前 60 日均值) / 前 60 日標準差
  retail5   = 近 5 日「散戶型」分點（非法人、非鎖碼、非隔日沖）淨買 / 全部淨買方淨買
  low_absorb   低檔異常承接：r20 ≤ −15% 且 N5 > 0 且 z ≥ 2
  high_distrib 高檔籌碼發散：r20 ≥ +20% 且 N5 < 0 且 z ≤ −1.5 且 retail5 ≥ 60%
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cluster import GeoMatcher
from .core import Analyzer, StockContext, iso, r, register_analyzer


def member_masks(ctx: StockContext, window: int = 20):
    tm = ctx._cache.get("tag_masks", {})
    lock = tm.get("lock")
    lock_recent = (lock.astype(int).rolling(window, min_periods=1).max().astype(bool)
                   if lock is not None else pd.DataFrame(False, index=ctx.net.index, columns=ctx.net.columns))
    geo = GeoMatcher(ctx.company, ctx.zip_table)
    geo_cols = [k for k in ctx.net.columns if geo.weight(ctx.meta.get(k, {})) > 1.0]
    geo_mask = pd.DataFrame(False, index=ctx.net.index, columns=ctx.net.columns)
    if geo_cols:
        geo_mask[geo_cols] = True
    return lock_recent, geo_mask


@register_analyzer
class CostZoneAnalyzer(Analyzer):
    key, order = "zones", 30

    def run(self, ctx: StockContext, results: dict):
        w = ctx.p("zone_window", 20)
        lock_m, geo_m = member_masks(ctx, w)
        pos = ctx.net.clip(lower=0)
        g_lock = pos.where(lock_m, 0).sum(axis=1)
        g_geo = pos.where(geo_m & ~lock_m, 0).sum(axis=1)
        if ctx.bp is None:                                                 # 沒有分點真實價格：不估算
            return {"latest": None, "series": [], "cost_basis": "none"}
        gm = pos.where(lock_m | geo_m, 0)                                  # 成員各自的建倉張數
        P = ctx.bp.reindex(index=gm.index, columns=gm.columns)
        ok = P.notna() & (P > 0)
        gm = gm.where(ok, 0)                                               # 沒有真實買價的建倉不列入
        P = P.where(ok, 0)
        g_lock = gm.where(lock_m, 0).sum(axis=1)
        g_geo = gm.where(geo_m & ~lock_m, 0).sum(axis=1)
        g = g_lock + g_geo
        gp, gp2 = (gm * P).sum(axis=1), (gm * P * P).sum(axis=1)
        den = g.rolling(w, min_periods=1).sum()
        mid = gp.rolling(w, min_periods=1).sum() / den.replace(0, np.nan)
        sd = np.sqrt((gp2.rolling(w, min_periods=1).sum() / den.replace(0, np.nan) - mid ** 2).clip(lower=0))
        half_min = mid * ctx.p("zone_min_half", 0.01)
        upper = mid + np.maximum(ctx.p("zone_up_sd", 0.5) * sd, half_min)
        lower = mid - np.maximum(ctx.p("zone_dn_sd", 1.0) * sd, half_min)
        lk = g_lock.rolling(w, min_periods=1).sum()
        ge = g_geo.rolling(w, min_periods=1).sum()
        kind = np.where((lk > 0) & (ge > 0), "lock+geo", np.where(lk > 0, "lock", "geo"))
        df = pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "lots": den, "kind": kind})
        df = ctx.in_range(df[df["lots"] >= ctx.p("zone_min_lots", 100)]).dropna(subset=["mid"])
        series = [{"date": iso(d), "price_upper": r(x.upper, 2), "price_lower": r(x.lower, 2), "price_mid": r(x.mid, 2),
                   "lots": int(x.lots), "signal_type": x.kind} for d, x in df.iterrows()]
        latest = series[-1] if series and series[-1]["date"] == ctx.asof else None
        if latest:
            c = float(ctx.px["close"].iloc[-1])
            latest = {**latest, "close": r(c, 2), "position": "above" if c > latest["price_upper"] else "below" if c < latest["price_lower"] else "inside",
                      "gap": r(c / latest["price_mid"] - 1, 4)}
        return {"latest": latest, "series": series, "cost_basis": "branch"}     # 分點真實買進均價（FinMind）


@register_analyzer
class DivergenceAnalyzer(Analyzer):
    key, order = "divergence", 40

    def run(self, ctx: StockContext, results: dict):
        c = ctx.px["close"]
        r20 = c / c.shift(20) - 1
        lock_m, _ = member_masks(ctx, ctx.p("zone_window", 20))
        n5 = ctx.net.where(lock_m, 0).sum(axis=1).rolling(5, min_periods=5).sum()
        base = n5.shift(5)
        z = (n5 - base.rolling(60, min_periods=20).mean()) / base.rolling(60, min_periods=20).std().replace(0, np.nan)
        tm = ctx._cache.get("tag_masks", {})
        flip = tm.get("overnight")
        flip_any = flip.any() if flip is not None else pd.Series(False, index=ctx.net.columns)
        inst = pd.Series({k: ctx.is_institutional(k) for k in ctx.net.columns})
        retail_cols = [k for k in ctx.net.columns if not inst[k] and not flip_any[k]]
        net5 = ctx.net.rolling(5, min_periods=1).sum()
        retail_net = net5[retail_cols].where(~lock_m[retail_cols], 0).clip(lower=0).sum(axis=1) if retail_cols else 0 * n5
        buyers = net5.clip(lower=0).sum(axis=1)
        retail5 = retail_net / buyers.replace(0, np.nan)
        low = (r20 <= -ctx.p("div_drop", 0.15)) & (n5 > 0) & (z >= ctx.p("div_z_in", 2.0))
        high = (r20 >= ctx.p("div_rise", 0.20)) & (n5 < 0) & (z <= -ctx.p("div_z_out", 1.5)) & (retail5 >= ctx.p("div_retail", 0.6))
        events = []
        for typ, mask, msg in (("low_absorb", low, "低檔異常承接：股價 20 日跌 {r20}，鎖碼群近 5 日淨買 {n5} 張（z={z}）"),
                               ("high_distrib", high, "高檔籌碼發散：股價 20 日漲 {r20}，鎖碼群近 5 日淨賣 {n5} 張，散戶型承接 {rt}")):
            hit = ctx.in_range(mask.fillna(False))
            for d in hit[hit].index[-ctx.p("div_max_events", 60):]:
                e = {"date": iso(d), "type": typ, "r20": r(r20[d]), "lock_net5": int(n5[d]), "z": r(z[d], 2),
                     "retail5": r(retail5[d], 3)}
                e["message"] = msg.format(r20=f"{r20[d] * 100:+.1f}%", n5=f"{abs(int(n5[d])):,}", z=f"{z[d]:.1f}",
                                          rt=f"{(retail5[d] or 0) * 100:.0f}%")
                events.append(e)
        events.sort(key=lambda e: e["date"])
        recent = set(iso(d) for d in c.index[-ctx.p("div_active_days", 3):])
        return {"active": [e for e in events if e["date"] in recent], "events": events,
                "now": {"r20": r(r20.iloc[-1]), "lock_net5": None if pd.isna(n5.iloc[-1]) else int(n5.iloc[-1]),
                        "z": r(z.iloc[-1], 2), "retail5": r(retail5.iloc[-1], 3)}}
