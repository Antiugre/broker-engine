"""
flowlab.tags — 模組一：分點行為標籤（每種標籤一個類別；新增標籤只要再寫一個類別）

  overnight  隔日沖   當日買進 ≥ 當日成交量 10%，且下一交易日賣出 ≥ 前一日買進的 70%
                      回看 120 日至少 2 次、且佔該分點大買日數一半以上 → 高頻隔日沖
  lock       波段鎖碼 近 10 個交易日 ≥ 7 天淨買，且淨買日張數的變異係數 ≤ 0.6（穩定）
  igniter    發動機   前 60 日交易天數 ≤ 6（平常極少交易），當日淨買 ≥ 前 60 日平均買量 × 5，
                      且至少 max(50 張, 當日量 1%)（避免 0.2 張 → 1 張也算 5 倍）
"""
from __future__ import annotations

import pandas as pd

from .core import Analyzer, StockContext, Tagger, TAGGERS, register_analyzer, register_tagger, rolling_cv


@register_tagger
class OvernightFlipper(Tagger):
    key, label = "overnight", "隔日沖"
    active_days, min_hits = 120, 2

    def mask(self, ctx: StockContext) -> pd.DataFrame:
        big = ctx.buy.div(ctx.vol.replace(0, pd.NA), axis=0) > ctx.p("overnight_buy_share", 0.10)
        flip = ctx.sell.shift(-1) >= ctx.p("overnight_sell_ratio", 0.70) * ctx.buy
        return (big & flip).fillna(False).astype(bool)

    def summarize(self, ctx, mask):
        big = (ctx.buy.div(ctx.vol.replace(0, pd.NA), axis=0) > ctx.p("overnight_buy_share", 0.10)).fillna(False)
        ratio = mask.tail(self.active_days).sum() / big.tail(self.active_days).sum().replace(0, pd.NA)
        keep = set(ratio[ratio >= 0.5].dropna().index)
        return [t | {"flip_ratio": round(float(ratio[t["key"]]), 3)} for t in super().summarize(ctx, mask) if t["key"] in keep]


@register_tagger
class SwingLock(Tagger):
    key, label = "lock", "波段鎖碼"
    active_days = 5

    def mask(self, ctx: StockContext) -> pd.DataFrame:
        w = ctx.p("lock_window", 10)
        pos = ctx.net > 0
        days = pos.astype(int).rolling(w, min_periods=w).sum()
        cv = rolling_cv(ctx.net, pos, w)
        return ((days >= ctx.p("lock_min_days", 7)) & (cv <= ctx.p("lock_max_cv", 0.6))).fillna(False)

    def strength(self, ctx, mask):
        return (ctx.net > 0).tail(ctx.p("lock_window", 10)).mean()


@register_tagger
class Igniter(Tagger):
    key, label = "igniter", "發動機"
    active_days = 5

    def mask(self, ctx: StockContext) -> pd.DataFrame:
        w = ctx.p("igniter_window", 60)
        active = (ctx.gross > 0).astype(int).shift(1).rolling(w, min_periods=w // 2).sum()
        base = ctx.buy.shift(1).rolling(w, min_periods=w // 2).mean()
        floor = (ctx.vol * 0.01).clip(lower=ctx.p("igniter_min_lots", 50))
        return ((active <= ctx.p("igniter_max_active_days", 6))
                & (ctx.net >= ctx.p("igniter_mult", 5.0) * base)
                & ctx.net.ge(floor, axis=0)).fillna(False)

    def strength(self, ctx, mask):
        return ctx.share.where(mask).tail(self.active_days).max().fillna(0).clip(0, 0.2) / 0.2


@register_analyzer
class TagAnalyzer(Analyzer):
    """執行所有註冊的 Tagger，輸出「每個分點帶哪些標籤」＋ 供後續模組使用的布林矩陣"""
    key, order = "tags", 10

    def run(self, ctx: StockContext, results: dict):
        masks, by_branch = {}, {}
        span = len(ctx.in_range(ctx.px))  # 分析範圍內的交易日數
        for key, cls in TAGGERS.items():
            t = cls()
            if key == "overnight":           # 隔日沖看「整個分析範圍」內的頻率
                t.active_days = max(20, span)
            m = t.mask(ctx)
            masks[key] = m
            for item in t.summarize(ctx, m):
                b = by_branch.setdefault(item["key"], {"key": item["key"], **_ident(ctx, item["key"]), "tags": []})
                b["tags"].append({k: v for k, v in item.items() if k != "key"})
        ctx._cache["tag_masks"] = masks  # 給群聚、成本區、背離模組用（不輸出到文件）
        rows = list(by_branch.values())
        for b in rows:
            b["net20"] = int(ctx.net[b["key"]].tail(20).sum())
        rows.sort(key=lambda b: (-len(b["tags"]), -abs(b["net20"])))
        return rows


def _ident(ctx: StockContext, key: str) -> dict:
    m = ctx.meta.get(key, {})
    return {k: m.get(k) for k in ("broker", "branch_id", "broker_id", "b_raw", "city", "district")}
