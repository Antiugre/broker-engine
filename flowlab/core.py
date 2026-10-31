"""
flowlab.core — Order Flow 分析管線的骨架（ETL + 策略模式）
==========================================================

  Extract   sources.extract_context()     本機快取（分點 5 年明細、個股日成交）→ StockContext
  Transform 依序執行 ANALYZERS 中註冊的分析器；每個分析器只讀 ctx 與前面分析器的結果，
            輸出自己那一段 JSON（tags、clusters、zones、divergence …）
  Load      schema.build_document()        把各段組成「前端可直接繪圖」的單一文件 flow/{stock_id}

擴充方式（不改主程式）
  ・新的分點行為標籤 → 寫一個繼承 Tagger 的類別，加上 @register_tagger
  ・新的分析段落     → 寫一個繼承 Analyzer 的類別，加上 @register_analyzer（key 即文件欄位名）

StockContext 的矩陣都是「交易日 × 分點」的 DataFrame（同一個 index / columns），
所有 rolling 計算一次對整個矩陣做，避免對分點逐一迴圈。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 外資／法人券商（用來把「散戶型」買盤分出來；名稱包含任一字即視為非散戶）
INSTITUTIONAL_HINTS = ("美林", "摩根", "高盛", "瑞銀", "花旗", "麥格理", "匯豐", "野村", "法銀", "巴黎", "德意志", "瑞士信貸",
                       "大和", "港商", "美商", "新加坡商", "英商", "法商", "德商", "日商", "台灣摩根", "上海匯豐", "自營")


@dataclass
class StockContext:
    stock_id: str
    stock_name: str | None
    asof: str
    px: pd.DataFrame                       # index=交易日，columns: vwap, close, vol（張）
    buy: pd.DataFrame                      # 交易日 × 分點（張）
    sell: pd.DataFrame
    meta: dict[str, dict]                  # 分點 key → {broker, branch_id, city, district, broker_id, b_raw}
    company: dict | None = None            # {city, district, zip}（地緣比對用）
    zip_table: dict | None = None          # {(city, district): zip3}
    params: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    # ── 常用衍生矩陣（第一次取用時計算，之後快取）──
    def _memo(self, k, fn):
        if k not in self._cache:
            self._cache[k] = fn()
        return self._cache[k]

    @property
    def net(self) -> pd.DataFrame:
        return self._memo("net", lambda: self.buy - self.sell)

    @property
    def gross(self) -> pd.DataFrame:
        return self._memo("gross", lambda: self.buy + self.sell)

    @property
    def vol(self) -> pd.Series:
        return self.px["vol"]

    @property
    def share(self) -> pd.DataFrame:
        """分點淨買 / 當日成交量"""
        return self._memo("share", lambda: self.net.div(self.vol.replace(0, np.nan), axis=0))

    def p(self, name: str, default):
        return self.params.get(name, default)

    # ── 分析範圍：rolling 需要暖機資料，所以矩陣保留 start 前 warmup 天，輸出只取 start 之後 ──
    @property
    def start(self) -> pd.Timestamp:
        s = self.params.get("range_start")
        return pd.Timestamp(s) if s else self.px.index[0]

    def in_range(self, obj):
        """DataFrame / Series 只取分析範圍內的日期"""
        return obj[obj.index >= self.start]

    def slice(self, start: str, warmup: int = 80, horizon: str | None = None) -> "StockContext":
        """回傳只含 [start − warmup 個交易日, asof] 的新 context（各範圍獨立快取）"""
        idx = self.px.index
        pos = max(0, int(idx.searchsorted(pd.Timestamp(start))) - warmup)
        keep = idx[pos:]
        return StockContext(self.stock_id, self.stock_name, self.asof, self.px.loc[keep], self.buy.loc[keep],
                            self.sell.loc[keep], self.meta, self.company, self.zip_table,
                            {**self.params, "range_start": start, "horizon": horizon})

    def is_institutional(self, key: str) -> bool:
        name = self.meta.get(key, {}).get("broker") or ""
        return any(h in name for h in INSTITUTIONAL_HINTS)


# ══════════════════════════════ 模組一：行為標籤（策略）══════════════════════════════
class Tagger(ABC):
    """一種分點行為標籤。mask() 回傳「交易日 × 分點」布林矩陣：該日是否符合此行為"""
    key: str = ""
    label: str = ""
    active_days: int = 10          # 最近幾個交易日內有命中，就算「目前帶有此標籤」
    min_hits: int = 1              # 回看期間至少命中幾次

    @abstractmethod
    def mask(self, ctx: StockContext) -> pd.DataFrame: ...

    def strength(self, ctx: StockContext, mask: pd.DataFrame) -> pd.Series:
        """每個分點的強度（0～1，前端排序與透明度用）；預設 = 回看期間命中比例"""
        return mask.tail(self.active_days).mean().clip(0, 1)

    def summarize(self, ctx: StockContext, mask: pd.DataFrame) -> list[dict]:
        recent = mask.tail(self.active_days)
        hits = recent.sum()
        st = self.strength(ctx, mask)
        last = mask.apply(lambda c: c[c].index.max() if c.any() else pd.NaT)
        out = []
        for k in hits[hits >= self.min_hits].index:
            out.append({"key": k, "type": self.key, "label": self.label, "hits": int(hits[k]),
                        "last": last[k].strftime("%Y-%m-%d") if pd.notna(last[k]) else None,
                        "strength": round(float(st.get(k, 0) or 0), 3)})
        return out


TAGGERS: dict[str, type[Tagger]] = {}


def register_tagger(cls: type[Tagger]) -> type[Tagger]:
    TAGGERS[cls.key] = cls
    return cls


# ══════════════════════════════ 分析器（策略）══════════════════════════════
class Analyzer(ABC):
    """管線中的一段 Transform；key = 輸出到 Firestore 文件的欄位名"""
    key: str = ""
    order: int = 100

    @abstractmethod
    def run(self, ctx: StockContext, results: dict) -> object: ...


ANALYZERS: dict[str, type[Analyzer]] = {}


def register_analyzer(cls: type[Analyzer]) -> type[Analyzer]:
    ANALYZERS[cls.key] = cls
    return cls


def run_pipeline(ctx: StockContext) -> dict:
    """Transform：依 order 執行所有註冊的分析器；某段失敗只影響那一段"""
    results: dict = {}
    for cls in sorted(ANALYZERS.values(), key=lambda c: c.order):
        try:
            results[cls.key] = cls().run(ctx, results)
        except Exception as ex:  # noqa: BLE001
            results[cls.key] = None
            results.setdefault("_errors", {})[cls.key] = f"{type(ex).__name__}: {ex}"
    return results


# ── 小工具 ──
def rolling_cv(values: pd.DataFrame, mask: pd.DataFrame, window: int) -> pd.DataFrame:
    """只看 mask 為 True 的日子，計算 rolling 變異係數（向量化：Σx、Σx² 的 rolling）"""
    x = values.where(mask, 0.0)
    n = mask.astype(float).rolling(window, min_periods=1).sum()
    s1 = x.rolling(window, min_periods=1).sum()
    s2 = (x * x).rolling(window, min_periods=1).sum()
    mean = s1 / n.replace(0, np.nan)
    var = (s2 / n.replace(0, np.nan) - mean ** 2).clip(lower=0)
    return np.sqrt(var) / mean.replace(0, np.nan)


def iso(d) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), n)
