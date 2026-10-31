"""
flowlab — Order Flow 分析（Ultra / Expectancy 面板中的第二套系統）

  python -m flowlab --date 2026-10-29 [--stocks 2330 2317]

模組
  core.py        StockContext、Tagger / Analyzer 基底與註冊表、run_pipeline（策略模式）
  tags.py        模組一 行為標籤：隔日沖、波段鎖碼、發動機
  cluster.py     模組二 群聚協同（共買矩陣、Jaccard、連通元件）與地緣比對 GeoMatcher
  zones.py       模組三 成本防守區、模組四 異常背離
  schema.py      模組五 Firestore 文件結構（flow/{stock_id}、flow/_index）
  sources.py     Extract：本機快取 → StockContext
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

import quant

from . import cluster, tags, zones  # noqa: F401  匯入即註冊 Tagger / Analyzer
from .core import ANALYZERS, TAGGERS, Analyzer, StockContext, Tagger, register_analyzer, register_tagger, run_pipeline
from .schema import build_document, index_entry

log = logging.getLogger("broker_scraper")
FLOW_DIR = quant.DATA_DIR / "flow"

__all__ = ["ANALYZERS", "TAGGERS", "Analyzer", "StockContext", "Tagger", "register_analyzer", "register_tagger",
           "run_pipeline", "build_document", "analyse_context", "run"]


def analyse_context(ctx: StockContext) -> dict:
    """Transform + Load（不連網，方便測試）"""
    return build_document(ctx, run_pipeline(ctx))


def run(day: str, stocks: list[str], fetcher=None, params: dict | None = None) -> dict:
    """每日批次：每檔 × 每個分析範圍輸出 data/flow/<day>/<代號>@<範圍>.json，另有 _index@<範圍>.json"""
    from .sources import extract_context

    own = fetcher is None
    if own:
        from scraper import RequestsFetcher
        fetcher = RequestsFetcher(1.0, 2.5)
    t0 = time.time()
    out_dir = FLOW_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    indexes = {h: {"asof": day, "horizon": h, "stocks": {}} for h in quant.HORIZONS}
    try:
        for sid in stocks:
            try:
                full = extract_context(fetcher, sid, day, params)
                if full is None:
                    continue
                for h, days in quant.HORIZONS.items():  # 同一份資料，依各分析範圍各產出一份文件
                    start = (dt.date.fromisoformat(day) - dt.timedelta(days=days)).isoformat()
                    doc = analyse_context(full.slice(start, horizon=h))
                    (out_dir / f"{sid}@{h}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                    indexes[h]["stocks"][sid] = index_entry(doc)
            except Exception as ex:  # noqa: BLE001
                log.warning("[%s] Order Flow 失敗：%s", sid, ex)
    finally:
        if own:
            fetcher.close()
    for h, idx in indexes.items():
        (out_dir / f"_index@{h}.json").write_text(json.dumps(idx, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    index = indexes[quant.DEFAULT_HORIZON]
    log.info("Order Flow：%d 檔（%.0f 秒）", len(index["stocks"]), time.time() - t0)
    return index
