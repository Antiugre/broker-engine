"""
flowlab.schema — Load：組成 Firestore 文件（前端一次讀取即可渲染）

  flow/{stock_id}@{horizon}  ← 每檔每個分析範圍（6m／1y／2y／3y／5y）每天 1 寫；
                               前端用 stock_id + 使用者選的範圍組出文件 ID，O(1) 讀到全部
  {
    schema: 1, stock_id, stock_name, asof, updated, horizon: "1y", range_start: "2025-10-31",
    price:      {close, r20},
    company:    {city, district, zip} | null,
    tag_counts: {lock: 3, overnight: 1, igniter: 0},
    tags:       [{key, broker, branch_id, city, district, net20,
                  tags: [{type, label, hits, last, strength, …}]}],        ← (1) 行為標籤陣列
    clusters:   [{members: [{key, broker, big_days, active, geo}], size, basis,
                  co_buys, cohesion, geo_weight, active_now, last_together, score}],
    zones:      {latest: {date, price_upper, price_lower, price_mid, lots, signal_type, close, position, gap},
                 series: [{date, price_upper, price_lower, price_mid, lots, signal_type}]},   ← (2) 主力成本區間
    divergence: {now: {r20, lock_net5, z, retail5},
                 active: [...], events: [{date, type, r20, lock_net5, z, retail5, message}]},  ← (3) 背離訊號
    errors:     {分析器: 錯誤訊息}（只有出錯時才有）
  }

  flow/_index@{horizon}      ← 每天每個範圍 1 寫；面板開頭的「全部股票摘要」（也是單次讀取）
  { asof, stocks: {代號: {name, tags: {lock: n, …}, alerts: ["low_absorb"], zone: {mid, position}}} }

設計取捨
  ・不壓縮、一般欄位：前端拿到就能畫，不必解壓；每檔約 10–40 KB，遠低於 1 MiB 上限
  ・陣列用 [{date, …}] 物件形式：與自訂繪圖引擎的資料介面一致、可讀性高
  ・所有欄位都是 JSON 原生型別（字串、數字、布林、null），不含任何圖表套件語法
"""
from __future__ import annotations

import datetime as dt

from .core import StockContext, TAGGERS, r

SCHEMA = 1


def build_document(ctx: StockContext, results: dict) -> dict:
    c = ctx.px["close"]
    tags = results.get("tags") or []
    counts = {k: 0 for k in TAGGERS}
    for b in tags:
        for t in b["tags"]:
            counts[t["type"]] = counts.get(t["type"], 0) + 1
    doc = {
        "schema": SCHEMA, "stock_id": ctx.stock_id, "stock_name": ctx.stock_name, "asof": ctx.asof,
        "horizon": ctx.params.get("horizon"), "range_start": ctx.start.strftime("%Y-%m-%d"),
        "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "price": {"close": r(c.iloc[-1], 2), "r20": r(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 20 else None},
        "company": {k: (ctx.company or {}).get(k) for k in ("city", "district", "zip")} if ctx.company else None,
        "tag_counts": counts, "tags": tags,
        "clusters": results.get("clusters") or [],
        "zones": results.get("zones") or {"latest": None, "series": []},
        "divergence": results.get("divergence") or {"now": None, "active": [], "events": []},
    }
    for k, v in results.items():          # 日後新增的分析器：自動成為文件的新欄位
        if k not in doc and not k.startswith("_"):
            doc[k] = v
    if results.get("_errors"):
        doc["errors"] = results["_errors"]
    return doc


def index_entry(doc: dict) -> dict:
    z = (doc.get("zones") or {}).get("latest") or {}
    return {"name": doc.get("stock_name"), "tags": doc.get("tag_counts", {}),
            "alerts": sorted({e["type"] for e in (doc.get("divergence") or {}).get("active", [])}),
            "clusters": len(doc.get("clusters") or []),
            "zone": {"mid": z.get("price_mid"), "position": z.get("position")} if z else None}
