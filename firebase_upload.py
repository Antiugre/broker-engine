"""
把本地清洗好的資料打包、壓縮後寫入 Firebase Firestore
====================================================

Firestore 結構（為了極致節省「寫入次數」）
----------------------------------------------------
免費額度：每天 20,000 次寫入 / 50,000 次讀取、單一文件上限 1 MiB。
寫入次數是「以文件計」，不是以資料筆數計，所以把一天的資料打包成少數幾份大文件最省。

  meta/latest                  ← 每次 1 寫。前端第一個讀的文件（目錄）
     date: "2026-10-28"         最新交易日
     dates: ["2026-10-28", …]   可選的日期（新→舊，最多 120 天）
     stocks: [{id, name}, …]    最新交易日有哪些股票
     updated: "2026-10-28T17:42:10+08:00"

  days/{YYYY-MM-DD}            ← 每天 1 寫（股票很多時自動切成 days/{日期}~1、~2…）
     date, parts: 1, part: 0
     hist: {"2330": "2026-10-28~0", …}   每檔的歷史在哪個 hist 文件
     payload: <bytes> gzip(JSON {代號: 當日分點紀錄})   ← 約 5–8 KB/檔

  hist/{YYYY-MM-DD}~{n}        ← 每天約 1 寫（約 25 檔塞一份；超過自動加一份）
     date, part
     payload: <bytes> gzip(JSON {代號: 近半年每日分點明細（欄位式陣列）})

  （以下 Ultra 文件都依分析範圍 h ∈ 6m／1y／2y／3y／5y 各一份，ID 後綴「@h」，例如 quant/latest@1y）
  quant/latest                 ← 每天 1 寫（Ultra 限定）追蹤股票的分點勝率統計
     date, params, payload: <bytes> gzip(JSON {代號: {分點key: {n, wr, ar, hd, style, high, ol, oc, …}}})
  signals/latest               ← 每天 1 寫（Ultra 限定）最新一天的跟單訊號，一般欄位、不壓縮
     date, count, items: [{stock_id, broker, key, wr, ar, style, net3, ol, oc, ur, …}]
  signals/{YYYY-MM-DD}         ← 同上的每日存檔（日後回測訊號用）
  edge/latest、edge/signals    ← Expectancy 引擎（quant_edge.py）的統計與「訊號＋緩步建倉」清單
  edge_hist/{YYYY-MM-DD}       ← Expectancy 清單的每日存檔
  flow/{stock_id}@h、flow/_index@h ← Order Flow（flowlab/schema.py）：行為標籤、群聚、成本防守區、異常背離

一天合計約 3 次寫入（meta + days + hist），與追蹤幾檔股票幾乎無關。
內容沒變就不寫（本地記錄 SHA-256），重跑腳本是 0 次寫入。

為什麼 payload 用 gzip 的 bytes 而不是一般欄位？
  1. 單一文件 1 MiB 上限 → 壓縮後可塞 5–10 倍資料
  2. Firestore 會為每個欄位/陣列元素建立索引（每份文件上限 40,000 個索引項），
     展開成欄位會撞上限且浪費儲存；一個 bytes 欄位只有 1 個（並建議在
     firestore.indexes.json 中排除索引）
  3. 前端用瀏覽器內建的 DecompressionStream('gzip') 解壓，不需額外套件

憑證
----------------------------------------------------
服務帳戶金鑰 JSON（Firebase 主控台 → 專案設定 → 服務帳戶 → 產生新的私密金鑰）
依序尋找：
  1. 環境變數 FIREBASE_CREDENTIALS
  2. 環境變數 GOOGLE_APPLICATION_CREDENTIALS
  3. secrets/firebase-service-account.json（已列入 .gitignore，絕對不要上傳）
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("broker_scraper")

ROOT = Path(__file__).resolve().parent
from paths import DATA_DIR  # noqa: E402  （可指向兩台電腦共用的同步資料夾）
STATE_FILE = DATA_DIR / ".firestore_state.json"
DEFAULT_CRED = ROOT / "secrets" / "firebase-service-account.json"
PART_LIMIT = 850_000  # 每份文件壓縮後的上限（保留餘裕給其他欄位，1 MiB = 1,048,576）
MAX_DATES = 120
SCHEMA = 1

# 前端用不到、或可由其他欄位推回的欄位，上傳前移除以縮小體積
DROP_RECORD_KEYS = {"url", "query_date", "history_range_url"}
DROP_ROW_KEYS: set[str] = set()  # b_raw 保留：網頁即時查詢分點歷史需要


# ──────────────────────────── 壓縮 / 雜湊 ────────────────────────────
def encode(obj) -> bytes:
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return gzip.compress(raw, compresslevel=9, mtime=0)  # mtime=0 → 內容相同時位元組相同，雜湊才穩定


def decode(blob: bytes):
    return json.loads(gzip.decompress(blob).decode("utf-8"))


def sha(doc: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(doc):
        if k == "updated":
            continue
        v = doc[k]
        h.update(k.encode())
        h.update(v if isinstance(v, bytes) else json.dumps(v, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


# ──────────────────────────── 打包 ────────────────────────────
def load_day_records(date: str, stocks: list[str] | None = None) -> dict[str, dict]:
    day_dir = DATA_DIR / date
    out = {}
    for f in sorted(day_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        if stocks and rec.get("stock_id") not in stocks:
            continue
        out[rec["stock_id"]] = rec
    return out


def compact_record(rec: dict) -> dict:
    r = {k: v for k, v in rec.items() if k not in DROP_RECORD_KEYS}
    for side in ("buy_side", "sell_side"):
        rows = []
        for row in rec.get(side, []):
            row = {k: v for k, v in row.items() if k not in DROP_ROW_KEYS and v is not None}
            if row.get("history"):  # "history/2330/9268.json" → "9268"
                row["hkey"] = Path(row.pop("history")).stem
            rows.append(row)
        r[side] = rows
    return r


def pack_history(sid: str, rec: dict) -> dict | None:
    """把該檔所有分點的半年明細合併成「欄位式」結構：日期只存一次，每個分點只存數字陣列"""
    hdir = DATA_DIR / "history" / sid
    keys = {Path(row["history"]).stem for side in ("buy_side", "sell_side")
            for row in rec.get(side, []) if row.get("history")}
    hists = {}
    for k in sorted(keys):
        f = hdir / f"{k}.json"
        if f.exists():
            hists[k] = json.loads(f.read_text(encoding="utf-8"))
    if not hists:
        return None

    dates = sorted({r["date"] for h in hists.values() for r in h.get("rows", [])})
    close, price = {}, {}
    for h in hists.values():
        for r in h.get("rows", []):
            if r.get("close") is not None:
                close.setdefault(r["date"], r["close"])
            if r.get("price") is not None:
                price.setdefault(r["date"], r["price"])

    branches = {}
    for k, h in hists.items():
        by_date = {r["date"]: r for r in h.get("rows", [])}
        buy, sell, lots, cost = [], [], [], []
        prev_lots, prev_cost = 0, None
        for d in dates:  # 沒有交易的日子補 0，持有沿用前一日
            r = by_date.get(d)
            if r:
                prev_lots, prev_cost = r.get("hold_lots", prev_lots), r.get("hold_cost")
            buy.append(r["buy"] if r else 0)
            sell.append(r["sell"] if r else 0)
            lots.append(prev_lots)
            cost.append(prev_cost)
        branches[k] = {
            "broker": h.get("broker"), "branch_id": h.get("branch_id"),
            "city": h.get("city"), "district": h.get("district"),
            "buy": buy, "sell": sell, "hold_lots": lots, "hold_cost": cost,  # 每日陣列
            **{x: h.get(x) for x in ("avg_buy_cost", "avg_sell_cost", "avg_buy_cost_1m", "avg_sell_cost_1m",
                                     "total_buy", "total_sell", "total_net")},
            "final_lots": h.get("hold_lots"), "final_cost": h.get("hold_cost"),  # 區間結束時的持有
        }
    any_h = next(iter(hists.values()))
    return {
        "stock_id": sid, "stock_name": rec.get("stock_name"),
        "start": any_h.get("start"), "end": any_h.get("end"), "price_source": any_h.get("price_source"),
        "dates": dates, "close": [close.get(d) for d in dates], "price": [price.get(d) for d in dates],
        "branches": branches,
    }


def split_parts(items: dict[str, object], limit: int = PART_LIMIT) -> list[dict[str, object]]:
    """依壓縮後大小把 {代號: 資料} 分組，每組壓縮後 ≤ limit"""
    parts, cur, cur_size = [], {}, 0
    for sid, obj in items.items():
        size = len(encode(obj))
        if size > limit:
            raise ValueError(f"{sid} 單檔壓縮後 {size:,} bytes 超過上限，請減少歷史天數")
        if cur and cur_size + size > limit:
            parts.append(cur)
            cur, cur_size = {}, 0
        cur[sid] = obj
        cur_size += size
    if cur:
        parts.append(cur)
    return parts


def build_documents(date: str, stocks: list[str] | None = None, prev_meta: dict | None = None) -> dict[str, dict]:
    """回傳 {文件路徑: 欄位}，meta/latest 一定放最後"""
    records = load_day_records(date, stocks)
    if not records:
        raise FileNotFoundError(f"data/{date}/ 沒有可上傳的資料")

    docs: dict[str, dict] = {}
    # hist
    hist_items = {sid: h for sid, rec in records.items() if (h := pack_history(sid, rec))}
    hist_index = {}
    for n, part in enumerate(split_parts(hist_items)):
        doc_id = f"{date}~{n}"
        docs[f"hist/{doc_id}"] = {"date": date, "part": n, "schema": SCHEMA, "payload": encode(part)}
        hist_index.update({sid: doc_id for sid in part})

    # days
    day_parts = split_parts({sid: compact_record(rec) for sid, rec in records.items()})
    for n, part in enumerate(day_parts):
        doc_id = date if n == 0 else f"{date}~{n}"
        docs[f"days/{doc_id}"] = {"date": date, "part": n, "parts": len(day_parts), "schema": SCHEMA,
                                   "hist": {s: hist_index[s] for s in part if s in hist_index},
                                   "payload": encode(part)}

    # 分析範圍（6m／1y／2y／3y／5y）：每個範圍各一份文件，ID 加上「@範圍」，前端依使用者選擇只讀那一份
    #   用固定的 latest 文件：雲端 cron 也會覆寫 meta/latest，不能把指標放在 meta 裡
    is_latest = date >= ((prev_meta or {}).get("date") or "")  # 補傳舊日期時不蓋掉 latest
    horizons_of = lambda d: d.get("horizons") or {"5y": d}  # noqa: E731  舊格式（無 horizons）視為 5 年

    # Hit Ratio（quant.py）：quant/latest@h（統計，壓縮）、signals/latest@h（訊號，一般欄位）
    qf = DATA_DIR / "quant" / f"{date}.json"
    if qf.exists():
        q = json.loads(qf.read_text(encoding="utf-8"))
        for h, hq in horizons_of(q).items():
            items = hq.get("signals", [])[:300]
            sig = {"date": date, "horizon": h, "schema": SCHEMA, "count": len(items), "items": items,
                   "params": q.get("params", {}), "updated": q.get("updated")}
            if is_latest:
                docs[f"quant/latest@{h}"] = {"date": date, "horizon": h, "schema": SCHEMA, "params": q.get("params", {}),
                                             "payload": encode(hq.get("stocks", {}))}
                docs[f"signals/latest@{h}"] = sig
            docs[f"signals/{date}@{h}"] = sig  # 每日存檔，日後可回測訊號的後續表現

    # Expectancy（quant_edge.py）：edge/latest@h（統計，壓縮）、edge/signals@h（訊號＋緩步建倉）
    ef = DATA_DIR / "edge" / f"{date}.json"
    if ef.exists():
        e = json.loads(ef.read_text(encoding="utf-8"))
        for h, he in horizons_of(e).items():
            lists = {"date": date, "horizon": h, "schema": SCHEMA, "params": e.get("params", {}), "prior": he.get("prior", {}),
                     "signals": he.get("signals", [])[:200], "stealth": he.get("stealth", [])[:200], "updated": e.get("updated")}
            if is_latest:
                docs[f"edge/latest@{h}"] = {"date": date, "horizon": h, "schema": SCHEMA, "payload": encode(he.get("stocks", {}))}
                docs[f"edge/signals@{h}"] = lists
            docs[f"edge_hist/{date}@{h}"] = lists

    # Order Flow（flowlab）：每檔每範圍一份 flow/{代號}@{範圍}（前端單次讀取即可渲染）＋ flow/_index@{範圍}
    fdir = DATA_DIR / "flow" / date
    if fdir.exists() and is_latest:
        for f in sorted(fdir.glob("*.json")):
            doc_id = f.stem  # "2330@1y" 或 "_index@1y"
            docs[f"flow/{doc_id}"] = json.loads(f.read_text(encoding="utf-8"))

    # meta（最後寫）
    prev_dates = (prev_meta or {}).get("dates", [])
    dates = sorted(set(prev_dates) | {date}, reverse=True)[:MAX_DATES]
    latest = dates[0]
    import socket
    meta = {
        "date": latest, "dates": dates, "schema": SCHEMA,
        "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        # 本機（MandM／Nunu）完成標記：備援機據此判斷要不要接手；
        # cloud_done 讓雲端排程（Vercel cron）跳過同一天，不會用較少的股票覆蓋本機結果
        "pc_done": date, "pc_host": socket.gethostname(), "cloud_done": date,
    }
    if latest == date:
        meta["stocks"] = [{"id": s, "name": records[s].get("stock_name")} for s in records]
    else:  # 補傳舊日期時，不覆蓋最新日期的股票清單
        meta["stocks"] = (prev_meta or {}).get("stocks", [])
    # 內容版本：任何 days/hist 內容改變，meta 也會跟著改 → 前端以此判斷快取是否過期
    meta["version"] = hashlib.sha256("".join(sha(d) for d in docs.values()).encode()).hexdigest()[:12]
    if prev_meta and prev_meta.get("date") == latest and latest != date:
        meta["version"] = prev_meta.get("version", meta["version"])
    docs["meta/latest"] = meta
    return docs


# ──────────────────────────── 寫入 ────────────────────────────
def get_client():
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        path = os.environ.get("FIREBASE_CREDENTIALS") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or str(DEFAULT_CRED)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"找不到 Firebase 服務帳戶金鑰：{path}\n"
                "請到 Firebase 主控台 → 專案設定 → 服務帳戶 → 產生新的私密金鑰，存成 secrets/firebase-service-account.json")
        firebase_admin.initialize_app(credentials.Certificate(path))
    return firestore.client()


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")


def upload(date: str, stocks: list[str] | None = None, *, dry_run: bool = False, force: bool = False,
           client=None, out_dir: Path | None = None) -> dict:
    """打包並上傳。回傳 {written: [...], skipped: [...], bytes: {...}}"""
    state = load_state()
    db = None
    prev_meta = None
    if not dry_run:
        db = client or get_client()
        snap = db.collection("meta").document("latest").get()  # 1 次讀取，用來合併日期清單
        prev_meta = snap.to_dict() if snap.exists else None
    else:
        prev_meta = state.get("_meta")

    docs = build_documents(date, stocks, prev_meta)
    changed = {p: d for p, d in docs.items() if force or state.get(p) != sha(d)}
    sizes = {p: sum(len(v) if isinstance(v, bytes) else len(json.dumps(v, ensure_ascii=False).encode())
                    for v in d.values()) for p, d in docs.items()}
    for p, size in sizes.items():
        log.info("  %-24s %8s bytes %s", p, f"{size:,}", "（將寫入）" if p in changed else "（內容未變，略過）")

    if dry_run:
        if out_dir:  # 輸出解壓後內容，方便檢查或給前端離線測試
            out_dir.mkdir(parents=True, exist_ok=True)
            for p, d in docs.items():
                view = {k: (decode(v) if k == "payload" else v) for k, v in d.items()}
                (out_dir / (p.replace("/", "__") + ".json")).write_text(
                    json.dumps(view, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("試跑模式：不寫入 Firestore（若正式執行將寫入 %d 份文件）", len(changed))
        state["_meta"] = {k: docs["meta/latest"][k] for k in ("dates", "stocks")}
        save_state(state)
        return {"written": [], "would_write": list(changed), "skipped": [p for p in docs if p not in changed], "bytes": sizes}

    if changed:
        batch = db.batch()  # 一個 batch 原子性寫入：前端不會讀到「meta 已更新但 days 還沒寫」的狀態
        for p, d in changed.items():
            col, doc_id = p.split("/", 1)
            batch.set(db.collection(col).document(doc_id), d)
        batch.commit()
        for p, d in changed.items():
            state[p] = sha(d)
        save_state(state)
    log.info("Firestore 寫入 %d 份文件、略過 %d 份", len(changed), len(docs) - len(changed))
    return {"written": list(changed), "skipped": [p for p in docs if p not in changed], "bytes": sizes}
