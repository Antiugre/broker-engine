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
  force/{stock_id}             ← 本土主力（mainforce.py）：本土分點 5 年累計庫存推算的主力持股（Ultra 限定）
  smallcap/latest              ← 中小型吃貨（smallcap.py）：中小型股本土分點吃貨訊號＋回測摘要（Ultra 限定）
  syndicate/{stock_id}、_index ← 主力集團偵測（broker_flow，測試版）：窗口、集團、驗證摘要（管理員限定）

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


DOC_PAYLOAD_LIMIT = 950_000   # Firestore 單一文件上限 1 MiB，留一點給其他欄位


def encode_fit(stocks: dict, label: str = "") -> bytes:
    """壓縮後超過單一文件上限時，從排序較後面的股票開始拿掉「近 180 日進出序列」（rs／typ），直到放得下"""
    blob = encode(stocks)
    if len(blob) <= DOC_PAYLOAD_LIMIT:
        return blob
    stocks = json.loads(json.dumps(stocks))
    for sid in reversed(list(stocks)):
        def strip(o):
            if isinstance(o, dict):
                o.pop("rs", None); o.pop("typ", None)
                for v in o.values():
                    strip(v)
            elif isinstance(o, list):
                for v in o:
                    strip(v)
        strip(stocks[sid])
        blob = encode(stocks)
        if len(blob) <= DOC_PAYLOAD_LIMIT:
            log.warning("%s 壓縮後太大：拿掉部分股票的近 N 日序列（從 %s 起）才放得下（%d bytes）", label, sid, len(blob))
            return blob
    log.warning("%s 拿掉全部序列後仍有 %d bytes", label, len(blob))
    return blob


def split_payload(stocks: dict, limit: int | None = None) -> list[bytes]:
    """{代號: …} 壓縮後超過單一文件上限 → 依股票順序切成多份，每份都 ≤ 上限（網站依 parts 依序讀回合併）"""
    limit = limit or DOC_PAYLOAD_LIMIT
    whole = encode(stocks)
    if len(whole) <= limit:
        return [whole]
    sizes = {s: len(encode({s: v})) for s, v in stocks.items()}
    parts, cur, size = [], {}, 0
    for s, v in stocks.items():
        if cur and size + sizes[s] > limit * 0.9:
            parts.append(cur)
            cur, size = {}, 0
        cur[s] = v
        size += sizes[s]
    if cur:
        parts.append(cur)
    out = []
    for part in parts:
        blob = encode(part)
        if len(blob) > limit and len(part) > 1:   # 估計失準：再對半切
            items = list(part.items())
            out += split_payload(dict(items[: len(items) // 2]), limit) + split_payload(dict(items[len(items) // 2:]), limit)
        else:
            out.append(encode_fit(part, "單一股票") if len(blob) > limit else blob)
    return out


def payload_docs(base_id: str, fields: dict, stocks: dict) -> dict[str, dict]:
    """→ {base_id: {…fields, parts, payload}, base_id~1: {part, payload}, …}"""
    blobs = split_payload(stocks)
    docs = {base_id: {**fields, "parts": len(blobs), "payload": blobs[0]}}
    for k, blob in enumerate(blobs[1:], 1):
        docs[f"{base_id}~{k}"] = {"date": fields.get("date"), "horizon": fields.get("horizon"), "part": k, "payload": blob}
    return docs


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
    # 人工更正安全閘（fm_store）：更正檔沒載入 → 直接失敗；有更正的股票這次對不到更正 → 那檔不上傳（不蓋掉網站上正確的結果）
    import fm_store
    corr = fm_store.corrected_stocks()
    blocked = {s for s in corr if not fm_store.upload_ok(s)}
    for s in sorted(blocked):
        log.warning("人工更正：%s 這次沒有套用到更正，不上傳這檔的 Hit Ratio／Expectancy／Order Flow／本土主力", s)
    keep = lambda d: {k: v for k, v in (d or {}).items() if k not in blocked}  # noqa: E731
    keep_items = lambda xs: [x for x in (xs or []) if str(x.get("stock_id")) not in blocked]  # noqa: E731
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
            items = keep_items(hq.get("signals", []))[:300]
            sig = {"date": date, "horizon": h, "schema": SCHEMA, "count": len(items), "items": items,
                   "params": q.get("params", {}), "updated": q.get("updated")}
            if is_latest:
                docs.update(payload_docs(f"quant/latest@{h}", {
                    "date": date, "horizon": h, "schema": SCHEMA, "params": q.get("params", {}),
                    "rdates": hq.get("rdates", [])},  # 近 180 交易日（網頁自訂「近 N 日」用）
                    keep(hq.get("stocks", {}))))
                docs[f"signals/latest@{h}"] = sig
            docs[f"signals/{date}@{h}"] = sig  # 每日存檔，日後可回測訊號的後續表現

    # Expectancy（quant_edge.py）：edge/latest@h（統計，壓縮）、edge/signals@h（訊號＋緩步建倉）
    ef = DATA_DIR / "edge" / f"{date}.json"
    if ef.exists():
        e = json.loads(ef.read_text(encoding="utf-8"))
        for h, he in horizons_of(e).items():
            lists = {"date": date, "horizon": h, "schema": SCHEMA, "params": e.get("params", {}), "prior": he.get("prior", {}),
                     "signals": keep_items(he.get("signals", []))[:200], "stealth": keep_items(he.get("stealth", []))[:200],
                     "loaded": keep_items(he.get("loaded", []))[:200], "updated": e.get("updated")}
            if is_latest:
                docs.update(payload_docs(f"edge/latest@{h}", {"date": date, "horizon": h, "schema": SCHEMA}, keep(he.get("stocks", {}))))
                docs[f"edge/signals@{h}"] = lists
            docs[f"edge_hist/{date}@{h}"] = lists

    # Order Flow（flowlab）：每檔每範圍一份 flow/{代號}@{範圍}（前端單次讀取即可渲染）＋ flow/_index@{範圍}
    fdir = DATA_DIR / "flow" / date
    if fdir.exists() and is_latest:
        for f in sorted(fdir.glob("*.json")):
            doc_id = f.stem  # "2330@1y" 或 "_index@1y"
            sid = doc_id.split("@")[0]
            if sid in blocked:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            if (st := fm_store.stamp(sid)):
                d["corrections"] = st          # 套用了哪一版人工更正（網站 Order Flow 會顯示）
            docs[f"flow/{doc_id}"] = d

    # 本土主力（mainforce.py）：每檔一份 force/{代號}（Ultra 限定）
    mdir = DATA_DIR / "force" / date
    if mdir.exists() and is_latest:
        for f in sorted(mdir.glob("*.json")):
            if f.stem in blocked:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            if (st := fm_store.stamp(f.stem)):
                d["corrections"] = st
            docs[f"force/{f.stem}"] = d
    # 中小型吃貨（smallcap.py）：今天的訊號＋回測摘要，一份 smallcap/latest（Ultra 限定）
    sfile = DATA_DIR / "smallcap" / f"{date}.json"
    if sfile.exists() and is_latest:
        d = json.loads(sfile.read_text(encoding="utf-8"))
        if blocked and isinstance(d.get("signals"), list):
            d["signals"] = keep_items(d["signals"])
        docs["smallcap/latest"] = d

    # 主力集團偵測（broker_flow，測試版，管理員限定）：每檔一份 syndicate/{代號} ＋ syndicate/_index
    bdir = DATA_DIR / "broker_flow"
    if bdir.exists() and is_latest:
        idx = []
        for f in sorted(bdir.glob("*/summary.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("asof") == date and d["stock_id"] not in blocked:
                docs[f"syndicate/{d['stock_id']}"] = d
            idx.append({"id": d["stock_id"], "name": d.get("stock_name") or "", "asof": d.get("asof"), "kind": d.get("kind", "stock")})
        if idx:
            docs["syndicate/_index"] = {"stock_id": "_index", "stocks": idx}

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


MAX_BATCH_BYTES = 3_500_000   # 每批約 3.5 MB（Firestore 單次請求上限 10 MiB，留足餘裕）
MAX_BATCH_DOCS = 200          # 每批最多 200 份（上限 500）


def _chunks(paths: list[str], sizes: dict[str, int]):
    cur, size = [], 0
    for p in paths:
        n = sizes.get(p, 0)
        if cur and (size + n > MAX_BATCH_BYTES or len(cur) >= MAX_BATCH_DOCS):
            yield cur
            cur, size = [], 0
        cur.append(p)
        size += n
    if cur:
        yield cur


def _commit(db, items: list[tuple[str, dict]], *, label: str = "", tries: int = 5) -> None:
    """寫入一批；逾時／暫時性錯誤時等一下再重試（最多 5 次）"""
    import time
    for attempt in range(1, tries + 1):
        batch = db.batch()
        for p, d in items:
            col, doc_id = p.split("/", 1)
            batch.set(db.collection(col).document(doc_id), d)
        try:
            batch.commit(timeout=300)
            log.info("  Firestore 第 %s 批：%d 份完成", label, len(items))
            return
        except Exception as e:  # noqa: BLE001 — DeadlineExceeded / ServiceUnavailable / Aborted 等
            if attempt == tries:
                raise
            wait = 15 * attempt
            log.warning("  Firestore 第 %s 批寫入失敗（%s），%d 秒後重試（%d/%d）", label, type(e).__name__, wait, attempt, tries)
            time.sleep(wait)


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
        # 分批寫入：Order Flow 每天數百份、每份 20～170 KB，一次全塞進同一個 batch 會超過 Firestore 的
        # 請求大小／時間上限（2026-10-31 雲端執行即因此 504 Deadline Exceeded）。
        # meta/* 放最後一批：前端看到新的日期清單時，其他文件都已寫好。
        order = sorted(changed, key=lambda p: (p.startswith("meta/"), p))
        chunks = list(_chunks(order, sizes))
        for i, chunk in enumerate(chunks, 1):
            _commit(db, [(p, changed[p]) for p in chunk], label=f"{i}/{len(chunks)}")
            for p in chunk:
                state[p] = sha(changed[p])
            save_state(state)  # 每批寫完就記錄：中途失敗時，下次只補沒寫成功的
    log.info("Firestore 寫入 %d 份文件、略過 %d 份", len(changed), len(docs) - len(changed))
    return {"written": list(changed), "skipped": [p for p in docs if p not in changed], "bytes": sizes}
