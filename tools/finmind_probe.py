"""
FinMind 整天下載探測（Sponsor Pro）：只印「方案額度、每個資料集一天的檔案大小、筆數、欄位、股票數、分點數」，不存任何資料
  python tools/finmind_probe.py [--date 2026-10-30]
token 從環境變數 FINMIND_TOKEN 讀（雲端：GitHub secret；本機：自己設定），程式不會印出 token。
FinMind 的分點資料限訂閱者自己使用：本程式只輸出統計數字，下載的檔案用完即刪。
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

import requests

API = "https://api.finmindtrade.com/api/v4"
DATASETS = ["TaiwanStockTradingDailyReport", "TaiwanStockTradingDailyReportSecIdAgg",
            "TaiwanStockWarrantTradingDailyReport", "TaiwanStockKBar", "TaiwanStockPriceTick"]


def headers() -> dict:
    tok = os.environ.get("FINMIND_TOKEN", "").strip()
    if not tok:
        sys.exit("沒有設定 FINMIND_TOKEN")
    return {"Authorization": f"Bearer {tok}"}


def user_info() -> None:
    try:
        r = requests.get("https://api.web.finmindtrade.com/v2/user_info", headers=headers(), timeout=30)
        j = r.json()
        print(f"[帳號] HTTP {r.status_code}；使用次數 {j.get('user_count')}，每小時上限 {j.get('api_request_limit')}"
              "（Sponsor Pro 應為 20000）")
    except Exception as e:  # noqa: BLE001
        print(f"[帳號] 查詢失敗：{e}")


def fetch_object(dataset: str, date: str) -> tuple[bytes | None, str]:
    """回傳 (檔案內容, 說明)。回應可能是 JSON（內含簽名網址）或直接轉址到檔案"""
    r = requests.get(f"{API}/storage_objects", params={"dataset": dataset, "date": date},
                     headers=headers(), timeout=120, allow_redirects=True)
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}：{r.text[:200]}"
    if "json" in ctype:
        j = r.json()
        url = None
        for k in ("url", "signed_url", "data", "download_url"):
            v = j.get(k)
            if isinstance(v, str) and v.startswith("http"):
                url = v
            elif isinstance(v, dict):
                url = next((x for x in v.values() if isinstance(x, str) and x.startswith("http")), None)
            if url:
                break
        if not url:
            return None, f"HTTP {r.status_code}，JSON 欄位 {sorted(j.keys())}，msg={j.get('msg')}"
        f = requests.get(url, timeout=600)
        return f.content, f"簽名網址下載 HTTP {f.status_code}"
    return r.content, f"直接回傳 HTTP {r.status_code}（{ctype}）"


def describe(blob: bytes) -> str:
    import pandas as pd
    df = pd.read_parquet(io.BytesIO(blob))
    cols = list(df.columns)
    parts = [f"{len(df):,} 筆", f"欄位 {cols}"]
    for c in ("stock_id", "securities_trader_id", "price", "date"):
        if c in df:
            parts.append(f"{c} 不重複 {df[c].nunique():,}")
    mem = df.memory_usage(deep=True).sum() / 1e6
    parts.append(f"記憶體 {mem:,.0f} MB")
    return "；".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-30")
    a = ap.parse_args()
    user_info()
    for ds in DATASETS:
        t0 = time.time()
        try:
            blob, note = fetch_object(ds, a.date)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] 失敗：{e}")
            continue
        if blob is None:
            print(f"[{ds}] {note}")
            continue
        info = describe(blob) if blob[:4] == b"PAR1" else "不是 parquet（不印內容）"
        print(f"[{ds}] {a.date}：檔案 {len(blob) / 1e6:,.1f} MB，{time.time() - t0:.0f} 秒；{note}；{info}")
        del blob
    return 0


if __name__ == "__main__":
    sys.exit(main())
