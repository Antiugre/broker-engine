"""
第一次建檔：把今天要算的股票（Ultra 會員自選 → 成交值前 N 名）的約 5 年分點明細與價格抓進本機快取
  python backfill.py                  # 跑到完成（可能十幾小時）
  python backfill.py --max-hours 6    # 每次最多 6 小時，中斷後再執行會從快取接著補
完成後每天的 pipeline 只需增量更新（約 30 分鐘～1 小時）。建檔期間不上傳 Firestore。
"""
import sys

import pipeline

if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(pipeline.main(["--no-upload", *args]))
