"""
共用路徑與本機設定（兩台電腦當運算中心時，指向同一個雲端同步資料夾）

  local_settings.json（放在專案資料夾，各台電腦各自一份，不上傳）
  {
    "data_dir": "C:/Users/me/OneDrive/broker_data",   ← 共用資料夾（不填 = 專案下的 data/）
    "role": "backup",                                  ← 雲端（GitHub Actions）主跑時，兩台電腦都設 backup
    "top_n": 100                                       ← 每天計算成交值前幾名（另加 Ultra 會員自選）
  }
  環境變數 BB_DATA_DIR 會覆蓋 data_dir。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = ROOT / "local_settings.json"


def _load() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


SETTINGS = _load()
DATA_DIR = Path(os.environ.get("BB_DATA_DIR") or SETTINGS.get("data_dir") or (ROOT / "data")).expanduser()
ROLE = os.environ.get("BB_ROLE") or SETTINGS.get("role", "primary")   # cloud（GitHub Actions 主跑）／primary／backup
TOP_N = int(SETTINGS.get("top_n", 100))
