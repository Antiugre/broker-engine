"""v3 參數：只從 config/params.yaml 讀取"""
from __future__ import annotations

from pathlib import Path

import yaml

PARAMS_FILE = Path(__file__).resolve().parent / "config" / "params.yaml"


class _Params(dict):
    def __getattr__(self, k):
        v = self[k]
        return _Params(v) if isinstance(v, dict) else v


def load(path: Path = PARAMS_FILE) -> _Params:
    return _Params(yaml.safe_load(path.read_text(encoding="utf-8")))


P = load()
