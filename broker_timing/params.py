"""擇時偵測參數：broker_flow_v3/config/params.yaml 的 timing: 區塊（§10）"""
from __future__ import annotations

from broker_flow_v3.params import P

TP = P.timing


def horizons() -> list[int]:
    return [int(h) for h in TP.horizons]
