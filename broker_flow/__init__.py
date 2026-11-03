"""
台股分點主力集團偵測（規格書：台股分點主力集團偵測系統：實作規格書，2026-11-03）

本版是「可行性測試版」，只在本專案現有資料上能做的部分照規格實作；
做不到的部分依第 0 節第 5 條標 SPEC-GAP（彙整在 SPEC_GAPS.md），不自行發明替代方法。

各層（每層輸出寫成 Parquet，下一層只讀檔案；第 0 節第 4 條）
  data_layer   longhist → branch_daily、daily_prices（股）、交易日曆、hold-out 防護
  accounting   當沖拆分、移動平均成本、已實現損益、uncovered_sell／left_censored
  features     行為分類遮罩、scale_i、z_net、活躍日
  dim_a        logistic 活躍機率 → Poisson-binomial p_act；活躍日殘差 Spearman p_mag
  fusion       CCT、全局 BH-FDR、連通分量集團、集團成本 C_G
  validation   樸素基準線、7.4 校正檢查、9.3 合成注入（含 C2ST）
  run          一檔股票跑完全部層 → 網頁用的摘要 JSON
"""
from .params import P  # noqa: F401
