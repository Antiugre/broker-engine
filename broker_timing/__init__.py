"""
單一分點擇時偵測 v1.1.2（凍結版；docs/timing_spec_v1.1.2.md）
  prep      §3   前處理：雙因子超額報酬（股票 × 交易日預先計算）、股數基準 g_t、ñ_t、樣本資格、反應函數特徵
  zigzag    §6.2 金額門檻的 zigzag 波段計數
  reaction  §4.2 Ridge 反應函數與殘差（含 E+ 版本）
  hac       §4.3、§4.4 堆疊式 Newey–West、G_eff、Cheverud–Nyholt 多期間合併
  ei        §4.5 EI 分解與移動區塊 bootstrap
  classify  §5、§7 分類規則與 BH
  report    §8 描述欄位
  run       逐檔協調、BH 族群、輸出文件
  validate  §9 安慰劑、合成注入、案例報告
參數：broker_flow_v3/config/params.yaml 的 timing: 區塊
"""
from __future__ import annotations

VERSION = "1.1.2"
