"""
台股分點主力集團偵測 v3（《主力集團偵測 v3 修正規格》2026-11-04）。v2（broker_flow/）保留不動，兩版並存到驗收完成。

各層：split（§2 切分）→ data_layer（§3.1）→ accounting（§3.2～3.4）→ features（§4）→ dim_a（§5）、dim_b（§6）、dim_c（§7）
     → fusion（§8 雙通道、逐窗口 BH、雙連通分量）→ group（§9）→ run
"""
from .params import P  # noqa: F401
