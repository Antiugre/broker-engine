"""
§3 資料與前處理
  ・金額一律用原始價 × 原始股數，報酬一律用還原價（§3.2）
  ・股數基準因子 g_t = Π_{τ > t} k_τ；n^cb = g·n；ñ = n^cb / median{|n^cb| : 活躍日}（§3.3）
  ・雙因子超額報酬（§3.4）：r^ex = r^stk − β^m r^mkt − β^ind (r^ind − r^mkt)，β 用 [t−120, t−1] 日報酬 OLS；
    「股票 × 交易日」層級預先計算一次（compute），分點迴圈只讀取
  ・樣本資格：t 之前已有 ≥ 120 個有效交易日（C > 0）且當日 C_t > 0（§3.1）
  ・反應函數特徵（§4.2）也在股票層級預先算好（與分點無關）
大盤指數：TaiwanStockTotalReturnIndex（上市 TAIEX、上櫃 TPEx），經 broker_flow_v3.auxdata.stock 的快取取得；
  呼叫端可以注入 market_fn(sid) → pd.Series(index=date str, value=指數)（測試用）
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .params import TP, horizons

log = logging.getLogger("broker_scraper")

FEATURES = ["min_r", "r_lag1", "sum_r5", "sum_r20", "pos20", "log_vol", "min_rmkt"]
FEATURE_EPLUS = "max_r"                 # §4.5 E+：再加入 max(r_t, 0)


# ───────────────────────── 股數基準（§3.3） ─────────────────────────
def share_factor(k_kind: str, value: float) -> float:
    """v3 §3.2：配股 1 + r；分割、面額變更、減資 P_before / P_ref；現金股利 1"""
    if k_kind == "stock":
        return 1.0 + float(value)
    if k_kind == "ratio":
        return float(value)
    return 1.0


def share_basis(dates: np.ndarray, events: pd.DataFrame | None) -> tuple[np.ndarray, np.ndarray]:
    """g_t = Π_{τ > t} k_τ（事件日 τ 為生效日、開盤前調整，所以 t 日在 τ 之前才乘 k_τ）；最新一日 g = 1。
    只用最後一日（含）之前生效的事件。回傳 (g, approx)；approx[t] = 該日是 ca_approx 事件生效日"""
    dates = np.asarray(dates).astype(str)
    T = len(dates)
    g = np.ones(T)
    approx = np.zeros(T, bool)
    if events is None or not len(events) or T == 0:
        return g, approx
    last = dates[-1]
    for _, e in events.iterrows():
        d = str(e["date"])[:10]
        if d > last:
            continue
        k = share_factor(str(e["kind"]), float(e["value"]))
        pos = int(np.searchsorted(dates, d, side="left"))     # 第一個 ≥ 生效日的交易日
        if k != 1.0 and np.isfinite(k) and k > 0:
            g[:pos] *= k
        if bool(e.get("approx", False)) and pos < T:
            approx[pos] = True
    return g, approx


def standardize(ncb: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float]:
    """ñ_t = n^cb_t / median{|n^cb_τ| : τ 活躍日}；沒有活躍日 → 全 0"""
    a = np.abs(ncb[active])
    if not len(a):
        return np.zeros_like(ncb, dtype=float), np.nan
    med = float(np.median(a))
    if med <= 0:
        return np.zeros_like(ncb, dtype=float), med
    return np.where(active, ncb / med, 0.0), med


# ───────────────────────── 股票 × 交易日預先計算（§3.4、§4.2） ─────────────────────────
@dataclass
class Prep:
    dates: np.ndarray            # 交易日（price_load_start 起）
    valid: np.ndarray            # C_t > 0
    eligible: np.ndarray         # §3.1：t 之前已有 ≥ 120 個有效交易日，且 t 有效
    close: np.ndarray            # 原始收盤（無效日 NaN）
    adj: np.ndarray              # 還原收盤（向前補值）
    volume: np.ndarray
    r: np.ndarray                # 日報酬（還原價）
    rm: np.ndarray               # 大盤日報酬
    beta_m: np.ndarray
    beta_ind: np.ndarray
    rex: dict                    # h → r^ex_{t→t+h}
    feats: np.ndarray            # (T, 8)：FEATURES + max_r
    industry_factor: bool
    n_peers: int
    inputs: dict = field(default_factory=dict, repr=False)   # 原始輸入（§9.2 注入後重算用）
    ex1: np.ndarray | None = None  # 第 2 輪 §3.4：單日超額對數報酬 e_s（rex[h][t] = Σ_{s=t+1}^{t+h} e_s）
    lind: np.ndarray | None = None  # 產業單日對數報酬 ln(1 + 等權單日報酬)（無產業項為 None）

    def align(self, dates: np.ndarray) -> np.ndarray:
        """面板日期 → 本表的索引（找不到為 −1）"""
        pos = {d: i for i, d in enumerate(self.dates)}
        return np.array([pos.get(str(d), -1) for d in dates], int)


def _ffill(a: np.ndarray) -> np.ndarray:
    return pd.Series(a, dtype=float).ffill().to_numpy(float)


def _fwd(level: np.ndarray, h: int) -> np.ndarray:
    # SPEC-GAP-timing：t+h 日停牌時用向前補值的最後有效價（level 已向前補值）
    T = len(level)
    out = np.full(T, np.nan)
    if T > h:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:T - h] = level[h:] / level[:T - h] - 1
    return out


def _daily(level: np.ndarray) -> np.ndarray:
    out = np.full(len(level), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = level[1:] / level[:-1] - 1
    return out


def rolling_betas(y: np.ndarray, x1: np.ndarray, x2: np.ndarray | None, lookback: int, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """β_t 用 [t−lookback, t−1] 的日資料 OLS：y = α + β1 x1 (+ β2 x2)。只用 t 以前的資料（不含 t）。
    以累積和向量化（不在分點迴圈內做）；有效筆數 < min_obs 或設計矩陣退化 → NaN"""
    T = len(y)
    use2 = x2 is not None
    m = np.isfinite(y) & np.isfinite(x1) & (np.isfinite(x2) if use2 else True)
    Y = np.where(m, y, 0.0); X1 = np.where(m, x1, 0.0); X2 = np.where(m, x2, 0.0) if use2 else np.zeros(T)
    M = m.astype(float)

    def win(a):
        c = np.r_[0.0, np.cumsum(a)]
        t = np.arange(T)
        lo = np.maximum(0, t - lookback)
        return c[t] - c[lo]                      # Σ_{τ=t−L}^{t−1}
    n = win(M)
    s1, s2, sy = win(X1), win(X2), win(Y)
    s11, s22, s12 = win(X1 * X1), win(X2 * X2), win(X1 * X2)
    s1y, s2y = win(X1 * Y), win(X2 * Y)
    with np.errstate(invalid="ignore", divide="ignore"):
        c11 = s11 - s1 * s1 / n
        c22 = s22 - s2 * s2 / n
        c12 = s12 - s1 * s2 / n
        c1y = s1y - s1 * sy / n
        c2y = s2y - s2 * sy / n
        if use2:
            det = c11 * c22 - c12 * c12
            ok = (n >= min_obs) & (np.abs(det) > 1e-14 * np.maximum(c11 * c22, 1e-300))
            b1 = np.where(ok, (c22 * c1y - c12 * c2y) / det, np.nan)
            b2 = np.where(ok, (c11 * c2y - c12 * c1y) / det, np.nan)
        else:
            ok = (n >= min_obs) & (c11 > 1e-300)
            b1 = np.where(ok, c1y / c11, np.nan)
            b2 = np.zeros(T)
    return b1, b2


def compute(dates, close, adj, volume, mkt, peer_adj: np.ndarray | None = None, n_peers: int | None = None) -> Prep:
    """股票 × 交易日一次算好（§3.4 預先計算）。
    dates：交易日（price_load_start 起）；close：原始收盤（≤ 0 或 NaN 為無效）；adj：還原收盤；volume：股；
    mkt：大盤報酬指數（同一日期軸，可有 NaN）；peer_adj：(同產業其他股票數, T) 的還原收盤（排除 j 本身），可為 None"""
    dates = np.asarray(dates).astype(str)
    close = np.asarray(close, float)
    adj_raw = np.asarray(adj, float)
    vol = np.nan_to_num(np.asarray(volume, float), nan=0.0)
    mkt = np.asarray(mkt, float)
    T = len(dates)
    valid = np.isfinite(close) & (close > 0) & np.isfinite(adj_raw) & (adj_raw > 0)
    adj_f = _ffill(np.where(valid, adj_raw, np.nan))
    mk_f = _ffill(np.where(np.isfinite(mkt) & (mkt > 0), mkt, np.nan))
    r = _daily(adj_f)
    rm = _daily(mk_f)
    n_peers = int(0 if peer_adj is None else peer_adj.shape[0]) if n_peers is None else int(n_peers)
    ind = bool(peer_adj is not None and n_peers >= int(TP.industry_min_peers))
    if ind:
        pa = np.where(np.isfinite(peer_adj) & (peer_adj > 0), peer_adj, np.nan)
        pf = pd.DataFrame(pa.T).ffill().to_numpy(float).T       # 每檔各自向前補值（上市前為 NaN）
        with np.errstate(invalid="ignore", divide="ignore"):
            pdr = np.full_like(pf, np.nan)
            pdr[:, 1:] = pf[:, 1:] / pf[:, :-1] - 1
            pdr[:, 1:][~np.isfinite(pa[:, 1:])] = np.nan          # 該檔當日無價 → 不列入當日平均
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rind = np.nanmean(pdr, axis=0)       # SPEC-GAP-timing：當日有價的同產業股票等權平均；產業檔數以宇宙內成員數（不逐日）判斷 ≥ 5
    else:
        rind = None
    # β：只用有效日（C_t > 0）的日報酬（SPEC-GAP-timing：停牌後復牌日的報酬跨越多日，照常使用）
    y = np.where(valid, r, np.nan)
    bm, bi = rolling_betas(y, rm, (rind - rm) if ind else None, int(TP.beta_lookback), int(TP.beta_min_obs))
    if not ind:
        bi = np.zeros(T)
    # 第 2 輪（CHANGELOG 1.1.2+r2）§3.4：單日超額對數報酬 e_s = ln(C_s/C_{s−1}) − β^m_{s−1} ln(M_s/M_{s−1}) − β^ind_{s−1} (ind_s − mkt_s)
    #   β_{s−1} 用 [s−121, s−2]（rolling_betas 不含當日），s 日之前已知；停牌日：價格向前補值 → 單日對數報酬 0、β 沿用最後有效值
    #   r^ex_{t→t+h} := Σ_{s=t+1}^{t+h} e_s（重疊可搬到解釋變數上，§4.4 反向迴歸）
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.r_[np.nan, np.log(adj_f[1:] / adj_f[:-1])]
        lm = np.r_[np.nan, np.log(mk_f[1:] / mk_f[:-1])]
        lind = np.log1p(rind) if ind else None           # SPEC-GAP-timing：產業單日報酬先等權平均再取對數
    bm1 = np.r_[np.nan, _ffill(bm)[:-1]]
    ex1 = lr - bm1 * lm
    if ind:
        bi1 = np.r_[np.nan, _ffill(bi)[:-1]]
        ex1 = ex1 - bi1 * (lind - lm)
    c = np.r_[0.0, np.cumsum(np.nan_to_num(ex1))]
    bad = np.r_[0, np.cumsum(~np.isfinite(ex1))]
    rex = {}
    for h in horizons():
        e = np.full(T, np.nan)
        if T > h:
            k = np.arange(T - h)
            s = c[k + 1 + h] - c[k + 1]                      # Σ_{s=t+1}^{t+h} e_s
            nb = bad[k + 1 + h] - bad[k + 1]
            e[:T - h] = np.where(nb == 0, s, np.nan)
        e[~valid] = np.nan
        rex[h] = e
    cnt_before = np.cumsum(valid) - valid
    eligible = valid & (cnt_before >= int(TP.min_history_days))
    feats = features(adj_f, r, rm, vol)
    return Prep(dates, valid, eligible, np.where(valid, close, np.nan), adj_f, vol, r, rm, bm, bi, rex, feats, ind, n_peers,
                {"close": close, "adj": adj_raw, "volume": vol, "mkt": mkt, "peer_adj": peer_adj, "n_peers": n_peers}, ex1=ex1, lind=lind)


def features(adj_f: np.ndarray, r: np.ndarray, rm: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """§4.2 的 x_t（未標準化；標準化在活躍日上做，見 reaction.py），最後一欄是 E+ 的 max(r_t, 0)"""
    s = pd.Series(r)
    r0 = s.fillna(0.0)                             # SPEC-GAP-timing：停牌日（向前補值）報酬視為 0 加總
    lag1 = s.shift(1).to_numpy(float)
    sum5 = r0.rolling(5, min_periods=5).sum().shift(1).to_numpy(float)
    sum20 = r0.rolling(20, min_periods=20).sum().shift(1).to_numpy(float)
    a = pd.Series(adj_f)
    L20 = a.rolling(int(TP.pos20_window), min_periods=int(TP.pos20_window)).min().shift(1).to_numpy(float)   # [t−20, t−1]
    H20 = a.rolling(int(TP.pos20_window), min_periods=int(TP.pos20_window)).max().shift(1).to_numpy(float)
    prev = a.shift(1).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos20 = np.where(H20 > L20, (prev - L20) / (H20 - L20), np.where(np.isfinite(H20), 0.5, np.nan))
    w0, w1 = (int(x) for x in TP.volume_median_window)            # [t−61, t−2]
    v = pd.Series(vol)
    vmed = v.rolling(w0 - w1 + 1, min_periods=w0 - w1 + 1).median().shift(w1).to_numpy(float)
    vprev = v.shift(1).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        lv = np.log((vprev + 1) / (vmed + 1))
    return np.c_[np.minimum(r, 0), lag1, sum5, sum20, pos20, lv, np.minimum(rm, 0), np.maximum(r, 0)]


# ───────────────────────── 資料來源（正式執行用；測試一律注入） ─────────────────────────
def stock_info() -> pd.DataFrame:
    from broker_flow_v3 import auxdata as aux
    info = aux.static("TaiwanStockInfo")
    if info is None or not len(info):
        return pd.DataFrame(columns=["stock_id", "industry_category", "type"])
    return info.drop_duplicates("stock_id", keep="last")


def market_level(sid: str) -> pd.Series:
    """§3.1 大盤報酬指數：上市 TAIEX、上櫃 TPEx（TaiwanStockTotalReturnIndex，經 auxdata 的快取）。
    SPEC-GAP-timing：市場別取 TaiwanStockInfo.type（tpex → TPEx，其餘 → TAIEX）；拿不到指數 → 空序列（呼叫端跳過該股）"""
    from broker_flow_v3 import auxdata as aux
    info = stock_info()
    typ = ""
    if len(info) and "type" in info:
        m = info[info["stock_id"].astype(str) == sid]
        typ = str(m["type"].iloc[-1]).lower() if len(m) else ""
    did = "TPEx" if typ == "tpex" else "TAIEX"
    df = aux.stock("TaiwanStockTotalReturnIndex", did, "2019-01-01")
    if df is None or not len(df) or "price" not in df:
        return pd.Series(dtype=float)
    return pd.Series(aux.num(df, "price").to_numpy(float), index=df["date"].astype(str)).groupby(level=0).last()


class Industry:
    """產業成員（§3.4：TaiwanStockInfo.industry_category；成員限股票宇宙）與同產業其他股票的還原收盤"""

    def __init__(self, universe: list[str], info: pd.DataFrame | None = None, adj_fn: Callable[[str], pd.Series] | None = None):
        self.universe = sorted(set(map(str, universe)))
        info = stock_info() if info is None else info
        self.ind = {}
        if len(info) and "industry_category" in info:
            self.ind = info.set_index(info["stock_id"].astype(str))["industry_category"].astype(str).to_dict()
        self.adj_fn = adj_fn or _adj_series
        self._cache: dict[str, pd.Series] = {}

    def category(self, sid: str) -> str | None:
        return self.ind.get(sid)

    def peers(self, sid: str) -> list[str]:
        c = self.ind.get(sid)
        if c is None:
            return []
        return [s for s in self.universe if s != sid and self.ind.get(s) == c]

    def adj(self, sid: str) -> pd.Series:
        if sid not in self._cache:
            try:
                self._cache[sid] = self.adj_fn(sid)
            except Exception as ex:  # noqa: BLE001
                log.warning("擇時 產業成員 %s 價格讀取失敗：%s", sid, ex)
                self._cache[sid] = pd.Series(dtype=float)
        return self._cache[sid]

    def peer_matrix(self, sid: str, dates: np.ndarray) -> tuple[np.ndarray | None, int]:
        ps = self.peers(sid)
        if len(ps) < int(TP.industry_min_peers):
            return None, len(ps)
        mat = np.vstack([self.adj(p).reindex(dates).to_numpy(float) for p in ps])
        return mat, len(ps)


def _adj_series(sid: str) -> pd.Series:
    from broker_flow_v3.data_layer import prices
    px = prices(sid)
    return pd.Series(px["adj_close"].where(px["close"] > 0).to_numpy(float), index=px["date"].astype(str))


def calendar(end: str, px_dates) -> np.ndarray:
    from broker_flow_v3 import auxdata as aux
    start = str(TP.price_load_start)
    cal = aux.trading_calendar()
    src = cal if cal else sorted(map(str, px_dates))          # SPEC-GAP（同 v3）：沒有官方日曆 → 個股有價格的日子
    return np.array([d for d in src if start <= d <= end])


def stock_prep(sid: str, end: str, industry: Industry | None = None, market_fn: Callable[[str], pd.Series] | None = None,
               cache_dir=None) -> Prep:
    """正式執行：載入價格（2020-11-01 起）、大盤、同產業 → compute；結果依 (股票, 最後一日, 版本) 快取"""
    key = f"{sid}|{end}|{TP.timing_version}|{len(industry.peers(sid)) if industry else 0}"
    f = None
    if cache_dir is not None:
        f = cache_dir / f"prep_{sid}.pkl"
        if f.exists():
            try:
                k, pr = pickle.loads(f.read_bytes())
                if k == key:
                    return pr
            except Exception:  # noqa: BLE001
                pass
    from broker_flow_v3.data_layer import prices
    px = prices(sid)
    dates = calendar(end, px["date"])
    pxi = px.set_index(px["date"].astype(str)).reindex(dates)
    mk = (market_fn or market_level)(sid)
    if mk is None or not len(mk):
        raise RuntimeError(f"{sid}：沒有大盤報酬指數（TaiwanStockTotalReturnIndex），無法計算超額報酬")
    mkt = mk.reindex(dates).to_numpy(float)
    peer, npeer = (industry.peer_matrix(sid, dates) if industry else (None, 0))
    pr = compute(dates, pxi["close"].to_numpy(float), pxi["adj_close"].to_numpy(float), pxi["volume"].to_numpy(float), mkt, peer, npeer)
    if f is not None:
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps((key, pr)))
        except Exception:  # noqa: BLE001
            pass
    return pr
