"""
smallcap.py — 中小型股・本土主力吃貨（Expectancy「中小型吃貨」，Ultra 限定）
==========================================================================

  對象    追蹤名單（Ultra 自選＋成交值前 N 名）中，實收資本額 ≤ CAP_MAX 的中小型股
  分點    本土券商分點（排除外資、自營、隔日沖型、散戶混合型大分點；判斷只用到當天為止的資料）
  吃貨    近 WIN 個交易日，分點庫存增加 ≥ 發行股數 MIN_ADD_PCT%（且 ≥ MIN_ADD_LOTS 張），
          目前庫存 ≥ 發行股數 MIN_HOLD_PCT%，期間淨買天數 ≥ 淨賣天數的 STEADY 倍（穩定吃，不是一兩天大買）
  個股    吃貨分點的增加量合計 ≥ 發行股數 SCORE_MIN%，且股價近 WIN 日漲幅 ≤ MAX_RUN（還沒被拉開）
  庫存    與 mainforce 相同：累積淨買超、以到當天為止的累計最低點當 0（不偷看未來）

  回測    過去約 2 年，每 5 個交易日檢查一次；隔天以成交均價進場，持有 20／40／60 個交易日；
          超額報酬 = 訊號股報酬 − 同一天全部中小型股（追蹤名單內）的平均報酬
          限制：追蹤名單是「現在」的成交值前 N 名＋自選（倖存者偏差，結果會偏樂觀）

  輸出    data/smallcap/<日>.json → Firestore smallcap/latest（Ultra 限定）

  python smallcap.py 2026-10-30            # 今天的訊號＋回測摘要
  python smallcap.py 2026-10-30 --report   # 另外印出回測報告
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

import numpy as np
import pandas as pd

import quant
from mainforce import _holdings, _institutional, FLIP_SHARE, FLIP_SELL, FLIP_RATIO, MIXED_TURNOVER

log = logging.getLogger("broker_scraper")

OUT_DIR = quant.DATA_DIR / "smallcap"
CAP_FILE = quant.DATA_DIR / "capital.json"
CAP_MAX = 100e8         # 實收資本額 ≤ 100 億元 → 中小型
CAP_SMALL = 20e8        # ≤ 20 億元標示「小型」（股本版本）
SMALL_MCAP = 100e8      # 市值 ≤ 100 億元標示「小型」
WIN = 20                # 吃貨觀察期（交易日）
MIN_ADD_PCT = 0.25      # 單一分點 WIN 日庫存增加 ≥ 發行股數 0.25%
MIN_ADD_LOTS = 100      # …且 ≥ 100 張
MIN_HOLD_PCT = 0.5      # 目前庫存 ≥ 發行股數 0.5%
STEADY = 1.5            # 淨買天數 ≥ 淨賣天數 × 1.5
SCORE_MIN = 0.5         # 個股：吃貨分點增加量合計 ≥ 發行股數 0.5%
MAX_RUN = 0.15          # 個股：近 WIN 日漲幅 ≤ 15%
BT_DAYS = 500           # 回測期間（交易日，約 2 年）
BT_STEP = 5             # 每 5 個交易日檢查一次
HORIZONS = (20, 40, 60)
NEW_GAP = 20            # 同一檔 20 個交易日內再出現不算新訊號
PARAMS = {k: v for k, v in dict(cap_max=CAP_MAX, cap_small=CAP_SMALL, win=WIN, min_add_pct=MIN_ADD_PCT, min_add_lots=MIN_ADD_LOTS,
                                min_hold_pct=MIN_HOLD_PCT, steady=STEADY, score_min=SCORE_MIN, max_run=MAX_RUN,
                                bt_days=BT_DAYS, bt_step=BT_STEP, horizons=list(HORIZONS)).items()}


# ──────────────────────────── 股本 ────────────────────────────
def capital_directory(max_age_days: int = 7) -> dict[str, dict]:
    """{代號: {capital: 實收資本額（元）, shares: 已發行普通股數}}；上市＋上櫃公開資料，快取 7 天"""
    from flowlab.sources import COMPANY_URLS
    try:
        cached = json.loads(CAP_FILE.read_text(encoding="utf-8"))
        if time.time() - cached.get("t", 0) < max_age_days * 86400 and cached.get("data"):
            return cached["data"]
    except Exception:  # noqa: BLE001
        cached = None
    import requests
    data: dict[str, dict] = {}
    for url in COMPANY_URLS:
        try:
            for x in requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json():
                code = x.get("公司代號") or x.get("SecuritiesCompanyCode") or x.get("CompanyCode")
                cap = sh = None
                for k, v in x.items():
                    try:
                        n = float(str(v).replace(",", "").strip())
                    except ValueError:
                        continue
                    if ("實收資本" in k or "Paidin" in k or "Capital" in k) and cap is None:
                        cap = n
                    elif ("已發行普通股" in k or "IssueShares" in k or "Shares" in k) and sh is None:
                        sh = n
                if code and (cap or sh):
                    data[str(code).strip()] = {"capital": cap or (sh * 10 if sh else None), "shares": sh or (cap / 10 if cap else None)}
        except Exception as ex:  # noqa: BLE001
            log.info("股本資料 %s 失敗：%s", url, ex)
    if data:
        CAP_FILE.parent.mkdir(parents=True, exist_ok=True)
        CAP_FILE.write_text(json.dumps({"t": time.time(), "data": data}, ensure_ascii=False), encoding="utf-8")
        return data
    return (cached or {}).get("data", {})


# ──────────────────────────── 版本（回測並列比較，LIVE 用在網站） ────────────────────────────
#   size: ("capital", 元) 用實收資本額；("mcap", 元) 用當天市值（股數 × 收盤價）
#   win: 吃貨觀察期；no_hq: 排除總公司／網路下單為主的大戶頭（名稱沒有「-」）；mixed: 散戶混合型門檻（近 250 日成交 ÷ 庫存）
#   geo_only: 只算地緣分點；flat: 股價盤整（近 win 日漲跌在 [-10%, +5%]，且收盤距吃貨成本 ±8%）
VARIANTS = {
    "base":  {"name": "原始（股本 ≤ 100 億）", "size": ("capital", 100e8), "win": 20, "no_hq": False, "mixed": 12, "geo_only": False, "flat": False},
    "nohq":  {"name": "排除總公司／網路大戶頭", "size": ("capital", 100e8), "win": 20, "no_hq": True, "mixed": 6, "geo_only": False, "flat": False},
    "mcap":  {"name": "＋市值 ≤ 300 億", "size": ("mcap", 300e8), "win": 20, "no_hq": True, "mixed": 6, "geo_only": False, "flat": False},
    "geo":   {"name": "＋只看地緣分點", "size": ("mcap", 300e8), "win": 20, "no_hq": True, "mixed": 6, "geo_only": True, "flat": False},
    "w40":   {"name": "＋吃貨期間 40 日", "size": ("mcap", 300e8), "win": 40, "no_hq": True, "mixed": 6, "geo_only": False, "flat": False},
    "flat":  {"name": "＋股價在成本附近盤整", "size": ("mcap", 300e8), "win": 20, "no_hq": True, "mixed": 6, "geo_only": False, "flat": True},
}
LIVE = "mcap"            # 網站顯示的版本（依回測結果決定）
WINS_ALL = (20, 40)


def is_hq(name: str) -> bool:
    """總公司或網路下單為主的大戶頭：富邦名稱沒有「-分點」（凱基、國泰證券、中國信託、群益金鼎…）"""
    return "-" not in (name or "")


# ──────────────────────────── 一檔股票：分點 × 日期 矩陣 ────────────────────────────
def _roll(x: np.ndarray, w: int) -> np.ndarray:
    """尾端 w 天加總（含當天）；x 可為 1 維或 (分點, 日期) 2 維"""
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1, dtype=float)], axis=-1)
    i = np.arange(1, x.shape[-1] + 1)
    return c[..., i] - c[..., np.maximum(0, i - w)]


def branch_frame(net: np.ndarray, gross: np.ndarray, vwap: np.ndarray, vol: np.ndarray) -> dict:
    """單一分點的逐日陣列：hold, cost, since, add{w}, pos{w}, neg{w}, flipper, turn（近 250 日成交 ÷ 庫存）；只用到當天為止的資料"""
    hold, cost, since = _holdings(net, vwap)
    n = len(net)
    out = {"hold": hold, "cost": cost, "since": since}
    for w in WINS_ALL:
        prev = np.concatenate([np.zeros(w), hold[:-w]]) if n > w else np.zeros(n)
        out[f"add{w}"] = hold - prev
        out[f"pos{w}"], out[f"neg{w}"] = _roll((net > 0).astype(float), w), _roll((net < 0).astype(float), w)
    # 隔日沖（近 250 日）：大買日（淨買 ≥ 當日量 10%）隔天賣掉 ≥ 70% 的比例過半；第 i 天只知道 i−1 天以前的大買結果
    big = (vol > 0) & (net >= FLIP_SHARE * vol)
    flip = np.zeros(n, dtype=bool)
    flip[:-1] = big[:-1] & (-net[1:] >= FLIP_SELL * net[:-1])
    nb = _roll(np.concatenate([[False], big[:-1]]).astype(float), 250)
    nf = _roll(np.concatenate([[False], flip[:-1]]).astype(float), 250)
    out["flipper"] = (nb >= 3) & (nf > FLIP_RATIO * np.maximum(nb, 1))
    out["turn"] = np.where(hold > 0, _roll(gross, 250) / np.maximum(hold, 1e-9), np.inf)
    # 舊欄位（相容）
    out["add"], out["pos"], out["neg"] = out[f"add{WIN}"], out[f"pos{WIN}"], out[f"neg{WIN}"]
    out["ok"] = ~(out["flipper"] | ((hold > 0) & (out["turn"] > MIXED_TURNOVER)))
    return out


def load_stock(sid: str, day: str, geo_of=None) -> dict | None:
    """讀 longhist → {idx, vwap, close, vol, keys, names, M: {欄位: (分點, 日期) 矩陣}, hq, geo}"""
    d = quant.LONG_DIR / sid
    if not d.exists():
        return None
    prices = quant._read(d / "_prices.json") or {}
    px = pd.DataFrame.from_dict(prices.get("data") or {}, orient="index")
    if px.empty or len(px) < 120:
        return None
    px.index = pd.to_datetime(px.index)
    px = px[px.index <= pd.Timestamp(day)].sort_index()
    for c in ("vwap", "close", "vol"):
        if c not in px:
            px[c] = np.nan
    px["vwap"] = px["vwap"].fillna(px["close"])
    idx = px.index
    vwap, close, vol = px["vwap"].to_numpy(float), px["close"].to_numpy(float), px["vol"].to_numpy(float)
    keys, names, frames = [], [], []
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        cache = quant._read(f) or {}
        name = cache.get("broker") or ""
        if not cache.get("rows") or _institutional(name):
            continue
        df = pd.DataFrame(cache["rows"])
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.groupby("date")[["buy", "sell"]].sum().reindex(idx).fillna(0.0)
        net = (df["buy"] - df["sell"]).to_numpy(float)
        if not np.any(net > 0):
            continue
        gross = (df["buy"] + df["sell"]).to_numpy(float)
        keys.append(f.stem); names.append(name); frames.append(branch_frame(net, gross, vwap, vol))
    if not frames:
        M = {}
    else:
        M = {k: np.vstack([F[k] for F in frames]) for k in frames[0] if k not in ("add", "pos", "neg", "ok")}
    return {"idx": idx, "vwap": vwap, "close": close, "vol": vol, "keys": keys, "names": names, "M": M,
            "hq": np.array([is_hq(x) for x in names], dtype=bool),
            "geo": np.array([geo_of(sid, x) if geo_of else 1.0 for x in names], dtype=float)}


def size_ok(S: dict, t: int, cap: dict, cfg: dict) -> bool:
    kind, lim = cfg["size"]
    if kind == "capital":
        return bool(cap.get("capital")) and cap["capital"] <= lim
    sh = cap.get("shares") or (cap.get("capital") or 0) / 10
    return sh > 0 and sh * S["close"][t] <= lim


def eval_at(S: dict, t: int, cap: dict, cfg: dict) -> dict | None:
    """第 t 天（索引）、某個版本的個股訊號；不符合回傳 None"""
    w = cfg["win"]
    M = S["M"]
    lots = ((cap.get("shares") or (cap.get("capital") or 0) / 10)) / 1000
    if t < w or lots <= 0 or not M or not size_ok(S, t, cap, cfg):
        return None
    add, hold, pos, neg = M[f"add{w}"][:, t], M["hold"][:, t], M[f"pos{w}"][:, t], M[f"neg{w}"][:, t]
    min_add = max(MIN_ADD_LOTS * w / WIN, lots * MIN_ADD_PCT / 100 * w / WIN)
    ok = (~M["flipper"][:, t]) & (M["turn"][:, t] <= cfg["mixed"]) & (add >= min_add) & (hold >= lots * MIN_HOLD_PCT / 100) \
        & (pos >= STEADY * np.maximum(neg, 1))
    if cfg["no_hq"]:
        ok &= ~S["hq"]
    if cfg["geo_only"]:
        ok &= S["geo"] > 1
    sel = np.where(ok)[0]
    if not len(sel):
        return None
    score = float(add[sel].sum() / lots * 100) * WIN / w
    c0, c1 = S["close"][t - w], S["close"][t]
    run = c1 / c0 - 1 if c0 > 0 else None
    if score < SCORE_MIN or run is None or run > MAX_RUN:
        return None
    cost = M["cost"][sel, t]
    wts = hold[sel]
    good = np.isfinite(cost)
    wcost = float((cost[good] * wts[good]).sum() / wts[good].sum()) if good.any() and wts[good].sum() > 0 else None
    if cfg["flat"]:
        if not (-0.10 <= run <= 0.05) or wcost is None or abs(c1 / wcost - 1) > 0.08:
            return None
    return {"t": t, "score": score, "run": run, "sel": sel, "cost": wcost, "lots": lots}


def forward(S: dict, t: int, h: int) -> float | None:
    """隔天成交均價進場，持有 h 個交易日後以成交均價出場"""
    a, b = t + 1, t + 1 + h
    if b >= len(S["vwap"]):
        return None
    p0, p1 = S["vwap"][a], S["vwap"][b]
    return p1 / p0 - 1 if p0 > 0 and p1 > 0 else None


# ──────────────────────────── 全部 ────────────────────────────
def _names(day: str, stocks: list[str]) -> dict[str, str]:
    """股票名稱：從最近幾天的個股檔找（當天可能還在下載中）"""
    days = sorted((p.name for p in quant.DATA_DIR.glob("????-??-??") if p.is_dir() and p.name <= day), reverse=True)[:5]
    out = {}
    for sid in stocks:
        out[sid] = ""
        for d in days:
            try:
                out[sid] = json.loads((quant.DATA_DIR / d / f"{sid}.json").read_text(encoding="utf-8")).get("stock_name") or ""
            except Exception:  # noqa: BLE001
                continue
            if out[sid]:
                break
    return out


def _branch_ids() -> dict[str, dict]:
    """history key → {broker_id, b_raw, branch_id}（富邦分點清單，網頁點選後要用來查明細）"""
    try:
        from sweep import LIST_CACHE, _key
        lst = json.loads(LIST_CACHE.read_text(encoding="utf-8")).get("branches") or []
        return {_key(x["b_raw"]): {k: x.get(k) for k in ("broker_id", "b_raw", "branch_id")} for x in lst}
    except Exception:  # noqa: BLE001
        return {}


def _stats(evs: list[dict], bmean: dict) -> dict:
    out = {"n": len(evs)}
    for h in HORIZONS:
        xs = [(e["ret"][h], e["ret"][h] - bmean[h][e["d"]]) for e in evs if e["ret"][h] is not None and e["d"] in bmean[h]]
        if not xs:
            out[f"h{h}"] = None
            continue
        r, x = np.array([a for a, _ in xs]), np.array([b for _, b in xs])
        sd = x.std(ddof=1) if len(x) > 2 else 0
        out[f"h{h}"] = {"n": len(xs), "ret": round(float(r.mean()), 4), "excess": round(float(x.mean()), 4),
                       "median_excess": round(float(np.median(x)), 4), "win": round(float((x > 0).mean()), 3),
                       "t": round(float(x.mean() / (sd / np.sqrt(len(x)))), 2) if sd > 0 else None}
    return out


def _geo_fn():
    """(代號, 分點名稱) → 地緣權重（>1 = 同郵遞區號／區／縣市）"""
    try:
        from broker_location import Locator, load_directory
        directory = load_directory(quiet=True)
        loc = Locator(directory) if directory else None
    except Exception:  # noqa: BLE001
        loc = None
    from flowlab.cluster import GeoMatcher
    from flowlab.sources import company_directory, zip_table
    cdir, ztab = company_directory(), zip_table()
    gm, where = {}, {}

    def place(name):
        if name not in where:
            e = loc.lookup(None, name) if loc else None
            where[name] = {"broker": name, "city": e["city"] if e else None, "district": e["district"] if e else None}
        return where[name]

    def geo_of(sid, name):
        if sid not in gm:
            gm[sid] = GeoMatcher(cdir.get(sid), ztab)
        return float(gm[sid].weight(place(name)))
    return geo_of, place


def run(day: str, stocks: list[str], report: bool = False) -> dict | None:
    t0 = time.time()
    caps = capital_directory()
    if not caps:
        log.warning("中小型吃貨：讀不到股本資料，略過")
        return None
    pool = [s for s in stocks if caps.get(s, {}).get("capital")]
    names = _names(day, pool)
    ids = _branch_ids()
    geo_of, place = _geo_fn()

    # 共同交易日曆（只讀價格檔）
    cal = set()
    for sid in pool:
        pr = quant._read(quant.LONG_DIR / sid / "_prices.json") or {}
        cal |= {d for d in (pr.get("data") or {}) if d <= day}
    if not cal:
        log.info("中小型吃貨：沒有價格資料")
        return None
    cal = [pd.Timestamp(d) for d in sorted(cal)]
    last_day = cal[-1]
    cal = cal[-(BT_DAYS + max(HORIZONS) + 2):]
    evals = cal[-1::-BT_STEP][::-1]          # 從最新一天往回每 5 天（最新一天一定在內）

    events = {v: [] for v in VARIANTS}
    base = {v: {h: {} for h in HORIZONS} for v in VARIANTS}
    sigs, recent_px, n_loaded = [], {}, 0
    for sid in pool:                          # 逐檔讀、算完就丟（記憶體只放一檔）
        try:
            S = load_stock(sid, day, geo_of)
        except Exception as ex:  # noqa: BLE001
            log.warning("[%s] 中小型吃貨讀取失敗：%s", sid, ex)
            continue
        if not S:
            continue
        n_loaded += 1
        cap = caps[sid]
        pos = {d: i for i, d in enumerate(S["idx"])}
        fw = {}
        for d in evals:
            t = pos.get(d)
            if t is not None:
                fw[d] = (t, {h: forward(S, t, h) for h in HORIZONS})
        for v, cfg in VARIANTS.items():
            last_sig = -10**9
            for d, (t, ret) in fw.items():
                if not size_ok(S, t, cap, cfg):
                    continue
                for h, r in ret.items():
                    if r is not None:
                        base[v][h].setdefault(d, []).append(r)
                e = eval_at(S, t, cap, cfg)
                if e is None:
                    continue
                events[v].append({"sid": sid, "d": d, "t": t, "new": t - last_sig > NEW_GAP, "score": e["score"], "ret": ret})
                last_sig = t
        t1 = len(S["idx"]) - 1
        recent_px[sid] = (S["idx"], S["vwap"], S["close"][t1])
        # 今天（這檔最新一天，最多落後一週）網站版本的訊號
        if (last_day - S["idx"][t1]).days > 7:
            continue
        cfg = VARIANTS[LIVE]
        e = eval_at(S, t1, cap, cfg)
        if e is None:
            continue
        M, lots, w = S["M"], e["lots"], cfg["win"]
        brs = []
        for i in sorted(e["sel"], key=lambda i: -M[f"add{w}"][i, t1]):
            k, nm = S["keys"][i], S["names"][i]
            c = M["cost"][i, t1]
            brs.append({"key": k, **place(nm), **ids.get(k, {"b_raw": k}), "geo": round(float(S["geo"][i]), 2),
                        "add": int(round(M[f"add{w}"][i, t1])), "lots": int(round(M["hold"][i, t1])),
                        "pct": round(float(M["hold"][i, t1]) / lots * 100, 2), "add_pct": round(float(M[f"add{w}"][i, t1]) / lots * 100, 2),
                        "buy_days": int(M[f"pos{w}"][i, t1]), "sell_days": int(M[f"neg{w}"][i, t1]),
                        "cost": None if not np.isfinite(c) else round(float(c), 2),
                        "since": S["idx"][int(M["since"][i, t1])].strftime("%Y-%m-%d")})
        prev = [ev for ev in events[LIVE] if ev["sid"] == sid and ev["t"] < t1]
        first = min([ev["d"] for ev in prev if t1 - ev["t"] <= NEW_GAP * 3] + [S["idx"][t1]])
        sh = cap.get("shares") or cap["capital"] / 10
        mcap = sh * float(S["close"][t1])
        sigs.append({"stock_id": sid, "stock_name": names.get(sid, ""), "asof": S["idx"][t1].strftime("%Y-%m-%d"),
                     "capital": cap["capital"], "mcap": round(mcap), "size": "小型" if mcap <= SMALL_MCAP else "中型",
                     "score": round(e["score"], 2), "run": round(e["run"], 4), "n": len(brs), "geo": any(x["geo"] > 1 for x in brs),
                     "close": float(S["close"][t1]), "cost": round(e["cost"], 2) if e["cost"] else None,
                     "gap": round(float(S["close"][t1]) / e["cost"] - 1, 4) if e["cost"] else None,
                     "hold_pct": round(sum(x["lots"] for x in brs) / lots * 100, 2), "first": pd.Timestamp(first).strftime("%Y-%m-%d"),
                     "branches": brs[:6]})
    if not n_loaded:
        log.info("中小型吃貨：沒有可用的股票（追蹤名單 %d 檔）", len(stocks))
        return None

    variants = {}
    for v, cfg in VARIANTS.items():
        bmean = {h: {d: float(np.mean(x)) for d, x in base[v][h].items()} for h in HORIZONS}
        new = [e for e in events[v] if e["new"]]
        by_year = {}
        for e in new:
            by_year.setdefault(str(e["d"].year), []).append(e)
        variants[v] = {"name": cfg["name"], "new": _stats(new, bmean), "all": _stats(events[v], bmean),
                       "by_year": {y: _stats(x, bmean) for y, x in sorted(by_year.items())},
                       "baseline": {f"h{h}": round(float(np.mean([x for xs in base[v][h].values() for x in xs])), 4) if base[v][h] else None for h in HORIZONS},
                       "n_stocks": len({e["sid"] for e in events[v]})}
    L = variants[LIVE]
    bt = {"from": evals[0].strftime("%Y-%m-%d"), "to": evals[-1].strftime("%Y-%m-%d"), "n_stocks": n_loaded,
          "variant": LIVE, "variant_name": VARIANTS[LIVE]["name"], **{k: L[k] for k in ("new", "all", "by_year", "baseline")},
          "note": "追蹤名單是現在的成交值前 N 名＋Ultra 自選（倖存者偏差，結果偏樂觀）；同一檔 20 個交易日內重複出現只算一次新訊號；基準＝同一天符合大小條件的全部股票平均"}
    sigs.sort(key=lambda x: (-x["geo"], -x["score"]))
    recent = []
    for e in sorted((e for e in events[LIVE] if e["new"]), key=lambda x: x["d"], reverse=True)[:40]:
        idx, vw, c1 = recent_px[e["sid"]]
        p0 = vw[e["t"] + 1] if e["t"] + 1 < len(idx) else None
        recent.append({"stock_id": e["sid"], "stock_name": names.get(e["sid"], ""), "date": e["d"].strftime("%Y-%m-%d"),
                       "score": round(e["score"], 2), "since_ret": round(float(c1 / p0 - 1), 4) if p0 else None,
                       "h20": None if e["ret"][20] is None else round(e["ret"][20], 4)})
    cfg = VARIANTS[LIVE]
    params = {**PARAMS, "win": cfg["win"], "size_kind": cfg["size"][0], "size_max": cfg["size"][1], "small_mcap": SMALL_MCAP,
              "no_hq": cfg["no_hq"], "mixed": cfg["mixed"], "geo_only": cfg["geo_only"], "flat": cfg["flat"], "variant": LIVE}
    doc = {"date": last_day.strftime("%Y-%m-%d"), "updated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "params": params, "universe": {"tracked": len(stocks), "small_mid": len({e["sid"] for e in events[LIVE]} | {s["stock_id"] for s in sigs}), "loaded": n_loaded},
           "signals": sigs, "recent": recent, "backtest": bt, "variants": variants}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{day}.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log.info("中小型吃貨：%d 檔、今日訊號 %d、版本 %s 回測新訊號 %d（%.0f 秒）", n_loaded, len(sigs), LIVE, L["new"]["n"], time.time() - t0)
    if report:
        def line(x):
            return "  ".join(f"{h}日 超額{x[f'h{h}']['excess']*100:+.1f}% 中位{x[f'h{h}']['median_excess']*100:+.1f}% 勝{x[f'h{h}']['win']*100:.0f}% t={x[f'h{h}']['t']}"
                             if x.get(f"h{h}") else f"{h}日 –" for h in HORIZONS)
        print("\n===== 各版本回測（新訊號；超額＝訊號股 − 同日同大小股票平均）=====")
        for v, x in variants.items():
            print(f"[{v}] {x['name']}：{x['new']['n']} 次，{x['n_stocks']} 檔\n    {line(x['new'])}")
            for y, yy in x["by_year"].items():
                print(f"      {y}（{yy['n']}）  {line(yy)}")
        print(f"\n今日訊號（{LIVE}）：", ", ".join(f"{s['stock_id']}{s['stock_name']}" for s in sigs) or "無")
    return doc


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    day = args[0]
    stocks = args[1:] or sorted(p.name for p in quant.LONG_DIR.iterdir() if p.is_dir())
    run(day, stocks, report="--report" in sys.argv)
