"""
FinMind 分點資料（Sponsor Pro 整天下載，私人倉庫 broker-data 的月檔）→ 給各引擎用的「分點每日買賣」來源

  資料夾  DATA_DIR/finmind/SecIdAgg/YYYY-MM.parquet（依 股票、分點、日期 排序；雲端由工作流程從私人倉庫下載）
  欄位    stock_id, securities_trader_id, securities_trader, date, buy_volume, sell_volume（股）, buy_price, sell_price（分點當日買／賣均價）

  與富邦的關係（使用者要求：富邦路徑保留作備用）
    ・有 FinMind 資料、而且找得到這個分點 → 用 FinMind（不連富邦，也不寫進 longhist 快取，避免 FinMind 資料進到公開快取）
    ・沒有 FinMind 資料夾、資料太舊、或這個分點對不到 → 回傳 None，呼叫端照原本的方式向富邦補
    ・環境變數 BROKER_HISTORY_SOURCE=fubon 可整個關掉 FinMind（強制走富邦）

  分點代碼：富邦的 b_raw 可能是 UTF-16BE 十六進位（"003900380031006a" → "981j"），代碼區分大小寫
  FinMind 分點資料限訂閱者自己使用：本模組的結果只供本系統計算與 Ultra（目前只有站長本人）顯示，不得公開原始資料
"""
from __future__ import annotations

import datetime as dt
import logging
import os
from functools import lru_cache
from pathlib import Path

from paths import DATA_DIR

log = logging.getLogger("broker_scraper")
FM_DIR = Path(os.environ.get("FM_DIR") or (DATA_DIR / "finmind"))
SEC_DIR = FM_DIR / "SecIdAgg"
STALE_DAYS = 7           # FinMind 最後一天比要算的日子舊超過這麼多天 → 視為不可用（改走富邦）
LOT = 1000


def enabled() -> bool:
    if os.environ.get("BROKER_HISTORY_SOURCE", "").lower() == "fubon":
        return False
    return SEC_DIR.exists() and any(SEC_DIR.glob("*.parquet"))


def decode_key(key: str) -> str:
    """富邦 b_raw／history key → 分點代碼（區分大小寫）"""
    if key and len(key) >= 8 and len(key) % 4 == 0:
        try:
            return bytes.fromhex(key).decode("utf-16-be")
        except (ValueError, UnicodeDecodeError):
            pass
    return key


def _files() -> list[Path]:
    return sorted(SEC_DIR.glob("????-??.parquet"))


@lru_cache(maxsize=1)
def last_date() -> str | None:
    import pyarrow.parquet as pq
    fs = _files()
    if not fs:
        return None
    t = pq.read_table(fs[-1], columns=["date"])
    return max(t.column("date").to_pylist()) if t.num_rows else None


@lru_cache(maxsize=48)
def stock_table(sid: str):
    """一檔股票的全部月份（pandas DataFrame）。月檔依股票排序 → 只讀到少數 row group"""
    import pandas as pd
    import pyarrow.dataset as ds
    cols = ["securities_trader_id", "securities_trader", "date", "buy_volume", "sell_volume", "buy_price", "sell_price"]
    parts = []
    for f in _files():
        t = ds.dataset(f, format="parquet").to_table(columns=cols, filter=ds.field("stock_id") == sid)
        if t.num_rows:
            parts.append(t.to_pandas())
    if not parts:
        return pd.DataFrame(columns=cols)
    return _apply_corrections(sid, pd.concat(parts, ignore_index=True))


# ─────────── 人工更正（使用者本人確認過的交易；任何資料來源都不能蓋過它）───────────
#   檔案：私人倉庫 broker-data Release「TaiwanStockTradingDailyReportSecIdAgg」的 corrections.json（不放公開倉庫）
#   位置：環境變數 BB_CORRECTIONS ＞ DATA_DIR/corrections.json（本機 MandM）＞ FM_DIR/SecIdAgg/corrections.json
#   必要：雲端與本機正式執行時檔案一定要在（缺檔或格式錯誤 → 直接失敗，不照算）；只有 pytest 或 BB_ALLOW_NO_CORRECTIONS=1 可以沒有
#   套用：FinMind（stock_table）與富邦備用路徑（apply_rows）都在讀資料這一層套用，各引擎吃到同一份
#   記錄：applied(sid) 這次執行是否真的套用到；上傳前據此把「沒有套用到更正的那檔」擋下（firebase_upload）
def _corr_path() -> Path:
    env = os.environ.get("BB_CORRECTIONS")
    if env:
        return Path(env)
    for p in (DATA_DIR / "corrections.json", SEC_DIR / "corrections.json"):
        if p.exists():
            return p
    return DATA_DIR / "corrections.json"


def corrections_required() -> bool:
    if os.environ.get("BB_ALLOW_NO_CORRECTIONS") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
        return os.environ.get("BB_REQUIRE_CORRECTIONS") == "1"
    return True


@lru_cache(maxsize=1)
def _corrections_raw() -> tuple[list[dict], str]:
    """(更正清單, 檔案雜湊前 12 碼)。必要時缺檔或格式錯誤 → RuntimeError（工作流程該段失敗，不照算）"""
    import hashlib
    import json
    p = _corr_path()
    if not p.exists():
        if corrections_required():
            raise RuntimeError(f"找不到人工更正檔 {p}：不能照算（請確認已從私人倉庫下載 corrections.json）")
        return [], ""
    blob = p.read_bytes()
    try:
        items = json.loads(blob.decode("utf-8"))
        assert isinstance(items, list) and all(isinstance(x, dict) and x.get("stock_id") and x.get("date") for x in items)
    except Exception as ex:  # noqa: BLE001
        raise RuntimeError(f"人工更正檔 {p} 格式錯誤：{ex}") from ex
    h = hashlib.sha256(blob).hexdigest()[:12]
    log.info("人工更正檔 %s：%d 筆、版本 %s", p, len(items), h)
    return items, h


def corrections_hash() -> str:
    return _corrections_raw()[1]


def corrected_stocks() -> set[str]:
    return {str(x["stock_id"]) for x in _corrections_raw()[0]}


_APPLIED: dict[str, set] = {}        # 代號 → {(日期, 分點名稱)}：這次執行真的套用到的更正
_MISSED: set[str] = set()            # 這次執行有更正卻對不到分點的股票（上傳前擋下）


def _mark(sid: str, date: str, name: str) -> None:
    _APPLIED.setdefault(sid, set()).add((str(date), _norm_name(name)))


def applied(sid: str) -> bool:
    """這檔的每一筆更正，這次執行都套用到了嗎（沒有更正的股票一律 True）"""
    need = {(str(x["date"]), _norm_name(x.get("securities_trader", ""))) for x in _corrections_raw()[0] if str(x["stock_id"]) == sid}
    return need <= _APPLIED.get(sid, set())


def upload_ok(sid: str) -> bool:
    """上傳前檢查：有更正的股票，更正檔必須已載入、而且這次沒有發生「讀到這檔卻對不到更正分點」"""
    if sid not in corrected_stocks():
        return True
    return bool(corrections_hash()) and sid not in _MISSED


def stamp(sid: str) -> dict | None:
    """寫進輸出文件的更正紀錄（網站上可確認套用了哪一版）"""
    if sid not in corrected_stocks():
        return None
    return {"version": corrections_hash(), "applied": applied(sid), "items": applied_items(sid)}


def applied_items(sid: str) -> list[dict]:
    return [{k: x[k] for k in ("date", "securities_trader", "buy_volume", "sell_volume", "buy_price", "sell_price") if k in x}
            for x in _corrections_raw()[0] if str(x["stock_id"]) == sid]


@lru_cache(maxsize=1)
def _corrections():
    import pandas as pd
    return pd.DataFrame(_corrections_raw()[0])


def _norm_name(x) -> str:
    return "".join(str(x).replace("證券", "").split()).replace("-", "").replace("－", "").replace("—", "")


def _corr_bid(sid: str, r, df=None) -> str:
    bid = str(r.get("securities_trader_id") or "")
    if bid in ("", "nan", "None"):
        want = _norm_name(r.get("securities_trader", ""))       # 名稱寫法不同（統一-仁愛／統一仁愛）也對得到
        if df is not None and len(df):
            hit = df.loc[df["securities_trader"].astype(str).map(_norm_name) == want, "securities_trader_id"]
            bid = str(hit.iloc[-1]) if len(hit) else ""
        if not bid:                                              # 這檔股票從沒出現過這個分點 → 用富邦分點清單找代碼
            bid = next((k for k, x in _fubon_ids().items() if _norm_name(x.get("broker") or x.get("name") or "") == want), "")
    return bid


def _apply_corrections(sid: str, df):
    import numpy as np
    import pandas as pd
    items = [x for x in _corrections_raw()[0] if str(x["stock_id"]) == sid]
    for r in items:
        bid = _corr_bid(sid, r, df)
        if not bid:
            log.warning("人工更正 %s %s %s：找不到這個分點的代碼（這檔不會上傳）", sid, r["date"], r.get("securities_trader"))
            _MISSED.add(sid)
            continue
        m = (df["securities_trader_id"].astype(str) == bid) & (df["date"].astype(str) == str(r["date"]))
        if not m.any():
            df = pd.concat([df, pd.DataFrame([{"securities_trader_id": bid, "securities_trader": str(r.get("securities_trader")), "date": str(r["date"]),
                                               "buy_volume": 0.0, "sell_volume": 0.0, "buy_price": np.nan, "sell_price": np.nan}])],
                           ignore_index=True)
            m = (df["securities_trader_id"].astype(str) == bid) & (df["date"].astype(str) == str(r["date"]))
        for col in ("buy_volume", "sell_volume", "buy_price", "sell_price"):
            v = r.get(col)
            if v is not None and pd.notna(v):
                df.loc[m, col] = float(v)
        _mark(sid, r["date"], r.get("securities_trader", ""))
        log.info("人工更正 %s %s %s（%s）：買 %s 股、賣 %s 股（版本 %s）", sid, r["date"], r.get("securities_trader"), bid,
                 df.loc[m, "buy_volume"].iloc[0], df.loc[m, "sell_volume"].iloc[0], corrections_hash())
    return df.sort_values(["securities_trader_id", "date"], kind="stable").reset_index(drop=True)


def apply_rows(sid: str, name: str, key: str, rows: list[dict]) -> list[dict]:
    """富邦備用路徑（longhist）用：rows＝[{date, buy, sell(, buy_price, sell_price)}]（張）。
    分點以名稱或代碼對應；回傳更正後的新清單（依日期排序）"""
    items = [x for x in _corrections_raw()[0] if str(x["stock_id"]) == sid]
    if not items:
        return rows
    bid = decode_key(key or "")
    out = {r["date"]: dict(r) for r in rows}
    for r in items:
        same = _norm_name(r.get("securities_trader", "")) == _norm_name(name or "") or (r.get("securities_trader_id") and str(r["securities_trader_id"]) == bid) \
            or (_corr_bid(sid, r) and _corr_bid(sid, r) == bid)
        if not same:
            continue
        d = str(r["date"])
        x = out.setdefault(d, {"date": d, "buy": 0, "sell": 0})
        if r.get("buy_volume") is not None:
            x["buy"] = float(r["buy_volume"]) / LOT
        if r.get("sell_volume") is not None:
            x["sell"] = float(r["sell_volume"]) / LOT
        if r.get("buy_price") is not None:
            x["buy_price"] = float(r["buy_price"])
        if r.get("sell_price") is not None:
            x["sell_price"] = float(r["sell_price"])
        _mark(sid, d, r.get("securities_trader", ""))
        log.info("人工更正（富邦備用路徑）%s %s %s：買 %s 張、賣 %s 張（版本 %s）", sid, d, name, x["buy"], x["sell"], corrections_hash())
    return sorted(out.values(), key=lambda r: r["date"])


@lru_cache(maxsize=48)
def _by_branch(sid: str) -> dict:
    df = stock_table(sid)
    out: dict[str, object] = {}
    for bid, g in df.groupby("securities_trader_id", sort=False):
        out[str(bid)] = g
    return out


def _lookup(sid: str, key: str):
    bb = _by_branch(sid)
    bid = decode_key(key)
    if bid in bb:
        return bb[bid]
    alt = [k for k in bb if k.upper() == bid.upper()]
    return bb[alt[0]] if len(alt) == 1 else None


def history(sid: str, key: str, floor: dt.date, end: dt.date) -> list[dict] | None:
    """與 quant.load_branch_history 相同格式：[{date, buy, sell}]（張，可有小數＝零股），另附 buy_price / sell_price。
    None ＝ 這次不能用 FinMind（呼叫端改走富邦）"""
    if not enabled():
        return None
    last = last_date()
    if not last or (end - dt.date.fromisoformat(last)).days > STALE_DAYS:
        return None
    g = _lookup(sid, key)
    if g is None:
        # 對不到（分點代碼不同、裁撤合併等）→ 交給富邦補，不當成「沒有成交」
        return None
    lo, hi = floor.isoformat(), end.isoformat()
    g = g[(g["date"] >= lo) & (g["date"] <= hi)]
    return [{"date": r.date, "buy": r.buy_volume / LOT, "sell": r.sell_volume / LOT,
             "buy_price": float(r.buy_price) if r.buy_volume else None,
             "sell_price": float(r.sell_price) if r.sell_volume else None}
            for r in g.sort_values("date").itertuples()]


def branches(sid: str) -> list[tuple[str, str]]:
    """這檔股票在 FinMind 出現過的全部分點：[(分點代碼, 名稱)]"""
    df = stock_table(sid)
    if not len(df):
        return []
    last = df.drop_duplicates("securities_trader_id", keep="last")
    return list(zip(last["securities_trader_id"].astype(str), last["securities_trader"].astype(str)))


@lru_cache(maxsize=1)
def _fubon_ids() -> dict[str, dict]:
    """富邦分點清單（sweep 快取）：分點代碼 → {broker, broker_id, b_raw}（網頁點進分點明細要用）"""
    import json
    try:
        lst = json.loads((DATA_DIR / "fubon_branches.json").read_text(encoding="utf-8")).get("branches") or []
    except Exception:  # noqa: BLE001
        return {}
    return {x["branch_id"]: x for x in lst if x.get("branch_id")}


def fubon_row(bid: str, name: str) -> dict:
    """FinMind 分點代碼 → 與富邦排行頁相同的欄位（broker_id = 總公司代碼、b_raw = 富邦網址參數）"""
    x = _fubon_ids().get(bid)
    if x:
        return {"broker": x.get("broker") or name, "broker_id": x["broker_id"], "branch_id": bid, "b_raw": x["b_raw"]}
    raw = bid if (bid.isdigit() or bid.endswith("00")) else bid.encode("utf-16-be").hex()
    return {"broker": name, "broker_id": bid[:2] + "00", "branch_id": bid, "b_raw": raw}


def range_top(sid: str, start: str, end: str, n: int = 15) -> dict | None:
    """區間合計買超／賣超前 n 名（取代富邦「區間」頁；格式同 scraper.parse_page 的 buy_side / sell_side）。
    None ＝ 不能用（FinMind 還沒有 end 那天的資料等）"""
    if not enabled() or (last_date() or "") < end:
        return None
    df = stock_table(sid)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if not len(df):
        return None
    g = df.groupby("securities_trader_id").agg(buy=("buy_volume", "sum"), sell=("sell_volume", "sum"),
                                               name=("securities_trader", "last"))
    g["net"] = (g["buy"] - g["sell"]) / LOT
    tot = float(g["buy"].sum() / LOT) or 1.0

    def rows(part):
        out = []
        for k, (bid, r) in enumerate(part.iterrows(), 1):
            out.append({**fubon_row(str(bid), str(r["name"])), "buy": round(r["buy"] / LOT), "sell": round(r["sell"] / LOT),
                        "net": round(r["net"]), "pct": round(abs(r["net"]) / tot * 100, 2), "rank": k})
        return out
    return {"stock_id": sid, "date": end, "unit": "張", "source": "finmind",
            "buy_side": rows(g[g["net"] > 0].nlargest(n, "net")), "sell_side": rows(g[g["net"] < 0].nsmallest(n, "net")),
            "query_date": end, "range": {"start": start, "end": end}}


TDR_DIR = FM_DIR / "TDR"     # 分點價位明細（私人倉庫 TDR-by-stock：hist-<代號>.parquet＋cur.parquet）


@lru_cache(maxsize=8)
def tdr(sid: str):
    """一檔股票的分點價位明細：date, securities_trader_id, price, buy, sell（股）。沒有資料 → 空表"""
    import pandas as pd
    import pyarrow.dataset as ds
    cols = ["date", "securities_trader_id", "price", "buy", "sell"]
    parts = []
    f = TDR_DIR / f"hist-{sid}.parquet"
    if f.exists():
        parts.append(ds.dataset(f, format="parquet").to_table(columns=cols).to_pandas())
    c = TDR_DIR / "cur.parquet"
    if c.exists():
        t = ds.dataset(c, format="parquet").to_table(columns=cols, filter=ds.field("stock_id") == sid)
        if t.num_rows:
            parts.append(t.to_pandas())
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def branch_series(sid: str):
    """這檔股票每個分點的完整每日買賣（張）＋分點買進均價：[(longhist 檔名 key, 富邦名稱, DataFrame[date, buy, sell, buy_price])]。
    給直接讀 longhist 的引擎（本土主力、中小型吃貨）改讀 FinMind 用；None ＝ 不能用（改讀 longhist）"""
    if not enabled():
        return None
    from history import history_key
    df = stock_table(sid)
    if not len(df):
        return None
    out = []
    for bid, g in df.groupby("securities_trader_id", sort=False):
        name = str(g["securities_trader"].iloc[-1])
        row = fubon_row(str(bid), name)
        x = g[["date"]].copy()
        x["buy"] = g["buy_volume"].to_numpy() / LOT
        x["sell"] = g["sell_volume"].to_numpy() / LOT
        x["buy_price"] = g["buy_price"].where(g["buy_volume"] > 0).astype(float).to_numpy()
        out.append((history_key({"b_raw": row["b_raw"]}), row["broker"], x))
    return out
