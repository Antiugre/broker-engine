"""
集團模式（使用者 2026-11-03 指定的流程）
  第一步  維度 A 逐檔算集團股票；維度 C 用全市場跑，再取出集團股票
  第二步  每檔股票 k、每個分點 i：p_i,k ＝ 它涉及的所有配對 p_meta 以 CCT 合併（等權重）
  第三步  p_i,group ＝ 分點 i 在所有集團股票上的 p_i,k 以 CCT 合併；只在一檔出現的分點就用那一檔
  第四步  假集團對照：從全市場挑 50 組報酬相關性、市值、流動性相近的股票組合，跑同樣的第二、三步
  第五步  局部 FDR（Efron）：以假集團分數為虛無分布 f0，lfdr(z) = π0·f0(z)/f(z)，π0 取 1（最保守）；
          可能性 = 1 − lfdr

  集團成員依時間切分：窗口必須整段落在成員期間內；成員組合不同的窗口分成不同段落，不跨段合併
  exclude_foreign（修正規格第 1 節，預設 false）：true 時外資分點不當維度 A 候選、維度 C 配對也拿掉外資後重算（假集團也一樣）；
  兩個版本並列（消融比較）

SPEC-GAP
  - 「沒有股權關係」無法驗證（沒有股權資料），只排除集團本身的股票
  - 假集團配對特徵：成員間日報酬相關係數平均、各成員 log 市值與 log 日均成交值（由小到大排），標準化後取歐氏距離最近的 50 組
  - lfdr 的密度以高斯核估計（z = Φ⁻¹(1 − p)），再做保守的單調化
  - 每個分點合併前 p 值上限截在 0.99（見 cct_many）
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
from scipy import stats

from .data_layer import build_panel, holdout_start_index
from .detect import Context, attach_c, run_window
from .dim_c import Market, window_c
from .fusion import cct
from .params import P

log = logging.getLogger("broker_scraper")
DETAIL_WINDOWS = 6
TOP = 30
DEADLINE: list = [None]      # time.time() 截止點（run.run_all 設定）


def deadline_passed() -> bool:
    import time
    return DEADLINE[0] is not None and time.time() > DEADLINE[0]


def _active(members: list[dict], start: str, end: str, have: set) -> tuple:
    return tuple(m["sid"] for m in members if m["from"] <= start and end <= m["to"] and m["sid"] in have)


def branch_p(ctx: dict, sids: tuple, end: str, cres: dict | None, nofx: bool, holdout: bool = True) -> dict:
    """第二、三步：回傳 {分點 key: {"k": {代號: p_i,k}, "p": p_i,group, "name", "type"}}"""
    per = {}
    for sid in sids:
        cx = ctx[sid]
        pn = cx.pn
        e = int(np.searchsorted(pn.dates, end, side="right")) - 1
        if e < int(P.detection.window_days) - 1:
            continue
        w = run_window(cx, e, allow_holdout=holdout, exclude_foreign=nofx)
        if w["res"] is None:
            continue
        attach_c(w, pn, cres)
        r = w["res"]
        cand = w["cand"]
        bucket: dict[int, list] = {}
        for k, (a, b) in enumerate(r["pairs"]):
            ia, ib = cand[a], cand[b]
            if nofx and (pn.btype[ia] == "foreign" or pn.btype[ib] == "foreign"):
                continue
            bucket.setdefault(ia, []).append(r["p_meta"][k])
            bucket.setdefault(ib, []).append(r["p_meta"][k])
        for i, ps in bucket.items():
            key = pn.branches[i]
            d = per.setdefault(key, {"k": {}, "name": pn.names[i], "type": pn.btype[i]})
            d["k"][sid] = cct_many(ps)
    for d in per.values():
        d["p"] = cct_many(list(d["k"].values()))
    return per


P_CAP = 0.99   # SPEC-GAP：合併前把 p 值上限截在 0.99


def cct_many(ps) -> float:
    """多個 p 值等權重 CCT。
    離散檢定（共同活躍天數為 0 的配對 p_act = 1）會讓 tan((0.5 − p)π) 趨近負無窮、把所有證據蓋掉，
    所以合併前把 p 截在 0.99（SPEC-GAP）。這會讓 p 集團的絕對值偏小，但最後的判斷是跟假集團比（第四、五步），
    假集團用完全相同的算法，所以這個偏差會被抵銷。"""
    return float(cct([np.minimum(np.asarray(p, float), P_CAP) for p in ps]))


def lfdr(real: np.ndarray, null: np.ndarray) -> np.ndarray:
    """第五步：lfdr = f0/f（π0 = 1），回傳可能性 1 − lfdr"""
    e = float(P.fusion.p_clip)
    zr = stats.norm.isf(np.clip(real, e, 1 - e))
    z0 = stats.norm.isf(np.clip(null, e, 1 - e))
    if len(zr) < 3 or len(z0) < 10 or np.std(zr) == 0 or np.std(z0) == 0:
        return np.full(len(real), np.nan)
    f0 = stats.gaussian_kde(z0)(zr)
    f = stats.gaussian_kde(zr)(zr)
    prob = np.clip(1 - f0 / np.maximum(f, 1e-300), 0, 1)
    # 單調化（保守）：分數越高的分點，可能性不得高於任何分數更高者的最小值
    o = np.argsort(-zr)
    prob[o] = np.minimum.accumulate(prob[o])
    return prob


def match_fakes(sids: tuple, pool: list[str], ctx_px: dict, start: str, end: str, n: int, cap: dict) -> list[tuple]:
    """第四步：挑 n 組報酬相關性、市值、流動性和真實集團相近的股票組合"""
    def feat(s):
        d, c, v = ctx_px[s]
        m = (d >= start) & (d <= end)
        if m.sum() < 60:
            return None
        sh = (cap.get(s) or {}).get("shares")
        mc = np.log(sh * np.nanmean(c[m])) if sh else np.nan
        liq = np.log(np.nanmean(v[m] * c[m]) + 1)
        ret = np.r_[0, np.diff(np.log(c))][m]
        return {"d": d[m], "ret": ret, "mc": mc, "liq": liq}

    F = {s: feat(s) for s in set(pool) | set(sids)}
    F = {s: f for s, f in F.items() if f is not None and np.isfinite(f["mc"])}
    if any(s not in F for s in sids):
        return []

    def corr(a, b):
        da, db = F[a], F[b]
        common, ia, ib = np.intersect1d(da["d"], db["d"], return_indices=True)
        return float(np.corrcoef(da["ret"][ia], db["ret"][ib])[0, 1]) if len(common) > 20 else np.nan

    def vec(g):
        cs = [corr(a, b) for a, b in itertools.combinations(g, 2)]
        return np.r_[np.nanmean(cs) if cs else 0.0, sorted(F[s]["mc"] for s in g), sorted(F[s]["liq"] for s in g)]

    cand = [s for s in pool if s in F and s not in sids]
    k = len(sids)
    combos = list(itertools.combinations(cand, k)) if k <= 2 else \
        [tuple(np.random.default_rng(0).choice(cand, k, replace=False)) for _ in range(5000)]
    target = vec(sids)
    X = np.array([vec(g) for g in combos])
    ok = np.all(np.isfinite(X), axis=1)
    X, combos = X[ok], [g for g, o in zip(combos, ok) if o]
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    dist = np.sqrt((((X - target) / sd) ** 2).sum(axis=1))
    order = np.argsort(dist)[:n]
    return [tuple(combos[i]) for i in order]


def run_group(g: dict, mk: Market, ctx: dict, cap: dict, holdout: bool = True) -> dict:
    members = g["members"]
    have = set(ctx)
    cal = mk.cal
    W = int(P.detection.window_days)
    pos = {d: i for i, d in enumerate(cal)}
    segs: dict[tuple, list] = {}
    for end in mk.ends:
        i = pos.get(end)
        if i is None or i - W + 1 < 0:
            continue
        act = _active(members, str(cal[i - W + 1]), end, have)
        if act:
            segs.setdefault(act, []).append(end)
    hold = str(cal[holdout_start_index(cal)])
    try:
        from smallcap import _branch_ids          # 分點代碼（網頁點選後在 K 線標記用）
        ids = _branch_ids()
    except Exception:  # noqa: BLE001
        ids = {}
    px = {}
    for s in mk.sids:
        st = mk.stocks[s]
        px[s] = (st.dates, st.close, st.volume)
    out_segs = []
    for act, ends in segs.items():
        lo_i = pos[ends[0]] - W + 1
        fakes = match_fakes(act, [s for s in mk.sids if s not in g.get("exclude_from_fake", [])],
                            px, str(cal[lo_i]), ends[-1], int(g.get("fake_groups", 50)), cap)
        need = sorted({s for f in fakes for s in f} | set(act))
        for s in need:
            if s not in ctx:
                try:
                    ctx[s] = Context(build_panel(s, write=False))
                except Exception:  # noqa: BLE001
                    pass
        seg = {"members": list(act), "fake_groups": ["+".join(f) for f in fakes], "windows": [], "partial": False}
        log.info("集團 %s 段落 %s：%d 個窗口、%d 組假集團", g["id"], "+".join(act), len(ends), len(fakes))
        for n_end, end in sorted(enumerate(ends), reverse=True):      # 新的窗口先算（時間不夠時至少有最近的結果）
            if deadline_passed():
                seg["partial"] = True
                log.warning("集團 %s：時間到，%s 之前的窗口下次再算", g["id"], end)
                break
            cres = window_c(mk, end, []) if mk.fit_for(end) is not None else None
            item = {"end": end, "start": str(cal[pos[end] - W + 1]), "has_c": cres is not None, "holdout": end >= hold}
            for var, nofx in (("all", False), ("nofx", True)):
                real = branch_p(ctx, act, end, cres, nofx, holdout)
                null_p, null_max = [], []
                for f in fakes:
                    fp = branch_p(ctx, tuple(s for s in f if s in ctx), end, cres, nofx, holdout)
                    v = np.array([d["p"] for d in fp.values()])
                    if len(v):
                        null_p.append(v)
                        null_max.append(-np.log10(v.min()))
                null = np.concatenate(null_p) if null_p else np.array([])
                keys = list(real)
                rp = np.array([real[k]["p"] for k in keys])
                prob = lfdr(rp, null) if len(rp) else rp
                emp = np.array([(1 + (null <= p).sum()) / (1 + len(null)) for p in rp]) if len(null) else np.full(len(rp), np.nan)
                top_s = float(-np.log10(rp.min())) if len(rp) else None
                v = {"n_branches": len(keys), "n_null": int(len(null)), "n_fake": len(null_max),
                     "n_prob50": int(np.nansum(prob >= 0.5)) if len(rp) else 0,
                     "max_score": top_s,
                     "fake_max_rank": float((np.array(null_max) >= top_s).mean()) if null_max and top_s is not None else None}
                if n_end >= len(ends) - DETAIL_WINDOWS:
                    order = np.argsort(rp)[:TOP]
                    known = set(g.get("known_branches") or [])
                    v["top"] = [{"key": keys[i], "name": real[keys[i]]["name"], "type": real[keys[i]]["type"], **ids.get(keys[i], {}),
                                 "p_k": {s: float(f"{x:.3g}") for s, x in real[keys[i]]["k"].items()},
                                 "p": float(f"{rp[i]:.3g}"), "p_emp": float(f"{emp[i]:.3g}"),
                                 "prob": None if not np.isfinite(prob[i]) else round(float(prob[i]), 3),
                                 "known": keys[i] in known} for i in order]
                item[var] = v
            seg["windows"].append(item)
        seg["windows"].sort(key=lambda x: x["end"])
        out_segs.append(seg)
    return {"id": g["id"], "name": g["name"], "members": members, "segments": out_segs,
            "known_branches": g.get("known_branches") or []}
