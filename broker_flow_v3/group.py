"""
§9 集團模式（v3）
  p_(i,i'),k = min(1, 2·min(p_same, p_opp))（兩通道 Bonferroni）
  p_i,k = Simes（分點 i 在股票 k 的所有配對）；p_i,group = CCT_k(p_i,k)
  假集團：股票數相同，從股票池排除集團本身與保留池 H；特徵＝產業（不符加懲罰 3）、成員間日報酬相關、log 市值、log 日均成交值；
          標準化特徵的馬氏距離，取最近 50 組
  SPEC-GAP-v3（今晚上線版）：「log 候選分點數」「分點成交集中度 HHI」兩個特徵尚未加入；假集團只算最近 fake_windows 個窗口、只算預設版本（不排除外資）
  §9.3 某段期間只剩 1 檔 → 不輸出集團模式，狀態 single_stock
"""
from __future__ import annotations

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from . import auxdata as aux, split
from .data_layer import build_panel
from .detect import Context, run_window
from .params import P
from .pvals import cct, simes

log = logging.getLogger("broker_scraper")
TOP = 30


def branch_p(cx: Context, e: int, nofx: bool) -> dict:
    """{分點代碼: (p_i,k, 名稱, 類型)}"""
    w = run_window(cx, e, nofx)
    r = w["res"]
    if r is None:
        return {}
    pn = cx.pn
    pp = np.minimum(1.0, 2 * np.fmin(r["p_same"], r["p_opp"]))
    bucket: dict[int, list] = {}
    for k, (a, b) in enumerate(r["pairs"]):
        if np.isfinite(pp[k]):
            bucket.setdefault(w["cand"][a], []).append(pp[k])
            bucket.setdefault(w["cand"][b], []).append(pp[k])
    return {pn.branches[i]: (simes(v), pn.names[i], pn.btype[i]) for i, v in bucket.items()}


def lfdr(real: np.ndarray, null: np.ndarray) -> np.ndarray:
    e = float(P.fusion.p_clip)
    zr = stats.norm.isf(np.clip(real, e, 1 - e))
    z0 = stats.norm.isf(np.clip(null, e, 1 - e))
    if len(zr) < 3 or len(z0) < 10 or np.std(zr) == 0 or np.std(z0) == 0:
        return np.full(len(real), np.nan)
    prob = np.clip(1 - stats.gaussian_kde(z0)(zr) / np.maximum(stats.gaussian_kde(zr)(zr), 1e-300), 0, 1)
    o = np.argsort(-zr)
    prob[o] = np.minimum.accumulate(prob[o])
    return prob


def _px(sid: str):
    from .data_layer import _prices_longhist
    d = _prices_longhist(sid)
    return d["date"].to_numpy(str), d["close"].to_numpy(float), d["volume"].to_numpy(float)


def match_fakes(sids: tuple, pool: list[str], start: str, end: str, n: int) -> list[tuple]:
    from smallcap import capital_directory
    cap = capital_directory()
    info = aux.static("TaiwanStockInfo")
    ind = {}
    if len(info) and "industry_category" in info:
        ind = info.drop_duplicates("stock_id", keep="last").set_index("stock_id")["industry_category"].astype(str).to_dict()

    def feat(s):
        d, c, v = _px(s)
        m = (d >= start) & (d <= end)
        if m.sum() < 60:
            return None
        sh = (cap.get(s) or {}).get("shares")
        mc = np.log(sh * np.nanmean(c[m])) if sh else np.nan
        return {"d": d[m], "ret": np.r_[0, np.diff(np.log(c))][m], "mc": mc, "liq": np.log(np.nanmean(v[m] * c[m]) + 1)}
    F = {s: feat(s) for s in set(pool) | set(sids)}
    F = {s: f for s, f in F.items() if f is not None and np.isfinite(f["mc"])}
    if any(s not in F for s in sids):
        return []

    def corr(a, b):
        _, ia, ib = np.intersect1d(F[a]["d"], F[b]["d"], return_indices=True)
        return float(np.corrcoef(F[a]["ret"][ia], F[b]["ret"][ib])[0, 1]) if len(ia) > 20 else np.nan

    def vec(g):
        cs = [corr(a, b) for a, b in itertools.combinations(g, 2)]
        return np.r_[np.nanmean(cs) if cs else 0.0, sorted(F[s]["mc"] for s in g), sorted(F[s]["liq"] for s in g)]
    cand = [s for s in pool if s in F and s not in sids]
    k = len(sids)
    combos = list(itertools.combinations(cand, k)) if k <= 2 else \
        [tuple(np.random.default_rng(0).choice(cand, k, replace=False)) for _ in range(5000)]
    X = np.array([vec(g) for g in combos])
    ok = np.all(np.isfinite(X), axis=1)
    X, combos = X[ok], [g for g, o in zip(combos, ok) if o]
    if not len(combos):
        return []
    target = vec(sids)
    C = np.cov(X.T) + 1e-9 * np.eye(X.shape[1])
    Ci = np.linalg.pinv(C)
    D = X - target
    dist = np.sqrt(np.einsum("ij,jk,ik->i", D, Ci, D))
    tind = sorted(ind.get(s, "?") for s in sids)
    pen = np.array([0.0 if sorted(ind.get(s, "?") for s in g) == tind else 3.0 for g in combos])
    order = np.argsort(dist + pen)[:n]
    return [tuple(combos[i]) for i in order]


_CTX: dict = {}


def _ctx(sid: str) -> Context | None:
    if sid not in _CTX:
        try:
            _CTX[sid] = Context(build_panel(sid))
        except Exception as ex:  # noqa: BLE001
            log.warning("v3 假集團 %s：%s", sid, ex)
            _CTX[sid] = None
    return _CTX[sid]


def _fake_job(args):
    sid, ends = args
    cx = _ctx(sid)
    if cx is None:
        return sid, {}
    out = {}
    for end in ends:
        e = int(np.searchsorted(cx.pn.dates, end, side="right")) - 1
        if e >= int(P.accounting.warmup_days) + int(P.detection.window_days) - 1:
            out[end] = {k: v[0] for k, v in branch_p(cx, e, False).items()}
    return sid, out


def run_group(g: dict, ctx: dict, pool: list[str], ends_all: list[str]) -> dict:
    W = int(P.detection.window_days)
    members = g["members"]
    have = set(ctx)
    cal = ctx[next(iter(ctx))].pn.dates
    pos = {d: i for i, d in enumerate(cal)}
    segs: dict[tuple, list] = {}
    single = []
    for end in ends_all:
        i = pos.get(end)
        if i is None or i - W + 1 < 0:
            continue
        st = str(cal[i - W + 1])
        act = tuple(m["sid"] for m in members if m["from"] <= st and end <= m["to"] and m["sid"] in have)
        if len(act) >= 2:
            segs.setdefault(act, []).append(end)
        elif act:
            single.append(end)
    hold = set(split.holdout().get("stocks") or [])
    excl = set(g.get("exclude_from_fake", [])) | hold
    out_segs = []
    for act, ends in segs.items():
        lo = str(cal[pos[ends[0]] - W + 1])
        fakes = match_fakes(act, [s for s in pool if s not in excl], lo, ends[-1], int(g.get("fake_groups", 50)))
        fw = sorted(ends)[-int(P.group_mode.fake_windows):]
        need = sorted({s for f in fakes for s in f})
        log.info("v3 集團 %s 段落 %s：%d 個窗口、%d 組假集團（%d 檔，算最近 %d 個窗口）", g["id"], "+".join(act), len(ends), len(fakes), len(need), len(fw))
        fake_res: dict = {}
        with ProcessPoolExecutor(4) as ex:
            for sid, res in ex.map(_fake_job, [(s, fw) for s in need]):
                fake_res[sid] = res
        seg = {"members": list(act), "fake_groups": ["+".join(f) for f in fakes], "windows": []}
        for end in sorted(ends):
            item = {"end": end, "start": str(cal[pos[end] - W + 1]), "has_c": False, "fake": end in fw}
            for var, nofx in (("all", False), ("nofx", True)):
                per = {}
                for sid in act:
                    cx = ctx[sid]
                    e = int(np.searchsorted(cx.pn.dates, end, side="right")) - 1
                    for k, (p, nm, tp) in branch_p(cx, e, nofx).items():
                        d = per.setdefault(k, {"k": {}, "name": nm, "type": tp})
                        d["k"][sid] = p
                keys = list(per)
                rp = np.array([cct(list(per[k]["k"].values())) for k in keys])
                v = {"n_branches": len(keys)}
                null = np.array([])
                if var == "all" and end in fw:
                    null_p, null_max = [], []
                    for f in fakes:
                        fp: dict = {}
                        for s in f:
                            for k, p in (fake_res.get(s) or {}).get(end, {}).items():
                                fp.setdefault(k, []).append(p)
                        vv = np.array([cct(x) for x in fp.values()])
                        if len(vv):
                            null_p.append(vv)
                            null_max.append(-np.log10(vv.min()))
                    null = np.concatenate(null_p) if null_p else np.array([])
                    top_s = float(-np.log10(rp.min())) if len(rp) else None
                    v.update(n_null=int(len(null)), n_fake=len(null_max), max_score=top_s,
                             fake_max_rank=float((np.array(null_max) >= top_s).mean()) if null_max and top_s is not None else None)
                prob = lfdr(rp, null) if len(rp) and len(null) else np.full(len(rp), np.nan)
                emp = np.array([(1 + (null <= p).sum()) / (1 + len(null)) for p in rp]) if len(null) else np.full(len(rp), np.nan)
                v["n_prob50"] = int(np.nansum(prob >= 0.5)) if len(rp) else 0
                if end in fw:
                    order = np.argsort(rp)[:TOP]
                    v["top"] = [{"key": keys[i], "name": per[keys[i]]["name"], "type": per[keys[i]]["type"],
                                 "p_k": {s: float(f"{x:.3g}") for s, x in per[keys[i]]["k"].items()},
                                 "p": float(f"{rp[i]:.3g}"), "p_emp": None if not np.isfinite(emp[i]) else float(f"{emp[i]:.3g}"),
                                 "prob": None if not np.isfinite(prob[i]) else round(float(prob[i]), 3)} for i in order]
                item[var] = v
            seg["windows"].append(item)
        out_segs.append(seg)
    return {"id": g["id"], "name": g["name"], "members": members, "segments": out_segs,
            "single_stock": {"status": "single_stock", "windows": len(single), "from": single[0] if single else None,
                             "to": single[-1] if single else None},
            "known_branches": g.get("known_branches") or []}
