"""
§9 驗證（全部只在調參集上跑；§9.1、§9.2 是方法診斷，§9.3 是案例說明、不得據以改參數）
  §9.1 安慰劑：分點在 j 的 ñ_t 不變，配上不同產業 j′ 的特徵與超額報酬（含 j′ 自己的產業項）→ p^E、p^R 應 super-uniform
       F̂(α) ≤ 1.2α + 2√(α(1−α)/N)，α ∈ {0.001, 0.01, 0.05}；依 G_eff（8–11、12–19、≥20）與活躍日數（40–79、80–159、≥160）分層
  §9.2 合成注入（平方根衝擊）：log C^adj_τ += 0.5 · sign(Δ_t) · σ_t · √(|Δ_t| / ADV_t)，τ ≥ t
  §9.3 案例報告：指定（股票、分點名稱）的門檻、三個 p 值、Δ 與區間、ρ、類型、旗標、循環數、h*
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from . import classify, prep, run
from .params import TP, horizons

log = logging.getLogger("broker_scraper")
ALPHAS = (0.001, 0.01, 0.05)
GEFF_STRATA = [(8, 11), (12, 19), (20, 10 ** 9)]
ACT_STRATA = [(40, 79), (80, 159), (160, 10 ** 9)]


# ───────────────────────── §9.1 ─────────────────────────
def bound(alpha: float, N: int) -> float:
    return 1.2 * alpha + 2 * np.sqrt(alpha * (1 - alpha) / max(N, 1))


def super_uniform(p, alphas=ALPHAS) -> dict:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    N = len(p)
    rows = {}
    for a in alphas:
        F = float((p <= a).mean()) if N else np.nan
        rows[str(a)] = {"F": F, "bound": float(bound(a, N)), "ok": bool(N and F <= bound(a, N))}
    return {"N": N, "alphas": rows, "ok": bool(N) and all(r["ok"] for r in rows.values())}


def stratified(p, geff, nact, alphas=ALPHAS) -> dict:
    p, geff, nact = (np.asarray(a, float) for a in (p, geff, nact))
    out = {"all": super_uniform(p, alphas), "G_eff": {}, "active_days": {}}
    for lo, hi in GEFF_STRATA:
        m = (geff >= lo) & (geff <= hi)
        out["G_eff"][f"{lo}-{hi if hi < 10 ** 9 else ''}"] = super_uniform(p[m], alphas)
    for lo, hi in ACT_STRATA:
        m = (nact >= lo) & (nact <= hi)
        out["active_days"][f"{lo}-{hi if hi < 10 ** 9 else ''}"] = super_uniform(p[m], alphas)
    strata = list(out["G_eff"].values()) + list(out["active_days"].values())
    out["ok"] = out["all"]["ok"] and all(s["ok"] for s in strata if s["N"])
    return out


def placebo_stats(stocks: list[str], prep_of: Callable[[str], prep.Prep], industry_of: Callable[[str], str | None],
                  shares_of: Callable[[str], float | None] | None = None, start: str | None = None, end: str | None = None) -> dict[str, dict]:
    """§9.1 配對特徵：sid → {industry, log_mcap, log_value, beta_m}，全部只用調參集期間 [start, end]（預設 2021-06-30～T_cut）。
      log_mcap  = log(已發行股數 × 期間平均收盤)：股數取 smallcap.capital_directory()（同 broker_flow_v3/group.py match_fakes）
      log_value = log(期間平均（收盤 × 成交量）+ 1)（同 match_fakes 的 liq）
      beta_m    = 期間內有效日的日報酬對大盤日報酬的 OLS 斜率
    SPEC-GAP-timing：股數用目前的已發行股數（不回推期間內的股本變動）；缺任何一項 → 該股不列入（不會被選為 j′）"""
    from broker_flow_v3.params import P
    start = str(start or P.data.finmind_start)
    end = str(end or P.governance.t_cut)
    if shares_of is None:
        from smallcap import capital_directory
        cap = capital_directory()

        def shares_of(s):
            return (cap.get(s) or {}).get("shares")
    out = {}
    for s in stocks:
        try:
            pr = prep_of(s)
        except Exception as ex:  # noqa: BLE001
            log.warning("擇時 安慰劑特徵 %s 略過：%s", s, ex)
            continue
        m = (pr.dates >= start) & (pr.dates <= end) & pr.valid
        if m.sum() < 60:
            continue
        c = pr.close[m]
        v = pr.volume[m]
        sh = shares_of(s)
        mc = float(np.log(sh * np.nanmean(c))) if sh and sh > 0 else np.nan
        val = float(np.log(np.nanmean(c * v) + 1))
        y, xm = pr.r[m], pr.rm[m]
        ok = np.isfinite(y) & np.isfinite(xm)
        bm = float(np.polyfit(xm[ok], y[ok], 1)[0]) if ok.sum() >= 60 and np.ptp(xm[ok]) > 0 else np.nan
        out[s] = {"industry": industry_of(s), "log_mcap": mc, "log_value": val, "beta_m": bm}
    return out


def match_placebo(sid: str, stats: dict[str, dict], exclude: set[str] = frozenset()) -> str | None:
    """j′：industry 與 j 不同、不在 H、不在 D（呼叫端放進 exclude）；以 log 市值、log 日均成交值、β^m 標準化後的歐氏距離最近者。
    stats：sid → {"industry", "log_mcap", "log_value", "beta_m"}"""
    me = stats.get(sid)
    if not me:
        return None
    keys = ["log_mcap", "log_value", "beta_m"]
    pool = [s for s, v in stats.items() if s != sid and s not in exclude and v.get("industry") != me.get("industry")
            and all(np.isfinite(v.get(k, np.nan)) for k in keys)]
    if not pool or not all(np.isfinite(me.get(k, np.nan)) for k in keys):
        return None
    M = np.array([[stats[s][k] for k in keys] for s in pool] + [[me[k] for k in keys]])
    mu, sd = M.mean(0), M.std(0)
    Z = (M - mu) / np.where(sd > 0, sd, 1.0)
    d = np.sqrt(((Z[:-1] - Z[-1]) ** 2).sum(1))
    return pool[int(np.argmin(d))]


def placebo_branch(n: np.ndarray, active: np.ndarray, dates: np.ndarray, suspended: np.ndarray, pr_alt: prep.Prep, events,
                   t_cut: str | None = None) -> dict:
    """分點在 j 的交易（ñ 用 j 的股數基準）配上 j′ 的特徵與超額報酬"""
    sa = run.arrays_for(dates, suspended, pr_alt, events, t_cut)
    bt = run.branch_test(n, active, sa)
    s = sa["sample"]
    return {"pE": float(bt["E"]["p"]), "pR": float(bt["R"]["p"]), "G_eff": int(bt["E"]["G_eff"]),
            "active_days": int(active[:sa["T"]].sum()), "N": int(len(s))}


def placebo(results: list[dict], prep_of: Callable[[str], prep.Prep], stats: dict[str, dict], exclude: set[str],
            t_cut: str | None = None) -> dict:
    """results：run.analyze_stock 的輸出（只用 split == tune 的股票）；prep_of(sid) → j′ 的 Prep（日期軸需涵蓋 j 的面板日期）"""
    ps = []
    for res in results:
        if res["split"] != "tune" or not res["rows"]:
            continue
        jp = match_placebo(res["sid"], stats, exclude)
        if jp is None:
            continue
        st = run._load_state(res)
        sub = st["sub"]
        pr = prep_of(jp)
        ev = None
        for r in res["rows"]:
            k = st["keep"].index(r["i"])
            n = sub.B[k] - sub.S[k]
            act = (n != 0) & ~sub.suspended
            # ñ 用 j 的股數基準：arrays_for 的 g 由 j 的公司行動決定（state 裡已有）
            sa = run.arrays_for(sub.dates, sub.suspended, pr, ev, t_cut)
            sa["g"] = st["sa"]["g"]
            bt = run.branch_test(n, act, sa)
            ps.append({"sid": res["sid"], "placebo": jp, "branch": r["branch"], "pE": float(bt["E"]["p"]), "pR": float(bt["R"]["p"]),
                       "G_eff": int(bt["E"]["G_eff"]), "active_days": int(act[:sa["T"]].sum())})
    pE = [x["pE"] for x in ps]
    pR = [x["pR"] for x in ps]
    g = [x["G_eff"] for x in ps]
    na = [x["active_days"] for x in ps]
    a = float(TP.fdr_q)
    return {"n": len(ps), "E": stratified(pE, g, na), "R": stratified(pR, g, na),
            "bh_discoveries_E": int((classify.bh(pE) <= a).sum()) if ps else 0,
            "bh_discoveries_R": int((classify.bh(pR) <= a).sum()) if ps else 0, "rows": ps}


# ───────────────────────── §9.2 ─────────────────────────
def impact(adj: np.ndarray, volume: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """平方根衝擊（係數 impact_coef，其中 impact_permanent_frac 永久）：從 t 日收盤起對所有 τ ≥ t 生效。
    SPEC-GAP-timing：σ_t、ADV_t 用注入前的原始序列計算（不把先前注入的衝擊算進波動度）"""
    adj = np.asarray(adj, float)
    vol = np.asarray(volume, float)
    T = len(adj)
    r = np.r_[np.nan, adj[1:] / adj[:-1] - 1]
    lr = np.log(adj)
    add = np.zeros(T)
    c = float(TP.impact_coef) * float(TP.impact_permanent_frac)
    for t in np.flatnonzero(delta != 0):
        if t < 60:
            continue
        sig = np.nanstd(r[t - 20:t], ddof=1)
        adv = np.nanmean(vol[t - 60:t])
        if not (np.isfinite(sig) and adv > 0):
            continue
        add[t] += c * np.sign(delta[t]) * sig * np.sqrt(abs(delta[t]) / adv)
    return np.exp(lr + np.cumsum(add))


def injection_flow(group: str, rex20: np.ndarray, r: np.ndarray, adv: np.ndarray, phi: float, pi: float, psi: float,
                   rng: np.random.Generator, ok: np.ndarray) -> np.ndarray:
    """注入淨量 Δ_t（股）。|Δ_t| = φ · ADV_t。ok：可注入的日子（有效、有 ADV）。
    SPEC-GAP-timing：純衝擊、先拉後倒的段落起點以「閒置日每天機率 π」產生；逆勢組每天以機率 π 交易"""
    T = len(rex20)
    d = np.zeros(T)
    if group == "timing":
        for t in np.flatnonzero(ok & np.isfinite(rex20)):
            if rng.random() < pi and rex20[t] != 0:
                sg = np.sign(rex20[t]) * (1 if rng.random() < psi else -1)
                d[t] = sg * phi * adv[t]
    elif group in ("impact", "pump_dump"):
        t = 0
        while t < T:
            if ok[t] and rng.random() < pi:
                if group == "impact":
                    L = int(rng.integers(5, 16))
                    sg = 1 if rng.random() < 0.5 else -1
                    seq = [sg] * L
                else:
                    seq = [1] * 5 + [-1] * 10
                for k, sg in enumerate(seq):
                    if t + k < T and ok[t + k]:
                        d[t + k] = sg * phi * adv[t + k]
                t += len(seq)
            else:
                t += 1
    elif group == "contrarian":
        s5 = np.r_[np.full(5, np.nan), np.convolve(np.nan_to_num(r), np.ones(5), "valid")[:-1]]
        for t in np.flatnonzero(ok & np.isfinite(s5)):
            if rng.random() < pi and s5[t] != 0:
                d[t] = -np.sign(s5[t]) * phi * adv[t]
    else:
        raise ValueError(group)
    return d


def inject_once(host: dict, group: str, phi: float, pi: float, psi: float, rng: np.random.Generator, t_cut: str | None = None) -> dict:
    """host：{"pr": Prep（股票層級，含 inputs）、"dates"、"suspended"（面板日期軸）、"n"（宿主分點原始淨量，面板日期軸）、
             "events"、"close"、"volume"、"branch"、"sid"}"""
    pr: prep.Prep = host["pr"]
    inp = pr.inputs
    idx = pr.align(host["dates"])
    ok_idx = idx >= 0
    vol_p = np.asarray(inp["volume"], float)
    adv_p = np.r_[np.full(60, np.nan), np.convolve(vol_p, np.ones(60) / 60, "valid")[:-1]] if len(vol_p) > 60 else np.full(len(vol_p), np.nan)
    hz = horizons()
    rex20 = np.full(len(idx), np.nan); r = np.full(len(idx), np.nan); adv = np.full(len(idx), np.nan)
    rex20[ok_idx] = pr.rex[20][idx[ok_idx]]
    r[ok_idx] = pr.r[idx[ok_idx]]
    adv[ok_idx] = adv_p[idx[ok_idx]]
    te = int(np.searchsorted(host["dates"], str(t_cut or run._t_cut()), side="right")) - 1
    ok = ok_idx & np.isfinite(adv) & (adv > 0) & ~host["suspended"] & (np.arange(len(idx)) <= te)
    d = np.round(injection_flow(group, rex20, r, adv, phi, pi, psi, rng, ok))
    dp = np.zeros(len(pr.dates))
    dp[idx[ok_idx]] = d[ok_idx]
    adj2 = impact(pr.adj, vol_p, dp) if group != "timing" else pr.adj.copy()
    adj2 = np.where(np.isfinite(inp["adj"]), adj2, np.nan)
    pr2 = prep.compute(pr.dates, inp["close"], adj2, inp["volume"], inp["mkt"], inp["peer_adj"], inp["n_peers"])
    n2 = np.asarray(host["n"], float) + d
    act = (n2 != 0) & ~host["suspended"]
    sa = run.arrays_for(host["dates"], host["suspended"], pr2, host["events"], t_cut)
    th = run.thresholds(n2, act, sa, host["close"], host["volume"])
    out = {"group": group, "phi": phi, "pi": pi, "psi": psi, "eligible": th["eligible"], "pE": np.nan, "pR": np.nan,
           "n_injected": int((d != 0).sum())}
    if not th["eligible"]:
        return out
    bt = run.branch_test(n2, act, sa)
    out.update(pE=float(bt["E"]["p"]), pR=float(bt["R"]["p"]), _n=n2, _act=act, _sa=sa,
               _bs={"eps": bt["eps"], "h_star_idx": bt["E"]["h_star_idx"]})
    return out


def inject(host: dict, group: str, phi: float, pi: float, psi: float = 0.6, reps: int = 50, seed: int = 0,
           t_cut: str | None = None, B: int | None = None) -> dict:
    """一組注入設定跑 reps 次。SPEC-GAP-timing：q 值在同一組設定的 reps 個注入資料集之間做 BH"""
    rng = np.random.default_rng(seed)
    res = [inject_once(host, group, phi, pi, psi, rng, t_cut) for _ in range(reps)]
    el = [x for x in res if x["eligible"]]
    qE = classify.bh([x["pE"] for x in el]) if el else []
    qR = classify.bh([x["pR"] for x in el]) if el else []
    a = float(TP.fdr_q)
    kinds = []
    for x, q1, q2 in zip(el, qE, qR):
        x["qE"], x["qR"] = float(q1), float(q2)
        if q1 <= a or q2 <= a:
            r = {"branch": f"{host.get('branch', 'host')}#inj{len(kinds)}", "qE": x["qE"], "qR": x["qR"]}
            ev = run.evaluate(host.get("sid", "inj"), r, x["_n"], x["_act"], x["_sa"], x["_bs"], B=B)
            x["kind"] = ev["kind"]
        else:
            x["kind"] = classify.NONE
        kinds.append(x["kind"])
    for x in res:
        for k in ("_n", "_act", "_sa", "_bs"):
            x.pop(k, None)
    return {"group": group, "phi": phi, "pi": pi, "psi": psi, "reps": reps, "eligible": len(el), "results": res,
            "summary": confusion(group, el)}


def confusion(group: str, rows: list[dict]) -> dict:
    """混淆率的分母只算已顯著者（§9.2）；已顯著樣本 < 20 → 測試無效"""
    a = float(TP.fdr_q)
    mn = int(TP.injection_min_detected)
    mx = float(TP.max_confusion_rate)
    kinds = [x.get("kind", classify.NONE) for x in rows]
    dist = {k: kinds.count(k) for k in classify.TYPES}
    if group in ("impact", "pump_dump"):
        sig = [x for x in rows if x.get("qE") is not None and x["qE"] <= a]
        bad = sum(x.get("kind") == classify.TIMING for x in sig)
    elif group == "contrarian":
        sig = [x for x in rows if x.get("qR") is not None and x["qR"] <= a]
        bad = sum(x.get("kind") in (classify.TIMING, classify.PUMP) for x in sig)
    else:
        sig = [x for x in rows if x.get("qE") is not None and x["qE"] <= a]
        return {"detected": len(sig), "power": len(sig) / len(rows) if rows else None, "distribution": dist}
    n = len(sig)
    rate = bad / n if n else None
    return {"detected": n, "confused": bad, "rate": rate, "distribution": dist, "valid": n >= mn,
            "ok": (n >= mn and rate is not None and rate <= mx) if n >= mn else None, "note": None if n >= mn else "測試無效"}


# ───────────────────────── §9.3 ─────────────────────────
def case_report(sid: str, branch_name: str, res: dict | None = None, B: int | None = None, **kw) -> dict:
    """富邦-二林在 2601 之類的案例：通過哪些門檻、三個 p 值、Δ 與區間、ρ、類型、旗標、循環數、h*。
    res：run.analyze_stock 的結果（含 state）；沒給就現算（開發集：q 在該股票內 BH，標「開發集，非驗證」）"""
    if res is None:
        res = run.analyze_stock(sid, **kw)
        run.assign_q([res])
    st = run._load_state(res)
    out = {"stock": sid, "branch_name": branch_name, "split": res["split"],
           "note": "案例說明，不得據以改參數" + ("；開發集，非驗證" if res["split"] == "dev" else "")}
    rows = [r for r in res["rows"] if branch_name in r["name"]]
    if not rows:
        out["eligible"] = False
        out["reason"] = "沒有通過 §6.1 門檻（或面板中找不到此分點）；請用 thresholds_for 查看各項條件"
        return out
    r = rows[0]
    k = st["keep"].index(r["i"])
    sub = st["sub"]
    n = sub.B[k] - sub.S[k]
    ev = run.evaluate(sid, r, n, (n != 0) & ~sub.suspended, st["sa"], st["branch"][r["i"]], B=B)
    eir = ev["EI"] or {}
    out.update(eligible=True, thresholds=r["thr"], pE=r["pE"], qE=r.get("qE"), pR=r["pR"], qR=r.get("qR"), p_Eplus=float(ev["Eplus"]["p"]),
               delta=eir.get("delta"), delta_ci90=[eir.get("delta_lo"), eir.get("delta_hi")], rho=eir.get("rho"),
               rho_ci90=[eir.get("rho_lo"), eir.get("rho_hi")], kind=ev["kind"], label=classify.LABEL[ev["kind"]],
               flags=classify.flags(ev["kind"], ev["Eplus"]["p"], bool(eir.get("unstable")), r["thr"]["cycles"]),
               cycles=r["thr"]["cycles"], h_star=r["E"]["h_star"])
    return out


def thresholds_for(pn, i: int, pr: prep.Prep, events, t_cut: str | None = None) -> dict:
    """單一分點的 §6.1 各項門檻（案例報告用，不論是否通過）"""
    sa = run.stock_arrays(pn, pr, events, t_cut)
    n = pn.B[i] - pn.S[i]
    return run.thresholds(n, (n != 0) & ~pn.suspended, sa, pn.close, pn.volume)
