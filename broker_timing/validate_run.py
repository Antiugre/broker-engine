"""
§9 真實資料驗證的執行器（方法診斷；結果只寫 Firestore syndicate/_validation@timing（管理員限定），
log 只印彙總數字，不印個別分點）
  §9.1 安慰劑：調參集每個通過門檻的「股票 × 分點」配上不同產業的 j′（不在 H、不在 D）→ p^E、p^R 分層 super-uniform
  §9.2 合成注入：宿主＝調參集 p^E > 0.5 的「股票 × 分點」；四組 × 參數格；同一組設定的所有宿主 × 重複合併後做 BH
  §9.3 案例：富邦-二林 @ 2601（開發集，非驗證）
SPEC-GAP-timing：
  - 注入宿主最多 injection_hosts 個（固定種子抽樣），每個宿主每組設定 injection_reps 次；合併成一組再算混淆率
  - 已顯著 < 20 且 φ = 0.02 → 改用 φ = 0.05 的結果判斷（§9.2「改用 φ = 0.05 加跑」；φ = 0.05 本來就在參數格內）
  - EI bootstrap 次數用呼叫端給的 B（驗證用，預設 199），只影響分類，不影響 p 值
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import classify, prep, run, validate
from .params import TP

log = logging.getLogger("broker_scraper")
GROUPS = ("timing", "impact", "pump_dump", "contrarian")
_CTX: dict = {}


def _end(results: list[dict]) -> str:
    return max(run._load_state(r)["data"]["end"] for r in results)


def _prep_of(sid: str) -> prep.Prep:
    if "industry" not in _CTX:
        _CTX["industry"] = prep.Industry(run.universe())
    return prep.stock_prep(sid, _CTX["end"], _CTX["industry"], None, run.out_dir() / "_cache")


def placebo_part(results: list[dict]) -> dict:
    from broker_flow_v3 import split
    from broker_flow_v3.params import P
    stocks = [r["sid"] for r in results]
    ind = _CTX.setdefault("industry", prep.Industry(run.universe()))
    stats = validate.placebo_stats(stocks, _prep_of, ind.category)
    exclude = set(P.governance.dev_stocks) | set(split.holdout().get("stocks") or [])
    out = validate.placebo(results, _prep_of, stats, exclude)
    out.pop("rows", None)                                      # 個別分點不輸出
    return out


def _hosts(results: list[dict], k: int, seed: int) -> list[dict]:
    from broker_flow_v3.accounting import corp_actions
    cand = [(res, r) for res in results if res["split"] == "tune" for r in res["rows"] if r["pE"] > 0.5]
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(cand), min(k, len(cand)), replace=False).tolist()) if cand else []
    hosts = []
    for i in pick:
        res, r = cand[i]
        st = run._load_state(res)
        sub = st["sub"]
        j = st["keep"].index(r["i"])
        hosts.append({"sid": res["sid"], "branch": r["branch"], "pr": _prep_of(res["sid"]), "dates": sub.dates,
                      "suspended": sub.suspended, "n": sub.B[j] - sub.S[j], "events": corp_actions(res["sid"]),
                      "close": sub.close, "volume": sub.volume})
    return hosts


def _one_setting(args) -> dict:
    group, phi, pi, psi, reps, seed, B = args
    hosts = _CTX["hosts"]
    rng = np.random.default_rng(seed)
    rows = []
    for h in hosts:
        for _ in range(reps):
            x = validate.inject_once(h, group, phi, pi, psi, rng)
            x["_host"] = h
            rows.append(x)
    el = [x for x in rows if x["eligible"]]
    a = float(TP.fdr_q)
    if el:
        qE = classify.bh([x["pE"] for x in el])
        qR = classify.bh([x["pR"] for x in el])
        for i, (x, q1, q2) in enumerate(zip(el, qE, qR)):
            x["qE"], x["qR"] = float(q1), float(q2)
            if q1 <= a or q2 <= a:
                r = {"branch": f"{x['_host']['branch']}#inj{i}", "qE": x["qE"], "qR": x["qR"]}
                x["kind"] = run.evaluate(x["_host"]["sid"], r, x["_n"], x["_act"], x["_sa"], x["_bs"], B=B)["kind"]
            else:
                x["kind"] = classify.NONE
    summ = validate.confusion(group, el)
    return {"group": group, "phi": phi, "pi": pi, "psi": psi, "runs": len(rows), "eligible": len(el), "summary": summ}


def _init(ctx):
    _CTX.update(ctx)


def injection_part(results: list[dict], hosts_n: int, reps: int, B: int, processes: int, seed: int = 20261105) -> dict:
    hosts = _hosts(results, hosts_n, seed)
    if not hosts:
        return {"note": "沒有 p^E > 0.5 的宿主", "settings": []}
    grid = []
    k = 0
    for g in GROUPS:
        for phi in TP.injection_phi:
            for pi in TP.injection_pi:
                for psi in (TP.injection_psi if g == "timing" else [None]):
                    grid.append((g, float(phi), float(pi), None if psi is None else float(psi), reps, seed + k, B))
                    k += 1
    ctx = {"hosts": hosts, "end": _CTX["end"]}
    if processes > 1:
        with ProcessPoolExecutor(processes, initializer=_init, initargs=(ctx,)) as ex:
            settings = list(ex.map(_one_setting, grid))
    else:
        _init(ctx)
        settings = [_one_setting(a) for a in grid]
    verdict = {}
    for g in GROUPS:
        if g == "timing":
            continue
        rs = [s for s in settings if s["group"] == g]
        ok = []
        for pi in TP.injection_pi:
            s02 = next((s for s in rs if s["phi"] == min(TP.injection_phi) and s["pi"] == pi), None)
            s05 = next((s for s in rs if s["phi"] == max(TP.injection_phi) and s["pi"] == pi), None)
            use = s02 if (s02 and s02["summary"].get("valid")) else s05
            ok.append(None if use is None or not use["summary"].get("valid") else bool(use["summary"].get("ok")))
        verdict[g] = {"ok": all(o for o in ok if o is not None) and any(o is not None for o in ok),
                      "invalid_settings": sum(o is None for o in ok)}
    return {"hosts": len(hosts), "reps_per_host": reps, "B": B, "settings": settings, "verdict": verdict}


def case_part(results: list[dict], B: int) -> dict:
    res = next((r for r in results if r["sid"] == "2601"), None)
    if res is None:
        return {"note": "2601 不在這次計算的股票中"}
    return validate.case_report("2601", "二林", res=res, B=B)


def run_validation(results: list[dict], processes: int = 4, hosts: int = 40, reps: int = 5, B: int = 199) -> dict:
    t0 = time.time()
    _CTX["end"] = _end(results)
    out = {"version": run.VERSION, "timing_version": str(TP.timing_version), "asof": _CTX["end"]}
    log.info("擇時驗證 §9.1 安慰劑…")
    out["placebo"] = placebo_part(results)
    pl = out["placebo"]
    log.info("擇時驗證 §9.1：N = %d；E 全部分層通過 = %s；R = %s；BH 發現數 E/R = %d/%d", pl["n"], pl["E"]["ok"], pl["R"]["ok"],
             pl["bh_discoveries_E"], pl["bh_discoveries_R"])
    for dim in ("G_eff", "active_days"):
        for k, s in pl["E"][dim].items():
            if s["N"]:
                log.info("  E %s %s：N = %d，F̂ = %s，%s", dim, k, s["N"],
                         " / ".join(f"{s['alphas'][a]['F']:.4f}" for a in sorted(s["alphas"], key=float)), "ok" if s["ok"] else "FAIL")
    log.info("擇時驗證 §9.2 合成注入…")
    out["injection"] = injection_part(results, hosts, reps, B, processes)
    for g, v in (out["injection"].get("verdict") or {}).items():
        log.info("擇時驗證 §9.2 %s：%s（測試無效的設定 %d 個）", g, "通過" if v["ok"] else "未通過", v["invalid_settings"])
    for s in out["injection"].get("settings", []):
        sm = s["summary"]
        log.info("  %s φ=%.2f π=%.1f ψ=%s：可檢定 %d、已顯著 %d、%s", s["group"], s["phi"], s["pi"], s["psi"], s["eligible"],
                 sm.get("detected", 0), f"混淆率 {sm['rate']:.3f}" if sm.get("rate") is not None else
                 (f"檢定力 {sm['power']:.3f}" if sm.get("power") is not None else "—"))
    out["case"] = case_part(results, B)
    out["seconds"] = round(time.time() - t0)
    out["ok"] = bool(pl["E"]["ok"] and pl["R"]["ok"] and all(v["ok"] for v in (out["injection"].get("verdict") or {}).values()))
    log.info("擇時驗證完成：%s（%d 秒）", "全部通過" if out["ok"] else "未全部通過", out["seconds"])
    return out
