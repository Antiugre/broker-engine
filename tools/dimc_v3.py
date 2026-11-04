"""
主力集團 v3 維度 C 分工計算（§7）：每一次擬合（觀測＋20 次虛無，每個擬合週）都是獨立工作，可以分給很多台雲端機器同時算
  python tools/dimc_v3.py plan    --out DIR [--sids ...]   全市場彙總 → DIR/market.pkl、DIR/rank.json、DIR/tasks.json
  python tools/dimc_v3.py fit     --dir DIR --part K --of N   算第 K 份工作（4 核同時）→ DIR/fits/fit_<週>_<b>.pkl
  python tools/dimc_v3.py windows --dir DIR [--only-fit-done]   所有窗口的 p_C（兩通道）→ DIR/windows/dimc_<結束日>.pkl
方法與單機完全相同，只是分工；中間結果含 FinMind 衍生資料，只存私人倉庫
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
log = logging.getLogger("broker_scraper")
_MK = None


def _ends() -> list[str]:
    from broker_flow_v3 import auxdata
    from broker_flow_v3.params import P
    cal = [d for d in (auxdata.trading_calendar() or []) if d >= str(P.data.finmind_start)]
    import datetime as dt
    cal = [d for d in cal if d <= dt.date.today().isoformat()]
    W, step = int(P.detection.window_days), int(P.detection.window_step)
    first = int(P.accounting.warmup_days) + W - 1
    ends = [cal[i] for i in range(first, len(cal), step)]
    return ends


def plan(out: Path, sids: list[str] | None) -> int:
    from broker_flow_v3 import dim_c
    from paths import DATA_DIR
    t0 = time.time()
    sids = sids or sorted(p.parent.name for p in (DATA_DIR / "longhist").glob("*/_prices.json"))
    ends = _ends()
    mk = dim_c.build_market(sids, ends)
    log.info("全市場 %d 檔、%d 個分點、極端日週計數 %d 筆、窗口 %d（%.0f 秒）", len(mk["sids"]), len(mk["keys"]), len(mk["tab"]), len(ends), time.time() - t0)
    from broker_flow_v3.params import P as _P
    fws = dim_c.fit_weeks(mk)
    sizes = {fw: len(dim_c.tensor(mk, fw)[1]) for fw in fws}
    log.info("v3.1 各擬合週張量非零格數：%s", sizes)
    big = {fw: n for fw, n in sizes.items() if n > int(_P.dim_c.max_tensor_nnz)}
    if big:
        log.error("v3.1：張量非零格數超過上限 %d（%s），依規格停止並回報，不自行調整門檻", int(_P.dim_c.max_tensor_nnz), big)
        return 2
    dim_c.save(mk, out / "market.pkl")
    rank_f = out / "rank.json"
    if not rank_f.exists():
        R = dim_c.choose_rank(mk)
        rank_f.write_text(json.dumps({"rank": R, "universe": mk["universe"]}))
    R = json.loads(rank_f.read_text())["rank"]
    need = sorted({dim_c.fit_for(mk, e) for e in ends} - {None})
    from broker_flow_v3.params import P
    tasks = [[fw, b] for fw in need for b in [-1] + list(range(int(P.dim_c.null_cp)))]
    (out / "tasks.json").write_text(json.dumps({"rank": R, "tasks": tasks, "ends": ends}))
    log.info("秩 %d；擬合週 %d 個、工作 %d 件（%.0f 秒）", R, len(need), len(tasks), time.time() - t0)
    return 0


def _job(t):
    from broker_flow_v3 import dim_c
    fw, b, R, d = t
    f = Path(d) / "fits" / f"fit_{fw}_{b}.pkl"
    if f.exists():
        return fw, b, 0.0
    s = time.time()
    res = dim_c.fit_task(_MK, fw, b, R)
    dim_c.save(res, f)
    return fw, b, time.time() - s


def fit(d: Path, part: int, of: int, workers: int) -> int:
    global _MK
    from broker_flow_v3 import dim_c
    _MK = dim_c.load(d / "market.pkl")
    T = json.loads((d / "tasks.json").read_text())
    mine = T["tasks"][part::of]
    log.info("第 %d/%d 份：%d 件工作，秩 %d", part, of, len(mine), T["rank"])
    with ProcessPoolExecutor(workers) as ex:
        for fw, b, sec in ex.map(_job, [(fw, b, T["rank"], str(d)) for fw, b in mine]):
            log.info("擬合 週 %d、b = %d 完成（%.0f 秒）", fw, b, sec)
    return 0


def windows(d: Path) -> int:
    from broker_flow_v3 import dim_c
    from broker_flow_v3.params import P
    mk = dim_c.load(d / "market.pkl")
    T = json.loads((d / "tasks.json").read_text())
    done = 0
    for end in T["ends"]:
        fw = dim_c.fit_for(mk, end)
        fits = {}
        if fw is not None:
            for b in [-1] + list(range(int(P.dim_c.null_cp))):
                f = d / "fits" / f"fit_{fw}_{b}.pkl"
                if f.exists():
                    fits[b] = dim_c.load(f)
            if -1 not in fits or len(fits) < 1 + int(P.dim_c.null_cp):
                log.warning("窗口 %s：擬合週 %s 的擬合還沒齊（%d/%d），略過", end, fw, len(fits), 1 + int(P.dim_c.null_cp))
                continue
        s = time.time()
        res = dim_c.window_c(mk, end, fits)
        if res:
            dim_c.save(res, d / "windows" / f"dimc_{end}.pkl")
            done += 1
        log.info("窗口 %s 完成（%.0f 秒）", end, time.time() - s)
    log.info("維度 C：%d 個窗口完成", done)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan"); a.add_argument("--out", required=True); a.add_argument("--sids", nargs="*")
    b = sub.add_parser("fit"); b.add_argument("--dir", required=True); b.add_argument("--part", type=int, required=True)
    b.add_argument("--of", type=int, required=True); b.add_argument("--workers", type=int, default=4)
    c = sub.add_parser("windows"); c.add_argument("--dir", required=True)
    x = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if x.cmd == "plan":
        return plan(Path(x.out), x.sids)
    if x.cmd == "fit":
        return fit(Path(x.dir), x.part, x.of, x.workers)
    return windows(Path(x.dir))


if __name__ == "__main__":
    sys.exit(main())
