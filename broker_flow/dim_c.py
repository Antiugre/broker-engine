"""
6.3 維度 C：跨股集團結構（一定要用全市場資料跑，再取出要看的股票）
  修正二（反向配合）：每個分點拆成買方角色 i⁺ 與賣方角色 i⁻，放進同一個張量、同一張圖
  C1 稀疏 Poisson CP-APR：單一張量 X，(2 × 分點數) × 股票 × 週；i⁺ 列＝該週 z_net ≥ 2 的天數，i⁻ 列＝z_net ≤ −2 的天數；
     每 13 週用最近 104 週重擬；秩在第一個訓練區塊選一次；10 組初始值取概似最高，分量需在 ≥ 8 組找到 FMS ≥ 0.8 的對應才保留；
     S_CP = 出現分量中，四種角色組合 (i⁺,i'⁺)(i⁻,i'⁻)(i⁻,i'⁺)(i⁺,i'⁻) 的 u_r[a]·u_r[b] 最大值；記錄 dir_C
  C2 Fraudar：單一二分圖，左＝角色 (i,+)、(i,−)，右＝(股票, 窗口)；(i,+) 連到 (j,w) 若窗口內 z_net 累積 ≥ 3，(i,−) 若 ≤ −3；
     欄權重 1/log(deg+5)；最多 20 個稠密區塊；S_Fr = 四種角色組合中 max_b 1[兩角色都連到 (j,w)]·min(τ_a, τ_b)
  同一分點的兩個角色不配對
  虛無分布（7.1、修正 4.3）：S_CP 循環平移，同一分點的 i⁺、i⁻ 用相同位移量，重擬 20 次；S_Fr 在合併後的單一圖上做 Curveball 50 次
  p_C = CCT(p_CP, p_Fr)

SPEC-GAP
  - 「週」取日曆週（週一～週五）
  - 行為分類（排除當沖、隔日沖）取擬合日之前最近一個窗口結束日的分類
  - 擬合日之後才出現的窗口週數，在擬合結果裡沒有對應（w_r 記 0）
  - S_CP 虛無分數池化範圍＝同一次擬合涵蓋的所有 (股票, 窗口)；S_Fr＝同一個窗口的所有股票
  - Fraudar 不在任何區塊、或兩者沒有同時連到 (j,w) 時 S_Fr = 0；區塊內最先被剝除者 τ = 0
  - dir_C：S_CP 有定義時取 S_CP 的方向，否則取 S_Fr 的方向
  - exclude_foreign：張量與圖不變（外資照常計入），只把外資從配對候選中拿掉
  - CP-APR 以乘法更新（KL-NMF 形式）實作，收斂門檻：相對概似變化 < 1e-4 或 300 次
  - 計算量大：擬合結果按擬合週存檔；張量非零格數變動超過 5% 才重擬；過去窗口的結果算過就存檔，不再重算
"""
from __future__ import annotations

import heapq
import json
import logging
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
from scipy import sparse

from . import features
from .fusion import cct
from .market import Stock
from .nulls import Pool, deciles, stratum
from .params import P
from .store import OUT

log = logging.getLogger("broker_scraper")
CACHE = OUT / "_cache"


# ──────────────────────────── 共用：候選分點、週 ────────────────────────────
def week_index(dates: np.ndarray) -> np.ndarray:
    d = dates.astype("datetime64[D]")
    return ((d - np.datetime64("1970-01-05")).astype(int) // 7)       # 1970-01-05 是週一


def candidates(st: Stock, end: str, exclude_foreign: bool = False) -> tuple[int, int, list[int]]:
    """6.0：(j, w) 的候選分點（未被排除、窗口內活躍日 ≥ 5；exclude_foreign 時再拿掉外資）；回傳 (s, e, 候選索引)"""
    e = st.end_index(end)
    W = int(P.detection.window_days)
    s = e - W + 1
    cls = st.cls.get(end)
    if s < 0 or cls is None:
        return s, e, []
    na = st.active[:, s:e + 1].sum(axis=1)
    return s, e, [i for i in range(len(st.keys)) if cls[i] not in features.EXCLUDED and na[i] >= int(P.detection.min_candidate_active)
                  and not (exclude_foreign and st.btype[i] == "foreign")]


# ──────────────────────────── C1：CP-APR ────────────────────────────
def cp_apr(idx: np.ndarray, val: np.ndarray, shape: tuple, R: int, rng, mask_idx: np.ndarray | None = None,
           max_iter: int = 300, tol: float = 1e-4):
    """稀疏 Poisson CP（乘法更新）。idx: (nnz, 3)；mask_idx：留出（不參與擬合）的座標。回傳 (factors, loglik)"""
    F = [rng.random((n, R)) + 0.1 for n in shape]
    S = [sparse.csr_matrix((np.ones(len(val)), (idx[:, m], np.arange(len(val)))), shape=(shape[m], len(val))) for m in range(3)]
    if mask_idx is not None and len(mask_idx):
        Sm = [sparse.csr_matrix((np.ones(len(mask_idx)), (mask_idx[:, m], np.arange(len(mask_idx)))), shape=(shape[m], len(mask_idx))) for m in range(3)]
    else:
        Sm = None
    prev = -np.inf
    ll = prev
    for it in range(max_iter):
        for m in range(3):
            a, b = [x for x in range(3) if x != m]
            KR = F[a][idx[:, a]] * F[b][idx[:, b]]
            model = np.einsum("nr,nr->n", F[m][idx[:, m]], KR)
            Phi = S[m] @ (KR * (val / np.maximum(model, 1e-300))[:, None])
            den = np.outer(np.ones(shape[m]), F[a].sum(axis=0) * F[b].sum(axis=0))
            if Sm is not None:
                den = den - Sm[m] @ (F[a][mask_idx[:, a]] * F[b][mask_idx[:, b]])
            F[m] = F[m] * Phi / np.maximum(den, 1e-300)
        model = np.einsum("nr,nr,nr->n", F[0][idx[:, 0]], F[1][idx[:, 1]], F[2][idx[:, 2]])
        total = float((F[0].sum(0) * F[1].sum(0) * F[2].sum(0)).sum())
        if Sm is not None:
            total -= float(np.einsum("nr,nr,nr->", F[0][mask_idx[:, 0]], F[1][mask_idx[:, 1]], F[2][mask_idx[:, 2]]))
        ll = float((val * np.log(np.maximum(model, 1e-300))).sum() - total)
        if abs(ll - prev) <= tol * max(1.0, abs(ll)):
            break
        prev = ll
    return F, ll


def normalize(F: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """因子各自 L1 正規化；λ_r = 三個模式欄和的乘積"""
    s = [f.sum(axis=0) for f in F]
    lam = s[0] * s[1] * s[2]
    return lam, [f / np.maximum(sv, 1e-300) for f, sv in zip(F, s)]


def heldout_ll(idx, val, shape, R, rng) -> float:
    """秩選擇：隨機遮蔽 10% 非零格與等量零格，回傳留出 Poisson 對數概似"""
    nnz = len(val)
    k = max(1, int(0.1 * nnz))
    pick = rng.choice(nnz, k, replace=False)
    keep = np.ones(nnz, bool); keep[pick] = False
    occupied = set(map(tuple, idx))
    zeros = []
    while len(zeros) < k:
        c = tuple(int(rng.integers(n)) for n in shape)
        if c not in occupied:
            zeros.append(c)
    mask = np.vstack([idx[pick], np.array(zeros)])
    mval = np.r_[val[pick], np.zeros(len(zeros))]
    F, _ = cp_apr(idx[keep], val[keep], shape, R, rng, mask_idx=mask)
    m = np.einsum("nr,nr,nr->n", F[0][mask[:, 0]], F[1][mask[:, 1]], F[2][mask[:, 2]])
    return float((mval * np.log(np.maximum(m, 1e-300)) - m).sum())


def fms(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    """分量對分量的 factor match score（三個模式餘弦相似度的乘積）"""
    out = np.ones((a[0].shape[1], b[0].shape[1]))
    for x, y in zip(a, b):
        xn = x / np.maximum(np.linalg.norm(x, axis=0), 1e-300)
        yn = y / np.maximum(np.linalg.norm(y, axis=0), 1e-300)
        out *= np.abs(xn.T @ yn)
    return out


def stable_fit(idx, val, shape, R, seed: int):
    """10 組初始值，取概似最高；分量在 ≥ 8 組找到 FMS ≥ 0.8 的對應才保留。回傳 (正規化因子, 保留的分量)"""
    c = P.dim_c
    rng = np.random.default_rng(seed)
    fits = [cp_apr(idx, val, shape, R, rng) for _ in range(int(c.inits))]
    best = max(range(len(fits)), key=lambda k: fits[k][1])
    _, Fb = normalize(fits[best][0])
    cnt = np.zeros(R, int)
    for F, _ in fits:
        cnt += (fms(Fb, normalize(F)[1]).max(axis=1) >= float(c.fms_min))
    return Fb, np.nonzero(cnt >= int(c.stable_min))[0]


def choose_rank(idx, val, shape, seed: int) -> int:
    c = P.dim_c
    rng = np.random.default_rng(seed)
    cand = [r for r in c.rank_candidates if r <= min(shape)] or [min(shape)]
    ll = {r: heldout_ll(idx, val, shape, r, rng) for r in cand}
    log.info("維度 C 秩選擇：%s", {k: round(v) for k, v in ll.items()})
    return max(ll, key=ll.get)


# ──────────────────────────── C2：Fraudar ────────────────────────────
def fraudar_blocks(edges: list[tuple[int, int]], n_left: int, n_right: int, max_blocks: int) -> list[dict]:
    """貪婪剝除，依序取出最多 max_blocks 個稠密區塊；每個區塊回傳 {left: {分點: τ}, right: set}"""
    E = set(edges)
    blocks = []
    for _ in range(max_blocks):
        if not E:
            break
        adjL: dict[int, list] = {}
        adjR: dict[int, list] = {}
        for i, j in E:
            adjL.setdefault(i, []).append(j)
            adjR.setdefault(j, []).append(i)
        wcol = {j: 1.0 / np.log(len(v) + 5) for j, v in adjR.items()}
        # 節點：左 ('L', i)、右 ('R', j)；優先值＝與剩餘節點相連邊權重和
        pri = {("L", i): sum(wcol[j] for j in v) for i, v in adjL.items()}
        pri.update({("R", j): wcol[j] * len(v) for j, v in adjR.items()})
        alive = set(pri)
        f = sum(wcol[j] for _, j in E)
        heap = [(p, k) for k, p in pri.items()]
        heapq.heapify(heap)
        best, best_k, order = f / len(alive), 0, []
        while alive:
            p, k = heapq.heappop(heap)
            if k not in alive or p != pri[k]:
                continue
            alive.discard(k)
            order.append(k)
            f -= pri[k]
            if k[0] == "L":
                for j in adjL[k[1]]:
                    if ("R", j) in alive:
                        pri[("R", j)] -= wcol[j]; heapq.heappush(heap, (pri[("R", j)], ("R", j)))
            else:
                for i in adjR[k[1]]:
                    if ("L", i) in alive:
                        pri[("L", i)] -= wcol[k[1]]; heapq.heappush(heap, (pri[("L", i)], ("L", i)))
            if alive and f / len(alive) > best:
                best, best_k = f / len(alive), len(order)
        members = order[best_k:]
        Ls = [k[1] for k in members if k[0] == "L"]
        Rs = {k[1] for k in members if k[0] == "R"}
        if not Ls or not Rs:
            break
        tau = {i: (r / (len(Ls) - 1) if len(Ls) > 1 else 1.0) for r, i in enumerate(Ls)}
        blocks.append({"left": tau, "right": Rs})
        E = {(i, j) for i, j in E if not (i in tau and j in Rs)}
    return blocks


def curveball(edges: list[tuple[int, int]], rng, rounds: int | None = None) -> list[tuple[int, int]]:
    """Curveball 邊交換：保持每個左、右節點的度數，打亂配對"""
    rows: dict[int, set] = {}
    for i, j in edges:
        rows.setdefault(i, set()).add(j)
    keys = list(rows)
    if len(keys) < 2:
        return list(edges)
    for _ in range(rounds or 5 * len(keys)):
        a, b = rng.choice(len(keys), 2, replace=False)
        A, B = rows[keys[a]], rows[keys[b]]
        ua, ub = list(A - B), list(B - A)
        if not ua and not ub:
            continue
        pool = ua + ub
        rng.shuffle(pool)
        na = len(ua)
        common = A & B
        rows[keys[a]] = common | set(pool[:na])
        rows[keys[b]] = common | set(pool[na:])
    return [(i, j) for i, v in rows.items() for j in v]




ROLE_PAIRS = ((0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))   # (a 的角色, b 的角色, 是否反向)；0 = 買方 +、1 = 賣方 −


def s_fraudar(blocks: list[dict], node: int, cand_g: list[int], connected: set) -> dict:
    """(j,w) 內每對候選分點的 S_Fr：{(a, b): (分數, 是否反向)}；cand_g＝候選分點的全域分點索引"""
    best: dict = {}
    for bl in blocks:
        if node not in bl["right"]:
            continue
        tau = bl["left"]
        for a in range(len(cand_g)):
            for b in range(a + 1, len(cand_g)):
                for ra, rb, opp in ROLE_PAIRS:
                    x, y = 2 * cand_g[a] + ra, 2 * cand_g[b] + rb
                    if x in tau and y in tau and x in connected and y in connected:
                        v = min(tau[x], tau[y])
                        if v > best.get((a, b), (0.0, 0))[0]:
                            best[(a, b)] = (v, opp)
    return best


# ──────────────────────────── 主程式 ────────────────────────────
class Market:
    def __init__(self, stocks: dict[str, Stock], cal: np.ndarray, ends: list[str]):
        self.stocks, self.cal, self.ends = stocks, cal, ends
        self.keys = sorted({k for s in stocks.values() for k in s.keys})
        self.kix = {k: i for i, k in enumerate(self.keys)}
        self.sids = sorted(stocks)
        self.six = {s: i for i, s in enumerate(self.sids)}
        self.week = week_index(cal)
        self.weeks = np.unique(self.week)
        c = P.dim_c
        span, every = int(c.fit_weeks), int(c.refit_weeks)
        self.fit_weeks = [int(self.weeks[k]) for k in range(span - 1, len(self.weeks), every)]
        CACHE.mkdir(parents=True, exist_ok=True)

    def fit_for(self, end: str) -> int | None:
        """不得前視：窗口結束日所在週（含）之前最近一次擬合"""
        w = int(week_index(np.array([end]))[0])
        ok = [f for f in self.fit_weeks if f <= w]
        return ok[-1] if ok else None

    def _cls_at(self, st: Stock, date: str):
        prev = [d for d in self.ends if d <= date and d in st.cls]
        return st.cls[prev[-1]] if prev else None

    def cal_index_week_end(self, fw: int) -> int:
        return int(np.searchsorted(self.week, fw, side="right")) - 1

    def tensor(self, fw: int, shift: np.ndarray | None = None):
        """修正二：單一張量的稀疏座標（角色 2i/2i+1, 股票, 週）。shift：每個分點的循環平移量（兩個角色相同）"""
        span = int(P.dim_c.fit_weeks)
        w0 = fw - span + 1
        thr = float(P.detection.znet_extreme)
        rows = []
        for sid, st in self.stocks.items():
            wk = week_index(st.dates)
            sel = (wk >= w0) & (wk <= fw)
            if not sel.any():
                continue
            cls = self._cls_at(st, str(self.cal[self.cal_index_week_end(fw)]))
            wloc = wk[sel] - w0
            z = st.z[:, sel]
            for i in range(len(st.keys)):
                if cls is not None and cls[i] in features.EXCLUDED:
                    continue
                gi = self.kix[st.keys[i]]
                for role, hit in ((0, z[i] >= thr), (1, z[i] <= -thr)):
                    if not hit.any():
                        continue
                    t, c = np.unique(wloc[hit], return_counts=True)
                    if shift is not None:
                        t = (t + shift[gi]) % span
                    rows.append(np.c_[np.full(len(t), 2 * gi + role), np.full(len(t), self.six[sid]), t, c])
        arr = np.vstack(rows) if rows else np.zeros((0, 4), int)
        return (arr[:, :3].astype(int), arr[:, 3].astype(float)), (2 * len(self.keys), len(self.sids), span)

    def nnz(self, fw: int) -> int:
        return int(len(self.tensor(fw)[0][1]))

    def signature(self, end: str) -> str:
        """窗口結果的指紋＝擬合週（SPEC-GAP：過去窗口的結果以第一次計算時的全市場資料為準，之後不再重算）"""
        return f"roles:fit:{self.fit_for(end)}"

    def rank(self) -> int:
        """修正二：張量結構改變 → 在第一個訓練區塊重新選一次秩"""
        f = CACHE / "dim_c_rank_roles.json"
        if f.exists():
            return int(json.loads(f.read_text())["rank"])
        (idx, val), shape = self.tensor(self.fit_weeks[0])
        r = choose_rank(idx, val, shape, seed=0) if len(val) else 10
        f.write_text(json.dumps({"rank": r, "fit_week": self.fit_weeks[0], "note": "單一張量（買賣角色合併），只在第一個訓練區塊選一次"}))
        return r

    def fit(self, fw: int, shift=None, seed: int = 0):
        (idx, val), shape = self.tensor(fw, shift)
        R = self.rank()
        if len(val) < R:
            return None
        return stable_fit(idx, val, shape, R, seed=seed)

    def s_cp(self, fit, fw: int, sid: str, end: str, cand: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(j,w) 內每對候選分點的 S_CP 與方向（1＝反向）；沒有任何分量在 (j,w) 出現時為 NaN"""
        c = P.dim_c
        n = len(cand)
        iu = np.triu_indices(n, 1)
        if fit is None:
            return np.full(len(iu[0]), np.nan), np.full(len(iu[0]), -1, np.int8)
        (U, V, Wt), keep = fit
        st = self.stocks[sid]
        j = self.six[sid]
        span = int(c.fit_weeks)
        w0 = fw - span + 1
        e = st.end_index(end)
        wk = week_index(st.dates[max(0, e - int(P.detection.window_days) + 1): e + 1])
        wloc = np.unique(wk[(wk >= w0) & (wk <= fw)] - w0)
        gi = np.array([self.kix[st.keys[i]] for i in cand])
        best = np.full((n, n), np.nan)
        opp = np.zeros((n, n), bool)
        for k in keep:
            if V[j, k] < float(c.appear_stock) or Wt[wloc, k].sum() < float(c.appear_weeks):
                continue
            up, un = U[2 * gi, k], U[2 * gi + 1, k]
            same = np.maximum(np.outer(up, up), np.outer(un, un))
            rev = np.maximum(np.outer(un, up), np.outer(up, un))
            val = np.maximum(same, rev)
            better = np.isnan(best) | (val > best)
            best = np.where(better, val, best)
            opp = np.where(better, rev > same, opp)
        d = np.where(np.isnan(best[iu]), -1, opp[iu].astype(np.int8)).astype(np.int8)
        return best[iu], d


_MK = None


def _null_job(job):
    fw, shift, seed = job
    return _MK.fit(fw, shift=shift, seed=seed)


def _parallel(mk: Market, jobs: list) -> list:
    """虛無重擬可以平行（每次重擬各自獨立）；雲端 4 核"""
    global _MK
    _MK = mk
    n = min(len(jobs), os.cpu_count() or 1)
    if n <= 1 or "fork" not in mp.get_all_start_methods():
        return [_null_job(j) for j in jobs]
    with mp.get_context("fork").Pool(n) as pool:
        return pool.map(_null_job, jobs)


def window_c_cached(mk: Market, end: str) -> dict | None:
    """只讀存檔（時間到之後仍可用已算好的窗口）"""
    cf = CACHE / f"dim_c_{end}.pkl"
    if cf.exists():
        cached = pickle.loads(cf.read_bytes())
        if cached.get("sig") == mk.signature(end):
            return cached
    return None


def window_c(mk: Market, end: str, sids: list[str], seed: int = 0) -> dict:
    """某個窗口結束日：全市場每檔 (j,w) 配對的 p_C、dir_C（結果存檔，之後重用）。
    候選＝主規格 6.0 的候選（含外資）；exclude_foreign 版本在使用時把外資配對拿掉"""
    cf = CACHE / f"dim_c_{end}.pkl"
    sig = mk.signature(end)
    if cf.exists():
        cached = pickle.loads(cf.read_bytes())
        if cached.get("sig") == sig and set(sids) <= set(cached.get("detail", [])):
            return cached
    c = P.dim_c
    fw = mk.fit_for(end)
    info = {}
    for sid in mk.sids:
        s, e, cand = candidates(mk.stocks[sid], end)
        if len(cand) >= 2:
            st = mk.stocks[sid]
            info[sid] = (cand, deciles(st.gross[cand, s:e + 1].sum(axis=1)))
    out = {"end": end, "fit_week": fw, "pairs": {}, "sig": sig}
    # ── C1：CP-APR（擬合結果按擬合週快取）──
    cp_obs, cp_pool = {}, None
    if fw is not None:
        ff = CACHE / f"dim_c_fit_{fw}.pkl"
        nnz = mk.nnz(fw)
        fit = None
        if ff.exists():
            fit, nulls_fit, nnz0 = pickle.loads(ff.read_bytes())
            if abs(nnz - nnz0) > float(c.refit_change) * max(nnz0, 1):    # 資料變動太大（新增很多股票或分點）才重擬
                fit = None
        if fit is None:
            t0 = time.time()
            fit = mk.fit(fw, seed=seed)
            rng = np.random.default_rng(seed + 1)
            jobs = [(fw, rng.integers(0, int(c.fit_weeks), len(mk.keys)), seed + 10 + b)   # 同一分點（兩個角色）在所有股票用同一位移量
                    for b in range(int(c.null_cp))]
            nulls_fit = _parallel(mk, jobs)
            for old in CACHE.glob("dim_c_fit_*.pkl"):      # 只留最近用到的擬合（舊窗口的結果已經存檔）
                old.unlink()
            ff.write_bytes(pickle.dumps((fit, nulls_fit, nnz)))
            log.info("維度 C 擬合 %s（含 %d 次虛無重擬）：%.0f 秒", fw, len(jobs), time.time() - t0)
        cp_pool = Pool()
        for sid, (cand, dec) in info.items():
            iu = np.triu_indices(len(cand), 1)
            strata = stratum(dec[iu[0]], dec[iu[1]])
            cp_obs[sid] = mk.s_cp(fit, fw, sid, end, cand)
            for nf in nulls_fit:
                cp_pool.add(strata, mk.s_cp(nf, fw, sid, end, cand)[0])
    # ── C2：Fraudar（買賣角色合併成單一圖）──
    thr = float(c.fraudar_edge)
    edges = []
    for sid in mk.sids:
        st = mk.stocks[sid]
        s, e, _ = candidates(st, end)
        cls = st.cls.get(end)
        if s < 0 or cls is None:
            continue
        cz = st.z[:, s:e + 1].sum(axis=1)
        for i in range(len(st.keys)):
            if cls[i] in features.EXCLUDED:
                continue
            gi = mk.kix[st.keys[i]]
            if cz[i] >= thr:
                edges.append((2 * gi, mk.six[sid]))
            elif cz[i] <= -thr:
                edges.append((2 * gi + 1, mk.six[sid]))
    rng = np.random.default_rng(seed + 2)
    nb = int(c.fraudar_blocks)

    def score(E: list) -> dict:
        blocks = fraudar_blocks(E, 2 * len(mk.keys), len(mk.sids), nb)
        conn: dict[int, set] = {}
        for i, j in E:
            conn.setdefault(j, set()).add(i)
        res = {}
        for sid, (cand, _) in info.items():
            j = mk.six[sid]
            res[sid] = s_fraudar(blocks, j, [mk.kix[mk.stocks[sid].keys[i]] for i in cand], conn.get(j, set()))
        return res

    fr_obs = score(edges)
    fr_pool = Pool()
    for _ in range(int(c.null_fr)):
        tmp = score(curveball(edges, rng))
        for sid, (cand, dec) in info.items():
            iu = np.triu_indices(len(cand), 1)
            v = np.zeros(len(iu[0]))
            pos = {(a, b): n for n, (a, b) in enumerate(zip(*iu))}
            for k, (x, _) in tmp[sid].items():
                v[pos[k]] = x
            fr_pool.add(stratum(dec[iu[0]], dec[iu[1]]), v)
    diag = {}
    for sid, (cand, dec) in info.items():
        iu = np.triu_indices(len(cand), 1)
        strata = stratum(dec[iu[0]], dec[iu[1]])
        pos = {(a, b): n for n, (a, b) in enumerate(zip(*iu))}
        sfr = np.zeros(len(iu[0]))
        dfr = np.full(len(iu[0]), -1, np.int8)
        for k, (x, o) in fr_obs[sid].items():
            sfr[pos[k]] = x
            dfr[pos[k]] = o
        p_fr, d1 = fr_pool.pvalues(strata, sfr)
        if sid in cp_obs:
            scp, dcp = cp_obs[sid]
            p_cp, d2 = cp_pool.pvalues(strata, scp)
            p_cp = np.where(np.isfinite(scp), p_cp, np.nan)
        else:
            scp, dcp, p_cp, d2 = np.full(len(sfr), np.nan), np.full(len(sfr), -1, np.int8), np.full(len(sfr), np.nan), {}
        if sid in sids:
            diag.update({f"{sid}/fr{k}": v for k, v in d1.items()}); diag.update({f"{sid}/cp{k}": v for k, v in d2.items()})
        p_c = cct([p_cp, p_fr], [0.5, 0.5])
        dir_c = np.where(dcp >= 0, dcp, dfr).astype(np.int8)
        # 全部股票存 p_C、dir_C（float32／int8，給假集團用）；指定的股票另存明細
        out["pairs"][sid] = {"keys": [mk.stocks[sid].keys[i] for i in cand], "p_C": np.asarray(p_c, np.float32), "dir_C": dir_c}
        if sid in sids:
            out["pairs"][sid].update({"S_CP": scp, "S_Fr": sfr, "p_CP": p_cp, "p_Fr": p_fr})
    out["gpd"] = diag
    out["detail"] = sorted(sids)
    cf.write_bytes(pickle.dumps(out))
    return out
