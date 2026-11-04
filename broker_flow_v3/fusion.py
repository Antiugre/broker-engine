"""
§8 融合與集團（v3）
  §8.1 雙通道：p_same = CCT(p_A_same, p_C_same)；p_opp = CCT(p_A_opp, p_B, p_C_opp)；等權、缺的移除；不截斷（G-CAP 移除）
       反向佐證：反向邊另需 p_B ≤ 0.05 或 p_C_opp ≤ 0.05
  §8.2 逐窗口 BH：族群＝一個窗口內全部配對 × 兩個通道
  §8.3 同向核心＝只用同向顯著邊的雙連通分量（大小 ≥ 3）；割點＝橋接分點（不算成員）；大小 2 ＝配對；
       與核心成員有顯著反向邊者＝反向對手；報告成員數、邊密度、割點、反向對手
"""
from __future__ import annotations

import numpy as np

from .params import P
from .pvals import bh, cct


def channels(r: dict) -> tuple[np.ndarray, np.ndarray]:
    nP = len(r["pairs"])
    nan = np.full(nP, np.nan)
    p_same = cct([r["p_A_same"], r.get("p_C_same", nan)])
    p_opp = cct([r["p_A_opp"], r.get("p_B", nan), r.get("p_C_opp", nan)])
    return p_same, p_opp


def window_fdr(r: dict) -> dict:
    """逐窗口 BH（兩個通道一起）＋反向佐證"""
    f = P.fusion
    ps, po = channels(r)
    nP = len(ps)
    q = bh(np.r_[ps, po])
    q_same, q_opp = q[:nP], q[nP:]
    a = float(f.opp_corroboration_alpha)
    nan = np.full(nP, np.nan)
    corr = (np.nan_to_num(r.get("p_B", nan), nan=1) <= a) | (np.nan_to_num(r.get("p_C_opp", nan), nan=1) <= a)
    sig_same = np.nan_to_num(q_same, nan=1) <= float(f.fdr_q)
    sig_opp = (np.nan_to_num(q_opp, nan=1) <= float(f.fdr_q)) & (corr if f.opp_require_corroboration else True)
    return {"p_same": ps, "p_opp": po, "q_same": q_same, "q_opp": q_opp, "sig_same": sig_same, "sig_opp": sig_opp,
            "opp_corroborated": corr}


def biconnected(n_nodes: list[int], edges: list[tuple[int, int]]):
    """Hopcroft–Tarjan（迭代版）：回傳 (雙連通分量的節點集合 list, 割點 set)"""
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        if a == b:
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    disc, low = {}, {}
    comps, arts = [], set()
    timer = [0]
    estack: list[tuple[int, int]] = []
    for root in adj:
        if root in disc:
            continue
        disc[root] = low[root] = timer[0]; timer[0] += 1
        children = 0
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            v, parent, it = stack[-1]
            advanced = False
            for w in it:
                if w not in disc:
                    disc[w] = low[w] = timer[0]; timer[0] += 1
                    estack.append((v, w))
                    stack.append((w, v, iter(adj[w])))
                    if v == root:
                        children += 1
                    advanced = True
                    break
                elif w != parent and disc[w] < disc[v]:
                    low[v] = min(low[v], disc[w])
                    estack.append((v, w))
            if advanced:
                continue
            stack.pop()
            if stack:
                u = stack[-1][0]
                low[u] = min(low[u], low[v])
                if low[v] >= disc[u]:
                    if u != root:
                        arts.add(u)
                    comp = set()
                    while estack:
                        e = estack.pop()
                        comp.update(e)
                        if e == (u, v):
                            break
                    comps.append(comp)
        if children > 1:
            arts.add(root)
    return comps, arts


def cores(n: int, sig_same_edges: list[tuple[int, int]], sig_opp_edges: list[tuple[int, int]]) -> dict:
    comps, arts = biconnected(list(range(n)), sig_same_edges)
    m = int(P.fusion.min_core_size)
    E = set(map(lambda e: (min(e), max(e)), sig_same_edges))
    out_cores, pairs = [], []
    for c in comps:
        if len(c) >= m:
            mem = sorted(x for x in c if x not in arts) or sorted(c)
            bridges = sorted(x for x in c if x in arts)
            k = len(c)
            ne = sum(1 for a, b in E if a in c and b in c)
            out_cores.append({"nodes": sorted(c), "members": mem, "bridges": bridges, "density": ne / (k * (k - 1) / 2)})
        elif len(c) == 2:
            pairs.append(sorted(c))
    for co in out_cores:
        s = set(co["nodes"])
        co["opp_rivals"] = sorted({b if a in s else a for a, b in sig_opp_edges if (a in s) ^ (b in s)})
    out_cores.sort(key=lambda c: (-len(c["nodes"]), c["nodes"]))
    return {"cores": out_cores, "pairs": pairs, "articulation": sorted(arts)}


def group_cost(members, inv_end, cost_end, window_volume: float):
    thr = float(P.fusion.group_inventory_share) * window_volume
    m = [i for i in members if inv_end[i] >= thr and inv_end[i] > 0]
    invG = float(sum(inv_end[i] for i in members))
    if not m:
        return None, invG
    w = np.array([inv_end[i] for i in m])
    return float((w * np.array([cost_end[i] for i in m])).sum() / w.sum()), invG
