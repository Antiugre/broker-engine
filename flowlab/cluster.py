"""
flowlab.cluster — 模組二：群聚與協同作戰、地緣比對

時間協同（矩陣運算）
  X[d, i]  = 分點 i 在 d 日「大買」（淨買 ≥ 當日量 big_share 且 ≥ min_lots）
  Xe       = X ∨ X(d−1) ∨ X(d+1)                    容許 ±1 日的時間差
  w[d]     = 1 若 d 落在「發動前」窗口（未來 1～5 日內出現 5 日漲幅 ≥ launch_ret 的發動日）
  C        = (X ∘ w)ᵀ · Xe                           i 在發動前大買時，j 在 ±1 日內也大買的次數
  C_sym    = min(C, Cᵀ)                              雙向都成立才算協同
  J        = C_sym / (nᵢ + nⱼ − C_sym)               Jaccard 相似度（nᵢ = 發動前大買次數）
  邊：C_sym ≥ min_co 且 J ≥ min_jaccard → 連通元件（union-find）＝ 主力集團
  沒有發動事件時退回全期間（basis = "all"）

地緣比對（GeoMatcher）
  公司地址 → 郵遞區號（地址開頭的數字，或 zip_table 查表）／縣市／鄉鎮市區
  分點與公司：同郵遞區號 ×1.5、同鄉鎮市區 ×1.3、同縣市 ×1.1，其他 ×1.0
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .core import Analyzer, StockContext, iso, r, register_analyzer

ZIP_RE = re.compile(r"^\s*(\d{3})(?:\d{2,3})?")


class GeoMatcher:
    def __init__(self, company: dict | None, zip_table: dict | None = None,
                 weights: tuple[float, float, float] = (1.5, 1.3, 1.1)):
        self.c = company or {}
        self.zip_table = zip_table or {}
        self.w_zip, self.w_dist, self.w_city = weights

    @staticmethod
    def parse(address: str | None) -> dict:
        """地址 → {zip, city, district}（縣市與區用 broker_location.parse_address）"""
        if not address:
            return {}
        out = {}
        m = ZIP_RE.match(address)
        if m:
            out["zip"] = m.group(1)
            address = address[m.end():]
        try:
            from broker_location import parse_address
            p = parse_address(address) or {}
            out.update({"city": p.get("city"), "district": p.get("district")})
        except Exception:  # noqa: BLE001
            pass
        return out

    def zip_of(self, loc: dict) -> str | None:
        return loc.get("zip") or self.zip_table.get((loc.get("city"), loc.get("district")))

    def weight(self, branch: dict) -> float:
        if not self.c or not branch:
            return 1.0
        cz, bz = self.zip_of(self.c), self.zip_of(branch)
        if cz and bz and cz == bz:
            return self.w_zip
        if self.c.get("city") and branch.get("city") == self.c.get("city"):
            if self.c.get("district") and branch.get("district") == self.c.get("district"):
                return self.w_dist
            return self.w_city
        return 1.0


def big_buy_matrix(ctx: StockContext) -> pd.DataFrame:
    return ((ctx.share >= ctx.p("cluster_big_share", 0.02)) & (ctx.net >= ctx.p("cluster_min_lots", 30))).fillna(False)


def launch_window(ctx: StockContext) -> pd.Series:
    """發動前窗口：未來 1～H 日內，存在某日其後 5 日漲幅 ≥ launch_ret"""
    c = ctx.px["close"]
    fwd = c.shift(-5) / c - 1
    launch = (fwd >= ctx.p("launch_ret", 0.10)).astype(int)
    h = ctx.p("launch_lead", 5)
    # w[d] = 1 若 launch[d+1 … d+h] 任一為 1
    ahead = launch[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)
    return ahead.fillna(0).astype(bool)


def co_buy(X: pd.DataFrame, w: pd.Series, tol: int = 1):
    """回傳 (C_sym, n, J) numpy 陣列"""
    A = X.to_numpy(dtype=np.int32)
    Xe = A.copy()
    for k in range(1, tol + 1):
        Xe[k:] |= A[:-k]
        Xe[:-k] |= A[k:]
    Aw = A * w.to_numpy(dtype=np.int32)[:, None]
    C = Aw.T @ Xe                                  # 分點 × 分點
    Cs = np.minimum(C, C.T)
    n = Aw.sum(axis=0)
    denom = n[:, None] + n[None, :] - Cs
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(denom > 0, Cs / denom, 0.0)
    np.fill_diagonal(Cs, 0)
    np.fill_diagonal(J, 0)
    return Cs, n, J


def components(adj: np.ndarray) -> list[list[int]]:
    parent = list(range(len(adj)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in zip(*np.nonzero(np.triu(adj, 1))):
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(adj)):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


@register_analyzer
class ClusterAnalyzer(Analyzer):
    key, order = "clusters", 20

    def run(self, ctx: StockContext, results: dict):
        X = ctx.in_range(big_buy_matrix(ctx))
        X = X.loc[:, X.any()]
        if X.shape[1] < 2:
            return []
        w = ctx.in_range(launch_window(ctx))
        basis = "pre_launch"
        if w.sum() < ctx.p("cluster_min_launch_days", 5):
            w, basis = pd.Series(True, index=X.index), "all"
        Cs, n, J = co_buy(X, w, ctx.p("cluster_tol_days", 1))
        adj = (Cs >= ctx.p("cluster_min_co", 2)) & (J >= ctx.p("cluster_min_jaccard", 0.3))
        geo = GeoMatcher(ctx.company, ctx.zip_table)
        cols = list(X.columns)
        recent = X.tail(ctx.p("cluster_recent_days", 3)).any()
        out = []
        for g in components(adj):
            sub = J[np.ix_(g, g)]
            pairs = sub[np.triu_indices(len(g), 1)]
            members = []
            for i in g:
                k = cols[i]
                m = ctx.meta.get(k, {})
                members.append({"key": k, "broker": m.get("broker"), "branch_id": m.get("branch_id"), "broker_id": m.get("broker_id"),
                                "b_raw": m.get("b_raw"), "city": m.get("city"), "district": m.get("district"),
                                "big_days": int(n[i]), "active": bool(recent[k]), "geo": geo.weight(m)})
            geo_w = float(np.mean([x["geo"] for x in members]))
            cohesion = float(pairs.mean()) if len(pairs) else 0.0
            co = int(Cs[np.ix_(g, g)].sum() // 2)
            active = sum(x["active"] for x in members)
            out.append({"members": sorted(members, key=lambda x: -x["big_days"]), "size": len(g), "basis": basis,
                        "co_buys": co, "cohesion": r(cohesion, 3), "geo_weight": r(geo_w, 3), "active_now": active,
                        "last_together": _last_together(X, [cols[i] for i in g]),
                        "score": r(cohesion * geo_w * (1 + active / len(g)) * np.log1p(co), 3)})
        out.sort(key=lambda c: -(c["score"] or 0))
        return out[: ctx.p("cluster_max", 8)]


def _last_together(X: pd.DataFrame, keys: list[str]) -> str | None:
    both = X[keys].rolling(3, min_periods=1).max().sum(axis=1) >= 2
    return iso(both[both].index.max()) if both.any() else None
