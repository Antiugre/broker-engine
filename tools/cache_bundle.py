"""
長期資料快取打包／還原（本地建檔 → 上傳給雲端；雲端每天跑完再更新）
  python tools/cache_bundle.py pack   [--out data_cache.tar.gz]   打包 data/ 中雲端需要的部分
  python tools/cache_bundle.py unpack [--file data_cache.tar.gz]  還原到 data/
  python tools/cache_bundle.py upload --repo Antiugre/broker-engine   打包並上傳成 GitHub Release「cache」的附件（需要已登入的 gh）

打包內容（不含任何金鑰）
  longhist/            每檔約 5 年分點明細與價格（最大的一塊）
  universe/            每日運算範圍、Ultra 自選匯總
  最近 70 個交易日的 data/<日期>/<代號>.json   （候選分點與排行頁張數）
  brokers.json、company_info.json               （分點／公司所在地）
  最近一天的 quant/、edge/、flow/ 運算結果          （上傳失敗時可只補傳）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA_DIR  # noqa: E402

KEEP_DAYS = 70


def pack(out: Path) -> Path:
    days = sorted(p for p in DATA_DIR.glob("????-??-??") if p.is_dir())[-KEEP_DAYS:]
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for name in ("longhist", "universe", "brokers.json", "company_info.json"):
            p = DATA_DIR / name
            if p.exists():
                tar.add(p, arcname=name)
        for d in days:
            for f in d.glob("*.json"):
                if not f.name.startswith("_"):
                    tar.add(f, arcname=f"{d.name}/{f.name}")
        # 最近一天的運算結果（Hit Ratio／Expectancy／Order Flow）：上傳失敗時可用 --upload-only 補傳，不必重算
        if days:
            last = days[-1].name
            for rel in (f"quant/{last}.json", f"edge/{last}.json", f"flow/{last}"):
                p = DATA_DIR / rel
                if p.exists():
                    tar.add(p, arcname=rel)
    print(f"打包完成：{out}（{out.stat().st_size / 1e6:.1f} MB，{len(days)} 個交易日）")
    return out


def unpack(file: Path) -> None:
    if not file.exists():
        print(f"找不到 {file}，略過還原（雲端會從頭建檔）")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(file, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter="data")
    print(f"已還原到 {DATA_DIR}")


def upload(repo: str, out: Path) -> None:
    pack(out)
    subprocess.run(["gh", "release", "view", "cache", "--repo", repo], check=False, capture_output=True) \
        .returncode == 0 or subprocess.run(["gh", "release", "create", "cache", "--repo", repo, "--title", "data cache",
                                            "--notes", "長期資料快取（自動更新）", "--prerelease"], check=True)
    subprocess.run(["gh", "release", "upload", "cache", str(out), "--repo", repo, "--clobber"], check=True)
    print("已上傳到 GitHub Release「cache」")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pack", "unpack", "upload"])
    ap.add_argument("--out", default="data_cache.tar.gz")
    ap.add_argument("--file", default="data_cache.tar.gz")
    ap.add_argument("--repo", default="Antiugre/broker-engine")
    a = ap.parse_args()
    {"pack": lambda: pack(Path(a.out)), "unpack": lambda: unpack(Path(a.file)),
     "upload": lambda: upload(a.repo, Path(a.out))}[a.cmd]()
