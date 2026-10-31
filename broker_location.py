#!/usr/bin/env python3
"""
券商分點所在地（縣市 / 鄉鎮市區）
--------------------------------------------------
資料來源：臺灣證券交易所 OpenAPI
  分公司：https://openapi.twse.com.tw/v1/opendata/OpenData_BRK02   （證券商分公司基本資料）
  總公司：https://openapi.twse.com.tw/v1/brokerService/brokerList  （證券商總公司基本資料）

證交所的地址格式不一致，本模組負責清洗：
  - 舊縣市名改制：台北縣板橋市 → 新北市板橋區、台中縣沙鹿鎮 → 台中市沙鹿區 …
  - 省略縣名：彰化市… → 彰化縣彰化市；簡寫：北市… → 台北市…
  - 缺行政區：以同一條路在其他地址出現的行政區推論（location_source = "road"）
  - 「同上 / 無 / 免設營業廳」：用分點名稱推論（"name"），自營/國際證券等用總公司地址（"hq"）

輸出快取：data/brokers.json（預設 7 天自動更新）

用法：
  python broker_location.py            # 下載/更新目錄，並列印解析統計
  python broker_location.py --enrich   # 幫 data/ 下既有的 JSON 補上地區欄位
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from paths import DATA_DIR  # noqa: E402  （可指向兩台電腦共用的同步資料夾）
CACHE = DATA_DIR / "brokers.json"
BRANCH_URL = "https://openapi.twse.com.tw/v1/opendata/OpenData_BRK02"
HQ_URL = "https://openapi.twse.com.tw/v1/brokerService/brokerList"
CACHE_DAYS = 7

# ──────────────────────────── 行政區基礎資料 ────────────────────────────
MUNICIPALITIES = ["台北市", "新北市", "桃園市", "台中市", "台南市", "高雄市"]  # 直轄市：下轄「區」
CITIES = MUNICIPALITIES + [
    "基隆市", "新竹市", "嘉義市",
    "新竹縣", "苗栗縣", "彰化縣", "南投縣", "雲林縣", "嘉義縣",
    "屏東縣", "宜蘭縣", "花蓮縣", "台東縣", "澎湖縣", "金門縣", "連江縣",
]
OLD_COUNTY = {"台北縣": "新北市", "桃園縣": "桃園市", "台中縣": "台中市", "台南縣": "台南市", "高雄縣": "高雄市"}
SHORT_PREFIX = {"北市": "台北市", "南市": "台南市", "中市": "台中市", "高市": "高雄市"}
# 只寫縣轄市、沒寫縣名的情況
COUNTY_SEAT = {
    "彰化市": "彰化縣", "員林市": "彰化縣", "屏東市": "屏東縣", "宜蘭市": "宜蘭縣", "花蓮市": "花蓮縣",
    "台東市": "台東縣", "苗栗市": "苗栗縣", "頭份市": "苗栗縣", "南投市": "南投縣", "竹北市": "新竹縣",
    "斗六市": "雲林縣", "馬公市": "澎湖縣", "太保市": "嘉義縣", "朴子市": "嘉義縣",
}
TOWN_RENAME = {"員林鎮": "員林市", "頭份鎮": "頭份市"}  # 已升格的縣轄市
PROVINCIAL_DISTRICTS = {
    "基隆市": ["中正區", "七堵區", "暖暖區", "仁愛區", "中山區", "安樂區", "信義區"],
    "新竹市": ["東區", "北區", "香山區"],
    "嘉義市": ["東區", "西區"],
}
CITY_SHORT = {c[:2]: c for c in CITIES}  # 台北→台北市 … （新竹/嘉義 取「市」）
CITY_SHORT.update({"新竹": "新竹市", "嘉義": "嘉義市"})
# 地址沒寫行政區時使用的路段對照表（人工整理，路段完整落在單一行政區者才收錄）
ROAD_DISTRICT = {
    "台北市": {
        "延平南路": "中正區", "衡陽路": "中正區", "南海路": "中正區", "館前路": "中正區", "許昌街": "中正區",
        "重慶南路1段": "中正區", "博愛路": "中正區", "武昌街1段": "中正區", "開封街1段": "中正區",
        "忠孝東路1段": "中正區", "忠孝東路2段": "中正區",
        "塔城街": "大同區", "重慶北路1段": "大同區", "重慶北路2段": "大同區", "重慶北路3段": "大同區",
        "迪化街1段": "大同區", "南京西路": "大同區", "承德路1段": "大同區", "承德路2段": "大同區", "承德路3段": "大同區",
        "明水路": "中山區", "南京東路1段": "中山區", "南京東路2段": "中山區", "松江路": "中山區", "建國北路2段": "中山區",
        "長安東路1段": "中山區", "長安東路2段": "中山區", "中山北路1段": "中山區", "中山北路2段": "中山區", "中山北路3段": "中山區",
        "東興路": "松山區", "光復北路": "松山區", "南京東路4段": "松山區", "南京東路5段": "松山區", "八德路4段": "松山區",
        "敦化北路": "松山區", "民生東路4段": "松山區", "民生東路5段": "松山區",
        "松仁路": "信義區", "松壽路": "信義區", "松德路": "信義區", "松智路": "信義區", "松山路": "信義區",
        "信義路5段": "信義區", "忠孝東路5段": "信義區", "基隆路1段": "信義區",
        "信義路3段": "大安區", "信義路4段": "大安區", "忠孝東路3段": "大安區", "忠孝東路4段": "大安區",
        "敦化南路1段": "大安區", "敦化南路2段": "大安區", "仁愛路3段": "大安區", "仁愛路4段": "大安區",
        "和平東路2段": "大安區", "和平東路3段": "大安區", "羅斯福路3段": "大安區", "安和路": "大安區", "復興南路1段": "大安區",
        "興隆路3段": "文山區", "木柵路": "文山區", "景文街": "文山區", "景中街": "文山區",
        "民權東路6段": "內湖區", "內湖路1段": "內湖區", "成功路4段": "內湖區", "成功路5段": "內湖區",
        "忠誠路1段": "士林區", "中山北路5段": "士林區", "中山北路6段": "士林區", "中山北路7段": "士林區",
        "承德路4段": "士林區", "文林路": "士林區", "天母西路": "士林區",
        "承德路7段": "北投區", "裕民六路": "北投區",
    },
    "台中市": {"民族路": "中區", "三民路3段": "北區", "向上路1段": "西區"},
    "台南市": {"中正路": "中西區", "永福路2段": "中西區", "西門路1段": "中西區", "西門路2段": "中西區",
             "開元路": "北區", "長榮路1段": "東區", "林森路1段": "東區", "林森路2段": "東區", "安中路2段": "安南區"},
    "高雄市": {"大勇路": "鹽埕區", "五福三路": "前金區", "中正二路": "苓雅區", "忠孝一路": "新興區",
             "新光路": "苓雅區", "博愛一路": "三民區"},
    "新竹市": {"光復路1段": "東區", "光復路2段": "東區", "林森路": "東區", "英明街": "東區", "中正路": "東區", "東大路1段": "東區", "關新路": "東區",
             "北大路": "北區", "經國路2段": "北區", "信義街": "北區", "四維路": "北區"},
    "嘉義市": {"國華街": "西區", "光華路": "西區", "興業西路": "西區", "蘭井街": "西區", "文化路": "東區"},
    "基隆市": {"仁一路": "仁愛區", "仁二路": "仁愛區", "仁三路": "仁愛區", "愛四路": "仁愛區",
             "信一路": "信義區", "義一路": "中正區"},
    "桃園市": {"南平路": "桃園區", "成功路1段": "桃園區"},
}
# 分點名稱中的地名（地址欄為「同上 / 無」時使用）
NAME_ALIAS = {
    "天母": ("台北市", "士林區"), "埔墘": ("新北市", "板橋區"), "松江": ("台北市", "中山區"),
    "敦南": ("台北市", "大安區"), "敦北": ("台北市", "松山區"), "雙和": ("新北市", "中和區"),
    "竹科": ("新竹市", "東區"), "南科": ("台南市", "新市區"), "左營": ("高雄市", "左營區"),
}

INVALID_ADDR = re.compile(r"^(同上|無|免設|無設營業廳|不接受|-|—)?$|免設營業廳|無設營業廳")
CN_NUM = str.maketrans("一二三四五六七八九０１２３４５６７８９", "123456789012345678" + "9")


def norm_text(s: str) -> str:
    s = (s or "").replace("臺", "台").replace("巿", "市")
    return re.sub(r"[\s　]+", "", s)


def _strip_district_suffix_rest(rest: str):
    """從直轄市後的字串取出「XX區」，或舊制「XX市/鎮/鄉」轉成「XX區」"""
    m = re.match(r"^([^\d段號巷弄樓之]{1,3}?區)", rest)  # 允許「路竹區」這類含「路」的區名
    if m:
        return m.group(1), rest[m.end():]
    m = re.match(r"^([^\d路街段號巷弄樓之]{1,3}?)(市|鎮|鄉)(?=[^\d])", rest)
    if m and not m.group(1).endswith(("路", "街")):
        return m.group(1) + "區", rest[m.end():]
    return None, rest


def parse_address(addr: str) -> dict | None:
    """回傳 {city, district, road}；無法判斷縣市時回傳 None"""
    s = norm_text(addr)
    if not s or INVALID_ADDR.search(s):
        return None
    for k, v in SHORT_PREFIX.items():
        if s.startswith(k):
            s = v + s[len(k):]
            break
    for k, v in OLD_COUNTY.items():
        if s.startswith(k):
            s = v + s[len(k):]
            break

    city = district = None
    for c in CITIES:
        if s.startswith(c):
            city, rest = c, s[len(c):]
            break
    if city is None:
        for seat, county in COUNTY_SEAT.items():
            if s.startswith(seat):
                city, district, rest = county, seat, s[len(seat):]
                break
    if city is None:
        return None

    if district is None:
        if city in MUNICIPALITIES:
            district, rest = _strip_district_suffix_rest(rest)
        elif city in PROVINCIAL_DISTRICTS:
            for d in PROVINCIAL_DISTRICTS[city]:
                if rest.startswith(d):
                    district, rest = d, rest[len(d):]
                    break
        else:
            m = re.match(r"^([^\d路街段號巷弄樓之]{1,3}?[市鎮鄉])", rest)
            if m:
                district, rest = m.group(1), rest[m.end():]
                district = TOWN_RENAME.get(district, district)

    rest = re.sub(r"^羅斯福(\d|[一二三四五])段", r"羅斯福路\1段", rest)
    road = None
    m = re.match(r"^(?:[^\d]{1,4}?[里村]\d*鄰?)?([^\d]{1,8}?(?:路|街|大道))([\d一二三四五六七八九]段)?", rest)
    if m:
        road = m.group(1) + (m.group(2) or "").translate(CN_NUM)
    return {"city": city, "district": district, "road": road}


# ──────────────────────────── 目錄建置 ────────────────────────────
def download() -> tuple[list, list]:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 broker-scraper", "Accept": "application/json"}
    branches = requests.get(BRANCH_URL, headers=headers, timeout=30).json()
    hqs = requests.get(HQ_URL, headers=headers, timeout=30).json()
    return branches, hqs


def build_directory(branches: list, hqs: list) -> dict:
    raw = {}
    for r in hqs:
        code = (r.get("Code") or "").strip()
        if code:
            raw[code] = {"code": code, "name": norm_text(r.get("Name")), "address": (r.get("Address") or "").strip(), "is_hq": True}
    for r in branches:
        code = (r.get("證券商代號") or "").strip()
        if code:
            name = norm_text(r.get("證券商名稱")).replace("&#29319;", "犇")
            raw[code] = {"code": code, "name": name, "address": (r.get("地址") or "").strip(), "is_hq": False}

    # 1) 直接解析
    for e in raw.values():
        p = parse_address(e["address"])
        e.update(city=p["city"] if p else None, district=p["district"] if p else None,
                 _road=p["road"] if p else None, source="address" if p else None)

    # 2) 缺行政區：先查人工路段表，再用「同縣市、同路段」在其他地址出現的行政區投票
    road_votes: dict[str, Counter] = defaultdict(Counter)
    district_city: dict[str, Counter] = defaultdict(Counter)
    for e in raw.values():
        if e["city"] and e["district"]:
            district_city[e["district"]][e["city"]] += 1
            if e["_road"]:
                road_votes[e["city"] + e["_road"]][e["district"]] += 1

    for e in raw.values():
        if e["city"] and not e["district"] and e["_road"]:
            table = ROAD_DISTRICT.get(e["city"], {})
            road_nosec = re.sub(r"\d段$", "", e["_road"])
            if e["_road"] in table or road_nosec in table:
                e["district"], e["source"] = table.get(e["_road"]) or table[road_nosec], "road"
                continue
            votes = road_votes.get(e["city"] + e["_road"])
            if votes:
                d, n = votes.most_common(1)[0]
                if n / sum(votes.values()) >= 0.6:
                    e["district"], e["source"] = d, "road"

    # 3) 沒有地址：自營/國際證券等用總公司；其餘以分點名稱推論
    hq_of = lambda code: raw.get(code[:3] + "0") or raw.get(code[:2] + "00")  # noqa: E731
    for e in raw.values():
        if e["city"]:
            continue
        suffix = re.split(r"[-－]+", e["name"])[-1]
        suffix = re.sub(r"(簡易|法人|證券股份有限公司|綜合)$", "", suffix)
        hq = hq_of(e["code"])
        if re.search(r"自營|國際證券|網路|數位|總公司", e["name"]) and hq and hq is not e and hq["city"]:
            e.update(city=hq["city"], district=hq["district"], source="hq")
            continue
        if suffix in NAME_ALIAS:
            city, dist = NAME_ALIAS[suffix]
            e.update(city=city, district=dist, source="name")
            continue
        for cand in (suffix + "區", suffix + "市", suffix + "鎮", suffix + "鄉"):
            if cand in district_city:
                e.update(city=district_city[cand].most_common(1)[0][0], district=cand, source="name")
                break
        else:
            if suffix[:2] in CITY_SHORT:
                e.update(city=CITY_SHORT[suffix[:2]], district=None, source="name")
            elif hq and hq is not e and hq["city"]:
                e.update(city=hq["city"], district=None, source="hq")  # 分點地點不明，只取總公司所在縣市

    out = {}
    for code, e in raw.items():
        e.pop("_road", None)
        out[code] = e
    return out


def load_directory(refresh: bool = False, quiet: bool = False) -> dict:
    """讀取快取；過期或不存在時自動下載。下載失敗則沿用舊快取。"""
    fresh = CACHE.exists() and (time.time() - CACHE.stat().st_mtime) < CACHE_DAYS * 86400
    if CACHE.exists() and fresh and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))["brokers"]
    try:
        branches, hqs = download()
        directory = build_directory(branches, hqs)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps({"updated": time.strftime("%Y-%m-%d %H:%M"), "source": [BRANCH_URL, HQ_URL],
                                     "brokers": directory}, ensure_ascii=False, indent=1), encoding="utf-8")
        if not quiet:
            print(f"券商目錄已更新：{len(directory)} 筆 → {CACHE}")
        return directory
    except Exception as ex:  # noqa: BLE001
        if CACHE.exists():
            if not quiet:
                print(f"券商目錄下載失敗（{ex}），沿用舊快取", file=sys.stderr)
            return json.loads(CACHE.read_text(encoding="utf-8"))["brokers"]
        print(f"券商目錄下載失敗且無快取：{ex}", file=sys.stderr)
        return {}


# ──────────────────────────── 套用到紀錄 ────────────────────────────
def _name_key(name: str) -> str:
    s = norm_text(name)
    s = re.sub(r"^\(.*?\)", "", s)               # (牛牛牛)亞-鑫豐 → 亞-鑫豐
    s = re.sub(r"[-－]", "", s)
    return re.sub(r"(證券|綜合)$", "", s)


class Locator:
    def __init__(self, directory: dict):
        self.by_code = directory
        self.by_lower = {k.lower(): v for k, v in directory.items()}
        self.by_name = {}
        for v in directory.values():
            self.by_name.setdefault(_name_key(v["name"]), v)

    def lookup(self, branch_id: str | None, broker: str | None) -> dict | None:
        if branch_id:
            e = self.by_code.get(branch_id) or self.by_lower.get(branch_id.lower())
            if e:
                return e
        if broker:
            key = _name_key(broker)
            e = self.by_name.get(key)
            if e:
                return e
            # 「亞-鑫豐」對「犇亞-鑫豐」這類前綴差異
            hits = [v for k, v in self.by_name.items() if k.endswith(key) and len(key) >= 3]
            if len(hits) == 1:
                return hits[0]
        return None

    def enrich_row(self, row: dict) -> dict:
        e = self.lookup(row.get("branch_id"), row.get("broker"))
        row["city"] = e["city"] if e else None
        row["district"] = e["district"] if e else None
        row["address"] = (e["address"] or None) if e else None
        row["location_source"] = e["source"] if e else None
        return row

    def enrich_record(self, rec: dict) -> dict:
        for side in ("buy_side", "sell_side"):
            for row in rec.get(side, []):
                self.enrich_row(row)
        rec["by_region"] = region_summary(rec)
        return rec


def region_summary(rec: dict) -> list[dict]:
    """依 縣市 / 鄉鎮市區 加總淨買賣張數（僅涵蓋本頁列出的分點）"""
    agg: dict[tuple, dict] = {}
    for side in ("buy_side", "sell_side"):
        for r in rec.get(side, []):
            key = (r.get("city") or "未知", r.get("district") or "未標示")
            a = agg.setdefault(key, {"city": key[0], "district": key[1], "net": 0, "buy": 0, "sell": 0, "brokers": 0})
            a["net"] += r.get("net") or 0
            a["buy"] += r.get("buy") or 0
            a["sell"] += r.get("sell") or 0
            a["brokers"] += 1
    return sorted(agg.values(), key=lambda x: -abs(x["net"]))


def enrich_existing(locator: Locator) -> int:
    n = 0
    for f in DATA_DIR.glob("????-??-??/*.json"):
        data = json.loads(f.read_text(encoding="utf-8"))
        recs = data if isinstance(data, list) else [data]
        for r in recs:
            locator.enrich_record(r)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="券商分點所在地目錄")
    ap.add_argument("--refresh", action="store_true", help="強制重新下載")
    ap.add_argument("--enrich", action="store_true", help="幫 data/ 下既有 JSON 補上地區欄位")
    args = ap.parse_args(argv)

    directory = load_directory(refresh=args.refresh or not args.enrich)
    if not directory:
        return 1
    src = Counter(v["source"] for v in directory.values())
    no_dist = [v for v in directory.values() if v["city"] and not v["district"]]
    unknown = [v for v in directory.values() if not v["city"]]
    print(f"共 {len(directory)} 筆｜地址直接解析 {src['address']}、路名推論 {src['road']}、"
          f"名稱推論 {src['name']}、沿用總公司 {src['hq']}｜只有縣市 {len(no_dist)}、未知 {len(unknown)}")
    for v in (unknown + no_dist)[:60]:
        print(f"  {v['code']:5} {v['name']:<12} {v['city'] or '-':<4} {v['address']}")
    if args.enrich:
        print(f"已補上地區欄位：{enrich_existing(Locator(directory))} 個檔案")
    return 0


if __name__ == "__main__":
    sys.exit(main())
