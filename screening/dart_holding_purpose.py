"""
5% 대량보유상황보고서 -> 보유목적/지분율 추출 (로컬 실행용)

경영권분쟁 스크리닝의 '보유목적 확정' 단계. d001.csv 의 종목을 DART 고유번호로
매핑해 대량보유 상황보고(majorstock)를 받아 detail_purpose.csv 로 저장한다.
이후 screen_control_disputes.py 를 재실행하면 rcp(접수번호) 기준으로 자동 join 되어
경영참여/경영권 문구가 확정 스코어(+5)에 반영된다.

두 가지 경로
------------
  * USE_API=True  (권장) : OpenDART API. 키 하나로 corp_code 매핑까지 자동.
                          접수번호/보고사유/보유비율/직전비율을 한 번에 받음.
  * USE_API=False        : 웹 스크래핑. 키 없을 때. 본문의 '보유목적'(경영참여/
                          단순투자) 분류 텍스트까지 뽑지만 느리고 스로틀링 위험.

주의
----
* 키는 코드에 하드코딩 금지. 이 폴더 .env 에 DART_API_KEY=... (이미 .gitignore 처리).
* 웹 컨테이너에서는 dart.fss.or.kr / opendart.fss.or.kr 모두 egress 정책으로 차단됨.
  -> DART 가 열려 있는 로컬에서 실행할 것.
"""
import os
import re
import time
import zipfile
import io
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_env():
    """.env 를 읽어 os.environ 에 주입 (python-dotenv 없이도 동작)."""
    p = os.path.join(HERE, ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

USE_API = True                                 # 키 있으면 API 경로 권장
KEY = os.environ.get("DART_API_KEY")           # 코드에 하드코딩 금지
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://dart.fss.or.kr/"}
S = requests.Session()
S.headers.update(UA)


# ---------- corp_code 매핑 (API 경로) ----------
def build_corp_map() -> dict:
    """OpenDART corpCode.xml 전체를 받아 {회사명: corp_code} 로. corp_map.csv 캐시."""
    cache = os.path.join(HERE, "corp_map.csv")
    if os.path.exists(cache):
        m = pd.read_csv(cache, dtype=str).fillna("")
        return dict(zip(m["corp_name"], m["corp_code"]))
    r = S.get("https://opendart.fss.or.kr/api/corpCode.xml",
              params={"crtfc_key": KEY}, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    root = ET.fromstring(z.read(z.namelist()[0]))
    rows = []
    for it in root.iter("list"):
        name = (it.findtext("corp_name") or "").strip()
        code = (it.findtext("corp_code") or "").strip()
        stock = (it.findtext("stock_code") or "").strip()
        if name and code:
            rows.append({"corp_name": name, "corp_code": code, "stock_code": stock})
    df = pd.DataFrame(rows)
    # 상장사(종목코드 있음) 우선 - 동명이인 비상장 제거
    df = df.sort_values("stock_code", ascending=False).drop_duplicates("corp_name")
    df.to_csv(cache, index=False)
    return dict(zip(df["corp_name"], df["corp_code"]))


def norm_name(name: str) -> str:
    """d001.csv 종목명 정리 ('케이카 IR' 등 UI 잔여 ' IR' 접미어 제거)."""
    return re.sub(r"\s+IR$", "", str(name)).strip()


def via_api(corp_code: str, company: str):
    """majorstock: 대량보유 상황보고. 접수번호(rcp)까지 반환 -> rcp 기준 join 가능."""
    r = S.get("https://opendart.fss.or.kr/api/majorstock.json",
              params={"crtfc_key": KEY, "corp_code": corp_code}, timeout=20).json()
    if r.get("status") != "000":
        return []
    out = []
    for x in r.get("list", []):
        out.append({
            "rcp": x.get("rcept_no", ""),
            "company": company,
            "filer": x.get("repror", ""),
            "date": x.get("rcept_dt", ""),
            "보유비율": x.get("stkrt", ""),
            "직전비율": x.get("stkrt_irds", ""),
            "보고사유": (x.get("report_resn", "") or "")[:120],
            "보고구분": x.get("report_tp", ""),
            "보유목적": "",   # majorstock 요약 API 에는 경영참여 분류 텍스트 없음 -> 보고사유로 판별
        })
    return out


def run_api():
    if not KEY:
        raise SystemExit("DART_API_KEY 없음 (.env 확인)")
    df = pd.read_csv(os.path.join(HERE, "d001.csv"), dtype=str).fillna("")
    companies = sorted({norm_name(c) for c in df["company"]})
    cmap = build_corp_map()
    hits = [(c, cmap[c]) for c in companies if c in cmap]
    miss = [c for c in companies if c not in cmap]
    print(f"종목 {len(companies)} / 매핑 {len(hits)} / 미매핑 {len(miss)}")
    if miss:
        print("미매핑(수기확인):", ", ".join(miss[:30]), "..." if len(miss) > 30 else "")

    def fetch(pair):
        comp, code = pair
        time.sleep(0.05)
        try:
            return via_api(code, comp)
        except Exception as e:
            return [{"rcp": "", "company": comp, "err": str(e)[:60]}]

    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(fetch, hits):
            rows.extend(r)
    return pd.DataFrame(rows)


# ---------- 경로 2: 웹 스크래핑 (키 없을 때) ----------
KEYS = ("text", "rcpNo", "dcmNo", "eleId", "offset", "length", "dtd")


def first_node(html: str):
    toks = re.findall(r"node\d\['(\w+)'\]\s*=\s*\"([^\"]*)\"", html)
    blk = {}
    for k, v in toks:
        if k in KEYS and k not in blk:
            blk[k] = v
        if len(blk) == len(KEYS):
            break
    return blk if "dcmNo" in blk else None


def via_web(rcp: str, sleep=0.4):
    """dtd 는 문서마다 dart3/dart4 로 다름 - 하드코딩하면 절반이 빈다."""
    time.sleep(sleep)
    m = S.get(f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp}", timeout=25).text
    b = first_node(m)
    if not b:
        return {"rcp": rcp, "err": "NONODE"}
    u = ("https://dart.fss.or.kr/report/viewer.do"
         f"?rcpNo={rcp}&dcmNo={b['dcmNo']}&eleId={b.get('eleId', 0)}"
         f"&offset={b.get('offset', 0)}&length={b.get('length', 0)}&dtd={b.get('dtd', 'dart3.xsd')}")
    t = BeautifulSoup(S.get(u, timeout=25).text, "lxml").get_text("|", strip=True)
    g = lambda p: (re.search(p, t).group(1).strip() if re.search(p, t) else "")
    prev = re.search(r"직전 보고서\|([\d,\-]+)\|([\d.\-]+)", t)
    cur = re.search(r"이번 보고서\|([\d,\-]+)\|([\d.\-]+)", t)
    return {"rcp": rcp,
            "보유목적": g(r"보유목적\|([^|]+)"),
            "보고사유": g(r"보고사유\|([^|]+)")[:120],
            "보고구분": g(r"보고구분\|([^|]+)"),
            "직전비율": prev.group(2) if prev else "",
            "금번비율": cur.group(2) if cur else "",
            "발행주식총수": g(r"의결권있는 발행주식 총수\(주\)\|([\d,]+)")}


def run_web():
    df = pd.read_csv(os.path.join(HERE, "d001.csv"), dtype={"rcp": str}).fillna("")
    tgt = df[df.report.str.contains("일반")]["rcp"].tolist()   # 일반서식 = 경영권 후보군
    print(len(tgt), "건 (웹 스크래핑, 저속)")
    with ThreadPoolExecutor(max_workers=4) as ex:              # 스로틀링 회피용 저동시성
        return pd.DataFrame(list(ex.map(via_web, tgt)))


if __name__ == "__main__":
    d = run_api() if USE_API else run_web()
    out = os.path.join(HERE, "detail_purpose.csv")
    d.to_csv(out, index=False)
    # 경영권 영향 목적으로 보이는 건 (보유목적/보고사유 문구 기준)
    txt = d.get("보유목적", pd.Series("", index=d.index)).fillna("") + " " + \
          d.get("보고사유", pd.Series("", index=d.index)).fillna("")
    hit = d[txt.str.contains("경영")]
    print(f"경영권/경영참여 문구 감지: {len(hit)}건")
    print(hit.head(50).to_string())
    print(f"\n-> {out} 저장. 이제 screen_control_disputes.py 재실행.")
