"""
5% 대량보유상황보고서 -> 보유목적/지분율 추출 (로컬 실행용)

경영권분쟁 스크리닝의 '보유목적 확정' 단계. d001.csv 의 '일반' 보고 rcp 를 돌며
보고서 본문에서 보유목적/보고사유/지분율을 뽑아 detail_purpose.csv 로 저장한다.
이후 screen_control_disputes.py 를 재실행하면 rcp 기준으로 자동 join 되어
경영참여/경영권 문구가 확정 스코어에 반영된다.

주의
----
* 이 파일이 도는 환경(웹 컨테이너)에서는 DART 웹 조회가 egress 정책으로 차단되고,
  대량 조회는 스로틀링에 걸린다. -> DART 는 로컬에서 실행할 것.
* OpenDART API 키를 쓰는 경로(USE_API=True)가 훨씬 안정적. corp_code 매핑 필요.
* 키는 코드에 하드코딩 금지. .env 에 DART_API_KEY=... 로 두고 os.environ 으로 읽는다.
  .env 가 .gitignore 에 있는지 먼저 확인 (이 폴더 .gitignore 에 포함되어 있음).
"""
import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))

USE_API = False                                # API 키+corp_code 준비되면 True 권장
KEY = os.environ.get("DART_API_KEY")           # 코드에 하드코딩 금지
UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://dart.fss.or.kr/"}
S = requests.Session()
S.headers.update(UA)


# ---------- 경로 1: OpenDART API (권장) ----------
def via_api(corp_code: str):
    """majorstock: 대량보유 상황보고. corp_code 는 DART 고유번호(8자리)."""
    r = S.get("https://opendart.fss.or.kr/api/majorstock.json",
              params={"crtfc_key": KEY, "corp_code": corp_code}, timeout=20).json()
    if r.get("status") != "000":
        return []
    return [{"corp": x["corp_name"], "보고자": x["repror"], "보고일": x["rcept_dt"],
             "보유비율": x["stkrt"], "직전비율": x.get("stkrt_irds"),
             "보유목적": x.get("report_resn")} for x in r.get("list", [])]


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
            "보고사유": g(r"보고사유\|([^|]+)")[:70],
            "보고구분": g(r"보고구분\|([^|]+)"),
            "직전비율": prev.group(2) if prev else "",
            "금번비율": cur.group(2) if cur else "",
            "발행주식총수": g(r"의결권있는 발행주식 총수\(주\)\|([\d,]+)")}


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(HERE, "d001.csv"), dtype={"rcp": str}).fillna("")
    tgt = df[df.report.str.contains("일반")]["rcp"].tolist()   # 일반서식 = 경영권 후보군
    print(len(tgt), "건")
    with ThreadPoolExecutor(max_workers=4) as ex:              # 스로틀링 회피용 저동시성
        out = list(ex.map(via_web, tgt))
    d = pd.DataFrame(out)
    d.to_csv(os.path.join(HERE, "detail_purpose.csv"), index=False)
    # 최종 스크린: 경영권 영향 목적
    hit = d[d.get("보유목적", pd.Series(dtype=str)).fillna("").str.contains("경영")]
    print(hit.head(50).to_string())
    print("\n-> detail_purpose.csv 저장. 이제 screen_control_disputes.py 재실행.")
