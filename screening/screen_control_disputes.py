"""
경영권분쟁(경영권 영향 목적) 스크리닝
====================================

입력: DART 공시통합검색 D001 '주식등의대량보유상황보고서' 전수 (d001.csv)
      컬럼 = market, company, report, filer, date, rcp

핵심 아이디어
-------------
5% 대량보유보고는 보고서식 자체가 1차 필터가 된다.
  * '약식' = 전문투자자/특정 기관의 단순투자·일반투자 목적  -> 경영권과 무관, 제외
  * '일반' = 그 외 전부. '경영권에 영향을 주기 위한 목적'인 경우 반드시 일반서식.
따라서 '일반' 서식만 남기면 경영권 관련 후보군으로 좁혀진다.

한 종목에 서로 다른 보고자가 여러 명 '일반' 보고를 내면 = 블록이 경합하는 형태
(= 전형적인 경영권 분쟁/행동주의 구도). 이 구조 신호로 후보를 스코어링한다.

한계 (반드시 로컬에서 보완할 것)
--------------------------------
서식(일반/약식)은 '보유목적'의 프록시일 뿐 확정값이 아니다. 확정 판정에는
각 보고서 본문의 '보유목적'(경영참여 / 단순투자 / 일반투자) 텍스트가 필요하다.
이 컨테이너에서는 dart.fss.or.kr 이 egress 정책상 차단되어 본문을 못 긁는다.
-> dart_holding_purpose.py 를 DART API 키로 로컬 실행해 detail_purpose.csv 를
   만든 뒤, 이 스크립트가 자동으로 join 하여 '보유목적' 확정 컬럼을 채운다.
   (detail_purpose.csv 가 없으면 서식 기반 프록시 스코어만 출력)

또한 특수관계인은 각자 별도로 대량보유보고를 내므로, '보고자 여러 명'이
반드시 경합을 뜻하지는 않는다(한 편일 수 있음). 최종 확인은 개별 종목 단위로.
"""
import os
import re
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "d001.csv")
DETAIL = os.path.join(HERE, "detail_purpose.csv")   # dart_holding_purpose.py 산출물
OUT = os.path.join(HERE, "경영권분쟁_스크리닝_v2.xlsx")

# 보고자 상호를 '기관/펀드'로 추정하는 키워드 (경영참여형 전략투자자와 구분용)
INST_KW = (
    "자산운용", "투자운용", "인베스트", "인베스먼트", "인베스트먼트", "캐피탈", "캐피털",
    "증권", "은행", "저축은행", "보험", "연금", "공제", "신협", "금융", "파트너스",
    "PE", "프라이빗에쿼티", "에쿼티", "사모투자", "신기술투자", "벤처", "조합", "펀드",
    "FUND", "CAPITAL", "PARTNERS", "ADVISORS", "MANAGEMENT", "SECURITIES", "LLC",
    "LIMITED", "LTD", "L.P", "LP", "HOLDINGS", "일임", "trust", "TRUST",
)


def is_institution(name: str) -> bool:
    n = name.upper()
    return any(k.upper() in n for k in INST_KW)


def load_reports(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["is_ilban"] = df["report"].str.contains("일반")
    df["is_amend"] = df["report"].str.contains("정정")
    return df


def merge_purpose(ilban: pd.DataFrame) -> pd.DataFrame:
    """detail_purpose.csv 가 있으면 rcp 기준으로 보유목적을 붙인다."""
    if not os.path.exists(DETAIL):
        ilban["보유목적"] = ""
        ilban["경영권목적확정"] = pd.NA  # 미확인
        return ilban
    d = pd.read_csv(DETAIL, dtype=str).fillna("")
    key = "rcp" if "rcp" in d.columns else d.columns[0]
    pcol = "보유목적" if "보유목적" in d.columns else None
    if pcol is None:
        ilban["보유목적"] = ""
        ilban["경영권목적확정"] = pd.NA
        return ilban
    d = d[[key, pcol]].rename(columns={key: "rcp", pcol: "보유목적"})
    ilban = ilban.merge(d, on="rcp", how="left")
    ilban["보유목적"] = ilban["보유목적"].fillna("")
    # '경영참여' / '경영권' 문구가 있으면 확정 True, 텍스트는 있으나 없으면 False, 없으면 NA
    def flag(t):
        if not t:
            return pd.NA
        return bool(re.search(r"경영권|경영참여|경영 참여", t))
    ilban["경영권목적확정"] = ilban["보유목적"].map(flag)
    return ilban


def build_candidates(ilban: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comp, g in ilban.groupby("company"):
        filers = sorted(g["filer"].unique())
        n_filers = len(filers)
        n_reports = len(g)
        market = g["market"].iloc[0]
        has_amend = bool(g["is_amend"].any())
        n_inst = sum(is_institution(f) for f in filers)
        n_strat = n_filers - n_inst  # 개인/사업회사 = 경영참여 성격이 강한 쪽
        # 단일 보고자가 같은 종목에 반복 보고(집중 매집)한 최대 횟수
        max_rep = g.groupby("filer").size().max()
        # 보유목적 확정 신호 (detail_purpose.csv 있을 때만)
        confirmed = g["경영권목적확정"].dropna()
        n_confirmed_ctrl = int(confirmed.sum()) if len(confirmed) else 0
        purpose_known = len(confirmed) > 0

        # ---- 구조 기반 스코어 (보유목적 미확정 상태의 프록시) ----
        score = 0
        score += 3 * max(0, n_filers - 1)      # 서로 다른 블록이 경합할수록 ↑
        score += 2 if n_strat >= 1 else 0      # 전략투자자(개인/사업회사) 존재 = 경영참여 성격
        score += 2 if has_amend else 0         # 정정보고 = 지분 변동/국면 전개
        score += 1 if max_rep >= 3 else 0      # 집중 매집
        # 보유목적이 확인된 경우 확정 신호를 크게 가산
        score += 5 * n_confirmed_ctrl

        rows.append({
            "종목": comp,
            "시장": market,
            "일반보고자수": n_filers,
            "전략투자자수(개인/사업회사)": n_strat,
            "기관/펀드수": n_inst,
            "일반보고건수": n_reports,
            "최대반복보고(1인)": int(max_rep),
            "정정보고": "Y" if has_amend else "",
            "경영권목적확정건": n_confirmed_ctrl if purpose_known else "",
            "보유목적확인여부": "확인" if purpose_known else "미확인(로컬DART필요)",
            "스코어": score,
            "보고자(최대6인)": ", ".join(filers[:6]),
        })
    out = pd.DataFrame(rows).sort_values(
        ["스코어", "일반보고자수", "일반보고건수"], ascending=False
    ).reset_index(drop=True)
    out.insert(0, "순위", out.index + 1)
    return out


def build_accumulation(ilban: pd.DataFrame, min_rep: int = 4) -> pd.DataFrame:
    """단일 보고자가 한 종목에 일반보고를 반복 = 매집 추적 (최대주주 포지션 유지 포함될 수 있음)."""
    g = ilban.groupby(["company", "filer"])
    acc = g.agg(보고건수=("rcp", "size"),
                최초보고=("date", "min"),
                최근보고=("date", "max"),
                시장=("market", "first")).reset_index()
    acc = acc[acc["보고건수"] >= min_rep].copy()
    acc["보고자유형"] = acc["filer"].map(lambda f: "기관/펀드" if is_institution(f) else "개인/사업회사")
    acc = acc.rename(columns={"company": "종목", "filer": "보고자"})
    acc = acc[["종목", "시장", "보고자", "보고자유형", "보고건수", "최초보고", "최근보고"]]
    return acc.sort_values("보고건수", ascending=False).reset_index(drop=True)


def build_methodology(purpose_known: bool) -> pd.DataFrame:
    lines = [
        "경영권분쟁(경영권 영향 목적) 스크리닝 v2",
        "",
        "출처: DART 공시통합검색 D001 '주식등의대량보유상황보고서' 전수 (d001.csv)",
        "대상: 2026.06.01~2026.09.08 접수분 4,010건",
        "",
        "[1차 필터] 보고서식",
        "  · '약식' = 전문투자자/기관의 단순·일반투자 목적 -> 경영권 무관, 제외",
        "  · '일반' = 그 외 전부. 경영권 영향 목적이면 반드시 일반서식으로 제출",
        "  -> '일반'(정정 포함)만 후보군으로 사용",
        "",
        "[2차 신호] 구조 기반 스코어 (보유목적 프록시)",
        "  · 일반보고자수(서로 다른 블록 경합) x3",
        "  · 전략투자자(개인/사업회사) 존재 +2  (단순 패시브펀드와 구분)",
        "  · 정정보고 존재 +2  (지분 변동/국면 전개)",
        "  · 단일 보고자 3회+ 반복(집중 매집) +1",
        "  · [보유목적 확정 시] 경영참여/경영권 문구 확정 1건당 +5",
        "",
        "[한계 - 반드시 로컬 보완]",
        "  · 서식은 보유목적의 프록시일 뿐. 확정 판정은 보고서 본문 '보유목적' 필요",
        "  · 이 환경은 dart.fss.or.kr 차단 -> 본문 미수집",
        "  · dart_holding_purpose.py 를 DART API키로 로컬 실행 -> detail_purpose.csv 생성",
        "    -> 재실행 시 '보유목적' 자동 join, 확정 스코어(+5) 반영",
        "  · 특수관계인은 각자 별도 보고 -> '보고자 다수'가 반드시 경합은 아님(한편일 수 있음)",
        "    최종 판단은 개별 종목 단위로 확인",
        "",
        f"현재 보유목적 확인 상태: {'확인됨 (detail_purpose.csv 반영)' if purpose_known else '미확인 (구조 프록시 스코어만)'}",
    ]
    return pd.DataFrame({"경영권분쟁 스크리닝 - 방법론 및 한계": lines})


def main():
    if not os.path.exists(RAW):
        sys.exit(f"입력 파일 없음: {RAW}")
    df = load_reports(RAW)
    ilban = df[df["is_ilban"]].copy()
    ilban = merge_purpose(ilban)
    purpose_known = ilban["경영권목적확정"].notna().any()

    cand = build_candidates(ilban)
    contested = cand[cand["일반보고자수"] >= 2].copy()
    acc = build_accumulation(ilban, min_rep=4)
    method = build_methodology(purpose_known)

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        method.to_excel(xw, sheet_name="0_방법론·한계", index=False)
        cand.to_excel(xw, sheet_name="1_경영권분쟁 후보(스코어)", index=False)
        contested.to_excel(xw, sheet_name="2_다중 일반보고자(경합)", index=False)
        acc.to_excel(xw, sheet_name="3_집중매집(반복보고)", index=False)

        # 열 너비 정리
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 60)

    print(f"일반보고: {len(ilban)}건 / 후보종목: {len(cand)}")
    print(f"경합(일반보고자>=2) 종목: {len(contested)}")
    print(f"집중매집(4회+): {len(acc)}행")
    print(f"보유목적 확인: {'예' if purpose_known else '아니오(로컬 DART 필요)'}")
    print(f"저장: {OUT}")
    print()
    print("=== 상위 20 경영권분쟁 후보 ===")
    cols = ["순위", "종목", "시장", "일반보고자수", "전략투자자수(개인/사업회사)",
            "정정보고", "스코어", "보고자(최대6인)"]
    print(cand[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
