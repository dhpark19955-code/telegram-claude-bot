"""기관 순매수 스크리너 v2용 parquet 증분 캐시.

일자 × 시장 × 종류(투자자 구분 또는 시세 스냅샷) 단위로 하나의 parquet 파일을
저장한다. 이미 캐시된 날짜는 다시 조회하지 않으므로, 스크리너를 매일 실행해도
새로 추가된 거래일만 pykrx를 호출한다.

파일명 규칙: ``{date}_{market}_{key}.parquet`` (date=YYYYMMDD)
  - key 가 투자자명(기관합계/금융투자/투신/연기금/사모)이면 해당 일자의
    투자자별 순매수 스냅샷
  - key 가 "시세" 이면 해당 일자의 종가/시가총액/거래대금 스냅샷
"""
from __future__ import annotations

import os

import pandas as pd

DEFAULT_CACHE_DIR = "./cache"


def _path(cache_dir: str, date: str, market: str, key: str) -> str:
    return os.path.join(cache_dir, f"{date}_{market}_{key}.parquet")


def load(cache_dir: str, date: str, market: str, key: str) -> pd.DataFrame | None:
    path = _path(cache_dir, date, market, key)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def save(cache_dir: str, date: str, market: str, key: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(_path(cache_dir, date, market, key))


def clear(cache_dir: str, market: str | None = None) -> int:
    """캐시 파일을 삭제한다. market을 지정하면 해당 시장 파일만 삭제한다."""
    if not os.path.isdir(cache_dir):
        return 0
    removed = 0
    for name in os.listdir(cache_dir):
        if not name.endswith(".parquet"):
            continue
        if market and f"_{market}_" not in name:
            continue
        os.remove(os.path.join(cache_dir, name))
        removed += 1
    return removed
