"""
Direct daily-price crawling from Naver Finance for Korean listed stocks.

Serves as a resilient fallback between the official KRX API and
FinanceDataReader: it hits Naver's ``siseJson`` endpoint, which returns a
date-ranged OHLCV array, so it keeps working even if FDR's internal Naver
endpoint (``fchart``) changes.

Returned DataFrames match ``price_history.fetch_history`` so summary/CSV
helpers are reused unchanged.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import requests
import pandas as pd

logger = logging.getLogger(__name__)

# siseJson returns a date-ranged array (unlike fchart's count-based feed).
ENDPOINT = "https://api.finance.naver.com/siseJson.naver"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/",
}


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fetch_naver_daily(code: str, start: date, end: date) -> pd.DataFrame:
    """Crawl daily OHLCV for a 6-digit KRX code over [start, end] (inclusive).

    Raises ValueError on a failed/empty response so callers can fall back.
    """
    params = {
        "symbol": code,
        "requestType": "1",
        "startTime": start.strftime("%Y%m%d"),
        "endTime": end.strftime("%Y%m%d"),
        "timeframe": "day",
    }
    resp = requests.get(ENDPOINT, params=params, headers=_HEADERS, timeout=25)
    resp.raise_for_status()

    text = resp.text.strip()
    if not text:
        raise ValueError(f"네이버 응답이 비어 있습니다 (code={code}).")

    # The body is a JS array literal: header row uses single quotes, so
    # normalise to valid JSON before parsing.
    try:
        rows = json.loads(text.replace("'", '"'))
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"네이버 응답 파싱 실패 (code={code}): {str(e)[:150]}")

    # rows[0] is the header: [날짜, 시가, 고가, 저가, 종가, 거래량, 외국인소진율]
    data_rows = [r for r in rows[1:] if r and len(r) >= 6]
    if not data_rows:
        raise ValueError(f"네이버: code {code} 데이터가 해당 기간에 없습니다.")

    recs = []
    for r in data_rows:
        recs.append({
            "Date": pd.to_datetime(str(r[0]), format="%Y%m%d"),
            "Open": _to_float(r[1]),
            "High": _to_float(r[2]),
            "Low": _to_float(r[3]),
            "Close": _to_float(r[4]),
            "Volume": _to_float(r[5]),
        })

    df = pd.DataFrame(recs).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.attrs["source"] = "Naver (crawl)"
    logger.info("Naver crawl: code=%s rows=%d %s..%s", code, len(df), start, end)
    return df
