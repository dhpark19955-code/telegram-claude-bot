"""
Official Korean-stock daily prices via the data.go.kr (공공데이터포털) open API.

Uses the Financial Services Commission "주식시세정보 / getStockPriceInfo"
endpoint, which serves KRX-sourced daily OHLCV for listed Korean securities.
Requires a service key, read from the ``KRX_SERVICE_KEY`` environment variable
— never hardcode it.

Returned DataFrames match the shape produced by ``price_history.fetch_history``
(index = date, columns Open/High/Low/Close/Volume [+ Change, Amount, MarketCap])
so the summary/CSV helpers can be reused unchanged.
"""

from __future__ import annotations

import os
import logging
from datetime import date

import requests
import pandas as pd

logger = logging.getLogger(__name__)

ENDPOINT = (
    "https://apis.data.go.kr/1160100/service/"
    "GetStockSecuritiesInfoService/getStockPriceInfo"
)


def get_service_key() -> str | None:
    """Return the KRX service key from the environment, or None if unset."""
    key = os.environ.get("KRX_SERVICE_KEY", "").strip()
    return key or None


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fetch_krx_daily(code: str, start: date, end: date, service_key: str) -> pd.DataFrame:
    """Fetch daily OHLCV for a 6-digit KRX code over [start, end] (inclusive).

    Raises ValueError on an API error or empty result so callers can fall back.
    """
    rows: list[dict] = []
    page = 1
    num_rows = 1000

    while True:
        params = {
            "serviceKey": service_key,
            "resultType": "json",
            "numOfRows": num_rows,
            "pageNo": page,
            "beginBasDt": start.strftime("%Y%m%d"),
            "endBasDt": end.strftime("%Y%m%d"),
            "likeSrtnCd": code,
        }
        resp = requests.get(ENDPOINT, params=params, timeout=25)
        resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            # data.go.kr returns an XML error envelope (not JSON) on auth/quota
            # problems even when resultType=json is requested.
            raise ValueError(
                f"KRX API가 JSON이 아닌 오류를 반환했습니다 (인증키/쿼터 확인): "
                f"{resp.text[:200]}"
            )

        response = data.get("response", {})
        header = response.get("header", {})
        result_code = header.get("resultCode")
        if result_code not in (None, "00", "000"):
            raise ValueError(
                f"KRX API 오류 {result_code}: {header.get('resultMsg', '')}"
            )

        body = response.get("body", {})
        items = body.get("items", {})
        item = items.get("item", []) if isinstance(items, dict) else []
        if isinstance(item, dict):
            item = [item]
        rows.extend(item)

        total = int(body.get("totalCount", 0) or 0)
        got = int(body.get("numOfRows", num_rows) or num_rows)
        if not item or page * got >= total:
            break
        page += 1

    if not rows:
        raise ValueError(f"KRX API: 코드 {code}에 대한 데이터가 없습니다.")

    # A substring code match can return unrelated tickers; keep the exact one.
    exact = [r for r in rows if str(r.get("srtnCd", "")).zfill(6) == code]
    if exact:
        rows = exact

    recs = []
    for it in rows:
        recs.append({
            "Date": pd.to_datetime(str(it["basDt"]), format="%Y%m%d"),
            "Open": _to_float(it.get("mkp")),
            "High": _to_float(it.get("hipr")),
            "Low": _to_float(it.get("lopr")),
            "Close": _to_float(it.get("clpr")),
            "Volume": _to_float(it.get("trqu")),
            "Change": _to_float(it.get("fltRt")),
            "Amount": _to_float(it.get("trPrc")),
            "MarketCap": _to_float(it.get("mrktTotAmt")),
        })

    df = pd.DataFrame(recs).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.attrs["source"] = "KRX (data.go.kr)"
    logger.info("KRX API: code=%s rows=%d %s..%s", code, len(df), start, end)
    return df
