"""
Historical price data fetching for stocks / indices / FX / crypto.

Backed by FinanceDataReader, which aggregates KRX, Naver Finance and global
sources (Yahoo/Stooq) behind a single API. Given an asset name (Korean or
English) or a raw symbol/code plus a date range, this returns an OHLCV
DataFrame together with a human-readable summary and a CSV export.

Used by the Telegram bot's ``/history`` command, but every function here is
plain and reusable from a CLI or notebook as well.
"""

from __future__ import annotations

import re
import io
import logging
from datetime import date, datetime, timedelta

import FinanceDataReader as fdr
import pandas as pd

import krx_api
import naver_crawler

logger = logging.getLogger(__name__)

_KRX_CODE_RE = re.compile(r"^\d{6}$")


# ─── Name → FinanceDataReader symbol aliases ───────────────────────────────
# FDR symbol conventions:
#   • Korean stocks : 6-digit KRX code, e.g. "005930"  (NO .KS/.KQ suffix)
#   • Korean indices: "KS11" KOSPI, "KQ11" KOSDAQ, "KS200" KOSPI200
#   • US indices    : "DJI" Dow, "IXIC" Nasdaq, "US500" S&P500
#   • US stocks     : plain ticker, e.g. "AAPL"
#   • FX            : "USD/KRW", "USD/EUR", ...
#   • Crypto        : "BTC/USD", "BTC/KRW", "ETH/USD", ...
NAME_TO_SYMBOL: dict[str, str] = {
    # ── Korean stocks ──
    "삼성전자": "005930", "삼성": "005930",
    "sk하이닉스": "000660", "하이닉스": "000660",
    "네이버": "035420", "naver": "035420",
    "카카오": "035720",
    "현대차": "005380", "현대자동차": "005380",
    "기아": "000270",
    "lg에너지솔루션": "373220", "엘지엔솔": "373220",
    "삼성바이오로직스": "207940", "삼바": "207940",
    "셀트리온": "068270",
    "포스코홀딩스": "005490", "포스코": "005490",
    "kb금융": "105560", "신한지주": "055550",
    "삼성sdi": "006400", "lg화학": "051910",
    "현대모비스": "012330", "sk이노베이션": "096770",
    "삼성물산": "028260", "한화에어로스페이스": "012450",
    # ── Korean indices / ETF ──
    "코스피": "KS11", "kospi": "KS11",
    "코스닥": "KQ11", "kosdaq": "KQ11",
    "코스피200": "KS200", "kospi200": "KS200",
    "코덱스200": "069500", "kodex200": "069500",
    # ── US stocks ──
    "테슬라": "TSLA", "애플": "AAPL", "엔비디아": "NVDA",
    "마이크로소프트": "MSFT", "구글": "GOOGL", "아마존": "AMZN",
    "메타": "META", "넷플릭스": "NFLX", "오라클": "ORCL",
    "코인베이스": "COIN", "마이크로스트래티지": "MSTR",
    "amd": "AMD", "인텔": "INTC", "팔란티어": "PLTR",
    "쿠팡": "CPNG",
    # ── US / global indices ──
    "나스닥": "IXIC", "nasdaq": "IXIC",
    "s&p500": "US500", "sp500": "US500", "에스앤피": "US500",
    "다우": "DJI", "다우존스": "DJI", "dow": "DJI",
    "vix": "VIX", "빅스": "VIX",
    # ── FX ──
    "원달러": "USD/KRW", "달러": "USD/KRW", "달러원": "USD/KRW",
    "유로달러": "EUR/USD", "엔달러": "USD/JPY", "달러엔": "USD/JPY",
    "위안": "USD/CNY", "원엔": "JPY/KRW",
    # ── Commodities (FDR/Yahoo futures) ──
    "금": "GC=F", "gold": "GC=F",
    "은": "SI=F", "silver": "SI=F",
    "원유": "CL=F", "wti": "CL=F", "crude": "CL=F",
    "천연가스": "NG=F",
    # ── Crypto ──
    "비트코인": "BTC/USD", "비코": "BTC/USD", "btc": "BTC/USD",
    "이더리움": "ETH/USD", "이더": "ETH/USD", "eth": "ETH/USD",
    "리플": "XRP/USD", "xrp": "XRP/USD",
    "솔라나": "SOL/USD", "솔": "SOL/USD", "sol": "SOL/USD",
    "도지코인": "DOGE/USD", "도지": "DOGE/USD", "doge": "DOGE/USD",
    "카르다노": "ADA/USD", "에이다": "ADA/USD", "ada": "ADA/USD",
    "폴카닷": "DOT/USD", "dot": "DOT/USD",
    "아발란체": "AVAX/USD", "avax": "AVAX/USD",
    "체인링크": "LINK/USD", "링크": "LINK/USD", "link": "LINK/USD",
    "라이트코인": "LTC/USD", "ltc": "LTC/USD",
    "트론": "TRX/USD", "trx": "TRX/USD",
    # Crypto priced in KRW (Korean-won pairs)
    "비트코인원화": "BTC/KRW", "이더리움원화": "ETH/KRW",
}


# ─── Date & window parsing ─────────────────────────────────────────────────
# Relative windows accepted when no explicit start date is given.
WINDOW_ALIASES: dict[str, int] = {
    "1w": 7, "1주": 7, "1주일": 7,
    "1m": 30, "1개월": 30, "한달": 30,
    "3m": 91, "3개월": 91, "분기": 91,
    "6m": 182, "6개월": 182, "반년": 182,
    "1y": 365, "1년": 365, "ytd": -1,  # ytd handled specially
    "2y": 730, "2년": 730,
    "3y": 1095, "3년": 1095,
    "5y": 1825, "5년": 1825,
    "10y": 3650, "10년": 3650,
    "max": 20000, "전체": 20000,
}

_DATE_PATTERNS = [
    ("%Y-%m-%d", re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")),
    ("%Y.%m.%d", re.compile(r"^\d{4}\.\d{1,2}\.\d{1,2}$")),
    ("%Y/%m/%d", re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$")),
    ("%Y%m%d",   re.compile(r"^\d{8}$")),
]


def parse_date(token: str) -> date | None:
    """Parse a single date token in several common formats, else None."""
    token = token.strip()
    for fmt, pat in _DATE_PATTERNS:
        if pat.match(token):
            try:
                return datetime.strptime(token, fmt).date()
            except ValueError:
                return None
    return None


def resolve_symbol(name: str) -> str:
    """Map a Korean/English asset name to an FDR symbol.

    Unknown names fall through unchanged so raw symbols/codes (``005930``,
    ``AAPL``, ``BTC/USD``) work directly.
    """
    key = name.strip().lower()
    if key in NAME_TO_SYMBOL:
        return NAME_TO_SYMBOL[key]
    # Bare 6-digit number → treat as a KRX code as-is.
    return name.strip()


def parse_query(tokens: list[str]) -> tuple[str, date, date]:
    """Split raw ``/history`` arguments into (name, start_date, end_date).

    Recognises explicit dates (any supported format) and relative window
    keywords anywhere in the argument list; everything else forms the name.

    Rules:
      • two dates            → [start, end]
      • one date             → [that date, today]
      • one window keyword   → [today - window, today]
      • nothing time-related → default to the last 1 year
    """
    dates: list[date] = []
    window_days: int | None = None
    ytd = False
    name_parts: list[str] = []

    for tok in tokens:
        d = parse_date(tok)
        if d is not None:
            dates.append(d)
            continue
        low = tok.strip().lower()
        if low in WINDOW_ALIASES:
            if low == "ytd" or WINDOW_ALIASES[low] == -1:
                ytd = True
            else:
                window_days = WINDOW_ALIASES[low]
            continue
        name_parts.append(tok)

    name = " ".join(name_parts).strip()
    if not name:
        raise ValueError("종목/자산 이름이 없습니다.")

    today = date.today()

    if len(dates) >= 2:
        dates.sort()
        start, end = dates[0], dates[-1]
    elif len(dates) == 1:
        start, end = dates[0], today
    elif ytd:
        start, end = date(today.year, 1, 1), today
    elif window_days is not None:
        start, end = today - timedelta(days=window_days), today
    else:
        start, end = today - timedelta(days=365), today  # default: 1 year

    return name, start, end


# ─── Fetch & format ────────────────────────────────────────────────────────
def fetch_history(name: str, start: date, end: date) -> tuple[str, str, pd.DataFrame]:
    """Fetch OHLCV history for ``name`` between ``start`` and ``end``.

    Returns (resolved_symbol, display_name, DataFrame). Raises ValueError with
    a user-friendly Korean message when nothing is found.
    """
    symbol = resolve_symbol(name)
    logger.info("fetch_history: %r → symbol=%r %s..%s", name, symbol, start, end)

    # Domestic 6-digit codes → try sources in order of authority, each falling
    # back to the next on failure: official KRX API → Naver crawl → FDR.
    if _KRX_CODE_RE.match(symbol):
        service_key = krx_api.get_service_key()
        if service_key:
            try:
                df = krx_api.fetch_krx_daily(symbol, start, end, service_key)
                return symbol, name, df.dropna(how="all")
            except Exception as e:  # noqa: BLE001
                logger.warning("KRX API failed for %s, trying Naver: %s", symbol, e)

        try:
            df = naver_crawler.fetch_naver_daily(symbol, start, end)
            return symbol, name, df.dropna(how="all")
        except Exception as e:  # noqa: BLE001
            logger.warning("Naver crawl failed for %s, falling back to FDR: %s", symbol, e)

    try:
        df = fdr.DataReader(symbol, start.isoformat(), end.isoformat())
    except Exception as e:  # network / symbol errors
        raise ValueError(f"'{name}'({symbol}) 데이터를 가져오지 못했습니다: {str(e)[:150]}")

    if df is None or df.empty:
        raise ValueError(
            f"'{name}'({symbol})에 대한 데이터가 해당 기간에 없습니다. "
            f"종목 코드/이름 또는 기간을 확인해 주세요."
        )

    df = df.dropna(how="all")
    return symbol, name, df


def summarize(name: str, symbol: str, df: pd.DataFrame) -> str:
    """Build a compact plaintext summary of an OHLCV DataFrame."""
    close_col = "Close" if "Close" in df.columns else df.columns[-1]
    closes = df[close_col].dropna()
    first_close = closes.iloc[0]
    last_close = closes.iloc[-1]
    pct = (last_close - first_close) / first_close * 100 if first_close else float("nan")

    period_high = df["High"].max() if "High" in df.columns else closes.max()
    period_low = df["Low"].min() if "Low" in df.columns else closes.min()
    idx_start = df.index[0]
    idx_end = df.index[-1]

    def fmt(v: float) -> str:
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:,.4f}".rstrip("0").rstrip(".")

    source = df.attrs.get("source", "FinanceDataReader")

    lines = [
        f"📈 {name}  (심볼: {symbol})",
        f"기간: {idx_start.date()} ~ {idx_end.date()}  ({len(df)} 거래일)",
        f"출처: {source}",
        "",
        f"시작 종가 : {fmt(first_close)}",
        f"최종 종가 : {fmt(last_close)}",
        f"기간 수익률: {pct:+.2f}%",
        f"기간 고가 : {fmt(period_high)}",
        f"기간 저가 : {fmt(period_low)}",
    ]

    if "Volume" in df.columns and df["Volume"].notna().any():
        avg_vol = df["Volume"].mean()
        lines.append(f"평균 거래량: {avg_vol:,.0f}")

    # Last few rows preview
    preview_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    tail = df[preview_cols].tail(5) if preview_cols else df.tail(5)
    lines.append("")
    lines.append("최근 데이터:")
    for idx, row in tail.iterrows():
        cells = "  ".join(
            f"{c[0]}={fmt(row[c])}" if c != "Volume" else f"V={row[c]:,.0f}"
            for c in tail.columns
        )
        lines.append(f"  {idx.date()}  {cells}")

    return "\n".join(lines)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize the DataFrame (with its date index) to CSV bytes."""
    buf = io.StringIO()
    df.to_csv(buf, index=True, index_label="Date")
    return buf.getvalue().encode("utf-8-sig")  # BOM so Excel reads Korean fine


def csv_filename(symbol: str, df: pd.DataFrame) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_") or "history"
    start = df.index[0].date()
    end = df.index[-1].date()
    return f"{safe}_{start}_{end}.csv"


# ─── CLI entry point (optional standalone use) ─────────────────────────────
def _main() -> None:
    import sys

    args = sys.argv[1:]
    if not args:
        print("사용법: python price_history.py <종목/자산> [시작일] [종료일|기간]")
        print("예시  : python price_history.py 삼성전자 2024-01-01 2024-06-30")
        print("        python price_history.py BTC/USD 1y")
        raise SystemExit(1)

    logging.basicConfig(level=logging.INFO)
    name, start, end = parse_query(args)
    symbol, _, df = fetch_history(name, start, end)
    print(summarize(name, symbol, df))
    out = csv_filename(symbol, df)
    df.to_csv(out, index=True, index_label="Date", encoding="utf-8-sig")
    print(f"\nCSV 저장: {out}")


if __name__ == "__main__":
    _main()
