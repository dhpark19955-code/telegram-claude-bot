"""
Telegram bot for price data only.

Two things:
  • /price   — real-time / delayed quote snapshot (Yahoo Finance via yfinance)
  • /history — historical OHLCV over a date range (KRX official API → Naver
               crawl → FinanceDataReader), returned as a summary + CSV file

News / chat / LLM features intentionally live in a separate bot.
"""

import os
import io
import re
import html
import logging
from pathlib import Path

import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import price_history

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── Load .env (for local dev) ────────────────────────────
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# ─── Config ────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_USERS = [int(x) for x in os.environ.get("ALLOWED_USERS", "").split(",") if x.strip()]


# ─── Real-time quote via yfinance ──────────────────────────
# Aliases (Korean/English names → Yahoo Finance tickers) for /price snapshots.
TICKER_ALIASES = {
    # Korean stocks
    "삼성전자": "005930.KS", "삼성": "005930.KS",
    "sk하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "네이버": "035420.KS", "카카오": "035720.KS",
    "현대차": "005380.KS", "현대자동차": "005380.KS",
    "기아": "000270.KS", "lg에너지솔루션": "373220.KS",
    "셀트리온": "068270.KS", "포스코홀딩스": "005490.KS",
    "쿠팡": "CPNG",
    # US stocks
    "테슬라": "TSLA", "애플": "AAPL", "엔비디아": "NVDA",
    "마이크로소프트": "MSFT", "구글": "GOOGL", "아마존": "AMZN",
    "메타": "META", "넷플릭스": "NFLX", "오라클": "ORCL",
    "코인베이스": "COIN", "마이크로스트래티지": "MSTR",
    # Crypto - Major
    "비트코인": "BTC-USD", "비코": "BTC-USD", "btc": "BTC-USD",
    "이더리움": "ETH-USD", "이더": "ETH-USD", "eth": "ETH-USD",
    "리플": "XRP-USD", "xrp": "XRP-USD",
    "솔라나": "SOL-USD", "솔": "SOL-USD", "sol": "SOL-USD",
    "도지코인": "DOGE-USD", "도지": "DOGE-USD", "doge": "DOGE-USD",
    "카르다노": "ADA-USD", "에이다": "ADA-USD", "ada": "ADA-USD",
    "폴카닷": "DOT-USD", "dot": "DOT-USD",
    "아발란체": "AVAX-USD", "avax": "AVAX-USD",
    "체인링크": "LINK-USD", "링크": "LINK-USD", "link": "LINK-USD",
    "라이트코인": "LTC-USD", "ltc": "LTC-USD",
    "트론": "TRX-USD", "trx": "TRX-USD",
    "시바이누": "SHIB-USD", "시바": "SHIB-USD", "shib": "SHIB-USD",
    "페페": "PEPE-USD", "pepe": "PEPE-USD",
    # Indices & FX & Commodities
    "코스피": "^KS11", "나스닥": "^IXIC", "s&p500": "^GSPC",
    "다우": "^DJI", "달러": "KRW=X", "원달러": "KRW=X",
    "금": "GC=F", "원유": "CL=F", "wti": "CL=F",
}


def extract_tickers(text: str) -> list[str]:
    """Extract potential ticker symbols from a message."""
    tickers = []
    lower = text.lower()

    for alias, ticker in TICKER_ALIASES.items():
        if alias in lower:
            tickers.append(ticker)

    explicit = re.findall(r'\$?([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b', text)
    for t in explicit:
        if len(t) >= 2 and t not in ("OR", "AN", "IS", "IT", "AT", "ON", "IN", "TO", "IF", "NO", "DO", "SO", "BY", "UP"):
            tickers.append(t)

    return list(dict.fromkeys(tickers))  # dedupe, preserve order


def get_price_data(ticker: str) -> str:
    """Fetch a real-time/delayed quote snapshot for a ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            fi = stock.fast_info
            if fi and hasattr(fi, "last_price") and fi.last_price:
                return (
                    f"Ticker: {ticker}\n"
                    f"Price: {fi.last_price:.2f}\n"
                    f"Previous Close: {fi.previous_close:.2f}\n"
                    f"Change: {((fi.last_price - fi.previous_close) / fi.previous_close * 100):.2f}%"
                )
            return f"Ticker {ticker}: No data available"

        parts = [f"Ticker: {ticker}"]
        name = info.get("shortName") or info.get("longName", "")
        if name:
            parts.append(f"Name: {name}")

        price = info.get("regularMarketPrice") or info.get("currentPrice")
        prev = info.get("regularMarketPreviousClose") or info.get("previousClose")
        if price:
            parts.append(f"Price: {price}")
        if prev and price:
            chg = (price - prev) / prev * 100
            parts.append(f"Change: {chg:+.2f}%")

        for key, label in [
            ("regularMarketDayHigh", "Day High"),
            ("regularMarketDayLow", "Day Low"),
            ("regularMarketVolume", "Volume"),
            ("marketCap", "Market Cap"),
            ("fiftyTwoWeekHigh", "52W High"),
            ("fiftyTwoWeekLow", "52W Low"),
            ("fiftyDayAverage", "50D MA"),
            ("twoHundredDayAverage", "200D MA"),
        ]:
            val = info.get(key)
            if val is not None:
                if key == "marketCap":
                    if val >= 1e12:
                        parts.append(f"{label}: {val/1e12:.2f}T")
                    elif val >= 1e9:
                        parts.append(f"{label}: {val/1e9:.2f}B")
                    else:
                        parts.append(f"{label}: {val/1e6:.2f}M")
                elif key == "regularMarketVolume":
                    parts.append(f"{label}: {val:,.0f}")
                else:
                    parts.append(f"{label}: {val}")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}")
        return f"Ticker {ticker}: Error fetching data - {str(e)[:100]}"


# ─── Helpers ───────────────────────────────────────────────
def is_authorized(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


_DATE_TOKEN_RE = re.compile(r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{8}")


def _looks_like_history(tokens: list[str]) -> bool:
    """True if any token is a date or a relative-window keyword."""
    for tok in tokens:
        if _DATE_TOKEN_RE.fullmatch(tok):
            return True
        if tok.strip().lower() in price_history.WINDOW_ALIASES:
            return True
    return False


async def _reply_price(update: Update, query: str):
    tickers = extract_tickers(query)
    if not tickers:
        tickers = [query.upper()]
    await update.message.chat.send_action("typing")
    results = [get_price_data(t) for t in tickers[:3]]
    text = "\n\n---\n\n".join(results)
    await update.message.reply_text(f"<pre>{html.escape(text)}</pre>", parse_mode="HTML")


async def _reply_history(update: Update, tokens: list[str]):
    try:
        name, start, end = price_history.parse_query(tokens)
    except ValueError as e:
        await update.message.reply_text(f"입력 오류: {e}")
        return

    await update.message.chat.send_action("typing")
    try:
        symbol, _, df = price_history.fetch_history(name, start, end)
    except ValueError as e:
        await update.message.reply_text(str(e))
        return
    except Exception as e:  # noqa: BLE001
        logger.error(f"history fetch error: {e}", exc_info=True)
        await update.message.reply_text(f"데이터 조회 중 오류가 발생했습니다: {str(e)[:200]}")
        return

    summary = price_history.summarize(name, symbol, df)
    await update.message.reply_text(f"<pre>{html.escape(summary)}</pre>", parse_mode="HTML")

    try:
        csv_bytes = price_history.to_csv_bytes(df)
        filename = price_history.csv_filename(symbol, df)
        await update.message.reply_document(
            document=io.BytesIO(csv_bytes),
            filename=filename,
            caption=f"{name} ({symbol}) — {len(df)} 거래일 CSV",
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"csv send error: {e}", exc_info=True)


# ─── Handlers ──────────────────────────────────────────────
async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Real-time quote snapshot: /price AAPL"""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    if not context.args:
        await update.message.reply_text("사용법: /price AAPL  또는  /price 삼성전자")
        return
    await _reply_price(update, " ".join(context.args).strip())


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Historical OHLCV over a date range: /history 삼성전자 2024-01-01 2024-06-30"""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    if not context.args:
        await update.message.reply_text(
            "사용법: /history <종목/자산> [시작일] [종료일 또는 기간]\n\n"
            "예시:\n"
            "• /history 삼성전자 2024-01-01 2024-06-30\n"
            "• /history 비트코인 1y\n"
            "• /history 코스피 ytd\n"
            "• /history AAPL 2024-03-01\n"
            "• /history 005930 20240101 20240301\n\n"
            "기간 키워드: 1w, 1m, 3m, 6m, 1y, 3y, 5y, ytd, max (또는 1개월/6개월/1년 …)\n"
            "날짜 형식: YYYY-MM-DD / YYYY.MM.DD / YYYYMMDD"
        )
        return
    await _reply_history(update, context.args)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Plain text → history if it carries a date/window, else a price snapshot."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Not authorized.")
        return
    tokens = (update.message.text or "").split()
    if not tokens:
        return
    if _looks_like_history(tokens):
        await _reply_history(update, tokens)
    else:
        await _reply_price(update, update.message.text.strip())


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    krx = "Enabled (data.go.kr)" if price_history.krx_api.get_service_key() else "Disabled (set KRX_SERVICE_KEY)"
    await update.message.reply_text(
        f"User ID: {update.effective_user.id}\n"
        f"Quote (real-time): yfinance\n"
        f"History (time series): KRX API → Naver → FinanceDataReader\n"
        f"KRX official API: {krx}\n"
        f"Allowed users: {ALLOWED_USERS or 'Everyone'}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>가격 데이터 봇</b>\n\n"
        "<b>Commands</b>\n"
        "/price 삼성전자 — 실시간 시세 스냅샷\n"
        "/history 삼성전자 2024-01-01 2024-06-30 — 기간별 시계열 + CSV\n"
        "/history 비트코인 1y — 최근 1년\n"
        "/status — 데이터 소스 상태\n"
        "/help — 이 도움말\n\n"
        "<b>Tip</b>\n"
        "• 그냥 <code>삼성전자</code> 라고 보내면 실시간 시세\n"
        "• <code>삼성전자 2024-01-01 2024-06-30</code> 처럼 날짜를 넣으면 시계열\n"
        "• 지원: 국내/미국 주식, 지수, 환율, 원자재, 코인",
        parse_mode="HTML",
    )


# ─── Main ──────────────────────────────────────────────────
def main():
    if not BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
        return

    logger.info("Starting price bot")
    logger.info(f"KRX official API: {'on' if price_history.krx_api.get_service_key() else 'off'}")
    logger.info(f"Allowed users: {ALLOWED_USERS or 'Everyone'}")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("hist", cmd_history))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running.")
    app.run_polling()


if __name__ == "__main__":
    main()
