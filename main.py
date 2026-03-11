"""
Telegram bot powered by Anthropic Claude API with web search + price analysis.
Deployable on Railway / any cloud platform.
"""

import os
import re
import html
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import anthropic
import mistune
import yfinance as yf
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

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
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ALLOWED_USERS = [int(x) for x in os.environ.get("ALLOWED_USERS", "").split(",") if x.strip()]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "50"))

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", """You are an expert financial analyst assistant on Telegram with real-time data access.

## Core Capabilities
- Real-time stock price analysis (provided via [PRICE_DATA] blocks)
- Web search for latest news, earnings, filings, macro data
- Technical & fundamental analysis

## News Source Priority
When searching for news, ALWAYS prioritize these sources in order:
1. Bloomberg (bloomberg.com)
2. Reuters (reuters.com)
3. Financial Times (ft.com)
4. Wall Street Journal (wsj.com)
5. SEC filings (sec.gov) for US companies
6. Company IR pages for earnings data

When searching, include "bloomberg OR reuters" in your search queries to prioritize these sources.

## Analysis Guidelines
- Always provide specific numbers: price, % change, volume, P/E, market cap
- Compare against sector peers and indices when relevant
- Note key support/resistance levels for price analysis
- Identify catalysts: earnings, macro events, sector trends
- Give both bull and bear case when analyzing
- Use tables for comparing multiple data points
- Mention data timestamps so user knows how current the info is

## Price Data
When [PRICE_DATA] is provided in the user message, use it for your analysis.
This data comes from Yahoo Finance and includes real-time/delayed quotes.

## Formatting
- Use markdown for clean formatting
- Keep responses focused and data-driven
- Use bullet points for key takeaways
- Bold important numbers and conclusions
""")

# ─── Anthropic Client ─────────────────────────────────────
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── In-memory conversation history ───────────────────────
conversations: dict[int, list] = defaultdict(list)

# ─── Telegram-compatible Markdown renderer ─────────────────
class TelegramRenderer(mistune.HTMLRenderer):
    def heading(self, text, level, **attrs):
        return f"<b>{text}</b>\n\n"

    def paragraph(self, text):
        return f"{text}\n\n"

    def list(self, text, ordered, **attrs):
        return text + "\n"

    def list_item(self, text, **attrs):
        return f"• {text}\n"

    def block_code(self, code, info=None):
        return f"<pre>{html.escape(code.strip())}</pre>\n\n"

    def codespan(self, text):
        return f"<code>{html.escape(text)}</code>"

    def emphasis(self, text):
        return f"<i>{text}</i>"

    def strong(self, text):
        return f"<b>{text}</b>"

    def strikethrough(self, text):
        return f"<s>{text}</s>"

    def link(self, text, url, title=None):
        return f'<a href="{html.escape(url)}">{text}</a>'

    def image(self, text, url, title=None):
        return f"[Image: {text}]"

    def block_quote(self, text):
        return f"<blockquote>{text}</blockquote>\n"

    def thematic_break(self):
        return "\n---\n\n"

    def linebreak(self):
        return "\n"

    def table(self, text):
        return f"<pre>{text}</pre>\n\n"

    def table_head(self, text):
        return text + "─" * 20 + "\n"

    def table_body(self, text):
        return text

    def table_row(self, text):
        return text + "\n"

    def table_cell(self, text, align=None, head=False):
        if head:
            return f"<b>{text}</b> │ "
        return f"{text} │ "


md = mistune.create_markdown(
    renderer=TelegramRenderer(escape=False),
    plugins=["strikethrough", "table", "task_lists", "url"],
)


def md_to_tg(text: str) -> str:
    return md(text).strip()


# ─── Price Data via yfinance ───────────────────────────────
# Common ticker aliases (Korean/English names → Yahoo Finance tickers)
TICKER_ALIASES = {
    "삼성전자": "005930.KS", "삼성": "005930.KS",
    "sk하이닉스": "000660.KS", "하이닉스": "000660.KS",
    "네이버": "035420.KS", "카카오": "035720.KS",
    "현대차": "005380.KS", "현대자동차": "005380.KS",
    "기아": "000270.KS", "lg에너지솔루션": "373220.KS",
    "셀트리온": "068270.KS", "포스코홀딩스": "005490.KS",
    "야놀자": "YNLJA", "쿠팡": "CPNG",
    "테슬라": "TSLA", "애플": "AAPL", "엔비디아": "NVDA",
    "마이크로소프트": "MSFT", "구글": "GOOGL", "아마존": "AMZN",
    "메타": "META", "넷플릭스": "NFLX", "오라클": "ORCL",
    "비트코인": "BTC-USD", "이더리움": "ETH-USD",
    "코스피": "^KS11", "나스닥": "^IXIC", "s&p500": "^GSPC",
    "다우": "^DJI", "달러": "KRW=X", "원달러": "KRW=X",
    "금": "GC=F", "원유": "CL=F", "wti": "CL=F",
}


def extract_tickers(text: str) -> list[str]:
    """Extract potential ticker symbols from user message."""
    tickers = []
    lower = text.lower()

    # Check aliases
    for alias, ticker in TICKER_ALIASES.items():
        if alias in lower:
            tickers.append(ticker)

    # Check for explicit tickers (e.g., $AAPL, TSLA)
    explicit = re.findall(r'\$?([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b', text)
    for t in explicit:
        if len(t) >= 2 and t not in ("OR", "AN", "IS", "IT", "AT", "ON", "IN", "TO", "IF", "NO", "DO", "SO", "BY", "UP"):
            tickers.append(t)

    return list(dict.fromkeys(tickers))  # deduplicate, preserve order


def get_price_data(ticker: str) -> str:
    """Fetch price data for a ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            # Try fast_info as fallback
            fi = stock.fast_info
            if fi and hasattr(fi, "last_price") and fi.last_price:
                return (
                    f"Ticker: {ticker}\n"
                    f"Price: {fi.last_price:.2f}\n"
                    f"Previous Close: {fi.previous_close:.2f}\n"
                    f"Change: {((fi.last_price - fi.previous_close) / fi.previous_close * 100):.2f}%\n"
                    f"Market Cap: {fi.market_cap:,.0f}\n" if hasattr(fi, "market_cap") and fi.market_cap else ""
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
            ("trailingPE", "P/E (TTM)"),
            ("forwardPE", "P/E (Fwd)"),
            ("fiftyTwoWeekHigh", "52W High"),
            ("fiftyTwoWeekLow", "52W Low"),
            ("fiftyDayAverage", "50D MA"),
            ("twoHundredDayAverage", "200D MA"),
            ("dividendYield", "Div Yield"),
            ("beta", "Beta"),
            ("trailingEps", "EPS (TTM)"),
            ("revenueGrowth", "Revenue Growth"),
            ("earningsGrowth", "Earnings Growth"),
        ]:
            val = info.get(key)
            if val is not None:
                if key == "marketCap":
                    if val >= 1e12:
                        parts.append(f"{label}: ${val/1e12:.2f}T")
                    elif val >= 1e9:
                        parts.append(f"{label}: ${val/1e9:.2f}B")
                    else:
                        parts.append(f"{label}: ${val/1e6:.2f}M")
                elif key == "regularMarketVolume":
                    parts.append(f"{label}: {val:,.0f}")
                elif key in ("dividendYield", "revenueGrowth", "earningsGrowth"):
                    parts.append(f"{label}: {val*100:.2f}%")
                else:
                    parts.append(f"{label}: {val}")

        # Recent price history (5 days)
        try:
            hist = stock.history(period="5d")
            if not hist.empty:
                parts.append("\nRecent 5-day prices:")
                for date, row in hist.iterrows():
                    d = date.strftime("%m/%d")
                    parts.append(f"  {d}: O={row['Open']:.2f} H={row['High']:.2f} L={row['Low']:.2f} C={row['Close']:.2f} V={row['Volume']:,.0f}")
        except Exception:
            pass

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"yfinance error for {ticker}: {e}")
        return f"Ticker {ticker}: Error fetching data - {str(e)[:100]}"


def enrich_with_price_data(message: str) -> str:
    """If tickers are detected, fetch price data and append to message."""
    tickers = extract_tickers(message)
    if not tickers:
        return message

    price_blocks = []
    for ticker in tickers[:5]:  # max 5 tickers per message
        data = get_price_data(ticker)
        price_blocks.append(data)

    if price_blocks:
        enriched = message + "\n\n[PRICE_DATA]\n" + "\n---\n".join(price_blocks) + "\n[/PRICE_DATA]"
        return enriched

    return message


# ─── Helpers ───────────────────────────────────────────────
def is_authorized(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


def trim_history(history: list) -> list:
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


async def send_long_message(update: Update, text: str, parse_mode: str = "HTML"):
    if len(text) <= 4000:
        await update.message.reply_text(text, parse_mode=parse_mode)
        return

    chunks = []
    current = ""
    for part in text.split("\n\n"):
        if len(current) + len(part) + 2 > 4000:
            if current:
                chunks.append(current.strip())
            current = part
        else:
            current = current + "\n\n" + part if current else part
    if current:
        chunks.append(current.strip())

    for chunk in chunks:
        await update.message.reply_text(chunk, parse_mode=parse_mode)


def call_claude_with_search(messages: list) -> str:
    """Call Claude API with web search tool enabled."""
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
        ],
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})

        tool_result_blocks = []
        for block in response.content:
            if block.type == "server_tool_use":
                tool_result_blocks.append({
                    "type": "server_tool_result",
                    "tool_use_id": block.id,
                })

        if tool_result_blocks:
            messages.append({"role": "user", "content": tool_result_blocks})

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 5,
                }
            ],
            messages=messages,
        )

    result_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            result_parts.append(block.text)

    return "\n".join(result_parts) if result_parts else "No response generated."


# ─── Handlers ──────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    message = update.message.text
    history = conversations[user_id]

    # Enrich message with price data if tickers are detected
    enriched_message = enrich_with_price_data(message)

    # Store original message in history (not enriched)
    history.append({"role": "user", "content": message})
    history = trim_history(history)
    conversations[user_id] = history

    await update.message.chat.send_action("typing")

    try:
        # Build API messages: use enriched message for the latest one
        api_messages = []
        for msg in history[:-1]:
            api_messages.append(msg.copy())
        # Last message uses enriched version
        api_messages.append({"role": "user", "content": enriched_message})

        assistant_text = call_claude_with_search(api_messages)

        history.append({"role": "assistant", "content": assistant_text})
        conversations[user_id] = trim_history(history)

        tg_html = md_to_tg(assistant_text)
        await send_long_message(update, tg_html)

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        await update.message.reply_text(f"API error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        try:
            await update.message.reply_text(str(e)[:4000])
        except Exception:
            await update.message.reply_text("An error occurred. Please try again.")


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversations[user_id] = []
    await update.message.reply_text("Session cleared. Next message starts fresh.")


async def cmd_price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick price lookup: /price AAPL"""
    user_id = update.effective_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    args = context.args
    if not args:
        await update.message.reply_text("Usage: /price AAPL or /price 삼성전자")
        return

    query = " ".join(args).strip()
    tickers = extract_tickers(query)
    if not tickers:
        # Try as raw ticker
        tickers = [query.upper()]

    await update.message.chat.send_action("typing")

    results = []
    for ticker in tickers[:3]:
        data = get_price_data(ticker)
        results.append(data)

    text = "\n\n---\n\n".join(results)
    await update.message.reply_text(f"<pre>{html.escape(text)}</pre>", parse_mode="HTML")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    msg_count = len(conversations.get(user_id, []))
    await update.message.reply_text(
        f"User ID: {user_id}\n"
        f"Model: {CLAUDE_MODEL}\n"
        f"Web search: Enabled\n"
        f"Price data: Enabled (yfinance)\n"
        f"News priority: Bloomberg > Reuters > FT > WSJ\n"
        f"Messages in session: {msg_count}\n"
        f"Max history: {MAX_HISTORY}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Commands</b>\n\n"
        "/new - Start a new conversation\n"
        "/price AAPL - Quick price lookup\n"
        "/price 삼성전자 - Korean stock lookup\n"
        "/status - Show session info\n"
        "/help - Show this message\n\n"
        "<b>Features</b>\n"
        "• Real-time price data (stocks, crypto, indices, FX)\n"
        "• Web search (Bloomberg, Reuters priority)\n"
        "• Financial analysis & earnings review\n\n"
        "Just send any message to chat with Claude!",
        parse_mode="HTML",
    )


# ─── Main ──────────────────────────────────────────────────
def main():
    if not BOT_TOKEN:
        print("ERROR: Set TELEGRAM_BOT_TOKEN environment variable")
        return
    if not ANTHROPIC_API_KEY:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        return

    logger.info(f"Starting bot with model: {CLAUDE_MODEL}")
    logger.info("Web search: Enabled | Price data: Enabled | News: Bloomberg/Reuters priority")
    logger.info(f"Allowed users: {ALLOWED_USERS or 'Everyone'}")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running.")
    app.run_polling()


if __name__ == "__main__":
    main()
