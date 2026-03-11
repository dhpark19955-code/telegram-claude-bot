"""
Telegram bot powered by Anthropic Claude API.
Deployable on Railway / any cloud platform.

Setup:
1. pip install -r requirements.txt
2. Set environment variables:
   - TELEGRAM_BOT_TOKEN: from @BotFather
   - ANTHROPIC_API_KEY: from console.anthropic.com
   - ALLOWED_USERS: comma-separated Telegram user IDs (optional)
   - CLAUDE_MODEL: model name (default: claude-sonnet-4-20250514)
   - SYSTEM_PROMPT: custom system prompt (optional)
"""

import os
import html
import logging
from pathlib import Path
from collections import defaultdict

import anthropic
import mistune
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
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "50"))  # max messages per session

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You are a helpful assistant on Telegram. "
    "Keep responses concise and well-formatted. "
    "Use markdown for formatting when helpful."
))

# ─── Anthropic Client ─────────────────────────────────────
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ─── In-memory conversation history ───────────────────────
# {user_id: [{"role": "user"/"assistant", "content": "..."}]}
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
    """Convert markdown to Telegram-compatible HTML."""
    return md(text).strip()


# ─── Helpers ───────────────────────────────────────────────
def is_authorized(user_id: int) -> bool:
    return not ALLOWED_USERS or user_id in ALLOWED_USERS


def trim_history(history: list) -> list:
    """Keep conversation within MAX_HISTORY messages."""
    if len(history) > MAX_HISTORY:
        return history[-MAX_HISTORY:]
    return history


async def send_long_message(update: Update, text: str, parse_mode: str = "HTML"):
    """Send message, splitting if over Telegram's 4096 char limit."""
    if len(text) <= 4000:
        await update.message.reply_text(text, parse_mode=parse_mode)
        return

    # Split on double newlines to keep formatting intact
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


# ─── Handlers ──────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        await update.message.reply_text("Not authorized.")
        return

    message = update.message.text
    history = conversations[user_id]
    history.append({"role": "user", "content": message})
    history = trim_history(history)
    conversations[user_id] = history

    # Typing indicator
    await update.message.chat.send_action("typing")

    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=history,
        )

        assistant_text = response.content[0].text

        # Save assistant reply to history
        history.append({"role": "assistant", "content": assistant_text})
        conversations[user_id] = trim_history(history)

        # Convert and send
        tg_html = md_to_tg(assistant_text)
        await send_long_message(update, tg_html)

    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {e}")
        await update.message.reply_text(f"API error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Fallback: send raw text if HTML parsing fails
        try:
            await update.message.reply_text(assistant_text)
        except Exception:
            await update.message.reply_text("An error occurred. Please try again.")


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history."""
    user_id = update.effective_user.id
    conversations[user_id] = []
    await update.message.reply_text("Session cleared. Next message starts fresh.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status."""
    user_id = update.effective_user.id
    msg_count = len(conversations.get(user_id, []))
    await update.message.reply_text(
        f"User ID: {user_id}\n"
        f"Model: {CLAUDE_MODEL}\n"
        f"Messages in session: {msg_count}\n"
        f"Max history: {MAX_HISTORY}"
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help message."""
    await update.message.reply_text(
        "<b>Commands</b>\n\n"
        "/new - Start a new conversation\n"
        "/status - Show session info\n"
        "/help - Show this message\n\n"
        "Just send any text to chat with Claude!",
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
    logger.info(f"Allowed users: {ALLOWED_USERS or 'Everyone'}")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running.")
    app.run_polling()


if __name__ == "__main__":
    main()
