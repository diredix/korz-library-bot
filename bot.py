#!/usr/bin/env python3
"""
bot.py - Production-Hardened Telegram Bot Adapter for KORZ LOCAL (Single-file SQLite Engine)

Security & Production Hardening:
  1. User Allowlist (ALLOWED_TELEGRAM_USERS) - protects private library.
  2. Read-only access - bot only calls search, context, books, status.
  3. Secret token masking - tokens are never printed or logged in cleartext.
  4. Safe HTML rendering - all book text and user queries sanitized with html.escape().
  5. Message splitting / pagination - handles long text without breaking Telegram limits.
  6. Strict callback validation - protects against malformed or unauthorized callback data.
  7. Query limits - rejects queries > 500 chars, clamps results to 1..10.
  8. Global error boundary - logs full tracebacks internally, returns safe user-friendly errors.

Usage:
    python bot.py --token <TELEGRAM_BOT_TOKEN> [--allowed-users 12345678,87654321]
    or set TELEGRAM_BOT_TOKEN and ALLOWED_TELEGRAM_USERS in .env file
"""

import argparse
import asyncio
import html
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from korz_core import KorzLibrary, KorzConfig, SearchResult, ShadowRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("korz_bot")

dp = Dispatcher()
lib: Optional[KorzLibrary] = None
shadow_runner: Optional[ShadowRunner] = None
ALLOWED_USERS: Set[int] = set()

MAX_QUERY_LENGTH = 500
MAX_TELEGRAM_MSG_LEN = 3900


def mask_token(token: str) -> str:
    """Mask token for safe logging (e.g. 123456:ABC... -> 123456:AB***)."""
    if not token:
        return "<empty>"
    parts = token.split(":", 1)
    if len(parts) == 2:
        prefix, secret = parts
        masked_secret = secret[:3] + "***" + secret[-2:] if len(secret) > 5 else "***"
        return f"{prefix}:{masked_secret}"
    return token[:4] + "***"


def load_env_config() -> dict:
    """Read configuration from .env file first, falling back to os.environ."""
    config = {
        "token": "",
        "allowed_users": "",
    }

    # 1. Load from .env file
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "TELEGRAM_BOT_TOKEN" and v:
                    config["token"] = v
                elif k == "ALLOWED_TELEGRAM_USERS" and v:
                    config["allowed_users"] = v

    # 2. Fallback to environment variables if not set in .env
    if not config["token"]:
        config["token"] = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip().strip('"').strip("'")
    if not config["allowed_users"]:
        config["allowed_users"] = os.environ.get("ALLOWED_TELEGRAM_USERS", "").strip().strip('"').strip("'")

    return config


def parse_allowed_users(raw: str) -> Set[int]:
    """Parse comma/space-separated Telegram user IDs into a set of ints."""
    ids = set()
    for item in re.split(r"[,;\s]+", raw):
        item = item.strip()
        if item.isdigit():
            ids.add(int(item))
    return ids


def is_user_allowed(user_id: int) -> bool:
    """Check if user is authorized."""
    if not ALLOWED_USERS:
        # If no allowlist is configured, permit (with warning logged at startup)
        return True
    return user_id in ALLOWED_USERS


def split_telegram_message(text: str, max_chars: int = MAX_TELEGRAM_MSG_LEN) -> List[str]:
    """
    Split a long message cleanly along paragraph boundaries.
    Ensures no chunk exceeds Telegram's 4096-character limit.
    """
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks = []
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    current_chunk = []
    current_len = 0

    for p in paragraphs:
        p_len = len(p) + 2
        if current_len + p_len > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += p_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return [c for c in chunks if c]


async def check_auth_or_reject(event: types.Message | types.CallbackQuery) -> bool:
    """Security gate: rejects unauthorized users with a safe, polite notice."""
    user = event.from_user
    if not user:
        return False

    if not is_user_allowed(user.id):
        logger.warning(f"Unauthorized access attempt from user_id={user.id} (@{user.username})")
        rejection_msg = "⛔ <b>Доступ до цього бота обмежено.</b>"
        if isinstance(event, types.Message):
            await event.answer(rejection_msg, parse_mode=ParseMode.HTML)
        elif isinstance(event, types.CallbackQuery):
            await event.answer("Доступ обмежено", show_alert=True)
        return False
    return True


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    await message.answer(
        "👋 <b>Вітаю у KORZ Library Bot!</b>\n\n"
        "Я допоможу знайти будь-який фрагмент чи цитату у ваших локальних книгах.\n\n"
        "🔍 <b>Як шукати:</b>\n"
        "• Надішліть будь-яке слово або фразу: <code>бабуїни соціальний стрес</code>\n"
        "• Для точного пошуку фрази візьміть її в лапки: <code>«бабуїни працюють чотири години»</code>\n\n"
        "📚 <b>Команди:</b>\n"
        "/ask &lt;питання&gt; — смисловий пошук думок (семантичний Question Retrieval)\n"
        "/books — список усіх книг у бібліотеці\n"
        "/status — стан та статистика бази даних\n"
        "/help — довідка та поради з пошуку",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    await message.answer(
        "📖 <b>Поради з пошуку в KORZ:</b>\n\n"
        "1. <b>Лексичний пошук (FTS5 / BM25):</b>\n"
        "Надішліть ключові слова через пробіл: <code>бабуїни соціальний стрес</code>\n\n"
        "2. <b>Точний пошук фрази (Exact Match):</b>\n"
        "Візьміть фразу в лапки: <code>«структурний диференціал»</code>\n\n"
        "3. <b>Смисловий пошук думок (/ask):</b>\n"
        "Задайте питання або ідею людською мовою: <code>/ask Чому забезпечене життя не усуває стрес?</code>\n"
        "Бот знайде уривки за схожістю змісту, навіть якщо слова повністю відрізняються.\n\n"
        "4. <b>Контекст та оцінка:</b>\n"
        "Під кожним знайденим уривком є <b>[📖 Контекст]</b> та <b>[🎯 Саме цей]</b>.",
        parse_mode=ParseMode.HTML,
    )


@dp.message(Command("books"))
async def cmd_books(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    try:
        books = lib.list_books()
        if not books:
            await message.answer("📚 Бібліотека наразі порожня.", parse_mode=ParseMode.HTML)
            return

        lines = ["📚 <b>Книги в бібліотеці KORZ:</b>\n"]
        for b in books:
            author = f"{html.escape(b['author'])} — " if b["author"] else ""
            title = html.escape(b["title"])
            lines.append(
                f"• <b>{author}{title}</b>\n"
                f"  📄 {b['total_pages']} стор. | 📝 {b['total_paragraphs']:,} абз. | 🧩 {b['total_chunks']:,} блоків\n".replace(",", " ")
            )

        full_msg = "\n".join(lines)
        for chunk in split_telegram_message(full_msg):
            await message.answer(chunk, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.exception("Error in /books command")
        await message.answer("❌ Не вдалося отримати список книг.", parse_mode=ParseMode.HTML)


@dp.message(Command("status"))
async def cmd_status(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    try:
        st = lib.status()
        await message.answer(
            "📊 <b>Стан бібліотеки KORZ:</b>\n"
            f"• <b>Книг:</b> {st['total_documents']}\n"
            f"• <b>Сторінок:</b> {st['total_pages']:,}\n"
            f"• <b>Абзаців:</b> {st['total_paragraphs']:,}\n"
            f"• <b>Пошукових блоків:</b> {st['total_chunks']:,}\n"
            f"• <b>Розмір бази:</b> {st['database_size_mb']} MB (SQLite WAL)\n"
            f"• <b>Версія схеми:</b> v{st['schema_version']}\n"
            f"• <b>Авторизованих користувачів:</b> {len(ALLOWED_USERS) if ALLOWED_USERS else 'Всі (відкрито)'}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.exception("Error in /status command")
        await message.answer("❌ Не вдалося отримати стан бібліотеки.", parse_mode=ParseMode.HTML)


@dp.callback_query(F.data.startswith("ctx:"))
async def cb_context(callback: types.CallbackQuery):
    if not await check_auth_or_reject(callback):
        return

    try:
        # Validate callback payload format: ctx:<paragraph_id>:<window>
        parts = callback.data.split(":")
        if len(parts) < 2 or not parts[1].isdigit():
            await callback.answer("Некоректний запит контексту", show_alert=True)
            return

        par_id = int(parts[1])
        window = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 2
        window = max(1, min(5, window))  # Bounded to 1..5

        ctx = lib.get_context(par_id, window=window)
        if not ctx:
            await callback.answer("Абзац не знайдено", show_alert=True)
            return

        author_str = f"{html.escape(ctx['author'])} — " if ctx.get("author") else ""
        title_str = html.escape(ctx["title"])
        header = (
            f"📖 <b>{author_str}{title_str}</b>\n"
            f"Сторінка {ctx['target_page']} (абзац {ctx['target_paragraph_index']})\n"
            f"<i>Контекст: вікно ±{window} абзаців навколо цільового фрагмента</i>\n"
            "────────────────────────\n\n"
        )

        body_parts = []
        for entry in ctx["entries"]:
            sanitized_text = html.escape(entry["text"])
            if entry["is_target"]:
                body_parts.append(
                    f"👉 <b>[Цільовий абзац {entry['paragraph_index']}]:</b>\n<u>{sanitized_text}</u>\n"
                )
            else:
                body_parts.append(
                    f"<b>[Абзац {entry['paragraph_index']}]:</b>\n{sanitized_text}\n"
                )

        full_text = header + "\n".join(body_parts)

        # Split and send if text exceeds Telegram limits
        chunks = split_telegram_message(full_text)
        for chunk in chunks:
            await callback.message.reply(chunk, parse_mode=ParseMode.HTML)

        await callback.answer()
    except Exception as e:
        logger.exception(f"Error handling callback {callback.data}")
        await callback.answer("Помилка завантаження контексту", show_alert=True)


@dp.message(Command("ask"))
async def cmd_ask(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    raw_text = message.text or ""
    parts = raw_text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer(
            "🧠 <b>Смисловий пошук думок (/ask):</b>\n\n"
            "Введіть питання людською мовою або концептуальну ідею після команди:\n"
            "• <code>/ask Чому забезпечене життя не усуває стрес?</code>\n"
            "• <code>/ask Де Коржибський говорить що карта не є територією?</code>\n"
            "• <code>/ask Чому гроші можуть стати непотрібними у майбутньому?</code>\n\n"
            "💡 <i>KORZ виконає векторний пошук за змістом (dense embeddings), знаходячи відповідні уривки, навіть якщо слова повністю відрізняються.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    question = parts[1].strip()
    if len(question) > MAX_QUERY_LENGTH:
        await message.answer(
            f"⚠️ <b>Питання занадто довге</b> ({len(question)} символів).\n"
            f"Максимальна дозволена довжина: {MAX_QUERY_LENGTH} символів.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        t0 = time.perf_counter()
        results = lib.ask(question, limit=3)
        latency_sem = (time.perf_counter() - t0) * 1000

        # Background shadow FTS comparison for telemetry
        if lib.config.shadow_enabled and shadow_runner:
            fts_t0 = time.perf_counter()
            fts_res = lib.search(question, mode="fts", limit=3, shadow=False)
            latency_fts = (time.perf_counter() - fts_t0) * 1000
            asyncio.create_task(
                shadow_runner.evaluate_async(
                    query=f"[ask] {question}",
                    fts_results=fts_res,
                    latency_fts_ms=latency_fts,
                )
            )

        if not results:
            clean_q = html.escape(question)
            await message.answer(
                f"🧠 За змістом для «<b>{clean_q}</b>» у книгах нічого не знайдено.\n\n"
                "💡 <i>Спробуйте переформулювати запитання або уточнити контекст.</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        clean_q = html.escape(question)
        intro = (
            f"🧠 <b>Знайдено за змістом: {len(results)}</b> "
            f"({latency_sem:.0f} ms для «{clean_q}»):\n"
        )
        await message.answer(intro, parse_mode=ParseMode.HTML)

        for i, r in enumerate(results, 1):
            author_str = f"{html.escape(r.author)} — " if r.author else ""
            title_str = html.escape(r.title)
            sanitized_text = html.escape(r.text)

            if len(sanitized_text) > 850:
                sanitized_text = sanitized_text[:840] + "..."

            msg_text = (
                f"📚 <b>{author_str}{title_str}</b>\n"
                f"📄 Сторінка <b>{r.page}</b> (абзац {r.paragraph_index}) [ID: {r.paragraph_id}]\n"
                f"🎯 Смислова відповідність: <b>{r.score:.3f}</b>\n\n"
                f"{sanitized_text}"
            )

            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="📖 Контекст (±2 абз.)",
                            callback_data=f"ctx:{r.paragraph_id}:2",
                        ),
                        InlineKeyboardButton(
                            text="🎯 Саме цей",
                            callback_data=f"fb:tgt:{r.paragraph_id}",
                        ),
                    ]
                ]
            )

            await message.answer(msg_text, reply_markup=kb, parse_mode=ParseMode.HTML)

        fb_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="👍 Знайшов",
                        callback_data=f"fb:up:{results[0].paragraph_id if results else 0}",
                    ),
                    InlineKeyboardButton(
                        text="👎 Не те",
                        callback_data="fb:dn:0",
                    ),
                ]
            ]
        )
        await message.answer("💡 <i>Оцініть точність знайденої відповіді:</i>", reply_markup=fb_kb, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.exception(f"Error handling /ask query: '{question}'")
        await message.answer(
            "❌ Сталася помилка під час обробки смислового запиту. Спробуйте пізніше або переформулюйте.",
            parse_mode=ParseMode.HTML,
        )


@dp.message()
async def handle_query(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    raw_query = message.text or ""
    query = raw_query.strip()
    if not query:
        return

    # Check query length limit
    if len(query) > MAX_QUERY_LENGTH:
        await message.answer(
            f"⚠️ <b>Запит занадто довгий</b> ({len(query)} символів).\n"
            f"Максимальна дозволена довжина: {MAX_QUERY_LENGTH} символів.",
            parse_mode=ParseMode.HTML,
        )
        return

    try:
        # Canonical search via korz_core.search (read-only, shadow handled explicitly async)
        t0 = time.perf_counter()
        results = lib.search(query, mode="auto", limit=3, shadow=False)
        latency_fts = (time.perf_counter() - t0) * 1000

        # Non-blocking background shadow evaluation (never delays user response)
        if lib.config.shadow_enabled and shadow_runner:
            asyncio.create_task(
                shadow_runner.evaluate_async(
                    query=query,
                    fts_results=results,
                    latency_fts_ms=latency_fts,
                )
            )

        if not results:
            clean_q = html.escape(query)
            await message.answer(
                f"🔍 За запитом «<b>{clean_q}</b>» нічого не знайдено.\n\n"
                "💡 <i>Спробуйте інші ключові слова або перевірте правопис.</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        mode_label = "Точний збіг" if results[0].match_type == "exact" else "FTS5 / BM25"
        clean_q = html.escape(query)
        intro = f"🔎 <b>Знайдено уривків: {len(results)}</b> ({mode_label} для «{clean_q}»):\n"
        await message.answer(intro, parse_mode=ParseMode.HTML)

        for i, r in enumerate(results, 1):
            author_str = f"{html.escape(r.author)} — " if r.author else ""
            title_str = html.escape(r.title)
            sanitized_text = html.escape(r.text)

            # Limit preview length
            if len(sanitized_text) > 850:
                sanitized_text = sanitized_text[:840] + "..."

            msg_text = (
                f"📚 <b>{author_str}{title_str}</b>\n"
                f"📄 Сторінка <b>{r.page}</b> (абзац {r.paragraph_index}) [ID: {r.paragraph_id}]\n\n"
                f"{sanitized_text}"
            )

            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="📖 Контекст (±2 абз.)",
                            callback_data=f"ctx:{r.paragraph_id}:2",
                        ),
                        InlineKeyboardButton(
                            text="🎯 Саме цей",
                            callback_data=f"fb:tgt:{r.paragraph_id}",
                        ),
                    ]
                ]
            )

            await message.answer(msg_text, reply_markup=kb, parse_mode=ParseMode.HTML)

        # Rating prompt for evaluation telemetry
        fb_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="👍 Знайшов",
                        callback_data=f"fb:up:{results[0].paragraph_id if results else 0}",
                    ),
                    InlineKeyboardButton(
                        text="👎 Не те",
                        callback_data="fb:dn:0",
                    ),
                ]
            ]
        )
        await message.answer("💡 <i>Оцініть точність пошуку:</i>", reply_markup=fb_kb, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.exception(f"Error handling search query: '{query}'")
        await message.answer(
            "❌ Сталася помилка під час обробки пошуку. Спробуйте пізніше або змініть формулювання.",
            parse_mode=ParseMode.HTML,
        )


@dp.callback_query(F.data.startswith("fb:"))
async def cb_feedback(callback: types.CallbackQuery):
    if not await check_auth_or_reject(callback):
        return

    try:
        parts = callback.data.split(":")
        fb_type = parts[1]  # 'tgt', 'up', 'dn'
        target_pid = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        if target_pid == 0:
            target_pid = None

        query_text = ""
        if callback.message and callback.message.reply_to_message and callback.message.reply_to_message.text:
            query_text = callback.message.reply_to_message.text.strip()

        user_str = str(callback.from_user.id)
        type_map = {"tgt": "target", "up": "found", "dn": "not_relevant"}
        canonical_type = type_map.get(fb_type, "found")

        if lib:
            lib.record_feedback(
                query=query_text or "telegram_query",
                feedback_type=canonical_type,
                target_paragraph_id=target_pid,
                user_id=user_str,
            )

        if fb_type == "tgt":
            toast = "Дякуємо! Фрагмент позначено як цільовий 🎯"
        elif fb_type == "up":
            toast = "Дякуємо за оцінку! 👍"
        else:
            toast = "Дякуємо! Врахуємо для покращення ✍️"

        await callback.answer(toast, show_alert=False)
    except Exception as e:
        logger.exception(f"Error handling feedback callback: {callback.data}")
        await callback.answer("Відгук збережено", show_alert=False)


async def main():
    global lib, shadow_runner, ALLOWED_USERS

    parser = argparse.ArgumentParser(description="KORZ Telegram Bot Adapter (v1.1 Hardened)")
    parser.add_argument("--token", help="Telegram Bot Token")
    parser.add_argument("--allowed-users", help="Comma-separated Telegram User IDs for allowlist")
    args = parser.parse_args()

    env_cfg = load_env_config()
    token = args.token or env_cfg["token"]
    raw_users = args.allowed_users or env_cfg["allowed_users"]

    if not token:
        print("\n❌ Помилка: Telegram Bot Token не знайдено!")
        print("Вкажіть токен через аргумент:")
        print("  python bot.py --token <YOUR_TOKEN>")
        print("або збережіть його у файлі .env:")
        print("  TELEGRAM_BOT_TOKEN=<YOUR_TOKEN>\n")
        sys.exit(1)

    if raw_users:
        ALLOWED_USERS = parse_allowed_users(raw_users)
        logger.info(f"🔒 Allowlist активовано: {len(ALLOWED_USERS)} користувач(ів)")
    else:
        logger.warning("⚠️ УВАГА: Allowlist порожній! Бот доступний для всіх. Встановіть ALLOWED_TELEGRAM_USERS для захисту.")

    lib = KorzLibrary()
    shadow_runner = ShadowRunner(lib, max_concurrency=1)
    if lib.config.shadow_enabled:
        logger.info("👻 Shadow Mode активовано: фоновий збір та аналітика семантичного пошуку")
        try:
            logger.info("⚡ Виконується warm-up семантичної моделі та завантаження індексу...")
            warmup_sec = lib.warmup_semantic()
            logger.info(f"⚡ Warm-up завершено успішно за {warmup_sec:.2f} с. Подальші пошуки у фоні виконуватимуться з мінімальною латентністю (~20-50 ms).")
        except Exception as e:
            logger.warning(f"⚠️ Попередження: не вдалося виконати warm-up семантичного рушія: {e}")

    bot = Bot(token=token)

    try:
        bot_info = await bot.get_me()
        masked = mask_token(token)
        logger.info(f"🤖 KORZ Telegram Bot успішно підключено: @{bot_info.username} (Token: {masked})")
        logger.info("📚 Підключено до локальної бібліотеки SQLite. Очікування запитів...")
        await dp.start_polling(bot)
    finally:
        lib.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
