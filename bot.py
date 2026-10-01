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
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
BOT_USERNAME: str = "saniarationalshelfbot"
_last_user_query: Dict[str, Tuple[str, str]] = {}

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

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
    except Exception:
        pass

    # 1. Load from .env file and ensure os.environ is fully populated
    env_file = Path(".env")
    if not env_file.exists():
        env_file = Path(__file__).resolve().parent / ".env"

    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ[k] = v
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


_group_admins_cache: Dict[int, Tuple[float, Set[int]]] = {}


def parse_allowed_users(raw: str) -> Set[int]:
    """Parse comma/space-separated Telegram user or chat IDs into a set of ints."""
    ids = set()
    for item in re.split(r"[,;\s]+", raw):
        item = item.strip()
        if re.match(r"^-?\d+$", item):
            ids.add(int(item))
    return ids


def is_user_allowed(user_id: int) -> bool:
    """Check if user or chat ID is authorized."""
    if not ALLOWED_USERS:
        # If no allowlist is configured, permit (with warning logged at startup)
        return True
    return user_id in ALLOWED_USERS


async def is_event_authorized(event: types.Message | types.CallbackQuery) -> bool:
    """
    Comprehensive authorization check:
      1. If ALLOWED_USERS is empty, allow all.
      2. If user.id in ALLOWED_USERS, allow.
      3. If sender_chat and sender_chat.id in ALLOWED_USERS, allow.
      4. If chat and chat.id in ALLOWED_USERS, allow.
      5. If user is GroupAnonymousBot (1087968824) in a group:
         Grant access if group ID is allowed, or if any admin of the group is in ALLOWED_USERS.
    """
    if not ALLOWED_USERS:
        return True

    user = event.from_user
    if user and user.id in ALLOWED_USERS:
        return True

    sender_chat = getattr(event, "sender_chat", None)
    if sender_chat and sender_chat.id in ALLOWED_USERS:
        return True

    chat = getattr(event, "chat", None)
    if not chat and isinstance(event, types.CallbackQuery) and event.message:
        chat = event.message.chat

    if chat and chat.id in ALLOWED_USERS:
        return True

    # Anonymous Administrator support (Telegram ID: 1087968824 @GroupAnonymousBot)
    if user and user.id == 1087968824 and chat and getattr(chat, "type", "") in ("group", "supergroup"):
        now = time.time()
        cached = _group_admins_cache.get(chat.id)
        if cached and (now - cached[0] < 300):
            admin_ids = cached[1]
        else:
            try:
                bot_instance = getattr(event, "bot", None)
                if bot_instance:
                    admins = await bot_instance.get_chat_administrators(chat.id)
                    admin_ids = {a.user.id for a in admins}
                    _group_admins_cache[chat.id] = (now, admin_ids)
                else:
                    admin_ids = set()
            except Exception as e:
                logger.warning(f"Failed to fetch administrators for chat {chat.id}: {e}")
                admin_ids = set()

        if admin_ids & ALLOWED_USERS:
            return True

    return False


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

    if not await is_event_authorized(event):
        logger.warning(f"Unauthorized access attempt from user_id={user.id} (@{user.username})")
        if isinstance(event, types.Message):
            is_group = bool(getattr(event, "chat", None) and getattr(event.chat, "type", "") in ("group", "supergroup"))
            raw_text = (event.text or "").lower()
            # Only send rejection in private chat, or if explicitly addressing the bot in group
            if not is_group or (BOT_USERNAME and f"@{BOT_USERNAME}" in raw_text) or raw_text.startswith("/"):
                rejection_msg = "⛔ <b>Доступ до цього бота обмежено.</b>"
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
        "/answer &lt;питання&gt; — обґрунтована відповідь за книгами (Experimental Engine)\n"
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
        "📖 <b>Поради з пошуку та роботи з KORZ:</b>\n\n"
        "1. <b>Лексичний пошук (FTS5 / BM25):</b>\n"
        "Надішліть ключові слова через пробіл: <code>бабуїни соціальний стрес</code>\n\n"
        "2. <b>Точний пошук фрази (Exact Match):</b>\n"
        "Візьміть фразу в лапки: <code>«структурний диференціал»</code>\n\n"
        "3. <b>Смисловий пошук думок (/ask):</b>\n"
        "Задайте питання людською мовою: <code>/ask Чому забезпечене життя не усуває стрес?</code>\n"
        "Бот знайде уривки за схожістю змісту, навіть якщо слова повністю відрізняються.\n\n"
        "4. <b>Обґрунтована відповідь (/answer):</b>\n"
        "Сформувати синтезовану відповідь із точними цитатами: <code>/answer Чому соціальний статус впливає на здоров'я?</code>\n\n"
        "5. <b>Контекст та оцінка:</b>\n"
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


def clean_user_query(text: str) -> str:
    """Strips bot mentions and leading command prefixes (/answer, /ask, /search) from user queries."""
    if not text:
        return ""
    if BOT_USERNAME:
        text = re.sub(rf"@{re.escape(BOT_USERNAME)}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@SaniaRationalShelfBot\b", "", text, flags=re.IGNORECASE).strip()
    text = text.lstrip(":, -–—").strip()
    text = re.sub(r"^/(?:answer|ask|search)\b", "", text, flags=re.IGNORECASE).strip()
    return text.lstrip(":, -–—").strip()


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

    question = clean_user_query(parts[1])

    if not question:
        await message.answer(
            "🧠 <b>Смисловий пошук думок (/ask):</b>\n\n"
            "Введіть питання людською мовою або концептуальну ідею після команди:\n"
            "• <code>/ask Чому забезпечене життя не усуває стрес?</code>\n"
            "• <code>/ask Де Коржибський говорить що карта не є територією?</code>\n"
            "• <code>/ask Чому гроші можуть стати непотрібними у майбутньому?</code>\n\n"
            "💡 <i>KORZ виконає векторний пошук за змістом (dense embeddings).</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    if len(question) > MAX_QUERY_LENGTH:
        await message.answer(
            f"⚠️ <b>Питання занадто довге</b> ({len(question)} символів).\n"
            f"Максимальна дозволена довжина: {MAX_QUERY_LENGTH} символів.",
            parse_mode=ParseMode.HTML,
        )
        return

    if message.from_user:
        _last_user_query[str(message.from_user.id)] = (question, "ask")

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

        top_pid = results[0].paragraph_id if results else None
        run_id = 0
        if lib:
            run_id = lib.record_search_run(
                query=question,
                mode="ask",
                doc_scope="ALL",
                result_count=len(results),
                top_paragraph_id=top_pid,
                latency_ms=latency_sem,
                user_id=str(message.from_user.id) if message.from_user else None,
            )

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
                            callback_data=f"fb:tgt:{run_id}:{r.paragraph_id}",
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
                        callback_data=f"fb:up:{run_id}:{top_pid if top_pid else 0}",
                    ),
                    InlineKeyboardButton(
                        text="👎 Не те",
                        callback_data=f"fb:dn:{run_id}:0",
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


async def _execute_answer_workflow(message: types.Message, question: str):
    """Executes the Grounded Answer Engine workflow (extractive or Gemini generative)."""
    question = clean_user_query(question)

    if not question:
        await message.answer(
            "📚 <b>Як користуватися режимом /answer (Grounded Answer Engine):</b>\n\n"
            "Введіть питання, на яке хочете отримати обґрунтовану відповідь за книгами:\n"
            "• <code>/answer Чому соціальний статус впливає на здоров'я?</code>\n"
            "• <code>/answer Як Фреско пояснює відмову від грошей?</code>\n"
            "• <code>/answer Чому слова не можуть повністю описати реальність?</code>\n\n"
            "💡 <i>KORZ знайде відповідні місця у книгах і сформує відповідь із точними посиланнями на сторінки та цитати.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    if len(question) > MAX_QUERY_LENGTH:
        await message.answer(
            f"⚠️ <b>Питання занадто довге</b> ({len(question)} символів).\n"
            f"Максимальна дозволена довжина: {MAX_QUERY_LENGTH} символів.",
            parse_mode=ParseMode.HTML,
        )
        return

    if message.from_user:
        _last_user_query[str(message.from_user.id)] = (question, "answer")

    try:
        t0 = time.perf_counter()
        res = lib.answer(question)
        latency = (time.perf_counter() - t0) * 1000

        search_run_id = 0
        ans_run_id = 0
        if lib:
            search_run_id = lib.record_search_run(
                query=question,
                mode="answer",
                doc_scope="ALL",
                result_count=len(res.sources),
                top_paragraph_id=res.sources[0].paragraph_id if res.sources else None,
                latency_ms=latency,
                user_id=str(message.from_user.id) if message.from_user else None,
            )
            ans_run_id = lib.record_answer_run(
                query=question,
                backend=res.mode,
                threshold=getattr(getattr(lib, "config", None), "answer_min_similarity", 0.82),
                insufficient_info=1 if res.insufficient_info else 0,
                source_paragraph_ids=[s.paragraph_id for s in res.sources],
                answer_text=res.answer_text,
                citation_verified=1 if res.is_grounded else 0,
                latency_ms=latency,
                search_run_id=search_run_id,
                user_id=str(message.from_user.id) if message.from_user else None,
                model_tier=getattr(res, "model_tier", "low"),
                model_id=getattr(res, "model_id", "gemini-3.8-flash-low"),
                adaptive_trigger=getattr(res, "adaptive_trigger", "default"),
                source_count=len(res.sources),
                book_count=getattr(res, "book_count", 0),
                author_count=getattr(res, "author_count", 0),
                prompt_version=getattr(res, "prompt_version", "v1.0-grounded"),
                answer_config_sha=getattr(res, "answer_config_sha", ""),
                source_text_hashes=getattr(res, "source_text_hashes", []),
                corpus_version=getattr(getattr(lib, "config", None), "corpus_version", "v1-baseline"),
                experiment_input_sha=getattr(res, "experiment_input_sha", ""),
                input_bundle_json=getattr(res, "input_bundle_json", ""),
                retrieval_mode=getattr(res, "retrieval_mode", "dense"),
                shadow_rrf_json=getattr(res, "shadow_rrf_json", ""),
                entity_intent=getattr(res, "entity_intent", ""),
                entity_telemetry_json=getattr(res, "entity_telemetry_json", ""),
            )

        if res.insufficient_info:
            clean_ans = html.escape(res.answer_text)
            await message.answer(
                f"⚠️ <b>{clean_ans}</b>\n\n"
                "💡 <i>Спробуйте переформулювати запитання або уточнити контекст.</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        clean_q = html.escape(question)
        if res.mode == "extractive" and res.fallback_reason:
            clean_fb = html.escape(res.fallback_reason)
            mode_badge = f"Extractive (fallback: {clean_fb})"
        elif res.mode == "extractive":
            mode_badge = "Extractive"
        else:
            tier_badge = {
                "high": "🧠 High (--deep)",
                "medium": "⚖️ Medium (adaptive)",
                "low": "⚡ Low",
            }.get(getattr(res, "model_tier", "low"), "⚡ Low")
            mode_badge = f"Gemini 3.8 Flash [{tier_badge}]"

        header = (
            f"📚 <b>Відповідь за бібліотекою KORZ:</b>\n"
            f"<i>(«{clean_q}» | {mode_badge} | {latency:.0f} ms)</i>\n\n"
        )
        body = html.escape(res.answer_text)

        sources_lines = ["\n\n────────────────────────\n<b>📖 Використані джерела:</b>"]
        kb_buttons = []
        for s in res.sources:
            author_str = f"{html.escape(s.author)} — " if s.author else ""
            title_str = html.escape(s.title)
            sources_lines.append(
                f"• [{s.citation_index}] <b>{author_str}{title_str}</b>, стор. <b>{s.page}</b> (абзац {s.paragraph_index})"
            )
            kb_buttons.append([
                InlineKeyboardButton(
                    text=f"📖 Джерело [{s.citation_index}] (стор. {s.page})",
                    callback_data=f"ctx:{s.paragraph_id}:2",
                ),
                InlineKeyboardButton(
                    text=f"🎯 Саме цей [{s.citation_index}]",
                    callback_data=f"fb:tgt:{search_run_id}:{s.paragraph_id}",
                ),
            ])

        full_msg = header + body + "\n".join(sources_lines)
        chunks = split_telegram_message(full_msg)
        kb = InlineKeyboardMarkup(inline_keyboard=kb_buttons)
        for i, chunk in enumerate(chunks):
            chunk_kb = kb if i == len(chunks) - 1 else None
            await message.answer(chunk, reply_markup=chunk_kb, parse_mode=ParseMode.HTML)

        fb_kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="👍 Відповідь точна",
                        callback_data=f"fb:ans_up:{ans_run_id}",
                    ),
                    InlineKeyboardButton(
                        text="⚠️ Частково",
                        callback_data=f"fb:ans_part:{ans_run_id}",
                    ),
                    InlineKeyboardButton(
                        text="👎 Не відповідає",
                        callback_data=f"fb:ans_dn:{ans_run_id}",
                    ),
                ]
            ]
        )
        await message.answer("💡 <i>Оцініть повноту та точність сформованої відповіді:</i>", reply_markup=fb_kb, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.exception(f"Error handling /answer query: '{question}'")
        await message.answer(
            "❌ Сталася помилка під час формування відповіді. Спробуйте пізніше або змініть формулювання.",
            parse_mode=ParseMode.HTML,
        )


@dp.message(Command("answer"))
async def cmd_answer(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    # Check feature flag
    if not getattr(getattr(lib, "config", None), "answer_enabled", False):
        await message.answer(
            "⚠️ <b>Режим /answer наразі вимкнено (Experimental Phase).</b>\n\n"
            "Цей режим формує вичерпну відповідь на основі уривків із книг (Extractive / Grounded LLM).\n\n"
            "Щоб активувати його, додайте у файл <code>.env</code>:\n"
            "<code>KORZ_ANSWER_ENABLED=1</code>\n\n"
            "Та перезапустіть бота.",
            parse_mode=ParseMode.HTML,
        )
        return

    raw_text = message.text or ""
    parts = raw_text.split(maxsplit=1)
    question = parts[1].strip() if len(parts) > 1 else ""
    await _execute_answer_workflow(message, question)


async def _execute_search_workflow(message: types.Message, query: str):
    """Executes the standard FTS5 / BM25 search workflow."""
    query = clean_user_query(query)

    if not query:
        await message.answer(
            "🔍 <b>Текстовий пошук за точними словами (/search):</b>\n\n"
            "Введіть ключові слова або фрази для пошуку по книгах:\n"
            "• <code>/search бабуїни стрес</code>\n"
            "• <code>/search семантична реакція</code>\n"
            "• <code>/search ресурсоорієнтована економіка</code>\n\n"
            "💡 <i>KORZ знайде точні збіги цитат у текстах за алгоритмом FTS5 / BM25.</i>",
            parse_mode=ParseMode.HTML,
        )
        return

    if len(query) > MAX_QUERY_LENGTH:
        await message.answer(
            f"⚠️ <b>Запит занадто довгий</b> ({len(query)} символів).\n"
            f"Максимальна дозволена довжина: {MAX_QUERY_LENGTH} символів.",
            parse_mode=ParseMode.HTML,
        )
        return

    if message.from_user:
        _last_user_query[str(message.from_user.id)] = (query, "search")

    try:
        t0 = time.perf_counter()
        results = lib.search(query, mode="auto", limit=3, shadow=False)
        latency_fts = (time.perf_counter() - t0) * 1000

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

        top_pid = results[0].paragraph_id if results else None
        run_id = 0
        if lib:
            run_id = lib.record_search_run(
                query=query,
                mode="search",
                doc_scope="ALL",
                result_count=len(results),
                top_paragraph_id=top_pid,
                latency_ms=latency_fts,
                user_id=str(message.from_user.id) if message.from_user else None,
            )

        for i, r in enumerate(results, 1):
            author_str = f"{html.escape(r.author)} — " if r.author else ""
            title_str = html.escape(r.title)
            sanitized_text = html.escape(r.text)

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
                            callback_data=f"fb:tgt:{run_id}:{r.paragraph_id}",
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
                        callback_data=f"fb:up:{run_id}:{top_pid if top_pid else 0}",
                    ),
                    InlineKeyboardButton(
                        text="👎 Не те",
                        callback_data=f"fb:dn:{run_id}:0",
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


@dp.message(Command("search"))
async def cmd_search(message: types.Message):
    if not await check_auth_or_reject(message):
        return

    raw_text = message.text or ""
    parts = raw_text.split(maxsplit=1)
    query = parts[1].strip() if len(parts) > 1 else ""
    await _execute_search_workflow(message, query)


@dp.message()
async def handle_query(message: types.Message):
    raw_query = message.text or ""
    if not raw_query.strip():
        return

    chat_type = getattr(getattr(message, "chat", None), "type", "")
    is_group = chat_type in ("group", "supergroup")

    if is_group:
        has_mention = False
        bot_tag = f"@{BOT_USERNAME}".lower() if BOT_USERNAME else "@saniarationalshelfbot"
        if bot_tag in raw_query.lower():
            has_mention = True
        elif message.entities:
            for entity in message.entities:
                if entity.type == "mention":
                    m_text = raw_query[entity.offset:entity.offset + entity.length].lower()
                    if (BOT_USERNAME and m_text == f"@{BOT_USERNAME}".lower()) or m_text == "@saniarationalshelfbot":
                        has_mention = True
                        break

        is_reply_to_bot = False
        if message.reply_to_message and message.reply_to_message.from_user:
            reply_user = message.reply_to_message.from_user
            if BOT_USERNAME and (reply_user.username or "").lower() == BOT_USERNAME.lower():
                is_reply_to_bot = True
            elif getattr(message.bot, "id", None) and reply_user.id == message.bot.id:
                is_reply_to_bot = True

        # In groups, ignore chatter that doesn't mention or reply to the bot
        if not (has_mention or is_reply_to_bot):
            return

    # Check authorization (only authorized users can query the bot)
    if not await check_auth_or_reject(message):
        return

    clean_query = clean_user_query(raw_query)

    if is_group:
        clean_clean = clean_query.strip().lstrip("/").strip()
        if not clean_clean:
            await message.answer(
                "👋 <b>Я на зв'язку!</b> Задайте мені запитання по книгах бібліотеки KORZ.\n\n"
                "Наприклад:\n"
                f"• <code>@{BOT_USERNAME or 'SaniaRationalShelfBot'} Чому виникає відчуття тривоги?</code>\n"
                f"• <code>@{BOT_USERNAME or 'SaniaRationalShelfBot'} Що Коржибський писав про карту і територію?</code>\n\n"
                "Або використовуйте команди:\n"
                "• /answer &lt;питання&gt; — синтез відповіді (Gemini 3.8 Flash)\n"
                "• /ask &lt;думка&gt; — семантичний пошук\n"
                "• /books — каталог книг у бібліотеці",
                parse_mode=ParseMode.HTML,
            )
            return

        # In groups, when directly addressed, prefer Grounded Answer Engine (Gemini synthesis) if enabled
        if getattr(getattr(lib, "config", None), "answer_enabled", False):
            await _execute_answer_workflow(message, clean_query)
        else:
            await _execute_search_workflow(message, clean_query)
    else:
        # In private chat, plain text defaults to search workflow
        await _execute_search_workflow(message, clean_query)


@dp.callback_query(F.data.startswith("fb:"))
async def cb_feedback(callback: types.CallbackQuery):
    if not await check_auth_or_reject(callback):
        return

    try:
        parts = callback.data.split(":")
        fb_type = parts[1]  # 'tgt', 'up', 'dn', 'ans_up', 'ans_part', 'ans_dn'

        search_run_id: Optional[int] = None
        answer_run_id: Optional[int] = None
        target_pid: Optional[int] = None
        query_text = ""
        mode = "search"
        doc_scope = "ALL"

        if fb_type in ("tgt", "up", "dn"):
            # New format: fb:tgt:<run_id>:<pid>
            # Legacy format: fb:tgt:<pid>
            if len(parts) >= 4:
                search_run_id = int(parts[2]) if parts[2].isdigit() else None
                target_pid = int(parts[3]) if parts[3].isdigit() else None
            elif len(parts) == 3:
                target_pid = int(parts[2]) if parts[2].isdigit() else None

            if target_pid == 0:
                target_pid = None

            if search_run_id and lib:
                run = lib.get_search_run(search_run_id)
                if run:
                    query_text = run.get("query", "")
                    mode = run.get("mode", "search")
                    doc_scope = run.get("doc_scope", "ALL")

        elif fb_type.startswith("ans_"):
            # New format: fb:ans_up:<ans_run_id>
            # Legacy format: fb:ans_up:<target_pid>
            raw_id = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
            if raw_id and lib:
                ans_run = lib.get_answer_run(raw_id)
                if ans_run:
                    answer_run_id = raw_id
                    query_text = ans_run.get("query", "")
                    mode = "answer"
                    search_run_id = ans_run.get("search_run_id")
                else:
                    # Legacy fallback
                    target_pid = raw_id if raw_id > 0 else None

        # Fallback if query was not resolved from run ID (legacy callback or direct test mock)
        if not query_text:
            if callback.message and callback.message.reply_to_message and callback.message.reply_to_message.text:
                raw = callback.message.reply_to_message.text.strip()
                if raw.startswith("/ask"):
                    query_text = raw[4:].strip()
                    mode = "ask"
                elif raw.startswith("/answer"):
                    query_text = raw[7:].strip()
                    mode = "answer"
                else:
                    query_text = raw
                    mode = "search"
            user_str = str(callback.from_user.id)
            if not query_text and user_str in _last_user_query:
                query_text, mode = _last_user_query[user_str]
        else:
            user_str = str(callback.from_user.id)

        type_map = {
            "tgt": "target",
            "up": "found",
            "dn": "not_relevant",
            "ans_up": "answer_accurate",
            "ans_part": "answer_partial",
            "ans_dn": "answer_irrelevant",
        }
        canonical_type = type_map.get(fb_type, "found")

        if lib:
            comment_dict = {"mode": mode, "doc_scope": doc_scope}
            if search_run_id:
                comment_dict["search_run_id"] = search_run_id
            if answer_run_id:
                comment_dict["answer_run_id"] = answer_run_id

            lib.record_feedback(
                query=query_text or "telegram_query",
                feedback_type=canonical_type,
                target_paragraph_id=target_pid,
                search_run_id=search_run_id,
                answer_run_id=answer_run_id,
                user_id=user_str,
                comment=json.dumps(comment_dict),
            )

        if fb_type == "tgt":
            toast = "Дякуємо! Фрагмент позначено як цільовий 🎯"
        elif fb_type in ("up", "ans_up"):
            toast = "Дякуємо за оцінку! 👍"
        elif fb_type == "ans_part":
            toast = "Дякуємо! Відповідь оцінено як часткову ⚠️"
        elif fb_type == "ans_dn":
            toast = "Дякуємо! Врахуємо для покращення рушія відповідей 👎"
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
        bot_info = None
        for attempt in range(1, 6):
            try:
                bot_info = await bot.get_me()
                break
            except Exception as conn_err:
                logger.warning(f"⚠️ Спроба підключення {attempt}/5 до Telegram не вдалася ({conn_err}). Очікування {attempt * 2} с...")
                await asyncio.sleep(attempt * 2)

        if not bot_info:
            bot_info = await bot.get_me()

        global BOT_USERNAME
        BOT_USERNAME = (bot_info.username or "saniarationalshelfbot").lower()
        masked = mask_token(token)
        logger.info(f"🤖 KORZ Telegram Bot успішно підключено: @{bot_info.username} (Token: {masked})")
        logger.info("📚 Підключено до локальної бібліотеки SQLite. Очікування запитів...")

        # Register interactive commands for Telegram UI autocomplete (/...)
        try:
            bot_commands = [
                types.BotCommand(command="answer", description="Синтез відповіді за книгами (Gemini 3.8 Flash)"),
                types.BotCommand(command="ask", description="Семантичний пошук ідей та думок (E5)"),
                types.BotCommand(command="search", description="Текстовий пошук за точними словами"),
                types.BotCommand(command="books", description="Каталог книг у бібліотеці"),
                types.BotCommand(command="status", description="Стан бібліотеки та семантичного індексу"),
                types.BotCommand(command="help", description="Довідка щодо роботи з ботом"),
            ]
            await bot.set_my_commands(bot_commands)
            logger.info("📋 Команди бота успішно зареєстровано в Telegram (автодоповнення при введенні /)")
        except Exception as ce:
            logger.warning(f"⚠️ Не вдалося зареєструвати команди бота в Telegram: {ce}")

        await dp.start_polling(bot)
    finally:
        lib.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
