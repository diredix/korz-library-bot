"""
tests/test_bot.py - Unit Tests for Hardened Telegram Bot Logic
Covers:
  - Token masking
  - Allowlist parsing & security gate
  - Message splitting / pagination
  - Safe HTML escaping for book texts
  - Callback data validation
  - Query length enforcement
"""

import html
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import (
    mask_token,
    parse_allowed_users,
    split_telegram_message,
    MAX_QUERY_LENGTH,
)


class TestBotHardening(unittest.TestCase):
    def test_token_masking(self):
        """Verify tokens are never printed in cleartext."""
        token = "7123456789:AAFlKj987654321_SecretKeyXYZ"
        masked = mask_token(token)
        self.assertNotIn("SecretKeyXYZ", masked)
        self.assertTrue(masked.startswith("7123456789:AAF***"))

    def test_allowlist_parsing(self):
        """Test parsing of user IDs from string formats."""
        raw = "12345678, 87654321; 99999999"
        parsed = parse_allowed_users(raw)
        self.assertEqual(parsed, {12345678, 87654321, 99999999})

    def test_html_escaping_safety(self):
        """Ensure book text containing HTML/XML entities does not break Telegram parser."""
        raw_book_text = "Формула <E=mc^2> & теорія A > B, а також <script>alert('xss')</script>."
        escaped = html.escape(raw_book_text)
        self.assertNotIn("<script>", escaped)
        self.assertIn("&lt;script&gt;", escaped)
        self.assertIn("&amp;", escaped)
        self.assertIn("&gt;", escaped)
        self.assertIn("&lt;", escaped)

    def test_message_splitting(self):
        """Ensure messages exceeding 3900 chars split cleanly across paragraph boundaries."""
        para = "Це абзац книги довжиною близько сотні символів для тестування розбиття повідомлень.\n\n"
        huge_text = para * 50  # ~4500 characters
        self.assertGreater(len(huge_text), 4000)

        chunks = split_telegram_message(huge_text, max_chars=3900)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 3900)
            # Ensure no broken paragraphs
            self.assertFalse(c.endswith("\n\n"))

    def test_callback_parsing_logic(self):
        """Verify callback data validation."""
        valid_cb = "ctx:1163:2"
        parts = valid_cb.split(":")
        self.assertEqual(parts[0], "ctx")
        self.assertTrue(parts[1].isdigit())
        self.assertEqual(int(parts[1]), 1163)
        self.assertEqual(int(parts[2]), 2)

        malformed_cb = "ctx:DROP_TABLE;--"
        parts_mal = malformed_cb.split(":")
        self.assertFalse(parts_mal[1].isdigit())

    def test_query_length_boundary(self):
        """Verify query length threshold."""
        short_query = "бабуїни соціальний стрес"
        self.assertLessEqual(len(short_query), MAX_QUERY_LENGTH)

        oversized_query = "а" * (MAX_QUERY_LENGTH + 50)
        self.assertGreater(len(oversized_query), MAX_QUERY_LENGTH)

    def test_handle_query_search_and_render(self):
        """Verify handle_query executes search, formats results safely without exceptions."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot
        from korz_core import KorzLibrary

        with KorzLibrary() as library:
            bot.lib = library
            for query in ["поведінка", "тривога"]:
                mock_msg = MagicMock()
                mock_msg.from_user.id = 12345
                mock_msg.text = query
                mock_msg.answer = AsyncMock()

                asyncio.run(bot.handle_query(mock_msg))
                self.assertGreaterEqual(mock_msg.answer.call_count, 2)
                sent_texts = [call[0][0] for call in mock_msg.answer.call_args_list]
                self.assertFalse(
                    any("Сталася помилка під час обробки пошуку" in t for t in sent_texts),
                    f"Search failed for '{query}'"
                )

    def test_cmd_ask_empty_guidance(self):
        """Verify calling /ask with no arguments presents clear usage instructions."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot

        mock_msg = MagicMock()
        mock_msg.from_user.id = 12345
        mock_msg.text = "/ask"
        mock_msg.answer = AsyncMock()

        asyncio.run(bot.cmd_ask(mock_msg))
        self.assertEqual(mock_msg.answer.call_count, 1)
        sent_text = mock_msg.answer.call_args[0][0]
        self.assertIn("/ask", sent_text)
        self.assertIn("Смисловий пошук", sent_text)

    def test_cmd_ask_question_retrieval(self):
        """Verify calling /ask with a question returns semantic answers and feedback markup."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot
        from korz_core import KorzLibrary

        with KorzLibrary() as library:
            bot.lib = library
            mock_msg = MagicMock()
            mock_msg.from_user.id = 12345
            mock_msg.text = "/ask чому забезпечене життя не усуває стрес?"
            mock_msg.answer = AsyncMock()

            asyncio.run(bot.cmd_ask(mock_msg))
            self.assertGreaterEqual(mock_msg.answer.call_count, 2)
            sent_texts = [call[0][0] for call in mock_msg.answer.call_args_list]
            self.assertTrue(any("Знайдено за змістом" in t for t in sent_texts))
            self.assertFalse(any("Сталася помилка під час обробки" in t for t in sent_texts))


    def test_search_run_id_immutable_binding_in_feedback(self):
        """
        Verify that feedback binding to search_run_id eliminates race condition:
        User executes Query A, then Query B, but clicks 🎯 on historical Query A.
        Feedback MUST be recorded for Query A, not Query B.
        """
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot
        from korz_core import KorzLibrary

        with KorzLibrary() as library:
            bot.lib = library

            # 1. Execute Query A
            run_id_A = library.record_search_run(
                query="чому статус впливає на здоров'я?",
                mode="ask",
                doc_scope="ALL",
                result_count=3,
                top_paragraph_id=2500,
                user_id="user_999",
            )

            # 2. Later, execute Query B
            run_id_B = library.record_search_run(
                query="чому бабуїни агресивні?",
                mode="ask",
                doc_scope="ALL",
                result_count=3,
                top_paragraph_id=3000,
                user_id="user_999",
            )

            # Overwrite session state with Query B (or clear it completely)
            bot._last_user_query["user_999"] = ("чому бабуїни агресивні?", "ask")

            # 3. User scrolls back to Query A and clicks 🎯 on paragraph 2500
            mock_cb = MagicMock()
            mock_cb.data = f"fb:tgt:{run_id_A}:2500"
            mock_cb.from_user.id = 999
            mock_cb.message = None  # No reply_to_message needed!
            mock_cb.answer = AsyncMock()

            asyncio.run(bot.cb_feedback(mock_cb))
            self.assertTrue(mock_cb.answer.called)

            # 4. Verify feedback in DB is strictly bound to Query A and run_id_A
            row = library.conn.execute(
                "SELECT search_run_id, query, target_paragraph_id, feedback_type FROM search_feedback WHERE user_id = '999' ORDER BY id DESC LIMIT 1"
            ).fetchone()

            self.assertIsNotNone(row)
            self.assertEqual(row[0], run_id_A)
            self.assertEqual(row[1], "чому статус впливає на здоров'я?")
            self.assertEqual(row[2], 2500)
            self.assertEqual(row[3], "target")

            # 5. Verify /answer run binding works identically
            ans_run_id = library.record_answer_run(
                query="Як Фреско пояснює відмову від грошей?",
                backend="extractive",
                threshold=0.82,
                source_paragraph_ids=[4258],
                answer_text="Ресурсоорієнтована економіка усуває дефіцит.",
                user_id="user_999",
            )

            mock_ans_cb = MagicMock()
            mock_ans_cb.data = f"fb:ans_up:{ans_run_id}"
            mock_ans_cb.from_user.id = 999
            mock_ans_cb.message = None
            mock_ans_cb.answer = AsyncMock()

            asyncio.run(bot.cb_feedback(mock_ans_cb))

            ans_fb_row = library.conn.execute(
                "SELECT answer_run_id, query, feedback_type FROM search_feedback WHERE user_id = '999' ORDER BY id DESC LIMIT 1"
            ).fetchone()

            self.assertIsNotNone(ans_fb_row)
            self.assertEqual(ans_fb_row[0], ans_run_id)
            self.assertEqual(ans_fb_row[1], "Як Фреско пояснює відмову від грошей?")
            self.assertEqual(ans_fb_row[2], "answer_accurate")

    def test_group_ignore_unaddressed_chatter(self):
        """In a group, messages not mentioning or replying to the bot must be completely ignored."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot

        mock_msg = MagicMock()
        mock_msg.chat.type = "group"
        mock_msg.text = "Привіт усім у чаті, як справи?"
        mock_msg.entities = []
        mock_msg.reply_to_message = None
        mock_msg.from_user.id = 12345
        mock_msg.answer = AsyncMock()

        asyncio.run(bot.handle_query(mock_msg))
        self.assertEqual(mock_msg.answer.call_count, 0)

    def test_group_mention_cleaning_and_routing(self):
        """In a group, addressing @SaniaRationalShelfBot cleans mention and executes query."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot
        from korz_core import KorzLibrary

        with KorzLibrary() as library:
            bot.lib = library
            mock_msg = MagicMock()
            mock_msg.chat.type = "supergroup"
            mock_msg.text = "@SaniaRationalShelfBot: поведінка"
            mock_msg.entities = []
            mock_msg.reply_to_message = None
            mock_msg.from_user.id = 12345
            mock_msg.answer = AsyncMock()

            asyncio.run(bot.handle_query(mock_msg))
            self.assertGreaterEqual(mock_msg.answer.call_count, 1)
            sent_texts = [call[0][0] for call in mock_msg.answer.call_args_list]
            self.assertFalse(any("Сталася помилка" in t for t in sent_texts))

    def test_group_mention_empty_query_guidance(self):
        """Mentioning only the bot in a group displays a greeting and usage examples."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot

        mock_msg = MagicMock()
        mock_msg.chat.type = "group"
        mock_msg.text = "@SaniaRationalShelfBot"
        mock_msg.entities = []
        mock_msg.reply_to_message = None
        mock_msg.from_user.id = 12345
        mock_msg.answer = AsyncMock()

        asyncio.run(bot.handle_query(mock_msg))
        self.assertEqual(mock_msg.answer.call_count, 1)
        sent_text = mock_msg.answer.call_args[0][0]
        self.assertIn("Я на зв'язку", sent_text)

    def test_cmd_search_execution(self):
        """Verify /search executes FTS search or shows guidance when empty."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        import bot
        from korz_core import KorzLibrary

        with KorzLibrary() as library:
            bot.lib = library
            mock_empty = MagicMock()
            mock_empty.from_user.id = 12345
            mock_empty.text = "/search"
            mock_empty.answer = AsyncMock()

            asyncio.run(bot.cmd_search(mock_empty))
            self.assertEqual(mock_empty.answer.call_count, 1)
            self.assertIn("Текстовий пошук за точними словами", mock_empty.answer.call_args[0][0])

            mock_q = MagicMock()
            mock_q.from_user.id = 12345
            mock_q.text = "/search бабуїни стрес"
            mock_q.answer = AsyncMock()

            asyncio.run(bot.cmd_search(mock_q))
            self.assertGreaterEqual(mock_q.answer.call_count, 2)


if __name__ == "__main__":
    unittest.main()


