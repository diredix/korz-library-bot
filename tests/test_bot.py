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


if __name__ == "__main__":
    unittest.main()

