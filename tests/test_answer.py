"""
tests/test_answer.py - Unit and Integration Tests for Grounded Answer Engine

Covers:
  - Extractive answering with verified sentence selection and citation indices.
  - Insufficient information gate (low similarity rejection threshold).
  - Strict paragraph_id attribution to SQLite records.
  - Telegram bot /answer command handling and formatting.
  - Feature flag (KORZ_ANSWER_ENABLED) guard behavior.
"""

import asyncio
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot
from korz_core import KorzLibrary, KorzConfig, SearchResult
from korz_answer import AnswerEngine, AnswerConfig, AnswerResult


class TestAnswerEngine(unittest.TestCase):
    def setUp(self):
        self.config = KorzConfig()
        self.lib = KorzLibrary(self.config)

    def tearDown(self):
        self.lib.close()

    def test_extractive_answer_attribution_and_sources(self):
        """Verify extractive answer contains authentic quotes and verifiable sources."""
        cfg = AnswerConfig(enabled=True, backend="extractive", min_similarity=0.75)
        engine = AnswerEngine(self.lib, cfg)

        question = "Чому соціальний статус впливає на здоров'я?"
        res = engine.answer(question)

        self.assertFalse(res.insufficient_info)
        self.assertEqual(res.mode, "extractive")
        self.assertGreater(len(res.sources), 0)
        self.assertIn("«", res.answer_text)
        self.assertIn("»", res.answer_text)

        # Check source attributes
        for s in res.sources:
            self.assertGreater(s.paragraph_id, 0)
            self.assertGreater(s.page, 0)
            self.assertTrue(bool(s.title))

    def test_insufficient_information_gate(self):
        """Verify out-of-domain query is rejected cleanly without synthetic hallucination."""
        cfg = AnswerConfig(enabled=True, backend="extractive", min_similarity=0.84)
        engine = AnswerEngine(self.lib, cfg)

        irrelevant_q = "Як приготувати полуничний коктейль з льодом?"
        res = engine.answer(irrelevant_q)

        self.assertTrue(res.insufficient_info)
        self.assertIn("недостатньо релевантної інформації", res.answer_text)

    def test_empty_question_handling(self):
        """Verify empty question returns early without error."""
        engine = AnswerEngine(self.lib, AnswerConfig(enabled=True))
        res = engine.answer("   ")
        self.assertTrue(res.insufficient_info)

    def test_bot_cmd_answer_integration(self):
        """Verify bot.cmd_answer formats messages with source buttons and sends to user."""
        bot.lib = self.lib
        self.lib.config.answer_enabled = True

        mock_msg = MagicMock()
        mock_msg.from_user.id = 12345
        mock_msg.text = "/answer Чому соціальний статус впливає на здоров'я?"
        mock_msg.answer = AsyncMock()

        asyncio.run(bot.cmd_answer(mock_msg))

        self.assertGreaterEqual(mock_msg.answer.call_count, 1)
        sent_texts = [call[0][0] for call in mock_msg.answer.call_args_list]
        self.assertTrue(any("Відповідь за бібліотекою KORZ" in t for t in sent_texts))
        self.assertTrue(any("Використані джерела" in t for t in sent_texts))

    def test_bot_cmd_answer_disabled_notice(self):
        """Verify bot.cmd_answer returns polite notice when KORZ_ANSWER_ENABLED=0."""
        bot.lib = self.lib
        self.lib.config.answer_enabled = False

        mock_msg = MagicMock()
        mock_msg.from_user.id = 12345
        mock_msg.text = "/answer тест"
        mock_msg.answer = AsyncMock()

        asyncio.run(bot.cmd_answer(mock_msg))

        self.assertEqual(mock_msg.answer.call_count, 1)
        sent_text = mock_msg.answer.call_args[0][0]
        self.assertIn("Режим /answer наразі вимкнено", sent_text)


if __name__ == "__main__":
    unittest.main()
