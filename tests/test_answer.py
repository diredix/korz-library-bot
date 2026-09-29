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

    def test_citation_verification_pass(self):
        """Verify Citation Verification Pass sanitizes out-of-bounds citations and detects ungrounded claims."""
        from korz_answer import AnswerSource
        engine = AnswerEngine(self.lib, AnswerConfig(enabled=True))

        mock_sources = [
            AnswerSource(1, 101, "doc1", "Автор", "Книга", 10, 1, "Текст джерела 1", 0.88),
            AnswerSource(2, 102, "doc1", "Автор", "Книга", 11, 2, "Текст джерела 2", 0.85),
        ]

        # Text with valid [1], invalid [9], and an unsupported claim
        raw_llm_text = (
            "Соціальний статус викликає фізіологічний стрес [1]. "
            "Цей стрес може призводити до тривожності [9]. "
            "Це загальновідомий психологічний факт."
        )

        verified_text, used_sources, unsupported_cnt, is_valid = engine._verify_and_filter_citations(
            raw_llm_text, mock_sources
        )

        self.assertTrue(is_valid)
        self.assertEqual(len(used_sources), 1)
        self.assertEqual(used_sources[0].citation_index, 1)
        self.assertNotIn("[9]", verified_text)
        self.assertIn("[1]", verified_text)
        self.assertGreaterEqual(unsupported_cnt, 1)

    def test_generative_fallback_when_ollama_unavailable(self):
        """Verify graceful fallback with explicit fallback_reason when generative backend is requested."""
        cfg = AnswerConfig(enabled=True, backend="generative", min_similarity=0.75, ollama_url="http://localhost:65432")
        engine = AnswerEngine(self.lib, cfg)

        res = engine.answer("Чому соціальний статус впливає на здоров'я?")
        self.assertEqual(res.mode, "extractive")
        self.assertIsNotNone(res.fallback_reason)
        self.assertIn("Ollama", res.fallback_reason)
        self.assertGreater(len(res.sources), 0)

    def test_bot_answer_feedback_callbacks(self):
        """Verify bot cb_feedback records 3-tier answer quality feedback."""
        bot.lib = self.lib

        # Test ans_up
        cb1 = MagicMock()
        cb1.data = "fb:ans_up:42"
        cb1.from_user.id = 12345
        cb1.message.reply_to_message.text = "тестове питання"
        cb1.answer = AsyncMock()

        asyncio.run(bot.cb_feedback(cb1))
        self.assertTrue(cb1.answer.called)

        # Test ans_part
        cb2 = MagicMock()
        cb2.data = "fb:ans_part:42"
        cb2.from_user.id = 12345
        cb2.message.reply_to_message.text = "тестове питання"
        cb2.answer = AsyncMock()

        asyncio.run(bot.cb_feedback(cb2))
        self.assertTrue(cb2.answer.called)

        # Test ans_dn
        cb3 = MagicMock()
        cb3.data = "fb:ans_dn:0"
        cb3.from_user.id = 12345
        cb3.message.reply_to_message.text = "тестове питання"
        cb3.answer = AsyncMock()

        asyncio.run(bot.cb_feedback(cb3))
        self.assertTrue(cb3.answer.called)

        # Check in DB
        rows = self.lib.conn.execute(
            "SELECT feedback_type FROM search_feedback WHERE user_id = '12345' ORDER BY id DESC LIMIT 3"
        ).fetchall()
        types = [r[0] for r in rows]
        self.assertIn("answer_irrelevant", types)
        self.assertIn("answer_partial", types)
        self.assertIn("answer_accurate", types)

    def test_gemini_generative_answer_mocked(self):
        """Verify Gemini backend formats prompt, runs agy, and verifies citations."""
        from unittest.mock import patch, MagicMock
        cfg = AnswerConfig(enabled=True, backend="gemini", min_similarity=0.75, gemini_model="gemini-3.8-flash-low")
        engine = AnswerEngine(self.lib, cfg)

        mock_sub = MagicMock()
        mock_sub.returncode = 0
        mock_sub.stdout = "Бабуїни хворіють через стрес [1].".encode("utf-8")

        with patch("subprocess.run", return_value=mock_sub):
            res = engine.answer("Чому соціальний статус впливає на здоров'я?")
            self.assertEqual(res.mode, "gemini")
            self.assertIn("[1]", res.answer_text)
            self.assertEqual(len(res.sources), 1)


if __name__ == "__main__":
    unittest.main()

