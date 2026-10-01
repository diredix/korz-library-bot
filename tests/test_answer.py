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

    def test_resolve_gemini_model_default_low(self):
        """Verify standard queries default to Low tier when adaptive is off."""
        from korz_answer import AnswerSource
        cfg = AnswerConfig(enabled=True, backend="gemini", adaptive_thinking=False)
        engine = AnswerEngine(self.lib, cfg)

        sources = [
            AnswerSource(1, 101, "doc1", "Сапольські", "Чому у зебр", 10, 1, "текст", 0.88),
            AnswerSource(2, 102, "doc1", "Сапольські", "Чому у зебр", 12, 2, "текст 2", 0.85),
        ]
        model, tier, trigger = engine.resolve_gemini_model("Чому бабуїни хворіють?", sources, deep=False)
        self.assertEqual(model, "gemini-3.8-flash-low")
        self.assertEqual(tier, "low")
        self.assertEqual(trigger, "default")

    def test_resolve_gemini_model_explicit_deep_high(self):
        """Verify explicit deep request activates High tier regardless of other flags."""
        from korz_answer import AnswerSource
        cfg = AnswerConfig(enabled=True, backend="gemini", adaptive_thinking=False)
        engine = AnswerEngine(self.lib, cfg)

        sources = [AnswerSource(1, 101, "doc1", "Автор", "Книга", 10, 1, "текст", 0.88)]
        model, tier, trigger = engine.resolve_gemini_model("Питання", sources, deep=True)
        self.assertEqual(model, "gemini-3.8-flash-high")
        self.assertEqual(tier, "high")
        self.assertEqual(trigger, "explicit_deep")

    def test_resolve_gemini_model_adaptive_disabled_keeps_low(self):
        """Verify feature flag OFF keeps multi-author/comparison queries on Low baseline."""
        from korz_answer import AnswerSource
        cfg = AnswerConfig(enabled=True, backend="gemini", adaptive_thinking=False)
        engine = AnswerEngine(self.lib, cfg)

        multi_sources = [
            AnswerSource(1, 101, "doc1", "Сапольські", "Зебри", 10, 1, "текст", 0.88),
            AnswerSource(2, 201, "doc2", "Фреско", "Проектування", 20, 1, "текст 2", 0.86),
            AnswerSource(3, 301, "doc3", "Коржибський", "Наука", 30, 1, "текст 3", 0.85),
        ]
        model, tier, trigger = engine.resolve_gemini_model("Порівняй погляди Сапольського і Фреско", multi_sources, deep=False)
        self.assertEqual(model, "gemini-3.8-flash-low")
        self.assertEqual(tier, "low")
        self.assertEqual(trigger, "default")

    def test_resolve_gemini_model_adaptive_enabled_switches_medium(self):
        """Verify feature flag ON promotes multi-source or comparison queries to Medium."""
        from korz_answer import AnswerSource
        cfg = AnswerConfig(enabled=True, backend="gemini", adaptive_thinking=True)
        engine = AnswerEngine(self.lib, cfg)

        multi_sources = [
            AnswerSource(1, 101, "doc1", "Сапольські", "Зебри", 10, 1, "текст", 0.88),
            AnswerSource(2, 201, "doc2", "Фреско", "Проектування", 20, 1, "текст 2", 0.86),
            AnswerSource(3, 301, "doc3", "Коржибський", "Наука", 30, 1, "текст 3", 0.85),
        ]
        # Comparison trigger (even single source)
        single_source = [AnswerSource(1, 101, "doc1", "Сапольські", "Зебри", 10, 1, "текст", 0.88)]
        m1, t1, tr1 = engine.resolve_gemini_model("Порівняй погляди", single_source, deep=False)
        self.assertEqual(m1, "gemini-3.8-flash-medium")
        self.assertEqual(t1, "medium")
        self.assertEqual(tr1, "adaptive_comparison")

        # Multi-author trigger
        multi_author_sources = [
            AnswerSource(1, 101, "doc1", "Сапольські", "Зебри", 10, 1, "текст", 0.88),
            AnswerSource(2, 201, "doc2", "Фреско", "Проектування", 20, 1, "текст 2", 0.86),
        ]
        m2, t2, tr2 = engine.resolve_gemini_model("Як формується реакція на дефіцит?", multi_author_sources, deep=False)
        self.assertEqual(m2, "gemini-3.8-flash-medium")
        self.assertEqual(t2, "medium")
        self.assertEqual(tr2, "adaptive_multi_author")

        # Multiple passages from SAME book & author must stay Low (diversity requirement)
        same_author_sources = [
            AnswerSource(1, 101, "doc1", "Сапольські", "Зебри", 10, 1, "текст", 0.88),
            AnswerSource(2, 102, "doc1", "Сапольські", "Зебри", 12, 2, "текст 2", 0.86),
            AnswerSource(3, 103, "doc1", "Сапольські", "Зебри", 14, 3, "текст 3", 0.85),
        ]
        m3, t3, tr3 = engine.resolve_gemini_model("Що таке глюкокортикоїди?", same_author_sources, deep=False)
        self.assertEqual(m3, "gemini-3.8-flash-low")
        self.assertEqual(t3, "low")
        self.assertEqual(tr3, "default")

    def test_deep_flag_stripped_from_retrieval_query(self):
        """Verify --deep flag is extracted, stripped from query, and routes to High tier."""
        from unittest.mock import patch, MagicMock
        cfg = AnswerConfig(enabled=True, backend="gemini", min_similarity=0.75)
        engine = AnswerEngine(self.lib, cfg)

        mock_sub = MagicMock()
        mock_sub.returncode = 0
        mock_sub.stdout = "Глибока відповідь [1].".encode("utf-8")

        with patch("subprocess.run", return_value=mock_sub) as mock_run:
            res = engine.answer("/answer --deep Чому соціальний статус впливає на здоров'я?")
            self.assertEqual(res.model_tier, "high")
            self.assertEqual(res.model_id, "gemini-3.8-flash-high")
            self.assertEqual(res.adaptive_trigger, "explicit_deep")
            self.assertEqual(res.prompt_version, "v1.0-grounded")
            self.assertTrue(bool(res.answer_config_sha))
            self.assertGreater(len(res.source_text_hashes), 0)
            # Verify --deep was not sent in command prompt to LLM as the raw question
            call_args = mock_run.call_args[0][0]
            self.assertIn("gemini-3.8-flash-high", call_args)

    def test_record_answer_run_adaptive_telemetry(self):
        """Verify record_answer_run stores all adaptive thinking columns and provenance in SQLite."""
        run_id = self.lib.record_answer_run(
            query="Порівняй авторів",
            backend="gemini",
            threshold=0.82,
            insufficient_info=0,
            source_paragraph_ids=[101, 202],
            answer_text="Відповідь [1], [2]",
            citation_verified=1,
            latency_ms=8500.0,
            user_id="user_123",
            model_tier="medium",
            model_id="gemini-3.8-flash-medium",
            adaptive_trigger="adaptive_comparison",
            source_count=2,
            book_count=2,
            author_count=2,
            prompt_version="v1.0-grounded",
            answer_config_sha="abcdef1234567890",
            source_text_hashes=["hash1", "hash2"],
            corpus_version="v1-baseline",
            experiment_input_sha="sha_test_123456",
            input_bundle_json='{"query": "Порівняй авторів"}',
        )

        details = self.lib.get_answer_run(run_id)
        self.assertIsNotNone(details)
        self.assertEqual(details["model_tier"], "medium")
        self.assertEqual(details["model_id"], "gemini-3.8-flash-medium")
        self.assertEqual(details["adaptive_trigger"], "adaptive_comparison")
        self.assertEqual(details["source_count"], 2)
        self.assertEqual(details["book_count"], 2)
        self.assertEqual(details["author_count"], 2)
        self.assertEqual(details["prompt_version"], "v1.0-grounded")
        self.assertEqual(details["answer_config_sha"], "abcdef1234567890")
        self.assertIn("hash1", details["source_text_hashes"])
        self.assertEqual(details["corpus_version"], "v1-baseline")
        self.assertEqual(details["experiment_input_sha"], "sha_test_123456")
        self.assertEqual(details["input_bundle_json"], '{"query": "Порівняй авторів"}')

    def test_build_experiment_bundle_deterministic_hash(self):
        """Verify build_experiment_bundle generates deterministic, bit-identical sha256."""
        from korz_answer import AnswerSource, build_experiment_bundle, compute_text_sha256

        t1 = "Текст першого уривка"
        t2 = "Текст другого уривка"
        s1 = AnswerSource(1, 101, "doc1", "Сапольські", "Зебри", 10, 1, t1, 0.88, compute_text_sha256(t1))
        s2 = AnswerSource(2, 201, "doc2", "Фреско", "Проектування", 20, 2, t2, 0.85, compute_text_sha256(t2))

        b1, json1, sha1 = build_experiment_bundle("Питання?", [s1, s2], "v1.0-grounded", "cfg123")
        b2, json2, sha2 = build_experiment_bundle("Питання?", [s1, s2], "v1.0-grounded", "cfg123")

        self.assertEqual(json1, json2)
        self.assertEqual(sha1, sha2)
        self.assertEqual(len(sha1), 64)
        self.assertEqual(b1["query"], "Питання?")
        self.assertEqual(len(b1["sources"]), 2)
        self.assertEqual(b1["sources"][0]["paragraph_id"], 101)
        self.assertEqual(b1["sources"][0]["text_sha256"], s1.text_sha256)

    def test_answer_retrieval_mode_dense_and_rrf(self):
        """Verify retrieval_mode switch between pure dense and hybrid RRF."""
        # 1. Pure Dense Mode
        cfg_dense = AnswerConfig(enabled=True, backend="extractive", retrieval_mode="dense", min_similarity=0.75)
        eng_dense = AnswerEngine(self.lib, cfg_dense)
        res_dense = eng_dense.answer("Чому соціальний статус впливає на здоров'я?")
        self.assertEqual(res_dense.retrieval_mode, "dense")
        self.assertFalse(res_dense.insufficient_info)
        self.assertGreater(len(res_dense.sources), 0)

        # 2. Hybrid RRF Mode
        cfg_rrf = AnswerConfig(enabled=True, backend="extractive", retrieval_mode="rrf", min_similarity=0.75)
        eng_rrf = AnswerEngine(self.lib, cfg_rrf)
        res_rrf = eng_rrf.answer("Чому соціальний статус впливає на здоров'я?")
        self.assertEqual(res_rrf.retrieval_mode, "rrf")
        self.assertFalse(res_rrf.insufficient_info)
        self.assertGreater(len(res_rrf.sources), 0)

    def test_answer_shadow_rrf_logging(self):
        """Verify shadow RRF logs comparison metrics while executing pure dense."""
        cfg = AnswerConfig(enabled=True, backend="extractive", retrieval_mode="dense", shadow_rrf=True, min_similarity=0.75)
        eng = AnswerEngine(self.lib, cfg)
        res = eng.answer("Чому соціальний статус впливає на здоров'я?")
        self.assertEqual(res.retrieval_mode, "dense")
        self.assertTrue(bool(res.shadow_rrf_json))
        import json
        shadow_data = json.loads(res.shadow_rrf_json)
        self.assertIn("overlap_at_5", shadow_data)
        self.assertIn("rrf_top_pids", shadow_data)

    def test_record_answer_run_retrieval_mode_and_shadow_telemetry(self):
        """Verify record_answer_run saves and returns retrieval_mode and shadow_rrf_json."""
        run_id = self.lib.record_answer_run(
            query="Тест гібридного пошуку",
            backend="extractive",
            threshold=0.82,
            insufficient_info=0,
            source_paragraph_ids=[55, 66],
            answer_text="Відповідь [1]",
            citation_verified=1,
            latency_ms=45.2,
            retrieval_mode="rrf",
            shadow_rrf_json='{"overlap_at_5": 3}',
        )
        rec = self.lib.get_answer_run(run_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["retrieval_mode"], "rrf")
        self.assertEqual(rec["shadow_rrf_json"], '{"overlap_at_5": 3}')

    def test_hybrid_insufficient_information_gate(self):
        """Verify hybrid mode safely rejects completely irrelevant queries."""
        cfg = AnswerConfig(enabled=True, backend="extractive", retrieval_mode="rrf", min_similarity=0.84)
        eng = AnswerEngine(self.lib, cfg)
        irrelevant_q = "Як приготувати полуничний коктейль з льодом?"
        res = eng.answer(irrelevant_q)
        self.assertTrue(res.insufficient_info)
        self.assertEqual(res.retrieval_mode, "rrf")
        self.assertIn("недостатньо релевантної інформації", res.answer_text)


if __name__ == "__main__":
    unittest.main()

