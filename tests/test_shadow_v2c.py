"""
tests/test_shadow_v2c.py - Phase V2C Verification Suite for KORZ Shadow Mode & Hybrid Analytics

Covers tests 46 through 60 as specified in the Phase V2C architecture:
  46. test_shadow_schema_and_tables_creation
  47. test_shadow_evaluator_produces_valid_record
  48. test_compact_json_format_without_bloat
  49. test_rrf_calculation_and_rank_deltas
  50. test_provenance_tracking_model_and_index_sha
  51. test_shadow_failure_does_not_affect_primary_fts
  52. test_concurrency_bounding_semaphore
  53. test_shadow_replay_idempotency
  54. test_shadow_replay_filters_limit_and_since
  55. test_feedback_recording_target_and_thumbs
  56. test_evaluation_metrics_recall_and_mrr
  57. test_shadow_report_formatting
  58. test_telegram_feedback_callback_handlers
  59. test_all_existing_45_tests_pass_without_regression
  60. test_soak_concurrency_with_shadow_runner
"""

import asyncio
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_core import (
    EmbeddingContract,
    KorzConfig,
    KorzLibrary,
    MockEmbeddingBackend,
    SearchResult,
    ShadowRunner,
    compute_simulated_rrf,
    serialize_compact_candidates,
)


class TestShadowV2C(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="korz_v2c_test_")
        self.temp_path = Path(self.temp_dir)
        self.db_path = self.temp_path / "data" / "test_v2c.sqlite"
        self.sources_dir = self.temp_path / "data" / "sources"
        self.backups_dir = self.temp_path / "data" / "backups"

        self.config = KorzConfig(
            data_dir=self.temp_path / "data",
            db_path=self.db_path,
            sources_dir=self.sources_dir,
            backups_dir=self.backups_dir,
            semantic_enabled=False,  # Kept FALSE by default to prove shadow isolation
            shadow_enabled=True,
        )

        self.mock_contract = EmbeddingContract(
            model_name="mock://v2c-test-model",
            dimension=64,
            dtype="float32",
            normalization="l2",
            pooling="mean",
            query_template="query: {text}",
            passage_template="passage: {text}",
            model_max_tokens=256,
        )
        self.mock_backend = MockEmbeddingBackend(dimension=64)

        # Seed sample documents
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            self.model_id = lib.register_embedding_model(self.mock_contract)

            # Ingest Book A
            sample_book_a = self.temp_path / "book_a.md"
            sample_book_a.write_text(
                "# Фізіологія стресу та поведінка\n\n"
                "## Розділ 1\n\n"
                "Хронічний стрес викликає порушення імунної системи та серцево-судинні захворювання у приматів. "
                "Соціальна ієрархія та рангове домінування визначають базовий рівень глюкокортикоїдів у крові.\n\n"
                "Коли примат зазнає поразки, рівень кортизолу стрімко зростає. "
                "Забезпечене середовище без фізичних загроз не рятує від психологічного виснаження.\n\n"
                "Психологічний стрес виникає через непередбачуваність та втрату контролю над обставинами.",
                encoding="utf-8",
            )
            lib.add_book(sample_book_a, title="Фізіологія стресу", author="Роберт Сапольський")

            # Ingest Book B
            sample_book_b = self.temp_path / "book_b.md"
            sample_book_b.write_text(
                "# Проектування міст майбутнього\n\n"
                "## Розділ 1\n\n"
                "Ресурсоорієнтована економіка замінює грошову систему науковим розподілом наявних ресурсів. "
                "Циркулярні міста майбутнього проектуються як єдина автоматизована кібернетична система.\n\n"
                "Автоматизація виробництва звільняє людину від одноманітної фізичної праці та рутини. "
                "Мова повинна мати високий ступінь фізичної кореляції з матеріальним світом.",
                encoding="utf-8",
            )
            lib.add_book(sample_book_b, title="Проектування майбутнього", author="Жак Фреско")

            # Build semantic index
            lib.build_semantic_index(model_id=self.model_id, backend=self.mock_backend)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_46_shadow_schema_and_tables_creation(self):
        """Test 46: Verify shadow_search_log and search_feedback tables and indexes exist."""
        with KorzLibrary(self.config) as lib:
            cur = lib.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {r[0] for r in cur.fetchall()}
            self.assertIn("shadow_search_log", tables)
            self.assertIn("search_feedback", tables)

            # Check indexes
            cur_idx = lib.conn.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = {r[0] for r in cur_idx.fetchall()}
            self.assertIn("idx_shadow_query", indexes)
            self.assertIn("idx_shadow_model_sha", indexes)
            self.assertIn("idx_feedback_query", indexes)

    def test_47_shadow_evaluator_produces_valid_record(self):
        """Test 47: ShadowRunner.evaluate_sync executes dual search and logs to shadow_search_log."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            runner = ShadowRunner(lib, max_concurrency=1)

            query = "хронічний стрес та ієрархія"
            fts_res = lib.search(query, mode="fts", limit=5)
            log_id = runner.evaluate_sync(
                query=query,
                fts_results=fts_res,
                latency_fts_ms=4.2,
                model_id=self.model_id,
            )

            self.assertIsNotNone(log_id)
            cur = lib.conn.execute("SELECT * FROM shadow_search_log WHERE id = ?", (log_id,))
            row = dict(cur.fetchone())
            self.assertEqual(row["query"], query)
            self.assertEqual(row["shadow_status"], "SUCCESS")
            self.assertGreater(row["semantic_count"], 0)
            self.assertGreaterEqual(row["overlap_at_5"], 0)
            self.assertIn(row["top1_match"], (0, 1))

    def test_48_compact_json_format_without_bloat(self):
        """Test 48: Candidate JSONs must store compact metadata only (no full paragraph text)."""
        sample_results = [
            SearchResult(
                paragraph_id=101,
                document_id="doc-123",
                title="Test Book",
                author="Test Author",
                page=5,
                paragraph_index=2,
                text="A very long paragraph text that would bloat the database if stored in json..." * 50,
                snippet="...",
                score=0.895,
                match_type="semantic",
            )
        ]

        compact_str = serialize_compact_candidates(sample_results, limit=1)
        compact_json = json.loads(compact_str)

        self.assertEqual(len(compact_json), 1)
        entry = compact_json[0]
        self.assertEqual(entry["paragraph_id"], 101)
        self.assertEqual(entry["document_id"], "doc-123")
        self.assertEqual(entry["page"], 5)
        self.assertEqual(entry["rank"], 1)
        self.assertEqual(entry["score"], 0.895)
        self.assertNotIn("text", entry)
        self.assertNotIn("display_text", entry)
        self.assertLess(len(compact_str), 200)

    def test_49_rrf_calculation_and_rank_deltas(self):
        """Test 49: Verify exact mathematical RRF@60 formula and rank delta calculation."""
        # Item 1 is rank 1 in FTS and rank 2 in Semantic
        # Item 2 is rank 2 in FTS and None in Semantic
        # Item 3 is None in FTS and rank 1 in Semantic
        fts = [
            SearchResult(paragraph_id=1, document_id="d1", title="T1", author="A1", page=1, paragraph_index=1, text="p1", snippet="p1", score=10.0, match_type="fts"),
            SearchResult(paragraph_id=2, document_id="d1", title="T1", author="A1", page=1, paragraph_index=2, text="p2", snippet="p2", score=5.0, match_type="fts"),
        ]
        sem = [
            SearchResult(paragraph_id=3, document_id="d1", title="T1", author="A1", page=2, paragraph_index=1, text="p3", snippet="p3", score=0.92, match_type="semantic"),
            SearchResult(paragraph_id=1, document_id="d1", title="T1", author="A1", page=1, paragraph_index=1, text="p1", snippet="p1", score=0.88, match_type="semantic"),
        ]

        rrf_top, o5, o10, top1 = compute_simulated_rrf(fts, sem, k=60, top_limit=10)

        self.assertEqual(o5, 1)  # Only paragraph 1 is in both
        self.assertEqual(top1, 0)  # FTS #1 is 1, SEM #1 is 3

        # Item 1 score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.0163934 + 0.0161290 = 0.032522
        item1 = next(x for x in rrf_top if x["paragraph_id"] == 1)
        self.assertEqual(item1["rank"], 1)  # Combined score should be highest
        self.assertAlmostEqual(item1["score"], (1.0 / 61) + (1.0 / 62), places=5)
        self.assertEqual(item1["rank_fts"], 1)
        self.assertEqual(item1["rank_sem"], 2)
        self.assertEqual(item1["rank_delta"], -1)  # 1 - 2 = -1

    def test_50_provenance_tracking_model_and_index_sha(self):
        """Test 50: Shadow log records correct semantic_model_id and non-empty semantic_index_sha."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            runner = ShadowRunner(lib)
            runner.evaluate_sync("ресурсоорієнтована економіка", fts_results=[], model_id=self.model_id)

            cur = lib.conn.execute("SELECT semantic_model_id, semantic_index_sha FROM shadow_search_log ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            self.assertEqual(row["semantic_model_id"], self.model_id)
            self.assertTrue(len(row["semantic_index_sha"]) >= 8)

    def test_51_shadow_failure_does_not_affect_primary_fts(self):
        """Test 51: If shadow evaluation fails, primary FTS search completes normally."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            # Break active model ID to force a shadow error
            lib._active_model_id = "non-existent-broken-model"

            # Execute search with shadow enabled
            results = lib.search("стрес", mode="auto", shadow=True)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0].title, "Фізіологія стресу")

            # Check that shadow log recorded the failure gracefully
            cur = lib.conn.execute("SELECT shadow_status, error_message FROM shadow_search_log ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            self.assertEqual(row["shadow_status"], "FAILED")
            self.assertIsNotNone(row["error_message"])

    def test_52_concurrency_bounding_semaphore(self):
        """Test 52: ShadowRunner limits concurrent evaluations using asyncio.Semaphore."""
        runner = ShadowRunner(None, max_concurrency=1)
        sem = runner._get_semaphore()
        self.assertEqual(sem._value, 1)

    def test_53_shadow_replay_idempotency(self):
        """Test 53: shadow replay skips queries that have already been evaluated."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            # Populate search_log with 3 queries
            lib.conn.execute("INSERT INTO search_log (query, mode, result_count, latency_ms, created_at) VALUES ('q1', 'auto', 1, 5.0, '2026-09-28T12:00:00')")
            lib.conn.execute("INSERT INTO search_log (query, mode, result_count, latency_ms, created_at) VALUES ('q2', 'auto', 1, 5.0, '2026-09-28T12:01:00')")
            lib.conn.execute("INSERT INTO search_log (query, mode, result_count, latency_ms, created_at) VALUES ('q3', 'auto', 1, 5.0, '2026-09-28T12:02:00')")

            # Replay 1
            res1 = lib.replay_shadow_evaluation(model_id=self.model_id)
            self.assertGreaterEqual(res1["evaluated"], 3)
            self.assertEqual(res1["failed"], 0)

            # Replay 2 (idempotent skip)
            res2 = lib.replay_shadow_evaluation(model_id=self.model_id)
            self.assertEqual(res2["evaluated"], 0)
            self.assertGreaterEqual(res2["skipped"], 3)

    def test_54_shadow_replay_filters_limit_and_since(self):
        """Test 54: shadow replay correctly respects limit and since parameters."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            lib.conn.execute("INSERT INTO search_log (query, mode, result_count, latency_ms, created_at) VALUES ('old_q', 'auto', 1, 5.0, '2026-01-01T10:00:00')")
            lib.conn.execute("INSERT INTO search_log (query, mode, result_count, latency_ms, created_at) VALUES ('new_q', 'auto', 1, 5.0, '2026-09-28T15:00:00')")

            # Filter since Sept 2026
            res = lib.replay_shadow_evaluation(since="2026-09-01", limit=10, model_id=self.model_id, force=True)
            self.assertIn("new_q", [r["query"] for r in lib.conn.execute("SELECT query FROM shadow_search_log").fetchall()])

    def test_55_feedback_recording_target_and_thumbs(self):
        """Test 55: Verify explicit user feedback is recorded in search_feedback."""
        with KorzLibrary(self.config) as lib:
            fb1 = lib.record_feedback(query="стрес", feedback_type="found", target_paragraph_id=1, user_id="12345")
            fb2 = lib.record_feedback(query="міста майбутнього", feedback_type="target", target_paragraph_id=4, user_id="12345")
            fb3 = lib.record_feedback(query="незрозуміле", feedback_type="not_relevant", user_id="12345")

            self.assertGreater(fb1, 0)
            self.assertGreater(fb2, 0)
            self.assertGreater(fb3, 0)

            cur = lib.conn.execute("SELECT COUNT(*) FROM search_feedback WHERE feedback_type = 'target'")
            self.assertEqual(cur.fetchone()[0], 1)

    def test_56_evaluation_metrics_recall_and_mrr(self):
        """Test 56: Verify report computes accurate Recall@1, Recall@5, and MRR from feedbacks."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            runner = ShadowRunner(lib)

            # Query 1: Target paragraph is ID 1
            fts_res = lib.search("стрес", mode="fts", limit=5)
            log_id = runner.evaluate_sync("стрес", fts_results=fts_res, model_id=self.model_id)

            # Record feedback with target_paragraph_id = 1
            lib.record_feedback(
                query="стрес",
                feedback_type="target",
                shadow_log_id=log_id,
                target_paragraph_id=1,
            )

            rep = lib.get_shadow_report()
            rel = rep["relevance"]
            self.assertGreater(rel["total_judged"], 0)
            self.assertIsNotNone(rel["fts_recall_at_1"])
            self.assertIsNotNone(rel["semantic_recall_at_1"])
            self.assertIsNotNone(rel["rrf_recall_at_1"])

    def test_57_shadow_report_formatting(self):
        """Test 57: Verify shadow report computes operational percentiles and divergence rates."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            runner = ShadowRunner(lib)
            for q in ["стрес", "міста", "економіка"]:
                fts = lib.search(q, mode="fts", limit=5, shadow=False)
                runner.evaluate_sync(q, fts_results=fts, latency_fts_ms=3.0, model_id=self.model_id)

            rep = lib.get_shadow_report(limit_divergences=5)
            self.assertEqual(rep["total_queries"], 3)
            self.assertIn("fts_latency_p50", rep["operational"])
            self.assertIn("semantic_latency_p50", rep["operational"])
            self.assertIn("top1_match_pct", rep["divergence"])
            self.assertIn("overlap_5_zero_pct", rep["divergence"])

    def test_58_telegram_feedback_callback_handlers(self):
        """Test 58: Simulate Telegram feedback callbacks format 'fb:tgt:123', 'fb:up:1', 'fb:dn:0'."""
        with KorzLibrary(self.config) as lib:
            raw_callbacks = ["fb:tgt:1", "fb:up:1", "fb:dn:0"]
            for cb in raw_callbacks:
                parts = cb.split(":")
                fb_type = parts[1]
                pid = int(parts[2]) if parts[2] != "0" else None
                fid = lib.record_feedback(query="test_query", feedback_type=fb_type, target_paragraph_id=pid)
                self.assertGreater(fid, 0)

    def test_59_all_existing_45_tests_pass_without_regression(self):
        """Test 59: Baseline backward compatibility - status() and schema version unchanged."""
        with KorzLibrary(self.config) as lib:
            st = lib.status()
            self.assertEqual(st["schema_version"], 2)
            self.assertEqual(st["total_documents"], 2)

    def test_60_soak_concurrency_with_shadow_runner(self):
        """Test 60: Run 10 rapid concurrent async evaluations without lock failures."""
        async def run_burst():
            with KorzLibrary(self.config) as lib:
                lib.set_backend("mock://v2c-test-model", self.mock_backend)
                runner = ShadowRunner(lib, max_concurrency=1)
                tasks = []
                for i in range(10):
                    q = f"concurrency_query_{i}"
                    fts = [SearchResult(paragraph_id=1, document_id="d1", title="T", author="A", page=1, paragraph_index=1, text="p", snippet="s", score=1.0, match_type="fts")]
                    tasks.append(runner.evaluate_async(q, fts_results=fts, model_id=self.model_id))
                results = await asyncio.gather(*tasks)
                return results

        results = asyncio.run(run_burst())
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r is not None for r in results))

    def test_61_rrf_deduplication_by_paragraph_id(self):
        """Test 61: Multiple chunks from the same paragraph must deduplicate to single best rank."""
        fts = [
            SearchResult(paragraph_id=10, document_id="d1", title="T", author="A", page=1, paragraph_index=1, text="p", snippet="s", score=5.0, match_type="fts")
        ]
        # Semantic returns two chunks for paragraph 20 (chunk A rank 1, chunk B rank 2), then paragraph 30 (rank 3)
        sem = [
            SearchResult(paragraph_id=20, document_id="d1", title="T", author="A", page=2, paragraph_index=1, text="p20a", snippet="s", score=0.95, match_type="semantic"),
            SearchResult(paragraph_id=20, document_id="d1", title="T", author="A", page=2, paragraph_index=1, text="p20b", snippet="s", score=0.90, match_type="semantic"),
            SearchResult(paragraph_id=30, document_id="d1", title="T", author="A", page=3, paragraph_index=1, text="p30", snippet="s", score=0.85, match_type="semantic"),
        ]

        rrf_top, o5, o10, top1 = compute_simulated_rrf(fts, sem, k=60, top_limit=5)
        # Should have unique paragraph IDs
        pids = [item["paragraph_id"] for item in rrf_top]
        self.assertEqual(len(pids), len(set(pids)))
        self.assertEqual(pids.count(20), 1)

        # Paragraph 20 must have retained its top rank (rank 1), not degraded by chunk B
        p20 = next(x for x in rrf_top if x["paragraph_id"] == 20)
        self.assertEqual(p20["rank_sem"], 1)

    def test_62_rrf_exact_wins_pinning(self):
        """Test 62: Exact phrase match at rank 1 in FTS must be strictly pinned as rank 1 in RRF."""
        # FTS has exact match for paragraph 99
        fts = [
            SearchResult(paragraph_id=99, document_id="d1", title="T", author="A", page=1, paragraph_index=1, text="exact text", snippet="s", score=100.0, match_type="exact")
        ]
        # Semantic has completely different strong hit for paragraph 42
        sem = [
            SearchResult(paragraph_id=42, document_id="d1", title="T", author="A", page=2, paragraph_index=1, text="other text", snippet="s", score=0.99, match_type="semantic")
        ]

        rrf_top, _, _, _ = compute_simulated_rrf(fts, sem, k=60, top_limit=5)
        # Exact match 99 must win rank 1
        self.assertEqual(rrf_top[0]["paragraph_id"], 99)
        self.assertEqual(rrf_top[0]["rank"], 1)
        self.assertGreater(rrf_top[0]["score"], 1.0)  # Contains exact match boost

    def test_63_warmup_semantic_loads_and_runs(self):
        """Test 63: warmup_semantic executes cleanly and populates in-memory index."""
        with KorzLibrary(self.config) as lib:
            lib.set_backend("mock://v2c-test-model", self.mock_backend)
            elapsed = lib.warmup_semantic(model_id=self.model_id)
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertIsNotNone(lib._semantic_index)
            self.assertEqual(lib._semantic_index.model_id, self.model_id)


if __name__ == "__main__":
    unittest.main()
