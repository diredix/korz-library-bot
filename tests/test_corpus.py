"""
tests/test_corpus.py - Unit and Integration Tests for KORZ Corpus Quality Pipeline

Tests:
  - CorpusAuditor noise detection (headers, page numbers, image markers, duplicate captions).
  - CorpusCleaner deterministic transformations (stripping layout noise, stitching broken sentences).
  - CorpusCleaner compound hyphen protection vs safe dehyphenation.
  - CorpusVerifier text-loss safety gate and content preservation gate.
  - Provenance tracking (clean_manifest.json and clean_diff.jsonl).
  - CandidateLab A/B evaluation metric calculation.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import sqlite3
from types import SimpleNamespace

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_corpus import (
    CorpusAuditor,
    CorpusCleaner,
    CorpusVerifier,
    CandidateLab,
    NoiseItem,
    CorpusParagraphMap,
    CorpusABShadowRunner,
)


class TestCorpusQualityPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_dirty_text = (
            "{10}------------------------------------------------\n\n"
            "10 • Роберт М. Сапольський\n\n"
            "![Бабуїн сидить на дереві](img1.jpg)\n\n"
            "Бабуїн сидить на дереві\n\n"
            "Бабуїни живуть у складних соціально-\n"
            "економічних умовах і мають чітку ієрар-\n"
            "хію підпорядкування. У них рідко виникають прямі сутички\n\n"
            "10\n\n"
            "{11}------------------------------------------------\n\n"
            "*Спогади примата* • 11\n\n"
            ". Вони вирішують конфлікти психологічним залякуванням.\n\n"
            "11\n"
        )

    def test_auditor_detects_noise_types(self):
        """Verify CorpusAuditor identifies headers, numbers, image tags, duplicate captions, and broken sentences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_book.md"
            file_path.write_text(self.sample_dirty_text, encoding="utf-8")

            auditor = CorpusAuditor()
            report = auditor.audit_path(file_path)

            self.assertEqual(report.total_files, 1)
            self.assertEqual(report.page_separators, 2)
            self.assertGreaterEqual(report.standalone_numbers, 2)
            self.assertGreaterEqual(report.running_headers, 2)
            self.assertEqual(report.image_markers, 1)
            self.assertEqual(report.duplicate_captions, 1)
            self.assertGreaterEqual(report.broken_sentences, 1)
            self.assertGreaterEqual(report.high_confidence_noise, 7)

    def test_cleaner_removes_noise_and_repairs_sentences(self):
        """Verify CorpusCleaner strips layout noise and stitches broken sentences across page transitions."""
        cleaner = CorpusCleaner()
        cleaned, stats, actions = cleaner.clean_text(self.sample_dirty_text)

        # Standalone numbers, running headers, image markers must be gone
        self.assertNotIn("10 • Роберт М. Сапольський", cleaned)
        self.assertNotIn("*Спогади примата* • 11", cleaned)
        self.assertNotIn("img1.jpg", cleaned)
        self.assertNotIn("\n10\n", cleaned)
        self.assertNotIn("\n11\n", cleaned)

        # Duplicate caption must be deduplicated
        self.assertEqual(cleaned.count("Бабуїн сидить на дереві"), 0)

        # Compound hyphen protection check: 'соціально-\nекономічних' -> 'соціально-економічних'
        self.assertIn("соціально-економічних", cleaned)
        self.assertEqual(stats["ambiguous_hyphens_preserved"], 1)

        # Dehyphenation check: 'ієрар-\nхію' -> 'ієрархію'
        self.assertIn("ієрархію", cleaned)
        self.assertEqual(stats["safe_dehyphenations"], 1)

        # Sentence repair check: '. Вони вирішують' attached to previous sentence
        self.assertIn("прямі сутички.", cleaned)
        self.assertIn("Вони вирішують конфлікти", cleaned)
        self.assertNotIn(". Вони вирішують", cleaned)

        # Page separators must be normalized and preserved
        self.assertIn("{10}--------------------------------", cleaned)
        self.assertIn("{11}--------------------------------", cleaned)

        self.assertGreaterEqual(stats["running_headers_removed"], 2)
        self.assertGreaterEqual(stats["standalone_numbers_removed"], 2)
        self.assertGreaterEqual(stats["cross_page_sentence_joins"], 1)
        self.assertGreater(len(actions), 5)

    def test_verifier_text_loss_and_content_preservation_gate(self):
        """Verify normal layout noise cleaning passes the text-loss and token preservation gate."""
        cleaner = CorpusCleaner()
        cleaned, stats, _ = cleaner.clean_text(self.sample_dirty_text)

        verifier = CorpusVerifier(max_allowed_loss_pct=60.0, max_unclassified_loss_pct=5.0, min_token_retention_pct=50.0)
        res = verifier.verify(self.sample_dirty_text, cleaned, stats)

        self.assertTrue(res.is_safe)
        self.assertEqual(res.status, "PASS")
        self.assertGreaterEqual(res.meaningful_token_retention_pct, 50.0)

    def test_verifier_text_loss_gate_fail_on_excessive_loss(self):
        """Verify massive content deletion triggers FAIL in text loss safety gate."""
        original = "Цінний змістовний текст книги про поведінку приматів. " * 100
        # Dropping 50% without corresponding stats
        cleaned = "Цінний змістовний текст книги. " * 50
        fake_stats = {"standalone_numbers_removed": 1}

        verifier = CorpusVerifier(max_allowed_loss_pct=3.5, max_unclassified_loss_pct=1.0, min_token_retention_pct=99.5)
        res = verifier.verify(original, cleaned, fake_stats)

        self.assertFalse(res.is_safe)
        self.assertEqual(res.status, "FAIL")
        self.assertGreater(res.unclassified_loss_pct, 1.0)

    def test_clean_file_dry_run_and_provenance_manifest(self):
        """Verify apply mode creates clean file, manifest JSON, and diff JSONL log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_file = Path(tmpdir) / "source.md"
            orig_file.write_text(self.sample_dirty_text, encoding="utf-8")

            cleaner = CorpusCleaner()

            # Dry run
            diff = cleaner.clean_file(orig_file, dry_run=True)
            self.assertGreater(len(diff.diff_text), 0)
            self.assertFalse((Path(tmpdir) / "source.cleaned.md").exists())

            # Apply run
            clean_dest = Path(tmpdir) / "clean_dir" / "clean_source.md"
            diff_applied = cleaner.clean_file(orig_file, dry_run=False, output_path=clean_dest)
            self.assertTrue(clean_dest.exists())

            # Verify manifest
            manifest_file = Path(tmpdir) / "clean_dir" / "clean_source.clean_manifest.json"
            self.assertTrue(manifest_file.exists())
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["cleaner_version"], "korz-corpus-cleaner-v1.0")
            self.assertGreater(manifest["meaningful_token_retention_pct"], 50.0)

            # Verify diff JSONL
            diff_jsonl_file = Path(tmpdir) / "clean_dir" / "clean_source.clean_diff.jsonl"
            self.assertTrue(diff_jsonl_file.exists())
            lines = diff_jsonl_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreater(len(lines), 3)
            actions = [json.loads(l) for l in lines]
            action_types = {a["action"] for a in actions}
            self.assertIn("REMOVE_HEADER", action_types)
            self.assertIn("SAFE_DEHYPHEN", action_types)
            self.assertIn("PRESERVE_COMPOUND_HYPHEN", action_types)



class TestCorpusABAlignmentAndShadow(unittest.TestCase):
    def test_paragraph_map_bidirectional_lookup(self):
        """Verify CorpusParagraphMap accurately translates baseline and clean IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cand.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("""
            CREATE TABLE corpus_paragraph_map (
                baseline_paragraph_id INTEGER PRIMARY KEY,
                clean_paragraph_id INTEGER,
                document_key TEXT,
                baseline_document_id TEXT,
                clean_document_id TEXT,
                page_number INTEGER,
                match_type TEXT,
                similarity REAL,
                confidence REAL,
                baseline_text_preview TEXT,
                clean_text_preview TEXT
            );
            """)
            conn.execute("INSERT INTO corpus_paragraph_map VALUES (101, 201, 'book1', 'b1', 'c1', 1, 'exact_normalized', 1.0, 1.0, 'text', 'text');")
            conn.execute("INSERT INTO corpus_paragraph_map VALUES (102, 202, 'book1', 'b1', 'c1', 1, 'minor_variation', 0.95, 0.95, 'text2', 'text2');")
            conn.execute("INSERT INTO corpus_paragraph_map VALUES (103, NULL, 'book1', 'b1', 'c1', 1, 'deleted_ocr_noise', 0.0, 1.0, '10 • author', NULL);")
            conn.commit()
            conn.close()

            pmap = CorpusParagraphMap(db_path)
            # Baseline to Clean
            self.assertEqual(pmap.get_clean_id(101), 201)
            self.assertEqual(pmap.get_clean_id(102), 202)
            self.assertIsNone(pmap.get_clean_id(103))
            self.assertIsNone(pmap.get_clean_id(999))

            # Clean to Baseline
            self.assertEqual(pmap.get_baseline_id(201), 101)
            self.assertEqual(pmap.get_baseline_id(202), 102)
            self.assertIsNone(pmap.get_baseline_id(999))

    def test_noise_detection(self):
        """Verify is_noise_text detects digit lines and running headers."""
        self.assertTrue(CorpusABShadowRunner.is_noise_text("14"))
        self.assertTrue(CorpusABShadowRunner.is_noise_text("  xiv  "))
        self.assertTrue(CorpusABShadowRunner.is_noise_text("10 • Роберт Сапольський"))
        self.assertTrue(CorpusABShadowRunner.is_noise_text("Спогади примата — 25"))
        self.assertFalse(CorpusABShadowRunner.is_noise_text("Бабуїни живуть у складній соціальній ієрархії."))

    def test_ab_shadow_runner_evaluate_and_rank_lift(self):
        """Verify shadow runner evaluation, noise counting, overlap, and rank lift."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_cand.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("""
            CREATE TABLE corpus_paragraph_map (
                baseline_paragraph_id INTEGER PRIMARY KEY,
                clean_paragraph_id INTEGER
            );
            """)
            conn.execute("INSERT INTO corpus_paragraph_map VALUES (10, 100);")
            conn.execute("INSERT INTO corpus_paragraph_map VALUES (11, 101);")
            conn.execute("INSERT INTO corpus_paragraph_map VALUES (12, 102);")
            conn.commit()
            conn.close()

            pmap = CorpusParagraphMap(db_path)

            class MockResult:
                def __init__(self, pid, text, score=0.85):
                    self.paragraph_id = pid
                    self.text = text
                    self.score = score

            class MockLib:
                def __init__(self, fts_results, sem_results):
                    self._fts = fts_results
                    self._sem = sem_results

                def search(self, query, mode="fts", limit=10):
                    return self._fts[:limit] if mode == "fts" else self._sem[:limit]

            # Baseline has noise at #1 in FTS and target at #3
            b_fts = [
                MockResult(1, "12 • Заголовок"),
                MockResult(2, "Випадковий абзац"),
                MockResult(10, "Цільовий абзац про павіанів"),
            ]
            b_sem = [
                MockResult(10, "Цільовий абзац про павіанів", 0.88),
                MockResult(11, "Другий абзац", 0.84),
            ]

            # Clean has noise removed, target elevated to #1 in FTS
            c_fts = [
                MockResult(100, "Цільовий абзац про павіанів"),
                MockResult(200, "Випадковий абзац"),
            ]
            c_sem = [
                MockResult(100, "Цільовий абзац про павіанів", 0.89),
                MockResult(101, "Другий абзац", 0.85),
            ]

            runner = CorpusABShadowRunner(
                baseline_lib=MockLib(b_fts, b_sem),
                candidate_lib=MockLib(c_fts, c_sem),
                para_map=pmap,
            )

            res = runner.evaluate_query("павіани", target_baseline_id=10, top_k=5)

            # Target baseline #10 mapped to clean #100
            self.assertEqual(res["target_baseline_id"], 10)
            self.assertEqual(res["target_clean_id"], 100)

            # FTS rank improved from #3 to #1 (+2 lift)
            self.assertEqual(res["baseline_fts_rank"], 3)
            self.assertEqual(res["clean_fts_rank"], 1)
            self.assertEqual(res["fts_rank_lift"], 2)

            # Semantic rank stayed #1 (0 lift)
            self.assertEqual(res["baseline_sem_rank"], 1)
            self.assertEqual(res["clean_sem_rank"], 1)
            self.assertEqual(res["sem_rank_lift"], 0)

            # Noise metrics
            self.assertEqual(res["baseline_noise_at_5"], 1)
            self.assertEqual(res["clean_noise_at_5"], 0)

            # Content Overlap@5: both 100 and 101 are matched (10->100, 11->101)
            self.assertEqual(res["content_overlap_at_5_pct"], 100.0)

    def test_promotion_gate_verdict(self):
        """Verify Promotion Gate checks all operational & relevance criteria."""
        class MockCandLib:
            def __init__(self, admin_ok=True, sem_ok=True):
                self._admin_ok = admin_ok
                self._sem_ok = sem_ok

            def admin_verify(self):
                return {"all_healthy": self._admin_ok}

            def verify_semantic_integrity(self, model_id=None):
                return {"semantic_ready": self._sem_ok, "embeddings_present": 100, "total_chunks": 100}

        runner = CorpusABShadowRunner(baseline_lib=None, candidate_lib=MockCandLib(True, True))

        # PASS case (n>=30 triggers strict 10% regression threshold)
        good_summary = {
            "clean_noise_at_10_pct": 0.0,
            "baseline_noise_at_10_pct": 5.0,
            "regressed_pct": 0.0,
            "median_semantic_rank_lift": 1,
            "judged_queries": 100,
        }
        verdict = runner.check_promotion_gate(good_summary)
        self.assertTrue(verdict["all_gates_passed"])
        self.assertEqual(verdict["status"], "PASS")

        # FAIL cases:
        # 1. High clean noise
        bad_noise = dict(good_summary, clean_noise_at_10_pct=3.5)
        self.assertFalse(runner.check_promotion_gate(bad_noise)["all_gates_passed"])

        # 2. Too many regressions (>10% with n>=30)
        bad_reg = dict(good_summary, regressed_pct=15.0)
        self.assertFalse(runner.check_promotion_gate(bad_reg)["all_gates_passed"])

        # 3. Negative rank lift
        bad_lift = dict(good_summary, median_semantic_rank_lift=-1)
        self.assertFalse(runner.check_promotion_gate(bad_lift)["all_gates_passed"])

        # 4. Corrupted candidate DB
        corrupt_runner = CorpusABShadowRunner(baseline_lib=None, candidate_lib=MockCandLib(admin_ok=False))
        self.assertFalse(corrupt_runner.check_promotion_gate(good_summary)["all_gates_passed"])


if __name__ == "__main__":
    unittest.main()
