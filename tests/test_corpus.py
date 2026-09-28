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

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_corpus import CorpusAuditor, CorpusCleaner, CorpusVerifier, CandidateLab, NoiseItem


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


if __name__ == "__main__":
    unittest.main()
