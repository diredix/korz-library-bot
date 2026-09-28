"""
tests/test_semantic_v2a.py - Phase V2A Verification Suite for KORZ Semantic Infrastructure

Covers tests 19 through 30 as specified in the Phase V2A architecture:
  19. test_migration_v1_to_v2_preserves_existing_data
  20. test_migration_is_idempotent
  21. test_schema_version_is_2
  22. test_v1_search_results_identical_after_migration (20 control queries)
  23. test_embedding_model_contract_is_deterministic
  24. test_embedding_blob_roundtrip_float32
  25. test_invalid_vector_dimension_rejected
  26. test_chunk_delete_cascades_embedding
  27. test_model_delete_restricted_when_vectors_exist
  28. test_text_sha_change_marks_embedding_stale
  29. test_semantic_state_counts_are_consistent
  30. test_backup_restore_preserves_schema_v2
"""

import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_core import (
    EmbeddingContract,
    KorzConfig,
    KorzLibrary,
    _apply_migration_v1,
    _apply_migration_v2,
    blob_to_vector,
    build_embedding_text,
    compute_text_sha256,
    get_connection,
    vector_to_blob,
)


class TestSemanticV2A(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="korz_v2a_test_")
        self.temp_path = Path(self.temp_dir)
        self.db_path = self.temp_path / "data" / "test_v2a.sqlite"
        self.sources_dir = self.temp_path / "data" / "sources"
        self.backups_dir = self.temp_path / "data" / "backups"

        self.config = KorzConfig(
            data_dir=self.temp_path / "data",
            db_path=self.db_path,
            sources_dir=self.sources_dir,
            backups_dir=self.backups_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_v1_to_v2_preserves_existing_data(self):
        """Test 19: Verify that upgrading from schema v1 to v2 preserves all existing data."""
        # 1. Manually initialize a pure v1 database
        conn = get_connection(self.db_path)
        _apply_migration_v1(conn)
        self.assertEqual(conn.execute("PRAGMA user_version;").fetchone()[0], 1)

        # 2. Insert dummy book and search logs
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?)",
            ("sha123", "test.md", 100, "text/markdown", "sources/test.md", now_iso),
        )
        conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("doc1", "Тестова Книга", "Автор", "uk", "1969", "sha123", 1, 2, 1, now_iso),
        )
        conn.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?)", ("doc1:p1", "doc1", 1, "raw", "clean", 2))
        conn.execute(
            """
            INSERT INTO paragraphs (
                id, paragraph_key, document_id, page_id, page_number,
                paragraph_index, char_start, char_end, raw_text, display_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "doc1:p1:par1", "doc1", "doc1:p1", 1, 1, 0, 10, "raw", "clean"),
        )
        conn.execute(
            """
            INSERT INTO chunks (
                id, chunk_key, document_id, page_start, page_end,
                paragraph_start_id, paragraph_end_id, display_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "doc1:p1-1:c1", "doc1", 1, 1, 1, 1, "chunk text"),
        )
        conn.execute("INSERT INTO search_log VALUES (?, ?, ?, ?, ?, ?, ?)", (1, "тест", "auto", 1, 1, 2.5, now_iso))
        conn.close()

        # 3. Apply v2 migration
        conn2 = get_connection(self.db_path)
        _apply_migration_v2(conn2, backups_dir=self.backups_dir)

        # 4. Verify all tables and counts preserved
        self.assertEqual(conn2.execute("PRAGMA user_version;").fetchone()[0], 2)
        self.assertEqual(conn2.execute("SELECT COUNT(*) FROM documents;").fetchone()[0], 1)
        self.assertEqual(conn2.execute("SELECT COUNT(*) FROM pages;").fetchone()[0], 1)
        self.assertEqual(conn2.execute("SELECT COUNT(*) FROM paragraphs;").fetchone()[0], 1)
        self.assertEqual(conn2.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0], 1)
        self.assertEqual(conn2.execute("SELECT COUNT(*) FROM search_log;").fetchone()[0], 1)
        conn2.close()

    def test_migration_is_idempotent(self):
        """Test 20: Calling _apply_migration_v2 multiple times does not error or corrupt state."""
        conn = get_connection(self.db_path)
        _apply_migration_v1(conn)
        _apply_migration_v2(conn)
        self.assertEqual(conn.execute("PRAGMA user_version;").fetchone()[0], 2)

        # Re-apply migration v2
        _apply_migration_v2(conn)
        self.assertEqual(conn.execute("PRAGMA user_version;").fetchone()[0], 2)

        # Verify tables still exist and are accessible
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_embeddings'")
        self.assertIsNotNone(cur.fetchone())
        conn.close()

    def test_schema_version_is_2(self):
        """Test 21: KorzLibrary initializes schema version 2 and registers migration log."""
        with KorzLibrary(self.config) as lib:
            st = lib.status()
            self.assertEqual(st["schema_version"], 2)

            cur_mig = lib.conn.execute("SELECT version, description FROM schema_migrations WHERE version = 2")
            row = cur_mig.fetchone()
            self.assertIsNotNone(row)
            self.assertIn("v2 additive semantic schema", row["description"])

    def test_v1_search_results_identical_after_migration(self):
        """Test 22: Verify that 20 diverse control queries produce 100% identical results before and after migration."""
        # 1. Create a pure v1 library and add rich test content
        v1_db_path = self.temp_path / "v1_control.sqlite"
        conn_v1 = get_connection(v1_db_path)
        _apply_migration_v1(conn_v1)
        conn_v1.close()

        v1_config = KorzConfig(
            data_dir=self.temp_path,
            db_path=v1_db_path,
            sources_dir=self.sources_dir,
            backups_dir=self.backups_dir,
        )

        with KorzLibrary(v1_config) as lib_v1:
            test_doc = self.temp_path / "control_book.md"
            test_doc.write_text(
                "{1}------------------------------------------------\n"
                "# Розділ 1: Стрес, біологія та суспільство\n\n"
                "Robert Sapolsky досліджував павіанів та зв'язок між соціальним статусом і глюкокортикоїдами.\n\n"
                "Коли примат стикається з психологічним стресом, його тиск піднімається.\n\n"
                "{2}------------------------------------------------\n"
                "# Розділ 2: Технологічний устрій Жака Фреско\n\n"
                "Jacque Fresco запропонував проєкт «Венера» та ресурсоорієнтовану економіку.\n\n"
                "Нам потрібна мова з високим ступенем фізичної кореляції з реальним світом.\n\n"
                "Пам'ять про минуле формує будь-який вибір у 1969-2026 роках.",
                encoding="utf-8",
            )
            lib_v1.add_book(test_doc, title="Control Book", author="Sapolsky & Fresco")

            # 2. Run 20 control queries on v1 baseline and snapshot outputs
            control_queries = [
                "стрес",
                "павіанів",
                "соціальним статусом",
                "глюкокортикоїдами",
                "Robert Sapolsky",
                "«Венера»",
                "Jacque Fresco",
                "ресурсоорієнтовану економіку",
                "фізичної кореляції",
                "пам'ять",
                "будь-який",
                "1969",
                "тиск піднімається",
                "мова з високим ступенем",
                "«фізичної кореляції»",
                "примат",
                "реальним світом",
                "психологічним",
                "досліджував",
                "неіснуюче_слово_контроль",
            ]

            baseline_results = {}
            for q in control_queries:
                res = lib_v1.search(q, mode="auto", limit=5)
                baseline_results[q] = [(r.paragraph_id, r.score, r.match_type, r.text) for r in res]

            # 3. Apply migration to v2 on the same database
            _apply_migration_v2(lib_v1.conn)

            # 4. Re-run all 20 queries and assert exact identical parity
            for q in control_queries:
                post_res = lib_v1.search(q, mode="auto", limit=5)
                post_tuples = [(r.paragraph_id, r.score, r.match_type, r.text) for r in post_res]
                self.assertEqual(
                    baseline_results[q],
                    post_tuples,
                    f"Mismatch on query '{q}' after migration v2!",
                )

    def test_embedding_model_contract_is_deterministic(self):
        """Test 23: EmbeddingContract generates deterministic config_sha256 and model_id."""
        c1 = EmbeddingContract(
            model_name="intfloat/multilingual-e5-large",
            dimension=1024,
            normalization="L2",
            query_template="query: {text}",
            passage_template="passage: {text}",
        )
        c2 = EmbeddingContract(
            model_name="intfloat/multilingual-e5-large",
            dimension=1024,
            normalization="l2",
            query_template="query: {text}",
            passage_template="passage: {text}",
        )

        self.assertEqual(c1.compute_config_sha256(), c2.compute_config_sha256())
        self.assertEqual(c1.generate_model_id(), c2.generate_model_id())
        self.assertTrue(c1.generate_model_id().startswith("emb-multilingual-e5-large-"))

    def test_embedding_blob_roundtrip_float32(self):
        """Test 24: Float32 numpy array serializes to BLOB and deserializes with exact fidelity."""
        dim = 1024
        rng = np.random.default_rng(42)
        raw_vec = rng.standard_normal(dim).astype(np.float32)

        blob = vector_to_blob(raw_vec, expected_dim=dim, normalize=True)
        self.assertEqual(len(blob), dim * 4)  # 4096 bytes

        restored = blob_to_vector(blob, expected_dim=dim)
        self.assertEqual(restored.shape, (dim,))
        self.assertEqual(restored.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(restored)), 1.0, places=5)

        # Check values match normalized original
        expected_norm = raw_vec / np.linalg.norm(raw_vec)
        np.testing.assert_allclose(restored, expected_norm, atol=1e-6)

    def test_invalid_vector_dimension_rejected(self):
        """Test 25: Incorrect dimensions or non-finite vectors are rejected with ValueError."""
        # 1. Dimension mismatch
        vec_512 = np.ones(512, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            vector_to_blob(vec_512, expected_dim=1024)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        # 2. Non-finite values
        vec_nan = np.ones(1024, dtype=np.float32)
        vec_nan[10] = np.nan
        with self.assertRaises(ValueError) as ctx:
            vector_to_blob(vec_nan, expected_dim=1024)
        self.assertIn("non-finite", str(ctx.exception).lower())

    def test_chunk_delete_cascades_embedding(self):
        """Test 26: Deleting a chunk cascades and removes its embeddings automatically."""
        with KorzLibrary(self.config) as lib:
            # Seed book
            doc_file = self.temp_path / "cascade_test.md"
            doc_file.write_text("{1}------------------------------------------------\nАбзац для каскадного видалення.", encoding="utf-8")
            res = lib.add_book(doc_file, title="Cascade Book")
            doc_id = res["document_id"]

            contract = EmbeddingContract(model_name="test-model", dimension=4)
            model_id = lib.register_embedding_model(contract)

            # Get chunk id
            cur = lib.conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
            chunk_id = cur.fetchone()[0]

            # Save embedding
            vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            lib.save_chunk_embedding(chunk_id, model_id, vec, "sha_test")

            # Verify embedding exists
            self.assertIsNotNone(lib.get_chunk_embedding(chunk_id, model_id))

            # Delete book (which deletes chunk)
            lib.remove_book(doc_id)

            # Verify chunk_embedding is automatically deleted via CASCADE
            cur_emb = lib.conn.execute("SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,))
            self.assertEqual(cur_emb.fetchone()[0], 0)

    def test_model_delete_restricted_when_vectors_exist(self):
        """Test 27: Deleting a model that has existing chunk embeddings raises IntegrityError (RESTRICT)."""
        with KorzLibrary(self.config) as lib:
            doc_file = self.temp_path / "restrict_test.md"
            doc_file.write_text("{1}------------------------------------------------\nТекст для restrict тесту.", encoding="utf-8")
            res = lib.add_book(doc_file, title="Restrict Book")
            doc_id = res["document_id"]

            contract = EmbeddingContract(model_name="restrict-model", dimension=4)
            model_id = lib.register_embedding_model(contract)

            cur = lib.conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
            chunk_id = cur.fetchone()[0]

            lib.save_chunk_embedding(chunk_id, model_id, np.array([1, 0, 0, 0], dtype=np.float32), "sha_r")

            # Deleting model must fail due to ON DELETE RESTRICT
            with self.assertRaises(sqlite3.IntegrityError):
                lib.delete_embedding_model(model_id)

    def test_text_sha_change_marks_embedding_stale(self):
        """Test 28: Stored text_sha256 mismatch with current chunk text is detected as STALE."""
        with KorzLibrary(self.config) as lib:
            doc_file = self.temp_path / "stale_test.md"
            doc_file.write_text("{1}------------------------------------------------\nПочатковий текст чанку.", encoding="utf-8")
            res = lib.add_book(doc_file, title="Stale Book")
            doc_id = res["document_id"]

            contract = EmbeddingContract(model_name="stale-model", dimension=4)
            model_id = lib.register_embedding_model(contract)

            cur = lib.conn.execute("SELECT id, display_text FROM chunks WHERE document_id = ?", (doc_id,))
            row = cur.fetchone()
            chunk_id, raw_txt = row["id"], row["display_text"]

            # Save with wrong/outdated SHA
            wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"
            lib.save_chunk_embedding(chunk_id, model_id, np.array([1, 0, 0, 0], dtype=np.float32), wrong_sha)

            # verify_semantic_integrity should detect stale embedding
            report = lib.verify_semantic_integrity(model_id)
            self.assertFalse(report["all_healthy"])
            self.assertEqual(report["stale_embeddings"], 1)

            # Update with correct SHA
            correct_emb_text = build_embedding_text(raw_txt)
            correct_sha = compute_text_sha256(correct_emb_text)
            lib.save_chunk_embedding(chunk_id, model_id, np.array([1, 0, 0, 0], dtype=np.float32), correct_sha)

            report2 = lib.verify_semantic_integrity(model_id)
            self.assertTrue(report2["all_healthy"])
            self.assertEqual(report2["stale_embeddings"], 0)

    def test_semantic_state_counts_are_consistent(self):
        """Test 29: semantic_state accurately tracks indexing status and chunk counts."""
        with KorzLibrary(self.config) as lib:
            doc_file = self.temp_path / "state_test.md"
            doc_file.write_text("{1}------------------------------------------------\nТекст для state тесту.", encoding="utf-8")
            res = lib.add_book(doc_file, title="State Book")
            doc_id = res["document_id"]

            contract = EmbeddingContract(model_name="state-model", dimension=4)
            model_id = lib.register_embedding_model(contract)

            lib.update_semantic_state(doc_id, model_id, status="INDEXING", embedded_chunks=10, total_chunks=50)

            cur = lib.conn.execute("SELECT * FROM semantic_state WHERE document_id = ?", (doc_id,))
            row = cur.fetchone()
            self.assertEqual(row["status"], "INDEXING")
            self.assertEqual(row["embedded_chunks"], 10)
            self.assertEqual(row["total_chunks"], 50)

            # Update to READY
            lib.update_semantic_state(doc_id, model_id, status="READY", embedded_chunks=50, total_chunks=50)
            cur2 = lib.conn.execute("SELECT * FROM semantic_state WHERE document_id = ?", (doc_id,))
            row2 = cur2.fetchone()
            self.assertEqual(row2["status"], "READY")
            self.assertEqual(row2["embedded_chunks"], 50)

    def test_backup_restore_preserves_schema_v2(self):
        """Test 30: Backup and restore preserves schema version 2 and semantic tables/vectors."""
        with KorzLibrary(self.config) as lib:
            doc_file = self.temp_path / "backup_v2.md"
            doc_file.write_text("{1}------------------------------------------------\nТекст для бекапу V2.", encoding="utf-8")
            res = lib.add_book(doc_file, title="Backup V2 Book")
            doc_id = res["document_id"]

            contract = EmbeddingContract(model_name="backup-model", dimension=4)
            model_id = lib.register_embedding_model(contract)

            cur = lib.conn.execute("SELECT id FROM chunks WHERE document_id = ?", (doc_id,))
            chunk_id = cur.fetchone()[0]
            test_vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
            lib.save_chunk_embedding(chunk_id, model_id, test_vec, "sha_bck")

            # Create backup
            backup_path = lib.backup()

        # Restore into a fresh location
        restore_db = self.temp_path / "restored_v2.sqlite"
        restore_config = KorzConfig(
            data_dir=self.temp_path,
            db_path=restore_db,
            sources_dir=self.sources_dir,
            backups_dir=self.backups_dir,
        )

        with KorzLibrary(restore_config) as lib_restored:
            lib_restored.restore(backup_path)

            st = lib_restored.status()
            self.assertEqual(st["schema_version"], 2)

            models = lib_restored.list_embedding_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["model_id"], model_id)

            restored_vec = lib_restored.get_chunk_embedding(chunk_id, model_id)
            self.assertIsNotNone(restored_vec)
            expected_norm = test_vec / np.linalg.norm(test_vec)
            np.testing.assert_allclose(restored_vec, expected_norm, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
