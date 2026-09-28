#!/usr/bin/env python3
"""
test_semantic_v2b.py - Comprehensive Unit & Regression Test Suite for KORZ V2 Phase V2B

Tests 31 to 45:
  31. test_embedding_backend_contract
  32. test_query_and_passage_templates_are_distinct
  33. test_build_all_chunks
  34. test_build_resume_after_interruption
  35. test_unchanged_chunk_is_skipped
  36. test_changed_chunk_is_reembedded
  37. test_embedding_input_sha_matches_actual_input
  38. test_vectors_are_float32_and_normalized
  39. test_semantic_matrix_row_mapping_is_stable
  40. test_semantic_search_returns_expected_chunk
  41. test_doc_id_filter_works_in_semantic_mode
  42. test_semantic_search_disabled_by_default
  43. test_v1_auto_results_remain_identical
  44. test_semantic_index_reload_after_restart
  45. test_corrupted_vector_blocks_ready
"""

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_core import (
    DEFAULT_E5_BASE_CONTRACT,
    EmbeddingBackend,
    EmbeddingContract,
    FastEmbedBackend,
    KorzConfig,
    KorzLibrary,
    MockEmbeddingBackend,
    SemanticIndex,
    blob_to_vector,
    build_embedding_input,
    build_embedding_text,
    compute_text_sha256,
    get_connection,
    vector_to_blob,
)


class TestSemanticV2B(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="korz_v2b_test_")
        self.temp_path = Path(self.temp_dir)
        self.db_path = self.temp_path / "data" / "test_v2b.sqlite"
        self.sources_dir = self.temp_path / "data" / "sources"
        self.backups_dir = self.temp_path / "data" / "backups"

        self.config = KorzConfig(
            data_dir=self.temp_path / "data",
            db_path=self.db_path,
            sources_dir=self.sources_dir,
            backups_dir=self.backups_dir,
            semantic_enabled=True,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sample_book(self, filename: str, content: str) -> Path:
        p = self.temp_path / filename
        p.write_text(content, encoding="utf-8")
        return p

    def test_embedding_backend_contract(self):
        """Test 31: EmbeddingBackend contract guarantees float32, L2 normalization and token count."""
        backend = MockEmbeddingBackend(dimension=768)
        passages = ["Це перший тестовий уривок", "Це другий тестовий уривок"]
        queries = ["тестовий запит"]

        p_vecs = backend.encode_passages(passages)
        self.assertEqual(p_vecs.shape, (2, 768))
        self.assertEqual(p_vecs.dtype, np.float32)
        for v in p_vecs:
            norm = np.linalg.norm(v)
            self.assertAlmostEqual(norm, 1.0, places=4)

        q_vecs = backend.encode_queries(queries)
        self.assertEqual(q_vecs.shape, (1, 768))
        self.assertEqual(q_vecs.dtype, np.float32)
        self.assertAlmostEqual(np.linalg.norm(q_vecs[0]), 1.0, places=4)

        # Token counting contract
        count = backend.count_tokens("одне два три чотири п'ять")
        self.assertEqual(count, 5)

    def test_query_and_passage_templates_are_distinct(self):
        """Test 32: E5 contract ensures query and passage templates produce distinct inputs and distinct vectors."""
        contract = DEFAULT_E5_BASE_CONTRACT
        self.assertEqual(contract.query_template, "query: {text}")
        self.assertEqual(contract.passage_template, "passage: {text}")

        raw_text = "фізична кореляція з навколишнім середовищем"
        q_text = build_embedding_input(raw_text, contract.query_template)
        p_text = build_embedding_input(raw_text, contract.passage_template)

        self.assertNotEqual(q_text, p_text)
        self.assertTrue(q_text.startswith("query: "))
        self.assertTrue(p_text.startswith("passage: "))

        backend = MockEmbeddingBackend(dimension=768)
        q_vec = backend.encode_queries([q_text])[0]
        p_vec = backend.encode_passages([p_text])[0]

        # Vectors must not be identical because prefixes differ
        self.assertFalse(np.allclose(q_vec, p_vec))

    def test_build_all_chunks(self):
        """Test 33: build_semantic_index embeds all chunks and transitions state from INDEXING to READY."""
        book_content = (
            "{1}------------------------------------------------\n"
            "Перший абзац про фізіологію стресу та поведінку приматів.\n\n"
            "Другий абзац про соціальну ієрархію та гормони наднирників.\n\n"
            "{2}------------------------------------------------\n"
            "Третій абзац про наслідки хронічного навантаження на серце.\n\n"
            "Четвертий абзац про ресурсоорієнтовану економіку Жака Фреско.\n"
        )
        book_file = self._create_sample_book("book1.md", book_content)

        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Книга 1", author="Автор 1")
            st_before = lib.status()
            total_chunks = st_before["total_chunks"]
            self.assertGreater(total_chunks, 0)

            # Build semantic index using mock backend
            mock_backend = MockEmbeddingBackend(dimension=768)
            res = lib.build_semantic_index(backend=mock_backend)

            self.assertEqual(res["total_chunks"], total_chunks)
            self.assertEqual(res["embedded"], total_chunks)
            self.assertEqual(res["cached"], 0)
            self.assertTrue(res["ready"])

            # Verify semantic state in DB
            cur = lib.conn.execute("SELECT status, embedded_chunks, total_chunks FROM semantic_state")
            state_row = cur.fetchone()
            self.assertEqual(state_row["status"], "READY")
            self.assertEqual(state_row["embedded_chunks"], total_chunks)
            self.assertEqual(state_row["total_chunks"], total_chunks)

    def test_build_resume_after_interruption(self):
        """Test 34: Interrupted build resumes safely without re-embedding previously completed chunks."""
        pages = []
        for p in range(1, 7):
            para = f"Це сторінка {p}. " + ("Довгий науковий текст з детальними поясненнями біології та поведінки. " * 15)
            pages.append(f"{{{p}}}------------------------------------------------\n{para}")
        book_file = self._create_sample_book("large_book.md", "\n\n".join(pages))

        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Велика Книга", author="Дослідник")
            total_chunks = lib.status()["total_chunks"]
            self.assertGreaterEqual(total_chunks, 4)

            model_id = lib.register_embedding_model(DEFAULT_E5_BASE_CONTRACT)
            mock_backend = MockEmbeddingBackend(dimension=768)

            # Simulate interruption: manually embed only the first 2 chunks
            cur = lib.conn.execute("SELECT id, display_text FROM chunks ORDER BY id ASC LIMIT 2")
            first_chunks = cur.fetchall()
            for r in first_chunks:
                emb_input = build_embedding_input(r["display_text"], "passage: {text}")
                sha = compute_text_sha256(emb_input)
                vec = mock_backend.encode_passages([emb_input])[0]
                blob = vector_to_blob(vec, 768)
                lib.conn.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, model_id, text_sha256, vector, created_at)
                    VALUES (?, ?, ?, ?, '2026-09-28T00:00:00Z')
                    """,
                    (r["id"], model_id, sha, blob),
                )

            # Resume build: should skip the 2 chunks and only embed the remaining chunks
            res = lib.build_semantic_index(model_id=model_id, backend=mock_backend)
            self.assertEqual(res["cached"], 2)
            self.assertEqual(res["embedded"], total_chunks - 2)
            self.assertTrue(res["ready"])

    def test_unchanged_chunk_is_skipped(self):
        """Test 35: Re-running build on an already-indexed library skips 100% of chunks."""
        book_file = self._create_sample_book(
            "skip_test.md",
            "{1}------------------------------------------------\nАбзац для перевірки пропуску незмінених чанків.",
        )
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Пропуск", author="Тестер")
            total_chunks = lib.status()["total_chunks"]
            mock_backend = MockEmbeddingBackend(dimension=768)

            # First build
            res1 = lib.build_semantic_index(backend=mock_backend)
            self.assertEqual(res1["embedded"], total_chunks)

            # Second build without changes
            res2 = lib.build_semantic_index(backend=mock_backend)
            self.assertEqual(res2["embedded"], 0)
            self.assertEqual(res2["cached"], total_chunks)
            self.assertTrue(res2["ready"])

    def test_changed_chunk_is_reembedded(self):
        """Test 36: When a chunk's text changes in SQLite, only that specific chunk is re-embedded."""
        book_file = self._create_sample_book(
            "change_test.md",
            "{1}------------------------------------------------\nАбзац 1 текст.\n\n{2}------------------------------------------------\nАбзац 2 текст.",
        )
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Зміна", author="Тестер")
            mock_backend = MockEmbeddingBackend(dimension=768)
            res1 = lib.build_semantic_index(backend=mock_backend)
            total = res1["total_chunks"]

            # Modify text of first chunk in SQLite
            cur = lib.conn.execute("SELECT id FROM chunks ORDER BY id ASC LIMIT 1")
            first_id = cur.fetchone()[0]
            lib.conn.execute(
                "UPDATE chunks SET display_text = 'Оновлений змінений текст для перевірки re-embed' WHERE id = ?",
                (first_id,),
            )

            # Re-run build: exactly 1 chunk should be embedded, remaining should be cached
            res2 = lib.build_semantic_index(backend=mock_backend)
            self.assertEqual(res2["embedded"], 1)
            self.assertEqual(res2["cached"], total - 1)
            self.assertTrue(res2["ready"])

    def test_embedding_input_sha_matches_actual_input(self):
        """Test 37: text_sha256 matches exact preprocessed embedding_input even after token truncation."""
        backend = MockEmbeddingBackend(dimension=768)

        # Max tokens = 4 words
        long_raw_text = "слово1 слово2 слово3 слово4 слово5 слово6 слово7"
        template = "passage: {text}"
        truncated_input = build_embedding_input(long_raw_text, template, model_max_tokens=4, backend=backend)

        # Must fit within 4 tokens
        self.assertLessEqual(backend.count_tokens(truncated_input), 4)

        computed_sha = compute_text_sha256(truncated_input)
        expected_direct_sha = hashlib.sha256(truncated_input.encode("utf-8")).hexdigest()
        self.assertEqual(computed_sha, expected_direct_sha)

    def test_vectors_are_float32_and_normalized(self):
        """Test 38: Stored vector blobs deserialize strictly to float32 with L2 norm == 1.0."""
        book_file = self._create_sample_book(
            "norm_test.md",
            "{1}------------------------------------------------\nПеревірка норми векторів float32.",
        )
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Норма", author="Тестер")
            mock_backend = MockEmbeddingBackend(dimension=768)
            lib.build_semantic_index(backend=mock_backend)

            cur = lib.conn.execute("SELECT vector FROM chunk_embeddings")
            rows = cur.fetchall()
            self.assertGreater(len(rows), 0)
            for r in rows:
                dim = 768
                vec = blob_to_vector(r["vector"], dim)
                self.assertEqual(vec.dtype, np.float32)
                norm = float(np.linalg.norm(vec))
                self.assertAlmostEqual(norm, 1.0, places=4)
                self.assertTrue(np.all(np.isfinite(vec)))

    def test_semantic_matrix_row_mapping_is_stable(self):
        """Test 39: In SemanticIndex, matrix row i strictly maps to chunk_ids[i] and paragraph_ids[i]."""
        book_file = self._create_sample_book(
            "stable_map.md",
            "{1}------------------------------------------------\nБлок А.\n\n{2}------------------------------------------------\nБлок Б.\n\n{3}------------------------------------------------\nБлок В.",
        )
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Стабільність", author="Тестер")
            mock_backend = MockEmbeddingBackend(dimension=768)
            lib.build_semantic_index(backend=mock_backend)

            index = lib.load_semantic_index()
            self.assertIsInstance(index, SemanticIndex)
            self.assertEqual(index.matrix.shape[0], len(index.chunk_ids))
            self.assertEqual(len(index.chunk_ids), len(index.paragraph_ids))
            self.assertEqual(len(index.chunk_ids), len(index.doc_ids))

            # Verify that matrix[i] matches the stored blob for chunk_ids[i]
            for i, c_id in enumerate(index.chunk_ids):
                stored_vec = lib.get_chunk_embedding(int(c_id), index.model_id)
                self.assertIsNotNone(stored_vec)
                self.assertTrue(np.allclose(index.matrix[i], stored_vec, atol=1e-5))

    def test_semantic_search_returns_expected_chunk(self):
        """Test 40: Semantic search retrieves correct paragraph with SearchResult contract."""
        book_content = (
            "{1}------------------------------------------------\n"
            + ("Звичайний вступ про погоду та природу в далеких горах Карпат. " * 25)
            + "\n\n{2}------------------------------------------------\n"
            + ("Сапольський докладно пояснює як примати реагують на соціальний стрес у дикій зграї. " * 25)
            + "\n\n{3}------------------------------------------------\n"
            + ("Жак Фреско описує концепцію ресурсної економіки та повну автоматизацію виробництва. " * 25)
        )
        book_file = self._create_sample_book("search_book.md", book_content)

        contract = EmbeddingContract(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            dimension=384,
            normalization="l2",
            query_template="{text}",
            passage_template="{text}",
            model_max_tokens=512,
        )
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Пошукова книга", author="Роберт Сапольський")
            m_id = lib.register_embedding_model(contract)
            lib.build_semantic_index(model_id=m_id)

            # Query targeting page 2 content
            results = lib.search("примати і соціальний стрес у зграї", mode="semantic", limit=3)
            self.assertGreater(len(results), 0)
            top = results[0]

            self.assertEqual(top.match_type, "semantic")
            self.assertEqual(top.title, "Пошукова книга")
            self.assertEqual(top.author, "Роберт Сапольський")
            self.assertEqual(top.page, 2)
            self.assertIn("соціальний стрес", top.text)
            self.assertGreater(top.score, 0.7)

    def test_doc_id_filter_works_in_semantic_mode(self):
        """Test 41: doc_id filter in semantic search strictly filters out results from other documents."""
        b1 = self._create_sample_book("doc1.md", "{1}------------------------------------------------\nТема приматів у першій книзі.")
        b2 = self._create_sample_book("doc2.md", "{1}------------------------------------------------\nТема приматів у другій книзі.")

        with KorzLibrary(self.config) as lib:
            doc1_res = lib.add_book(b1, title="Книга 1", author="Автор 1")
            doc2_res = lib.add_book(b2, title="Книга 2", author="Автор 2")
            mock_backend = MockEmbeddingBackend(dimension=768)
            lib.build_semantic_index(backend=mock_backend)

            # Filter search strictly to doc2
            results = lib.search("примати", mode="semantic", limit=10, doc_id=doc2_res["document_id"])
            self.assertGreater(len(results), 0)
            for r in results:
                self.assertEqual(r.document_id, doc2_res["document_id"])
                self.assertEqual(r.title, "Книга 2")

    def test_semantic_search_disabled_by_default(self):
        """Test 42: When semantic_enabled=False, search(mode='semantic') raises RuntimeError."""
        disabled_config = KorzConfig(
            data_dir=self.temp_path / "data",
            db_path=self.db_path,
            sources_dir=self.sources_dir,
            backups_dir=self.backups_dir,
            semantic_enabled=False,
        )
        book_file = self._create_sample_book("test_dis.md", "{1}------------------------------------------------\nТекст книги.")
        with KorzLibrary(disabled_config) as lib:
            lib.add_book(book_file, title="Вимкнено", author="Автор")
            with self.assertRaises(RuntimeError) as ctx:
                lib.search("будь-який запит", mode="semantic")
            self.assertIn("Semantic search is disabled", str(ctx.exception))

    def test_v1_auto_results_remain_identical(self):
        """Test 43: mode='auto' results remain 100% identical before and after building semantic index."""
        book_content = (
            "{1}------------------------------------------------\n"
            "Щоб покращити комунікацію, нам потрібна мова з високим ступенем фізичної кореляції з навколишнім середовищем.\n\n"
            "{2}------------------------------------------------\n"
            "Психологічний стрес виникає навіть за відсутності прямої фізичної загрози для організму.\n"
        )
        book_file = self._create_sample_book("control_v1.md", book_content)

        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Контроль", author="Автор")

            queries = [
                "фізичної кореляції",
                "комунікацію",
                "психологічний стрес",
                "«фізичної кореляції»",
            ]

            # 1. Baseline search before semantic index
            baseline = {}
            for q in queries:
                res = lib.search(q, mode="auto", limit=5)
                baseline[q] = [(r.paragraph_id, r.score, r.match_type, r.text) for r in res]

            # 2. Build semantic index
            mock_backend = MockEmbeddingBackend(dimension=768)
            lib.build_semantic_index(backend=mock_backend)

            # 3. Search again with mode='auto'
            for q in queries:
                post_res = lib.search(q, mode="auto", limit=5)
                post_tuples = [(r.paragraph_id, r.score, r.match_type, r.text) for r in post_res]
                self.assertEqual(
                    baseline[q],
                    post_tuples,
                    f"Auto mode results diverted for query '{q}' after building semantic index!",
                )

    def test_semantic_index_reload_after_restart(self):
        """Test 44: Closing and reopening KorzLibrary reloads SemanticIndex without rebuilding."""
        book_file = self._create_sample_book("restart.md", "{1}------------------------------------------------\nПеревірка збереження індексу після перезапуску.")
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Перезапуск", author="Автор")
            mock_backend = MockEmbeddingBackend(dimension=768)
            lib.build_semantic_index(backend=mock_backend)

        # Open completely fresh library connection
        with KorzLibrary(self.config) as lib2:
            mock_backend2 = MockEmbeddingBackend(dimension=768)
            lib2.set_backend("intfloat/multilingual-e5-base", mock_backend2)

            # Search should work automatically without running build again
            results = lib2.search("перевірка збереження індексу", mode="semantic", limit=3)
            self.assertGreater(len(results), 0)
            self.assertIn("перезапуску", results[0].text)

    def test_corrupted_vector_blocks_ready(self):
        """Test 45: Corrupted vector in chunk_embeddings causes verify_semantic_integrity to report unhealthiness."""
        book_file = self._create_sample_book("corrupt.md", "{1}------------------------------------------------\nТекст для перевірки пошкодження вектора.")
        with KorzLibrary(self.config) as lib:
            lib.add_book(book_file, title="Пошкодження", author="Автор")
            mock_backend = MockEmbeddingBackend(dimension=768)
            lib.build_semantic_index(backend=mock_backend)

            # Baseline check: healthy
            res_clean = lib.verify_semantic_integrity()
            self.assertTrue(res_clean["all_healthy"])
            self.assertTrue(res_clean["semantic_ready"])

            # Corrupt vector with invalid bytes
            cur = lib.conn.execute("SELECT chunk_id, model_id FROM chunk_embeddings LIMIT 1")
            row = cur.fetchone()
            bad_blob = b"corrupted_bad_blob_data"
            lib.conn.execute(
                "UPDATE chunk_embeddings SET vector = ? WHERE chunk_id = ? AND model_id = ?",
                (bad_blob, row["chunk_id"], row["model_id"]),
            )

            # Verify must catch the invalid blob and report unhealthy
            res_corrupt = lib.verify_semantic_integrity(row["model_id"])
            self.assertFalse(res_corrupt["all_healthy"])
            self.assertFalse(res_corrupt["semantic_ready"])
            self.assertGreater(res_corrupt["invalid_blob"], 0)


if __name__ == "__main__":
    unittest.main()
