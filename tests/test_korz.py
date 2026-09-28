"""
tests/test_korz.py - Comprehensive Unit & Integration Tests for KORZ LOCAL (SQLite Engine)
Covers:
  - Database init, schema_version, and migrations
  - Separator and markdown image elimination
  - Transactional ingestion and duplicate prevention
  - Crash-safety and transactional rollback on failure
  - Exact substring find and FTS5 BM25 search
  - Ukrainian apostrophes, letters (і/ї/є/ґ), hyphens, quotes, digits, and mixed UA/EN
  - Surround context window
  - Cascade removal and FTS cleanup
  - Online atomic backup, hot restore, and integrity verification
  - Canonical SearchResult contract
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_core import KorzConfig, KorzLibrary, TextProcessor, SourceExtractor, SearchResult


class TestKorzCore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="korz_test_")
        self.temp_path = Path(self.temp_dir)
        self.config = KorzConfig(
            data_dir=self.temp_path / "data",
            db_path=self.temp_path / "data" / "korz_test.sqlite",
            sources_dir=self.temp_path / "data" / "sources",
            backups_dir=self.temp_path / "data" / "backups",
        )
        self.lib = KorzLibrary(self.config)

    def tearDown(self):
        self.lib.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_initialization_and_schema_version(self):
        """Test tables, FTS5 virtual table, schema_version, and WAL pragmas."""
        cur = self.lib.conn.execute("PRAGMA journal_mode;")
        journal = cur.fetchone()[0]
        self.assertEqual(journal.lower(), "wal")

        cur = self.lib.conn.execute("PRAGMA user_version;")
        v = cur.fetchone()[0]
        self.assertEqual(v, 2)

        cur = self.lib.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {r[0] for r in cur.fetchall()}
        expected = {"schema_migrations", "sources", "documents", "pages", "paragraphs", "chunks", "paragraphs_fts", "ingest_history", "embedding_models", "chunk_embeddings", "semantic_state"}
        self.assertTrue(expected.issubset(tables))

        cur = self.lib.conn.execute("SELECT version, description FROM schema_migrations ORDER BY version;")
        migs = cur.fetchall()
        self.assertEqual(len(migs), 2)
        self.assertEqual(migs[0][0], 1)
        self.assertEqual(migs[1][0], 2)

    def test_text_processor_and_separator_elimination(self):
        """Test that separators and markdown images are eliminated from indexing."""
        sample_page = """
{1}------------------------------------------------

# Глава 1: Бабуїни

![Ілюстрація бабуїна](img.jpg)

Бабуїни живуть у великих соціальних групах. Це створює постійний стрес.

---

Самці борються за домінування щодня.
***
"""
        clean = TextProcessor.clean_page_text(sample_page)
        self.assertNotIn("![", clean)
        self.assertNotIn("{1}---", clean)
        self.assertNotIn("***", clean)

        paras = TextProcessor.segment_paragraphs(clean, page_number=1, max_chars=1400)
        self.assertEqual(len(paras), 3)
        self.assertEqual(paras[0].display_text, "# Глава 1: Бабуїни")
        self.assertEqual(paras[1].display_text, "Бабуїни живуть у великих соціальних групах. Це створює постійний стрес.")
        self.assertEqual(paras[2].display_text, "Самці борються за домінування щодня.")

    def test_add_book_and_duplicate_prevention(self):
        """Test transactional ingestion and duplicate protection."""
        book_file = self.temp_path / "test_book.md"
        content = """
{1}------------------------------------------------
Бабуїни живуть у савані і досліджують соціальний стрес.

{2}------------------------------------------------
Роберт Сапольський проводив польові дослідження в Кенії багато років.
"""
        book_file.write_text(content.strip(), encoding="utf-8")

        # 1. First import
        res = self.lib.add_book(book_file, title="Тестова Книга", author="Роберт Сапольський")
        self.assertEqual(res["status"], "READY")
        self.assertEqual(res["pages"], 2)
        self.assertEqual(res["paragraphs"], 2)
        doc_id = res["document_id"]

        # 2. Duplicate import of same file
        dup_res = self.lib.add_book(book_file)
        self.assertEqual(dup_res["status"], "DUPLICATE")
        self.assertEqual(dup_res["document_id"], doc_id)

    def test_transactional_rollback_on_crash(self):
        """Test that a crash midway during add_book rolls back completely, leaving 0 orphan records."""
        book_file = self.temp_path / "crash_book.md"
        book_file.write_text(
            "{1}------------------------------------------------\nЦей текст не повинен залишитися в базі після краху.",
            encoding="utf-8",
        )

        # Inject a crash midway inside the transaction
        # At this point, documents, pages, and paragraphs have already been inserted into the SQLite transaction,
        # but chunks are about to be processed and serialized with json.dumps.
        with patch("korz_core.json.dumps", side_effect=RuntimeError("Simulated Mid-Transaction Ingestion Crash")):
            with self.assertRaises(RuntimeError):
                self.lib.add_book(book_file, title="Крах Книга")

        # Verify database has ZERO documents, pages, paragraphs, or chunks (transaction was rolled back)
        cur_docs = self.lib.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        cur_pages = self.lib.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        cur_paras = self.lib.conn.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0]
        cur_chunks = self.lib.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        cur_fts = self.lib.conn.execute("SELECT COUNT(*) FROM paragraphs_fts").fetchone()[0]

        self.assertEqual(cur_docs, 0)
        self.assertEqual(cur_pages, 0)
        self.assertEqual(cur_paras, 0)
        self.assertEqual(cur_chunks, 0)
        self.assertEqual(cur_fts, 0)

        # Ingest history should record the failure outside the rolled-back transaction
        cur_hist = self.lib.conn.execute("SELECT action, status FROM ingest_history WHERE status = 'FAILED'").fetchall()
        self.assertEqual(len(cur_hist), 1)

    def test_search_punctuation_and_language_robustness(self):
        """
        Verify search handles:
          - Ukrainian apostrophes: ', ’, ʼ, `
          - Ukrainian characters: і, ї, є, ґ
          - Hyphenated words: будь-який, соціально-економічний
          - Quotes: «...», "..."
          - Digits: 1969, 121
          - Mixed UA/EN queries
        """
        book_file = self.temp_path / "test_punct.md"
        book_file.write_text(
            "{1}------------------------------------------------\n"
            "У 1969 році Jacque Fresco описав соціально-економічний устрій та «новий світ» "
            "для будь-якої людини з дивовижною пам'яттю про минуле.\n\n"
            "{2}------------------------------------------------\n"
            "Ґрунтовні дослідження вчених підтверджують зв'язок між їжею, імунітетом та єдністю організму.",
            encoding="utf-8",
        )
        self.lib.add_book(book_file, title="Тест Пунктуації", author="Жак Фреско")

        # 1. Ukrainian apostrophe variations in query
        for q in ["пам'яттю", "пам’яттю", "памʼяттю", "пам`яттю"]:
            res = self.lib.search(q)
            self.assertEqual(len(res), 1, f"Failed for apostrophe query: {q}")
            self.assertIn("людини з дивовижною пам'яттю", res[0].text)

        # 2. Hyphenated word query
        res_hyphen = self.lib.search("соціально-економічний")
        self.assertEqual(len(res_hyphen), 1)

        res_hyphen2 = self.lib.search("будь-якої")
        self.assertEqual(len(res_hyphen2), 1)

        # 3. Quotes (angle brackets and double quotes)
        res_quotes1 = self.lib.search("«новий світ»")
        self.assertEqual(len(res_quotes1), 1)

        res_quotes2 = self.lib.search('"новий світ"')
        self.assertEqual(len(res_quotes2), 1)

        # 4. Digits
        res_digit = self.lib.search("1969")
        self.assertEqual(len(res_digit), 1)

        # 5. Mixed UA / EN
        res_mixed = self.lib.search("Jacque Fresco соціально-економічний")
        self.assertEqual(len(res_mixed), 1)

        # 6. Ukrainian specific letters: ґ, ї, і, є
        res_letters = self.lib.search("ґрунтовні")
        self.assertEqual(len(res_letters), 1)

        res_letters2 = self.lib.search("імунітетом")
        self.assertEqual(len(res_letters2), 1)

    def test_canonical_search_contract(self):
        """Test that SearchResult dataclass matches the canonical contract."""
        book_file = self.temp_path / "test_contract.md"
        book_file.write_text(
            "{1}------------------------------------------------\n"
            "Бабуїни живуть у великих соціальних групах.",
            encoding="utf-8",
        )
        self.lib.add_book(book_file, title="Контрактна Книга", author="Роберт Сапольський")

        results = self.lib.search("бабуїни")
        self.assertTrue(len(results) >= 1)
        r = results[0]

        # Verify all contract fields
        self.assertIsInstance(r.paragraph_id, int)
        self.assertIsInstance(r.document_id, str)
        self.assertEqual(r.title, "Контрактна Книга")
        self.assertEqual(r.author, "Роберт Сапольський")
        self.assertEqual(r.page, 1)
        self.assertEqual(r.paragraph_index, 1)
        self.assertIn("Бабуїни", r.text)
        self.assertIsInstance(r.snippet, str)
        self.assertIsInstance(r.score, float)
        self.assertEqual(r.match_type, "fts")

        # Verify backward-compatibility aliases
        self.assertEqual(r.page_number, r.page)
        self.assertEqual(r.display_text, r.text)

    def test_get_context(self):
        """Test context expansion around target paragraph."""
        book_file = self.temp_path / "test_ctx.md"
        book_file.write_text(
            "{1}------------------------------------------------\n"
            "Перший абзац книги.\n\n"
            "Другий абзац книги: цільовий фрагмент.\n\n"
            "Третій абзац книги.\n\n"
            "Четвертий абзац книги.",
            encoding="utf-8",
        )
        self.lib.add_book(book_file, title="Контекстна Книга")

        hits = self.lib.find("цільовий фрагмент")
        self.assertEqual(len(hits), 1)
        target_id = hits[0].paragraph_id

        ctx = self.lib.get_context(target_id, window=1)
        self.assertIsNotNone(ctx)
        self.assertEqual(len(ctx["entries"]), 3)
        self.assertEqual(ctx["entries"][0]["text"], "Перший абзац книги.")
        self.assertEqual(ctx["entries"][1]["text"], "Другий абзац книги: цільовий фрагмент.")
        self.assertTrue(ctx["entries"][1]["is_target"])
        self.assertEqual(ctx["entries"][2]["text"], "Третій абзац книги.")

    def test_remove_book_and_fts_cleanup(self):
        """Test cascade removal and FTS cleanup."""
        book_file = self.temp_path / "test_del.md"
        book_file.write_text(
            "{1}------------------------------------------------\n"
            "Текст для видалення з бази даних.",
            encoding="utf-8",
        )
        res = self.lib.add_book(book_file, title="Видаляла")
        doc_id = res["document_id"]

        self.assertEqual(len(self.lib.search("видалення")), 1)

        ok = self.lib.remove_book(doc_id)
        self.assertTrue(ok)

        self.assertEqual(len(self.lib.list_books()), 0)
        self.assertEqual(len(self.lib.search("видалення")), 0)
        self.assertEqual(len(self.lib.find("видалення")), 0)

    def test_backup_restore_and_admin_verify(self):
        """Test online backup -> mutation -> hot restore -> admin verify."""
        book_file1 = self.temp_path / "book1.md"
        book_file1.write_text("{1}----\nПерша збережена книга.", encoding="utf-8")
        self.lib.add_book(book_file1, title="Книга 1")

        # 1. Create backup
        backup_path = self.lib.backup()
        self.assertTrue(Path(backup_path).exists())

        # 2. Mutate database: add a second book
        book_file2 = self.temp_path / "book2.md"
        book_file2.write_text("{1}----\nДруга книга перед відновленням.", encoding="utf-8")
        self.lib.add_book(book_file2, title="Книга 2")
        self.assertEqual(len(self.lib.list_books()), 2)

        # 3. Restore from backup
        restore_status = self.lib.restore(backup_path)
        self.assertEqual(restore_status["total_documents"], 1)

        # 4. Verify restored state
        books = self.lib.list_books()
        self.assertEqual(len(books), 1)
        self.assertEqual(books[0]["title"], "Книга 1")

        # 5. Admin verify
        verify_res = self.lib.admin_verify()
        self.assertTrue(verify_res["all_healthy"])
        self.assertTrue(verify_res["parity_ok"])


if __name__ == "__main__":
    unittest.main()
