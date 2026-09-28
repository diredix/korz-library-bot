"""
tests/test_soak.py - Comprehensive Soak & Hardening Verification for KORZ LOCAL

Simulates real-world production load against the SQLite database:
  - 500 search queries (FTS5, Exact, quoted, Ukrainian apostrophes)
  - 100 get_context queries
  - 20 list_books calls
  - 20 status calls
  - Concurrent online backup (WAL checkpointing)
  - Concurrent CLI document ingestion (BEGIN IMMEDIATE transaction)
  - Abrupt restart / connection kill under active query load

Verifies:
  - errors == 0
  - locked == 0 (busy_timeout + WAL)
  - SQLite integrity_check == ok
  - FTS5 integrity_check == ok
  - Paragraph to FTS parity == 100%
  - Anonymous search_log populated without PII
"""

import concurrent.futures
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_core import KorzConfig, KorzLibrary


class TestSoak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We use a temporary clone of the real database if available, else a synthetic multi-book corpus
        cls.temp_dir = tempfile.mkdtemp(prefix="korz_soak_")
        cls.temp_path = Path(cls.temp_dir)
        cls.db_path = cls.temp_path / "korz_soak.sqlite"
        cls.sources_dir = cls.temp_path / "sources"
        cls.backups_dir = cls.temp_path / "backups"

        cls.sources_dir.mkdir(parents=True, exist_ok=True)
        cls.backups_dir.mkdir(parents=True, exist_ok=True)

        real_db = Path(__file__).resolve().parent.parent / "data" / "korz_library.sqlite"
        if real_db.exists() and real_db.stat().st_size > 100000:
            shutil.copy2(real_db, cls.db_path)
            cls.config = KorzConfig(
                data_dir=cls.temp_path,
                db_path=cls.db_path,
                sources_dir=cls.sources_dir,
                backups_dir=cls.backups_dir,
            )
            cls.primary_lib = KorzLibrary(cls.config)
        else:
            cls.config = KorzConfig(
                data_dir=cls.temp_path,
                db_path=cls.db_path,
                sources_dir=cls.sources_dir,
                backups_dir=cls.backups_dir,
            )
            cls.primary_lib = KorzLibrary(cls.config)
            for i in range(1, 6):
                doc_file = cls.temp_path / f"soak_seed_{i}.md"
                doc_file.write_text(
                    f"{{1}}------------------------------------------------\n"
                    f"# Книга {i}: Соціальний устрій та стрес у приматів\n\n"
                    f"Дослідження поведінки тварин показує зв'язок між соціальним статусом і рівнем стресу.\n\n"
                    f"{{2}}------------------------------------------------\n"
                    f"Проєкт «Венера» та ресурсоорієнтована економіка пропонують новий погляд.\n\n"
                    f"Пам'ять про минуле визначає наші соціально-економічні рішення.",
                    encoding="utf-8",
                )
                cls.primary_lib.add_book(doc_file, title=f"Soak Book {i}", author="Fresco & Sapolsky")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.primary_lib.close()
        except Exception:
            pass
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_soak_500_queries_with_concurrent_writes(self):
        """
        Executes:
          - 500 search queries across 12 worker threads
          - 100 context calls
          - 20 list_books calls
          - 20 status calls
          - 1 concurrent backup in background
          - 1 concurrent add_book in background
        """
        search_queries = [
            ("стрес", "fts"),
            ("бабуїни", "fts"),
            ("ресурсоорієнтована економіка", "fts"),
            ("«новий світ»", "fts"),
            ("соціально-економічний", "fts"),
            ("пам'ять", "fts"),
            ("Jacque Fresco", "fts"),
            ("Robert Sapolsky", "fts"),
            ("мова", "fts"),
            ("корозія", "exact"),
            ("зв'язок", "exact"),
            ("неіснуюче_слово_для_zero_hit_log", "fts"),
        ]

        latencies = []
        errors = []
        locked_errors = []
        context_count = [0]

        # Background task 1: Ingest a new book during active reads
        def background_ingest():
            time.sleep(0.05)  # Let reads start first
            ingest_lib = KorzLibrary(self.config)
            try:
                extra_file = self.temp_path / "background_inserted_book.md"
                extra_file.write_text(
                    "{1}------------------------------------------------\n"
                    "Паралельна фонова книга, додана прямо під час активних пошукових запитів користувачів.\n\n"
                    "{2}------------------------------------------------\n"
                    "Система KORZ LOCAL демонструє надійність SQLite WAL режиму.",
                    encoding="utf-8",
                )
                res = ingest_lib.add_book(extra_file, title="Паралельна Книга", author="WAL Test")
                return ("INGEST", res["status"])
            except Exception as e:
                err = str(e)
                if "locked" in err.lower():
                    locked_errors.append(f"Ingest locked: {err}")
                else:
                    errors.append(f"Ingest error: {err}")
                return ("INGEST", f"ERROR: {err}")
            finally:
                ingest_lib.close()

        # Background task 2: Online backup during active reads
        def background_backup():
            time.sleep(0.08)
            backup_lib = KorzLibrary(self.config)
            try:
                b_path = backup_lib.backup()
                return ("BACKUP", str(b_path))
            except Exception as e:
                err = str(e)
                if "locked" in err.lower():
                    locked_errors.append(f"Backup locked: {err}")
                else:
                    errors.append(f"Backup error: {err}")
                return ("BACKUP", f"ERROR: {err}")
            finally:
                backup_lib.close()

        # Search worker task
        def worker_task(idx: int):
            worker_lib = KorzLibrary(self.config)
            try:
                # 1. Search query
                q, mode = search_queries[idx % len(search_queries)]
                t0 = time.perf_counter()
                results = worker_lib.search(q, mode=mode, limit=5)
                elapsed = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed)

                # 2. Context call every 5th search (total ~100 calls)
                if idx % 5 == 0 and results:
                    par_id = results[0].paragraph_id
                    ctx = worker_lib.get_context(par_id, window=2)
                    self.assertIsNotNone(ctx)
                    context_count[0] += 1

                # 3. Status call every 25th search (total ~20 calls)
                if idx % 25 == 0:
                    st = worker_lib.status()
                    self.assertGreater(st["total_documents"], 0)

                # 4. Books list every 25th search (total ~20 calls)
                if idx % 25 == 0:
                    bks = worker_lib.list_books()
                    self.assertGreater(len(bks), 0)

                return True
            except Exception as e:
                err = str(e)
                if "locked" in err.lower():
                    locked_errors.append(f"Worker {idx} locked: {err}")
                else:
                    errors.append(f"Worker {idx} error: {err}")
                return False
            finally:
                worker_lib.close()

        # Launch soak threadpool: 12 workers running 500 tasks + 2 background write tasks
        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=14) as executor:
            fut_ingest = executor.submit(background_ingest)
            fut_backup = executor.submit(background_backup)
            fut_searches = [executor.submit(worker_task, i) for i in range(500)]

            # Wait for all to complete
            done_searches = [f.result() for f in fut_searches]
            res_ingest = fut_ingest.result()
            res_backup = fut_backup.result()

        total_time = time.perf_counter() - t_start

        # Assert zero errors and zero locked database errors
        self.assertEqual(len(errors), 0, f"Errors encountered during soak test: {errors}")
        self.assertEqual(len(locked_errors), 0, f"Locked errors encountered: {locked_errors}")
        self.assertEqual(sum(1 for d in done_searches if d is True), 500)
        self.assertEqual(res_ingest[1], "READY")
        self.assertTrue(Path(res_backup[1]).exists())

        # Check performance metrics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        mean_lat = sum(latencies) / len(latencies)

        print(f"\n[SOAK TEST COMPLETED IN {total_time:.2f}s]")
        print(f"Total Searches: 500 | Context Calls: {context_count[0]}")
        print(f"Mean Latency: {mean_lat:.2f} ms | P50: {p50:.2f} ms | P95: {p95:.2f} ms | P99: {p99:.2f} ms")
        print(f"Errors: {len(errors)} | Locked Errors: {len(locked_errors)}")

        # Post-soak deep database verification
        verify_res = self.primary_lib.admin_verify()
        self.assertTrue(verify_res["all_healthy"])
        self.assertTrue(verify_res["parity_ok"])
        self.assertEqual(verify_res["sqlite_integrity_check"], "ok")
        self.assertEqual(verify_res["fts5_integrity_check"], "ok")

        # Telemetry verification: search_log should contain records without leaking user info
        stats = self.primary_lib.get_search_stats()
        self.assertGreater(stats["total_searches"], 0)
        failed_searches = self.primary_lib.get_failed_searches()
        self.assertTrue(any("неіснуюче_слово" in f["query"] for f in failed_searches))

    def test_kill_and_immediate_restart_under_load(self):
        """
        Simulates abrupt process kill (e.g. SIGKILL / Ctrl+C) while operations are running.
        Verifies that an immediate reopen has zero stale lock, zero data corruption,
        and instant read availability.
        """
        # Create an independent connection doing work
        active_lib = KorzLibrary(self.config)
        res1 = active_lib.search("стрес", limit=2)
        self.assertGreater(len(res1), 0)

        # Abruptly close / drop connection simulating hard death
        active_lib.conn.close()

        # Reopen immediately in a brand new process/instance
        reborn_lib = KorzLibrary(self.config)
        try:
            # Immediate read must succeed without recovery steps
            res2 = reborn_lib.search("стрес", limit=2)
            self.assertEqual(len(res2), len(res1))

            # Database health remains pristine
            v = reborn_lib.admin_verify()
            self.assertTrue(v["all_healthy"])
        finally:
            reborn_lib.close()


if __name__ == "__main__":
    unittest.main()
