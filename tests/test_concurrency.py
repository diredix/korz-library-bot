"""
tests/test_concurrency.py - High-Concurrency Stress Test for KORZ LOCAL (SQLite WAL Engine)

Simulates realistic multi-user Telegram read load (50-100 concurrent searches/contexts)
occurring concurrently with database writes (online backups and document ingestion).
Verifies:
  - Zero database deadlocks
  - Zero 'database is locked' errors (WAL + busy_timeout=30000)
  - 100% query success rate
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


class TestConcurrency(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="korz_stress_")
        self.temp_path = Path(self.temp_dir)
        self.config = KorzConfig(
            data_dir=self.temp_path / "data",
            db_path=self.temp_path / "data" / "korz_stress.sqlite",
            sources_dir=self.temp_path / "data" / "sources",
            backups_dir=self.temp_path / "data" / "backups",
        )
        self.lib = KorzLibrary(self.config)

        # Seed database with sample books
        for i in range(1, 4):
            doc_file = self.temp_path / f"seed_book_{i}.md"
            doc_file.write_text(
                f"{{1}}------------------------------------------------\n"
                f"# Книга {i}: Дослідження приматів та суспільства\n\n"
                f"Бабуїни {i} живуть у савані та стикаються з соціальним стресом щодня.\n\n"
                f"{{2}}------------------------------------------------\n"
                f"Ресурсоорієнтована економіка {i} створює суспільство без грошей та бідності.\n\n"
                f"Пам'ять про минуле формує наше сприйняття майбутнього.",
                encoding="utf-8",
            )
            self.lib.add_book(doc_file, title=f"Тестова Книга {i}", author="Тестовий Автор")

    def tearDown(self):
        self.lib.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_concurrent_reads_with_background_writes(self):
        """
        Run 100 concurrent search and context queries across 10 threads
        simultaneously with continuous background backups and document ingestion.
        """
        NUM_QUERIES = 100
        NUM_WORKERS = 10

        query_latencies = []
        errors = []

        # Queries pool
        test_queries = [
            ("соціальний стрес", "fts"),
            ("ресурсоорієнтована", "fts"),
            ("бабуїни", "fts"),
            ("пам'ять", "fts"),
            ("суспільство без грошей", "fts"),
            ("стикаються з соціальним стресом", "exact"),
            ("Дослідження приматів", "exact"),
        ]

        def worker_read(idx: int):
            # Each worker thread uses its own connection to simulate distinct processes/bot instances
            thread_lib = KorzLibrary(self.config)
            try:
                q, mode = test_queries[idx % len(test_queries)]
                t0 = time.perf_counter()
                results = thread_lib.search(q, mode=mode, limit=3)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                # Also test context fetch if hits found
                if results:
                    par_id = results[0].paragraph_id
                    ctx = thread_lib.get_context(par_id, window=2)
                    self.assertIsNotNone(ctx)

                query_latencies.append(elapsed_ms)
                return True
            except Exception as e:
                errors.append(f"Worker {idx} error: {e}")
                return False
            finally:
                thread_lib.close()

        def worker_write_backup():
            write_lib = KorzLibrary(self.config)
            try:
                for _ in range(3):
                    bck = write_lib.backup()
                    self.assertTrue(Path(bck).exists())
                    time.sleep(0.05)
            except Exception as e:
                errors.append(f"Backup writer error: {e}")
            finally:
                write_lib.close()

        def worker_write_add():
            write_lib = KorzLibrary(self.config)
            try:
                for b_idx in range(10, 13):
                    extra_file = self.temp_path / f"extra_{b_idx}.md"
                    extra_file.write_text(f"{{1}}----\nДодатковий текст книги {b_idx}.", encoding="utf-8")
                    res = write_lib.add_book(extra_file, title=f"Екстра {b_idx}")
                    self.assertIn(res["status"], ["READY", "DUPLICATE"])
                    time.sleep(0.05)
            except Exception as e:
                errors.append(f"Add writer error: {e}")
            finally:
                write_lib.close()

        # Launch concurrent execution
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS + 2) as executor:
            # Launch background writer threads
            fut_bck = executor.submit(worker_write_backup)
            fut_add = executor.submit(worker_write_add)

            # Launch 100 concurrent reader queries
            read_futures = [executor.submit(worker_read, i) for i in range(NUM_QUERIES)]

            # Wait for all reads
            concurrent.futures.wait(read_futures)
            concurrent.futures.wait([fut_bck, fut_add])

        total_duration = time.perf_counter() - start_time

        # Assert zero errors
        self.assertEqual(len(errors), 0, f"Concurrency errors encountered: {errors}")
        self.assertEqual(len(query_latencies), NUM_QUERIES)

        avg_latency = sum(query_latencies) / len(query_latencies)
        p95_latency = sorted(query_latencies)[int(len(query_latencies) * 0.95)]

        print(f"\n[STRESS TEST] Results ({NUM_QUERIES} queries, {NUM_WORKERS} workers):")
        print(f"   * Successful queries: {len(query_latencies)} / {NUM_QUERIES} (100%)")
        print(f"   * Concurrency errors: {len(errors)} (0)")
        print(f"   * Total wall time:    {total_duration:.2f} s")
        print(f"   * Mean latency:       {avg_latency:.2f} ms")
        print(f"   * P95 latency:        {p95_latency:.2f} ms")


if __name__ == "__main__":
    unittest.main()
