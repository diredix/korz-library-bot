import sys
import os
import sqlite3

sys.path.insert(0, os.path.abspath("."))
from korz_core import KorzLibrary
import korz_entity
from korz_answer import AnswerEngine

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("data/korz_library.sqlite")
korz_entity.seed_all_pilots(conn)
conn.close()

lib = KorzLibrary()
engine = AnswerEngine(lib)

queries = [
    "Яким був бабуїн Соломон?",
    "Якою була бабуїниця Рут?",
    "Ким була Наомі у зграї?",
]

for q in queries:
    print("\n" + "="*60)
    print(f"QUERY: {q}")
    print("="*60)
    res = engine.answer(q)
    print(f"Retrieval Mode: {res.retrieval_mode}")
    print(f"Book count: {res.book_count}, Sources count: {len(res.sources)}")
    print("\n--- ANSWER ---")
    print(res.answer_text)
    print("\n--- SOURCES ---")
    for s in res.sources[:4]:
        print(f"[{s.citation_index}] {s.author} — {s.title}, p.{s.page} (PID {s.paragraph_id})")

lib.close()
