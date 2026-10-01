import sys
import os
import sqlite3
sys.path.insert(0, os.path.abspath("."))
from korz_core import KorzLibrary
import korz_entity
from korz_answer import AnswerEngine, AnswerSource

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("data/korz_library.sqlite")
korz_entity.seed_pilot_nick(conn)

lib = KorzLibrary()
engine = AnswerEngine(lib)

query = "Яким був бабуїн Нік?"
entity_info = korz_entity.resolve_entity(conn, query)
intent = korz_entity.detect_character_intent(query)
print(f"Matched Entity: {entity_info}")
print(f"Detected Intent: {intent}")

bundle = korz_entity.build_entity_evidence_bundle(conn, entity_info["id"], query, max_paragraphs=12)
print(f"Bundle paragraphs count: {len(bundle)}")
for b in bundle:
    print(f"  [{b['citation_index']}] PID {b['paragraph_id']} (p.{b['page']} #{b['paragraph_index']}): {b['text'][:80]}...")

sources = [
    AnswerSource(
        citation_index=b["citation_index"],
        paragraph_id=b["paragraph_id"],
        document_id=b["document_id"],
        author=b["author"],
        title=b["title"],
        page=b["page"],
        paragraph_index=b["paragraph_index"],
        text=b["text"],
        score=b["score"],
        text_sha256=b["text_sha256"],
    )
    for b in bundle
]

res = engine._generative_gemini_answer(query, sources, deep=False)
print("\n" + "="*50)
print("=== GEMINI FLASH LOW ANSWER ===")
print("="*50 + "\n")
print(res.answer_text)
print("\n" + "="*50)
print("=== VERIFIED CITATIONS ===")
print("="*50 + "\n")
for s in res.sources:
    print(f"[{s.citation_index}] {s.author} — {s.title}, p.{s.page} (PID {s.paragraph_id})")
