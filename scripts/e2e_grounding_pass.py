"""
scripts/e2e_grounding_pass.py
End-to-End Grounding Pass for Phase 4.2:
  Test A — Profile:      «Яким був бабуїн Нік?»
  Test B — Evolution:    «Як змінилася Рут з часом?»
  Test C — Relationship: «Який конфлікт був між Соломоном і Урією?»
  Test D — Safety:       «Чому Нік жорстоко розправився з Рубеном?»
"""
import sys
import os
import sqlite3

# Ensure utf-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))
import korz_entity
from korz_core import KorzLibrary
from korz_answer import AnswerEngine

def main():
    conn = sqlite3.connect("data/korz_library.sqlite")
    korz_entity.seed_all_pilots(conn)
    conn.close()

    lib = KorzLibrary()
    engine = AnswerEngine(lib)

    test_queries = [
        ("Test A — Profile", "Яким був бабуїн Нік?"),
        ("Test B — Evolution", "Як змінилася Рут з часом?"),
        ("Test C — Relationship", "Який конфлікт був між Соломоном і Урією?"),
        ("Test D — Safety", "Чому Нік жорстоко розправився з Рубеном?"),
    ]

    for label, q in test_queries:
        print("=" * 70)
        print(f"[{label}] QUERY: {q}")
        print("=" * 70)

        # Check entity and intent classification
        conn = sqlite3.connect("data/korz_library.sqlite")
        ent = korz_entity.resolve_entity(conn, q)
        top_intent = korz_entity.detect_character_intent(q)
        co = korz_entity.resolve_co_entity(conn, q, ent["id"]) if ent else None
        fine_intent = korz_entity.detect_entity_intent(q, has_co_entity=(co is not None)) if ent else None
        conn.close()

        print(f"  Primary Entity: {ent.get('canonical_name') if ent else 'None'}")
        print(f"  Co-Entity:      {co.get('canonical_name') if co else 'None'}")
        print(f"  Top Intent:     {top_intent}")
        print(f"  Fine Intent:    {fine_intent}")

        # Run through AnswerEngine
        res = engine.answer(q, mode="gemini")
        print(f"\n  Retrieval Mode:  {res.retrieval_mode}")
        print(f"  Entity Intent:   {res.entity_intent}")
        print(f"  Entity Telemetry:{res.entity_telemetry_json}")
        print(f"  Model Tier:      {res.model_tier} ({res.model_id})")
        print(f"  Citations Count: {len(res.sources)}")
        print(f"  Is Grounded:     {res.is_grounded}")
        print(f"\n  ANSWER TEXT:\n  {res.answer_text}\n")
        print("  SOURCES:")
        for s in res.sources:
            print(f"    [{s.citation_index}] {s.author} — {s.title}, p.{s.page} (PID {s.paragraph_id}): {s.text[:90]}...")
        print()

if __name__ == "__main__":
    main()
