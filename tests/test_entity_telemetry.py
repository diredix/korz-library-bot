"""
tests/test_entity_telemetry.py
Tests for Entity Telemetry and Morphology Layer (Step 3).
"""
import json
import sqlite3
import pytest

import korz_entity
from korz_core import KorzLibrary, KorzConfig
from korz_answer import AnswerEngine, AnswerResult, AnswerSource


@pytest.fixture
def library():
    lib = KorzLibrary()
    korz_entity.seed_all_pilots(lib.conn)
    yield lib
    lib.close()


def test_morphology_generator_precision():
    """Verify nominal case declension endings are produced without false substrings."""
    variants_solomon = korz_entity.generate_ukrainian_name_variants("Соломон")
    assert "Соломона" in variants_solomon
    assert "Соломону" in variants_solomon
    assert "Соломоном" in variants_solomon
    assert "Соломоні" in variants_solomon
    assert "Соломонові" in variants_solomon

    variants_nick = korz_entity.generate_ukrainian_name_variants("Нік")
    assert "Ніка" in variants_nick
    assert "Ніком" in variants_nick
    assert "Ніку" in variants_nick
    # Crucial safety check: 'Ніколи' must never be generated
    assert "Ніколи" not in variants_nick
    assert "Ніксон" not in variants_nick


def test_resolve_entity_with_generated_morphology():
    """Verify that inflected names resolve with alias_type='generated_morphology'."""
    conn = sqlite3.connect(":memory:")
    # Initialize basic tables needed for seeding
    conn.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, title TEXT);")
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'A Primate Memoir');")
    conn.execute("CREATE TABLE IF NOT EXISTS paragraphs (id INTEGER PRIMARY KEY, document_id TEXT, page_number INTEGER, paragraph_index INTEGER, display_text TEXT);")
    korz_entity.seed_all_pilots(conn)

    # Nominative (manual alias)
    res_nom = korz_entity.resolve_entity(conn, "Яким був Соломон?")
    assert res_nom is not None
    assert res_nom["canonical_name"] == "Solomon"
    assert res_nom["alias_type"] == "manual"

    # Instrumental (generated morphology)
    res_inst = korz_entity.resolve_entity(conn, "Який конфлікт був між Соломоном і Урією?")
    assert res_inst is not None
    assert res_inst["canonical_name"] == "Solomon"
    assert res_inst["matched_alias"] == "Соломоном"
    assert res_inst["alias_type"] == "generated_morphology"

    conn.close()


def test_record_answer_run_entity_telemetry(library):
    """Verify record_answer_run persists and retrieves entity_intent and entity_telemetry_json."""
    sample_telemetry = {
        "entity_intent": "relationship",
        "primary_entity": "Solomon",
        "co_entities": ["Uriah"],
        "bundle_builder": "build_relationship_bundle",
        "scene_count": 2,
        "pid_count": 6,
        "pids": [9108, 9109, 9110, 9463, 9464, 9465],
        "entity_resolution": {
            "confidence": 0.98,
            "matched_alias": "Соломоном",
            "alias_type": "generated_morphology",
        },
    }
    telemetry_json = json.dumps(sample_telemetry, ensure_ascii=False)

    run_id = library.record_answer_run(
        query="Який конфлікт був між Соломоном і Урією?",
        backend="gemini",
        threshold=0.82,
        insufficient_info=0,
        source_paragraph_ids=[9108, 9463],
        answer_text="Урія кидав виклик старіючому Соломону...",
        citation_verified=1,
        latency_ms=125.4,
        retrieval_mode="entity_dossier",
        entity_intent="relationship",
        entity_telemetry_json=telemetry_json,
    )

    assert run_id > 0
    logged = library.get_answer_run(run_id)
    assert logged is not None
    assert logged["retrieval_mode"] == "entity_dossier"
    assert logged["entity_intent"] == "relationship"
    assert logged["entity_telemetry_json"] == telemetry_json

    parsed = json.loads(logged["entity_telemetry_json"])
    assert parsed["primary_entity"] == "Solomon"
    assert parsed["bundle_builder"] == "build_relationship_bundle"
    assert parsed["entity_resolution"]["matched_alias"] == "Соломоном"


def test_entity_trace_end_to_end():
    """
    Test Entity Trace:
    Verifies that AnswerEngine.answer() resolves an inflected entity,
    routes to specialized builder, builds telemetry bundle provenance,
    and returns full trace info on AnswerResult.
    """
    conn = sqlite3.connect("data/korz_library.sqlite")
    korz_entity.seed_all_pilots(conn)
    conn.close()

    lib = KorzLibrary()
    engine = AnswerEngine(lib)

    # Use extractive mode to avoid calling external agy in unit test
    res = engine.answer("Який конфлікт був між Соломоном і Урією?", mode="extractive")

    assert res.retrieval_mode == "entity_dossier"
    assert res.entity_intent == "relationship"
    assert res.entity_telemetry_json != ""

    telemetry = json.loads(res.entity_telemetry_json)
    assert telemetry["primary_entity"] == "Solomon"
    assert "Урією" in telemetry["co_entities"] or "Uriah" in telemetry["co_entities"] or "Урія" in telemetry["co_entities"]
    assert telemetry["bundle_builder"] == "build_relationship_bundle"
    assert telemetry["entity_resolution"]["matched_alias"] == "Соломоном"
    assert telemetry["entity_resolution"]["alias_type"] == "generated_morphology"
    assert len(telemetry["pids"]) > 0

    # Also verify that logging to DB preserves this trace
    run_id = lib.record_answer_run(
        query=res.question,
        backend=res.mode,
        retrieval_mode=res.retrieval_mode,
        entity_intent=res.entity_intent,
        entity_telemetry_json=res.entity_telemetry_json,
        answer_text=res.answer_text,
    )
    saved = lib.get_answer_run(run_id)
    assert saved["entity_intent"] == "relationship"
    assert json.loads(saved["entity_telemetry_json"])["primary_entity"] == "Solomon"
