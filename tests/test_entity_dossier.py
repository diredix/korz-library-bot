import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pytest
from unittest.mock import patch, MagicMock

import korz_entity
from korz_core import KorzLibrary
from korz_answer import AnswerEngine, AnswerConfig, AnswerResult, AnswerSource


@pytest.fixture
def mem_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    korz_entity.ensure_entity_schema(conn)

    # Minimal mock documents and paragraphs table for foreign keys
    conn.execute("""
    CREATE TABLE documents (
        id TEXT PRIMARY KEY,
        title TEXT,
        author TEXT
    );
    """)
    conn.execute("""
    CREATE TABLE paragraphs (
        id INTEGER PRIMARY KEY,
        document_id TEXT,
        page_number INTEGER,
        paragraph_index INTEGER,
        display_text TEXT
    );
    """)
    conn.execute("INSERT INTO documents VALUES ('doc-test', 'Test Book', 'Test Author');")
    conn.commit()
    yield conn
    conn.close()


def test_ensure_entity_schema(mem_db):
    tables = [r[0] for r in mem_db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "entities" in tables
    assert "entity_aliases" in tables
    assert "entity_scenes" in tables
    assert "entity_mentions" in tables


def test_register_entity_and_alias(mem_db):
    eid = korz_entity.register_entity(
        mem_db,
        canonical_name="Nick",
        entity_type="baboon",
        book_id="doc-test",
        aliases=["Нік", "бабуїн Нік"],
    )
    assert eid > 0

    dossier = korz_entity.get_entity_dossier(mem_db, eid)
    assert dossier["canonical_name"] == "Nick"
    assert "Нік" in dossier["aliases"]
    assert "бабуїн Нік" in dossier["aliases"]


def test_resolve_entity_word_boundaries(mem_db):
    korz_entity.register_entity(
        mem_db,
        canonical_name="Nick",
        entity_type="baboon",
        book_id="doc-test",
        aliases=["Нік", "Nick", "той негідник"],
    )

    # True matches
    assert korz_entity.resolve_entity(mem_db, "Яким був Нік?") is not None
    assert korz_entity.resolve_entity(mem_db, "Розповісти про бабуїна на ім'я Nick.") is not None
    assert korz_entity.resolve_entity(mem_db, "Хто такий той негідник?") is not None

    # False matches (substring contamination prevention)
    assert korz_entity.resolve_entity(mem_db, "Ніколи не кажи ніколи") is None
    assert korz_entity.resolve_entity(mem_db, "Президент Ніксон відвідав саміт") is None


def test_detect_character_intent():
    assert korz_entity.detect_character_intent("Яким був бабуїн Нік?") == "character_profiling"
    assert korz_entity.detect_character_intent("Хто такий Нік?") == "character_profiling"
    assert korz_entity.detect_character_intent("Опиши характер та особистість персонажа") == "character_profiling"
    assert korz_entity.detect_character_intent("Чому впала гілка?") == "causal_narrative"
    assert korz_entity.detect_character_intent("Скільки бабуїнів у зграї?") == "pinpoint_fact"


def test_seed_pilot_nick_real_db():
    conn = sqlite3.connect("data/korz_library.sqlite")
    eid = korz_entity.seed_pilot_nick(conn)
    assert eid > 0

    dossier = korz_entity.get_entity_dossier(conn, eid)
    assert dossier["canonical_name"] == "Nick"
    assert len(dossier["scenes"]) >= 7
    assert len(dossier["mentions"]) >= 15
    conn.close()


def test_build_entity_evidence_bundle():
    conn = sqlite3.connect("data/korz_library.sqlite")
    korz_entity.seed_pilot_nick(conn)
    entity_info = korz_entity.resolve_entity(conn, "Яким був бабуїн Нік?")
    assert entity_info is not None

    bundle = korz_entity.build_entity_evidence_bundle(
        conn,
        entity_id=entity_info["id"],
        query="Яким був бабуїн Нік?",
        max_paragraphs=12,
    )
    assert len(bundle) > 0
    pids = [b["paragraph_id"] for b in bundle]
    # Check that core character PIDs are included
    assert 10209 in pids or 10208 in pids
    assert 10211 in pids or 10213 in pids
    conn.close()


def test_answer_engine_routes_entity_dossier(monkeypatch):
    """Verify AnswerEngine seamlessly routes character_profiling queries to entity dossier."""
    conn = sqlite3.connect("data/korz_library.sqlite")
    korz_entity.seed_pilot_nick(conn)
    conn.close()

    lib = KorzLibrary()
    try:
        cfg = AnswerConfig(enabled=True, backend="extractive", retrieval_mode="rrf")
        engine = AnswerEngine(lib, config=cfg)

        # Test extractive character profiling answer
        res = engine.answer("Яким був бабуїн Нік?")
        assert res.retrieval_mode == "entity_dossier"
        assert len(res.sources) > 0
        assert any(s.paragraph_id in (10208, 10209, 10211, 10212, 10213, 10214) for s in res.sources)
    finally:
        lib.close()


def test_seed_all_pilots_and_routing():
    """Verify all 5 pilot entities (Nick, Solomon, Ruth, Naomi, Fresco) seed and route correctly."""
    conn = sqlite3.connect("data/korz_library.sqlite")
    pilots = korz_entity.seed_all_pilots(conn)
    conn.close()

    assert "Nick" in pilots
    assert "Solomon" in pilots
    assert "Ruth" in pilots
    assert "Naomi" in pilots
    assert "Fresco" in pilots

    lib = KorzLibrary()
    try:
        cfg = AnswerConfig(enabled=True, backend="extractive", retrieval_mode="rrf")
        engine = AnswerEngine(lib, config=cfg)

        # Test Solomon
        res_sol = engine.answer("Яким був бабуїн Соломон?")
        assert res_sol.retrieval_mode == "entity_dossier"
        assert any(s.paragraph_id in (9070, 9079, 9081, 9086, 9104) for s in res_sol.sources)

        # Test Ruth
        res_ruth = engine.answer("Якою була бабуїниця Рут?")
        assert res_ruth.retrieval_mode == "entity_dossier"
        assert any(s.paragraph_id in (9088, 9091, 9092, 9093) for s in res_ruth.sources)

        # Test Naomi
        res_naomi = engine.answer("Ким була Наомі у зграї?")
        assert res_naomi.retrieval_mode == "entity_dossier"
        assert any(s.paragraph_id in (9070, 9071, 9083, 9084, 9102) for s in res_naomi.sources)
    finally:
        lib.close()

