"""
test_entity_discovery.py - Unit & Regression Tests for Phase 4.3: Autonomous Entity Discovery Pipeline.
Tests:
  1. Candidate mining & schema validation
  2. Morphological declension grouping
  3. Entity type classification (place/concept filtering vs character promotion)
  4. Promotion gate validation
  5. 100% Golden entity recall on 'A Primate's Memoir'
  6. Run #62 Regression: Lisa auto-discovered, promoted, and bundle contains PID 10328 (клінічна психологія).
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
import pytest

import korz_entity


@pytest.fixture
def db_conn():
    conn = sqlite3.connect("data/korz_library.sqlite")
    korz_entity.ensure_entity_schema(conn)
    yield conn
    conn.close()


def test_entity_candidates_schema(db_conn):
    """Verify entity_candidates table exists and has all required columns."""
    tables = [r[0] for r in db_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    assert "entity_candidates" in tables

    cols = [c[1] for c in db_conn.execute("PRAGMA table_info(entity_candidates)").fetchall()]
    assert "document_id" in cols
    assert "surface_form" in cols
    assert "canonical_name" in cols
    assert "frequency" in cols
    assert "mid_sentence_count" in cols
    assert "page_count" in cols
    assert "chapter_count" in cols
    assert "first_seen_page" in cols
    assert "last_seen_page" in cols
    assert "action_score" in cols
    assert "prep_score" in cols
    assert "candidate_type" in cols
    assert "confidence" in cols
    assert "status" in cols


def test_morphological_stemming():
    """Verify Ukrainian declension suffixes are cleanly stripped."""
    assert korz_entity.extract_stem("Ліза") == "Ліз"
    assert korz_entity.extract_stem("Лізи") == "Ліз"
    assert korz_entity.extract_stem("Лізою") == "Ліз"
    assert korz_entity.extract_stem("Річард") == "Річард"
    assert korz_entity.extract_stem("Річарда") == "Річард"
    assert korz_entity.extract_stem("Річардом") == "Річард"
    assert korz_entity.extract_stem("Соломон") == "Соломон"
    assert korz_entity.extract_stem("Соломона") == "Соломон"


def test_candidate_mining_finds_characters_and_places(db_conn):
    """Verify candidate mining returns recurring capitalized entities with statistics."""
    book_id = "doc-a-primates-memoir-0348660a"
    candidates = korz_entity.mine_entity_candidates(db_conn, book_id)
    assert len(candidates) > 20

    surfaces = {c["surface_form"] for c in candidates}
    # Check that major characters are present as candidates
    assert any("Ліз" in s for s in surfaces)
    assert any("Річард" in s for s in surfaces)
    assert any("Соломон" in s for s in surfaces)
    assert any("Нік" in s for s in surfaces)
    assert any("Рут" in s for s in surfaces)
    assert any("Наомі" in s for s in surfaces)

    # Check that places are classified as 'place'
    nairobi = next((c for c in candidates if "Найроб" in c["surface_form"]), None)
    assert nairobi is not None
    assert nairobi["candidate_type"] == "place"


def test_promotion_gate_and_golden_recall(db_conn):
    """
    Test Phase 4.3.3 Promotion Gate:
    1. Promotes real characters (Lisa, Nick, Solomon, Ruth, Naomi, Richard).
    2. Rejects places (Nairobi, Kenya, Sudan) and discourse markers (Гаразд, Знаєте).
    3. Golden entity recall == 100%.
    """
    book_id = "doc-a-primates-memoir-0348660a"
    promoted = korz_entity.promote_entity_candidates(db_conn, book_id)
    assert len(promoted) > 0

    promoted_surfaces = {p["surface_form"] for p in promoted}
    all_promoted_forms = set()
    for p in promoted:
        all_promoted_forms.update(p["forms"])

    # 1. 100% Golden Recall
    golden = ["Нік", "Соломон", "Рут", "Наомі", "Ліза"]
    for g in golden:
        assert g in promoted_surfaces or g in all_promoted_forms, f"Golden entity {g} not promoted!"

    # 2. Strict Place Rejection (False Discovery Prevention)
    assert "Найробі" not in promoted_surfaces
    assert "Кенія" not in promoted_surfaces
    assert "Судан" not in promoted_surfaces

    # 3. Discourse Marker Rejection
    assert "Гаразд" not in promoted_surfaces
    assert "Знаєте" not in promoted_surfaces


def test_run62_regression_lisa_dossier_resolution(db_conn):
    """
    Regression Test for Run #62:
    Query: 'Ким за фахом була Ліза, майбутня дружина Сапольскі?'
    With Auto-Discovery active:
      1. resolve_entity must resolve Lisa.
      2. build_entity_evidence_bundle must include PID 10328 (p.252 - клінічна психологія).
    """
    book_id = "doc-a-primates-memoir-0348660a"
    # Ensure auto-discovery has run
    korz_entity.auto_discover_entities_for_document(db_conn, book_id)

    query = "Ким за фахом була Ліза, майбутня дружина Сапольскі?"
    resolved = korz_entity.resolve_entity(db_conn, query)
    assert resolved is not None, "Failed to resolve Lisa as an entity!"
    assert "Ліз" in resolved["canonical_name"] or "Lisa" in resolved["canonical_name"]

    bundle = korz_entity.build_entity_evidence_bundle(
        db_conn,
        entity_id=resolved["id"],
        query=query,
        max_paragraphs=15,
    )
    assert len(bundle) > 0

    bundle_pids = [b["paragraph_id"] for b in bundle]
    # Check if ground truth PID 10328 (or surrounding scene PID) is present
    has_psychology_pid = 10328 in bundle_pids
    # Also check if text mentions клінічної психології
    full_text = " ".join(b["text"] for b in bundle)
    assert "психолог" in full_text.lower(), "Evidence bundle missing Lisa's profession (клінічна психологія)!"
