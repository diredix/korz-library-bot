"""
korz_entity.py - Entity Dossier & Scene Cluster Knowledge Graph Engine.
Phase 4 Implementation: Deterministic Entity Provenance Index & Query-Specific Evidence Bundling.

Design Invariant:
  «Досьє персонажа не повинно бути новим джерелом істини. Воно має бути індексом і навігатором до першоджерел.»
  Corpus ──> Evidence Extraction (Deterministic) ──> Entity Dossier (PIDs, Scenes, Hashes) ──> Live Verification
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("korz_entity")


def compute_text_sha256(text: str) -> str:
    """Deterministic truncated SHA256 of cleaned text (first 16 hex chars)."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def ensure_entity_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure the 4 core Entity Dossier tables and indices exist in SQLite:
      1. entities
      2. entity_aliases
      3. entity_scenes
      4. entity_mentions
    """
    conn.execute("""
    CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_name TEXT NOT NULL UNIQUE,
        entity_type TEXT NOT NULL,
        book_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (book_id) REFERENCES documents(id) ON DELETE CASCADE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS entity_aliases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id INTEGER NOT NULL,
        alias TEXT NOT NULL,
        language TEXT NOT NULL DEFAULT 'uk',
        alias_type TEXT NOT NULL DEFAULT 'manual',
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
        UNIQUE(entity_id, alias)
    );
    """)

    # Safe column migration for entity_aliases
    try:
        cur_cols = [c[1] for c in conn.execute("PRAGMA table_info(entity_aliases);").fetchall()]
        if "alias_type" not in cur_cols:
            conn.execute("ALTER TABLE entity_aliases ADD COLUMN alias_type TEXT DEFAULT 'manual';")
    except Exception:
        pass

    conn.execute("""
    CREATE TABLE IF NOT EXISTS entity_scenes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id TEXT NOT NULL UNIQUE,
        entity_id INTEGER NOT NULL,
        start_pid INTEGER NOT NULL,
        end_pid INTEGER NOT NULL,
        start_page INTEGER NOT NULL,
        end_page INTEGER NOT NULL,
        scene_type TEXT NOT NULL,
        description TEXT,
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
        FOREIGN KEY (start_pid) REFERENCES paragraphs(id) ON DELETE CASCADE,
        FOREIGN KEY (end_pid) REFERENCES paragraphs(id) ON DELETE CASCADE
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS entity_mentions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity_id INTEGER NOT NULL,
        paragraph_id INTEGER NOT NULL,
        scene_id TEXT,
        mention_type TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
        FOREIGN KEY (paragraph_id) REFERENCES paragraphs(id) ON DELETE CASCADE,
        FOREIGN KEY (scene_id) REFERENCES entity_scenes(scene_id) ON DELETE SET NULL,
        UNIQUE(entity_id, paragraph_id)
    );
    """)

    # Indices for high-performance lookup
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_canonical ON entities(canonical_name);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(alias);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_scenes_entity ON entity_scenes(entity_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_scenes_pids ON entity_scenes(start_pid, end_pid);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_mentions_pid ON entity_mentions(paragraph_id);")

    # Phase 4.3: Entity Candidates Table for Auto-Discovery Pipeline
    conn.execute("""
    CREATE TABLE IF NOT EXISTS entity_candidates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL,
        surface_form TEXT NOT NULL,
        canonical_name TEXT NOT NULL,
        frequency INTEGER NOT NULL,
        mid_sentence_count INTEGER NOT NULL,
        page_count INTEGER NOT NULL,
        chapter_count INTEGER NOT NULL DEFAULT 1,
        first_seen_page INTEGER,
        last_seen_page INTEGER,
        action_score REAL NOT NULL DEFAULT 0.0,
        prep_score REAL NOT NULL DEFAULT 0.0,
        candidate_type TEXT NOT NULL,
        confidence REAL NOT NULL,
        promoted_entity_id INTEGER,
        status TEXT NOT NULL DEFAULT 'candidate',
        created_at TEXT NOT NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        FOREIGN KEY (promoted_entity_id) REFERENCES entities(id) ON DELETE SET NULL,
        UNIQUE(document_id, surface_form)
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_candidates_doc ON entity_candidates(document_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_candidates_status ON entity_candidates(status);")
    conn.commit()


def register_entity(
    conn: sqlite3.Connection,
    canonical_name: str,
    entity_type: str,
    book_id: str,
    aliases: Optional[List[str]] = None,
) -> int:
    """Register or retrieve an entity by canonical name."""
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO entities (canonical_name, entity_type, book_id, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(canonical_name) DO UPDATE SET
            entity_type = excluded.entity_type,
            book_id = excluded.book_id
        RETURNING id;
        """,
        (canonical_name, entity_type, book_id, now_iso),
    )
    entity_id = cur.fetchone()[0]

    if aliases:
        for alias in aliases:
            register_alias(conn, entity_id, alias)
    conn.commit()
    return entity_id


def register_alias(
    conn: sqlite3.Connection,
    entity_id: int,
    alias: str,
    language: str = "uk",
    alias_type: str = "manual",
) -> None:
    """Register an alias for an entity with provenance tracking (manual vs generated_morphology)."""
    conn.execute(
        """
        INSERT OR IGNORE INTO entity_aliases (entity_id, alias, language, alias_type)
        VALUES (?, ?, ?, ?);
        """,
        (entity_id, alias.strip(), language, alias_type),
    )
    conn.commit()


def generate_ukrainian_name_variants(name: str) -> List[str]:
    """
    Generate deterministic grammatical case variants for Ukrainian masculine/feminine names.
    Provides rigorous nominal declension without heuristic prefix/substring matching.
    Strictly generates standard 2nd declension masculine hard-consonant case endings:
      -а (gen/acc), -у (dat/loc), -ом (inst), -і (loc), -е (voc), -ові (dat)
    """
    variants = []
    clean_name = name.strip()
    # Hard consonant endings: Соломон, Нік, Рубен, Аарон
    if re.search(r"[бвгґджзклмнпрстфхцчшщ]$", clean_name, re.IGNORECASE):
        for end in ["а", "у", "ом", "і", "е", "ові"]:
            variants.append(clean_name + end)
    return variants


def register_scene(
    conn: sqlite3.Connection,
    scene_id: str,
    entity_id: int,
    start_pid: int,
    end_pid: int,
    start_page: int,
    end_page: int,
    scene_type: str,
    description: Optional[str] = None,
) -> None:
    """Register a contiguous scene cluster."""
    conn.execute(
        """
        INSERT INTO entity_scenes (scene_id, entity_id, start_pid, end_pid, start_page, end_page, scene_type, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scene_id) DO UPDATE SET
            entity_id = excluded.entity_id,
            start_pid = excluded.start_pid,
            end_pid = excluded.end_pid,
            start_page = excluded.start_page,
            end_page = excluded.end_page,
            scene_type = excluded.scene_type,
            description = excluded.description;
        """,
        (scene_id, entity_id, start_pid, end_pid, start_page, end_page, scene_type, description),
    )
    conn.commit()


def register_mention(
    conn: sqlite3.Connection,
    entity_id: int,
    paragraph_id: int,
    scene_id: Optional[str] = None,
    mention_type: str = "explicit_name",
    confidence: float = 1.0,
) -> None:
    """Register a paragraph mention linking directly to primary corpus PID."""
    conn.execute(
        """
        INSERT INTO entity_mentions (entity_id, paragraph_id, scene_id, mention_type, confidence)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(entity_id, paragraph_id) DO UPDATE SET
            scene_id = excluded.scene_id,
            mention_type = excluded.mention_type,
            confidence = excluded.confidence;
        """,
        (entity_id, paragraph_id, scene_id, mention_type, confidence),
    )
    conn.commit()


def resolve_entity(conn: sqlite3.Connection, text: str) -> Optional[Dict[str, Any]]:
    """
    Resolve an entity from user query text using word-boundary matching.
    Prevents false substrings (e.g. 'Ніколи' != 'Нік', 'Ніксон' != 'Нік').
    Tracks provenance via alias_type ('manual' vs 'generated_morphology').
    """
    cur = conn.execute("""
        SELECT e.id, e.canonical_name, e.entity_type, e.book_id, a.alias, COALESCE(a.alias_type, 'manual')
        FROM entities e
        JOIN entity_aliases a ON a.entity_id = e.id
        ORDER BY LENGTH(a.alias) DESC
    """)
    rows = cur.fetchall()

    normalized_text = f" {text.lower()} "
    for eid, cname, etype, bid, alias, atype in rows:
        alias_clean = alias.strip().lower()
        # Word boundary pattern for Cyrillic and Latin
        pattern = rf"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ]){re.escape(alias_clean)}(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])"
        if re.search(pattern, normalized_text, re.IGNORECASE):
            return {
                "id": eid,
                "canonical_name": cname,
                "entity_type": etype,
                "book_id": bid,
                "matched_alias": alias,
                "alias_type": atype,
            }
    return None


def detect_character_intent(query: str) -> str:
    """
    Classify user query intent into cognitive categories (top-level routing):
      - 'character_profiling': any entity-centric query (profile, evolution, relationship, etc.)
      - 'causal_narrative': 'чому', 'як сталося'
      - 'pinpoint_fact': factual point query
    """
    q_lower = query.lower()
    profile_patterns = [
        r"\bяким\s+був\b",
        r"\bяка\s+була\b",
        r"\bякою\s+була\b",
        r"\bяке\s+було\b",
        r"\bким\s+був\b",
        r"\bким\s+була\b",
        r"\bким\b.*?\bбул[а-я]*\b",
        r"\bфах[а-я]*\b",
        r"\bпрофесі[а-я]*\b",
        r"\bспеціальност[а-я]*\b",
        r"\bким\s+працював[а-я]*\b",
        r"\bхто\s+такий\b",
        r"\bхто\s+така\b",
        r"\bщо\s+за\s+персонаж\b",
        r"\bопиши\s+персонажа\b",
        r"\bхарактер\b",
        r"\bособистість\b",
        r"\bяк\s+поводився\b",
        r"\bяк\s+поводилася\b",
        r"\bрозкажи\s+про\b",
        r"\bпортрет\b",
        r"\bщо\s+відомо\s+про\b",
        r"\bщо\s+думав\b",
        r"\bякі\s+ідеї\b",
        r"\bпогляди\b",
        # Evolution / timeline triggers
        r"\bяк\s+змінився\b",
        r"\bяк\s+змінилася\b",
        r"\bяк\s+змінювал\b",
        r"\bяк\s+склалася\s+доля\b",
        r"\bщо\s+сталося\s+з\b",
        r"\bнаприкінці\b",
        r"\bз\s+роками\b",
        r"\bз\s+часом\b",
        r"\bпротягом\b",
        r"\bхронолог\b",
        # Relationship triggers
        r"\bміж\b.*\bі\b",
        r"\bконфлікт\b",
        r"\bпротистояння\b",
        r"\bворог\b",
        r"\bінцидент\b",
        r"\bсутичк\b",
        # Affiliation triggers
        r"\bстосунк\b",
        r"\bродин\b",
        r"\bхто\s+належав\b",
        r"\bхто\s+входив\b",
        r"\bколо\b",
        r"\bдіти\b",
    ]
    for p in profile_patterns:
        if re.search(p, q_lower):
            return "character_profiling"

    if re.search(r"\bчому\b|\bз\s+якої\s+причини\b|\bяк\s+так\s+сталося\b", q_lower):
        return "causal_narrative"

    return "pinpoint_fact"


def detect_entity_intent(query: str, has_co_entity: bool = False) -> str:
    """
    Fine-grained Entity Intent Classification (Phase 4.2).

    Five entity-centric query modes:
      - 'profile':       Static portrait — personality, behavior, defining traits
      - 'evolution':     Dynamic trajectory — how the entity changed over time
      - 'relationship':  Cross-entity dynamics — conflict, alliance, interaction between two entities
      - 'affiliation':   Social/group membership — family, troop, hierarchy, clan
      - 'timeline':      Chronological arc — full life story from first to last appearance

    Args:
        query: user query text
        has_co_entity: True if a second entity was detected in the query
    """
    q_lower = query.lower()

    # ── 1. Relationship: two entities interacting ──
    # Must check first because it depends on co-entity detection
    if has_co_entity:
        rel_patterns = [
            r"\bміж\b", r"\bконфлікт", r"\bпротистоянн", r"\bворог",
            r"\bінцидент", r"\bсутичк", r"\bбійк", r"\bпоєдин",
            r"\bстосунк", r"\bсоюз", r"\bпроти\b", r"\bнапад",
            r"\bображ", r"\bатак", r"\bі\b",
        ]
        if any(re.search(p, q_lower) for p in rel_patterns):
            return "relationship"

    # ── 2. Evolution: dynamic trajectory over time ──
    evo_patterns = [
        r"\bяк\s+змінивс", r"\bяк\s+змінилас", r"\bяк\s+змінювал",
        r"\bяк\s+склалас.+\bдол", r"\bщо\s+сталос",
        r"\bнаприкінці\b", r"\bпід\s+кінець\b",
        r"\bпішов\s+з\s+життя", r"\bпішла\s+з\s+життя",
        r"\bпомер", r"\bзагинув", r"\bзагинул",
        r"\bвтратив\s+влад", r"\bвтратила",
        r"\bз\s+роками\b", r"\bз\s+часом\b",
    ]
    if any(re.search(p, q_lower) for p in evo_patterns):
        return "evolution"

    # ── 3. Timeline: chronological sweep ──
    timeline_patterns = [
        r"\bпротягом\b", r"\bхронолог", r"\bвід\s+початку\b",
        r"\bвід\s+першої\b", r"\bдо\s+останньої\b",
        r"\bісторія\s+життя", r"\bжиттєвий\s+шлях",
        r"\bвсе\s+про\b", r"\bповна\s+історія",
    ]
    if any(re.search(p, q_lower) for p in timeline_patterns):
        return "timeline"

    # ── 4. Affiliation: social/group membership ──
    affil_patterns = [
        r"\bродин", r"\bхто\s+належав", r"\bхто\s+входив",
        r"\bколо\b", r"\bдіти\b", r"\bсім'", r"\bклан",
        r"\bдинаст", r"\bматріарх", r"\bпатріарх",
        r"\bматеринськ", r"\bбатьківськ",
        r"\bстосунк",
    ]
    if any(re.search(p, q_lower) for p in affil_patterns):
        return "affiliation"

    # ── 5. Profile: default for entity-centric queries ──
    return "profile"


def get_entity_dossier(conn: sqlite3.Connection, entity_id: int) -> Dict[str, Any]:
    """Retrieve full deterministic structural dossier for an entity."""
    entity_row = conn.execute(
        "SELECT id, canonical_name, entity_type, book_id, created_at FROM entities WHERE id = ?",
        (entity_id,),
    ).fetchone()
    if not entity_row:
        return {}

    aliases = [
        r[0]
        for r in conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias",
            (entity_id,),
        ).fetchall()
    ]

    scenes_rows = conn.execute(
        """
        SELECT scene_id, start_pid, end_pid, start_page, end_page, scene_type, description
        FROM entity_scenes
        WHERE entity_id = ?
        ORDER BY start_pid
        """,
        (entity_id,),
    ).fetchall()

    scenes = []
    for r in scenes_rows:
        scenes.append({
            "scene_id": r[0],
            "start_pid": r[1],
            "end_pid": r[2],
            "start_page": r[3],
            "end_page": r[4],
            "scene_type": r[5],
            "description": r[6],
        })

    mentions_rows = conn.execute(
        """
        SELECT paragraph_id, scene_id, mention_type, confidence
        FROM entity_mentions
        WHERE entity_id = ?
        ORDER BY paragraph_id
        """,
        (entity_id,),
    ).fetchall()

    mentions = [
        {
            "paragraph_id": r[0],
            "scene_id": r[1],
            "mention_type": r[2],
            "confidence": r[3],
        }
        for r in mentions_rows
    ]

    return {
        "id": entity_row[0],
        "canonical_name": entity_row[1],
        "entity_type": entity_row[2],
        "book_id": entity_row[3],
        "created_at": entity_row[4],
        "aliases": aliases,
        "scenes": scenes,
        "mentions": mentions,
    }


def detect_query_aspect(query: str) -> str:
    """
    DEPRECATED in Phase 4.2 — use detect_entity_intent() instead.
    Kept for backward compatibility with existing test suite.
    """
    q_lower = query.lower()
    if any(k in q_lower for k in ["як змінився", "як змінилося", "як змінювалася", "що сталося", "наприкінці", "доля", "з роками", "з часом"]):
        return "trajectory"
    if any(k in q_lower for k in ["конфлікт", "бійк", "протистояння", "ворог", "інцидент", "поєдинок", "сутичк", "переміг"]):
        return "conflict"
    if any(k in q_lower for k in ["стосунк", "родин", "союзник", "партнер", "хто належав", "діти", "сім'"]):
        return "relational"
    return "personality"


def find_co_mentioned_terms(conn: sqlite3.Connection, query: str, exclude_entity_id: int) -> List[str]:
    """Find other character names or terms mentioned in the query."""
    cur = conn.execute("SELECT alias FROM entity_aliases WHERE entity_id != ?", (exclude_entity_id,))
    aliases = [r[0] for r in cur.fetchall()]
    extra_names = [
        "Урія", "Урії", "Урією",
        "Девора", "Девори", "Деворою",
        "Рубен", "Рубена", "Рубеном",
        "Лея", "Леї", "Леєю",
        "Аарон", "Джошуа", "Гідеон", "Нат", "Рейчел", "Сара", "Джоб"
    ]
    aliases.extend(extra_names)

    q_lower = query.lower()
    matched = []
    for a in aliases:
        if len(a) > 2 and re.search(rf"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ]){re.escape(a.lower())}(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])", q_lower):
            matched.append(a)
    return list(set(matched))


def resolve_co_entity(conn: sqlite3.Connection, query: str, primary_entity_id: int) -> Optional[Dict[str, Any]]:
    """Resolve a second entity from the query (excluding the primary one and its aliases)."""
    primary_aliases = set(r[0].lower() for r in conn.execute(
        "SELECT alias FROM entity_aliases WHERE entity_id = ?", (primary_entity_id,)
    ).fetchall())
    primary_cname = conn.execute("SELECT canonical_name FROM entities WHERE id = ?", (primary_entity_id,)).fetchone()
    if primary_cname:
        primary_aliases.add(primary_cname[0].lower())

    cur = conn.execute("""
        SELECT e.id, e.canonical_name, e.entity_type, e.book_id, a.alias
        FROM entities e
        JOIN entity_aliases a ON a.entity_id = e.id
        WHERE e.id != ?
        ORDER BY LENGTH(a.alias) DESC
    """, (primary_entity_id,))
    rows = cur.fetchall()
    normalized_text = f" {query.lower()} "
    for eid, cname, etype, bid, alias in rows:
        alias_clean = alias.strip().lower()
        if alias_clean in primary_aliases or cname.lower() in primary_aliases:
            continue
        pattern = rf"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ]){re.escape(alias_clean)}(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])"
        if re.search(pattern, normalized_text, re.IGNORECASE):
            return {"id": eid, "canonical_name": cname, "entity_type": etype, "book_id": bid}
    return None


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.2: Specialized Evidence Bundle Builders
# ═══════════════════════════════════════════════════════════════════

def _fetch_paragraphs_by_pids(conn: sqlite3.Connection, pids: List[int]) -> List[Dict[str, Any]]:
    """Fetch authentic paragraph texts from corpus by PID list. Shared helper."""
    if not pids:
        return []
    placeholders = ",".join(["?"] * len(pids))
    sql = f"""
    SELECT p.id, p.document_id, p.page_number, p.paragraph_index, p.display_text,
           d.title, d.author
    FROM paragraphs p
    JOIN documents d ON d.id = p.document_id
    WHERE p.id IN ({placeholders})
    ORDER BY p.id
    """
    rows = conn.execute(sql, pids).fetchall()
    scene_map = {}
    try:
        sc_rows = conn.execute(
            f"SELECT paragraph_id, scene_id FROM entity_mentions WHERE paragraph_id IN ({placeholders}) AND scene_id IS NOT NULL", pids
        ).fetchall()
        for pid_val, sc_id in sc_rows:
            scene_map[pid_val] = sc_id
    except Exception:
        pass

    bundle_items: List[Dict[str, Any]] = []
    for idx, r in enumerate(rows, 1):
        clean_text = (r[4] or "").strip()
        bundle_items.append({
            "citation_index": idx,
            "paragraph_id": r[0],
            "document_id": r[1],
            "page": r[2],
            "paragraph_index": r[3],
            "text": clean_text,
            "title": r[5],
            "author": r[6],
            "text_sha256": compute_text_sha256(clean_text),
            "score": 1.0,
            "scene_id": scene_map.get(r[0], ""),
        })
    return bundle_items


def _gather_pids_from_scenes(scenes: List[Dict[str, Any]], max_paragraphs: int, max_per_scene: Optional[int] = None) -> List[int]:
    """Gather contiguous PID runs from an ordered list of scenes, up to max_paragraphs."""
    gathered: List[int] = []
    limit_per_scene = max_per_scene or max_paragraphs
    for sc in scenes:
        pids = list(range(sc["start_pid"], sc["end_pid"] + 1))[:limit_per_scene]
        for pid in pids:
            if pid not in gathered:
                gathered.append(pid)
            if len(gathered) >= max_paragraphs:
                return gathered
    return gathered


def _score_scene_co_occurrence(conn: sqlite3.Connection, sc: Dict[str, Any], co_mentions: List[str]) -> float:
    """Score a scene by co-occurrence of mentioned entity names in its text."""
    if not co_mentions:
        return 0.0
    scene_text_row = conn.execute(
        "SELECT GROUP_CONCAT(display_text, ' ') FROM paragraphs WHERE id BETWEEN ? AND ?",
        (sc["start_pid"], sc["end_pid"]),
    ).fetchone()
    scene_text = (scene_text_row[0] or "").lower()
    score = 0.0
    for term in co_mentions:
        if term.lower() in scene_text:
            score += 50.0
    return score


def build_profile_bundle(
    conn: sqlite3.Connection, entity_id: int, query: str, max_paragraphs: int = 12,
) -> List[Dict[str, Any]]:
    """
    Profile Bundle: Static portrait — personality, behavior, defining traits.
    Prioritizes: personality_profile > dominance_bluff_resolution > alpha_reign > other.
    Includes query focus keyword boost (career, profession, relationships, traits) and
    samples across top scenes to prevent a single long scene from starving other key aspects.
    """
    dossier = get_entity_dossier(conn, entity_id)
    if not dossier or not dossier.get("scenes"):
        return []

    focus_keywords = ["фах", "професі", "робот", "дисертац", "психолог", "освіт", "кар'єр", "дружин"]
    q_lower = query.lower()

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for sc in dossier["scenes"]:
        score = 0.0
        stype = sc["scene_type"]
        if stype == "personality_profile":
            score += 30.0
        elif stype in ("dominance_bluff_resolution", "confrontation"):
            score += 20.0  # defining behavioral events
        elif stype == "alpha_reign":
            score += 15.0
        elif stype == "social_interaction":
            score += 10.0

        # Query focus keyword boost
        scene_text_row = conn.execute(
            "SELECT GROUP_CONCAT(display_text, ' ') FROM paragraphs WHERE id BETWEEN ? AND ?",
            (sc["start_pid"], sc["end_pid"]),
        ).fetchone()
        scene_text = (scene_text_row[0] or "").lower()
        for fk in focus_keywords:
            if fk in scene_text and (fk in q_lower or any(term in q_lower for term in ["фах", "професі", "ким", "робот"])):
                score += 25.0

        scored.append((score, sc))

    ordered = [sc for _, sc in sorted(scored, key=lambda x: x[0], reverse=True)]
    limit_per_scene = max(3, max_paragraphs // 3)
    pids = _gather_pids_from_scenes(ordered, max_paragraphs, max_per_scene=limit_per_scene)
    return _fetch_paragraphs_by_pids(conn, pids)


def build_evolution_bundle(
    conn: sqlite3.Connection, entity_id: int, query: str, max_paragraphs: int = 12,
) -> List[Dict[str, Any]]:
    """
    Evolution Bundle: Dynamic trajectory — how the entity changed over time.
    Guarantees representation across EARLY (intro), MIDDLE (peak/turning point), LATE (demise/exit).
    """
    dossier = get_entity_dossier(conn, entity_id)
    if not dossier or not dossier.get("scenes"):
        return []

    scenes = dossier["scenes"]  # already ordered by start_pid
    n = len(scenes)

    # Split into thirds: early / middle / late
    early_end = max(1, n // 3)
    late_start = n - max(1, n // 3)

    early = scenes[:early_end]
    middle = scenes[early_end:late_start] if early_end < late_start else []
    late = scenes[late_start:]

    # Prioritize demise/transition scenes within late
    def _evo_score(sc: Dict[str, Any], phase: str) -> float:
        stype = sc["scene_type"]
        base = 0.0
        if phase == "late":
            if stype == "historical_demise":
                base += 40.0
            elif stype in ("confrontation", "alpha_reign"):
                base += 20.0
            else:
                base += 10.0
        elif phase == "early":
            if stype == "personality_profile":
                base += 30.0
            else:
                base += 15.0
        else:  # middle
            if stype in ("confrontation", "dominance_event", "dominance_bluff_resolution"):
                base += 25.0
            elif stype == "alpha_reign":
                base += 20.0
            else:
                base += 10.0
        return base

    # Budget: allocate ~1/3 of max_paragraphs to each phase
    budget_per_phase = max(2, max_paragraphs // 3)
    remainder = max_paragraphs - budget_per_phase * 3

    all_pids: List[int] = []

    for phase_scenes, phase_name in [(early, "early"), (middle, "middle"), (late, "late")]:
        scored = sorted(phase_scenes, key=lambda s: _evo_score(s, phase_name), reverse=True)
        phase_budget = budget_per_phase + (remainder if phase_name == "late" else 0)
        pids = _gather_pids_from_scenes(scored, phase_budget)
        for pid in pids:
            if pid not in all_pids:
                all_pids.append(pid)

    return _fetch_paragraphs_by_pids(conn, all_pids[:max_paragraphs])


def build_relationship_bundle(
    conn: sqlite3.Connection, entity_id: int, query: str, max_paragraphs: int = 12,
    co_entity_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Relationship Bundle: Cross-entity dynamics — scenes where two entities co-occur.
    Maximizes co-occurrence score, then falls back to individual entity scenes.
    """
    dossier = get_entity_dossier(conn, entity_id)
    if not dossier or not dossier.get("scenes"):
        return []

    co_mentions = find_co_mentioned_terms(conn, query, entity_id)
    # If we resolved a co-entity, add its aliases to co_mentions
    if co_entity_id:
        co_aliases = [r[0] for r in conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?", (co_entity_id,)
        ).fetchall()]
        co_mentions = list(set(co_mentions + co_aliases))

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for sc in dossier["scenes"]:
        score = _score_scene_co_occurrence(conn, sc, co_mentions)
        # Boost confrontation/interaction scenes
        stype = sc["scene_type"]
        if stype in ("confrontation", "dominance_event", "dominance_bluff_resolution"):
            score += 15.0
        elif stype == "social_interaction":
            score += 10.0
        scored.append((score, sc))

    ordered = [sc for _, sc in sorted(scored, key=lambda x: x[0], reverse=True)]
    pids = _gather_pids_from_scenes(ordered, max_paragraphs)
    return _fetch_paragraphs_by_pids(conn, pids)


def build_affiliation_bundle(
    conn: sqlite3.Connection, entity_id: int, query: str, max_paragraphs: int = 12,
) -> List[Dict[str, Any]]:
    """
    Affiliation Bundle: Social/group membership — family, troop, hierarchy, clan.
    Prioritizes social_interaction > alpha_reign > other, with co-occurrence boost.
    """
    dossier = get_entity_dossier(conn, entity_id)
    if not dossier or not dossier.get("scenes"):
        return []

    co_mentions = find_co_mentioned_terms(conn, query, entity_id)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for sc in dossier["scenes"]:
        score = _score_scene_co_occurrence(conn, sc, co_mentions)
        stype = sc["scene_type"]
        if stype == "social_interaction":
            score += 30.0
        elif stype == "alpha_reign":
            score += 20.0
        elif stype == "personality_profile":
            score += 10.0
        scored.append((score, sc))

    ordered = [sc for _, sc in sorted(scored, key=lambda x: x[0], reverse=True)]
    pids = _gather_pids_from_scenes(ordered, max_paragraphs)
    return _fetch_paragraphs_by_pids(conn, pids)


def build_timeline_bundle(
    conn: sqlite3.Connection, entity_id: int, query: str, max_paragraphs: int = 12,
) -> List[Dict[str, Any]]:
    """
    Timeline Bundle: Chronological arc — even distribution from first to last appearance.
    Samples scenes evenly across the full PID range to reconstruct the entity's life story.
    """
    dossier = get_entity_dossier(conn, entity_id)
    if not dossier or not dossier.get("scenes"):
        return []

    scenes = dossier["scenes"]  # ordered by start_pid
    n = len(scenes)

    if n <= 4:
        # Few scenes: take all, ordered chronologically
        pids = _gather_pids_from_scenes(scenes, max_paragraphs)
        return _fetch_paragraphs_by_pids(conn, pids)

    # Even sampling: pick scenes at regular intervals
    budget_per_scene = max(1, max_paragraphs // min(n, 6))
    step = max(1, n // 6)  # aim for ~6 sample points
    sampled_indices = list(range(0, n, step))
    # Always include first and last
    if 0 not in sampled_indices:
        sampled_indices.insert(0, 0)
    if (n - 1) not in sampled_indices:
        sampled_indices.append(n - 1)

    all_pids: List[int] = []
    for idx in sampled_indices:
        sc = scenes[idx]
        pids = list(range(sc["start_pid"], sc["end_pid"] + 1))[:budget_per_scene]
        for pid in pids:
            if pid not in all_pids:
                all_pids.append(pid)
            if len(all_pids) >= max_paragraphs:
                break
        if len(all_pids) >= max_paragraphs:
            break

    return _fetch_paragraphs_by_pids(conn, all_pids[:max_paragraphs])


def build_entity_evidence_bundle(
    conn: sqlite3.Connection,
    entity_id: int,
    query: str,
    max_paragraphs: int = 12,
) -> List[Dict[str, Any]]:
    """
    Phase 4.2 Dispatcher: Routes to specialized evidence bundle builder
    based on fine-grained Entity Intent Classification.

    Intent → Builder mapping:
      profile       → build_profile_bundle
      evolution     → build_evolution_bundle
      relationship  → build_relationship_bundle
      affiliation   → build_affiliation_bundle
      timeline      → build_timeline_bundle
    """
    co_mentions = find_co_mentioned_terms(conn, query, entity_id)
    co_entity = resolve_co_entity(conn, query, entity_id) if co_mentions else None
    has_co_entity = (co_entity is not None) or bool(co_mentions)

    entity_intent = detect_entity_intent(query, has_co_entity=has_co_entity)
    logger.info(f"Entity intent for '{query[:50]}': {entity_intent} (co_entity={co_entity})")

    if entity_intent == "profile":
        return build_profile_bundle(conn, entity_id, query, max_paragraphs)
    elif entity_intent == "evolution":
        return build_evolution_bundle(conn, entity_id, query, max_paragraphs)
    elif entity_intent == "relationship":
        co_eid = co_entity["id"] if co_entity else None
        return build_relationship_bundle(conn, entity_id, query, max_paragraphs, co_entity_id=co_eid)
    elif entity_intent == "affiliation":
        return build_affiliation_bundle(conn, entity_id, query, max_paragraphs)
    elif entity_intent == "timeline":
        return build_timeline_bundle(conn, entity_id, query, max_paragraphs)
    else:
        # Fallback to profile
        return build_profile_bundle(conn, entity_id, query, max_paragraphs)


def seed_pilot_nick(conn: sqlite3.Connection) -> int:
    """
    Seed Golden Pilot Entity: Baboon Nick from Sapolsky's 'A Primate's Memoir'.
    Maps out all verified scenes, mentions, and aliases.
    """
    ensure_entity_schema(conn)

    # Document ID for A Primate's Memoir
    doc_row = conn.execute(
        "SELECT id FROM documents WHERE title LIKE '%Primate%' LIMIT 1"
    ).fetchone()
    book_id = doc_row[0] if doc_row else "doc-a-primates-memoir-0348660a"

    entity_id = register_entity(
        conn,
        canonical_name="Nick",
        entity_type="baboon",
        book_id=book_id,
        aliases=[
            "Нік",
            "Nick",
            "бабуїн Нік",
            "той негідник",
        ],
    )
    for v in generate_ukrainian_name_variants("Нік"):
        register_alias(conn, entity_id, v, language="uk", alias_type="generated_morphology")

    # Register Scene Clusters
    scenes = [
        (
            "nick-s1-personality",
            10208, 10214, 228, 229,
            "personality_profile",
            "Введення Ніка, тип А особистості, найзліший бабуїн, бійка з Рубеном і розривання заду, інцидент з Рут на дереві",
        ),
        (
            "nick-s2-reign-terror",
            10215, 10218, 229, 230,
            "dominance_event",
            "Правління терору, протистояння з Небуханецаром і Гідеоном, спроби створення коаліцій",
        ),
        (
            "nick-s3-absolom-crash",
            10221, 10221, 231, 231,
            "social_interaction",
            "Вилизування з Деворою в затишному місці, падіння Абсолома з гілки",
        ),
        (
            "nick-s4-troop-reign",
            10234, 10234, 234, 234,
            "alpha_reign",
            "Зміни в складі зграї підлітків за часів правління Ніка, перехід струмків на двох ногах",
        ),
        (
            "nick-s5-ruben-blowdart-bluff",
            10237, 10242, 235, 236,
            "dominance_bluff_resolution",
            "Постріл дротиком у Рубена, кидок Ніка на непритомного суперника, перемога-показуха, крик ва-ху, приписування заслуги собі",
        ),
        (
            "nick-s6-reign-era",
            10245, 10245, 238, 238,
            "alpha_reign",
            "Повернення Сапольскі до масаїв за часів правління Ніка як альфи",
        ),
        (
            "nick-s7-school-visit",
            10379, 10379, 264, 264,
            "camp_encounter",
            "Екскурсія школярів до табору для спостереження за знерухомленим Ніком",
        ),
        (
            "nick-s8-demise",
            10557, 10557, 298, 298,
            "historical_demise",
            "Ретроспектива зграї: Нік, Рут, Ісаак та інші пішли з життя, вижив лише Джошуа",
        ),
    ]

    for sid, spid, epid, spg, epg, stype, desc in scenes:
        register_scene(conn, sid, entity_id, spid, epid, spg, epg, stype, desc)

    # Register Deterministic Mentions
    mentions = [
        (10208, "nick-s1-personality", "section_header", 1.0),
        (10209, "nick-s1-personality", "explicit_name", 1.0),
        (10211, "nick-s1-personality", "explicit_name", 1.0),
        (10213, "nick-s1-personality", "explicit_name", 1.0),
        (10214, "nick-s1-personality", "explicit_name", 1.0),
        (10215, "nick-s2-reign-terror", "explicit_name", 1.0),
        (10216, "nick-s2-reign-terror", "explicit_name", 1.0),
        (10217, "nick-s2-reign-terror", "explicit_name", 1.0),
        (10218, "nick-s2-reign-terror", "explicit_name", 1.0),
        (10221, "nick-s3-absolom-crash", "explicit_name", 1.0),
        (10234, "nick-s4-troop-reign", "explicit_name", 1.0),
        (10237, "nick-s5-ruben-blowdart-bluff", "explicit_name", 1.0),
        (10238, "nick-s5-ruben-blowdart-bluff", "explicit_name", 1.0),
        (10239, "nick-s5-ruben-blowdart-bluff", "explicit_name", 1.0),
        (10240, "nick-s5-ruben-blowdart-bluff", "explicit_name", 1.0),
        (10241, "nick-s5-ruben-blowdart-bluff", "explicit_name", 1.0),
        (10242, "nick-s5-ruben-blowdart-bluff", "pronoun_reference", 1.0),
        (10245, "nick-s6-reign-era", "explicit_name", 1.0),
        (10379, "nick-s7-school-visit", "explicit_name", 1.0),
        (10557, "nick-s8-demise", "explicit_name", 1.0),
    ]

    for pid, sid, mtype, conf in mentions:
        register_mention(conn, entity_id, pid, sid, mtype, conf)

    conn.commit()
    return entity_id


def infer_scene_type(texts: List[str]) -> str:
    """Infer semantic scene type from raw Ukrainian narrative text."""
    combined = " ".join(texts).lower()
    if any(k in combined for k in ["пішов з життя", "пішла з життя", "помер", "померла", "загинув", "загинула", "смерт"]):
        return "historical_demise"
    if any(k in combined for k in ["особистість", "характер", "був добрим", "була нервов", "тип а", "найзліш", "поведінк", "дисертац", "психолог", "професі", "освіт", "кар'єр", "фах"]):
        return "personality_profile"
    if any(k in combined for k in ["альфа", "правління", "вожак", "влада", "статус", "ієрархі"]):
        return "alpha_reign"
    if any(k in combined for k in ["бійк", "ікла", "урія", "рубен", "напад", "сутичк", "дротик"]):
        return "confrontation"
    if any(k in combined for k in ["дерево", "сеч", "тічк", "еструс", "спаров", "догляд"]):
        return "social_interaction"
    return "general_scene"


def auto_cluster_and_register_scenes(
    conn: sqlite3.Connection,
    entity_id: int,
    canonical_name: str,
    max_gap: int = 2,
) -> List[str]:
    """
    Deterministically cluster an entity's paragraph mentions into cohesive scenes,
    infer their scene types, and register them in entity_scenes & entity_mentions.
    """
    mentions = conn.execute(
        "SELECT paragraph_id FROM entity_mentions WHERE entity_id = ? ORDER BY paragraph_id",
        (entity_id,),
    ).fetchall()
    if not mentions:
        return []

    pids = [m[0] for m in mentions]
    clusters = []
    c_start = pids[0]
    c_end = pids[0]
    for pid in pids[1:]:
        if pid - c_end <= (max_gap + 1):
            c_end = pid
        else:
            clusters.append((c_start, c_end))
            c_start = pid
            c_end = pid
    clusters.append((c_start, c_end))

    registered_scene_ids = []
    for idx, (s_pid, e_pid) in enumerate(clusters, 1):
        scene_id = f"{canonical_name.lower()}-s{idx}"
        pages_and_texts = conn.execute(
            "SELECT min(page_number), max(page_number), GROUP_CONCAT(display_text, ' ') FROM paragraphs WHERE id BETWEEN ? AND ?",
            (s_pid, e_pid),
        ).fetchone()
        s_page = pages_and_texts[0]
        e_page = pages_and_texts[1]
        raw_text = pages_and_texts[2] or ""
        stype = infer_scene_type([raw_text])

        register_scene(
            conn,
            scene_id=scene_id,
            entity_id=entity_id,
            start_pid=s_pid,
            end_pid=e_pid,
            start_page=s_page,
            end_page=e_page,
            scene_type=stype,
            description=f"Auto-clustered scene {idx} ({s_pid}..{e_pid})",
        )
        conn.execute(
            "UPDATE entity_mentions SET scene_id = ? WHERE entity_id = ? AND paragraph_id BETWEEN ? AND ?",
            (scene_id, entity_id, s_pid, e_pid),
        )
        registered_scene_ids.append(scene_id)

    conn.commit()
    return registered_scene_ids


def seed_pilot_solomon(conn: sqlite3.Connection) -> int:
    """Seed Golden Pilot Entity: Baboon Solomon (Imposing Alpha, Power Transition, Demise)."""
    ensure_entity_schema(conn)
    doc_row = conn.execute("SELECT id FROM documents WHERE title LIKE '%Primate%' LIMIT 1").fetchone()
    book_id = doc_row[0] if doc_row else "doc-a-primates-memoir-0348660a"

    entity_id = register_entity(
        conn,
        canonical_name="Solomon",
        entity_type="baboon",
        book_id=book_id,
        aliases=["Соломон", "Solomon", "вожак Соломон", "альфа Соломон"],
    )
    for v in generate_ukrainian_name_variants("Соломон"):
        register_alias(conn, entity_id, v, language="uk", alias_type="generated_morphology")

    # Deterministic Mention Extraction
    rows = conn.execute("SELECT id, display_text FROM paragraphs WHERE document_id = ? ORDER BY id", (book_id,)).fetchall()
    regex = re.compile(r"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ])(?:Соломон[а-я]*|Solomon)(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])", re.IGNORECASE)
    for pid, text in rows:
        if regex.search(text):
            register_mention(conn, entity_id, pid, mention_type="explicit_name")

    auto_cluster_and_register_scenes(conn, entity_id, "Solomon")
    return entity_id


def seed_pilot_ruth(conn: sqlite3.Connection) -> int:
    """Seed Golden Pilot Entity: Baboon Ruth (Anxious Female, Low Status, Victim, Mother)."""
    ensure_entity_schema(conn)
    doc_row = conn.execute("SELECT id FROM documents WHERE title LIKE '%Primate%' LIMIT 1").fetchone()
    book_id = doc_row[0] if doc_row else "doc-a-primates-memoir-0348660a"

    entity_id = register_entity(
        conn,
        canonical_name="Ruth",
        entity_type="baboon",
        book_id=book_id,
        aliases=["Рут", "Руфі", "Руф", "Ruth", "бідна Рут", "Руті"],
    )

    rows = conn.execute("SELECT id, display_text FROM paragraphs WHERE document_id = ? ORDER BY id", (book_id,)).fetchall()
    regex = re.compile(r"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ])(?:Рут[а-я]*|Руф[а-я]*|Ruth)(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])", re.IGNORECASE)
    for pid, text in rows:
        if regex.search(text):
            register_mention(conn, entity_id, pid, mention_type="explicit_name")

    auto_cluster_and_register_scenes(conn, entity_id, "Ruth")
    return entity_id


def seed_pilot_naomi(conn: sqlite3.Connection) -> int:
    """Seed Golden Pilot Entity: Baboon Naomi (High-Ranking Matriarch, Matrilineal Dynasty)."""
    ensure_entity_schema(conn)
    doc_row = conn.execute("SELECT id FROM documents WHERE title LIKE '%Primate%' LIMIT 1").fetchone()
    book_id = doc_row[0] if doc_row else "doc-a-primates-memoir-0348660a"

    entity_id = register_entity(
        conn,
        canonical_name="Naomi",
        entity_type="baboon",
        book_id=book_id,
        aliases=["Наомі", "Naomi", "стара Наомі", "матріарх Наомі"],
    )

    rows = conn.execute("SELECT id, display_text FROM paragraphs WHERE document_id = ? ORDER BY id", (book_id,)).fetchall()
    regex = re.compile(r"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ])(?:Наомі|Naomi)(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])", re.IGNORECASE)
    for pid, text in rows:
        if regex.search(text):
            register_mention(conn, entity_id, pid, mention_type="explicit_name")

    auto_cluster_and_register_scenes(conn, entity_id, "Naomi")
    return entity_id


def seed_pilot_fresco(conn: sqlite3.Connection) -> int:
    """Seed Golden Pilot Entity: Jacque Fresco (Social Cybernetics & Ideas)."""
    ensure_entity_schema(conn)
    doc_row = conn.execute("SELECT id FROM documents WHERE title LIKE '%Проектування%' OR title LIKE '%Проєктування%' LIMIT 1").fetchone()
    book_id = doc_row[0] if doc_row else "doc-proektuvannya-maybutnoho-546d9dcc"

    entity_id = register_entity(
        conn,
        canonical_name="Jacque Fresco",
        entity_type="person",
        book_id=book_id,
        aliases=["Жак Фреско", "Фреско", "Jacque Fresco", "Fresco"],
    )

    rows = conn.execute("SELECT id, display_text FROM paragraphs WHERE document_id = ? ORDER BY id", (book_id,)).fetchall()
    regex = re.compile(r"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ])(?:Фреско|Fresco)(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])", re.IGNORECASE)
    for pid, text in rows:
        if regex.search(text):
            register_mention(conn, entity_id, pid, mention_type="explicit_name")

    auto_cluster_and_register_scenes(conn, entity_id, "Fresco")
    return entity_id


def seed_all_pilots(conn: sqlite3.Connection) -> Dict[str, int]:
    """Seed all Golden Pilot Entities (Nick, Solomon, Ruth, Naomi, Fresco)."""
    return {
        "Nick": seed_pilot_nick(conn),
        "Solomon": seed_pilot_solomon(conn),
        "Ruth": seed_pilot_ruth(conn),
        "Naomi": seed_pilot_naomi(conn),
        "Fresco": seed_pilot_fresco(conn),
    }


# ═══════════════════════════════════════════════════════════════════
#  Phase 4.3: Autonomous Entity Discovery & Promotion Pipeline
# ═══════════════════════════════════════════════════════════════════

GAZETTEER_PLACES: Set[str] = {
    "найробі", "кенія", "кенії", "кенію", "кенією",
    "африка", "африки", "африці", "африку", "африкою",
    "танзанія", "танзанії", "танзанію", "танзанією",
    "серенгеті", "масаї-мара", "уганда", "уганді", "уганд",
    "судан", "судані", "нью-йорк", "бостон", "гарвард",
    "стенфорд", "лондон", "париж", "америка", "сша", "європа",
    "кіліманджаро", "амбоселі", "моші", "аруша", "мояле",
    "руанда", "руанди", "руандою", "заїр", "конго", "сомалі",
    "ефіопія", "джуби", "джубі", "кампалі", "кампали",
    "йорк", "йорку", "йорка", "бруклін", "брукліні", "брукліна",
    "шотландія", "шотландії", "юдея", "юдеї", "захід", "заходу", "заході",
}

CONCEPT_STOPWORDS: Set[str] = {
    "розділ", "глава", "частина", "примітка", "автор", "книга", "стор",
    "понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя",
    "січень", "лютий", "березень", "квітень", "травень", "червень",
    "липень", "серпень", "вересень", "жовтень", "листопад", "грудень",
    "година", "хвилина", "секунда", "тиждень", "місяць", "року", "роки", "років",
    "масаї", "масаїв", "масаями", "масаям",
}

DISCOURSE_MARKERS: Set[str] = {
    "гаразд", "знаєте", "може", "можу", "хіба", "нічого", "старий",
    "бувай", "слухай", "дивись", "правда", "добре", "буде", "нехай",
    "кажуть", "видно", "здається", "малюк", "інше",
}

GENERAL_STOPWORDS: Set[str] = {
    "я", "ми", "він", "вона", "вони", "воно", "ти", "ви", "це", "той", "та", "те", "ті",
    "але", "і", "й", "також", "коли", "як", "якщо", "де", "куди", "звідки", "чому", "тому",
    "так", "ні", "навіть", "лише", "тільки", "ще", "вже", "після", "перед", "під", "над",
    "через", "для", "про", "до", "від", "із", "зі", "із-за", "що", "хто", "однак", "проте",
    "наприклад", "зокрема", "хоча", "нарешті", "наостанок", "пізніше", "раптом", "тоді",
    "одного", "раз", "інший", "інша", "інші", "кожний", "кожен", "кожна", "кожне",
    "дуже", "більш", "менш", "багато", "мало", "тут", "там", "сюди", "туди", "зараз",
    "тепер", "вранці", "ввечері", "вдень", "вночі", "сьогодні", "вчора", "завтра",
    "спершу", "спочатку", "наприкінці", "згодом", "разом", "сам", "сама", "самі", "саме",
}

PAST_VERB_PATTERN = re.compile(
    r"\b(сказав[а-я]*|подививс[а-я]*|пішов[а-я]*|пішл[а-я]*|зробив[а-я]*|напав[а-я]*|"
    r"вирішив[а-я]*|вирушив[а-я]*|побачив[а-я]*|закричав[а-я]*|втік[а-я]*|помер[а-я]*|"
    r"народивс[а-я]*|кинувс[а-я]*|взяв[а-я]*|був[а-я]*|була[а-я]*|жив[а-я]*|жил[а-я]*|"
    r"відчув[а-я]*|знав[а-я]*|сидів[а-я]*|стояв[а-я]*|біг[а-я]*|повернувс[а-я]*)\b",
    re.IGNORECASE
)

PREPOSITION_PATTERN = re.compile(r"\b(у|в|до|з|зі|із|біля|навколо|поблизу)\s+$", re.IGNORECASE)

UKR_CASE_SUFFIXES = [
    "ові", "еві", "ою", "ею", "єю", "ям", "ем", "ом",
    "ів", "ях", "ах", "их", "ами", "ями", "а", "я",
    "у", "ю", "і", "и", "е", "ї"
]


def extract_stem(word: str) -> str:
    """Strip Ukrainian noun declension suffixes down to canonical stem."""
    w = word.strip()
    for suffix in UKR_CASE_SUFFIXES:
        if w.lower().endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def mine_entity_candidates(conn: sqlite3.Connection, document_id: str) -> List[Dict[str, Any]]:
    """
    Phase 4.3.1: Candidate Mining.
    Scans corpus paragraphs for recurring capitalized entities, computes occurrence statistics,
    groups morphological declensions into canonical clusters, classifies types, and stores in SQLite.
    """
    ensure_entity_schema(conn)

    rows = conn.execute(
        "SELECT id, page_number, display_text FROM paragraphs WHERE document_id = ? ORDER BY id",
        (document_id,)
    ).fetchall()

    if not rows:
        return []

    # Map paragraph IDs to chapter indices based on ## headings
    current_chapter = 1
    pid_to_chapter = {}
    for pid, pg, text in rows:
        if text.strip().startswith("##"):
            current_chapter += 1
        pid_to_chapter[pid] = current_chapter

    from collections import defaultdict
    word_stats = defaultdict(lambda: {
        "total": 0,
        "mid_sentence": 0,
        "pages": set(),
        "pids": set(),
        "chapters": set(),
        "actions": 0,
        "prep_preceded": 0,
    })

    for pid, pg, text in rows:
        ch = pid_to_chapter[pid]
        for m in re.finditer(r"(?<!\w)([А-ЯІЇЄҐA-Z][а-яіїєґa-z\u02bc\u2019\x27-]+)(?!\w)", text):
            w = m.group(1)
            start = m.start()
            end = m.end()

            w_lower = w.lower()
            if w_lower in GENERAL_STOPWORDS or w_lower in CONCEPT_STOPWORDS or w_lower in DISCOURSE_MARKERS or len(w) < 3:
                continue

            preceding_str = text[:start]
            is_sentence_start = (start == 0) or bool(re.search(r"[.!?\n]\s*$", preceding_str))
            has_prep = bool(PREPOSITION_PATTERN.search(preceding_str))

            following_str = text[end:end+30]
            has_action = bool(PAST_VERB_PATTERN.match(following_str.strip()))

            st = word_stats[w]
            st["total"] += 1
            if not is_sentence_start:
                st["mid_sentence"] += 1
            if has_prep:
                st["prep_preceded"] += 1
            if has_action:
                st["actions"] += 1
            st["pages"].add(pg)
            st["pids"].add(pid)
            st["chapters"].add(ch)

    # Group inflected forms into canonical entity clusters by stem
    stem_clusters = defaultdict(list)
    for w in word_stats.keys():
        s = extract_stem(w)
        stem_clusters[s.lower()].append(w)

    candidates = []
    now_str = datetime.datetime.utcnow().isoformat()

    for stem_lower, forms in stem_clusters.items():
        total_freq = sum(word_stats[f]["total"] for f in forms)
        total_mid = sum(word_stats[f]["mid_sentence"] for f in forms)
        all_pages = set().union(*(word_stats[f]["pages"] for f in forms))
        all_chapters = set().union(*(word_stats[f]["chapters"] for f in forms))
        total_actions = sum(word_stats[f]["actions"] for f in forms)
        total_preps = sum(word_stats[f]["prep_preceded"] for f in forms)

        canonical_surface = max(forms, key=lambda f: (word_stats[f]["total"], -len(f)))

        action_score = total_actions / max(1, total_freq)
        prep_score = total_preps / max(1, total_freq)

        # Phase 4.3.2: Type Classification
        is_known_place = any(f.lower() in GAZETTEER_PLACES for f in forms)
        if is_known_place or (prep_score > 0.4 and action_score < 0.05):
            candidate_type = "place"
        elif any(f.lower() in CONCEPT_STOPWORDS for f in forms) or any(f.lower() in DISCOURSE_MARKERS for f in forms):
            candidate_type = "concept"
        else:
            candidate_type = "person"

        mid_ratio = total_mid / max(1, total_freq)
        confidence = round(0.5 * mid_ratio + 0.3 * min(1.0, len(all_pages) / 5) + 0.2 * min(1.0, action_score * 3), 3)

        cand_data = {
            "document_id": document_id,
            "surface_form": canonical_surface,
            "canonical_stem": stem_lower,
            "forms": sorted(forms, key=lambda f: word_stats[f]["total"], reverse=True),
            "frequency": total_freq,
            "mid_sentence_count": total_mid,
            "page_count": len(all_pages),
            "chapter_count": len(all_chapters),
            "first_seen_page": min(all_pages) if all_pages else 0,
            "last_seen_page": max(all_pages) if all_pages else 0,
            "action_score": round(action_score, 3),
            "prep_score": round(prep_score, 3),
            "candidate_type": candidate_type,
            "confidence": confidence,
            "status": "candidate",
            "created_at": now_str,
        }
        candidates.append(cand_data)

        # Store in entity_candidates table
        conn.execute("""
        INSERT INTO entity_candidates (
            document_id, surface_form, canonical_name, frequency, mid_sentence_count,
            page_count, chapter_count, first_seen_page, last_seen_page, action_score,
            prep_score, candidate_type, confidence, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, surface_form) DO UPDATE SET
            canonical_name = excluded.canonical_name,
            frequency = excluded.frequency,
            mid_sentence_count = excluded.mid_sentence_count,
            page_count = excluded.page_count,
            chapter_count = excluded.chapter_count,
            first_seen_page = excluded.first_seen_page,
            last_seen_page = excluded.last_seen_page,
            action_score = excluded.action_score,
            prep_score = excluded.prep_score,
            candidate_type = excluded.candidate_type,
            confidence = excluded.confidence,
            created_at = excluded.created_at;
        """, (
            document_id, canonical_surface, canonical_surface, total_freq, total_mid,
            len(all_pages), len(all_chapters), cand_data["first_seen_page"], cand_data["last_seen_page"],
            cand_data["action_score"], cand_data["prep_score"], candidate_type, confidence,
            "candidate", now_str,
        ))

    conn.commit()
    return candidates


def promote_entity_candidates(
    conn: sqlite3.Connection,
    document_id: str,
    min_frequency: int = 5,
    min_pages: int = 3,
    min_chapters: int = 2,
    min_confidence: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Phase 4.3.3: Promotion Gate & Phase 4.3.4: Automatic Dossier Construction.
    Evaluates mined candidates against strict promotion criteria:
      - candidate_type in ('person', 'animal')
      - frequency >= min_frequency
      - page_count >= min_pages
      - chapter_count >= min_chapters
      - mid_sentence_count >= 3
      - confidence >= min_confidence
    Promotes approved candidates into canonical `entities`, registers morphological aliases,
    indexes primary corpus paragraph mentions, and clusters cohesive narrative scenes.
    """
    candidates = mine_entity_candidates(conn, document_id)
    promoted_entities = []

    rows = conn.execute(
        "SELECT id, display_text FROM paragraphs WHERE document_id = ? ORDER BY id",
        (document_id,)
    ).fetchall()

    for cand in candidates:
        is_qualified = (
            cand["candidate_type"] in ("person", "animal")
            and cand["frequency"] >= min_frequency
            and cand["page_count"] >= min_pages
            and cand["chapter_count"] >= min_chapters
            and cand["mid_sentence_count"] >= 3
            and cand["confidence"] >= min_confidence
        )

        if not is_qualified:
            conn.execute(
                "UPDATE entity_candidates SET status = 'rejected' WHERE document_id = ? AND surface_form = ?",
                (document_id, cand["surface_form"])
            )
            continue

        cname = cand["surface_form"]
        forms = cand["forms"]

        # Check if entity already exists
        existing = conn.execute(
            "SELECT id FROM entities WHERE canonical_name = ? AND book_id = ?",
            (cname, document_id)
        ).fetchone()

        if existing:
            eid = existing[0]
        else:
            eid = register_entity(
                conn,
                canonical_name=cname,
                entity_type=cand["candidate_type"],
                book_id=document_id,
                aliases=forms,
            )

        # Register auto-discovered aliases
        for f in forms:
            register_alias(conn, entity_id=eid, alias=f, language="uk", alias_type="auto_discovered")

        # Scan document paragraphs for mentions
        pattern_str = r"(?:^|[^\wа-яА-ЯіїєґІЇЄҐ])(?:" + "|".join(re.escape(f) for f in forms) + r")(?:$|[^\wа-яА-ЯіїєґІЇЄҐ])"
        regex = re.compile(pattern_str, re.IGNORECASE)

        for pid, ptext in rows:
            if regex.search(ptext):
                register_mention(conn, entity_id=eid, paragraph_id=pid, mention_type="explicit_name")

        # Auto-cluster contiguous narrative scenes
        auto_cluster_and_register_scenes(conn, entity_id=eid, canonical_name=cname)

        # Mark candidate as approved
        conn.execute(
            "UPDATE entity_candidates SET status = 'approved', promoted_entity_id = ? WHERE document_id = ? AND surface_form = ?",
            (eid, document_id, cname)
        )

        cand["status"] = "approved"
        cand["promoted_entity_id"] = eid
        promoted_entities.append(cand)

    conn.commit()
    logger.info(f"Promoted {len(promoted_entities)} entities for document {document_id}")
    return promoted_entities


def auto_discover_entities_for_document(
    conn: sqlite3.Connection,
    document_id: str,
    min_frequency: int = 5,
    min_pages: int = 3,
    min_chapters: int = 2,
    min_confidence: float = 0.50,
) -> Dict[str, Any]:
    """
    High-level entry point for Autonomous Entity Discovery on any document.
    Executes:
      1. Candidate Mining
      2. Type Classification & Filtering
      3. Promotion Gate
      4. Scene Clustering & Dossier Generation
    Returns audit summary.
    """
    promoted = promote_entity_candidates(
        conn,
        document_id=document_id,
        min_frequency=min_frequency,
        min_pages=min_pages,
        min_chapters=min_chapters,
        min_confidence=min_confidence,
    )
    total_cand = conn.execute(
        "SELECT count(*) FROM entity_candidates WHERE document_id = ?", (document_id,)
    ).fetchone()[0]
    approved_count = len(promoted)
    rejected_count = total_cand - approved_count

    return {
        "document_id": document_id,
        "total_candidates": total_cand,
        "approved_count": approved_count,
        "rejected_count": rejected_count,
        "promoted_entities": [p["surface_form"] for p in promoted],
    }

