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
        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
        UNIQUE(entity_id, alias)
    );
    """)

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
) -> None:
    """Register an alias for an entity."""
    conn.execute(
        """
        INSERT OR IGNORE INTO entity_aliases (entity_id, alias, language)
        VALUES (?, ?, ?);
        """,
        (entity_id, alias.strip(), language),
    )
    conn.commit()


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
    """
    cur = conn.execute("""
        SELECT e.id, e.canonical_name, e.entity_type, e.book_id, a.alias
        FROM entities e
        JOIN entity_aliases a ON a.entity_id = e.id
        ORDER BY LENGTH(a.alias) DESC
    """)
    rows = cur.fetchall()

    normalized_text = f" {text.lower()} "
    for eid, cname, etype, bid, alias in rows:
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
            }
    return None


def detect_character_intent(query: str) -> str:
    """
    Classify user query intent into cognitive categories:
      - 'character_profiling': 'Яким був...', 'Якою була...', 'Ким був...', 'Хто такий...', 'характер', 'поведінка', 'ідеї'
      - 'pinpoint_fact': factual point query
      - 'causal_narrative': 'чому', 'як сталося'
      - 'general': standard retrieval
    """
    q_lower = query.lower()
    profile_patterns = [
        r"\bяким\s+був\b",
        r"\bяка\s+була\b",
        r"\bякою\s+була\b",
        r"\bяке\s+було\b",
        r"\bким\s+був\b",
        r"\bким\s+була\b",
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
    ]
    for p in profile_patterns:
        if re.search(p, q_lower):
            return "character_profiling"

    if re.search(r"\bчому\b|\bз\s+якої\s+причини\b|\bяк\s+так\s+сталося\b", q_lower):
        return "causal_narrative"

    return "pinpoint_fact"


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
    Detect the specific biographical aspect requested:
      - 'trajectory': 'як змінився', 'як склалася доля', 'що сталося', 'наприкінці', 'з роками'
      - 'conflict': 'конфлікт', 'бійка', 'протистояння', 'ворог', 'інцидент', 'поєдинок'
      - 'relational': 'стосунки', 'родина', 'союзник', 'партнер', 'хто належав', 'діти'
      - 'personality': 'яким був', 'характер', 'особистість', 'поведінка', 'портрет'
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


def build_entity_evidence_bundle(
    conn: sqlite3.Connection,
    entity_id: int,
    query: str,
    max_paragraphs: int = 12,
) -> List[Dict[str, Any]]:
    """
    Assemble the primary text Evidence Bundle for Gemini generation.
    Strictly queries authentic corpus paragraphs by PID ranges defined in Entity Scenes.
    Applies Query-Specific Aspect Filtering and Cross-Entity Co-occurrence boosts.
    """
    intent = detect_character_intent(query)
    dossier = get_entity_dossier(conn, entity_id)
    if not dossier or not dossier.get("scenes"):
        return []

    scenes = dossier["scenes"]
    aspect = detect_query_aspect(query)
    co_mentions = find_co_mentioned_terms(conn, query, entity_id)

    scored_scenes: List[Tuple[float, Dict[str, Any]]] = []
    for sc in scenes:
        score = 0.0

        # 1. Co-occurrence boost: if scene paragraphs mention co-mentioned terms
        if co_mentions:
            scene_text_row = conn.execute(
                "SELECT GROUP_CONCAT(display_text, ' ') FROM paragraphs WHERE id BETWEEN ? AND ?",
                (sc["start_pid"], sc["end_pid"]),
            ).fetchone()
            scene_text = (scene_text_row[0] or "").lower()
            for term in co_mentions:
                if term.lower() in scene_text:
                    score += 50.0

        # 2. Aspect match boost
        stype = sc["scene_type"]
        if aspect == "conflict":
            if stype in ("confrontation", "dominance_event", "dominance_bluff_resolution"):
                score += 25.0
        elif aspect == "trajectory":
            if stype in ("historical_demise", "alpha_reign"):
                score += 20.0
            elif stype == "personality_profile":
                score += 10.0
        elif aspect == "relational":
            if stype in ("social_interaction", "alpha_reign"):
                score += 20.0
        elif aspect == "personality":
            if stype == "personality_profile":
                score += 30.0
            elif stype in ("dominance_bluff_resolution", "alpha_reign"):
                score += 15.0

        scored_scenes.append((score, sc))

    # Sort descending by score
    selected_scenes = [sc for _, sc in sorted(scored_scenes, key=lambda x: x[0], reverse=True)]

    gathered_pids: List[int] = []
    for sc in selected_scenes:
        # Fetch all contiguous paragraphs within this scene cluster
        pids = list(range(sc["start_pid"], sc["end_pid"] + 1))
        for pid in pids:
            if pid not in gathered_pids:
                gathered_pids.append(pid)
            if len(gathered_pids) >= max_paragraphs:
                break
        if len(gathered_pids) >= max_paragraphs:
            break

    if not gathered_pids:
        return []

    # Query authentic paragraph texts directly from SQLite
    placeholders = ",".join(["?"] * len(gathered_pids))
    sql = f"""
    SELECT p.id, p.document_id, p.page_number, p.paragraph_index, p.display_text,
           d.title, d.author
    FROM paragraphs p
    JOIN documents d ON d.id = p.document_id
    WHERE p.id IN ({placeholders})
    ORDER BY p.id
    """
    rows = conn.execute(sql, gathered_pids).fetchall()

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
            "score": 1.0,  # Deterministic graph match
        })

    return bundle_items


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
    if any(k in combined for k in ["особистість", "характер", "був добрим", "була нервов", "тип а", "найзліш", "поведінк"]):
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

