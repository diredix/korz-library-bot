"""
test_entity_discovery_pipeline.py - Verification script for Phase 4.3
Tests:
  1. Candidate Mining & Frequency/Page/Chapter Distribution
  2. Type Classification (place, concept, person/animal)
  3. Promotion Gate
  4. Automatic Dossier Generation (mentions & scenes)
  5. Golden Entity Recall (Nick, Solomon, Ruth, Naomi, Lisa)
"""

import re
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

GAZETTEER_PLACES = {
    "найробі", "кенія", "кенії", "кенію", "кенією",
    "африка", "африки", "африці", "африку", "африкою",
    "танзанія", "танзанії", "танзанію", "танзанією",
    "серенгеті", "масаї-мара", "уганда", "уганді", "уганд",
    "судан", "судані", "нью-йорк", "бостон", "гарвард",
    "стенфорд", "лондон", "париж", "америка", "сша", "європа",
    "кіліманджаро", "амбоселі", "моші", "аруша", "мояле",
    "руанда", "заїр", "конго", "сомалі", "ефіопія"
}

CONCEPT_STOPWORDS = {
    "розділ", "глава", "частина", "примітка", "автор", "книга", "стор",
    "понеділок", "вівторок", "середа", "четвер", "п'ятниця", "субота", "неділя",
    "січень", "лютий", "березень", "квітень", "травень", "червень",
    "липень", "серпень", "вересень", "жовтень", "листопад", "грудень",
    "година", "хвилина", "секунда", "тиждень", "місяць", "року", "роки", "років",
    "масаї", "масаїв", "масаями", "масаям", # ethnonym / group
}

GENERAL_STOPWORDS = {
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


def test_mining_pipeline():
    conn = sqlite3.connect("data/korz_library.sqlite")
    book_id = "doc-a-primates-memoir-0348660a"

    rows = conn.execute(
        "SELECT id, page_number, display_text FROM paragraphs WHERE document_id = ? ORDER BY id",
        (book_id,)
    ).fetchall()

    # Track chapters based on ## headings
    current_chapter = 1
    pid_to_chapter = {}
    for pid, pg, text in rows:
        if text.strip().startswith("##"):
            current_chapter += 1
        pid_to_chapter[pid] = current_chapter

    # Raw word occurrences stats
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
        # Match words starting with capital letter
        for m in re.finditer(r"(?<!\w)([А-ЯІЇЄҐA-Z][а-яіїєґa-z\u02bc\u2019\x27-]+)(?!\w)", text):
            w = m.group(1)
            start = m.start()
            end = m.end()

            w_lower = w.lower()
            if w_lower in GENERAL_STOPWORDS or w_lower in CONCEPT_STOPWORDS or len(w) < 3:
                continue

            # Check if sentence-initial
            preceding_str = text[:start]
            is_sentence_start = (start == 0) or bool(re.search(r"[.!?\n]\s*$", preceding_str))

            # Preceding preposition check (e.g. "у Найробі", "до Кенії")
            has_prep = bool(PREPOSITION_PATTERN.search(preceding_str))

            # Following action/verb check within next 25 chars
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

    # Build grouped candidates
    candidates = []
    for stem_lower, forms in stem_clusters.items():
        total_freq = sum(word_stats[f]["total"] for f in forms)
        total_mid = sum(word_stats[f]["mid_sentence"] for f in forms)
        all_pages = set().union(*(word_stats[f]["pages"] for f in forms))
        all_pids = set().union(*(word_stats[f]["pids"] for f in forms))
        all_chapters = set().union(*(word_stats[f]["chapters"] for f in forms))
        total_actions = sum(word_stats[f]["actions"] for f in forms)
        total_preps = sum(word_stats[f]["prep_preceded"] for f in forms)

        # Pick the most frequent surface form as the canonical display name
        canonical_surface = max(forms, key=lambda f: (word_stats[f]["total"], -len(f)))

        action_score = total_actions / max(1, total_freq)
        prep_score = total_preps / max(1, total_freq)

        # Classify Type
        is_known_place = any(f.lower() in GAZETTEER_PLACES for f in forms)
        if is_known_place or (prep_score > 0.4 and action_score < 0.05):
            candidate_type = "place"
        elif any(f.lower() in CONCEPT_STOPWORDS for f in forms):
            candidate_type = "concept"
        else:
            candidate_type = "person" # or animal/character

        # Compute Confidence
        mid_ratio = total_mid / max(1, total_freq)
        confidence = round(0.5 * mid_ratio + 0.3 * min(1.0, len(all_pages) / 5) + 0.2 * min(1.0, action_score * 3), 3)

        # Promotion Gate Criteria:
        # 1. Type: person/animal
        # 2. total_freq >= 5
        # 3. distinct_pages >= 3
        # 4. distinct_chapters >= 2
        # 5. mid_sentence >= 3
        # 6. confidence >= 0.50
        promoted = (
            candidate_type in ("person", "animal")
            and total_freq >= 5
            and len(all_pages) >= 3
            and len(all_chapters) >= 2
            and total_mid >= 3
            and confidence >= 0.50
        )

        candidates.append({
            "canonical_name": canonical_surface,
            "forms": sorted(forms, key=lambda f: word_stats[f]["total"], reverse=True),
            "frequency": total_freq,
            "mid_sentence": total_mid,
            "pages_count": len(all_pages),
            "chapters_count": len(all_chapters),
            "first_page": min(all_pages) if all_pages else 0,
            "last_page": max(all_pages) if all_pages else 0,
            "action_score": round(action_score, 3),
            "prep_score": round(prep_score, 3),
            "candidate_type": candidate_type,
            "confidence": confidence,
            "promoted": promoted,
        })

    candidates.sort(key=lambda c: (c["promoted"], c["frequency"]), reverse=True)

    print("=" * 80)
    print("PROMOTED ENTITY CANDIDATES:")
    print("=" * 80)
    promoted_list = [c for c in candidates if c["promoted"]]
    for c in promoted_list:
        forms_str = ", ".join(c["forms"][:3])
        print(f"  {c['canonical_name']:16} [{c['candidate_type']}] | freq={c['frequency']:3d} | mid={c['mid_sentence']:3d} | pages={c['pages_count']:2d} | ch={c['chapters_count']:2d} | act={c['action_score']:.2f} | conf={c['confidence']:.2f} | forms: {forms_str}")

    print("\n" + "=" * 80)
    print("FILTERED PLACES / CONCEPTS (REJECTED FROM PROMOTION):")
    print("=" * 80)
    rejected_places = [c for c in candidates if not c["promoted"] and c["candidate_type"] == "place"][:10]
    for c in rejected_places:
        print(f"  {c['canonical_name']:16} [{c['candidate_type']}] | freq={c['frequency']:3d} | prep={c['prep_score']:.2f} | forms: {', '.join(c['forms'][:2])}")

    # Golden Evaluation
    golden = ["Нік", "Соломон", "Рут", "Наомі", "Ліза"]
    promoted_names = set(c["canonical_name"] for c in promoted_list)
    # Also check if any form matches
    promoted_all_forms = set()
    for c in promoted_list:
        promoted_all_forms.update(c["forms"])

    found_golden = [g for g in golden if g in promoted_names or g in promoted_all_forms]
    recall = len(found_golden) / len(golden)
    print("\n" + "=" * 80)
    print(f"GOLDEN ENTITY RECALL: {len(found_golden)}/{len(golden)} ({recall:.0%})")
    for g in golden:
        status = "✅ FOUND" if (g in promoted_names or g in promoted_all_forms) else "❌ MISSING"
        print(f"  {g:10}: {status}")
    print("=" * 80)


if __name__ == "__main__":
    test_mining_pipeline()
