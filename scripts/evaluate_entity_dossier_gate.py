"""
evaluate_entity_dossier_gate.py
────────────────────────────────────────
Entity Dossier Evaluation Gate — Phase 4.1

Compares two retrieval strategies head-to-head:
  1. RRF-only (Simple Hybrid RRF, k=60)
  2. Entity Dossier (deterministic scene-graph evidence bundle)

Runs 15 golden benchmark queries across 5 archetypes × 3 entities (Nick, Solomon, Ruth):
  A1 — Static Portrait          ("Яким був X?")
  A2 — Dynamic Trajectory       ("Як змінився X з часом?")
  A3 — Conflict Dynamics         ("Який конфлікт мав X з Y?")
  A4 — Relational Graph          ("Які стосунки мав X?")
  A5 — Demise / Historical Fate  ("Що сталося з X наприкінці?")

Metrics per retrieval mode:
  • PID Count           — total unique paragraph IDs in evidence bundle
  • Page Count          — unique page numbers covered
  • Scene Continuity    — avg length of contiguous PID runs (higher = less fragmented)
  • Max Gap             — largest gap between adjacent PIDs (lower = better narrative flow)
  • Cliffhanger Rate    — % of PID runs that are singletons (isolated fragments)
"""

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import korz_entity
from korz_core import KorzLibrary, KorzConfig, compute_simulated_rrf


# ═══════════════════════════════════════════════════════════════════
#  GOLDEN BENCHMARK QUERIES
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_QUERIES: List[Dict[str, str]] = [
    # ── A1: Static Portrait ──
    {"id": "A1-nick",    "archetype": "Static Portrait",      "query": "Яким був бабуїн Нік?",          "entity": "Nick"},
    {"id": "A1-solomon", "archetype": "Static Portrait",      "query": "Яким був бабуїн Соломон?",      "entity": "Solomon"},
    {"id": "A1-ruth",    "archetype": "Static Portrait",      "query": "Якою була бабуїн Рут?",          "entity": "Ruth"},

    # ── A2: Dynamic Trajectory ──
    {"id": "A2-nick",    "archetype": "Dynamic Trajectory",   "query": "Як змінився Нік з часом?",       "entity": "Nick"},
    {"id": "A2-solomon", "archetype": "Dynamic Trajectory",   "query": "Як склалася доля Соломона?",     "entity": "Solomon"},
    {"id": "A2-ruth",    "archetype": "Dynamic Trajectory",   "query": "Що сталося з Рут наприкінці?",   "entity": "Ruth"},

    # ── A3: Conflict Dynamics ──
    {"id": "A3-nick",    "archetype": "Conflict Dynamics",    "query": "Який інцидент стався між Ніком і Рубеном?",         "entity": "Nick"},
    {"id": "A3-solomon", "archetype": "Conflict Dynamics",    "query": "Який конфлікт був між Соломоном і Урією?",          "entity": "Solomon"},
    {"id": "A3-ruth",    "archetype": "Conflict Dynamics",    "query": "Хто ображав Рут і як це відбувалося?",              "entity": "Ruth"},

    # ── A4: Relational Graph ──
    {"id": "A4-nick",    "archetype": "Relational Graph",     "query": "Які стосунки мав Нік з іншими бабуїнами?",          "entity": "Nick"},
    {"id": "A4-solomon", "archetype": "Relational Graph",     "query": "Хто належав до кола Соломона?",                     "entity": "Solomon"},
    {"id": "A4-ruth",    "archetype": "Relational Graph",     "query": "Які стосунки мала Рут з родиною?",                  "entity": "Ruth"},

    # ── A5: Demise / Historical Fate ──
    {"id": "A5-nick",    "archetype": "Demise/Historical",    "query": "Як Нік пішов з життя?",                             "entity": "Nick"},
    {"id": "A5-solomon", "archetype": "Demise/Historical",    "query": "Як Соломон пішов з життя?",                         "entity": "Solomon"},
    {"id": "A5-ruth",    "archetype": "Demise/Historical",    "query": "Що сталося з Рут під кінець книги?",                "entity": "Ruth"},
]


# ═══════════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    """Structural metrics for a single query's evidence bundle."""
    query_id: str
    archetype: str
    entity: str
    mode: str  # "rrf_only" | "entity_dossier"
    pid_count: int = 0
    page_count: int = 0
    scene_continuity: float = 0.0  # avg contiguous-run length
    max_gap: int = 0
    cliffhanger_rate: float = 0.0  # fraction of singleton runs
    pids: List[int] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)


def compute_metrics(pids: List[int], pages: List[int], query_id: str, archetype: str, entity: str, mode: str) -> RetrievalMetrics:
    """Compute structural retrieval metrics from a list of PIDs."""
    m = RetrievalMetrics(
        query_id=query_id,
        archetype=archetype,
        entity=entity,
        mode=mode,
        pid_count=len(pids),
        page_count=len(set(pages)),
        pids=sorted(pids),
        pages=sorted(set(pages)),
    )

    if not pids:
        return m

    sorted_pids = sorted(pids)
    # Compute contiguous runs
    runs = []
    run_start = sorted_pids[0]
    run_len = 1
    max_gap = 0
    for i in range(1, len(sorted_pids)):
        gap = sorted_pids[i] - sorted_pids[i - 1]
        if gap == 1:
            run_len += 1
        else:
            runs.append(run_len)
            if gap > max_gap:
                max_gap = gap
            run_start = sorted_pids[i]
            run_len = 1
    runs.append(run_len)

    m.scene_continuity = sum(runs) / len(runs) if runs else 0.0
    m.max_gap = max_gap
    singletons = sum(1 for r in runs if r == 1)
    m.cliffhanger_rate = singletons / len(runs) if runs else 0.0

    return m


# ═══════════════════════════════════════════════════════════════════
#  RRF-ONLY RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

def retrieve_rrf_only(lib: KorzLibrary, query: str, k: int = 12) -> Tuple[List[int], List[int]]:
    """Retrieve evidence via pure RRF (no entity dossier)."""
    dense_candidates = lib.ask(query, limit=max(k * 2, 20))
    fts_candidates = lib._search_fts(query, limit=max(k * 2, 20))

    if not dense_candidates and not fts_candidates:
        return [], []

    # compute_simulated_rrf(fts_results, semantic_results, k=60) -> (merged, overlap5, overlap10, top1_match)
    merged_tuple = compute_simulated_rrf(fts_candidates, dense_candidates, k=60, top_limit=k)
    merged = merged_tuple[0]  # List of dicts with 'paragraph_id', 'page', etc.

    pids = [r["paragraph_id"] for r in merged[:k]]
    pages = [r.get("page", 0) for r in merged[:k]]
    return pids, pages


# ═══════════════════════════════════════════════════════════════════
#  ENTITY DOSSIER RETRIEVAL
# ═══════════════════════════════════════════════════════════════════

def retrieve_entity_dossier(conn: sqlite3.Connection, query: str, entity_name: str, k: int = 12) -> Tuple[List[int], List[int]]:
    """Retrieve evidence via Entity Dossier scene-graph navigation."""
    entity_info = korz_entity.resolve_entity(conn, query)
    if not entity_info:
        # Fallback: try to resolve by canonical name directly
        row = conn.execute(
            "SELECT id FROM entities WHERE canonical_name = ?", (entity_name,)
        ).fetchone()
        if not row:
            return [], []
        entity_id = row[0]
    else:
        entity_id = entity_info["id"]

    bundle = korz_entity.build_entity_evidence_bundle(
        conn, entity_id=entity_id, query=query, max_paragraphs=k,
    )
    if not bundle:
        return [], []

    pids = [b["paragraph_id"] for b in bundle]
    pages = [b["page"] for b in bundle]
    return pids, pages


# ═══════════════════════════════════════════════════════════════════
#  MAIN EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════

def run_evaluation(db_path: str, k: int = 12) -> List[RetrievalMetrics]:
    """Run all 15 benchmark queries × 2 modes and collect metrics."""
    config = KorzConfig(db_path=Path(db_path))
    lib = KorzLibrary(config)
    conn = sqlite3.connect(db_path)

    # Ensure pilots are seeded
    korz_entity.ensure_entity_schema(conn)
    existing = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    if existing == 0:
        print("[SEED] Seeding all pilot entities...")
        korz_entity.seed_all_pilots(conn)
    else:
        print(f"[SEED] {existing} entities already present, skipping seed.")

    all_metrics: List[RetrievalMetrics] = []

    for bq in BENCHMARK_QUERIES:
        qid = bq["id"]
        query = bq["query"]
        archetype = bq["archetype"]
        entity = bq["entity"]

        print(f"\n{'─'*60}")
        print(f"  {qid}: {query}")
        print(f"  Archetype: {archetype} | Entity: {entity}")
        print(f"{'─'*60}")

        # Mode 1: RRF-only
        t0 = time.perf_counter()
        rrf_pids, rrf_pages = retrieve_rrf_only(lib, query, k=k)
        rrf_ms = (time.perf_counter() - t0) * 1000
        rrf_m = compute_metrics(rrf_pids, rrf_pages, qid, archetype, entity, "rrf_only")
        all_metrics.append(rrf_m)
        print(f"  RRF:     {rrf_m.pid_count:3d} PIDs | {rrf_m.page_count:2d} pages | continuity {rrf_m.scene_continuity:.1f} | max_gap {rrf_m.max_gap:4d} | cliffhanger {rrf_m.cliffhanger_rate:.0%} | {rrf_ms:.0f}ms")

        # Mode 2: Entity Dossier
        t0 = time.perf_counter()
        ed_pids, ed_pages = retrieve_entity_dossier(conn, query, entity, k=k)
        ed_ms = (time.perf_counter() - t0) * 1000
        ed_m = compute_metrics(ed_pids, ed_pages, qid, archetype, entity, "entity_dossier")
        all_metrics.append(ed_m)
        print(f"  Dossier: {ed_m.pid_count:3d} PIDs | {ed_m.page_count:2d} pages | continuity {ed_m.scene_continuity:.1f} | max_gap {ed_m.max_gap:4d} | cliffhanger {ed_m.cliffhanger_rate:.0%} | {ed_ms:.0f}ms")

        # Compare
        if ed_m.scene_continuity > rrf_m.scene_continuity:
            print(f"  ✅ Dossier wins: +{ed_m.scene_continuity - rrf_m.scene_continuity:.1f} continuity")
        elif rrf_m.scene_continuity > ed_m.scene_continuity:
            print(f"  ⚠️  RRF wins: +{rrf_m.scene_continuity - ed_m.scene_continuity:.1f} continuity")
        else:
            print(f"  ⚖️  Tie on continuity")

    conn.close()
    return all_metrics


def print_summary_table(metrics: List[RetrievalMetrics]) -> str:
    """Print and return a formatted summary comparison table."""
    lines = []
    header = f"{'Query':<14} | {'Mode':<16} | {'PIDs':>5} | {'Pages':>5} | {'Continuity':>10} | {'MaxGap':>6} | {'Cliffhanger':>11}"
    sep = "─" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    # Group by query_id
    by_qid = {}
    for m in metrics:
        by_qid.setdefault(m.query_id, []).append(m)

    rrf_wins = 0
    ed_wins = 0
    ties = 0

    for qid in sorted(by_qid.keys()):
        pair = by_qid[qid]
        for m in pair:
            lines.append(
                f"{m.query_id:<14} | {m.mode:<16} | {m.pid_count:5d} | {m.page_count:5d} | {m.scene_continuity:10.2f} | {m.max_gap:6d} | {m.cliffhanger_rate:10.0%}"
            )
        # Compare continuity
        rrf_m = [m for m in pair if m.mode == "rrf_only"]
        ed_m = [m for m in pair if m.mode == "entity_dossier"]
        if rrf_m and ed_m:
            if ed_m[0].scene_continuity > rrf_m[0].scene_continuity:
                ed_wins += 1
            elif rrf_m[0].scene_continuity > ed_m[0].scene_continuity:
                rrf_wins += 1
            else:
                ties += 1
        lines.append("")

    lines.append(sep)
    lines.append(f"VERDICT: Entity Dossier wins {ed_wins}/15 | RRF wins {rrf_wins}/15 | Ties {ties}/15")

    # Aggregate averages
    rrf_all = [m for m in metrics if m.mode == "rrf_only"]
    ed_all = [m for m in metrics if m.mode == "entity_dossier"]

    def avg(lst, attr):
        vals = [getattr(m, attr) for m in lst]
        return sum(vals) / len(vals) if vals else 0.0

    lines.append(f"\nAGGREGATE AVERAGES:")
    lines.append(f"  RRF-Only:       avg PIDs={avg(rrf_all, 'pid_count'):.1f}  avg Pages={avg(rrf_all, 'page_count'):.1f}  avg Continuity={avg(rrf_all, 'scene_continuity'):.2f}  avg Cliffhanger={avg(rrf_all, 'cliffhanger_rate'):.0%}")
    lines.append(f"  Entity Dossier: avg PIDs={avg(ed_all, 'pid_count'):.1f}  avg Pages={avg(ed_all, 'page_count'):.1f}  avg Continuity={avg(ed_all, 'scene_continuity'):.2f}  avg Cliffhanger={avg(ed_all, 'cliffhanger_rate'):.0%}")

    report = "\n".join(lines)
    print("\n" + report)
    return report


def export_metrics_json(metrics: List[RetrievalMetrics], path: str) -> None:
    """Export raw metrics as JSON for post-analysis."""
    data = []
    for m in metrics:
        data.append({
            "query_id": m.query_id,
            "archetype": m.archetype,
            "entity": m.entity,
            "mode": m.mode,
            "pid_count": m.pid_count,
            "page_count": m.page_count,
            "scene_continuity": round(m.scene_continuity, 3),
            "max_gap": m.max_gap,
            "cliffhanger_rate": round(m.cliffhanger_rate, 3),
            "pids": m.pids,
            "pages": m.pages,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[EXPORT] Metrics saved to {path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    db_path = str(PROJECT_ROOT / "data" / "korz_library.sqlite")
    k = 12

    print("=" * 70)
    print("  KORZ Phase 4.1 — Entity Dossier Evaluation Gate")
    print(f"  DB: {db_path}")
    print(f"  k={k} paragraphs per mode")
    print("=" * 70)

    metrics = run_evaluation(db_path, k=k)
    report_text = print_summary_table(metrics)

    # Export raw metrics
    json_path = str(PROJECT_ROOT / "data" / "entity_dossier_eval_metrics.json")
    export_metrics_json(metrics, json_path)

    # Write markdown report
    report_md_path = str(PROJECT_ROOT / "data" / "entity_dossier_eval_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# KORZ Phase 4.1 — Entity Dossier Evaluation Gate Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**k**: {k}\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")
    print(f"\n[REPORT] Markdown report saved to {report_md_path}")
