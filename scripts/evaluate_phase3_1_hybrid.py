"""
evaluate_phase3_1_hybrid.py - Comprehensive evaluation of Phase 3.1:
  Candidate A: Dense only (v2 baseline)
  Candidate B: FTS only (BM25 v2)
  Candidate C: Simple RRF (k=60, equal weights 1:1)
Evaluated across the 11 Golden Benchmark Cases.
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\diredix\Downloads\korz_local")

from korz_core import KorzLibrary, KorzConfig, compute_simulated_rrf

# Canonical target mapping in v2 (post-firewall IDs)
TARGETS_V2 = {
    35: {"pids": [9107, 9108, 9109, 9111], "type": "recovery", "name": "Соломон виклик"},
    38: {"pids": [9116, 9118, 9122, 9148, 9151, 9920], "type": "recovery", "name": "Опис Найробі"},
    45: {"pids": [9089, 9090, 9115], "type": "recovery", "name": "Мрія Сапольскі"},
    46: {"pids": [9104, 9105], "type": "recovery", "name": "Травми Аарона"},
    47: {"pids": [9116, 9117], "type": "recovery", "name": "Багаж Найробі"},
    48: {"pids": [], "type": "safety", "name": "Зовнішність Аарона (відсутня)"},
    49: {"pids": [9115], "type": "recovery", "name": "Підготовка до Африки"},
    50: {"pids": [9118], "type": "recovery", "name": "Не розумів суахілі"},
    51: {"pids": [9151, 9153, 9155], "type": "recovery", "name": "Виживання без грошей"},
    52: {"pids": [9130, 9552], "type": "recovery", "name": "Здивування в слонах"},
    53: {"pids": [7061, 7089], "type": "safety", "name": "Сорочка в кібернетиці (межі)"},
}

def evaluate_phase3_1():
    db_path = Path(r"C:\Users\diredix\Downloads\korz_local\data\korz_library.sqlite")
    cases_path = Path(r"C:\Users\diredix\Downloads\korz_local\data\benchmark_cases.jsonl")

    lib = KorzLibrary(KorzConfig(db_path=db_path, semantic_enabled=True))
    
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    print("=" * 110)
    print("PHASE 3.1 SCIENTIFIC EVALUATION: DENSE vs FTS vs SIMPLE RRF (k=60, 1:1)")
    print("Active Corpus: data/korz_library.sqlite (v2-clean-body-only)")
    print("=" * 110)

    results = []

    for c in cases:
        run_id = c["run_id"]
        q = c["query"]
        target_info = TARGETS_V2.get(run_id, {"pids": [], "type": "recovery", "name": "Unknown"})
        valid_targets = target_info["pids"]
        is_safety = target_info["type"] == "safety"

        # 1. Retrieve candidates (top 15 to assess recall and plateau)
        dense_res = lib.ask(q, limit=15)
        fts_res = lib._search_fts(q, limit=15)
        rrf_res, ov5, ov10, top1_match = compute_simulated_rrf(fts_res, dense_res, k=60, top_limit=15)

        # Extract PIDs
        dense_pids = [r.paragraph_id for r in dense_res]
        fts_pids = [r.paragraph_id for r in fts_res]
        rrf_pids = [r["paragraph_id"] for r in rrf_res]

        # Determine target ranks
        def find_first_rank(pids_list, targets):
            for rank, pid in enumerate(pids_list, 1):
                if pid in targets:
                    return rank
            return None

        rank_dense = find_first_rank(dense_pids, valid_targets)
        rank_fts = find_first_rank(fts_pids, valid_targets)
        rank_rrf = find_first_rank(rrf_pids, valid_targets)

        hit5_dense = rank_dense is not None and rank_dense <= 5
        hit5_fts = rank_fts is not None and rank_fts <= 5
        hit5_rrf = rank_rrf is not None and rank_rrf <= 5

        results.append({
            "run_id": run_id,
            "name": target_info["name"],
            "type": target_info["type"],
            "query": q,
            "rank_dense": rank_dense,
            "rank_fts": rank_fts,
            "rank_rrf": rank_rrf,
            "hit5_dense": hit5_dense,
            "hit5_fts": hit5_fts,
            "hit5_rrf": hit5_rrf,
            "overlap_5": ov5,
            "overlap_10": ov10,
        })

    # Summary table
    print(f"\n{'Run':<8} | {'Type':<8} | {'Topic':<28} | {'Dense (A)':<12} | {'FTS (B)':<12} | {'Simple RRF (C)':<15} | {'Overlap@5'}")
    print("-" * 110)

    recovery_dense_hits = 0
    recovery_fts_hits = 0
    recovery_rrf_hits = 0
    total_recovery = 0

    for r in results:
        t_type = r["type"]
        if t_type == "recovery":
            total_recovery += 1
            if r["hit5_dense"]: recovery_dense_hits += 1
            if r["hit5_fts"]: recovery_fts_hits += 1
            if r["hit5_rrf"]: recovery_rrf_hits += 1

        d_str = f"#{r['rank_dense']}" if r['rank_dense'] else "miss (>15)"
        f_str = f"#{r['rank_fts']}" if r['rank_fts'] else "miss (>15)"
        rrf_str = f"#{r['rank_rrf']}" if r['rank_rrf'] else "miss (>15)"
        
        # Add visual checkmarks
        d_badge = "✅" if r['hit5_dense'] else "❌"
        f_badge = "✅" if r['hit5_fts'] else "❌"
        rrf_badge = "✅" if r['hit5_rrf'] else "❌"

        if t_type == "safety":
            d_badge = f_badge = rrf_badge = "🛡️"
            d_str = f_str = rrf_str = "safe-bound"

        print(f"#{r['run_id']:<7} | {t_type:<8} | {r['name']:<28} | {d_str:<8} {d_badge} | {f_str:<8} {f_badge} | {rrf_str:<10} {rrf_badge}    | {r['overlap_5']}/5")

    print("-" * 110)
    print(f"Pillar 1 (Recovery Recall@5) Summary across {total_recovery} target cases:")
    print(f"  • Candidate A (Dense only) : {recovery_dense_hits}/{total_recovery} ({recovery_dense_hits/total_recovery*100:.1f}%)")
    print(f"  • Candidate B (FTS only)   : {recovery_fts_hits}/{total_recovery} ({recovery_fts_hits/total_recovery*100:.1f}%)")
    print(f"  • Candidate C (Simple RRF) : {recovery_rrf_hits}/{total_recovery} ({recovery_rrf_hits/total_recovery*100:.1f}%)")
    print("=" * 110)

    lib.close()

if __name__ == "__main__":
    evaluate_phase3_1()
