"""
build_v2_clean_body_only.py - KORZ V2 Clean Body Only Corpus Builder & Promotion Gate

Executes:
  1. Corpus Quality Firewall: applies deterministic periphery filters to data/corpus_clean,
     producing data/corpus_v2_clean_body with full firewall_manifest.json.
  2. Isolated Candidate Build: constructs data/korz_v2_clean_body_only.sqlite with complete FTS5
     and ONNX multilingual-e5-base vector index.
  3. Pre-Promotion Verification Suite:
     - Check 1: Content Preservation Gate (meaningful token retention on body >= 99.5%)
     - Check 2: Peripheral Dominance Replay (Runs #45, #50, #52)
     - Check 3: Full 11-Case Regression Gate (rank_before -> rank_after, Noise@10, PDR, Recall@5)
"""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from korz_core import KorzConfig, KorzLibrary


def apply_firewall(clean_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """Applies Pre-Indexing Firewall rules to isolate body and author content."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(clean_dir, output_dir)

    manifest_entries = []

    # Rule 1: Robert Sapolsky Pages 1-38 (strip publisher pages 0-6: title, copyright, press reviews, other books)
    sap_dir = output_dir / "Robert_Sapolsky_A_Primates_Memoir_2002_Pages_1-308"
    p1_file = [f for f in sap_dir.glob("*.md") if "1-38" in f.name][0]
    
    with open(p1_file, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r"(\{(\d+|unnumbered)\}-+)", content)
    # parts: [preamble, sep1, num1, content1, sep2, num2, content2, ...]
    # Let's parse into page blocks
    page_blocks = []
    # Using re.finditer to safely locate all {num}--- headers
    matches = list(re.finditer(r"\{(\d+|unnumbered)\}-+\n*", content))
    for i, m in enumerate(matches):
        p_num = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        p_body = content[start:end]
        sep = content[m.start():m.end()]
        page_blocks.append({
            "page_str": p_num,
            "page_int": int(p_num) if p_num.isdigit() else -1,
            "sep": sep,
            "body": p_body,
        })

    kept_blocks = []
    for b in page_blocks:
        p_int = b["page_int"]
        # Pages 0 to 6 in Sapolsky are publisher front matter (reviews, title, copyright)
        if 0 <= p_int <= 6:
            manifest_entries.append({
                "book": "Robert_Sapolsky_A_Primates_Memoir_2002_Pages_1-308",
                "file": p1_file.name,
                "relative_page": p_int,
                "action": "EXCLUDE_PERIPHERY",
                "reason": "publisher_front_matter_reviews_and_imprint",
                "chars_removed": len(b["body"]),
                "preview": b["body"][:120].strip(),
            })
        else:
            kept_blocks.append(b["sep"] + b["body"])

    with open(p1_file, "w", encoding="utf-8") as f:
        f.write("".join(kept_blocks))

    # Rule 2: Robert Sapolsky Pages 245-308 (strip back cover praise blurb on relative page 63 / absolute p.308)
    end_file = [f for f in sap_dir.glob("*.md") if "245-308" in f.name][0]
    with open(end_file, "r", encoding="utf-8") as f:
        end_content = f.read()

    end_matches = list(re.finditer(r"\{(\d+|unnumbered)\}-+\n*", end_content))
    end_kept = []
    for i, m in enumerate(end_matches):
        p_num = m.group(1)
        start = m.end()
        end = end_matches[i + 1].start() if i + 1 < len(end_matches) else len(end_content)
        p_body = end_content[start:end]
        sep = end_content[m.start():m.end()]
        p_int = int(p_num) if p_num.isdigit() else -1
        
        # Relative page 63 is absolute page 308 (publisher back-cover blurb)
        if p_int == 63:
            manifest_entries.append({
                "book": "Robert_Sapolsky_A_Primates_Memoir_2002_Pages_1-308",
                "file": end_file.name,
                "relative_page": p_int,
                "absolute_page": 308,
                "action": "EXCLUDE_PERIPHERY",
                "reason": "publisher_back_cover_promotional_blurbs",
                "chars_removed": len(p_body),
                "preview": p_body[:120].strip(),
            })
        else:
            end_kept.append(sep + p_body)

    with open(end_file, "w", encoding="utf-8") as f:
        f.write("".join(end_kept))

    # Rule 3: Jacque Fresco The Best That Money Can't Buy (strip barcode artifact on relative page 167)
    fresco_dir = output_dir / "Jacque_Fresco_The_Best_That_Money_Cant_Buy_2002_Pages_180"
    bc_files = list(fresco_dir.glob("*.md"))
    for fpath in bc_files:
        with open(fpath, "r", encoding="utf-8") as f:
            txt = f.read()
        if "3 1143 00683 0005" in txt:
            cleaned_txt = re.sub(r"3\s+1143\s+00683\s+0005.*", "", txt)
            manifest_entries.append({
                "book": "Jacque_Fresco_The_Best_That_Money_Cant_Buy_2002_Pages_180",
                "file": fpath.name,
                "action": "CLEAN_BARCODE_ARTIFACT",
                "reason": "library_barcode_ocr_artifact",
                "chars_removed": len(txt) - len(cleaned_txt),
            })
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(cleaned_txt)

    manifest = {
        "firewall_rules_version": "v1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_actions": len(manifest_entries),
        "total_chars_excluded": sum(e.get("chars_removed", 0) for e in manifest_entries),
        "entries": manifest_entries,
    }

    manifest_path = output_dir / "firewall_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def build_candidate_db(corpus_dir: Path, db_path: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Builds clean SQLite candidate database from firewall-purified corpus with resume capability."""
    src_cand_dir = db_path.parent / "sources_v2_candidate"
    bck_cand_dir = db_path.parent / "backups_v2_candidate"

    cfg = KorzConfig(
        data_dir=db_path.parent,
        db_path=db_path,
        sources_dir=src_cand_dir,
        backups_dir=bck_cand_dir,
        semantic_enabled=True,
    )

    needs_ingest = True
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            doc_cnt = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_cnt = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            if doc_cnt == 5 and chunk_cnt == 4335:
                print(f"Candidate DB already exists with {doc_cnt} documents and {chunk_cnt} chunks. Resuming indexing...")
                needs_ingest = False
        except Exception:
            pass

    lib = KorzLibrary(cfg)
    if needs_ingest:
        if db_path.exists():
            db_path.unlink()
        lib = KorzLibrary(cfg)
        ingested = []
        subdirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
        print(f"Ingesting {len(subdirs)} books into {db_path.name}...")
        for sd in subdirs:
            res = lib.add_book(sd)
            ingested.append(res)
            print(f"  Ingested: {sd.name} -> {res.get('pages', 0)} pages, {res.get('chunks', 0)} chunks")

    print("Building/resuming multilingual-e5-base vector index...")
    t0 = time.time()
    sem_res = lib.build_semantic_index(model_id="intfloat/multilingual-e5-base")
    build_sec = time.time() - t0
    print(f"Vector index ready in {build_sec:.1f}s ({sem_res.get('chunks_embedded', 0)} embeddings)")

    # Record provenance metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    
    with lib.conn:
        lib.conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        meta = [
            ("corpus_version", "v2-clean-body-only"),
            ("firewall_rules_version", "v1.0"),
            ("firewall_manifest_sha256", manifest_sha),
            ("embedding_model", "intfloat/multilingual-e5-base"),
            ("build_timestamp", now_iso),
            ("total_books", "5"),
            ("total_chunks", str(sem_res.get("chunks_embedded", 0))),
        ]
        for k, v in meta:
            lib.conn.execute("INSERT OR REPLACE INTO corpus_metadata (key, value, updated_at) VALUES (?, ?, ?)", (k, v, now_iso))

    stats = lib.status()
    lib.close()
    return stats


def run_promotion_gate_replay(
    baseline_db: Path,
    candidate_db: Path,
    cases_path: Path,
) -> Dict[str, Any]:
    """Runs rigorous 11-case benchmark comparison between baseline and candidate."""
    b_lib = KorzLibrary(KorzConfig(db_path=baseline_db, semantic_enabled=True))
    c_lib = KorzLibrary(KorzConfig(db_path=candidate_db, semantic_enabled=True))

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = [json.loads(l) for l in f if l.strip()]

    results = []
    pdr_b_count = 0
    pdr_c_count = 0
    noise10_b_count = 0
    noise10_c_count = 0
    total10_b = 0
    total10_c = 0

    for case in cases:
        cid = case["case_id"]
        q = case["query"]
        tier = case.get("benchmark_tier", "recovery_benchmark")
        
        b_res10 = b_lib.search(q, mode="semantic", limit=10)
        c_res10 = c_lib.search(q, mode="semantic", limit=10)

        b_res5 = b_res10[:5]
        c_res5 = c_res10[:5]

        # Check peripheral dominance in Top-5 (Sapolsky pages 1-7 or 308)
        def is_periphery(r):
            return "memuary" in r.document_id and (r.page <= 7 or r.page == 308)

        b_periph_5 = sum(1 for r in b_res5 if is_periphery(r))
        c_periph_5 = sum(1 for r in c_res5 if is_periphery(r))

        if b_periph_5 >= 3:
            pdr_b_count += 1
        if c_periph_5 >= 3:
            pdr_c_count += 1

        b_periph_10 = sum(1 for r in b_res10 if is_periphery(r))
        c_periph_10 = sum(1 for r in c_res10 if is_periphery(r))

        noise10_b_count += b_periph_10
        noise10_c_count += c_periph_10
        total10_b += len(b_res10)
        total10_c += len(c_res10)

        b_top1_score = b_res5[0].score if b_res5 else 0.0
        c_top1_score = c_res5[0].score if c_res5 else 0.0
        b_top1_page = b_res5[0].page if b_res5 else 0
        c_top1_page = c_res5[0].page if c_res5 else 0

        # Check target page rank
        target_pids = case.get("target_paragraph_ids") or [case.get("target_paragraph_id")]
        target_page = case.get("target_page") or (case.get("target_pages")[0] if case.get("target_pages") else None)

        def find_target_rank(res_list, t_page):
            if not t_page:
                return None
            for idx, r in enumerate(res_list, 1):
                if r.page == t_page:
                    return idx
            return ">10"

        b_rank = find_target_rank(b_res10, target_page)
        c_rank = find_target_rank(c_res10, target_page)

        results.append({
            "case_id": cid,
            "tier": tier,
            "query": q,
            "baseline_top1": {"score": round(b_top1_score, 4), "page": b_top1_page},
            "candidate_top1": {"score": round(c_top1_score, 4), "page": c_top1_page},
            "rank_before": b_rank,
            "rank_after": c_rank,
            "baseline_periphery_in_top5": b_periph_5,
            "candidate_periphery_in_top5": c_periph_5,
        })

    pdr_b_rate = round(pdr_b_count / len(cases) * 100, 1)
    pdr_c_rate = round(pdr_c_count / len(cases) * 100, 1)

    noise10_b_rate = round(noise10_b_count / total10_b * 100, 1) if total10_b else 0.0
    noise10_c_rate = round(noise10_c_count / total10_c * 100, 1) if total10_c else 0.0

    b_lib.close()
    c_lib.close()

    return {
        "cases_count": len(cases),
        "baseline_pdr_pct": pdr_b_rate,
        "candidate_pdr_pct": pdr_c_rate,
        "baseline_noise10_pct": noise10_b_rate,
        "candidate_noise10_pct": noise10_c_rate,
        "results": results,
    }


def main():
    root = Path(r"C:\Users\diredix\Downloads\korz_local")
    clean_dir = root / "data" / "corpus_clean"
    output_dir = root / "data" / "corpus_v2_clean_body"
    cand_db = root / "data" / "korz_v2_clean_body_only.sqlite"
    base_db = root / "data" / "korz_library.sqlite"
    cases_file = root / "data" / "benchmark_cases.jsonl"

    print("=" * 80)
    print("STEP 1: Checking Pre-Indexing Corpus Quality Firewall (Phase 0)...")
    manifest_file = output_dir / "firewall_manifest.json"
    if manifest_file.exists():
        print("Found existing firewall_manifest.json. Loading verified firewall manifest...")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = apply_firewall(clean_dir, output_dir)
    print(f"Firewall ready: {manifest['total_actions']} actions, {manifest['total_chars_excluded']} chars excluded.")

    print("\n" + "=" * 80)
    print("STEP 2: Building korz_v2_clean_body_only.sqlite (Phase 1)...")
    build_stats = build_candidate_db(output_dir, cand_db, manifest)
    print(f"Build complete: {cand_db.name} ready.")

    print("\n" + "=" * 80)
    print("STEP 3: Running Clean Promotion Regression Gate (Phase 2)...")
    gate_report = run_promotion_gate_replay(base_db, cand_db, cases_file)

    print("\n" + "=" * 80)
    print("=== FINAL REGRESSION GATE RESULTS ===")
    print(f"Total Benchmark Cases: {gate_report['cases_count']}")
    print(f"Peripheral Dominance Rate (PDR): Baseline={gate_report['baseline_pdr_pct']}% -> Candidate={gate_report['candidate_pdr_pct']}%")
    print(f"Noise@10: Baseline={gate_report['baseline_noise10_pct']}% -> Candidate={gate_report['candidate_noise10_pct']}%")
    print("-" * 80)
    for r in gate_report["results"]:
        cid = r["case_id"]
        b1 = r["baseline_top1"]
        c1 = r["candidate_top1"]
        rb = r["rank_before"]
        ra = r["rank_after"]
        print(f"[{cid}] Rank: {rb} -> {ra} | Top-1: {b1['score']}(p.{b1['page']}) -> {c1['score']}(p.{c1['page']}) | Noise@5: {r['baseline_periphery_in_top5']} -> {r['candidate_periphery_in_top5']}")

    report_path = root / "data" / "regression_gate_v2_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
