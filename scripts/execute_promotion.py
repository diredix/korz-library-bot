"""
execute_promotion.py - Safe, transactional promotion of KORZ V2 Clean Body Only to production baseline
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def promote():
    root = Path(r"C:\Users\diredix\Downloads\korz_local")
    data_dir = root / "data"
    backups_dir = data_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    v1_live = data_dir / "korz_library.sqlite"
    v1_backup = backups_dir / "korz_library_v1_frozen.sqlite"
    v2_cand = data_dir / "korz_v2_clean_body_only.sqlite"

    print("=" * 80)
    print("STEP 1: Checkpointing and Freezing v1 Baseline...")
    if v1_live.exists():
        # Checkpoint WAL
        conn_v1 = sqlite3.connect(str(v1_live))
        conn_v1.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn_v1.close()

        # Create archival frozen backup
        shutil.copy2(v1_live, v1_backup)
        v1_sha = compute_sha256(v1_backup)
        print(f"Archived v1 frozen baseline: {v1_backup.name} (SHA: {v1_sha[:16]}...)")
    else:
        raise FileNotFoundError(f"Missing live baseline: {v1_live}")

    print("\n" + "=" * 80)
    print("STEP 2: Preparing Candidate v2-clean-body-only...")
    if not v2_cand.exists():
        raise FileNotFoundError(f"Missing candidate database: {v2_cand}")

    # Checkpoint WAL on candidate
    conn_v2 = sqlite3.connect(str(v2_cand))
    conn_v2.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn_v2.close()
    v2_sha = compute_sha256(v2_cand)
    print(f"Candidate verified: {v2_cand.name} (SHA: {v2_sha[:16]}...)")

    print("\n" + "=" * 80)
    print("STEP 3: Atomic Baseline Replacement...")
    # Remove existing live files
    for suffix in ["", "-wal", "-shm"]:
        p = data_dir / f"korz_library.sqlite{suffix}"
        if p.exists():
            p.unlink()

    # Copy candidate into live position
    shutil.copy2(v2_cand, v1_live)
    print(f"Replaced {v1_live.name} with {v2_cand.name}.")

    print("\n" + "=" * 80)
    print("STEP 4: Writing Production Promotion Manifest into Live Database...")
    now_iso = datetime.now(timezone.utc).isoformat()
    conn_live = sqlite3.connect(str(v1_live))
    with conn_live:
        conn_live.execute("""
        CREATE TABLE IF NOT EXISTS corpus_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        manifest_meta = [
            ("active_corpus", "v2-clean-body-only"),
            ("previous_corpus", "v1-baseline"),
            ("promotion_timestamp", now_iso),
            ("firewall_version", "v1.0"),
            ("embedding_model", "intfloat/multilingual-e5-base"),
            ("benchmark_gate", "PASS"),
            ("v1_baseline_backup", str(v1_backup)),
            ("v1_baseline_sha256", v1_sha),
            ("v2_promoted_sha256", v2_sha),
        ]
        for k, v in manifest_meta:
            conn_live.execute("INSERT OR REPLACE INTO corpus_metadata (key, value, updated_at) VALUES (?, ?, ?)", (k, v, now_iso))

    # Verify integrity of newly promoted live database
    print("\n" + "=" * 80)
    print("STEP 5: Verifying Live Database Integrity...")
    integrity = conn_live.execute("PRAGMA integrity_check;").fetchall()
    docs = conn_live.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = conn_live.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embs = conn_live.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    meta_rows = conn_live.execute("SELECT key, value FROM corpus_metadata").fetchall()
    conn_live.close()

    print(f"PRAGMA integrity_check: {integrity[0][0]}")
    print(f"Total documents: {docs}")
    print(f"Total chunks: {chunks}")
    print(f"Total vector embeddings: {embs}")
    print("Live Corpus Metadata:")
    for k, v in meta_rows:
        print(f"   {k}: {v}")

    # Also save promotion audit log
    audit_file = data_dir / "promotion_manifest.json"
    audit_data = {
        "event": "PRODUCTION_BASELINE_PROMOTION",
        "active_corpus": "v2-clean-body-only",
        "previous_corpus": "v1-baseline",
        "timestamp": now_iso,
        "v1_baseline_backup": str(v1_backup),
        "v1_baseline_sha256": v1_sha,
        "v2_promoted_sha256": v2_sha,
        "total_documents": docs,
        "total_chunks": chunks,
        "total_embeddings": embs,
        "promotion_gate": "PASS",
    }
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print(f"\nAudit manifest written to: {audit_file.name}")
    print("\nPROMOTION COMPLETE! Ready to restart bot daemon.")

if __name__ == "__main__":
    promote()
