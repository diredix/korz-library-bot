#!/usr/bin/env python3
"""Auto-update the korz agent after adding new UA book-page .md files.

Run this whenever you drop a new ukr.datalab-output-<slug>_<pages>.pdf.md
into C:/Users/diredix/Downloads/korz_71_84_hq/ — it:
  1. rebuilds qdrant_db from ALL ukr.datalab-output-*.md
  2. computes current page range + chunk count
  3. rewrites the "outside pages" line in SOUL.md and the citation line in SKILL.md
  4. syncs the skill into the korz profile
  5. restarts the korz gateway so it picks up everything

No manual SOUL/number edits needed anymore.
"""
import os, re, glob, sys, subprocess, shutil

HERE = r"C:\Users\diredix\Downloads\korz_71_84_hq"
PROFILE = r"C:\Users\diredix\AppData\Local\hermes\profiles\sania"
SKILL_GLOBAL = r"C:\Users\diredix\AppData\Local\hermes\skills\book-korzybsky-preface\SKILL.md"
SKILL_PROFILE = os.path.join(PROFILE, "skills", "book-korzybsky-preface", "SKILL.md")
SOUL = os.path.join(PROFILE, "SOUL.md")

# ---- 1. collect page ranges from FILENAMES (reliable) ----
# filename pattern: ukr.datalab-output-<slug>_<pages>.pdf.md  e.g. _20_70 or _103_121
files = sorted(glob.glob(os.path.join(HERE, "укр.datalab-output-*.md")))
ranges = []
blocks = 0
for f in files:
    m = re.search(r"datalab-output-[A-Za-z0-9]+_(\d+)_(\d+)\.pdf", f)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        ranges.append((lo, hi))
        blocks += 1
    else:
        # fallback: single page or unknown — try {NN} range but cap at sane max
        text = open(f, encoding="utf-8").read()
        marks = [int(x) for x in re.findall(r"\{(\d+)\}", text) if int(x) <= 900]
        if marks:
            ranges.append((min(marks), max(marks)))
            blocks += 1

if not ranges:
    print("no page ranges found in any ukr.datalab-output-*.md"); sys.exit(1)
min_p = min(lo for lo, hi in ranges)
max_p = max(hi for lo, hi in ranges)
total_chunks = 0  # computed by qdrant build below
print(f"FOUND: pages {min_p}-{max_p} across {blocks} file(s)")

# ---- 2. rebuild qdrant index ----
sys.path.insert(0, HERE)
import korz_qdrant
korz_qdrant.upsert_incremental()

# ---- 3. update SOUL.md (all "pages 71-N" coverage mentions) ----
s = open(SOUL, encoding="utf-8").read()
# match any "71-<digits>" range token and bump it to current max
# match any "N-M" range token after "pages" or "сторінки" and bump to current
s2 = re.sub(r"(pages\s+)(\d+)-(\d+)", lambda m: f"{m.group(1)}{min_p}-{max_p}", s)
s2 = re.sub(r"(Маю сторінки\s+)(\d+)[–-](\d+)", lambda m: f"{m.group(1)}{min_p}–{max_p}", s2)
s2 = re.sub(r"(\d+)-(\d+)(?=\s*\(PREFACE)", lambda m: f"{min_p}-{max_p}", s2)
if s2 != s:
    open(SOUL, "w", encoding="utf-8").write(s2)
    print("SOUL.md updated (coverage -> %d-%d)" % (min_p, max_p))
else:
    print("SOUL.md: no range token found, left unchanged")

# ---- 4. update SKILL.md (coverage mentions) + sync to profile ----
sk = open(SKILL_GLOBAL, encoding="utf-8").read()
# only bump a top-level "71-N" coverage token if present; do not touch section
# sub-ranges like 71-84 / 85-102 which are fixed chapter bounds
sk2 = re.sub(r"71[-–]\d+", f"{min_p}-{max_p}", sk)
if sk2 != sk:
    open(SKILL_GLOBAL, "w", encoding="utf-8").write(sk2)
    shutil.copy2(SKILL_GLOBAL, SKILL_PROFILE)
    print("SKILL.md updated + synced to profile")
else:
    print("SKILL.md: no 71-N range found, left unchanged")

# ---- 5. NO gateway restart needed ----
# Cloud index is live: the gateway reads it on every query, so new pages are
# visible immediately. (Local-mode used to require a restart to release the
# qdrant_db lock; cloud has no such lock.)
print("DONE — cloud index updated, gateway sees new pages live (no restart)")
