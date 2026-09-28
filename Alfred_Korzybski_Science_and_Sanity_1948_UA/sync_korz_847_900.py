#!/usr/bin/env python3
"""Delete English idx for 847-900 and upsert the Ukrainian one (incremental)."""
import sys, os

HERE = r"C:\Users\diredix\Downloads\korz_71_84_hq"
sys.path.insert(0, HERE)

# do NOT set QDRANT_API_KEY here — korz_qdrant._load_key() reads from .env files
# ensure sania .env or local .env contains: QDRANT_API_KEY=<actual_key>
from korz_qdrant import get_client, COLL, upsert_incremental, make_filter
from qdrant_client import models

URL = "https://71e4ac02-a9fb-47ed-94ed-14fc36b0d98e.europe-west3-0.gcp.cloud.qdrant.io:6333"
print(f"Connecting to Qdrant: {URL}")

client = get_client()
print(f"Collection: {COLL}")

# 1. Delete English points for pages == "847-900"
filt = make_filter(pages="847-900")
deleted = 0
offset = None
while True:
    res = client.scroll(COLL, scroll_filter=filt, limit=100, with_payload=False, with_vectors=False, offset=offset)
    pts, offset = res
    if not pts:
        break
    ids = [p.id for p in pts]
    client.delete(COLL, points_selector=models.PointIdsList(points=ids))
    deleted += len(ids)
    print(f"  Deleted {len(ids)} English points (total so far: {deleted})")
    if offset is None:
        break
print(f"Total English points deleted: {deleted}")

# 2. Upsert Ukrainian file (incremental)
print("Running incremental upsert for Ukrainian file(s)...")
upsert_incremental()
print("Done.")
