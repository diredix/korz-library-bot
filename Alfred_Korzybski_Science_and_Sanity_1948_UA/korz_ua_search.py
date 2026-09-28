#!/usr/bin/env python3
"""Korzybsky UKRAINIAN translation index (pages 71-102).

Searches the verbatim UA translation and returns matching lines so the korz
agent can quote the book in Ukrainian with context.

Usage:
  python korz_ua_search.py "запит словами"
  python korz_ua_search.py --raw          # dump whole text
"""
import sys, os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
# all UA translation .md files (pages 71-84, 85-102, ...)
SRCS = sorted(glob.glob(os.path.join(HERE, "укр.datalab-output-korz_*.md")))

def load():
    parts = []
    for s in SRCS:
        parts.append(open(s, encoding="utf-8").read())
    return "\n".join(parts)

def search(q):
    text = load()
    terms = [t for t in re.findall(r"\w+", q.lower()) if len(t) > 2]
    out = []
    if not terms:
        return out
    for i, line in enumerate(text.splitlines()):
        ll = line.lower()
        hits = sum(ll.count(t) for t in terms)
        if hits:
            out.append((hits, i, line.strip()))
    out.sort(key=lambda x: -x[0])
    return out

if __name__ == "__main__":
    if "--raw" in sys.argv:
        print(load()); sys.exit(0)
    q = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
    if not q:
        print("usage: python korz_ua_search.py \"запит\""); sys.exit(1)
    res = search(q)
    print(f"=== UA query: {q!r} ({len(res)} line matches) ===")
    for hits, i, line in res[:25]:
        print(f"[L{i}] {line}")
