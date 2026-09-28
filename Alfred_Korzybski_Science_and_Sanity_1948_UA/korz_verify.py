#!/usr/bin/env python3
"""Verify a quoted chunk actually exists (verbatim) in the UA translation files.

Usage:
  python korz_verify.py "текст цитати яку видав бот"
Exit 0 + prints the file + page if found, else exit 1 + "NOT FOUND".
The bot runs this AFTER quoting, to grade its own claim (arc-skill style:
grade the prediction against the frame that came back).
"""
import sys, os, re, glob

HERE = os.path.dirname(os.path.abspath(__file__))
SRCS = sorted(glob.glob(os.path.join(HERE, "укр.datalab-output-*.md")))

def norm(s):
    # keep only letters/digits (any language), drop everything else
    return re.sub(r"[^a-zа-яіїєґ0-9]", "", s.lower())

def page_of(text, pos):
    pre = text[:pos]
    idx = pre.rfind("{")
    if idx == -1:
        return "?"
    m = re.search(r"\{(\d+)\}", pre[idx:])
    return m.group(1) if m else "?"

def verify(quote, threshold=0.85):
    q = norm(quote)
    ql = len(q)
    if ql < 10:
        return None
    for src in SRCS:
        text = open(src, encoding="utf-8").read()
        nt = norm(text)
        pos = nt.find(q)
        if pos != -1:
            # map normalized pos back to original text pos
            ratio = pos / max(1, len(nt))
            orig_pos = int(ratio * len(text))
            return (1.0, os.path.basename(src), page_of(text, orig_pos))
    return None

if __name__ == "__main__":
    q = " ".join(sys.argv[1:])
    if not q.strip():
        print(__doc__); sys.exit(2)
    res = verify(q)
    if res and res[0] >= 0.85:
        print(f"VERIFIED ratio={res[0]:.2f} file={res[1]} page={res[2]}")
        sys.exit(0)
    else:
        print("NOT FOUND — quote not verbatim in index; re-search or say 'outside my pages'")
        sys.exit(1)
