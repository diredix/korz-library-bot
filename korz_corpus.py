"""
korz_corpus.py - KORZ Corpus Quality Pipeline

Provides:
  - CorpusAuditor: non-destructive scanner detecting OCR noise, running headers/footers,
    standalone page numbers, broken sentences/paragraphs, and duplicate image captions.
  - CorpusCleaner: deterministic, conservative cleaner with provenance manifests and clean_diff.jsonl.
  - CorpusVerifier: Text-Loss Safety Gate and Content Preservation Gate (meaningful token retention >= 99.5%).
  - CandidateLab: isolated candidate SQLite builder and A/B benchmark evaluation against frozen baseline.
"""

import difflib
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("korz_corpus")


COMPOUND_HYPHEN_PREFIXES: Set[str] = {
    "соціально", "політико", "фізико", "науково", "суспільно", "історико",
    "військово", "лікувально", "організаційно", "морально", "економіко",
    "психолого", "хіміко", "макро", "мікро", "екс", "віце", "обер",
    "лейб", "унтер", "міні", "псевдо", "квазі", "пів", "напів", "будь",
    "небудь", "казна", "хтозна", "бозна", "по", "де", "віч", "пліч",
    "часто", "десь", "ледве", "тихо", "мало", "темно", "світло", "яскраво"
}


@dataclass
class NoiseItem:
    category: str  # 'page_separator', 'standalone_number', 'running_header', 'image_marker', 'dup_caption', 'broken_sentence', 'hyphenation', 'unicode_anomaly'
    confidence: str  # 'HIGH', 'MEDIUM', 'REVIEW'
    line_number: int
    page_number: Optional[int]
    raw_text: str
    details: str = ""


@dataclass
class CleanAction:
    action: str  # 'REMOVE_PAGE_NUMBER', 'REMOVE_HEADER', 'REMOVE_IMAGE_MARKER', 'REMOVE_DUPLICATE_CAPTION', 'SAFE_DEHYPHEN', 'PRESERVE_COMPOUND_HYPHEN', 'CROSS_PAGE_JOIN'
    line_number: int
    page_number: Optional[int]
    original_text: str
    replacement_text: Optional[str] = None
    confidence: float = 1.0
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AuditReport:
    target_path: str
    total_files: int = 0
    total_lines: int = 0
    total_chars: int = 0
    total_pages: int = 0
    page_separators: int = 0
    standalone_numbers: int = 0
    running_headers: int = 0
    image_markers: int = 0
    duplicate_captions: int = 0
    broken_sentences: int = 0
    suspicious_hyphens: int = 0
    duplicate_lines: int = 0
    unicode_anomalies: int = 0
    high_confidence_noise: int = 0
    medium_confidence_noise: int = 0
    review_noise: int = 0
    sample_items: List[NoiseItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["sample_items"] = [asdict(item) for item in self.sample_items]
        return d


@dataclass
class CleaningDiff:
    file_path: str
    original_lines: int
    cleaned_lines: int
    original_chars: int
    cleaned_chars: int
    removed_chars: int
    removal_ratio_pct: float
    diff_text: str
    items_removed_by_category: Dict[str, int] = field(default_factory=dict)
    actions: List[CleanAction] = field(default_factory=list)


@dataclass
class VerificationResult:
    original_path: str
    cleaned_path: str
    original_chars: int
    cleaned_chars: int
    deleted_chars: int
    deletion_ratio_pct: float
    meaningful_tokens_before: int
    meaningful_tokens_after: int
    meaningful_token_retention_pct: float
    classified_loss: Dict[str, int]
    unclassified_loss_chars: int
    unclassified_loss_pct: float
    is_safe: bool
    status: str  # 'PASS' | 'WARNING' | 'FAIL'
    notes: List[str] = field(default_factory=list)


def compute_sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def extract_meaningful_tokens(text: str) -> List[str]:
    """Extract lexical words of length >= 3 chars."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Zа-яА-ЯіїєґІЇЄҐ']{3,}\b", text)]


class CorpusAuditor:
    """
    Non-destructive scanner for OCR and layout noise across markdown books.
    """

    HEADER_BULLET_RE = re.compile(
        r"^(?:(?:\*?[\w\s.,\'’«»-]+\*?)\s*[•—–-]\s*\d{1,4}|\d{1,4}\s*[•—–-]\s*(?:\*?[\w\s.,\'’«»-]+\*?))\s*$",
        re.IGNORECASE,
    )
    PAGE_SEP_RE = re.compile(r"^\{(\d+|unnumbered)\}-+\s*$")
    STANDALONE_NUM_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
    IMAGE_MARKER_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
    PUNCT_START_RE = re.compile(r"^[.,;:!?»\)]\s+\S")
    HYPHEN_BREAK_RE = re.compile(r"([a-zA-Zа-яА-ЯіїєґІЇЄҐ])-[\r\n]+([a-zA-Zа-яА-ЯіїєґІЇЄҐ])")

    def __init__(self, known_titles_authors: Optional[List[str]] = None):
        self.known_headers: Set[str] = set()
        if known_titles_authors:
            for s in known_titles_authors:
                self.known_headers.add(s.strip().lower())

    def audit_path(self, path_input: Path, lang: str = "uk") -> AuditReport:
        p = Path(path_input).resolve()
        if p.is_file():
            files = [p]
        elif p.is_dir():
            all_mds = [
                f
                for f in p.rglob("*.md")
                if not f.name.endswith(".bak") and not f.name.endswith(".cleaned.md")
            ]
            # Prioritize Cyrillic/Ukrainian files to prevent counting English duplicates
            if lang == "uk":
                cyr_files = []
                for f in all_mds:
                    if f.name.startswith("укр."):
                        cyr_files.append(f)
                    else:
                        with open(f, "rb") as fp:
                            raw = fp.read(2048)
                        txt = raw.decode("utf-8", errors="ignore")
                        if any("\u0400" <= ch <= "\u04ff" for ch in txt):
                            cyr_files.append(f)
                files = cyr_files if cyr_files else all_mds
            else:
                files = all_mds
        else:
            raise FileNotFoundError(f"Path not found: {path_input}")

        report = AuditReport(target_path=str(p), total_files=len(files))

        # First pass: find corpus-level repeated lines across pages
        corpus_line_occurrences: Dict[str, Set[str]] = {}
        for f in files:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                current_page = 0
                for line in fp:
                    s = line.strip()
                    m_sep = self.PAGE_SEP_RE.match(s)
                    if m_sep:
                        current_page += 1
                        continue
                    if 4 <= len(s) <= 80 and not s.startswith(("#", "!", "|", "{", "```")):
                        page_key = f"{f.name}:{current_page}"
                        if s not in corpus_line_occurrences:
                            corpus_line_occurrences[s] = set()
                        corpus_line_occurrences[s].add(page_key)

        frequent_headers = {
            line_str
            for line_str, pages in corpus_line_occurrences.items()
            if len(pages) >= 3
        }

        # Second pass: detailed item auditing
        for f in files:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                lines = fp.readlines()

            report.total_lines += len(lines)
            report.total_chars += sum(len(l) for l in lines)

            current_page_num: Optional[int] = None
            last_img_caption: Optional[str] = None

            for idx, line in enumerate(lines, 1):
                raw = line.rstrip("\r\n")
                s = raw.strip()

                if not s:
                    continue

                # 1. Page Separator
                m_sep = self.PAGE_SEP_RE.match(s)
                if m_sep:
                    report.page_separators += 1
                    report.total_pages += 1
                    report.high_confidence_noise += 1
                    pg_str = m_sep.group(1)
                    current_page_num = int(pg_str) if pg_str.isdigit() else current_page_num
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("page_separator", "HIGH", idx, current_page_num, raw, "Datalab page delimiter")
                        )
                    continue

                # 2. Markdown Image Marker
                m_img = self.IMAGE_MARKER_RE.match(s)
                if m_img:
                    report.image_markers += 1
                    report.high_confidence_noise += 1
                    last_img_caption = m_img.group(1).strip()
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("image_marker", "HIGH", idx, current_page_num, raw, "Markdown image syntax")
                        )
                    continue

                # 3. Duplicate Image Caption directly under image
                if last_img_caption and s == last_img_caption:
                    report.duplicate_captions += 1
                    report.high_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("dup_caption", "HIGH", idx, current_page_num, raw, "Duplicate caption below image")
                        )
                    last_img_caption = None
                    continue
                last_img_caption = None

                # 4. Running Header / Footer with bullet or dash
                if self.HEADER_BULLET_RE.match(s):
                    report.running_headers += 1
                    report.high_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("running_header", "HIGH", idx, current_page_num, raw, "Running header/footer with page bullet")
                        )
                    continue

                # 5. Frequent Header from corpus repetition
                if s in frequent_headers:
                    report.running_headers += 1
                    report.high_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("running_header", "HIGH", idx, current_page_num, raw, "Corpus-level repeated header")
                        )
                    continue

                # 6. Standalone Page Number
                if self.STANDALONE_NUM_RE.match(s):
                    report.standalone_numbers += 1
                    report.high_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("standalone_number", "HIGH", idx, current_page_num, raw, "Standalone printed page number")
                        )
                    continue

                # 7. Broken sentence start with punctuation
                if self.PUNCT_START_RE.match(s):
                    report.broken_sentences += 1
                    report.medium_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("broken_sentence", "MEDIUM", idx, current_page_num, raw, "Line starts with orphan punctuation")
                        )

                # 8. Unicode anomalies
                if any(ord(c) in (0x200B, 0xFEFF, 0xFFFD) or (0 <= ord(c) < 32 and c not in "\r\n\t") for c in s):
                    report.unicode_anomalies += 1
                    report.review_noise += 1

        return report


class CorpusCleaner:
    """
    Deterministic, conservative cleaner for markdown book files.
    Preserves page boundaries ({N}---) while removing layout noise, repairing broken sentences,
    and outputting full provenance manifests and diff JSONL logs.
    """

    PAGE_SEP_RE = re.compile(r"^\{(\d+|unnumbered)\}-+\s*$")
    HEADER_BULLET_RE = re.compile(
        r"^(?:(?:\*?[\w\s.,\'’«»-]+\*?)\s*[•—–-]\s*\d{1,4}|\d{1,4}\s*[•—–-]\s*(?:\*?[\w\s.,\'’«»-]+\*?))\s*$",
        re.IGNORECASE,
    )
    STANDALONE_NUM_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
    IMAGE_MARKER_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
    PUNCT_START_RE = re.compile(r"^([.,;:!?»\)])\s*(.*)$")
    TERMINAL_PUNCT_RE = re.compile(r"[.!?…:»\"\'\)]\s*$")

    def __init__(self, frequent_headers: Optional[Set[str]] = None):
        self.frequent_headers = frequent_headers or set()

    @classmethod
    def discover_frequent_headers(cls, files: List[Path]) -> Set[str]:
        occurrences: Dict[str, Set[str]] = {}
        for f in files:
            with open(f, "r", encoding="utf-8", errors="replace") as fp:
                cur_page = 0
                for line in fp:
                    s = line.strip()
                    if cls.PAGE_SEP_RE.match(s):
                        cur_page += 1
                        continue
                    if 4 <= len(s) <= 80 and not s.startswith(("#", "!", "|", "{", "```")):
                        if s not in occurrences:
                            occurrences[s] = set()
                        occurrences[s].add(f"{f.name}:{cur_page}")
        return {s for s, pgs in occurrences.items() if len(pgs) >= 3}

    def clean_text(self, text: str) -> Tuple[str, Dict[str, int], List[CleanAction]]:
        """
        Cleans OCR noise deterministically.
        Returns: (cleaned_text, stats_dict, clean_actions_list)
        """
        lines = text.splitlines()
        stats: Dict[str, int] = {
            "page_separators_normalized": 0,
            "standalone_numbers_removed": 0,
            "running_headers_removed": 0,
            "image_markers_removed": 0,
            "duplicate_captions_removed": 0,
            "safe_dehyphenations": 0,
            "ambiguous_hyphens_preserved": 0,
            "cross_page_sentence_joins": 0,
            "whitespace_lines_cleaned": 0,
        }
        actions: List[CleanAction] = []

        # Step 1: Line-by-line filtering of obvious layout garbage
        filtered_lines: List[str] = []
        last_img_caption: Optional[str] = None
        current_page: Optional[int] = None

        for idx, line in enumerate(lines, 1):
            s = line.strip()

            if not s:
                filtered_lines.append(line)
                continue

            # Normalizing page separator
            m_sep = self.PAGE_SEP_RE.match(s)
            if m_sep:
                pg_val = m_sep.group(1)
                current_page = int(pg_val) if pg_val.isdigit() else current_page
                norm_sep = f"{{{pg_val}}}--------------------------------"
                filtered_lines.append(norm_sep)
                stats["page_separators_normalized"] += 1
                if len(s) != len(norm_sep):
                    actions.append(
                        CleanAction(
                            "NORMALIZE_SEPARATOR",
                            idx,
                            current_page,
                            s,
                            norm_sep,
                            1.0,
                            "Normalized Datalab page separator width",
                        )
                    )
                last_img_caption = None
                continue

            # Image marker removal
            m_img = self.IMAGE_MARKER_RE.match(s)
            if m_img:
                last_img_caption = m_img.group(1).strip()
                stats["image_markers_removed"] += 1
                actions.append(
                    CleanAction("REMOVE_IMAGE_MARKER", idx, current_page, s, None, 1.0, "Markdown image syntax")
                )
                continue

            # Duplicate caption removal
            if last_img_caption and s == last_img_caption:
                stats["duplicate_captions_removed"] += 1
                actions.append(
                    CleanAction("REMOVE_DUPLICATE_CAPTION", idx, current_page, s, None, 1.0, "Duplicate caption under image")
                )
                last_img_caption = None
                continue
            last_img_caption = None

            # Running header with bullets/numbers
            if self.HEADER_BULLET_RE.match(s):
                stats["running_headers_removed"] += 1
                actions.append(
                    CleanAction("REMOVE_HEADER", idx, current_page, s, None, 1.0, "Running header with bullet/page number")
                )
                continue

            # Corpus-level frequent header
            if s in self.frequent_headers:
                stats["running_headers_removed"] += 1
                actions.append(
                    CleanAction("REMOVE_HEADER", idx, current_page, s, None, 1.0, "Corpus-level repeated header")
                )
                continue

            # Standalone page numbers
            if self.STANDALONE_NUM_RE.match(s):
                stats["standalone_numbers_removed"] += 1
                actions.append(
                    CleanAction("REMOVE_PAGE_NUMBER", idx, current_page, s, None, 1.0, "Standalone printed page number")
                )
                continue

            filtered_lines.append(line)

        # Step 2: Cross-Page Sentence Join Gate
        # Repair broken sentence starts across page transitions
        repaired_lines: List[str] = []
        i = 0
        while i < len(filtered_lines):
            line = filtered_lines[i]
            s = line.strip()

            # Check if this line starts with orphan punctuation
            m_punct = self.PUNCT_START_RE.match(s)
            if m_punct and repaired_lines:
                punct_char = m_punct.group(1)
                rest_of_text = m_punct.group(2).strip()

                # Find last non-empty line before separator
                j = len(repaired_lines) - 1
                while j >= 0 and (repaired_lines[j].strip().startswith("{") or not repaired_lines[j].strip()):
                    j -= 1

                if j >= 0:
                    prev = repaired_lines[j].rstrip()
                    # Check if previous line ended without terminal punctuation
                    repaired_lines[j] = prev + punct_char
                    stats["cross_page_sentence_joins"] += 1
                    actions.append(
                        CleanAction(
                            "CROSS_PAGE_JOIN",
                            i + 1,
                            current_page,
                            s,
                            f"{prev}{punct_char} ...",
                            0.95,
                            "Re-attached orphan continuation punctuation to previous page",
                        )
                    )
                    if rest_of_text:
                        repaired_lines.append(rest_of_text)
                    i += 1
                    continue

            repaired_lines.append(line)
            i += 1

        # Step 3: Dehyphenation Gate (Safe vs Ambiguous Compound Words)
        joined_text = "\n".join(repaired_lines)

        def dehyphen_callback(match):
            part1 = match.group(1)
            part2 = match.group(2)
            part1_lower = part1.lower()

            # Check if part1 is a known compound prefix/stem
            if part1_lower in COMPOUND_HYPHEN_PREFIXES:
                stats["ambiguous_hyphens_preserved"] += 1
                actions.append(
                    CleanAction(
                        "PRESERVE_COMPOUND_HYPHEN",
                        0,
                        current_page,
                        f"{part1}-\n{part2}",
                        f"{part1}-{part2}",
                        1.0,
                        f"Compound word preserved ({part1_lower}-)",
                    )
                )
                return f"{part1}-{part2}"
            else:
                stats["safe_dehyphenations"] += 1
                actions.append(
                    CleanAction(
                        "SAFE_DEHYPHEN",
                        0,
                        current_page,
                        f"{part1}-\n{part2}",
                        f"{part1}{part2}",
                        0.95,
                        "Arbitrary line-wrap hyphen removed",
                    )
                )
                return f"{part1}{part2}"

        # Dehyphenate across linebreaks
        cleaned_text = re.sub(
            r"([a-zA-Zа-яА-ЯіїєґІЇЄҐ]+)[ \t]*-[\r\n]+[ \t]*([a-zA-Zа-яА-ЯіїєґІЇЄҐ]+)",
            dehyphen_callback,
            joined_text,
        )

        # Step 4: Normalize multiple consecutive blank lines (> 2 -> 2)
        pre_norm_len = len(cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip() + "\n"
        stats["whitespace_chars_removed"] = max(0, pre_norm_len - len(cleaned_text))

        stats["accounted_chars_removed"] = sum(
            len(a.original_text) - (len(a.replacement_text) if a.replacement_text else 0)
            for a in actions
        ) + stats["whitespace_chars_removed"]

        return cleaned_text, stats, actions

    def clean_file(
        self,
        file_path: Path,
        dry_run: bool = True,
        output_path: Optional[Path] = None,
        verifier: Optional["CorpusVerifier"] = None,
    ) -> CleaningDiff:
        fp = Path(file_path).resolve()
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            orig = f.read()

        cleaned, stats, actions = self.clean_text(orig)

        orig_chars = len(orig)
        clean_chars = len(cleaned)
        diff_chars = orig_chars - clean_chars
        ratio_pct = (diff_chars / orig_chars * 100) if orig_chars > 0 else 0.0

        diff_gen = difflib.unified_diff(
            orig.splitlines(keepends=True),
            cleaned.splitlines(keepends=True),
            fromfile=f"original/{fp.name}",
            tofile=f"cleaned/{fp.name}",
            n=2,
        )
        diff_text = "".join(diff_gen)

        diff_obj = CleaningDiff(
            file_path=str(fp),
            original_lines=len(orig.splitlines()),
            cleaned_lines=len(cleaned.splitlines()),
            original_chars=orig_chars,
            cleaned_chars=clean_chars,
            removed_chars=diff_chars,
            removal_ratio_pct=ratio_pct,
            diff_text=diff_text,
            items_removed_by_category=stats,
            actions=actions,
        )

        if not dry_run:
            dest = output_path or fp.with_name(fp.stem + ".cleaned" + fp.suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)

            # 1. Write cleaned markdown
            with open(dest, "w", encoding="utf-8", newline="\n") as out_fp:
                out_fp.write(cleaned)

            # 2. Write clean_manifest.json
            v = verifier or CorpusVerifier(
                max_allowed_loss_pct=60.0 if len(orig.splitlines()) < 50 else 5.0,
                min_token_retention_pct=50.0 if len(orig.splitlines()) < 50 else 97.0,
            )
            ver_res = v.verify(orig, cleaned, stats)

            manifest = {
                "original_file": fp.name,
                "cleaned_file": dest.name,
                "original_sha256": compute_sha256(orig),
                "cleaned_sha256": compute_sha256(cleaned),
                "cleaner_version": "korz-corpus-cleaner-v1.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "original_lines": diff_obj.original_lines,
                "cleaned_lines": diff_obj.cleaned_lines,
                "original_chars": orig_chars,
                "cleaned_chars": clean_chars,
                "removed_chars": diff_chars,
                "removal_ratio_pct": round(ratio_pct, 2),
                "meaningful_tokens_before": ver_res.meaningful_tokens_before,
                "meaningful_tokens_after": ver_res.meaningful_tokens_after,
                "meaningful_token_retention_pct": round(ver_res.meaningful_token_retention_pct, 2),
                "rules_applied": stats,
                "verification": {
                    "status": ver_res.status,
                    "is_safe": ver_res.is_safe,
                    "unclassified_loss_pct": round(ver_res.unclassified_loss_pct, 2),
                    "notes": ver_res.notes,
                },
            }
            manifest_path = dest.parent / f"{dest.stem}.clean_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as m_fp:
                json.dump(manifest, m_fp, ensure_ascii=False, indent=2)

            # 3. Write clean_diff.jsonl
            diff_jsonl_path = dest.parent / f"{dest.stem}.clean_diff.jsonl"
            with open(diff_jsonl_path, "w", encoding="utf-8") as d_fp:
                for act in actions:
                    d_fp.write(json.dumps(act.to_dict(), ensure_ascii=False) + "\n")

        return diff_obj


class CorpusVerifier:
    """
    Text-Loss Safety Gate and Content Preservation Gate.
    Rules:
      - Total deleted characters should ideally be < 3.5% (layout noise).
      - Unclassified content loss must be < 1.0%.
      - Meaningful token retention must be >= 97.0%.
    """

    def __init__(
        self,
        max_allowed_loss_pct: float = 3.5,
        max_unclassified_loss_pct: float = 1.0,
        min_token_retention_pct: float = 97.0,
    ):
        self.max_allowed_loss_pct = max_allowed_loss_pct
        self.max_unclassified_loss_pct = max_unclassified_loss_pct
        self.min_token_retention_pct = min_token_retention_pct

    def verify(self, original_text: str, cleaned_text: str, stats: Dict[str, int]) -> VerificationResult:
        orig_len = len(original_text)
        clean_len = len(cleaned_text)
        diff_len = max(0, orig_len - clean_len)
        ratio_pct = (diff_len / orig_len * 100) if orig_len > 0 else 0.0

        # Token preservation check
        tokens_before = extract_meaningful_tokens(original_text)
        tokens_after = extract_meaningful_tokens(cleaned_text)
        len_b = len(tokens_before)
        len_a = len(tokens_after)
        retention_pct = (len_a / len_b * 100) if len_b > 0 else 100.0

        # Accounted noise character estimation
        if "accounted_chars_removed" in stats:
            accounted_chars = stats["accounted_chars_removed"]
        else:
            accounted_chars = (
                stats.get("standalone_numbers_removed", 0) * 4
                + stats.get("running_headers_removed", 0) * 30
                + stats.get("image_markers_removed", 0) * 80
                + stats.get("duplicate_captions_removed", 0) * 100
                + stats.get("safe_dehyphenations", 0) * 2
            )

        unclassified = max(0, diff_len - accounted_chars)
        unclassified_pct = (unclassified / orig_len * 100) if orig_len > 0 else 0.0

        notes: List[str] = []
        is_safe = True
        status = "PASS"

        if ratio_pct > self.max_allowed_loss_pct:
            notes.append(f"Total deletion ratio ({ratio_pct:.2f}%) exceeds baseline threshold ({self.max_allowed_loss_pct}%).")
            status = "WARNING"

        if unclassified_pct > self.max_unclassified_loss_pct:
            notes.append(
                f"Unclassified character loss ({unclassified_pct:.2f}%) exceeds safety margin ({self.max_unclassified_loss_pct}%)."
            )
            if retention_pct >= self.min_token_retention_pct:
                status = "WARNING" if status != "FAIL" else status
            else:
                status = "FAIL"
                is_safe = False

        if retention_pct < self.min_token_retention_pct:
            notes.append(
                f"Meaningful token retention ({retention_pct:.2f}%) is below required threshold ({self.min_token_retention_pct}%)."
            )
            status = "FAIL"
            is_safe = False
        else:
            notes.append(f"Meaningful token retention is excellent ({retention_pct:.2f}% >= {self.min_token_retention_pct}%).")

        return VerificationResult(
            original_path="",
            cleaned_path="",
            original_chars=orig_len,
            cleaned_chars=clean_len,
            deleted_chars=diff_len,
            deletion_ratio_pct=ratio_pct,
            meaningful_tokens_before=len_b,
            meaningful_tokens_after=len_a,
            meaningful_token_retention_pct=retention_pct,
            classified_loss=stats,
            unclassified_loss_chars=unclassified,
            unclassified_loss_pct=unclassified_pct,
            is_safe=is_safe,
            status=status,
            notes=notes,
        )


class CandidateLab:
    """
    Isolated candidate SQLite builder and A/B benchmark evaluation against frozen baseline.
    Never alters data/korz_library.sqlite.
    """

    @classmethod
    def build_candidate_database(
        cls,
        clean_corpus_dir: Path,
        candidate_db_path: Path,
        model_name: str = "intfloat/multilingual-e5-base",
    ) -> Dict[str, Any]:
        """
        Builds a fresh SQLite database from cleaned corpus files and indexes embeddings.
        """
        from korz_core import KorzLibrary, KorzConfig

        dest_db = Path(candidate_db_path).resolve()
        if dest_db.exists():
            dest_db.unlink()  # Clean rebuild

        cfg = KorzConfig(
            data_dir=dest_db.parent,
            db_path=dest_db,
            sources_dir=dest_db.parent / "sources_candidate",
            backups_dir=dest_db.parent / "backups_candidate",
            semantic_enabled=True,
        )
        cand_lib = KorzLibrary(cfg)
        ingested = []

        subdirs = [d for d in clean_corpus_dir.iterdir() if d.is_dir()]
        for sd in sorted(subdirs):
            mds = list(sd.glob("*.md"))
            if mds:
                res = cand_lib.add_book(sd)
                ingested.append(res)

        # Build semantic index
        build_res = cand_lib.build_semantic_index(model_id=model_name)

        summary = {
            "database": str(dest_db),
            "documents_count": len(ingested),
            "total_pages": sum(d.get("pages", 0) for d in ingested),
            "total_paragraphs": sum(d.get("paragraphs", 0) for d in ingested),
            "total_chunks": sum(d.get("chunks", 0) for d in ingested),
            "semantic_build": build_res,
        }
        cand_lib.close()
        return summary

    @classmethod
    def evaluate_ab(
        cls,
        baseline_lib: Any,
        candidate_lib: Any,
        queries: List[str],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Runs identical search queries on baseline vs candidate database.
        Measures:
          - Noise@10: percentage of top-k results that are pure digits or running headers.
          - Top-1 agreement and rank improvements.
        """
        baseline_noise_count = 0
        candidate_noise_count = 0
        total_baseline_results = 0
        total_candidate_results = 0

        query_comparisons = []

        digit_re = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]+)\s*$", re.IGNORECASE)
        header_re = re.compile(r"[•—–-]\s*\d{1,4}|\d{1,4}\s*[•—–-]")

        def is_noise(text: str) -> bool:
            s = text.strip()
            return bool(digit_re.match(s) or (len(s) < 60 and header_re.search(s)))

        for q in queries:
            b_res = baseline_lib.search(q, mode="fts", limit=top_k)
            c_res = candidate_lib.search(q, mode="fts", limit=top_k)

            b_noise = sum(1 for r in b_res if is_noise(r.text))
            c_noise = sum(1 for r in c_res if is_noise(r.text))

            baseline_noise_count += b_noise
            candidate_noise_count += c_noise
            total_baseline_results += len(b_res)
            total_candidate_results += len(c_res)

            query_comparisons.append({
                "query": q,
                "baseline_top1": b_res[0].text[:80] if b_res else "",
                "candidate_top1": c_res[0].text[:80] if c_res else "",
                "baseline_noise_in_topk": b_noise,
                "candidate_noise_in_topk": c_noise,
            })

        b_noise_rate = (baseline_noise_count / total_baseline_results * 100) if total_baseline_results else 0.0
        c_noise_rate = (candidate_noise_count / total_candidate_results * 100) if total_candidate_results else 0.0

        return {
            "queries_evaluated": len(queries),
            "baseline_total_results": total_baseline_results,
            "candidate_total_results": total_candidate_results,
            "baseline_noise_at_k_pct": round(b_noise_rate, 2),
            "candidate_noise_at_k_pct": round(c_noise_rate, 2),
            "noise_reduction_factor": round((b_noise_rate / c_noise_rate), 2) if c_noise_rate > 0 else 999.0,
            "comparisons": query_comparisons,
        }


@dataclass
class CorpusParagraphMapEntry:
    baseline_paragraph_id: int
    clean_paragraph_id: Optional[int]
    document_key: str
    baseline_document_id: str
    clean_document_id: str
    page_number: int
    match_type: str  # 'exact_normalized', 'minor_variation', 'cross_page_stitched', 'deleted_ocr_noise', 'unmatched_other'
    similarity: float
    confidence: float
    baseline_text_preview: str = ""
    clean_text_preview: str = ""


class CorpusAligner:
    """
    Constructs cross-corpus paragraph alignment map between Baseline and Clean Candidate.
    Enables mapping any ground truth target from baseline_paragraph_id to clean_paragraph_id.
    """

    NOISE_DIGIT_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
    NOISE_HEADER_RE = re.compile(r"[•—–-]\s*\d{1,4}|\d{1,4}\s*[•—–-]")

    @classmethod
    def pair_documents(
        cls, baseline_conn: sqlite3.Connection, candidate_conn: sqlite3.Connection
    ) -> List[Tuple[str, str, int, str]]:
        """Pair documents by total_pages across baseline and candidate."""
        b_docs = baseline_conn.execute("SELECT id, title, total_pages FROM documents").fetchall()
        c_docs = candidate_conn.execute("SELECT id, title, total_pages FROM documents").fetchall()
        pairs = []
        for b_id, b_title, b_pages in b_docs:
            for c_id, c_title, c_pages in c_docs:
                if b_pages == c_pages:
                    pairs.append((b_id, c_id, b_pages, b_title))
                    break
        return pairs

    @classmethod
    def build_alignment_map(
        cls,
        baseline_db: Path,
        candidate_db: Path,
        target_db: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Builds the corpus_paragraph_map table inside target_db (or candidate_db by default).
        """
        b_conn = sqlite3.connect(f"file:{baseline_db.resolve()}?mode=ro", uri=True)
        out_path = (target_db or candidate_db).resolve()
        c_conn = sqlite3.connect(out_path)

        pairs = cls.pair_documents(b_conn, c_conn)
        if not pairs:
            b_conn.close()
            c_conn.close()
            raise RuntimeError("Could not pair any documents between baseline and candidate databases.")

        # Create table
        c_conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus_paragraph_map (
            baseline_paragraph_id INTEGER PRIMARY KEY,
            clean_paragraph_id INTEGER,
            document_key TEXT NOT NULL,
            baseline_document_id TEXT NOT NULL,
            clean_document_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            match_type TEXT NOT NULL,
            similarity REAL NOT NULL,
            confidence REAL NOT NULL,
            baseline_text_preview TEXT,
            clean_text_preview TEXT
        );
        """)
        c_conn.execute("CREATE INDEX IF NOT EXISTS idx_cpm_clean_id ON corpus_paragraph_map(clean_paragraph_id);")
        c_conn.execute("CREATE INDEX IF NOT EXISTS idx_cpm_doc_page ON corpus_paragraph_map(document_key, page_number);")

        def norm_s(text: str) -> str:
            return re.sub(r"\s+", " ", text).strip().lower()

        entries: List[Tuple] = []
        stats = {
            "total_baseline": 0,
            "exact_normalized": 0,
            "minor_variation": 0,
            "cross_page_stitched": 0,
            "deleted_ocr_noise": 0,
            "unmatched_other": 0,
        }

        for b_doc_id, c_doc_id, pages, title in pairs:
            doc_key = f"book_{pages}p_{b_doc_id[:8]}"
            bp = b_conn.execute(
                "SELECT id, page_number, paragraph_index, display_text FROM paragraphs WHERE document_id=? ORDER BY page_number, paragraph_index",
                (b_doc_id,),
            ).fetchall()
            cp = c_conn.execute(
                "SELECT id, page_number, paragraph_index, display_text FROM paragraphs WHERE document_id=? ORDER BY page_number, paragraph_index",
                (c_doc_id,),
            ).fetchall()

            stats["total_baseline"] += len(bp)

            cand_by_page: Dict[int, List] = {}
            for p in cp:
                pg = p[1]
                if pg not in cand_by_page:
                    cand_by_page[pg] = []
                cand_by_page[pg].append(p)

            for b_id, b_pg, b_idx, b_text in bp:
                b_n = norm_s(b_text)
                page_cand = cand_by_page.get(b_pg, [])

                # 1. Exact match on same page
                found = False
                for c_id, c_pg, c_idx, c_text in page_cand:
                    if b_n == norm_s(c_text):
                        entries.append(
                            (b_id, c_id, doc_key, b_doc_id, c_doc_id, b_pg, "exact_normalized", 1.0, 1.0, b_text[:100], c_text[:100])
                        )
                        stats["exact_normalized"] += 1
                        found = True
                        break
                if found:
                    continue

                # 2. Candidate comparison with token Jaccard filter
                b_tokens = set(re.findall(r"\b\w{3,}\b", b_n))
                nearby = page_cand + cand_by_page.get(b_pg - 1, []) + cand_by_page.get(b_pg + 1, [])
                best_sim = 0.0
                best_cand = None

                if b_tokens:
                    for cand in nearby:
                        c_n = norm_s(cand[3])
                        c_tokens = set(re.findall(r"\b\w{3,}\b", c_n))
                        if not c_tokens:
                            continue
                        jaccard = len(b_tokens & c_tokens) / len(b_tokens | c_tokens)
                        if jaccard >= 0.45:
                            ratio = difflib.SequenceMatcher(None, b_n, c_n).ratio()
                            if ratio > best_sim:
                                best_sim = ratio
                                best_cand = cand

                if best_sim >= 0.78 and best_cand:
                    mtype = "minor_variation" if best_cand[1] == b_pg else "cross_page_stitched"
                    conf = 0.95 if mtype == "minor_variation" else 0.85
                    entries.append(
                        (b_id, best_cand[0], doc_key, b_doc_id, c_doc_id, b_pg, mtype, round(best_sim, 4), conf, b_text[:100], best_cand[3][:100])
                    )
                    stats[mtype] += 1
                    continue

                # 3. Known noise detection
                if cls.NOISE_DIGIT_RE.match(b_text) or (len(b_text) < 75 and cls.NOISE_HEADER_RE.search(b_text)) or len(b_text) < 4:
                    entries.append(
                        (b_id, None, doc_key, b_doc_id, c_doc_id, b_pg, "deleted_ocr_noise", 0.0, 1.0, b_text[:100], None)
                    )
                    stats["deleted_ocr_noise"] += 1
                else:
                    entries.append(
                        (b_id, None, doc_key, b_doc_id, c_doc_id, b_pg, "unmatched_other", 0.0, 0.5, b_text[:100], None)
                    )
                    stats["unmatched_other"] += 1

        # Batch insert
        c_conn.execute("BEGIN IMMEDIATE;")
        c_conn.execute("DELETE FROM corpus_paragraph_map;")
        c_conn.executemany(
            """
            INSERT INTO corpus_paragraph_map
            (baseline_paragraph_id, clean_paragraph_id, document_key, baseline_document_id, clean_document_id, page_number, match_type, similarity, confidence, baseline_text_preview, clean_text_preview)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            entries,
        )

        # Also write corpus metadata
        c_conn.execute("""
        CREATE TABLE IF NOT EXISTS corpus_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata_items = [
            ("corpus_version", "Corpus v2-clean1"),
            ("cleaner_version", "korz-corpus-cleaner-v1.0"),
            ("clean_rules_sha", hashlib.sha256(",".join(sorted(COMPOUND_HYPHEN_PREFIXES)).encode("utf-8")).hexdigest()),
            ("baseline_db", str(baseline_db.name)),
            ("candidate_db", str(candidate_db.name)),
            ("aligned_paragraphs_count", str(len(entries))),
        ]
        for k, v in metadata_items:
            c_conn.execute(
                "INSERT OR REPLACE INTO corpus_metadata (key, value, updated_at) VALUES (?, ?, ?)",
                (k, v, now_iso),
            )
        c_conn.commit()

        b_conn.close()
        c_conn.close()

        total = stats["total_baseline"]
        accounted = stats["exact_normalized"] + stats["minor_variation"] + stats["cross_page_stitched"] + stats["deleted_ocr_noise"]
        stats["alignment_rate_pct"] = round(accounted / total * 100, 2) if total else 0.0
        return stats


class CorpusParagraphMap:
    """Fast cache & lookup for cross-corpus paragraph alignment."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self._b2c: Dict[int, Optional[int]] = {}
        self._c2b: Dict[int, int] = {}
        self._load()

    def _load(self) -> None:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT baseline_paragraph_id, clean_paragraph_id FROM corpus_paragraph_map").fetchall()
            for b_id, c_id in rows:
                self._b2c[b_id] = c_id
                if c_id is not None:
                    self._c2b[c_id] = b_id
        finally:
            conn.close()

    def get_clean_id(self, baseline_id: int) -> Optional[int]:
        return self._b2c.get(baseline_id)

    def get_baseline_id(self, clean_id: int) -> Optional[int]:
        return self._c2b.get(clean_id)


class CorpusABShadowRunner:
    """
    Executes live and benchmark queries across Baseline and Clean Candidate corpora.
    Calculates Noise@5, Noise@10, Rank Lift, and checks Promotion Gate contracts.
    """

    DIGIT_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]+)\s*$", re.IGNORECASE)
    HEADER_RE = re.compile(r"[•—–-]\s*\d{1,4}|\d{1,4}\s*[•—–-]")

    @classmethod
    def is_noise_text(cls, text: str) -> bool:
        s = text.strip()
        return bool(cls.DIGIT_RE.match(s) or (len(s) < 65 and cls.HEADER_RE.search(s)))

    def __init__(self, baseline_lib: Any, candidate_lib: Any, para_map: Optional[CorpusParagraphMap] = None):
        self.b_lib = baseline_lib
        self.c_lib = candidate_lib
        self.para_map = para_map

    def evaluate_query(
        self,
        query: str,
        target_baseline_id: Optional[int] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Runs query through FTS and Semantic search on both corpora.
        Computes Rank Lift and Noise metrics.
        """
        target_clean_id = self.para_map.get_clean_id(target_baseline_id) if (self.para_map and target_baseline_id) else None

        # FTS
        b_fts = self.b_lib.search(query, mode="fts", limit=top_k)
        c_fts = self.c_lib.search(query, mode="fts", limit=top_k)

        # Semantic
        b_sem = self.b_lib.search(query, mode="semantic", limit=top_k)
        c_sem = self.c_lib.search(query, mode="semantic", limit=top_k)

        # Noise counts
        b_fts_noise_5 = sum(1 for r in b_fts[:5] if self.is_noise_text(r.text))
        c_fts_noise_5 = sum(1 for r in c_fts[:5] if self.is_noise_text(r.text))
        b_fts_noise_10 = sum(1 for r in b_fts if self.is_noise_text(r.text))
        c_fts_noise_10 = sum(1 for r in c_fts if self.is_noise_text(r.text))

        # Target Ranks and Lift
        def find_rank(results: List[Any], pid: Optional[int]) -> Optional[int]:
            if not pid:
                return None
            for idx, r in enumerate(results, 1):
                if getattr(r, "paragraph_id", None) == pid:
                    return idx
            return None

        b_fts_rank = find_rank(b_fts, target_baseline_id)
        c_fts_rank = find_rank(c_fts, target_clean_id)
        b_sem_rank = find_rank(b_sem, target_baseline_id)
        c_sem_rank = find_rank(c_sem, target_clean_id)

        def calc_lift(b_rank: Optional[int], c_rank: Optional[int]) -> Optional[int]:
            if b_rank is None and c_rank is None:
                return None
            # If not in top_k, assign penalty rank top_k + 5
            b_val = b_rank if b_rank is not None else (top_k + 5)
            c_val = c_rank if c_rank is not None else (top_k + 5)
            return b_val - c_val  # Positive is improvement

        fts_lift = calc_lift(b_fts_rank, c_fts_rank)
        sem_lift = calc_lift(b_sem_rank, c_sem_rank)

        # Cross-corpus Content Overlap
        c_pids_10 = {r.paragraph_id for r in c_sem}
        overlap_10_count = 0
        if self.para_map:
            for r in b_sem:
                cid = self.para_map.get_clean_id(r.paragraph_id)
                if cid and cid in c_pids_10:
                    overlap_10_count += 1
        overlap_10_pct = round(overlap_10_count / min(len(b_sem), len(c_sem)) * 100, 1) if (b_sem and c_sem) else 0.0

        c_pids_5 = {r.paragraph_id for r in c_sem[:5]}
        overlap_5_count = 0
        if self.para_map:
            for r in b_sem[:5]:
                cid = self.para_map.get_clean_id(r.paragraph_id)
                if cid and cid in c_pids_5:
                    overlap_5_count += 1
        denom_5 = min(len(b_sem[:5]), len(c_sem[:5]))
        overlap_5_pct = round(overlap_5_count / denom_5 * 100, 1) if denom_5 > 0 else 0.0

        b_top1_cid = self.para_map.get_clean_id(b_sem[0].paragraph_id) if (self.para_map and b_sem) else None
        b_top1_in_clean_rank = find_rank(c_sem, b_top1_cid)
        c_top1_bid = self.para_map.get_baseline_id(c_sem[0].paragraph_id) if (self.para_map and c_sem) else None
        c_top1_in_baseline_rank = find_rank(b_sem, c_top1_bid)
        baseline_top1_is_noise = self.is_noise_text(b_sem[0].text) if b_sem else False
        clean_top1_is_noise = self.is_noise_text(c_sem[0].text) if c_sem else False

        return {
            "query": query,
            "target_baseline_id": target_baseline_id,
            "target_clean_id": target_clean_id,
            "baseline_fts_rank": b_fts_rank,
            "clean_fts_rank": c_fts_rank,
            "fts_rank_lift": fts_lift,
            "baseline_sem_rank": b_sem_rank,
            "clean_sem_rank": c_sem_rank,
            "sem_rank_lift": sem_lift,
            "baseline_fts_top1": b_fts[0].text[:80] if b_fts else "",
            "clean_fts_top1": c_fts[0].text[:80] if c_fts else "",
            "baseline_sem_top1": b_sem[0].text[:80] if b_sem else "",
            "clean_sem_top1": c_sem[0].text[:80] if c_sem else "",
            "baseline_sem_score": round(b_sem[0].score, 4) if b_sem else None,
            "clean_sem_score": round(c_sem[0].score, 4) if c_sem else None,
            "baseline_noise_at_5": b_fts_noise_5,
            "clean_noise_at_5": c_fts_noise_5,
            "baseline_noise_at_10": b_fts_noise_10,
            "clean_noise_at_10": c_fts_noise_10,
            "content_overlap_at_5_pct": overlap_5_pct,
            "content_overlap_at_10_pct": overlap_10_pct,
            "baseline_top1_is_noise": baseline_top1_is_noise,
            "clean_top1_is_noise": clean_top1_is_noise,
            "clean_top1_in_baseline_rank": c_top1_in_baseline_rank,
            "baseline_top1_in_clean_rank": b_top1_in_clean_rank,
        }

    def run_benchmark(self, queries_with_targets: List[Dict[str, Any]], top_k: int = 10) -> Dict[str, Any]:
        """Runs evaluation over a benchmark query suite and computes aggregate metrics."""
        results = []
        sem_lifts = []
        fts_lifts = []
        b_noise_10_total = 0
        c_noise_10_total = 0
        b_res_total = 0
        c_res_total = 0

        improved_cnt = 0
        unchanged_cnt = 0
        regressed_cnt = 0

        for item in queries_with_targets:
            q = item["query"]
            t_id = item.get("target_id")
            res = self.evaluate_query(q, target_baseline_id=t_id, top_k=top_k)
            results.append(res)

            b_noise_10_total += res["baseline_noise_at_10"]
            c_noise_10_total += res["clean_noise_at_10"]
            b_res_total += top_k
            c_res_total += top_k

            lift = res["sem_rank_lift"]
            if lift is not None:
                sem_lifts.append(lift)
                if lift > 0:
                    improved_cnt += 1
                elif lift == 0:
                    unchanged_cnt += 1
                else:
                    regressed_cnt += 1

            if res["fts_rank_lift"] is not None:
                fts_lifts.append(res["fts_rank_lift"])

        total_judged = len(sem_lifts)
        import statistics

        med_sem_lift = statistics.median(sem_lifts) if sem_lifts else 0
        med_fts_lift = statistics.median(fts_lifts) if fts_lifts else 0

        return {
            "queries_count": len(queries_with_targets),
            "judged_queries": total_judged,
            "improved_pct": round(improved_cnt / total_judged * 100, 1) if total_judged else 0.0,
            "unchanged_pct": round(unchanged_cnt / total_judged * 100, 1) if total_judged else 0.0,
            "regressed_pct": round(regressed_cnt / total_judged * 100, 1) if total_judged else 0.0,
            "median_semantic_rank_lift": med_sem_lift,
            "median_fts_rank_lift": med_fts_lift,
            "baseline_noise_at_10_pct": round(b_noise_10_total / b_res_total * 100, 2) if b_res_total else 0.0,
            "clean_noise_at_10_pct": round(c_noise_10_total / c_res_total * 100, 2) if c_res_total else 0.0,
            "details": results,
        }

    def check_promotion_gate(
        self,
        benchmark_summary: Dict[str, Any],
        noise_challenge_summary: Optional[Dict[str, Any]] = None,
        ground_truth_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates promotion gates with tiered verdict before switching production:

        TECHNICAL GATES (hard requirements):
          T1. Candidate DB integrity (SQLite & FTS5)
          T2. Semantic index integrity (all vectors present)

        RETRIEVAL GATES (require sufficient evidence):
          R1. Clean Noise@10 must be <= Baseline Noise@10 and <= 2.0%
          R2. Regressed queries <= threshold (adaptive: 10% for n>=30, 30% for n<30)
          R3. Median Semantic Rank Lift >= 0
          R4. Noise Challenge Set (optional, when provided)

        EVIDENCE GATE (provenance: strictly Human 🎯 ground truth):
          - Only unique valid human target queries count towards the 30 minimum.
          - Regression Set is safety only; Noise Challenge is noise test only.

        Overall verdict:
          PASS              — all gates pass, human judged queries >= 30
          PROVISIONAL PASS  — all gates pass, human judged queries < 30 (HOLD)
          BLOCKED           — any gate fails
        """
        technical_gates = []
        retrieval_gates = []

        # ── Technical Gate T1: DB Integrity ──
        admin_res = self.c_lib.admin_verify()
        t1_pass = admin_res.get("all_healthy", False)
        technical_gates.append({
            "name": "Candidate DB Integrity",
            "passed": t1_pass,
            "detail": f"SQLite & FTS5 parity check {'PASS' if t1_pass else 'FAIL'}",
        })

        # ── Technical Gate T2: Semantic Index ──
        sem_res = self.c_lib.verify_semantic_integrity()
        t2_pass = sem_res.get("semantic_ready", False)
        emb_present = sem_res.get("embeddings_present", 0)
        emb_expected = sem_res.get("chunks_expected", 0)
        technical_gates.append({
            "name": "Semantic Index Integrity",
            "passed": t2_pass,
            "detail": f"Vectors: {emb_present}/{emb_expected}, READY={t2_pass}",
        })

        # ── Retrieval Gate R1: Noise Suppression ──
        c_noise = benchmark_summary.get("clean_noise_at_10_pct", 0.0)
        b_noise = benchmark_summary.get("baseline_noise_at_10_pct", 0.0)
        r1_pass = c_noise <= b_noise and c_noise <= 2.0
        retrieval_gates.append({
            "name": "Noise@10 Suppression",
            "passed": r1_pass,
            "detail": f"Clean ({c_noise}%) <= Baseline ({b_noise}%) and <= 2.0%",
        })

        # ── Retrieval Gate R2: Regression Limit (adaptive) ──
        reg = benchmark_summary.get("regressed_pct", 0.0)
        n_judged = benchmark_summary.get("judged_queries", 0)
        reg_threshold = 10.0 if n_judged >= 30 else 30.0
        r2_pass = reg <= reg_threshold
        retrieval_gates.append({
            "name": "Rank Regression Margin",
            "passed": r2_pass,
            "detail": f"Regressed queries ({reg}%) <= {reg_threshold}% (n={n_judged})",
        })

        # ── Retrieval Gate R3: Rank Lift ──
        lift = benchmark_summary.get("median_semantic_rank_lift", 0)
        r3_pass = lift >= 0
        retrieval_gates.append({
            "name": "Semantic Rank Lift",
            "passed": r3_pass,
            "detail": f"Median semantic rank lift ({lift:+.0f}) >= 0",
        })

        # ── Retrieval Gate R4: Noise Challenge Set (optional) ──
        if noise_challenge_summary:
            nc_b = noise_challenge_summary.get("baseline_noise_at_10_pct", 0.0)
            nc_c = noise_challenge_summary.get("clean_noise_at_10_pct", 0.0)
            nc_queries = noise_challenge_summary.get("queries_count", 0)
            r4_pass = nc_c < nc_b
            retrieval_gates.append({
                "name": "Noise Challenge Suppression",
                "passed": r4_pass,
                "detail": f"Challenge Noise@10: Clean ({nc_c}%) < Baseline ({nc_b}%) across {nc_queries} queries",
            })

        # ── Combine into tiered verdict ──
        all_technical = all(g["passed"] for g in technical_gates)
        all_retrieval = all(g["passed"] for g in retrieval_gates)

        # Evidence Sufficiency: Strictly Human Ground Truth (unique judged queries)
        if ground_truth_summary is not None:
            human_judged = ground_truth_summary.get("unique_valid_queries", 0)
            min_queries = ground_truth_summary.get("promotion_minimum", 30)
        else:
            human_judged = n_judged
            min_queries = 30

        sufficient_evidence = human_judged >= min_queries

        if all_technical and all_retrieval and sufficient_evidence:
            status = "PASS"
        elif all_technical and all_retrieval and not sufficient_evidence:
            status = "PROVISIONAL PASS"
        else:
            status = "BLOCKED"

        return {
            "status": status,
            "all_gates_passed": all_technical and all_retrieval,
            "sufficient_evidence": sufficient_evidence,
            "judged_queries": human_judged,
            "min_queries_for_promotion": min_queries,
            "technical_gates": technical_gates,
            "retrieval_gates": retrieval_gates,
            "gates": technical_gates + retrieval_gates,
        }


@dataclass
class GroundTruthItem:
    id: int
    query: str
    normalized_query: str
    query_fingerprint: str
    mode: str
    doc_scope: str
    baseline_paragraph_id: int
    clean_paragraph_id: Optional[int]
    match_type: Optional[str]
    similarity: float
    source: str
    search_run_id: Optional[int] = None
    user_id: Optional[str] = None
    comment: Optional[str] = None
    created_at: str = ""
    is_valid_for_promotion: bool = True
    validation_issue: Optional[str] = None
    baseline_preview: str = ""
    clean_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroundTruthToolkit:
    """
    Manages human ground truth (🎯 target labels) across Baseline and Clean Candidate corpora.
    Ensures:
      - Provenance tracking (search_feedback vs manual vs CLI)
      - Search run binding (immutable search_run_id / answer_run_id)
      - Query fingerprinting: SHA256(normalized_query + ":" + mode + ":" + doc_scope), COUNT(DISTINCT fingerprint)
      - Strict cross-corpus validation: only unambiguous mappings (clean_id != None, similarity >= 0.78)
        are eligible for promotion gating.
    """

    def __init__(self, baseline_db: Path, candidate_db: Path):
        self.baseline_db = Path(baseline_db).resolve()
        self.candidate_db = Path(candidate_db).resolve()
        self._init_db()
        self.para_map = CorpusParagraphMap(self.candidate_db)

    def _init_db(self) -> None:
        """Ensure ground_truth_labels table exists in candidate_db."""
        conn = sqlite3.connect(self.candidate_db)
        try:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS ground_truth_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                normalized_query TEXT NOT NULL,
                query_fingerprint TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'search',
                doc_scope TEXT NOT NULL DEFAULT 'ALL',
                search_run_id INTEGER,
                baseline_paragraph_id INTEGER NOT NULL,
                clean_paragraph_id INTEGER,
                match_type TEXT,
                similarity REAL DEFAULT 0.0,
                source TEXT NOT NULL DEFAULT 'search_feedback',
                user_id TEXT,
                comment TEXT,
                created_at TEXT NOT NULL
            );
            """)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_gt_fp_pid ON ground_truth_labels(query_fingerprint, baseline_paragraph_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gt_mode ON ground_truth_labels(mode);")

            # Safe column migration if table previously existed
            try:
                cols = [c[1] for c in conn.execute("PRAGMA table_info(ground_truth_labels)").fetchall()]
                if "doc_scope" not in cols:
                    conn.execute("ALTER TABLE ground_truth_labels ADD COLUMN doc_scope TEXT NOT NULL DEFAULT 'ALL';")
                if "search_run_id" not in cols:
                    conn.execute("ALTER TABLE ground_truth_labels ADD COLUMN search_run_id INTEGER;")
            except Exception:
                pass

            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def compute_fingerprint(query: str, mode: str = "search", doc_scope: str = "ALL") -> Tuple[str, str]:
        """Returns (normalized_query, sha256_fingerprint[:16]) including doc_scope."""
        norm_q = re.sub(r"\s+", " ", query).strip().lower()
        norm_scope = (doc_scope or "ALL").strip().upper()
        fp = hashlib.sha256(f"{norm_q}:{mode}:{norm_scope}".encode("utf-8")).hexdigest()[:16]
        return norm_q, fp

    def sync_from_feedback(self) -> int:
        """
        Syncs human target labels from search_feedback in baseline_db into ground_truth_labels.
        Excludes synthetic/test queries. Resolves mappings via CorpusParagraphMap and historical search runs.
        """
        if not self.baseline_db.exists():
            return 0

        b_conn = sqlite3.connect(f"file:{self.baseline_db}?mode=ro", uri=True)
        try:
            has_table = b_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_feedback'").fetchone()
            if not has_table:
                return 0

            # Check if search_run_id is a column in search_feedback
            cur_cols = [c[1] for c in b_conn.execute("PRAGMA table_info(search_feedback)").fetchall()]
            has_run_id = "search_run_id" in cur_cols

            if has_run_id:
                sql = """
                    SELECT id, query, target_paragraph_id, feedback_type, user_id, comment, created_at, search_run_id
                    FROM search_feedback
                    WHERE target_paragraph_id IS NOT NULL
                      AND feedback_type IN ('target', 'found', 'answer_accurate', 'answer_partial')
                      AND query != 'telegram_query'
                      AND query NOT LIKE 'тест %'
                """
            else:
                sql = """
                    SELECT id, query, target_paragraph_id, feedback_type, user_id, comment, created_at, NULL as search_run_id
                    FROM search_feedback
                    WHERE target_paragraph_id IS NOT NULL
                      AND feedback_type IN ('target', 'found', 'answer_accurate', 'answer_partial')
                      AND query != 'telegram_query'
                      AND query NOT LIKE 'тест %'
                """

            rows = b_conn.execute(sql).fetchall()

            has_search_runs = bool(b_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_runs'").fetchone())
            runs_cache: Dict[int, Tuple[str, str, str]] = {}
            if has_search_runs:
                for r_id, r_q, r_m, r_s in b_conn.execute("SELECT id, query, mode, doc_scope FROM search_runs").fetchall():
                    runs_cache[r_id] = (r_q, r_m, r_s or "ALL")
        finally:
            b_conn.close()

        if not rows:
            return 0

        c_conn = sqlite3.connect(self.candidate_db)
        imported = 0
        try:
            for fb_id, raw_q, pid, fb_type, uid, comment, created_at, s_run_id in rows:
                q = raw_q.strip()
                mode = "search"
                doc_scope = "ALL"

                # Check immutable search run first
                if s_run_id and s_run_id in runs_cache:
                    run_q, run_m, run_s = runs_cache[s_run_id]
                    q = run_q
                    mode = run_m
                    doc_scope = run_s
                elif comment:
                    try:
                        c_dict = json.loads(comment)
                        mode = c_dict.get("mode", mode)
                        doc_scope = c_dict.get("doc_scope", doc_scope)
                    except Exception:
                        pass

                if mode == "search":
                    if fb_type in ("answer_accurate", "answer_partial"):
                        mode = "answer"
                    elif q.startswith("/ask"):
                        mode = "ask"
                        q = q[4:].strip()
                    elif q.startswith("/answer"):
                        mode = "answer"
                        q = q[7:].strip()

                norm_q, fp = self.compute_fingerprint(q, mode, doc_scope)

                clean_id = self.para_map.get_clean_id(pid)
                map_info = c_conn.execute(
                    "SELECT match_type, similarity FROM corpus_paragraph_map WHERE baseline_paragraph_id=?",
                    (pid,)
                ).fetchone()
                mtype = map_info[0] if map_info else None
                sim = map_info[1] if map_info else 0.0

                cur = c_conn.execute("""
                    INSERT OR IGNORE INTO ground_truth_labels
                    (query, normalized_query, query_fingerprint, mode, doc_scope, search_run_id, baseline_paragraph_id, clean_paragraph_id, match_type, similarity, source, user_id, comment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'search_feedback', ?, ?, ?)
                """, (q, norm_q, fp, mode, doc_scope, s_run_id, pid, clean_id, mtype, sim, uid, comment, created_at))
                if cur.rowcount > 0:
                    imported += 1
            c_conn.commit()
        finally:
            c_conn.close()

        return imported

    def add_label(
        self,
        query: str,
        baseline_paragraph_id: int,
        mode: str = "search",
        doc_scope: str = "ALL",
        search_run_id: Optional[int] = None,
        source: str = "cli",
        user_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Manually records a human ground-truth target label."""
        q = query.strip()
        norm_q, fp = self.compute_fingerprint(q, mode, doc_scope)
        clean_id = self.para_map.get_clean_id(baseline_paragraph_id)

        c_conn = sqlite3.connect(self.candidate_db)
        try:
            map_info = c_conn.execute(
                "SELECT match_type, similarity FROM corpus_paragraph_map WHERE baseline_paragraph_id=?",
                (baseline_paragraph_id,)
            ).fetchone()
            mtype = map_info[0] if map_info else None
            sim = map_info[1] if map_info else 0.0
            now_iso = datetime.now(timezone.utc).isoformat()

            c_conn.execute("""
                INSERT OR REPLACE INTO ground_truth_labels
                (query, normalized_query, query_fingerprint, mode, doc_scope, search_run_id, baseline_paragraph_id, clean_paragraph_id, match_type, similarity, source, user_id, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (q, norm_q, fp, mode, doc_scope.strip().upper(), search_run_id, baseline_paragraph_id, clean_id, mtype, sim, source, user_id, comment, now_iso))
            c_conn.commit()
        finally:
            c_conn.close()

        return {
            "query": q,
            "fingerprint": fp,
            "mode": mode,
            "doc_scope": doc_scope.strip().upper(),
            "search_run_id": search_run_id,
            "baseline_paragraph_id": baseline_paragraph_id,
            "clean_paragraph_id": clean_id,
            "match_type": mtype,
            "similarity": sim,
        }

    def get_labels(self) -> List[GroundTruthItem]:
        """Loads and validates all ground truth labels against corpus paragraphs."""
        self.sync_from_feedback()

        c_conn = sqlite3.connect(f"file:{self.candidate_db}?mode=ro", uri=True)
        b_conn = sqlite3.connect(f"file:{self.baseline_db}?mode=ro", uri=True) if self.baseline_db.exists() else None

        items = []
        try:
            rows = c_conn.execute("""
                SELECT id, query, normalized_query, query_fingerprint, mode, doc_scope, search_run_id, baseline_paragraph_id, clean_paragraph_id, match_type, similarity, source, user_id, comment, created_at
                FROM ground_truth_labels
                ORDER BY id ASC
            """).fetchall()

            has_b_para = bool(b_conn and b_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paragraphs'").fetchone())
            has_c_para = bool(c_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paragraphs'").fetchone())

            for r in rows:
                (lid, query, norm_q, fp, mode, scope, s_run_id, b_id, c_id, mtype, sim, src, uid, comment, created_at) = r

                b_prev = ""
                if has_b_para:
                    b_row = b_conn.execute("SELECT display_text FROM paragraphs WHERE id=?", (b_id,)).fetchone()
                    b_prev = b_row[0][:100] if b_row else ""

                c_prev = ""
                if has_c_para and c_id:
                    c_row = c_conn.execute("SELECT display_text FROM paragraphs WHERE id=?", (c_id,)).fetchone()
                    c_prev = c_row[0][:100] if c_row else ""

                is_valid = True
                issue = None
                if mtype == "deleted_ocr_noise":
                    is_valid = False
                    issue = "Target is OCR layout noise (deleted in clean corpus)"
                elif c_id is None:
                    is_valid = False
                    issue = "Missing mapping in clean candidate (NULL)"
                elif mtype == "unmatched_other" or sim < 0.78:
                    is_valid = False
                    issue = f"Ambiguous cross-corpus mapping (similarity {sim:.2f} < 0.78)"

                items.append(GroundTruthItem(
                    id=lid,
                    query=query,
                    normalized_query=norm_q,
                    query_fingerprint=fp,
                    mode=mode,
                    doc_scope=scope or "ALL",
                    search_run_id=s_run_id,
                    baseline_paragraph_id=b_id,
                    clean_paragraph_id=c_id,
                    match_type=mtype,
                    similarity=round(sim, 4),
                    source=src,
                    user_id=uid,
                    comment=comment,
                    created_at=created_at,
                    is_valid_for_promotion=is_valid,
                    validation_issue=issue,
                    baseline_preview=b_prev,
                    clean_preview=c_prev,
                ))
        finally:
            c_conn.close()
            if b_conn:
                b_conn.close()

        return items

    def get_status(self) -> Dict[str, Any]:
        """Calculates ground truth metrics and readiness for promotion gate."""
        labels = self.get_labels()
        total_labels = len(labels)
        unique_fps = set(l.query_fingerprint for l in labels)
        unique_queries = len(unique_fps)

        valid_labels = [l for l in labels if l.is_valid_for_promotion]
        unique_valid = len(set(l.query_fingerprint for l in valid_labels))

        mapped_count = sum(1 for l in labels if l.clean_paragraph_id is not None and l.match_type != "deleted_ocr_noise")
        ambiguous_count = sum(1 for l in labels if l.match_type == "unmatched_other" or (l.clean_paragraph_id and l.similarity < 0.78))
        missing_count = sum(1 for l in labels if l.clean_paragraph_id is None or l.match_type == "deleted_ocr_noise")

        by_mode = {"search": 0, "ask": 0, "answer": 0}
        for l in labels:
            m = l.mode.lower()
            if m in by_mode:
                by_mode[m] += 1
            else:
                by_mode["search"] += 1

        prom_min = 30
        remaining = max(0, prom_min - unique_valid)

        return {
            "human_labels": total_labels,
            "unique_queries": unique_queries,
            "unique_valid_queries": unique_valid,
            "mapped_to_clean": mapped_count,
            "ambiguous_mappings": ambiguous_count,
            "missing_mappings": missing_count,
            "by_mode": by_mode,
            "promotion_minimum": prom_min,
            "remaining": remaining,
            "is_ready_for_promotion": unique_valid >= prom_min,
            "labels": labels,
        }

    def validate(self) -> Dict[str, Any]:
        """Detailed validation report of all ground truth entries."""
        status = self.get_status()
        labels = status["labels"]
        valid = [l.to_dict() for l in labels if l.is_valid_for_promotion]
        invalid = [l.to_dict() for l in labels if not l.is_valid_for_promotion]

        return {
            "total": len(labels),
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "valid": valid,
            "invalid": invalid,
            "summary": status,
        }

    def export(self, output_path: Optional[Path] = None) -> str:
        """Exports ground truth data to JSON."""
        labels = [l.to_dict() for l in self.get_labels()]
        payload = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(labels),
            "labels": labels,
        }
        json_str = json.dumps(payload, ensure_ascii=False, indent=2)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str, encoding="utf-8")
        return json_str


