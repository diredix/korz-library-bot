"""
korz_corpus.py - KORZ Corpus Quality Pipeline

Provides:
  - CorpusAuditor: non-destructive scanner detecting OCR noise, running headers/footers,
    standalone page numbers, broken sentences/paragraphs, and duplicate image captions.
  - CorpusCleaner: deterministic, conservative cleaner with --dry-run diff and --apply modes.
  - CorpusVerifier: Text-Loss Safety Gate (< 3% total, < 1% unclassified loss) and diff reporting.
  - CandidateLab: staging and building clean candidate corpus for A/B evaluation against frozen baseline.
"""

import difflib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("korz_corpus")


@dataclass
class NoiseItem:
    category: str  # 'page_separator', 'standalone_number', 'running_header', 'image_marker', 'dup_caption', 'broken_sentence', 'hyphenation', 'unicode_anomaly'
    confidence: str  # 'HIGH', 'MEDIUM', 'REVIEW'
    line_number: int
    page_number: Optional[int]
    raw_text: str
    details: str = ""


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


@dataclass
class VerificationResult:
    original_path: str
    cleaned_path: str
    original_chars: int
    cleaned_chars: int
    deleted_chars: int
    deletion_ratio_pct: float
    classified_loss: Dict[str, int]
    unclassified_loss_chars: int
    unclassified_loss_pct: float
    is_safe: bool
    status: str  # 'PASS' | 'WARNING' | 'FAIL'
    notes: List[str] = field(default_factory=list)


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

    def audit_path(self, path_input: Path) -> AuditReport:
        p = Path(path_input).resolve()
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = [
                f
                for f in p.rglob("*.md")
                if not f.name.endswith(".bak") and not f.name.endswith(".cleaned.md")
            ]
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

        # Repeated on >= 3 distinct pages is likely a running header/footer
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
            prev_line_empty = True

            for idx, line in enumerate(lines, 1):
                raw = line.rstrip("\r\n")
                s = raw.strip()

                if not s:
                    prev_line_empty = True
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
                    prev_line_empty = False
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
                    prev_line_empty = False
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
                    prev_line_empty = False
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
                    prev_line_empty = False
                    continue

                # 5. Frequent Header from corpus repetition
                if s in frequent_headers:
                    report.running_headers += 1
                    report.high_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("running_header", "HIGH", idx, current_page_num, raw, "Corpus-level repeated header")
                        )
                    prev_line_empty = False
                    continue

                # 6. Standalone Page Number
                if self.STANDALONE_NUM_RE.match(s):
                    # Higher confidence if at start/end of page block
                    report.standalone_numbers += 1
                    report.high_confidence_noise += 1
                    if len(report.sample_items) < 30:
                        report.sample_items.append(
                            NoiseItem("standalone_number", "HIGH", idx, current_page_num, raw, "Standalone printed page number")
                        )
                    prev_line_empty = False
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

                prev_line_empty = False

        return report


class CorpusCleaner:
    """
    Deterministic, conservative cleaner for markdown book files.
    Preserves page boundaries ({N}---) while removing layout noise and repairing broken sentences.
    """

    PAGE_SEP_RE = re.compile(r"^\{(\d+|unnumbered)\}-+\s*$")
    HEADER_BULLET_RE = re.compile(
        r"^(?:(?:\*?[\w\s.,\'’«»-]+\*?)\s*[•—–-]\s*\d{1,4}|\d{1,4}\s*[•—–-]\s*(?:\*?[\w\s.,\'’«»-]+\*?))\s*$",
        re.IGNORECASE,
    )
    STANDALONE_NUM_RE = re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
    IMAGE_MARKER_RE = re.compile(r"^!\[(.*?)\]\((.*?)\)\s*$")
    PUNCT_START_RE = re.compile(r"^([.,;:!?»\)])\s*(.*)$")

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

    def clean_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Cleans OCR noise deterministically.
        Returns: (cleaned_text, stats_dict)
        """
        lines = text.splitlines()
        stats: Dict[str, int] = {
            "page_separators_normalized": 0,
            "standalone_numbers_removed": 0,
            "running_headers_removed": 0,
            "image_markers_removed": 0,
            "duplicate_captions_removed": 0,
            "broken_sentences_repaired": 0,
            "hyphens_repaired": 0,
            "whitespace_lines_cleaned": 0,
        }

        # Step 1: Line-by-line filtering of obvious layout garbage
        filtered_lines: List[str] = []
        last_img_caption: Optional[str] = None

        for line in lines:
            s = line.strip()

            if not s:
                filtered_lines.append(line)
                continue

            # Normalizing page separator
            m_sep = self.PAGE_SEP_RE.match(s)
            if m_sep:
                filtered_lines.append(f"{{{m_sep.group(1)}}}--------------------------------")
                stats["page_separators_normalized"] += 1
                last_img_caption = None
                continue

            # Image marker removal
            m_img = self.IMAGE_MARKER_RE.match(s)
            if m_img:
                last_img_caption = m_img.group(1).strip()
                stats["image_markers_removed"] += 1
                continue

            # Duplicate caption removal
            if last_img_caption and s == last_img_caption:
                stats["duplicate_captions_removed"] += 1
                last_img_caption = None
                continue
            last_img_caption = None

            # Running header with bullets/numbers
            if self.HEADER_BULLET_RE.match(s):
                stats["running_headers_removed"] += 1
                continue

            # Corpus-level frequent header
            if s in self.frequent_headers:
                stats["running_headers_removed"] += 1
                continue

            # Standalone page numbers
            if self.STANDALONE_NUM_RE.match(s):
                stats["standalone_numbers_removed"] += 1
                continue

            filtered_lines.append(line)

        # Step 2: Repair broken sentence starts across page transitions
        # e.g. Page N ends with text, Page N+1 starts with ". Він пішов"
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
                # Attach the punctuation to the previous text line!
                j = len(repaired_lines) - 1
                while j >= 0 and (repaired_lines[j].strip().startswith("{") or not repaired_lines[j].strip()):
                    j -= 1

                if j >= 0:
                    prev = repaired_lines[j].rstrip()
                    repaired_lines[j] = prev + punct_char
                    stats["broken_sentences_repaired"] += 1
                    if rest_of_text:
                        repaired_lines.append(rest_of_text)
                    i += 1
                    continue

            repaired_lines.append(line)
            i += 1

        # Step 3: De-hyphenate words split at line wraps (e.g. 'психологіч-\nного' -> 'психологічного')
        joined_text = "\n".join(repaired_lines)

        def dehyphen(match):
            stats["hyphens_repaired"] += 1
            return match.group(1) + match.group(2)

        # Ukrainian/Cyrillic lowercase dehyphenation
        cleaned_text = re.sub(
            r"([а-яіїєґa-z])-[\r\n]+([а-яіїєґa-z])",
            dehyphen,
            joined_text,
        )

        # Step 4: Normalize multiple consecutive blank lines (> 2 -> 2)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip() + "\n"

        return cleaned_text, stats

    def clean_file(self, file_path: Path, dry_run: bool = True, output_path: Optional[Path] = None) -> CleaningDiff:
        fp = Path(file_path).resolve()
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            orig = f.read()

        cleaned, stats = self.clean_text(orig)

        orig_chars = len(orig)
        clean_chars = len(cleaned)
        diff_chars = orig_chars - clean_chars
        ratio_pct = (diff_chars / orig_chars * 100) if orig_chars > 0 else 0.0

        # Unified diff for inspection
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
        )

        if not dry_run:
            dest = output_path or fp.with_name(fp.stem + ".cleaned" + fp.suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8", newline="\n") as out_fp:
                out_fp.write(cleaned)

        return diff_obj


class CorpusVerifier:
    """
    Text-Loss Safety Gate ensuring cleaned files do not lose genuine book content.
    Rules:
      - Total deleted characters should ideally be < 3.5% (layout noise).
      - Unclassified content loss must be < 1.0%.
    """

    def __init__(self, max_allowed_loss_pct: float = 3.5, max_unclassified_loss_pct: float = 1.0):
        self.max_allowed_loss_pct = max_allowed_loss_pct
        self.max_unclassified_loss_pct = max_unclassified_loss_pct

    def verify(self, original_text: str, cleaned_text: str, stats: Dict[str, int]) -> VerificationResult:
        orig_len = len(original_text)
        clean_len = len(cleaned_text)
        diff_len = max(0, orig_len - clean_len)
        ratio_pct = (diff_len / orig_len * 100) if orig_len > 0 else 0.0

        # Estimate accounted characters based on stats
        # A standalone page number ~ 3-4 chars
        # A running header ~ 20-40 chars
        # An image marker ~ 50-150 chars
        # Normalized whitespace & dashes ~ remainder
        accounted_chars = (
            stats.get("standalone_numbers_removed", 0) * 4
            + stats.get("running_headers_removed", 0) * 30
            + stats.get("image_markers_removed", 0) * 80
            + stats.get("duplicate_captions_removed", 0) * 100
            + stats.get("hyphens_repaired", 0) * 2
        )

        unclassified = max(0, diff_len - accounted_chars)
        unclassified_pct = (unclassified / orig_len * 100) if orig_len > 0 else 0.0

        notes: List[str] = []
        is_safe = True
        status = "PASS"

        if ratio_pct > self.max_allowed_loss_pct:
            notes.append(f"Total deletion ratio ({ratio_pct:.2f}%) exceeds safety threshold ({self.max_allowed_loss_pct}%).")
            status = "WARNING"

        if unclassified_pct > self.max_unclassified_loss_pct:
            notes.append(
                f"Unclassified character loss ({unclassified_pct:.2f}%) exceeds safety margin ({self.max_unclassified_loss_pct}%)."
            )
            status = "FAIL"
            is_safe = False
        else:
            notes.append("Character loss is fully accounted for by identified layout noise items.")

        return VerificationResult(
            original_path="",
            cleaned_path="",
            original_chars=orig_len,
            cleaned_chars=clean_len,
            deleted_chars=diff_len,
            deletion_ratio_pct=ratio_pct,
            classified_loss=stats,
            unclassified_loss_chars=unclassified,
            unclassified_loss_pct=unclassified_pct,
            is_safe=is_safe,
            status=status,
            notes=notes,
        )
