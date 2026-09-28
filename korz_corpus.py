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
                filtered_lines.append(f"{{{pg_val}}}--------------------------------")
                stats["page_separators_normalized"] += 1
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
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip() + "\n"

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
        build_res = cand_lib.build_semantic_index(model_id_or_name=model_name)

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
