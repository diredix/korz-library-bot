"""
korz_core.py - Core Library for KORZ LOCAL (Single-file SQLite Engine)

Zero external server daemons, zero Qdrant, zero Docker.
Everything operates within a local SQLite database:
  - Document -> Page -> Paragraph hierarchy
  - Search Chunks for multi-paragraph window
  - Native SQLite FTS5 with unicode61 tokenizer & BM25 ranking
  - Case-insensitive literal substring search (Exact find)
  - Punctuation & character normalization (Ukrainian apostrophes ’, ʼ, `, quotes «», hyphens, digits, mixed UA/EN)
  - Surround context expansion (get_context)
  - Content-addressed source archiving & duplicate prevention
  - Atomic transactions with WAL mode, busy timeout, foreign keys
  - Schema versioning and migration framework
  - Hot backup & restore APIs
"""

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("korz")

# Default directories
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "korz_library.sqlite"
DEFAULT_SOURCES_DIR = DEFAULT_DATA_DIR / "sources"
DEFAULT_BACKUPS_DIR = DEFAULT_DATA_DIR / "backups"

CURRENT_SCHEMA_VERSION = 1


@dataclass
class KorzConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    db_path: Path = DEFAULT_DB_PATH
    sources_dir: Path = DEFAULT_SOURCES_DIR
    backups_dir: Path = DEFAULT_BACKUPS_DIR
    max_chunk_chars: int = 1400
    target_chunk_chars: int = 1200
    target_paragraph_chars: int = 1400

    @classmethod
    def from_env(cls) -> "KorzConfig":
        data_dir = Path(os.environ.get("KORZ_DATA_DIR", str(DEFAULT_DATA_DIR)))
        db_path = Path(os.environ.get("KORZ_DB_PATH", str(data_dir / "korz_library.sqlite")))
        sources_dir = data_dir / "sources"
        backups_dir = data_dir / "backups"
        return cls(data_dir=data_dir, db_path=db_path, sources_dir=sources_dir, backups_dir=backups_dir)


@dataclass
class ExtractedPage:
    page_number: int
    raw_text: str
    display_text: str


@dataclass
class ExtractedParagraph:
    page_number: int
    paragraph_index: int
    char_start: int
    char_end: int
    raw_text: str
    display_text: str


@dataclass
class ExtractedDocument:
    source_sha256: str
    source_name: str
    file_size: int
    mime_type: str
    title: str
    author: Optional[str]
    edition: Optional[str]
    language: str
    pages: List[ExtractedPage]


@dataclass
class SearchResult:
    """Canonical search result contract across CLI, Telegram, Web."""
    paragraph_id: int
    document_id: str
    title: str
    author: Optional[str]
    page: int
    paragraph_index: int
    text: str
    snippet: str
    score: float = 0.0
    match_type: str = "fts"  # "exact" | "fts"

    # Backward compatibility aliases
    @property
    def page_number(self) -> int:
        return self.page

    @property
    def display_text(self) -> str:
        return self.text


# Custom SQLite functions
def _instr_ci(text: Optional[str], sub: Optional[str]) -> int:
    """Case-insensitive and apostrophe-normalized substring search for SQLite."""
    if not text or not sub:
        return 0
    t = re.sub(r"[\u2019\u02bc`]", "'", text.lower())
    s = re.sub(r"[\u2019\u02bc`]", "'", sub.lower())
    return 1 if s in t else 0


def _regexp(pattern: Optional[str], text: Optional[str]) -> int:
    """Regex pattern match for SQLite."""
    if not pattern or text is None:
        return 0
    try:
        return 1 if re.search(pattern, text, re.IGNORECASE) else 0
    except re.error:
        return 0


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Create and configure a SQLite connection with WAL mode and custom functions."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # autocommit mode (isolation_level=None) prevents unintended implicit transactions & lock deadlocks
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.create_function("INSTR_CI", 2, _instr_ci)
    conn.create_function("REGEXP", 2, _regexp)
    return conn


def init_database(conn: sqlite3.Connection) -> None:
    """Initialize database tables, migrations, FTS5 index, and triggers."""
    cur = conn.execute("PRAGMA user_version;")
    version_row = cur.fetchone()
    current_version = version_row[0] if version_row else 0

    if current_version < 1:
        _apply_migration_v1(conn)
    else:
        # Ensure search_log table exists on existing v1 databases
        conn.execute("""
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            mode TEXT NOT NULL,
            result_count INTEGER NOT NULL,
            top_result_id INTEGER,
            latency_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_search_log_created ON search_log(created_at);")


def _apply_migration_v1(conn: sqlite3.Connection) -> None:
    """Initial schema (v1) with explicit atomic transaction."""
    conn.execute("BEGIN IMMEDIATE;")
    try:
        # Schema migrations log
        conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        );
        """)

        # Sources table (content-addressed)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            source_sha256 TEXT PRIMARY KEY,
            original_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            mime_type TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        # Documents table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            language TEXT DEFAULT 'uk',
            edition TEXT,
            source_sha256 TEXT NOT NULL REFERENCES sources(source_sha256) ON DELETE CASCADE,
            total_pages INTEGER NOT NULL DEFAULT 0,
            total_paragraphs INTEGER NOT NULL DEFAULT 0,
            total_chunks INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_sha256);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);")

        # Pages table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            display_text TEXT NOT NULL,
            paragraphs_count INTEGER NOT NULL DEFAULT 0
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pages_doc_num ON pages(document_id, page_number);")

        # Paragraphs table (primary unit of storage & retrieval)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paragraph_key TEXT UNIQUE NOT NULL,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            paragraph_index INTEGER NOT NULL,
            char_start INTEGER NOT NULL,
            char_end INTEGER NOT NULL,
            raw_text TEXT NOT NULL,
            display_text TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_doc_page ON paragraphs(document_id, page_number, paragraph_index);")

        # Chunks table (auxiliary multi-paragraph retrieval window)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_key TEXT UNIQUE NOT NULL,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_start INTEGER NOT NULL,
            page_end INTEGER NOT NULL,
            paragraph_start_id INTEGER NOT NULL REFERENCES paragraphs(id) ON DELETE CASCADE,
            paragraph_end_id INTEGER NOT NULL REFERENCES paragraphs(id) ON DELETE CASCADE,
            display_text TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, page_start, page_end);")

        # FTS5 Virtual Table for Paragraphs (external content table)
        conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS paragraphs_fts USING fts5(
            display_text,
            content='paragraphs',
            content_rowid='id',
            tokenize='unicode61'
        );
        """)

        # Triggers to keep FTS5 synchronized with paragraphs table
        conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paragraphs_ai AFTER INSERT ON paragraphs BEGIN
            INSERT INTO paragraphs_fts(rowid, display_text) VALUES (new.id, new.display_text);
        END;
        """)
        conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paragraphs_ad AFTER DELETE ON paragraphs BEGIN
            INSERT INTO paragraphs_fts(paragraphs_fts, rowid, display_text) VALUES('delete', old.id, old.display_text);
        END;
        """)
        conn.execute("""
        CREATE TRIGGER IF NOT EXISTS paragraphs_au AFTER UPDATE ON paragraphs BEGIN
            INSERT INTO paragraphs_fts(paragraphs_fts, rowid, display_text) VALUES('delete', old.id, old.display_text);
            INSERT INTO paragraphs_fts(rowid, display_text) VALUES (new.id, new.display_text);
        END;
        """)

        # Ingestion history log
        conn.execute("""
        CREATE TABLE IF NOT EXISTS ingest_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_sha256 TEXT,
            document_id TEXT,
            action TEXT NOT NULL,
            status TEXT NOT NULL,
            details TEXT,
            created_at TEXT NOT NULL
        );
        """)

        # Search Query Log (anonymous diagnostics for tuning V2)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS search_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            mode TEXT NOT NULL,
            result_count INTEGER NOT NULL,
            top_result_id INTEGER,
            latency_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_search_log_created ON search_log(created_at);")

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            (1, now_iso, "v1 baseline schema: documents, pages, paragraphs, chunks, FTS5, search_log"),
        )
        conn.execute("PRAGMA user_version = 1;")
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise


class TextProcessor:
    """Cleaning, paragraph segmentation, and chunk grouping."""

    @staticmethod
    def normalize_apostrophes(text: str) -> str:
        """Normalize all apostrophe variants (’, ʼ, `) to standard ASCII apostrophe '."""
        if not text:
            return ""
        return re.sub(r"[\u2019\u02bc`]", "'", text)

    @classmethod
    def clean_page_text(cls, text: str) -> str:
        """Strip markdown images, divider lines, and normalize whitespace & apostrophes."""
        if not text:
            return ""
        # Remove markdown image tags ![...](...)
        t = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # Remove html tags like <img ...>, <div>, etc.
        t = re.sub(r"<[^>]+>", "", t)
        # Remove standalone divider lines (e.g., ***, ---, ===, ___, {1}-----)
        t = re.sub(r"^\s*[-*=_]{3,}\s*$", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\s*\{\d+\}[-*=_]{3,}\s*$", "", t, flags=re.MULTILINE)
        # Normalize multiple spaces and non-breaking spaces
        t = t.replace("\u00a0", " ").replace("\ufeff", "")
        # Normalize apostrophes
        t = cls.normalize_apostrophes(t)
        # Remove consecutive blank lines
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    @staticmethod
    def _split_long_paragraph(para: str, max_chars: int = 1400) -> List[str]:
        """Split a long paragraph on sentence boundaries with fallback to words or characters."""
        if len(para) <= max_chars:
            return [para]

        # Sentence split on punctuation followed by whitespace and uppercase/quote
        sentences = re.split(r"(?<=[.!?])\s+(?=[А-ЯІЇЄҐA-Z0-9\"'«])", para)
        if len(sentences) <= 1:
            # Fallback to general sentence end
            sentences = re.split(r"(?<=[.!?])\s+", para)

        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            if len(s_clean) > max_chars:
                # Flush current buffer first
                if buf:
                    chunks.append(" ".join(buf))
                    buf = []
                    buf_len = 0
                # Word-level split
                words = s_clean.split()
                w_buf: List[str] = []
                w_len = 0
                for w in words:
                    cand = w_len + len(w) + (1 if w_buf else 0)
                    if cand > max_chars and w_buf:
                        chunks.append(" ".join(w_buf))
                        w_buf = [w]
                        w_len = len(w)
                    else:
                        w_buf.append(w)
                        w_len = cand
                if w_buf:
                    chunks.append(" ".join(w_buf))
                continue

            cand_len = buf_len + len(s_clean) + (1 if buf else 0)
            if cand_len > max_chars and buf:
                chunks.append(" ".join(buf))
                buf = [s_clean]
                buf_len = len(s_clean)
            else:
                buf.append(s_clean)
                buf_len = cand_len

        if buf:
            chunks.append(" ".join(buf))

        return [c for c in chunks if c.strip()]

    @classmethod
    def segment_paragraphs(cls, page_text: str, page_number: int, max_chars: int = 1400) -> List[ExtractedParagraph]:
        """Segment cleaned page text into logical paragraphs."""
        raw_paras = re.split(r"\n\s*\n+", page_text)
        result: List[ExtractedParagraph] = []
        idx = 1
        current_offset = 0

        for raw_p in raw_paras:
            clean_p = raw_p.strip()
            if not clean_p:
                current_offset += len(raw_p) + 2
                continue

            # Skip pure separators or headers like "{unnumbered}"
            if re.match(r"^\{.*\}$", clean_p) or re.match(r"^[-*=_]{3,}$", clean_p):
                current_offset += len(raw_p) + 2
                continue

            # Check if paragraph needs sentence splitting
            if len(clean_p) > max_chars:
                sub_paras = cls._split_long_paragraph(clean_p, max_chars=max_chars)
                sub_offset = current_offset
                for sp in sub_paras:
                    result.append(
                        ExtractedParagraph(
                            page_number=page_number,
                            paragraph_index=idx,
                            char_start=sub_offset,
                            char_end=sub_offset + len(sp),
                            raw_text=sp,
                            display_text=sp,
                        )
                    )
                    idx += 1
                    sub_offset += len(sp) + 1
            else:
                result.append(
                    ExtractedParagraph(
                        page_number=page_number,
                        paragraph_index=idx,
                        char_start=current_offset,
                        char_end=current_offset + len(clean_p),
                        raw_text=clean_p,
                        display_text=clean_p,
                    )
                )
                idx += 1

            current_offset += len(raw_p) + 2

        return result

    @classmethod
    def build_chunks(
        cls, paragraphs: List[ExtractedParagraph], target_chars: int = 1200, max_chars: int = 1400
    ) -> List[Dict[str, Any]]:
        """Group 2-4 consecutive paragraphs into search chunks."""
        chunks: List[Dict[str, Any]] = []
        if not paragraphs:
            return chunks

        i = 0
        chunk_idx = 1
        while i < len(paragraphs):
            cur_pars = [paragraphs[i]]
            cur_len = len(paragraphs[i].display_text)
            j = i + 1

            while j < len(paragraphs):
                next_p = paragraphs[j]
                cand_len = cur_len + len(next_p.display_text) + 2
                if cand_len > max_chars:
                    break
                cur_pars.append(next_p)
                cur_len = cand_len
                j += 1
                if cur_len >= target_chars:
                    break

            display_text = "\n\n".join(p.display_text for p in cur_pars)
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "page_start": cur_pars[0].page_number,
                    "page_end": cur_pars[-1].page_number,
                    "paragraph_start_idx": cur_pars[0].paragraph_index,
                    "paragraph_end_idx": cur_pars[-1].paragraph_index,
                    "paragraph_start_key": f"p{cur_pars[0].page_number}:par{cur_pars[0].paragraph_index}",
                    "paragraph_end_key": f"p{cur_pars[-1].page_number}:par{cur_pars[-1].paragraph_index}",
                    "display_text": display_text,
                }
            )
            chunk_idx += 1

            # Advance by at least 1 paragraph, or overlap by 1 paragraph if multi-paragraph
            if len(cur_pars) > 2:
                i += len(cur_pars) - 1
            else:
                i = j

        return chunks


class SourceExtractor:
    """Extracts pages and metadata from PDFs, Markdown files, or directories."""

    @staticmethod
    def infer_metadata(path_obj: Path, first_page_text: str = "") -> Tuple[str, Optional[str], Optional[str]]:
        """Infer Title, Author, and Edition from path and header text."""
        name = path_obj.stem
        # Clean common prefixes/suffixes
        name_clean = re.sub(r"^(\.)?datalab-output-", "", name)
        name_clean = re.sub(r"_Pages_.*$", "", name_clean, flags=re.IGNORECASE)
        name_clean = re.sub(r"_UA$", "", name_clean)

        author: Optional[str] = None
        title: str = name_clean.replace("_", " ")
        edition: Optional[str] = None

        m_year = re.search(r"\b(19\d\d|20\d\d)\b", name)
        if m_year:
            edition = m_year.group(1)

        if "Sapolsky" in name:
            author = "Robert Sapolsky"
            title = "A Primate's Memoir"
        elif "Looking_Forward" in name or "Looking Forward" in name:
            author = "Jacque Fresco"
            title = "Looking Forward"
        elif "Future_Design" in name or "Future Design" in name:
            author = "Jacque Fresco"
            title = "Future Design"
        elif "Best_Money" in name or "Best_That_Money" in name or "Найкраще, що не купиш" in first_page_text:
            author = "Jacque Fresco"
            title = "The Best That Money Can't Buy"
        elif "Korzybski" in name or "Science_and_Sanity" in name:
            author = "Alfred Korzybski"
            title = "Science and Sanity"
        else:
            parts = name_clean.split("_")
            if len(parts) >= 3:
                author = f"{parts[0]} {parts[1]}"
                title = " ".join(parts[2:])

        if first_page_text:
            m_h1 = re.search(r"^#\s+(.+)$", first_page_text, re.MULTILINE)
            if m_h1 and len(m_h1.group(1).strip()) > 3:
                candidate = m_h1.group(1).strip()
                if candidate.upper() not in ["КНИГА І", "ЧАСТИНА 1", "РОЗДІЛ 1", "TABLE OF CONTENTS"]:
                    title = candidate

        return title, author, edition

    @classmethod
    def compute_sha256(cls, path_obj: Path) -> Tuple[str, int]:
        """Compute SHA256 and total size for file or directory."""
        hasher = hashlib.sha256()
        total_size = 0

        if path_obj.is_file():
            with open(path_obj, "rb") as fp:
                while chunk := fp.read(1024 * 1024):
                    hasher.update(chunk)
                    total_size += len(chunk)
            return hasher.hexdigest(), total_size

        elif path_obj.is_dir():
            all_files = sorted([f for f in path_obj.rglob("*") if f.is_file() and not f.name.endswith(".bak")])
            for f in all_files:
                rel = str(f.relative_to(path_obj)).encode("utf-8")
                hasher.update(rel)
                f_size = f.stat().st_size
                total_size += f_size
                with open(f, "rb") as fp:
                    while chunk := fp.read(1024 * 1024):
                        hasher.update(chunk)
            return hasher.hexdigest(), total_size

        raise ValueError(f"Path does not exist: {path_obj}")

    @classmethod
    def extract_from_pdf(cls, pdf_path: Path) -> List[ExtractedPage]:
        """Extract pages from PDF file using PyMuPDF (fitz) or pypdf."""
        pages: List[ExtractedPage] = []
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                txt = page.get_text()
                clean = TextProcessor.clean_page_text(txt)
                pages.append(ExtractedPage(page_number=i + 1, raw_text=txt, display_text=clean))
            doc.close()
            return pages
        except ImportError:
            pass

        import pypdf

        reader = pypdf.PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            clean = TextProcessor.clean_page_text(txt)
            pages.append(ExtractedPage(page_number=i + 1, raw_text=txt, display_text=clean))
        return pages

    @classmethod
    def extract_from_markdown_dir(cls, dir_path: Path) -> List[ExtractedPage]:
        """Extract pages from a directory containing batch markdown files."""
        md_files = [f for f in dir_path.glob("*.md") if not f.name.endswith(".bak")]
        if not md_files:
            return []

        cyr_files = []
        for f in md_files:
            with open(f, "rb") as fp:
                raw = fp.read(2048)
            txt = raw.decode("utf-8", errors="ignore")
            if any("\u0400" <= ch <= "\u04ff" for ch in txt):
                cyr_files.append(f)

        target_files = cyr_files if cyr_files else md_files

        def sort_key(p: Path) -> int:
            m = re.search(r"(\d+)-(\d+)", p.name)
            if m:
                return int(m.group(1))
            m2 = re.search(r"(\d+)", p.name)
            if m2:
                return int(m2.group(1))
            return 0

        target_files = sorted(target_files, key=sort_key)

        pages: List[ExtractedPage] = []
        current_page_num = 1

        for f in target_files:
            with open(f, "rb") as fp:
                raw_bytes = fp.read()
            txt = raw_bytes.decode("utf-8", errors="replace")

            parts = re.split(r"\n*\{(\d+|unnumbered)\}-+\n*", txt)
            if len(parts) > 1:
                for i in range(1, len(parts), 2):
                    content = parts[i + 1]
                    clean = TextProcessor.clean_page_text(content)
                    pages.append(ExtractedPage(page_number=current_page_num, raw_text=content, display_text=clean))
                    current_page_num += 1
            else:
                alt_parts = re.split(r"\n*---+\s*(?:Page|Сторінка)\s*(\d+)\s*---+\n*", txt, flags=re.IGNORECASE)
                if len(alt_parts) > 1:
                    for i in range(1, len(alt_parts), 2):
                        content = alt_parts[i + 1]
                        clean = TextProcessor.clean_page_text(content)
                        pages.append(ExtractedPage(page_number=current_page_num, raw_text=content, display_text=clean))
                        current_page_num += 1
                else:
                    clean = TextProcessor.clean_page_text(txt)
                    pages.append(ExtractedPage(page_number=current_page_num, raw_text=txt, display_text=clean))
                    current_page_num += 1

        return pages

    @classmethod
    def extract_document(
        cls, path_input: Union[str, Path], title: Optional[str] = None, author: Optional[str] = None
    ) -> ExtractedDocument:
        """Extract complete document structure from file or directory."""
        path_obj = Path(path_input).resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"Input path not found: {path_input}")

        sha256, file_size = cls.compute_sha256(path_obj)
        pages: List[ExtractedPage] = []
        mime_type = "application/octet-stream"

        if path_obj.is_file():
            suffix = path_obj.suffix.lower()
            if suffix == ".pdf":
                mime_type = "application/pdf"
                pages = cls.extract_from_pdf(path_obj)
            elif suffix in [".md", ".txt"]:
                mime_type = "text/markdown" if suffix == ".md" else "text/plain"
                with open(path_obj, "rb") as fp:
                    raw_bytes = fp.read()
                txt = raw_bytes.decode("utf-8", errors="replace")
                parts = re.split(r"\n*\{(\d+|unnumbered)\}-+\n*", txt)
                if len(parts) > 1:
                    p_num = 1
                    for i in range(1, len(parts), 2):
                        content = parts[i + 1]
                        pages.append(
                            ExtractedPage(
                                page_number=p_num,
                                raw_text=content,
                                display_text=TextProcessor.clean_page_text(content),
                            )
                        )
                        p_num += 1
                else:
                    pages.append(
                        ExtractedPage(page_number=1, raw_text=txt, display_text=TextProcessor.clean_page_text(txt))
                    )
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

        elif path_obj.is_dir():
            mime_type = "application/x-directory"
            master_pdf = list(path_obj.glob("*.pdf"))
            pages = cls.extract_from_markdown_dir(path_obj)
            if not pages and master_pdf:
                pages = cls.extract_from_pdf(master_pdf[0])

        if not pages:
            raise ValueError(f"No readable text or pages could be extracted from: {path_obj}")

        first_page_text = pages[0].display_text if pages else ""
        inferred_title, inferred_author, inferred_edition = cls.infer_metadata(path_obj, first_page_text)

        final_title = title or inferred_title
        final_author = author or inferred_author

        sample_text = " ".join(p.display_text[:200] for p in pages[:5])
        cyr_chars = len(re.findall(r"[\u0400-\u04ff]", sample_text))
        lang = "uk" if cyr_chars > 20 else "en"

        return ExtractedDocument(
            source_sha256=sha256,
            source_name=path_obj.name,
            file_size=file_size,
            mime_type=mime_type,
            title=final_title,
            author=final_author,
            edition=inferred_edition,
            language=lang,
            pages=pages,
        )


class KorzLibrary:
    """Unified API for KORZ Local Library."""

    def __init__(self, config: Optional[KorzConfig] = None):
        self.config = config or KorzConfig.from_env()
        self.conn = get_connection(self.config.db_path)
        init_database(self.conn)

    def close(self) -> None:
        """Close SQLite database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "KorzLibrary":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def add_book(
        self,
        path_input: Union[str, Path],
        title: Optional[str] = None,
        author: Optional[str] = None,
        edition: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transactional ingestion of a book into the local SQLite library.
        Prevents duplicate imports by source SHA256.
        Rolls back completely if an error occurs.
        """
        path_obj = Path(path_input).resolve()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1: Extract document structure
        doc = SourceExtractor.extract_document(path_obj, title=title, author=author)
        if edition:
            doc.edition = edition

        # Step 2: Check for duplicates by source_sha256
        cur = self.conn.execute("SELECT id, title FROM documents WHERE source_sha256 = ?", (doc.source_sha256,))
        existing = cur.fetchone()
        if existing:
            return {
                "status": "DUPLICATE",
                "message": f"Document already imported. (ID: {existing['id']}, Title: '{existing['title']}')",
                "document_id": existing["id"],
                "title": existing["title"],
                "source_sha256": doc.source_sha256,
            }

        # Step 3: Archive source to data/sources/
        sha_prefix = doc.source_sha256[:2]
        dest_dir = self.config.sources_dir / sha_prefix
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext = path_obj.suffix or ".bundle"
        stored_path = dest_dir / f"{doc.source_sha256}{ext}"

        if not stored_path.exists():
            if path_obj.is_file():
                shutil.copy2(path_obj, stored_path)
            else:
                zip_path = dest_dir / f"{doc.source_sha256}.zip"
                shutil.make_archive(str(dest_dir / doc.source_sha256), "zip", path_obj)
                stored_path = zip_path

        # Step 4: Generate document ID with Ukrainian transliteration
        translit_map = {
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'h', 'ґ': 'g', 'д': 'd', 'е': 'e', 'є': 'ye',
            'ж': 'zh', 'з': 'z', 'и': 'y', 'і': 'i', 'ї': 'yi', 'й': 'y', 'к': 'k', 'л': 'l',
            'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ь': '', 'ю': 'yu',
            'я': 'ya', "'": '', '’': ''
        }
        clean_title = doc.title.lower()
        translit_title = "".join(translit_map.get(ch, ch) for ch in clean_title)
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", translit_title).strip("-")[:40]
        if not slug:
            slug = "book"
        doc_id = f"doc-{slug}-{doc.source_sha256[:8]}"

        # Step 5: Process pages and segment paragraphs
        total_pages = len(doc.pages)
        all_paragraphs: List[ExtractedParagraph] = []
        page_records: List[Tuple[str, str, int, str, str, int]] = []

        for p in doc.pages:
            page_id = f"{doc_id}:p{p.page_number}"
            paras = TextProcessor.segment_paragraphs(
                p.display_text, page_number=p.page_number, max_chars=self.config.target_paragraph_chars
            )
            all_paragraphs.extend(paras)
            page_records.append((page_id, doc_id, p.page_number, p.raw_text, p.display_text, len(paras)))

        if not all_paragraphs:
            raise ValueError(f"Document contains 0 valid paragraphs: {doc.title}")

        # Step 6: Build auxiliary search chunks
        chunks = TextProcessor.build_chunks(
            all_paragraphs, target_chars=self.config.target_chunk_chars, max_chars=self.config.max_chunk_chars
        )

        # Step 7: Atomic SQLite Transaction with explicit BEGIN IMMEDIATE
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            # Insert Source
            self.conn.execute(
                """
                INSERT OR REPLACE INTO sources (source_sha256, original_name, file_size, mime_type, stored_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc.source_sha256, doc.source_name, doc.file_size, doc.mime_type, str(stored_path), now_iso),
            )

            # Insert Document
            self.conn.execute(
                """
                INSERT INTO documents (id, title, author, language, edition, source_sha256, total_pages, total_paragraphs, total_chunks, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    doc.title,
                    doc.author,
                    doc.language,
                    doc.edition,
                    doc.source_sha256,
                    total_pages,
                    len(all_paragraphs),
                    len(chunks),
                    now_iso,
                ),
            )

            # Batch insert pages
            self.conn.executemany(
                """
                INSERT INTO pages (id, document_id, page_number, raw_text, display_text, paragraphs_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                page_records,
            )

            # Batch insert paragraphs (FTS5 populated automatically via triggers)
            para_records = []
            for p in all_paragraphs:
                p_key = f"{doc_id}:p{p.page_number}:par{p.paragraph_index}"
                page_id = f"{doc_id}:p{p.page_number}"
                para_records.append(
                    (
                        p_key,
                        doc_id,
                        page_id,
                        p.page_number,
                        p.paragraph_index,
                        p.char_start,
                        p.char_end,
                        p.raw_text,
                        p.display_text,
                    )
                )

            self.conn.executemany(
                """
                INSERT INTO paragraphs (paragraph_key, document_id, page_id, page_number, paragraph_index, char_start, char_end, raw_text, display_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                para_records,
            )

            # Fetch inserted paragraph rowids to link chunks
            para_id_map: Dict[str, int] = {}
            cur = self.conn.execute(
                "SELECT id, page_number, paragraph_index FROM paragraphs WHERE document_id = ?", (doc_id,)
            )
            for row in cur.fetchall():
                para_id_map[f"p{row['page_number']}:par{row['paragraph_index']}"] = row["id"]

            # Batch insert chunks
            chunk_records = []
            for ch in chunks:
                ch_key = f"{doc_id}:p{ch['page_start']}-{ch['page_end']}:c{ch['chunk_index']}"
                p_start_id = para_id_map.get(ch["paragraph_start_key"])
                p_end_id = para_id_map.get(ch["paragraph_end_key"])
                if p_start_id and p_end_id:
                    chunk_records.append(
                        (
                            ch_key,
                            doc_id,
                            ch["page_start"],
                            ch["page_end"],
                            p_start_id,
                            p_end_id,
                            ch["display_text"],
                        )
                    )

            self.conn.executemany(
                """
                INSERT INTO chunks (chunk_key, document_id, page_start, page_end, paragraph_start_id, paragraph_end_id, display_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_records,
            )

            # Record in ingest history
            self.conn.execute(
                """
                INSERT INTO ingest_history (source_sha256, document_id, action, status, details, created_at)
                VALUES (?, ?, 'ADD', 'SUCCESS', ?, ?)
                """,
                (
                    doc.source_sha256,
                    doc_id,
                    json.dumps(
                        {
                            "pages": total_pages,
                            "paragraphs": len(all_paragraphs),
                            "chunks": len(chunks),
                            "title": doc.title,
                        },
                        ensure_ascii=False,
                    ),
                    now_iso,
                ),
            )

            # Explicit atomic commit
            self.conn.execute("COMMIT;")

        except Exception as e:
            # Explicit rollback wipes all partial inserts
            try:
                self.conn.execute("ROLLBACK;")
            except Exception:
                pass

            # Ingest failure recorded outside the aborted transaction
            try:
                self.conn.execute(
                    """
                    INSERT INTO ingest_history (source_sha256, document_id, action, status, details, created_at)
                    VALUES (?, ?, 'ADD', 'FAILED', ?, ?)
                    """,
                    (doc.source_sha256, doc_id, str(e), now_iso),
                )
            except Exception:
                pass
            raise e

        return {
            "status": "READY",
            "document_id": doc_id,
            "title": doc.title,
            "author": doc.author,
            "pages": total_pages,
            "paragraphs": len(all_paragraphs),
            "chunks": len(chunks),
            "source_sha256": doc.source_sha256,
        }

    def _build_fts_query(self, query: str) -> str:
        """
        Build a robust, syntax-safe FTS5 query.
        Handles Ukrainian apostrophes, hyphens, quotes, digits, and mixed UA/EN.
        """
        clean_q = TextProcessor.normalize_apostrophes(query.strip())

        # Exact phrase if wrapped in quotes ("..." or «...»)
        if (clean_q.startswith('"') and clean_q.endswith('"')) or (clean_q.startswith('«') and clean_q.endswith('»')):
            inner = clean_q[1:-1].replace('"', '""').replace('«', '').replace('»', '').strip()
            if inner:
                return f'"{inner}"'

        # Tokenize preserving hyphenated words and words with internal apostrophes
        # e.g., 'будь-який', "пам'ять", '1969', 'Sapolsky'
        tokens = re.findall(r"[\w]+(?:['\-][\w]+)*", clean_q, re.UNICODE)
        if not tokens:
            return ""

        clauses = []
        for t in tokens:
            # Hyphenated or apostrophe words are wrapped in double quotes
            if "'" in t or "-" in t:
                clauses.append(f'"{t}"')
            elif len(t) >= 3 and not t.isdigit():
                clauses.append(f'("{t}" OR "{t}"*)')
            else:
                clauses.append(f'"{t}"')

        return " AND ".join(clauses)

    def search(
        self, query: str, mode: str = "auto", limit: int = 10, doc_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Unified canonical search API for KORZ library.
        Modes:
          - 'auto'  (default): exact phrase if in quotes, otherwise FTS5 BM25 with exact fallback
          - 'exact': case-insensitive literal substring search
          - 'fts'  : FTS5 full-text search with BM25 ranking
        Logs anonymous diagnostics to search_log for tuning & V2 design.
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        t0 = time.perf_counter()

        # If mode is exact, or auto with explicit quotes
        is_quoted = (clean_q.startswith('"') and clean_q.endswith('"')) or (clean_q.startswith('«') and clean_q.endswith('»'))
        if mode == "exact" or (mode == "auto" and is_quoted):
            phrase = clean_q[1:-1] if is_quoted else clean_q
            results = self._search_exact(phrase, limit=limit, doc_id=doc_id)
            if not results and mode == "auto":
                results = self._search_fts(clean_q, limit=limit, doc_id=doc_id)
        else:
            # FTS5 search
            results = self._search_fts(clean_q, limit=limit, doc_id=doc_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Record anonymous query diagnostic
        try:
            top_id = results[0].paragraph_id if results else None
            now_iso = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """
                INSERT INTO search_log (query, mode, result_count, top_result_id, latency_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean_q[:250], mode, len(results), top_id, round(elapsed_ms, 2), now_iso),
            )
        except Exception:
            pass  # Logging never disrupts search operations

        return results

    def _search_exact(self, phrase: str, limit: int = 10, doc_id: Optional[str] = None) -> List[SearchResult]:
        """Exact case-insensitive literal substring search."""
        clean_phrase = TextProcessor.normalize_apostrophes(phrase.strip())
        sql = """
        SELECT
            p.id,
            p.paragraph_key,
            p.document_id,
            d.title,
            d.author,
            p.page_number,
            p.paragraph_index,
            p.display_text
        FROM paragraphs p
        JOIN documents d ON d.id = p.document_id
        WHERE INSTR_CI(p.display_text, ?) = 1
        """
        params: List[Any] = [clean_phrase]

        if doc_id:
            sql += " AND p.document_id = ?"
            params.append(doc_id)

        sql += " ORDER BY p.document_id, p.page_number, p.paragraph_index LIMIT ?"
        params.append(limit)

        cur = self.conn.execute(sql, params)
        results: List[SearchResult] = []

        for row in cur.fetchall():
            text = row["display_text"]
            idx = text.lower().find(clean_phrase.lower())
            start = max(0, idx - 60)
            end = min(len(text), idx + len(clean_phrase) + 60)
            sub = text[start:end]
            if start > 0:
                sub = "..." + sub
            if end < len(text):
                sub = sub + "..."

            results.append(
                SearchResult(
                    paragraph_id=row["id"],
                    document_id=row["document_id"],
                    title=row["title"],
                    author=row["author"],
                    page=row["page_number"],
                    paragraph_index=row["paragraph_index"],
                    text=text,
                    snippet=sub,
                    score=1.0,
                    match_type="exact",
                )
            )

        return results

    def _search_fts(self, query: str, limit: int = 10, doc_id: Optional[str] = None) -> List[SearchResult]:
        """FTS5 search with BM25 ranking."""
        fts_match_query = self._build_fts_query(query)
        if not fts_match_query:
            return []

        sql = """
        SELECT
            p.id,
            p.paragraph_key,
            p.document_id,
            d.title,
            d.author,
            p.page_number,
            p.paragraph_index,
            p.display_text,
            bm25(paragraphs_fts) AS rank,
            snippet(paragraphs_fts, 0, '«', '»', '...', 20) AS snip
        FROM paragraphs_fts
        JOIN paragraphs p ON p.id = paragraphs_fts.rowid
        JOIN documents d ON d.id = p.document_id
        WHERE paragraphs_fts MATCH ?
        """
        params: List[Any] = [fts_match_query]

        if doc_id:
            sql += " AND p.document_id = ?"
            params.append(doc_id)

        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # Fallback to simple phrase or quoted terms on any FTS operational error
            safe_terms = " ".join(f'"{t}"' for t in re.findall(r"[\w]+", query, re.UNICODE))
            params[0] = safe_terms
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()

        # If strict conjunction had 0 hits, try OR ranking
        if not rows and len(re.findall(r"[\w]+", query, re.UNICODE)) > 1:
            tokens = re.findall(r"[\w]+(?:['\-][\w]+)*", TextProcessor.normalize_apostrophes(query), re.UNICODE)
            or_clauses = [f'"{t}"' if ("'" in t or "-" in t or len(t) < 3 or t.isdigit()) else f'("{t}" OR "{t}"*)' for t in tokens]
            params[0] = " OR ".join(or_clauses)
            cur = self.conn.execute(sql, params)
            rows = cur.fetchall()

        results: List[SearchResult] = []
        for r in rows:
            results.append(
                SearchResult(
                    paragraph_id=r["id"],
                    document_id=r["document_id"],
                    title=r["title"],
                    author=r["author"],
                    page=r["page_number"],
                    paragraph_index=r["paragraph_index"],
                    text=r["display_text"],
                    snippet=r["snip"] or r["display_text"][:160] + "...",
                    score=float(r["rank"]),
                    match_type="fts",
                )
            )

        return results

    def find(self, phrase: str, limit: int = 10, doc_id: Optional[str] = None) -> List[SearchResult]:
        """Convenience alias for search(phrase, mode='exact')."""
        return self.search(phrase, mode="exact", limit=limit, doc_id=doc_id)

    def get_context(self, paragraph_id: int, window: int = 2) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paragraph with surrounding context (W paragraphs before, W after).
        """
        cur = self.conn.execute(
            """
            SELECT p.*, d.title, d.author
            FROM paragraphs p
            JOIN documents d ON d.id = p.document_id
            WHERE p.id = ?
            """,
            (paragraph_id,),
        )
        target = cur.fetchone()
        if not target:
            return None

        doc_id = target["document_id"]
        target_id = target["id"]

        cur_prev = self.conn.execute(
            """
            SELECT id, page_number, paragraph_index, display_text
            FROM paragraphs
            WHERE document_id = ? AND id < ?
            ORDER BY id DESC LIMIT ?
            """,
            (doc_id, target_id, window),
        )
        prev_pars = list(reversed(cur_prev.fetchall()))

        cur_next = self.conn.execute(
            """
            SELECT id, page_number, paragraph_index, display_text
            FROM paragraphs
            WHERE document_id = ? AND id > ?
            ORDER BY id ASC LIMIT ?
            """,
            (doc_id, target_id, window),
        )
        next_pars = cur_next.fetchall()

        all_entries = []
        for p in prev_pars:
            all_entries.append(
                {
                    "id": p["id"],
                    "page": p["page_number"],
                    "paragraph_index": p["paragraph_index"],
                    "text": p["display_text"],
                    "is_target": False,
                }
            )

        all_entries.append(
            {
                "id": target["id"],
                "page": target["page_number"],
                "paragraph_index": target["paragraph_index"],
                "text": target["display_text"],
                "is_target": True,
            }
        )

        for p in next_pars:
            all_entries.append(
                {
                    "id": p["id"],
                    "page": p["page_number"],
                    "paragraph_index": p["paragraph_index"],
                    "text": p["display_text"],
                    "is_target": False,
                }
            )

        full_text = "\n\n".join(e["text"] for e in all_entries)

        return {
            "document_id": doc_id,
            "title": target["title"],
            "author": target["author"],
            "target_id": target["id"],
            "target_page": target["page_number"],
            "target_paragraph_index": target["paragraph_index"],
            "entries": all_entries,
            "combined_text": full_text,
        }

    def list_books(self) -> List[Dict[str, Any]]:
        """List all registered books in library."""
        cur = self.conn.execute("""
            SELECT
                id,
                title,
                author,
                edition,
                language,
                total_pages,
                total_paragraphs,
                total_chunks,
                source_sha256,
                created_at
            FROM documents
            ORDER BY title ASC
        """)
        return [dict(r) for r in cur.fetchall()]

    def get_book(self, doc_id_or_title: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific book by ID or title."""
        cur = self.conn.execute(
            """
            SELECT * FROM documents
            WHERE id = ? OR title = ? OR id LIKE ? OR title LIKE ?
            LIMIT 1
            """,
            (doc_id_or_title, doc_id_or_title, f"%{doc_id_or_title}%", f"%{doc_id_or_title}%"),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def remove_book(self, doc_id_or_title: str) -> bool:
        """
        Transactional deletion of a book from the library.
        Cascades to pages, paragraphs, chunks, and FTS5 triggers.
        """
        book = self.get_book(doc_id_or_title)
        if not book:
            return False

        doc_id = book["id"]
        now_iso = datetime.now(timezone.utc).isoformat()

        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.execute(
                """
                INSERT INTO ingest_history (source_sha256, document_id, action, status, details, created_at)
                VALUES (?, ?, 'REMOVE', 'SUCCESS', ?, ?)
                """,
                (book["source_sha256"], doc_id, f"Removed book '{book['title']}'", now_iso),
            )
            self.conn.execute("COMMIT;")
        except Exception:
            try:
                self.conn.execute("ROLLBACK;")
            except Exception:
                pass
            raise

        return True

    def status(self) -> Dict[str, Any]:
        """Return library health, counts, schema version, and database file size."""
        db_file = self.config.db_path
        db_size_bytes = db_file.stat().st_size if db_file.exists() else 0
        db_size_mb = round(db_size_bytes / (1024 * 1024), 2)

        cur_v = self.conn.execute("PRAGMA user_version;")
        schema_version = cur_v.fetchone()[0]

        cur_docs = self.conn.execute("SELECT COUNT(*) FROM documents")
        docs_count = cur_docs.fetchone()[0]

        cur_pages = self.conn.execute("SELECT COUNT(*) FROM pages")
        pages_count = cur_pages.fetchone()[0]

        cur_paras = self.conn.execute("SELECT COUNT(*) FROM paragraphs")
        paras_count = cur_paras.fetchone()[0]

        cur_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks")
        chunks_count = cur_chunks.fetchone()[0]

        return {
            "database_path": str(db_file),
            "database_size_mb": db_size_mb,
            "schema_version": schema_version,
            "total_documents": docs_count,
            "total_pages": pages_count,
            "total_paragraphs": paras_count,
            "total_chunks": chunks_count,
            "sources_dir": str(self.config.sources_dir),
            "backups_dir": str(self.config.backups_dir),
        }

    def backup(self, dest_path: Optional[Union[str, Path]] = None) -> str:
        """
        Atomic online backup of SQLite database using sqlite3.Connection.backup().
        Safe to run even while read/writes are happening in WAL mode.
        """
        self.config.backups_dir.mkdir(parents=True, exist_ok=True)
        if dest_path:
            target = Path(dest_path)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = self.config.backups_dir / f"korz_library_backup_{ts}.sqlite"

        target.parent.mkdir(parents=True, exist_ok=True)

        bck_conn = sqlite3.connect(str(target), isolation_level=None)
        try:
            self.conn.backup(bck_conn)
        finally:
            bck_conn.close()

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO ingest_history (source_sha256, document_id, action, status, details, created_at)
                VALUES (NULL, NULL, 'BACKUP', 'SUCCESS', ?, ?)
                """,
                (f"Backup created at {target}", now_iso),
            )

        return str(target)

    def restore(self, backup_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Atomically restore database from an existing backup file.
        Verifies backup integrity before applying.
        """
        source_bck = Path(backup_path).resolve()
        if not source_bck.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")

        # Verify backup integrity first
        test_conn = sqlite3.connect(str(source_bck), isolation_level=None)
        try:
            cur = test_conn.execute("PRAGMA integrity_check;")
            res = cur.fetchall()
            if not res or res[0][0] != "ok":
                raise ValueError(f"Corrupted backup file: {backup_path} ({res})")
        finally:
            test_conn.close()

        # Atomic restore into current active database connection
        bck_conn = sqlite3.connect(str(source_bck), isolation_level=None)
        try:
            bck_conn.backup(self.conn)
        finally:
            bck_conn.close()

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO ingest_history (source_sha256, document_id, action, status, details, created_at)
                VALUES (NULL, NULL, 'RESTORE', 'SUCCESS', ?, ?)
                """,
                (f"Restored database from {source_bck}", now_iso),
            )

        return self.status()

    def admin_verify(self) -> Dict[str, Any]:
        """Verify database and FTS5 integrity and consistency."""
        cur = self.conn.execute("PRAGMA integrity_check;")
        sqlite_check = [r[0] for r in cur.fetchall()]
        sqlite_ok = sqlite_check == ["ok"]

        fts_ok = False
        fts_error = None
        try:
            self.conn.execute("INSERT INTO paragraphs_fts(paragraphs_fts) VALUES('integrity-check');")
            fts_ok = True
        except Exception as e:
            fts_error = str(e)

        cur_p = self.conn.execute("SELECT COUNT(*) FROM paragraphs")
        p_count = cur_p.fetchone()[0]

        cur_fts = self.conn.execute("SELECT COUNT(*) FROM paragraphs_fts")
        fts_count = cur_fts.fetchone()[0]

        parity_ok = p_count == fts_count

        return {
            "sqlite_integrity_check": "ok" if sqlite_ok else str(sqlite_check),
            "fts5_integrity_check": "ok" if fts_ok else fts_error,
            "paragraphs_count": p_count,
            "fts_indexed_count": fts_count,
            "parity_ok": parity_ok,
            "all_healthy": sqlite_ok and fts_ok and parity_ok,
        }

    def admin_reindex(self) -> Dict[str, Any]:
        """Rebuild the FTS5 virtual table."""
        with self.conn:
            self.conn.execute("INSERT INTO paragraphs_fts(paragraphs_fts) VALUES('rebuild');")

        cur_fts = self.conn.execute("SELECT COUNT(*) FROM paragraphs_fts")
        fts_count = cur_fts.fetchone()[0]

        return {"status": "REBUILT", "total_indexed_paragraphs": fts_count}

    def get_search_stats(self) -> Dict[str, Any]:
        """Return search diagnostics from search_log."""
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM search_log;")
            total_searches = cur.fetchone()[0]

            cur_zero = self.conn.execute("SELECT COUNT(*) FROM search_log WHERE result_count = 0;")
            zero_hits = cur_zero.fetchone()[0]

            cur_avg = self.conn.execute("SELECT AVG(latency_ms) FROM search_log;")
            avg_lat = cur_avg.fetchone()[0] or 0.0

            return {
                "total_searches": total_searches,
                "zero_hit_searches": zero_hits,
                "hit_rate_pct": round(100.0 * (total_searches - zero_hits) / total_searches, 1) if total_searches > 0 else 100.0,
                "avg_latency_ms": round(avg_lat, 2),
            }
        except Exception:
            return {"total_searches": 0, "zero_hit_searches": 0, "hit_rate_pct": 100.0, "avg_latency_ms": 0.0}

    def get_failed_searches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return queries that returned 0 results for tuning search & designing V2."""
        try:
            cur = self.conn.execute(
                """
                SELECT query, mode, created_at
                FROM search_log
                WHERE result_count = 0
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

