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

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("korz")

# Default directories
DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "korz_library.sqlite"
DEFAULT_SOURCES_DIR = DEFAULT_DATA_DIR / "sources"
DEFAULT_BACKUPS_DIR = DEFAULT_DATA_DIR / "backups"

CURRENT_SCHEMA_VERSION = 2


@dataclass
class KorzConfig:
    data_dir: Path = DEFAULT_DATA_DIR
    db_path: Path = DEFAULT_DB_PATH
    sources_dir: Path = DEFAULT_SOURCES_DIR
    backups_dir: Path = DEFAULT_BACKUPS_DIR
    max_chunk_chars: int = 1400
    target_chunk_chars: int = 1200
    target_paragraph_chars: int = 1400
    semantic_enabled: bool = field(
        default_factory=lambda: os.environ.get("KORZ_SEMANTIC_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    )
    shadow_enabled: bool = field(
        default_factory=lambda: os.environ.get("KORZ_SHADOW_ENABLED", "0").strip().lower() in ("1", "true", "yes")
    )
    fts_engine_version: str = "sqlite-fts5-bm25-v1.1"

    @classmethod
    def from_env(cls) -> "KorzConfig":
        data_dir = Path(os.environ.get("KORZ_DATA_DIR", str(DEFAULT_DATA_DIR)))
        db_path = Path(os.environ.get("KORZ_DB_PATH", str(data_dir / "korz_library.sqlite")))
        sources_dir = data_dir / "sources"
        backups_dir = data_dir / "backups"
        semantic_enabled = os.environ.get("KORZ_SEMANTIC_ENABLED", "0").strip().lower() in ("1", "true", "yes")
        shadow_enabled = os.environ.get("KORZ_SHADOW_ENABLED", "0").strip().lower() in ("1", "true", "yes")
        return cls(
            data_dir=data_dir,
            db_path=db_path,
            sources_dir=sources_dir,
            backups_dir=backups_dir,
            semantic_enabled=semantic_enabled,
            shadow_enabled=shadow_enabled,
        )


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
    match_type: str = "fts"  # "exact" | "fts" | "semantic" | "hybrid"

    # Backward compatibility aliases
    @property
    def page_number(self) -> int:
        return self.page

    @property
    def display_text(self) -> str:
        return self.text


@dataclass(frozen=True)
class EmbeddingContract:
    """Canonical specification of an embedding model configuration."""
    model_name: str
    model_revision: Optional[str] = None
    dimension: int = 768
    dtype: str = "float32"
    normalization: str = "l2"
    pooling: Optional[str] = "mean"
    query_template: str = "{text}"
    passage_template: str = "{text}"
    document_template: Optional[str] = None
    model_max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.document_template is not None and self.passage_template == "{text}":
            object.__setattr__(self, "passage_template", self.document_template)
        elif self.document_template is None:
            object.__setattr__(self, "document_template", self.passage_template)

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "dimension": int(self.dimension),
            "dtype": self.dtype,
            "normalization": self.normalization.lower(),
            "pooling": self.pooling.lower() if self.pooling else None,
            "query_template": self.query_template,
            "passage_template": self.passage_template,
            "model_max_tokens": int(self.model_max_tokens),
        }

    def compute_config_sha256(self) -> str:
        data = self.to_canonical_dict()
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def generate_model_id(self) -> str:
        clean_name = re.sub(r"[^a-zA-Z0-9]+", "-", self.model_name.split("/")[-1].lower()).strip("-")
        config_hash = self.compute_config_sha256()[:12]
        return f"emb-{clean_name}-{config_hash}"


DEFAULT_E5_BASE_CONTRACT = EmbeddingContract(
    model_name="intfloat/multilingual-e5-base",
    dimension=768,
    dtype="float32",
    normalization="l2",
    pooling="mean",
    query_template="query: {text}",
    passage_template="passage: {text}",
    model_max_tokens=512,
)


class EmbeddingBackend(ABC):
    """Abstract interface for local embedding model inference."""

    @abstractmethod
    def encode_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode list of passage texts into an (N, D) float32 L2-normalized numpy array."""
        pass

    @abstractmethod
    def encode_queries(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode list of query texts into an (N, D) float32 L2-normalized numpy array."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens according to model tokenizer."""
        pass


class FastEmbedBackend(EmbeddingBackend):
    """Fast local CPU ONNX embedding backend via FastEmbed."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import PoolingType, ModelSource
        except ImportError as e:
            raise ImportError("fastembed is not installed. Run: pip install fastembed") from e

        self.model_name = model_name
        supported = [m["model"] for m in TextEmbedding.list_supported_models()]
        if model_name == "intfloat/multilingual-e5-base" and model_name not in supported:
            try:
                TextEmbedding.add_custom_model(
                    model="intfloat/multilingual-e5-base",
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf="Xenova/multilingual-e5-base"),
                    dim=768,
                    model_file="onnx/model.onnx",
                    description="Multilingual E5 base model via Xenova ONNX",
                )
            except Exception:
                pass

        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def encode_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embs = list(self._model.embed(texts, batch_size=batch_size))
        arr = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def encode_queries(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embs = list(self._model.embed(texts, batch_size=batch_size))
        arr = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def count_tokens(self, text: str) -> int:
        if hasattr(self._model, "token_count"):
            try:
                return int(self._model.token_count(text))
            except Exception:
                pass
        return max(1, len(text) // 4)


class MockEmbeddingBackend(EmbeddingBackend):
    """Deterministic, fast mock backend for testing without external models."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def encode_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self._pseudo_vectors(texts)

    def encode_queries(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self._pseudo_vectors(texts)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _pseudo_vectors(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vecs = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:4], "little")
            rng = np.random.default_rng(seed)
            arr = rng.standard_normal(self.dimension).astype(np.float32)
            norm = float(np.linalg.norm(arr))
            if norm == 0 or not np.isfinite(norm):
                arr[0] = 1.0
                norm = 1.0
            vecs.append(arr / norm)
        return np.vstack(vecs).astype(np.float32)


def get_embedding_backend(model_name: str, cache_dir: Optional[str] = None) -> EmbeddingBackend:
    if model_name.startswith("mock://") or model_name == "mock":
        return MockEmbeddingBackend()
    return FastEmbedBackend(model_name, cache_dir=cache_dir)


@dataclass(frozen=True)
class SemanticIndex:
    """Immutable snapshot of the in-memory semantic vector index."""
    model_id: str
    matrix: np.ndarray          # shape (N, D), float32, L2-normalized
    chunk_ids: np.ndarray       # shape (N,), int64
    paragraph_ids: np.ndarray   # shape (N,), int64 (paragraph_start_id)
    doc_ids: np.ndarray         # shape (N,), object (document_id)
    index_sha: str = ""         # SHA-256 fingerprint of the index matrix and structure

    @property
    def total_vectors(self) -> int:
        return len(self.chunk_ids)

    @property
    def dimension(self) -> int:
        return self.matrix.shape[1] if self.matrix.ndim == 2 else 0

    @property
    def memory_size_bytes(self) -> int:
        return int(self.matrix.nbytes + self.chunk_ids.nbytes + self.paragraph_ids.nbytes + self.doc_ids.nbytes)

    @property
    def memory_size_mb(self) -> float:
        return round(self.memory_size_bytes / (1024 * 1024), 2)


def build_embedding_input(
    raw_text: str,
    template: str = "{text}",
    model_max_tokens: int = 512,
    backend: Optional[EmbeddingBackend] = None,
) -> str:
    """
    Build canonical embedding input with deterministic whitespace normalization
    and strict token limit compliance.
    """
    normalized_raw = " ".join(raw_text.split())
    formatted = template.format(text=normalized_raw) if "{text}" in template else normalized_raw

    if backend is None or model_max_tokens <= 0:
        return formatted

    tokens_count = backend.count_tokens(formatted)
    if tokens_count <= model_max_tokens:
        return formatted

    # Deterministic character-based binary search truncation so formatted text <= model_max_tokens
    low, high = 0, len(normalized_raw)
    best_formatted = formatted
    while low <= high:
        mid = (low + high) // 2
        cand_raw = normalized_raw[:mid].rstrip()
        cand_formatted = template.format(text=cand_raw) if "{text}" in template else cand_raw
        if backend.count_tokens(cand_formatted) <= model_max_tokens:
            best_formatted = cand_formatted
            low = mid + 1
        else:
            high = mid - 1

    return best_formatted


def build_embedding_text(text: str, template: str = "{text}") -> str:
    """Build canonical embedding text with deterministic whitespace normalization."""
    normalized_text = " ".join(text.split())
    if "{text}" in template:
        return template.format(text=normalized_text)
    return normalized_text


def compute_text_sha256(embedding_text: str) -> str:
    """Compute deterministic SHA-256 for chunk embedding caching."""
    return hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()


def vector_to_blob(vector: Any, expected_dim: int, normalize: bool = True) -> bytes:
    """
    Convert a vector to float32 little-endian BLOB with optional L2 normalization.
    Ensures vector is 1D, finite, and strictly matches expected dimension.
    """
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise ValueError(f"Vector dimension mismatch: expected ({expected_dim},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vector contains non-finite values (NaN or Inf)")

    if normalize:
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm

    return arr.tobytes(order="C")


def blob_to_vector(blob: bytes, expected_dim: int) -> np.ndarray:
    """
    Convert BLOB back to 1D float32 numpy array.
    Validates byte length and finiteness.
    """
    expected_bytes = expected_dim * 4  # 4 bytes per float32
    if len(blob) != expected_bytes:
        raise ValueError(
            f"BLOB size mismatch: expected {expected_bytes} bytes for dim {expected_dim}, got {len(blob)} bytes"
        )

    arr = np.frombuffer(blob, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Deserialized vector contains non-finite values")
    return arr


def serialize_compact_candidates(results: List["SearchResult"], limit: int = 10) -> str:
    """Format compact candidate JSON without bulky text payloads."""
    entries = []
    for rank, r in enumerate(results[:limit], 1):
        entries.append({
            "paragraph_id": r.paragraph_id,
            "document_id": r.document_id,
            "page": r.page,
            "rank": rank,
            "score": round(float(r.score), 4),
        })
    return json.dumps(entries, ensure_ascii=False)


def compute_simulated_rrf(
    fts_results: List["SearchResult"],
    semantic_results: List["SearchResult"],
    k: int = 60,
    top_limit: int = 10,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Compute Reciprocal Rank Fusion (RRF) simulation with fixed k parameter (default: 60).
    RRF Score: sum(1.0 / (k + rank_m)) for m in {fts, sem}.
    Returns:
      (simulated_rrf_top, overlap_at_5, overlap_at_10, top1_match)
    """
    fts_top10 = fts_results[:10]
    sem_top10 = semantic_results[:10]

    fts_ranks: Dict[int, Tuple[int, "SearchResult"]] = {r.paragraph_id: (idx + 1, r) for idx, r in enumerate(fts_top10)}
    sem_ranks: Dict[int, Tuple[int, "SearchResult"]] = {r.paragraph_id: (idx + 1, r) for idx, r in enumerate(sem_top10)}

    fts_pids_5 = set(r.paragraph_id for r in fts_top10[:5])
    sem_pids_5 = set(r.paragraph_id for r in sem_top10[:5])
    overlap_at_5 = len(fts_pids_5 & sem_pids_5)

    fts_pids_10 = set(fts_ranks.keys())
    sem_pids_10 = set(sem_ranks.keys())
    overlap_at_10 = len(fts_pids_10 & sem_pids_10)

    top1_match = 1 if (fts_results and semantic_results and fts_results[0].paragraph_id == semantic_results[0].paragraph_id) else 0

    all_pids = set(fts_ranks.keys()) | set(sem_ranks.keys())
    scored_items = []
    for pid in all_pids:
        rrf_score = 0.0
        rank_fts = None
        rank_sem = None
        sample_r = None

        if pid in fts_ranks:
            rank_fts, sample_r = fts_ranks[pid]
            rrf_score += 1.0 / (k + rank_fts)
        if pid in sem_ranks:
            rank_sem, sem_r = sem_ranks[pid]
            sample_r = sample_r or sem_r
            rrf_score += 1.0 / (k + rank_sem)

        rank_delta = (rank_fts - rank_sem) if (rank_fts is not None and rank_sem is not None) else None

        scored_items.append({
            "paragraph_id": pid,
            "document_id": sample_r.document_id if sample_r else "",
            "page": sample_r.page if sample_r else 0,
            "score": round(rrf_score, 6),
            "rank_fts": rank_fts,
            "rank_sem": rank_sem,
            "rank_delta": rank_delta,
        })

    # Sort descending by rrf score
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(scored_items[:top_limit], 1):
        item["rank"] = idx

    return scored_items[:top_limit], overlap_at_5, overlap_at_10, top1_match


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
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.create_function("INSTR_CI", 2, _instr_ci)
    conn.create_function("REGEXP", 2, _regexp)
    return conn


def init_database(conn: sqlite3.Connection, backups_dir: Optional[Path] = None) -> None:
    """Initialize database tables, migrations, FTS5 index, and triggers."""
    cur = conn.execute("PRAGMA user_version;")
    version_row = cur.fetchone()
    current_version = version_row[0] if version_row else 0

    if current_version < 1:
        _apply_migration_v1(conn)
    if current_version < 2:
        _apply_migration_v2(conn, backups_dir=backups_dir)

    _ensure_shadow_tables(conn)


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


def _apply_migration_v2(conn: sqlite3.Connection, backups_dir: Optional[Path] = None) -> None:
    """
    Apply additive Schema v2 migration:
      - embedding_models
      - chunk_embeddings (BLOB float32 vectors)
      - semantic_state
    Pre-migration safety: checks integrity and optionally creates pre-migration backup.
    Post-migration safety: verifies integrity.
    """
    # Pre-migration safety checks
    try:
        cur_check = conn.execute("PRAGMA quick_check;")
        res = cur_check.fetchone()[0]
        if res != "ok":
            logger.warning(f"Pre-migration quick_check returned: {res}")
    except Exception as e:
        logger.warning(f"Pre-migration check exception: {e}")

    # Optional hot snapshot backup before schema upgrade
    if backups_dir:
        try:
            backups_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_file = backups_dir / f"korz_library_pre_v2_migration_{ts}.sqlite"
            b_conn = sqlite3.connect(str(backup_file))
            try:
                conn.backup(b_conn)
                logger.info(f"Pre-migration v2 backup created: {backup_file}")
            finally:
                b_conn.close()
        except Exception as e:
            logger.warning(f"Could not create pre-migration backup: {e}")

    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute("BEGIN IMMEDIATE;")
    try:
        # Table 1: Registered embedding models
        conn.execute("""
        CREATE TABLE IF NOT EXISTS embedding_models (
            model_id TEXT PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_revision TEXT,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            dtype TEXT NOT NULL DEFAULT 'float32',
            normalization TEXT NOT NULL,
            pooling TEXT,
            query_template TEXT,
            passage_template TEXT,
            config_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        # Table 2: Chunk embeddings
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            text_sha256 TEXT NOT NULL,
            vector BLOB NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (chunk_id, model_id),
            FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
            FOREIGN KEY (model_id) REFERENCES embedding_models(model_id) ON DELETE RESTRICT
        );
        """)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model ON chunk_embeddings(model_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_sha ON chunk_embeddings(text_sha256);")

        # Table 3: Document semantic indexing progress
        conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_state (
            document_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('PENDING', 'INDEXING', 'READY', 'STALE', 'FAILED')),
            total_chunks INTEGER NOT NULL DEFAULT 0,
            embedded_chunks INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (document_id, model_id),
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (model_id) REFERENCES embedding_models(model_id) ON DELETE CASCADE
        );
        """)

        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at, description) VALUES (?, ?, ?)",
            (2, now_iso, "v2 additive semantic schema: embedding_models, chunk_embeddings, semantic_state"),
        )
        conn.execute("PRAGMA user_version = 2;")
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise

    # Post-migration sanity check
    try:
        cur_post = conn.execute("PRAGMA quick_check;")
        res_post = cur_post.fetchone()[0]
        if res_post != "ok":
            raise RuntimeError(f"Post-migration quick_check failed: {res_post}")
    except Exception as e:
        logger.error(f"Post-migration check error: {e}")
        raise


def _ensure_shadow_tables(conn: sqlite3.Connection) -> None:
    """
    Ensure additive Phase V2C shadow evaluation & feedback tables exist:
      - shadow_search_log (captures FTS vs Semantic dual retrieval, simulated RRF@60, provenance)
      - search_feedback (captures explicit human relevance signal: thumbs up/down, target fragment)
    """
    conn.execute("""
    CREATE TABLE IF NOT EXISTS shadow_search_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        doc_id TEXT,
        fts_engine_version TEXT,
        semantic_model_id TEXT NOT NULL,
        semantic_index_sha TEXT,
        latency_fts_ms REAL,
        latency_semantic_ms REAL,
        fts_count INTEGER NOT NULL,
        semantic_count INTEGER NOT NULL,
        overlap_at_5 INTEGER NOT NULL,
        overlap_at_10 INTEGER NOT NULL,
        top1_match INTEGER NOT NULL,
        fts_top_json TEXT NOT NULL,
        semantic_top_json TEXT NOT NULL,
        simulated_rrf_json TEXT,
        shadow_status TEXT NOT NULL,
        error_message TEXT,
        created_at TEXT NOT NULL
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_query ON shadow_search_log(query);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_created ON shadow_search_log(created_at);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shadow_model_sha ON shadow_search_log(semantic_model_id, semantic_index_sha);")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS search_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shadow_log_id INTEGER,
        query TEXT NOT NULL,
        target_paragraph_id INTEGER,
        target_rank INTEGER,
        feedback_type TEXT NOT NULL,
        user_id TEXT,
        comment TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (shadow_log_id) REFERENCES shadow_search_log(id) ON DELETE SET NULL
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_query ON search_feedback(query);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created ON search_feedback(created_at);")


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
        init_database(self.conn, backups_dir=self.config.backups_dir)
        self._semantic_index: Optional[SemanticIndex] = None
        self._active_model_id: Optional[str] = None
        self._backends: Dict[str, EmbeddingBackend] = {}

    def get_backend(self, model_name: str) -> EmbeddingBackend:
        """Get or initialize embedding backend for model_name."""
        if model_name not in self._backends:
            self._backends[model_name] = get_embedding_backend(model_name)
        return self._backends[model_name]

    def set_backend(self, model_name: str, backend: EmbeddingBackend) -> None:
        """Inject or override an embedding backend (useful for testing and mocks)."""
        self._backends[model_name] = backend

    def get_default_model_id(self) -> Optional[str]:
        """Return the default/primary embedding model ID if any model is registered."""
        cur = self.conn.execute("SELECT model_id FROM embedding_models ORDER BY created_at ASC LIMIT 1")
        row = cur.fetchone()
        return row["model_id"] if row else None

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
        self,
        query: str,
        mode: str = "auto",
        limit: int = 10,
        doc_id: Optional[str] = None,
        shadow: Optional[bool] = None,
    ) -> List[SearchResult]:
        """
        Unified canonical search API for KORZ library.
        Modes:
          - 'auto'  (default): exact phrase if in quotes, otherwise FTS5 BM25 with exact fallback
          - 'exact': case-insensitive literal substring search
          - 'fts'  : FTS5 full-text search with BM25 ranking
          - 'semantic': dense vector retrieval
        Logs anonymous diagnostics to search_log and triggers shadow evaluation if enabled.
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        if mode == "semantic":
            t0 = time.perf_counter()
            results = self._search_semantic(clean_q, limit=limit, doc_id=doc_id)
            elapsed_ms = (time.perf_counter() - t0) * 1000
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
                pass
            return results

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

        # Phase V2C: Synchronous shadow evaluation if requested or shadow_enabled in config
        should_shadow = shadow if shadow is not None else self.config.shadow_enabled
        if should_shadow and mode in ("auto", "fts", "exact"):
            try:
                runner = ShadowRunner(self)
                runner.evaluate_sync(
                    query=clean_q,
                    fts_results=results,
                    doc_id=doc_id,
                    latency_fts_ms=elapsed_ms,
                )
            except Exception as e:
                logger.warning(f"Shadow evaluation failed: {e}")

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

        semantic_info = self.get_semantic_status() if schema_version >= 2 else None

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
            "semantic": semantic_info,
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

    # =========================================================================
    # Phase V2A: Semantic Infrastructure & Embedding Storage APIs
    # =========================================================================

    def register_embedding_model(self, contract: EmbeddingContract) -> str:
        """Register an embedding model contract in embedding_models table."""
        model_id = contract.generate_model_id()
        config_sha = contract.compute_config_sha256()
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO embedding_models (
                    model_id, model_name, model_revision, dimension, dtype,
                    normalization, pooling, query_template, passage_template,
                    config_sha256, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    contract.model_name,
                    contract.model_revision,
                    contract.dimension,
                    contract.dtype,
                    contract.normalization,
                    contract.pooling,
                    contract.query_template,
                    contract.passage_template,
                    config_sha,
                    now_iso,
                ),
            )
        return model_id

    def get_embedding_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch details of a registered embedding model."""
        cur = self.conn.execute("SELECT * FROM embedding_models WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_embedding_models(self) -> List[Dict[str, Any]]:
        """List all registered embedding models."""
        cur = self.conn.execute("SELECT * FROM embedding_models ORDER BY created_at ASC")
        return [dict(r) for r in cur.fetchall()]

    def delete_embedding_model(self, model_id: str) -> None:
        """
        Delete an embedding model.
        Raises sqlite3.IntegrityError if vectors exist referencing this model (ON DELETE RESTRICT).
        """
        with self.conn:
            self.conn.execute("DELETE FROM embedding_models WHERE model_id = ?", (model_id,))

    def save_chunk_embedding(self, chunk_id: int, model_id: str, vector: Any, text_sha256: str) -> None:
        """Save a float32 vector blob for a chunk."""
        model = self.get_embedding_model(model_id)
        if not model:
            raise ValueError(f"Unknown embedding model: {model_id}")

        dim = model["dimension"]
        should_norm = model["normalization"].lower() == "l2"
        blob = vector_to_blob(vector, expected_dim=dim, normalize=should_norm)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO chunk_embeddings (chunk_id, model_id, text_sha256, vector, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, model_id, text_sha256, blob, now_iso),
            )

    def get_chunk_embedding(self, chunk_id: int, model_id: str) -> Optional[np.ndarray]:
        """Fetch and deserialize a chunk's float32 embedding vector."""
        model = self.get_embedding_model(model_id)
        if not model:
            return None

        cur = self.conn.execute(
            "SELECT vector FROM chunk_embeddings WHERE chunk_id = ? AND model_id = ?",
            (chunk_id, model_id),
        )
        row = cur.fetchone()
        if not row:
            return None

        return blob_to_vector(row["vector"], expected_dim=model["dimension"])

    def get_chunk_embeddings_for_model(self, model_id: str) -> Tuple[List[int], np.ndarray]:
        """
        Load all chunk embeddings for a model into an in-memory NumPy matrix (N, D).
        Returns (chunk_ids, matrix).
        """
        model = self.get_embedding_model(model_id)
        if not model:
            raise ValueError(f"Unknown embedding model: {model_id}")

        dim = model["dimension"]
        cur = self.conn.execute(
            "SELECT chunk_id, vector FROM chunk_embeddings WHERE model_id = ? ORDER BY chunk_id ASC",
            (model_id,),
        )
        rows = cur.fetchall()
        if not rows:
            return [], np.empty((0, dim), dtype=np.float32)

        chunk_ids = [r["chunk_id"] for r in rows]
        vectors = [blob_to_vector(r["vector"], expected_dim=dim) for r in rows]
        matrix = np.vstack(vectors).astype(np.float32)
        return chunk_ids, matrix

    def update_semantic_state(
        self,
        document_id: str,
        model_id: str,
        status: str,
        embedded_chunks: int,
        total_chunks: int,
        last_error: Optional[str] = None,
    ) -> None:
        """Update or insert semantic state for a document and model."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO semantic_state (
                    document_id, model_id, status, embedded_chunks, total_chunks, last_error, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (document_id, model_id, status, embedded_chunks, total_chunks, last_error, now_iso),
            )

    def get_semantic_status(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Return comprehensive status of the semantic infrastructure."""
        models = self.list_embedding_models()
        active_model_id = model_id or (models[0]["model_id"] if models else None)

        cur_tot_docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        cur_tot_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        embedded_count = 0
        ready_docs = 0
        if active_model_id:
            cur_emb = self.conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = ?", (active_model_id,)
            )
            embedded_count = cur_emb.fetchone()[0]

            cur_ready = self.conn.execute(
                "SELECT COUNT(*) FROM semantic_state WHERE model_id = ? AND status = 'READY'", (active_model_id,)
            )
            ready_docs = cur_ready.fetchone()[0]

        return {
            "schema_version": 2,
            "semantic_enabled": getattr(self.config, "semantic_enabled", False),
            "models_registered": len(models),
            "active_model_id": active_model_id,
            "models": models,
            "total_documents": cur_tot_docs,
            "documents_ready": ready_docs,
            "ready_documents": ready_docs,
            "total_chunks": cur_tot_chunks,
            "chunks_embedded": embedded_count,
            "fts5_baseline": "READY",
        }

    def load_semantic_index(self, model_id: Optional[str] = None) -> SemanticIndex:
        """
        Load all chunk embeddings for model_id from SQLite into an immutable in-memory SemanticIndex.
        Deterministic ordering by chunk_id ensures stable row mapping:
        matrix[row] <-> chunk_ids[row] <-> paragraph_ids[row] <-> doc_ids[row].
        """
        target_model = model_id or self._active_model_id or self.get_default_model_id()
        if not target_model:
            raise RuntimeError("No embedding model registered in KORZ library.")

        model_info = self.get_embedding_model(target_model)
        if not model_info:
            raise ValueError(f"Embedding model '{target_model}' not found in library.")

        dim = model_info["dimension"]
        cur = self.conn.execute(
            """
            SELECT ce.chunk_id, ce.vector, c.paragraph_start_id, c.document_id
            FROM chunk_embeddings ce
            JOIN chunks c ON ce.chunk_id = c.id
            WHERE ce.model_id = ?
            ORDER BY ce.chunk_id ASC
            """,
            (target_model,),
        )
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError(f"No embeddings found for model '{target_model}'. Run 'python korz.py semantic build'.")

        chunk_ids_list: List[int] = []
        paragraph_ids_list: List[int] = []
        doc_ids_list: List[str] = []
        vectors_list: List[np.ndarray] = []

        for r in rows:
            chunk_ids_list.append(r["chunk_id"])
            paragraph_ids_list.append(r["paragraph_start_id"])
            doc_ids_list.append(r["document_id"])
            vectors_list.append(blob_to_vector(r["vector"], dim))

        matrix = np.vstack(vectors_list).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        # Fingerprint the matrix & chunk structure
        hasher = hashlib.sha256()
        hasher.update(matrix.tobytes()[:65536])
        hasher.update(f"{target_model}:{len(chunk_ids_list)}:{chunk_ids_list[0] if chunk_ids_list else 0}".encode("utf-8"))
        idx_sha = hasher.hexdigest()[:16]

        index = SemanticIndex(
            model_id=target_model,
            matrix=matrix,
            chunk_ids=np.array(chunk_ids_list, dtype=np.int64),
            paragraph_ids=np.array(paragraph_ids_list, dtype=np.int64),
            doc_ids=np.array(doc_ids_list, dtype=object),
            index_sha=idx_sha,
        )

        self._semantic_index = index
        self._active_model_id = target_model
        return index

    def build_semantic_index(
        self,
        model_id: Optional[str] = None,
        backend: Optional[EmbeddingBackend] = None,
        batch_size: int = 32,
        force: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Resume-safe offline semantic build with short transactional batch commits.
        Transitions semantic_state from INDEXING to READY only after final verification.
        """
        # 1. Resolve or auto-register model
        target_model = model_id
        if target_model is None:
            target_model = self.get_default_model_id()
            if not target_model:
                target_model = self.register_embedding_model(DEFAULT_E5_BASE_CONTRACT)

        model_info = self.get_embedding_model(target_model)
        if not model_info:
            for m in self.list_embedding_models():
                if m["model_name"] == target_model:
                    target_model = m["model_id"]
                    model_info = m
                    break
            if not model_info:
                if "e5-base" in str(target_model).lower():
                    target_model = self.register_embedding_model(DEFAULT_E5_BASE_CONTRACT)
                    model_info = self.get_embedding_model(target_model)
                else:
                    raise ValueError(f"Model '{target_model}' not registered in library.")

        dim = model_info["dimension"]
        passage_tpl = model_info.get("passage_template") or model_info.get("document_template") or "passage: {text}"
        model_max_tokens = model_info.get("model_max_tokens", 512)

        if backend is None:
            backend = self.get_backend(model_info["model_name"])

        # 2. Query all chunks
        cur = self.conn.execute(
            """
            SELECT c.id, c.document_id, c.paragraph_start_id, c.display_text
            FROM chunks c
            ORDER BY c.document_id, c.id ASC
            """
        )
        all_chunks = cur.fetchall()
        total_chunks = len(all_chunks)
        if total_chunks == 0:
            return {"model_id": target_model, "total_chunks": 0, "embedded": 0, "cached": 0, "failed": 0, "ready": False}

        # 3. Transition document states to INDEXING
        doc_ids = sorted(list({c["document_id"] for c in all_chunks}))
        for d_id in doc_ids:
            cur_d = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (d_id,))
            doc_total = cur_d.fetchone()[0]
            self.update_semantic_state(d_id, target_model, status="INDEXING", embedded_chunks=0, total_chunks=doc_total)

        # 4. Check existing embeddings for resume-safety
        cur_existing = self.conn.execute(
            "SELECT chunk_id, text_sha256 FROM chunk_embeddings WHERE model_id = ?",
            (target_model,),
        )
        existing_map = {r["chunk_id"]: r["text_sha256"] for r in cur_existing.fetchall()}

        cached_chunks: List[int] = []
        to_embed: List[Tuple[int, str, str, str]] = []

        for c in all_chunks:
            c_id = c["id"]
            d_id = c["document_id"]
            raw_text = c["display_text"]
            emb_input = build_embedding_input(raw_text, passage_tpl, model_max_tokens, backend)
            sha = compute_text_sha256(emb_input)

            if not force and c_id in existing_map and existing_map[c_id] == sha:
                cached_chunks.append(c_id)
            else:
                to_embed.append((c_id, d_id, emb_input, sha))

        if progress_callback:
            progress_callback({
                "status": "indexing",
                "total": total_chunks,
                "cached": len(cached_chunks),
                "to_embed": len(to_embed),
                "embedded": 0,
            })

        # 5. Batch embedding generation with short transactional commits
        embedded_count = 0
        failed_count = 0

        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            texts = [item[2] for item in batch]
            try:
                vectors = backend.encode_passages(texts, batch_size=len(texts))
            except Exception as e:
                failed_count += len(batch)
                logger.error(f"Error embedding batch starting at {i}: {e}")
                continue

            now_iso = datetime.now(timezone.utc).isoformat()
            with self.conn:
                for (c_id, d_id, _, sha), vec in zip(batch, vectors):
                    blob = vector_to_blob(vec, dim, normalize=True)
                    self.conn.execute(
                        """
                        INSERT INTO chunk_embeddings (chunk_id, model_id, text_sha256, vector, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(chunk_id, model_id) DO UPDATE SET
                            text_sha256 = excluded.text_sha256,
                            vector = excluded.vector,
                            created_at = excluded.created_at
                        """,
                        (c_id, target_model, sha, blob, now_iso),
                    )

            embedded_count += len(batch)
            if progress_callback:
                progress_callback({
                    "status": "indexing",
                    "total": total_chunks,
                    "cached": len(cached_chunks),
                    "to_embed": len(to_embed),
                    "embedded": embedded_count,
                })

        # 6. Verification and Status Transition
        integrity = self.verify_semantic_integrity(target_model)
        is_ready = integrity["all_healthy"] and integrity["missing_embeddings"] == 0 and failed_count == 0

        for d_id in doc_ids:
            cur_d = self.conn.execute(
                """
                SELECT COUNT(ce.chunk_id)
                FROM chunks c
                JOIN chunk_embeddings ce ON ce.chunk_id = c.id AND ce.model_id = ?
                WHERE c.document_id = ?
                """,
                (target_model, d_id),
            )
            emb_for_doc = cur_d.fetchone()[0]
            cur_tot = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE document_id = ?", (d_id,))
            tot_for_doc = cur_tot.fetchone()[0]

            new_status = "READY" if (is_ready and emb_for_doc == tot_for_doc) else "FAILED"
            err = None if new_status == "READY" else "Integrity check failed"
            self.update_semantic_state(
                d_id, target_model, status=new_status, embedded_chunks=emb_for_doc, total_chunks=tot_for_doc, last_error=err
            )

        if is_ready:
            self.load_semantic_index(target_model)

        return {
            "model_id": target_model,
            "total_chunks": total_chunks,
            "cached": len(cached_chunks),
            "embedded": embedded_count,
            "failed": failed_count,
            "ready": is_ready,
        }

    def _search_semantic(
        self,
        query: str,
        limit: int = 10,
        doc_id: Optional[str] = None,
        model_id: Optional[str] = None,
        internal_call: bool = False,
    ) -> List[SearchResult]:
        """
        Execute dense vector retrieval using in-memory matrix multiplication (matrix @ q_vec).
        Maps top chunk hits to their anchor paragraph (paragraph_start_id) and returns
        canonical SearchResults with original paragraph text and metadata.
        """
        if not internal_call and not self.config.semantic_enabled:
            raise RuntimeError("Semantic search is disabled. Set KORZ_SEMANTIC_ENABLED=1 to enable.")

        target_model = model_id or self._active_model_id or self.get_default_model_id()
        if not target_model:
            raise RuntimeError("Semantic index is not ready. Use FTS5 or run 'python korz.py semantic build'.")

        if self._semantic_index is None or self._semantic_index.model_id != target_model:
            try:
                self.load_semantic_index(target_model)
            except Exception as e:
                raise RuntimeError(
                    f"Semantic index is not ready. Use FTS5 or run 'python korz.py semantic build'. Details: {e}"
                ) from e

        # Ensure index is in READY state
        status = self.get_semantic_status(target_model)
        if status["ready_documents"] < status["total_documents"] or status["ready_documents"] == 0:
            raise RuntimeError(
                f"Semantic index is not ready ({status['ready_documents']}/{status['total_documents']} docs ready). "
                f"Use FTS5 or run 'python korz.py semantic build'."
            )

        model_info = self.get_embedding_model(target_model)
        query_tpl = model_info.get("query_template") or "query: {text}"
        clean_q = " ".join(query.strip().split())
        query_input = query_tpl.format(text=clean_q) if "{text}" in query_tpl else clean_q

        backend = self.get_backend(model_info["model_name"])
        q_vec = backend.encode_queries([query_input])[0]

        # Dot product with normalized matrix yields cosine similarity
        scores = self._semantic_index.matrix @ q_vec

        # Apply document filter if requested
        if doc_id is not None:
            mask = (self._semantic_index.doc_ids == doc_id)
            scores = np.where(mask, scores, -1e9)

        # Retrieve candidates and deduplicate by paragraph
        fetch_k = min(len(scores), max(limit * 3, 20))
        top_indices = np.argsort(scores)[::-1][:fetch_k]

        results: List[SearchResult] = []
        seen_paras: set = set()

        for idx in top_indices:
            score = float(scores[idx])
            if score <= -1e8:
                continue

            p_id = int(self._semantic_index.paragraph_ids[idx])
            if p_id in seen_paras:
                continue
            seen_paras.add(p_id)

            cur = self.conn.execute(
                """
                SELECT p.id, p.document_id, d.title, d.author, p.page_number, p.paragraph_index, p.display_text
                FROM paragraphs p
                JOIN documents d ON d.id = p.document_id
                WHERE p.id = ?
                """,
                (p_id,),
            )
            row = cur.fetchone()
            if not row:
                continue

            text = row["display_text"]
            snippet = text[:280] + ("..." if len(text) > 280 else "")

            results.append(
                SearchResult(
                    paragraph_id=row["id"],
                    document_id=row["document_id"],
                    title=row["title"],
                    author=row["author"],
                    page=row["page_number"],
                    paragraph_index=row["paragraph_index"],
                    text=text,
                    snippet=snippet,
                    score=round(score, 4),
                    match_type="semantic",
                )
            )

            if len(results) >= limit:
                break

        return results

    def verify_semantic_integrity(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Deep integrity check of semantic infrastructure:
          - Model contracts validity
          - Vector blob size = dimension * 4 bytes
          - Finite values and L2 norms ≈ 1.0
          - text_sha256 freshness against current chunk text
          - Absence of orphan embeddings
          - Ready documents chunk parity
        """
        models = self.list_embedding_models()
        target_model = model_id or self.get_default_model_id()
        if not target_model:
            return {
                "all_healthy": True,
                "model_id": None,
                "model_name": None,
                "dimension": 0,
                "chunks_expected": 0,
                "embeddings_present": 0,
                "missing_embeddings": 0,
                "stale_embeddings": 0,
                "invalid_blob": 0,
                "nan_inf_count": 0,
                "non_unit_vectors": 0,
                "documents_ready": 0,
                "total_documents": 0,
                "matrix_mib": 0.0,
                "semantic_ready": False,
                "issues": [],
            }

        m = self.get_embedding_model(target_model)
        if not m:
            return {"all_healthy": False, "issues": [f"Model '{target_model}' not found."]}

        dim = m["dimension"]
        expected_bytes = dim * 4
        passage_tpl = m.get("passage_template") or m.get("document_template") or "passage: {text}"
        model_max_tokens = m.get("model_max_tokens", 512)

        cur_tot = self.conn.execute("SELECT COUNT(*) FROM chunks")
        chunks_expected = cur_tot.fetchone()[0]

        cur_docs = self.conn.execute("SELECT COUNT(*) FROM documents")
        total_documents = cur_docs.fetchone()[0]

        cur_ready = self.conn.execute(
            "SELECT COUNT(*) FROM semantic_state WHERE model_id = ? AND status = 'READY'",
            (target_model,),
        )
        documents_ready = cur_ready.fetchone()[0]

        cur = self.conn.execute(
            """
            SELECT ce.chunk_id, ce.text_sha256, ce.vector, c.display_text
            FROM chunk_embeddings ce
            JOIN chunks c ON ce.chunk_id = c.id
            WHERE ce.model_id = ?
            """,
            (target_model,),
        )
        rows = cur.fetchall()
        embeddings_present = len(rows)
        missing_embeddings = max(0, chunks_expected - embeddings_present)

        stale_embeddings = 0
        invalid_blob = 0
        nan_inf_count = 0
        non_unit_vectors = 0
        issues: List[str] = []

        if missing_embeddings > 0:
            issues.append(f"Missing {missing_embeddings} chunk embeddings")

        for row in rows:
            b = row["vector"]
            if len(b) != expected_bytes:
                invalid_blob += 1
                issues.append(f"Chunk {row['chunk_id']}: blob size {len(b)} != expected {expected_bytes}")
                continue

            arr = np.frombuffer(b, dtype=np.float32)
            if not np.all(np.isfinite(arr)):
                nan_inf_count += 1
                issues.append(f"Chunk {row['chunk_id']}: vector contains NaN or Inf")
                continue

            norm = float(np.linalg.norm(arr))
            if abs(norm - 1.0) > 1e-4:
                non_unit_vectors += 1
                issues.append(f"Chunk {row['chunk_id']}: vector norm {norm:.6f} != 1.0")

            expected_text = build_embedding_input(row["display_text"], passage_tpl, model_max_tokens)
            expected_sha = compute_text_sha256(expected_text)
            if row["text_sha256"] != expected_sha:
                stale_embeddings += 1

        if stale_embeddings > 0:
            issues.append(f"Found {stale_embeddings} stale chunk embeddings (text/template mismatch)")

        cur_orphans = self.conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE model_id = ? AND chunk_id NOT IN (SELECT id FROM chunks)",
            (target_model,),
        )
        orphan_count = cur_orphans.fetchone()[0]
        if orphan_count > 0:
            issues.append(f"Found {orphan_count} orphan embeddings for model {target_model}")

        matrix_mib = round((embeddings_present * dim * 4) / (1024 * 1024), 2)
        all_healthy = (
            len(issues) == 0
            and missing_embeddings == 0
            and stale_embeddings == 0
            and invalid_blob == 0
            and nan_inf_count == 0
            and non_unit_vectors == 0
        )
        semantic_ready = all_healthy and (documents_ready == total_documents and total_documents > 0)

        return {
            "all_healthy": all_healthy,
            "semantic_ready": semantic_ready,
            "model_id": target_model,
            "model_name": m["model_name"],
            "dimension": dim,
            "chunks_expected": chunks_expected,
            "embeddings_present": embeddings_present,
            "missing_embeddings": missing_embeddings,
            "stale_embeddings": stale_embeddings,
            "invalid_blob": invalid_blob,
            "nan_inf_count": nan_inf_count,
            "non_unit_vectors": non_unit_vectors,
            "documents_ready": documents_ready,
            "total_documents": total_documents,
            "matrix_mib": matrix_mib,
            "issues_count": len(issues),
            "issues": issues[:10],
        }

    # =========================================================================
    # Phase V2C: Shadow Mode & Hybrid Analytics APIs
    # =========================================================================

    def record_feedback(
        self,
        query: str,
        feedback_type: str,
        shadow_log_id: Optional[int] = None,
        target_paragraph_id: Optional[int] = None,
        target_rank: Optional[int] = None,
        user_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> int:
        """
        Record explicit user relevance feedback (thumbs up/down, target fragment match).
        Types:
          - 'found' / 'up': User found the answer in shown results
          - 'not_relevant' / 'down': Shown results were not relevant
          - 'target': User pinpointed the exact desired paragraph
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO search_feedback (
                    shadow_log_id, query, target_paragraph_id, target_rank, feedback_type, user_id, comment, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (shadow_log_id, query.strip(), target_paragraph_id, target_rank, feedback_type, user_id, comment, now_iso),
            )
            return cur.lastrowid

    def get_shadow_stats(self) -> Dict[str, Any]:
        """Summary of shadow mode telemetry."""
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM shadow_search_log;")
            total = cur.fetchone()[0]

            cur_fail = self.conn.execute("SELECT COUNT(*) FROM shadow_search_log WHERE shadow_status = 'FAILED';")
            failures = cur_fail.fetchone()[0]

            cur_fb = self.conn.execute("SELECT COUNT(*) FROM search_feedback;")
            feedback_count = cur_fb.fetchone()[0]

            return {
                "total_shadowed_queries": total,
                "shadow_failures": failures,
                "success_rate_pct": round(100.0 * (total - failures) / total, 1) if total > 0 else 100.0,
                "feedback_count": feedback_count,
            }
        except Exception:
            return {"total_shadowed_queries": 0, "shadow_failures": 0, "success_rate_pct": 100.0, "feedback_count": 0}

    def get_shadow_report(self, limit_divergences: int = 20) -> Dict[str, Any]:
        """
        Produce comprehensive operational and retrieval divergence metrics from shadow_search_log
        along with feedback-grounded Recall@1, Recall@5, and MRR.
        """
        cur = self.conn.execute("""
            SELECT
                id, query, doc_id, fts_engine_version, semantic_model_id, semantic_index_sha,
                latency_fts_ms, latency_semantic_ms, fts_count, semantic_count,
                overlap_at_5, overlap_at_10, top1_match, fts_top_json, semantic_top_json,
                simulated_rrf_json, shadow_status, error_message, created_at
            FROM shadow_search_log
            ORDER BY id ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        total_queries = len(rows)

        if total_queries == 0:
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "operational": {
                    "queries_shadowed": 0,
                    "shadow_failures": 0,
                    "fts_latency_p50": 0.0,
                    "fts_latency_p95": 0.0,
                    "fts_latency_p99": 0.0,
                    "semantic_latency_p50": 0.0,
                    "semantic_latency_p95": 0.0,
                    "semantic_latency_p99": 0.0,
                },
                "divergence": {
                    "top1_match_count": 0,
                    "top1_match_pct": 0.0,
                    "overlap_5_zero_count": 0,
                    "overlap_5_zero_pct": 0.0,
                    "overlap_10_zero_count": 0,
                    "overlap_10_zero_pct": 0.0,
                    "fts_zero_semantic_hits": 0,
                },
                "relevance": {},
                "top_divergences": [],
            }

        successful = [r for r in rows if r["shadow_status"] == "SUCCESS"]
        failed = [r for r in rows if r["shadow_status"] != "SUCCESS"]

        fts_lats = sorted([r["latency_fts_ms"] for r in successful if r["latency_fts_ms"] is not None])
        sem_lats = sorted([r["latency_semantic_ms"] for r in successful if r["latency_semantic_ms"] is not None])

        def percentile(arr, p):
            if not arr:
                return 0.0
            k = (len(arr) - 1) * p
            f = int(k)
            c = min(f + 1, len(arr) - 1)
            d0 = arr[f] * (c - k)
            d1 = arr[c] * (k - f)
            return round(d0 + d1, 2)

        op_metrics = {
            "queries_shadowed": total_queries,
            "shadow_failures": len(failed),
            "fts_latency_p50": percentile(fts_lats, 0.50),
            "fts_latency_p95": percentile(fts_lats, 0.95),
            "fts_latency_p99": percentile(fts_lats, 0.99),
            "semantic_latency_p50": percentile(sem_lats, 0.50),
            "semantic_latency_p95": percentile(sem_lats, 0.95),
            "semantic_latency_p99": percentile(sem_lats, 0.99),
        }

        top1_matches = sum(1 for r in successful if r["top1_match"] == 1)
        overlap_5_zero = sum(1 for r in successful if r["overlap_at_5"] == 0)
        overlap_10_zero = sum(1 for r in successful if r["overlap_at_10"] == 0)
        fts_zero_sem_hits = sum(1 for r in successful if r["fts_count"] == 0 and r["semantic_count"] > 0)

        n_succ = max(1, len(successful))
        div_metrics = {
            "top1_match_count": top1_matches,
            "top1_match_pct": round(100.0 * top1_matches / n_succ, 1),
            "overlap_5_zero_count": overlap_5_zero,
            "overlap_5_zero_pct": round(100.0 * overlap_5_zero / n_succ, 1),
            "overlap_10_zero_count": overlap_10_zero,
            "overlap_10_zero_pct": round(100.0 * overlap_10_zero / n_succ, 1),
            "fts_zero_semantic_hits": fts_zero_sem_hits,
        }

        # Feedback evaluation
        cur_fb = self.conn.execute("SELECT * FROM search_feedback ORDER BY id ASC")
        feedbacks = [dict(r) for r in cur_fb.fetchall()]

        rel_metrics = {}
        if feedbacks:
            judged_queries = len(feedbacks)
            target_feedbacks = [f for f in feedbacks if f["feedback_type"] in ("target", "found") and f.get("target_paragraph_id")]

            fts_rec1 = 0
            fts_rec5 = 0
            sem_rec1 = 0
            sem_rec5 = 0
            rrf_rec1 = 0
            rrf_rec5 = 0

            fts_mrr_sum = 0.0
            sem_mrr_sum = 0.0
            rrf_mrr_sum = 0.0

            log_map = {r["id"]: r for r in rows}
            query_map = {r["query"]: r for r in rows}

            evaluated_judgments = 0
            for fb in target_feedbacks:
                target_pid = fb["target_paragraph_id"]
                log_entry = log_map.get(fb.get("shadow_log_id")) or query_map.get(fb.get("query"))
                if not log_entry:
                    continue

                evaluated_judgments += 1
                try:
                    fts_top = json.loads(log_entry["fts_top_json"])
                    sem_top = json.loads(log_entry["semantic_top_json"])
                    rrf_top = json.loads(log_entry["simulated_rrf_json"]) if log_entry["simulated_rrf_json"] else []
                except Exception:
                    continue

                f_rank = next((item["rank"] for item in fts_top if item["paragraph_id"] == target_pid), None)
                if f_rank == 1:
                    fts_rec1 += 1
                if f_rank and f_rank <= 5:
                    fts_rec5 += 1
                if f_rank:
                    fts_mrr_sum += 1.0 / f_rank

                s_rank = next((item["rank"] for item in sem_top if item["paragraph_id"] == target_pid), None)
                if s_rank == 1:
                    sem_rec1 += 1
                if s_rank and s_rank <= 5:
                    sem_rec5 += 1
                if s_rank:
                    sem_mrr_sum += 1.0 / s_rank

                r_rank = next((item["rank"] for item in rrf_top if item["paragraph_id"] == target_pid), None)
                if r_rank == 1:
                    rrf_rec1 += 1
                if r_rank and r_rank <= 5:
                    rrf_rec5 += 1
                if r_rank:
                    rrf_mrr_sum += 1.0 / r_rank

            n_eval = max(1, evaluated_judgments)
            thumbs_up = sum(1 for f in feedbacks if f["feedback_type"] in ("found", "up", "target"))
            thumbs_down = sum(1 for f in feedbacks if f["feedback_type"] in ("not_relevant", "down"))

            rel_metrics = {
                "total_judged": judged_queries,
                "target_judged": evaluated_judgments,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "user_satisfaction_pct": round(100.0 * thumbs_up / judged_queries, 1) if judged_queries else 0.0,
                "fts_recall_at_1": round(fts_rec1 / n_eval, 3) if evaluated_judgments else None,
                "fts_recall_at_5": round(fts_rec5 / n_eval, 3) if evaluated_judgments else None,
                "fts_mrr": round(fts_mrr_sum / n_eval, 3) if evaluated_judgments else None,
                "semantic_recall_at_1": round(sem_rec1 / n_eval, 3) if evaluated_judgments else None,
                "semantic_recall_at_5": round(sem_rec5 / n_eval, 3) if evaluated_judgments else None,
                "semantic_mrr": round(sem_mrr_sum / n_eval, 3) if evaluated_judgments else None,
                "rrf_recall_at_1": round(rrf_rec1 / n_eval, 3) if evaluated_judgments else None,
                "rrf_recall_at_5": round(rrf_rec5 / n_eval, 3) if evaluated_judgments else None,
                "rrf_mrr": round(rrf_mrr_sum / n_eval, 3) if evaluated_judgments else None,
            }

        divergences = []
        for r in successful:
            if r["overlap_at_5"] == 0 or (r["fts_count"] == 0 and r["semantic_count"] > 0):
                divergences.append({
                    "id": r["id"],
                    "query": r["query"],
                    "fts_count": r["fts_count"],
                    "semantic_count": r["semantic_count"],
                    "overlap_at_5": r["overlap_at_5"],
                    "overlap_at_10": r["overlap_at_10"],
                    "top1_match": bool(r["top1_match"]),
                    "latency_fts": r["latency_fts_ms"],
                    "latency_sem": r["latency_semantic_ms"],
                })

        return {
            "total_queries": total_queries,
            "successful_queries": len(successful),
            "failed_queries": len(failed),
            "operational": op_metrics,
            "divergence": div_metrics,
            "relevance": rel_metrics,
            "top_divergences": divergences[:limit_divergences],
        }

    def replay_shadow_evaluation(
        self,
        limit: int = 300,
        since: Optional[str] = None,
        model_id: Optional[str] = None,
        force: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Replay historical queries from search_log through shadow evaluation.
        Idempotent: skips queries already evaluated for (query, model_id, semantic_index_sha) unless force=True.
        """
        target_model = model_id or self._active_model_id or self.get_default_model_id()
        if not target_model:
            raise ValueError("No embedding model registered for shadow replay.")

        if self._semantic_index is None or self._semantic_index.model_id != target_model:
            self.load_semantic_index(target_model)

        index = self._semantic_index
        index_sha = getattr(index, "index_sha", "") if index else ""

        # Fetch queries from search_log
        sql = "SELECT DISTINCT query FROM search_log WHERE 1=1"
        params: List[Any] = []
        if since:
            sql += " AND created_at >= ?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cur = self.conn.execute(sql, params)
        queries = [r["query"].strip() for r in cur.fetchall() if r["query"] and r["query"].strip()]

        if not queries:
            return {"total": 0, "evaluated": 0, "skipped": 0, "failed": 0, "model_id": target_model, "index_sha": index_sha}

        existing_queries = set()
        if not force:
            cur_ex = self.conn.execute(
                """
                SELECT DISTINCT query FROM shadow_search_log
                WHERE semantic_model_id = ? AND semantic_index_sha = ? AND shadow_status = 'SUCCESS'
                """,
                (target_model, index_sha),
            )
            existing_queries = {r["query"].strip() for r in cur_ex.fetchall()}

        runner = ShadowRunner(self)
        evaluated = 0
        skipped = 0
        failed = 0

        for idx, q in enumerate(queries, 1):
            if q in existing_queries:
                skipped += 1
                if progress_callback:
                    progress_callback({
                        "current": idx,
                        "total": len(queries),
                        "evaluated": evaluated,
                        "skipped": skipped,
                        "failed": failed,
                    })
                continue

            t0 = time.perf_counter()
            fts_res = self._search_fts(q, limit=10)
            lat_fts = (time.perf_counter() - t0) * 1000

            res_id = runner.evaluate_sync(
                query=q,
                fts_results=fts_res,
                latency_fts_ms=lat_fts,
                model_id=target_model,
            )
            if res_id is not None:
                evaluated += 1
            else:
                failed += 1

            if progress_callback:
                progress_callback({
                    "current": idx,
                    "total": len(queries),
                    "evaluated": evaluated,
                    "skipped": skipped,
                    "failed": failed,
                })

        return {
            "total": len(queries),
            "evaluated": evaluated,
            "skipped": skipped,
            "failed": failed,
            "model_id": target_model,
            "index_sha": index_sha,
        }


class ShadowRunner:
    """
    Non-blocking background runner for shadow semantic evaluation and divergence logging.
    Enforces bounded concurrency via asyncio.Semaphore so query bursts do not overload CPU.
    Guarantees strict failure isolation: shadow errors NEVER affect primary user search.
    """

    def __init__(self, lib: "KorzLibrary", max_concurrency: int = 1):
        self.lib = lib
        self.max_concurrency = max_concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore

    def evaluate_sync(
        self,
        query: str,
        fts_results: List[SearchResult],
        doc_id: Optional[str] = None,
        latency_fts_ms: float = 0.0,
        model_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Synchronously compute semantic retrieval, compare divergence, and record in shadow_search_log.
        Safe execution: all exceptions caught and logged without propagating.
        """
        if not query or not query.strip():
            return None

        clean_q = query.strip()
        t0 = time.perf_counter()
        target_model = model_id or self.lib._active_model_id or self.lib.get_default_model_id()
        if not target_model:
            logger.warning("Shadow evaluation skipped: no embedding model registered.")
            return None

        index_sha = ""
        latency_sem_ms = 0.0
        sem_results: List[SearchResult] = []
        status = "SUCCESS"
        err_msg = None

        try:
            if self.lib._semantic_index is None or self.lib._semantic_index.model_id != target_model:
                self.lib.load_semantic_index(target_model)

            index = self.lib._semantic_index
            index_sha = getattr(index, "index_sha", "") if index else ""

            # Perform semantic search with internal_call=True to bypass user-facing disabled flag
            sem_results = self.lib._search_semantic(
                clean_q, limit=10, doc_id=doc_id, model_id=target_model, internal_call=True
            )
            latency_sem_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            latency_sem_ms = (time.perf_counter() - t0) * 1000
            status = "FAILED"
            err_msg = str(e)
            logger.warning(f"Shadow semantic evaluation failed for '{clean_q[:50]}': {e}")

        # Compute RRF and divergence metrics
        simulated_rrf, overlap_at_5, overlap_at_10, top1_match = compute_simulated_rrf(
            fts_results=fts_results,
            semantic_results=sem_results,
            k=60,
            top_limit=10,
        )

        fts_json = serialize_compact_candidates(fts_results, limit=10)
        sem_json = serialize_compact_candidates(sem_results, limit=10)
        rrf_json = json.dumps(simulated_rrf, ensure_ascii=False)
        now_iso = datetime.now(timezone.utc).isoformat()
        engine_ver = getattr(self.lib.config, "fts_engine_version", "sqlite-fts5-bm25-v1.1")

        try:
            with self.lib.conn:
                cur = self.lib.conn.execute(
                    """
                    INSERT INTO shadow_search_log (
                        query, doc_id, fts_engine_version, semantic_model_id, semantic_index_sha,
                        latency_fts_ms, latency_semantic_ms, fts_count, semantic_count,
                        overlap_at_5, overlap_at_10, top1_match, fts_top_json, semantic_top_json,
                        simulated_rrf_json, shadow_status, error_message, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_q,
                        doc_id,
                        engine_ver,
                        target_model,
                        index_sha,
                        round(latency_fts_ms, 2),
                        round(latency_sem_ms, 2),
                        len(fts_results),
                        len(sem_results),
                        overlap_at_5,
                        overlap_at_10,
                        top1_match,
                        fts_json,
                        sem_json,
                        rrf_json,
                        status,
                        err_msg,
                        now_iso,
                    ),
                )
                return cur.lastrowid
        except Exception as e:
            logger.error(f"Failed to write shadow search log: {e}")
            return None

    async def evaluate_async(
        self,
        query: str,
        fts_results: List[SearchResult],
        doc_id: Optional[str] = None,
        latency_fts_ms: float = 0.0,
        model_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Asynchronously evaluate query in shadow mode with bounded concurrency.
        Runs in background thread to never block asyncio event loop or Telegram polling.
        """
        sem = self._get_semaphore()
        async with sem:
            return await asyncio.to_thread(
                self.evaluate_sync,
                query=query,
                fts_results=fts_results,
                doc_id=doc_id,
                latency_fts_ms=latency_fts_ms,
                model_id=model_id,
            )

