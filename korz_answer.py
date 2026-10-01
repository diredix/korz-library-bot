"""
korz_answer.py - Experimental Grounded Answer Engine for KORZ

Architectural Pipeline:
    Question
       │
       ▼
    E5 Semantic Retrieval (lib.ask)
       │
       ▼
    Top K Candidate Passages (with cosine scores)
       │
       ▼
    Sufficient Information Gate (experimental min_similarity threshold)
       │
       ├─ [Insufficient] ──► Graceful refusal ("У бібліотеці немає достатньо даних...")
       │
       ▼
    Answer Modality:
       ├─ Extractive Engine  ──► Source-bound, selects exact salience sentences from SQLite (no generative drift)
       └─ Generative (Local) ──► Grounded LLM (Ollama) with Citation Verification Pass
       │
       ▼
    Structured AnswerResult with verifiable source paragraph_ids and page numbers.
"""

import hashlib
import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from korz_core import KorzLibrary, SearchResult, compute_simulated_rrf

logger = logging.getLogger("korz_answer")

PROMPT_VERSION = "v1.0-grounded"


def compute_text_sha256(text: str) -> str:
    """Computes a 16-char sha256 of passage text for provenance verification."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def compute_config_sha(config: "AnswerConfig", prompt_version: str = PROMPT_VERSION) -> str:
    """Computes a deterministic hash of answer parameters for provenance and replayability."""
    payload = f"{config.backend}:{config.retrieval_mode}:{config.gemini_model}:{config.min_similarity:.4f}:{config.top_k}:{config.max_sentences}:{prompt_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_experiment_bundle(
    query: str,
    sources: List["AnswerSource"],
    prompt_version: str = PROMPT_VERSION,
    config_sha: str = "",
    retrieval_mode: str = "dense",
) -> Tuple[Dict[str, Any], str, str]:
    """
    Builds the canonical immutable input bundle and computes experiment_input_sha256.
    Returns: (bundle_dict, canonical_json_string, experiment_input_sha256)
    """
    bundle = {
        "query": query,
        "prompt_version": prompt_version,
        "config_sha": config_sha,
        "retrieval_mode": retrieval_mode,
        "sources": [
            {
                "citation_index": s.citation_index,
                "paragraph_id": s.paragraph_id,
                "document_id": s.document_id,
                "author": s.author,
                "title": s.title,
                "page": s.page,
                "paragraph_index": s.paragraph_index,
                "text": s.text,
                "text_sha256": s.text_sha256,
            }
            for s in sources
        ],
    }
    canonical_json = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    input_sha = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return bundle, canonical_json, input_sha


@dataclass(frozen=True)
class AnswerSource:
    """Attributed source passage backing an answer statement."""
    citation_index: int
    paragraph_id: int
    document_id: str
    author: Optional[str]
    title: str
    page: int
    paragraph_index: int
    text: str
    score: float
    text_sha256: str = ""


@dataclass
class AnswerResult:
    """Complete structured output from AnswerEngine."""
    question: str
    answer_text: str
    sources: List[AnswerSource] = field(default_factory=list)
    mode: str = "extractive"  # 'extractive' | 'gemini' | 'generative'
    retrieval_mode: str = "dense"  # 'dense' | 'rrf' | 'hybrid'
    shadow_rrf_json: str = ""
    model_tier: str = "low"   # 'low' | 'medium' | 'high'
    model_id: str = ""        # e.g. 'gemini-3.8-flash-low'
    adaptive_trigger: str = "default"  # 'default' | 'explicit_deep' | 'adaptive_comparison' | 'adaptive_multi_author' | 'adaptive_multi_book'
    book_count: int = 0
    author_count: int = 0
    prompt_version: str = PROMPT_VERSION
    answer_config_sha: str = ""
    source_text_hashes: List[str] = field(default_factory=list)
    experiment_input_sha: str = ""
    input_bundle_json: str = ""
    latency_ms: float = 0.0
    confidence_score: float = 0.0
    is_grounded: bool = True
    insufficient_info: bool = False
    error: Optional[str] = None
    fallback_reason: Optional[str] = None
    unsupported_claims_count: int = 0


@dataclass
class AnswerConfig:
    """Configuration for AnswerEngine."""
    enabled: bool = False
    backend: str = "extractive"  # 'extractive' | 'gemini' | 'ollama' | 'auto'
    retrieval_mode: str = "dense"  # 'dense' | 'rrf' | 'hybrid'
    shadow_rrf: bool = False       # Log shadow RRF when retrieval_mode is dense
    gemini_model: str = "gemini-3.8-flash-low"
    adaptive_thinking: bool = False  # Feature flag: default False (0)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    min_similarity: float = 0.82
    top_k: int = 5
    max_sentences: int = 3

    @classmethod
    def from_env(cls) -> "AnswerConfig":
        env_file = Path(".env")
        if not env_file.exists():
            env_file = Path(__file__).resolve().parent / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ or not os.environ[k]:
                            os.environ[k] = v
            except Exception:
                pass

        return cls(
            enabled=os.environ.get("KORZ_ANSWER_ENABLED", "0").strip() in ("1", "true", "True"),
            backend=os.environ.get("KORZ_ANSWER_BACKEND", "extractive").strip().lower(),
            retrieval_mode=os.environ.get("KORZ_RETRIEVAL_MODE", os.environ.get("KORZ_RETRIEVAL_BACKEND", "dense")).strip().lower(),
            shadow_rrf=os.environ.get("KORZ_SHADOW_RRF", "0").strip().lower() in ("1", "true", "yes"),
            gemini_model=os.environ.get("KORZ_GEMINI_MODEL", "gemini-3.8-flash-low").strip(),
            adaptive_thinking=os.environ.get("KORZ_ADAPTIVE_THINKING", "0").strip() in ("1", "true", "True"),
            ollama_url=os.environ.get("KORZ_OLLAMA_URL", "http://localhost:11434").strip(),
            ollama_model=os.environ.get("KORZ_OLLAMA_MODEL", "llama3.2:3b").strip(),
            min_similarity=float(os.environ.get("KORZ_ANSWER_MIN_SIMILARITY", "0.82")),
            top_k=int(os.environ.get("KORZ_ANSWER_TOP_K", "5")),
            max_sentences=int(os.environ.get("KORZ_ANSWER_MAX_SENTENCES", "3")),
        )


class AnswerEngine:
    """
    Grounded Answering Engine on top of KORZ retrieval.
    Guarantees strict attribution to authentic SQLite book passages.
    """

    def __init__(self, lib: KorzLibrary, config: Optional[AnswerConfig] = None):
        self.lib = lib
        self.config = config or AnswerConfig.from_env()

    def answer(
        self,
        question: str,
        mode: Optional[str] = None,
        limit: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> AnswerResult:
        """
        Produce a grounded answer with verified source citations.
        Modes:
          - 'extractive': Pure sentence extraction directly from SQLite text (zero hallucinations).
          - 'generative': Synthesized grounded answer via local Ollama LLM.
          - 'auto': Use generative if Ollama is available, otherwise extractive.
        """
        # Detect explicit --deep flag
        deep_flag = bool(re.search(r"(?:^|\s)--deep(?:\s|$)", question, re.IGNORECASE))
        clean_q = re.sub(r"(?:^|\s)--deep(?:\s|$)", " ", question, flags=re.IGNORECASE).strip()
        retrieval_mode = (self.config.retrieval_mode or "dense").lower()
        if not clean_q:
            return AnswerResult(
                question="",
                answer_text="Питання не може бути порожнім.",
                retrieval_mode=retrieval_mode,
                insufficient_info=True,
            )

        t0 = time.perf_counter()
        target_mode = mode or self.config.backend
        k = limit or self.config.top_k
        is_hybrid = retrieval_mode in ("rrf", "hybrid")

        sources: List[AnswerSource] = []
        top_score = 0.0
        shadow_rrf_json = ""

        # Phase 4: Entity Dossier & Character Profiling Routing
        entity_info = None
        intent = "general"
        try:
            import korz_entity
            entity_info = korz_entity.resolve_entity(self.lib.conn, clean_q)
            intent = korz_entity.detect_character_intent(clean_q)
        except Exception as e_err:
            logger.debug(f"Entity resolution check skipped: {e_err}")

        if entity_info and intent == "character_profiling":
            import korz_entity
            entity_bundle = korz_entity.build_entity_evidence_bundle(
                self.lib.conn,
                entity_id=entity_info["id"],
                query=clean_q,
                max_paragraphs=max(k, 12),
            )
            if entity_bundle:
                sources = [
                    AnswerSource(
                        citation_index=b["citation_index"],
                        paragraph_id=b["paragraph_id"],
                        document_id=b["document_id"],
                        author=b["author"],
                        title=b["title"],
                        page=b["page"],
                        paragraph_index=b["paragraph_index"],
                        text=b["text"],
                        score=b["score"],
                        text_sha256=b["text_sha256"],
                    )
                    for b in entity_bundle
                ]
                retrieval_mode = "entity_dossier"
                top_score = 1.0

        if sources and retrieval_mode == "entity_dossier":
            pass  # Already resolved via deterministic Entity Dossier
        elif is_hybrid:
            # 1. Dual Retrieval: Dense Semantic + FTS5 BM25
            dense_candidates = self.lib.ask(clean_q, limit=max(k * 2, 10), doc_id=doc_id)
            fts_candidates = self.lib._search_fts(clean_q, limit=max(k * 2, 10), doc_id=doc_id)

            if not dense_candidates and not fts_candidates:
                elapsed = (time.perf_counter() - t0) * 1000
                return AnswerResult(
                    question=clean_q,
                    answer_text="У бібліотеці не знайдено матеріалів, пов'язаних із цим запитанням.",
                    sources=[],
                    retrieval_mode=retrieval_mode,
                    latency_ms=elapsed,
                    insufficient_info=True,
                )

            dense_top_score = dense_candidates[0].score if dense_candidates else 0.0
            top_score = dense_top_score

            # Insufficient Information Gate for Hybrid:
            if dense_top_score < self.config.min_similarity:
                elapsed = (time.perf_counter() - t0) * 1000
                return AnswerResult(
                    question=clean_q,
                    answer_text=(
                        "У знайдених фрагментах недостатньо релевантної інформації, "
                        f"щоб надійно відповісти на це питання (максимальна схожість: {dense_top_score:.3f} < {self.config.min_similarity:.2f})."
                    ),
                    sources=[],
                    retrieval_mode=retrieval_mode,
                    latency_ms=elapsed,
                    confidence_score=dense_top_score,
                    insufficient_info=True,
                )

            # Simple RRF Fusion (k=60)
            rrf_items, ov5, ov10, top1_match = compute_simulated_rrf(
                fts_results=fts_candidates,
                semantic_results=dense_candidates,
                k=60,
                top_limit=k,
            )

            if not rrf_items:
                elapsed = (time.perf_counter() - t0) * 1000
                return AnswerResult(
                    question=clean_q,
                    answer_text="У бібліотеці не знайдено матеріалів, пов'язаних із цим запитанням.",
                    sources=[],
                    retrieval_mode=retrieval_mode,
                    latency_ms=elapsed,
                    insufficient_info=True,
                )

            cand_map: Dict[int, SearchResult] = {r.paragraph_id: r for r in (dense_candidates + fts_candidates)}

            for idx, item in enumerate(rrf_items, 1):
                pid = item["paragraph_id"]
                cand = cand_map.get(pid)
                if not cand:
                    continue
                clean_p_text = cand.text.strip()
                p_sha = compute_text_sha256(clean_p_text)
                sources.append(
                    AnswerSource(
                        citation_index=idx,
                        paragraph_id=cand.paragraph_id,
                        document_id=cand.document_id,
                        author=cand.author,
                        title=cand.title,
                        page=cand.page,
                        paragraph_index=cand.paragraph_index,
                        text=clean_p_text,
                        score=float(item.get("score", cand.score)),
                        text_sha256=p_sha,
                    )
                )

        else:
            # 1. Pure Dense Retrieval (E5 Semantic)
            passages = self.lib.ask(clean_q, limit=k, doc_id=doc_id)
            if not passages:
                elapsed = (time.perf_counter() - t0) * 1000
                return AnswerResult(
                    question=clean_q,
                    answer_text="У бібліотеці не знайдено матеріалів, пов'язаних із цим запитанням.",
                    sources=[],
                    retrieval_mode=retrieval_mode,
                    latency_ms=elapsed,
                    insufficient_info=True,
                )

            top_score = passages[0].score
            if top_score < self.config.min_similarity:
                elapsed = (time.perf_counter() - t0) * 1000
                return AnswerResult(
                    question=clean_q,
                    answer_text=(
                        "У знайдених фрагментах недостатньо релевантної інформації, "
                        f"щоб надійно відповісти на це питання (максимальна схожість: {top_score:.3f} < {self.config.min_similarity:.2f})."
                    ),
                    sources=[],
                    retrieval_mode=retrieval_mode,
                    latency_ms=elapsed,
                    confidence_score=top_score,
                    insufficient_info=True,
                )

            for idx, p in enumerate(passages, 1):
                clean_p_text = p.text.strip()
                p_sha = compute_text_sha256(clean_p_text)
                sources.append(
                    AnswerSource(
                        citation_index=idx,
                        paragraph_id=p.paragraph_id,
                        document_id=p.document_id,
                        author=p.author,
                        title=p.title,
                        page=p.page,
                        paragraph_index=p.paragraph_index,
                        text=clean_p_text,
                        score=p.score,
                        text_sha256=p_sha,
                    )
                )

            if self.config.shadow_rrf:
                try:
                    fts_candidates = self.lib._search_fts(clean_q, limit=max(k * 2, 10), doc_id=doc_id)
                    rrf_items, ov5, ov10, top1_match = compute_simulated_rrf(
                        fts_results=fts_candidates,
                        semantic_results=passages,
                        k=60,
                        top_limit=k,
                    )
                    shadow_rrf_json = json.dumps({
                        "overlap_at_5": ov5,
                        "overlap_at_10": ov10,
                        "top1_match": top1_match,
                        "rrf_top_pids": [item["paragraph_id"] for item in rrf_items],
                    })
                except Exception as shadow_e:
                    logger.warning(f"Shadow RRF evaluation failed: {shadow_e}")

        # 4. Dispatch to requested answering strategy
        if target_mode in ("gemini", "agy_gemini") or (target_mode == "auto" and self._is_agy_ready()):
            res = self._generative_gemini_answer(clean_q, sources, deep=deep_flag)
        elif target_mode == "generative" or (target_mode == "auto" and self._is_ollama_ready()):
            res = self._generative_llm_answer(clean_q, sources)
        else:
            res = self._extractive_answer(clean_q, sources)

        cfg_sha = compute_config_sha(self.config, PROMPT_VERSION)
        _, bundle_json, exp_sha = build_experiment_bundle(
            clean_q, sources, PROMPT_VERSION, cfg_sha, retrieval_mode=retrieval_mode
        )

        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.confidence_score = top_score
        res.retrieval_mode = retrieval_mode
        res.shadow_rrf_json = shadow_rrf_json
        res.book_count = len({s.document_id for s in sources if s.document_id})
        res.author_count = len({s.author for s in sources if s.author})
        res.prompt_version = PROMPT_VERSION
        res.answer_config_sha = cfg_sha
        res.source_text_hashes = [s.text_sha256 for s in res.sources]
        res.experiment_input_sha = exp_sha
        res.input_bundle_json = bundle_json
        return res

    def _extractive_answer(self, question: str, sources: List[AnswerSource]) -> AnswerResult:
        """
        Extractive QA: Source-bound selection of high-salience sentences directly from authentic text.
        Guarantees zero generative hallucinations, but subject to retrieval and relevance accuracy.
        """
        q_words = set(re.findall(r"\w+", question.lower()))
        stop_words = {"і", "та", "й", "або", "чи", "що", "як", "де", "чому", "коли", "хто", "який", "яка", "яке", "які", "про", "для", "на", "у", "в", "з", "із", "зі", "до", "від", "це", "той", "ті", "цей", "ця"}
        content_q_words = {w for w in (q_words - stop_words) if len(w) > 2}
        if not content_q_words:
            content_q_words = {w for w in q_words if len(w) > 1}

        overlapping_candidates: List[Tuple[float, int, str, AnswerSource]] = []
        all_candidates: List[Tuple[float, int, str, AnswerSource]] = []

        for src in sources:
            raw_sentences = re.split(r"(?<=[.!?])\s+", src.text)
            for s in raw_sentences:
                s_clean = s.strip()
                if len(s_clean) < 30:
                    continue

                s_words = set(re.findall(r"\w+", s_clean.lower()))
                overlap = len(content_q_words & s_words)
                salience = src.score + (overlap * 0.05)
                cand = (salience, src.citation_index, s_clean, src)
                all_candidates.append(cand)
                if overlap >= 1:
                    overlapping_candidates.append(cand)

        # Prefer sentences with keyword overlap; fallback to top semantic candidates
        candidates = overlapping_candidates if overlapping_candidates else all_candidates
        candidates.sort(key=lambda x: x[0], reverse=True)

        selected = candidates[: self.config.max_sentences]
        if not selected:
            top_src = sources[0]
            answer_body = f"«{top_src.text}» [{top_src.citation_index}]"
            used_sources = [top_src]
        else:
            body_parts = []
            used_indexes = set()
            for score, cit_idx, sent, src in selected:
                body_parts.append(f"«{sent}» [{cit_idx}]")
                used_indexes.add(cit_idx)

            answer_body = "\n\n".join(body_parts)
            used_sources = [s for s in sources if s.citation_index in used_indexes]

        return AnswerResult(
            question=question,
            answer_text=answer_body,
            sources=used_sources,
            mode="extractive",
            is_grounded=True,
        )

    def _verify_and_filter_citations(
        self,
        generated_text: str,
        sources: List[AnswerSource],
    ) -> Tuple[str, List[AnswerSource], int, bool]:
        """
        Citation Verification Pass:
        1. Checks that all cited indices [N] exist in candidate sources.
        2. Filters or removes hallucinated out-of-bounds citations (e.g. [8] when only 3 sources).
        3. Identifies claims lacking citation support.
        Returns: (verified_text, valid_used_sources, unsupported_claims_count, is_valid)
        """
        valid_indices = {s.citation_index for s in sources}
        found_cits = [int(m) for m in re.findall(r"\[(\d+)\]", generated_text)]

        # Check for invalid citations
        invalid_cits = [c for c in found_cits if c not in valid_indices]
        verified_text = generated_text
        if invalid_cits:
            logger.warning(f"Citation Verification: Found out-of-bounds citations {invalid_cits}; sanitizing.")
            # Remove or replace invalid citation marks
            for inv in set(invalid_cits):
                verified_text = re.sub(rf"\[{inv}\]", "", verified_text)

        # Re-check remaining valid citations
        remaining_cits = set(int(m) for m in re.findall(r"\[(\d+)\]", verified_text))
        used_sources = [s for s in sources if s.citation_index in remaining_cits]

        # Analyze sentences for ungrounded claims
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", verified_text) if len(s.strip()) > 15]
        unsupported = 0
        for sent in sentences:
            if not re.search(r"\[\d+\]", sent):
                unsupported += 1

        is_valid = len(used_sources) > 0 and len(remaining_cits) > 0
        return verified_text, used_sources, unsupported, is_valid

    def _generative_llm_answer(self, question: str, sources: List[AnswerSource]) -> AnswerResult:
        """
        Generative Answering via local LLM (Ollama) with strict citation contract
        and an automated Citation Verification Pass.
        If verification fails or LLM is unavailable, falls back to extractive mode with explicit notice.
        """
        # Prepare context blocks with clear citation markers
        context_blocks = []
        for s in sources:
            author_str = f"{s.author} — " if s.author else ""
            context_blocks.append(
                f"[{s.citation_index}] ДЖЕРЕЛО: {author_str}«{s.title}», стор. {s.page}\n«{s.text}»"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = (
            "Ти — точний помічник локальної бібліотеки KORZ. "
            "Твоє завдання — надати чітку відповідь на запитання користувача ВИКЛЮЧНО на основі наведених нижче джерел з книг.\n\n"
            "СУВОРІ ПРАВИЛА:\n"
            "1. Відповідай спираючись ТІЛЬКИ на факти з наданих фрагментів. Не додавай нічого від себе і не вигадуй фактів.\n"
            "2. Обов'язково вказуй номери джерел [1], [2] після кожного твердження чи факту.\n"
            "3. Якщо в наданих текстах немає прямої відповіді, прямо напиши: "
            "\"У наданих джерелах недостатньо інформації для відповіді на це питання.\"\n"
            "4. Відповідай українською мовою лаконічно та по суті.\n\n"
            f"ФРАГМЕНТИ З КНИГ:\n{context_text}\n\n"
            f"ЗАПИТАННЯ: {question}\n\n"
            "ВІДПОВІДЬ ЗА ДЖЕРЕЛАМИ:"
        )

        fallback_reason = None
        try:
            payload = {
                "model": self.config.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for strict factual grounding
                    "num_predict": 400,
                },
            }
            req = urllib.request.Request(
                f"{self.config.ollama_url}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                generated = data.get("response", "").strip()

            if generated:
                # Check for explicit refusal by model
                if "недостатньо інформації" in generated.lower():
                    return AnswerResult(
                        question=question,
                        answer_text=generated,
                        sources=sources[:2],
                        mode="generative",
                        is_grounded=True,
                        insufficient_info=True,
                    )

                # Citation Verification Pass
                verified_text, used_sources, unsupported_cnt, is_valid = self._verify_and_filter_citations(
                    generated, sources
                )

                if is_valid:
                    return AnswerResult(
                        question=question,
                        answer_text=verified_text,
                        sources=used_sources,
                        mode="generative",
                        is_grounded=True,
                        unsupported_claims_count=unsupported_cnt,
                    )
                else:
                    fallback_reason = "Генеративна LLM повернула текст без дійсних посилань на джерела"
            else:
                fallback_reason = "Генеративна LLM повернула порожню відповідь"

        except Exception as e:
            fallback_reason = f"Ollama недоступна ({e.__class__.__name__})"
            logger.warning(f"Generative LLM answer failed: {e}; falling back to extractive answer.")

        # Fallback to extractive
        res = self._extractive_answer(question, sources)
        res.fallback_reason = fallback_reason or "Ollama fallback"
        res.error = fallback_reason
        return res

    def _is_ollama_ready(self) -> bool:
        """Check if local Ollama server is responsive and has models available."""
        try:
            req = urllib.request.Request(f"{self.config.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                return len(models) > 0
        except Exception:
            return False

    def _is_agy_ready(self) -> bool:
        """Check if local agy CLI binary is available on PATH."""
        import shutil
        return shutil.which("agy") is not None

    def resolve_gemini_model(
        self,
        question: str,
        sources: List[AnswerSource],
        deep: bool = False,
    ) -> Tuple[str, str, str]:
        """
        Determines the Gemini model, thinking tier, and trigger reason.
        Returns: (model_id, tier, trigger)
        Tiers:
          - 'high': explicit --deep request (gemini-3.8-flash-high)
          - 'medium': comparison or multi-source reasoning (gemini-3.8-flash-medium) when adaptive_thinking is ON
          - 'low': fast default factual synthesis (gemini-3.8-flash-low)
        """
        if deep:
            return "gemini-3.8-flash-high", "high", "explicit_deep"

        if self.config.adaptive_thinking:
            unique_docs = len({s.document_id for s in sources if s.document_id})
            unique_authors = len({s.author for s in sources if s.author})

            comparison_patterns = [
                r"\bпорівняй",
                r"\bпорівняння",
                r"\bрізниц[яіею]",
                r"\bвідмінност",
                r"\bпротистав",
                r"\bспіввідношен",
                r"\bпротилежн",
                r"\bвзаємозв",
                r"\bяк пов'язан",
                r"\bчи згоден\b",
                r"\bсуперечност",
            ]
            is_comp = any(re.search(p, question, re.IGNORECASE) for p in comparison_patterns)

            # Diversity-based escalation: strictly comparison or distinct authors/books
            if is_comp:
                return "gemini-3.8-flash-medium", "medium", "adaptive_comparison"
            elif unique_authors >= 2:
                return "gemini-3.8-flash-medium", "medium", "adaptive_multi_author"
            elif unique_docs >= 2:
                return "gemini-3.8-flash-medium", "medium", "adaptive_multi_book"

        base_model = self.config.gemini_model or "gemini-3.8-flash-low"
        return base_model, "low", "default"

    def _generative_gemini_answer(
        self,
        question: str,
        sources: List[AnswerSource],
        deep: bool = False,
    ) -> AnswerResult:
        """
        Generative Answering via local Antigravity CLI (Gemini 3.8 Flash)
        without requiring external API keys.
        Strict citation contract with automated Citation Verification Pass.
        Falls back to extractive mode if agy fails or citations are invalid.
        """
        import subprocess

        # Resolve tier and model
        model_id, tier, trigger = self.resolve_gemini_model(question, sources, deep=deep)

        # Prepare context blocks with clear citation markers
        context_blocks = []
        for s in sources:
            author_str = f"{s.author} — " if s.author else ""
            context_blocks.append(
                f"[{s.citation_index}] ДЖЕРЕЛО: {author_str}«{s.title}», стор. {s.page}\n«{s.text}»"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = (
            "Ти — точний помічник локальної бібліотеки KORZ.\n"
            "Твоє завдання — надати чітку відповідь на запитання користувача ВИКЛЮЧНО на основі наведених нижче джерел з книг.\n\n"
            "СУВОРІ ПРАВИЛА:\n"
            "1. Відповідай спираючись ТІЛЬКИ на факти з наданих фрагментів. Не додавай нічого від себе і не вигадуй фактів.\n"
            "2. Обов'язково вказуй номери джерел [1], [2] після кожного твердження чи факту.\n"
            "3. Якщо в наданих текстах немає прямої відповіді, прямо напиши: "
            "\"У наданих джерелах недостатньо інформації для відповіді на це питання.\"\n"
            "4. Відповідай українською мовою лаконічно (2-4 речення) та по суті.\n\n"
            f"ФРАГМЕНТИ З КНИГ:\n{context_text}\n\n"
            f"ЗАПИТАННЯ: {question}\n\n"
            "ВІДПОВІДЬ ЗА ДЖЕРЕЛАМИ:"
        )

        fallback_reason = None
        timeout_sec = 65 if tier == "high" else 45
        try:
            cmd = ["agy", "--model", model_id, "-p", prompt]
            res = subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
            if res.returncode == 0:
                generated = res.stdout.decode("utf-8", errors="replace").strip()
                if generated:
                    if "недостатньо інформації" in generated.lower():
                        return AnswerResult(
                            question=question,
                            answer_text=generated,
                            sources=sources[:2],
                            mode="gemini",
                            model_tier=tier,
                            model_id=model_id,
                            adaptive_trigger=trigger,
                            is_grounded=True,
                            insufficient_info=True,
                        )

                    verified_text, used_sources, unsupported_cnt, is_valid = self._verify_and_filter_citations(
                        generated, sources
                    )
                    if is_valid:
                        return AnswerResult(
                            question=question,
                            answer_text=verified_text,
                            sources=used_sources,
                            mode="gemini",
                            model_tier=tier,
                            model_id=model_id,
                            adaptive_trigger=trigger,
                            is_grounded=True,
                            unsupported_claims_count=unsupported_cnt,
                        )
                    else:
                        fallback_reason = "Gemini повернула текст без валідних посилань на джерела"
                else:
                    fallback_reason = "Gemini повернула порожню відповідь"
            else:
                err_msg = res.stderr.decode("utf-8", errors="replace").strip()
                fallback_reason = f"agy CLI завершився з кодом {res.returncode}: {err_msg[:100]}"
        except Exception as e:
            fallback_reason = f"Помилка виклику agy ({e.__class__.__name__})"
            logger.warning(f"Gemini answer via agy failed: {e}; falling back to extractive.")

        # Fallback to extractive
        res = self._extractive_answer(question, sources)
        res.model_tier = tier
        res.model_id = model_id
        res.adaptive_trigger = trigger
        res.fallback_reason = fallback_reason or "Gemini fallback"
        res.error = fallback_reason
        return res


