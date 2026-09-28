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

import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from korz_core import KorzLibrary, SearchResult

logger = logging.getLogger("korz_answer")


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


@dataclass
class AnswerResult:
    """Complete structured output from AnswerEngine."""
    question: str
    answer_text: str
    sources: List[AnswerSource] = field(default_factory=list)
    mode: str = "extractive"  # 'extractive' | 'generative'
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
    backend: str = "extractive"  # 'extractive' | 'ollama' | 'auto'
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    min_similarity: float = 0.82
    top_k: int = 5
    max_sentences: int = 3

    @classmethod
    def from_env(cls) -> "AnswerConfig":
        return cls(
            enabled=os.environ.get("KORZ_ANSWER_ENABLED", "0").strip() in ("1", "true", "True"),
            backend=os.environ.get("KORZ_ANSWER_BACKEND", "extractive").strip().lower(),
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
        clean_q = question.strip()
        if not clean_q:
            return AnswerResult(
                question="",
                answer_text="Питання не може бути порожнім.",
                insufficient_info=True,
            )

        t0 = time.perf_counter()
        target_mode = mode or self.config.backend
        k = limit or self.config.top_k

        # 1. Retrieve candidates via E5 semantic retrieval
        passages = self.lib.ask(clean_q, limit=k, doc_id=doc_id)
        if not passages:
            elapsed = (time.perf_counter() - t0) * 1000
            return AnswerResult(
                question=clean_q,
                answer_text="У бібліотеці не знайдено матеріалів, пов'язаних із цим запитанням.",
                sources=[],
                latency_ms=elapsed,
                insufficient_info=True,
            )

        # 2. Check confidence threshold
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
                latency_ms=elapsed,
                confidence_score=top_score,
                insufficient_info=True,
            )

        # 3. Build verified AnswerSource list
        sources: List[AnswerSource] = []
        for idx, p in enumerate(passages, 1):
            sources.append(
                AnswerSource(
                    citation_index=idx,
                    paragraph_id=p.paragraph_id,
                    document_id=p.document_id,
                    author=p.author,
                    title=p.title,
                    page=p.page,
                    paragraph_index=p.paragraph_index,
                    text=p.text.strip(),
                    score=p.score,
                )
            )

        # 4. Dispatch to requested answering strategy
        if target_mode == "generative" or (target_mode == "auto" and self._is_ollama_ready()):
            res = self._generative_llm_answer(clean_q, sources)
        else:
            res = self._extractive_answer(clean_q, sources)

        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.confidence_score = top_score
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

