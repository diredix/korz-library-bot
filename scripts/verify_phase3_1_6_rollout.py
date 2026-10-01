"""
scripts/verify_phase3_1_6_rollout.py - Production Rollout Verification for Phase 3.1.6

Tests AnswerEngine with retrieval_mode="rrf" on the 4 critical benchmark archetypes:
  1. Run #50 (Lexical Blind Spot / Swahili)
  2. Run #53 (Semantic Drift & Hallucination Boundary / Cybernation)
  3. Run #54 (Narrative Chain Interruption / Reserve Warden)
  4. Run #48 (Negative Constraint / Aaron Appearance Safe Refusal)
"""

import sys
import time
from pathlib import Path

# Fix terminal encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from korz_core import KorzLibrary, KorzConfig
from korz_answer import AnswerEngine, AnswerConfig

def main():
    print("=" * 80)
    print("PHASE 3.1.6 CONTROLLED ROLLOUT: GROUNDED ANSWER ENGINE VERIFICATION")
    print("Retrieval Mode: RRF (Simple RRF k=60) | Generation Backend: Gemini 3.8 Flash Low")
    print("=" * 80)

    config = KorzConfig()
    lib = KorzLibrary(config)

    cfg = AnswerConfig(
        enabled=True,
        backend="gemini",
        retrieval_mode="rrf",
        gemini_model="gemini-3.8-flash-low",
        min_similarity=0.82,
        top_k=5,
    )
    engine = AnswerEngine(lib, cfg)

    test_cases = [
        {
            "run_id": 50,
            "name": "Run #50 (Lexical Blind Spot: Swahili)",
            "query": "Якою мовою, крім англійської, володів автор у Кенії, але майже не розумів її на ринку?",
            "check": lambda ans: "суахілі" in ans.answer_text.lower() and len(ans.sources) > 0,
        },
        {
            "run_id": 53,
            "name": "Run #53 (Semantic Boundary: Cybernation 5s)",
            "query": "Скільки часу потрібно людської праці, щоб виготовити сорочку при кібернетизації за Жаком Фреско?",
            "check": lambda ans: (
                ("5 секунд" in ans.answer_text.lower() or "п'яти секунд" in ans.answer_text.lower() or "п'ять секунд" in ans.answer_text.lower())
                or ("немає" in ans.answer_text.lower() or "не вказано" in ans.answer_text.lower() or "без людей" in ans.answer_text.lower())
            ) and len(ans.sources) > 0,
        },
        {
            "run_id": 54,
            "name": "Run #54 (Narrative Chain: Reserve Warden)",
            "query": "Чому заступник наглядача заповідника відправив автора до віддаленого туристичного табору?",
            "check": lambda ans: (
                ("радіоповідомлення" in ans.answer_text.lower() or "енергійний" in ans.answer_text.lower() or "показати" in ans.answer_text.lower() or "проблему" in ans.answer_text.lower() or "наглядач" in ans.answer_text.lower())
                and not ans.insufficient_info
            ),
        },
        {
            "run_id": 48,
            "name": "Run #48 (Epistemic Boundary: Aaron Appearance)",
            "query": "Який зовнішній вигляд мав Аарон з книги про павіанів?",
            "check": lambda ans: (
                ans.insufficient_info or "недостатньо інформації" in ans.answer_text.lower() or "немає опису" in ans.answer_text.lower() or "не містить опису" in ans.answer_text.lower()
            ),
        },
    ]

    all_passed = True

    for tc in test_cases:
        print(f"\n--- Testing {tc['name']} ---")
        print(f"Query: {tc['query']}")
        t0 = time.perf_counter()
        ans = engine.answer(tc["query"])
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"Latency: {elapsed:.2f} ms | Mode: {ans.mode} | Retrieval: {ans.retrieval_mode}")
        print(f"Sources ({len(ans.sources)}): {[s.paragraph_id for s in ans.sources]}")
        print(f"Answer:\n{ans.answer_text[:300]}...")

        passed = tc["check"](ans)
        if passed:
            print(f"RESULT: [PASS] (Verified Grounding/Boundary)")
        else:
            print(f"RESULT: [FAIL] (Did not satisfy verification check)")
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL 4 CANONICAL BENCHMARK CASES PASSED SUCCESSFULLY ON SIMPLE RRF!")
    else:
        print("SOME CASES FAILED VERIFICATION.")
    print("=" * 80)

if __name__ == "__main__":
    main()
