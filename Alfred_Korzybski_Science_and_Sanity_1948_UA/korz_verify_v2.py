"""
korz_verify_v2.py — verify a Ukrainian Korzybski quote using LLM-as-a-Verifier
backed by a free OpenAI-compatible endpoint (nemotron 3.5 lightning free).

Reads OPENAI_API_KEY + OPENAI_BASE_URL from local .env (same folder).
Usage:
  python korz_verify_v2.py "<quote text>" <page_number>
"""
import os, re, sys, dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
dotenv.load_dotenv(os.path.join(HERE, ".env"))

from openai import OpenAI
import llm_verifier as v

MODEL = os.getenv("VERIFIER_MODEL", "nemotron 3.5 lightning free")  # override via .env VERIFIER_MODEL
BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")

if not BASE_URL or not API_KEY:
    sys.exit("Missing OPENAI_BASE_URL / OPENAI_API_KEY in .env")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

CRITERIA_FILE = os.path.join(HERE, "korz_criteria.md")
CRITERIA = CRITERIA_FILE  # llm_verifier accepts a path to a .md criteria file

def verify(quote: str, page: int, fakes: list[str]) -> tuple[bool, int, list[float]]:
    """Verbatim check via boN voting: candidate[0] is the real quote,
    fakes[1..] are paraphrased/altered variants. Verified if the verifier
    ranks the real quote first (index == 0)."""
    candidates = [quote] + list(fakes)
    res = v.select(
        problem=f"Which candidate is the verbatim Ukrainian quote from Korzybski 'Science and Sanity' page {page}?",
        candidates=candidates,
        criteria=CRITERIA,
        model=MODEL,
        n_evaluations=3,
        client=client,
    )
    scores = getattr(res, "scores", None)
    ok = res.index == 0
    return ok, res.index, scores


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Usage: python korz_verify_v2.py "<quote>" <page> ["fake1" "fake2" ...]')
    quote = sys.argv[1]
    page = int(sys.argv[2])
    fakes = sys.argv[3:]
    if not fakes:
        # default weak distractors if none supplied
        fakes = [
            "Коржибський стверджував, що карта не є територією, але вона необхідна.",
            "За Коржибським, територія важливіша за карту, бо карта — це лише спрощення.",
        ]
    ok, idx, scores = verify(quote, page, fakes)
    print(f"page={page} verdict={'VERBATIM_OK' if ok else 'FAKE_DETECTED'} (index={idx}) scores={scores}")
