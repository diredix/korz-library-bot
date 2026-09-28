# Verbatim Ukrainian Quote Verification

Verify whether a supplied Ukrainian text is a verbatim (word-for-word) excerpt
from Alfred Korzybski's "Science and Sanity" (Ukrainian translation).

## Criteria

### Verbatim Accuracy
Score 1.0 if the candidate is an exact verbatim match: coherent Ukrainian,
matches the stated page's theme, no hallucinated or paraphrased text.
Score 0.7–0.9 for minor wording differences still clearly the same passage.
Score 0.4–0.6 for partial match with some paraphrasing or missing clauses.
Score 0.0–0.3 if hallucinated, heavily altered, or not from the book.
Do NOT reward fluent paraphrasing — the task is verbatim accuracy, not
semantic similarity.
