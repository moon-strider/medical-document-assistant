# Synthetic-document evaluation method

The [main report](../../README.md#evaluation-results) contains the design, results and costs. [Worked examples](analysis.md) illustrate individual verdicts.

## What the scores mean

Status describes completeness of the answer to the *question*, not certainty of a diagnosis. `supported` can describe a complete answer about an unresolved issue; `partial` leaves a requested part unresolved. `conflicting` preserves material source disagreement; `not_documented` requires complete inspection of the relevant scope, beyond a limited retrieval packet; `needs_clarification` means the reference cannot be safely resolved.

Strict grounded success requires the exact gold status, every required claim, complete citations for material assertions, valid coverage of scope-wide conclusions, a judge pass and passing machine citation checks. *Judge assessable* means the judge returned pass or fail after reviewing every required claim, candidate claim and citation link; it does not independently establish correctness. Completion, judge assessability and strict success use **planned cases** as their denominator, including the one that produced no answer. Latency uses **completed cases** only.

An *annotated evidence set* is a complete group of source spans marked sufficient for one positive required claim. The final-packet diagnostic requires one complete annotated set for **every** positive required claim in the generator's final input. It measures delivery of those particular passages, not semantic recall or answer quality: another valid citation may support a passing answer. Its denominator includes completed cases with positive annotated claims and a known final packet, excluding the case without a target answer.

## Comparing cells

The paired bootstrap resamples the 20 questions by distinct question-family labels. Questions can share source documents, so its interval describes sensitivity within this constructed set, not independent patient archives. The compared configurations change several components together; their difference cannot isolate reranking.

Main-table latency includes both dialogue turns where applicable. The application's answer-latency view covers only the target turn. Both omit queue wait, judging and trace delivery; four workers shared one local runtime.
