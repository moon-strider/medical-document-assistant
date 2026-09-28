import hashlib
import math
import random
from collections import Counter, defaultdict

JUDGE_SEVERITIES = {"none", "S0", "S1", "S2"}
EXECUTION_STATES = {"pending", "first_turn_failed", "target_failed", "completed", "needs_review"}
TRACE_DELIVERED = {"original_materialized", "recovered_from_audit"}
SLICE_FIELDS = (
    "question_class",
    "anchor_document_class",
    "required_document_class",
    "anchor_format",
    "difficulty",
    "gold_status",
    "turn_count",
    "coverage_policy",
)


def ratio(numerator, denominator):
    """Represent a metric together with its explicit eligible population.

    Args:
        numerator (int | float): Count or sum meeting the metric criterion.
        denominator (int): Number of eligible cases, claims, edges, or groups.

    Returns:
        dict[str, int | float | None]: ``numerator``, ``denominator``, and
        ``value``. An empty population has value None; a populated population
        with no successes has value 0. No input values are changed."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency(values):
    return {
        "n": len(values),
        "p50_ms": _percentile(values, 0.5),
        "p90_ms": _percentile(values, 0.9),
    }


def judge_coverage(judge, gold_ids, claim_count, expected_edges):
    """Check whether the judge reviewed the complete declared target answer.

    Exact multiset matching rejects missing, repeated, or extra required-claim
    reviews, candidate-claim reviews, and declared citation edges. Negative
    claim indices represent unlisted prose and do not satisfy declared coverage.
    This is review eligibility, not a correctness verdict.

    Args:
        judge (dict[str, object]): Required ``required_claim_reviews`` with
            ``gold_claim_id``, ``claim_reviews`` with integer ``claim_index``,
            and ``citation_reviews`` with ``claim_index`` and ``evidence_id``.
        gold_ids (list[str]): Unique required gold claim IDs.
        claim_count (int): Number of declared candidate claims, indexed from 0.
        expected_edges (list[tuple[int, str]]): Unique candidate claim/evidence
            links that the judge must review.

    Returns:
        bool: Whether all three declared inventories are reviewed exactly once."""
    review_ids = [review["gold_claim_id"] for review in judge["required_claim_reviews"]]
    claim_indices = [
        review["claim_index"] for review in judge["claim_reviews"] if review["claim_index"] >= 0
    ]
    reviewed_edges = [
        (edge["claim_index"], edge["evidence_id"])
        for edge in judge["citation_reviews"]
        if edge["claim_index"] >= 0
    ]
    return (
        len(gold_ids) == len(set(gold_ids))
        and Counter(review_ids) == Counter(gold_ids)
        and Counter(claim_indices) == Counter(range(claim_count))
        and len(expected_edges) == len(set(expected_edges))
        and Counter(reviewed_edges) == Counter(expected_edges)
    )


def _judge_state(row):
    """Classify whether one planned case has a usable judge verdict.

    Failed or unfinished execution, a missing/not-assessable verdict, and
    incomplete review coverage remain outside conditional judge metrics while
    staying in planned-case denominators.

    Args:
        row (dict[str, object]): Evaluation row with ``execution.state``,
            ``judge_na_reason``, optional ``judge``, ``gold_required_claims``
            containing ``claim_id``, ``candidate_claims_count``, and
            ``candidate_citation_edges`` containing claim/evidence pairs.

    Returns:
        tuple[str, str | None]: ``assessable`` with None, or
        ``not_assessable`` with the execution/missingness/coverage reason.

    Raises:
        ValueError: A present judge verdict is neither pass, fail, nor
            not_assessable."""
    if row["execution"]["state"] != "completed":
        return "not_assessable", row["judge_na_reason"] or row["execution"]["state"]
    judge = row["judge"]
    if judge is None:
        return "not_assessable", row["judge_na_reason"] or "missing_judge"
    if judge["verdict"] == "not_assessable":
        return "not_assessable", row["judge_na_reason"] or "judge_not_assessable"
    if judge["verdict"] not in ("pass", "fail"):
        raise ValueError("unsupported judge verdict")
    gold_ids = [claim["claim_id"] for claim in row["gold_required_claims"]]
    expected_edges = [
        (edge["claim_index"], edge["evidence_id"]) for edge in row["candidate_citation_edges"]
    ]
    if not judge_coverage(judge, gold_ids, row["candidate_claims_count"], expected_edges):
        return "not_assessable", "invalid_judge_coverage"
    return "assessable", None


def _confirmed(row, judge_state):
    """Apply the strict grounded-success gate to an eligible case.

    Success requires a completed, assessable judge pass, exact gold status,
    valid scope coverage, no S0/S1 severity, every required claim present, all
    candidate claims supported, acceptable citations, no unlisted prose claims,
    and all three machine citation/coverage checks. An unassessable case counts
    as unsuccessful, rather than disappearing from the planned population.

    Args:
        row (dict[str, object]): Evaluation row with execution and answer/gold
            status, the judge verdict/status/coverage/severity and review lists,
            and boolean ``machine.citation_ids_valid``,
            ``required_coverage_valid``, and ``citation_edges_complete``.
        judge_state (str): Eligibility classification from ``_judge_state``.

    Returns:
        bool: Whether every strict success condition holds."""
    if judge_state != "assessable":
        return False
    judge = row["judge"]
    return (
        row["execution"]["state"] == "completed"
        and judge["verdict"] == "pass"
        and row["answer_status"] == row["gold_status"]
        and judge["status_correct"]
        and judge["coverage_valid"]
        and judge["severity"] not in ("S0", "S1")
        and all(review["claim_index"] >= 0 for review in judge["claim_reviews"])
        and all(review["label"] == "present" for review in judge["required_claim_reviews"])
        and all(review["support_label"] == "supported" for review in judge["claim_reviews"])
        and all(review["label"] in ("useful", "redundant") for review in judge["citation_reviews"])
        and all(
            row["machine"][name]
            for name in ("citation_ids_valid", "required_coverage_valid", "citation_edges_complete")
        )
    )


def _validate(rows):
    """Reject rows that would make planned-case metrics inconsistent.

    Validate unique case/variant/generator cells, execution and turn states,
    answer/judge absence after noncompletion, trace/run alignment, document
    class membership, completed-case machine checks and finite nonnegative
    latency in milliseconds. This does not fully validate judge review labels
    or the complete row schema.

    Args:
        rows (Iterable[dict[str, object]]): Rows in the shape documented by
            ``aggregate``; direct field access requires those fields to exist.

    Raises:
        ValueError: A checked identity, lifecycle, population, or latency
            invariant is violated."""
    seen = set()
    for row in rows:
        key = (row["case_id"], row["variant"], row["generator"])
        if key in seen:
            raise ValueError("duplicate planned cell row")
        seen.add(key)
        if row["execution"]["state"] not in EXECUTION_STATES:
            raise ValueError("unsupported execution state")
        if row["execution"]["state"] != "completed":
            if row["answer_status"] is not None or row["judge"] is not None:
                raise ValueError("noncompleted row has answer or judge")
            if row["candidate_claims_count"] or row["candidate_citation_edges"]:
                raise ValueError("noncompleted row has target candidate claims")
        if row["execution"]["state"] == "first_turn_failed" and row["turn_count"] != 2:
            raise ValueError("first-turn failure on single-turn case")
        if len(row["execution"]["run_ids"]) != len(row["execution"]["trace_statuses"]):
            raise ValueError("run and trace counts differ")
        if row["execution"]["state"] == "completed" and row["answer_status"] is None:
            raise ValueError("completed row lacks answer status")
        if row["candidate_claims_count"] < 0:
            raise ValueError("negative candidate claim count")
        if row["turn_count"] not in (1, 2):
            raise ValueError("unsupported turn count")
        if len(row["required_document_classes"]) != len(set(row["required_document_classes"])):
            raise ValueError("duplicate required document class")
        if row["anchor_document_class"] not in row["required_document_classes"]:
            raise ValueError("anchor absent from required document classes")
        if row["execution"]["state"] == "completed":
            for field in (
                "citation_ids_valid",
                "required_coverage_valid",
                "citation_edges_complete",
            ):
                if not isinstance(row["machine"][field], bool):
                    raise ValueError("completed row missing machine check")
            for field in ("total_latency_ms", "target_latency_ms"):
                value = row["execution"][field]
                if value is not None and (
                    not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value)
                ):
                    raise ValueError("invalid latency")
        if row["judge"] is not None and row["judge"]["severity"] not in JUDGE_SEVERITIES:
            raise ValueError("unsupported severity")


def _slice_values(row, field):
    """Assign a case to the requested reporting population.

    Args:
        row (dict[str, object]): Evaluation row with slice fields,
            ``required_document_classes``, and optional diagnostics containing
            ``coverage_policy``.
        field (str): One of ``SLICE_FIELDS``.

    Returns:
        list[object]: All required document classes, which may overlap across
        slices, or a single field value. Missing coverage policy is ``unknown``."""
    if field == "coverage_policy":
        diagnostic = row.get("diagnostics") or {}
        return [diagnostic.get("coverage_policy") or "unknown"]
    if field == "required_document_class":
        return row["required_document_classes"]
    return [row[field]]


def _evidence_availability(rows, field):
    """Measure complete annotated support present in a known span inventory.

    One positive gold claim is available when any nonempty acceptable evidence
    set is wholly present. Claims without annotated alternatives are excluded;
    a case is eligible only if it has at least one such claim, and succeeds only
    if all of those claims are available. This measures annotated span delivery,
    not semantic recall or answer correctness.

    Args:
        rows (Iterable[dict[str, object]]): Rows with ``gold_required_claims``
            containing ``acceptable_evidence_sets`` as lists of span-ID lists.
        field (str): Row key whose value is a known iterable of span IDs, such
            as ``runtime_available_span_ids``.

    Returns:
        tuple[dict[str, object], dict[str, object]]: Claim and case ratios, each
        with numerator, eligible denominator, and None for an empty denominator."""
    claim_n = 0
    claim_available = 0
    case_n = 0
    case_available = 0
    for row in rows:
        available = set(row[field])
        flags = [
            any(
                bool(option) and set(option) <= available
                for option in claim["acceptable_evidence_sets"]
            )
            for claim in row["gold_required_claims"]
            if claim["acceptable_evidence_sets"]
        ]
        claim_n += len(flags)
        claim_available += sum(flags)
        if flags:
            case_n += 1
            case_available += all(flags)
    return ratio(claim_available, claim_n), ratio(case_available, case_n)


def _diagnostic_evidence_availability(rows, field):
    """Measure annotated support without treating unknown delivery as empty.

    A missing or None diagnostic inventory excludes that positive case and its
    claims from the measured denominator; an empty known list includes them as
    failures. Cases without positive annotated claims are counted separately.
    Any complete acceptable set suffices for a claim, and every positive claim
    must have one for case availability.

    Args:
        rows (Iterable[dict[str, object]]): Rows with gold required claims and
            optional ``diagnostics`` mapping packet/stage fields to span-ID
            lists or None. Claims carry ``acceptable_evidence_sets``.
        field (str): Diagnostic span inventory key to measure.

    Returns:
        dict[str, object]: ``claims`` and ``cases`` ratios, plus integer
        ``unknown_positive_cases`` and ``no_positive_set_cases`` counts."""
    claims = 0
    claim_available = 0
    cases = 0
    case_available = 0
    unknown_cases = 0
    no_positive_set_cases = 0
    for row in rows:
        relevant = [
            claim for claim in row["gold_required_claims"] if claim["acceptable_evidence_sets"]
        ]
        if not relevant:
            no_positive_set_cases += 1
            continue
        diagnostic = row.get("diagnostics") or {}
        delivered = diagnostic.get(field)
        if delivered is None:
            unknown_cases += 1
            continue
        available = set(delivered)
        flags = [
            any(
                bool(option) and set(option) <= available
                for option in claim["acceptable_evidence_sets"]
            )
            for claim in relevant
        ]
        claims += len(flags)
        claim_available += sum(flags)
        cases += 1
        case_available += all(flags)
    return {
        "claims": ratio(claim_available, claims),
        "cases": ratio(case_available, cases),
        "unknown_positive_cases": unknown_cases,
        "no_positive_set_cases": no_positive_set_cases,
    }


def _diagnostic_receipt(rows, field):
    """Report complete evidence receipts over cases with boolean receipts.

    Args:
        rows (Iterable[dict[str, object]]): Rows with optional diagnostics.
        field (str): Receipt-completeness key under ``diagnostics``.

    Returns:
        dict[str, object]: ``complete`` ratio over strictly boolean values and
        ``unknown_cases`` for missing or nonboolean values. False is a known
        incomplete receipt, not unknown."""
    values = [(row.get("diagnostics") or {}).get(field) for row in rows]
    known = [value for value in values if isinstance(value, bool)]
    return {"complete": ratio(sum(known), len(known)), "unknown_cases": len(values) - len(known)}


def _core(rows):
    """Aggregate one population while preserving each metric's eligibility.

    Strict success, completion, and judge assessability use all planned rows.
    Claim/edge quality uses assessable judge reviews; conservative gold recall
    retains every required claim. Exact status and trace-delivery rates use
    completed cases. Latency percentiles use known completed-case milliseconds;
    total latency covers both turns and target latency only the evaluated turn.
    Final-packet availability excludes unknown positive inventories, while
    indexed availability uses the known runtime span inventory. Trace delivery
    is a separate diagnostic and never a success-population filter.

    Args:
        rows (list[dict[str, object]]): Validated evaluation rows in the shape
            documented by ``aggregate``.

    Returns:
        dict[str, object]: Planned/state counts, explicit ratio objects,
        status-confusion and missingness counts, evidence/receipt diagnostics,
        judge severity, trace status, and p50/p90 latency summaries. Empty
        ratio populations and absent latency percentiles have value None."""
    planned = len(rows)
    states = Counter(row["execution"]["state"] for row in rows)
    judged = [(_judge_state(row), row) for row in rows]
    assessable = [(row, row["judge"]) for (state, _), row in judged if state == "assessable"]
    confirmed = sum(_confirmed(row, state) for (state, _), row in judged)
    completed = [row for row in rows if row["execution"]["state"] == "completed"]
    status_pairs = Counter((row["gold_status"], row["answer_status"]) for row in completed)
    status_confusion = {}
    for (gold, predicted), count in sorted(status_pairs.items()):
        status_confusion.setdefault(gold, {})[predicted] = count
    missing_answer = Counter(row["gold_status"] for row in rows if row["answer_status"] is None)
    gold_claims = sum(len(row["gold_required_claims"]) for row in rows)
    gold_claims_without_evidence_sets = sum(
        not claim["acceptable_evidence_sets"]
        for row in rows
        for claim in row["gold_required_claims"]
    )
    present = sum(
        review["label"] == "present"
        for _, judge in assessable
        for review in judge["required_claim_reviews"]
    )
    assessable_gold_claims = sum(
        review["label"] != "not_assessable"
        for _, judge in assessable
        for review in judge["required_claim_reviews"]
    )
    own_supported = sum(
        review["support_label"] == "supported"
        for _, judge in assessable
        for review in judge["claim_reviews"]
    )
    own_assessable = sum(
        review["support_label"] != "not_assessable"
        for _, judge in assessable
        for review in judge["claim_reviews"]
    )
    citation_useful = sum(
        review["label"] == "useful"
        for _, judge in assessable
        for review in judge["citation_reviews"]
    )
    citation_assessable = sum(
        review["label"] != "not_assessable"
        for _, judge in assessable
        for review in judge["citation_reviews"]
    )
    final_packet_evidence = _diagnostic_evidence_availability(rows, "final_packet_span_ids")
    evidence_claims = final_packet_evidence["claims"]
    evidence_cases = final_packet_evidence["cases"]
    indexed_claims, indexed_cases = _evidence_availability(rows, "runtime_available_span_ids")
    prose_claims = sum(
        review["claim_index"] == -1 for _, judge in assessable for review in judge["claim_reviews"]
    )
    prose_edges = sum(
        review["claim_index"] == -1
        for _, judge in assessable
        for review in judge["citation_reviews"]
    )
    trace_counts = Counter(status for row in rows for status in row["execution"]["trace_statuses"])
    trace_original = sum(
        bool(row["execution"]["trace_statuses"])
        and all(status == "original_materialized" for status in row["execution"]["trace_statuses"])
        for row in completed
    )
    trace_delivered = sum(
        bool(row["execution"]["trace_statuses"])
        and all(status in TRACE_DELIVERED for status in row["execution"]["trace_statuses"])
        for row in completed
    )
    judge_na = Counter(reason for (state, reason), _ in judged if state == "not_assessable")
    severity = Counter(judge["severity"] for _, judge in assessable)
    total_latencies = [
        row["execution"]["total_latency_ms"]
        for row in completed
        if row["execution"]["total_latency_ms"] is not None
    ]
    target_latencies = [
        row["execution"]["target_latency_ms"]
        for row in completed
        if row["execution"]["target_latency_ms"] is not None
    ]
    two_turn = [row for row in rows if row["turn_count"] == 2]
    diagnostic_rows = [
        row["diagnostics"] for row in rows if isinstance(row.get("diagnostics"), dict)
    ]
    return {
        "planned": planned,
        "execution_state_counts": dict(sorted(states.items())),
        "confirmed_grounded_success": ratio(confirmed, planned),
        "confirmed_grounded_success_on_assessable": ratio(confirmed, len(assessable)),
        "technical_completion": ratio(len(completed), planned),
        "judge_assessable": ratio(len(assessable), planned),
        "judge_not_assessable": ratio(planned - len(assessable), planned),
        "judge_na_reason_counts": dict(sorted(judge_na.items())),
        "first_turn_failure": ratio(states["first_turn_failed"], len(two_turn)),
        "answer_status_exact": ratio(
            sum(row["answer_status"] == row["gold_status"] for row in completed), len(completed)
        ),
        "status_confusion": status_confusion,
        "status_missing_answer_by_gold": dict(sorted(missing_answer.items())),
        "required_claim_recall_conditional": ratio(present, assessable_gold_claims),
        "required_claim_present_conservative": ratio(present, gold_claims),
        "required_claim_na_count": gold_claims - assessable_gold_claims,
        "own_claim_support": ratio(own_supported, own_assessable),
        "own_claim_na_count": sum(row["candidate_claims_count"] for row in rows)
        + prose_claims
        - own_assessable,
        "unlisted_prose_claim_count": prose_claims,
        "citation_edge_precision": ratio(citation_useful, citation_assessable),
        "citation_edge_na_count": sum(len(row["candidate_citation_edges"]) for row in rows)
        + prose_edges
        - citation_assessable,
        "evidence_claim_availability": evidence_claims,
        "evidence_case_availability": evidence_cases,
        "evidence_unknown_positive_cases": final_packet_evidence["unknown_positive_cases"],
        "indexed_evidence_claim_availability": indexed_claims,
        "indexed_evidence_case_availability": indexed_cases,
        "diagnostic_evidence": {
            "indexed": {"claims": indexed_claims, "cases": indexed_cases},
            "initial_candidate_block": _diagnostic_evidence_availability(
                rows, "candidate_block_span_ids"
            ),
            "initial_packet": _diagnostic_evidence_availability(rows, "initial_packet_span_ids"),
            "final_packet": final_packet_evidence,
        },
        "diagnostic_policy_counts": dict(
            sorted(Counter(item["coverage_policy"] for item in diagnostic_rows).items())
        ),
        "diagnostic_initial_method_counts": dict(
            sorted(Counter(item["initial_method"] for item in diagnostic_rows).items())
        ),
        "diagnostic_second_batch_counts": dict(
            sorted(Counter(item["second_batch_kind"] or "none" for item in diagnostic_rows).items())
        ),
        "diagnostic_guard_counts": dict(
            sorted(Counter(item["guard_action"] for item in diagnostic_rows).items())
        ),
        "diagnostic_initial_receipt": _diagnostic_receipt(rows, "initial_receipt_complete"),
        "diagnostic_final_receipt": _diagnostic_receipt(rows, "final_receipt_complete"),
        "diagnostic_missing_cases": planned - len(diagnostic_rows),
        "evidence_claim_no_set_count": gold_claims_without_evidence_sets,
        "severity_counts_assessable": dict(sorted(severity.items())),
        "trace_status_counts": dict(sorted(trace_counts.items())),
        "trace_original_completed": ratio(trace_original, len(completed)),
        "trace_delivered_completed": ratio(trace_delivered, len(completed)),
        "latency_total_completed": _latency(total_latencies),
        "latency_target_completed": _latency(target_latencies),
        "latency_missing_completed_count": len(completed) - len(total_latencies),
    }


def aggregate(rows):
    """Build outcome metrics and slices for planned evaluation attempts.

    Materialize and validate the input, then calculate overall and per-slice
    populations. Required-document-class slices overlap when a question needs
    several classes; their counts must not be summed as disjoint populations.
    Macro success weights each nonempty question-class or anchor-document-class
    slice equally rather than weighting by its number of cases. No row is
    changed and no trace service or inference provider is consulted.

    Args:
        rows (Iterable[dict[str, object]]): One row per case/variant/generator.
            Required identity/slice fields are ``case_id``, ``family_id``,
            ``variant``, ``generator``, ``question_class``,
            ``anchor_document_class``, ``required_document_classes`` (unique
            list containing the anchor), ``anchor_format``, ``difficulty``,
            ``gold_status``, and ``turn_count`` (1 or 2). ``execution`` requires
            ``state``, matching ``run_ids``/``trace_statuses`` lists, and
            ``total_latency_ms``/``target_latency_ms`` (number or None).
            ``answer_status`` and ``judge`` are None for noncompleted cases;
            ``judge_na_reason`` records unavailable judging. Gold claims contain
            ``claim_id`` and ``acceptable_evidence_sets`` (span-ID lists).
            Candidate fields are ``candidate_claims_count`` and
            ``candidate_citation_edges`` with ``claim_index``/``evidence_id``.
            Completed rows require the three boolean ``machine`` checks used by
            ``_confirmed``. Judges contain verdict, status_correct,
            coverage_valid, severity, ``required_claim_reviews`` with
            gold_claim_id/label, ``claim_reviews`` with claim_index/support_label,
            and ``citation_reviews`` with claim_index/evidence_id/label.
            ``runtime_available_span_ids`` is a known span-ID list. Optional
            ``diagnostics`` contains stage span-ID lists or None, boolean/None
            receipts, and coverage_policy, initial_method, second_batch_kind,
            and guard_action labels.

    Returns:
        dict[str, object]: ``summary`` metrics, ``slices`` indexed by field and
        stringified value, and ``macro_confirmed`` explicit ratios. Ratios retain
        numerator/denominator; None means no eligible observations, not zero.

    Raises:
        ValueError: A checked row invariant or judge verdict is invalid."""
    rows = list(rows)
    _validate(rows)
    slices = {}
    for field in SLICE_FIELDS:
        groups = defaultdict(list)
        for row in rows:
            for value in _slice_values(row, field):
                groups[str(value)].append(row)
        slices[field] = {value: _core(group) for value, group in sorted(groups.items())}
    summary = _core(rows)
    macro = {}
    for field in ("question_class", "anchor_document_class"):
        rates = [
            result["confirmed_grounded_success"]["value"]
            for result in slices[field].values()
            if result["planned"]
        ]
        macro[field] = ratio(sum(rates), len(rates))
    return {"summary": summary, "slices": slices, "macro_confirmed": macro}


def _bootstrap_difference(pairs, seed, replicates):
    """Resample matched question families to quantify success-rate sensitivity.

    Sample the distinct family labels with replacement, retaining every paired
    case in each sampled family. Each replicate divides its summed B-minus-A
    successes by the sampled number of cases, not the number of families. The
    percentile interval describes this constructed question set; shared source
    documents do not become independent clinical samples.

    Args:
        pairs (list[tuple[dict[str, object], dict[str, object]]]): Nonempty,
            validated A/B case pairs with matching family_id and success fields.
        seed (int): Seed for a local reproducible random generator.
        replicates (int): Positive number of resampled populations.

    Returns:
        dict[str, object]: Seed, replicate/family-cluster counts, method label,
        and two-element ``ci95_percentage_points`` for B minus A."""
    by_family = defaultdict(list)
    for a, b in pairs:
        by_family[a["family_id"]].append((a, b))
    families = sorted(by_family)
    family_values = []
    for family in families:
        members = by_family[family]
        change = sum(
            int(_confirmed(b, _judge_state(b)[0])) - int(_confirmed(a, _judge_state(a)[0]))
            for a, b in members
        )
        family_values.append((change, len(members)))
    generator = random.Random(seed)
    differences = []
    for _ in range(replicates):
        selected = [family_values[generator.randrange(len(families))] for _ in families]
        differences.append(
            100 * sum(change for change, _ in selected) / sum(count for _, count in selected)
        )
    return {
        "seed": seed,
        "replicates": replicates,
        "method": "family_cluster_percentile",
        "family_clusters": len(families),
        "ci95_percentage_points": [
            _percentile(differences, 0.025),
            _percentile(differences, 0.975),
        ],
    }


def paired_comparison(a_rows, b_rows, seed=20260925, replicates=10000):
    """Compare strict success in two cells over exactly the same planned cases.

    Pair by unique case ID and verify family, question class, anchor class, and
    gold status. Failed or unassessable attempts count as zero successes in both
    the paired rate and family-cluster bootstrap. Return discordant-pair counts
    to distinguish gained from lost successes. The difference is B minus A in
    percentage points and does not attribute causality to one changed component.

    Args:
        a_rows (Iterable[dict[str, object]]): Nonempty A-cell evaluation rows in
            the ``aggregate`` shape, with one row per unique case_id.
        b_rows (Iterable[dict[str, object]]): B-cell rows with identical case IDs
            and matching pairing metadata.
        seed (int): Reproducible bootstrap seed.
        replicates (int): Positive bootstrap replicate count.

    Returns:
        dict[str, object]: ``paired_n``, ``a_confirmed``/``b_confirmed`` ratios
        over all pairs, gained/lost-success counts, B-minus-A
        ``difference_percentage_points``, and family-cluster ``bootstrap``.

    Raises:
        ValueError: Cells are empty, duplicated, incompatible, invalid, or
            replicates is not a positive integer."""
    a_rows = list(a_rows)
    b_rows = list(b_rows)
    _validate(a_rows)
    _validate(b_rows)
    if not a_rows or not b_rows:
        raise ValueError("paired comparison needs nonempty cells")
    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be positive")
    a_by_id = {row["case_id"]: row for row in a_rows}
    b_by_id = {row["case_id"]: row for row in b_rows}
    if (
        len(a_by_id) != len(a_rows)
        or len(b_by_id) != len(b_rows)
        or a_by_id.keys() != b_by_id.keys()
    ):
        raise ValueError("paired cells need identical unique case IDs")
    pairs = [(a_by_id[case_id], b_by_id[case_id]) for case_id in sorted(a_by_id)]
    for a, b in pairs:
        for field in ("family_id", "question_class", "anchor_document_class", "gold_status"):
            if a[field] != b[field]:
                raise ValueError("paired metadata mismatch")
    a_success = [int(_confirmed(a, _judge_state(a)[0])) for a, _ in pairs]
    b_success = [int(_confirmed(b, _judge_state(b)[0])) for _, b in pairs]
    b_positive_a_negative = sum(b and not a for a, b in zip(a_success, b_success))
    b_negative_a_positive = sum(a and not b for a, b in zip(a_success, b_success))
    return {
        "paired_n": len(pairs),
        "a_confirmed": ratio(sum(a_success), len(pairs)),
        "b_confirmed": ratio(sum(b_success), len(pairs)),
        "b_positive_a_negative": b_positive_a_negative,
        "b_negative_a_positive": b_negative_a_positive,
        "difference_percentage_points": 100 * (sum(b_success) - sum(a_success)) / len(pairs),
        "bootstrap": _bootstrap_difference(pairs, seed, replicates),
    }


def campaign_report(rows, seed=20260925, replicates=10000):
    """Report each observed configuration and the predefined paired contrasts.

    Group planned rows by variant/generator, compare Sol minus Luna within each
    variant where both exist, and V3 minus V0 within each generator where both
    exist. Missing cells produce no contrast. Present cells must have identical
    paired cases; they are not silently intersected. Derive a stable seed per
    comparison label so each interval is reproducible independently.

    Args:
        rows (Iterable[dict[str, object]]): Planned evaluation rows in the
            ``aggregate`` shape, spanning any observed variant/generator cells.
        seed (int): Base seed used to derive comparison-specific seeds.
        replicates (int): Positive bootstrap replicate count for comparisons.

    Returns:
        dict[str, object]: ``cells`` keyed as variant/generator with aggregate
        reports, and ``comparisons`` keyed by the named paired contrasts.

    Raises:
        ValueError: Row validation or an emitted paired comparison fails."""
    rows = list(rows)
    _validate(rows)
    cells = defaultdict(list)
    for row in rows:
        cells[(row["variant"], row["generator"])].append(row)
    cell_reports = {
        variant + "/" + generator: aggregate(group)
        for (variant, generator), group in sorted(cells.items())
    }
    comparisons = {}
    variants = sorted({variant for variant, _ in cells})
    generators = sorted({generator for _, generator in cells})
    for variant in variants:
        a = cells.get((variant, "gpt-6-luna"))
        b = cells.get((variant, "gpt-6-sol"))
        if a and b:
            label = variant + ":sol_minus_luna"
            pair_seed = int.from_bytes(
                hashlib.sha256((str(seed) + label).encode()).digest()[:8], "big"
            )
            comparisons[label] = paired_comparison(a, b, pair_seed, replicates)
    for generator in generators:
        a = cells.get(("V0", generator))
        b = cells.get(("V3", generator))
        if a and b:
            label = generator + ":V3_minus_V0"
            pair_seed = int.from_bytes(
                hashlib.sha256((str(seed) + label).encode()).digest()[:8], "big"
            )
            comparisons[label] = paired_comparison(a, b, pair_seed, replicates)
    return {"cells": cell_reports, "comparisons": comparisons}
