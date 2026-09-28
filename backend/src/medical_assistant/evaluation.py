import argparse
import asyncio
import copy
import csv
import hashlib
import json
import logging
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
from langfuse import Langfuse
from sqlalchemy import text

from medical_assistant.evaluation_metrics import (
    EXECUTION_STATES,
    aggregate,
    campaign_report,
    judge_coverage,
)
from medical_assistant.evaluation_release import Release, canonical, digest, file_hash, ref_key
from medical_assistant.graph import coverage_policy
from medical_assistant.providers import get_provider
from medical_assistant.providers.common import ProviderError, instruction_hash
from medical_assistant.providers.contracts import validate_output
from medical_assistant.settings import get_settings
from medical_assistant.storage import Store, _record
from medical_assistant.telemetry import Telemetry, estimate_api_equivalent_cost

VARIANTS = ("V0", "V3")
GENERATORS = ("gpt-6-sol", "gpt-6-luna")
COMPLETE_STATES = {"completed", "first_turn_failed", "target_failed"}
PROVIDER_OUTAGE_CODES = {
    "bridge_transport_unavailable",
    "bridge_unavailable",
    "bridge_unauthorized",
    "bridge_token_missing",
    "authentication_missing",
}
JUDGE_RECONCILE_INTERVAL = timedelta(seconds=60)


def case_worker_count(config):
    """Resolve the bounded concurrency frozen into an evaluation.

    Args:
        config (dict): Campaign settings; case_workers defaults to 1.

    Returns:
        int: Worker count from 1 through 4; booleans are not accepted.

    Raises:
        ValueError: The configured count is not an integer in the allowed range.
    """
    count = config.get("case_workers", 1)
    if type(count) is not int or not 1 <= count <= 4:
        raise ValueError("Frozen case_workers must be an integer from 1 to 4")
    return count


def execution_order(cases):
    """Rotate comparison cells across families in a reproducible order.

    Interleave sorted question families and rotate the four variant/generator
    cells by family and within-family case position. Every selected case still
    receives every cell, while execution is less concentrated by one cell.

    Args:
        cases (dict[str, dict]): Cases keyed by ID with family_id metadata.

    Returns:
        list[list[str]]: Ordered [case_id, variant, generator] cells; empty
            when no cases are supplied.
    """
    families = defaultdict(list)
    for case_id, case in sorted(cases.items()):
        families[case["family_id"]].append(case_id)
    family_ids = sorted(families)
    cells = [(variant, generator) for variant in VARIANTS for generator in GENERATORS]
    order = []
    for case_index in range(max(map(len, families.values()), default=0)):
        for cell_index in range(len(cells)):
            for family_index, family_id in enumerate(family_ids):
                if case_index < len(families[family_id]):
                    variant, generator = cells[
                        (cell_index + family_index + case_index) % len(cells)
                    ]
                    order.append([families[family_id][case_index], variant, generator])
    return order


def selected_cases(release, config):
    """Enforce the frozen case selection and acceptance balance.

    Args:
        release (Release): Loaded evaluation partition and its cases.
        config (dict): Nonempty, sorted, unique selected_case_ids. Acceptance
            requires 20 selected cases, four in each Q1 through Q5 class.

    Returns:
        dict[str, dict]: Selected release cases in configured ID order.

    Raises:
        ValueError: Selection is malformed, outside the partition or unbalanced.
    """
    ids = config.get("selected_case_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(case_id, str) for case_id in ids)
        or ids != sorted(set(ids))
        or not set(ids) <= set(release.cases)
    ):
        raise ValueError("Frozen selected_case_ids must be sorted, unique partition case IDs")
    if release.partition == "acceptance":
        if len(ids) != 20 or Counter(
            release.cases[case_id]["question_class"] for case_id in ids
        ) != {f"Q{index}": 4 for index in range(1, 6)}:
            raise ValueError("Acceptance selection must contain 20 cases, four per question class")
    return {case_id: release.cases[case_id] for case_id in ids}


def now():
    return datetime.now(UTC).isoformat()


def code_hash():
    """Bind campaign reproducibility to application and dependency artifact bytes.

    Hash sorted Python, prompt, migration, dependency-lock and pricing files
    together with their repository paths. Python docstrings are included, so
    documentation changes also alter the frozen campaign code identity.

    Returns:
        str: SHA-256 hex digest of the current reproducibility artifacts.

    Raises:
        OSError: A required artifact cannot be read.
    """
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "backend/src/medical_assistant").rglob("*.py"))
    paths += sorted((root / "prompts").rglob("*.md"))
    paths += sorted((root / "prompts").rglob("*.json"))
    paths += sorted((root / "migrations").glob("*.sql"))
    paths += [
        root / "pyproject.toml",
        root / "uv.lock",
        root / "backend/src/medical_assistant/pricing_snapshot.json",
    ]
    value = hashlib.sha256()
    for path in paths:
        value.update(str(path.relative_to(root)).encode())
        value.update(b"\0")
        value.update(path.read_bytes())
        value.update(b"\0")
    return value.hexdigest()


def source_map_contract(mapping, release=None, require_ready=False):
    """Validate the single shared collection used by all campaign cells.

    Args:
        mapping (dict): Exactly collection_id (str | None), collection_revision
            (nonnegative int | None), and documents mapping doc_id to exactly
            source_id, sha256 and status. Source IDs must be distinct.
        release (Release | None): Optional manifest for document/hash checks.
        require_ready (bool): Require collection identity/revision, nonempty
            ready documents and, with release, the complete document inventory.

    Returns:
        dict: The original mapping, without copying or normalizing it.

    Raises:
        ValueError: Shape, identity, inventory or readiness violates the contract.
    """
    if not isinstance(mapping, dict) or set(mapping) != {
        "collection_id",
        "collection_revision",
        "documents",
    }:
        raise ValueError("Campaign source map does not use the shared collection contract")
    collection_id = mapping["collection_id"]
    revision = mapping["collection_revision"]
    documents = mapping["documents"]
    if not isinstance(documents, dict) or not isinstance(collection_id, (str, type(None))):
        raise ValueError("Invalid shared collection source map")
    if revision is not None and (type(revision) is not int or revision < 0):
        raise ValueError("Invalid shared collection revision")
    if require_ready and (not collection_id or revision is None or not documents):
        raise ValueError("Shared collection import is incomplete")
    if release is not None and require_ready and set(documents) != set(release.documents):
        raise ValueError("Shared collection does not contain every release document")
    source_ids = []
    for doc_id, item in documents.items():
        if not isinstance(item, dict) or set(item) != {"source_id", "sha256", "status"}:
            raise ValueError("Invalid imported document mapping")
        if not isinstance(item["source_id"], str) or not isinstance(item["sha256"], str):
            raise ValueError("Invalid imported document identity")
        if release is not None and (
            doc_id not in release.documents or item["sha256"] != release.documents[doc_id]["sha256"]
        ):
            raise ValueError("Imported document differs from release manifest")
        if require_ready and item["status"] != "ready":
            raise ValueError("Imported document is not ready")
        source_ids.append(item["source_id"])
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Imported documents reuse a source ID")
    return mapping


def lock_and_verify_collection(conn, source_map):
    """Protect corpus readiness or campaign freeze with an inventory check.

    Hold a shared collection-row lock in the caller's transaction, verify the
    frozen revision/counters, then require exactly the mapped ready sources
    with their original hashes. This operation does not commit the transaction.

    Args:
        conn (sqlalchemy.engine.Connection): Open transactional connection.
        source_map (dict): Ready shared collection map accepted by
            source_map_contract, including all document/source/hash identities.

    Raises:
        ValueError: Collection scope or active source inventory changed.
    """
    source_map_contract(source_map, require_ready=True)
    collection = (
        conn.execute(
            text(
                "SELECT id,revision,source_count,ready_count,unavailable_count FROM collections WHERE id=:id FOR SHARE"
            ),
            {"id": source_map["collection_id"]},
        )
        .mappings()
        .first()
    )
    frozen_collection_scope(source_map, collection)
    active_sources = (
        conn.execute(
            text(
                "SELECT id,sha256,status FROM sources WHERE collection_id=:id "
                "AND status<>'deleted' AND deleted_at IS NULL"
            ),
            {"id": source_map["collection_id"]},
        )
        .mappings()
        .all()
    )
    expected = {item["source_id"]: item["sha256"] for item in source_map["documents"].values()}
    if len(active_sources) != len(expected) or any(
        str(source["id"]) not in expected
        or source["status"] != "ready"
        or source["sha256"] != expected[str(source["id"])]
        for source in active_sources
    ):
        raise ValueError("Shared collection source inventory changed before commit")


def frozen_collection_scope(mapping, collection):
    """Build the run-scope snapshot only for an unchanged ready corpus.

    Args:
        mapping (dict): Frozen collection_id, collection_revision and documents.
        collection (Mapping | None): Current id, revision, source_count,
            ready_count and unavailable_count; None means it no longer exists.

    Returns:
        dict: collection_id, revision, all three counters and snapshot_hash;
            every mapped document must be ready and none unavailable.

    Raises:
        ValueError: Identity, revision or counters differ from the frozen scope.
    """
    expected_count = len(mapping["documents"])
    if (
        collection is None
        or str(collection["id"]) != mapping["collection_id"]
        or collection["revision"] != mapping["collection_revision"]
        or collection["source_count"] != expected_count
        or collection["ready_count"] != expected_count
        or collection["unavailable_count"] != 0
    ):
        raise ValueError("Shared collection changed from the frozen release")
    snapshot = {
        "collection_id": mapping["collection_id"],
        "revision": mapping["collection_revision"],
        "source_count": expected_count,
        "ready_count": expected_count,
        "unavailable_count": 0,
    }
    return {**snapshot, "snapshot_hash": digest(snapshot)}


def validate_projection_row(attempt):
    """Select an auditable primary result without hiding later corrections.

    Check the current projection against the row's cell, execution state and
    run IDs, and validate any frozen first-attempt projection and judge identity.
    Terminal first attempts and all retried rows must retain a primary record.

    Args:
        attempt (dict): Planned row with id, cell identity, state, run IDs,
            attempts history and metrics containing projection, cost_estimate
            and optional primary with its projection digest and attempt ID.

    Returns:
        dict: Frozen primary projection when present, otherwise the current
            projection; no projection is recomputed or persisted here.

    Raises:
        ValueError: Current or primary metrics are absent, stale or malformed.
    """
    metrics = attempt.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Current metric record is malformed: {attempt['id']}")
    projection = metrics.get("projection")
    if projection is None:
        reason = metrics.get("projection_unavailable") or "projection_missing"
        raise ValueError(f"Current metric projection unavailable for {attempt['id']}: {reason}")
    if not isinstance(projection, dict):
        raise ValueError(f"Current metric projection is malformed: {attempt['id']}")
    expected_runs = [
        str(run_id) for run_id in (attempt["first_run_id"], attempt["target_run_id"]) if run_id
    ]
    expected_state = attempt["state"] if attempt["state"] in EXECUTION_STATES else "pending"
    execution = projection.get("execution")
    if (
        projection.get("case_id") != attempt["case_id"]
        or projection.get("variant") != attempt["variant"]
        or projection.get("generator") != attempt["generator"]
        or not isinstance(execution, dict)
        or execution.get("run_ids") != expected_runs
        or execution.get("state") != expected_state
        or not isinstance(projection.get("machine"), dict)
        or not isinstance(projection.get("gold_required_claims"), list)
        or not isinstance(projection.get("runtime_available_span_ids"), list)
        or not isinstance(metrics.get("cost_estimate"), dict)
    ):
        raise ValueError(f"Current metric projection is stale or malformed: {attempt['id']}")
    primary = metrics.get("primary")
    if primary is not None:
        if not isinstance(primary, dict) or not attempt["attempts"]:
            raise ValueError(f"Current primary metric record is malformed: {attempt['id']}")
        first = attempt["attempts"][0]
        first_run_ids = [turn["run_id"] for turn in first.get("turns", []) if turn.get("run_id")]
        if (
            primary.get("attempt_id") != first.get("attempt_id")
            or primary.get("projection_sha256") != digest(primary.get("projection"))
            or not isinstance(primary.get("projection"), dict)
            or primary["projection"].get("execution", {}).get("run_ids") != first_run_ids
            or primary.get("judge_request_id")
            != (
                first.get("judge_history", [{}])[0].get("request_id")
                if first.get("judge_history")
                else first.get("judge_request_id")
                or (first.get("judge_result") or {}).get("request_id")
            )
        ):
            raise ValueError(f"Current primary metric record is stale: {attempt['id']}")
    elif len(attempt["attempts"]) > 1:
        raise ValueError(f"Retry lacks frozen primary metric: {attempt['id']}")
    elif attempt["attempts"] and _primary_terminal(attempt):
        raise ValueError(f"Terminal first attempt lacks frozen primary metric: {attempt['id']}")
    return primary["projection"] if primary is not None else projection


def _primary_terminal(row):
    """Determine when the first result can be frozen for comparison.

    Args:
        row (dict): Execution state and judge_status for the planned cell.

    Returns:
        bool: True for execution failure or needs_review, or a completed
            answer whose judge is assessed or not_assessable. Pending or
            in-flight judging is not final.
    """
    return row["state"] in {"first_turn_failed", "target_failed", "needs_review"} or (
        row["state"] == "completed" and row["judge_status"] in {"assessed", "not_assessable"}
    )


def validate_campaign_rows(campaign, attempts, release=None):
    """Require the full planned matrix before computing report denominators.

    Args:
        campaign (dict): Frozen status, corpus identity, source_map,
            planned_count and config.selected_case_ids.
        attempts (list[dict]): One planned row per selected case and each of
            the four variant/generator cells, including unsuccessful rows.
        release (Release | None): Optional selected-partition validation.

    Returns:
        list[dict]: Validated primary projections in input row order.

    Raises:
        ValueError: Matrix, planned denominator, inventory or metrics changed.
    """
    source_map_contract(campaign["source_map"], release, require_ready=True)
    if not campaign.get("corpus_row_id") or not campaign.get("fingerprint"):
        raise ValueError(f"Campaign lacks corpus identity: {campaign['id']}")
    if campaign["status"] not in {"frozen", "running", "incomplete", "complete"}:
        raise ValueError(f"Unsupported campaign status: {campaign['id']}")
    if len(attempts) != campaign["planned_count"] or not attempts:
        raise ValueError(
            f"Campaign planned denominator differs from attempt rows: {campaign['id']}"
        )
    cells = {(row["case_id"], row["variant"], row["generator"]) for row in attempts}
    case_ids = {row["case_id"] for row in attempts}
    configured_ids = campaign["config"].get("selected_case_ids")
    if (
        not isinstance(configured_ids, list)
        or any(not isinstance(case_id, str) for case_id in configured_ids)
        or configured_ids != sorted(set(configured_ids))
    ):
        raise ValueError(f"Campaign selected case IDs are invalid: {campaign['id']}")
    if (
        len(cells) != len(attempts)
        or cells
        != {
            (case_id, variant, generator)
            for case_id in case_ids
            for variant in VARIANTS
            for generator in GENERATORS
        }
        or case_ids != set(configured_ids or ())
        or campaign["planned_count"] != 4 * len(case_ids)
    ):
        raise ValueError(f"Campaign cells differ from current complete matrix: {campaign['id']}")
    if release is not None:
        selected_cases(release, campaign["config"])
    return [validate_projection_row(attempt) for attempt in attempts]


def validate_campaign_corpus(campaign, corpus, release=None):
    """Bind a frozen campaign to its ready corpus and artifact fingerprint.

    Args:
        campaign (dict): Corpus/release/partition identities, artifact/config/
            rubric/code hashes, source_map and config source inventory digest.
        corpus (dict | None): Ready corpus row with id, corpus_id, manifest
            hashes and source_map; None denotes a missing ledger record.
        release (Release | None): Optional current release identity checks.

    Raises:
        ValueError: Inventory, artifact identity or recomputed fingerprint
            differs from the immutable campaign contract.
    """
    if corpus is None or campaign["corpus_row_id"] != corpus["id"]:
        raise ValueError("Campaign corpus ledger is missing")
    if corpus["status"] != "ready" or campaign["source_map"] != corpus["source_map"]:
        raise ValueError("Campaign corpus inventory differs from ready corpus")
    if (
        campaign["manifest_sha256"] != corpus["manifest_sha256"]
        or campaign["artifact_manifest_sha256"] != corpus["artifact_manifest_sha256"]
    ):
        raise ValueError("Campaign corpus manifest differs from ledger")
    source_map_contract(campaign["source_map"], release, require_ready=True)
    if campaign["config"].get("source_inventory_sha256") != digest(
        campaign["source_map"]["documents"]
    ):
        raise ValueError("Frozen release source inventory digest changed")
    if release is not None and (
        corpus["corpus_id"] != release.manifest["corpus_id"]
        or campaign["release_id"] != release.release_id
        or campaign["split"] != release.partition
    ):
        raise ValueError("Campaign release or case partition differs from corpus")
    fingerprint = digest(
        {
            "corpus_id": corpus["corpus_id"],
            "partition": campaign["split"],
            "release_id": campaign["release_id"],
            "manifest_sha256": campaign["manifest_sha256"],
            "questions_sha256": campaign["questions_sha256"],
            "gold_sha256": campaign["gold_sha256"],
            "extraction_map_sha256": campaign["extraction_map_sha256"],
            "config_sha256": campaign["config_sha256"],
            "rubric_sha256": campaign["rubric_sha256"],
            "code_sha256": campaign["code_sha256"],
        }
    )
    if campaign["fingerprint"] != fingerprint:
        raise ValueError("Campaign fingerprint differs from current contract")


class CorpusLedger:
    """Persist resumable ownership of one shared evaluation corpus.

    Documents and collection identity may be recorded while importing. Ready
    publication verifies the live inventory and prevents further source-map
    changes through this ledger; campaigns reuse that ready corpus.
    """

    def __init__(self, store):
        self.store = store

    def by_corpus_id(self, corpus_id):
        with self.store.engine.connect() as conn:
            return _record(
                conn.execute(
                    text("SELECT * FROM evaluation_corpora WHERE corpus_id=:corpus_id"),
                    {"corpus_id": corpus_id},
                )
                .mappings()
                .first()
            )

    def corpus(self, corpus_row_id):
        with self.store.engine.connect() as conn:
            return _record(
                conn.execute(
                    text("SELECT * FROM evaluation_corpora WHERE id=:id"), {"id": corpus_row_id}
                )
                .mappings()
                .first()
            )

    @contextmanager
    def coordinator(self, corpus_id):
        """Exclude concurrent import coordinators for the same corpus.

        Args:
            corpus_id (str): Content-derived manifest corpus identity.

        Yields:
            None: While a session advisory lock is held; released on exit.

        Raises:
            RuntimeError: Another importer already holds the corpus lock.
        """
        key = int.from_bytes(hashlib.sha256(corpus_id.encode()).digest()[:8], "big", signed=True)
        with self.store.engine.connect() as conn:
            if not conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
            ).scalar_one():
                raise RuntimeError("Another importer is running for this corpus")
            try:
                yield
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})

    def create_import(self, release):
        """Create or reuse an import ledger without uploading originals.

        Args:
            release (Release): Validated corpus manifest and retained file hashes.

        Returns:
            dict: Existing or inserted corpus row with id, corpus_id, status,
                manifest hashes and source_map. A new row is importing with null
                collection identity/revision and an empty document map.

        Raises:
            ValueError: The same corpus ID is associated with different manifest
                bytes or an invalid existing map.
        """
        existing = self.by_corpus_id(release.manifest["corpus_id"])
        if existing:
            if (
                existing["manifest_sha256"] != release.hashes["manifest_sha256"]
                or existing["artifact_manifest_sha256"]
                != release.manifest["artifact_manifest_sha256"]
            ):
                raise ValueError("Corpus ID has different manifest bytes")
            source_map_contract(existing["source_map"], release)
            return existing
        corpus_row_id = str(uuid.uuid4())
        with self.store.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO evaluation_corpora(id,corpus_id,status,manifest_sha256,artifact_manifest_sha256,source_map) "
                    "VALUES (:id,:corpus_id,'importing',:manifest_sha256,:artifact_manifest_sha256,CAST(:source_map AS jsonb)) "
                    "ON CONFLICT(corpus_id) DO NOTHING"
                ),
                {
                    "id": corpus_row_id,
                    "corpus_id": release.manifest["corpus_id"],
                    "manifest_sha256": release.hashes["manifest_sha256"],
                    "artifact_manifest_sha256": release.manifest["artifact_manifest_sha256"],
                    "source_map": canonical(
                        {"collection_id": None, "collection_revision": None, "documents": {}}
                    ),
                },
            )
        result = self.by_corpus_id(release.manifest["corpus_id"])
        if (
            result is None
            or result["manifest_sha256"] != release.hashes["manifest_sha256"]
            or result["artifact_manifest_sha256"] != release.manifest["artifact_manifest_sha256"]
        ):
            raise ValueError("Corpus ID has different manifest bytes")
        source_map_contract(result["source_map"], release)
        return result

    def source_map(
        self,
        corpus_row_id,
        *,
        collection_id=None,
        collection_revision=None,
        doc_id=None,
        document=None,
    ):
        """Persist import progress while keeping document identities stable.

        Lock the corpus row and record optional collection, document or revision
        progress. Status can advance for an existing source, but its source ID and
        original hash cannot be replaced. Mutations commit together.

        Args:
            corpus_row_id (str): Corpus ledger row UUID.
            collection_id (str | None): Shared collection UUID; None leaves it unchanged.
            collection_revision (int | None): Nonnegative revision to record;
                None leaves it unchanged.
            doc_id (str | None): Manifest key required when document is supplied.
            document (dict | None): source_id, sha256 and status to record;
                None leaves document mappings unchanged.

        Returns:
            dict: Persisted collection_id, collection_revision and documents map.

        Raises:
            ValueError: Corpus is no longer importing, identities change or the
                resulting map violates the shared collection contract.
        """
        with self.store.engine.begin() as conn:
            row = (
                conn.execute(
                    text(
                        "SELECT source_map,status FROM evaluation_corpora WHERE id=:id FOR UPDATE"
                    ),
                    {"id": corpus_row_id},
                )
                .mappings()
                .one()
            )
            if row["status"] != "importing":
                raise ValueError("Source map can only change during import")
            mapping = source_map_contract(row["source_map"])
            if collection_id is not None:
                if mapping["collection_id"] not in (None, collection_id):
                    raise ValueError("Shared collection identity changed")
                mapping["collection_id"] = collection_id
            if document is not None:
                if doc_id is None or mapping["collection_id"] is None:
                    raise ValueError("Imported document needs a shared collection")
                old = mapping["documents"].get(doc_id)
                if old and (
                    old["source_id"] != document["source_id"] or old["sha256"] != document["sha256"]
                ):
                    raise ValueError("Imported document identity changed")
                mapping["documents"][doc_id] = document
            if collection_revision is not None:
                mapping["collection_revision"] = collection_revision
            source_map_contract(mapping)
            conn.execute(
                text(
                    "UPDATE evaluation_corpora SET source_map=CAST(:source_map AS jsonb),updated_at=now() WHERE id=:id"
                ),
                {"id": corpus_row_id, "source_map": canonical(mapping)},
            )
            return mapping

    def ensure_collection(self, corpus):
        """Claim a deterministic collection for an importing corpus.

        Create its search partition and record ownership in the same transaction.
        An existing collection cannot be claimed merely because its deterministic
        ID matches; a resumed import must already record that ownership.

        Args:
            corpus (dict): Corpus row with id and corpus_id.

        Returns:
            str: Deterministic collection UUID, reusable by this importing corpus.

        Raises:
            ValueError: Import state, collection title or prior ownership conflicts.
        """
        collection_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "pfl-corpus:" + corpus["corpus_id"]))
        with self.store.engine.begin() as conn:
            existing = conn.execute(
                text("SELECT id FROM collections WHERE id=:id"), {"id": collection_id}
            ).first()
            inserted = conn.execute(
                text(
                    "INSERT INTO collections(id,title,description) VALUES (:id,:title,'Synthetic evaluation corpus') "
                    "ON CONFLICT(id) DO NOTHING"
                ),
                {"id": collection_id, "title": f"Evaluation corpus {corpus['corpus_id']}"},
            )
            row = (
                conn.execute(
                    text("SELECT id,title FROM collections WHERE id=:id"), {"id": collection_id}
                )
                .mappings()
                .one()
            )
            if row["title"] != f"Evaluation corpus {corpus['corpus_id']}":
                raise ValueError("Deterministic collection ID belongs to another corpus")
            conn.execute(
                text("SELECT ensure_collection_search_partition(CAST(:id AS uuid))"),
                {"id": collection_id},
            )
            current = (
                conn.execute(
                    text(
                        "SELECT status,source_map FROM evaluation_corpora WHERE id=:id FOR UPDATE"
                    ),
                    {"id": corpus["id"]},
                )
                .mappings()
                .one()
            )
            if current["status"] != "importing":
                raise ValueError("Only an importing corpus can claim a collection")
            mapping = source_map_contract(current["source_map"])
            if mapping["collection_id"] not in (None, collection_id):
                raise ValueError("Corpus collection identity changed")
            if mapping["collection_id"] is None and (
                existing is not None or inserted.rowcount == 0
            ):
                raise ValueError("Deterministic collection ID is not owned by this corpus")
            mapping["collection_id"] = collection_id
            conn.execute(
                text(
                    "UPDATE evaluation_corpora SET source_map=CAST(:source_map AS jsonb),updated_at=now() WHERE id=:id"
                ),
                {"id": corpus["id"], "source_map": canonical(mapping)},
            )
        return collection_id

    def mark_ready(self, corpus_row_id, source_map):
        """Commit corpus readiness only against the unchanged complete inventory.

        Args:
            corpus_row_id (str): Importing corpus row UUID.
            source_map (dict): Complete ready collection/documents map with its
                final revision, equal to the map currently stored in the ledger.

        Raises:
            ValueError: Live inventory or ledger state changed before publication.
        """
        source_map_contract(source_map, require_ready=True)
        with self.store.engine.begin() as conn:
            lock_and_verify_collection(conn, source_map)
            current = (
                conn.execute(
                    text(
                        "SELECT status,source_map FROM evaluation_corpora WHERE id=:id FOR UPDATE"
                    ),
                    {"id": corpus_row_id},
                )
                .mappings()
                .one()
            )
            if current["status"] != "importing" or current["source_map"] != source_map:
                raise ValueError("Corpus source map changed before import commit")
            conn.execute(
                text("UPDATE evaluation_corpora SET status='ready',updated_at=now() WHERE id=:id"),
                {"id": corpus_row_id},
            )


class Ledger:
    """Own frozen planned cells, durable attempt history and primary results.

    Retries update current execution/projections while the first finalized
    result remains immutable. Planned rows therefore retain a stable comparison
    denominator across failures and explicit corrections.
    """

    def __init__(self, store):
        self.store = store

    def by_fingerprint(self, fingerprint):
        with self.store.engine.connect() as conn:
            return _record(
                conn.execute(
                    text("SELECT * FROM evaluation_campaigns WHERE fingerprint=:fingerprint"),
                    {"fingerprint": fingerprint},
                )
                .mappings()
                .first()
            )

    def campaign(self, campaign_id):
        with self.store.engine.connect() as conn:
            return _record(
                conn.execute(
                    text("SELECT * FROM evaluation_campaigns WHERE id=:id"), {"id": campaign_id}
                )
                .mappings()
                .first()
            )

    def freeze(self, release, config, corpus, service):
        """Freeze artifacts and all four comparison cells over one ready corpus.

        Resolve deterministic execution order, source inventory, code and rubric
        hashes, then prepare pending metric projections against reviewed evidence.
        Under transaction locks, recheck corpus ownership and insert campaign plus
        planned rows, or reuse the identical fingerprint. No inference is performed.

        Args:
            release (Release): Gold-bearing selected partition and extraction review.
            config (dict): selected_case_ids, selected_variant, selected_generator,
                judge_model=gpt-6-sol and price_snapshot_id, plus frozen runtime
                settings. Mutated to add inventory digest, execution_order and a
                default case_workers count.
            corpus (dict): Ready corpus row, matching manifest hashes and source_map.
            service (Service): Reads live scope and extracted evidence via HTTP.

        Returns:
            dict: Frozen campaign row with artifact hashes, fingerprint, source_map,
                config and planned_count=4 times the selected case count.

        Raises:
            ValueError: Corpus, configuration, selection, evidence or locked
                inventory fails the frozen contract.
        """
        if corpus["status"] != "ready" or corpus["corpus_id"] != release.manifest["corpus_id"]:
            raise ValueError("Corpus import must be ready before freeze")
        if (
            corpus["manifest_sha256"] != release.hashes["manifest_sha256"]
            or corpus["artifact_manifest_sha256"] != release.manifest["artifact_manifest_sha256"]
        ):
            raise ValueError("Corpus manifest changed before freeze")
        source_map = source_map_contract(corpus["source_map"], release, require_ready=True)
        config["source_inventory_sha256"] = digest(source_map["documents"])
        case_worker_count(config)
        if (
            config.get("judge_model") != "gpt-6-sol"
            or config.get("selected_variant") not in VARIANTS
            or config.get("selected_generator") not in GENERATORS
        ):
            raise ValueError("Freeze config must select a development choice and Sol judge")
        cases = selected_cases(release, config)
        planned_order = execution_order(cases)
        if "execution_order" in config and config["execution_order"] != planned_order:
            raise ValueError(
                "Configured execution order differs from deterministic family rotation"
            )
        config["execution_order"] = planned_order
        config.setdefault("case_workers", 1)
        current_code = code_hash()
        current_rubric = instruction_hash("judge")
        current_config = digest(config)
        fingerprint = digest(
            {
                "corpus_id": corpus["corpus_id"],
                "partition": release.partition,
                "release_id": release.release_id,
                "manifest_sha256": release.hashes["manifest_sha256"],
                "questions_sha256": release.hashes["questions_sha256"],
                "gold_sha256": release.hashes["gold_sha256"],
                "extraction_map_sha256": release.hashes["extraction_map_sha256"],
                "config_sha256": current_config,
                "rubric_sha256": current_rubric,
                "code_sha256": current_code,
            }
        )
        campaign_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "pfl-eval:" + fingerprint))
        rows = []
        evidence = CorpusEvidence(release, source_map, service)
        for case_id, case in sorted(cases.items()):
            case_data = evidence.case_data(case)
            metadata = {
                "family_id": case["family_id"],
                "patient_id": case.get("patient_id"),
                "question_class": case["question_class"],
                "anchor_document_class": case["anchor_document_class"],
                "anchor_doc_id": case["anchor_doc_id"],
                "query_language": case["query_language"],
                "difficulty": case["difficulty"],
                "turn_count": len(case["turns"]),
            }
            for variant in VARIANTS:
                for generator in GENERATORS:
                    planned = {
                        "id": str(uuid.uuid4()),
                        "campaign_id": campaign_id,
                        "case_id": case_id,
                        "variant": variant,
                        "generator": generator,
                        "metadata": canonical(metadata),
                    }
                    projection = evidence.metric_row(
                        {
                            **planned,
                            "state": "pending",
                            "first_run_id": None,
                            "target_run_id": None,
                            "judge_output": None,
                            "judge_status": None,
                            "judge_na_reason": None,
                        },
                        self.store,
                        case_data=case_data,
                    )
                    planned["metrics"] = canonical(
                        {
                            "projection": projection,
                            "cost_estimate": cost_for_attempt(
                                {**planned, "attempts": []}, self.store, config["price_snapshot_id"]
                            ),
                        }
                    )
                    rows.append(planned)
        with self.store.engine.begin() as conn:
            lock_key = int.from_bytes(bytes.fromhex(fingerprint)[:8], "big", signed=True)
            conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_key})
            lock_and_verify_collection(conn, source_map)
            locked_corpus = (
                conn.execute(
                    text("SELECT status,source_map FROM evaluation_corpora WHERE id=:id FOR SHARE"),
                    {"id": corpus["id"]},
                )
                .mappings()
                .one()
            )
            if locked_corpus["status"] != "ready" or locked_corpus["source_map"] != source_map:
                raise ValueError("Corpus import changed during freeze")
            existing = (
                conn.execute(
                    text(
                        "SELECT * FROM evaluation_campaigns WHERE fingerprint=:fingerprint FOR UPDATE"
                    ),
                    {"fingerprint": fingerprint},
                )
                .mappings()
                .first()
            )
            if existing:
                if (
                    existing["corpus_row_id"] != uuid.UUID(corpus["id"])
                    or existing["source_map"] != source_map
                ):
                    raise ValueError("Frozen campaign corpus identity changed")
                return _record(existing)
            conn.execute(
                text(
                    "INSERT INTO evaluation_campaigns(id,corpus_row_id,fingerprint,release_id,split,status,manifest_sha256,questions_sha256,gold_sha256,extraction_map_sha256,artifact_manifest_sha256,config_sha256,rubric_sha256,code_sha256,selected_variant,selected_generator,planned_count,source_map,config,frozen_at) "
                    "VALUES (:id,:corpus_row_id,:fingerprint,:release_id,:split,'frozen',:manifest_sha256,:questions_sha256,:gold_sha256,:extraction_map_sha256,:artifact_manifest_sha256,:config_sha256,:rubric_sha256,:code_sha256,:selected_variant,:selected_generator,:planned_count,CAST(:source_map AS jsonb),CAST(:config AS jsonb),now())"
                ),
                {
                    "id": campaign_id,
                    "corpus_row_id": corpus["id"],
                    "fingerprint": fingerprint,
                    "release_id": release.release_id,
                    "split": release.partition,
                    "manifest_sha256": release.hashes["manifest_sha256"],
                    "questions_sha256": release.hashes["questions_sha256"],
                    "gold_sha256": release.hashes["gold_sha256"],
                    "extraction_map_sha256": release.hashes["extraction_map_sha256"],
                    "artifact_manifest_sha256": release.manifest["artifact_manifest_sha256"],
                    "config_sha256": current_config,
                    "rubric_sha256": current_rubric,
                    "code_sha256": current_code,
                    "selected_variant": config["selected_variant"],
                    "selected_generator": config["selected_generator"],
                    "planned_count": len(rows),
                    "source_map": canonical(source_map),
                    "config": canonical(config),
                },
            )
            for row in rows:
                conn.execute(
                    text(
                        "INSERT INTO evaluation_attempts(id,campaign_id,case_id,variant,generator,metadata,state,metrics) VALUES (:id,:campaign_id,:case_id,:variant,:generator,CAST(:metadata AS jsonb),'pending',CAST(:metrics AS jsonb))"
                    ),
                    row,
                )
        return self.campaign(campaign_id)

    def rows(self, campaign_id, states=None):
        """Return planned cells in their frozen dispatch order.

        Args:
            campaign_id (str): Campaign UUID.
            states (Iterable[str] | None): Optional execution-state filter; empty
                or None selects all planned rows.

        Returns:
            list[dict]: Attempt rows ordered by config.execution_order, or by cell
                identity for campaigns without that order.

        Raises:
            ValueError: Frozen order repeats a cell or omits a returned planned row.
        """
        with self.store.engine.connect() as conn:
            sql = "SELECT * FROM evaluation_attempts WHERE campaign_id=:campaign_id"
            params = {"campaign_id": campaign_id}
            if states:
                sql += " AND state=ANY(:states)"
                params["states"] = list(states)
            rows = [_record(row) for row in conn.execute(text(sql), params).mappings()]
        campaign = self.campaign(campaign_id)
        order = campaign["config"].get("execution_order") if campaign else None
        if order:
            positions = {tuple(cell): index for index, cell in enumerate(order)}
            if len(positions) != len(order):
                raise ValueError("Frozen execution order contains duplicate cells")
            if any(
                (row["case_id"], row["variant"], row["generator"]) not in positions for row in rows
            ):
                raise ValueError("Planned row absent from frozen execution order")
            rows.sort(key=lambda row: positions[(row["case_id"], row["variant"], row["generator"])])
        else:
            rows.sort(key=lambda row: (row["case_id"], row["variant"], row["generator"]))
        return rows

    @contextmanager
    def coordinator(self, campaign_id):
        """Exclude overlapping execution, resume, retry and reconciliation.

        Args:
            campaign_id (str): Campaign UUID used to derive the advisory lock.

        Yields:
            None: While the campaign session lock is held, until context exit.

        Raises:
            RuntimeError: Another coordinator holds this campaign lock.
        """
        key = int.from_bytes(
            hashlib.sha256(uuid.UUID(campaign_id).bytes).digest()[:8], "big", signed=True
        )
        with self.store.engine.connect() as conn:
            acquired = conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
            ).scalar_one()
            if not acquired:
                raise RuntimeError("Another coordinator is running this campaign")
            try:
                yield
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})

    def row(self, row_id):
        with self.store.engine.connect() as conn:
            return _record(
                conn.execute(text("SELECT * FROM evaluation_attempts WHERE id=:id"), {"id": row_id})
                .mappings()
                .first()
            )

    def mutate(self, row_id, operation):
        """Apply one durable attempt transition while protecting the primary result.

        Args:
            row_id (str): Planned attempt-row UUID to lock for update.
            operation (Callable[[dict], dict]): Transition receiving the row with
                attempts, metrics, state, run IDs and judge fields; returns the
                updated row to persist. Existing metrics.primary cannot change.

        Returns:
            dict: Updated row after its execution/history/judge/metric fields commit.

        Raises:
            ValueError: The transition modifies an already frozen primary record.
        """
        with self.store.engine.begin() as conn:
            row = _record(
                conn.execute(
                    text("SELECT * FROM evaluation_attempts WHERE id=:id FOR UPDATE"),
                    {"id": row_id},
                )
                .mappings()
                .one()
            )
            frozen_primary = copy.deepcopy(row["metrics"].get("primary"))
            updated = operation(row)
            if frozen_primary is not None and updated["metrics"].get("primary") != frozen_primary:
                raise ValueError("Frozen primary result cannot change")
            conn.execute(
                text(
                    "UPDATE evaluation_attempts SET state=:state,attempts=CAST(:attempts AS jsonb),first_run_id=:first_run_id,target_run_id=:target_run_id,judge_output=CAST(:judge_output AS jsonb),judge_status=:judge_status,judge_na_reason=:judge_na_reason,metrics=CAST(:metrics AS jsonb),updated_at=now() WHERE id=:id"
                ),
                {
                    "id": row_id,
                    "state": updated["state"],
                    "attempts": canonical(updated["attempts"]),
                    "first_run_id": updated["first_run_id"],
                    "target_run_id": updated["target_run_id"],
                    "judge_output": canonical(updated["judge_output"]),
                    "judge_status": updated["judge_status"],
                    "judge_na_reason": updated["judge_na_reason"],
                    "metrics": canonical(updated["metrics"]),
                },
            )
            return updated

    def campaign_status(self, campaign_id):
        """Persist completion of execution and judge resolution, irrespective of pass.

        Args:
            campaign_id (str): Campaign UUID whose planned rows are inspected.

        Returns:
            str: complete only when every row is completed, first_turn_failed or
                target_failed and judging is assessed or not_assessable; otherwise
                incomplete, including needs_review. Updates the campaign row.
        """
        rows = self.rows(campaign_id)
        status = (
            "complete"
            if rows
            and all(
                row["state"] in COMPLETE_STATES
                and row["judge_status"] in {"assessed", "not_assessable"}
                for row in rows
            )
            else "incomplete"
        )
        with self.store.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE evaluation_campaigns SET status=:status,updated_at=now() WHERE id=:id"
                ),
                {"id": campaign_id, "status": status},
            )
        return status

    def mark_running(self, campaign_id):
        with self.store.engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE evaluation_campaigns SET status='running',updated_at=now() WHERE id=:id AND status IN ('frozen','incomplete','running','complete')"
                ),
                {"id": campaign_id},
            )


class Service:
    """Use authenticated application HTTP paths for corpus and dialogue execution.

    The service token is required. Import shares one collection across cases;
    model runs use normal conversation/run endpoints and their idempotency IDs.
    """

    def __init__(self, settings):
        if not settings.service_token:
            raise ValueError("PFL_SERVICE_TOKEN is required")
        self.settings = settings
        self.client = httpx.Client(
            base_url=settings.app_origin,
            headers={"Authorization": f"Bearer {settings.service_token}"},
            timeout=60,
        )

    def close(self):
        self.client.close()

    def request(self, method, path, **kwargs):
        response = self.client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code} at {path}: {response.text[:300]}")
        return response.json()

    def import_release(self, ledger, corpus, release):
        """Resume a corpus import under exclusive importer coordination.

        Reuse stored document identities and deterministic upload request IDs,
        wait for ingestion, verify the complete shared inventory, then publish
        readiness. A ready corpus is verified and reused. Earlier uploads and
        progress records may remain committed if a later step fails.

        Args:
            ledger (CorpusLedger): Durable corpus ownership/progress store.
            corpus (dict): Corpus row with id; reloaded after acquiring its lock.
            release (Release): Validated manifest and local original files.

        Returns:
            dict: Verified ready corpus row with complete source_map and revision.

        Raises:
            ValueError: Import state, document hashes, ingestion or inventory fails.
            TimeoutError: A source fails to reach an ingestion terminal state in time.
            RuntimeError: Another importer is active for the same corpus.
        """
        with ledger.coordinator(release.manifest["corpus_id"]):
            return self._import_release(ledger, ledger.corpus(corpus["id"]), release)

    def _import_release(self, ledger, corpus, release):
        if corpus["status"] not in {"importing", "ready"}:
            raise ValueError("Invalid corpus import state")
        mapping = source_map_contract(corpus["source_map"], release)
        if corpus["status"] == "ready":
            self.verify_inventory(release, mapping)
            return corpus
        collection_id = mapping["collection_id"]
        if collection_id is None:
            collection_id = ledger.ensure_collection(corpus)
        for doc_id, document in sorted(release.documents.items()):
            mapping = source_map_contract(ledger.corpus(corpus["id"])["source_map"], release)
            stored = mapping["documents"].get(doc_id)
            if stored:
                source = self._source(collection_id, stored["source_id"])
                if source is None or source["sha256"] != document["sha256"]:
                    raise ValueError("Imported source disappeared or changed")
                if source["status"] != "ready":
                    source = self.wait_source(collection_id, stored["source_id"])
                if source["status"] != "ready":
                    raise ValueError(f"Ingestion failed for {doc_id}")
                if stored["status"] != "ready":
                    ledger.source_map(
                        corpus["id"],
                        doc_id=doc_id,
                        document={
                            "source_id": source["id"],
                            "sha256": document["sha256"],
                            "status": "ready",
                        },
                    )
                continue
            path = release.directory / document["path"]
            request_id = (
                "eval-"
                + hashlib.sha256(f"{release.manifest['corpus_id']}:{doc_id}".encode()).hexdigest()[
                    :48
                ]
            )
            with path.open("rb") as handle:
                uploaded = self.request(
                    "POST",
                    f"/api/collections/{collection_id}/sources",
                    files={
                        "file": (
                            path.name,
                            handle,
                            "application/pdf" if document["format"] == "pdf" else "text/plain",
                        )
                    },
                    data={
                        "document_class": document["document_class"],
                        "request_id": request_id,
                    },
                )
            if uploaded["sha256"] != document["sha256"]:
                raise ValueError("Uploaded document hash mismatch")
            ledger.source_map(
                corpus["id"],
                doc_id=doc_id,
                document={
                    "source_id": uploaded["id"],
                    "sha256": document["sha256"],
                    "status": uploaded["status"],
                },
            )
            source = self.wait_source(collection_id, uploaded["id"])
            ledger.source_map(
                corpus["id"],
                doc_id=doc_id,
                document={
                    "source_id": source["id"],
                    "sha256": document["sha256"],
                    "status": source["status"],
                },
            )
            if source["status"] != "ready":
                raise ValueError(f"Ingestion failed for {doc_id}")
        mapping = ledger.corpus(corpus["id"])["source_map"]
        collection = self.verify_inventory(release, mapping, check_revision=False)
        ledger.source_map(corpus["id"], collection_revision=collection["revision"])
        self.verify_inventory(release, ledger.corpus(corpus["id"])["source_map"])
        ledger.mark_ready(corpus["id"], ledger.corpus(corpus["id"])["source_map"])
        return ledger.corpus(corpus["id"])

    def verify_inventory(self, release, mapping, check_revision=True):
        """Verify all paginated sources and stable collection counters via HTTP.

        Args:
            release (Release): Expected complete document inventory and hashes.
            mapping (dict): Shared collection and every manifest document mapped
                to source_id, sha256 and ready status.
            check_revision (bool): Require the frozen revision; False permits
                discovering the final import revision before it is recorded.

        Returns:
            dict: Initial collection record with id, revision and source/ready/
                unavailable counters, unchanged after source pagination.

        Raises:
            ValueError: Documents, readiness, counters or revision change, including
                a collection change during pagination.
        """
        source_map_contract(mapping, release, require_ready=False)
        if not mapping["collection_id"] or set(mapping["documents"]) != set(release.documents):
            raise ValueError("Shared collection import is incomplete")
        collection = self.request("GET", f"/api/collections/{mapping['collection_id']}")
        if check_revision and collection["revision"] != mapping["collection_revision"]:
            raise ValueError("Shared collection revision changed")
        if (
            collection["source_count"] != len(release.documents)
            or collection["ready_count"] != len(release.documents)
            or collection["unavailable_count"] != 0
        ):
            raise ValueError("Shared collection counters differ from release")
        expected = {
            item["source_id"]: (doc_id, release.documents[doc_id]["sha256"])
            for doc_id, item in mapping["documents"].items()
        }
        cursor = None
        while True:
            params = {"limit": 100}
            if cursor is not None:
                params["cursor"] = cursor
            page = self.request(
                "GET", f"/api/collections/{mapping['collection_id']}/sources", params=params
            )
            for source in page["items"]:
                item = expected.pop(source["id"], None)
                if item is None:
                    raise ValueError("Shared collection source inventory differs from release")
                doc_id, sha256 = item
                mapped = mapping["documents"][doc_id]
                if (
                    mapped["sha256"] != sha256
                    or mapped["status"] != "ready"
                    or source["status"] != "ready"
                    or source["sha256"] != sha256
                    or source["collection_id"] != mapping["collection_id"]
                ):
                    raise ValueError("Shared collection document differs from release")
            cursor = page["next_cursor"]
            if cursor is None:
                break
        if expected:
            raise ValueError("Shared collection source inventory differs from release")
        after = self.request("GET", f"/api/collections/{mapping['collection_id']}")
        if (
            after["revision"] != collection["revision"]
            or after["source_count"] != collection["source_count"]
            or after["ready_count"] != collection["ready_count"]
            or after["unavailable_count"] != collection["unavailable_count"]
        ):
            raise ValueError("Shared collection changed during inventory verification")
        return collection

    def verify_collection_scope(self, mapping):
        """Require the frozen collection revision before using runtime evidence.

        Args:
            mapping (dict): Frozen collection identity/revision and document map.

        Returns:
            dict: Verified collection snapshot with counters and snapshot_hash.

        Raises:
            ValueError: HTTP request fails or current scope differs from the map.
        """
        collection = self.request("GET", f"/api/collections/{mapping['collection_id']}")
        return frozen_collection_scope(mapping, collection)

    def _source(self, collection_id, source_id):
        source = self.request("GET", f"/api/sources/{source_id}")
        return source if source["collection_id"] == collection_id else None

    def wait_source(self, collection_id, source_id, timeout_seconds=600):
        """Wait for indexing to make an imported source ready or failed.

        Args:
            collection_id (str): Expected shared collection UUID.
            source_id (str): Uploaded source UUID.
            timeout_seconds (int | float): Maximum polling duration in seconds.

        Returns:
            dict: Source record in ready or failed state; failed ingestion is
                returned for the importer to reject rather than raised here.

        Raises:
            ValueError: Source is outside the collection or an HTTP request fails.
            TimeoutError: Neither terminal ingestion state is observed by the deadline.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            source = self._source(collection_id, source_id)
            if source is None:
                raise ValueError("Imported source disappeared")
            if source["status"] in {"ready", "failed"}:
                return source
            time.sleep(1)
        raise TimeoutError("Source ingestion timed out")

    def spans(self, source_id):
        spans = []
        cursor = None
        while True:
            params = {"limit": 100}
            if cursor is not None:
                params["cursor"] = cursor
            page = self.request("GET", f"/api/sources/{source_id}/spans", params=params)
            spans.extend(page["items"])
            cursor = page["next_cursor"]
            if cursor is None:
                return spans

    def run(self, conversation_id, request_id, question, model, variant):
        """Submit one evaluation turn through the normal run idempotency boundary.

        Args:
            conversation_id (str): Conversation over the shared corpus collection.
            request_id (str): Durable attempt/turn key for reusing an existing run.
            question (str): Authored user turn text, without gold or scope filters.
            model (str): Generator model for the comparison cell.
            variant (str): Retrieval architecture, V0 or V3 for this campaign.

        Returns:
            dict: Application run record with id, request_id and execution status;
                completion requires polling. retry_of_run_id is submitted as None.

        Raises:
            ValueError: Application rejects the request.
        """
        return self.request(
            "POST",
            f"/api/conversations/{conversation_id}/runs",
            json={
                "request_id": request_id,
                "question": question,
                "model": model,
                "variant": variant,
                "retry_of_run_id": None,
            },
        )

    def wait_run(self, run_id, timeout_seconds=900):
        """Observe the existing run until execution reaches a terminal state.

        Args:
            run_id (str): Previously submitted application run UUID.
            timeout_seconds (int | float): Maximum polling duration in seconds.

        Returns:
            dict: Run with succeeded, failed, cancelled or interrupted status;
                execution failure is returned, not converted into another model call.

        Raises:
            TimeoutError: No terminal state is observed by the deadline; the run
                is not automatically cancelled by this method.
            ValueError: Application rejects a polling request.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            run = self.request("GET", f"/api/runs/{run_id}")
            if run["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                return run
            time.sleep(1)
        raise TimeoutError("Run did not reach a terminal state")


class CorpusEvidence:
    """Resolve reviewed gold references to spans in the frozen shared corpus.

    Offline scope selects evidence for judging and diagnostics only. Generator
    retrieval still searches the entire shared collection. Extracted documents
    are cached within this object while live collection revision checks guard use.
    """

    def __init__(self, release, source_map, service):
        self.release = release
        self.source_map = source_map
        self.service = service
        self.document_cache = {}

    def load(self, case):
        """Load runtime evidence needed for one case's offline assessment.

        Combine case scope, required gold scope and referenced gold documents,
        fetch uncached spans, validate unique IDs/source ownership and check the
        frozen collection before and after loading.

        Args:
            case (dict): case_id and scope.doc_ids for a validated release case.

        Returns:
            dict: source_map, spans keyed by runtime span ID, and by_document
                mapping manifest doc_id to extracted span lists. Span records
                include source_id, page, line_start, line_end and text.

        Raises:
            ValueError: Collection scope or extracted span ownership/uniqueness changed.
        """
        self.service.verify_collection_scope(self.source_map)
        spans = {}
        by_document = {}
        gold = self.release.golds[case["case_id"]]
        doc_ids = set(case["scope"]["doc_ids"]) | set(gold.get("required_scope_doc_ids", []))
        doc_ids.update(
            reference["doc_id"]
            for claim in gold.get("required_claims", [])
            for option in claim.get("acceptable_evidence_sets", [])
            for reference in option
        )
        for doc_id in sorted(doc_ids):
            source = self.source_map["documents"][doc_id]
            if doc_id not in self.document_cache:
                self.document_cache[doc_id] = self.service.spans(source["source_id"])
            document_spans = self.document_cache[doc_id]
            ids = [span["id"] for span in document_spans]
            if (
                any(span["source_id"] != source["source_id"] for span in document_spans)
                or len(ids) != len(set(ids))
                or set(ids) & set(spans)
            ):
                raise ValueError("Extracted spans do not match the imported source")
            by_document[doc_id] = document_spans
            spans.update({span["id"]: span for span in document_spans})
        self.service.verify_collection_scope(self.source_map)
        return {"source_map": self.source_map, "spans": spans, "by_document": by_document}

    def ids_for(self, reference, loaded):
        """Translate a reviewed authored reference into complete runtime span IDs.

        Match every intact extracted locator uniquely and verify the normalized
        authored quote appears across the selected spans. Non-intact review
        outcomes deliberately produce an unavailable sentinel rather than an
        empty set that could appear to satisfy a claim.

        Args:
            reference (dict): doc_id, page, line_start, line_end and quote.
            loaded (dict): Evidence load containing by_document and spans maps.

        Returns:
            list[str]: Sorted unique matched IDs, or one unavailable:<digest>
                sentinel for mismatch/unavailable reviewed extraction.

        Raises:
            ValueError: Intact locators, document membership, uniqueness or quote
                support fail to resolve against extracted evidence.
        """
        entry = self.release.entries[ref_key(reference)]
        if entry["outcome"] != "intact":
            return ["unavailable:" + digest(reference)]
        doc_id = reference["doc_id"]
        if doc_id not in loaded["by_document"]:
            raise ValueError("Gold reference is outside the imported release")
        document_spans = loaded["by_document"][doc_id]
        locators = entry.get("extracted_locators")
        if not isinstance(locators, list) or not locators:
            raise ValueError("Intact reviewed reference lacks extracted locators")
        matched = []
        for locator in locators:
            options = [span for span in document_spans if self._matches(span, locator)]
            if len(options) != 1:
                raise ValueError("Reviewed extracted locator does not resolve uniquely")
            matched.append(options[0]["id"])
        if not matched:
            raise ValueError("Intact reference has no matching span")
        selected_text = " ".join(span["text"] for span in document_spans if span["id"] in matched)
        if " ".join(reference["quote"].split()) not in " ".join(selected_text.split()):
            raise ValueError("Reviewed quote does not appear in matched extracted spans")
        return sorted(set(matched))

    def _matches(self, span, locator):
        """Apply only the locator constraints recorded by extraction review.

        Args:
            span (dict): Extracted page, line_start, line_end and text fields.
            locator (dict): Any subset of page, line_start, line_end, text_sha256
                and quote. Coordinates/hash must match exactly and quote must be
                a literal substring; omitted fields impose no restriction.

        Returns:
            bool: True when all supplied constraints match, including an empty
                locator. Uniqueness and authored-quote support are checked by ids_for.
        """
        if "page" in locator and span["page"] != locator["page"]:
            return False
        if "line_start" in locator and span["line_start"] != locator["line_start"]:
            return False
        if "line_end" in locator and span["line_end"] != locator["line_end"]:
            return False
        if (
            "text_sha256" in locator
            and hashlib.sha256(span["text"].encode()).hexdigest() != locator["text_sha256"]
        ):
            return False
        if "quote" in locator and locator["quote"] not in span["text"]:
            return False
        return True

    def validate_all(self, cases=None):
        """Resolve every reviewed gold alternative before execution begins.

        Args:
            cases (dict[str, dict] | None): Case selection; None or an empty map
                checks all cases in the loaded partition. May fill the span cache.

        Raises:
            ValueError: Frozen scope or an intact gold reference cannot be resolved.
        """
        for case_id, case in (cases or self.release.cases).items():
            loaded = self.load(case)
            gold = self.release.golds[case_id]
            for claim in gold.get("required_claims", []):
                for option in claim.get("acceptable_evidence_sets", []):
                    for reference in option:
                        self.ids_for(reference, loaded)

    def case_data(self, case):
        """Prepare reusable span-level gold requirements for a planned case.

        Args:
            case (dict): Validated release case with case_id and offline scope.

        Returns:
            tuple[dict, list[dict]]: Loaded evidence and claim records containing
                claim_id plus acceptable_evidence_sets of runtime span IDs. Every
                set combines all references in that alternative; non-intact
                references retain unavailable sentinels.
        """
        gold = self.release.golds[case["case_id"]]
        loaded = self.load(case)
        gold_claims = []
        for claim in gold.get("required_claims", []):
            options = []
            for option in claim.get("acceptable_evidence_sets", []):
                ids = sorted(
                    {span_id for reference in option for span_id in self.ids_for(reference, loaded)}
                )
                options.append(ids)
            gold_claims.append({"claim_id": claim["claim_id"], "acceptable_evidence_sets": options})
        return loaded, gold_claims

    def metric_row(self, attempt, store, case_data=None):
        """Project one planned cell into execution, evidence and judge metrics.

        Use the target answer only after completed/succeeded execution, validate
        citation membership and any required complete-scope receipt, and retain
        offline gold alternatives for delivery diagnostics. Two-turn latency sums
        both successful turns; target latency measures only the evaluated turn.

        Args:
            attempt (dict): case_id, variant, generator, state, nullable first/
                target_run_id and judge_output/status/na_reason.
            store (Store): Reads run, citation span and source records.
            case_data (tuple[dict, list[dict]] | None): Optional preloaded evidence
                and gold claim records from case_data to avoid repeat loading.

        Returns:
            dict: Cell metadata, gold_required_claims, available/delivered IDs,
                execution with run_ids, trace_statuses and millisecond latencies,
                candidate citation edges, judge, machine checks and diagnostics.
                For incomplete execution answer/judge/latencies and machine check
                values are None. Missing diagnostics stay None rather than guessed.
        """
        case = self.release.cases[attempt["case_id"]]
        gold = self.release.golds[attempt["case_id"]]
        loaded, gold_claims = case_data if case_data is not None else self.case_data(case)
        anchor = self.release.documents[case["anchor_doc_id"]]
        run_ids = [value for value in (attempt["first_run_id"], attempt["target_run_id"]) if value]
        runs = [store.get_run(run_id) for run_id in run_ids]
        trace_statuses = [(run or {}).get("trace_status") or "missing" for run in runs]
        target = store.get_run(attempt["target_run_id"]) if attempt["target_run_id"] else None
        completed = (
            attempt["state"] == "completed"
            and target is not None
            and target["status"] == "succeeded"
            and target["answer"] is not None
        )
        answer = target["answer"] if completed else None
        candidate_claims = answer.get("claims", []) if answer else []
        edges = [
            {"claim_index": index, "evidence_id": evidence_id}
            for index, claim in enumerate(candidate_claims)
            for evidence_id in claim.get("evidence_ids", [])
        ]
        cited = {span["id"] for span in answer.get("citations", [])} if answer else set()
        all_cited = {edge["evidence_id"] for edge in edges}
        required_source_ids = {
            loaded["source_map"]["documents"][doc_id]["source_id"]
            for doc_id in gold.get("required_scope_doc_ids", [])
        }
        receipt = answer.get("coverage", {}) if answer else {}
        exhaustive = (
            coverage_policy(case["turns"][-1]["text"]) == "complete_scope"
            or gold["gold_status"] == "not_documented"
        )
        requires_complete = bool(answer) and (
            answer["status"] == "not_documented"
            or answer.get("coverage_requirement") == "complete_scope"
            or (answer["status"] == "supported" and exhaustive)
            or any(
                coverage_policy(claim.get("text", "")) == "complete_scope"
                for claim in candidate_claims
            )
        )
        resolved_citations = store.get_spans(sorted(all_cited)) if completed else []
        citation_sources_valid = all(
            (source := store.get_source(span["source_id"])) is not None
            and source["collection_id"] == self.source_map["collection_id"]
            for span in resolved_citations
        )
        machine = (
            {
                "citation_ids_valid": all_cited <= cited
                and all_cited <= {span["id"] for span in resolved_citations}
                and citation_sources_valid,
                "required_coverage_valid": (not requires_complete or bool(receipt.get("complete")))
                and (
                    not requires_complete
                    or required_source_ids <= set(receipt.get("delivered_source_ids", []))
                ),
                "citation_edges_complete": all(
                    bool(claim.get("evidence_ids")) for claim in candidate_claims
                )
                and all_cited <= cited,
            }
            if completed
            else {
                "citation_ids_valid": None,
                "required_coverage_valid": None,
                "citation_edges_complete": None,
            }
        )
        target_latency = target["metrics"].get("end_to_end_ms") if target else None
        first_run = (
            runs[0]
            if len(case["turns"]) == 2
            and attempt["first_run_id"]
            and attempt["first_run_id"] != attempt["target_run_id"]
            and runs
            else None
        )
        first_latency = (
            first_run["metrics"].get("end_to_end_ms")
            if first_run and first_run["status"] == "succeeded"
            else None
        )
        total_latency = (
            target_latency + (first_latency if len(case["turns"]) == 2 else 0)
            if target_latency is not None and (len(case["turns"]) == 1 or first_latency is not None)
            else None
        )
        state = attempt["state"] if attempt["state"] in EXECUTION_STATES else "pending"
        return {
            "case_id": attempt["case_id"],
            "family_id": case["family_id"],
            "variant": attempt["variant"],
            "generator": attempt["generator"],
            "question_class": case["question_class"],
            "anchor_document_class": case["anchor_document_class"],
            "required_document_classes": sorted(
                {anchor["document_class"]}
                | {
                    self.release.documents[doc_id]["document_class"]
                    for doc_id in gold.get("required_scope_doc_ids", [])
                }
            ),
            "anchor_format": anchor["format"],
            "difficulty": case["difficulty"],
            "gold_status": gold["gold_status"],
            "turn_count": len(case["turns"]),
            "gold_required_claims": gold_claims,
            "runtime_available_span_ids": sorted(loaded["spans"]),
            "generator_delivered_span_ids": target["metrics"].get("delivered_span_ids", [])
            if target
            else [],
            "diagnostics": _diagnostic_projection(target, completed),
            "execution": {
                "state": state,
                "total_latency_ms": total_latency if completed else None,
                "target_latency_ms": target_latency if completed else None,
                "trace_statuses": trace_statuses,
                "run_ids": run_ids,
            },
            "answer_status": answer.get("status") if answer else None,
            "candidate_claims_count": len(candidate_claims),
            "candidate_citation_edges": edges,
            "judge": attempt["judge_output"]
            if completed and attempt["judge_status"] == "assessed"
            else None,
            "judge_na_reason": attempt["judge_na_reason"]
            or ("needs_review" if attempt["state"] == "needs_review" else None),
            "machine": machine,
        }


def _diagnostic_projection(target, completed):
    """Preserve packet-delivery diagnostics only when the audit is complete.

    Args:
        target (dict | None): Target run whose metrics contain initial/final
            packets, retrieval, receipt, second batch and guard fields.
        completed (bool): Whether a target answer was successfully completed.

    Returns:
        dict | None: Diagnostic fields when all required metrics exist; None
            for incomplete execution or any absent audit field.
    """
    if not completed:
        return None
    metrics = target.get("metrics") or {}
    fields = {
        "candidate_chunk_ids": "initial_candidate_chunk_ids",
        "candidate_block_span_ids": "initial_candidate_block_span_ids",
        "initial_packet_span_ids": "initial_provider_span_ids",
        "initial_receipt_complete": "initial_receipt_complete",
        "initial_method": "initial_method",
        "coverage_policy": "coverage_policy",
        "second_batch_kind": "second_batch_kind",
        "final_packet_span_ids": "final_provider_span_ids",
        "final_receipt_complete": "final_receipt_complete",
        "guard_action": "guard_action",
    }
    if any(field not in metrics for field in fields.values()):
        return None
    return {name: metrics[field] for name, field in fields.items()}


def build_judge_payload(case, gold, answer, loaded, evidence, conversation_context=()):
    """Assemble candidate, gold and actual source excerpts for target-turn judging.

    Resolve and deduplicate authored gold references, include their extracted
    spans and all reviewed scope-document spans, then attach the candidate and
    actual preceding conversation. These assessment excerpts do not alter the
    generator's retrieval scope and are not a model invocation themselves.

    Args:
        case (dict): Turns and offline scope.doc_ids; the last turn is judged.
        gold (dict): gold_status, required_claims with text/claim_id and
            acceptable_evidence_sets, plus optional item/order/inference rules.
        answer (dict): Candidate status, answer, claims with evidence_ids,
            citations, limitations and coverage receipt.
        loaded (dict): Runtime spans and by_document from evidence.load.
        evidence (CorpusEvidence): Resolves reviewed references to runtime IDs.
        conversation_context (Sequence[dict]): Actual preceding user/assistant
            turn records; empty for single-turn cases.

    Returns:
        dict: judge-v4 live_candidate request containing question, candidate,
            gold, conversation_context, visible_source_excerpts and
            scope_source_excerpts. Unavailable references retain sentinel IDs
            and have no extracted spans.
    """
    excerpts = []
    seen = set()
    for claim in gold.get("required_claims", []):
        for option in claim.get("acceptable_evidence_sets", []):
            for reference in option:
                key = ref_key(reference)
                if key in seen:
                    continue
                runtime_ids = evidence.ids_for(reference, loaded)
                excerpts.append(
                    {
                        "authored_reference": reference,
                        "runtime_span_ids": runtime_ids,
                        "extracted_spans": [
                            {
                                "id": span_id,
                                "source_id": loaded["spans"][span_id]["source_id"],
                                "page": loaded["spans"][span_id]["page"],
                                "text": loaded["spans"][span_id]["text"],
                            }
                            for span_id in runtime_ids
                            if span_id in loaded["spans"]
                        ],
                    }
                )
                seen.add(key)
    reviewed_doc_ids = set(case["scope"]["doc_ids"]) | set(gold.get("required_scope_doc_ids", []))
    scope_excerpts = [
        {
            "doc_id": doc_id,
            "id": span["id"],
            "source_id": span["source_id"],
            "page": span["page"],
            "line_start": span["line_start"],
            "line_end": span["line_end"],
            "text": span["text"],
        }
        for doc_id in sorted(reviewed_doc_ids)
        for spans in [loaded["by_document"][doc_id]]
        for span in sorted(
            spans, key=lambda item: (item["page"] or 0, item["line_start"] or 0, item["id"])
        )
    ]
    return {
        "evaluation_mode": "live_candidate",
        "question": case["turns"][-1]["text"],
        "conversation_context": list(conversation_context),
        "candidate": {
            "status": answer.get("status"),
            "answer": answer.get("answer"),
            "claims": answer.get("claims", []),
            "limitations": answer.get("limitations", []),
            "coverage": answer.get("coverage", {}),
            "citations": answer.get("citations", []),
        },
        "gold": {
            "status": gold["gold_status"],
            "required_claims": [
                {
                    "claim_id": claim["claim_id"],
                    "text": claim["text"],
                    "acceptable_evidence_sets": [
                        sorted(
                            {
                                span_id
                                for reference in option
                                for span_id in evidence.ids_for(reference, loaded)
                            }
                        )
                        for option in claim.get("acceptable_evidence_sets", [])
                    ],
                }
                for claim in gold.get("required_claims", [])
            ],
            "expected_items": gold.get("expected_items", []),
            "temporal_order": gold.get("temporal_order", []),
            "prohibited_inferences": gold.get("prohibited_inferences", []),
            "required_scope_doc_ids": gold.get("required_scope_doc_ids", []),
        },
        "visible_source_excerpts": excerpts,
        "scope_source_excerpts": scope_excerpts,
        "rubric_version": "judge-v4",
    }


async def judge_payload(
    provider,
    telemetry,
    payload,
    request_id,
    price_snapshot_id,
    metadata=None,
    trace_id=None,
    operation_id=None,
):
    """Make one Sol judge call and retain raw output, usage and price estimate.

    Record a generation observation, invoke the judge final-only, then validate
    its structured output. Schema errors remain data in the returned result;
    provider errors propagate. Ordinary telemetry delivery failures are recorded
    without aborting the judge call. Cost is hypothetical API-equivalent
    usage pricing, not a claim about subscription charges or trace delivery.

    Args:
        provider (Provider): Inference adapter supporting judge generation.
        telemetry (Telemetry): Observation recorder.
        payload (dict): Target candidate/gold/source-excerpt request from
            build_judge_payload.
        request_id (str): Durable identity for this specific judge invocation.
        price_snapshot_id (str): Frozen token-pricing schedule identifier.
        metadata (dict | None): Optional campaign/attempt/run attribution fields.
        trace_id (str | None): Optional stable judge trace identity.
        operation_id (str | None): Optional stable observation identity.

    Returns:
        dict: Provider result with output, usage and billing metadata plus
            api_equivalent_estimate, validated_output and validation_error.
            validated_output is None on validation failure, with an error string;
            on success validation_error is None.
    """
    observation_context = telemetry.observation(
        "judge.generate",
        "generation",
        payload,
        model="gpt-6-sol",
        metadata={
            "request_id": request_id,
            "role": "judge",
            "operation_id": operation_id,
            **(metadata or {}),
        },
    )
    if trace_id is not None:
        observation_context.trace_id = trace_id
    async with observation_context as observation:
        result = await provider.generate(
            payload,
            role="judge",
            model="gpt-6-sol",
            request_id=request_id,
            final_only=True,
        )
        estimate = estimate_api_equivalent_cost(
            "gpt-6-sol",
            result.get("usage", {}),
            price_snapshot_id,
            billing_mode=result.get("billing_mode", "unknown"),
        )
        result["api_equivalent_estimate"] = estimate
        if observation:
            usage = result.get("usage", {})
            reported = {
                key: usage[name]
                for key, name in (("input", "input_tokens"), ("output", "output_tokens"))
                if isinstance(usage.get(name), int)
            }
            components = estimate.get("cost_components_usd")
            cost_details = (
                {
                    "input": float(
                        Decimal(components["uncached_input"]) + Decimal(components["cached_input"])
                    ),
                    "output": float(components["output_including_reasoning"]),
                }
                if components
                else None
            )
            observation.update(
                output=result["output"],
                usage_details=reported or None,
                cost_details=cost_details,
                metadata={
                    "usage": usage,
                    "billing_mode": result.get("billing_mode"),
                    "api_equivalent_estimate": estimate,
                    "pricing_status": "hypothetical_api_estimate",
                    "actual_billed_cost": "unknown",
                },
            )
    try:
        result["validated_output"] = validate_output("judge", result["output"])
        result["validation_error"] = None
    except Exception as exc:
        result["validated_output"] = None
        result["validation_error"] = f"{type(exc).__name__}:{str(exc)[:300]}"
    return result


def _start_attempt(row):
    """Record the first execution intent before any external run is submitted.

    Args:
        row (dict): Pending planned row with mutable attempts history and judge fields.

    Returns:
        dict: Same mutated row, running with a new attempt ID, empty turns and
            no conversation; clears judge status/reason. Persistence is the caller's job.

    Raises:
        ValueError: The planned row has already left pending state.
    """
    if row["state"] != "pending":
        raise ValueError("Only a pending planned row can start")
    record = {
        "attempt_id": str(uuid.uuid4()),
        "started_at": now(),
        "state": "running",
        "conversation_id": None,
        "turns": [],
        "error": None,
    }
    row["attempts"].append(record)
    row["state"] = "running"
    row["judge_status"] = None
    row["judge_na_reason"] = None
    return row


def _record_conversation(row, conversation_id):
    row["attempts"][-1]["conversation_id"] = conversation_id
    return row


def _turn_intent(row, index, request_id):
    """Record a durable per-turn request key before submitting inference.

    Args:
        row (dict): Running row with an active attempts[-1].turns list.
        index (int): Zero-based authored turn position.
        request_id (str): Application idempotency key for that turn.

    Returns:
        dict: Same row; append a requested turn only when index is the next
            list position. Existing positions leave the row unchanged.
    """
    record = row["attempts"][-1]
    if len(record["turns"]) == index:
        record["turns"].append(
            {"request_id": request_id, "run_id": None, "status": "requested", "requested_at": now()}
        )
    return row


def _turn_started(row, index, run_id):
    turn = row["attempts"][-1]["turns"][index]
    turn["run_id"] = run_id
    turn["status"] = "running"
    if index == 0 and row["metadata"]["turn_count"] == 2:
        row["first_run_id"] = run_id
    else:
        row["target_run_id"] = run_id
    return row


def _turn_finished(row, index, run):
    """Classify execution failures and distinguish context from target turns.

    Args:
        row (dict): Mutable row with metadata.turn_count and active turn history.
        index (int): Zero-based turn index.
        run (dict): Terminal application run with id and status.

    Returns:
        dict: Same mutated row with run identity/status and finish time. A failed
            context turn becomes first_turn_failed; any other failed turn becomes
            target_failed, both not_assessable. Successful final turns become
            completed with pending judge; earlier successful turns stay running.
    """
    turn = row["attempts"][-1]["turns"][index]
    turn["run_id"] = run["id"]
    turn["status"] = run["status"]
    turn["finished_at"] = now()
    if index == 0 and row["metadata"]["turn_count"] == 2:
        row["first_run_id"] = run["id"]
    else:
        row["target_run_id"] = run["id"]
    if run["status"] != "succeeded":
        row["state"] = (
            "first_turn_failed"
            if index == 0 and row["metadata"]["turn_count"] == 2
            else "target_failed"
        )
        row["judge_status"] = "not_assessable"
        row["judge_na_reason"] = row["state"]
        row["attempts"][-1]["state"] = row["state"]
    elif index == row["metadata"]["turn_count"] - 1:
        row["state"] = "completed"
        row["judge_status"] = "pending"
        row["attempts"][-1]["state"] = "completed"
    return row


def _needs_review(row, reason):
    """Retain an execution/preparation error as an unresolved planned result.

    Args:
        row (dict): Planned row with an active mutable attempt history.
        reason (str): Diagnostic error retained in judge_na_reason and the
            current attempt error field.

    Returns:
        dict: Same row with needs_review execution and not_assessable judge.
            It remains in the planned denominator and requires explicit retry
            for another answer attempt; persistence is the caller's job.
    """
    row["state"] = "needs_review"
    row["judge_status"] = "not_assessable"
    row["judge_na_reason"] = reason
    row["attempts"][-1]["state"] = "needs_review"
    row["attempts"][-1]["error"] = reason
    return row


def _judge_interrupted(row):
    """Mark an uncertain in-flight judge outcome for explicit retry.

    Args:
        row (dict): Completed/calling row with active attempt history, or any
            other state that should remain unchanged.

    Returns:
        dict: Same row; a completed/calling judge becomes not_assessable with
            judge_call_interrupted and an error record. No model call is repeated.
    """
    if row["state"] == "completed" and row["judge_status"] == "calling":
        row["judge_status"] = "not_assessable"
        row["judge_na_reason"] = "judge_call_interrupted"
        row["attempts"][-1]["judge_finished_at"] = now()
        row["attempts"][-1]["judge_error"] = {
            "type": "Interrupted",
            "message": "Judge request outcome is uncertain; explicit retry required",
        }
    return row


def _retry_attempt(row, reason, seen_set):
    """Start an explicit answer correction while preserving the primary result.

    Archive the latest judge summary, append a fresh answer attempt and clear
    current run/judge/projection answer fields. The frozen primary projection
    stays intact and continues to define the main comparison result.

    Args:
        row (dict): Terminal or needs_review row with finalized metrics.primary.
        reason (str): Caller-supplied correction rationale stored in history.
        seen_set (bool): Whether this is an acceptance rerun of already seen cases.

    Returns:
        dict: Same mutated row in running state, ready for a new conversation
            and inference; no HTTP/model call or persistence occurs here.

    Raises:
        ValueError: Execution is nonterminal or the primary result is not finalized.
    """
    if row["state"] not in COMPLETE_STATES | {"needs_review"}:
        raise ValueError("Only an observed terminal row can be retried explicitly")
    if row["metrics"].get("primary") is None:
        raise ValueError("Primary result must be finalized before answer retry")
    if row["attempts"]:
        row["attempts"][-1]["judge_result"] = {
            "status": row["judge_status"],
            "output": row["judge_output"],
            "na_reason": row["judge_na_reason"],
            "usage": row["attempts"][-1].get("judge_usage"),
            "request_id": row["attempts"][-1].get("judge_request_id"),
            "operation_id": row["attempts"][-1].get("judge_operation_id"),
            "trace_id": row["attempts"][-1].get("judge_trace_id"),
            "delivery_status": row["attempts"][-1].get("judge_delivery_status"),
        }
    row["attempts"].append(
        {
            "attempt_id": str(uuid.uuid4()),
            "started_at": now(),
            "state": "running",
            "conversation_id": None,
            "turns": [],
            "error": None,
            "retry_reason": reason,
            "seen_set_rerun": seen_set,
        }
    )
    row["state"] = "running"
    row["first_run_id"] = None
    row["target_run_id"] = None
    row["judge_output"] = None
    row["judge_status"] = None
    row["judge_na_reason"] = None
    projection = row["metrics"].get("projection")
    if projection:
        projection["execution"] = {
            "state": "pending",
            "total_latency_ms": None,
            "target_latency_ms": None,
            "trace_statuses": [],
            "run_ids": [],
        }
        projection["answer_status"] = None
        projection["candidate_claims_count"] = 0
        projection["candidate_citation_edges"] = []
        projection["generator_delivered_span_ids"] = []
        projection["diagnostics"] = None
        projection["judge"] = None
        projection["judge_na_reason"] = None
        projection["machine"] = {
            "citation_ids_valid": None,
            "required_coverage_valid": None,
            "citation_edges_complete": None,
        }
    return row


def _retry_judge(row, reason):
    """Archive judge evidence and queue an explicit rejudgment of the same answer.

    Args:
        row (dict): Completed row with assessed, not_assessable or calling judge;
            its first attempt must already have metrics.primary.
        reason (str): Retry rationale stored with prior payload/output/usage,
            error, delivery identity and cost attribution in judge_history.

    Returns:
        dict: Same row with pending judge, cleared current judge fields and
            retained answer/run IDs and primary result. No inference occurs here.

    Raises:
        ValueError: Candidate or observed judge state is not eligible for retry.
    """
    if row["state"] != "completed" or row["judge_status"] not in {
        "assessed",
        "not_assessable",
        "calling",
    }:
        raise ValueError("Judge retry needs a completed candidate and observed judge attempt")
    if len(row["attempts"]) == 1 and row["metrics"].get("primary") is None:
        raise ValueError("Primary judge result must be finalized before judge retry")
    record = row["attempts"][-1]
    record.setdefault("judge_history", []).append(
        {
            "status": row["judge_status"],
            "request_id": record.get("judge_request_id"),
            "judge_operation_id": record.get("judge_operation_id"),
            "judge_trace_id": record.get("judge_trace_id"),
            "judge_metadata": record.get("judge_metadata"),
            "judge_delivery_status": record.get("judge_delivery_status"),
            "judge_delivery_checked_at": record.get("judge_delivery_checked_at"),
            "started_at": record.get("judge_started_at"),
            "finished_at": record.get("judge_finished_at"),
            "payload": record.get("judge_payload"),
            "payload_sha256": record.get("judge_payload_sha256"),
            "output": record.get("judge_raw_output") or row["judge_output"],
            "na_reason": row["judge_na_reason"],
            "usage": record.get("judge_usage") or record.get("judge_raw_usage"),
            "raw_provider_usage": record.get("judge_raw_provider_usage"),
            "error": record.get("judge_error"),
            "billing_mode": record.get("judge_billing_mode"),
            "cost_state": record.get("judge_cost_state"),
            "api_equivalent_estimate": record.get("judge_api_equivalent_estimate"),
            "retry_reason": reason,
            "at": now(),
        }
    )
    record["judge_request_id"] = None
    record["judge_operation_id"] = None
    record["judge_trace_id"] = None
    record["judge_metadata"] = None
    record["judge_delivery_status"] = None
    record["judge_delivery_checked_at"] = None
    record["judge_started_at"] = None
    record["judge_finished_at"] = None
    record["judge_usage"] = None
    record["judge_raw_usage"] = None
    record["judge_raw_provider_usage"] = None
    record["judge_raw_output"] = None
    record["judge_error"] = None
    record["judge_billing_mode"] = None
    record["judge_cost_state"] = None
    record["judge_api_equivalent_estimate"] = None
    row["judge_output"] = None
    row["judge_status"] = "pending"
    row["judge_na_reason"] = None
    row["metrics"]["judge_usage"] = None
    row["metrics"]["judge_elapsed_ms"] = None
    row["metrics"]["judge_billing_mode"] = None
    row["metrics"]["judge_cost_state"] = None
    projection = row["metrics"].get("projection")
    if projection:
        projection["judge"] = None
        projection["judge_na_reason"] = "judge_retry_pending"
    return row


def _judge_delivery_due(record, instant):
    """Throttle delivery reconciliation without changing judge assessability.

    Args:
        record (dict): Judge history/current record with optional
            judge_delivery_checked_at as datetime or ISO timestamp.
        instant (datetime): Current time for the reconciliation decision.

    Returns:
        bool: True when never checked or at least 60 seconds since the last
            check; naive stored timestamps are interpreted as UTC.
    """
    checked_at = record.get("judge_delivery_checked_at")
    if not checked_at:
        return True
    if isinstance(checked_at, str):
        checked_at = datetime.fromisoformat(checked_at.replace("Z", "+00:00"))
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=UTC)
    return checked_at + JUDGE_RECONCILE_INTERVAL <= instant


class Runner:
    """Coordinate durable evaluation execution over an immutable campaign contract.

    Answer attempts, judge attempts and trace delivery have separate state.
    Normal resume recovers recorded requests and avoids repeating uncertain
    judge calls; corrections require explicit retry and retain primary results.
    """

    def __init__(self, ledger, release, campaign, service, settings):
        self.ledger = ledger
        self.release = release
        self.campaign = campaign
        self.service = service
        self.settings = settings
        self.evidence = CorpusEvidence(release, campaign["source_map"], service)
        self.provider = None
        self.telemetry = None
        self.outage_code = None

    def _verify_run_scope(self, run):
        """Require a run to use exactly the frozen shared collection snapshot.

        Args:
            run (dict): Application run with id, status and scope containing its
                run id plus collection identity, revision, counters and hash.

        Raises:
            ValueError: Live collection or saved scope differs. A queued/running
                mismatched run is cancelled through HTTP before the error propagates.
        """
        mapping = self.campaign["source_map"]
        try:
            expected = self.service.verify_collection_scope(mapping)
        except ValueError:
            if run.get("status") in {"queued", "running"}:
                self.service.request("POST", f"/api/runs/{run['id']}/cancel")
            raise
        scope = run.get("scope") or {}
        if isinstance(scope, dict) and scope == {"id": run["id"], **expected}:
            return
        if run.get("status") in {"queued", "running"}:
            self.service.request("POST", f"/api/runs/{run['id']}/cancel")
        raise ValueError("Answer run does not use the frozen release collection")

    def flush_telemetry(self):
        client = getattr(self.telemetry, "client", None)
        if client is not None:
            try:
                client.flush()
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Evaluation telemetry flush failed: %s", type(exc).__name__
                )

    def _save_projection(self, row_id):
        """Persist current metrics and freeze the first finalized result once.

        Args:
            row_id (str): Planned-row UUID. Reads current evidence/runs, calculates
                all-attempt cost and updates projection and cost_estimate. The
                first terminal attempt gains a copied, hashed metrics.primary.

        Raises:
            ValueError: Evidence projection cannot be built. Even then persists
                projection=None, projection_unavailable and current cost attribution
                before raising; an existing primary result remains protected.
        """
        row = self.ledger.row(row_id)
        try:
            projection = self.evidence.metric_row(row, self.ledger.store)
        except (ValueError, KeyError) as exc:
            reason = f"current_projection_unavailable:{type(exc).__name__}:{str(exc)[:200]}"
            costs = cost_for_attempt(
                row, self.ledger.store, self.campaign["config"]["price_snapshot_id"]
            )
            self.ledger.mutate(
                row_id,
                lambda value: {
                    **value,
                    "metrics": {
                        **value["metrics"],
                        "projection": None,
                        "projection_unavailable": reason,
                        "cost_estimate": costs,
                    },
                },
            )
            raise ValueError(f"Projection unavailable for {row_id}: {reason}") from exc
        costs = cost_for_attempt(
            row, self.ledger.store, self.campaign["config"]["price_snapshot_id"]
        )

        def update(value):
            value["metrics"] = {
                **value["metrics"],
                "projection": projection,
                "projection_unavailable": None,
                "cost_estimate": costs,
            }
            if (
                len(value["attempts"]) == 1
                and _primary_terminal(value)
                and value["metrics"].get("primary") is None
            ):
                first_projection = copy.deepcopy(projection)
                value["metrics"]["primary"] = {
                    "attempt_id": value["attempts"][0]["attempt_id"],
                    "judge_request_id": value["attempts"][0].get("judge_request_id"),
                    "projection": first_projection,
                    "projection_sha256": digest(first_projection),
                    "finalized_at": now(),
                }
            return value

        self.ledger.mutate(row_id, update)

    def verify(self):
        """Validate frozen artifacts and live prerequisites without calling a model.

        Check corpus/release identities, execution matrix, code/rubric/config,
        embedding and pricing snapshots, healthy non-paid provider and every
        reviewed gold locator. May populate the evidence cache via HTTP.

        Raises:
            ValueError: Any frozen dependency, runtime provider or evidence contract
                no longer matches the campaign.
        """
        self.release.verify_hashes(self.campaign)
        validate_campaign_corpus(
            self.campaign,
            CorpusLedger(self.ledger.store).corpus(self.campaign["corpus_row_id"]),
            self.release,
        )
        self.service.verify_collection_scope(self.campaign["source_map"])
        case_worker_count(self.campaign["config"])
        expected_order = execution_order(selected_cases(self.release, self.campaign["config"]))
        if self.campaign["config"].get("execution_order") != expected_order:
            raise ValueError("Frozen execution order differs from release family rotation")
        actual_cells = {
            (row["case_id"], row["variant"], row["generator"])
            for row in self.ledger.rows(self.campaign["id"])
        }
        if len(actual_cells) != len(expected_order) or actual_cells != {
            tuple(cell) for cell in expected_order
        }:
            raise ValueError("Frozen execution order differs from planned attempt rows")
        if (
            self.campaign["code_sha256"] != code_hash()
            or self.campaign["rubric_sha256"] != instruction_hash("judge")
            or self.campaign["config_sha256"] != digest(self.campaign["config"])
        ):
            raise ValueError("Frozen code, rubric, or configuration changed")
        if (
            self.campaign["config"].get("embedding_model") != self.settings.embedding_model
            or self.campaign["config"].get("embedding_revision") != self.settings.embedding_revision
        ):
            raise ValueError("Frozen embedding model changed")
        if self.campaign["config"].get("price_snapshot_sha256") != file_hash(
            Path(__file__).with_name("pricing_snapshot.json")
        ):
            raise ValueError("Frozen API-equivalent price snapshot changed")
        health = self.service.request("GET", "/api/health")
        if (
            health["status"] != "ok"
            or health["provider"] == "api"
            or self.settings.provider == "api"
        ):
            raise ValueError("Evaluation requires a healthy non-paid provider")
        expected = self.campaign["config"].get("provider")
        if expected and expected != health["provider"]:
            raise ValueError("Runtime provider differs from frozen configuration")
        self.evidence.validate_all(selected_cases(self.release, self.campaign["config"]))

    def execute_pending(self, limit=None):
        """Run pending planned cells under the campaign coordinator lock.

        Args:
            limit (int | None): Optional maximum pending cells selected in frozen
                order; None selects all. Each dispatched cell may generate one or
                two dialogue turns and a separate judge call.

        Returns:
            str: Persisted complete/incomplete campaign status after dispatch;
                detected provider outage stops later batches and sets outage_code.

        Raises:
            RuntimeError: Another coordinator owns the campaign.
        """
        with self.ledger.coordinator(self.campaign["id"]):
            self.outage_code = None
            return self._execute_pending(limit)

    @contextmanager
    def _worker(self):
        store = Store(self.settings)
        service = None
        worker = None
        try:
            service = Service(self.settings)
            worker = Runner(Ledger(store), self.release, self.campaign, service, self.settings)
            yield worker
        finally:
            if worker is not None:
                worker.flush_telemetry()
            try:
                if service is not None:
                    service.close()
            finally:
                store.engine.dispose()

    def _worker_call(self, row, judge_only):
        with self._worker() as worker:
            if judge_only:
                worker._judge(row)
            else:
                worker._continue(row)

    def _provider_outage_code(self, row_id):
        """Identify provider outages that should stop later dispatch batches.

        Args:
            row_id (str): Observed planned row with an active attempt.

        Returns:
            str | None: Recognized transport/authentication outage from the latest
                failed run audit or completed/unassessable judge error; None for
                other failures or no such recorded outage.
        """
        row = self.ledger.row(row_id)
        attempt = row["attempts"][-1]
        if row["state"] in {"first_turn_failed", "target_failed"}:
            run_id = attempt["turns"][-1].get("run_id") if attempt["turns"] else None
            if run_id:
                for event in self.ledger.store.audit(run_id):
                    if event["stage"] == "run.exception":
                        code = event["payload"].get("code")
                        if code in PROVIDER_OUTAGE_CODES:
                            return code
        if row["state"] == "completed" and row["judge_status"] == "not_assessable":
            code = (attempt.get("judge_error") or {}).get("code")
            if code in PROVIDER_OUTAGE_CODES:
                return code
        return None

    def _dispatch(self, rows, judge_only=False, start_pending=False):
        """Dispatch frozen-order cells with bounded parallelism and outage stopping.

        Args:
            rows (list[dict]): Selected planned rows in dispatch order.
            judge_only (bool): Judge existing target answers instead of continuing turns.
            start_pending (bool): Persist first-attempt intent before dispatch.

        Returns:
            bool: True if a recorded provider outage stopped future batches and
                set outage_code; False after all selected rows finish. Parallel
                batches finish their already submitted work before outage inspection.

        Worker errors propagate after all futures in the current parallel batch
        have been collected; previous row transitions remain durable.
        """
        count = case_worker_count(self.campaign["config"])
        if count == 1:
            for row in rows:
                if start_pending:
                    row = self.ledger.mutate(row["id"], _start_attempt)
                if judge_only:
                    self._judge(row)
                else:
                    self._continue(row)
                code = self._provider_outage_code(row["id"])
                if code:
                    self.outage_code = code
                    return True
            return False
        with ThreadPoolExecutor(max_workers=count) as executor:
            for offset in range(0, len(rows), count):
                batch = rows[offset : offset + count]
                if start_pending:
                    batch = [self.ledger.mutate(row["id"], _start_attempt) for row in batch]
                futures = [executor.submit(self._worker_call, row, judge_only) for row in batch]
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    raise errors[0]
                for row in batch:
                    code = self._provider_outage_code(row["id"])
                    if code:
                        self.outage_code = code
                        return True
        return False

    def _execute_pending(self, limit=None):
        self.ledger.mark_running(self.campaign["id"])
        rows = self.ledger.rows(self.campaign["id"])
        pending = [row for row in rows if row["state"] == "pending"]
        if limit is not None:
            pending = pending[:limit]
        self._dispatch(pending, start_pending=True)
        return self.ledger.campaign_status(self.campaign["id"])

    def resume(self, limit=None):
        """Recover durable work and reconcile telemetry without implicit judge retry.

        Continue running answer attempts by recorded request/run IDs, judge
        completed/pending answers, then start remaining pending cells. Interrupted
        calling judges become not_assessable and require explicit retry. Finally
        check delivery for up to 25 due rows without another LLM call.

        Args:
            limit (int | None): Bounds resumed running plus newly started pending
                answer cells; does not bound completed/pending judge dispatch.

        Returns:
            str: Persisted complete/incomplete execution/judge status.

        Raises:
            RuntimeError: Another coordinator owns the campaign.
        """
        with self.ledger.coordinator(self.campaign["id"]):
            self.outage_code = None
            status = self._resume(limit)
            self._reconcile_judges(25)
            return status

    def reconcile_judges(self, limit=25):
        """Recover missing judge observations from the ledger without inference.

        Args:
            limit (int): Maximum due planned rows checked, from 1 through 100;
                each row can contain multiple answer/judge-history observations.

        Returns:
            int: Number of selected rows checked, not number of model calls or
                newly delivered observations. Persists delivery status/check times.

        Raises:
            ValueError: Reconciliation row limit is outside the permitted range.
            RuntimeError: Another coordinator owns the campaign.
        """
        with self.ledger.coordinator(self.campaign["id"]):
            return self._reconcile_judges(limit)

    def _reconcile_judges(self, limit):
        if not 1 <= limit <= 100:
            raise ValueError("Judge reconciliation limit must be 1–100")
        pending = []
        instant = datetime.now(UTC)
        for row in self.ledger.rows(self.campaign["id"]):
            checks = [
                record.get("judge_delivery_checked_at") or ""
                for attempt in row["attempts"]
                for record in [*attempt.get("judge_history", []), attempt]
                if record.get("judge_operation_id")
                and record.get("judge_delivery_status")
                not in {"original_materialized", "recovered_from_ledger", "unrecoverable"}
                and _judge_delivery_due(record, instant)
            ]
            if checks:
                pending.append((min(checks), row["id"]))
        pending.sort(key=lambda item: item[0])
        for _, row_id in pending[:limit]:
            self._refresh_judge_delivery(row_id, due_only=True)
        return min(limit, len(pending))

    def _resume(self, limit=None):
        self.ledger.mark_running(self.campaign["id"])
        rows = self.ledger.rows(self.campaign["id"])
        for row in rows:
            if row["state"] == "completed" and row["judge_status"] == "calling":
                self.ledger.mutate(row["id"], _judge_interrupted)
                self._save_projection(row["id"])
        running = [row for row in rows if row["state"] == "running"]
        if limit is not None:
            running = running[:limit]
        if self._dispatch(running):
            return self.ledger.campaign_status(self.campaign["id"])
        completed = [
            row
            for row in self.ledger.rows(self.campaign["id"])
            if row["state"] == "completed" and row["judge_status"] == "pending"
        ]
        if self._dispatch(completed, judge_only=True):
            return self.ledger.campaign_status(self.campaign["id"])
        remaining = None if limit is None else max(0, limit - len(running))
        if remaining:
            self._execute_pending(remaining)
        elif limit is None:
            self._execute_pending()
        return self.ledger.campaign_status(self.campaign["id"])

    def retry(self, row_id, reason, judge_only=False):
        """Perform an explicit answer or judge correction under exclusive coordination.

        Args:
            row_id (str): Planned-row UUID belonging to this campaign.
            reason (str): Rationale retained with correction history.
            judge_only (bool): True invokes a new judge for the existing answer;
                False starts a fresh conversation/answer attempt and then judges it.

        Returns:
            str: Persisted complete/incomplete status after correction; primary
                results remain frozen, current metrics record the latest correction
                and outage_code reflects any recognized provider outage.

        Raises:
            ValueError: Row ownership or retry eligibility fails.
            RuntimeError: Another coordinator owns the campaign.
        """
        with self.ledger.coordinator(self.campaign["id"]):
            self.outage_code = None
            return self._retry(row_id, reason, judge_only)

    def _retry(self, row_id, reason, judge_only=False):
        row = self.ledger.row(row_id)
        if row is None or row["campaign_id"] != self.campaign["id"]:
            raise ValueError("Attempt row does not belong to campaign")
        self.ledger.mark_running(self.campaign["id"])
        if judge_only:
            row = self.ledger.mutate(row_id, lambda value: _retry_judge(value, reason))
            self._judge(row)
        else:
            row = self.ledger.mutate(
                row_id,
                lambda value: _retry_attempt(value, reason, self.campaign["split"] == "acceptance"),
            )
            self._continue(row)
        self.outage_code = self._provider_outage_code(row_id)
        return self.ledger.campaign_status(self.campaign["id"])

    def _continue(self, row):
        """Recover or execute the active answer attempt, then judge its target turn.

        Recover a conversation by durable title and a run by request ID before
        creating either, skip successful prior turns, verify frozen scope and wait
        for each terminal run. First-turn failure prevents the target submission.
        Execution/preparation errors are recorded as needs_review or an
        unassessable judge, with projections saved afterward.

        Args:
            row (dict): Running planned row with active attempts[-1], cell model/
                variant and authored case identity. Durable updates occur through
                the ledger; HTTP submissions can invoke real generator inference.
        """
        case = self.release.cases[row["case_id"]]
        attempt = row["attempts"][-1]
        try:
            conversation_id = attempt.get("conversation_id")
            if conversation_id is None:
                collection_id = self.campaign["source_map"]["collection_id"]
                title = f"Evaluation {self.campaign['id']} {attempt['attempt_id']}"
                existing = self.service.request(
                    "GET", "/api/conversations", params={"collection_id": collection_id}
                )["items"]
                matching = [item for item in existing if item["title"] == title]
                if len(matching) > 1:
                    raise ValueError("Multiple conversations match durable attempt")
                conversation = (
                    matching[0]
                    if matching
                    else self.service.request(
                        "POST",
                        "/api/conversations",
                        json={"collection_id": collection_id, "title": title},
                    )
                )
                conversation_id = conversation["id"]
                row = self.ledger.mutate(
                    row["id"], lambda value: _record_conversation(value, conversation_id)
                )
            for index, turn in enumerate(case["turns"]):
                record = row["attempts"][-1]
                if index < len(record["turns"]) and record["turns"][index]["status"] == "succeeded":
                    continue
                if index >= len(record["turns"]):
                    request_id = f"eval-{record['attempt_id'].replace('-', '')}-t{index}"
                    row = self.ledger.mutate(
                        row["id"], lambda value: _turn_intent(value, index, request_id)
                    )
                turn_record = row["attempts"][-1]["turns"][index]
                run_id = turn_record.get("run_id")
                if run_id is None:
                    self.service.verify_collection_scope(self.campaign["source_map"])
                    conversation = self.service.request(
                        "GET", f"/api/conversations/{conversation_id}"
                    )
                    found = next(
                        (
                            run
                            for run in conversation["runs"]
                            if run["request_id"] == turn_record["request_id"]
                        ),
                        None,
                    )
                    run = found or self.service.run(
                        conversation_id,
                        turn_record["request_id"],
                        turn["text"],
                        row["generator"],
                        row["variant"],
                    )
                    run_id = run["id"]
                    row = self.ledger.mutate(
                        row["id"], lambda value: _turn_started(value, index, run_id)
                    )
                run = self.service.request("GET", f"/api/runs/{run_id}")
                self._verify_run_scope(run)
                run = self.service.wait_run(run_id)
                row = self.ledger.mutate(row["id"], lambda value: _turn_finished(value, index, run))
                if run["status"] != "succeeded":
                    self._save_projection(row["id"])
                    return
            self._judge(row)
        except Exception as exc:
            reason = f"{type(exc).__name__}:{str(exc)[:300]}"
            current = self.ledger.row(row["id"])
            if current["state"] == "completed":
                error_type = type(exc).__name__
                error_message = str(exc)[:300]

                def mark_judge_unassessable(value):
                    value["judge_status"] = "not_assessable"
                    value["judge_na_reason"] = f"judge_preparation_error:{error_type}"
                    value["attempts"][-1]["judge_error"] = {
                        "type": error_type,
                        "message": error_message,
                    }
                    return value

                self.ledger.mutate(row["id"], mark_judge_unassessable)
            else:
                self.ledger.mutate(row["id"], lambda value: _needs_review(value, reason))
            self._save_projection(row["id"])

    def _judge(self, row):
        """Judge one completed target answer with durable payload and call attribution.

        Ignore rows outside completed/pending. Prepare reviewed excerpts and actual
        first-turn context, persist calling intent with payload digest/request/
        operation/trace IDs, then invoke Sol once. Preserve raw output/usage before
        schema and full claim/citation coverage checks. Preparation/provider errors
        become not_assessable; a missing target becomes needs_review. Save primary/
        current metrics, attempt score delivery and reconcile the judge observation.

        Args:
            row (dict): Planned row identified by id, reloaded before judging.
                Mutates ledger judge/history/metrics fields and may emit telemetry;
                partial usage from ProviderError remains attributed even on failure.
        """
        row = self.ledger.row(row["id"])
        if row["judge_status"] != "pending" or row["state"] != "completed":
            return
        case = self.release.cases[row["case_id"]]
        gold = self.release.golds[row["case_id"]]
        target = self.ledger.store.get_run(row["target_run_id"])
        if target is None or target["status"] != "succeeded":
            self.ledger.mutate(row["id"], lambda value: _needs_review(value, "target_run_missing"))
            self._save_projection(row["id"])
            return
        try:
            self.service.verify_collection_scope(self.campaign["source_map"])
            loaded = self.evidence.load(case)
        except Exception as exc:
            reason = f"judge_evidence_unavailable:{type(exc).__name__}"

            def unavailable(value):
                value["judge_status"] = "not_assessable"
                value["judge_na_reason"] = reason
                return value

            self.ledger.mutate(row["id"], unavailable)
            self._save_projection(row["id"])
            return
        try:
            conversation_context = []
            if len(case["turns"]) == 2:
                first = self.ledger.store.get_run(row["first_run_id"])
                if first is None or first["status"] != "succeeded" or first["answer"] is None:
                    raise ValueError("Actual first turn is unavailable for judge")
                first_answer = first["answer"]
                conversation_context = [
                    {"role": "user", "text": first["question"]},
                    {
                        "role": "assistant",
                        "status": first_answer.get("status"),
                        "answer": first_answer.get("answer"),
                        "claims": first_answer.get("claims", []),
                        "limitations": first_answer.get("limitations", []),
                        "coverage": first_answer.get("coverage", {}),
                        "citations": first_answer.get("citations", []),
                    },
                ]
            payload = build_judge_payload(
                case,
                gold,
                target["answer"],
                loaded,
                self.evidence,
                conversation_context,
            )
        except Exception as exc:
            reason = f"judge_payload_unavailable:{type(exc).__name__}"

            def unavailable(value):
                value["judge_status"] = "not_assessable"
                value["judge_na_reason"] = reason
                return value

            self.ledger.mutate(row["id"], unavailable)
            self._save_projection(row["id"])
            return
        answer_attempt_id = row["attempts"][-1]["attempt_id"]
        judge_ordinal = len(row["attempts"][-1].get("judge_history", []))
        request_id = f"judge-{answer_attempt_id.replace('-', '')}-r{judge_ordinal}"
        operation_id = f"{row['id']}:judge:{answer_attempt_id}:{judge_ordinal}"
        judge_trace_id = Langfuse.create_trace_id(seed=f"pfl-judge:{operation_id}")
        judge_metadata = {
            "campaign_id": self.campaign["id"],
            "case_id": row["case_id"],
            "answer_attempt_id": answer_attempt_id,
            "judge_ordinal": judge_ordinal,
            "target_run_id": row["target_run_id"],
            "target_trace_id": target.get("trace_id"),
            "app_url": self.settings.app_origin.rstrip("/") + "/runs/" + row["target_run_id"],
        }

        def mark_calling(value):
            value["judge_status"] = "calling"
            value["attempts"][-1]["judge_request_id"] = request_id
            value["attempts"][-1]["judge_started_at"] = now()
            value["attempts"][-1]["judge_payload"] = payload
            value["attempts"][-1]["judge_payload_sha256"] = digest(payload)
            value["attempts"][-1]["judge_operation_id"] = operation_id
            value["attempts"][-1]["judge_trace_id"] = judge_trace_id
            value["attempts"][-1]["judge_delivery_status"] = "pending"
            value["attempts"][-1]["judge_metadata"] = judge_metadata
            return value

        self.ledger.mutate(row["id"], mark_calling)
        store = self.ledger.store
        try:
            if self.settings.provider == "api":
                raise ValueError("Paid API provider is disabled for evaluation")
            if self.provider is None:
                self.provider = get_provider(self.settings)
            if self.telemetry is None:
                self.telemetry = Telemetry(self.settings, store)
            result = asyncio.run(
                judge_payload(
                    self.provider,
                    self.telemetry,
                    payload,
                    request_id,
                    self.campaign["config"]["price_snapshot_id"],
                    metadata=judge_metadata,
                    trace_id=judge_trace_id,
                    operation_id=operation_id,
                )
            )

            def preserve_raw(value):
                record = value["attempts"][-1]
                record["judge_raw_output"] = result.get("output")
                record["judge_raw_usage"] = result.get("usage", {})
                record["judge_raw_provider_usage"] = result.get("raw_provider_usage")
                record["judge_billing_mode"] = result.get("billing_mode")
                record["judge_api_equivalent_estimate"] = result["api_equivalent_estimate"]
                return value

            self.ledger.mutate(row["id"], preserve_raw)
            if result["validation_error"] is not None:
                raise ValueError(result["validation_error"])
            output = result["validated_output"]
            answer_claims = target["answer"].get("claims", [])
            coverage = judge_coverage(
                output,
                [claim["claim_id"] for claim in gold.get("required_claims", [])],
                len(answer_claims),
                [
                    (index, evidence_id)
                    for index, claim in enumerate(answer_claims)
                    for evidence_id in claim.get("evidence_ids", [])
                ],
            )
            status = (
                "assessed"
                if coverage and output["verdict"] != "not_assessable"
                else "not_assessable"
            )
            reason = (
                None
                if status == "assessed"
                else "invalid_judge_coverage"
                if not coverage
                else "judge_not_assessable"
            )

            def finish(value):
                value["judge_output"] = output
                value["judge_status"] = status
                value["judge_na_reason"] = reason
                value["metrics"] = {
                    **value["metrics"],
                    "judge_usage": result.get("usage", {}),
                    "judge_elapsed_ms": result.get("elapsed_ms"),
                    "judge_billing_mode": result.get("billing_mode"),
                    "judge_cost_state": result.get("cost_state"),
                }
                value["attempts"][-1]["judge_finished_at"] = now()
                value["attempts"][-1]["judge_usage"] = result.get("usage", {})
                value["attempts"][-1]["judge_billing_mode"] = result.get("billing_mode")
                value["attempts"][-1]["judge_cost_state"] = result.get("cost_state")
                value["attempts"][-1]["judge_api_equivalent_estimate"] = result[
                    "api_equivalent_estimate"
                ]
                return value

            row = self.ledger.mutate(row["id"], finish)
            metric = self.evidence.metric_row(row, store)
            self._save_projection(row["id"])
            scored = aggregate([metric])["summary"]["confirmed_grounded_success"]["numerator"]
            try:
                self.telemetry.score(
                    row["target_run_id"],
                    "eval_confirmed_grounded_success",
                    bool(scored),
                    data_type="BOOLEAN",
                    metadata={"campaign_id": self.campaign["id"], "case_id": row["case_id"]},
                )
                self.telemetry.score(
                    row["target_run_id"],
                    "eval_judge_verdict",
                    output["verdict"],
                    data_type="CATEGORICAL",
                    metadata={"campaign_id": self.campaign["id"], "case_id": row["case_id"]},
                )
            except Exception as exc:
                store.save_audit(
                    row["target_run_id"],
                    "judge.score_delivery_failed",
                    {"type": type(exc).__name__},
                )
        except Exception as exc:
            reason = f"judge_error:{type(exc).__name__}"
            error_detail = {"type": type(exc).__name__, "message": str(exc)[:300]}
            failure_usage = None
            failure_raw_provider_usage = None
            failure_estimate = None
            if isinstance(exc, ProviderError):
                error_detail["code"] = exc.code
                error_detail["response_status"] = exc.response_status
                error_detail["response_reason"] = exc.response_reason
                failure_usage = exc.usage
                failure_raw_provider_usage = exc.raw_provider_usage
                failure_estimate = estimate_api_equivalent_cost(
                    "gpt-6-sol",
                    failure_usage,
                    self.campaign["config"]["price_snapshot_id"],
                    billing_mode="unknown",
                )

            def fail(value):
                value["judge_status"] = "not_assessable"
                value["judge_na_reason"] = reason
                value["attempts"][-1]["judge_finished_at"] = now()
                value["attempts"][-1]["judge_error"] = error_detail
                if failure_usage is not None:
                    value["attempts"][-1]["judge_raw_usage"] = failure_usage
                    value["attempts"][-1]["judge_raw_provider_usage"] = failure_raw_provider_usage
                    value["attempts"][-1]["judge_billing_mode"] = "unknown"
                    value["attempts"][-1]["judge_cost_state"] = "unknown"
                    value["attempts"][-1]["judge_api_equivalent_estimate"] = failure_estimate
                    value["metrics"]["judge_usage"] = failure_usage
                    value["metrics"]["judge_billing_mode"] = "unknown"
                    value["metrics"]["judge_cost_state"] = "unknown"
                return value

            self.ledger.mutate(row["id"], fail)
            self._save_projection(row["id"])
        self._refresh_judge_delivery(row["id"])

    def _refresh_judge_delivery(self, row_id, due_only=False):
        """Reconcile current and historic judge observations from stored results.

        Args:
            row_id (str): Planned-row UUID with current attempts and judge_history.
            due_only (bool): Respect the 60-second reconciliation interval when True.

        Records without operation/trace IDs or already in a final delivery state
        are skipped.
        Finished requests with no stored output become unrecoverable; others are
        checked/replayed through telemetry with preserved payload/usage. Delivery
        status and check times commit to the ledger. No LLM call or judge verdict
        correction is performed, and rows with no updates remain unchanged.
        """
        row = self.ledger.row(row_id)
        if self.telemetry is None:
            self.telemetry = Telemetry(self.settings, self.ledger.store)
        updates = {}
        target = self.ledger.store.get_run(row["target_run_id"]) if row["target_run_id"] else None
        instant = datetime.now(UTC)
        for attempt in row["attempts"]:
            records = [*attempt.get("judge_history", []), attempt]
            for ordinal, record in enumerate(records):
                operation_id = record.get("judge_operation_id")
                trace_id = record.get("judge_trace_id")
                if not operation_id or not trace_id:
                    continue
                if record.get("judge_delivery_status") in {
                    "original_materialized",
                    "recovered_from_ledger",
                    "unrecoverable",
                }:
                    continue
                if due_only and not _judge_delivery_due(record, instant):
                    continue
                output = record.get("output")
                if output is None:
                    output = record.get("judge_raw_output")
                finished_at = record.get("finished_at") or record.get("judge_finished_at")
                if output is None and finished_at:
                    updates[operation_id] = "unrecoverable"
                    continue
                metadata = record.get("judge_metadata") or {
                    "campaign_id": self.campaign["id"],
                    "case_id": row["case_id"],
                    "answer_attempt_id": attempt["attempt_id"],
                    "judge_ordinal": ordinal,
                    "target_run_id": row["target_run_id"],
                    "target_trace_id": (target or {}).get("trace_id"),
                    "app_url": self.settings.app_origin.rstrip("/")
                    + "/runs/"
                    + row["target_run_id"],
                }
                updates[operation_id] = self.telemetry.reconcile_judge_observation(
                    trace_id=trace_id,
                    operation_id=operation_id,
                    payload=record.get("payload") or record.get("judge_payload"),
                    output=output,
                    usage=record.get("usage")
                    or record.get("judge_usage")
                    or record.get("judge_raw_usage")
                    or {},
                    estimate=record.get("api_equivalent_estimate")
                    or record.get("judge_api_equivalent_estimate"),
                    billing_mode=record.get("billing_mode")
                    or record.get("judge_billing_mode")
                    or "unknown",
                    metadata=metadata,
                    finished_at=finished_at,
                )
        if not updates:
            return

        def save(value):
            for attempt in value["attempts"]:
                for record in [*attempt.get("judge_history", []), attempt]:
                    operation_id = record.get("judge_operation_id")
                    if operation_id in updates:
                        record["judge_delivery_status"] = updates[operation_id]
                        record["judge_delivery_checked_at"] = now()
            return value

        self.ledger.mutate(row_id, save)


def report_data(ledger, campaign, release):
    """Build a ledger-based comparison with stable primary denominators.

    Validate release/corpus/matrix and frozen projections, read current trace
    facts, aggregate primary results and present latest corrections separately.
    Completion, assessability and strict success retain all planned cells;
    latency uses completed cases and sums successful dialogue turns. Costs
    include all answer and judge attempts, including retries and failures.

    Args:
        ledger (Ledger): Reads planned rows and persisted run facts.
        campaign (dict): Frozen identities, config, planned_count and source_map.
        release (Release): Matching authored partition, gold and extraction review.

    Returns:
        dict: campaign, cells, comparisons, latest_correction, state_counts,
            api_equivalent_cost and generated_at. Cells expose primary summary
            metrics and separate latest_correction/all-attempt cost. Refreshes
            trace statuses in loaded primary projection objects, without
            persisting ledger changes or calling inference.

    Raises:
        ValueError: Frozen contracts/projections or generation cost attribution
            fail validation; no stale projection is silently recomputed.
    """
    release.verify_hashes(campaign)
    validate_campaign_corpus(
        campaign, CorpusLedger(ledger.store).corpus(campaign["corpus_row_id"]), release
    )
    attempts = ledger.rows(campaign["id"])
    rows = validate_campaign_rows(campaign, attempts, release)
    run_ids = []
    for attempt, projection in zip(attempts, rows, strict=True):
        run_ids.extend(projection["execution"]["run_ids"])
        run_ids.extend(
            turn["run_id"]
            for item in attempt["attempts"]
            for turn in item.get("turns", [])
            if turn.get("run_id")
        )
    run_facts = report_run_facts(ledger.store, [run_id for run_id in run_ids if run_id])
    for projection in rows:
        execution = projection["execution"]
        execution["trace_statuses"] = [
            (run_facts.get(run_id) or {}).get("trace_status") or "missing"
            for run_id in execution["run_ids"]
        ]
    if len(rows) != campaign["planned_count"]:
        raise ValueError("Planned ledger denominator changed")
    result = campaign_report(rows)
    latest_rows = [attempt["metrics"]["projection"] for attempt in attempts]
    costs = [
        cost_for_attempt(row, ledger.store, campaign["config"]["price_snapshot_id"], run_facts)
        for row in attempts
    ]
    for cell, value in result["cells"].items():
        variant, generator = cell.split("/", 1)
        corrected = [
            row for row in attempts if row["variant"] == variant and row["generator"] == generator
        ]
        value["latest_correction"] = {
            "attempted_rows": sum(
                len(row["attempts"]) > 1
                or any(item.get("judge_history") for item in row["attempts"])
                for row in corrected
            ),
            "completed": sum(row["state"] == "completed" for row in corrected),
            "state_counts": dict(Counter(row["state"] for row in corrected)),
            "confirmed_grounded_success": aggregate(
                [row["metrics"]["projection"] for row in corrected]
            )["summary"]["confirmed_grounded_success"],
        }
        value["api_equivalent_cost"] = combine_costs(
            [
                cost
                for row, cost in zip(attempts, costs, strict=True)
                if row["variant"] == variant and row["generator"] == generator
            ]
        )
    return {
        "campaign": {
            "id": campaign["id"],
            "corpus_row_id": campaign["corpus_row_id"],
            "corpus_id": release.manifest["corpus_id"],
            "release_id": campaign["release_id"],
            "partition": campaign["split"],
            "status": campaign["status"],
            "planned_count": campaign["planned_count"],
            "selected_variant": campaign["selected_variant"],
            "selected_generator": campaign["selected_generator"],
            "manifest_sha256": campaign["manifest_sha256"],
            "questions_sha256": campaign["questions_sha256"],
            "gold_sha256": campaign["gold_sha256"],
            "extraction_map_sha256": campaign["extraction_map_sha256"],
            "config_sha256": campaign["config_sha256"],
            "rubric_sha256": campaign["rubric_sha256"],
            "code_sha256": campaign["code_sha256"],
        },
        "cells": result["cells"],
        "comparisons": result["comparisons"],
        "latest_correction": {
            "attempted_rows": sum(
                len(row["attempts"]) > 1
                or any(item.get("judge_history") for item in row["attempts"])
                for row in attempts
            ),
            "completed": sum(row["state"] == "completed" for row in attempts),
            "state_counts": dict(Counter(row["state"] for row in attempts)),
            "confirmed_grounded_success": aggregate(latest_rows)["summary"][
                "confirmed_grounded_success"
            ],
        },
        "api_equivalent_cost": combine_costs(costs),
        "state_counts": dict(Counter(row["execution"]["state"] for row in rows)),
        "generated_at": now(),
    }


def write_report(value, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"{value['campaign']['id']}.json"
    csv_path = destination / f"{value['campaign']['id']}.csv"
    report_path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "campaign_id",
                "variant",
                "generator",
                "planned",
                "completed",
                "assessable",
                "confirmed",
                "confirmed_rate",
                "technical_rate",
                "latency_p50_ms",
                "latency_p90_ms",
                "trace_delivered_rate",
                "generation_api_estimate_usd",
                "generation_unestimated_calls",
                "judge_api_estimate_usd",
                "judge_unestimated_calls",
            ],
        )
        writer.writeheader()
        for cell, report in sorted(value["cells"].items()):
            variant, generator = cell.split("/", 1)
            summary = report["summary"]
            writer.writerow(
                {
                    "campaign_id": value["campaign"]["id"],
                    "variant": variant,
                    "generator": generator,
                    "planned": summary["planned"],
                    "completed": summary["technical_completion"]["numerator"],
                    "assessable": summary["judge_assessable"]["numerator"],
                    "confirmed": summary["confirmed_grounded_success"]["numerator"],
                    "confirmed_rate": summary["confirmed_grounded_success"]["value"],
                    "technical_rate": summary["technical_completion"]["value"],
                    "latency_p50_ms": summary["latency_total_completed"]["p50_ms"],
                    "latency_p90_ms": summary["latency_total_completed"]["p90_ms"],
                    "trace_delivered_rate": summary["trace_delivered_completed"]["value"],
                    "generation_api_estimate_usd": report["api_equivalent_cost"]["generation"][
                        "estimated_total_api_usd"
                    ],
                    "generation_unestimated_calls": report["api_equivalent_cost"]["generation"][
                        "unestimated_call_count"
                    ],
                    "judge_api_estimate_usd": report["api_equivalent_cost"]["judge"][
                        "estimated_total_api_usd"
                    ],
                    "judge_unestimated_calls": report["api_equivalent_cost"]["judge"][
                        "unestimated_call_count"
                    ],
                }
            )
    return report_path, csv_path


def _parser():
    parser = argparse.ArgumentParser(prog="medical_assistant.evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "import",
        "freeze",
        "run",
        "resume",
        "retry",
        "retry-judge",
        "reconcile-judge",
        "report",
    ):
        command = commands.add_parser(name)
        command.add_argument("--release-dir", required=True)
        if name != "import":
            command.add_argument(
                "--partition", required=True, choices=("development", "acceptance")
            )
            command.add_argument("--extraction-map", required=True)
        if name not in {"import", "freeze"}:
            command.add_argument("--campaign-id", required=True)
        if name == "freeze":
            command.add_argument("--config", required=True)
        if name in {"run", "resume", "reconcile-judge"}:
            command.add_argument("--limit", type=int)
        if name in {"retry", "retry-judge"}:
            command.add_argument("--row-id", required=True)
            command.add_argument("--reason", required=True)
        if name == "report":
            command.add_argument("--output-dir", default="tmp/work/runtime/evaluation/reports")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    settings = get_settings()
    store = Store(settings)
    store.migrate()
    ledger = Ledger(store)
    corpus_ledger = CorpusLedger(store)
    release = Release(
        args.release_dir,
        getattr(args, "extraction_map", None),
        require_gold=args.command != "import",
        partition=getattr(args, "partition", None),
    )
    service = Service(settings) if args.command != "report" else None
    runner = None
    try:
        if args.command == "import":
            corpus = corpus_ledger.create_import(release)
            result = service.import_release(corpus_ledger, corpus, release)
            print(
                json.dumps(
                    {
                        "corpus_id": result["corpus_id"],
                        "status": result["status"],
                        "collection_id": result["source_map"]["collection_id"],
                        "collection_revision": result["source_map"]["collection_revision"],
                        "document_count": len(result["source_map"]["documents"]),
                    }
                )
            )
            return 0
        if args.command == "freeze":
            corpus = corpus_ledger.by_corpus_id(release.manifest["corpus_id"])
            if corpus is None:
                raise ValueError("Shared corpus has not been imported")
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            config["embedding_model"] = settings.embedding_model
            config["embedding_revision"] = settings.embedding_revision
            price_path = Path(__file__).with_name("pricing_snapshot.json")
            config["price_snapshot_id"] = json.loads(price_path.read_text(encoding="utf-8"))["id"]
            config["price_snapshot_sha256"] = file_hash(price_path)
            if (
                service.request("GET", "/api/health")["provider"] != config.get("provider")
                or config.get("provider") == "api"
            ):
                raise ValueError("Frozen provider must match healthy non-paid runtime")
            evidence = CorpusEvidence(release, corpus["source_map"], service)
            evidence.validate_all(selected_cases(release, config))
            result = ledger.freeze(release, config, corpus, service)
            print(
                json.dumps(
                    {
                        "campaign_id": result["id"],
                        "status": result["status"],
                        "planned_count": result["planned_count"],
                    }
                )
            )
            return 0
        campaign = ledger.campaign(args.campaign_id)
        if campaign is None:
            raise ValueError("Campaign does not exist")
        release.verify_hashes(campaign)
        validate_campaign_corpus(campaign, corpus_ledger.corpus(campaign["corpus_row_id"]), release)
        if campaign["status"] not in {"frozen", "running", "incomplete", "complete"}:
            raise ValueError("Campaign is not frozen")
        if args.command in {"run", "resume"}:
            if args.limit is not None and args.limit < 1:
                raise ValueError("Limit must be positive")
            runner = Runner(ledger, release, campaign, service, settings)
            runner.verify()
            status = (
                runner.execute_pending(args.limit)
                if args.command == "run"
                else runner.resume(args.limit)
            )
            print(
                json.dumps(
                    {
                        "campaign_id": campaign["id"],
                        "status": status,
                        "provider_outage": runner.outage_code,
                        "state_counts": dict(
                            Counter(row["state"] for row in ledger.rows(campaign["id"]))
                        ),
                    },
                    sort_keys=True,
                )
            )
            return 2 if runner.outage_code else 0
        if args.command in {"retry", "retry-judge"}:
            runner = Runner(ledger, release, campaign, service, settings)
            runner.verify()
            status = runner.retry(
                args.row_id, args.reason, judge_only=args.command == "retry-judge"
            )
            print(
                json.dumps(
                    {
                        "campaign_id": campaign["id"],
                        "status": status,
                        "row_id": args.row_id,
                        "provider_outage": runner.outage_code,
                    },
                    sort_keys=True,
                )
            )
            return 2 if runner.outage_code else 0
        if args.command == "reconcile-judge":
            runner = Runner(ledger, release, campaign, service, settings)
            runner.verify()
            checked = runner.reconcile_judges(args.limit or 25)
            print(json.dumps({"campaign_id": campaign["id"], "checked": checked}))
            return 0
        if args.command == "report":
            release.verify_hashes(campaign)
            if campaign["code_sha256"] != code_hash() or campaign[
                "rubric_sha256"
            ] != instruction_hash("judge"):
                raise ValueError("Frozen code or judge rubric changed")
            value = report_data(ledger, campaign, release)
            report_path, csv_path = write_report(value, args.output_dir)
            print(
                json.dumps(
                    {
                        "campaign_id": campaign["id"],
                        "json": str(report_path),
                        "csv": str(csv_path),
                        "status": campaign["status"],
                    },
                    sort_keys=True,
                )
            )
            return 0
    finally:
        if runner is not None:
            runner.flush_telemetry()
        if service is not None:
            service.close()
    return 1


def _cost_group(estimates):
    """Summarize token-price estimates without presenting missing usage as free work.

    Args:
        estimates (list[dict]): Per-call estimated_api_cost_usd decimal string
            or None, billing_mode and estimate_limitation.

    Returns:
        dict: Call counts, known_subtotal_api_usd decimal string, limitation/
            billing-mode counts and actual_billed_usd=None. estimated_total_api_usd
            is None when empty or any call is unestimated; otherwise the known
            subtotal. The subtotal is still retained with incomplete coverage.
    """
    known = [
        Decimal(value["estimated_api_cost_usd"])
        for value in estimates
        if value["estimated_api_cost_usd"] is not None
    ]
    unknown = [value for value in estimates if value["estimated_api_cost_usd"] is None]
    modes = Counter(value["billing_mode"] for value in estimates)
    return {
        "call_count": len(estimates),
        "estimated_call_count": len(known),
        "unestimated_call_count": len(unknown),
        "known_subtotal_api_usd": str(sum(known, Decimal(0))),
        "estimated_total_api_usd": str(sum(known, Decimal(0)))
        if estimates and not unknown
        else None,
        "estimate_limitation_counts": dict(
            Counter(value["estimate_limitation"] for value in unknown)
        ),
        "actual_billed_usd": None,
        "billing_mode_counts": dict(modes),
    }


def report_run_facts(store, run_ids, batch_size=200):
    """Read bounded run/audit facts for cost and trace reporting.

    Args:
        store (Store): Database access for runs and generation audit events.
        run_ids (Iterable[str]): Referenced run UUIDs; deduplicated in input order.
        batch_size (int): Maximum IDs per database batch, from 1 through 200.

    Returns:
        dict[str, dict]: Existing run IDs mapped to model, trace_status,
            generation_started operation-ID list and ordered generation_events
            with operation_id, usage and billing_mode. Missing runs are omitted;
            completed and failed generation events both remain attributable.

    Raises:
        ValueError: Batch size is outside the bounded reporting range.
    """
    if not 1 <= batch_size <= 200:
        raise ValueError("report fact batch size must be between 1 and 200")
    facts = {}
    unique_ids = list(dict.fromkeys(run_ids))
    with store.engine.connect() as connection:
        for offset in range(0, len(unique_ids), batch_size):
            ids = unique_ids[offset : offset + batch_size]
            for row in connection.execute(
                text("SELECT id,model,trace_status FROM runs WHERE id=ANY(CAST(:ids AS uuid[]))"),
                {"ids": ids},
            ).mappings():
                facts[str(row["id"])] = {
                    "model": row["model"],
                    "trace_status": row["trace_status"],
                    "generation_started": [],
                    "generation_events": [],
                }
            for row in connection.execute(
                text(
                    "SELECT run_id,stage,payload->>'operation_id' AS operation_id,"
                    "payload->'usage' AS usage,"
                    "payload->>'billing_mode' AS billing_mode FROM run_audit "
                    "WHERE run_id=ANY(CAST(:ids AS uuid[])) "
                    "AND stage IN ('generation.started','generation.completed','generation.failed') "
                    "ORDER BY run_id,at,id"
                ),
                {"ids": ids},
            ).mappings():
                fact = facts.get(str(row["run_id"]))
                if fact is None:
                    continue
                if row["stage"] == "generation.started":
                    fact["generation_started"].append(row["operation_id"])
                else:
                    fact["generation_events"].append(
                        {
                            "operation_id": row["operation_id"],
                            "usage": row["usage"],
                            "billing_mode": row["billing_mode"],
                        }
                    )
    return facts


def cost_for_attempt(row, store, snapshot_id="openai_standard_short_2026-09-25", run_facts=None):
    """Attribute all generation and judge work across corrections to one planned cell.

    Deduplicate dialogue run IDs across answer history, require unique started
    generation operations and at most one matching terminal event, then price
    every started operation. Missing run facts or terminal usage remain unknown
    estimates. Include every historic judge and current recorded request,
    retaining usage from failures and validation-rejected outputs.

    Args:
        row (dict): generator and attempts list; each attempt includes turns
            with optional run_id, optional judge_history, judge_request_id,
            judge_usage/judge_raw_usage and judge_billing_mode.
        store (Store): Reads run/audit facts when run_facts is None.
        snapshot_id (str): Frozen API-equivalent token-pricing schedule.
        run_facts (dict[str, dict] | None): Optional bulk facts from
            report_run_facts; missing IDs are treated as unknown, not reread.

    Returns:
        dict: price_snapshot_id and separate generation/judge cost groups with
            call counts, decimal-string known subtotal, nullable complete total
            and missing-estimate reasons. Actual billed charges stay None.

    Raises:
        ValueError: Generation start IDs repeat/are missing, a terminal lacks
            its start or multiple terminal events claim the same operation.
    """
    run_ids = list(
        dict.fromkeys(
            turn["run_id"]
            for attempt in row["attempts"]
            for turn in attempt.get("turns", [])
            if turn.get("run_id")
        )
    )
    generation = []
    for run_id in run_ids:
        if run_facts is None:
            run = store.get_run(run_id)
            if run is None:
                fact = None
            else:
                audit = store.audit(run_id)
                fact = {
                    "model": run["model"],
                    "generation_started": [
                        item["payload"].get("operation_id")
                        for item in audit
                        if item["stage"] == "generation.started"
                    ],
                    "generation_events": [
                        {
                            "operation_id": item["payload"].get("operation_id"),
                            "usage": item["payload"].get("usage"),
                            "billing_mode": item["payload"].get("billing_mode"),
                        }
                        for item in audit
                        if item["stage"] in {"generation.completed", "generation.failed"}
                    ],
                }
        else:
            fact = run_facts.get(run_id)
        if fact is None:
            generation.append(
                estimate_api_equivalent_cost(
                    row["generator"], {}, snapshot_id, billing_mode="unknown"
                )
            )
            continue
        started = fact["generation_started"]
        if len(started) != len(set(started)) or None in started:
            raise ValueError(f"Generation start operation IDs are invalid: {run_id}")
        terminal = {}
        for event in fact["generation_events"]:
            operation_id = event["operation_id"]
            if operation_id in terminal:
                raise ValueError(f"Generation operation has multiple terminal events: {run_id}")
            terminal[operation_id] = event
        if set(terminal) - set(started):
            raise ValueError(f"Generation terminal operation has no start: {run_id}")
        for operation_id in started:
            event = terminal.get(operation_id) or {}
            generation.append(
                estimate_api_equivalent_cost(
                    fact["model"],
                    event.get("usage") or {},
                    snapshot_id,
                    billing_mode=event.get("billing_mode") or "unknown",
                )
            )
    judge = []
    for attempt in row["attempts"]:
        for prior in attempt.get("judge_history", []):
            judge.append(
                estimate_api_equivalent_cost(
                    "gpt-6-sol",
                    prior.get("usage") or {},
                    snapshot_id,
                    billing_mode=prior.get("billing_mode") or "unknown",
                )
            )
        if attempt.get("judge_request_id"):
            judge.append(
                estimate_api_equivalent_cost(
                    "gpt-6-sol",
                    attempt.get("judge_usage") or attempt.get("judge_raw_usage") or {},
                    snapshot_id,
                    billing_mode=attempt.get("judge_billing_mode") or "unknown",
                )
            )
    return {
        "price_snapshot_id": snapshot_id,
        "generation": _cost_group(generation),
        "judge": _cost_group(judge),
    }


def combine_costs(values):
    """Combine cell costs while retaining incomplete usage coverage.

    Args:
        values (list[dict]): Cost records from cost_for_attempt, each with
            price_snapshot_id and generation/judge groups. Callers supply a
            common snapshot; this function does not verify snapshot equality.

    Returns:
        dict: Summed generation/judge call counts, decimal USD known subtotals,
            limitation and billing-mode counts. Complete totals are None for
            zero calls or any unestimated call; actual_billed_usd is always None.
            price_snapshot_id comes from the first record, or None when empty.
    """
    result = {}
    for kind in ("generation", "judge"):
        calls = [value[kind] for value in values]
        known = sum((Decimal(value["known_subtotal_api_usd"]) for value in calls), Decimal(0))
        missing = sum(value["unestimated_call_count"] for value in calls)
        count = sum(value["call_count"] for value in calls)
        reasons = Counter()
        modes = Counter()
        for value in calls:
            reasons.update(value["estimate_limitation_counts"])
            modes.update(value.get("billing_mode_counts", {}))
        result[kind] = {
            "call_count": count,
            "estimated_call_count": count - missing,
            "unestimated_call_count": missing,
            "known_subtotal_api_usd": str(known),
            "estimated_total_api_usd": str(known) if count and not missing else None,
            "estimate_limitation_counts": dict(reasons),
            "actual_billed_usd": None,
            "billing_mode_counts": dict(modes),
        }
    result["price_snapshot_id"] = values[0]["price_snapshot_id"] if values else None
    return result


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
