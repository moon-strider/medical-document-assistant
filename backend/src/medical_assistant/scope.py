import hashlib
import json
import uuid

_COUNTERS = ("revision", "source_count", "ready_count", "unavailable_count")
_SCOPE_KEYS = {"id", "collection_id", *_COUNTERS, "snapshot_hash"}


def collection_snapshot(collection):
    """Describe the collection version that a run is allowed to read.

    Args:
        collection (Mapping): Collection row with ``id`` and integer
            ``revision``, ``source_count``, ``ready_count``, and
            ``unavailable_count`` fields.

    Returns:
        dict: String ``collection_id`` and the four counters, copied without
        validation. No source IDs, file hashes, or document text are included.
    """
    return {"collection_id": str(collection["id"]), **{key: collection[key] for key in _COUNTERS}}


def build_run_scope(collection, run_id):
    """Freeze collection version metadata for retrieval and answer publication.

    The digest identifies canonical JSON of the collection ID and counters.
    It is neither a content hash of the archive nor an authentication signature;
    the run ID is attached separately and does not affect the digest. Callers
    must update the collection revision when searchable content changes.

    Args:
        collection (Mapping): Collection row accepted by ``collection_snapshot``.
        run_id (str | UUID): Run identity, stored as its string representation.

    Returns:
        dict: Exactly ``id``, ``collection_id``, ``revision``, ``source_count``,
        ``ready_count``, ``unavailable_count``, and hexadecimal ``snapshot_hash``.
        Building this value does not validate counters or persist a snapshot.
    """
    snapshot = collection_snapshot(collection)
    digest = hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"id": str(run_id), **snapshot, "snapshot_hash": digest}


def scope_matches(scope, collection, run_id, conversation_collection_id):
    """Decide whether frozen run metadata still permits access to a collection.

    A match requires the exact scope shape, matching run and conversation
    collection identities, nonnegative integer counters (excluding booleans),
    consistent source totals, and equality to a newly built collection scope.
    The caller separately enforces run status; this predicate does not inspect it.

    Args:
        scope (object): Expected scope dict produced by ``build_run_scope``.
        collection (Mapping | None): Current row with ``id`` and the four
            snapshot counters, or None when the collection no longer exists.
        run_id (str | UUID): Expected run identity.
        conversation_collection_id (str | UUID): Collection owning the run's
            conversation, preventing a scope from crossing collections.

    Returns:
        bool: False for a missing collection, malformed scope, identity mismatch,
        invalid or inconsistent counters, or changed metadata/hash. True checks
        metadata freshness, not document contents or complete evidence coverage.
        Neither outcome changes the scope or database state.
    """
    if not isinstance(scope, dict) or set(scope) != _SCOPE_KEYS:
        return False
    try:
        if str(uuid.UUID(scope["id"])) != str(run_id):
            return False
        collection_id = str(uuid.UUID(scope["collection_id"]))
    except (ValueError, TypeError, AttributeError):
        return False
    if collection is None or collection_id != str(collection["id"]):
        return False
    if collection_id != str(conversation_collection_id):
        return False
    if any(type(scope[key]) is not int or scope[key] < 0 for key in _COUNTERS):
        return False
    if scope["source_count"] != scope["ready_count"] + scope["unavailable_count"]:
        return False
    return scope == build_run_scope(collection, run_id)
