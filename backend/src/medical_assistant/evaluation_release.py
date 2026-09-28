import hashlib
import json
import re
from pathlib import Path

_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def ref_key(reference):
    return canonical(
        {key: reference[key] for key in ("doc_id", "page", "line_start", "line_end", "quote")}
    )


class Release:
    """Validate a local corpus release before import or a frozen evaluation.

    Import mode reads the shared document inventory without partition gold.
    Evaluation mode additionally selects one partition and binds its gold to
    reviewed extraction locators. Validation reads originals and release JSON
    files; it neither imports sources nor calls a model. File-byte hashes and
    the derived release identity are retained for later campaign verification.
    """

    def __init__(self, directory, extraction_map=None, require_gold=False, partition=None):
        """Load and validate release artifacts for the requested workflow.

        Args:
            directory (str | Path): Release root containing manifest.json,
                questions.json, corpus originals and partition gold files.
            extraction_map (str | Path | None): Reviewed map with corpus_id and
                an entries list identifying authored document/page/line/quote references.
                Required when require_gold is True; optional for import.
            require_gold (bool): Whether to validate partition gold and extraction
                references. Must be True exactly when partition is provided.
            partition (str | None): development or acceptance for evaluation;
                None for importing the complete shared corpus.

        Raises:
            ValueError: Mode, inventory, hashes, cases, gold or reviewed references
                violate the release contract.
            OSError: A required artifact cannot be read.
        """
        if (partition is None) == require_gold:
            raise ValueError("Import needs no partition; evaluation needs a partition and gold")
        if partition is not None and partition not in {"development", "acceptance"}:
            raise ValueError("Invalid case partition")
        self.partition = partition
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.questions_path = self.directory / "questions.json"
        self.gold_path = self.directory / "gold" / f"{partition}.json" if partition else None
        self.extraction_path = Path(extraction_map).resolve() if extraction_map else None
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.questions = json.loads(self.questions_path.read_text(encoding="utf-8"))
        self.gold = json.loads(self.gold_path.read_text(encoding="utf-8")) if require_gold else None
        self.extraction = (
            json.loads(self.extraction_path.read_text(encoding="utf-8"))
            if self.extraction_path
            else None
        )
        self.hashes = {
            "manifest_sha256": file_hash(self.manifest_path),
            "questions_sha256": file_hash(self.questions_path),
            "gold_sha256": file_hash(self.gold_path) if require_gold else None,
            "extraction_map_sha256": file_hash(self.extraction_path)
            if self.extraction_path
            else None,
        }
        self.documents = self._documents()
        self.all_cases = self._cases()
        self.cases = {
            case_id: case
            for case_id, case in self.all_cases.items()
            if partition is None or case["partition"] == partition
        }
        if partition is not None and not self.cases:
            raise ValueError("Selected case partition is empty")
        self.golds = self._gold() if require_gold else {}
        self.entries = self._entries() if self.extraction is not None else {}
        if require_gold:
            self._validate_gold_refs()
            self.release_id = (
                "eval-"
                + digest(
                    {
                        "corpus_id": self.manifest["corpus_id"],
                        "partition": partition,
                        "questions_sha256": self.hashes["questions_sha256"],
                        "gold_sha256": self.hashes["gold_sha256"],
                        "extraction_map_sha256": self.hashes["extraction_map_sha256"],
                    }
                )[:20]
            )
        else:
            self.release_id = None

    def _documents(self):
        """Verify original-file identity and complete family membership.

        Check the manifest digest, safe single-format paths, English document
        metadata, original bytes, declared counts and one family per document.
        Derive corpus identity from documents and families so case partitioning
        cannot silently change the shared corpus.

        Returns:
            dict[str, dict]: Manifest documents keyed by doc_id; each includes
                path, format, sha256, document_class and family_id, with an
                optional patient_id.

        Raises:
            ValueError: Identity, file availability/hash, metadata or family
                inventory differs from the declared corpus.
        """
        manifest = self.manifest
        if manifest.get("schema_version") != 2 or not _ID.fullmatch(manifest.get("corpus_id", "")):
            raise ValueError("Invalid corpus manifest identity")
        if (
            not isinstance(manifest.get("case_counts"), dict)
            or set(manifest["case_counts"]) != {"development", "acceptance"}
            or any(
                type(value) is not int or value < 0 for value in manifest["case_counts"].values()
            )
        ):
            raise ValueError("Invalid case partition counts")
        if not _HEX.fullmatch(manifest.get("artifact_manifest_sha256", "")):
            raise ValueError("Invalid artifact manifest digest")
        if (
            digest(
                {key: value for key, value in manifest.items() if key != "artifact_manifest_sha256"}
            )
            != manifest["artifact_manifest_sha256"]
        ):
            raise ValueError("Artifact manifest digest mismatch")
        documents = {}
        for document in manifest.get("documents", []):
            doc_id = document.get("doc_id", "")
            format_name = document.get("format")
            if (
                not _ID.fullmatch(doc_id)
                or format_name not in {"txt", "pdf"}
                or document.get("path") != f"corpus/{doc_id}.{format_name}"
            ):
                raise ValueError("Unsafe or inconsistent document path")
            if (
                doc_id in documents
                or document.get("language") != "en"
                or document.get("document_class") not in {"D1", "D2", "D3", "D4", "D5", "other"}
            ):
                raise ValueError("Invalid document metadata")
            path = (self.directory / document["path"]).resolve()
            if not path.is_relative_to(self.directory / "corpus") or not path.is_file():
                raise ValueError("Corpus document is unavailable")
            if file_hash(path) != document.get("sha256"):
                raise ValueError(f"Corpus hash changed: {doc_id}")
            documents[doc_id] = document
        if not documents:
            raise ValueError("Release contains no documents")
        if type(manifest.get("document_count")) is not int or manifest["document_count"] != len(
            documents
        ):
            raise ValueError("Declared document count differs from corpus")
        family_ids = set()
        member_ids = []
        for family in manifest.get("families", []):
            family_id = family["family_id"]
            if family_id in family_ids:
                raise ValueError("Duplicate family")
            family_ids.add(family_id)
            for doc_id in family["doc_ids"]:
                member_ids.append(doc_id)
                if (
                    doc_id not in documents
                    or documents[doc_id]["family_id"] != family_id
                    or documents[doc_id].get("patient_id") != family.get("patient_id")
                ):
                    raise ValueError("Family document membership mismatch")
        if (
            family_ids != {document["family_id"] for document in documents.values()}
            or len(member_ids) != len(documents)
            or set(member_ids) != set(documents)
        ):
            raise ValueError("Document family inventory mismatch")
        corpus_digest = digest(
            {
                "documents": sorted(manifest["documents"], key=lambda item: item["doc_id"]),
                "families": sorted(manifest["families"], key=lambda item: item["family_id"]),
            }
        )
        if manifest["corpus_id"] != "corpus-" + corpus_digest[:20]:
            raise ValueError("Corpus ID differs from document inventory")
        return documents

    def _cases(self):
        """Validate authored questions without narrowing runtime retrieval.

        Check partition identity, anchor/family metadata and patient-consistent
        offline scope. Each case has one or two user turns and evaluates the last
        turn; scope annotations remain evaluation metadata rather than a filter
        on the shared runtime collection.

        Returns:
            dict[str, dict]: All cases by case_id, including partition, scope
                with doc_ids, anchor_doc_id, turns, target_turn and two_turn.

        Raises:
            ValueError: Cases, English language, dialogue structure or partition
                counts violate the authored release contract.
        """
        if not isinstance(self.questions, list):
            raise ValueError("questions.json must contain a list")
        cases = {}
        for case in self.questions:
            case_id = case.get("case_id", "")
            if (
                not _ID.fullmatch(case_id)
                or case_id in cases
                or case.get("partition") not in {"development", "acceptance"}
            ):
                raise ValueError("Invalid case identity")
            anchor = self.documents.get(case.get("anchor_doc_id"))
            scope_ids = case.get("scope", {}).get("doc_ids", [])
            if (
                anchor is None
                or not scope_ids
                or len(scope_ids) != len(set(scope_ids))
                or case["anchor_doc_id"] not in scope_ids
            ):
                raise ValueError("Invalid case scope or anchor")
            if any(
                doc_id not in self.documents
                or (
                    case.get("patient_id") is not None
                    and self.documents[doc_id].get("patient_id") != case["patient_id"]
                )
                for doc_id in scope_ids
            ):
                raise ValueError("Case scope crosses patients or references missing document")
            if (
                case.get("patient_id") is not None
                and case["scope"].get("patient") != case["patient_id"]
            ):
                raise ValueError("Case scope patient mismatch")
            if (
                anchor["document_class"] != case["anchor_document_class"]
                or anchor["family_id"] != case["family_id"]
            ):
                raise ValueError("Case anchor metadata mismatch")
            turns = case.get("turns", [])
            if (
                len(turns) not in {1, 2}
                or case.get("target_turn") != len(turns) - 1
                or case.get("two_turn") != (len(turns) == 2)
                or any(turn.get("role") != "user" or not turn.get("text") for turn in turns)
            ):
                raise ValueError("Invalid case turn sequence")
            if case.get("query_language") != "en" or case.get("answer_language") != "en":
                raise ValueError("Invalid case language")
            cases[case_id] = case
        if not cases:
            raise ValueError("Release contains no cases")
        for partition, count in self.manifest["case_counts"].items():
            if sum(case["partition"] == partition for case in cases.values()) != count:
                raise ValueError("Declared case partition count differs from questions")
        return cases

    def _gold(self):
        """Require exactly one supported rubric record per selected case.

        Validate the answer-status vocabulary and unique required-claim IDs;
        calculator annotations are rejected. Claim content and evidence support
        are not independently judged here.

        Returns:
            dict[str, dict]: Gold by case_id with gold_status and optional
                required_claims, acceptable evidence alternatives and scope rules.

        Raises:
            ValueError: Gold is duplicated, missing, outside the partition or
                uses unsupported status/calculation annotations.
        """
        if not isinstance(self.gold, list):
            raise ValueError("gold.json must contain a list")
        golds = {}
        for value in self.gold:
            case_id = value.get("case_id")
            if case_id not in self.cases or case_id in golds:
                raise ValueError("Gold case mismatch")
            if "calculation" in value:
                raise ValueError("Calculator annotations are not part of the current gold contract")
            if value.get("gold_status") not in {
                "supported",
                "partial",
                "conflicting",
                "not_documented",
                "needs_clarification",
            }:
                raise ValueError("Invalid gold status")
            claims = value.get("required_claims", [])
            ids = [claim.get("claim_id") for claim in claims]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate gold claim")
            golds[case_id] = value
        if set(golds) != set(self.cases):
            raise ValueError("Gold and question case IDs differ")
        return golds

    def _entries(self):
        """Validate portable extraction review independent of database IDs.

        Reviewed outcomes may be intact, mismatch or unavailable. Intact entries
        must provide nonempty extracted_locators; runtime span_ids are forbidden
        so importing the same release into another database remains possible.

        Returns:
            dict[str, dict]: Entries keyed by canonical authored reference, with
                source_sha256, reviewed_by, outcome and optional locators.

        Raises:
            ValueError: Corpus identity, review completeness, reference uniqueness
                or source hashes differ from the release.
        """
        if self.extraction.get("corpus_id") != self.manifest[
            "corpus_id"
        ] or not self.extraction.get("schema_version"):
            raise ValueError("Extraction map corpus identity mismatch")
        entries = {}
        for entry in self.extraction.get("entries", []):
            key = ref_key(entry["reference"])
            if (
                key in entries
                or entry.get("outcome") not in {"intact", "mismatch", "unavailable"}
                or not entry.get("reviewed_by")
                or "span_ids" in entry
                or (
                    entry.get("outcome") == "intact"
                    and (
                        not isinstance(entry.get("extracted_locators"), list)
                        or not entry["extracted_locators"]
                    )
                )
            ):
                raise ValueError("Extraction entry lacks a unique reviewed outcome")
            document = self.documents.get(entry["reference"]["doc_id"])
            if document is None or entry.get("source_sha256") != document["sha256"]:
                raise ValueError("Extraction entry source hash mismatch")
            entries[key] = entry
        return entries

    def _validate_gold_refs(self):
        """Require a reviewed extraction outcome for every gold reference.

        A mismatch or unavailable outcome is permitted; this verifies review
        coverage, not that every required claim has intact runtime evidence.

        Raises:
            ValueError: The extraction map is absent or a referenced document or
                reviewed entry is missing.
        """
        if self.extraction is None:
            raise ValueError("Reviewed extraction map is required")
        for gold in self.golds.values():
            references = [
                reference
                for claim in gold.get("required_claims", [])
                for option in claim.get("acceptable_evidence_sets", [])
                for reference in option
            ]
            for reference in references:
                if (
                    reference["doc_id"] not in self.documents
                    or ref_key(reference) not in self.entries
                ):
                    raise ValueError("Gold reference lacks reviewed extraction entry")

    def verify_hashes(self, campaign):
        """Reject changed release artifacts when reopening a campaign.

        Args:
            campaign (dict): Frozen manifest_sha256, questions_sha256,
                gold_sha256, extraction_map_sha256 and artifact_manifest_sha256.

        Raises:
            ValueError: Any retained artifact hash differs from this loaded release.
        """
        for name in ("manifest_sha256", "questions_sha256", "gold_sha256", "extraction_map_sha256"):
            if campaign[name] != self.hashes[name]:
                raise ValueError(f"Frozen release file changed: {name}")
        if campaign["artifact_manifest_sha256"] != self.manifest["artifact_manifest_sha256"]:
            raise ValueError("Artifact manifest digest changed")
