import asyncio
import base64
import hashlib
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from medical_assistant.api import create_app
from medical_assistant.evaluation import (
    CorpusEvidence,
    CorpusLedger,
    Ledger,
    Runner,
    Service,
    report_data,
    write_report,
)
from medical_assistant.evaluation_release import Release, digest, file_hash
from medical_assistant.graph import packet_hash
from medical_assistant.mcp_client import MCPToolFailure
from medical_assistant.providers import bridge_client
from medical_assistant.providers.bridge_client import BridgeProvider
from medical_assistant.settings import Settings
from medical_assistant.storage import Store
from medical_assistant.telemetry import Telemetry
from medical_assistant.worker import run_once
from reportlab.pdfgen import canvas
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "tmp/work/runtime/.env"
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def admin(database, sql):
    container = os.environ.get("PFL_E2E_PG_CONTAINER")
    command = (
        ["docker", "exec", "-i", container]
        if container
        else [
            "docker",
            "compose",
            "--env-file",
            str(ENV_FILE),
            "-f",
            str(ROOT / "deploy/compose.yaml"),
            "exec",
            "-T",
            "postgres",
        ]
    ) + [
        "psql",
        "-U",
        "postgres",
        "-d",
        database,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Local PostgreSQL test database setup failed")


@pytest.fixture(scope="session")
def database():
    name = "pfl_e2e_" + uuid.uuid4().hex
    admin("postgres", f"CREATE DATABASE {name} OWNER assistant")
    try:
        admin(name, "CREATE EXTENSION IF NOT EXISTS vector")
        admin(name, "CREATE EXTENSION IF NOT EXISTS pg_textsearch")
        admin(name, "GRANT CONNECT ON DATABASE " + name + " TO assistant_readonly")
        admin(name, "GRANT USAGE ON SCHEMA public TO assistant_readonly")
        admin(
            name,
            "ALTER DEFAULT PRIVILEGES FOR ROLE assistant IN SCHEMA public GRANT SELECT ON TABLES TO assistant_readonly",
        )
        yield name
    finally:
        admin("postgres", f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


@pytest.fixture(scope="session")
def settings(database):
    original = Settings()
    port = int(os.environ.get("PFL_E2E_PG_PORT", "5439"))
    with tempfile.TemporaryDirectory(prefix="pfl-e2e-", dir=ROOT / "tmp/work") as directory:
        yield Settings(
            database_url=make_url(original.database_url)
            .set(database=database, port=port)
            .render_as_string(hide_password=False),
            read_database_url=make_url(original.read_database_url)
            .set(database=database, port=port)
            .render_as_string(hide_password=False),
            data_dir=Path(directory),
            provider="fixture",
            app_origin="http://127.0.0.1:8080",
            launch_token="test-launch-token-0123456789",
            service_token="test-service-token-0123456789",
            langfuse_public_key="",
            langfuse_secret_key="",
        )


@pytest.fixture(scope="session")
def store(settings):
    value = Store(settings)
    value.migrate()
    yield value
    value.engine.dispose()


class NoopObservation:
    trace_id = None
    trace_url = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def update(self, **_):
        return None


class NoopTelemetry:
    def __init__(self):
        self.observation_calls = []
        self.observation_metadata = []

    def trace(self, *_, **__):
        return NoopObservation()

    def observation(self, *args, **kwargs):
        self.observation_calls.append(args[:2])
        self.observation_metadata.append(kwargs.get("metadata"))
        return NoopObservation()

    def reconcile_run(self, *_):
        return None

    def reconcile_judge_observation(self, **_):
        return "pending"

    def score(self, *_args, **_kwargs):
        return None


class Provider:
    def __init__(self, respond, usage=None):
        self.respond = respond
        self.plan_respond = None
        self.usage = usage or {}
        self.entered = threading.Event()
        self.completed = threading.Event()
        self.cancelled = []
        self.payloads = []
        self.calls = []

    async def generate(self, payload, *, role, model, request_id, final_only):
        self.payloads.append(payload)
        self.calls.append((request_id, payload))
        if payload.get("task_mode") == "query_planning":
            assert final_only is False
            assert payload["evidence"] == {"spans": []}
            assert set(payload["available_tools"]) == {"search_evidence"}
            if self.plan_respond is not None:
                output = await self.plan_respond(payload)
            else:
                query = " ".join(
                    [
                        message["content"]
                        for message in payload["history"]
                        if message["role"] == "user"
                    ][-1:]
                    + [payload["question"]]
                )[:512]
                output = {
                    "kind": "request_tools",
                    "status": "partial",
                    "answer": "",
                    "claims": [],
                    "limitations": [],
                    "coverage_requirement": "claims",
                    "requests": [
                        {
                            "tool": "search_evidence",
                            "query": query,
                            "span_ids": [],
                        }
                    ],
                }
        else:
            self.entered.set()
            output = await self.respond(payload)
            self.completed.set()
        return {
            "output": output,
            "usage": self.usage,
            "elapsed_ms": 1,
            "provider": "fixture",
            "model": model,
            "role": role,
            "instruction_hash": "test",
            "billing_mode": "fixture",
            "cost_state": "not_applicable",
        }

    async def cancel(self, request_id):
        self.cancelled.append(request_id)


class ASGIService(Service):
    def __init__(self, settings, client, store):
        self.settings = settings
        self.client = client
        self.store = store

    def request(self, method, path, **kwargs):
        headers = {**service(self.settings), **kwargs.pop("headers", {})}
        response = self.client.request(method, path, headers=headers, **kwargs)
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code} at {path}: {response.text[:300]}")
        return response.json()

    def wait_source(self, collection_id, source_id, timeout_seconds=600):
        ingest(self.store, self.settings)
        return super().wait_source(collection_id, source_id, timeout_seconds)


class SyntheticJudge:
    def __init__(self):
        self.calls = []
        self.fail_once = True

    async def generate(self, payload, *, role, model, request_id, final_only):
        assert role == "judge" and model == "gpt-6-sol" and final_only is True
        self.calls.append((request_id, payload))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic judge interruption")
        candidate = payload["candidate"]
        output = {
            "verdict": "pass",
            "severity": "none",
            "status_correct": True,
            "coverage_valid": True,
            "scores": {
                "claim_support": 4,
                "completeness": 4,
                "contradictions": 4,
                "temporal_subject_units": 4,
                "coverage": 4,
            },
            "claim_reviews": [
                {
                    "claim_index": index,
                    "claim_text": claim["text"],
                    "evidence_ids": claim["evidence_ids"],
                    "support_label": "supported",
                    "reason": "Synthetic source states the code.",
                }
                for index, claim in enumerate(candidate["claims"])
            ],
            "required_claim_reviews": [
                {
                    "gold_claim_id": claim["claim_id"],
                    "label": "present",
                    "evidence_ids": claim["acceptable_evidence_sets"][0],
                    "reason": "The cited source states the code.",
                }
                for claim in payload["gold"]["required_claims"]
            ],
            "citation_reviews": [
                {
                    "claim_index": index,
                    "evidence_id": span_id,
                    "label": "useful",
                    "reason": "The citation supports the claim.",
                }
                for index, claim in enumerate(candidate["claims"])
                for span_id in claim["evidence_ids"]
            ],
            "reason": "Synthetic answer matches the visible source.",
        }
        return {
            "output": output,
            "usage": {
                "input_tokens": 120,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 40,
            },
            "elapsed_ms": 1,
            "provider": "fixture",
            "model": model,
            "role": role,
            "instruction_hash": "test",
            "billing_mode": "fixture",
            "cost_state": "not_applicable",
        }


def answer(payload, *, status="supported", coverage=None, cited=True):
    spans = payload["evidence"]["spans"]
    ids = [spans[0]["id"]] if spans and cited else []
    return {
        "kind": "answer",
        "status": status,
        "answer": "Alpha is documented.",
        "claims": [{"text": "Alpha is documented.", "evidence_ids": ids}] if ids else [],
        "limitations": [],
        "coverage_requirement": coverage or payload.get("coverage_policy", "claims"),
        "requests": [],
    }


def client_for(settings, store, provider, *, mcp=None):
    return TestClient(
        create_app(settings, store=store, provider=provider, mcp=mcp, telemetry=NoopTelemetry()),
        base_url=settings.app_origin,
    )


def service(settings):
    return {"Authorization": "Bearer " + settings.service_token}


def collection(client, settings):
    response = client.post(
        "/api/collections", json={"title": "Synthetic fixtures"}, headers=service(settings)
    )
    assert response.status_code == 200, response.text
    return response.json()


def upload(client, settings, collection_id, content, filename="facts.txt", request_id=None):
    response = client.post(
        f"/api/collections/{collection_id}/sources",
        files={
            "file": (
                filename,
                content,
                "application/pdf" if filename.endswith(".pdf") else "text/plain",
            )
        },
        data={
            "document_class": "other",
            "request_id": request_id or uuid.uuid4().hex,
        },
        headers=service(settings),
    )
    return response


def ingest(store, settings):
    processed = False
    for _ in range(30):
        if not run_once(store, settings, "e2e-worker"):
            assert processed
            return
        processed = True
    pytest.fail("Worker queue did not drain")


def conversation(client, settings, collection_id):
    response = client.post(
        "/api/conversations",
        json={"collection_id": collection_id, "title": "Synthetic"},
        headers=service(settings),
    )
    assert response.status_code == 200, response.text
    return response.json()


def run(client, settings, conversation_id, question="What was the alpha value?", **extra):
    response = client.post(
        f"/api/conversations/{conversation_id}/runs",
        json={
            "request_id": uuid.uuid4().hex,
            "question": question,
            "model": "gpt-6-sol",
            "variant": "V3",
            **extra,
        },
        headers=service(settings),
    )
    assert response.status_code == 202, response.text
    return response.json()


def finished(client, settings, run_id):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        result = client.get(f"/api/runs/{run_id}", headers=service(settings)).json()
        if result["status"] in TERMINAL:
            return result
        time.sleep(0.1)
    pytest.fail("Run did not finish")


def execution_stopped(client, run_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if client.portal.call(lambda: run_id not in client.app.state.tasks):
            return
        time.sleep(0.05)
    pytest.fail("Run execution task did not stop")


def sse_entries(body):
    entries = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        entries.append((fields["id"], fields["event"]))
    return entries


def pdf_bytes():
    output = io.BytesIO()
    document = canvas.Canvas(output)
    document.drawString(72, 750, "SCREENING GUIDELINE")
    document.drawString(72, 730, "The screening interval is 21 days after enrollment.")
    document.showPage()
    document.save()
    return output.getvalue()


def synthetic_release(settings, tag="main"):
    directory = settings.data_dir / "synthetic-release"
    corpus = directory / "corpus"
    corpus.mkdir(parents=True)
    documents = (
        (
            "doc-e2e-willow",
            "family-willow",
            "subject-willow",
            "Object Atlas for Willow has code ALPHA.",
        ),
        (
            "doc-e2e-cedar",
            "family-cedar",
            "subject-cedar",
            "Object Boreal for Cedar has code BETA.",
        ),
        (
            "doc-e2e-ash",
            f"family-ash-{tag}",
            f"subject-ash-{tag}",
            "Object Atlas for Ash has code OMEGA.",
        ),
    )
    manifest_documents = []
    families = []
    references = {}
    source_hashes = {}
    for doc_id, family_id, patient_id, sentence in documents:
        document_path = corpus / f"{doc_id}.txt"
        document_path.write_text(sentence + "\n", encoding="utf-8")
        source_hash = file_hash(document_path)
        source_hashes[doc_id] = source_hash
        manifest_documents.append(
            {
                "doc_id": doc_id,
                "family_id": family_id,
                "patient_id": patient_id,
                "format": "txt",
                "path": f"corpus/{doc_id}.txt",
                "sha256": source_hash,
                "language": "en",
                "document_class": "D1",
            }
        )
        families.append({"family_id": family_id, "patient_id": patient_id, "doc_ids": [doc_id]})
        references[doc_id] = {
            "doc_id": doc_id,
            "page": None,
            "line_start": 1,
            "line_end": 1,
            "quote": sentence,
        }
    manifest = {
        "schema_version": 2,
        "documents": manifest_documents,
        "families": families,
        "document_count": len(manifest_documents),
        "case_counts": {"development": 2, "acceptance": 20},
    }
    manifest["corpus_id"] = (
        "corpus-"
        + digest(
            {
                "documents": sorted(manifest_documents, key=lambda item: item["doc_id"]),
                "families": sorted(families, key=lambda item: item["family_id"]),
            }
        )[:20]
    )
    manifest["artifact_manifest_sha256"] = digest(manifest)
    cases = []
    gold = {"development": [], "acceptance": []}
    for partition, case_id, family_id, patient_id, doc_id, turns, code in (
        (
            "development",
            "case-one",
            "family-willow",
            "subject-willow",
            "doc-e2e-willow",
            ["For Willow's object Atlas, what was its code?"],
            "ALPHA",
        ),
        (
            "development",
            "case-two",
            "family-cedar",
            "subject-cedar",
            "doc-e2e-cedar",
            ["For Cedar, what was the object?", "What was its code?"],
            "BETA",
        ),
        (
            "acceptance",
            "accept-one",
            "family-willow",
            "subject-willow",
            "doc-e2e-willow",
            ["For Willow's object Atlas, what was its code?"],
            "ALPHA",
        ),
        (
            "acceptance",
            "accept-two",
            "family-cedar",
            "subject-cedar",
            "doc-e2e-cedar",
            ["For Cedar, what was the object?", "What was its code?"],
            "BETA",
        ),
    ):
        cases.append(
            {
                "case_id": case_id,
                "partition": partition,
                "family_id": family_id,
                "patient_id": patient_id,
                "anchor_doc_id": doc_id,
                "anchor_document_class": "D1",
                "scope": {"patient": patient_id, "doc_ids": [doc_id]},
                "turns": [{"role": "user", "text": text} for text in turns],
                "target_turn": len(turns) - 1,
                "two_turn": len(turns) == 2,
                "query_language": "en",
                "answer_language": "en",
                "question_class": "Q1" if partition == "acceptance" else "lookup",
                "difficulty": "simple",
            }
        )
        gold[partition].append(
            {
                "case_id": case_id,
                "gold_status": "supported",
                "required_claims": [
                    {
                        "claim_id": case_id + "-code",
                        "text": f"The code is {code}.",
                        "acceptable_evidence_sets": [[references[doc_id]]],
                    }
                ],
                "required_scope_doc_ids": [doc_id],
                "prohibited_inferences": [],
            }
        )
    for index in range(2, 20):
        case_id = f"accept-{index + 1:02d}"
        template = cases[2 + index % 2]
        case = json.loads(json.dumps(template))
        case["case_id"] = case_id
        case["question_class"] = f"Q{index // 4 + 1}"
        cases.append(case)
        expected = json.loads(json.dumps(gold["acceptance"][index % 2]))
        expected["case_id"] = case_id
        expected["required_claims"][0]["claim_id"] = case_id + "-code"
        gold["acceptance"].append(expected)
    extraction = {
        "corpus_id": manifest["corpus_id"],
        "schema_version": "1",
        "entries": [
            {
                "reference": references[doc_id],
                "source_sha256": source_hashes[doc_id],
                "outcome": "intact",
                "reviewed_by": "synthetic-e2e",
                "extracted_locators": [
                    {
                        "line_start": 1,
                        "text_sha256": hashlib.sha256(
                            references[doc_id]["quote"].encode()
                        ).hexdigest(),
                    }
                ],
            }
            for doc_id in ("doc-e2e-willow", "doc-e2e-cedar")
        ],
    }
    (directory / "gold").mkdir(exist_ok=True)
    for name, value in (
        ("manifest.json", manifest),
        ("questions.json", cases),
        ("gold/development.json", gold["development"]),
        ("gold/acceptance.json", gold["acceptance"]),
        ("extraction-map.json", extraction),
    ):
        (directory / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return directory, directory / "extraction-map.json"


def test_upload_parser_worker_and_delete_lifecycle(settings, store):
    provider = Provider(lambda payload: asyncio.sleep(0, result=answer(payload)))
    with client_for(settings, store, provider) as client:
        group = collection(client, settings)
        assert (group["source_count"], group["ready_count"], group["unavailable_count"]) == (
            0,
            0,
            0,
        )
        request_id = uuid.uuid4().hex
        first = upload(
            client,
            settings,
            group["id"],
            b"# Sample\nAlpha value 10 units\n",
            request_id=request_id,
        )
        assert first.status_code == 202, first.text
        source = first.json()
        pending_collection = store.get_collection(group["id"])
        assert pending_collection["revision"] == group["revision"] + 1
        assert (
            pending_collection["source_count"],
            pending_collection["ready_count"],
            pending_collection["unavailable_count"],
        ) == (1, 0, 1)
        duplicate = upload(
            client,
            settings,
            group["id"],
            b"# Sample\nAlpha value 10 units\n",
            request_id=request_id,
        )
        assert duplicate.status_code == 202 and duplicate.json()["id"] == source["id"]
        assert store.get_collection(group["id"])["revision"] == pending_collection["revision"]
        assert (
            upload(
                client, settings, group["id"], b"Changed text", request_id=request_id
            ).status_code
            == 409
        )
        stale = store.claim_job("stale-e2e-worker")
        with store.engine.begin() as connection:
            connection.execute(
                text("UPDATE jobs SET lease_until=now()-interval '1 second' WHERE id=:id"),
                {"id": stale["id"]},
            )
        fresh = store.claim_job("fresh-e2e-worker")
        assert fresh["id"] == stale["id"] and fresh["attempt"] == stale["attempt"] + 1
        with pytest.raises(ValueError, match="lease"):
            store.publish_source(
                source["id"],
                [],
                [],
                1,
                job_id=stale["id"],
                worker_id="stale-e2e-worker",
                attempt=stale["attempt"],
            )
        with store.engine.begin() as connection:
            connection.execute(
                text("UPDATE jobs SET lease_until=now()-interval '1 second' WHERE id=:id"),
                {"id": fresh["id"]},
            )
        ingest(store, settings)
        ready = client.get(f"/api/sources/{source['id']}", headers=service(settings)).json()
        assert ready["status"] == "ready" and ready["page_count"] == 1
        ready_collection = store.get_collection(group["id"])
        assert ready_collection["revision"] == pending_collection["revision"] + 1
        assert (
            ready_collection["source_count"],
            ready_collection["ready_count"],
            ready_collection["unavailable_count"],
        ) == (1, 1, 0)
        spans = client.get(f"/api/sources/{source['id']}/spans", headers=service(settings)).json()[
            "items"
        ]
        assert [item["line_start"] for item in spans] == [1, 2]
        first_span_page = client.get(
            f"/api/sources/{source['id']}/spans",
            params={"limit": 1},
            headers=service(settings),
        ).json()
        assert [item["id"] for item in first_span_page["items"]] == [spans[0]["id"]]
        assert first_span_page["next_cursor"] == spans[0]["id"]
        second_span_page = client.get(
            f"/api/sources/{source['id']}/spans",
            params={"limit": 1, "cursor": first_span_page["next_cursor"]},
            headers=service(settings),
        ).json()
        assert [item["id"] for item in second_span_page["items"]] == [spans[1]["id"]]
        assert second_span_page["next_cursor"] is None
        anchored_span_page = client.get(
            f"/api/sources/{source['id']}/spans",
            params={"limit": 1, "anchor": spans[1]["id"]},
            headers=service(settings),
        ).json()
        assert [item["id"] for item in anchored_span_page["items"]] == [spans[1]["id"]]
        assert (
            client.get(
                f"/api/sources/{source['id']}/spans",
                params={"cursor": spans[0]["id"], "anchor": spans[1]["id"]},
                headers=service(settings),
            ).status_code
            == 400
        )
        assert (
            client.get(f"/api/sources/{source['id']}/file", headers=service(settings)).content
            == b"# Sample\nAlpha value 10 units\n"
        )
        with store.engine.connect() as connection:
            variants = (
                connection.execute(
                    text("SELECT DISTINCT variant FROM chunks WHERE source_id=:id"),
                    {"id": source["id"]},
                )
                .scalars()
                .all()
            )
        assert set(variants) == {"fixed", "structural"}

        pdf = upload(client, settings, group["id"], pdf_bytes(), filename="native.pdf")
        assert pdf.status_code == 202, pdf.text
        ingest(store, settings)
        pdf_spans = client.get(
            f"/api/sources/{pdf.json()['id']}/spans", headers=service(settings)
        ).json()["items"]
        assert any(item["page"] == 1 and item["bbox"] for item in pdf_spans)
        assert any("screening interval is 21 days" in item["text"] for item in pdf_spans)
        assert (
            client.get(
                f"/api/sources/{pdf.json()['id']}/spans",
                params={"limit": 1, "cursor": first_span_page["next_cursor"]},
                headers=service(settings),
            ).status_code
            == 400
        )
        pdf_collection = store.get_collection(group["id"])
        assert pdf_collection["revision"] == ready_collection["revision"] + 2
        assert (
            pdf_collection["source_count"],
            pdf_collection["ready_count"],
            pdf_collection["unavailable_count"],
        ) == (2, 2, 0)

        async def guideline_answer(payload):
            evidence = next(
                item
                for item in payload["evidence"]["spans"]
                if "screening interval is 21 days" in item["text"]
            )
            return {
                "kind": "answer",
                "status": "supported",
                "answer": "The guideline states a 21-day screening interval after enrollment.",
                "claims": [
                    {
                        "text": "The guideline states a 21-day screening interval after enrollment.",
                        "evidence_ids": [evidence["id"]],
                    }
                ],
                "limitations": [],
                "coverage_requirement": "claims",
                "requests": [],
            }

        provider.respond = guideline_answer
        pdf_chat = conversation(client, settings, group["id"])
        cited_pdf = finished(
            client,
            settings,
            run(
                client,
                settings,
                pdf_chat["id"],
                "According to the guideline, what is the screening interval?",
            )["id"],
        )
        assert cited_pdf["status"] == "succeeded", store.audit(cited_pdf["id"])
        assert {item["source_id"] for item in cited_pdf["answer"]["citations"]} == {
            pdf.json()["id"]
        }

        invalid = upload(
            client, settings, group["id"], b"%PDF-1.4\nnot a valid PDF", filename="broken.pdf"
        )
        assert invalid.status_code == 202, invalid.text
        ingest(store, settings)
        assert (
            client.get(f"/api/sources/{invalid.json()['id']}", headers=service(settings)).json()[
                "status"
            ]
            == "failed"
        )
        failed_collection = store.get_collection(group["id"])
        assert failed_collection["revision"] == pdf_collection["revision"] + 2
        assert (
            failed_collection["source_count"],
            failed_collection["ready_count"],
            failed_collection["unavailable_count"],
        ) == (3, 2, 1)
        source_ids = []
        cursor = ""
        while True:
            response = client.get(
                f"/api/collections/{group['id']}/sources",
                params={"limit": 1, **({"cursor": cursor} if cursor else {})},
                headers=service(settings),
            )
            assert response.status_code == 200, response.text
            page = response.json()
            assert page["collection"]["id"] == group["id"]
            assert page["collection"]["source_count"] == 3
            assert len(page["items"]) == 1
            source_ids.append(page["items"][0]["id"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert len(source_ids) == len(set(source_ids)) == 3
        assert set(source_ids) == {source["id"], pdf.json()["id"], invalid.json()["id"]}

        deleted = client.delete(f"/api/sources/{source['id']}", headers=service(settings))
        assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
        deleted_collection = store.get_collection(group["id"])
        assert deleted_collection["revision"] == failed_collection["revision"] + 1
        assert (
            deleted_collection["source_count"],
            deleted_collection["ready_count"],
            deleted_collection["unavailable_count"],
        ) == (2, 1, 1)
        after_delete = client.get(
            f"/api/collections/{group['id']}/sources", headers=service(settings)
        ).json()
        assert after_delete["collection"]["source_count"] == 2
        assert {item["id"] for item in after_delete["items"]} == {
            pdf.json()["id"],
            invalid.json()["id"],
        }
        assert (
            client.get(f"/api/sources/{source['id']}/file", headers=service(settings)).status_code
            == 410
        )
        assert (
            client.get(f"/api/sources/{source['id']}/spans", headers=service(settings)).status_code
            == 410
        )
        ingest(store, settings)
        assert (
            client.get(f"/api/spans/{spans[0]['id']}", headers=service(settings)).status_code == 404
        )
        assert not (settings.data_dir / "uploads" / source["file_key"]).exists()


def test_real_mcp_search_read_answer_replay_and_export(settings, store):
    async def respond(payload):
        spans = payload["evidence"]["spans"]
        output = answer(payload)
        output["answer"] = "Alpha is recorded as 10 units and Beta as 14 units."
        output["claims"] = [
            {
                "text": "Alpha is recorded as 10 units and Beta as 14 units.",
                "evidence_ids": [
                    span["id"]
                    for span in spans
                    if "Alpha 10 units" in span["text"] or "Beta 14 units" in span["text"]
                ],
            }
        ]
        return output

    provider = Provider(respond)
    with client_for(settings, store, provider) as client:
        group = collection(client, settings)
        source = upload(client, settings, group["id"], b"Alpha 10 units\nBeta 14 units\n").json()
        ingest(store, settings)
        chat = conversation(client, settings, group["id"])
        question = "What amounts are recorded for Alpha and Beta?"
        stats_before = client.get("/api/stats", headers=service(settings)).json()
        result = finished(client, settings, run(client, settings, chat["id"], question)["id"])
        assert result["status"] == "succeeded", store.audit(result["id"])
        assert result["answer"]["answer"] == "Alpha is recorded as 10 units and Beta as 14 units."
        assert result["metrics"]["initial_method"] == "search_read"
        assert result["metrics"]["initial_receipt_complete"] is True
        audit = store.audit(result["id"])
        expected_operations = Telemetry(settings, store)._expected_operations(audit, result["id"])
        assert expected_operations == {
            "run": {f"{result['id']}:root"},
            "search_evidence": {f"{result['id']}:tool:1"},
            "read_evidence": {f"{result['id']}:tool:2"},
            "answer.generate": {f"{result['id']}:generation:1"},
        }
        stats_after = client.get("/api/stats", headers=service(settings)).json()
        assert set(stats_after) == {
            "total_runs",
            "latency_n",
            "latency_p50_ms",
            "runtime_status_counts",
            "answer_status_counts",
            "trace_status_counts",
            "terminal_trace_status_counts",
            "latency_measure",
            "observed_at",
        }
        assert stats_after["total_runs"] == stats_before["total_runs"] + 1
        assert stats_after["latency_n"] == stats_before["latency_n"] + 1
        assert stats_after["runtime_status_counts"]["succeeded"] == (
            stats_before["runtime_status_counts"]["succeeded"] + 1
        )
        assert stats_after["answer_status_counts"]["supported"] == (
            stats_before["answer_status_counts"]["supported"] + 1
        )
        assert sum(stats_after["trace_status_counts"].values()) == stats_after["total_runs"]
        assert {item["source_id"] for item in result["answer"]["citations"]} == {source["id"]}
        assert len(result["answer"]["citations"]) == 2
        assert len(provider.payloads) == 1
        expected_ids = {span["id"] for span in store.source_spans(source["id"])}
        for generation, payload in enumerate(provider.payloads, 1):
            evidence = payload["evidence"]
            assert evidence["receipt"]["complete"] is True
            assert evidence["receipt"]["provider_invocation_id"] == f"{result['id']}:{generation}"
            assert {span["id"] for span in evidence["spans"]} == expected_ids
            assert evidence["receipt"]["provider_input_packet_sha256"] == packet_hash(
                evidence["spans"]
            )
            assert (
                evidence["span_inventory_sha256"]
                == hashlib.sha256(
                    json.dumps(sorted(expected_ids), separators=(",", ":")).encode()
                ).hexdigest()
            )
        events = sse_entries(
            client.get(f"/api/runs/{result['id']}/events", headers=service(settings)).text
        )
        assert events[0] == (f"{result['id']}:1", "run.created")
        assert events[-1][1] == "answer.ready"
        replay = client.get(
            f"/api/runs/{result['id']}/events",
            headers={**service(settings), "Last-Event-ID": f"{result['id']}:1"},
        ).text
        replay_entries = sse_entries(replay)
        assert replay_entries[-1][1] == "answer.ready"
        assert all(int(event_id.rsplit(":", 1)[1]) > 1 for event_id, _ in replay_entries)
        assert all(event_id.startswith(result["id"] + ":") for event_id, _ in replay_entries)
        exported = client.get(f"/api/runs/{result['id']}/export", headers=service(settings))
        assert exported.status_code == 200
        assert exported.text.startswith(f"# {question}\n")
        assert "Alpha is recorded as 10 units and Beta as 14 units.\n\n## Sources" in exported.text
        assert f"facts.txt (line 1; source {source['id']}" in exported.text
        assert (
            client.get(f"/api/conversations/{chat['id']}", headers=service(settings)).json()[
                "messages"
            ][0]["role"]
            == "assistant"
        )

        provider.plan_respond = lambda payload: asyncio.sleep(
            0,
            result={
                "kind": "answer",
                "status": "needs_clarification",
                "answer": "Which object do you mean?",
                "claims": [],
                "limitations": [],
                "coverage_requirement": "claims",
                "requests": [],
            },
        )
        long_chat = conversation(client, settings, group["id"])
        long_question = "What is its code? " + "Unspecified context. " * 30
        clarification = finished(
            client,
            settings,
            run(client, settings, long_chat["id"], long_question)["id"],
        )
        assert clarification["status"] == "succeeded"
        assert clarification["answer"]["status"] == "needs_clarification"
        assert clarification["answer"]["citations"] == []
        assert clarification["metrics"]["model_calls"] == 1
        assert clarification["metrics"]["tool_calls"] == 0
        assert clarification["metrics"]["initial_method"] == "query_clarification"
        assert [
            payload["task_mode"]
            for request_id, payload in provider.calls
            if request_id == clarification["id"]
        ] == ["query_planning"]
        provider.plan_respond = None

        async def extend_complete_packet(payload):
            if not payload["final_only"]:
                return {
                    "kind": "request_tools",
                    "status": "partial",
                    "answer": "",
                    "claims": [],
                    "limitations": [],
                    "coverage_requirement": "claims",
                    "requests": [
                        {
                            "tool": "read_evidence",
                            "query": "",
                            "span_ids": [payload["evidence"]["spans"][0]["id"]],
                        }
                    ],
                }
            return answer(payload, coverage="complete_scope")

        provider.respond = extend_complete_packet
        replacement = finished(
            client,
            settings,
            run(client, settings, chat["id"], "List all values", variant="V2")["id"],
        )
        assert replacement["status"] == "succeeded"
        assert replacement["metrics"]["initial_receipt_complete"] is True
        assert replacement["metrics"]["final_receipt_complete"] is True
        assert replacement["metrics"]["guard_action"] == "none"
        assert replacement["answer"]["status"] == "supported"
        replacement_payloads = [
            payload for request_id, payload in provider.calls if request_id == replacement["id"]
        ]
        assert {
            span["id"] for span in replacement_payloads[-1]["evidence"]["spans"]
        } == expected_ids
        assert replacement_payloads[-1]["evidence"]["receipt"][
            "provider_input_packet_sha256"
        ] == packet_hash(replacement_payloads[-1]["evidence"]["spans"])

        mixed_group = collection(client, settings)
        mixed_sources = []
        for index in range(16):
            genre = ("Journal article", "Clinical guideline", "Patient note")[index % 3]
            marker = f"marker_{index:02d}"
            content = f"{genre}: {marker} describes observation {index}.\n".encode()
            mixed_sources.append(
                (
                    marker,
                    upload(
                        client,
                        settings,
                        mixed_group["id"],
                        content,
                        filename=f"source-{index:02d}.txt",
                    ).json()["id"],
                )
            )
        pdf_source = upload(
            client,
            settings,
            mixed_group["id"],
            pdf_bytes(),
            filename="guideline-en.pdf",
        ).json()
        ingest(store, settings)
        marker_spans = {
            store.source_spans(source_id)[0]["id"]: marker for marker, source_id in mixed_sources
        }
        mixed_chat = conversation(client, settings, mixed_group["id"])
        selected_markers = []

        async def two_distinct_searches(payload):
            if not payload["final_only"]:
                seen = {span["id"] for span in payload["evidence"]["spans"]}
                selected_markers[:] = [
                    marker for sid, marker in marker_spans.items() if sid not in seen
                ][:2]
                assert len(selected_markers) == 2
                return {
                    "kind": "request_tools",
                    "status": "partial",
                    "answer": "",
                    "claims": [],
                    "limitations": [],
                    "coverage_requirement": "claims",
                    "requests": [
                        {
                            "tool": "search_evidence",
                            "query": marker,
                            "span_ids": [],
                        }
                        for marker in selected_markers
                    ],
                }
            found = [
                span
                for span in payload["evidence"]["spans"]
                if any(marker in span["text"] for marker in selected_markers)
            ]
            return {
                "kind": "answer",
                "status": "supported",
                "answer": "The two requested source observations are documented.",
                "claims": [
                    {
                        "text": f"{marker} is documented.",
                        "evidence_ids": [
                            next(span["id"] for span in found if marker in span["text"])
                        ],
                    }
                    for marker in selected_markers
                ],
                "limitations": [],
                "coverage_requirement": "claims",
                "requests": [],
            }

        provider.respond = two_distinct_searches
        mixed_run = finished(
            client,
            settings,
            run(
                client,
                settings,
                mixed_chat["id"],
                "What monitoring observations appear?",
                variant="V3",
            )["id"],
        )
        assert mixed_run["status"] == "succeeded", store.audit(mixed_run["id"])
        assert mixed_run["metrics"]["tool_calls"] == 5
        assert mixed_run["metrics"]["model_calls"] == 2
        assert mixed_run["answer"]["status"] == "supported"
        assert {
            marker
            for span in mixed_run["answer"]["citations"]
            for marker in selected_markers
            if marker in span["text"]
        } == set(selected_markers)
        assert store.get_source(pdf_source["id"])["status"] == "ready"
        assert mixed_run["scope"]["collection_id"] == mixed_group["id"]
        assert mixed_run["scope"]["source_count"] == 17
        assert mixed_run["scope"]["ready_count"] == 17
        assert mixed_run["scope"]["unavailable_count"] == 0
        mixed_payloads = [
            payload for request_id, payload in provider.calls if request_id == mixed_run["id"]
        ]
        assert mixed_payloads[-1]["evidence"]["retrieval_queries"] == selected_markers
        assert mixed_payloads[-1]["evidence"]["receipt"][
            "provider_input_packet_sha256"
        ] == packet_hash(mixed_payloads[-1]["evidence"]["spans"])

        excerpt_order = []

        async def overbroad_summary(payload):
            chosen = sorted(
                payload["evidence"]["spans"], key=lambda item: item["id"], reverse=True
            )[:4]
            excerpt_order[:] = [item["id"] for item in chosen]
            return {
                "kind": "answer",
                "status": "supported",
                "answer": "These are every observation in the collection.",
                "claims": [
                    {"text": f"Observation {index} is documented.", "evidence_ids": [item["id"]]}
                    for index, item in enumerate(chosen)
                ],
                "limitations": [],
                "coverage_requirement": "claims",
                "requests": [],
            }

        provider.respond = overbroad_summary
        broad_chat = conversation(client, settings, mixed_group["id"])
        bounded = finished(
            client,
            settings,
            run(client, settings, broad_chat["id"], "List all observations in the collection")[
                "id"
            ],
        )
        assert bounded["status"] == "succeeded"
        assert bounded["answer"]["status"] == "partial"
        assert [claim["evidence_ids"][0] for claim in bounded["answer"]["claims"]] == excerpt_order[
            :3
        ]
        assert "These are every observation" not in bounded["answer"]["answer"]

        original_call = client.app.state.mcp.call
        failed_chat = conversation(client, settings, group["id"])

        async def failed_search(name, arguments):
            if name == "search_evidence":
                raise MCPToolFailure("stale_source")
            return await original_call(name, arguments)

        client.app.state.mcp.call = failed_search
        try:
            failed_tool = finished(
                client,
                settings,
                run(client, settings, failed_chat["id"], "What is the alpha value?")["id"],
            )
        finally:
            client.app.state.mcp.call = original_call
        assert failed_tool["status"] == "failed" and failed_tool["answer"] is None
        assert not any(request_id == failed_tool["id"] for request_id, _ in provider.calls)
        assert "stale_source" in str(store.audit(failed_tool["id"]))


def test_scope_guards_foreign_citation_and_exhaustive_partial(settings, store):
    async def foreign(payload):
        output = answer(payload)
        output["claims"][0]["evidence_ids"] = [foreign_span]
        return output

    provider = Provider(foreign)
    with client_for(settings, store, provider) as client:
        own = collection(client, settings)
        other = collection(client, settings)
        own_source = upload(client, settings, own["id"], b"Alpha value 10 units\n").json()
        ingest(store, settings)
        other_source = upload(client, settings, other["id"], b"Foreign value 99 units\n").json()
        ingest(store, settings)
        foreign_span = store.source_spans(other_source["id"])[0]["id"]
        chat = conversation(client, settings, own["id"])
        bad = finished(client, settings, run(client, settings, chat["id"])["id"])
        assert bad["status"] == "failed" and bad["answer"] is None
        assert all(item["role"] != "assistant" for item in store.messages_page(chat["id"]))
        provider.respond = lambda payload: asyncio.sleep(0, result=answer(payload))
        long_text = b"Pending value 20 units " + b"marker " * 1400 + b"\n"
        pending = upload(client, settings, own["id"], long_text).json()
        assert pending["status"] == "pending"
        own_page = client.get(
            f"/api/collections/{own['id']}/sources",
            params={"limit": 1},
            headers=service(settings),
        ).json()
        assert own_page["collection"]["source_count"] == 2
        assert len(own_page["items"]) == 1 and own_page["next_cursor"]
        assert (
            client.get(
                f"/api/collections/{other['id']}/sources",
                params={"limit": 1, "cursor": own_page["next_cursor"]},
                headers=service(settings),
            ).status_code
            == 400
        )
        partial = finished(
            client,
            settings,
            run(
                client,
                settings,
                chat["id"],
                question="List all alpha values",
            )["id"],
        )
        assert partial["status"] == "succeeded"
        assert partial["answer"]["status"] == "partial"
        assert partial["answer"]["coverage"]["complete"] is False
        assert partial["answer"]["citations"]
        assert partial["answer"]["answer"] == "\n".join(
            claim["text"] for claim in partial["answer"]["claims"]
        )
        assert any(
            "smaller set to a new collection" in limitation
            for limitation in partial["answer"]["limitations"]
        )
        assert set(partial["scope"]) == {
            "id",
            "collection_id",
            "revision",
            "source_count",
            "ready_count",
            "unavailable_count",
            "snapshot_hash",
        }
        assert partial["scope"]["id"] == partial["id"]
        assert partial["scope"]["collection_id"] == own["id"]
        assert partial["scope"]["source_count"] == 2
        assert partial["scope"]["ready_count"] == 1
        assert partial["scope"]["unavailable_count"] == 1
        assert (
            partial["scope"]["snapshot_hash"]
            == hashlib.sha256(
                json.dumps(
                    {
                        key: partial["scope"][key]
                        for key in (
                            "collection_id",
                            "revision",
                            "source_count",
                            "ready_count",
                            "unavailable_count",
                        )
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        assert store.get_source(own_source["id"])["status"] == "ready"
        assert store.get_source(pending["id"])["status"] == "pending"

        async def partial_global_summary(payload):
            item = payload["evidence"]["spans"][0]
            return {
                "kind": "answer",
                "status": "partial",
                "answer": "These are all values in the library.",
                "claims": [
                    {"text": "Alpha value 10 units is recorded.", "evidence_ids": [item["id"]]}
                ],
                "limitations": ["Some documents were unavailable."],
                "coverage_requirement": "claims",
                "requests": [],
            }

        provider.respond = partial_global_summary
        cited_partial = finished(
            client,
            settings,
            run(client, settings, chat["id"], question="Show all recorded values")["id"],
        )
        assert cited_partial["status"] == "succeeded"
        assert cited_partial["answer"]["status"] == "partial"
        assert cited_partial["answer"]["claims"][0]["text"] == "Alpha value 10 units is recorded."
        assert cited_partial["answer"]["citations"]
        assert "These are all values" not in cited_partial["answer"]["answer"]
        provider.respond = lambda payload: asyncio.sleep(0, result=answer(payload))
        pending_chat = conversation(client, settings, own["id"])
        pending_scope = store.create_run(
            pending_chat["id"], uuid.uuid4().hex, "List all values", "gpt-6-sol", "V0"
        )
        pending_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {
                "run_scope_id": pending_scope["id"],
                "span_ids": [store.source_spans(own_source["id"])[0]["id"]],
            },
        )
        assert pending_read["inventory_complete"] is False

        async def undocumented(_):
            return {
                "kind": "answer",
                "status": "not_documented",
                "answer": "No puncture is documented in the collection.",
                "claims": [],
                "limitations": ["This applies only to the current collection."],
                "coverage_requirement": "complete_scope",
                "requests": [],
            }

        provider.respond = undocumented
        incomplete_absence = finished(
            client,
            settings,
            run(client, settings, chat["id"], question="List all documented punctures")["id"],
        )
        assert incomplete_absence["answer"]["status"] == "partial"
        assert incomplete_absence["answer"]["coverage"]["complete"] is False

        async def omission_in_limitations(_):
            return {
                "kind": "answer",
                "status": "partial",
                "answer": "The available information is incomplete.",
                "claims": [],
                "limitations": ["No records of puncture are present in the collection."],
                "coverage_requirement": "claims",
                "requests": [],
            }

        provider.respond = omission_in_limitations
        guarded_limitation = finished(
            client,
            settings,
            run(client, settings, chat["id"], question="What was the alpha value?")["id"],
        )
        assert guarded_limitation["answer"]["status"] == "partial"
        assert guarded_limitation["metrics"]["guard_action"] == "none"
        assert guarded_limitation["answer"]["limitations"] == [
            "No records of puncture are present in the collection."
        ]
        ingest(store, settings)
        newly_ready_ids = [
            span["id"]
            for source_id in (own_source["id"], pending["id"])
            for span in store.source_spans(source_id)
        ]
        with pytest.raises(MCPToolFailure, match="scope_expired"):
            client.portal.call(
                client.app.state.mcp.call,
                "read_evidence",
                {"run_scope_id": pending_scope["id"], "span_ids": newly_ready_ids},
            )
        store.cancel_run(pending_scope["id"])
        ready_scope = store.create_run(
            chat["id"], uuid.uuid4().hex, "List all values", "gpt-6-sol", "V0"
        )
        ready_ids = [
            span["id"]
            for source_id in (own_source["id"], pending["id"])
            for span in store.source_spans(source_id)
        ]
        subset = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": ready_scope["id"], "span_ids": ready_ids[:1]},
        )
        assert subset["inventory_complete"] is False
        clipped = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": ready_scope["id"], "span_ids": ready_ids, "budget_bytes": 5},
        )
        assert clipped["inventory_complete"] is False
        assert clipped["truncated"] is True
        complete_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": ready_scope["id"], "span_ids": ready_ids},
        )
        assert complete_read["inventory_complete"] is True
        assert complete_read["span_inventory_count"] == len(ready_ids)
        assert {span["id"] for span in complete_read["spans"]} == set(ready_ids)
        store.cancel_run(ready_scope["id"])
        provider.respond = undocumented
        complete_absence = finished(
            client,
            settings,
            run(client, settings, chat["id"], question="List all documented punctures")["id"],
        )
        assert complete_absence["answer"]["status"] == "not_documented"
        assert complete_absence["answer"]["claims"] == []
        assert complete_absence["answer"]["citations"] == []
        assert complete_absence["answer"]["coverage"]["complete"] is True

        async def uncited_absence(payload):
            result = await undocumented(payload)
            result["claims"] = [{"text": "Puncture is not documented.", "evidence_ids": []}]
            return result

        provider.respond = uncited_absence
        invalid_absence = finished(
            client,
            settings,
            run(client, settings, chat["id"], question="List all documented punctures")["id"],
        )
        assert invalid_absence["status"] == "failed"
        assert invalid_absence["answer"] is None
        provider.respond = lambda payload: asyncio.sleep(0, result=answer(payload))
        complete = finished(
            client,
            settings,
            run(client, settings, chat["id"], question="List all alpha values after ingestion")[
                "id"
            ],
        )
        assert complete["status"] == "succeeded"
        assert complete["answer"]["coverage"]["complete"] is True
        assert set(complete["answer"]["coverage"]["delivered_source_ids"]) == {
            own_source["id"],
            pending["id"],
        }
        delivered = [
            payload["evidence"]["spans"]
            for request_id, payload in provider.calls
            if request_id == complete["id"]
        ][-1]
        pending_fragments = store.source_spans(pending["id"])
        delivered_pending = {
            span["id"]: span["text"] for span in delivered if span["source_id"] == pending["id"]
        }
        assert delivered_pending == {span["id"]: span["text"] for span in pending_fragments}
        assert "".join(delivered_pending[span["id"]] for span in pending_fragments) == (
            long_text.decode().strip()
        )
        queued = store.create_run(
            chat["id"], uuid.uuid4().hex, "List all values", "gpt-6-sol", "V3"
        )
        first_page = client.portal.call(
            client.app.state.mcp.call,
            "collect_scope",
            {"run_scope_id": queued["id"], "page_bytes": 8192},
        )
        assert first_page["next_cursor"] and first_page["text_bytes"] <= 8192
        cursor = first_page["next_cursor"]
        for _ in range(3):
            next_page = client.portal.call(
                client.app.state.mcp.call,
                "collect_scope",
                {"run_scope_id": queued["id"], "cursor": cursor, "page_bytes": 8192},
            )
            assert next_page["spans"] and next_page["text_bytes"] <= 8192
            cursor = next_page["next_cursor"]
            if not cursor:
                break
        assert cursor == ""
        with pytest.raises(MCPToolFailure, match="unknown_span"):
            client.portal.call(
                client.app.state.mcp.call,
                "read_evidence",
                {"run_scope_id": queued["id"], "span_ids": [foreign_span]},
            )
        with pytest.raises(MCPToolFailure):
            client.portal.call(
                client.app.state.mcp.call,
                "collect_scope",
                {
                    "run_scope_id": queued["id"],
                    "cursor": first_page["next_cursor"],
                    "page_bytes": 4096,
                },
            )
        with pytest.raises(MCPToolFailure):
            client.portal.call(
                client.app.state.mcp.call,
                "collect_scope",
                {"run_scope_id": queued["id"], "cursor": first_page["next_cursor"] + "x"},
            )
        store.cancel_run(queued["id"])

        many_group = collection(client, settings)
        many_source = upload(
            client,
            settings,
            many_group["id"],
            b"".join(f"Row {index:02d}\n".encode() for index in range(65)),
        ).json()
        ingest(store, settings)
        many_chat = conversation(client, settings, many_group["id"])
        many_scope = store.create_run(
            many_chat["id"], uuid.uuid4().hex, "List all rows", "gpt-6-sol", "V0"
        )
        many_ids = [span["id"] for span in store.source_spans(many_source["id"])]
        assert len(many_ids) == 65
        many_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": many_scope["id"], "span_ids": many_ids[:64]},
        )
        assert many_read["inventory_complete"] is False
        assert many_read["span_inventory_count"] is None
        boundary_group = collection(client, settings)
        boundary_source = upload(
            client,
            settings,
            boundary_group["id"],
            b"".join(f"Boundary {index:02d}\n".encode() for index in range(64)),
        ).json()
        ingest(store, settings)
        boundary_chat = conversation(client, settings, boundary_group["id"])
        boundary_scope = store.create_run(
            boundary_chat["id"], uuid.uuid4().hex, "List all rows", "gpt-6-sol", "V0"
        )
        boundary_ids = [span["id"] for span in store.source_spans(boundary_source["id"])]
        assert len(boundary_ids) == 64
        boundary_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": boundary_scope["id"], "span_ids": boundary_ids},
        )
        assert boundary_read["inventory_complete"] is True
        assert boundary_read["span_inventory_count"] == 64
        store.cancel_run(boundary_scope["id"])

        utf_group = collection(client, settings)
        utf_source = upload(client, settings, utf_group["id"], "Ä value 10 units\n".encode()).json()
        ingest(store, settings)
        utf_chat = conversation(client, settings, utf_group["id"])
        utf_scope = store.create_run(
            utf_chat["id"], uuid.uuid4().hex, "List all text", "gpt-6-sol", "V0"
        )
        utf_id = store.source_spans(utf_source["id"])[0]["id"]
        exact_utf_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": utf_scope["id"], "span_ids": [utf_id], "budget_bytes": 17},
        )
        assert exact_utf_read["inventory_complete"] is True
        assert exact_utf_read["text_bytes"] == 17
        overflow_utf_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": utf_scope["id"], "span_ids": [utf_id], "budget_bytes": 16},
        )
        assert overflow_utf_read["inventory_complete"] is False
        store.cancel_run(utf_scope["id"])

        large_group = collection(client, settings)
        large_source = upload(
            client, settings, large_group["id"], b"a" * 9000 + b"\n" + b"b" * 9000 + b"\n"
        ).json()
        ingest(store, settings)
        large_chat = conversation(client, settings, large_group["id"])
        large_scope = store.create_run(
            large_chat["id"], uuid.uuid4().hex, "List all text", "gpt-6-sol", "V0"
        )
        large_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {
                "run_scope_id": large_scope["id"],
                "span_ids": [span["id"] for span in store.source_spans(large_source["id"])],
            },
        )
        assert large_read["inventory_complete"] is False
        assert large_read["span_inventory_count"] == len(store.source_spans(large_source["id"]))
        assert large_read["truncated"] is True
        store.cancel_run(large_scope["id"])
        store.cancel_run(many_scope["id"])


def test_cancel_late_publish_retry_and_frozen_scope(settings, store):
    async def returns_after_cancel(payload):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return answer(payload)

    provider = Provider(returns_after_cancel)
    with client_for(settings, store, provider) as client:
        group = collection(client, settings)
        first = upload(client, settings, group["id"], b"Alpha value 10 units\n").json()
        ingest(store, settings)
        chat = conversation(client, settings, group["id"])
        active = run(client, settings, chat["id"])
        assert provider.entered.wait(15)
        assert active["scope"]["collection_id"] == group["id"]
        assert active["scope"]["source_count"] == 1
        assert active["scope"]["ready_count"] == 1
        assert active["scope"]["unavailable_count"] == 0
        assert store.run_scope_state(active["id"]) == "running"
        later = upload(client, settings, group["id"], b"Later value 40 units\n").json()
        assert store.run_scope_state(active["id"]) == "scope_expired"
        assert store.get_source(later["id"])["status"] == "pending"
        assert store.get_collection(group["id"])["source_count"] == 2
        cancelled = client.post(
            f"/api/runs/{active['id']}/cancel", headers=service(settings)
        ).json()
        assert cancelled["status"] == "cancelled"
        assert provider.completed.wait(15)
        execution_stopped(client, active["id"])
        assert store.finish_run(active["id"], {"answer": "late"}, {}) is False
        assert store.get_run(active["id"])["answer"] is None
        assert all(message["role"] != "assistant" for message in store.messages_page(chat["id"]))
        assert active["id"] in provider.cancelled
        provider.respond = lambda payload: asyncio.sleep(0, result=answer(payload))
        retried = finished(
            client,
            settings,
            run(client, settings, chat["id"], retry_of_run_id=active["id"])["id"],
        )
        assert retried["status"] == "succeeded" and retried["retry_of_run_id"] == active["id"]
        assert retried["scope"]["collection_id"] == group["id"]
        assert retried["scope"]["source_count"] == 2
        assert retried["scope"]["ready_count"] == 1
        assert retried["scope"]["unavailable_count"] == 1
        assert retried["scope"]["revision"] > active["scope"]["revision"]
        corrected = finished(
            client,
            settings,
            run(client, settings, chat["id"], retry_of_run_id=retried["id"])["id"],
        )
        assert corrected["status"] == "succeeded"
        corrected_payloads = [
            payload for request_id, payload in provider.calls if request_id == corrected["id"]
        ]
        assert len(corrected_payloads) == 1
        assert corrected_payloads[0]["task_mode"] == "answer"
        assert corrected_payloads[0]["history"] == []
        preserved_chat = conversation(client, settings, group["id"])
        preserved = store.create_run(
            preserved_chat["id"], uuid.uuid4().hex, "What is in scope?", "gpt-6-sol", "V3"
        )
        assert store.start_run(preserved["id"])

        release_after_delete = asyncio.Event()

        async def blocked_until_delete(payload):
            await release_after_delete.wait()
            return answer(payload)

        provider.respond = blocked_until_delete
        provider.entered.clear()
        provider.completed.clear()
        deletion_run = run(client, settings, chat["id"])
        assert deletion_run["scope"]["source_count"] == 2
        assert deletion_run["scope"]["ready_count"] == 1
        assert deletion_run["scope"]["unavailable_count"] == 1
        assert provider.entered.wait(15)
        deleted = client.delete(f"/api/sources/{first['id']}", headers=service(settings))
        assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
        client.portal.call(release_after_delete.set)
        assert provider.completed.wait(15)
        failed = finished(client, settings, deletion_run["id"])
        assert failed["status"] == "failed" and failed["answer"] is None
        assert failed["error"]["code"] == "scope_expired"
        assert any(
            item["stage"] == "run.exception" and "scope_expired" in item["payload"]["message"]
            for item in store.audit(deletion_run["id"])
        )
        assert all(
            message["role"] != "assistant"
            for message in store.messages_page(chat["id"])
            if message["run_id"] == deletion_run["id"]
        )
        assert store.finish_run(preserved["id"], {"answer": "deleted"}, {}) is False
        assert store.get_run(preserved["id"])["error"] == {"code": "scope_expired"}

        race_group = collection(client, settings)
        race_chat = conversation(client, settings, race_group["id"])
        stale = store.create_run(
            race_chat["id"], uuid.uuid4().hex, "What is in scope?", "gpt-6-sol", "V3"
        )
        assert store.start_run(stale["id"])
        assert stale["scope"]["source_count"] == 0
        assert stale["scope"]["ready_count"] == 0
        assert stale["scope"]["unavailable_count"] == 0
        added = upload(client, settings, race_group["id"], b"Added after scope freeze\n")
        assert added.status_code == 202
        assert store.get_collection(race_group["id"])["unavailable_count"] == 1
        assert store.run_scope_state(stale["id"]) == "scope_expired"
        assert store.finish_run(stale["id"], {"answer": "stale"}, {}) is False
        stale_result = store.get_run(stale["id"])
        assert stale_result["status"] == "failed"
        assert stale_result["error"] == {"code": "scope_expired"}
        assert stale_result["answer"] is None
        assert not any(
            message["role"] == "assistant" and message["run_id"] == stale["id"]
            for message in store.messages_page(race_chat["id"])
        )

        ahead = store.create_run(
            race_chat["id"], uuid.uuid4().hex, "What is in scope?", "gpt-6-sol", "V3"
        )
        assert store.start_run(ahead["id"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            with store.engine.begin() as mutation:
                mutation.execute(
                    text("UPDATE collections SET revision=revision+1 WHERE id=:id"),
                    {"id": race_group["id"]},
                )
                pending_finish = pool.submit(store.finish_run, ahead["id"], {"answer": "stale"}, {})
                with pytest.raises(TimeoutError):
                    pending_finish.result(timeout=0.2)
            assert pending_finish.result(timeout=10) is False
        assert store.get_run(ahead["id"])["error"] == {"code": "scope_expired"}

        behind = store.create_run(
            race_chat["id"], uuid.uuid4().hex, "What is in scope?", "gpt-6-sol", "V3"
        )
        assert store.start_run(behind["id"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            with store.engine.begin() as conversation_lock:
                conversation_lock.execute(
                    text("SELECT id FROM conversations WHERE id=:id FOR UPDATE"),
                    {"id": race_chat["id"]},
                )
                publishing = pool.submit(store.finish_run, behind["id"], {"answer": "current"}, {})
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    with store.engine.connect() as observer:
                        waiting = observer.execute(
                            text(
                                "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                                "WHERE datname=current_database() AND wait_event_type='Lock' "
                                "AND (query LIKE 'INSERT INTO messages%' "
                                "OR query LIKE 'UPDATE conversations SET revision=revision+1%'))"
                            )
                        ).scalar_one()
                    if waiting:
                        break
                    time.sleep(0.02)
                assert waiting
                with store.engine.connect() as attempted_mutation:
                    attempted_mutation.execute(text("SET LOCAL lock_timeout='200ms'"))
                    with pytest.raises(OperationalError, match="lock timeout"):
                        attempted_mutation.execute(
                            text("UPDATE collections SET revision=revision+1 WHERE id=:id"),
                            {"id": race_group["id"]},
                        )
            assert publishing.result(timeout=10) is True
        assert store.get_run(behind["id"])["status"] == "succeeded"

        store.migrate()

        succeeded_ids = [str(uuid.UUID(int=(1 << 128) - index)) for index in range(1, 126)]
        failed_ids = [str(uuid.UUID(int=index)) for index in range(1, 4)]
        page_ids = succeeded_ids + list(reversed(failed_ids))
        with store.engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO runs(id,conversation_id,request_id,question,model,variant,status,scope,created_at) "
                    "VALUES (:id,:conversation_id,:request_id,'Pagination probe','gpt-6-sol','V3',:status,CAST(:scope AS jsonb),"
                    "TIMESTAMPTZ '2099-01-01 00:00:00+00')"
                ),
                [
                    {
                        "id": run_id,
                        "conversation_id": race_chat["id"],
                        "request_id": f"page-{run_id}",
                        "status": "succeeded" if run_id in succeeded_ids else "failed",
                        "scope": json.dumps(behind["scope"]),
                    }
                    for run_id in page_ids
                ],
            )
        first_page_response = client.get("/api/runs?limit=100", headers=service(settings))
        assert first_page_response.status_code == 200, first_page_response.text
        first_page = first_page_response.json()
        assert len(first_page["items"]) == 100
        assert [row["id"] for row in first_page["items"]] == succeeded_ids[:100]
        assert isinstance(first_page["next_cursor"], str)
        with store.engine.begin() as connection:
            newer_id = str(uuid.uuid4())
            connection.execute(
                text(
                    "INSERT INTO runs(id,conversation_id,request_id,question,model,variant,status,scope,created_at) "
                    "VALUES (:id,:conversation_id,:request_id,'Newer pagination probe','gpt-6-sol','V3','failed',"
                    "CAST(:scope AS jsonb),TIMESTAMPTZ '2099-01-02 00:00:00+00')"
                ),
                {
                    "id": newer_id,
                    "conversation_id": race_chat["id"],
                    "request_id": f"page-{newer_id}",
                    "scope": json.dumps(behind["scope"]),
                },
            )
        next_page_response = client.get(
            "/api/runs",
            params={"limit": 100, "cursor": first_page["next_cursor"]},
            headers=service(settings),
        )
        assert next_page_response.status_code == 200, next_page_response.text
        next_page = next_page_response.json()
        combined_ids = [row["id"] for row in first_page["items"] + next_page["items"]]
        assert combined_ids[: len(page_ids)] == page_ids
        assert len(combined_ids) == len(set(combined_ids))
        assert newer_id not in combined_ids
        assert (
            client.get("/api/runs?limit=1", headers=service(settings)).json()["items"][0]["id"]
            == newer_id
        )
        failed_page = client.get(
            "/api/runs", params={"status": "failed", "limit": 2}, headers=service(settings)
        ).json()
        assert [row["id"] for row in failed_page["items"]] == [newer_id, failed_ids[-1]]
        assert failed_page["next_cursor"]
        filtered_next = client.get(
            "/api/runs",
            params={"status": "failed", "limit": 2, "cursor": failed_page["next_cursor"]},
            headers=service(settings),
        ).json()
        assert [row["id"] for row in filtered_next["items"][:2]] == failed_ids[-2::-1]
        assert (
            client.get(
                "/api/runs",
                params={"status": "failed", "cursor": first_page["next_cursor"]},
                headers=service(settings),
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/api/runs", params={"cursor": "not*a*cursor"}, headers=service(settings)
            ).status_code
            == 400
        )
        invalid_payload = json.loads(base64.urlsafe_b64decode(first_page["next_cursor"] + "=="))
        invalid_payload["id"] = 1
        invalid_cursor = (
            base64.urlsafe_b64encode(
                json.dumps(invalid_payload, sort_keys=True, separators=(",", ":")).encode()
            )
            .decode()
            .rstrip("=")
        )
        assert (
            client.get(
                "/api/runs", params={"cursor": invalid_cursor}, headers=service(settings)
            ).status_code
            == 400
        )
        assert client.get("/api/runs?status=unknown", headers=service(settings)).status_code == 400
        assert client.get("/api/runs?limit=101", headers=service(settings)).status_code == 400


def test_browser_auth_host_origin_csrf_and_mcp_scope_guards(settings, store):
    provider = Provider(lambda payload: asyncio.sleep(0, result=answer(payload)))
    with client_for(settings, store, provider) as client:
        assert client.get("/api/collections").status_code == 401
        assert (
            client.post("/api/session", json={"token": "wrong-token-0123456789"}).status_code == 401
        )
        session = client.post("/api/session", json={"token": settings.launch_token})
        assert session.status_code == 200 and "httponly" in session.headers["set-cookie"].lower()
        assert client.get("/api/session").json()["authenticated"] is True
        assert client.post("/api/collections", json={"title": "Denied"}).status_code == 403
        csrf = session.json()["csrf_token"]
        assert (
            client.post(
                "/api/collections", json={"title": "Allowed"}, headers={"X-CSRF-Token": csrf}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/api/collections",
                json={"title": "Bad Origin"},
                headers={"X-CSRF-Token": csrf, "Origin": "http://evil.invalid"},
            ).status_code
            == 403
        )
        assert client.get("/api/collections", headers={"Host": "evil.invalid"}).status_code == 403
        assert (
            client.get(
                "/api/collections", headers={"Authorization": "Bearer incorrect"}
            ).status_code
            == 401
        )
        assert client.get("/api/collections", headers=service(settings)).status_code == 200
        before = len(client.get("/api/collections", headers=service(settings)).json()["items"])
        oversized_json = b'{"title":"' + b"A" * 70000 + b'"}'
        response = client.post(
            "/api/collections",
            content=iter([oversized_json[:200], oversized_json[200:]]),
            headers={
                **service(settings),
                "Content-Type": "application/json",
                "Content-Length": "100",
            },
        )
        assert response.status_code == 413
        assert (
            len(client.get("/api/collections", headers=service(settings)).json()["items"]) == before
        )

        group = collection(client, settings)
        boundary = "e2e-boundary"
        oversized_multipart = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="request_id"\r\n\r\n'
                f"{uuid.uuid4().hex}\r\n"
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="large.txt"\r\n'
                "Content-Type: text/plain\r\n\r\n"
            ).encode()
            + b"A" * (settings.max_upload_bytes + 1024 * 1024)
            + f"\r\n--{boundary}--\r\n".encode()
        )
        rejected_upload = client.post(
            f"/api/collections/{group['id']}/sources",
            content=iter([oversized_multipart[:500], oversized_multipart[500:]]),
            headers={
                **service(settings),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": "500",
            },
        )
        assert rejected_upload.status_code == 413
        assert (
            client.get(f"/api/collections/{group['id']}/sources", headers=service(settings)).json()[
                "items"
            ]
            == []
        )
        source = upload(
            client,
            settings,
            group["id"],
            b"Alpha is recorded as 5 mg.\nBeta is recorded as 6 mmol/L.\nTrial NCT04277731 reports these values.\n",
        ).json()
        filename_source = upload(
            client,
            settings,
            group["id"],
            b"The registry entry describes enrollment windows.\n",
            filename="registry-NCT05551234.txt",
        ).json()
        title_source = upload(
            client,
            settings,
            group["id"],
            b"The study reports an enrollment window.\n",
            filename="study-metadata.txt",
        ).json()
        with store.engine.begin() as connection:
            changed = connection.execute(
                text("UPDATE sources SET title=:title WHERE id=:id AND status='pending'"),
                {"title": "Study DOI 10.1234/ABCD", "id": title_source["id"]},
            )
            assert changed.rowcount == 1
        foreign_group = collection(client, settings)
        foreign_source = upload(
            client,
            settings,
            foreign_group["id"],
            b"Alpha is recorded as 5 mg.\nTrial NCT04277731 belongs to another collection.\n",
        ).json()
        ingest(store, settings)
        chat = conversation(client, settings, group["id"])
        running = store.create_run(
            chat["id"], uuid.uuid4().hex, "What was the alpha value?", "gpt-6-sol", "V3"
        )
        span_ids = [span["id"] for span in store.source_spans(source["id"])]
        candidates = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {"run_scope_id": running["id"], "query": "Alpha 5 mg", "variant": "V3"},
        )
        assert candidates["scope_applied"] == running["id"]
        assert any(span_ids[0] in item["span_ids"] for item in candidates["candidates"])
        assert {item["source_id"] for item in candidates["candidates"]} <= {
            source["id"],
            filename_source["id"],
            title_source["id"],
        }
        assert foreign_source["id"] not in {item["source_id"] for item in candidates["candidates"]}
        assert candidates["retrieval"]["dense_candidates"] > 0
        assert candidates["retrieval"]["bm25_candidates"] > 0
        assert candidates["retrieval"]["rrf_pool_size"] <= 64
        assert candidates["retrieval"]["index_scope"] == "collection_partition"
        assert candidates["rerank"]["status"] == "ok"
        assert candidates["rerank"]["input_count"] <= 64
        assert candidates["rerank"]["scored_pairs"] == candidates["rerank"]["input_count"]
        dense_only = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {"run_scope_id": running["id"], "query": "Alpha 5 mg", "variant": "V0"},
        )
        assert dense_only["retrieval"]["dense_candidates"] > 0
        assert dense_only["retrieval"]["bm25_candidates"] == 0
        assert all(item["branches"] == ["dense"] for item in dense_only["candidates"])
        literal = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {
                "run_scope_id": running["id"],
                "query": "NCT04277731 unrelatedzyxword",
                "variant": "V1",
            },
        )
        assert literal["retrieval"]["literal_candidates"] > 0
        assert any(
            item["source_id"] == source["id"] and "literal" in item["branches"]
            for item in literal["candidates"]
        )
        assert foreign_source["id"] not in {item["source_id"] for item in literal["candidates"]}
        filename_literal = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {
                "run_scope_id": running["id"],
                "query": "What does registry NCT05551234 report?",
                "variant": "V1",
            },
        )
        assert any(
            item["source_id"] == filename_source["id"] and "literal" in item["branches"]
            for item in filename_literal["candidates"]
        )
        assert all(
            "NCT05551234" not in span["text"] for span in store.source_spans(filename_source["id"])
        )
        title_literal = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {
                "run_scope_id": running["id"],
                "query": "What does 10.1234/ABCD report?",
                "variant": "V1",
            },
        )
        assert any(
            item["source_id"] == title_source["id"] and "literal" in item["branches"]
            for item in title_literal["candidates"]
        )
        assert all(
            "10.1234/ABCD" not in span["text"] for span in store.source_spans(title_source["id"])
        )
        lexical_or = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {"run_scope_id": running["id"], "query": "Alpha unrelatedzyxword", "variant": "V1"},
        )
        assert lexical_or["retrieval"]["bm25_candidates"] > 0
        assert any("bm25" in item["branches"] for item in lexical_or["candidates"])
        phrase = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {
                "run_scope_id": running["id"],
                "query": '"Alpha is recorded as 5 mg"',
                "variant": "V1",
            },
        )
        assert phrase["retrieval"]["phrase_candidates"] > 0
        assert any(
            item["source_id"] == source["id"] and "phrase" in item["branches"]
            for item in phrase["candidates"]
        )
        assert foreign_source["id"] not in {item["source_id"] for item in phrase["candidates"]}
        read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": running["id"], "span_ids": [span_ids[0]]},
        )
        assert read["spans"][0]["text"] == "Alpha is recorded as 5 mg."
        with pytest.raises(MCPToolFailure, match="invalid_search"):
            client.portal.call(
                client.app.state.mcp.call,
                "search_evidence",
                {"run_scope_id": running["id"], "query": ""},
            )
        with pytest.raises(MCPToolFailure, match="unknown_span"):
            client.portal.call(
                client.app.state.mcp.call,
                "read_evidence",
                {"run_scope_id": running["id"], "span_ids": [str(uuid.uuid4())]},
            )
        store.cancel_run(running["id"])

        long_group = collection(client, settings)
        late_fact = "Trial NCT09990001 reports 73 participants."
        long_line = "Background context repeated. " * 1500 + late_fact
        long_source = upload(
            client,
            settings,
            long_group["id"],
            (long_line + "\n").encode(),
            filename="long-research-record.txt",
        ).json()
        ingest(store, settings)
        fragments = store.source_spans(long_source["id"])
        assert len(fragments) > 8
        assert [item["ordinal"] for item in fragments] == list(range(1, len(fragments) + 1))
        assert all(len(item["text"].encode()) <= 4096 for item in fragments)
        assert "".join(item["text"] for item in fragments) == long_line
        late_fragment = next(item for item in fragments if late_fact in item["text"])
        long_chat = conversation(client, settings, long_group["id"])
        long_run = store.create_run(
            long_chat["id"], uuid.uuid4().hex, "What did NCT09990001 report?", "gpt-6-sol", "V1"
        )
        long_search = client.portal.call(
            client.app.state.mcp.call,
            "search_evidence",
            {
                "run_scope_id": long_run["id"],
                "query": "NCT09990001 participants",
                "variant": "V1",
            },
        )
        assert any(
            item["source_id"] == long_source["id"]
            and late_fragment["id"] in item["span_ids"]
            and "literal" in item["branches"]
            for item in long_search["candidates"]
        )
        long_read = client.portal.call(
            client.app.state.mcp.call,
            "read_evidence",
            {"run_scope_id": long_run["id"], "span_ids": [late_fragment["id"]]},
        )
        assert late_fact in long_read["spans"][0]["text"]
        assert long_read["text_bytes"] <= 4096
        collected = []
        cursor = ""
        while True:
            page = client.portal.call(
                client.app.state.mcp.call,
                "collect_scope",
                {"run_scope_id": long_run["id"], "cursor": cursor, "page_bytes": 8192},
            )
            assert page["spans"] and page["text_bytes"] <= 8192
            collected.extend(item["text"] for item in page["spans"])
            cursor = page["next_cursor"]
            if not cursor:
                break
        assert "".join(collected) == long_line
        store.cancel_run(long_run["id"])


def test_evaluation_release_runner_judge_latency_and_report(settings, store, monkeypatch):
    async def respond(payload):
        context = " ".join(
            [payload["question"]]
            + [message["content"] for message in payload["history"] if message["role"] == "user"]
        )
        subject = "Cedar" if "Cedar" in context else "Willow"
        span = next(
            item
            for item in payload["evidence"]["spans"]
            if f"for {subject} has code" in item["text"]
        )
        object_question = payload["question"] == "For Cedar, what was the object?"
        statement = (
            "The object is Boreal."
            if object_question
            else "The code is BETA."
            if subject == "Cedar"
            else "The code is ALPHA."
        )
        return {
            "kind": "answer",
            "status": "supported",
            "answer": statement,
            "claims": [{"text": statement, "evidence_ids": [span["id"]]}],
            "limitations": [],
            "coverage_requirement": "claims",
            "requests": [],
        }

    provider = Provider(
        respond,
        usage={
            "input_tokens": 100,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 20,
        },
    )
    judge = SyntheticJudge()
    release_dir, extraction_path = synthetic_release(settings)
    with client_for(settings, store, provider) as client:
        evaluator_service = ASGIService(settings, client, store)
        initial_release = Release(release_dir)
        corpus_ledger = CorpusLedger(store)
        ledger = Ledger(store)
        corpus = corpus_ledger.create_import(initial_release)
        imported = evaluator_service.import_release(corpus_ledger, corpus, initial_release)
        assert imported["status"] == "ready"
        assert (
            corpus_ledger.by_corpus_id(initial_release.manifest["corpus_id"])["id"]
            == imported["id"]
        )
        source_map = imported["source_map"]
        assert set(source_map) == {"collection_id", "collection_revision", "documents"}
        assert source_map["collection_id"]
        assert isinstance(source_map["collection_revision"], int)
        assert set(source_map["documents"]) == {
            "doc-e2e-willow",
            "doc-e2e-cedar",
            "doc-e2e-ash",
        }
        assert {item["status"] for item in source_map["documents"].values()} == {"ready"}
        collection_sources = evaluator_service.request(
            "GET", f"/api/collections/{source_map['collection_id']}/sources"
        )["items"]
        assert {item["id"] for item in collection_sources} == {
            item["source_id"] for item in source_map["documents"].values()
        }
        release = Release(release_dir, extraction_path, require_gold=True, partition="development")
        acceptance_release = Release(
            release_dir, extraction_path, require_gold=True, partition="acceptance"
        )
        assert len(initial_release.cases) == 22
        assert set(release.cases) == {"case-one", "case-two"}
        assert len(acceptance_release.cases) == 20
        assert {case["question_class"] for case in acceptance_release.cases.values()} == {
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
        }
        partial_release = Release(
            release_dir, extraction_path, require_gold=True, partition="development"
        )
        partial_release.cases["case-one"]["turns"] = [
            {"role": "user", "text": "List all observations across the whole collection."}
        ]
        partial_release.golds["case-one"]["gold_status"] = "partial"
        partial_release.golds["case-one"]["required_scope_doc_ids"] = [
            "doc-e2e-willow",
            "doc-e2e-cedar",
        ]
        source_id = source_map["documents"]["doc-e2e-willow"]["source_id"]
        span = evaluator_service.spans(source_id)[0]
        probe_run_id = str(uuid.uuid4())
        probe_run = {
            "status": "succeeded",
            "answer": {
                "status": "partial",
                "answer": "Only this cited excerpt was inspected; the collection is not fully reviewed.",
                "claims": [{"text": "The code is ALPHA.", "evidence_ids": [span["id"]]}],
                "citations": [{"id": span["id"]}],
                "coverage": {"complete": False, "delivered_source_ids": [source_id]},
                "coverage_requirement": "claims",
            },
            "metrics": {"end_to_end_ms": 1},
            "trace_status": "complete",
        }
        probe_attempt = {
            "case_id": "case-one",
            "variant": "V0",
            "generator": "gpt-6-sol",
            "first_run_id": None,
            "target_run_id": probe_run_id,
            "state": "completed",
            "judge_status": None,
            "judge_output": None,
            "judge_na_reason": None,
        }
        evidence = CorpusEvidence(partial_release, source_map, evaluator_service)
        original_get_run = store.get_run
        with monkeypatch.context() as patch:
            patch.setattr(
                store,
                "get_run",
                lambda run_id: probe_run if run_id == probe_run_id else original_get_run(run_id),
            )
            assert (
                evidence.metric_row(probe_attempt, store)["machine"]["required_coverage_valid"]
                is True
            )
            probe_run["answer"]["status"] = "supported"
            assert (
                evidence.metric_row(probe_attempt, store)["machine"]["required_coverage_valid"]
                is False
            )
            probe_run["answer"]["status"] = "not_documented"
            assert (
                evidence.metric_row(probe_attempt, store)["machine"]["required_coverage_valid"]
                is False
            )
            probe_run["answer"]["status"] = "partial"
            probe_run["answer"]["coverage_requirement"] = "complete_scope"
            assert (
                evidence.metric_row(probe_attempt, store)["machine"]["required_coverage_valid"]
                is False
            )
            probe_run["answer"]["coverage_requirement"] = "claims"
            probe_run["answer"]["claims"][0]["text"] = (
                "Across the whole collection, ALPHA is reported."
            )
            assert (
                evidence.metric_row(probe_attempt, store)["machine"]["required_coverage_valid"]
                is False
            )
        price_path = ROOT / "backend/src/medical_assistant/pricing_snapshot.json"
        config = {
            "selected_variant": "V3",
            "selected_generator": "gpt-6-sol",
            "selected_case_ids": ["case-one", "case-two"],
            "judge_model": "gpt-6-sol",
            "provider": "fixture",
            "embedding_model": settings.embedding_model,
            "embedding_revision": settings.embedding_revision,
            "price_snapshot_id": json.loads(price_path.read_text(encoding="utf-8"))["id"],
            "price_snapshot_sha256": file_hash(price_path),
        }
        frozen = ledger.freeze(release, dict(config), imported, evaluator_service)
        acceptance_frozen = ledger.freeze(
            acceptance_release,
            {**config, "selected_case_ids": sorted(acceptance_release.cases)},
            imported,
            evaluator_service,
        )
        campaign = frozen
        assert frozen["status"] == "frozen" and frozen["planned_count"] == 8
        assert acceptance_frozen["status"] == "frozen"
        assert acceptance_frozen["planned_count"] == 80
        assert frozen["id"] != acceptance_frozen["id"]
        assert frozen["fingerprint"] != acceptance_frozen["fingerprint"]
        assert frozen["corpus_row_id"] == acceptance_frozen["corpus_row_id"] == imported["id"]
        assert frozen["source_map"] == acceptance_frozen["source_map"] == source_map
        assert frozen["split"] == "development"
        assert acceptance_frozen["split"] == "acceptance"
        assert {row["case_id"] for row in ledger.rows(acceptance_frozen["id"])} == set(
            acceptance_release.cases
        )
        assert len(ledger.rows(acceptance_frozen["id"])) == 80
        planned = ledger.rows(campaign["id"])
        assert frozen["config"]["execution_order"] == [
            [row["case_id"], row["variant"], row["generator"]] for row in planned
        ]
        assert {
            case_id: sum(row["case_id"] == case_id for row in planned) for case_id in release.cases
        } == {
            "case-one": 4,
            "case-two": 4,
        }
        runner = Runner(ledger, release, frozen, evaluator_service, settings)
        runner.provider = judge
        runner.telemetry = NoopTelemetry()
        runner.verify()
        with ledger.coordinator(campaign["id"]):
            with pytest.raises(RuntimeError, match="Another coordinator"):
                runner.execute_pending(limit=1)
        assert runner.execute_pending(limit=5) == "incomplete"
        acceptance_runner = Runner(
            ledger, acceptance_release, acceptance_frozen, evaluator_service, settings
        )
        acceptance_judge = SyntheticJudge()
        acceptance_judge.fail_once = False
        acceptance_runner.provider = acceptance_judge
        acceptance_runner.telemetry = NoopTelemetry()
        acceptance_runner.verify()
        assert acceptance_runner.execute_pending(limit=1) == "incomplete"
        acceptance_attempt = ledger.rows(acceptance_frozen["id"])[0]
        assert acceptance_attempt["state"] == "completed"
        imported_ids = {item["source_id"] for item in source_map["documents"].values()}
        acceptance_run = store.get_run(acceptance_attempt["target_run_id"])
        assert acceptance_run["scope"]["collection_id"] == source_map["collection_id"]
        assert acceptance_run["scope"]["revision"] == source_map["collection_revision"]
        assert acceptance_run["scope"]["source_count"] == len(imported_ids)
        assert acceptance_run["scope"]["ready_count"] == len(imported_ids)
        assert acceptance_run["scope"]["unavailable_count"] == 0
        assert any(
            "Object Atlas for Ash has code OMEGA." in span["text"]
            for request_id, item in provider.calls
            if request_id == acceptance_run["id"] and item.get("task_mode") == "answer"
            for span in item["evidence"]["spans"]
        )
        acceptance_report = report_data(
            ledger, ledger.campaign(acceptance_frozen["id"]), acceptance_release
        )
        assert acceptance_report["campaign"]["planned_count"] == 80
        assert acceptance_report["state_counts"] == {"completed": 1, "pending": 79}
        assert all(cell["summary"]["planned"] == 20 for cell in acceptance_report["cells"].values())
        assert ("judge.generate", "generation") in runner.telemetry.observation_calls
        rows = ledger.rows(campaign["id"])
        assert all(
            store.get_run(row["target_run_id"])["scope"]["collection_id"]
            == source_map["collection_id"]
            and store.get_run(row["target_run_id"])["scope"]["ready_count"] == len(imported_ids)
            for row in rows
            if row["state"] == "completed"
        )
        assert all(row["state"] == "completed" for row in rows[:5]), [
            (row["case_id"], row["variant"], row["state"]) for row in rows[:5]
        ]
        one_turn = [row for row in rows[:5] if row["case_id"] == "case-one"]
        two_turn = next(row for row in rows[:5] if row["case_id"] == "case-two")
        assert two_turn["first_run_id"] and two_turn["target_run_id"]
        assert two_turn["first_run_id"] != two_turn["target_run_id"]
        assert any(
            metadata
            and metadata.get("campaign_id") == campaign["id"]
            and metadata.get("case_id") == "case-two"
            and metadata.get("target_run_id") == two_turn["target_run_id"]
            and metadata.get("app_url")
            == settings.app_origin + "/runs/" + two_turn["target_run_id"]
            for metadata in runner.telemetry.observation_metadata
        )
        two_turn_payload = two_turn["attempts"][-1]["judge_payload"]
        assert two_turn_payload["conversation_context"][0] == {
            "role": "user",
            "text": "For Cedar, what was the object?",
        }
        assert two_turn_payload["conversation_context"][1]["answer"] == "The object is Boreal."
        assert two_turn_payload["question"] == "What was its code?"
        assert [item["text"] for item in two_turn_payload["scope_source_excerpts"]] == [
            "Object Boreal for Cedar has code BETA."
        ]
        target_run = store.get_run(two_turn["target_run_id"])
        assert target_run["scope"]["collection_id"] == source_map["collection_id"]
        assert target_run["scope"]["source_count"] == len(imported_ids)
        assert target_run["scope"]["ready_count"] == len(imported_ids)
        assert target_run["scope"]["unavailable_count"] == 0
        assert any(
            "Object Atlas for Ash has code OMEGA." in span["text"]
            for request_id, item in provider.calls
            if request_id == two_turn["target_run_id"] and item.get("task_mode") == "answer"
            for span in item["evidence"]["spans"]
        )
        first_latency = store.get_run(two_turn["first_run_id"])["metrics"]["end_to_end_ms"]
        first_projection = one_turn[0]["metrics"]["projection"]["execution"]
        second_projection = two_turn["metrics"]["projection"]["execution"]
        assert first_projection["total_latency_ms"] == pytest.approx(
            first_projection["target_latency_ms"]
        )
        assert second_projection["total_latency_ms"] == pytest.approx(
            first_latency + second_projection["target_latency_ms"]
        )
        unjudged = next(row for row in rows[:5] if row["judge_status"] == "not_assessable")
        before = report_data(ledger, ledger.campaign(campaign["id"]), release)
        assert before["state_counts"] == {"completed": 5, "pending": 3}
        assert before["api_equivalent_cost"]["judge"]["unestimated_call_count"] == 1
        assert (
            runner.retry(unjudged["id"], "synthetic judge recovery", judge_only=True)
            == "incomplete"
        )
        recovered = ledger.row(unjudged["id"])
        assert recovered["judge_status"] == "assessed"
        assert len(recovered["attempts"][-1]["judge_history"]) == 1
        assert recovered["metrics"]["primary"]["projection"]["judge"] is None
        assert (
            recovered["attempts"][-1]["judge_history"][0]["judge_operation_id"]
            != recovered["attempts"][-1]["judge_operation_id"]
        )
        assert len(judge.calls) == 6
        report = report_data(ledger, ledger.campaign(campaign["id"]), release)
        assert report["campaign"]["planned_count"] == 8 and len(report["cells"]) == 4
        assert all(cell["summary"]["planned"] == 2 for cell in report["cells"].values())
        assert (
            sum(
                cell["summary"]["technical_completion"]["numerator"]
                for cell in report["cells"].values()
            )
            == 5
        )
        assert (
            sum(
                cell["latest_correction"]["confirmed_grounded_success"]["numerator"]
                for cell in report["cells"].values()
            )
            == 5
        )
        assert (
            sum(
                cell["summary"]["judge_assessable"]["numerator"]
                for cell in report["cells"].values()
            )
            == 4
        )
        costs = report["api_equivalent_cost"]
        assert (
            costs["generation"]["call_count"]
            == before["api_equivalent_cost"]["generation"]["call_count"]
        )
        assert costs["generation"]["unestimated_call_count"] == 0
        assert costs["judge"]["call_count"] == 6
        assert costs["judge"]["unestimated_call_count"] == 1
        assert costs["judge"]["estimated_total_api_usd"] is None
        assert costs["generation"]["actual_billed_usd"] is None
        report_path, csv_path = write_report(report, settings.data_dir / "reports")
        assert json.loads(report_path.read_text(encoding="utf-8"))["campaign"]["planned_count"] == 8
        assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 5
        experiments = evaluator_service.request("GET", "/api/experiments")["items"]
        assert len([item for item in experiments if item["campaign_id"] == campaign["id"]]) == 4
        experiment = next(item for item in experiments if item["campaign_id"] == campaign["id"])
        detail = evaluator_service.request("GET", f"/api/experiments/{experiment['id']}")
        assert len(detail["cases"]) == 2
        groups = detail["breakdown"]["anchor_document_class"]
        assert sum(group["planned"] for group in groups.values()) == 2
        assert (
            sum(group["confirmed_grounded_success"]["denominator"] for group in groups.values())
            == 2
        )
        parallel_settings = settings.model_copy(
            update={"data_dir": settings.data_dir / "parallel-release"}
        )
        parallel_dir, parallel_extraction = synthetic_release(parallel_settings, tag="parallel")
        parallel_initial = Release(parallel_dir)
        parallel_corpus = corpus_ledger.create_import(parallel_initial)
        parallel_imported = evaluator_service.import_release(
            corpus_ledger, parallel_corpus, parallel_initial
        )
        parallel_release = Release(
            parallel_dir, parallel_extraction, require_gold=True, partition="development"
        )
        parallel_config = {**config, "case_workers": 4}
        parallel_frozen = ledger.freeze(
            parallel_release,
            parallel_config,
            parallel_imported,
            evaluator_service,
        )
        parallel_campaign = parallel_frozen
        assert parallel_frozen["config"]["case_workers"] == 4
        parallel_runner = Runner(
            ledger, parallel_release, parallel_frozen, evaluator_service, settings
        )
        parallel_runner.verify()
        worker_lock = threading.Lock()
        workers = []
        active_workers = 0
        generation_lock = threading.Lock()
        generation_gate = asyncio.Event()
        active_generations = 0
        peak_generations = 0
        original_generate = provider.generate

        async def overlapping_generate(payload, *, role, model, request_id, final_only):
            nonlocal active_generations, peak_generations
            with generation_lock:
                active_generations += 1
                peak_generations = max(peak_generations, active_generations)
                if active_generations == 4:
                    generation_gate.set()
            try:
                await asyncio.wait_for(generation_gate.wait(), timeout=15)
                return await original_generate(
                    payload,
                    role=role,
                    model=model,
                    request_id=request_id,
                    final_only=final_only,
                )
            finally:
                with generation_lock:
                    active_generations -= 1

        provider.generate = overlapping_generate

        @contextmanager
        def fixture_worker():
            nonlocal active_workers
            worker_store = Store(settings)
            worker_service = ASGIService(settings, client, worker_store)
            worker = Runner(
                Ledger(worker_store),
                parallel_release,
                parallel_frozen,
                worker_service,
                settings,
            )
            worker.provider = SyntheticJudge()
            worker.provider.fail_once = False
            worker.telemetry = NoopTelemetry()
            with worker_lock:
                workers.append(worker)
                active_workers += 1
            try:
                yield worker
            finally:
                with worker_lock:
                    active_workers -= 1
                worker_store.engine.dispose()

        parallel_runner._worker = fixture_worker
        assert parallel_runner.execute_pending(limit=4) == "incomplete"
        provider.generate = original_generate
        assert active_workers == 0
        assert peak_generations == 4 and active_generations == 0
        first_batch = ledger.rows(parallel_campaign["id"])[:4]
        assert all(
            row["state"] == "completed" and row["judge_status"] == "assessed" for row in first_batch
        )
        first_ids = {row["id"]: row["target_run_id"] for row in first_batch}
        first_attempts = {row["id"]: len(row["attempts"]) for row in first_batch}
        assert sum(len(worker.provider.calls) for worker in workers) == 4
        resumed_rows = ledger.rows(parallel_campaign["id"])
        assert sum(row["state"] == "completed" for row in resumed_rows) == 4
        assert all(
            ledger.row(row_id)["target_run_id"] == run_id
            and len(ledger.row(row_id)["attempts"]) == first_attempts[row_id]
            for row_id, run_id in first_ids.items()
        )
        assert active_workers == 0
        prior_judges = sum(len(worker.provider.calls) for worker in workers)
        failure_rows = resumed_rows[4:8]
        failed_id = failure_rows[0]["id"]
        failure_barrier = threading.Barrier(4)
        entered = threading.Event()
        release_workers = threading.Event()
        original_worker_call = parallel_runner._worker_call

        def interrupted_worker_call(row, judge_only):
            failure_barrier.wait(timeout=15)
            entered.set()
            if row["id"] == failed_id:
                raise RuntimeError("synthetic worker interruption")
            assert release_workers.wait(timeout=15)
            original_worker_call(row, judge_only)

        parallel_runner._worker_call = interrupted_worker_call
        with ThreadPoolExecutor(max_workers=1) as coordinator_pool:
            coordinator_result = coordinator_pool.submit(parallel_runner.execute_pending, 4)
            try:
                assert entered.wait(timeout=15)
                with pytest.raises(RuntimeError, match="Another coordinator"):
                    parallel_runner.execute_pending(limit=1)
            finally:
                release_workers.set()
            with pytest.raises(RuntimeError, match="synthetic worker interruption"):
                coordinator_result.result(timeout=45)
        parallel_runner._worker_call = original_worker_call
        drained_rows = ledger.rows(parallel_campaign["id"])[4:8]
        assert (
            sum(
                row["state"] == "completed" and row["judge_status"] == "assessed"
                for row in drained_rows
            )
            == 3
        )
        assert ledger.row(failed_id)["state"] == "running"
        assert active_workers == 0
        assert sum(len(worker.provider.calls) for worker in workers) == prior_judges + 3
        completed_ids = {
            row["id"]: row["target_run_id"] for row in drained_rows if row["state"] == "completed"
        }
        assert parallel_runner.resume(limit=1) == "complete"
        assert ledger.row(failed_id)["state"] == "completed"
        assert all(
            ledger.row(row_id)["target_run_id"] == run_id
            for row_id, run_id in completed_ids.items()
        )
        assert parallel_runner.resume() == "complete"
        final_rows = ledger.rows(parallel_campaign["id"])
        assert len(final_rows) == 8
        assert all(
            row["state"] == "completed" and row["judge_status"] == "assessed" for row in final_rows
        )
        final_runs = {row["id"]: row["target_run_id"] for row in final_rows}
        final_attempts = {row["id"]: len(row["attempts"]) for row in final_rows}
        final_answer_calls = len(provider.calls)
        final_judge_calls = sum(len(worker.provider.calls) for worker in workers)
        assert parallel_runner.resume() == "complete"
        assert len(provider.calls) == final_answer_calls
        assert sum(len(worker.provider.calls) for worker in workers) == final_judge_calls == 8
        assert all(
            row["target_run_id"] == final_runs[row["id"]]
            and len(row["attempts"]) == final_attempts[row["id"]]
            for row in ledger.rows(parallel_campaign["id"])
        )

        def frozen_outage_campaign(name, case_workers=4):
            outage_settings = settings.model_copy(update={"data_dir": settings.data_dir / name})
            outage_dir, outage_extraction = synthetic_release(outage_settings, tag=name)
            initial = Release(outage_dir)
            created = corpus_ledger.create_import(initial)
            imported = evaluator_service.import_release(corpus_ledger, created, initial)
            release = Release(
                outage_dir, outage_extraction, require_gold=True, partition="development"
            )
            frozen = ledger.freeze(
                release,
                {**config, "case_workers": case_workers},
                imported,
                evaluator_service,
            )
            return release, frozen

        class UnavailableBridge:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def post(self, *_args, **_kwargs):
                raise httpx.ConnectError("synthetic bridge disconnect")

        class UnauthorizedBridge:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def post(self, *_args, **_kwargs):
                return httpx.Response(401)

        class FailedBridge:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def post(self, *_args, **_kwargs):
                return httpx.Response(500)

        class SemanticBridge:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_):
                return None

            async def post(self, *_args, **_kwargs):
                return httpx.Response(
                    422, json={"detail": {"code": "invalid_output", "message": "invalid output"}}
                )

        bridge_settings = settings.model_copy(
            update={"bridge_token": "synthetic-bridge-token", "bridge_url": "http://127.0.0.1:8768"}
        )
        bridge_provider = BridgeProvider(bridge_settings)
        answer_release, answer_frozen = frozen_outage_campaign("answer-outage", case_workers=2)
        answer_runner = Runner(ledger, answer_release, answer_frozen, evaluator_service, settings)
        answer_runner.verify()
        answer_lock = threading.Lock()
        answer_calls = 0

        async def fail_first_answer(payload):
            nonlocal answer_calls
            with answer_lock:
                answer_calls += 1
                fail = answer_calls == 1
            if fail:
                await bridge_provider.generate(payload, request_id="synthetic-transport")
            return await respond(payload)

        provider.respond = fail_first_answer

        @contextmanager
        def answer_outage_worker():
            worker_store = Store(settings)
            worker_service = ASGIService(settings, client, worker_store)
            worker = Runner(
                Ledger(worker_store), answer_release, answer_frozen, worker_service, settings
            )
            worker.provider = SyntheticJudge()
            worker.provider.fail_once = False
            worker.telemetry = NoopTelemetry()
            try:
                yield worker
            finally:
                worker_store.engine.dispose()

        answer_runner._worker = answer_outage_worker
        with monkeypatch.context() as patch:
            patch.setattr(bridge_client.httpx, "AsyncClient", lambda **_: UnavailableBridge())
            assert answer_runner.execute_pending() == "incomplete"
        provider.respond = respond
        assert answer_runner.outage_code == "bridge_transport_unavailable"
        answer_rows = ledger.rows(answer_frozen["id"])
        started = answer_rows[:2]
        waiting = answer_rows[2:]
        failed = [row for row in started if row["state"] in {"first_turn_failed", "target_failed"}]
        assert len(failed) == 1
        assert all(row["state"] == "completed" for row in started if row not in failed)
        assert all(row["state"] == "pending" and row["attempts"] == [] for row in waiting)
        failed_run_id = failed[0]["first_run_id"] or failed[0]["target_run_id"]
        assert any(
            item["stage"] == "run.exception"
            and item["payload"].get("code") == "bridge_transport_unavailable"
            for item in store.audit(failed_run_id)
        )
        assert sum(call[0] == failed_run_id for call in provider.calls) == 1
        assert answer_runner.resume(limit=0) == "incomplete"
        failed_after_resume = ledger.row(failed[0]["id"])
        assert failed_after_resume["state"] == failed[0]["state"]
        assert len(failed_after_resume["attempts"]) == 1
        assert sum(call[0] == failed_run_id for call in provider.calls) == 1

        pending_before_500 = [
            row for row in ledger.rows(answer_frozen["id"]) if row["state"] == "pending"
        ]
        response_count = 0

        async def fail_one_response(payload):
            nonlocal response_count
            with answer_lock:
                response_count += 1
                fail = response_count == 1
            if fail:
                await bridge_provider.generate(payload, request_id="synthetic-http500")
            return await respond(payload)

        provider.respond = fail_one_response
        with monkeypatch.context() as patch:
            patch.setattr(bridge_client.httpx, "AsyncClient", lambda **_: FailedBridge())
            assert answer_runner.execute_pending() == "incomplete"
        provider.respond = respond
        assert answer_runner.outage_code == "bridge_unavailable"
        after_500 = {row["id"]: row for row in ledger.rows(answer_frozen["id"])}
        assert all(after_500[row["id"]]["state"] != "pending" for row in pending_before_500[:2])
        assert all(
            after_500[row["id"]]["state"] == "pending" and after_500[row["id"]]["attempts"] == []
            for row in pending_before_500[2:]
        )
        response_failures = [
            after_500[row["id"]]
            for row in pending_before_500[:2]
            if after_500[row["id"]]["state"] in {"first_turn_failed", "target_failed"}
        ]
        assert len(response_failures) == 1
        response_run_id = (
            response_failures[0]["first_run_id"] or response_failures[0]["target_run_id"]
        )
        assert any(
            event["stage"] == "run.exception"
            and event["payload"].get("code") == "bridge_unavailable"
            for event in store.audit(response_run_id)
        )

        semantic_count = 0

        async def fail_one_semantic(payload):
            nonlocal semantic_count
            with answer_lock:
                semantic_count += 1
                fail = semantic_count == 1
            if fail:
                await bridge_provider.generate(payload, request_id="synthetic-invalid-output")
            return await respond(payload)

        provider.respond = fail_one_semantic
        with monkeypatch.context() as patch:
            patch.setattr(bridge_client.httpx, "AsyncClient", lambda **_: SemanticBridge())
            assert answer_runner.execute_pending() == "complete"
        provider.respond = respond
        assert answer_runner.outage_code is None
        semantic_pending_ids = {row["id"] for row in pending_before_500[2:]}
        semantic_failures = [
            row
            for row in ledger.rows(answer_frozen["id"])
            if row["id"] in semantic_pending_ids
            and row["state"] in {"first_turn_failed", "target_failed"}
        ]
        assert len(semantic_failures) == 1
        semantic_run_id = (
            semantic_failures[0]["target_run_id"] or semantic_failures[0]["first_run_id"]
        )
        assert any(
            event["stage"] == "run.exception" and event["payload"].get("code") == "invalid_output"
            for event in store.audit(semantic_run_id)
        )
        assert sum(call[0] == failed_run_id for call in provider.calls) == 1

        primary_outage_report = report_data(
            ledger, ledger.campaign(answer_frozen["id"]), answer_release
        )
        answer_runner.provider = SyntheticJudge()
        answer_runner.provider.fail_once = False
        answer_runner.telemetry = NoopTelemetry()
        assert answer_runner.retry(failed[0]["id"], "explicit synthetic correction") == "complete"
        corrected_row = ledger.row(failed[0]["id"])
        assert len(corrected_row["attempts"]) == 2
        assert corrected_row["metrics"]["primary"]["projection"]["execution"]["state"] in {
            "first_turn_failed",
            "target_failed",
        }
        assert failed_run_id in [
            turn["run_id"] for turn in corrected_row["attempts"][0]["turns"] if turn.get("run_id")
        ]
        corrected_outage_report = report_data(
            ledger, ledger.campaign(answer_frozen["id"]), answer_release
        )
        assert [
            cell["summary"]["confirmed_grounded_success"]
            for cell in corrected_outage_report["cells"].values()
        ] == [
            cell["summary"]["confirmed_grounded_success"]
            for cell in primary_outage_report["cells"].values()
        ]
        assert corrected_outage_report["state_counts"] == primary_outage_report["state_counts"]
        assert corrected_outage_report["latest_correction"]["attempted_rows"] == 1
        assert corrected_outage_report["latest_correction"]["completed"] == (
            primary_outage_report["latest_correction"]["completed"] + 1
        )
        corrected_cell = corrected_outage_report["cells"][
            f"{corrected_row['variant']}/{corrected_row['generator']}"
        ]["summary"]
        assert "confirmed_grounded_success_on_assessable" in corrected_cell
        experiment_item = next(
            item
            for item in evaluator_service.request("GET", "/api/experiments")["items"]
            if item["campaign_id"] == answer_frozen["id"]
            and item["variant"] == corrected_row["variant"]
            and item["model"] == corrected_row["generator"]
        )
        assert experiment_item["completed"] == corrected_cell["technical_completion"]["numerator"]
        assert experiment_item["latest_correction"]["completed"] == (
            experiment_item["completed"] + 1
        )

        judge_release, judge_frozen = frozen_outage_campaign("judge-outage")
        judge_runner = Runner(ledger, judge_release, judge_frozen, evaluator_service, settings)
        judge_runner.verify()

        @contextmanager
        def judge_outage_worker():
            worker_store = Store(settings)
            worker_service = ASGIService(settings, client, worker_store)
            worker = Runner(
                Ledger(worker_store), judge_release, judge_frozen, worker_service, settings
            )
            worker.provider = bridge_provider
            worker.telemetry = NoopTelemetry()
            try:
                yield worker
            finally:
                worker_store.engine.dispose()

        judge_runner._worker = judge_outage_worker
        with monkeypatch.context() as patch:
            patch.setattr(bridge_client.httpx, "AsyncClient", lambda **_: UnauthorizedBridge())
            assert judge_runner.execute_pending() == "incomplete"
        assert judge_runner.outage_code == "bridge_unauthorized"
        judge_rows = ledger.rows(judge_frozen["id"])
        assert all(
            row["state"] == "completed"
            and row["judge_status"] == "not_assessable"
            and row["attempts"][-1]["judge_error"]["code"] == "bridge_unauthorized"
            for row in judge_rows[:4]
        )
        assert all(row["state"] == "pending" and row["attempts"] == [] for row in judge_rows[4:])

        prior_cost = recovered["metrics"]["cost_estimate"]["judge"]
        prior_cost_state = recovered["attempts"][-1]["judge_cost_state"]
        runner.provider = bridge_provider
        with monkeypatch.context() as patch:
            patch.setattr(bridge_client.httpx, "AsyncClient", lambda **_: UnauthorizedBridge())
            assert runner.retry(
                recovered["id"], "synthetic judge auth outage", judge_only=True
            ) == ("incomplete")
        retried_judge = ledger.row(recovered["id"])
        latest = retried_judge["attempts"][-1]
        assert latest["judge_history"][-1]["cost_state"] == prior_cost_state
        assert latest["judge_cost_state"] == "unknown"
        assert retried_judge["metrics"]["judge_cost_state"] == "unknown"
        assert latest["judge_error"]["code"] == "bridge_unauthorized"
        judge_cost = retried_judge["metrics"]["cost_estimate"]["judge"]
        assert judge_cost["call_count"] == prior_cost["call_count"] + 1
        assert judge_cost["unestimated_call_count"] == prior_cost["unestimated_call_count"] + 1
