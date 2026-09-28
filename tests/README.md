# End-to-end tests

Six scenarios exercise the API through FastAPI's in-process `TestClient`, with real PostgreSQL, ingestion, local E5 embeddings and MiniLM reranking, and the stdio MCP server. Deterministic model fixtures also check the evaluation workflow. These tests verify integration behavior; they do not establish browser rendering, network serving, live-model answer quality, or large-corpus recall. The [root README](../README.md#verification-and-review) describes the separate network integration check.

Run from the repository root with `uv`, Docker, and the pinned local E5/MiniLM model cache available:

```sh
uv run --project . python deploy/configure.py
docker compose --env-file tmp/work/runtime/.env -f deploy/compose.yaml build postgres
docker compose --env-file tmp/work/runtime/.env -f deploy/compose.yaml up -d --wait postgres
uv run --project . python -m medical_assistant.reranking
uv run --project . pytest tests/test_e2e.py -q
```

The Compose setup provides PostgreSQL 17 with `vector`, preloaded `pg_textsearch`, and the required roles. The fixture creates and drops an isolated database for each run; it does not use the application's database or make paid model calls. Preparing the local model cache may download pinned public files.
