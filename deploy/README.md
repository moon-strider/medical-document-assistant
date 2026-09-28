# Native development and provider options

Follow [Run locally](../README.md#run-locally) for initial configuration and Docker setup. Run the commands below from the repository root.

## Responses API provider

Before starting Compose, set `PFL_PROVIDER=api` in the generated private `tmp/work/runtime/.env` and export `OPENAI_API_KEY` in the terminal running Compose. This provider makes paid API calls; only offline contract validation has been performed. It does not need the Codex CLI bridge.

For native development, export `OPENAI_API_KEY` in the API terminal as well.

## Native processes

Keep the PostgreSQL and Langfuse dependency containers running, but stop the Docker application services if they are running:

```sh
docker compose --env-file tmp/work/runtime/.env -f deploy/compose.yaml -f deploy/app.compose.yaml stop api worker reconciler
```

Prepare the host dependencies and model cache:

```sh
uv sync --project . --locked
uv run --project . python -m medical_assistant.reranking
npm ci --prefix frontend
npm run build --prefix frontend
```

In three separate terminals, load the generated environment and run one process per terminal:

```sh
set -a; . tmp/work/runtime/.env; set +a
uv run --project . uvicorn medical_assistant.api:app --host 127.0.0.1 --port 8080
```

```sh
set -a; . tmp/work/runtime/.env; set +a
uv run --project . python -m medical_assistant.worker
```

```sh
set -a; . tmp/work/runtime/.env; set +a
uv run --project . python -m medical_assistant.telemetry_cli
```

With the default CLI provider, also run the bridge as shown in [Run locally](../README.md#run-locally).

## Resource limit

The PostgreSQL container is limited to 768 MiB, including an approximate 128 MiB BM25 cache budget. Size larger deployments from measured chunk and index counts.
