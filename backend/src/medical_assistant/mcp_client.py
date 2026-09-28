import asyncio
import logging
import os
import secrets
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from pydantic import ValidationError

from medical_assistant.settings import Settings

_TOOLS = {
    "search_evidence",
    "read_evidence",
    "collect_scope",
}
_LOGGER = logging.getLogger(__name__)


class MCPToolFailure(Exception):
    pass


class MCPTransportFailure(Exception):
    pass


class MCPGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Client | None = None
        self._lock = asyncio.Lock()
        self._cursor_key = secrets.token_hex(32)
        self._owner_task: asyncio.Task | None = None
        self._owner_stop: asyncio.Event | None = None

    async def __aenter__(self) -> "MCPGateway":
        async with self._lock:
            await self._start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        async with self._lock:
            await self._stop()

    async def _start(self) -> None:
        if not self.settings.read_database_url:
            raise MCPTransportFailure("read_database_url_required")
        if self._owner_task is not None:
            await self._stop()
        project = Path(__file__).resolve().parents[3]
        env = {
            key: os.environ[key]
            for key in (
                "PATH",
                "HOME",
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "XDG_CACHE_HOME",
                "SSL_CERT_FILE",
            )
            if key in os.environ
        }
        env.update(
            {
                "PFL_READ_DATABASE_URL": self.settings.read_database_url,
                "PFL_EMBEDDING_MODEL": self.settings.embedding_model,
                "PFL_EMBEDDING_REVISION": self.settings.embedding_revision,
                "PFL_EMBEDDING_THREADS": str(self.settings.embedding_threads),
                "PFL_RERANKER_MODEL": self.settings.reranker_model,
                "PFL_RERANKER_REVISION": self.settings.reranker_revision,
                "PFL_RERANKER_THREADS": str(self.settings.reranker_threads),
                "PFL_MCP_CURSOR_KEY": self._cursor_key,
                "PYTHONPATH": str(project / "backend" / "src"),
            }
        )
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "medical_assistant.mcp_server"],
            env=env,
            cwd=project,
        )
        ready = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()
        self._owner_stop = stop
        self._owner_task = asyncio.create_task(self._run_client(params, ready, stop))
        try:
            await asyncio.shield(ready)
        except BaseException:
            if not ready.done():
                ready.cancel()
            await self._stop()
            raise

    async def _run_client(
        self, params: StdioServerParameters, ready: asyncio.Future, stop: asyncio.Event
    ) -> None:
        try:
            async with Client(params, read_timeout_seconds=120) as client:
                names = set()
                cursor = None
                while True:
                    listing = await client.list_tools(cursor=cursor)
                    names.update(tool.name for tool in listing.tools)
                    cursor = listing.next_cursor
                    if cursor is None:
                        break
                if names != _TOOLS:
                    raise MCPTransportFailure("unexpected_tool_catalog")
                self._client = client
                ready.set_result(None)
                await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                _LOGGER.warning("mcp_client_owner_failed cause=%s", type(exc).__name__)
        finally:
            self._client = None

    async def _stop(self) -> None:
        owner = self._owner_task
        if owner is None:
            return
        active = self._client is not None
        stopping = self._owner_stop is not None and self._owner_stop.is_set()
        self._client = None
        if self._owner_stop is not None:
            self._owner_stop.set()
        if not active and not stopping:
            owner.cancel()
        try:
            await asyncio.shield(owner)
        finally:
            if owner.done():
                self._owner_task = None
                self._owner_stop = None

    async def call(self, name: str, arguments: dict[str, Any]) -> dict:
        if name not in _TOOLS:
            raise MCPToolFailure("unknown_tool")
        started = perf_counter()
        acquired = False
        try:
            async with asyncio.timeout(120 if name == "search_evidence" else 30):
                async with self._lock:
                    acquired = True
                    queue_wait_ms = (perf_counter() - started) * 1000
                    _LOGGER.log(
                        logging.WARNING if queue_wait_ms >= 1000 else logging.INFO,
                        "mcp_queue_wait tool=%s wait_ms=%.3f",
                        name,
                        queue_wait_ms,
                    )
                    return await self._call_locked(name, arguments)
        except TimeoutError as exc:
            _LOGGER.warning(
                "mcp_call_timeout tool=%s phase=%s elapsed_ms=%.3f",
                name,
                "active" if acquired else "queue",
                (perf_counter() - started) * 1000,
            )
            raise MCPTransportFailure("mcp_timeout") from exc

    async def _call_locked(self, name: str, arguments: dict[str, Any]) -> dict:
        try:
            for attempt in range(2):
                if self._client is None:
                    try:
                        await self._start()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        _LOGGER.warning(
                            "mcp_failure tool=%s phase=start attempt=%d cause=%s",
                            name,
                            attempt + 1,
                            type(exc).__name__,
                        )
                        raise MCPTransportFailure("mcp_start_failed") from exc
                try:
                    result = await self._client.call_tool(name, arguments)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    _LOGGER.warning(
                        "mcp_failure tool=%s phase=call attempt=%d cause=%s",
                        name,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    await self._stop()
                    if attempt == 1:
                        raise MCPTransportFailure("mcp_transport_failed") from exc
                    continue
                if result.is_error:
                    message = "tool_error"
                    for item in result.content:
                        if getattr(item, "type", None) == "text":
                            message = item.text
                            break
                    if message in {
                        "scope_expired",
                        f"Error executing tool {name}: scope_expired",
                    }:
                        message = "scope_expired"
                    raise MCPToolFailure(message)
                if not isinstance(result.structured_content, dict):
                    raise MCPTransportFailure("missing_structured_content")
                from medical_assistant.mcp_server import (
                    CollectResult,
                    ReadResult,
                    SearchResult,
                )

                models = {
                    "search_evidence": SearchResult,
                    "read_evidence": ReadResult,
                    "collect_scope": CollectResult,
                }
                try:
                    validated = models[name].model_validate(result.structured_content).model_dump()
                except ValidationError as exc:
                    _LOGGER.warning(
                        "mcp_failure tool=%s phase=validate attempt=%d cause=%s",
                        name,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    raise MCPTransportFailure("invalid_structured_content") from exc
                return validated
        except asyncio.CancelledError:
            if self._client is not None:
                _LOGGER.warning("mcp_call_cancelled tool=%s phase=active", name)
                await self._stop()
            raise
        raise MCPTransportFailure("mcp_transport_failed")
