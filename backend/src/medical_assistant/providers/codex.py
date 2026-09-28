import asyncio
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

from pydantic import ValidationError

from .common import ProviderError, check_request, instruction_hash, instruction_paths, prompt_for
from .contracts import load_schema, validate_output

PINNED_VERSION = "codex-cli 0.156.1"
MAX_EVENT_BYTES = 1048576
MAX_STDERR_BYTES = 131072
MAX_PENDING_CANCELLATIONS = 4096


class CodexProvider:
    """Run each subscription-backed inference in a fresh role-configured CLI.

    Active IDs and bounded pre-start cancellation reservations are local to this
    instance. CLI configuration and visible event restrictions constrain model
    work; the temporary role directory is not operating-system file isolation.
    """

    def __init__(self, settings):
        self.settings = settings
        self.active: dict[str, asyncio.subprocess.Process] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.reserved: set[str] = set()
        self.cancelled: set[str] = set()
        self.pending_cancellations: dict[str, float] = {}
        self.lock = asyncio.Lock()
        self.version_checked = False

    def _solution_root(self) -> Path:
        return Path(__file__).resolve().parents[4]

    def _codex_home(self) -> Path:
        path = Path(self.settings.codex_home)
        if not path.is_absolute():
            path = self._solution_root() / path
        return path.resolve()

    def _binary(self) -> Path:
        return Path(self.settings.codex_binary).resolve()

    async def readiness(self) -> dict:
        """Inspect local CLI configuration without starting inference.

        Check executable presence, the application base configuration, every
        allowed role/model profile and login-file presence. This probe does not
        run the pinned-version check or validate the login with the service.

        Returns:
            dict: provider="codex", ready (bool), connection="local",
            configuration ("present" only on success, otherwise "missing"),
            inference="unverified" and reason. ready=True means reason="configured";
            failure reasons identify missing binary/configuration/authentication,
            base/profile drift or an unreadable configuration. No successful
            inference, model access or remaining subscription quota is guaranteed.
        """

        status = {
            "provider": "codex",
            "ready": False,
            "connection": "local",
            "configuration": "missing",
            "inference": "unverified",
            "reason": "binary_missing",
        }
        try:
            binary = self._binary()
            if not binary.is_file() or not os.access(binary, os.X_OK):
                return status
            home = self._codex_home()
            expected_base = (
                self._solution_root() / "config" / "codex" / "config.toml"
            ).read_bytes()
            if not home.is_dir() or not (home / "config.toml").is_file():
                return {**status, "reason": "configuration_missing"}
            if (home / "config.toml").read_bytes() != expected_base or (
                home / "AGENTS.md"
            ).exists():
                return {**status, "reason": "configuration_drift"}
            for role, model in (
                ("answer", "gpt-6-sol"),
                ("answer", "gpt-6-luna"),
                ("judge", "gpt-6-sol"),
            ):
                _, base = instruction_paths(role)
                profile = home / f"{role}-{model.removeprefix('gpt-6-')}.config.toml"
                expected_profile = (
                    f'model = "{model}"\n'
                    'model_reasoning_effort = "high"\n'
                    f'model_instructions_file = "{base}"\n'
                ).encode()
                if not profile.is_file() or profile.read_bytes() != expected_profile:
                    return {**status, "reason": "profile_drift"}
            if not (home / "auth.json").is_file():
                return {**status, "reason": "authentication_missing"}
        except (OSError, ValueError):
            return {**status, "reason": "configuration_unavailable"}
        return {**status, "ready": True, "configuration": "present", "reason": "configured"}

    def _environment(self) -> dict[str, str]:
        allowed = ("HOME", "USER", "LOGNAME", "LANG", "TMPDIR", "TERM")
        environment = {key: value for key in allowed if (value := os.environ.get(key))}
        environment["PATH"] = "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        environment["CODEX_HOME"] = str(self._codex_home())
        environment["TERM"] = "dumb"
        return environment

    async def _check_configuration(self, role: str, model: str) -> None:
        binary = self._binary()
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ProviderError("binary_missing", "pinned Codex binary is unavailable")
        if not self.version_checked:
            process = await asyncio.create_subprocess_exec(
                str(binary),
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._environment(),
                start_new_session=True,
            )
            try:
                stdout, _ = await process.communicate()
            except BaseException:
                await self._terminate(process)
                raise
            if process.returncode != 0 or stdout.decode().strip() != PINNED_VERSION:
                raise ProviderError("version_mismatch", "Codex CLI version is not pinned version")
            self.version_checked = True
        home = self._codex_home()
        expected_base = (self._solution_root() / "config" / "codex" / "config.toml").read_bytes()
        if not home.is_dir() or not (home / "config.toml").is_file():
            raise ProviderError("configuration_missing", "application Codex home is unavailable")
        if (home / "config.toml").read_bytes() != expected_base or (home / "AGENTS.md").exists():
            raise ProviderError("configuration_drift", "application Codex configuration changed")
        _, base = instruction_paths(role)
        profile = home / f"{role}-{model.removeprefix('gpt-6-')}.config.toml"
        expected_profile = (
            f'model = "{model}"\n'
            'model_reasoning_effort = "high"\n'
            f'model_instructions_file = "{base}"\n'
        ).encode()
        if not profile.is_file() or profile.read_bytes() != expected_profile:
            raise ProviderError("profile_drift", "application Codex role profile changed")
        if not (home / "auth.json").is_file():
            raise ProviderError("authentication_missing", "application Codex login is unavailable")

    def _arguments(self, role: str, model: str, cwd: Path, schema: Path) -> list[str]:
        profile = f"{role}-{model.removeprefix('gpt-6-')}"
        return [
            str(self._binary()),
            "exec",
            "-p",
            profile,
            "--ephemeral",
            "--json",
            "--output-schema",
            str(schema),
            "--skip-git-repo-check",
            "--ignore-rules",
            "-C",
            str(cwd),
            "-s",
            "read-only",
            "--color",
            "never",
            "-",
        ]

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def cancel(self, request_id: str) -> None:
        """Interrupt an active request or reserve rejection before it starts.

        Under the instance lock, mark a reserved ID as cancelled or retain a
        pending cancellation for provider_timeout_seconds + 15 seconds. Pending
        reservations survive rejected starts until expiry. Cancel another active
        inference task and wait for it to settle; its cleanup terminates any CLI
        process group and removes its private scratch directory.

        Args:
            request_id (str): Identifier matching a generate() request. The method
                itself does not validate its format.

        Returns:
            None: Cancellation has been recorded and any other registered task
            has settled. Calling repeatedly for an existing pending ID does not
            extend its expiry. For an unknown or completed ID, this changes
            future admission rather than proving a process was interrupted.
            Already transmitted task data and consumed tokens cannot be retracted.

        Raises:
            ProviderError: A new pending reservation would exceed the 4096-ID cap.
        """

        async with self.lock:
            if request_id in self.reserved:
                self.cancelled.add(request_id)
            else:
                now = time.monotonic()
                self.pending_cancellations = {
                    key: expires_at
                    for key, expires_at in self.pending_cancellations.items()
                    if expires_at > now
                }
                if request_id not in self.pending_cancellations:
                    if len(self.pending_cancellations) >= MAX_PENDING_CANCELLATIONS:
                        raise ProviderError(
                            "cancellation_capacity", "pending cancellation limit reached"
                        )
                    self.pending_cancellations[request_id] = (
                        now + self.settings.provider_timeout_seconds + 15
                    )
            task = self.tasks.get(request_id)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.wait({task})

    async def _read_stderr(self, process: asyncio.subprocess.Process) -> str:
        chunks = []
        total = 0
        while chunk := await process.stderr.read(8192):
            total += len(chunk)
            if total > MAX_STDERR_BYTES:
                raise ProviderError("stderr_too_large", "Codex stderr exceeded limit")
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def _stderr_category(self, stderr: str) -> str:
        lower = stderr.lower()
        for category, markers in (
            ("rate_limit", ("rate limit", "too many requests", "429")),
            ("authentication", ("unauthorized", "authentication", "login", "401")),
            ("permission", ("permission denied", "forbidden", "403")),
            ("network", ("connection", "network", "dns", "502", "503")),
            ("model", ("model", "404")),
            ("schema", ("schema", "json")),
        ):
            if any(marker in lower for marker in markers):
                return category
        return "other" if stderr else "none"

    async def _read_events(
        self, process: asyncio.subprocess.Process
    ) -> tuple[list[dict], dict, dict]:
        events: list[dict] = []
        final: dict | None = None
        usage: dict = {}
        completed = False
        total = 0
        while True:
            try:
                line = await process.stdout.readline()
            except ValueError as exc:
                raise ProviderError(
                    "events_too_large",
                    "Codex event line exceeded limit",
                    events,
                    usage=usage,
                ) from exc
            if not line:
                break
            total += len(line)
            if total > MAX_EVENT_BYTES or len(line) > MAX_EVENT_BYTES:
                raise ProviderError(
                    "events_too_large", "Codex event stream exceeded limit", events, usage=usage
                )
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProviderError(
                    "invalid_event", "Codex emitted invalid JSONL", events, usage=usage
                ) from exc
            if not isinstance(event, dict):
                raise ProviderError(
                    "invalid_event", "Codex emitted non-object event", events, usage=usage
                )
            events.append(event)
            event_type = event.get("type")
            item = event.get("item", {})
            if event_type in {"item.started", "item.updated", "item.completed"}:
                item_type = item.get("type") if isinstance(item, dict) else None
                if item_type not in {"agent_message", "error"}:
                    raise ProviderError(
                        "unexpected_tool_use",
                        f"Codex item type {item_type}",
                        events,
                        usage=usage,
                    )
                if event_type == "item.completed" and item_type == "agent_message":
                    if final is not None:
                        raise ProviderError(
                            "multiple_final_messages",
                            "Codex emitted multiple messages",
                            events,
                            usage=usage,
                        )
                    try:
                        text = item.get("text")
                        if not isinstance(text, str):
                            raise TypeError("Codex final message text is not a string")
                        final = json.loads(text)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ProviderError(
                            "invalid_output",
                            "Codex final message is not JSON",
                            events,
                            usage=usage,
                        ) from exc
            elif event_type == "turn.completed":
                if completed:
                    raise ProviderError(
                        "multiple_turn_completions",
                        "Codex completed twice",
                        events,
                        usage=usage,
                    )
                completed = True
                usage = event.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ProviderError("invalid_event", "Codex usage is invalid", events)
            elif event_type in {"turn.failed", "turn.interrupted"}:
                raise ProviderError(
                    "turn_failed",
                    f"Codex {event_type}",
                    events,
                    usage=(event.get("usage") or usage)
                    if isinstance(event.get("usage"), dict)
                    else usage,
                )
            elif event_type not in {"thread.started", "turn.started", "error"}:
                raise ProviderError(
                    "unexpected_event", f"Codex event type {event_type}", events, usage=usage
                )
        if not completed or final is None or not isinstance(final, dict):
            raise ProviderError(
                "incomplete_turn",
                "Codex did not return a final structured output",
                events,
                usage=usage,
            )
        return events, final, usage

    async def _complete(
        self, process: asyncio.subprocess.Process
    ) -> tuple[list[dict], dict, dict, str]:
        stderr_task = asyncio.create_task(self._read_stderr(process))
        events_task = asyncio.create_task(self._read_events(process))
        try:
            await asyncio.wait({stderr_task, events_task}, return_when=asyncio.FIRST_EXCEPTION)
            for task in (stderr_task, events_task):
                if not task.done():
                    continue
                error = task.exception()
                if error is not None:
                    await self._terminate(process)
                    if isinstance(error, ProviderError):
                        error.exit_code = process.returncode
                        if task is events_task:
                            try:
                                error.diagnostic = self._stderr_category(await stderr_task)
                            except ProviderError:
                                error.diagnostic = "stderr_too_large"
                        else:
                            try:
                                error.usage = (await events_task)[2]
                            except ProviderError as event_error:
                                error.usage = event_error.usage
                    raise error
            events, output, usage = await events_task
            await process.wait()
            stderr = await stderr_task
            return events, output, usage, stderr
        finally:
            for task in (stderr_task, events_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stderr_task, events_task, return_exceptions=True)

    async def generate(
        self,
        payload: dict,
        role: str = "answer",
        model: str = "gpt-6-sol",
        request_id: str = "",
        final_only: bool = False,
    ) -> dict:
        """Generate one role-valid result from a bounded ephemeral CLI session.

        Admit the request, reserve its ID, verify the role configuration, and start
        a fresh CLI process in a private temporary directory. Accept exactly one
        structured message and one completed turn, rejecting observed tool use.
        Validate the role output before returning; terminate subprocess work and
        remove the temporary directory on every exit. The adapter has no retry
        loop. Finished IDs can be reused for new inference and usage; reservations
        prevent only overlapping calls and pending cancellation, not durable
        replay or billing duplication.

        Args:
            payload (dict): Untrusted JSON task packet. Answer packets contain
                question, history, evidence.spans and coverage/tool context;
                task_mode="query_planning" uses the same answer schema. Judge
                packets contain candidate, gold, source excerpts and rubric_version.
                Admission checks transport shape/size, not these semantic fields.
            role (str): "answer" or "judge".
            model (str): gpt-6-sol for either role, or gpt-6-luna for answer only.
            request_id (str): Valid nonempty ID used to reserve work and correlate
                cancellation. An active ID or unexpired cancellation is rejected.
            final_only (bool): Require an answer action rather than request_tools
                for the answer role; judge output always uses its own verdict schema.

        Returns:
            dict: output (validated role mapping), usage (reported CLI token
            counts, possibly empty), elapsed_ms (integer milliseconds from
            admission), provider="codex", model, role, instruction_hash,
            billing_mode="subscription", cost_state="unknown", raw_visible_events
            (ordered CLI event list), input_packet (original payload), stderr (str)
            and effective_capabilities (configured tools disabled, no observed
            runtime tool use, no OS filesystem isolation and pinned CLI version).
            Token usage does not state a subscription charge; monetary cost is
            unknown. Output status describes evidence support, not process success.

        Raises:
            ProviderError: Admission/configuration fails, the ID is active or
                cancelled, timeout occurs, process/event bounds or completion
                rules fail, or output violates its role contract. Explicit cancel()
                becomes code="cancelled"; available events/usage/diagnostics are
                retained on failures that captured them, but may be absent.
            asyncio.CancelledError: External task cancellation not marked by
                cancel(); subprocess cleanup still runs. Work already sent to the
                provider may have consumed tokens even without reported usage.
            OSError: Instructions/schema cannot be read or process/scratch work
                fails at the operating-system boundary.
            ValueError: The installed role output schema differs from code.
        """

        check_request(payload, role, model, request_id)
        started = time.monotonic()
        async with self.lock:
            self.pending_cancellations = {
                key: expires_at
                for key, expires_at in self.pending_cancellations.items()
                if expires_at > started
            }
            if request_id in self.pending_cancellations:
                raise ProviderError("cancelled", "request was cancelled before start")
            if request_id in self.reserved:
                raise ProviderError("duplicate_request_id", "request_id is already active")
            self.reserved.add(request_id)
            self.tasks[request_id] = asyncio.current_task()
        try:
            try:
                async with asyncio.timeout(self.settings.provider_timeout_seconds):
                    return await self._generate_reserved(
                        payload, role, model, request_id, final_only, started
                    )
            except TimeoutError as exc:
                raise ProviderError("timeout", "Codex inference timed out") from exc
        except asyncio.CancelledError as exc:
            if request_id in self.cancelled:
                raise ProviderError("cancelled", "request was cancelled") from exc
            raise
        finally:
            async with self.lock:
                self.active.pop(request_id, None)
                self.tasks.pop(request_id, None)
                self.cancelled.discard(request_id)
                self.reserved.discard(request_id)

    async def _generate_reserved(
        self, payload, role, model, request_id, final_only, started
    ) -> dict:
        schema = load_schema(role)
        await self._check_configuration(role, model)
        prompt = prompt_for(payload, role, final_only)
        async with self.lock:
            if request_id in self.cancelled:
                raise ProviderError("cancelled", "request was cancelled before start")
        scratch = Path(tempfile.mkdtemp(prefix="pfl-provider-"))
        process = None
        events: list[dict] = []
        try:
            agents, _ = instruction_paths(role)
            (scratch / "AGENTS.md").write_bytes(agents.read_bytes())
            (scratch / ".role-root").write_text("", encoding="utf-8")
            schema_file = scratch / "output-schema.json"
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            os.chmod(scratch, 0o700)
            process = await asyncio.create_subprocess_exec(
                *self._arguments(role, model, scratch, schema_file),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._environment(),
                cwd=scratch,
                start_new_session=True,
                limit=MAX_EVENT_BYTES + 1,
            )
            async with self.lock:
                self.active[request_id] = process
                already_cancelled = request_id in self.cancelled
            if already_cancelled:
                await self._terminate(process)
                raise ProviderError("cancelled", "request was cancelled")
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            events, output, usage, stderr = await self._complete(process)
            if request_id in self.cancelled:
                raise ProviderError("cancelled", "request was cancelled", events, usage=usage)
            if process.returncode != 0:
                raise ProviderError(
                    "process_failed",
                    "Codex process failed",
                    events,
                    usage=usage,
                    diagnostic=self._stderr_category(stderr),
                    exit_code=process.returncode,
                )
            try:
                validated = validate_output(role, output, final_only=final_only)
            except (ValidationError, ValueError) as exc:
                raise ProviderError(
                    "invalid_output",
                    "Codex output violates role schema",
                    events,
                    usage=usage,
                ) from exc
            return {
                "output": validated,
                "usage": usage,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "provider": "codex",
                "model": model,
                "role": role,
                "instruction_hash": instruction_hash(role),
                "billing_mode": "subscription",
                "cost_state": "unknown",
                "raw_visible_events": events,
                "input_packet": payload,
                "stderr": stderr,
                "effective_capabilities": {
                    "configured_tools": "disabled",
                    "runtime_tool_use_observed": False,
                    "os_filesystem_isolation": False,
                    "cli_version": PINNED_VERSION,
                },
            }
        except BaseException as exc:
            if process is not None:
                await self._terminate(process)
            if request_id in self.cancelled and not isinstance(exc, asyncio.CancelledError):
                if isinstance(exc, ProviderError):
                    raise ProviderError(
                        "cancelled",
                        "request was cancelled",
                        exc.events or events,
                        usage=exc.usage,
                        raw_provider_usage=exc.raw_provider_usage,
                        response_status=exc.response_status,
                        response_reason=exc.response_reason,
                        diagnostic=exc.diagnostic,
                        exit_code=exc.exit_code,
                    ) from exc
                raise ProviderError("cancelled", "request was cancelled", events) from exc
            raise
        finally:
            async with self.lock:
                self.active.pop(request_id, None)
            shutil.rmtree(scratch)
