# Inference providers

Provider adapters give the dialogue graph one inference contract across the application-scoped Codex CLI, its Docker host bridge, and the Responses API. The bridge keeps the authenticated subscription CLI on the host while allowing the container to request bounded inference and cancellation. Each CLI call starts a fresh process, which makes role configuration and cancellation explicit but adds startup cost and depends on bridge availability.

Readiness makes no model call and does not guarantee successful inference. The Responses API adapter requires separate credentials and has only offline contract validation. The fixture provider is for deterministic verification and is never a live fallback.

Provider selection is explicit: a failed live provider does not silently switch billing modes or fall back to a fixture.
