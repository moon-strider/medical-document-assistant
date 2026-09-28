from .api import APIProvider
from .bridge_client import BridgeProvider
from .codex import CodexProvider
from .fixture import FixtureProvider


def get_provider(settings):
    """Select the configured inference and billing boundary without fallback.

    Args:
        settings (Settings): Provider configuration. provider="codex" selects
            the host bridge when bridge_url is nonempty and a local CLI otherwise;
            "api" selects Responses API and "fixture" selects explicit fixtures.

    Returns:
        CodexProvider | BridgeProvider | APIProvider | FixtureProvider: A new
        configured adapter. Construction does not perform
        inference or verify readiness. A live-provider failure never switches
        this selection or its billing mode automatically.

    Raises:
        ValueError: settings.provider names an unsupported adapter.
    """

    if settings.provider == "codex":
        if settings.bridge_url:
            return BridgeProvider(settings)
        return CodexProvider(settings)
    if settings.provider == "api":
        return APIProvider(settings)
    if settings.provider == "fixture":
        return FixtureProvider(settings)
    raise ValueError(f"unsupported provider: {settings.provider}")
