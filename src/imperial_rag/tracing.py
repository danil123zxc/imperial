from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

from imperial_rag.config import Settings


_CONFIGURED_PROVIDER: object | None = None
_CONFIGURED_KEY: tuple[str, str] | None = None


def configure_phoenix_tracing(settings: Settings | None = None, enabled: bool | None = None) -> object | None:
    """Configure Phoenix OpenTelemetry tracing once for the current process."""

    env_enabled = enabled is None
    if enabled is None:
        enabled = _env_flag("PHOENIX_TRACING_ENABLED") or _env_flag("IMPERIAL_RAG_TRACING_ENABLED")
    if not enabled:
        return None

    resolved_settings = settings or Settings()
    if env_enabled and not _collector_endpoint_reachable(resolved_settings.phoenix_collector_endpoint):
        return None

    key = (resolved_settings.phoenix_project_name, resolved_settings.phoenix_collector_endpoint)
    global _CONFIGURED_PROVIDER, _CONFIGURED_KEY
    if _CONFIGURED_PROVIDER is not None:
        if _CONFIGURED_KEY == key:
            return _CONFIGURED_PROVIDER
        raise RuntimeError(
            "Phoenix tracing is already configured for "
            f"project={_CONFIGURED_KEY[0]!r}, endpoint={_CONFIGURED_KEY[1]!r}; "
            f"cannot reconfigure to project={key[0]!r}, endpoint={key[1]!r} in the same process."
        )

    try:
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing dependencies are missing. Install arize-phoenix-otel and OpenInference instrumentors."
        ) from exc

    _CONFIGURED_PROVIDER = register(
        project_name=resolved_settings.phoenix_project_name,
        endpoint=resolved_settings.phoenix_collector_endpoint,
        auto_instrument=True,
        verbose=False,
    )
    _CONFIGURED_KEY = key
    return _CONFIGURED_PROVIDER


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _collector_endpoint_reachable(endpoint: str, timeout: float = 0.2) -> bool:
    parsed = urlparse(endpoint)
    if not parsed.hostname:
        return True
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def _reset_phoenix_tracing_for_tests() -> None:
    global _CONFIGURED_PROVIDER, _CONFIGURED_KEY
    _CONFIGURED_PROVIDER = None
    _CONFIGURED_KEY = None
