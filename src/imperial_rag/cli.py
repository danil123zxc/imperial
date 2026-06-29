from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any


def load_project_environment(workspace_root: Path | None) -> None:
    from imperial_rag.env import load_project_env

    load_project_env(workspace_root)


def build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    try:
        return Settings(workspace_root=workspace_root)
    except TypeError:
        os.environ["IMPERIAL_RAG_WORKSPACE_ROOT"] = str(workspace_root)
        return Settings()


def configure_observability(settings: Any) -> None:
    from imperial_rag.observability import configure_observability as configure

    configure(settings)


def configure_tracing(settings: Any, *, trace_phoenix: bool | None = None, enabled: bool | None = None) -> None:
    from imperial_rag.observability.phoenix import configure_phoenix_tracing

    if enabled is None and trace_phoenix is not None:
        enabled = True if trace_phoenix else None
    configure_phoenix_tracing(settings, enabled=enabled)


@contextmanager
def trace_context(session_id: str, *, entrypoint: str = "cli", tags: list[str] | None = None):
    from imperial_rag.observability.phoenix import phoenix_trace_context

    with phoenix_trace_context(
        session_id,
        metadata={"entrypoint": entrypoint},
        tags=tags or ["imperial-rag", "cli"],
    ):
        yield


def trace_session_id(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env_value = os.environ.get("IMPERIAL_RAG_TRACE_SESSION_ID", "").strip()
    if env_value:
        return env_value
    return f"cli_{uuid.uuid4()}"


def duration_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def log_failure(operation: str, exc: BaseException, started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_failure as emit_failure

    emit_failure(operation, exc, component="cli", duration_ms=duration_ms(started_at), **fields)
