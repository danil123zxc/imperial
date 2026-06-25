from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import uuid


APP_TITLE = "Imperial RAG"
PREVIEW_UNAVAILABLE_TEXT = "Preview is unavailable for this file."
AUTH_SESSION_EMAIL_KEY = "auth_user_email"
FILE_PREVIEW_CHAR_LIMIT = 12_000
FILE_DOWNLOAD_BYTE_LIMIT = 50 * 1024 * 1024
_RUNTIME_CACHE_WRAPPER: Any | None = None
_RUNTIME_CACHE_RESOURCE: Any | None = None


@dataclass(frozen=True)
class RetrievedFileGroup:
    file_key: str
    file_name: str
    display_path: str
    download_path: Path | None
    download_name: str
    download_mime: str
    preview_text: str
    can_download: bool


def build_status_summary(total_files: int, indexed_files: int, failed_files: int) -> str:
    return "\n".join(
        [
            f"Total files: {total_files}",
            f"Indexed files: {indexed_files}",
            f"Failed files: {failed_files}",
        ]
    )


def load_status_summary(settings: Any | None = None) -> str:
    try:
        from imperial_rag.config import Settings
        from imperial_rag.ingestion.manifest import FileStatus, ManifestStore
    except ImportError:
        return build_status_summary(total_files=0, indexed_files=0, failed_files=0)

    resolved_settings = settings or Settings()
    if not hasattr(resolved_settings, "manifest_db_path"):
        return build_status_summary(total_files=0, indexed_files=0, failed_files=0)
    manifest_path = Path(resolved_settings.manifest_db_path)
    if not manifest_path.exists():
        return build_status_summary(total_files=0, indexed_files=0, failed_files=0)
    with ManifestStore(manifest_path) as manifest_store:
        records = manifest_store.list_records()
    indexed = sum(1 for record in records if record.status == FileStatus.INDEXED)
    failed = sum(1 for record in records if record.status == FileStatus.FAILED)
    return build_status_summary(total_files=len(records), indexed_files=indexed, failed_files=failed)


def query_runtime(settings: Any, question: str) -> dict[str, Any]:
    try:
        from imperial_rag.answering.runtime import create_runtime as maybe_create_runtime
    except (ImportError, AttributeError):
        maybe_create_runtime = None
    if maybe_create_runtime is not None:
        return _coerce_result(_runtime_resource(settings).query(question))

    try:
        from imperial_rag.answering.runtime import Runtime as maybe_runtime_class
    except (ImportError, AttributeError):
        maybe_runtime_class = None

    if maybe_runtime_class is not None:
        return _coerce_result(maybe_runtime_class(settings=settings).query(question))

    from imperial_rag.answering.runtime import build_live_query_workflow

    return _coerce_result(build_live_query_workflow(settings).invoke({"question": question}))


def _runtime_resource(settings: Any) -> Any:
    streamlit_module: Any | None
    try:
        import streamlit as streamlit_module
    except ImportError:
        streamlit_module = None

    cache_resource = getattr(streamlit_module, "cache_resource", None) if streamlit_module is not None else None
    if cache_resource is None:
        from imperial_rag.answering.runtime import create_runtime

        return create_runtime(settings)

    global _RUNTIME_CACHE_WRAPPER, _RUNTIME_CACHE_RESOURCE
    if _RUNTIME_CACHE_WRAPPER is None or _RUNTIME_CACHE_RESOURCE is not cache_resource:
        _RUNTIME_CACHE_WRAPPER = cache_resource(_create_cached_runtime)
        _RUNTIME_CACHE_RESOURCE = cache_resource
    wrapper = _RUNTIME_CACHE_WRAPPER
    if wrapper is None:
        raise RuntimeError("Streamlit runtime cache wrapper was not initialized.")
    return wrapper(_runtime_cache_key(settings), settings)


def _create_cached_runtime(cache_key: tuple[Any, ...], _settings: Any) -> Any:
    from imperial_rag.answering.runtime import create_runtime

    return create_runtime(_settings)


def _runtime_cache_key(settings: Any) -> tuple[Any, ...]:
    env_names = (
        "DASHSCOPE_API_KEY",
        "IMPERIAL_RAG_QWEN_CHAT_MODEL",
        "IMPERIAL_RAG_QWEN_EMBEDDING_MODEL",
        "IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSION",
        "IMPERIAL_RAG_QWEN_RERANK_MODEL",
        "IMPERIAL_RAG_PRIMARY_RERANKER",
        "IMPERIAL_RAG_FALLBACK_RERANKER",
        "IMPERIAL_RAG_VECTOR_FETCH_K",
        "IMPERIAL_RAG_VECTOR_K",
        "IMPERIAL_RAG_KEYWORD_LIMIT",
        "IMPERIAL_RAG_RERANK_INPUT_LIMIT",
        "IMPERIAL_RAG_RERANK_TOP_N",
        "IMPERIAL_RAG_MMR_LAMBDA_MULT",
        "IMPERIAL_RAG_RRF_K",
    )
    return (
        str(getattr(settings, "workspace_root", "")),
        str(getattr(settings, "qdrant_url", "")),
        str(getattr(settings, "qdrant_collection", "")),
        str(getattr(settings, "elasticsearch_url", "")),
        str(getattr(settings, "elasticsearch_index", "")),
        tuple((name, _cache_safe_env_value(name)) for name in env_names),
    )


def _cache_safe_env_value(name: str) -> str | bool:
    value = os.environ.get(name, "")
    if "KEY" in name or "TOKEN" in name or "SECRET" in name:
        return bool(value.strip())
    return value.strip()


def build_retrieved_file_groups(evidence: list[Any], settings: Any) -> list[RetrievedFileGroup]:
    documents_root = Path(getattr(settings, "documents_root", Path.cwd())).resolve()
    builders: dict[str, dict[str, Any]] = {}

    for index, document in enumerate(evidence or []):
        metadata = dict(getattr(document, "metadata", {}) or {})
        file_key = _file_group_key(metadata, index, documents_root)
        if file_key not in builders:
            file_name = _file_name(metadata, index)
            download_path = _safe_download_path(metadata, documents_root)
            builders[file_key] = {
                "file_key": file_key,
                "file_name": file_name,
                "display_path": _display_path(metadata, documents_root, download_path, file_name),
                "download_path": download_path,
                "download_name": file_name,
                "download_mime": _download_mime(file_name),
                "preview_text": _file_preview_text(metadata, settings),
            }

    groups: list[RetrievedFileGroup] = []
    for builder in builders.values():
        download_path = builder["download_path"]
        groups.append(
            RetrievedFileGroup(
                file_key=builder["file_key"],
                file_name=builder["file_name"],
                display_path=builder["display_path"],
                download_path=download_path,
                download_name=builder["download_name"],
                download_mime=builder["download_mime"],
                preview_text=builder["preview_text"],
                can_download=download_path is not None,
            )
        )
    return groups


def normalize_retrieved_file_groups(groups: list[Any], settings: Any) -> list[RetrievedFileGroup]:
    documents_root = Path(getattr(settings, "documents_root", Path.cwd())).resolve()
    builders: dict[str, RetrievedFileGroup] = {}

    for index, group in enumerate(groups or []):
        file_key = _stored_group_key(group, index, documents_root)
        normalized = _coerce_retrieved_file_group(group, file_key, settings)
        existing = builders.get(file_key)
        if existing is None:
            builders[file_key] = normalized
            continue
        builders[file_key] = _merge_file_groups(existing, normalized)

    return list(builders.values())


def main() -> None:
    _ensure_src_on_path()

    from imperial_rag.env import load_project_env

    load_project_env()

    import streamlit as st

    from imperial_rag.config import Settings

    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    settings = Settings()
    from imperial_rag.observability import configure_observability
    from imperial_rag.tracing import configure_phoenix_tracing, phoenix_trace_context

    configure_observability(settings)
    configure_phoenix_tracing(settings)

    auth_store = _prepare_auth_store(settings)
    current_user = _current_authenticated_user(st, auth_store)
    if current_user is None:
        _render_auth_gate(st, auth_store)
        return

    with st.sidebar:
        st.caption(f"Signed in as {current_user.email}")
        if st.button("Log out", key="auth-logout", icon=":material/logout:"):
            st.session_state.pop(AUTH_SESSION_EMAIL_KEY, None)
            st.session_state.pop("messages", None)
            _rerun(st)
            return
        if current_user.is_admin:
            _render_admin_access_panel(st, auth_store, current_user)
        st.header("Ingestion status")
        st.text(load_status_summary(settings))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "phoenix_trace_session_id" not in st.session_state:
        st.session_state.phoenix_trace_session_id = str(uuid.uuid4())

    for message_index, message in enumerate(st.session_state.messages):
        _render_chat_message(st, message, message_index, settings)

    question = st.chat_input("Ask about the indexed documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    from imperial_rag.tracing import trace_user_id_from_email

    user_hash = trace_user_id_from_email(current_user.email)
    phoenix_session_id = st.session_state.phoenix_trace_session_id
    started_at = perf_counter()
    try:
        with phoenix_trace_context(
            phoenix_session_id,
            user_id=user_hash,
            metadata={"entrypoint": "streamlit"},
            tags=["imperial-rag", "streamlit"],
        ):
            result = query_runtime(settings, question)
    except Exception as exc:
        from imperial_rag.observability import log_failure

        log_failure(
            "web_query",
            exc,
            component="streamlit",
            duration_ms=_duration_ms(started_at),
            phoenix_session_id=phoenix_session_id,
            session_id=phoenix_session_id,
            user_hash=user_hash,
        )
        with st.chat_message("assistant"):
            st.error("Something went wrong while answering. Check local logs for details.")
        return
    from imperial_rag.observability import log_event

    log_event(
        "imperial_rag.web_query",
        operation="web_query",
        status="success",
        component="streamlit",
        duration_ms=_duration_ms(started_at),
        phoenix_session_id=phoenix_session_id,
        session_id=phoenix_session_id,
        user_hash=user_hash,
        **_query_log_fields(result),
    )
    answer = str(result.get("answer", ""))
    sources = result.get("sources") or result.get("citations") or []
    evidence = result.get("evidence") or result.get("retrieved_documents") or []
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "error": result.get("error"),
        "citations_valid": result.get("citations_valid"),
        "invalid_citations": result.get("invalid_citations") or [],
        "retrieved_files": build_retrieved_file_groups(evidence, settings),
    }
    st.session_state.messages.append(assistant_message)
    _render_chat_message(st, assistant_message, len(st.session_state.messages) - 1, settings)


def _prepare_auth_store(settings: Any) -> Any:
    from imperial_rag.app.auth import AuthStore

    auth_db_path = getattr(settings, "auth_db_path", None)
    if auth_db_path is None:
        processed_root = getattr(settings, "processed_root", Path.cwd() / ".imperial_rag")
        auth_db_path = Path(processed_root) / "auth.sqlite3"
    store = AuthStore(Path(auth_db_path))
    store.initialize()

    admin_email = os.environ.get("IMPERIAL_RAG_ADMIN_EMAIL", "").strip()
    admin_password = os.environ.get("IMPERIAL_RAG_ADMIN_PASSWORD", "")
    if admin_email and admin_password:
        store.bootstrap_admin(admin_email, admin_password)
    return store


def _current_authenticated_user(st: Any, auth_store: Any) -> Any | None:
    email = st.session_state.get(AUTH_SESSION_EMAIL_KEY)
    if not email:
        return None
    user = auth_store.get_user(str(email))
    if user is None or user.status != "approved":
        st.session_state.pop(AUTH_SESSION_EMAIL_KEY, None)
        return None
    return user


def _render_auth_gate(st: Any, auth_store: Any) -> None:
    from imperial_rag.app.auth import AuthenticationStatus

    st.subheader("Access required")
    mode = st.radio("Account action", ["Log in", "Sign up"], horizontal=True, key="auth-mode")

    if mode == "Log in":
        with st.form("auth-login-form"):
            email = st.text_input("Email", key="auth-login-email")
            password = st.text_input("Password", type="password", key="auth-login-password")
            submitted = st.form_submit_button("Log in", icon=":material/login:")
        if not submitted:
            return
        try:
            result = auth_store.authenticate(email, password)
        except ValueError as exc:
            st.error(str(exc))
            return
        if result.status == AuthenticationStatus.AUTHENTICATED and result.user is not None:
            st.session_state[AUTH_SESSION_EMAIL_KEY] = result.user.email
            _rerun(st)
            return
        if result.status == AuthenticationStatus.PENDING_APPROVAL:
            st.warning("Your access request is waiting for admin approval.")
            return
        if result.status == AuthenticationStatus.REJECTED:
            st.error("This access request was rejected.")
            return
        st.error("Invalid email or password.")
        return

    with st.form("auth-signup-form"):
        full_name = st.text_input("Full name", key="auth-signup-full-name")
        email = st.text_input("Email", key="auth-signup-email")
        password = st.text_input("Password", type="password", key="auth-signup-password")
        reason = st.text_area("Access reason", key="auth-signup-reason")
        submitted = st.form_submit_button("Request access", icon=":material/how_to_reg:")
    if not submitted:
        return
    try:
        user = auth_store.register_user(email=email, password=password, full_name=full_name, reason=reason)
    except ValueError as exc:
        st.error(str(exc))
        return
    if user.status == "approved":
        st.info("This account already has access. Log in to continue.")
        return
    st.success("Registration submitted. An admin can grant access from the access requests panel.")


def _render_admin_access_panel(st: Any, auth_store: Any, current_user: Any) -> None:
    pending_users = auth_store.list_pending_users()
    if not pending_users:
        return

    request_label = (
        "1 pending access request" if len(pending_users) == 1 else f"{len(pending_users)} pending access requests"
    )
    st.warning(request_label)
    for pending_user in pending_users:
        with st.container(border=True):
            st.markdown(f"**{pending_user.full_name or pending_user.email}**")
            st.caption(pending_user.email)
            if pending_user.reason:
                st.caption(pending_user.reason)
            if st.button(
                "Grant access",
                key=f"auth-approve-{pending_user.email}",
                icon=":material/check:",
                width="stretch",
            ):
                auth_store.approve_user(current_user.email, pending_user.email)
                st.success(f"Granted access to {pending_user.email}")
                _rerun(st)


def _rerun(st: Any) -> None:
    rerun = getattr(st, "rerun", None)
    if rerun is not None:
        rerun()


def _coerce_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "answer": getattr(result, "answer", ""),
        "sources": getattr(result, "sources", getattr(result, "citations", [])),
        "evidence": getattr(result, "evidence", getattr(result, "retrieved_documents", [])),
    }


def _query_log_fields(result: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    retrieval = result.get("retrieval") if isinstance(result, dict) else getattr(result, "retrieval", None)
    if isinstance(retrieval, dict):
        for key in (
            "final_evidence",
            "vector_candidates",
            "keyword_candidates",
            "merged_candidates",
            "rerank_input_candidates",
            "reranked_candidates",
            "reranker",
        ):
            if key in retrieval:
                fields[key] = retrieval[key]
        fallbacks = retrieval.get("fallbacks")
        if isinstance(fallbacks, list):
            fields["fallback_count"] = len(fallbacks)
    evidence = result.get("evidence") or result.get("retrieved_documents") if isinstance(result, dict) else None
    if evidence is not None and "final_evidence" not in fields:
        try:
            fields["final_evidence"] = len(evidence)
        except TypeError:
            pass
    error = result.get("error") if isinstance(result, dict) else None
    if isinstance(error, dict):
        fields["error_type"] = str(error.get("type") or "")
    return fields


def _duration_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _render_chat_message(st: Any, message: dict[str, Any], message_index: int, settings: Any) -> None:
    with st.chat_message(message["role"]):
        if _message_has_error(message):
            st.error(message["content"])
        else:
            st.write(message["content"])
        if message.get("citations_valid") is False:
            st.warning("Answer citations could not be verified. Treat this response as diagnostic.")
        retrieved_files = message.get("retrieved_files") or []
        if retrieved_files:
            _render_retrieved_files(st, normalize_retrieved_file_groups(retrieved_files, settings), message_index)
            return
        for source in message.get("sources", []):
            st.caption(str(source))


def _render_retrieved_files(st: Any, groups: list[RetrievedFileGroup], message_index: int) -> None:
    st.markdown("**Retrieved files**")
    for group_index, group in enumerate(groups):
        with st.container(border=True):
            info_col, download_col = st.columns([5, 1])
            with info_col:
                st.markdown(f"**{group.file_name}**")
                st.caption(group.display_path)
            with download_col:
                data, disabled = _download_button_payload(group)
                st.download_button(
                    "Download",
                    data=data,
                    file_name=group.download_name,
                    mime=group.download_mime,
                    key=f"download-{message_index}-{group_index}",
                    disabled=disabled,
                    width="stretch",
                    icon=":material/download:",
                )
            with st.expander("Preview"):
                st.text(group.preview_text)


def _download_button_payload(group: RetrievedFileGroup) -> tuple[bytes, bool]:
    if not group.can_download or group.download_path is None:
        return b"", True
    try:
        if group.download_path.stat().st_size > FILE_DOWNLOAD_BYTE_LIMIT:
            return b"", True
        return group.download_path.read_bytes(), False
    except OSError:
        return b"", True


def _file_group_key(metadata: dict[str, Any], index: int, documents_root: Path) -> str:
    relative_path = _metadata_relative_path(metadata, documents_root)
    if relative_path:
        return f"relative_path:{relative_path}"
    file_id = metadata.get("file_id")
    if file_id:
        return f"file_id:{file_id}"
    file_name = metadata.get("file_name")
    if file_name:
        return f"file_name:{file_name}"
    return f"unknown:{index}"


def _file_name(metadata: dict[str, Any], index: int) -> str:
    for field in ("file_name", "relative_path", "file_path"):
        value = metadata.get(field)
        if value:
            name = Path(str(value)).name
            if name:
                return name
    return f"retrieved-source-{index + 1}"


def _display_path(
    metadata: dict[str, Any],
    documents_root: Path,
    download_path: Path | None,
    file_name: str,
) -> str:
    relative_path = metadata.get("relative_path")
    if relative_path:
        return _shorten_path(_normalize_relative_path(relative_path))
    if download_path is not None:
        try:
            return _shorten_path(download_path.relative_to(documents_root).as_posix())
        except ValueError:
            return file_name
    parent_folder = metadata.get("parent_folder")
    if parent_folder:
        return _shorten_path(f"{parent_folder}/{file_name}")
    return file_name


def _metadata_relative_path(metadata: dict[str, Any], documents_root: Path) -> str | None:
    relative_path = metadata.get("relative_path")
    if relative_path:
        return _normalize_relative_path(relative_path)

    file_path = metadata.get("file_path")
    if not file_path:
        return None
    try:
        resolved = Path(str(file_path)).expanduser().resolve(strict=False)
        return resolved.relative_to(documents_root).as_posix()
    except (OSError, ValueError):
        return None


def _normalize_relative_path(path: Any) -> str:
    normalized = str(path).replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _shorten_path(path: str, max_parts: int = 3) -> str:
    parts = Path(path).parts
    if len(parts) <= max_parts:
        return path
    return str(Path(parts[0]) / "..." / Path(*parts[-(max_parts - 1) :]))


def _safe_download_path(metadata: dict[str, Any], documents_root: Path) -> Path | None:
    candidates: list[Path] = []
    file_path = metadata.get("file_path")
    if file_path:
        candidates.append(Path(str(file_path)))
    relative_path = metadata.get("relative_path")
    if relative_path:
        candidates.append(documents_root / str(relative_path))

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=False)
        except OSError:
            continue
        if not _is_within(resolved, documents_root):
            continue
        if resolved.is_file():
            return resolved
    return None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _download_mime(file_name: str) -> str:
    return mimetypes.guess_type(file_name)[0] or "application/octet-stream"


def _file_preview_text(metadata: dict[str, Any], settings: Any) -> str:
    file_id = metadata.get("file_id")
    if not file_id:
        return PREVIEW_UNAVAILABLE_TEXT
    return _file_preview_text_by_id(str(file_id), settings)


def _file_preview_text_by_id(file_id: str, settings: Any) -> str:
    extraction_root = _extraction_root(settings)
    if extraction_root is None or not _safe_artifact_id(file_id):
        return PREVIEW_UNAVAILABLE_TEXT
    artifact_path = extraction_root / "documents" / f"{file_id}.json"
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PREVIEW_UNAVAILABLE_TEXT

    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        return PREVIEW_UNAVAILABLE_TEXT
    parts: list[str] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        text = str(document.get("page_content") or "").strip()
        if text:
            parts.append(text)
    return _bounded_preview(parts)


def _bounded_preview(parts: list[str]) -> str:
    if not parts:
        return PREVIEW_UNAVAILABLE_TEXT
    preview = "\n\n".join(parts)
    if len(preview) <= FILE_PREVIEW_CHAR_LIMIT:
        return preview
    return f"{preview[:FILE_PREVIEW_CHAR_LIMIT].rstrip()}..."


def _message_has_error(message: dict[str, Any]) -> bool:
    error = message.get("error")
    return isinstance(error, dict) and bool(error.get("type"))


def _extraction_root(settings: Any) -> Path | None:
    extraction_root = getattr(settings, "extraction_root", None)
    if extraction_root:
        return Path(extraction_root)
    processed_root = getattr(settings, "processed_root", None)
    if processed_root:
        return Path(processed_root) / "extracted"
    return None


def _safe_artifact_id(file_id: str) -> bool:
    return bool(file_id) and "/" not in file_id and "\\" not in file_id and Path(file_id).name == file_id


def _stored_group_key(group: Any, index: int, documents_root: Path) -> str:
    download_path = getattr(group, "download_path", None)
    if download_path is not None:
        try:
            resolved = Path(download_path).expanduser().resolve(strict=False)
            return f"relative_path:{resolved.relative_to(documents_root).as_posix()}"
        except (OSError, ValueError):
            pass

    file_key = str(getattr(group, "file_key", "") or "")
    if file_key.startswith("relative_path:"):
        return f"relative_path:{_normalize_relative_path(file_key.removeprefix('relative_path:'))}"
    if file_key.startswith("file_path:"):
        return _file_group_key({"file_path": file_key.removeprefix("file_path:")}, index, documents_root)
    if file_key:
        return file_key

    file_name = getattr(group, "file_name", None)
    if file_name:
        return f"file_name:{file_name}"
    return f"unknown:{index}"


def _coerce_retrieved_file_group(group: Any, file_key: str, settings: Any) -> RetrievedFileGroup:
    file_name = str(getattr(group, "file_name", "") or "retrieved-source")
    download_path = getattr(group, "download_path", None)
    if download_path is not None:
        download_path = Path(download_path)
    preview_text = str(getattr(group, "preview_text", "") or "")
    if not preview_text:
        preview_text = _stored_group_preview_text(group, settings)
    return RetrievedFileGroup(
        file_key=file_key,
        file_name=file_name,
        display_path=str(getattr(group, "display_path", "") or file_name),
        download_path=download_path,
        download_name=str(getattr(group, "download_name", "") or file_name),
        download_mime=str(getattr(group, "download_mime", "") or _download_mime(file_name)),
        preview_text=preview_text,
        can_download=bool(getattr(group, "can_download", False)) and download_path is not None,
    )


def _stored_group_preview_text(group: Any, settings: Any) -> str:
    file_key = str(getattr(group, "file_key", "") or "")
    if file_key.startswith("file_id:"):
        return _file_preview_text_by_id(file_key.removeprefix("file_id:"), settings)
    return PREVIEW_UNAVAILABLE_TEXT


def _merge_file_groups(existing: RetrievedFileGroup, duplicate: RetrievedFileGroup) -> RetrievedFileGroup:
    preview_text = existing.preview_text
    if preview_text == PREVIEW_UNAVAILABLE_TEXT and duplicate.preview_text != PREVIEW_UNAVAILABLE_TEXT:
        preview_text = duplicate.preview_text
    download_path = existing.download_path or duplicate.download_path
    return RetrievedFileGroup(
        file_key=existing.file_key,
        file_name=existing.file_name,
        display_path=existing.display_path,
        download_path=download_path,
        download_name=existing.download_name,
        download_mime=existing.download_mime,
        preview_text=preview_text,
        can_download=existing.can_download or duplicate.can_download,
    )


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
