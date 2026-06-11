from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any


APP_TITLE = "Imperial RAG"
PREVIEW_UNAVAILABLE_TEXT = "Preview is unavailable for this file."


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
        from imperial_rag.manifest import FileStatus, ManifestStore
    except ImportError:
        return build_status_summary(total_files=0, indexed_files=0, failed_files=0)

    resolved_settings = settings or Settings()
    if not hasattr(resolved_settings, "manifest_db_path"):
        return build_status_summary(total_files=0, indexed_files=0, failed_files=0)
    manifest_path = Path(resolved_settings.manifest_db_path)
    if not manifest_path.exists():
        return build_status_summary(total_files=0, indexed_files=0, failed_files=0)
    records = ManifestStore(manifest_path).list_records()
    indexed = sum(1 for record in records if record.status == FileStatus.INDEXED)
    failed = sum(1 for record in records if record.status == FileStatus.FAILED)
    return build_status_summary(total_files=len(records), indexed_files=indexed, failed_files=failed)


def query_runtime(settings: Any, question: str) -> dict[str, Any]:
    try:
        from imperial_rag.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return _coerce_result(create_runtime(settings).query(question))

    try:
        from imperial_rag.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        return _coerce_result(Runtime(settings=settings).query(question))

    from imperial_rag.runtime import build_live_query_workflow

    return _coerce_result(build_live_query_workflow(settings).invoke({"question": question}))


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
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings)

    with st.sidebar:
        st.header("Ingestion status")
        st.text(load_status_summary(settings))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message_index, message in enumerate(st.session_state.messages):
        _render_chat_message(st, message, message_index, settings)

    question = st.chat_input("Ask about the indexed documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = query_runtime(settings, question)
    answer = str(result.get("answer", ""))
    sources = result.get("sources") or result.get("citations") or []
    evidence = result.get("evidence") or result.get("retrieved_documents") or []
    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieved_files": build_retrieved_file_groups(evidence, settings),
    }
    st.session_state.messages.append(assistant_message)
    _render_chat_message(st, assistant_message, len(st.session_state.messages) - 1, settings)


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


def _render_chat_message(st: Any, message: dict[str, Any], message_index: int, settings: Any) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
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
    return "\n\n".join(parts) if parts else PREVIEW_UNAVAILABLE_TEXT


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
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
