from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from pathlib import Path
import sys
from typing import Any


APP_TITLE = "Imperial RAG"
SOURCE_DETAIL_FIELDS = (
    "section_heading",
    "page_number",
    "sheet_name",
    "table_index",
    "row_range",
    "image_index",
    "embedded_media_name",
)


@dataclass(frozen=True)
class RetrievedSnippet:
    marker: str
    text: str
    source_type: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievedFileGroup:
    file_key: str
    file_name: str
    display_path: str
    download_path: Path | None
    download_name: str
    download_mime: str
    markers: tuple[str, ...]
    snippets: tuple[RetrievedSnippet, ...]
    can_download: bool

    @property
    def chunk_count(self) -> int:
        return len(self.snippets)


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
        file_key = _file_group_key(metadata, index)
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
                "markers": [],
                "snippets": [],
            }

        marker = f"[S{index + 1}]"
        builders[file_key]["markers"].append(marker)
        builders[file_key]["snippets"].append(
            RetrievedSnippet(
                marker=marker,
                text=str(getattr(document, "page_content", "")),
                source_type=str(metadata.get("source_type") or "unknown"),
                details=_snippet_details(metadata),
            )
        )

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
                markers=tuple(builder["markers"]),
                snippets=tuple(builder["snippets"]),
                can_download=download_path is not None,
            )
        )
    return groups


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
        _render_chat_message(st, message, message_index)

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
    _render_chat_message(st, assistant_message, len(st.session_state.messages) - 1)


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


def _render_chat_message(st: Any, message: dict[str, Any], message_index: int) -> None:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        retrieved_files = message.get("retrieved_files") or []
        if retrieved_files:
            _render_retrieved_files(st, retrieved_files, message_index)
            return
        for source in message.get("sources", []):
            st.caption(str(source))


def _render_retrieved_files(st: Any, groups: list[RetrievedFileGroup], message_index: int) -> None:
    st.markdown("**Retrieved files**")
    for group_index, group in enumerate(groups):
        with st.container(border=True):
            info_col, download_col = st.columns([5, 1])
            with info_col:
                marker_text = ", ".join(group.markers)
                st.markdown(f"**{group.file_name}**")
                st.caption(f"{group.display_path} | {group.chunk_count} chunks | {marker_text}")
            with download_col:
                data, disabled = _download_button_payload(group)
                st.download_button(
                    "Download",
                    data=data,
                    file_name=group.download_name,
                    mime=group.download_mime,
                    key=f"download-{message_index}-{group_index}",
                    disabled=disabled,
                    use_container_width=True,
                    icon=":material/download:",
                )
            with st.expander("Preview"):
                for snippet in group.snippets:
                    details = " | ".join((snippet.source_type, *snippet.details))
                    st.caption(f"{snippet.marker} {details}")
                    st.text(snippet.text)


def _download_button_payload(group: RetrievedFileGroup) -> tuple[bytes, bool]:
    if not group.can_download or group.download_path is None:
        return b"", True
    try:
        return group.download_path.read_bytes(), False
    except OSError:
        return b"", True


def _file_group_key(metadata: dict[str, Any], index: int) -> str:
    for field in ("file_id", "file_path", "relative_path", "file_name"):
        value = metadata.get(field)
        if value:
            return f"{field}:{value}"
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
        return _shorten_path(str(relative_path))
    if download_path is not None:
        try:
            return _shorten_path(download_path.relative_to(documents_root).as_posix())
        except ValueError:
            return file_name
    parent_folder = metadata.get("parent_folder")
    if parent_folder:
        return _shorten_path(f"{parent_folder}/{file_name}")
    return file_name


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


def _snippet_details(metadata: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{field}={metadata[field]}" for field in SOURCE_DETAIL_FIELDS if metadata.get(field))


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
