from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


APP_TITLE = "Imperial RAG"


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


def main() -> None:
    _ensure_src_on_path()

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            for source in message.get("sources", []):
                st.caption(str(source))

    question = st.chat_input("Ask about the indexed documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    result = query_runtime(settings, question)
    answer = str(result.get("answer", ""))
    sources = result.get("sources") or result.get("citations") or []
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    with st.chat_message("assistant"):
        st.write(answer)
        for source in sources:
            st.caption(str(source))


def _coerce_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "answer": getattr(result, "answer", ""),
        "sources": getattr(result, "sources", getattr(result, "citations", [])),
    }


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
