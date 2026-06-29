import json
import subprocess
import sys
import textwrap
import types
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document

from imperial_rag.app import web as web_app
from imperial_rag.app.web import (
    APP_TITLE,
    RetrievedFileGroup,
    build_retrieved_file_groups,
    build_status_summary,
    load_status_summary,
)


def _fake_module(name: str) -> Any:
    return types.ModuleType(name)


def test_status_summary_displays_manifest_counts():
    summary = build_status_summary(total_files=162, indexed_files=100, failed_files=3)

    assert APP_TITLE == "Imperial RAG"
    assert "Total files: 162" in summary
    assert "Indexed files: 100" in summary
    assert "Failed files: 3" in summary


def test_load_status_summary_is_importable_without_manifest_stack():
    summary = load_status_summary(settings=object())

    assert "Total files:" in summary


def test_build_retrieved_file_groups_groups_chunks_by_file_and_loads_file_preview(tmp_path):
    documents_root = tmp_path / "documents"
    extraction_root = tmp_path / ".imperial_rag" / "extracted"
    source_path = documents_root / "11. РЕГЛАМЕНТЫ" / "Регламент ЛОГИСТИКА.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"docx bytes")
    artifact_path = extraction_root / "documents" / "logistics.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"page_content": "Первый фрагмент полного файла."},
                    {"page_content": "Второй фрагмент полного файла."},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=extraction_root)

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Retrieved chunk should not be shown.",
                metadata={
                    "file_id": "logistics",
                    "file_path": str(source_path),
                    "relative_path": "11. РЕГЛАМЕНТЫ/Регламент ЛОГИСТИКА.docx",
                    "file_name": "Регламент ЛОГИСТИКА.docx",
                    "source_type": "body",
                },
            ),
            Document(
                page_content="Another retrieved chunk should not be shown.",
                metadata={
                    "file_id": "logistics",
                    "file_path": str(source_path),
                    "relative_path": "11. РЕГЛАМЕНТЫ/Регламент ЛОГИСТИКА.docx",
                    "file_name": "Регламент ЛОГИСТИКА.docx",
                    "source_type": "body",
                },
            ),
        ],
        settings,
    )

    assert len(groups) == 1
    assert groups[0].file_name == "Регламент ЛОГИСТИКА.docx"
    assert groups[0].display_path == "11. РЕГЛАМЕНТЫ/Регламент ЛОГИСТИКА.docx"
    assert groups[0].preview_text == "Первый фрагмент полного файла.\n\nВторой фрагмент полного файла."
    assert not hasattr(groups[0], "snippets")
    assert not hasattr(groups[0], "markers")
    assert not hasattr(groups[0], "chunk_count")
    assert groups[0].can_download is True


def test_build_retrieved_file_groups_bounds_loaded_file_preview(tmp_path, monkeypatch):
    documents_root = tmp_path / "documents"
    extraction_root = tmp_path / ".imperial_rag" / "extracted"
    source_path = documents_root / "docs" / "policy.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"docx bytes")
    artifact_path = extraction_root / "documents" / "policy.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps({"documents": [{"page_content": "abcdef"}, {"page_content": "ghijkl"}]}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=extraction_root)
    monkeypatch.setattr(web_app, "FILE_PREVIEW_CHAR_LIMIT", 10)

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="chunk",
                metadata={
                    "file_id": "policy",
                    "file_path": str(source_path),
                    "relative_path": "docs/policy.docx",
                    "file_name": "policy.docx",
                },
            )
        ],
        settings,
    )

    assert groups[0].preview_text == "abcdef\n\ngh..."


def test_download_button_payload_disables_large_files(tmp_path, monkeypatch):
    path = tmp_path / "large.pdf"
    path.write_bytes(b"12345")
    monkeypatch.setattr(web_app, "FILE_DOWNLOAD_BYTE_LIMIT", 4)
    group = RetrievedFileGroup(
        file_key="relative_path:large.pdf",
        file_name="large.pdf",
        display_path="large.pdf",
        download_path=path,
        download_name="large.pdf",
        download_mime="application/pdf",
        preview_text="preview",
        can_download=True,
    )

    assert web_app._download_button_payload(group) == (b"", True)


def test_build_retrieved_file_groups_groups_same_relative_path_without_matching_file_id(tmp_path):
    documents_root = tmp_path / "documents"
    source_path = documents_root / "docs" / "same.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"same")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="First chunk",
                metadata={
                    "file_id": "old-id",
                    "relative_path": "docs/same.docx",
                    "file_name": "same.docx",
                },
            ),
            Document(
                page_content="Second chunk",
                metadata={
                    "relative_path": "docs/same.docx",
                    "file_name": "same.docx",
                },
            ),
        ],
        settings,
    )

    assert len(groups) == 1
    assert groups[0].file_key == "relative_path:docs/same.docx"


def test_build_retrieved_file_groups_keeps_same_filename_in_different_folders_separate(tmp_path):
    documents_root = tmp_path / "documents"
    first = documents_root / "sales" / "policy.docx"
    second = documents_root / "hr" / "policy.docx"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"sales")
    second.write_bytes(b"hr")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Sales policy",
                metadata={"relative_path": "sales/policy.docx", "file_name": "policy.docx"},
            ),
            Document(
                page_content="HR policy",
                metadata={"relative_path": "hr/policy.docx", "file_name": "policy.docx"},
            ),
        ],
        settings,
    )

    assert [group.display_path for group in groups] == ["sales/policy.docx", "hr/policy.docx"]


def test_normalize_retrieved_file_groups_merges_stored_duplicate_cards(tmp_path):
    documents_root = tmp_path / "documents"
    extraction_root = tmp_path / "extracted"
    source_path = documents_root / "docs" / "same.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"same")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=extraction_root)
    first = RetrievedFileGroup(
        file_key="file_id:old",
        file_name="same.docx",
        display_path="docs/same.docx",
        download_path=source_path,
        download_name="same.docx",
        download_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        preview_text="First preview",
        can_download=True,
    )
    second = RetrievedFileGroup(
        file_key="relative_path:docs/same.docx",
        file_name="same.docx",
        display_path="docs/same.docx",
        download_path=source_path,
        download_name="same.docx",
        download_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        preview_text="Second preview",
        can_download=True,
    )

    groups = web_app.normalize_retrieved_file_groups([first, second], settings)

    assert len(groups) == 1
    assert groups[0].file_key == "relative_path:docs/same.docx"
    assert groups[0].preview_text == "First preview"


def test_build_retrieved_file_groups_uses_relative_path_for_safe_download(tmp_path):
    documents_root = tmp_path / "documents"
    source_path = documents_root / "forms" / "заявление на увольнение.docx"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"form")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Фрагмент заявления.",
                metadata={
                    "relative_path": "forms/заявление на увольнение.docx",
                    "file_name": "заявление на увольнение.docx",
                    "source_type": "body",
                },
            )
        ],
        settings,
    )

    assert groups[0].download_path == source_path.resolve()
    assert groups[0].download_name == "заявление на увольнение.docx"
    assert groups[0].can_download is True


def test_build_retrieved_file_groups_disables_missing_or_outside_downloads(tmp_path):
    documents_root = tmp_path / "documents"
    documents_root.mkdir()
    outside_path = tmp_path / "outside.docx"
    outside_path.write_bytes(b"outside")
    settings = SimpleNamespace(documents_root=documents_root, extraction_root=tmp_path / "extracted")

    groups = build_retrieved_file_groups(
        [
            Document(
                page_content="Missing file snippet.",
                metadata={"relative_path": "missing.docx", "file_name": "missing.docx"},
            ),
            Document(
                page_content="Outside file snippet.",
                metadata={"file_path": str(outside_path), "file_name": "outside.docx"},
            ),
        ],
        settings,
    )

    assert [group.can_download for group in groups] == [False, False]
    assert [group.download_path for group in groups] == [None, None]
    assert [group.preview_text for group in groups] == [
        web_app.PREVIEW_UNAVAILABLE_TEXT,
        web_app.PREVIEW_UNAVAILABLE_TEXT,
    ]


def test_build_retrieved_file_groups_returns_empty_list_without_evidence(tmp_path):
    settings = SimpleNamespace(documents_root=tmp_path / "documents")

    assert build_retrieved_file_groups([], settings) == []


def test_query_runtime_reuses_streamlit_cached_runtime(monkeypatch, tmp_path):
    calls = []

    class FakeRuntime:
        def __init__(self, settings):
            self.settings = settings

        def query(self, question):
            return {"answer": f"{question}:{len(calls)}"}

    def fake_create_runtime(settings):
        calls.append(settings)
        return FakeRuntime(settings)

    def fake_cache_resource(func):
        cache = {}

        def wrapper(cache_key, settings):
            if cache_key not in cache:
                cache[cache_key] = func(cache_key, settings)
            return cache[cache_key]

        return wrapper

    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace(cache_resource=fake_cache_resource))
    monkeypatch.setattr("imperial_rag.answering.runtime.create_runtime", fake_create_runtime)
    monkeypatch.setattr(web_app, "_RUNTIME_CACHE_WRAPPER", None, raising=False)
    settings = SimpleNamespace(
        workspace_root=tmp_path,
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="chunks",
        elasticsearch_url="http://127.0.0.1:9200",
        elasticsearch_index="keyword",
    )

    first = web_app.query_runtime(settings, "first")
    second = web_app.query_runtime(settings, "second")

    assert first == {"answer": "first:1"}
    assert second == {"answer": "second:1"}
    assert calls == [settings]


def test_render_chat_message_surfaces_invalid_citation_warning(tmp_path):
    class ChatMessage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    warnings = []
    writes = []
    streamlit = types.SimpleNamespace(
        chat_message=lambda *args, **kwargs: ChatMessage(),
        write=lambda message: writes.append(message),
        warning=lambda message: warnings.append(message),
    )
    message = {
        "role": "assistant",
        "content": "Unsupported answer without valid citations.",
        "citations_valid": False,
        "invalid_citations": ["S99"],
    }

    web_app._render_chat_message(streamlit, message, 0, SimpleNamespace(documents_root=tmp_path))

    assert writes == ["Unsupported answer without valid citations."]
    assert warnings == ["Answer citations could not be verified. Treat this response as diagnostic."]


def test_render_chat_message_surfaces_model_provider_error(tmp_path):
    class ChatMessage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    errors = []
    streamlit = types.SimpleNamespace(
        chat_message=lambda *args, **kwargs: ChatMessage(),
        error=lambda message: errors.append(message),
    )
    message = {
        "role": "assistant",
        "content": "The model provider failed while answering. Check local logs and provider credentials, then try again.",
        "error": {"type": "model_provider_error"},
    }

    web_app._render_chat_message(streamlit, message, 0, SimpleNamespace(documents_root=tmp_path))

    assert errors == [
        "The model provider failed while answering. Check local logs and provider credentials, then try again."
    ]


def test_main_loads_project_env_before_creating_settings(monkeypatch, tmp_path):
    from imperial_rag.app import web as web_app

    calls = []

    env_module = _fake_module("imperial_rag.env")
    env_module.load_project_env = lambda: calls.append("env")

    class FakeSettings:
        def __init__(self):
            calls.append("settings")
            self.auth_db_path = tmp_path / "auth.sqlite3"

    config_module = _fake_module("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = _fake_module("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: calls.append("tracing")
    tracing_module.phoenix_trace_context = _null_trace_context

    observability_module = _fake_module("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: calls.append("observability")

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Sidebar(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(),
        subheader=lambda *args, **kwargs: None,
        radio=lambda *args, **kwargs: "Log in",
        form=lambda *args, **kwargs: Sidebar(),
        text_input=lambda *args, **kwargs: "",
        form_submit_button=lambda *args, **kwargs: False,
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        success=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        chat_input=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    web_app.main()

    assert calls[:3] == ["env", "settings", "observability"]


def test_main_requires_authenticated_user_before_chat_input(monkeypatch, tmp_path):
    from imperial_rag.app import web as web_app

    env_module = _fake_module("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = tmp_path / "auth.sqlite3"

    config_module = _fake_module("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = _fake_module("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context

    observability_module = _fake_module("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Form:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    chat_inputs = []
    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Sidebar(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(),
        subheader=lambda *args, **kwargs: None,
        radio=lambda *args, **kwargs: "Log in",
        form=lambda *args, **kwargs: Form(),
        text_input=lambda *args, **kwargs: "",
        form_submit_button=lambda *args, **kwargs: False,
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        success=lambda *args, **kwargs: None,
        chat_input=lambda *args, **kwargs: chat_inputs.append(args),
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    web_app.main()

    assert chat_inputs == []


def test_main_notifies_admin_about_pending_access_requests(monkeypatch, tmp_path):
    from imperial_rag.app import web as web_app
    from imperial_rag.app.auth import AuthStore

    auth_db_path = tmp_path / "auth.sqlite3"
    store = AuthStore(auth_db_path)
    store.initialize()
    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password", "Test User", "Needs access")

    env_module = _fake_module("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path

    config_module = _fake_module("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = _fake_module("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context

    observability_module = _fake_module("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    warnings = []
    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Context(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(auth_user_email="admin@example.com"),
        caption=lambda *args, **kwargs: None,
        warning=lambda message, *args, **kwargs: warnings.append(message),
        button=lambda *args, **kwargs: False,
        markdown=lambda *args, **kwargs: None,
        container=lambda *args, **kwargs: Context(),
        chat_input=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    web_app.main()

    assert warnings == ["1 pending access request"]


def test_main_signup_form_creates_pending_access_request(monkeypatch, tmp_path):
    from imperial_rag.app import web as web_app
    from imperial_rag.app.auth import AuthStore

    auth_db_path = tmp_path / "auth.sqlite3"

    env_module = _fake_module("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path

    config_module = _fake_module("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = _fake_module("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context

    observability_module = _fake_module("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    values = {
        "auth-signup-full-name": "Test User",
        "auth-signup-email": "user@example.com",
        "auth-signup-password": "user-password",
        "auth-signup-reason": "Needs document access",
    }
    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        session_state=SessionState(),
        subheader=lambda *args, **kwargs: None,
        radio=lambda *args, **kwargs: "Sign up",
        form=lambda *args, **kwargs: Context(),
        text_input=lambda *args, **kwargs: values[kwargs["key"]],
        text_area=lambda *args, **kwargs: values[kwargs["key"]],
        form_submit_button=lambda *args, **kwargs: True,
        error=lambda *args, **kwargs: None,
        success=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    web_app.main()

    pending = AuthStore(auth_db_path).list_pending_users()
    assert [user.email for user in pending] == ["user@example.com"]
    assert pending[0].reason == "Needs document access"


def test_main_admin_grant_button_approves_user(monkeypatch, tmp_path):
    from imperial_rag.app import web as web_app
    from imperial_rag.app.auth import AuthStore, AuthenticationStatus

    auth_db_path = tmp_path / "auth.sqlite3"
    store = AuthStore(auth_db_path)
    store.initialize()
    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password", "Test User", "Needs access")

    env_module = _fake_module("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path

    config_module = _fake_module("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = _fake_module("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context

    observability_module = _fake_module("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Context(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(auth_user_email="admin@example.com"),
        caption=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: kwargs.get("key") == "auth-approve-user@example.com",
        markdown=lambda *args, **kwargs: None,
        container=lambda *args, **kwargs: Context(),
        success=lambda *args, **kwargs: None,
        rerun=lambda: None,
        chat_input=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)

    web_app.main()

    assert AuthStore(auth_db_path).authenticate("user@example.com", "user-password").status == AuthenticationStatus.AUTHENTICATED


def test_main_logs_web_query_failure_without_private_question(monkeypatch, tmp_path):
    from imperial_rag.app import web as web_app
    from imperial_rag.app.auth import AuthStore

    calls = []
    auth_db_path = tmp_path / "auth.sqlite3"
    chat_history_db_path = tmp_path / "chat_history.sqlite3"
    store = AuthStore(auth_db_path)
    store.initialize()
    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password", "User", "Testing")
    store.approve_user("admin@example.com", "user@example.com")

    env_module = _fake_module("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path
            self.chat_history_db_path = chat_history_db_path

    config_module = _fake_module("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = _fake_module("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    trace_contexts = []

    @contextmanager
    def trace_context(session_id, **kwargs):
        trace_contexts.append({"session_id": session_id, **kwargs})
        yield

    tracing_module.phoenix_trace_context = trace_context
    tracing_module.trace_user_id_from_email = lambda email: "user_sha256:testhash"

    observability_module = _fake_module("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda *args, **kwargs: calls.append(("event", args, kwargs))
    observability_module.log_failure = lambda *args, **kwargs: calls.append(("failure", args, kwargs))

    class SessionState(dict):
        def __getattr__(self, key):
            return self[key]

        def __setattr__(self, key, value):
            self[key] = value

    class Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class ChatMessage:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    errors = []
    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Sidebar(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(auth_user_email="user@example.com"),
        caption=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: False,
        chat_input=lambda *args, **kwargs: "private question",
        chat_message=lambda *args, **kwargs: ChatMessage(),
        write=lambda *args, **kwargs: None,
        error=lambda message: errors.append(message),
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setattr(web_app, "query_runtime", lambda settings, question: (_ for _ in ()).throw(RuntimeError("boom")))

    web_app.main()

    failure = [call for call in calls if call[0] == "failure"][0]
    assert failure[1][0] == "web_query"
    assert "question" not in failure[2]
    assert trace_contexts == [
        {
            "session_id": streamlit_module.session_state.phoenix_trace_session_id,
            "user_id": "user_sha256:testhash",
            "metadata": {"entrypoint": "streamlit"},
            "tags": ["imperial-rag", "streamlit"],
        }
    ]
    assert streamlit_module.session_state.auth_user_email not in str(trace_contexts)
    assert errors == ["Something went wrong while answering. Check local logs for details."]


def test_main_bootstraps_src_path_for_streamlit_script_launch():
    script = textwrap.dedent(
        """
        import os
        import runpy
        import sys
        import tempfile
        import types

        sys.path = [entry for entry in sys.path if not entry.endswith('/src')]
        for name in list(sys.modules):
            if name.startswith('imperial_rag'):
                del sys.modules[name]
        os.environ['IMPERIAL_RAG_WORKSPACE_ROOT'] = tempfile.mkdtemp()

        class SessionState(dict):
            def __getattr__(self, key):
                return self[key]

            def __setattr__(self, key, value):
                self[key] = value

        class Sidebar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        sys.modules['streamlit'] = types.SimpleNamespace(
            set_page_config=lambda **kwargs: None,
            title=lambda *args, **kwargs: None,
            sidebar=Sidebar(),
            header=lambda *args, **kwargs: None,
            text=lambda *args, **kwargs: None,
            session_state=SessionState(),
            subheader=lambda *args, **kwargs: None,
            radio=lambda *args, **kwargs: "Log in",
            form=lambda *args, **kwargs: Sidebar(),
            text_input=lambda *args, **kwargs: "",
            form_submit_button=lambda *args, **kwargs: False,
            error=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            success=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            chat_input=lambda *args, **kwargs: None,
        )
        namespace = runpy.run_path('src/imperial_rag/app/web.py', run_name='imperial_web_app_test')
        namespace['main']()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@contextmanager
def _null_trace_context(session_id, **kwargs):
    yield
