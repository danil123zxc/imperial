from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import sqlite3
import sys
import types

import pytest
from langchain_core.documents import Document

from imperial_rag.app import web as web_app
from imperial_rag.app.auth import AuthStore
from imperial_rag.app.chat_history import ChatHistoryStore
from imperial_rag.app import chat_history as chat_history_module


class SessionState(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class TrackingConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value) -> None:
        if name in {"_connection", "closed"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._connection, name, value)

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._connection.__exit__(exc_type, exc, traceback)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)
        self._connection.close()


def test_chat_history_store_persists_and_scopes_conversations_by_user(tmp_path):
    db_path = tmp_path / "chat_history.sqlite3"
    store = ChatHistoryStore(db_path)
    first = store.create_conversation(
        "USER@example.com",
        title="User policy question",
        phoenix_session_id="trace-user",
    )
    store.add_message("user@example.com", first.id, "user", "What is my policy?")
    store.add_message(
        "user@example.com",
        first.id,
        "assistant",
        "Use policy A.",
        payload={
            "sources": ["policy-a.docx"],
            "retrieved_documents": [
                {
                    "page_content": "Private retrieved text",
                    "metadata": {"chunk_id": "chunk-a"},
                }
            ],
            "retrieval": {"final_evidence": 1},
        },
    )
    other = store.create_conversation(
        "other@example.com",
        title="Other user question",
        phoenix_session_id="trace-other",
    )
    store.add_message("other@example.com", other.id, "user", "Other private question")

    reopened = ChatHistoryStore(db_path)

    assert [conversation.id for conversation in reopened.list_conversations("user@example.com")] == [first.id]
    assert reopened.get_conversation("other@example.com", first.id) is None
    assert reopened.list_messages("other@example.com", first.id) == []
    with pytest.raises(PermissionError):
        reopened.add_message("other@example.com", first.id, "user", "Should not be allowed")

    messages = reopened.list_messages("USER@example.com", first.id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].to_chat_message()["retrieved_documents"][0]["page_content"] == "Private retrieved text"
    assert messages[1].to_chat_message()["retrieval"] == {"final_evidence": 1}


def test_chat_history_store_closes_short_lived_connections(monkeypatch, tmp_path):
    real_connect = sqlite3.connect
    opened: list[TrackingConnection] = []

    def tracking_connect(*args, **kwargs):
        connection = TrackingConnection(real_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(chat_history_module.sqlite3, "connect", tracking_connect)
    store = ChatHistoryStore(tmp_path / "chat_history.sqlite3")

    conversation = store.create_conversation("user@example.com", "Question")
    store.add_message("user@example.com", conversation.id, "user", "Hello")
    store.list_messages("user@example.com", conversation.id)

    assert opened
    assert all(connection.closed for connection in opened)


def test_chat_history_store_lists_messages_with_one_read_connection(monkeypatch, tmp_path):
    db_path = tmp_path / "chat_history.sqlite3"
    setup_store = ChatHistoryStore(db_path)
    conversation = setup_store.create_conversation("user@example.com", "Question")
    setup_store.add_message("user@example.com", conversation.id, "user", "Hello")

    real_connect = sqlite3.connect
    opened: list[TrackingConnection] = []

    def tracking_connect(*args, **kwargs):
        connection = TrackingConnection(real_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(chat_history_module.sqlite3, "connect", tracking_connect)
    store = ChatHistoryStore(db_path)
    store.initialize()
    opened.clear()

    messages = store.list_messages("user@example.com", conversation.id)

    assert [message.content for message in messages] == ["Hello"]
    assert len(opened) == 1
    assert all(connection.closed for connection in opened)


def test_chat_history_state_loads_only_signed_in_users_latest_chat(tmp_path):
    store = ChatHistoryStore(tmp_path / "chat_history.sqlite3")
    older = store.create_conversation("user@example.com", "Older chat", phoenix_session_id="trace-old")
    store.add_message("user@example.com", older.id, "user", "Older question")
    other = store.create_conversation("other@example.com", "Other chat", phoenix_session_id="trace-other")
    store.add_message("other@example.com", other.id, "user", "Other user's private question")
    latest = store.create_conversation("user@example.com", "Latest chat", phoenix_session_id="trace-new")
    store.add_message("user@example.com", latest.id, "user", "Latest question")

    streamlit = SimpleNamespace(session_state=SessionState())

    web_app._sync_chat_history_state(streamlit, store, "USER@example.com")

    assert streamlit.session_state.chat_history_user_email == "user@example.com"
    assert streamlit.session_state.active_conversation_id == latest.id
    assert streamlit.session_state.phoenix_trace_session_id == "trace-new"
    assert streamlit.session_state.messages == [
        {"role": "user", "content": "Latest question"},
        {
            "role": "assistant",
            "content": "The previous answer was not saved. Ask again to regenerate it.",
            "error": {"type": "incomplete_assistant_turn"},
        },
    ]
    assert "Other user's private question" not in str(streamlit.session_state.messages)


def test_pending_chat_turn_finishes_user_only_saved_turn(monkeypatch, tmp_path):
    store = ChatHistoryStore(tmp_path / "chat_history.sqlite3")
    conversation = store.create_conversation("user@example.com", "Pending chat", phoenix_session_id="trace-pending")
    user_message = store.add_message("user@example.com", conversation.id, "user", "Pending question")
    streamlit = SimpleNamespace(
        session_state=SessionState(
            active_conversation_id=conversation.id,
            phoenix_trace_session_id="trace-pending",
            messages=[
                {"role": "user", "content": "Pending question"},
                web_app._build_incomplete_assistant_message(),
            ],
            pending_chat_turn={
                "user_email": "user@example.com",
                "conversation_id": conversation.id,
                "question": "Pending question",
                "phoenix_session_id": "trace-pending",
                "user_message_id": user_message.id,
            },
        )
    )
    tracing_module = types.ModuleType("imperial_rag.observability.phoenix")
    tracing_module.phoenix_trace_context = _null_trace_context
    tracing_module.trace_user_id_from_email = lambda email: "user_sha256:testhash"
    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.log_event = lambda *args, **kwargs: None
    observability_module.log_failure = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setattr(
        web_app,
        "query_runtime",
        lambda settings, question: {
            "answer": "Recovered answer.",
            "sources": [],
            "retrieval": {"final_evidence": 0},
        },
    )

    completed = web_app._complete_pending_chat_turn(
        streamlit,
        store,
        SimpleNamespace(documents_root=tmp_path / "documents", extraction_root=tmp_path / "extracted"),
        "user@example.com",
    )

    assert completed is not None
    assert streamlit.session_state.messages == [
        {"role": "user", "content": "Pending question"},
        {
            "role": "assistant",
            "content": "Recovered answer.",
            "sources": [],
            "error": None,
            "citations_valid": None,
            "invalid_citations": [],
            "retrieved_files": [],
            "retrieved_documents": [],
            "retrieval": {"final_evidence": 0},
        },
    ]
    assert web_app.PENDING_CHAT_TURN_KEY not in streamlit.session_state
    assert [message.content for message in store.list_messages("user@example.com", conversation.id)] == [
        "Pending question",
        "Recovered answer.",
    ]


def test_build_assistant_message_preserves_debug_retrieval_payload(tmp_path):
    source_path = tmp_path / "documents" / "policy.docx"
    source_path.parent.mkdir()
    source_path.write_bytes(b"policy")
    settings = SimpleNamespace(documents_root=source_path.parent, extraction_root=tmp_path / "extracted")
    result = {
        "answer": "Use policy A.",
        "sources": ["policy.docx"],
        "citations_valid": True,
        "evidence": [
            Document(
                page_content="Private chunk text for future debugging.",
                metadata={
                    "relative_path": "policy.docx",
                    "file_name": "policy.docx",
                    "chunk_id": "chunk-a",
                },
            )
        ],
        "retrieval": {"final_evidence": 1, "reranker": "qwen"},
    }

    message = web_app._build_assistant_message(result, settings)

    assert message["content"] == "Use policy A."
    assert message["sources"] == ["policy.docx"]
    assert message["retrieved_documents"] == [
        {
            "page_content": "Private chunk text for future debugging.",
            "metadata": {
                "relative_path": "policy.docx",
                "file_name": "policy.docx",
                "chunk_id": "chunk-a",
            },
        }
    ]
    assert message["retrieval"] == {"final_evidence": 1, "reranker": "qwen"}


def test_main_persists_submitted_question_to_signed_in_users_chat_history(monkeypatch, tmp_path):
    auth_db_path = tmp_path / "auth.sqlite3"
    chat_history_db_path = tmp_path / "chat_history.sqlite3"
    auth_store = AuthStore(auth_db_path)
    auth_store.initialize()
    auth_store.bootstrap_admin("admin@example.com", "admin-password")
    auth_store.register_user("user@example.com", "user-password", "User", "Testing")
    auth_store.approve_user("admin@example.com", "user@example.com")

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path
            self.chat_history_db_path = chat_history_db_path
            self.documents_root = tmp_path / "documents"
            self.extraction_root = tmp_path / "extracted"

    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = types.ModuleType("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context
    tracing_module.trace_user_id_from_email = lambda email: "user_sha256:testhash"

    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda *args, **kwargs: None
    observability_module.log_failure = lambda *args, **kwargs: None

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
        session_state=SessionState(auth_user_email="user@example.com"),
        caption=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: False,
        chat_input=lambda *args, **kwargs: "How do I expense travel?",
        chat_message=lambda *args, **kwargs: Context(),
        write=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setattr(
        web_app,
        "query_runtime",
        lambda settings, question: {
            "answer": "Use the travel expense policy.",
            "sources": ["travel-policy.docx"],
            "retrieval": {"final_evidence": 0},
        },
    )

    web_app.main()

    history = ChatHistoryStore(chat_history_db_path)
    conversations = history.list_conversations("user@example.com")
    assert len(conversations) == 1
    assert conversations[0].title == "How do I expense travel?"
    messages = history.list_messages("user@example.com", conversations[0].id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == [
        "How do I expense travel?",
        "Use the travel expense policy.",
    ]
    assert messages[1].to_chat_message()["sources"] == ["travel-policy.docx"]
    assert history.list_conversations("other@example.com") == []


def test_main_persists_prompt_to_original_chat_when_sidebar_selection_changes(monkeypatch, tmp_path):
    auth_db_path = tmp_path / "auth.sqlite3"
    chat_history_db_path = tmp_path / "chat_history.sqlite3"
    auth_store = AuthStore(auth_db_path)
    auth_store.initialize()
    auth_store.bootstrap_admin("admin@example.com", "admin-password")
    auth_store.register_user("user@example.com", "user-password", "User", "Testing")
    auth_store.approve_user("admin@example.com", "user@example.com")

    history = ChatHistoryStore(chat_history_db_path)
    older = history.create_conversation("user@example.com", "Older chat", phoenix_session_id="trace-old")
    history.add_message("user@example.com", older.id, "user", "Older question")
    history.add_message("user@example.com", older.id, "assistant", "Older answer")
    active = history.create_conversation("user@example.com", "Active chat", phoenix_session_id="trace-active")
    history.add_message("user@example.com", active.id, "user", "Active question")
    history.add_message("user@example.com", active.id, "assistant", "Active answer")

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path
            self.chat_history_db_path = chat_history_db_path
            self.documents_root = tmp_path / "documents"
            self.extraction_root = tmp_path / "extracted"

    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = types.ModuleType("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context
    tracing_module.trace_user_id_from_email = lambda email: "user_sha256:testhash"

    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda *args, **kwargs: None
    observability_module.log_failure = lambda *args, **kwargs: None

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
        session_state=SessionState(
            auth_user_email="user@example.com",
            chat_history_user_email="user@example.com",
            active_conversation_id=active.id,
            phoenix_trace_session_id="trace-active",
        ),
        caption=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: kwargs.get("key") == f"chat-history-select-{older.id}",
        chat_input=lambda *args, **kwargs: "Question typed in the active chat",
        chat_message=lambda *args, **kwargs: Context(),
        write=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        rerun=lambda: None,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setattr(
        web_app,
        "query_runtime",
        lambda settings, question: {
            "answer": "Answer for the active chat.",
            "sources": [],
            "retrieval": {"final_evidence": 0},
        },
    )

    web_app.main()

    updated_history = ChatHistoryStore(chat_history_db_path)
    active_messages = updated_history.list_messages("user@example.com", active.id)
    older_messages = updated_history.list_messages("user@example.com", older.id)
    assert [message.content for message in active_messages] == [
        "Active question",
        "Active answer",
        "Question typed in the active chat",
        "Answer for the active chat.",
    ]
    assert [message.content for message in older_messages] == ["Older question", "Older answer"]


def test_main_persists_assistant_error_when_signed_in_query_fails(monkeypatch, tmp_path):
    auth_db_path = tmp_path / "auth.sqlite3"
    chat_history_db_path = tmp_path / "chat_history.sqlite3"
    auth_store = AuthStore(auth_db_path)
    auth_store.initialize()
    auth_store.bootstrap_admin("admin@example.com", "admin-password")
    auth_store.register_user("user@example.com", "user-password", "User", "Testing")
    auth_store.approve_user("admin@example.com", "user@example.com")

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda: None

    class FakeSettings:
        def __init__(self):
            self.auth_db_path = auth_db_path
            self.chat_history_db_path = chat_history_db_path
            self.documents_root = tmp_path / "documents"
            self.extraction_root = tmp_path / "extracted"

    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = FakeSettings

    tracing_module = types.ModuleType("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings: None
    tracing_module.phoenix_trace_context = _null_trace_context
    tracing_module.trace_user_id_from_email = lambda email: "user_sha256:testhash"

    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda *args, **kwargs: None
    observability_module.log_failure = lambda *args, **kwargs: None

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    errors = []
    streamlit_module = types.SimpleNamespace(
        set_page_config=lambda **kwargs: None,
        title=lambda *args, **kwargs: None,
        sidebar=Context(),
        header=lambda *args, **kwargs: None,
        text=lambda *args, **kwargs: None,
        session_state=SessionState(auth_user_email="user@example.com"),
        caption=lambda *args, **kwargs: None,
        button=lambda *args, **kwargs: False,
        chat_input=lambda *args, **kwargs: "How do I expense travel?",
        chat_message=lambda *args, **kwargs: Context(),
        write=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda message: errors.append(message),
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "streamlit", streamlit_module)
    monkeypatch.setattr(web_app, "query_runtime", lambda settings, question: (_ for _ in ()).throw(RuntimeError("boom")))

    web_app.main()

    history = ChatHistoryStore(chat_history_db_path)
    conversations = history.list_conversations("user@example.com")
    assert len(conversations) == 1
    messages = history.list_messages("user@example.com", conversations[0].id)
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].content == "Something went wrong while answering. Check local logs for details."
    assert messages[1].to_chat_message()["error"] == {
        "type": "web_query_error",
        "exception_type": "RuntimeError",
    }
    assert errors == ["Something went wrong while answering. Check local logs for details."]


@contextmanager
def _null_trace_context(session_id, **kwargs):
    yield
