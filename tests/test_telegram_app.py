from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from langchain_core.documents import Document

from imperial_rag.app.chat_history import ChatHistoryStore
from imperial_rag.app.telegram import (
    ACCESS_DENIED_TEXT,
    FILE_DOWNLOAD_BYTE_LIMIT,
    TelegramBotState,
    handle_text,
    load_bot_configuration,
    parse_allowed_user_ids,
    split_telegram_text,
    start_health_server,
    telegram_user_email,
)


class FakeMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.events: list[tuple[str, str]] = []

    async def reply_text(self, text: str) -> None:
        self.events.append(("text", text))

    async def reply_document(self, document, *, filename: str, caption: str) -> None:
        assert document.read(1)
        self.events.append(("document", f"{filename}|{caption}"))


class FakeRuntime:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or {}
        self.error = error
        self.questions: list[str] = []

    def query(self, question: str):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.result


def _state(tmp_path: Path, runtime: FakeRuntime, *, allowed_user_ids: set[int] | None = None) -> TelegramBotState:
    chat_store = ChatHistoryStore(tmp_path / "chat.sqlite3")
    chat_store.initialize()
    settings = SimpleNamespace(
        documents_root=tmp_path / "documents",
        extraction_root=tmp_path / "extracted",
    )
    settings.documents_root.mkdir(exist_ok=True)
    return TelegramBotState(
        settings=settings,
        runtime=runtime,
        chat_store=chat_store,
        allowed_user_ids=frozenset(allowed_user_ids or {123}),
    )


def _update(message: FakeMessage, user_id: int, chat_type: str = "private"):
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(type=chat_type),
    )


def _context(state: TelegramBotState):
    return SimpleNamespace(application=SimpleNamespace(bot_data={"imperial_rag_state": state}))


def test_configuration_requires_token_and_numeric_allowlist() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        load_bot_configuration({})
    with pytest.raises(ValueError, match="at least one"):
        load_bot_configuration({"TELEGRAM_BOT_TOKEN": "token"})
    with pytest.raises(ValueError, match="comma-separated"):
        parse_allowed_user_ids("12,nope")
    with pytest.raises(ValueError, match="positive"):
        parse_allowed_user_ids("0")

    assert load_bot_configuration(
        {
            "TELEGRAM_BOT_TOKEN": " token ",
            "IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS": "123, 456,123",
        }
    ) == ("token", frozenset({123, 456}))
    assert telegram_user_email(123) == "telegram-123@users.invalid"


def test_split_telegram_text_respects_limit() -> None:
    text = ("word " * 1200).strip()

    chunks = split_telegram_text(text)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 4096 for chunk in chunks)
    assert " ".join(chunks) == text


def test_allowed_private_question_persists_answer_sources_and_one_safe_document(tmp_path: Path) -> None:
    document_path = tmp_path / "documents" / "manual.pdf"
    document_path.parent.mkdir(exist_ok=True)
    document_path.write_bytes(b"pdf")
    evidence = [
        Document(page_content="first", metadata={"relative_path": "manual.pdf", "source_type": "body"}),
        Document(page_content="second", metadata={"relative_path": "manual.pdf", "source_type": "body"}),
    ]
    runtime = FakeRuntime(
        {
            "answer": "Ответ. [S1]",
            "sources": ["[S1] manual.pdf body"],
            "evidence": evidence,
            "citations_valid": True,
            "retrieval": {"final_evidence": 2},
        }
    )
    state = _state(tmp_path, runtime)
    message = FakeMessage("Вопрос?")

    asyncio.run(handle_text(_update(message, 123), _context(state)))

    assert runtime.questions == ["Вопрос?"]
    assert message.events == [
        ("text", "Ответ. [S1]"),
        ("text", "Источники:\n[S1] manual.pdf body"),
        ("document", "manual.pdf|manual.pdf"),
    ]
    user_email = telegram_user_email(123)
    conversations = state.chat_store.list_conversations(user_email)
    assert len(conversations) == 1
    history = state.chat_store.list_messages(user_email, conversations[0].id)
    assert [item.role for item in history] == ["user", "assistant"]
    assert history[1].payload["sources"] == ["[S1] manual.pdf body"]
    assert history[1].payload["retrieval"]["final_evidence"] == 2


def test_unauthorized_and_group_messages_never_query_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime({"answer": "unused"})
    state = _state(tmp_path, runtime)
    unauthorized = FakeMessage("private")
    group = FakeMessage("group")

    asyncio.run(handle_text(_update(unauthorized, 999), _context(state)))
    asyncio.run(handle_text(_update(group, 123, chat_type="group"), _context(state)))

    assert runtime.questions == []
    assert unauthorized.events == [("text", ACCESS_DENIED_TEXT)]
    assert group.events == []


def test_query_failure_persists_and_sends_generic_error(tmp_path: Path) -> None:
    runtime = FakeRuntime(error=RuntimeError("private failure"))
    state = _state(tmp_path, runtime)
    message = FakeMessage("Sensitive question")

    asyncio.run(handle_text(_update(message, 123), _context(state)))

    assert len(message.events) == 1
    assert "private failure" not in message.events[0][1]
    conversations = state.chat_store.list_conversations(telegram_user_email(123))
    history = state.chat_store.list_messages(telegram_user_email(123), conversations[0].id)
    assert [item.role for item in history] == ["user", "assistant"]
    assert history[1].payload["error"]["type"] == "telegram_query_error"


def test_missing_outside_and_oversized_documents_are_not_uploaded(tmp_path: Path) -> None:
    documents_root = tmp_path / "documents"
    documents_root.mkdir()
    oversized = documents_root / "large.pdf"
    with oversized.open("wb") as file:
        file.truncate(FILE_DOWNLOAD_BYTE_LIMIT + 1)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"private")
    evidence = [
        Document(page_content="large", metadata={"relative_path": "large.pdf"}),
        Document(page_content="outside", metadata={"file_path": str(outside)}),
        Document(page_content="missing", metadata={"relative_path": "missing.pdf"}),
    ]
    runtime = FakeRuntime({"answer": "Ответ.", "sources": [], "evidence": evidence})
    state = _state(tmp_path, runtime)
    message = FakeMessage("Вопрос")

    asyncio.run(handle_text(_update(message, 123), _context(state)))

    assert all(event_type != "document" for event_type, _ in message.events)
    notices = "\n".join(text for event_type, text in message.events if event_type == "text")
    assert "large.pdf" in notices
    assert "outside.pdf" in notices
    assert "missing.pdf" in notices


def test_health_server_reports_readiness() -> None:
    from threading import Event

    ready = Event()
    server = start_health_server(ready, host="127.0.0.1", port=0)
    url = f"http://127.0.0.1:{server.server_address[1]}/healthz"
    try:
        with pytest.raises(HTTPError) as exc:
            urlopen(url, timeout=2)
        assert exc.value.code == 503

        ready.set()
        with urlopen(url, timeout=2) as response:
            assert response.status == 200
            assert response.read() == b"ok\n"
    finally:
        server.shutdown()
        server.server_close()
