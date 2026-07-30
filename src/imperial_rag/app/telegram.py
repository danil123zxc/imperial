from __future__ import annotations

import asyncio
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, Mapping

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from imperial_rag.app.chat_history import ChatHistoryStore
from imperial_rag.app.web import (
    FILE_DOWNLOAD_BYTE_LIMIT,
    _build_assistant_message,
    _build_query_failure_message,
    _chat_message_payload,
    _query_log_fields,
    build_retrieved_file_groups,
)
from imperial_rag.observability import log_event, log_failure
from imperial_rag.observability.phoenix import phoenix_trace_context, trace_user_id_from_email

MAX_TELEGRAM_TEXT_LENGTH = 4096
STATE_KEY = "imperial_rag_state"
READY_KEY = "imperial_rag_ready"
ACCESS_DENIED_TEXT = "Доступ к боту не предоставлен."
WELCOME_TEXT = "Задайте вопрос по проиндексированным документам."
SOURCE_UNAVAILABLE_TEXT = "Источник недоступен для отправки: {name}"


@dataclass(frozen=True)
class TelegramBotState:
    settings: Any
    runtime: Any
    chat_store: ChatHistoryStore
    allowed_user_ids: frozenset[int]


def load_bot_configuration(environ: Mapping[str, str] | None = None) -> tuple[str, frozenset[int]]:
    values = os.environ if environ is None else environ
    token = values.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is required")
    return token, parse_allowed_user_ids(values.get("IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS"))


def parse_allowed_user_ids(raw: str | None) -> frozenset[int]:
    values = [value.strip() for value in str(raw or "").split(",") if value.strip()]
    if not values:
        raise ValueError("IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS must contain at least one numeric user ID")
    try:
        user_ids = frozenset(int(value) for value in values)
    except ValueError as exc:
        raise ValueError("IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS must be a comma-separated list of integers") from exc
    if any(user_id <= 0 for user_id in user_ids):
        raise ValueError("Telegram user IDs must be positive integers")
    return user_ids


def telegram_user_email(user_id: int) -> str:
    if int(user_id) <= 0:
        raise ValueError("Telegram user ID must be positive")
    return f"telegram-{int(user_id)}@users.invalid"


def split_telegram_text(text: str, limit: int = MAX_TELEGRAM_TEXT_LENGTH) -> list[str]:
    if limit <= 0:
        raise ValueError("message limit must be positive")
    remaining = str(text or "").strip()
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = max(
            remaining.rfind("\n\n", 0, limit + 1),
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind(" ", 0, limit + 1),
        )
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_private(update):
        return
    message = update.effective_message
    if message is None:
        return
    if not _is_allowed(update, _state(context)):
        await message.reply_text(ACCESS_DENIED_TEXT)
        return
    await message.reply_text(WELCOME_TEXT)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_private(update):
        return
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return
    state = _state(context)
    if not _is_allowed(update, state):
        await message.reply_text(ACCESS_DENIED_TEXT)
        return
    question = str(message.text or "").strip()
    if not question:
        return

    user_email = telegram_user_email(user.id)
    conversation = _conversation_for_question(state.chat_store, user_email, question)
    user_message = state.chat_store.add_message(user_email, conversation.id, "user", question)
    if not state.chat_store.claim_assistant_response(user_email, conversation.id, user_message.id):
        return

    user_hash = trace_user_id_from_email(user_email)
    started_at = perf_counter()
    try:
        with phoenix_trace_context(
            conversation.phoenix_session_id,
            user_id=user_hash,
            metadata={"entrypoint": "telegram"},
            tags=["imperial-rag", "telegram"],
        ):
            result = await asyncio.to_thread(state.runtime.query, question)
    except Exception as exc:
        log_failure(
            "telegram_query",
            exc,
            component="telegram",
            duration_ms=_duration_ms(started_at),
            phoenix_session_id=conversation.phoenix_session_id,
            session_id=conversation.phoenix_session_id,
            user_hash=user_hash,
        )
        assistant_message = _build_query_failure_message(exc)
        assistant_message["error"]["type"] = "telegram_query_error"
        _persist_assistant_message(state.chat_store, user_email, conversation.id, assistant_message)
        await _send_text(message, assistant_message["content"])
        return

    result = result if isinstance(result, dict) else {"answer": getattr(result, "answer", str(result))}
    assistant_message = _build_assistant_message(result, state.settings)
    _persist_assistant_message(state.chat_store, user_email, conversation.id, assistant_message)
    await _send_text(message, assistant_message["content"])
    await _send_sources(message, result.get("sources") or result.get("citations") or [])
    await _send_documents(message, result.get("evidence") or result.get("retrieved_documents") or [], state.settings)
    log_event(
        "imperial_rag.telegram_query",
        operation="telegram_query",
        status="success",
        component="telegram",
        duration_ms=_duration_ms(started_at),
        phoenix_session_id=conversation.phoenix_session_id,
        session_id=conversation.phoenix_session_id,
        user_hash=user_hash,
        **_query_log_fields(result),
    )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    error = context.error
    if isinstance(error, BaseException):
        log_failure("telegram_update", error, component="telegram")


def create_application(
    token: str,
    state: TelegramBotState,
    ready: Event,
) -> Application[Any, Any, Any, Any, Any, Any]:
    application = (
        Application.builder()
        .token(token)
        # ponytail: sequential updates; add bounded per-user concurrency only if queue latency becomes measurable.
        .concurrent_updates(False)
        .post_init(_mark_ready)
        .post_shutdown(_mark_not_ready)
        .build()
    )
    application.bot_data[STATE_KEY] = state
    application.bot_data[READY_KEY] = ready
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("help", handle_start))
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.PRIVATE & ~filters.COMMAND, handle_text))
    application.add_error_handler(handle_error)
    return application


def start_health_server(
    ready: Event,
    host: str = "0.0.0.0",
    port: int = 8501,
) -> ThreadingHTTPServer:
    handler = _health_handler(ready)
    server = ThreadingHTTPServer((host, port), handler)
    Thread(target=server.serve_forever, name="telegram-health", daemon=True).start()
    return server


def main() -> None:
    from imperial_rag.answering.runtime import create_runtime
    from imperial_rag.config import Settings, apply_active_index_pointer
    from imperial_rag.env import load_project_env
    from imperial_rag.observability import configure_observability
    from imperial_rag.observability.phoenix import configure_phoenix_tracing

    load_project_env()
    token, allowed_user_ids = load_bot_configuration()
    settings = apply_active_index_pointer(Settings())
    configure_observability(settings)
    configure_phoenix_tracing(settings)
    chat_store = ChatHistoryStore(Path(settings.chat_history_db_path))
    chat_store.initialize()
    state = TelegramBotState(
        settings=settings,
        runtime=create_runtime(settings),
        chat_store=chat_store,
        allowed_user_ids=allowed_user_ids,
    )
    ready = Event()
    health_server = start_health_server(ready)
    application = create_application(token, state, ready)
    try:
        application.run_polling(allowed_updates=["message"])
    finally:
        ready.clear()
        health_server.shutdown()
        health_server.server_close()


async def _mark_ready(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    application.bot_data[READY_KEY].set()


async def _mark_not_ready(application: Application[Any, Any, Any, Any, Any, Any]) -> None:
    application.bot_data[READY_KEY].clear()


def _state(context: ContextTypes.DEFAULT_TYPE) -> TelegramBotState:
    return context.application.bot_data[STATE_KEY]


def _is_private(update: Update) -> bool:
    chat = update.effective_chat
    return chat is not None and chat.type == ChatType.PRIVATE


def _is_allowed(update: Update, state: TelegramBotState) -> bool:
    user = update.effective_user
    return user is not None and user.id in state.allowed_user_ids


def _conversation_for_question(chat_store: ChatHistoryStore, user_email: str, question: str) -> Any:
    conversations = chat_store.list_conversations(user_email)
    if conversations:
        return conversations[0]
    return chat_store.create_conversation(user_email, title=question)


def _persist_assistant_message(
    chat_store: ChatHistoryStore,
    user_email: str,
    conversation_id: str,
    message: dict[str, Any],
) -> None:
    chat_store.add_message(
        user_email,
        conversation_id,
        "assistant",
        str(message["content"]),
        payload=_chat_message_payload(message),
    )


async def _send_text(message: Any, text: str) -> None:
    for chunk in split_telegram_text(text) or ["Ответ отсутствует."]:
        await message.reply_text(chunk)


async def _send_sources(message: Any, sources: Any) -> None:
    normalized = [str(source) for source in sources if str(source).strip()]
    if normalized:
        await _send_text(message, "Источники:\n" + "\n".join(normalized))


async def _send_documents(message: Any, evidence: list[Any], settings: Any) -> None:
    unavailable: list[str] = []
    for index, group in enumerate(build_retrieved_file_groups(evidence, settings)):
        path = group.download_path
        try:
            if not group.can_download or path is None or path.stat().st_size > FILE_DOWNLOAD_BYTE_LIMIT:
                unavailable.append(group.display_path)
                continue
            with path.open("rb") as source:
                await message.reply_document(
                    document=source,
                    filename=group.download_name,
                    caption=group.display_path[:1024],
                )
        except Exception as exc:
            log_failure("telegram_source_send", exc, component="telegram", source_index=index)
            unavailable.append(group.display_path)
    if unavailable:
        await _send_text(
            message,
            "\n".join(SOURCE_UNAVAILABLE_TEXT.format(name=name) for name in unavailable),
        )


def _health_handler(ready: Event) -> type[BaseHTTPRequestHandler]:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/healthz":
                self.send_error(404)
                return
            status = 200 if ready.is_set() else 503
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok\n" if status == 200 else b"not ready\n")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return HealthHandler


def _duration_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


if __name__ == "__main__":
    main()
