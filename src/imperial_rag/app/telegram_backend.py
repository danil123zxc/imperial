from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
from secrets import compare_digest
import sqlite3
from time import perf_counter, time
from typing import Any, AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from imperial_rag.app.auth import AuthStore
from imperial_rag.app.chat_history import ChatHistoryStore
from imperial_rag.app.telegram import (
    MAX_TELEGRAM_TEXT_LENGTH,
    NEW_COMMAND,
    NEW_CONVERSATION_TEXT,
    NEW_CONVERSATION_TITLE,
    PayloadError,
    _json_body,
    _validated_secret,
    parse_allowed_user_ids,
    split_telegram_text,
)
from imperial_rag.app.web import _build_assistant_message, _build_query_failure_message, _chat_message_payload, _query_log_fields
from imperial_rag.observability import log_event, log_failure
from imperial_rag.observability.phoenix import phoenix_trace_context, trace_user_id_from_email

PROCESSING_LEASE_SECONDS = 30 * 60
DELIVERY_LEASE_SECONDS = 60
FAILURE_TEXT = "Не удалось подготовить ответ. Подробности доступны в локальных журналах."


@dataclass(frozen=True)
class TelegramBackendConfig:
    service_token: str
    allowed_user_ids: frozenset[int]
    phone_hash_secret: str


@dataclass(frozen=True)
class TelegramJob:
    update_id: int
    user_id: int
    question: str
    status: str
    result: dict[str, Any] | None
    attempts: int
    processing_lease_until: float | None
    delivery_lease_until: float | None
    created_at: float
    updated_at: float


class TelegramJobStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_jobs (
                    update_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'deliverable', 'delivering', 'delivered')),
                    result_json TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    processing_lease_until REAL,
                    delivery_lease_until REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS telegram_jobs_status_updated_idx
                ON telegram_jobs(status, updated_at, update_id)
                """
            )

    def create(self, update_id: int, user_id: int, question: str, *, now: float | None = None) -> tuple[TelegramJob, bool]:
        timestamp = time() if now is None else now
        self.initialize()
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO telegram_jobs(
                    update_id, user_id, question, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?)
                """,
                (update_id, user_id, question, timestamp, timestamp),
            )
            row = conn.execute("SELECT * FROM telegram_jobs WHERE update_id = ?", (update_id,)).fetchone()
        if row is None:
            raise RuntimeError("failed to create Telegram job")
        return _job_from_row(row), cursor.rowcount == 1

    def get(self, update_id: int) -> TelegramJob | None:
        self.initialize()
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM telegram_jobs WHERE update_id = ?", (update_id,)).fetchone()
        return _job_from_row(row) if row is not None else None

    def claim_processing(self, *, now: float | None = None) -> TelegramJob | None:
        timestamp = time() if now is None else now
        self.initialize()
        with self._immediate_connection() as conn:
            self._recover_expired(conn, timestamp)
            row = conn.execute(
                "SELECT * FROM telegram_jobs WHERE status = 'pending' ORDER BY created_at, update_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE telegram_jobs
                SET status = 'processing', attempts = attempts + 1,
                    processing_lease_until = ?, updated_at = ?
                WHERE update_id = ? AND status = 'pending'
                """,
                (timestamp + PROCESSING_LEASE_SECONDS, timestamp, int(row["update_id"])),
            )
            claimed = conn.execute("SELECT * FROM telegram_jobs WHERE update_id = ?", (int(row["update_id"]),)).fetchone()
        return _job_from_row(claimed) if claimed is not None else None

    def complete_processing(
        self,
        update_id: int,
        attempt: int,
        result: dict[str, Any],
        *,
        now: float | None = None,
    ) -> bool:
        timestamp = time() if now is None else now
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE telegram_jobs
                SET status = 'deliverable', result_json = ?, processing_lease_until = NULL, updated_at = ?
                WHERE update_id = ? AND status = 'processing' AND attempts = ?
                """,
                (json.dumps(result, ensure_ascii=False), timestamp, update_id, attempt),
            )
        return cursor.rowcount == 1

    def claim_delivery(self, *, now: float | None = None) -> TelegramJob | None:
        timestamp = time() if now is None else now
        self.initialize()
        with self._immediate_connection() as conn:
            self._recover_expired(conn, timestamp)
            row = conn.execute(
                "SELECT * FROM telegram_jobs WHERE status = 'deliverable' ORDER BY updated_at, update_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE telegram_jobs
                SET status = 'delivering', delivery_lease_until = ?, updated_at = ?
                WHERE update_id = ? AND status = 'deliverable'
                """,
                (timestamp + DELIVERY_LEASE_SECONDS, timestamp, int(row["update_id"])),
            )
            claimed = conn.execute("SELECT * FROM telegram_jobs WHERE update_id = ?", (int(row["update_id"]),)).fetchone()
        return _job_from_row(claimed) if claimed is not None else None

    def complete_delivery(self, update_id: int, *, now: float | None = None) -> bool:
        timestamp = time() if now is None else now
        with self._connection() as conn:
            row = conn.execute("SELECT status FROM telegram_jobs WHERE update_id = ?", (update_id,)).fetchone()
            if row is not None and row["status"] == "delivered":
                return True
            cursor = conn.execute(
                """
                UPDATE telegram_jobs
                SET status = 'delivered', delivery_lease_until = NULL, updated_at = ?
                WHERE update_id = ? AND status = 'delivering'
                """,
                (timestamp, update_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _recover_expired(conn: sqlite3.Connection, timestamp: float) -> None:
        conn.execute(
            """
            UPDATE telegram_jobs
            SET status = 'pending', processing_lease_until = NULL, updated_at = ?
            WHERE status = 'processing' AND processing_lease_until <= ?
            """,
            (timestamp, timestamp),
        )
        conn.execute(
            """
            UPDATE telegram_jobs
            SET status = 'deliverable', delivery_lease_until = NULL, updated_at = ?
            WHERE status = 'delivering' AND delivery_lease_until <= ?
            """,
            (timestamp, timestamp),
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def _immediate_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


class TelegramJobWorker:
    def __init__(self, store: TelegramJobStore, chat_store: ChatHistoryStore, runtime: Any, settings: Any):
        self.store = store
        self.chat_store = chat_store
        self.runtime = runtime
        self.settings = settings

    async def process_once(self) -> bool:
        job = self.store.claim_processing()
        if job is None:
            return False
        result = await self._query(job)
        self.store.complete_processing(job.update_id, job.attempts, result)
        return True

    async def _query(self, job: TelegramJob) -> dict[str, Any]:
        user_email = telegram_user_email(job.user_id)
        if job.question == NEW_COMMAND:
            self.chat_store.create_conversation(user_email, title=NEW_CONVERSATION_TITLE)
            return {"messages": [NEW_CONVERSATION_TEXT]}
        conversations = self.chat_store.list_conversations(user_email)
        conversation = conversations[0] if conversations else self.chat_store.create_conversation(user_email, title=job.question)
        user_message = self.chat_store.add_message(user_email, conversation.id, "user", job.question)
        if not self.chat_store.claim_assistant_response(user_email, conversation.id, user_message.id):
            return {"messages": [FAILURE_TEXT]}

        user_hash = trace_user_id_from_email(user_email)
        started_at = perf_counter()
        try:
            with phoenix_trace_context(
                conversation.phoenix_session_id,
                user_id=user_hash,
                metadata={"entrypoint": "telegram-backend"},
                tags=["imperial-rag", "telegram"],
            ):
                result = await asyncio.to_thread(self.runtime.query, job.question)
        except Exception as exc:
            log_failure(
                "telegram_query",
                exc,
                component="telegram-backend",
                duration_ms=_duration_ms(started_at),
                phoenix_session_id=conversation.phoenix_session_id,
                session_id=conversation.phoenix_session_id,
                user_hash=user_hash,
            )
            assistant_message = _build_query_failure_message(exc)
            assistant_message["error"]["type"] = "telegram_query_error"
            self._persist_assistant(user_email, conversation.id, assistant_message)
            return {"messages": split_telegram_text(assistant_message["content"])}

        normalized = result if isinstance(result, dict) else {"answer": getattr(result, "answer", str(result))}
        assistant_message = _build_assistant_message(normalized, self.settings)
        self._persist_assistant(user_email, conversation.id, assistant_message)
        messages = split_telegram_text(assistant_message["content"]) or ["Ответ отсутствует."]
        sources = [str(source).strip() for source in normalized.get("sources") or normalized.get("citations") or [] if str(source).strip()]
        if sources:
            messages.extend(split_telegram_text("Источники:\n" + "\n".join(sources)))
        log_event(
            "imperial_rag.telegram_query",
            operation="telegram_query",
            status="success",
            component="telegram-backend",
            duration_ms=_duration_ms(started_at),
            phoenix_session_id=conversation.phoenix_session_id,
            session_id=conversation.phoenix_session_id,
            user_hash=user_hash,
            **_query_log_fields(normalized),
        )
        return {"messages": messages}

    def _persist_assistant(self, user_email: str, conversation_id: str, message: dict[str, Any]) -> None:
        self.chat_store.add_message(
            user_email,
            conversation_id,
            "assistant",
            str(message["content"]),
            payload=_chat_message_payload(message),
        )


def load_configuration(environ: Mapping[str, str] | None = None) -> TelegramBackendConfig:
    if environ is None:
        from imperial_rag.env import load_project_env

        load_project_env()
    values = os.environ if environ is None else environ
    return TelegramBackendConfig(
        service_token=_validated_secret(values, "IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN"),
        allowed_user_ids=parse_allowed_user_ids(values.get("IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS")),
        phone_hash_secret=_validated_secret(values, "IMPERIAL_RAG_TELEGRAM_PHONE_HASH_SECRET"),
    )


def create_app(
    config: TelegramBackendConfig | None = None,
    store: TelegramJobStore | None = None,
    worker: TelegramJobWorker | None = None,
    access_store: AuthStore | None = None,
) -> Starlette:
    application = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/internal/telegram/access", check_access, methods=["POST"]),
            Route("/internal/telegram/jobs", create_job, methods=["POST"]),
            Route("/internal/telegram/deliveries/claim", claim_delivery, methods=["POST"]),
            Route("/internal/telegram/deliveries/{update_id:int}/complete", complete_delivery, methods=["POST"]),
        ],
        lifespan=_lifespan,
    )
    application.state.config = config
    application.state.store = store
    application.state.worker = worker
    application.state.access_store = access_store
    application.state.ready = bool(config and store and access_store)
    return application


async def healthz(request: Request) -> Response:
    if not _authorized(request):
        return PlainTextResponse("unauthorized\n", status_code=401)
    return PlainTextResponse("ok\n" if request.app.state.ready else "not ready\n", status_code=200 if request.app.state.ready else 503)


async def check_access(request: Request) -> Response:
    if not _authorized(request):
        return PlainTextResponse("unauthorized\n", status_code=401)
    try:
        payload = await _json_body(request)
        user_id = _positive_int(payload.get("user_id"), "user_id")
        username = payload.get("username")
        if username is not None and (not isinstance(username, str) or len(username) > 32):
            raise PayloadError("username must be a string of at most 32 characters", 400)
        contact = payload.get("contact")
        if contact is not None and not isinstance(contact, dict):
            raise PayloadError("contact must be an object", 400)
        phone_number = contact.get("phone_number") if contact else None
        contact_user_id = contact.get("user_id") if contact else None
        if phone_number is not None and (not isinstance(phone_number, str) or len(phone_number) > 32):
            raise PayloadError("contact phone number must be a string of at most 32 characters", 400)
        if contact_user_id is not None:
            contact_user_id = _positive_int(contact_user_id, "contact user_id")
    except PayloadError as exc:
        return PlainTextResponse(f"{exc}\n", status_code=exc.status_code)

    authorization = _access_store(request.app).authorize_telegram_user(
        user_id,
        username=username,
        phone_number=phone_number,
        contact_user_id=contact_user_id,
        phone_hash_secret=_config(request.app).phone_hash_secret,
    )
    result = {
        "authorized": authorization.authorized or user_id in _config(request.app).allowed_user_ids,
        "newly_bound": authorization.newly_bound,
    }
    return JSONResponse(result, status_code=200 if result["authorized"] else 403)


async def create_job(request: Request) -> Response:
    if not _authorized(request):
        return PlainTextResponse("unauthorized\n", status_code=401)
    try:
        payload = await _json_body(request)
        update_id = _positive_int(payload.get("update_id"), "update_id")
        user_id = _positive_int(payload.get("user_id"), "user_id")
        question = payload.get("question")
        if not _telegram_user_allowed(request.app, user_id):
            return PlainTextResponse("forbidden\n", status_code=403)
        if not isinstance(question, str) or not question.strip():
            raise PayloadError("question must be a non-empty string", 400)
        question = question.strip()
        if len(question) > MAX_TELEGRAM_TEXT_LENGTH:
            raise PayloadError("question is too long", 400)
    except PayloadError as exc:
        return PlainTextResponse(f"{exc}\n", status_code=exc.status_code)
    job, created = _store(request.app).create(update_id, user_id, question)
    return JSONResponse(_job_payload(job), status_code=202 if created else 200)


async def claim_delivery(request: Request) -> Response:
    if not _authorized(request):
        return PlainTextResponse("unauthorized\n", status_code=401)
    try:
        await _json_body(request)
    except PayloadError as exc:
        return PlainTextResponse(f"{exc}\n", status_code=exc.status_code)
    job = _store(request.app).claim_delivery()
    if job is None:
        return Response(status_code=204)
    result = job.result or {}
    return JSONResponse({"update_id": job.update_id, "user_id": job.user_id, "messages": result.get("messages") or []})


async def complete_delivery(request: Request) -> Response:
    if not _authorized(request):
        return PlainTextResponse("unauthorized\n", status_code=401)
    try:
        await _json_body(request)
        update_id = _positive_int(request.path_params.get("update_id"), "update_id")
    except PayloadError as exc:
        return PlainTextResponse(f"{exc}\n", status_code=exc.status_code)
    if not _store(request.app).complete_delivery(update_id):
        return PlainTextResponse("delivery is not claimable\n", status_code=409)
    return JSONResponse({"update_id": update_id, "status": "delivered"})


@asynccontextmanager
async def _lifespan(application: Starlette) -> AsyncIterator[None]:
    if (
        application.state.config is None
        or application.state.store is None
        or application.state.worker is None
        or application.state.access_store is None
    ):
        from imperial_rag.answering.runtime import create_runtime
        from imperial_rag.config import Settings, apply_active_index_pointer
        from imperial_rag.env import load_project_env
        from imperial_rag.observability import configure_observability
        from imperial_rag.observability.phoenix import configure_phoenix_tracing

        load_project_env()
        config = application.state.config or load_configuration()
        settings = apply_active_index_pointer(Settings())
        configure_observability(settings)
        configure_phoenix_tracing(settings)
        store = application.state.store or TelegramJobStore(Path(settings.chat_history_db_path))
        store.initialize()
        chat_store = ChatHistoryStore(Path(settings.chat_history_db_path))
        chat_store.initialize()
        application.state.config = config
        application.state.store = store
        application.state.access_store = application.state.access_store or AuthStore(Path(settings.auth_db_path))
        application.state.access_store.initialize()
        application.state.worker = application.state.worker or TelegramJobWorker(
            store,
            chat_store,
            create_runtime(settings),
            settings,
        )
    application.state.ready = True
    task = asyncio.create_task(_worker_loop(application))
    try:
        yield
    finally:
        application.state.ready = False
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _worker_loop(application: Starlette) -> None:
    while True:
        try:
            processed = await application.state.worker.process_once()
        except Exception as exc:
            log_failure("telegram_worker", exc, component="telegram-backend")
            processed = False
        if not processed:
            await asyncio.sleep(1)


def telegram_user_email(user_id: int) -> str:
    if user_id <= 0:
        raise ValueError("Telegram user ID must be positive")
    return f"telegram-{user_id}@users.invalid"


def _authorized(request: Request) -> bool:
    supplied = request.headers.get("Authorization", "")
    expected = f"Bearer {_config(request.app).service_token}"
    return compare_digest(supplied, expected)


def _telegram_user_allowed(application: Starlette, user_id: int) -> bool:
    if user_id in _config(application).allowed_user_ids:
        return True
    return _access_store(application).authorize_telegram_user(
        user_id,
        phone_hash_secret=_config(application).phone_hash_secret,
    ).authorized


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise PayloadError(f"{name} must be a positive integer", 400)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"{name} must be a positive integer", 400) from exc
    if parsed <= 0:
        raise PayloadError(f"{name} must be a positive integer", 400)
    return parsed


def _job_from_row(row: sqlite3.Row) -> TelegramJob:
    result = json.loads(row["result_json"]) if row["result_json"] else None
    return TelegramJob(
        update_id=int(row["update_id"]),
        user_id=int(row["user_id"]),
        question=str(row["question"]),
        status=str(row["status"]),
        result=result if isinstance(result, dict) else None,
        attempts=int(row["attempts"]),
        processing_lease_until=float(row["processing_lease_until"]) if row["processing_lease_until"] is not None else None,
        delivery_lease_until=float(row["delivery_lease_until"]) if row["delivery_lease_until"] is not None else None,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _job_payload(job: TelegramJob) -> dict[str, Any]:
    return {"update_id": job.update_id, "user_id": job.user_id, "status": job.status, "attempts": job.attempts}


def _config(application: Starlette) -> TelegramBackendConfig:
    config = application.state.config
    if not isinstance(config, TelegramBackendConfig):
        raise RuntimeError("Telegram backend configuration is unavailable")
    return config


def _store(application: Starlette) -> TelegramJobStore:
    store = application.state.store
    if not isinstance(store, TelegramJobStore):
        raise RuntimeError("Telegram job store is unavailable")
    return store


def _access_store(application: Starlette) -> AuthStore:
    store = application.state.access_store
    if not isinstance(store, AuthStore):
        raise RuntimeError("Telegram access store is unavailable")
    return store


def _duration_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


app = create_app()
