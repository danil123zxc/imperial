from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from imperial_rag.app.auth import AuthStore
from imperial_rag.app.chat_history import ChatHistoryStore
from imperial_rag.app.telegram import (
    NEW_COMMAND,
    NEW_CONVERSATION_TEXT,
    NEW_CONVERSATION_TITLE,
    TelegramWebhookConfig,
    create_app as create_webhook_app,
)
from imperial_rag.app.telegram_backend import (
    TelegramBackendConfig,
    deliver_once,
    TelegramJobStore,
    TelegramJobWorker,
    create_app as create_backend_app,
    telegram_user_email,
)

TOKEN = "t" * 32
WEBHOOK_SECRET = "w" * 32
PHONE_SECRET = "p" * 32


class FakeRuntime:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result or {}
        self.error = error
        self.questions: list[str] = []

    def query(self, question: str):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.result


class FakeTelegramClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self.messages.append(kwargs["json"])
        return httpx.Response(200, json={"ok": True, "result": {}})


def _stores(tmp_path: Path) -> tuple[TelegramJobStore, ChatHistoryStore]:
    path = tmp_path / "chat_history.sqlite3"
    jobs = TelegramJobStore(path)
    jobs.initialize()
    history = ChatHistoryStore(path)
    history.initialize()
    return jobs, history


def _worker(tmp_path: Path, runtime: FakeRuntime) -> tuple[TelegramJobStore, TelegramJobWorker, ChatHistoryStore]:
    jobs, history = _stores(tmp_path)
    settings = SimpleNamespace(documents_root=tmp_path / "documents", extraction_root=tmp_path / "extracted")
    return jobs, TelegramJobWorker(jobs, history, runtime, settings), history


def _access_store(tmp_path: Path) -> AuthStore:
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.initialize()
    return store


async def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://backend.example") as client:
        return await client.request(method, path, **kwargs)


def test_duplicate_jobs_and_expired_leases_recover_after_restart(tmp_path: Path) -> None:
    store, _ = _stores(tmp_path)
    original, created = store.create(10, 123, "first", now=100)
    duplicate, duplicate_created = store.create(10, 123, "changed", now=101)
    assert created is True and duplicate_created is False
    assert original.question == duplicate.question == "first"

    processing = store.claim_processing(now=110)
    assert processing and processing.status == "processing" and processing.attempts == 1
    reopened = TelegramJobStore(store.db_path)
    recovered = reopened.claim_processing(now=110 + 30 * 60 + 1)
    assert recovered and recovered.update_id == 10 and recovered.attempts == 2
    assert reopened.complete_processing(10, recovered.attempts, {"messages": ["answer"]}, now=2000)

    delivery = reopened.claim_delivery(now=2001)
    assert delivery and delivery.status == "delivering"
    redelivery = reopened.claim_delivery(now=2067)
    assert redelivery and redelivery.update_id == 10
    assert reopened.complete_delivery(10, redelivery.delivery_attempts, now=2068)
    assert reopened.complete_delivery(10, redelivery.delivery_attempts, now=2069)
    assert reopened.get(10).status == "delivered"


def test_internal_api_requires_bearer_json_positive_ids_and_allowlist(tmp_path: Path) -> None:
    store, _ = _stores(tmp_path)
    app = create_backend_app(
        TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"),
        store,
        access_store=_access_store(tmp_path),
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}

    assert asyncio.run(_request(app, "GET", "/healthz")).status_code == 401
    assert asyncio.run(_request(app, "GET", "/healthz", headers=headers)).status_code == 200
    assert asyncio.run(_request(app, "POST", "/internal/telegram/jobs", json={}, headers={"Authorization": "Bearer wrong"})).status_code == 401
    assert asyncio.run(
        _request(app, "POST", "/internal/telegram/jobs", content=b"{}", headers={**headers, "content-type": "text/plain"})
    ).status_code == 415
    assert asyncio.run(
        _request(app, "POST", "/internal/telegram/jobs", json={"update_id": -1, "user_id": 123, "question": "q"}, headers=headers)
    ).status_code == 400
    assert asyncio.run(
        _request(app, "POST", "/internal/telegram/jobs", json={"update_id": 1, "user_id": 999, "question": "q"}, headers=headers)
    ).status_code == 403

    accepted = asyncio.run(
        _request(app, "POST", "/internal/telegram/jobs", json={"update_id": 1, "user_id": 123, "question": "q"}, headers=headers)
    )
    duplicate = asyncio.run(
        _request(app, "POST", "/internal/telegram/jobs", json={"update_id": 1, "user_id": 123, "question": "q"}, headers=headers)
    )
    assert accepted.status_code == 202
    assert duplicate.status_code == 200


def test_access_api_binds_username_and_phone_then_revocation_blocks_jobs(tmp_path: Path) -> None:
    jobs, _ = _stores(tmp_path)
    access = _access_store(tmp_path)
    admin = access.bootstrap_admin("admin@example.com", "admin-password")
    username_grant = access.add_telegram_access_grant(admin.email, "@Alice_User", PHONE_SECRET)
    phone_grant = access.add_telegram_access_grant(admin.email, "+7 999 123-45-67", PHONE_SECRET)
    app = create_backend_app(
        TelegramBackendConfig(TOKEN, frozenset(), PHONE_SECRET, "https://render.example"),
        jobs,
        access_store=access,
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}

    username = asyncio.run(
        _request(
            app,
            "POST",
            "/internal/telegram/access",
            json={"user_id": 101, "username": "ALICE_USER"},
            headers=headers,
        )
    )
    assert username.status_code == 200
    assert username.json() == {"authorized": True, "newly_bound": True}
    assert asyncio.run(
        _request(
            app,
            "POST",
            "/internal/telegram/jobs",
            json={"update_id": 1, "user_id": 101, "question": "q"},
            headers=headers,
        )
    ).status_code == 202

    forwarded_contact = asyncio.run(
        _request(
            app,
            "POST",
            "/internal/telegram/access",
            json={
                "user_id": 202,
                "contact": {"user_id": 999, "phone_number": "+79991234567"},
            },
            headers=headers,
        )
    )
    assert forwarded_contact.status_code == 403
    own_contact = asyncio.run(
        _request(
            app,
            "POST",
            "/internal/telegram/access",
            json={
                "user_id": 202,
                "contact": {"user_id": 202, "phone_number": "+79991234567"},
            },
            headers=headers,
        )
    )
    assert own_contact.status_code == 200
    assert own_contact.json() == {"authorized": True, "newly_bound": True}

    assert access.revoke_telegram_access_grant(admin.email, username_grant.id)
    assert asyncio.run(
        _request(
            app,
            "POST",
            "/internal/telegram/jobs",
            json={"update_id": 2, "user_id": 101, "question": "blocked"},
            headers=headers,
        )
    ).status_code == 403
    assert access.revoke_telegram_access_grant(admin.email, phone_grant.id)


def test_worker_persists_answer_source_labels_and_generic_failures(tmp_path: Path) -> None:
    runtime = FakeRuntime({"answer": "Ответ [S1]", "sources": ["[S1] manual.pdf"], "retrieval": {"final_evidence": 1}})
    store, worker, history = _worker(tmp_path, runtime)
    store.create(1, 123, "Вопрос")
    assert asyncio.run(worker.process_once()) is True
    job = store.get(1)
    assert job and job.status == "deliverable"
    assert job.result == {"messages": ["Ответ [S1]", "Источники:\n[S1] manual.pdf"]}
    conversation = history.list_conversations(telegram_user_email(123))[0]
    assert [message.role for message in history.list_messages(telegram_user_email(123), conversation.id)] == ["user", "assistant"]

    failed_runtime = FakeRuntime(error=RuntimeError("private provider failure"))
    failed_store, failed_worker, _ = _worker(tmp_path / "failed", failed_runtime)
    failed_store.create(2, 123, "Sensitive")
    assert asyncio.run(failed_worker.process_once()) is True
    result = failed_store.get(2).result
    assert result and "private provider failure" not in str(result)
    assert "Не удалось подготовить ответ" in result["messages"][0]


def test_new_command_creates_one_fresh_conversation_without_querying(tmp_path: Path) -> None:
    runtime = FakeRuntime({"answer": "Ответ"})
    store, worker, history = _worker(tmp_path, runtime)
    _, created = store.create(1, 123, NEW_COMMAND)
    _, duplicate_created = store.create(1, 123, "changed")

    assert created is True and duplicate_created is False
    assert asyncio.run(worker.process_once()) is True
    assert asyncio.run(worker.process_once()) is False
    assert runtime.questions == []
    conversations = history.list_conversations(telegram_user_email(123))
    assert len(conversations) == 1
    assert conversations[0].title == NEW_CONVERSATION_TITLE
    assert history.list_messages(telegram_user_email(123), conversations[0].id) == []
    assert store.get(1).result == {"messages": [NEW_CONVERSATION_TEXT]}

    store.create(2, 123, "Следующий вопрос")
    assert asyncio.run(worker.process_once()) is True
    assert runtime.questions == ["Следующий вопрос"]
    assert len(history.list_conversations(telegram_user_email(123))) == 1
    assert [message.role for message in history.list_messages(telegram_user_email(123), conversations[0].id)] == [
        "user",
        "assistant",
    ]


def test_webhook_to_durable_worker_to_delivery_integration(tmp_path: Path) -> None:
    runtime = FakeRuntime({"answer": "Ответ", "sources": ["[S1] handbook.pdf"]})
    store, worker, _ = _worker(tmp_path, runtime)
    backend_app = create_backend_app(
        TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"),
        store,
        worker,
        _access_store(tmp_path),
    )
    webhook_app = create_webhook_app(
        TelegramWebhookConfig(
            bot_token="bot-token",
            webhook_url="https://render.example/telegram/webhook",
            webhook_secret=WEBHOOK_SECRET,
            backend_url="https://backend.example",
            service_token=TOKEN,
        )
    )
    telegram = FakeTelegramClient()

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=backend_app), base_url="https://backend.example"
        ) as backend_client:
            webhook_app.state.backend_client = backend_client
            webhook_app.state.telegram_client = telegram
            response = await _request(
                webhook_app,
                "POST",
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": WEBHOOK_SECRET},
                json={
                    "update_id": 7,
                    "message": {"from": {"id": 123}, "chat": {"id": 123, "type": "private"}, "text": "Вопрос"},
                },
            )
            assert response.status_code == 200
            assert await worker.process_once()
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=webhook_app), base_url="https://render.example"
            ) as delivery_client:
                backend_app.state.delivery_client = delivery_client
                assert await deliver_once(backend_app)

    asyncio.run(exercise())
    assert [message["text"] for message in telegram.messages] == [
        "Вопрос принят. Готовлю ответ.",
        "Ответ",
        "Источники:\n[S1] handbook.pdf",
    ]
    assert store.get(7).status == "delivered"


def test_additive_migration_preserves_existing_job_and_is_repeatable(tmp_path: Path) -> None:
    import sqlite3
    from contextlib import closing

    path = tmp_path / "legacy.sqlite3"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""CREATE TABLE telegram_jobs (
            update_id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, question TEXT NOT NULL,
            status TEXT NOT NULL, result_json TEXT, attempts INTEGER NOT NULL DEFAULT 0,
            processing_lease_until REAL, delivery_lease_until REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL
        )""")
        conn.execute("""INSERT INTO telegram_jobs VALUES
            (1, 123, 'question', 'deliverable', '{"messages": ["answer"]}', 2, NULL, NULL, 10, 20)""")
    store = TelegramJobStore(path)
    store.initialize()
    store.initialize()
    job = store.get(1)
    assert job and job.result == {"messages": ["answer"]} and job.question == "question"
    assert job.attempts == 2 and job.delivery_attempts == 0 and job.next_delivery_at == 0
    assert job.created_at == 10 and job.updated_at == 20
    assert store.claim_delivery(now=100).delivery_attempts == 1


def _ready_job(store: TelegramJobStore, update_id: int = 1, messages=None) -> None:
    store.create(update_id, 123, "question", now=1)
    job = store.claim_processing(now=2)
    assert job and store.complete_processing(update_id, job.attempts, {"messages": messages or ["answer"]}, now=3)


def test_concurrent_delivery_claims_and_stale_attempts(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store, _ = _stores(tmp_path)
    _ready_job(store, messages=["answer", "sources"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: TelegramJobStore(store.db_path).claim_delivery(now=100), range(2)))
    claimed = [job for job in claims if job is not None]
    assert len(claimed) == 1
    first = claimed[0]
    assert first.delivery_lease_until == 185  # 20N + 45, beyond the 20N + 15 request deadline.
    assert store.claim_delivery(now=184) is None
    assert not store.complete_delivery(1, first.delivery_attempts, now=185)
    second = TelegramJobStore(store.db_path).claim_delivery(now=185)
    assert second and second.delivery_attempts == 2
    assert not store.complete_delivery(1, first.delivery_attempts, now=186)
    assert not store.retry_delivery(1, first.delivery_attempts, 900, now=186)
    assert store.complete_delivery(1, second.delivery_attempts, now=187)
    assert store.complete_delivery(1, second.delivery_attempts, now=188)
    assert not store.complete_delivery(1, first.delivery_attempts, now=189)


def test_retry_due_time_survives_reopening_and_does_not_block_other_jobs(tmp_path: Path) -> None:
    store, _ = _stores(tmp_path)
    _ready_job(store)
    first = store.claim_delivery(now=100)
    assert first and store.retry_delivery(1, first.delivery_attempts, 420, now=101)
    reopened = TelegramJobStore(store.db_path)
    assert reopened.get(1).next_delivery_at == 521
    assert reopened.claim_delivery(now=520) is None
    _ready_job(reopened, update_id=2)
    assert reopened.claim_delivery(now=520).update_id == 2
    assert reopened.claim_delivery(now=521).update_id == 1


def test_delivery_backoff_is_persisted_and_capped(tmp_path: Path, monkeypatch) -> None:
    from imperial_rag.app import telegram_backend

    clock = [100.0]
    monkeypatch.setattr(telegram_backend, "time", lambda: clock[0])
    store, _ = _stores(tmp_path)
    _ready_job(store)
    app = create_backend_app(TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"), store)

    class Client:
        async def post(self, url, **kwargs):
            assert url == "https://render.example/internal/telegram/deliver"
            assert kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}
            assert kwargs["timeout"] == 35
            raise httpx.ConnectError("private")

    app.state.delivery_client = Client()
    for attempt, delay in enumerate([5, 10, 20, 40, 80, 160, 300, 300], 1):
        assert asyncio.run(deliver_once(app))
        job = TelegramJobStore(store.db_path).get(1)
        assert job.status == "deliverable" and job.delivery_attempts == attempt
        assert job.next_delivery_at == clock[0] + delay
        assert asyncio.run(deliver_once(app)) is False
        clock[0] += delay


@pytest.mark.parametrize("response,delay", [
    (httpx.Response(200, json={"update_id": 999, "status": "delivered"}), 5),
    (httpx.Response(200, json={"update_id": True, "status": "delivered"}), 5),
    (httpx.Response(200, json={"update_id": 1, "status": "pending"}), 5),
    (httpx.Response(200, text="private"), 5),
    (httpx.Response(202, json={"update_id": 1, "status": "delivered"}), 5),
    (httpx.Response(502, json={"retryable": True, "retry_after": 600}), 600),
    (httpx.Response(502, json={"retryable": True, "retry_after": "600"}), 5),
    (httpx.Response(502, json={"retryable": False}), 900),
    (httpx.Response(401, text="private"), 900),
    (httpx.Response(413, text="private"), 900),
    (httpx.ReadTimeout("private"), 5),
])
def test_failed_or_invalid_confirmation_keeps_result_queued(tmp_path: Path, monkeypatch, response, delay) -> None:
    from imperial_rag.app import telegram_backend

    monkeypatch.setattr(telegram_backend, "time", lambda: 100)
    store, _ = _stores(tmp_path)
    _ready_job(store)
    app = create_backend_app(TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"), store)

    class Client:
        async def post(self, url, **kwargs):
            if isinstance(response, Exception):
                raise response
            return response

    app.state.delivery_client = Client()
    assert asyncio.run(deliver_once(app))
    job = TelegramJobStore(store.db_path).get(1)
    assert job.status == "deliverable" and job.next_delivery_at == 100 + delay
    assert job.result == {"messages": ["answer"]}


def test_render_url_is_required_and_must_be_https_origin() -> None:
    from imperial_rag.app.telegram_backend import load_configuration

    values = {"IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN": TOKEN, "IMPERIAL_RAG_TELEGRAM_PHONE_HASH_SECRET": PHONE_SECRET}
    for invalid in ("", "http://render.example", "https://render.example/path", "https://user:pass@render.example", "https://render.example?q=1"):
        with pytest.raises(ValueError, match="IMPERIAL_RAG_TELEGRAM_RENDER_URL"):
            load_configuration({**values, "IMPERIAL_RAG_TELEGRAM_RENDER_URL": invalid})
    assert load_configuration({**values, "IMPERIAL_RAG_TELEGRAM_RENDER_URL": "https://render.example/"}).render_url == "https://render.example"


def test_backend_lifespan_runs_delivery_independently_and_cancels_tasks(tmp_path: Path, monkeypatch) -> None:
    from imperial_rag.app import telegram_backend

    store, _ = _stores(tmp_path)
    app = create_backend_app(TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"),
                             store, access_store=_access_store(tmp_path))
    started = []
    stopped = []

    class Worker:
        async def process_once(self):
            started.append("worker")
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append("worker")

    async def delivery(application):
        assert application.state.delivery_client is not None
        started.append("delivery")
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append("delivery")

    app.state.worker = Worker()
    monkeypatch.setattr(telegram_backend, "deliver_once", delivery)

    async def exercise():
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)
            assert sorted(started) == ["delivery", "worker"]
            assert app.state.ready
        assert sorted(stopped) == ["delivery", "worker"]
        assert not app.state.ready
        assert app.state.delivery_client.is_closed
        for path in ("/internal/telegram/deliveries/claim", "/internal/telegram/deliveries/1/complete"):
            assert (await _request(app, "POST", path, json={}, headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 404

    asyncio.run(exercise())


def test_partial_send_retries_the_whole_answer_after_restart(tmp_path: Path, monkeypatch) -> None:
    from imperial_rag.app import telegram_backend

    clock = [100]
    monkeypatch.setattr(telegram_backend, "time", lambda: clock[0])
    store, _ = _stores(tmp_path)
    _ready_job(store, messages=["first", "second"])
    backend = create_backend_app(TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"), store)
    render = create_webhook_app(TelegramWebhookConfig("bot", "https://render.example/telegram/webhook", WEBHOOK_SECRET, "https://backend.example", TOKEN))
    sent = []

    class Telegram:
        async def post(self, url, **kwargs):
            sent.append(kwargs["json"]["text"])
            if len(sent) == 2:
                return httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 60}})
            return httpx.Response(200, json={"ok": True})

    render.state.telegram_client = Telegram()

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=render)) as client:
            backend.state.delivery_client = client
            assert await deliver_once(backend)
            backend.state.store = TelegramJobStore(store.db_path)
            assert store.get(1).status == "deliverable"
            assert store.get(1).next_delivery_at == 160
            assert not await deliver_once(backend)
            clock[0] = 160
            assert await deliver_once(backend)

    asyncio.run(exercise())
    assert sent == ["first", "second", "first", "second"]
    assert store.get(1).status == "delivered"


def test_backend_deadline_reschedules_and_cancellation_recovers_by_lease(tmp_path: Path, monkeypatch) -> None:
    from imperial_rag.app import telegram_backend

    clock = [100]
    monkeypatch.setattr(telegram_backend, "time", lambda: clock[0])
    real_timeout = asyncio.timeout
    deadlines = []

    def short_timeout(seconds):
        deadlines.append(seconds)
        return real_timeout(0.01)

    monkeypatch.setattr(telegram_backend.asyncio, "timeout", short_timeout)
    store, _ = _stores(tmp_path)
    _ready_job(store)
    app = create_backend_app(TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET, "https://render.example"), store)

    class Client:
        async def post(self, url, **kwargs):
            await asyncio.Event().wait()

    app.state.delivery_client = Client()
    assert asyncio.run(deliver_once(app))
    assert deadlines == [35] and store.get(1).next_delivery_at == 105
    monkeypatch.setattr(telegram_backend.asyncio, "timeout", real_timeout)
    clock[0] = 105

    async def cancel():
        task = asyncio.create_task(deliver_once(app))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    reopened = TelegramJobStore(store.db_path)
    assert reopened.get(1).status == "delivering"
    assert reopened.claim_delivery(now=169) is None
    assert reopened.claim_delivery(now=170).delivery_attempts == 3
