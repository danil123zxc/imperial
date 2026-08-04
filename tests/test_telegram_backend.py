from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from imperial_rag.app.auth import AuthStore
from imperial_rag.app.chat_history import ChatHistoryStore
from imperial_rag.app.telegram import (
    NEW_COMMAND,
    NEW_CONVERSATION_TEXT,
    NEW_CONVERSATION_TITLE,
    TelegramWebhookConfig,
    create_app as create_webhook_app,
    deliver_once,
)
from imperial_rag.app.telegram_backend import (
    TelegramBackendConfig,
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
    redelivery = reopened.claim_delivery(now=2062)
    assert redelivery and redelivery.update_id == 10
    assert reopened.complete_delivery(10, now=2063)
    assert reopened.complete_delivery(10, now=2064)
    assert reopened.get(10).status == "delivered"


def test_internal_api_requires_bearer_json_positive_ids_and_allowlist(tmp_path: Path) -> None:
    store, _ = _stores(tmp_path)
    app = create_backend_app(
        TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET),
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
        TelegramBackendConfig(TOKEN, frozenset(), PHONE_SECRET),
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
        TelegramBackendConfig(TOKEN, frozenset({123}), PHONE_SECRET),
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
            assert await deliver_once(webhook_app)

    asyncio.run(exercise())
    assert [message["text"] for message in telegram.messages] == [
        "Вопрос принят. Готовлю ответ.",
        "Ответ",
        "Источники:\n[S1] handbook.pdf",
    ]
    assert store.get(7).status == "delivered"
