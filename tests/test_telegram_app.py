from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from imperial_rag.app import telegram
from imperial_rag.app.telegram import (
    ACCESS_GRANTED_TEXT,
    ACCESS_REQUIRED_TEXT,
    ACCEPTED_TEXT,
    CONTACT_DENIED_TEXT,
    HELP_TEXT,
    NEW_COMMAND,
    TELEGRAM_COMMANDS,
    TelegramWebhookConfig,
    UNKNOWN_COMMAND_TEXT,
    WELCOME_TEXT,
    create_app,
    load_configuration,
    parse_allowed_user_ids,
    split_telegram_text,
)

SECRET = "s" * 32
TOKEN = "t" * 32


class FakeClient:
    def __init__(self, backend_status: int = 202, access_status: int = 200) -> None:
        self.backend_status = backend_status
        self.access_status = access_status
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self.calls.append((url, kwargs))
        if "api.telegram.org" in url:
            return httpx.Response(200, json={"ok": True, "result": True})
        if url.endswith("/internal/telegram/access"):
            return httpx.Response(
                self.access_status,
                json={"authorized": self.access_status == 200, "newly_bound": False},
            )
        return httpx.Response(self.backend_status, json={"status": "pending"})


def _config() -> TelegramWebhookConfig:
    return TelegramWebhookConfig(
        bot_token="bot-token",
        webhook_url="https://render.example/telegram/webhook",
        webhook_secret=SECRET,
        backend_url="https://backend.example",
        service_token=TOKEN,
    )


def _update(
    *,
    update_id: int = 1,
    user_id: int = 123,
    chat_type: str = "private",
    text: str | None = "Вопрос?",
    username: str | None = "alice_user",
    contact: dict | None = None,
) -> dict:
    user = {"id": user_id}
    if username is not None:
        user["username"] = username
    message = {
        "from": user,
        "chat": {"id": user_id, "type": chat_type},
    }
    if text is not None:
        message["text"] = text
    if contact is not None:
        message["contact"] = contact
    return {
        "update_id": update_id,
        "message": message,
    }


async def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://render.example") as client:
        return await client.request(method, path, **kwargs)


def test_configuration_requires_https_strong_secrets_and_optional_numeric_allowlist() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        load_configuration({})
    with pytest.raises(ValueError, match="comma-separated"):
        parse_allowed_user_ids("12,nope")
    with pytest.raises(ValueError, match="positive"):
        parse_allowed_user_ids("0")
    assert parse_allowed_user_ids("") == frozenset()
    assert parse_allowed_user_ids("123, 456,123") == frozenset({123, 456})

    values = {
        "TELEGRAM_BOT_TOKEN": " token ",
        "TELEGRAM_WEBHOOK_URL": "https://render.example/telegram/webhook",
        "TELEGRAM_WEBHOOK_SECRET": SECRET,
        "IMPERIAL_RAG_TELEGRAM_BACKEND_URL": "https://backend.example/",
        "IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN": TOKEN,
    }
    config = load_configuration(values)
    assert config.backend_url == "https://backend.example"

    values["IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN"] = "weak"
    with pytest.raises(ValueError, match="32-256"):
        load_configuration(values)

    values["IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN"] = TOKEN
    values["TELEGRAM_WEBHOOK_URL"] = "https://render.example/wrong"
    with pytest.raises(ValueError, match="/telegram/webhook"):
        load_configuration(values)


def test_split_telegram_text_respects_limit() -> None:
    text = ("word " * 1200).strip()
    chunks = split_telegram_text(text)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 4096 for chunk in chunks)
    assert " ".join(chunks) == text


def test_webhook_validates_secret_payload_access_and_groups() -> None:
    app = create_app(_config())
    backend = FakeClient(access_status=403)
    telegram_client = FakeClient()
    app.state.backend_client = backend
    app.state.telegram_client = telegram_client

    assert asyncio.run(_request(app, "POST", "/telegram/webhook", json=_update())).status_code == 403
    malformed = asyncio.run(
        _request(
            app,
            "POST",
            "/telegram/webhook",
            content=b"not-json",
            headers={"content-type": "application/json", "X-Telegram-Bot-Api-Secret-Token": SECRET},
        )
    )
    assert malformed.status_code == 400

    headers = {"X-Telegram-Bot-Api-Secret-Token": SECRET}
    unauthorized = asyncio.run(_request(app, "POST", "/telegram/webhook", json=_update(user_id=999), headers=headers))
    group = asyncio.run(_request(app, "POST", "/telegram/webhook", json=_update(chat_type="group"), headers=headers))
    oversized = json.dumps(_update(text="x" * 17000)).encode()
    too_large = asyncio.run(
        _request(app, "POST", "/telegram/webhook", content=oversized, headers={**headers, "content-type": "application/json"})
    )
    assert unauthorized.status_code == group.status_code == 200
    assert too_large.status_code == 413
    assert len(backend.calls) == 1
    assert backend.calls[0][1]["json"] == {"user_id": 999, "username": "alice_user"}
    assert telegram_client.calls[0][1]["json"] == {
        "chat_id": 999,
        "text": ACCESS_REQUIRED_TEXT,
        "reply_markup": {
            "keyboard": [[{"text": "Поделиться номером", "request_contact": True}]],
            "resize_keyboard": True,
            "one_time_keyboard": True,
        },
    }


def test_webhook_verifies_shared_contact_before_confirming_access() -> None:
    app = create_app(_config())
    backend = FakeClient(access_status=200)
    telegram_client = FakeClient()
    app.state.backend_client = backend
    app.state.telegram_client = telegram_client
    headers = {"X-Telegram-Bot-Api-Secret-Token": SECRET}
    contact = {"phone_number": "+79991234567", "user_id": 123, "first_name": "Private"}

    response = asyncio.run(
        _request(app, "POST", "/telegram/webhook", json=_update(text=None, contact=contact), headers=headers)
    )

    assert response.status_code == 200
    assert backend.calls[0][1]["json"] == {
        "user_id": 123,
        "username": "alice_user",
        "contact": {"phone_number": "+79991234567", "user_id": 123},
    }
    assert telegram_client.calls[0][1]["json"] == {
        "chat_id": 123,
        "text": ACCESS_GRANTED_TEXT,
        "reply_markup": {"remove_keyboard": True},
    }

    backend.access_status = 403
    telegram_client.calls.clear()
    response = asyncio.run(
        _request(app, "POST", "/telegram/webhook", json=_update(text=None, contact=contact), headers=headers)
    )
    assert response.status_code == 200
    assert telegram_client.calls[0][1]["json"] == {
        "chat_id": 123,
        "text": CONTACT_DENIED_TEXT,
        "reply_markup": {"remove_keyboard": True},
    }


def test_accepted_webhook_submits_job_before_acknowledging() -> None:
    app = create_app(_config())
    backend = FakeClient(202)
    telegram_client = FakeClient()
    app.state.backend_client = backend
    app.state.telegram_client = telegram_client
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/telegram/webhook",
            json=_update(update_id=99),
            headers={"X-Telegram-Bot-Api-Secret-Token": SECRET},
        )
    )
    assert response.status_code == 200
    assert backend.calls[0][1]["json"] == {"user_id": 123, "username": "alice_user"}
    assert backend.calls[1][1]["json"] == {"update_id": 99, "user_id": 123, "question": "Вопрос?"}
    assert backend.calls[1][1]["headers"] == {"Authorization": f"Bearer {TOKEN}"}
    assert telegram_client.calls[0][1]["json"] == {"chat_id": 123, "text": ACCEPTED_TEXT}

    backend.backend_status = 200
    telegram_client.calls.clear()
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/telegram/webhook",
            json=_update(update_id=99),
            headers={"X-Telegram-Bot-Api-Secret-Token": SECRET},
        )
    )
    assert response.status_code == 200
    assert telegram_client.calls == []


def test_webhook_handles_commands_before_rag_submission() -> None:
    app = create_app(_config())
    backend = FakeClient(202)
    telegram_client = FakeClient()
    app.state.backend_client = backend
    app.state.telegram_client = telegram_client
    headers = {"X-Telegram-Bot-Api-Secret-Token": SECRET}

    for update_id, text in [
        (1, "/start@imperial_bot payload"),
        (2, "/HELP details"),
        (3, "/unknown argument"),
        (4, "/invalid-command"),
    ]:
        response = asyncio.run(
            _request(app, "POST", "/telegram/webhook", json=_update(update_id=update_id, text=text), headers=headers)
        )
        assert response.status_code == 200

    assert all(not url.endswith("/internal/telegram/jobs") for url, _ in backend.calls)
    assert [call[1]["json"]["text"] for call in telegram_client.calls] == [
        WELCOME_TEXT,
        HELP_TEXT,
        UNKNOWN_COMMAND_TEXT,
        UNKNOWN_COMMAND_TEXT,
    ]

    telegram_client.calls.clear()
    response = asyncio.run(
        _request(app, "POST", "/telegram/webhook", json=_update(update_id=5, text="/new@imperial_bot now"), headers=headers)
    )
    assert response.status_code == 200
    job_calls = [call for call in backend.calls if call[0].endswith("/internal/telegram/jobs")]
    assert job_calls[0][1]["json"] == {"update_id": 5, "user_id": 123, "question": NEW_COMMAND}
    assert telegram_client.calls == []


def test_lifespan_sets_exact_webhook_and_health(monkeypatch) -> None:
    fake = FakeClient(204)

    class ClientContext:
        async def __aenter__(self):
            return fake

        async def __aexit__(self, *args):
            return None

    monkeypatch.setattr(telegram.httpx, "AsyncClient", lambda **kwargs: ClientContext())
    app = create_app(_config())

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            assert app.state.ready is True

    asyncio.run(exercise())
    assert len(fake.calls) == 2  # No backend polling during lifespan.
    commands_url, commands_call = fake.calls[0]
    assert commands_url.endswith("/setMyCommands")
    assert commands_call["json"] == {
        "commands": TELEGRAM_COMMANDS,
        "scope": {"type": "all_private_chats"},
    }
    webhook_url, webhook_call = fake.calls[1]
    assert webhook_url.endswith("/setWebhook")
    assert webhook_call["json"] == {
        "url": "https://render.example/telegram/webhook",
        "allowed_updates": ["message"],
        "drop_pending_updates": False,
        "secret_token": SECRET,
    }


@pytest.mark.parametrize("payload,status", [
    ({"update_id": True, "user_id": 123, "messages": ["answer"]}, 400),
    ({"update_id": 1.5, "user_id": 123, "messages": ["answer"]}, 400),
    ({"update_id": 1, "user_id": "123", "messages": ["answer"]}, 400),
    ({"update_id": 1, "user_id": 0, "messages": ["answer"]}, 400),
    ({"update_id": 1, "user_id": 123, "messages": []}, 400),
    ({"update_id": 1, "user_id": 123, "messages": ["valid", " "]}, 400),
    ({"update_id": 1, "user_id": 123, "messages": ["valid", 12]}, 400),
    ({"update_id": 1, "user_id": 123, "messages": ["valid", "x" * 4097]}, 400),
    ({"update_id": 1, "user_id": 123, "messages": ["x" * 4096] * 260}, 413),
])
def test_delivery_validates_whole_batch_before_sending(payload, status) -> None:
    app = create_app(_config())
    app.state.telegram_client = FakeClient()
    response = asyncio.run(_request(app, "POST", "/internal/telegram/deliver", json=payload,
                                    headers={"Authorization": f"Bearer {TOKEN}"}))
    assert response.status_code == status
    assert app.state.telegram_client.calls == []


def test_delivery_auth_body_limits_and_multichunk_success() -> None:
    app = create_app(_config())
    app.state.telegram_client = FakeClient()
    payload = {"update_id": 7, "user_id": 123, "messages": ["я" * 4096] * 5}
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        assert asyncio.run(_request(app, "POST", "/internal/telegram/deliver", json=payload, headers=headers)).status_code == 401
    assert app.state.telegram_client.calls == []
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = asyncio.run(_request(app, "POST", "/internal/telegram/deliver", json=payload, headers=headers))
    assert response.status_code == 200 and response.json() == {"update_id": 7, "status": "delivered"}
    assert [call[1]["json"]["text"] for call in app.state.telegram_client.calls] == payload["messages"]
    assert asyncio.run(_request(app, "POST", "/internal/telegram/deliver", content=b"bad", headers={**headers, "content-type": "application/json"})).status_code == 400
    assert asyncio.run(_request(app, "POST", "/internal/telegram/deliver", content=b"{}", headers={**headers, "content-type": "text/plain"})).status_code == 415
    # The larger delivery limit must not change the webhook limit.
    assert asyncio.run(_request(app, "POST", "/telegram/webhook", json=payload,
                               headers={"X-Telegram-Bot-Api-Secret-Token": SECRET})).status_code == 413


@pytest.mark.parametrize("failure,retryable,retry_after", [
    (httpx.Response(429, json={"ok": False, "parameters": {"retry_after": 420}, "description": "private"}), True, 420),
    (httpx.Response(403, json={"ok": False, "description": "private"}), False, None),
    (httpx.Response(500, text="private"), True, None),
    (httpx.Response(200, json={"ok": False}), True, None),
    (httpx.Response(200, text="private"), True, None),
    (httpx.ReadTimeout("private"), True, None),
    (TimeoutError("private"), True, None),
])
def test_partial_delivery_failure_never_acknowledges_success(failure, retryable, retry_after) -> None:
    app = create_app(_config())
    calls = []

    class Client:
        async def post(self, url, **kwargs):
            calls.append(kwargs["json"])
            if len(calls) == 2:
                if isinstance(failure, Exception):
                    raise failure
                return failure
            return httpx.Response(200, json={"ok": True})

    app.state.telegram_client = Client()
    response = asyncio.run(_request(app, "POST", "/internal/telegram/deliver",
                                    json={"update_id": 1, "user_id": 123, "messages": ["first", "second", "third"]},
                                    headers={"Authorization": f"Bearer {TOKEN}"}))
    assert response.status_code == 502
    assert response.json() == {"error": "telegram_delivery_failed", "retryable": retryable, "retry_after": retry_after}
    assert [call["text"] for call in calls] == ["first", "second"]
    assert "private" not in response.text


def test_streaming_delivery_body_limit_without_content_length() -> None:
    app = create_app(_config())
    app.state.telegram_client = FakeClient()

    async def body():
        yield b" " * telegram.MAX_DELIVERY_BODY_BYTES
        yield b"{}"

    response = asyncio.run(_request(app, "POST", "/internal/telegram/deliver", content=body(),
                                    headers={"Authorization": f"Bearer {TOKEN}", "content-type": "application/json"}))
    assert response.status_code == 413
    assert app.state.telegram_client.calls == []


def test_telegram_call_deadline_cancels_hung_send(monkeypatch) -> None:
    real_timeout = asyncio.timeout
    deadlines = []

    def short_timeout(seconds):
        deadlines.append(seconds)
        return real_timeout(0.01 if seconds == 15 else seconds)

    monkeypatch.setattr(telegram.asyncio, "timeout", short_timeout)
    app = create_app(_config())
    cancelled = []

    class Client:
        async def post(self, url, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)

    app.state.telegram_client = Client()
    response = asyncio.run(_request(app, "POST", "/internal/telegram/deliver",
                                    json={"update_id": 1, "user_id": 123, "messages": ["answer", "sources"]},
                                    headers={"Authorization": f"Bearer {TOKEN}"}))
    assert response.status_code == 502 and response.json()["retryable"] is True
    assert deadlines == [40, 15] and cancelled == [True]
