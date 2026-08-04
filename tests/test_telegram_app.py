from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from imperial_rag.app import telegram
from imperial_rag.app.telegram import (
    ACCEPTED_TEXT,
    TelegramWebhookConfig,
    create_app,
    load_configuration,
    parse_allowed_user_ids,
    split_telegram_text,
)

SECRET = "s" * 32
TOKEN = "t" * 32


class FakeClient:
    def __init__(self, backend_status: int = 202) -> None:
        self.backend_status = backend_status
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self.calls.append((url, kwargs))
        if "api.telegram.org" in url:
            return httpx.Response(200, json={"ok": True, "result": True})
        return httpx.Response(self.backend_status, json={"status": "pending"})


def _config() -> TelegramWebhookConfig:
    return TelegramWebhookConfig(
        bot_token="bot-token",
        allowed_user_ids=frozenset({123}),
        webhook_url="https://render.example/telegram/webhook",
        webhook_secret=SECRET,
        backend_url="https://backend.example",
        service_token=TOKEN,
    )


def _update(*, update_id: int = 1, user_id: int = 123, chat_type: str = "private", text: str = "Вопрос?") -> dict:
    return {
        "update_id": update_id,
        "message": {
            "from": {"id": user_id},
            "chat": {"id": user_id, "type": chat_type},
            "text": text,
        },
    }


async def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://render.example") as client:
        return await client.request(method, path, **kwargs)


def test_configuration_requires_https_strong_secrets_and_numeric_allowlist() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        load_configuration({})
    with pytest.raises(ValueError, match="comma-separated"):
        parse_allowed_user_ids("12,nope")
    with pytest.raises(ValueError, match="positive"):
        parse_allowed_user_ids("0")

    values = {
        "TELEGRAM_BOT_TOKEN": " token ",
        "IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS": "123, 456,123",
        "TELEGRAM_WEBHOOK_URL": "https://render.example/telegram/webhook",
        "TELEGRAM_WEBHOOK_SECRET": SECRET,
        "IMPERIAL_RAG_TELEGRAM_BACKEND_URL": "https://backend.example/",
        "IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN": TOKEN,
    }
    config = load_configuration(values)
    assert config.allowed_user_ids == frozenset({123, 456})
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


def test_webhook_validates_secret_payload_allowlist_and_groups() -> None:
    app = create_app(_config())
    backend = FakeClient()
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
    assert backend.calls == []


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
    assert backend.calls[0][1]["json"] == {"update_id": 99, "user_id": 123, "question": "Вопрос?"}
    assert backend.calls[0][1]["headers"] == {"Authorization": f"Bearer {TOKEN}"}
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
    url, call = fake.calls[0]
    assert url.endswith("/setWebhook")
    assert call["json"] == {
        "url": "https://render.example/telegram/webhook",
        "allowed_updates": ["message"],
        "drop_pending_updates": False,
        "secret_token": SECRET,
    }
