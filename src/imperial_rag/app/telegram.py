from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
import json
import os
import re
from secrets import compare_digest
from typing import Any, AsyncIterator
from urllib.parse import urlsplit

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

MAX_HTTP_BODY_BYTES = 16 * 1024
MAX_TELEGRAM_TEXT_LENGTH = 4096
ACCEPTED_TEXT = "Вопрос принят. Готовлю ответ."
WELCOME_TEXT = "Задайте вопрос по проиндексированным документам.\n\nИспользуйте /help, чтобы посмотреть доступные команды."
HELP_TEXT = "Команды:\n/start — начать работу\n/help — показать помощь\n/new — начать новый диалог"
UNKNOWN_COMMAND_TEXT = f"Неизвестная команда.\n\n{HELP_TEXT}"
NEW_COMMAND = "/new"
NEW_CONVERSATION_TITLE = "Новый диалог"
NEW_CONVERSATION_TEXT = "Новый диалог начат. Задайте вопрос."
TELEGRAM_COMMANDS = [
    {"command": "start", "description": "Начать работу"},
    {"command": "help", "description": "Показать помощь"},
    {"command": "new", "description": "Начать новый диалог"},
]
COMMAND_PATTERN = re.compile(r"^/([a-z0-9_]+)(?:@[a-z0-9_]+)?(?:\s|$)", re.IGNORECASE)


@dataclass(frozen=True)
class TelegramWebhookConfig:
    bot_token: str
    allowed_user_ids: frozenset[int]
    webhook_url: str
    webhook_secret: str
    backend_url: str
    service_token: str


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


def load_configuration(environ: Mapping[str, str] | None = None) -> TelegramWebhookConfig:
    if environ is None:
        from imperial_rag.env import load_project_env

        load_project_env()
    values = os.environ if environ is None else environ
    config = TelegramWebhookConfig(
        bot_token=_required(values, "TELEGRAM_BOT_TOKEN"),
        allowed_user_ids=parse_allowed_user_ids(values.get("IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS")),
        webhook_url=_required(values, "TELEGRAM_WEBHOOK_URL").rstrip("/"),
        webhook_secret=_validated_secret(values, "TELEGRAM_WEBHOOK_SECRET"),
        backend_url=_required(values, "IMPERIAL_RAG_TELEGRAM_BACKEND_URL").rstrip("/"),
        service_token=_validated_secret(values, "IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN"),
    )
    webhook_url = _https_url(config.webhook_url, "TELEGRAM_WEBHOOK_URL")
    backend_url = _https_url(config.backend_url, "IMPERIAL_RAG_TELEGRAM_BACKEND_URL")
    if webhook_url.path != "/telegram/webhook":
        raise ValueError("TELEGRAM_WEBHOOK_URL must end with /telegram/webhook")
    if backend_url.path not in {"", "/"}:
        raise ValueError("IMPERIAL_RAG_TELEGRAM_BACKEND_URL must be an HTTPS origin without a path")
    return config


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


def create_app(config: TelegramWebhookConfig | None = None) -> Starlette:
    application = Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/telegram/webhook", telegram_webhook, methods=["POST"]),
        ],
        lifespan=_lifespan,
    )
    application.state.config = config
    application.state.ready = False
    application.state.backend_client = None
    application.state.telegram_client = None
    return application


async def healthz(request: Request) -> Response:
    return PlainTextResponse("ok\n" if request.app.state.ready else "not ready\n", status_code=200 if request.app.state.ready else 503)


async def telegram_webhook(request: Request) -> Response:
    config = _config(request.app)
    supplied_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not compare_digest(supplied_secret, config.webhook_secret):
        return PlainTextResponse("forbidden\n", status_code=403)
    try:
        payload = await _json_body(request)
        job = _job_from_update(payload, config.allowed_user_ids)
    except PayloadError as exc:
        return PlainTextResponse(f"{exc}\n", status_code=exc.status_code)
    if job is None:
        return PlainTextResponse("ignored\n")

    command = _command_from_text(job["question"])
    if command is not None:
        if command == "start":
            reply = WELCOME_TEXT
        elif command == "help":
            reply = HELP_TEXT
        elif command == "new":
            job["question"] = NEW_COMMAND
            reply = None
        else:
            reply = UNKNOWN_COMMAND_TEXT
        if reply is not None:
            try:
                await _telegram_request(
                    request.app.state.telegram_client,
                    config,
                    "sendMessage",
                    {"chat_id": job["user_id"], "text": reply},
                )
            except RuntimeError:
                return PlainTextResponse("telegram unavailable\n", status_code=503)
            return PlainTextResponse("ok\n")

    client = request.app.state.backend_client
    try:
        response = await client.post(
            f"{config.backend_url}/internal/telegram/jobs",
            headers={"Authorization": f"Bearer {config.service_token}"},
            json=job,
        )
    except httpx.HTTPError:
        return PlainTextResponse("backend unavailable\n", status_code=503)
    if response.status_code not in {200, 202}:
        return PlainTextResponse("backend rejected job\n", status_code=503)
    if response.status_code == 202 and command != "new":
        try:
            await _telegram_request(request.app.state.telegram_client, config, "sendMessage", {"chat_id": job["user_id"], "text": ACCEPTED_TEXT})
        except RuntimeError:
            return PlainTextResponse("telegram unavailable\n", status_code=503)
    return PlainTextResponse("ok\n")


async def deliver_once(application: Starlette) -> bool:
    config = _config(application)
    response = await application.state.backend_client.post(
        f"{config.backend_url}/internal/telegram/deliveries/claim",
        headers={"Authorization": f"Bearer {config.service_token}"},
        json={},
    )
    if response.status_code == 204:
        return False
    if response.status_code != 200:
        raise RuntimeError("backend delivery claim failed")
    delivery = response.json()
    update_id = _positive_int(delivery.get("update_id"), "update_id")
    user_id = _positive_int(delivery.get("user_id"), "user_id")
    messages = delivery.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError("backend returned an invalid delivery")
    for message in messages:
        if not isinstance(message, str) or not message or len(message) > MAX_TELEGRAM_TEXT_LENGTH:
            raise RuntimeError("backend returned an invalid Telegram message")
        await _telegram_request(
            application.state.telegram_client,
            config,
            "sendMessage",
            {"chat_id": user_id, "text": message},
        )
    completion = await application.state.backend_client.post(
        f"{config.backend_url}/internal/telegram/deliveries/{update_id}/complete",
        headers={"Authorization": f"Bearer {config.service_token}"},
        json={},
    )
    if completion.status_code != 200:
        raise RuntimeError("backend delivery completion failed")
    return True


@asynccontextmanager
async def _lifespan(application: Starlette) -> AsyncIterator[None]:
    application.state.config = application.state.config or load_configuration()
    async with httpx.AsyncClient(timeout=15) as client:
        application.state.backend_client = client
        application.state.telegram_client = client
        await _telegram_request(
            client,
            _config(application),
            "setMyCommands",
            {"commands": TELEGRAM_COMMANDS, "scope": {"type": "all_private_chats"}},
        )
        await _telegram_request(
            client,
            _config(application),
            "setWebhook",
            {
                "url": _config(application).webhook_url,
                "allowed_updates": ["message"],
                "drop_pending_updates": False,
                "secret_token": _config(application).webhook_secret,
            },
        )
        application.state.ready = True
        task = asyncio.create_task(_delivery_loop(application))
        try:
            yield
        finally:
            application.state.ready = False
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


async def _delivery_loop(application: Starlette) -> None:
    from imperial_rag.observability import log_failure

    while True:
        try:
            await deliver_once(application)
        except Exception as exc:
            log_failure("telegram_delivery", exc, component="telegram-render")
        await asyncio.sleep(2)


async def _telegram_request(client: Any, config: TelegramWebhookConfig, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await client.post(f"https://api.telegram.org/bot{config.bot_token}/{method}", json=payload)
    except httpx.HTTPError as exc:
        raise RuntimeError("Telegram Bot API request failed") from exc
    if response.status_code != 200:
        raise RuntimeError("Telegram Bot API request failed")
    data = response.json()
    if not isinstance(data, dict) or data.get("ok") is not True:
        raise RuntimeError("Telegram Bot API request failed")
    return data


async def _json_body(request: Request) -> dict[str, Any]:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().casefold() != "application/json":
        raise PayloadError("content type must be application/json", 415)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            parsed_content_length = int(content_length)
        except ValueError as exc:
            raise PayloadError("invalid content length", 400) from exc
        if parsed_content_length > MAX_HTTP_BODY_BYTES:
            raise PayloadError("request body is too large", 413)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_HTTP_BODY_BYTES:
            raise PayloadError("request body is too large", 413)
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PayloadError("request body must be valid JSON", 400) from exc
    if not isinstance(payload, dict):
        raise PayloadError("request body must be a JSON object", 400)
    return payload


def _job_from_update(payload: dict[str, Any], allowed_user_ids: frozenset[int]) -> dict[str, Any] | None:
    update_id = _positive_int(payload.get("update_id"), "update_id")
    message = payload.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    user = message.get("from")
    if not isinstance(chat, dict) or chat.get("type") != "private" or not isinstance(user, dict):
        return None
    user_id = _positive_int(user.get("id"), "user_id")
    if user_id not in allowed_user_ids:
        return None
    question = message.get("text")
    if not isinstance(question, str) or not question.strip():
        return None
    question = question.strip()
    if len(question) > MAX_TELEGRAM_TEXT_LENGTH:
        raise PayloadError("question is too long", 400)
    return {"update_id": update_id, "user_id": user_id, "question": question}


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


def _command_from_text(text: str) -> str | None:
    if not text.startswith("/"):
        return None
    match = COMMAND_PATTERN.match(text)
    return match.group(1).casefold() if match else ""


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _validated_secret(values: Mapping[str, str], name: str) -> str:
    value = _required(values, name)
    if re.fullmatch(r"[A-Za-z0-9_-]{32,256}", value) is None:
        raise ValueError(f"{name} must contain 32-256 URL-safe characters")
    return value


def _https_url(value: str, name: str) -> Any:
    parsed = urlsplit(value)
    if parsed.scheme.casefold() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{name} must use HTTPS without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{name} must not contain a query or fragment")
    return parsed


def _config(application: Starlette) -> TelegramWebhookConfig:
    config = application.state.config
    if not isinstance(config, TelegramWebhookConfig):
        raise RuntimeError("Telegram webhook configuration is unavailable")
    return config


class PayloadError(ValueError):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


app = create_app()
