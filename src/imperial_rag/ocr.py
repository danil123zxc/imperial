from __future__ import annotations

import base64
import mimetypes
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OcrResult:
    text: str
    method: str
    cached: bool = False


class QwenOcrClient:
    def __init__(self, settings=None, conversation_client=None) -> None:
        from imperial_rag.providers import QwenProviderSettings

        self.settings = settings or QwenProviderSettings.from_env()
        self.api_key = self.settings.require_api_key()
        if conversation_client is None:
            import dashscope

            conversation_client = dashscope.MultiModalConversation
        self.conversation_client = conversation_client

    def extract_image_text(self, image_path: Path) -> OcrResult:
        from imperial_rag.providers import (
            DashScopeProviderError,
            _sanitize_provider_message,
            build_qwen_ocr_message,
            parse_qwen_ocr_response,
        )

        try:
            response = self.conversation_client.call(
                api_key=self.api_key,
                model=self.settings.vision_model,
                messages=[build_qwen_ocr_message(image_path, self.settings)],
                ocr_options={"task": self.settings.ocr_task},
            )
        except Exception as exc:
            message = _sanitize_provider_message(str(exc), self.api_key)
            raise DashScopeProviderError(
                f"DashScope OCR failed: exception={exc.__class__.__name__} message={message}"
            ) from None
        return OcrResult(
            text=parse_qwen_ocr_response(response, api_key=self.api_key),
            method=f"dashscope:{self.settings.vision_model}",
        )


class LegacyOpenAIOcrClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self._model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(model=self._model_name, temperature=0)
        return self._model

    def extract_image_text(self, image_path: Path) -> OcrResult:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "image/jpeg"
        response = self.model.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all visible Russian and English text verbatim. Do not summarize.",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                    ],
                }
            ]
        )
        return OcrResult(text=str(response.content).strip(), method="legacy_openai_vision")


OcrClient = QwenOcrClient


class OcrCache:
    def __init__(self, processed_root: Path) -> None:
        self.db_path = processed_root if processed_root.suffix == ".sqlite3" else processed_root / "ocr_cache.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_cache (
                file_hash TEXT NOT NULL,
                image_id TEXT NOT NULL,
                text TEXT NOT NULL,
                method TEXT NOT NULL,
                updated_ns INTEGER NOT NULL,
                PRIMARY KEY (file_hash, image_id)
            )
            """
        )

    def lookup(self, file_hash: str, image_id: str) -> OcrResult | None:
        row = self._conn.execute(
            "SELECT text, method FROM ocr_cache WHERE file_hash = ? AND image_id = ?",
            (file_hash, image_id),
        ).fetchone()
        if row is None:
            return None
        return OcrResult(text=row[0], method=row[1], cached=True)

    def store(self, file_hash: str, image_id: str, result: OcrResult) -> None:
        with self._conn:
            self._conn.execute(
                """
                REPLACE INTO ocr_cache(file_hash, image_id, text, method, updated_ns)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_hash, image_id, result.text, result.method, time.time_ns()),
            )

    def read(self, cache_key: str) -> OcrResult | None:
        return self.lookup("cache_key", cache_key)

    def write(self, cache_key: str, result: OcrResult) -> None:
        self.store("cache_key", cache_key, result)

    def close(self) -> None:
        self._conn.close()
