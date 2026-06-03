from __future__ import annotations

import os
from pathlib import Path

import pytest

from imperial_rag.config import Settings
from imperial_rag.indexing import qdrant_health


def test_qdrant_health_returns_false_when_unreachable(monkeypatch, tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self, url):
            self.url = url

        def get_collections(self):
            raise RuntimeError("offline")

    monkeypatch.setattr("imperial_rag.indexing.QdrantClient", FakeClient)
    settings = Settings(workspace_root=tmp_path, qdrant_url="http://127.0.0.1:6333")

    assert qdrant_health(settings) is False


@pytest.mark.skipif(
    os.environ.get("IMPERIAL_RAG_LIVE_QDRANT") != "1",
    reason="live Qdrant test is opt-in",
)
def test_qdrant_health_check_reaches_local_qdrant() -> None:
    settings = Settings()

    assert settings.qdrant_url.startswith(("http://localhost", "http://127.0.0.1"))
    assert qdrant_health(settings) is True
