from __future__ import annotations

import re

import pytest

from imperial_rag.document_ids import content_fingerprint_id, metadata_or_content_id


def test_content_fingerprint_id_rejects_none_content() -> None:
    with pytest.raises(ValueError, match="Cannot fingerprint empty or None content"):
        content_fingerprint_id(None)


def test_content_fingerprint_id_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="Cannot fingerprint empty or None content"):
        content_fingerprint_id("")


def test_content_fingerprint_id_returns_content_hash_prefix() -> None:
    value = content_fingerprint_id("hello")

    assert re.fullmatch(r"content_sha256:[0-9a-f]{12}", value)
    assert value == "content_sha256:2cf24dba5fb0"


def test_metadata_or_content_id_rejects_empty_content_after_metadata_exhausted() -> None:
    with pytest.raises(ValueError, match="No usable ID"):
        metadata_or_content_id(None, content=None)


def test_metadata_or_content_id_uses_metadata_before_content_guard() -> None:
    assert metadata_or_content_id("existing-id", content=None) == "existing-id"
