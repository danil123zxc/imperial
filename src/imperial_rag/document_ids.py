from __future__ import annotations

import hashlib
from typing import Any


def content_fingerprint_id(content: Any, *, length: int = 12) -> str:
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:length]
    return f"content_sha256:{digest}"


def first_nonempty_value(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        resolved = str(value).strip()
        if resolved:
            return resolved
    return None


def metadata_or_content_id(*values: Any, content: Any) -> str:
    return first_nonempty_value(*values) or content_fingerprint_id(content)
