from __future__ import annotations


def normalize_user_email(email: str) -> str:
    normalized = str(email or "").strip().casefold()
    if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        raise ValueError("valid email is required")
    return normalized
