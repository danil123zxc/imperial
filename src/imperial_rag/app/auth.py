from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
from pathlib import Path
import secrets
import sqlite3
import time


APPROVED = "approved"
PENDING = "pending"
REJECTED = "rejected"
PBKDF2_ITERATIONS = 390_000


class AuthenticationStatus(str, Enum):
    AUTHENTICATED = "authenticated"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"
    INVALID_PASSWORD = "invalid_password"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class UserRecord:
    email: str
    status: str
    is_admin: bool
    full_name: str
    reason: str
    created_at: int
    approved_at: int | None = None
    approved_by: str | None = None


@dataclass(frozen=True)
class AuthenticationResult:
    status: AuthenticationStatus
    user: UserRecord | None


class AuthStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    full_name TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    approved_at INTEGER,
                    approved_by TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS users_status_idx ON users(status)")

    def bootstrap_admin(self, email: str, password: str) -> UserRecord:
        normalized_email = _normalize_email(email)
        _validate_password(password)
        self.initialize()
        existing = self.get_user(normalized_email)
        now = time.time_ns()
        if existing is not None:
            salt, digest = _hash_password(password)
            with self._connection() as conn:
                conn.execute(
                    """
                    UPDATE users
                    SET password_salt = ?, password_hash = ?,
                        status = ?, is_admin = 1, updated_at = ?,
                        approved_at = COALESCE(approved_at, ?),
                        approved_by = COALESCE(approved_by, ?)
                    WHERE email = ?
                    """,
                    (salt, digest, APPROVED, now, now, normalized_email, normalized_email),
                )
            return self.get_user(normalized_email) or existing

        salt, digest = _hash_password(password)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO users(
                    email, password_salt, password_hash, status, is_admin,
                    full_name, reason, created_at, updated_at, approved_at, approved_by
                )
                VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized_email,
                    salt,
                    digest,
                    APPROVED,
                    "Administrator",
                    "Bootstrap admin",
                    now,
                    now,
                    now,
                    normalized_email,
                ),
            )
        user = self.get_user(normalized_email)
        if user is None:
            raise RuntimeError("failed to create bootstrap admin")
        return user

    def register_user(self, email: str, password: str, full_name: str = "", reason: str = "") -> UserRecord:
        normalized_email = _normalize_email(email)
        _validate_password(password)
        self.initialize()
        existing = self.get_user(normalized_email)
        if existing is not None and existing.status != REJECTED:
            return existing

        salt, digest = _hash_password(password)
        now = time.time_ns()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO users(
                    email, password_salt, password_hash, status, is_admin,
                    full_name, reason, created_at, updated_at, approved_at, approved_by
                )
                VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, NULL, NULL)
                ON CONFLICT(email) DO UPDATE SET
                    password_salt = excluded.password_salt,
                    password_hash = excluded.password_hash,
                    status = excluded.status,
                    is_admin = 0,
                    full_name = excluded.full_name,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at,
                    approved_at = NULL,
                    approved_by = NULL
                """,
                (normalized_email, salt, digest, PENDING, full_name.strip(), reason.strip(), now, now),
            )
        user = self.get_user(normalized_email)
        if user is None:
            raise RuntimeError("failed to register user")
        return user

    def authenticate(self, email: str, password: str) -> AuthenticationResult:
        normalized_email = _normalize_email(email)
        self.initialize()
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
        if row is None:
            return AuthenticationResult(AuthenticationStatus.NOT_FOUND, None)
        if not _verify_password(password, row["password_salt"], row["password_hash"]):
            return AuthenticationResult(AuthenticationStatus.INVALID_PASSWORD, None)

        user = _row_to_user(row)
        if user.status == APPROVED:
            return AuthenticationResult(AuthenticationStatus.AUTHENTICATED, user)
        if user.status == REJECTED:
            return AuthenticationResult(AuthenticationStatus.REJECTED, None)
        return AuthenticationResult(AuthenticationStatus.PENDING_APPROVAL, None)

    def approve_user(self, admin_email: str, target_email: str) -> UserRecord:
        normalized_admin = _normalize_email(admin_email)
        normalized_target = _normalize_email(target_email)
        self.initialize()
        admin = self.get_user(normalized_admin)
        if admin is None or not admin.is_admin or admin.status != APPROVED:
            raise PermissionError("only an approved admin can grant access")
        if self.get_user(normalized_target) is None:
            raise KeyError(normalized_target)

        now = time.time_ns()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE users
                SET status = ?, updated_at = ?, approved_at = ?, approved_by = ?
                WHERE email = ?
                """,
                (APPROVED, now, now, normalized_admin, normalized_target),
            )
        user = self.get_user(normalized_target)
        if user is None:
            raise RuntimeError("failed to approve user")
        return user

    def get_user(self, email: str) -> UserRecord | None:
        normalized_email = _normalize_email(email)
        self.initialize()
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (normalized_email,)).fetchone()
        return _row_to_user(row) if row is not None else None

    def list_pending_users(self) -> list[UserRecord]:
        self.initialize()
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM users WHERE status = ? ORDER BY created_at ASC, email ASC",
                (PENDING,),
            ).fetchall()
        return [_row_to_user(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def _normalize_email(email: str) -> str:
    normalized = str(email or "").strip().casefold()
    if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        raise ValueError("valid email is required")
    return normalized


def _validate_password(password: str) -> None:
    if len(password or "") < 8:
        raise ValueError("password must be at least 8 characters")


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    password_salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), password_salt, PBKDF2_ITERATIONS)
    return password_salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, digest_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(actual, expected)


def _row_to_user(row: sqlite3.Row) -> UserRecord:
    return UserRecord(
        email=str(row["email"]),
        status=str(row["status"]),
        is_admin=bool(row["is_admin"]),
        full_name=str(row["full_name"] or ""),
        reason=str(row["reason"] or ""),
        created_at=int(row["created_at"]),
        approved_at=int(row["approved_at"]) if row["approved_at"] is not None else None,
        approved_by=str(row["approved_by"]) if row["approved_by"] is not None else None,
    )
