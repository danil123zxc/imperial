from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
from pathlib import Path
import re
import secrets
import sqlite3
import time

from imperial_rag.app.users import normalize_user_email


APPROVED = "approved"
PENDING = "pending"
REJECTED = "rejected"
PBKDF2_ITERATIONS = 390_000
NANOSECONDS_PER_SECOND = 1_000_000_000
SESSION_TOKEN_BYTES = 32
SESSION_TOKEN_MAX_LENGTH = 512
TELEGRAM_PHONE_HASH_SECRET_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,256}")
TELEGRAM_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9_]{5,32}")
TELEGRAM_PHONE_PATTERN = re.compile(r"\+?[0-9\s().-]+")


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


@dataclass(frozen=True)
class TelegramAccessGrant:
    id: int
    identity_type: str
    display_label: str
    telegram_user_id: int | None
    created_by: str
    created_at: int
    bound_at: int | None


@dataclass(frozen=True)
class TelegramAuthorizationResult:
    authorized: bool
    newly_bound: bool


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY(user_email) REFERENCES users(email) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS auth_sessions_user_idx ON auth_sessions(user_email)")
            conn.execute("CREATE INDEX IF NOT EXISTS auth_sessions_expiry_idx ON auth_sessions(expires_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telegram_access_grants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_type TEXT NOT NULL CHECK(identity_type IN ('username', 'phone')),
                    identity_key TEXT NOT NULL,
                    display_label TEXT NOT NULL,
                    telegram_user_id INTEGER,
                    created_by TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    bound_at INTEGER,
                    UNIQUE(identity_type, identity_key)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS telegram_access_grants_user_idx
                ON telegram_access_grants(telegram_user_id)
                """
            )

    def bootstrap_admin(self, email: str, password: str) -> UserRecord:
        normalized_email = normalize_user_email(email)
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
        normalized_email = normalize_user_email(email)
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
        normalized_email = normalize_user_email(email)
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
        normalized_admin = normalize_user_email(admin_email)
        normalized_target = normalize_user_email(target_email)
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
        normalized_email = normalize_user_email(email)
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

    def add_telegram_access_grant(
        self,
        admin_email: str,
        identity: str,
        phone_hash_secret: str,
    ) -> TelegramAccessGrant:
        normalized_admin = normalize_user_email(admin_email)
        self._require_approved_admin(normalized_admin)
        identity_type, identity_key, display_label = _telegram_identity(identity, phone_hash_secret)
        now = time.time_ns()
        self.initialize()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO telegram_access_grants(
                    identity_type, identity_key, display_label, telegram_user_id,
                    created_by, created_at, bound_at
                )
                VALUES (?, ?, ?, NULL, ?, ?, NULL)
                """,
                (identity_type, identity_key, display_label, normalized_admin, now),
            )
            row = conn.execute(
                """
                SELECT * FROM telegram_access_grants
                WHERE identity_type = ? AND identity_key = ?
                """,
                (identity_type, identity_key),
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to create Telegram access grant")
        return _row_to_telegram_access_grant(row)

    def list_telegram_access_grants(self) -> list[TelegramAccessGrant]:
        self.initialize()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM telegram_access_grants
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [_row_to_telegram_access_grant(row) for row in rows]

    def revoke_telegram_access_grant(self, admin_email: str, grant_id: int) -> bool:
        normalized_admin = normalize_user_email(admin_email)
        self._require_approved_admin(normalized_admin)
        if isinstance(grant_id, bool) or not isinstance(grant_id, int) or grant_id <= 0:
            raise ValueError("Telegram access grant ID must be a positive integer")
        self.initialize()
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM telegram_access_grants WHERE id = ?", (grant_id,))
        return cursor.rowcount == 1

    def authorize_telegram_user(
        self,
        user_id: int,
        *,
        username: str | None = None,
        phone_number: str | None = None,
        contact_user_id: int | None = None,
        phone_hash_secret: str,
    ) -> TelegramAuthorizationResult:
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Telegram user ID must be a positive integer")
        self.initialize()
        with self._connection() as conn:
            candidates: list[tuple[str, str]] = []
            if username:
                try:
                    candidates.append(("username", _normalize_telegram_username(username, require_prefix=False)))
                except ValueError:
                    pass
            if phone_number and contact_user_id == user_id:
                try:
                    digits = _normalize_telegram_phone(phone_number, require_prefix=False)
                    candidates.append(("phone", _telegram_phone_digest(digits, phone_hash_secret)))
                except ValueError:
                    pass

            now = time.time_ns()
            newly_bound = False
            for identity_type, identity_key in candidates:
                cursor = conn.execute(
                    """
                    UPDATE telegram_access_grants
                    SET telegram_user_id = ?, bound_at = ?
                    WHERE identity_type = ? AND identity_key = ? AND telegram_user_id IS NULL
                    """,
                    (user_id, now, identity_type, identity_key),
                )
                if cursor.rowcount == 1:
                    newly_bound = True
            if newly_bound:
                return TelegramAuthorizationResult(True, True)
            if conn.execute(
                "SELECT 1 FROM telegram_access_grants WHERE telegram_user_id = ? LIMIT 1",
                (user_id,),
            ).fetchone() is not None:
                return TelegramAuthorizationResult(True, False)
        return TelegramAuthorizationResult(False, False)

    def create_session(self, email: str, ttl_seconds: int) -> str:
        if ttl_seconds <= 0:
            raise ValueError("session ttl must be positive")
        normalized_email = normalize_user_email(email)
        self.initialize()
        user = self.get_user(normalized_email)
        if user is None or user.status != APPROVED:
            raise PermissionError("only an approved user can create a session")

        token = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
        token_hash = _hash_session_token(token)
        now = time.time_ns()
        expires_at = now + ttl_seconds * NANOSECONDS_PER_SECOND
        with self._connection() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
            conn.execute(
                """
                INSERT INTO auth_sessions(token_hash, user_email, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token_hash, normalized_email, now, expires_at),
            )
        return token

    def authenticate_session(self, token: str) -> UserRecord | None:
        if not _valid_session_token(token):
            return None
        self.initialize()
        token_hash = _hash_session_token(token)
        now = time.time_ns()
        with self._connection() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE expires_at <= ?", (now,))
            row = conn.execute(
                """
                SELECT users.*
                FROM auth_sessions
                JOIN users ON users.email = auth_sessions.user_email
                WHERE auth_sessions.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if str(row["status"]) != APPROVED:
                conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
                return None
        return _row_to_user(row)

    def revoke_session(self, token: str) -> None:
        if not _valid_session_token(token):
            return
        self.initialize()
        with self._connection() as conn:
            conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_hash_session_token(token),))

    def _require_approved_admin(self, email: str) -> UserRecord:
        admin = self.get_user(email)
        if admin is None or not admin.is_admin or admin.status != APPROVED:
            raise PermissionError("only an approved admin can manage Telegram access")
        return admin

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()


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


def _valid_session_token(token: str) -> bool:
    return isinstance(token, str) and bool(token) and len(token) <= SESSION_TOKEN_MAX_LENGTH


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _telegram_identity(identity: str, phone_hash_secret: str) -> tuple[str, str, str]:
    value = str(identity or "").strip()
    if value.startswith("@"):
        username = _normalize_telegram_username(value, require_prefix=True)
        return "username", username, f"@{username}"
    if value.startswith("+"):
        digits = _normalize_telegram_phone(value, require_prefix=True)
        return "phone", _telegram_phone_digest(digits, phone_hash_secret), f"+••••{digits[-4:]}"
    raise ValueError("Telegram access must be an @username or +phone number")


def _normalize_telegram_username(value: str, *, require_prefix: bool) -> str:
    raw = str(value or "").strip()
    if require_prefix and not raw.startswith("@"):
        raise ValueError("Telegram username must start with @")
    username = raw.removeprefix("@").casefold()
    if TELEGRAM_USERNAME_PATTERN.fullmatch(username) is None:
        raise ValueError("Telegram username must contain 5-32 letters, digits, or underscores")
    return username


def _normalize_telegram_phone(value: str, *, require_prefix: bool) -> str:
    raw = str(value or "").strip()
    if require_prefix and not raw.startswith("+"):
        raise ValueError("Telegram phone number must start with +")
    if TELEGRAM_PHONE_PATTERN.fullmatch(raw) is None:
        raise ValueError("Telegram phone number must use international format")
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) < 8 or len(digits) > 15 or digits.startswith("0"):
        raise ValueError("Telegram phone number must contain 8-15 international digits")
    return digits


def _telegram_phone_digest(digits: str, secret: str) -> str:
    if TELEGRAM_PHONE_HASH_SECRET_PATTERN.fullmatch(str(secret or "")) is None:
        raise ValueError("IMPERIAL_RAG_TELEGRAM_PHONE_HASH_SECRET must contain 32-256 URL-safe characters")
    return hmac.new(secret.encode("utf-8"), digits.encode("ascii"), hashlib.sha256).hexdigest()


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


def _row_to_telegram_access_grant(row: sqlite3.Row) -> TelegramAccessGrant:
    return TelegramAccessGrant(
        id=int(row["id"]),
        identity_type=str(row["identity_type"]),
        display_label=str(row["display_label"]),
        telegram_user_id=int(row["telegram_user_id"]) if row["telegram_user_id"] is not None else None,
        created_by=str(row["created_by"]),
        created_at=int(row["created_at"]),
        bound_at=int(row["bound_at"]) if row["bound_at"] is not None else None,
    )
