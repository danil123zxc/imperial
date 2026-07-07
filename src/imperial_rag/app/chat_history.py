from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import sqlite3
import time
import uuid

from imperial_rag.app.users import normalize_user_email

ASSISTANT_RESPONSE_CLAIM_TTL_NS = 30 * 60 * 1_000_000_000


@dataclass(frozen=True)
class ConversationRecord:
    id: str
    user_email: str
    title: str
    created_at: int
    updated_at: int
    phoenix_session_id: str


@dataclass(frozen=True)
class MessageRecord:
    id: int
    conversation_id: str
    role: str
    content: str
    payload: dict[str, Any]
    created_at: int
    sequence: int

    def to_chat_message(self) -> dict[str, Any]:
        message = dict(self.payload)
        message["role"] = self.role
        message["content"] = self.content
        return message


class ChatHistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_email TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    phoenix_session_id TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
                ON conversations(user_email, updated_at DESC, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS messages_conversation_sequence_idx
                ON messages(conversation_id, sequence ASC, id ASC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_response_claims (
                    user_message_id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    claimed_at INTEGER NOT NULL,
                    FOREIGN KEY(user_message_id) REFERENCES messages(id) ON DELETE CASCADE,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
        self._initialized = True

    def create_conversation(
        self,
        user_email: str,
        title: str = "New chat",
        phoenix_session_id: str | None = None,
    ) -> ConversationRecord:
        normalized_email = normalize_user_email(user_email)
        self.initialize()
        now = time.time_ns()
        conversation_id = str(uuid.uuid4())
        clean_title = _clean_title(title)
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations(id, user_email, title, created_at, updated_at, phoenix_session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    normalized_email,
                    clean_title,
                    now,
                    now,
                    phoenix_session_id or str(uuid.uuid4()),
                ),
            )
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ? AND user_email = ?",
                (conversation_id, normalized_email),
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to create chat conversation")
        return _row_to_conversation(row)

    def list_conversations(self, user_email: str) -> list[ConversationRecord]:
        normalized_email = normalize_user_email(user_email)
        self.initialize()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversations
                WHERE user_email = ?
                ORDER BY updated_at DESC, created_at DESC, title ASC
                """,
                (normalized_email,),
            ).fetchall()
        return [_row_to_conversation(row) for row in rows]

    def get_conversation(self, user_email: str, conversation_id: str) -> ConversationRecord | None:
        normalized_email = normalize_user_email(user_email)
        self.initialize()
        with self._connection() as conn:
            row = self._find_conversation(conn, normalized_email, conversation_id)
        return _row_to_conversation(row) if row is not None else None

    def add_message(
        self,
        user_email: str,
        conversation_id: str,
        role: str,
        content: str,
        payload: dict[str, Any] | None = None,
    ) -> MessageRecord:
        normalized_email = normalize_user_email(user_email)
        normalized_role = _normalize_role(role)
        self.initialize()
        now = time.time_ns()
        payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
        with self._connection() as conn:
            conversation = self._find_conversation(conn, normalized_email, conversation_id)
            if conversation is None:
                exists = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
                if exists is not None:
                    raise PermissionError("conversation does not belong to this user")
                raise KeyError(conversation_id)
            sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence), -1) + 1 FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            cursor = conn.execute(
                """
                INSERT INTO messages(conversation_id, role, content, payload_json, created_at, sequence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, normalized_role, str(content), payload_json, now, sequence),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ? AND user_email = ?",
                (now, conversation_id, normalized_email),
            )
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        if row is None:
            raise RuntimeError("failed to append chat message")
        return _row_to_message(row)

    def claim_assistant_response(
        self,
        user_email: str,
        conversation_id: str,
        user_message_id: int,
        *,
        stale_after_ns: int = ASSISTANT_RESPONSE_CLAIM_TTL_NS,
    ) -> bool:
        normalized_email = normalize_user_email(user_email)
        self.initialize()
        now = time.time_ns()
        with self._connection() as conn:
            conversation = self._find_conversation(conn, normalized_email, conversation_id)
            if conversation is None:
                exists = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
                if exists is not None:
                    raise PermissionError("conversation does not belong to this user")
                raise KeyError(conversation_id)
            user_message = conn.execute(
                """
                SELECT id, role, sequence FROM messages
                WHERE id = ? AND conversation_id = ?
                """,
                (user_message_id, conversation_id),
            ).fetchone()
            if user_message is None or str(user_message["role"]) != "user":
                return False
            if self._assistant_after_user_exists(conn, conversation_id, user_message):
                return False
            if stale_after_ns > 0:
                conn.execute(
                    """
                    DELETE FROM assistant_response_claims
                    WHERE user_message_id = ? AND claimed_at < ?
                    """,
                    (user_message_id, now - stale_after_ns),
                )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO assistant_response_claims(
                    user_message_id,
                    conversation_id,
                    claimed_at
                )
                VALUES (?, ?, ?)
                """,
                (user_message_id, conversation_id, now),
            )
        return cursor.rowcount == 1

    def list_messages(self, user_email: str, conversation_id: str) -> list[MessageRecord]:
        normalized_email = normalize_user_email(user_email)
        self.initialize()
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT messages.* FROM messages
                INNER JOIN conversations ON conversations.id = messages.conversation_id
                WHERE messages.conversation_id = ? AND conversations.user_email = ?
                ORDER BY messages.sequence ASC, messages.id ASC
                """,
                (conversation_id, normalized_email),
            ).fetchall()
        return [_row_to_message(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _find_conversation(
        self,
        conn: sqlite3.Connection,
        user_email: str,
        conversation_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM conversations WHERE id = ? AND user_email = ?",
            (conversation_id, user_email),
        ).fetchone()

    def _assistant_after_user_exists(
        self,
        conn: sqlite3.Connection,
        conversation_id: str,
        user_message: sqlite3.Row,
    ) -> bool:
        row = conn.execute(
            """
            SELECT 1 FROM messages
            WHERE conversation_id = ?
              AND role = 'assistant'
              AND (
                  sequence > ?
                  OR (sequence = ? AND id > ?)
              )
            LIMIT 1
            """,
            (
                conversation_id,
                int(user_message["sequence"]),
                int(user_message["sequence"]),
                int(user_message["id"]),
            ),
        ).fetchone()
        return row is not None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def _clean_title(title: str) -> str:
    normalized = " ".join(str(title or "").split())
    if not normalized:
        return "New chat"
    return normalized[:80]


def _normalize_role(role: str) -> str:
    normalized = str(role or "").strip().casefold()
    if normalized not in {"user", "assistant", "system"}:
        raise ValueError("message role must be user, assistant, or system")
    return normalized


def _row_to_conversation(row: sqlite3.Row) -> ConversationRecord:
    return ConversationRecord(
        id=str(row["id"]),
        user_email=str(row["user_email"]),
        title=str(row["title"] or "New chat"),
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        phoenix_session_id=str(row["phoenix_session_id"] or ""),
    )


def _row_to_message(row: sqlite3.Row) -> MessageRecord:
    return MessageRecord(
        id=int(row["id"]),
        conversation_id=str(row["conversation_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        payload=_parse_payload(row["payload_json"]),
        created_at=int(row["created_at"]),
        sequence=int(row["sequence"]),
    )


def _parse_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
