from __future__ import annotations

import hashlib
import sqlite3

from imperial_rag.app.auth import AuthStore, AuthenticationStatus
from imperial_rag.app import auth as auth_module

PHONE_SECRET = "p" * 32


class TrackingConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value) -> None:
        if name in {"_connection", "closed"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._connection, name, value)

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._connection.__exit__(exc_type, exc, traceback)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)
        self._connection.close()


def test_register_authenticate_and_admin_approval_flow(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.initialize()
    admin = store.bootstrap_admin("admin@example.com", "admin-password")

    registration = store.register_user(
        email="User@Example.com",
        password="user-password",
        full_name="Test User",
        reason="Needs corpus access",
    )

    assert registration.status == "pending"
    assert registration.email == "user@example.com"
    assert registration.full_name == "Test User"
    assert [user.email for user in store.list_pending_users()] == ["user@example.com"]

    pending_login = store.authenticate("user@example.com", "user-password")
    assert pending_login.status == AuthenticationStatus.PENDING_APPROVAL
    assert pending_login.user is None

    approved = store.approve_user(admin.email, "user@example.com")
    assert approved.status == "approved"
    assert approved.approved_by == admin.email

    approved_login = store.authenticate("USER@example.com", "user-password")
    assert approved_login.status == AuthenticationStatus.AUTHENTICATED
    assert approved_login.user is not None
    assert approved_login.user.email == "user@example.com"


def test_only_approved_admins_can_grant_access(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.initialize()
    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("manager@example.com", "manager-password", "Manager", "Review docs")
    store.register_user("user@example.com", "user-password", "User", "Ask questions")
    store.approve_user("admin@example.com", "manager@example.com")

    try:
        store.approve_user("manager@example.com", "user@example.com")
    except PermissionError as exc:
        assert "admin" in str(exc)
    else:
        raise AssertionError("non-admin users must not be able to approve access")

    assert store.authenticate("user@example.com", "user-password").status == AuthenticationStatus.PENDING_APPROVAL


def test_bootstrap_admin_refreshes_password(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.initialize()
    store.bootstrap_admin("admin@example.com", "old-password")

    store.bootstrap_admin("admin@example.com", "new-password")

    assert store.authenticate("admin@example.com", "old-password").status == AuthenticationStatus.INVALID_PASSWORD
    assert store.authenticate("admin@example.com", "new-password").status == AuthenticationStatus.AUTHENTICATED


def test_auth_store_uses_shared_user_email_normalizer(monkeypatch, tmp_path):
    calls: list[str] = []

    def shared_normalizer(email: str) -> str:
        calls.append(email)
        return "shared@example.com"

    monkeypatch.setattr(auth_module, "normalize_user_email", shared_normalizer, raising=False)
    store = AuthStore(tmp_path / "auth.sqlite3")

    user = store.bootstrap_admin("Admin@Example.com", "admin-password")

    assert calls
    assert calls[0] == "Admin@Example.com"
    assert user.email == "shared@example.com"


def test_auth_store_closes_short_lived_connections(monkeypatch, tmp_path):
    real_connect = sqlite3.connect
    opened: list[TrackingConnection] = []

    def tracking_connect(*args, **kwargs):
        connection = TrackingConnection(real_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(auth_module.sqlite3, "connect", tracking_connect)
    store = AuthStore(tmp_path / "auth.sqlite3")

    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password", "User", "Need docs")
    store.authenticate("user@example.com", "user-password")

    assert opened
    assert all(connection.closed for connection in opened)


def test_session_tokens_are_hashed_and_restore_only_approved_users(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.bootstrap_admin("admin@example.com", "admin-password")

    token = store.create_session("admin@example.com", ttl_seconds=60)

    with sqlite3.connect(store.db_path) as conn:
        row = conn.execute("SELECT token_hash, user_email FROM auth_sessions").fetchone()

    assert row == (hashlib.sha256(token.encode("utf-8")).hexdigest(), "admin@example.com")
    assert token not in str(row)
    assert store.authenticate_session(token).email == "admin@example.com"
    assert store.authenticate_session(f"{token}tampered") is None


def test_session_tokens_expire_without_sliding(monkeypatch, tmp_path):
    now = 1_700_000_000_000_000_000
    monkeypatch.setattr(auth_module.time, "time_ns", lambda: now)
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.bootstrap_admin("admin@example.com", "admin-password")
    token = store.create_session("admin@example.com", ttl_seconds=30)

    monkeypatch.setattr(auth_module.time, "time_ns", lambda: now + 29_000_000_000)
    assert store.authenticate_session(token) is not None

    monkeypatch.setattr(auth_module.time, "time_ns", lambda: now + 30_000_000_000)
    assert store.authenticate_session(token) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_session_tokens_can_be_revoked_and_reject_invalid_values(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.bootstrap_admin("admin@example.com", "admin-password")
    token = store.create_session("admin@example.com", ttl_seconds=60)

    store.revoke_session(token)

    assert store.authenticate_session(token) is None
    assert store.authenticate_session("") is None
    assert store.authenticate_session("x" * 513) is None
    store.revoke_session("")


def test_session_creation_and_restore_require_approved_user(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("pending@example.com", "pending-password")

    try:
        store.create_session("pending@example.com", ttl_seconds=60)
    except PermissionError as exc:
        assert "approved" in str(exc)
    else:
        raise AssertionError("pending users must not receive sessions")

    token = store.create_session("admin@example.com", ttl_seconds=60)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE users SET status = 'rejected' WHERE email = 'admin@example.com'")

    assert store.authenticate_session(token) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_deleting_user_cascades_owned_sessions(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.bootstrap_admin("admin@example.com", "admin-password")
    token = store.create_session("admin@example.com", ttl_seconds=60)

    with store._connection() as conn:
        conn.execute("DELETE FROM users WHERE email = 'admin@example.com'")

    assert store.authenticate_session(token) is None
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0] == 0


def test_admin_manages_private_telegram_grants_and_bindings(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    admin = store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password")

    username = store.add_telegram_access_grant(admin.email, "@Alice_User", PHONE_SECRET)
    duplicate = store.add_telegram_access_grant(admin.email, "@alice_user", PHONE_SECRET)
    phone = store.add_telegram_access_grant(admin.email, "+7 (999) 123-45-67", PHONE_SECRET)

    assert duplicate.id == username.id
    assert username.display_label == "@alice_user"
    assert phone.display_label == "+••••4567"
    with sqlite3.connect(store.db_path) as conn:
        phone_row = conn.execute(
            "SELECT identity_key, display_label FROM telegram_access_grants WHERE id = ?",
            (phone.id,),
        ).fetchone()
    assert phone_row is not None
    assert phone_row[0] != "79991234567"
    assert "79991234567" not in str(phone_row)

    bound_username = store.authorize_telegram_user(101, username="ALICE_USER", phone_hash_secret=PHONE_SECRET)
    assert bound_username.authorized and bound_username.newly_bound
    assert store.authorize_telegram_user(101, username="renamed_user", phone_hash_secret=PHONE_SECRET).authorized
    assert not store.authorize_telegram_user(102, username="alice_user", phone_hash_secret=PHONE_SECRET).authorized

    forwarded = store.authorize_telegram_user(
        202,
        phone_number="+79991234567",
        contact_user_id=999,
        phone_hash_secret=PHONE_SECRET,
    )
    assert not forwarded.authorized
    bound_phone = store.authorize_telegram_user(
        202,
        phone_number="79991234567",
        contact_user_id=202,
        phone_hash_secret=PHONE_SECRET,
    )
    assert bound_phone.authorized and bound_phone.newly_bound

    assert store.revoke_telegram_access_grant(admin.email, username.id)
    assert not store.authorize_telegram_user(101, phone_hash_secret=PHONE_SECRET).authorized
    assert not store.revoke_telegram_access_grant(admin.email, username.id)


def test_only_approved_admin_can_manage_telegram_grants(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password")

    try:
        store.add_telegram_access_grant("user@example.com", "@alice_user", PHONE_SECRET)
    except PermissionError as exc:
        assert "admin" in str(exc)
    else:
        raise AssertionError("non-admin users must not manage Telegram access")


def test_one_contact_update_binds_matching_username_and_phone_grants(tmp_path):
    store = AuthStore(tmp_path / "auth.sqlite3")
    admin = store.bootstrap_admin("admin@example.com", "admin-password")
    username = store.add_telegram_access_grant(admin.email, "@multi_user", PHONE_SECRET)
    phone = store.add_telegram_access_grant(admin.email, "+82 10 1234 5678", PHONE_SECRET)

    result = store.authorize_telegram_user(
        303,
        username="multi_user",
        phone_number="+821012345678",
        contact_user_id=303,
        phone_hash_secret=PHONE_SECRET,
    )

    assert result.authorized and result.newly_bound
    assert {grant.telegram_user_id for grant in store.list_telegram_access_grants()} == {303}
    assert store.revoke_telegram_access_grant(admin.email, username.id)
    assert store.authorize_telegram_user(303, phone_hash_secret=PHONE_SECRET).authorized
    assert store.revoke_telegram_access_grant(admin.email, phone.id)
    assert not store.authorize_telegram_user(303, phone_hash_secret=PHONE_SECRET).authorized
