from __future__ import annotations

import sqlite3

from imperial_rag.app.auth import AuthStore, AuthenticationStatus
from imperial_rag.app import auth as auth_module


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
