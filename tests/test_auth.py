from __future__ import annotations

from imperial_rag.auth import AuthStore, AuthenticationStatus


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
