from __future__ import annotations

from types import SimpleNamespace

from imperial_rag.app import auth_cookie


def test_cookie_component_mounts_fixed_security_attributes(monkeypatch):
    registrations = []
    mounts = []

    def component(name, **kwargs):
        registrations.append((name, kwargs))

        def mount(**mount_kwargs):
            mounts.append(mount_kwargs)
            return SimpleNamespace(acknowledged={"operation_id": "op-1", "success": True})

        return mount

    streamlit = SimpleNamespace(components=SimpleNamespace(v2=SimpleNamespace(component=component)))
    monkeypatch.setattr(auth_cookie, "_COOKIE_COMPONENT", None)

    result = auth_cookie.render_cookie_operation(
        streamlit,
        action="set",
        operation_id="op-1",
        token="opaque-token",
        cookie_name="imperial_rag_session_v1",
        max_age_seconds=2_592_000,
        secure=True,
        key="auth-cookie-sync",
    )

    assert result.acknowledged == {"operation_id": "op-1", "success": True}
    assert registrations[0][0] == "imperial_rag_auth_cookie"
    javascript = registrations[0][1]["js"]
    assert "SameSite=Strict" in javascript
    assert "Max-Age" in javascript
    assert "document.cookie" in javascript
    assert "new WeakMap()" in javascript
    assert "parentElement.dataset" not in javascript
    assert mounts == [
        {
            "key": "auth-cookie-sync",
            "data": {
                "action": "set",
                "name": "imperial_rag_session_v1",
                "operation_id": "op-1",
                "token": "opaque-token",
                "max_age": 2_592_000,
                "secure": True,
            },
            "height": 0,
            "width": "content",
        }
    ]


def test_cookie_component_rejects_unknown_action(monkeypatch):
    streamlit = SimpleNamespace(components=SimpleNamespace(v2=SimpleNamespace(component=lambda *args, **kwargs: None)))
    monkeypatch.setattr(auth_cookie, "_COOKIE_COMPONENT", None)

    try:
        auth_cookie.render_cookie_operation(
            streamlit,
            action="rotate",
            operation_id="op-1",
            token="opaque-token",
            cookie_name="imperial_rag_session_v1",
            max_age_seconds=2_592_000,
            secure=False,
            key="auth-cookie-sync",
        )
    except ValueError as exc:
        assert "cookie action" in str(exc)
    else:
        raise AssertionError("unknown cookie actions must be rejected")
