from __future__ import annotations

from typing import Any


_COOKIE_COMPONENT: Any | None = None

_COOKIE_COMPONENT_HTML = """
<span aria-hidden="true" hidden></span>
"""

_COOKIE_COMPONENT_JS = """
const completedOperations = new WeakMap()

export default function (component) {
  const { data, parentElement, setStateValue } = component
  const operationId = String(data?.operation_id ?? "")
  if (!operationId || completedOperations.get(parentElement) === operationId) return

  const name = encodeURIComponent(String(data?.name ?? ""))
  const token = encodeURIComponent(String(data?.token ?? ""))
  const secure = data?.secure ? "; Secure" : ""
  let success = false

  try {
    if (data?.action === "set") {
      const maxAge = Number(data?.max_age ?? 0)
      document.cookie = `${name}=${token}; Path=/; Max-Age=${maxAge}; SameSite=Strict${secure}`
      success = document.cookie.split("; ").some(cookie => cookie === `${name}=${token}`)
    } else if (data?.action === "delete") {
      document.cookie = `${name}=; Path=/; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Strict${secure}`
      success = !document.cookie.split("; ").some(cookie => cookie.startsWith(`${name}=`))
    }
  } catch (_error) {
    success = false
  }

  completedOperations.set(parentElement, operationId)
  setStateValue("acknowledged", { operation_id: operationId, success })
}
"""


def render_cookie_operation(
    st: Any,
    *,
    action: str,
    operation_id: str,
    token: str,
    cookie_name: str,
    max_age_seconds: int,
    secure: bool,
    key: str,
) -> Any:
    if action not in {"set", "delete"}:
        raise ValueError("unsupported cookie action")
    component = _cookie_component(st)
    return component(
        key=key,
        data={
            "action": action,
            "name": cookie_name,
            "operation_id": operation_id,
            "token": token,
            "max_age": max_age_seconds,
            "secure": secure,
        },
        height=0,
        width="content",
    )


def _cookie_component(st: Any) -> Any:
    global _COOKIE_COMPONENT
    if _COOKIE_COMPONENT is None:
        _COOKIE_COMPONENT = st.components.v2.component(
            "imperial_rag_auth_cookie",
            html=_COOKIE_COMPONENT_HTML,
            js=_COOKIE_COMPONENT_JS,
        )
    return _COOKIE_COMPONENT
