from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_project_env(workspace_root: Path | str | None = None) -> bool:
    env_path = _env_path(workspace_root)
    if not env_path.exists():
        return False
    return bool(load_dotenv(dotenv_path=env_path, override=False))


def _env_path(workspace_root: Path | str | None = None) -> Path:
    if workspace_root is not None:
        return Path(workspace_root).expanduser().resolve() / ".env"
    return Path(__file__).resolve().parents[3] / ".env"
