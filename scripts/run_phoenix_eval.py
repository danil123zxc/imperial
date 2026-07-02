from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from imperial_rag.evals import phoenix_experiment as _phoenix_experiment  # noqa: E402
from imperial_rag.evals.phoenix_experiment import *  # noqa: E402,F403

_build_settings = _phoenix_experiment._build_settings
_configure_observability = _phoenix_experiment._configure_observability
_configure_tracing = _phoenix_experiment._configure_tracing
_duration_ms = _phoenix_experiment._duration_ms
_load_project_env = _phoenix_experiment._load_project_env
_log_failure = _phoenix_experiment._log_failure
main = _phoenix_experiment.main

for _name in dir(_phoenix_experiment):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_phoenix_experiment, _name)

__all__ = [name for name in dir(_phoenix_experiment) if not name.startswith("__")]


if __name__ == "__main__":
    main()
