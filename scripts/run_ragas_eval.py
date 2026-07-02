from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from imperial_rag.evals import ragas_runner as _ragas_runner  # noqa: E402
from imperial_rag.evals.ragas_runner import *  # noqa: E402,F403

_build_settings = _ragas_runner._build_settings
_configure_observability = _ragas_runner._configure_observability
_duration_ms = _ragas_runner._duration_ms
_load_project_env = _ragas_runner._load_project_env
_log_failure = _ragas_runner._log_failure
main = _ragas_runner.main

for _name in dir(_ragas_runner):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_ragas_runner, _name)

__all__ = [name for name in dir(_ragas_runner) if not name.startswith("__")]


if __name__ == "__main__":
    main()
