from __future__ import annotations

from pathlib import Path


def test_compose_defines_persistent_self_hosted_phoenix():
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "arizephoenix/phoenix:latest" in compose
    assert '"127.0.0.1:6006:6006"' in compose
    assert '"127.0.0.1:4317:4317"' in compose
    assert "PHOENIX_WORKING_DIR: /mnt/data" in compose
    assert "phoenix_data:/mnt/data" in compose


def test_old_superpowers_docs_point_to_phoenix_supersession_spec():
    spec = Path("docs/superpowers/specs/2026-06-02-local-rag-system-design.md").read_text(encoding="utf-8")
    plan = Path("docs/superpowers/plans/2026-06-02-local-rag-system.md").read_text(encoding="utf-8")

    assert "`2026-06-03-phoenix-observability-design.md` supersedes" in spec
    assert "`2026-06-03-phoenix-observability-design.md` supersedes" in plan
