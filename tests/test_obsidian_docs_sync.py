from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".agents"
    / "skills"
    / "sync-imperial-obsidian-docs"
    / "scripts"
    / "obsidian_docs.py"
)
SPEC = importlib.util.spec_from_file_location("imperial_obsidian_docs", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
obsidian_docs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = obsidian_docs
SPEC.loader.exec_module(obsidian_docs)


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["obsidian"], returncode, stdout=stdout, stderr=stderr)


def _eval_result(payload: Any) -> subprocess.CompletedProcess[str]:
    return _completed(stdout=f"=> {json.dumps(payload, ensure_ascii=False)}\n")


def _argument(command: Sequence[str], prefix: str) -> str:
    return next(argument[len(prefix) :] for argument in command if argument.startswith(prefix))


def test_note_registry_is_complete_and_allowlisted() -> None:
    assert tuple(obsidian_docs.NOTES) == (
        "index",
        "brief",
        "architecture",
        "ingestion",
        "retrieval",
        "observability",
        "decisions",
        "current-state",
        "roadmap",
        "runbook",
        "pipeline-schema",
        "database-schema",
    )
    assert len(set(obsidian_docs.NOTES.values())) == 12
    assert all(path.startswith("1. Projects/Imperial RAG/") for path in obsidian_docs.NOTES.values())
    assert all(path.endswith(".md") for path in obsidian_docs.NOTES.values())


def test_check_verifies_vault_identity_and_all_registered_notes() -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        assert command[:3] == ["obsidian", "vault=Second brain", "eval"]
        code = _argument(command, "code=")
        assert all(path in code for path in obsidian_docs.NOTES.values())
        return _eval_result(
            {
                "name": "Second brain",
                "path": "/Users/danil/Documents/Second brain",
                "missing": [],
            }
        )

    client = obsidian_docs.ObsidianDocsClient(runner=runner, which=lambda _: "/usr/local/bin/obsidian")

    assert client.check() == {
        "vault_name": "Second brain",
        "vault_path": "/Users/danil/Documents/Second brain",
        "note_count": 12,
        "missing": [],
    }


def test_check_rejects_missing_cli_without_running_a_command() -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"runner should not be called: {command}")

    client = obsidian_docs.ObsidianDocsClient(runner=runner, which=lambda _: None)

    with pytest.raises(obsidian_docs.ObsidianDocsError, match="not installed"):
        client.check()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"name": "Wrong vault", "path": "/tmp/wrong", "missing": []}, "Expected vault"),
        ({"name": "Second brain", "path": "/tmp/wrong", "missing": []}, "Expected vault path"),
        (
            {
                "name": "Second brain",
                "path": "/Users/danil/Documents/Second brain",
                "missing": ["1. Projects/Imperial RAG/missing.md"],
            },
            "Missing registered Imperial notes",
        ),
    ],
)
def test_check_rejects_wrong_vault_or_missing_notes(payload: dict[str, Any], message: str) -> None:
    client = obsidian_docs.ObsidianDocsClient(
        runner=lambda _: _eval_result(payload),
        which=lambda _: "/usr/local/bin/obsidian",
    )

    with pytest.raises(obsidian_docs.ObsidianDocsError, match=message):
        client.check()


def test_read_and_write_preserve_exact_markdown_and_use_optimistic_lock() -> None:
    state = {
        "content": "---\ntitle: Old\n---\n\n# Old\n",
        "writes": 0,
    }
    revised = (
        "---\n"
        "title: Imperial RAG — Retrieval\n"
        "updated: 2026-07-13\n"
        "---\n\n"
        "# Unicode 한국어 and Русский\n\n"
        "```mermaid\nflowchart LR\n  A[documents/] --> B[`code`]\n```\n\n"
        "Literal backslashes: C:\\tmp\\notes and \\n stays text.\n"
    )

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        code = _argument(command, "code=")
        if "await app.vault.modify" not in code:
            return _eval_result(state["content"])
        expected_json = code.split("const expected = ", 1)[1].split(";const content = ", 1)[0]
        content_json = code.split("const content = ", 1)[1].split(";const file = ", 1)[0]
        assert json.loads(expected_json) == state["content"]
        state["content"] = json.loads(content_json)
        state["writes"] += 1
        return _eval_result(
            {
                "ok": True,
                "path": obsidian_docs.NOTES["retrieval"],
                "characters": len(state["content"]),
            }
        )

    client = obsidian_docs.ObsidianDocsClient(runner=runner)
    old_hash = obsidian_docs._sha256(state["content"])

    result = client.write("retrieval", revised, old_hash)

    assert state == {"content": revised, "writes": 1}
    assert client.read("retrieval") == revised
    assert result == {
        "note": "retrieval",
        "path": obsidian_docs.NOTES["retrieval"],
        "characters": len(revised),
        "sha256": obsidian_docs._sha256(revised),
        "verified": True,
    }


def test_write_rejects_concurrent_edit_before_modifying_note() -> None:
    calls = 0

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _eval_result("newer note content")

    client = obsidian_docs.ObsidianDocsClient(runner=runner)

    with pytest.raises(obsidian_docs.ObsidianDocsError, match="Concurrent edit detected"):
        client.write("architecture", "replacement", "0" * 64)

    assert calls == 1


def test_write_accepts_exact_readback_when_cli_loses_eval_result() -> None:
    state = {"content": "old", "calls": 0}

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        state["calls"] += 1
        code = _argument(command, "code=")
        if "await app.vault.modify" in code:
            content_json = code.split("const content = ", 1)[1].split(";const file = ", 1)[0]
            state["content"] = json.loads(content_json)
            return _completed(stdout="undefined\n")
        return _eval_result(state["content"])

    client = obsidian_docs.ObsidianDocsClient(runner=runner)

    result = client.write("index", "new", obsidian_docs._sha256("old"))

    assert state == {"content": "new", "calls": 3}
    assert result["verified"] is True
    assert result["sha256"] == obsidian_docs._sha256("new")


def test_write_rejects_empty_content_and_unknown_note_id() -> None:
    client = obsidian_docs.ObsidianDocsClient(runner=lambda _: _eval_result("current"))

    with pytest.raises(obsidian_docs.ObsidianDocsError, match="empty content"):
        client.write("index", "", obsidian_docs._sha256("current"))
    with pytest.raises(obsidian_docs.ObsidianDocsError, match="Unknown Imperial note id"):
        client.read("../../arbitrary")


def test_nonzero_cli_failure_is_reported_without_echoing_command_content() -> None:
    client = obsidian_docs.ObsidianDocsClient(
        runner=lambda _: _completed(stderr="Obsidian is not running\nprivate second line", returncode=1)
    )

    with pytest.raises(obsidian_docs.ObsidianDocsError, match="Obsidian is not running") as error:
        client.read("index")

    assert "private second line" not in str(error.value)
