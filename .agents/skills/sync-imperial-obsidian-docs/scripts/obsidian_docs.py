#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


VAULT_NAME = "Second brain"
VAULT_PATH = Path("/Users/danil/Documents/Second brain")
NOTE_FOLDER = "1. Projects/Imperial RAG"
NOTES = {
    "index": f"{NOTE_FOLDER}/Imperial RAG.md",
    "brief": f"{NOTE_FOLDER}/Imperial RAG - 01 - Project Brief.md",
    "architecture": f"{NOTE_FOLDER}/Imperial RAG - 02 - Architecture.md",
    "ingestion": f"{NOTE_FOLDER}/Imperial RAG - 03 - Data and Ingestion.md",
    "retrieval": f"{NOTE_FOLDER}/Imperial RAG - 04 - Retrieval and Answering.md",
    "observability": f"{NOTE_FOLDER}/Imperial RAG - 05 - Observability and Evaluation.md",
    "decisions": f"{NOTE_FOLDER}/Imperial RAG - 06 - Architecture Decisions.md",
    "current-state": f"{NOTE_FOLDER}/Imperial RAG - 07 - Current State.md",
    "roadmap": f"{NOTE_FOLDER}/Imperial RAG - 08 - Roadmap and Planned Directions.md",
    "runbook": f"{NOTE_FOLDER}/Imperial RAG - 09 - Runbook.md",
    "pipeline-schema": f"{NOTE_FOLDER}/Imperial RAG - 10 - Pipeline Schema.md",
    "database-schema": f"{NOTE_FOLDER}/Imperial RAG - 11 - Database Schema.md",
}


class ObsidianDocsError(RuntimeError):
    pass


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _js(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _note_path(note_id: str) -> str:
    try:
        return NOTES[note_id]
    except KeyError as exc:
        raise ObsidianDocsError(f"Unknown Imperial note id: {note_id}") from exc


def _eval_payload(stdout: str) -> Any:
    for line in reversed(stdout.splitlines()):
        if line.startswith("=> "):
            try:
                return json.loads(line[3:])
            except json.JSONDecodeError as exc:
                raise ObsidianDocsError("Obsidian returned malformed JSON") from exc
    detail = next((line.strip() for line in stdout.splitlines() if line.strip()), "no output")
    raise ObsidianDocsError(f"Obsidian eval returned no result: {detail[:300]}")


@dataclass
class ObsidianDocsClient:
    runner: Runner = _default_runner
    which: Callable[[str], str | None] = shutil.which

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        command = ["obsidian", f"vault={VAULT_NAME}", *arguments]
        result = self.runner(command)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown CLI error").strip().splitlines()[0]
            raise ObsidianDocsError(f"Obsidian CLI failed ({result.returncode}): {detail[:300]}")
        return result

    def _eval(self, code: str) -> Any:
        return _eval_payload(self._run("eval", f"code={code}").stdout)

    def check(self) -> dict[str, Any]:
        if self.which("obsidian") is None:
            raise ObsidianDocsError("Obsidian CLI is not installed or not on PATH")
        paths = json.dumps(list(NOTES.values()), ensure_ascii=False)
        code = (
            "(() => {"
            f"const paths = {paths};"
            "const missing = paths.filter((path) => !app.vault.getAbstractFileByPath(path));"
            "return JSON.stringify({"
            "name: app.vault.getName(),"
            "path: app.vault.adapter.getBasePath(),"
            "missing"
            "});"
            "})()"
        )
        payload = self._eval(code)
        if not isinstance(payload, dict):
            raise ObsidianDocsError("Obsidian vault check returned an unexpected payload")
        if payload.get("name") != VAULT_NAME:
            raise ObsidianDocsError(f"Expected vault {VAULT_NAME!r}, got {payload.get('name')!r}")
        actual_path = Path(str(payload.get("path", ""))).expanduser()
        if os.path.realpath(actual_path) != os.path.realpath(VAULT_PATH):
            raise ObsidianDocsError(f"Expected vault path {VAULT_PATH}, got {actual_path}")
        missing = payload.get("missing")
        if not isinstance(missing, list):
            raise ObsidianDocsError("Obsidian vault check returned an invalid missing-note list")
        if missing:
            raise ObsidianDocsError("Missing registered Imperial notes: " + ", ".join(map(str, missing)))
        return {
            "vault_name": VAULT_NAME,
            "vault_path": str(VAULT_PATH),
            "note_count": len(NOTES),
            "missing": [],
        }

    def read(self, note_id: str) -> str:
        path = _note_path(note_id)
        code = (
            "(async () => {"
            f"const path = {_js(path)};"
            "const file = app.vault.getAbstractFileByPath(path);"
            "if (!file || file.extension !== 'md') throw new Error('registered note is missing');"
            "return JSON.stringify(await app.vault.read(file));"
            "})()"
        )
        payload = self._eval(code)
        if not isinstance(payload, str):
            raise ObsidianDocsError("Obsidian note read returned an unexpected payload")
        return payload

    def write(self, note_id: str, content: str, expected_sha256: str) -> dict[str, Any]:
        if content == "":
            raise ObsidianDocsError("Refusing to replace a registered note with empty content")
        current = self.read(note_id)
        current_sha256 = _sha256(current)
        if current_sha256 != expected_sha256:
            raise ObsidianDocsError(
                f"Concurrent edit detected for {note_id}: expected {expected_sha256}, found {current_sha256}"
            )
        path = _note_path(note_id)
        code = (
            "(async () => {"
            f"const path = {_js(path)};"
            f"const expected = {_js(current)};"
            f"const content = {_js(content)};"
            "const file = app.vault.getAbstractFileByPath(path);"
            "if (!file || file.extension !== 'md') throw new Error('registered note is missing');"
            "const before = await app.vault.read(file);"
            "if (before !== expected) throw new Error('note changed after optimistic-lock check');"
            "await app.vault.modify(file, content);"
            "const after = await app.vault.read(file);"
            "if (after !== content) throw new Error('exact read-back verification failed');"
            "return JSON.stringify({ok: true, path, characters: after.length});"
            "})()"
        )
        try:
            payload = self._eval(code)
        except ObsidianDocsError:
            # Large note writes can complete while the desktop CLI loses the eval return value.
            # Treat a separate exact read-back as authoritative; otherwise preserve the original error.
            if self.read(note_id) != content:
                raise
            payload = {"ok": True}
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ObsidianDocsError("Obsidian note write returned an unexpected payload")
        return {
            "note": note_id,
            "path": path,
            "characters": len(content),
            "sha256": _sha256(content),
            "verified": True,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely read and write the registered Imperial RAG Obsidian notes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Verify the CLI, vault identity, and registered note set.")
    check_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    read_parser = subparsers.add_parser("read", help="Read a registered note and report its SHA-256 on stderr.")
    read_parser.add_argument("--note", required=True, choices=tuple(NOTES))

    write_parser = subparsers.add_parser("write", help="Replace a registered note from stdin with optimistic locking.")
    write_parser.add_argument("--note", required=True, choices=tuple(NOTES))
    write_parser.add_argument("--expect-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None, *, client: ObsidianDocsClient | None = None) -> int:
    args = _parser().parse_args(argv)
    docs = client or ObsidianDocsClient()
    try:
        if args.command == "check":
            result = docs.check()
            if args.json:
                print(json.dumps(result, sort_keys=True))
            else:
                print(
                    f"Obsidian docs gate ready: {result['vault_name']} at {result['vault_path']} "
                    f"({result['note_count']} notes)"
                )
            return 0
        if args.command == "read":
            content = docs.read(args.note)
            print(f"note={args.note} sha256={_sha256(content)}", file=sys.stderr)
            print(content, end="")
            return 0
        if args.command == "write":
            result = docs.write(args.note, sys.stdin.read(), args.expect_sha256)
            print(json.dumps(result, sort_keys=True))
            return 0
    except ObsidianDocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
