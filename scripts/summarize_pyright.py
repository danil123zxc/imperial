from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _relative_file(diagnostic: dict[str, Any]) -> str:
    file_path = Path(str(diagnostic["file"]))
    try:
        return file_path.relative_to(ROOT).as_posix()
    except ValueError:
        return file_path.as_posix()


def _top_level(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else path


def summarize(data: dict[str, Any]) -> str:
    diagnostics = list(data.get("generalDiagnostics") or [])
    lines: list[str] = []
    summary = data.get("summary") or {}
    if summary:
        lines.append(
            "summary: "
            f"files={summary.get('filesAnalyzed', 0)} "
            f"errors={summary.get('errorCount', 0)} "
            f"warnings={summary.get('warningCount', 0)} "
            f"information={summary.get('informationCount', 0)}"
        )
        lines.append("")

    by_top_level = Counter(_top_level(_relative_file(diagnostic)) for diagnostic in diagnostics)
    if by_top_level:
        lines.append("by path:")
        for path, count in sorted(by_top_level.items()):
            lines.append(f"  {path}: {count}")
        lines.append("")

    by_rule = Counter(
        (
            str(diagnostic.get("severity", "")),
            str(diagnostic.get("rule") or "<none>"),
        )
        for diagnostic in diagnostics
    )
    if by_rule:
        lines.append("by rule:")
        for (severity, rule), count in by_rule.most_common():
            lines.append(f"  {count:4} {severity:11} {rule}")
        lines.append("")

    by_file = Counter(_relative_file(diagnostic) for diagnostic in diagnostics)
    if by_file:
        lines.append("top files:")
        for path, count in by_file.most_common(25):
            lines.append(f"  {count:4} {path}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Pyright JSON diagnostics by path, rule, and file.")
    parser.add_argument("json_path", nargs="?", type=Path, help="Pyright --outputjson file. Reads stdin when omitted.")
    args = parser.parse_args(argv)

    raw = args.json_path.read_text(encoding="utf-8") if args.json_path else sys.stdin.read()
    data = json.loads(raw)
    print(summarize(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
