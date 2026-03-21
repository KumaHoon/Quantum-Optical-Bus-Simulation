"""Project README health audit for markdown rendering and link sanity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urldefrag


ROOT = Path(__file__).resolve().parent.parent
READMEs: tuple[Path, ...] = (
    ROOT / "README.md",
    ROOT / "docs" / "README.ja.md",
    ROOT / "docs" / "README.ko.md",
    ROOT / "docs" / "README.zh.md",
)


IMAGE_EXTS = {".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp"}
MD_EXTS = {".md"}


@dataclass
class Failure:
    file: Path
    line: int
    message: str


def _is_reference_like(target: str) -> bool:
    return bool(
        target.startswith("#")
        or target.startswith("http://")
        or target.startswith("https://")
        or target.startswith("mailto:")
        or target.startswith("data:")
        or target.startswith("javascript:")
        or target.startswith("ftp://")
    )


def _parse_links(line: str) -> Iterable[tuple[bool, str, int]]:
    # Returns tuples of (is_image, target, start_col).
    pattern = re.compile(r"(!?\[[^\]]*\])\(\s*([^)\\s]+)(?:\s+\"[^\"]*\")?\s*\)")
    for m in pattern.finditer(line):
        full = m.group(1)
        target = m.group(2)
        yield (full.startswith("!"), target, m.start(2))


def _line_failures(file: Path, line: str, line_num: int, index_offset: int) -> list[Failure]:
    failures: list[Failure] = []
    if line.startswith("--- ##"):
        failures.append(Failure(file, line_num, "header marker on same line with divider"))

    if line.startswith("```mermaid"):
        # mermaid header should be on its own clean line (space after backticks not required here but valid)
        if line.strip() != "```mermaid":
            failures.append(Failure(file, line_num, "mermaid fence must be exactly ```mermaid"))

    # Guard fragile KaTeX/LaTeX math patterns that can break on underscore.
    if re.search(r"\\text\{[^}]*_[^}]*\}", line):
        failures.append(
            Failure(
                file,
                line_num,
                "math fragment uses \\\\text with underscore; prefer \\\\mathrm for unicode subscripts",
            )
        )

    # Guard accidental inline-code-looking headers that can hide broken structure.
    if re.search(r"^#{1,6}\\S", line):
        failures.append(Failure(file, line_num, "header missing space after hashes"))

    return failures


def _heading_failures(lines: list[str], file: Path) -> list[Failure]:
    failures: list[Failure] = []
    in_code = False
    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        if re.match(r"^#{1,6}\s*", stripped):
            if not re.match(r"^#{1,6}\s+\S", stripped):
                failures.append(Failure(file, i + 1, "non-space header after hashes"))
                continue

            prev = lines[i - 1].strip() if i > 0 else ""
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if i > 0 and prev not in {"", "---"}:
                failures.append(Failure(file, i + 1, "missing blank line before heading"))
            if nxt != "":
                failures.append(Failure(file, i + 1, "missing blank line after heading"))

    return failures


def _image_or_anchor_failures(lines: list[str], file: Path) -> list[Failure]:
    failures: list[Failure] = []
    in_code = False

    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        if '<p align="center">' in line:
            prev = lines[i - 2].strip() if i - 2 >= 0 else ""
            if prev != "":
                failures.append(
                    Failure(file, i, "center image block should have blank line before <p align>")
                )

            # find closing tag
            j = i
            while j < len(lines):
                if "</p>" in lines[j]:
                    break
                j += 1
            if j >= len(lines):
                failures.append(Failure(file, i, 'unclosed <p align="center"> block'))
            else:
                next_line = lines[j + 1].strip() if j + 1 < len(lines) else ""
                if next_line != "":
                    failures.append(
                        Failure(file, i, "center image block should have blank line after </p>")
                    )

        for is_img, target, col in _parse_links(line):
            if (
                target.startswith("assets/")
                or target.startswith("../assets/")
                or target.startswith("./assets/")
            ):
                cleaned, _ = urldefrag(target)
                candidate = (file.parent / cleaned).resolve()
                if not candidate.exists():
                    failures.append(Failure(file, i, f"missing linked file: {cleaned} (col {col})"))
                continue

            # markdown crosslinks and local files
            if _is_reference_like(target):
                continue

            if target.startswith("../") or target.startswith("./"):
                cleaned, _ = urldefrag(target)
                candidate = (file.parent / cleaned).resolve()
                if not candidate.exists():
                    failures.append(Failure(file, i, f"missing linked file: {cleaned} (col {col})"))
                continue

            if target.endswith(".md"):
                cleaned, _ = urldefrag(target)
                candidate = (file.parent / cleaned).resolve()
                if not candidate.exists():
                    failures.append(Failure(file, i, f"missing markdown link: {cleaned}"))
                continue

            if Path(target).suffix.lower() in IMAGE_EXTS | MD_EXTS:
                cleaned, _ = urldefrag(target)
                candidate = (file.parent / cleaned).resolve()
                if not candidate.exists():
                    failures.append(Failure(file, i, f"missing resource link: {cleaned}"))

            # explicit report for malformed mermaid fenced lines is handled separately
    return failures


def _mermaid_failures(lines: list[str], file: Path) -> list[Failure]:
    failures: list[Failure] = []
    i = 0
    in_mermaid = False

    while i < len(lines):
        line = lines[i].strip()

        if line == "```mermaid":
            if in_mermaid:
                failures.append(Failure(file, i + 1, "nested mermaid block"))
                i += 1
                continue

            if i + 1 >= len(lines):
                failures.append(Failure(file, i + 1, "empty mermaid block"))
                failures.append(Failure(file, i + 1, "mermaid block not closed"))
                break

            # Next non-empty line should define the diagram type.
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines):
                failures.append(Failure(file, i + 1, "mermaid block has no content"))
                break
            if not lines[j].lstrip().startswith(("flowchart", "graph")):
                failures.append(
                    Failure(
                        file,
                        j + 1,
                        "mermaid block should start with flowchart/graph on first content line",
                    )
                )

            in_mermaid = True
            i = j + 1
            continue

        if in_mermaid and line == "```":
            in_mermaid = False
            i += 1
            continue

        if in_mermaid and line.startswith("```") and line != "```":
            failures.append(Failure(file, i + 1, "unexpected fenced block inside mermaid block"))

        i += 1

    if in_mermaid:
        failures.append(Failure(file, len(lines), "mermaid block not closed"))

    return failures


def _bom_failures(file: Path) -> list[Failure]:
    data = file.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return [Failure(file, 1, "UTF-8 BOM found")]
    return []


def audit_file(path: Path) -> list[Failure]:
    lines = path.read_text(encoding="utf-8").splitlines()
    failures: list[Failure] = []
    failures.extend(_bom_failures(path))

    for idx, line in enumerate(lines, start=1):
        failures.extend(_line_failures(path, line, idx, idx))

    failures.extend(_heading_failures(lines, path))
    failures.extend(_image_or_anchor_failures(lines, path))
    failures.extend(_mermaid_failures(lines, path))
    return failures


def main() -> int:
    all_failures: list[Failure] = []
    for path in READMEs:
        if not path.exists():
            all_failures.append(Failure(path, 0, "README target file missing"))
            continue

        all_failures.extend(audit_file(path))

    if not all_failures:
        print("README health audit PASS")
        return 0

    print("README health audit FAIL")
    for f in all_failures:
        print(f"{f.file.as_posix()}:{f.line}: {f.message}")
    print(f"Total issues: {len(all_failures)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
