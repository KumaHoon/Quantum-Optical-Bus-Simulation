"""README style guard for rendering rules (non-clipping friendly)."""

from __future__ import annotations

from pathlib import Path
import re


README_PATH = Path("README.md")


def fail(message: str) -> None:
    print(f"FAIL: {message}")


def check_headings(lines: list[str]) -> int:
    failures = 0
    in_code_block = False
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if not line.startswith("#"):
            continue
        if not re.match(r"^#{1,6}\s+\S", line):
            fail(f"Header format should be standard markdown heading at line {i + 1}")
            continue

        prev_ok = i == 0 or lines[i - 1].strip() == "" or lines[i - 1].strip() == "---"
        next_ok = i == len(lines) - 1 or lines[i + 1].strip() == ""

        if not prev_ok or not next_ok:
            fail(f"Header is not on a standalone line at line {i + 1}")
            failures += 1

    return failures


def check_table_rows(lines: list[str]) -> int:
    failures = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "|---" in line and "|" in line:
            if line.count("|") < 3:
                fail(f"Malformed table separator at line {i + 1}")
                failures += 1
    return failures


def check_image_blocks(lines: list[str]) -> int:
    failures = 0
    for i, line in enumerate(lines):
        if line.strip() != '<p align="center">':
            continue

        before_ok = i == 0 or lines[i - 1].strip() == ""
        # find matching closing tag
        j = i + 1
        while j < len(lines) and lines[j].strip() != "</p>":
            j += 1

        if j == len(lines):
            fail(f'Unclosed <p align="center"> block at line {i + 1}')
            failures += 1
            continue

        after_ok = j == len(lines) - 1 or lines[j + 1].strip() == ""

        if not before_ok or not after_ok:
            fail(f"Center block missing blank lines at {i + 1}..{j + 1}")
            failures += 1

    return failures


def check_mermaid(lines: list[str]) -> int:
    failures = 0
    in_code_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```") and not stripped == "```mermaid":
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped != "```mermaid":
            continue

        in_code_block = True

        if i + 1 >= len(lines):
            fail("Mermaid block has no content line")
            failures += 1
            continue

        if not lines[i + 1].startswith("flowchart"):
            fail(f"Mermaid block does not start with flowchart at line {i + 1}")
            failures += 1

    return failures


def check_utf8_bom(path: Path) -> int:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        fail("README.md has UTF-8 BOM")
        return 1
    return 0


def main() -> int:
    text = README_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    failures = 0
    failures += check_headings(lines)
    failures += check_table_rows(lines)
    failures += check_image_blocks(lines)
    failures += check_mermaid(lines)
    failures += check_utf8_bom(README_PATH)

    if failures:
        print(f"Result: FAIL ({failures} rule issue(s))")
        return 1

    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
