"""Checks for HDL feedforward golden vector generation."""

from pathlib import Path

from scripts.export_golden_vectors import generate_vectors, write_mem_files


def test_export_golden_vectors_generates_expected_counts(tmp_path: Path) -> None:
    vectors = generate_vectors(count=64, seed=13)
    assert len(vectors) == 64

    write_mem_files(vectors, tmp_path)
    stim = (tmp_path / "stimulus.mem").read_text(encoding="utf-8").strip().splitlines()
    exp = (tmp_path / "expected.mem").read_text(encoding="utf-8").strip().splitlines()

    assert len(stim) == 64
    assert len(exp) == 64
    assert all(len(line.strip()) == 4 for line in stim)
    assert all(len(line.strip()) == 4 for line in exp)
