"""Checks for HDL feedforward golden vector generation."""

import json
from pathlib import Path

from scripts.export_golden_vectors import (
    _qformat_contract,
    generate_vectors,
    write_contract,
    write_mem_files,
)


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


def test_export_contract_contains_interface_metadata(tmp_path: Path) -> None:
    manifest = tmp_path / "contract.json"
    write_contract(manifest, count=16, seed=7)
    contract = json.loads(manifest.read_text(encoding="utf-8"))
    assert contract["format"]["format"] == "Q1.15"
    assert contract["format"]["pipeline_latency_cycles"] == 2
    assert contract["format"]["lut_depth"] == 16
    assert _qformat_contract()["data_width"] == 16
