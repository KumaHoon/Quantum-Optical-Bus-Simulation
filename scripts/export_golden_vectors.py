"""Generate fixed-point stimulus and expected vectors for the LUT feedforward TB."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


LUT = [
    0,
    6047,
    11695,
    16645,
    20747,
    23991,
    26462,
    28293,
    29620,
    30568,
    31237,
    31707,
    32034,
    32261,
    32418,
    32527,
]


def _q15_from_float(x: float) -> int:
    scaled = int(round(x * (1 << 15)))
    if scaled < -(1 << 15):
        return -(1 << 15)
    if scaled > (1 << 15) - 1:
        return (1 << 15) - 1
    return scaled


def apply_lut(sample: int) -> int:
    if sample < 0:
        sign = -1
        abs_sample = min(-sample, (1 << 15) - 1)
    else:
        sign = 1
        abs_sample = min(sample, (1 << 15) - 1)

    index = (abs_sample >> 12) & 0xF
    value = LUT[index]
    return sign * value


def generate_vectors(count: int, seed: int) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    for i in range(count):
        theta = (2.0 * math.pi * i) / max(count, 1)
        jitter = 0.03 * math.sin(0.17 * seed + 0.09 * i)
        sample = _q15_from_float(0.90 * math.sin(theta) + jitter)
        expected = apply_lut(sample)
        samples.append((sample, expected))
    return samples[:count]


def to_hex16(value: int) -> str:
    return f"{value & 0xFFFF:04x}"


def write_mem_files(vectors: list[tuple[int, int]], output_dir: Path) -> None:
    stimulus_path = output_dir / "stimulus.mem"
    expected_path = output_dir / "expected.mem"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        stimulus_path.open("w", encoding="utf-8") as stim_fh,
        expected_path.open("w", encoding="utf-8") as exp_fh,
    ):
        for sample, expected in vectors:
            stim_fh.write(to_hex16(sample) + "\n")
            exp_fh.write(to_hex16(expected) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LUT golden vectors for HDL feedforward testbench."
    )
    parser.add_argument(
        "--output-dir", default="hdl/vectors", help="Output directory for .mem files."
    )
    parser.add_argument("--count", type=int, default=128, help="Number of vectors to generate.")
    parser.add_argument(
        "--seed", type=int, default=13, help="Seed-like switch for deterministic variation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectors = generate_vectors(count=args.count, seed=args.seed)
    out_dir = Path(args.output_dir)
    write_mem_files(vectors, out_dir)
    print(
        f"[OK] wrote {len(vectors)} vectors to {out_dir / 'stimulus.mem'} and {out_dir / 'expected.mem'}"
    )


if __name__ == "__main__":
    main()
