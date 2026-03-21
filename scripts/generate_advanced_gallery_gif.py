"""Generate a crossfading advanced gallery GIF from dashboard snapshots."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
import sys
from pathlib import Path

from PIL import Image

try:
    from figstyle import canonical_canvas_px, canonical_dpi, write_figure_meta
except ModuleNotFoundError:
    from scripts.figstyle import canonical_canvas_px, canonical_dpi, write_figure_meta

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
DASHBOARD_SCRIPT = ROOT_DIR / "scripts" / "generate_advanced_dashboard_gallery.py"
DEFAULT_OUTPUT = ASSETS_DIR / "advanced_gallery.gif"
try:
    from asset_profile import (
        PROFILE_OPTIONS,
        candidate_output_paths,
        normalize_profiles,
        resolve_outputs,
    )
except ModuleNotFoundError:
    from scripts.asset_profile import (
        PROFILE_OPTIONS,
        candidate_output_paths,
        normalize_profiles,
        resolve_outputs,
    )


ADVANCED_IMAGES = (
    ("Advanced Tab 1: Multi-mode", "dashboard_multimode.png"),
    ("Advanced Tab 2: Topology", "dashboard_topology.png"),
    ("Advanced Tab 3: Digital twin", "dashboard_digital_twin.png"),
)


def _source_image(base_dir: Path, filename: str) -> Path:
    for candidate in candidate_output_paths(base_dir, filename):
        if candidate.exists():
            return candidate
    return base_dir / filename


@dataclass(frozen=True)
class RenderConfig:
    fps: float
    hold_seconds: float
    crossfade_seconds: float
    max_width: int
    label_size: int
    colors: int
    output: Path
    save_mp4: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output GIF path (default: assets/advanced_gallery.gif).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory where source PNGs and target GIF are stored.",
    )
    parser.add_argument("--fps", type=float, default=9.0, help="Frame rate.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.8,
        help="Seconds each advanced image is held (default: 1.8).",
    )
    parser.add_argument(
        "--crossfade-seconds",
        type=float,
        default=0.65,
        help="Duration of each crossfade transition in seconds.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
        help="Maximum output frame width before resizing (default: 960).",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=24,
        help="Corner label font size.",
    )
    parser.add_argument("--colors", type=int, default=144, help="GIF palette size.")
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Also save MP4 output when ffmpeg is available.",
    )
    parser.add_argument(
        "--profile",
        default="web",
        choices=PROFILE_OPTIONS,
        help="Render profile: web, paper, or both.",
    )
    return parser.parse_args()


def _advanced_paths(base_dir: Path) -> list[tuple[str, Path]]:
    return [(label, _source_image(base_dir, name)) for label, name in ADVANCED_IMAGES]


def ensure_images_exist(base_dir: Path, profile: str) -> None:
    missing = [
        name
        for name in [name for _, name in ADVANCED_IMAGES]
        if not (any(candidate.exists() for candidate in candidate_output_paths(base_dir, name)))
    ]
    if not missing:
        return

    regen_profile = "both" if profile in {"paper", "both"} else profile
    print("Missing advanced PNGs detected, regenerating advanced dashboard images...")
    subprocess.run(
        [
            sys.executable,
            str(DASHBOARD_SCRIPT),
            "--output-dir",
            str(base_dir),
            "--profile",
            regen_profile,
        ],
        check=True,
    )


def load_labeled_image(path: Path, target_width: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.width > target_width:
        scale = target_width / image.width
        image = image.resize(
            (
                target_width,
                max(1, int(image.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return image


def make_equal_canvas(images: list[Image.Image], bg=(10, 18, 30, 255)) -> list[Image.Image]:
    target_w = max(im.width for im in images)
    target_h = max(im.height for im in images)
    framed: list[Image.Image] = []
    for image in images:
        if image.width == target_w and image.height == target_h:
            framed.append(image)
            continue
        canvas = Image.new("RGBA", (target_w, target_h), bg)
        x = (target_w - image.width) // 2
        y = (target_h - image.height) // 2
        canvas.alpha_composite(image, (x, y))
        framed.append(canvas)
    return framed


def build_frames(
    images: list[Image.Image], config: RenderConfig
) -> tuple[list[Image.Image], list[int]]:
    hold_frames = max(1, int(round(config.hold_seconds * config.fps)))
    fade_frames = max(1, int(round(config.crossfade_seconds * config.fps)))
    frame_duration_ms = max(1, int(1000 / config.fps))
    frames: list[Image.Image] = []
    durations: list[int] = []

    for i, image in enumerate(images):
        for _ in range(hold_frames):
            frames.append(image)
            durations.append(frame_duration_ms)
        if i + 1 >= len(images):
            continue
        nxt = images[i + 1]
        for step in range(1, fade_frames + 1):
            alpha = step / (fade_frames + 1)
            frames.append(Image.blend(image, nxt, alpha))
            durations.append(frame_duration_ms)
    assert len(frames) > 2, "Generated GIF must contain more than one frame."
    return frames, durations


def optimize_and_save(
    frames: list[Image.Image],
    durations: list[int],
    config: RenderConfig,
    profile: str,
) -> None:
    if not frames:
        raise RuntimeError("No frames generated.")

    quantized = [
        frame.convert("RGB").quantize(
            colors=config.colors,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    for out in resolve_outputs(config.output, profile):
        quantized[0].save(
            out,
            save_all=True,
            append_images=quantized[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
            include_color_table=True,
        )
        print(f"[OK] Saved GIF: {out} ({out.stat().st_size} bytes)")
        target_profile = out.parent.name
        with Image.open(out) as frame:
            canvas_px = frame.size
        write_figure_meta(
            out,
            figure_id=out.stem,
            profile=target_profile,
            generator_script="scripts/generate_advanced_gallery_gif.py",
            generator_args=(f"--output={config.output}", f"--profile={target_profile}"),
            labels={
                "title": "Advanced gallery",
                "xlabel": "frame index",
                "ylabel": "N/A",
            },
            units={"x": "count", "time": "frame"},
            notes="Cross-fade GIF for advanced dashboard scenarios.",
            seed=11,
            dpi=canonical_dpi(target_profile),
            canvas_px=canvas_px
            if isinstance(canvas_px, tuple)
            else canonical_canvas_px(target_profile),
        )
        if config.save_mp4:
            save_mp4_if_available(out)


def save_mp4_if_available(gif_path: Path) -> None:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[INFO] ffmpeg not available; skipping MP4 generation.")
        return

    try:
        import imageio.v2 as imageio

        with imageio.get_reader(gif_path) as reader:
            frames = [f for f in reader]
        if not frames:
            print("[WARN] GIF has no frames; skipped MP4.")
            return
        imageio.mimsave(gif_path.with_suffix(".mp4"), frames, fps=9.0)
        print(f"[OK] Also saved MP4: {gif_path.with_suffix('.mp4')}")
    except Exception as exc:
        print(f"[WARN] MP4 generation failed: {exc}")


def main() -> None:
    args = parse_args()
    if args.output.parent == Path("."):
        args.output = args.output_dir / args.output.name
    ensure_images_exist(args.output_dir, args.profile)

    advanced_images = _advanced_paths(args.output_dir)
    raw_images = [load_labeled_image(path, args.max_width) for _, path in advanced_images]
    framed = make_equal_canvas(raw_images)
    config = RenderConfig(
        fps=args.fps,
        hold_seconds=args.hold_seconds,
        crossfade_seconds=args.crossfade_seconds,
        max_width=args.max_width,
        label_size=args.label_size,
        colors=args.colors,
        output=args.output,
        save_mp4=args.save_mp4,
    )
    profiles = normalize_profiles(args.profile)
    frames, durations = build_frames(framed, config)
    for profile in profiles:
        optimize_and_save(frames, durations, config, profile)


if __name__ == "__main__":
    main()
