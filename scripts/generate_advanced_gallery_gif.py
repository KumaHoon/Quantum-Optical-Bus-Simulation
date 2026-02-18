"""Generate a crossfading advanced gallery GIF from dashboard snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
DASHBOARD_SCRIPT = ROOT_DIR / "scripts" / "generate_advanced_dashboard_gallery.py"
DEFAULT_OUTPUT = ASSETS_DIR / "advanced_gallery.gif"


ADVANCED_IMAGES = (
    ("Advanced 1", ASSETS_DIR / "dashboard_multimode.png"),
    ("Advanced 2", ASSETS_DIR / "dashboard_topology.png"),
    ("Advanced 3", ASSETS_DIR / "dashboard_digital_twin.png"),
)


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
    parser.add_argument("--fps", type=float, default=9.0, help="GIF frame rate.")
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
    parser.add_argument("--label-size", type=int, default=24, help="Corner label font size.")
    parser.add_argument("--colors", type=int, default=144, help="GIF palette size.")
    parser.add_argument(
        "--save-mp4",
        action="store_true",
        help="Also save MP4 output when ffmpeg is available.",
    )
    return parser.parse_args()


def ensure_images_exist() -> None:
    missing = [str(p) for _, p in ADVANCED_IMAGES if not p.exists()]
    if not missing:
        return
    print("Missing advanced PNGs detected, regenerating advanced dashboard images...")
    subprocess.run([sys.executable, str(DASHBOARD_SCRIPT)], check=True)


def load_labeled_image(path: Path, label: str, target_width: int, label_size: int) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    if image.width > target_width:
        scale = target_width / image.width
        image = image.resize(
            (target_width, max(1, int(image.height * scale)),),
            Image.Resampling.LANCZOS,
        )

    try:
        label_font = ImageFont.truetype("DejaVuSans.ttf", size=label_size)
    except OSError:
        label_font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)
    text = label
    text_bbox = draw.textbbox((0, 0), text, font=label_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    pad = max(8, label_size // 3)
    x = 14
    y = image.height - text_height - (pad * 2) - 4

    bg_left = x - 6
    bg_top = y - 4
    bg_right = x + text_width + 12
    bg_bottom = y + text_height + 8
    draw.rounded_rectangle(
        (bg_left, bg_top, bg_right, bg_bottom),
        radius=max(6, label_size // 4),
        fill=(0, 0, 0, 175),
        outline=(255, 255, 255, 120),
        width=1,
    )
    draw.text((x, y), text, fill=(255, 255, 255, 235), font=label_font)

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


def build_frames(images: list[Image.Image], config: RenderConfig) -> tuple[list[Image.Image], list[int]]:
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

    return frames, durations


def optimize_and_save(
    frames: list[Image.Image],
    durations: list[int],
    config: RenderConfig,
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

    config.output.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        config.output,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
        include_color_table=True,
    )
    print(f"[OK] Saved GIF: {config.output} ({config.output.stat().st_size} bytes)")

    if config.save_mp4:
        save_mp4_if_available(config.output)


def save_mp4_if_available(gif_path: Path) -> None:
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[INFO] ffmpeg not available; skipping MP4 generation.")
        return

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        print(f"[WARN] imageio not available for MP4 generation: {exc}")
        return

    mp4_path = gif_path.with_suffix(".mp4")
    try:
        with imageio.get_reader(gif_path) as reader:
            frames = [f for f in reader]
        if not frames:
            print("[WARN] GIF has no frames; skipped MP4.")
            return

        imageio.mimsave(mp4_path, frames, fps=9.0)
        print(f"[OK] Also saved MP4: {mp4_path}")
    except Exception as exc:
        print(f"[WARN] MP4 generation failed: {exc}")


def main() -> None:
    args = parse_args()
    ensure_images_exist()

    raw_images = [
        load_labeled_image(path, label, args.max_width, args.label_size)
        for label, path in ADVANCED_IMAGES
    ]
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
    frames, durations = build_frames(framed, config)
    optimize_and_save(frames, durations, config)


if __name__ == "__main__":
    main()
