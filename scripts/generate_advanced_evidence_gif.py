"""Generate a compact research-evidence GIF for advanced dashboard documentation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
DEFAULT_OUTPUT = ASSETS_DIR / "advanced_evidence.gif"

EVIDENCE_IMAGES = (
    ("1", ASSETS_DIR / "sweep_latency.png"),
    ("2", ASSETS_DIR / "sweep_quantization.png"),
    ("3", ASSETS_DIR / "gkp_proxy.png"),
    ("4", ASSETS_DIR / "drift_recovery.png"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output GIF path (default: assets/advanced_evidence.gif).",
    )
    parser.add_argument("--fps", type=float, default=9.0, help="Output frame rate.")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.7,
        help="Seconds each slide is held (default: 1.7).",
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
        default=950,
        help="Maximum output frame width before resizing.",
    )
    parser.add_argument(
        "--label-size",
        type=int,
        default=20,
        help="Corner label font size (default: 20).",
    )
    parser.add_argument("--gif-colors", type=int, default=144, help="Palette colors for GIF quantization.")
    return parser.parse_args()


def ensure_evidence_images() -> None:
    missing = [str(path) for _, path in EVIDENCE_IMAGES if not path.exists()]
    if not missing:
        return

    print("Missing evidence PNGs required for advanced evidence GIF:")
    for path in missing:
        print(f"  - {path}")
    print("Please generate or restore these assets before re-running.")
    sys.exit(1)


def _label_slide(canvas: Image.Image, text: str, label_size: int) -> None:
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=label_size)
    except OSError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    w = text_bbox[2] - text_bbox[0]
    h = text_bbox[3] - text_bbox[1]
    x = 12
    y = canvas.height - h - 12
    draw.rounded_rectangle(
        (x - 5, y - 4, x + w + 9, y + h + 4),
        radius=max(6, label_size // 4),
        fill=(0, 0, 0, 170),
        outline=(255, 255, 255, 130),
        width=1,
    )
    draw.text((x, y), text, fill=(255, 255, 255, 235), font=font)


def _load_and_fit(path: Path, max_w: int, max_h: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    resized = ImageOps.contain(image, (max_w, max_h), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), "#0d1117")
    x = (max_w - resized.width) // 2
    y = (max_h - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _compose_sweep_slide(
    left_path: Path,
    right_path: Path,
    target_w: int,
    panel_pad: int = 12,
    gutter: int = 12,
    title: str = "Control Co-design Sweeps",
) -> Image.Image:
    target_h = 430
    title_h = 34
    panel_w = max(1, (target_w - 2 * panel_pad - gutter) // 2)
    panel_h = target_h - title_h - panel_pad * 2
    canvas = Image.new("RGB", (target_w, target_h), "#0d1117")

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        title_font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    title_x = max(panel_pad, (target_w - title_w) // 2)
    draw.text((title_x, 8), title, fill=(200, 209, 217), font=title_font)

    left_panel = _load_and_fit(left_path, panel_w, panel_h)
    right_panel = _load_and_fit(right_path, panel_w, panel_h)
    y = title_h
    x_left = panel_pad
    x_right = panel_pad + panel_w + gutter
    canvas.paste(left_panel, (x_left, y))
    canvas.paste(right_panel, (x_right, y))
    return canvas


def _compose_single_slide(
    path: Path,
    target_w: int,
    target_h: int,
    title: str,
) -> Image.Image:
    canvas = Image.new("RGB", (target_w, target_h), "#0d1117")
    image = _load_and_fit(path, target_w, target_h - 34)
    x = (target_w - image.width) // 2
    y = (target_h - image.height + 6) // 2
    canvas.paste(image, (x, y))

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    title_bbox = draw.textbbox((0, 0), title, font=font)
    tw = title_bbox[2] - title_bbox[0]
    draw.rectangle((12, 6, 16 + tw, 32), fill=(13, 17, 23))
    draw.text((14, 8), title, fill=(200, 209, 217), font=font)
    return canvas


def build_slides(config: RenderConfig) -> list[Image.Image]:
    left, right, gkp, drift = [path for _, path in EVIDENCE_IMAGES]
    slides: list[Image.Image] = []
    slide1 = _compose_sweep_slide(
        left_path=left,
        right_path=right,
        target_w=config.max_width,
    )
    _label_slide(slide1, "Evidence 1", config.label_size)
    slide2 = _compose_single_slide(
        path=gkp,
        target_w=config.max_width,
        target_h=410,
        title="Fault tolerance / GKP proxy",
    )
    _label_slide(slide2, "Evidence 2", config.label_size)
    slide3 = _compose_single_slide(
        path=drift,
        target_w=config.max_width,
        target_h=500,
        title="Stability / 24h drift automation",
    )
    _label_slide(slide3, "Evidence 3", config.label_size)
    slides.extend([slide1, slide2, slide3])
    return slides


def build_frames(
    images: list[Image.Image], config: RenderConfig
) -> tuple[list[Image.Image], list[int]]:
    hold_frames = max(1, int(round(config.hold_seconds * config.fps)))
    fade_frames = max(1, int(round(config.crossfade_seconds * config.fps)))
    frame_duration_ms = max(1, int(1000 / config.fps))
    frames: list[Image.Image] = []
    durations: list[int] = []

    for idx, image in enumerate(images):
        for _ in range(hold_frames):
            frames.append(image)
            durations.append(frame_duration_ms)
        if idx + 1 < len(images):
            nxt = images[idx + 1]
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
        raise RuntimeError("No frames generated for advanced evidence GIF.")

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


def main() -> None:
    args = parse_args()
    ensure_evidence_images()

    config = RenderConfig(
        fps=args.fps,
        hold_seconds=args.hold_seconds,
        crossfade_seconds=args.crossfade_seconds,
        max_width=args.max_width,
        label_size=args.label_size,
        colors=args.gif_colors,
        output=args.output,
    )

    slides = build_slides(config)
    target_w = max(slide.width for slide in slides)
    target_h = max(slide.height for slide in slides)
    for idx, slide in enumerate(slides):
        if slide.width != target_w or slide.height != target_h:
            canvas = Image.new("RGB", (target_w, target_h), "#0d1117")
            x = (target_w - slide.width) // 2
            y = (target_h - slide.height) // 2
            canvas.paste(slide, (x, y))
            slides[idx] = canvas

    frames, durations = build_frames(slides, config)
    optimize_and_save(frames, durations, config)


if __name__ == "__main__":
    main()
