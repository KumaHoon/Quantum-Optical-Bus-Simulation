from pathlib import Path
from PIL import Image


def _target_size_status(actual: int, target: int, label: str) -> str:
    if actual == target:
        return "PASS"
    delta = abs(actual - target)
    if delta <= 4:
        return "PASS"
    return f"WARN: {label} = {actual}px (target {target}px, delta={delta})"


def _check_png(path: Path) -> dict[str, str | int | float]:
    with Image.open(path) as im:
        return {
            "file": path.as_posix(),
            "type": "PNG",
            "exists": "PASS",
            "width": im.width,
            "height": im.height,
            "frames": 1,
            "size_kb": round(path.stat().st_size / 1024, 1),
        }


def _check_gif(path: Path) -> dict[str, str | int | float]:
    with Image.open(path) as im:
        return {
            "file": path.as_posix(),
            "type": "GIF",
            "exists": "PASS",
            "width": im.width,
            "height": im.height,
            "frames": im.n_frames,
            "size_kb": round(path.stat().st_size / 1024, 1),
        }


def audit(assets_dir: Path) -> int:
    targets_png = [
        assets_dir / "dashboard_vacuum.png",
        assets_dir / "dashboard_calibration.png",
        assets_dir / "dashboard_decoherence.png",
        assets_dir / "dashboard_multimode.png",
        assets_dir / "dashboard_topology.png",
        assets_dir / "dashboard_digital_twin.png",
        assets_dir / "scenario_gallery.gif",
        assets_dir / "advanced_gallery.gif",
        assets_dir / "advanced_evidence.gif",
        assets_dir / "calibration_demo.gif",
    ]

    print("README Figure Compliance Audit")
    print("Targets: PNG 2148 px width (7.16 in @ 300 dpi), GIF display width consistency check")
    print("=" * 72)

    failures = 0
    for path in targets_png:
        if not path.exists():
            print(f"{path.name:34} MISSING")
            failures += 1
            continue

        info = _check_png(path) if path.suffix == ".png" else _check_gif(path)
        width = int(info["width"])
        size_kb = float(info["size_kb"])
        h = int(info["height"])
        frames = int(info["frames"])

        if path.suffix == ".png":
            target = _target_size_status(width, 2148, "width")
            if target == "PASS":
                status = f"PASS: {width}x{h}px"
            else:
                status = target
                failures += 1
            print(f"{path.name:34} PNG  {status:<52} size={size_kb:8.1f} KB")
            continue

        # GIF checks
        gif_ok = width >= 900 and width <= 1050
        reason = "PASS" if gif_ok else f"FAIL: unexpected width {width}px (expected ~950)"
        if not gif_ok:
            failures += 1
        frame_note = f"{frames} frames"
        print(
            f"{path.name:34} GIF  {reason:<42} {width}x{h}px, {frame_note}, size={size_kb:8.1f} KB"
        )

    if failures:
        print(f"\nResult: FAIL ({failures} item(s) out of spec)")
        return 1

    print("\nResult: PASS")
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    assets_dir = repo / "assets"
    return audit(assets_dir)


if __name__ == "__main__":
    raise SystemExit(main())
