#!/usr/bin/env python3
"""Compose per-frame development visualizations into a two-row review image."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as pil_error:  # pragma: no cover - remote dependency.
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_IMPORT_ERROR = pil_error
else:
    PIL_IMPORT_ERROR = None


DEFAULT_DATA_DIR = Path("/nas/nfs/large-model/vince/data/xd-online-las-data/test-v3")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
BG_COLOR = (18, 22, 28)
CARD_COLOR = (31, 36, 46)
TEXT_COLOR = (235, 240, 248)
MUTED_COLOR = (148, 163, 184)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stack original, association BEV, and decision overlay views for each frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", "--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=None)
    parser.add_argument("--association-bev-dir", "--association_bev_dir", type=Path, default=None)
    parser.add_argument("--decision-overlay-dir", "--decision_overlay_dir", type=Path, default=None)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument("--frames", type=str, default="", help="Comma-separated frame ids or a text file with one id per line.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--canvas-width", "--canvas_width", type=int, default=2540)
    parser.add_argument("--margin", type=int, default=28)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--label-height", "--label_height", type=int, default=50)
    parser.add_argument("--font-size", "--font_size", type=int, default=32)
    parser.add_argument("--quality", type=int, default=92)
    return parser.parse_args()


def ensure_dependencies() -> None:
    if PIL_IMPORT_ERROR is not None:
        raise RuntimeError("Pillow is required for visualization.") from PIL_IMPORT_ERROR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_frames_arg(frames_arg: str) -> set[str]:
    if not frames_arg:
        return set()
    maybe_path = Path(frames_arg)
    if maybe_path.exists():
        return {Path(line.strip()).stem for line in maybe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return {Path(item.strip()).stem for item in frames_arg.split(",") if item.strip()}


def resolve_image_path(directory: Path, frame: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = directory / f"{frame}{extension}"
        if candidate.exists():
            return candidate
    return None


def frame_ids(image_dir: Path, frames_arg: str, limit: int) -> list[str]:
    wanted = parse_frames_arg(frames_arg)
    frames = sorted({path.stem for extension in IMAGE_EXTENSIONS for path in image_dir.glob(f"*{extension}")})
    if wanted:
        frames = [frame for frame in frames if frame in wanted]
    if limit > 0:
        frames = frames[:limit]
    return frames


def image_resampling() -> int:
    resampling = getattr(Image, "Resampling", None)
    return resampling.LANCZOS if resampling is not None else Image.LANCZOS


def fit_width(path: Path, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    scale = width / max(1, image.width)
    height = max(1, round(image.height * scale))
    return image.resize((width, height), image_resampling())


def shorten_path(path: Path, max_chars: int) -> str:
    text = str(path)
    if len(text) <= max_chars:
        return text
    return "..." + text[-max(1, max_chars - 3) :]


def draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    path: Path,
    width: int,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.text((x, y), title, fill=TEXT_COLOR, font=font)
    max_chars = max(20, int(width / 13))
    draw.text((x, y + 30), shorten_path(path, max_chars), fill=MUTED_COLOR, font=small_font)


def render_frame(frame: str, paths: dict[str, Path], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_path = output_dir / "frames" / f"{frame}.jpg"
    if output_path.exists() and not args.overwrite:
        return {"frame": frame, "output": str(output_path), "saved": False, "skipped_existing": True}

    font = load_font(args.font_size)
    small_font = load_font(max(14, args.font_size - 12))
    canvas_width = int(args.canvas_width)
    margin = int(args.margin)
    gap = int(args.gap)
    label_height = int(args.label_height)
    title_height = max(56, args.font_size + 24)
    cell_width = (canvas_width - 2 * margin - gap) // 2
    full_width = canvas_width - 2 * margin

    original = fit_width(paths["original"], cell_width)
    association = fit_width(paths["association"], cell_width)
    decision = fit_width(paths["decision"], full_width)

    row1_image_height = max(original.height, association.height)
    row1_height = label_height + row1_image_height
    row2_height = label_height + decision.height
    canvas_height = margin + title_height + row1_height + gap + row2_height + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), f"Dev visualization stack | {frame}", fill=TEXT_COLOR, font=font)

    row1_y = margin + title_height
    row1_items = (
        (margin, "1. Original image", paths["original"], original),
        (margin + cell_width + gap, "2. Association BEV debug", paths["association"], association),
    )
    for x, title, path, image in row1_items:
        draw.rectangle((x - 8, row1_y - 6, x + cell_width + 8, row1_y + row1_height + 8), fill=CARD_COLOR)
        draw_label(draw, x, row1_y, title, path, cell_width, font, small_font)
        canvas.paste(image, (x, row1_y + label_height))

    row2_y = row1_y + row1_height + gap
    draw.rectangle((margin - 8, row2_y - 6, margin + full_width + 8, row2_y + row2_height + 8), fill=CARD_COLOR)
    draw_label(draw, margin, row2_y, "3. Decision overlay", paths["decision"], full_width, font, small_font)
    canvas.paste(decision, (margin, row2_y + label_height))

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=max(1, min(100, int(args.quality))))
    return {
        "frame": frame,
        "output": str(output_path),
        "saved": not args.dry_run,
        "original": str(paths["original"]),
        "association": str(paths["association"]),
        "decision": str(paths["decision"]),
        "size": [canvas_width, canvas_height],
    }


def write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    figures = []
    for record in records:
        if record.get("error") or not record.get("output"):
            continue
        rel = Path(record["output"]).relative_to(output_dir)
        frame = html.escape(str(record.get("frame")))
        figures.append(f"<figure><a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}'></a><figcaption>{frame}</figcaption></figure>")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dev Visualization Stack</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #111827; padding: 10px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Dev Visualization Stack</h1>
  <p>Generated at {html.escape(utc_now())}. Layout: original and association on row one, decision overlay on row two.</p>
  <div class="grid">{''.join(figures)}</div>
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    args.data_dir = args.data_dir.expanduser()
    image_dir = (args.image_dir or args.data_dir / "images").expanduser()
    association_dir = (args.association_bev_dir or args.data_dir / "inference" / "association" / "vis" / "bev_debug").expanduser()
    decision_dir = (args.decision_overlay_dir or args.data_dir / "inference" / "decision" / "vis" / "overlay").expanduser()
    output_dir = (args.output_dir or args.data_dir / "vis").expanduser()

    frames = frame_ids(image_dir, args.frames, args.limit)
    if not frames:
        raise FileNotFoundError(f"No source images found under {image_dir}")

    records = []
    for idx, frame in enumerate(frames, 1):
        paths = {
            "original": resolve_image_path(image_dir, frame),
            "association": resolve_image_path(association_dir, frame),
            "decision": resolve_image_path(decision_dir, frame),
        }
        missing = [name for name, path in paths.items() if path is None]
        if missing:
            records.append({"frame": frame, "error": f"missing inputs: {', '.join(missing)}"})
        else:
            try:
                records.append(render_frame(frame, {key: value for key, value in paths.items() if value is not None}, output_dir, args))
            except Exception as exc:  # noqa: BLE001 - keep batch moving for diagnosis.
                records.append({"frame": frame, "error": f"{type(exc).__name__}: {exc}"})
        if idx % 10 == 0 or idx == len(frames):
            print(f"[info] rendered {idx}/{len(frames)}")

    summary = {
        "schema_version": "dev_visualization_stack/v1",
        "created_at": utc_now(),
        "data_dir": str(args.data_dir),
        "image_dir": str(image_dir),
        "association_bev_dir": str(association_dir),
        "decision_overlay_dir": str(decision_dir),
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "ok_count": sum(1 for item in records if not item.get("error")),
        "failed_count": sum(1 for item in records if item.get("error")),
        "records": records,
    }
    if not args.dry_run:
        dump_json(output_dir / "_summary.json", summary)
        write_index(output_dir, records)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] rendered={summary['ok_count']} failed={summary['failed_count']} output={output_dir}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
