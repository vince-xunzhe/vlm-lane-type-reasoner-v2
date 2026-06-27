#!/usr/bin/env python3
"""Visualize rule-based lane type decisions."""

from __future__ import annotations

import argparse
import html
import json
import math
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


DEFAULT_DECISION_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference/decision"
LANE_TYPE_COLORS = {
    "tidal": (239, 68, 68),
    "variable": (168, 85, 247),
    "bus": (37, 99, 235),
    "bicycle": (22, 163, 74),
    "normal": (100, 116, 139),
}
LANE_TYPE_LABELS = {
    "tidal": "tidal lane",
    "variable": "variable lane",
    "bus": "bus lane",
    "bicycle": "bicycle lane",
    "normal": "normal lane",
}
SPECIAL_SCORE_KEYS = ("tidal", "variable", "bus", "bicycle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render lane type decision JSON files as image overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--decision-dir", "--decision_dir", type=Path, default=Path(DEFAULT_DECISION_DIR))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument("--frames", type=str, default="", help="Comma-separated frame ids or a text file with one id per line.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--line-width", "--line_width", type=int, default=8)
    parser.add_argument("--font-size", "--font_size", type=int, default=22)
    parser.add_argument("--panel-width", "--panel_width", type=int, default=620)
    parser.add_argument("--quality", type=int, default=92)
    return parser.parse_args()


def ensure_dependencies() -> None:
    if PIL_IMPORT_ERROR is not None:
        raise RuntimeError("Pillow is required for visualization.") from PIL_IMPORT_ERROR


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
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


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox(xy, text, font=font)
    except AttributeError:
        width, height = draw.textsize(text, font=font)
        x, y = xy
        return x, y, x + width, y + height


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    bg: tuple[int, int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
    padding: int = 5,
) -> tuple[int, int, int, int]:
    x, y = xy
    bbox = text_bbox(draw, (x, y), text, font)
    box = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(box, fill=bg)
    draw.text((x, y), text, font=font, fill=fill)
    return box


def draw_text_box_clamped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
    *,
    bg: tuple[int, int, int],
    fill: tuple[int, int, int] = (255, 255, 255),
    padding: int = 5,
) -> tuple[int, int, int, int]:
    width, height = image_size
    bbox = text_bbox(draw, xy, text, font)
    box_width = bbox[2] - bbox[0] + padding * 2
    box_height = bbox[3] - bbox[1] + padding * 2
    x = int(clamp(xy[0], padding, max(padding, width - box_width - padding)))
    y = int(clamp(xy[1], padding, max(padding, height - box_height - padding)))
    return draw_text_box(draw, (x, y), text, font, bg=bg, fill=fill, padding=padding)


def parse_frames_arg(frames_arg: str) -> set[str]:
    if not frames_arg:
        return set()
    maybe_path = Path(frames_arg)
    if maybe_path.exists():
        return {Path(line.strip()).stem for line in maybe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return {Path(item.strip()).stem for item in frames_arg.split(",") if item.strip()}


def decision_paths(decision_dir: Path, frames_arg: str, limit: int) -> list[Path]:
    wanted = parse_frames_arg(frames_arg)
    paths = sorted((decision_dir / "frames").glob("*.json"))
    if wanted:
        paths = [path for path in paths if path.stem in wanted]
    if limit > 0:
        paths = paths[:limit]
    return paths


def as_points(value: Any) -> list[tuple[int, int]]:
    out = []
    if not isinstance(value, list):
        return out
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            out.append((int(round(float(point[0]))), int(round(float(point[1])))))
        except (TypeError, ValueError):
            continue
    return out


def bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return tuple(int(round(float(item))) for item in value[:4])  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def lane_color(lane_type: str) -> tuple[int, int, int]:
    return LANE_TYPE_COLORS.get(lane_type, LANE_TYPE_COLORS["normal"])


def score_summary(decision: dict[str, Any]) -> str:
    scores = decision.get("scores") or {}
    parts = []
    for key in SPECIAL_SCORE_KEYS:
        value = scores.get(key)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score > 0:
            parts.append(f"{key}={score:.2f}")
    return ", ".join(parts) if parts else "no special score"


def is_special_lane(lane_decision: dict[str, Any]) -> bool:
    return str(lane_decision.get("lane_type") or "normal") != "normal"


def has_special_lane(decision: dict[str, Any]) -> bool:
    return any(is_special_lane(item) for item in decision.get("lane_decisions") or [])


def evidence_summary(decision: dict[str, Any], max_items: int = 3) -> str:
    accepted = [item for item in decision.get("evidence") or [] if item.get("accepted")]
    if not accepted:
        return str(decision.get("decision_reason") or "no evidence")
    rules = [str(item.get("rule_id")) for item in accepted[:max_items]]
    if len(accepted) > max_items:
        rules.append(f"+{len(accepted) - max_items}")
    return ", ".join(rules)


def draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: tuple[int, int, int], width: int) -> None:
    if len(points) >= 2:
        draw.line(points, fill=(0, 0, 0), width=width + 4, joint="curve")
        draw.line(points, fill=(255, 255, 255), width=width + 1, joint="curve")
        draw.line(points, fill=color, width=width, joint="curve")
    radius = max(3, width // 2)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(15, 23, 42), width=2)


def load_association(decision: dict[str, Any]) -> dict[str, Any]:
    path_value = decision.get("source_association_json")
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return load_json(path)


def lane_geometry_by_id(association: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(lane.get("lane_id")): lane for lane in association.get("lanes") or []}


def objects_by_id(association: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(obj.get("object_id")): obj for obj in association.get("objects") or []}


def line_height(font: ImageFont.ImageFont) -> int:
    bbox = text_bbox(ImageDraw.Draw(Image.new("RGB", (10, 10))), (0, 0), "Ag", font)
    return max(1, bbox[3] - bbox[1] + 8)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    panel_xy: tuple[int, int],
    panel_size: tuple[int, int],
    decision: dict[str, Any],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    x0, y0 = panel_xy
    width, height = panel_size
    draw.rectangle((x0, y0, x0 + width, y0 + height), fill=(15, 23, 42))
    y = y0 + 18
    draw.text((x0 + 18, y), f"Frame {decision.get('frame')}", fill=(255, 255, 255), font=font)
    y += line_height(font) + 4
    draw.text((x0 + 18, y), "Lane type decision", fill=(203, 213, 225), font=small_font)
    y += line_height(small_font) + 12

    legend_items = [("tidal", "tidal"), ("variable", "variable"), ("bus", "bus"), ("bicycle", "bicycle"), ("normal", "normal")]
    lx = x0 + 18
    for lane_type, text in legend_items:
        color = lane_color(lane_type)
        draw.rectangle((lx, y, lx + 18, y + 18), fill=color)
        draw.text((lx + 24, y - 3), text, fill=(226, 232, 240), font=small_font)
        lx += 112
    y += 36

    for lane in decision.get("lane_decisions") or []:
        lane_type = str(lane.get("lane_type") or "normal")
        color = lane_color(lane_type)
        top = y
        draw.rectangle((x0 + 14, y, x0 + width - 14, y + 112), outline=color, width=3)
        draw.rectangle((x0 + 14, y, x0 + 28, y + 112), fill=color)
        title = f"L{lane.get('lane_id')} {LANE_TYPE_LABELS.get(lane_type, lane_type)}"
        draw.text((x0 + 38, y + 8), title, fill=(255, 255, 255), font=font)
        draw.text((x0 + 38, y + 38), score_summary(lane), fill=(226, 232, 240), font=small_font)
        boundary = lane.get("boundary_type") or {}
        boundary_text = f"left={boundary.get('left')} right={boundary.get('right')}"
        draw.text((x0 + 38, y + 62), boundary_text, fill=(203, 213, 225), font=small_font)
        evidence_text = evidence_summary(lane)
        draw.text((x0 + 38, y + 86), evidence_text[:74], fill=(148, 163, 184), font=small_font)
        y = top + 124
        if y > y0 + height - 126:
            draw.text((x0 + 18, y), "...", fill=(203, 213, 225), font=font)
            break


def draw_accepted_objects(
    draw: ImageDraw.ImageDraw,
    lane_decision: dict[str, Any],
    object_lookup: dict[str, dict[str, Any]],
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
) -> None:
    for item in lane_decision.get("accepted_objects") or []:
        object_id = str(item.get("object_id") or "")
        obj = object_lookup.get(object_id)
        if not obj:
            continue
        box = bbox(obj.get("bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle(box, outline=color, width=5)
        label = str(item.get("label_name") or object_id)
        draw_text_box_clamped(draw, (x1, max(0, y1 - 34)), label, font, image_size, bg=color, fill=(0, 0, 0), padding=4)


def render_frame(decision_path: Path, output_dir: Path, args: argparse.Namespace, font: ImageFont.ImageFont, small_font: ImageFont.ImageFont) -> dict[str, Any]:
    decision = load_json(decision_path)
    association = load_association(decision)
    lane_lookup = lane_geometry_by_id(association)
    object_lookup = objects_by_id(association)
    image_path = Path(str(decision.get("image_path") or association.get("image_path") or ""))
    if not image_path.exists():
        raise FileNotFoundError(f"missing image for {decision_path.name}: {image_path}")

    image = Image.open(image_path).convert("RGB")
    canvas = Image.new("RGB", (image.width + args.panel_width, image.height), (15, 23, 42))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    image_size = image.size

    for lane_decision in decision.get("lane_decisions") or []:
        lane_id = str(lane_decision.get("lane_id"))
        lane_type = str(lane_decision.get("lane_type") or "normal")
        color = lane_color(lane_type)
        lane = lane_lookup.get(lane_id, {})
        polyline = as_points(lane.get("points"))
        if not polyline:
            continue
        draw_polyline(draw, polyline, color, args.line_width)
        label = f"L{lane_id} {LANE_TYPE_LABELS.get(lane_type, lane_type)}"
        if score_summary(lane_decision) != "no special score":
            label += f" | {score_summary(lane_decision)}"
        label_xy = polyline[0]
        draw_text_box_clamped(draw, label_xy, label, font, image_size, bg=color, fill=(0, 0, 0), padding=5)
        draw_accepted_objects(draw, lane_decision, object_lookup, color, small_font, image_size)

    header = f"{decision.get('frame')} | lanes={len(decision.get('lane_decisions') or [])}"
    draw_text_box_clamped(draw, (10, 10), header, font, image_size, bg=(15, 23, 42), fill=(255, 255, 255), padding=6)
    draw_panel(draw, (image.width, 0), (args.panel_width, image.height), decision, font, small_font)

    frame = str(decision.get("frame"))
    output_path = output_dir / "overlay" / f"{frame}.jpg"
    major_output_path = output_dir / "major" / f"{frame}.jpg"
    special_lane = has_special_lane(decision)
    if args.dry_run:
        return {
            "frame": frame,
            "output": str(output_path),
            "major_output": str(major_output_path) if special_lane else None,
            "has_special_lane": special_lane,
            "dry_run": True,
        }

    saved = False
    major_saved = False
    if not output_path.exists() or args.overwrite:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=int(clamp(args.quality, 1, 100)))
        saved = True
    if special_lane and (not major_output_path.exists() or args.overwrite):
        major_output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(major_output_path, quality=int(clamp(args.quality, 1, 100)))
        major_saved = True

    record: dict[str, Any] = {
        "frame": frame,
        "output": str(output_path),
        "saved": saved,
        "has_special_lane": special_lane,
    }
    if output_path.exists() and not saved:
        record["reason"] = "exists"
    if special_lane:
        record["major_output"] = str(major_output_path)
        record["major_saved"] = major_saved
    return record


def write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    figures = []
    for record in records:
        output = record.get("output")
        if not output:
            continue
        rel = Path(output).relative_to(output_dir)
        frame = html.escape(str(record.get("frame")))
        figures.append(f"<figure><a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}'></a><figcaption>{frame}</figcaption></figure>")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lane Decision Visualization</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #111827; padding: 10px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Lane Decision Visualization</h1>
  <p>Generated at {html.escape(utc_now())}.</p>
  <div class="grid">{''.join(figures)}</div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_major_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    major_dir = output_dir / "major"
    figures = []
    for record in records:
        output = record.get("major_output")
        if not output:
            continue
        rel = Path(output).relative_to(major_dir)
        frame = html.escape(str(record.get("frame")))
        figures.append(f"<figure><a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}'></a><figcaption>{frame}</figcaption></figure>")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Major Lane Decision Visualization</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #111827; padding: 10px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Major Lane Decision Visualization</h1>
  <p>Generated at {html.escape(utc_now())}. Frames here contain at least one non-normal lane decision.</p>
  <div class="grid">{''.join(figures)}</div>
</body>
</html>
"""
    major_dir.mkdir(parents=True, exist_ok=True)
    (major_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    args.decision_dir = args.decision_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else args.decision_dir / "vis"
    paths = decision_paths(args.decision_dir, args.frames, args.limit)
    if not paths:
        raise FileNotFoundError(f"No decision frame JSONs found under {args.decision_dir / 'frames'}")
    print(f"[info] frames={len(paths)} output={output_dir}")
    font = load_font(args.font_size)
    small_font = load_font(max(14, args.font_size - 6))
    records = []
    for idx, path in enumerate(paths, 1):
        try:
            records.append(render_frame(path, output_dir, args, font, small_font))
        except Exception as exc:  # noqa: BLE001 - keep batch moving for diagnosis.
            print(f"[error] {path.name}: {type(exc).__name__}: {exc}")
            records.append({"frame": path.stem, "error": f"{type(exc).__name__}: {exc}", "saved": False})
        if idx % 10 == 0 or idx == len(paths):
            print(f"[info] rendered {idx}/{len(paths)}")
    summary = {
        "schema_version": "lane_decision_visualization/v1",
        "created_at": utc_now(),
        "decision_dir": str(args.decision_dir),
        "output_dir": str(output_dir),
        "frame_count": len(paths),
        "ok_count": sum(1 for item in records if not item.get("error")),
        "major_count": sum(1 for item in records if item.get("has_special_lane") and not item.get("error")),
        "records": records,
    }
    if not args.dry_run:
        dump_json(output_dir / "_summary.json", summary)
        dump_json(output_dir / "major" / "_summary.json", {**summary, "records": [item for item in records if item.get("has_special_lane")]})
        write_index(output_dir, records)
        write_major_index(output_dir, records)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] rendered={summary['ok_count']} output={output_dir}")
    return 0 if summary["ok_count"] == len(paths) else 1


if __name__ == "__main__":
    raise SystemExit(main())
