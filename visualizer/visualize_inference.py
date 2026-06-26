#!/usr/bin/env python3
"""Visualize VLM perception inference outputs.

The script reads inference JSON files and writes rendered images under:

    <inference_dir>/vis

It supports:
- prompt-box-* outputs with bbox_2d annotations.
- prompt-classification-boundary-type outputs with one JSON per lane.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as pil_error:
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_IMPORT_ERROR = pil_error
else:
    PIL_IMPORT_ERROR = None


DEFAULT_INFERENCE_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference"
DEFAULT_IMAGE_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/images"
BOX_PROMPT_PREFIX = "prompt-box-"
BOUNDARY_PROMPT_NAME = "prompt-classification-boundary-type"
COMBINED_PROMPT_NAME = "combined"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
PROMPT_COLORS = {
    "prompt-box-road-symbol": (255, 204, 0),
    "prompt-box-road-text": (0, 197, 255),
    "prompt-box-sign": (255, 95, 87),
    "prompt-box-signal": (117, 214, 93),
    BOUNDARY_PROMPT_NAME: (185, 124, 255),
}
LANE_COLORS = (
    (255, 204, 0),
    (0, 197, 255),
    (255, 95, 87),
    (117, 214, 93),
    (185, 124, 255),
    (255, 149, 0),
    (86, 156, 214),
    (255, 112, 166),
)
PROMPT_LABELS = {
    "prompt-box-road-symbol": "symbol",
    "prompt-box-road-text": "text",
    "prompt-box-sign": "sign",
    "prompt-box-signal": "signal",
    BOUNDARY_PROMPT_NAME: "boundary",
}
PROMPT_STYLES = {
    "prompt-box-road-symbol": "solid",
    "prompt-box-road-text": "dashed",
    "prompt-box-sign": "thick",
    "prompt-box-signal": "double",
}


@dataclass
class VisStats:
    rendered: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass
class CombinedGroup:
    image_path: Path
    box_results: list[dict[str, Any]] = field(default_factory=list)
    boundary_results: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize VLM inference JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--inference-dir", "--inference_dir", type=Path, default=Path(DEFAULT_INFERENCE_DIR))
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument("--prompts", nargs="+", default=["all"], help="'all' or selected prompt folder names.")
    parser.add_argument(
        "--mode",
        choices=("combined", "per-prompt", "both"),
        default="combined",
        help="Render all selected prompts on one image, per-prompt images, or both.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit rendered files/groups; <=0 means no limit. In combined mode this limits image groups.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--line-width", "--line_width", type=int, default=4)
    parser.add_argument("--font-size", "--font_size", type=int, default=22)
    parser.add_argument("--draw-empty", "--draw_empty", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-errors", "--draw_errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality", type=int, default=92)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(size: int) -> ImageFont.ImageFont:
    if ImageFont is None:
        raise RuntimeError("Pillow is required for visualization.") from PIL_IMPORT_ERROR
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


def safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def text_bbox(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> tuple[int, int, int, int]:
    try:
        return draw.textbbox(xy, text, font=font)
    except AttributeError:
        width, height = draw.textsize(text, font=font)
        x, y = xy
        return x, y, x + width, y + height


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    padding: int = 5,
) -> tuple[int, int, int, int]:
    x, y = xy
    bbox = text_bbox(draw, (x, y), text, font)
    box = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(box, fill=bg)
    draw.text((x, y), text, font=font, fill=fill)
    return box


def sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.strip("._")
    return clean or "item"


def selected_prompt(prompt_name: str, prompts: Iterable[str]) -> bool:
    prompt_set = set(prompts)
    return "all" in prompt_set or prompt_name in prompt_set


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_model_json(response: Any) -> Any:
    if isinstance(response, (list, dict)):
        return response
    if response is None:
        return None
    text = strip_markdown_fence(str(response))
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates: list[str] = []
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def resolve_image_path(payload: dict[str, Any], image_dir: Path) -> Path | None:
    image_value = payload.get("image_path") or payload.get("image")
    if isinstance(image_value, str) and image_value:
        path = Path(image_value)
        if path.exists():
            return path
        candidate = image_dir / path.name
        if candidate.exists():
            return candidate
    return None


def coord_mode(values: list[float], limit: int) -> str:
    if not values:
        return "normalized_1000"
    if all(0.0 <= value <= 1.0 for value in values):
        return "normalized_1"
    if all(0.0 <= value <= 1000.0 for value in values):
        return "normalized_1000"
    return "pixel"


def points_to_pixels(points: list[list[Any]], size: tuple[int, int]) -> list[tuple[int, int]]:
    width, height = size
    parsed: list[tuple[float, float]] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            parsed.append((float(point[0]), float(point[1])))
    mode = coord_mode([value for xy in parsed for value in xy], max(width, height))
    out: list[tuple[int, int]] = []
    for x, y in parsed:
        if mode == "normalized_1":
            px, py = x * width, y * height
        elif mode == "normalized_1000":
            px, py = x / 1000.0 * width, y / 1000.0 * height
        else:
            px, py = x, y
        out.append((max(0, min(width - 1, int(round(px)))), max(0, min(height - 1, int(round(py))))))
    return out


def box_to_pixels(box: list[Any], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    width, height = size
    values = [float(value) for value in box[:4]]
    mode = coord_mode(values, max(width, height))
    x1, y1, x2, y2 = values
    if mode == "normalized_1":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif mode == "normalized_1000":
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    x_min, x_max = sorted((int(round(x1)), int(round(x2))))
    y_min, y_max = sorted((int(round(y1)), int(round(y2))))
    x_min = max(0, min(width - 1, x_min))
    x_max = max(0, min(width - 1, x_max))
    y_min = max(0, min(height - 1, y_min))
    y_max = max(0, min(height - 1, y_max))
    return x_min, y_min, x_max, y_max


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    radius = max(3, width + 1)
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=1)


def save_image(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=quality)


def prompt_output_path(output_dir: Path, prompt_name: str, image_path: Path) -> Path:
    return output_dir / prompt_name / f"{image_path.stem}.jpg"


def find_image_by_stem(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(path for path in image_dir.iterdir() if path.is_file() and path.stem == stem)
    return matches[0] if matches else None


def extract_box_items(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    status = safe_text(payload.get("status"), "unknown")
    parsed = parse_model_json(payload.get("response"))
    boxes: list[dict[str, Any]] = []
    parse_note = ""
    if status == "ok":
        if isinstance(parsed, list):
            boxes = [item for item in parsed if isinstance(item, dict)]
        elif isinstance(parsed, dict):
            for key in ("boxes", "objects", "detections", "result"):
                value = parsed.get(key)
                if isinstance(value, list):
                    boxes = [item for item in value if isinstance(item, dict)]
                    break
            if not boxes and any(key in parsed for key in ("bbox_2d", "bbox", "box")):
                boxes = [parsed]
        elif payload.get("response") not in ("", None):
            parse_note = "parse_failed"
    else:
        parse_note = safe_text(payload.get("error_type"), "error")
    return boxes, parse_note


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    dash: int = 16,
    gap: int = 10,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if x1 == x2:
        direction = 1 if y2 >= y1 else -1
        y = y1
        while (direction > 0 and y <= y2) or (direction < 0 and y >= y2):
            y_end = y + direction * dash
            if direction > 0:
                y_end = min(y_end, y2)
            else:
                y_end = max(y_end, y2)
            draw.line((x1, y, x2, y_end), fill=color, width=width)
            y += direction * (dash + gap)
        return
    direction = 1 if x2 >= x1 else -1
    x = x1
    while (direction > 0 and x <= x2) or (direction < 0 and x >= x2):
        x_end = x + direction * dash
        if direction > 0:
            x_end = min(x_end, x2)
        else:
            x_end = max(x_end, x2)
        draw.line((x, y1, x_end, y2), fill=color, width=width)
        x += direction * (dash + gap)


def draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = box
    draw_dashed_line(draw, (x1, y1), (x2, y1), color, width)
    draw_dashed_line(draw, (x2, y1), (x2, y2), color, width)
    draw_dashed_line(draw, (x2, y2), (x1, y2), color, width)
    draw_dashed_line(draw, (x1, y2), (x1, y1), color, width)


def draw_box_outline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    prompt_name: str,
    color: tuple[int, int, int],
    width: int,
) -> None:
    style = PROMPT_STYLES.get(prompt_name, "solid")
    if style == "dashed":
        draw_dashed_rectangle(draw, box, color, max(2, width))
    elif style == "double":
        draw.rectangle(box, outline=color, width=max(2, width))
        x1, y1, x2, y2 = box
        inset = max(4, width + 2)
        if x2 - x1 > inset * 2 and y2 - y1 > inset * 2:
            draw.rectangle((x1 + inset, y1 + inset, x2 - inset, y2 - inset), outline=color, width=max(1, width // 2))
    else:
        draw.rectangle(box, outline=color, width=max(2, width + 2 if style == "thick" else width))


def box_label(prompt_name: str, item: dict[str, Any], index: int) -> str:
    prefix = PROMPT_LABELS.get(prompt_name, prompt_name.replace(BOX_PROMPT_PREFIX, ""))
    label = safe_text(item.get("label") or item.get("class") or item.get("category"), str(index))
    return f"{prefix}:{label}"


def draw_box_item(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    prompt_name: str,
    item: dict[str, Any],
    index: int,
    args: argparse.Namespace,
    font: ImageFont.ImageFont,
) -> None:
    color = PROMPT_COLORS.get(prompt_name, (255, 204, 0))
    draw_box_outline(draw, box, prompt_name, color, max(2, args.line_width))
    draw_text_box(draw, (box[0], max(0, box[1] - args.font_size - 10)), box_label(prompt_name, item, index), font, bg=color, fill=(0, 0, 0))


def visualize_box_result(
    result_path: Path,
    output_dir: Path,
    image_dir: Path,
    args: argparse.Namespace,
    font: ImageFont.ImageFont,
) -> tuple[str, Path | None]:
    payload = load_json(result_path)
    prompt_name = safe_text(payload.get("prompt_name"), result_path.parent.name)
    image_path = resolve_image_path(payload, image_dir)
    if image_path is None:
        return "missing_image", None

    out_path = prompt_output_path(output_dir, prompt_name, image_path)
    if out_path.exists() and not args.overwrite:
        return "skipped", out_path

    if args.dry_run:
        return "rendered", out_path

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    status = safe_text(payload.get("status"), "unknown")
    boxes, parse_note = extract_box_items(payload)

    if status != "ok":
        if not args.draw_errors:
            return "skipped", out_path
    if not boxes and status == "ok" and not args.draw_empty:
        return "skipped", out_path

    for index, item in enumerate(boxes, 1):
        raw_box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        box = box_to_pixels(raw_box, image.size) if raw_box is not None else None
        if box is None:
            continue
        draw_box_item(draw, box, prompt_name, item, index, args, font)

    header = f"{prompt_name} | {status} | boxes={len(boxes)}"
    if parse_note:
        header += f" | {parse_note}"
    draw_text_box(draw, (12, 12), header, font, bg=(0, 0, 0), fill=(255, 255, 255))
    save_image(image, out_path, args.quality)
    return "rendered", out_path


def box_prompt_dirs(inference_dir: Path, prompts: Iterable[str]) -> list[Path]:
    dirs: list[Path] = []
    for path in sorted(inference_dir.iterdir()):
        if not path.is_dir() or path.name == "vis" or path.name.startswith("_"):
            continue
        if path.name.startswith(BOX_PROMPT_PREFIX) and selected_prompt(path.name, prompts):
            dirs.append(path)
    return dirs


def iter_limited(paths: list[Path], limit: int) -> list[Path]:
    if limit > 0:
        return paths[:limit]
    return paths


def boundary_groups(boundary_dir: Path, limit: int) -> list[tuple[str, list[Path]]]:
    groups: list[tuple[str, list[Path]]] = []
    if not boundary_dir.exists():
        return groups
    for image_dir in sorted(path for path in boundary_dir.iterdir() if path.is_dir()):
        json_paths = sorted(path for path in image_dir.glob("*.json") if path.is_file())
        if json_paths:
            groups.append((image_dir.name, json_paths))
    if limit > 0:
        return groups[:limit]
    return groups


def boundary_label(payload: dict[str, Any]) -> str:
    lane_id = safe_text(payload.get("lane_id"), "?")
    lane_index = payload.get("lane_index")
    prefix = f"{lane_index}:{lane_id}" if lane_index is not None else f"lane:{lane_id}"
    if payload.get("status") != "ok":
        return f"{prefix} ERROR {safe_text(payload.get('error_type'), '')}".strip()
    parsed = parse_model_json(payload.get("response"))
    if isinstance(parsed, dict):
        left = safe_text(parsed.get("left_boundary"), "?")
        right = safe_text(parsed.get("right_boundary"), "?")
        return f"{prefix} L={left} R={right}"
    return f"{prefix} parse_failed"


def boundary_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    try:
        idx = int(item.get("lane_index", 0))
    except (TypeError, ValueError):
        idx = 0
    return idx, safe_text(item.get("lane_id"), "")


def visualize_boundary_group(
    image_key: str,
    result_paths: list[Path],
    output_dir: Path,
    image_dir: Path,
    args: argparse.Namespace,
    font: ImageFont.ImageFont,
) -> tuple[str, Path | None]:
    payloads: list[dict[str, Any]] = []
    image_path: Path | None = None
    for path in result_paths:
        payload = load_json(path)
        payload["_result_path"] = str(path)
        payloads.append(payload)
        if image_path is None:
            image_path = resolve_image_path(payload, image_dir)

    if image_path is None:
        return "missing_image", None
    out_path = output_dir / BOUNDARY_PROMPT_NAME / f"{sanitize_name(image_path.stem)}.jpg"
    if out_path.exists() and not args.overwrite:
        return "skipped", out_path
    if args.dry_run:
        return "rendered", out_path

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    legend_rows: list[tuple[tuple[int, int, int], str]] = []

    for index, payload in enumerate(sorted(payloads, key=boundary_sort_key)):
        status = safe_text(payload.get("status"), "unknown")
        if status != "ok" and not args.draw_errors:
            continue
        color = LANE_COLORS[index % len(LANE_COLORS)] if status == "ok" else (255, 64, 64)
        points = payload.get("center_line_points") or payload.get("center_line_points_raw") or []
        pixel_points = points_to_pixels(points, image.size)
        draw_polyline(draw, pixel_points, color, max(2, args.line_width))
        label = boundary_label(payload)
        legend_rows.append((color, label))
        if pixel_points:
            x, y = pixel_points[0]
            draw_text_box(draw, (min(image.size[0] - 80, x + 8), max(0, y - args.font_size - 8)), label.split(" ", 1)[0], font, bg=color, fill=(0, 0, 0))

    header = f"{BOUNDARY_PROMPT_NAME} | lanes={len(payloads)}"
    draw_text_box(draw, (12, 12), header, font, bg=(0, 0, 0), fill=(255, 255, 255))
    y = 52
    max_rows = max(1, (image.size[1] - y - 12) // (args.font_size + 12))
    for color, label in legend_rows[:max_rows]:
        draw_text_box(draw, (12, y), label, font, bg=color, fill=(0, 0, 0))
        y += args.font_size + 12
    if len(legend_rows) > max_rows:
        draw_text_box(draw, (12, y), f"+{len(legend_rows) - max_rows} more", font, bg=(0, 0, 0), fill=(255, 255, 255))

    save_image(image, out_path, args.quality)
    return "rendered", out_path


def result_image_stem(result_path: Path) -> str:
    return Path(result_path.stem).stem


def combined_group_for(groups: dict[str, CombinedGroup], image_path: Path) -> CombinedGroup:
    key = image_path.stem
    if key not in groups:
        groups[key] = CombinedGroup(image_path=image_path)
    return groups[key]


def collect_combined_groups(
    inference_dir: Path,
    image_dir: Path,
    prompts: Iterable[str],
    limit: int,
) -> list[tuple[str, CombinedGroup]]:
    groups: dict[str, CombinedGroup] = {}

    for prompt_dir in box_prompt_dirs(inference_dir, prompts):
        prompt_name = prompt_dir.name
        for result_path in sorted(path for path in prompt_dir.rglob("*.json") if path.is_file()):
            payload = load_json(result_path)
            image_path = resolve_image_path(payload, image_dir)
            if image_path is None:
                image_path = find_image_by_stem(image_dir, result_image_stem(result_path))
            if image_path is None:
                continue
            group = combined_group_for(groups, image_path)
            group.box_results.append({"prompt_name": prompt_name, "payload": payload, "result_path": str(result_path)})

    if selected_prompt(BOUNDARY_PROMPT_NAME, prompts):
        boundary_dir = inference_dir / BOUNDARY_PROMPT_NAME
        for image_key, result_paths in boundary_groups(boundary_dir, 0):
            payloads: list[dict[str, Any]] = []
            image_path: Path | None = None
            for result_path in result_paths:
                payload = load_json(result_path)
                payload["_result_path"] = str(result_path)
                payloads.append(payload)
                if image_path is None:
                    image_path = resolve_image_path(payload, image_dir)
            if image_path is None:
                image_path = find_image_by_stem(image_dir, image_key)
            if image_path is None:
                continue
            group = combined_group_for(groups, image_path)
            group.boundary_results.extend(payloads)

    items = sorted(groups.items(), key=lambda item: item[0])
    if limit > 0:
        return items[:limit]
    return items


def draw_combined_legend(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
    args: argparse.Namespace,
    box_totals: dict[str, int],
    boundary_rows: list[tuple[tuple[int, int, int], str]],
    issue_count: int,
) -> None:
    total_boxes = sum(box_totals.values())
    header = f"combined | boxes={total_boxes} | lanes={len(boundary_rows)}"
    if issue_count:
        header += f" | issues={issue_count}"
    draw_text_box(draw, (12, 12), header, font, bg=(0, 0, 0), fill=(255, 255, 255))

    y = 52
    for prompt_name in sorted(box_totals):
        color = PROMPT_COLORS.get(prompt_name, (255, 204, 0))
        label = PROMPT_LABELS.get(prompt_name, prompt_name)
        style = PROMPT_STYLES.get(prompt_name, "solid")
        draw_text_box(draw, (12, y), f"{label} {style} boxes={box_totals[prompt_name]}", font, bg=color, fill=(0, 0, 0))
        y += args.font_size + 12

    if boundary_rows:
        draw_text_box(draw, (12, y), "boundary lanes", font, bg=PROMPT_COLORS[BOUNDARY_PROMPT_NAME], fill=(0, 0, 0))
        y += args.font_size + 12
    max_rows = max(0, (image_size[1] - y - 12) // (args.font_size + 12))
    for color, label in boundary_rows[:max_rows]:
        draw_text_box(draw, (12, y), label, font, bg=color, fill=(0, 0, 0))
        y += args.font_size + 12
    if len(boundary_rows) > max_rows:
        draw_text_box(draw, (12, y), f"+{len(boundary_rows) - max_rows} more lanes", font, bg=(0, 0, 0), fill=(255, 255, 255))


def visualize_combined_group(
    image_key: str,
    group: CombinedGroup,
    output_dir: Path,
    args: argparse.Namespace,
    font: ImageFont.ImageFont,
) -> tuple[str, Path | None]:
    out_path = output_dir / COMBINED_PROMPT_NAME / f"{sanitize_name(group.image_path.stem)}.jpg"
    if out_path.exists() and not args.overwrite:
        return "skipped", out_path
    if args.dry_run:
        return "rendered", out_path

    image = Image.open(group.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    box_totals: dict[str, int] = {}
    issue_count = 0

    for record in group.box_results:
        prompt_name = safe_text(record.get("prompt_name"), "")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            issue_count += 1
            continue
        box_totals.setdefault(prompt_name, 0)
        status = safe_text(payload.get("status"), "unknown")
        boxes, parse_note = extract_box_items(payload)
        if status != "ok":
            issue_count += 1
            if not args.draw_errors:
                continue
        elif parse_note:
            issue_count += 1
        if not boxes and status == "ok" and not args.draw_empty:
            continue
        for index, item in enumerate(boxes, 1):
            raw_box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            box = box_to_pixels(raw_box, image.size) if raw_box is not None else None
            if box is None:
                issue_count += 1
                continue
            draw_box_item(draw, box, prompt_name, item, index, args, font)
            box_totals[prompt_name] += 1

    boundary_rows: list[tuple[tuple[int, int, int], str]] = []
    for index, payload in enumerate(sorted(group.boundary_results, key=boundary_sort_key)):
        status = safe_text(payload.get("status"), "unknown")
        if status != "ok" and not args.draw_errors:
            continue
        color = LANE_COLORS[index % len(LANE_COLORS)] if status == "ok" else (255, 64, 64)
        points = payload.get("center_line_points") or payload.get("center_line_points_raw") or []
        pixel_points = points_to_pixels(points, image.size)
        if pixel_points:
            draw_polyline(draw, pixel_points, color, max(2, args.line_width))
            x, y = pixel_points[0]
            lane_tag = boundary_label(payload).split(" ", 1)[0]
            draw_text_box(draw, (min(image.size[0] - 80, x + 8), max(0, y - args.font_size - 8)), lane_tag, font, bg=color, fill=(0, 0, 0))
        elif status == "ok":
            issue_count += 1
        if status != "ok":
            issue_count += 1
        boundary_rows.append((color, boundary_label(payload)))

    if not group.box_results and not group.boundary_results:
        draw_text_box(draw, (12, 12), f"{image_key} | no selected inference", font, bg=(0, 0, 0), fill=(255, 255, 255))
    else:
        draw_combined_legend(draw, image.size, font, args, box_totals, boundary_rows, issue_count)

    save_image(image, out_path, args.quality)
    return "rendered", out_path


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_index_html(output_dir: Path, summary: dict[str, Any]) -> None:
    rows: list[str] = []
    for prompt_name, prompt_summary in summary.get("prompts", {}).items():
        rows.append(f"<h2>{html.escape(prompt_name)}</h2>")
        rows.append("<ul>")
        for item in prompt_summary.get("outputs", []):
            rel = Path(item).relative_to(output_dir)
            rows.append(f'<li><a href="{html.escape(str(rel))}">{html.escape(str(rel))}</a></li>')
        rows.append("</ul>")
    body = "\n".join(rows)
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VLM Inference Visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; line-height: 1.4; }}
    a {{ color: #065fd4; }}
  </style>
</head>
<body>
  <h1>VLM Inference Visualization</h1>
  <p>Created at {html.escape(summary.get("created_at", ""))}</p>
  {body}
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def update_stats(stats: VisStats, status: str) -> None:
    if status == "rendered":
        stats.rendered += 1
    elif status == "skipped":
        stats.skipped += 1
    else:
        stats.failed += 1


def main() -> int:
    args = parse_args()
    if PIL_IMPORT_ERROR is not None:
        raise RuntimeError("Missing dependency: Pillow. Install pillow or run on the remote environment.") from PIL_IMPORT_ERROR
    inference_dir = args.inference_dir.expanduser()
    image_dir = args.image_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else inference_dir / "vis"
    font = load_font(args.font_size)
    if not inference_dir.exists():
        raise FileNotFoundError(f"inference dir not found: {inference_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "inference_dir": str(inference_dir),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "prompts": {},
    }

    exit_code = 0

    if args.mode in ("combined", "both"):
        stats = VisStats()
        outputs: list[str] = []
        try:
            combined_groups = collect_combined_groups(inference_dir, image_dir, args.prompts, args.limit)
        except Exception as exc:  # noqa: BLE001
            combined_groups = []
            stats.failed += 1
            exit_code = 2
            print(f"ERROR {COMBINED_PROMPT_NAME}: {type(exc).__name__}: {exc}")
        for image_key, group in combined_groups:
            try:
                status, out_path = visualize_combined_group(image_key, group, output_dir, args, font)
                update_stats(stats, status)
                if out_path is not None:
                    outputs.append(str(out_path))
            except Exception as exc:  # noqa: BLE001 - summarize visualization failures and continue.
                stats.failed += 1
                exit_code = 2
                print(f"ERROR {COMBINED_PROMPT_NAME}/{image_key}: {type(exc).__name__}: {exc}")
        summary["prompts"][COMBINED_PROMPT_NAME] = {**stats.__dict__, "outputs": outputs}
        print(f"{COMBINED_PROMPT_NAME}: rendered={stats.rendered} skipped={stats.skipped} failed={stats.failed}")

    if args.mode in ("per-prompt", "both"):
        for prompt_dir in box_prompt_dirs(inference_dir, args.prompts):
            prompt_name = prompt_dir.name
            stats = VisStats()
            outputs = []
            result_paths = sorted(path for path in prompt_dir.rglob("*.json") if path.is_file())
            for result_path in iter_limited(result_paths, args.limit):
                try:
                    status, out_path = visualize_box_result(result_path, output_dir, image_dir, args, font)
                    update_stats(stats, status)
                    if out_path is not None:
                        outputs.append(str(out_path))
                except Exception as exc:  # noqa: BLE001 - summarize visualization failures and continue.
                    stats.failed += 1
                    exit_code = 2
                    print(f"ERROR {result_path}: {type(exc).__name__}: {exc}")
            summary["prompts"][prompt_name] = {**stats.__dict__, "outputs": outputs}
            print(f"{prompt_name}: rendered={stats.rendered} skipped={stats.skipped} failed={stats.failed}")

        if selected_prompt(BOUNDARY_PROMPT_NAME, args.prompts):
            boundary_dir = inference_dir / BOUNDARY_PROMPT_NAME
            stats = VisStats()
            outputs = []
            for image_key, result_paths in boundary_groups(boundary_dir, args.limit):
                try:
                    status, out_path = visualize_boundary_group(image_key, result_paths, output_dir, image_dir, args, font)
                    update_stats(stats, status)
                    if out_path is not None:
                        outputs.append(str(out_path))
                except Exception as exc:  # noqa: BLE001
                    stats.failed += 1
                    exit_code = 2
                    print(f"ERROR {boundary_dir / image_key}: {type(exc).__name__}: {exc}")
            summary["prompts"][BOUNDARY_PROMPT_NAME] = {**stats.__dict__, "outputs": outputs}
            print(f"{BOUNDARY_PROMPT_NAME}: rendered={stats.rendered} skipped={stats.skipped} failed={stats.failed}")

    if not args.dry_run:
        write_summary(output_dir / "_summary.json", summary)
        write_index_html(output_dir, summary)
    else:
        print(f"dry_run_output_dir={output_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
