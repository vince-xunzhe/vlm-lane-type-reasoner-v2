#!/usr/bin/env python3
"""Visualize 3D lane-association probe outputs.

The expected input is produced by:
  associator/associate_elements_to_lanes.py

It renders four synchronized views per frame:
  1. association_overlay: original image + SAM road mask + lanes + objects
  2. depth_overlay: Depth-Anything depth heatmap + association evidence
  3. confidence_overlay: Depth-Anything confidence heatmap + association evidence
  4. bev_debug: camera-space BEV lane bands, lane samples, and object anchors
"""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as np_error:  # pragma: no cover - remote dependency.
    np = None
    NUMPY_IMPORT_ERROR = np_error
else:
    NUMPY_IMPORT_ERROR = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as pil_error:  # pragma: no cover - remote dependency.
    Image = None
    ImageDraw = None
    ImageFont = None
    PIL_IMPORT_ERROR = pil_error
else:
    PIL_IMPORT_ERROR = None


DEFAULT_ASSOCIATION_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference/association"
VIEW_NAMES = ("association_overlay", "depth_overlay", "confidence_overlay", "bev_debug")
LANE_COLORS = (
    (56, 189, 248),
    (34, 197, 94),
    (251, 191, 36),
    (244, 114, 182),
    (168, 85, 247),
    (45, 212, 191),
    (249, 115, 22),
    (129, 140, 248),
)
OBJECT_COLORS = {
    "sign_signal": (255, 95, 87),
    "road_marking": (255, 204, 0),
    "unknown": (148, 163, 184),
}
PROMPT_COLORS = {
    "prompt-box-road-symbol": (255, 204, 0),
    "prompt-box-road-text": (0, 197, 255),
    "prompt-box-sign": (255, 95, 87),
    "prompt-box-signal": (117, 214, 93),
}
SAM_MASK_COLOR = (56, 189, 248)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 3D lane association outputs as image and BEV debug overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--association-dir", "--association_dir", type=Path, default=Path(DEFAULT_ASSOCIATION_DIR))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit rendered frames; <=0 means no limit.")
    parser.add_argument("--frames", type=str, default="", help="Comma-separated frame ids or a text file of frame ids.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--top-k", "--top_k", type=int, default=3, help="Number of association scores to show per object.")
    parser.add_argument("--line-width", "--line_width", type=int, default=5)
    parser.add_argument("--font-size", "--font_size", type=int, default=22)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--sam-alpha", "--sam_alpha", type=int, default=48)
    parser.add_argument("--heat-alpha", "--heat_alpha", type=int, default=145)
    parser.add_argument("--bev-width", "--bev_width", type=int, default=1400)
    parser.add_argument("--bev-height", "--bev_height", type=int, default=920)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def ensure_dependencies() -> None:
    if NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError("Missing dependency: numpy. Run in the remote environment or install numpy.") from NUMPY_IMPORT_ERROR
    if PIL_IMPORT_ERROR is not None:
        raise RuntimeError("Missing dependency: Pillow. Run in the remote environment or install pillow.") from PIL_IMPORT_ERROR


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


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    bg: tuple[int, int, int] = (0, 0, 0),
    fill: tuple[int, int, int] = (255, 255, 255),
    padding: int = 5,
) -> tuple[int, int, int, int]:
    x, y = xy
    bbox = text_bbox(draw, (x, y), text, font)
    box = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    draw.rectangle(box, fill=bg)
    draw.text((x, y), text, font=font, fill=fill)
    return box


def shorten(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def draw_text_box_clamped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    image_size: tuple[int, int],
    *,
    bg: tuple[int, int, int] = (0, 0, 0),
    fill: tuple[int, int, int] = (255, 255, 255),
    padding: int = 5,
) -> tuple[int, int, int, int]:
    text = shorten(text, 110)
    width, height = image_size
    bbox_value = text_bbox(draw, xy, text, font)
    box_width = bbox_value[2] - bbox_value[0] + padding * 2
    box_height = bbox_value[3] - bbox_value[1] + padding * 2
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


def load_frame_payloads(association_dir: Path, frames_arg: str, limit: int) -> list[dict[str, Any]]:
    frames_dir = association_dir / "frames"
    wanted = parse_frames_arg(frames_arg)
    candidates: list[Path] = []
    summary_path = association_dir / "association_results.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        for item in summary.get("results", []):
            frame = str(item.get("frame", ""))
            if wanted and frame not in wanted:
                continue
            artifact = item.get("artifact_json")
            artifact_path = Path(artifact) if artifact else frames_dir / f"{frame}.json"
            if artifact_path.exists():
                candidates.append(artifact_path)
    if not candidates and frames_dir.exists():
        candidates = sorted(frames_dir.glob("*.json"))
        if wanted:
            candidates = [path for path in candidates if path.stem in wanted]
    candidates = sorted(dict.fromkeys(candidates))
    if limit > 0:
        candidates = candidates[:limit]
    return [load_json(path) for path in candidates]


def lane_color(index: int) -> tuple[int, int, int]:
    return LANE_COLORS[index % len(LANE_COLORS)]


def object_color(obj: dict[str, Any]) -> tuple[int, int, int]:
    prompt_color = PROMPT_COLORS.get(str(obj.get("prompt_name") or ""))
    return prompt_color or OBJECT_COLORS.get(str(obj.get("kind") or "unknown"), OBJECT_COLORS["unknown"])


def as_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def image_points(value: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not isinstance(value, list):
        return out
    for item in value:
        point = as_point(item)
        if point is not None:
            out.append((int(round(point[0])), int(round(point[1]))))
    return out


def bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def draw_polyline(
    draw: ImageDraw.ImageDraw,
    polyline: list[tuple[int, int]],
    color: tuple[int, int, int],
    width: int,
    *,
    dots: bool = False,
) -> None:
    if len(polyline) >= 2:
        draw.line(polyline, fill=color, width=width, joint="curve")
    if dots:
        radius = max(2, width // 2)
        for x, y in polyline:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=1)


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if mask.size == size:
        return mask
    resampling = getattr(Image, "Resampling", Image).NEAREST
    return mask.resize(size, resampling)


def overlay_sam_mask(image: Image.Image, mask_path_value: str | None, alpha: int) -> Image.Image:
    if not mask_path_value:
        return image.convert("RGB")
    mask_path = Path(mask_path_value)
    if not mask_path.exists():
        return image.convert("RGB")
    mask = resize_mask(Image.open(mask_path).convert("L"), image.size)
    color = Image.new("RGBA", image.size, (*SAM_MASK_COLOR, 0))
    alpha_mask = mask.point(lambda value: alpha if value > 0 else 0)
    color.putalpha(alpha_mask)
    return Image.alpha_composite(image.convert("RGBA"), color).convert("RGB")


def lane_center_x_at_z(lane: dict[str, Any], z: float) -> float | None:
    fit = lane.get("bev_fit") or {}
    if not fit.get("ok"):
        return None
    try:
        if fit.get("model") == "constant_x":
            return float(fit["x_const"])
        if fit.get("model") == "linear_x_of_z":
            return float(fit["slope_x_per_z"]) * float(z) + float(fit["intercept_x"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def lane_bands_at_z(lanes: list[dict[str, Any]], z: float, default_width_m: float = 3.5) -> list[dict[str, Any]]:
    centers = []
    for lane in lanes:
        x = lane_center_x_at_z(lane, z)
        if x is None or not math.isfinite(x):
            continue
        centers.append({"lane": lane, "lane_id": str(lane.get("lane_id")), "center_x": float(x)})
    centers.sort(key=lambda item: item["center_x"])
    if not centers:
        return []
    if len(centers) == 1:
        half = float(default_width_m) * 0.5
        centers[0]["left_x"] = centers[0]["center_x"] - half
        centers[0]["right_x"] = centers[0]["center_x"] + half
        return centers
    for idx, item in enumerate(centers):
        center = item["center_x"]
        if idx == 0:
            right = 0.5 * (center + centers[idx + 1]["center_x"])
            half = max(float(default_width_m) * 0.5, abs(right - center))
            left = center - half
        elif idx == len(centers) - 1:
            left = 0.5 * (centers[idx - 1]["center_x"] + center)
            half = max(float(default_width_m) * 0.5, abs(center - left))
            right = center + half
        else:
            left = 0.5 * (centers[idx - 1]["center_x"] + center)
            right = 0.5 * (center + centers[idx + 1]["center_x"])
        if left > right:
            left, right = right, left
        item["left_x"] = float(left)
        item["right_x"] = float(right)
    return centers


def association_label(obj: dict[str, Any], top_k: int) -> str:
    label = str(obj.get("label_name") or obj.get("object_id") or "object")
    assignment = obj.get("assignment") or {}
    if obj.get("filtered"):
        return f"{label} | filtered:{obj.get('filter_reason')}"
    scores = assignment.get("scores") if isinstance(assignment, dict) else None
    if not scores:
        reason = assignment.get("reason") if isinstance(assignment, dict) else "no_assignment"
        return f"{label} | {reason}"
    top = scores[0]
    head = f"{label} -> L{top.get('lane_id')} {float(top.get('score', 0.0)):.2f}"
    if top_k <= 1:
        return head
    tail = []
    for item in scores[1:top_k]:
        tail.append(f"L{item.get('lane_id')}:{float(item.get('score', 0.0)):.2f}")
    return head + (" | " + ", ".join(tail) if tail else "")


def draw_lane_depth_samples(draw: ImageDraw.ImageDraw, lane: dict[str, Any], width: int) -> None:
    for sample in lane.get("samples") or []:
        if not sample.get("ok"):
            continue
        point = as_point([sample.get("x"), sample.get("y")])
        if point is None:
            continue
        color = (15, 23, 42)
        source = str(sample.get("sample_source") or "")
        if source == "centerline_exact":
            color = (255, 255, 255)
        elif source == "forward_extension":
            color = (251, 146, 60)
        elif source == "ego_extension":
            color = (236, 72, 153)
        x, y = int(round(point[0])), int(round(point[1]))
        r = max(2, width // 2)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0), width=1)


def draw_image_evidence(
    base: Image.Image,
    frame_payload: dict[str, Any],
    font: ImageFont.ImageFont,
    *,
    top_k: int,
    line_width: int,
    sam_alpha: int,
    draw_sam: bool,
) -> Image.Image:
    image = overlay_sam_mask(base, frame_payload.get("sam_mask"), sam_alpha) if draw_sam else base.convert("RGB")
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    image_size = overlay.size

    for idx, lane in enumerate(frame_payload.get("lanes") or []):
        color = lane_color(idx)
        polyline = image_points(lane.get("points"))
        draw_polyline(draw, polyline, color, line_width)
        draw_lane_depth_samples(draw, lane, line_width)
        if polyline:
            fit = lane.get("bev_fit") or {}
            status = "ok" if fit.get("ok") else str(fit.get("reason") or lane.get("invalid_reason") or "invalid")
            depth_value = lane.get("depth_median")
            try:
                depth_text = f"{float(depth_value):.1f}m" if depth_value is not None else "-"
            except (TypeError, ValueError):
                depth_text = "-"
            label = f"L{lane.get('lane_id')} z={depth_text} {status}"
            draw_text_box_clamped(draw, polyline[0], label, font, image_size, bg=color, fill=(0, 0, 0), padding=4)

    for obj in frame_payload.get("objects") or []:
        color = object_color(obj)
        box = bbox(obj.get("bbox"))
        if box is None:
            continue
        width = max(2, line_width)
        draw.rectangle(box, outline=color, width=width)
        x1, y1, x2, y2 = box
        if obj.get("filtered"):
            draw.line((x1, y1, x2, y2), fill=color, width=max(2, width - 1))
            draw.line((x1, y2, x2, y1), fill=color, width=max(2, width - 1))

        anchor = obj.get("anchor") or {}
        anchor_xy = as_point(anchor.get("image_xy"))
        if anchor_xy is not None and anchor.get("ok"):
            ax, ay = int(round(anchor_xy[0])), int(round(anchor_xy[1]))
            r = max(7, width + 3)
            draw.line((ax - r, ay, ax + r, ay), fill=color, width=max(2, width - 2))
            draw.line((ax, ay - r, ax, ay + r), fill=color, width=max(2, width - 2))

        for candidate in obj.get("ground_candidates_preview") or []:
            point = as_point([candidate.get("x"), candidate.get("y")])
            if point is None:
                continue
            cx, cy = int(round(point[0])), int(round(point[1]))
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255), outline=color)

        for item in obj.get("road_marking_band_preview") or []:
            point = as_point([item.get("x"), item.get("y")])
            if point is None:
                continue
            cx, cy = int(round(point[0])), int(round(point[1]))
            fill = (34, 197, 94) if item.get("inside_band") else (248, 113, 113)
            draw.rectangle((cx - 3, cy - 3, cx + 3, cy + 3), fill=fill)

        label = association_label(obj, top_k)
        draw_text_box_clamped(draw, (x1, max(0, y1 - font.size - 12)), label, font, image_size, bg=color, fill=(0, 0, 0), padding=4)

    header = f"{frame_payload.get('frame')} | objects={len(frame_payload.get('objects') or [])} lanes={len(frame_payload.get('lanes') or [])}"
    draw_text_box_clamped(draw, (10, 10), header, font, image_size, bg=(15, 23, 42), fill=(255, 255, 255), padding=6)
    return overlay


def load_depth_arrays(frame_payload: dict[str, Any]) -> tuple[Any, Any] | None:
    npz_path_value = frame_payload.get("depth_npz")
    if not npz_path_value:
        return None
    npz_path = Path(npz_path_value)
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=True)
    return np.asarray(npz["depth"][0], dtype=np.float32), np.asarray(npz["conf"][0], dtype=np.float32)


def normalize_array(values: Any, valid_mask: Any | None = None, *, p_low: float = 2.0, p_high: float = 98.0) -> Any:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.isfinite(arr)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    samples = arr[mask]
    if samples.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.percentile(samples, p_low))
    high = float(np.percentile(samples, p_high))
    if high <= low:
        high = low + 1e-6
    out = (arr - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def heatmap_image(norm: Any, *, invert: bool = False) -> Image.Image:
    values = 1.0 - norm if invert else norm
    r = np.clip(255.0 * values, 0, 255)
    g = np.clip(255.0 * (1.0 - np.abs(values - 0.5) * 2.0), 0, 255)
    b = np.clip(255.0 * (1.0 - values), 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def blend_heatmap(base: Image.Image, heat: Image.Image, alpha: int) -> Image.Image:
    if heat.size != base.size:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        heat = heat.resize(base.size, resampling)
    return Image.blend(base.convert("RGB"), heat.convert("RGB"), clamp(float(alpha) / 255.0, 0.0, 1.0))


def render_depth_overlay(
    source_image: Image.Image,
    frame_payload: dict[str, Any],
    font: ImageFont.ImageFont,
    args: argparse.Namespace,
) -> Image.Image:
    arrays = load_depth_arrays(frame_payload)
    if arrays is None:
        return draw_image_evidence(source_image, frame_payload, font, top_k=args.top_k, line_width=args.line_width, sam_alpha=args.sam_alpha, draw_sam=True)
    depth, conf = arrays
    norm = normalize_array(depth, np.isfinite(depth) & (depth > 0) & np.isfinite(conf), p_low=2, p_high=98)
    heat = heatmap_image(norm, invert=True)
    base = blend_heatmap(source_image, heat, args.heat_alpha)
    return draw_image_evidence(base, frame_payload, font, top_k=args.top_k, line_width=args.line_width, sam_alpha=max(18, args.sam_alpha // 2), draw_sam=True)


def render_confidence_overlay(
    source_image: Image.Image,
    frame_payload: dict[str, Any],
    font: ImageFont.ImageFont,
    args: argparse.Namespace,
) -> Image.Image:
    arrays = load_depth_arrays(frame_payload)
    if arrays is None:
        return draw_image_evidence(source_image, frame_payload, font, top_k=args.top_k, line_width=args.line_width, sam_alpha=args.sam_alpha, draw_sam=True)
    _depth, conf = arrays
    norm = normalize_array(conf, np.isfinite(conf), p_low=1, p_high=99)
    heat = heatmap_image(norm, invert=False)
    base = blend_heatmap(source_image, heat, args.heat_alpha)
    return draw_image_evidence(base, frame_payload, font, top_k=args.top_k, line_width=args.line_width, sam_alpha=max(18, args.sam_alpha // 2), draw_sam=True)


def collect_bev_points(frame_payload: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for lane in frame_payload.get("lanes") or []:
        for sample in lane.get("samples") or []:
            xyz = sample.get("xyz")
            if isinstance(xyz, list) and len(xyz) >= 3:
                points.append((float(xyz[0]), float(xyz[2])))
    for obj in frame_payload.get("objects") or []:
        anchor = obj.get("anchor") or {}
        bev = as_point(anchor.get("bev_xz"))
        if bev is not None:
            points.append(bev)
        for candidate in obj.get("ground_candidates_preview") or []:
            bev = as_point(candidate.get("bev_xz"))
            if bev is not None:
                points.append(bev)
    return [(x, z) for x, z in points if math.isfinite(x) and math.isfinite(z)]


def bev_transform(points: list[tuple[float, float]], width: int, height: int) -> tuple[Any, dict[str, float]]:
    if points:
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(zs), max(zs)
    else:
        x_min, x_max, z_min, z_max = -10.0, 10.0, 0.0, 80.0
    x_margin = max(3.0, (x_max - x_min) * 0.12)
    z_margin = max(6.0, (z_max - z_min) * 0.10)
    x_min -= x_margin
    x_max += x_margin
    z_min = max(0.0, z_min - z_margin)
    z_max += z_margin
    left, right, top, bottom = 90, width - 40, 40, height - 90
    sx = (right - left) / max(1e-6, x_max - x_min)
    sz = (bottom - top) / max(1e-6, z_max - z_min)

    def project(x: float, z: float) -> tuple[int, int]:
        px = int(round(left + (x - x_min) * sx))
        py = int(round(bottom - (z - z_min) * sz))
        return px, py

    meta = {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }
    return project, meta


def draw_bev_grid(draw: ImageDraw.ImageDraw, project: Any, meta: dict[str, float], font: ImageFont.ImageFont) -> None:
    x_step = 2.0 if meta["x_max"] - meta["x_min"] <= 24 else 5.0
    z_step = 10.0
    x0 = math.floor(meta["x_min"] / x_step) * x_step
    x1 = math.ceil(meta["x_max"] / x_step) * x_step
    z0 = math.floor(meta["z_min"] / z_step) * z_step
    z1 = math.ceil(meta["z_max"] / z_step) * z_step
    x = x0
    while x <= x1 + 1e-6:
        p1 = project(x, meta["z_min"])
        p2 = project(x, meta["z_max"])
        color = (148, 163, 184) if abs(x) < 1e-6 else (51, 65, 85)
        draw.line((p1, p2), fill=color, width=2 if abs(x) < 1e-6 else 1)
        if meta["left"] <= p1[0] <= meta["right"]:
            draw.text((p1[0] + 4, meta["bottom"] + 8), f"{x:g}m", fill=(203, 213, 225), font=font)
        x += x_step
    z = z0
    while z <= z1 + 1e-6:
        p1 = project(meta["x_min"], z)
        p2 = project(meta["x_max"], z)
        draw.line((p1, p2), fill=(51, 65, 85), width=1)
        if meta["top"] <= p1[1] <= meta["bottom"]:
            draw.text((8, p1[1] - 8), f"{z:g}m", fill=(203, 213, 225), font=font)
        z += z_step
    draw.rectangle((meta["left"], meta["top"], meta["right"], meta["bottom"]), outline=(148, 163, 184), width=2)


def lane_bev_sample_points(lane: dict[str, Any]) -> list[tuple[float, float]]:
    out = []
    for sample in lane.get("samples") or []:
        if not sample.get("ok"):
            continue
        xyz = sample.get("xyz")
        if isinstance(xyz, list) and len(xyz) >= 3:
            x, z = float(xyz[0]), float(xyz[2])
            if math.isfinite(x) and math.isfinite(z):
                out.append((x, z))
    return sorted(out, key=lambda item: item[1])


def draw_lane_fit_bev(
    draw: ImageDraw.ImageDraw,
    lane: dict[str, Any],
    lanes: list[dict[str, Any]],
    project: Any,
    meta: dict[str, float],
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    samples = lane_bev_sample_points(lane)
    if len(samples) >= 2:
        draw.line([project(x, z) for x, z in samples], fill=color, width=4)
    for x, z in samples:
        px, py = project(x, z)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(15, 23, 42), width=1)

    fit = lane.get("bev_fit") or {}
    if fit.get("ok"):
        z_values = np.linspace(meta["z_min"], meta["z_max"], 80)
        fit_points = []
        for z in z_values:
            x = lane_center_x_at_z(lane, float(z))
            if x is not None:
                fit_points.append(project(x, float(z)))
        if len(fit_points) >= 2:
            draw.line(fit_points, fill=(255, 255, 255), width=2)

        band_left = []
        band_right = []
        for z in z_values:
            bands = lane_bands_at_z(lanes, float(z))
            band = next((item for item in bands if item["lane_id"] == str(lane.get("lane_id"))), None)
            if band is None:
                continue
            band_left.append(project(float(band["left_x"]), float(z)))
            band_right.append(project(float(band["right_x"]), float(z)))
        if len(band_left) >= 2:
            draw.line(band_left, fill=color, width=1)
            draw.line(band_right, fill=color, width=1)

    label_z = samples[-1][1] if samples else meta["z_max"] * 0.5
    label_x = lane_center_x_at_z(lane, label_z) if fit.get("ok") else (samples[-1][0] if samples else 0.0)
    label = f"L{lane.get('lane_id')} n={lane.get('depth_count', 0)}"
    px, py = project(label_x or 0.0, label_z)
    draw_text_box(draw, (px + 6, py - 12), label, font, bg=color, fill=(0, 0, 0), padding=4)


def render_bev_debug(frame_payload: dict[str, Any], font: ImageFont.ImageFont, args: argparse.Namespace) -> Image.Image:
    width, height = int(args.bev_width), int(args.bev_height)
    image = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(image)
    points = collect_bev_points(frame_payload)
    project, meta = bev_transform(points, width, height)
    draw_bev_grid(draw, project, meta, font)
    lanes = frame_payload.get("lanes") or []
    for idx, lane in enumerate(lanes):
        draw_lane_fit_bev(draw, lane, lanes, project, meta, lane_color(idx), font)

    for obj in frame_payload.get("objects") or []:
        color = object_color(obj)
        anchor = obj.get("anchor") or {}
        bev = as_point(anchor.get("bev_xz"))
        if bev is not None:
            px, py = project(bev[0], bev[1])
            r = 9
            draw.ellipse((px - r, py - r, px + r, py + r), fill=color, outline=(255, 255, 255), width=2)
            label = association_label(obj, args.top_k)
            draw_text_box_clamped(draw, (px + 12, py - 14), label, font, image.size, bg=color, fill=(0, 0, 0), padding=4)
        for candidate in obj.get("ground_candidates_preview") or []:
            cand_bev = as_point(candidate.get("bev_xz"))
            if cand_bev is None:
                continue
            px, py = project(cand_bev[0], cand_bev[1])
            draw.rectangle((px - 2, py - 2, px + 2, py + 2), fill=(226, 232, 240))

    title = f"{frame_payload.get('frame')} | BEV X-Z camera space"
    draw_text_box(draw, (90, 12), title, font, bg=(30, 41, 59), fill=(255, 255, 255), padding=6)
    legend_y = height - 42
    draw.text((90, legend_y), "X: lateral meters, Z: forward depth meters. Thin colored lines are inferred lane-band edges.", fill=(203, 213, 225), font=font)
    return image


def save_image(path: Path, image: Image.Image, quality: int, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=int(clamp(quality, 1, 100)))
    return True


def render_frame(frame_payload: dict[str, Any], output_dir: Path, font: ImageFont.ImageFont, args: argparse.Namespace) -> dict[str, Any]:
    frame = str(frame_payload.get("frame"))
    image_path = Path(str(frame_payload.get("image_path") or ""))
    if not image_path.exists():
        raise FileNotFoundError(f"missing source image for frame {frame}: {image_path}")
    source = Image.open(image_path).convert("RGB")
    outputs = {
        "association_overlay": output_dir / "association_overlay" / f"{frame}.jpg",
        "depth_overlay": output_dir / "depth_overlay" / f"{frame}.jpg",
        "confidence_overlay": output_dir / "confidence_overlay" / f"{frame}.jpg",
        "bev_debug": output_dir / "bev_debug" / f"{frame}.jpg",
    }
    if args.dry_run:
        return {"frame": frame, "image_path": str(image_path), "outputs": {key: str(path) for key, path in outputs.items()}, "dry_run": True}

    association = draw_image_evidence(source, frame_payload, font, top_k=args.top_k, line_width=args.line_width, sam_alpha=args.sam_alpha, draw_sam=True)
    depth = render_depth_overlay(source, frame_payload, font, args)
    confidence = render_confidence_overlay(source, frame_payload, font, args)
    bev = render_bev_debug(frame_payload, font, args)
    images = {
        "association_overlay": association,
        "depth_overlay": depth,
        "confidence_overlay": confidence,
        "bev_debug": bev,
    }
    saved = {}
    for key, image in images.items():
        saved[key] = save_image(outputs[key], image, args.quality, args.overwrite)
    return {
        "frame": frame,
        "image_path": str(image_path),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "saved": saved,
        "object_count": len(frame_payload.get("objects") or []),
        "lane_count": len(frame_payload.get("lanes") or []),
    }


def write_index(output_dir: Path, records: list[dict[str, Any]]) -> None:
    cards = []
    for record in records:
        frame = html.escape(str(record.get("frame")))
        outputs = record.get("outputs") or {}
        views = []
        for view in VIEW_NAMES:
            path_value = outputs.get(view)
            if not path_value:
                continue
            rel = Path(path_value).relative_to(output_dir)
            views.append(
                f"<figure><a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}'></a>"
                f"<figcaption>{html.escape(view)}</figcaption></figure>"
            )
        cards.append(f"<section><h2>{frame}</h2><div class='grid'>{''.join(views)}</div></section>")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lane Association Visualization</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    h1 {{ margin-bottom: 4px; }}
    section {{ margin: 28px 0 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #111827; padding: 10px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; color: #94a3b8; }}
  </style>
</head>
<body>
  <h1>Lane Association Visualization</h1>
  <p>Generated at {html.escape(utc_now())}. Views: association overlay, depth overlay, confidence overlay, BEV debug.</p>
  {''.join(cards)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    args.association_dir = args.association_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else args.association_dir / "vis"
    frames = load_frame_payloads(args.association_dir, args.frames, args.limit)
    if not frames:
        raise FileNotFoundError(f"No association frame JSONs found under {args.association_dir}")

    if not args.dry_run:
        for view in VIEW_NAMES:
            (output_dir / view).mkdir(parents=True, exist_ok=True)
    print(f"[info] frames={len(frames)} output={output_dir}")
    font = load_font(args.font_size)
    records = []
    for idx, frame_payload in enumerate(frames, 1):
        try:
            record = render_frame(frame_payload, output_dir, font, args)
            records.append(record)
        except Exception as exc:  # noqa: BLE001 - keep batch visualization moving.
            frame = frame_payload.get("frame")
            print(f"[error] frame {frame}: {type(exc).__name__}: {exc}")
            records.append({"frame": frame, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        if idx % 10 == 0 or idx == len(frames):
            print(f"[info] rendered {idx}/{len(frames)}")

    summary = {
        "schema_version": "camera_3d_lane_association_visualization/v1",
        "created_at": utc_now(),
        "association_dir": str(args.association_dir),
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "ok_count": sum(1 for item in records if item.get("ok", True)),
        "records": records,
    }
    if not args.dry_run:
        dump_json(output_dir / "_summary.json", summary)
        write_index(output_dir, records)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] rendered={len(records)} output={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
