#!/usr/bin/env python3
"""Depth-aware 3D association between perception elements and lane centerlines.

This is adapted from:
  /nas/nfs/large-model/vince/code/vlm-lane-type-reasoner/code/scripts/probe_3d_lane_association.py

The important change from the previous v1 associator is that association is
done in camera-space BEV using Depth-Anything-3 metric depth, confidence,
camera intrinsics, and SAM road-surface masks. It does not use 2D vertical
projection from signs to road masks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
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
    from PIL import Image
except ModuleNotFoundError as pil_error:  # pragma: no cover - remote dependency.
    Image = None
    PIL_IMPORT_ERROR = pil_error
else:
    PIL_IMPORT_ERROR = None


DEFAULT_DATA_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2"
DEFAULT_INFERENCE_DIR = f"{DEFAULT_DATA_DIR}/inference"
DEFAULT_IMAGE_DIR = f"{DEFAULT_DATA_DIR}/images"
DEFAULT_CENTER_LINE_DIR = f"{DEFAULT_DATA_DIR}/center_line_2d"
DEFAULT_DEPTH_DIR = f"{DEFAULT_DATA_DIR}/depth"
DEFAULT_SAM_DIR = f"{DEFAULT_DATA_DIR}/sam3"
DEFAULT_OUTPUT_DIR = f"{DEFAULT_DATA_DIR}/inference/association"
BOUNDARY_PROMPT_NAME = "prompt-classification-boundary-type"
BOX_PROMPT_PREFIX = "prompt-box-"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SIGN_SIGNAL_LABELS = {
    "bus_related_time_restriction_sign",
    "bus_sign",
    "bicycle_sign",
    "red_x",
    "variable_lane_signal",
}
BUS_RIGHTMOST_PRIOR_LABELS = {"bus_related_time_restriction_sign", "bus_sign"}
ROAD_MARKING_LABELS = {
    "bus_text_gong",
    "bus_text_jiao",
    "variable_text_ke",
    "variable_text_bian",
    "bicycle_icon",
    "bus_icon",
    "公",
    "交",
    "可",
    "变",
    "潮",
    "汐",
}
ALWAYS_FILTER_LABELS = {"mixed_lane_signal_candidate"}

LABEL_NAME_TO_ID = {
    "公": 0,
    "bus_text_gong": 0,
    "交": 1,
    "bus_text_jiao": 1,
    "bus_related_time_restriction_sign": 2,
    "可": 4,
    "variable_text_ke": 4,
    "变": 5,
    "variable_text_bian": 5,
    "mixed_lane_signal_candidate": 6,
    "bicycle_sign": 9,
    "bicycle_icon": 10,
    "bus_sign": 20,
    "bus_icon": 21,
    "red_x": 22,
    "variable_lane_signal": 23,
    "潮": 24,
    "汐": 25,
}


@dataclass
class DepthScene:
    depth: Any
    conf: Any
    intrinsics: Any
    road_mask: Any
    image_width: int
    image_height: int
    min_conf: float

    @property
    def depth_shape(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def image_to_depth_xy(self, x: float, y: float) -> tuple[int, int]:
        dh, dw = self.depth_shape
        u = int(round(float(x) / max(self.image_width - 1, 1) * (dw - 1)))
        v = int(round(float(y) / max(self.image_height - 1, 1) * (dh - 1)))
        return max(0, min(dw - 1, u)), max(0, min(dh - 1, v))

    def depth_to_camera(self, u: float, v: float, z: float) -> list[float]:
        intr = self.intrinsics
        fx = float(intr[0, 0]) if intr.size else 1.0
        fy = float(intr[1, 1]) if intr.size else fx
        cx = float(intr[0, 2]) if intr.size else self.depth.shape[1] / 2.0
        cy = float(intr[1, 2]) if intr.size else self.depth.shape[0] / 2.0
        fx = fx if abs(fx) > 1e-6 else 1.0
        fy = fy if abs(fy) > 1e-6 else 1.0
        return [float((u - cx) / fx * z), float((v - cy) / fy * z), float(z)]

    def robust_depth_at_image_xy(self, x: float, y: float, radius: int = 3) -> dict[str, Any]:
        u, v = self.image_to_depth_xy(x, y)
        dh, dw = self.depth_shape
        x0, x1 = max(0, u - radius), min(dw, u + radius + 1)
        y0, y1 = max(0, v - radius), min(dh, v + radius + 1)
        values = self.depth[y0:y1, x0:x1].reshape(-1)
        conf = self.conf[y0:y1, x0:x1].reshape(-1)
        good = np.isfinite(values) & (values > 0) & np.isfinite(conf) & (conf >= self.min_conf)
        if not np.any(good):
            return {"ok": False, "u": u, "v": v, "count": 0}
        vals = values[good].astype(float)
        z = float(np.median(vals))
        return {
            "ok": True,
            "depth": z,
            "u": u,
            "v": v,
            "count": int(vals.size),
            "conf_median": float(np.median(conf[good])),
            "xyz": self.depth_to_camera(u, v, z),
        }

    def robust_depth_in_bbox(self, bbox: list[float], shrink_ratio: float = 0.12) -> dict[str, Any]:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return {"ok": False, "reason": "invalid_bbox", "count": 0, "valid_ratio": 0.0}
        dx = (x2 - x1) * shrink_ratio
        dy = (y2 - y1) * shrink_ratio
        x1, x2 = x1 + dx, x2 - dx
        y1, y2 = y1 + dy, y2 - dy
        u0, v0 = self.image_to_depth_xy(x1, y1)
        u1, v1 = self.image_to_depth_xy(x2, y2)
        x0, xh = sorted([u0, u1])
        y0, yh = sorted([v0, v1])
        xh = min(self.depth.shape[1] - 1, max(x0, xh))
        yh = min(self.depth.shape[0] - 1, max(y0, yh))
        values = self.depth[y0 : yh + 1, x0 : xh + 1].reshape(-1)
        conf = self.conf[y0 : yh + 1, x0 : xh + 1].reshape(-1)
        total = int(values.size)
        good = np.isfinite(values) & (values > 0) & np.isfinite(conf) & (conf >= self.min_conf)
        if not np.any(good):
            return {"ok": False, "reason": "no_valid_depth", "count": 0, "valid_ratio": 0.0, "total": total}
        vals = values[good].astype(float)
        return {
            "ok": True,
            "depth_median": float(np.median(vals)),
            "depth_p25": float(np.percentile(vals, 25)),
            "depth_p75": float(np.percentile(vals, 75)),
            "count": int(vals.size),
            "total": total,
            "valid_ratio": float(vals.size / max(total, 1)),
            "conf_min": float(np.min(conf[good])),
            "conf_p25": float(np.percentile(conf[good], 25)),
            "conf_median": float(np.median(conf[good])),
            "conf_p75": float(np.percentile(conf[good], 75)),
            "conf_p90": float(np.percentile(conf[good], 90)),
            "conf_max": float(np.max(conf[good])),
        }

    def road_mask_at_image_xy(self, x: float, y: float) -> bool:
        xi = int(round(x))
        yi = int(round(y))
        h, w = self.road_mask.shape
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return False
        return bool(self.road_mask[yi, xi])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run camera-space 3D object-to-lane association for v1 inference outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", "--data_dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--inference-dir", "--inference_dir", type=Path, default=Path(DEFAULT_INFERENCE_DIR))
    parser.add_argument("--image-dir", "--image_dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--center-line-dir", "--center_line_dir", type=Path, default=Path(DEFAULT_CENTER_LINE_DIR))
    parser.add_argument("--depth-dir", "--depth_dir", type=Path, default=Path(DEFAULT_DEPTH_DIR))
    parser.add_argument("--sam-dir", "--sam_dir", type=Path, default=Path(DEFAULT_SAM_DIR))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompts", nargs="+", default=["all"], help="'all' or selected prompt-box-* folder names.")
    parser.add_argument("--frames", default="", help="Comma-separated frame stems or a text file with one stem per line.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--sign-max-depth-m", "--sign_max_depth_m", type=float, default=100.0)
    parser.add_argument("--min-conf", "--min_conf", type=float, default=1.0)
    parser.add_argument("--min-object-depth-valid-ratio", "--min_object_depth_valid_ratio", type=float, default=0.03)
    parser.add_argument("--lane-samples", "--lane_samples", type=int, default=96)
    parser.add_argument("--lane-min-depth-samples", "--lane_min_depth_samples", type=int, default=4)
    parser.add_argument("--lane-depth-radius", "--lane_depth_radius", type=int, default=3)
    parser.add_argument("--lane-extension-samples", "--lane_extension_samples", type=int, default=96)
    parser.add_argument("--lane-extension-max-px", "--lane_extension_max_px", type=float, default=900.0)
    parser.add_argument("--lane-band-default-width-m", "--lane_band_default_width_m", type=float, default=3.5)
    parser.add_argument("--lane-band-distance-scale-m", "--lane_band_distance_scale_m", type=float, default=0.75)
    parser.add_argument("--lane-band-center-scale-m", "--lane_band_center_scale_m", type=float, default=2.0)
    parser.add_argument("--sign-lateral-scale-m", "--sign_lateral_scale_m", type=float, default=1.8)
    parser.add_argument("--lane-repair-reference-threshold", "--lane_repair_reference_threshold", type=float, default=0.55)
    parser.add_argument("--lane-repair-target-threshold", "--lane_repair_target_threshold", type=float, default=0.45)
    parser.add_argument("--lane-repair-min-z-span-m", "--lane_repair_min_z_span_m", type=float, default=6.0)
    parser.add_argument("--sign-rightmost-prior", "--sign_rightmost_prior", type=float, default=9.0)
    parser.add_argument("--sign-z-extrapolation-scale-m", "--sign_z_extrapolation_scale_m", type=float, default=30.0)
    parser.add_argument("--max-ground-candidates", "--max_ground_candidates", type=int, default=220)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


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
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def selected_prompt(prompt_name: str, prompts: list[str]) -> bool:
    return "all" in set(prompts) or prompt_name in set(prompts)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def coord_mode(values: list[float]) -> str:
    if not values:
        return "normalized_1000"
    if all(0.0 <= value <= 1.0 for value in values):
        return "normalized_1"
    if all(0.0 <= value <= 1000.0 for value in values):
        return "normalized_1000"
    return "pixel"


def box_to_pixels(box: list[Any] | tuple[Any, ...], size: tuple[int, int]) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    width, height = size
    values = [float(value) for value in box[:4]]
    mode = coord_mode(values)
    x1, y1, x2, y2 = values
    if mode == "normalized_1":
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif mode == "normalized_1000":
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    x_min, x_max = sorted((clamp(x1, 0, width - 1), clamp(x2, 0, width - 1)))
    y_min, y_max = sorted((clamp(y1, 0, height - 1), clamp(y2, 0, height - 1)))
    return [x_min, y_min, x_max, y_max]


def parse_points(value: Any) -> list[list[float]]:
    points: list[list[float]] = []
    for item in value or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            points.append([float(item[0]), float(item[1])])
        except (TypeError, ValueError):
            continue
    return points


def lane_image_bottom_x(lane: dict[str, Any]) -> float | None:
    points = lane.get("points") or []
    if not points:
        return None
    try:
        return float(max(points, key=lambda point: float(point[1]))[0])
    except (TypeError, ValueError, IndexError):
        return None


def lane_image_top_x(lane: dict[str, Any]) -> float | None:
    points = lane.get("points") or []
    if not points:
        return None
    try:
        return float(min(points, key=lambda point: float(point[1]))[0])
    except (TypeError, ValueError, IndexError):
        return None


def image_path_for_frame(image_dir: Path, frame: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{frame}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_lanes(center_path: Path) -> list[dict[str, Any]]:
    payload = load_json(center_path, default={})
    raw_lanes = payload.get("lane") or payload.get("lanes") or payload.get("lines") or []
    lanes: list[dict[str, Any]] = []
    for idx, lane in enumerate(raw_lanes):
        if not isinstance(lane, dict):
            continue
        points = parse_points(lane.get("points") or lane.get("points_2d") or lane.get("polyline"))
        if not points:
            continue
        lane_id = str(lane.get("id", idx))
        lanes.append({"lane_id": lane_id, "index": idx, "points": points, "attribute": lane.get("attribute")})
    return lanes


def object_kind(label_name: str) -> str:
    if label_name in SIGN_SIGNAL_LABELS:
        return "sign_signal"
    if label_name in ROAD_MARKING_LABELS:
        return "road_marking"
    return "unknown"


def prompt_dirs(inference_dir: Path, prompts: list[str]) -> list[Path]:
    out = []
    for path in sorted(inference_dir.iterdir()):
        if not path.is_dir() or path.name.startswith("_") or path.name == "vis":
            continue
        if path.name.startswith(BOX_PROMPT_PREFIX) and selected_prompt(path.name, prompts):
            out.append(path)
    return out


def extract_box_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("status") != "ok":
        return []
    parsed = parse_model_json(payload.get("response"))
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        if any(key in parsed for key in ("bbox_2d", "bbox", "box")):
            return [parsed]
        for key in ("boxes", "objects", "detections", "result"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def load_objects_for_frame(inference_dir: Path, image_dir: Path, frame: str, prompts: list[str]) -> list[dict[str, Any]]:
    image_path = image_path_for_frame(image_dir, frame)
    if image_path is None:
        return []
    with Image.open(image_path) as image:
        image_size = image.size
    objects: list[dict[str, Any]] = []
    for prompt_dir in prompt_dirs(inference_dir, prompts):
        result_path = prompt_dir / f"{frame}.jpg.json"
        if not result_path.exists():
            result_path = prompt_dir / f"{frame}.json"
        if not result_path.exists():
            continue
        payload = load_json(result_path, default={})
        for idx, item in enumerate(extract_box_items(payload)):
            raw_box = item.get("bbox_2d") or item.get("bbox") or item.get("box")
            bbox = box_to_pixels(raw_box, image_size) if raw_box is not None else None
            if bbox is None:
                continue
            label_name = str(item.get("label") or item.get("class") or item.get("category") or "unknown")
            score_value = item.get("score") or item.get("det_score")
            try:
                score = float(score_value) if score_value is not None else None
            except (TypeError, ValueError):
                score = None
            object_id = f"{prompt_dir.name}:{idx:03d}"
            objects.append(
                {
                    "object_id": object_id,
                    "index": len(objects),
                    "prompt_name": prompt_dir.name,
                    "label_id": LABEL_NAME_TO_ID.get(label_name),
                    "label_name": label_name,
                    "kind_hint": object_kind(label_name),
                    "score": score,
                    "bbox": [float(v) for v in bbox],
                    "source_result": str(result_path),
                }
            )
    return objects


def load_boundary_types(inference_dir: Path, frame: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    root = inference_dir / BOUNDARY_PROMPT_NAME / frame
    if not root.exists():
        return out
    for path in sorted(root.glob("*.json")):
        payload = load_json(path, default={})
        lane_id = str(payload.get("lane_id") or path.stem)
        parsed = parse_model_json(payload.get("response"))
        left = parsed.get("left_boundary") if isinstance(parsed, dict) else None
        right = parsed.get("right_boundary") if isinstance(parsed, dict) else None
        out[lane_id] = {
            "left": left,
            "right": right,
            "status": payload.get("status"),
            "path": str(path),
        }
    return out


def sam_mask_path(sam_dir: Path, frame: str) -> Path | None:
    json_path = sam_dir / "json" / f"{frame}.json"
    data = load_json(json_path, default={})
    selected = (((data or {}).get("result") or {}).get("selected") or []) if isinstance(data, dict) else []
    if selected:
        rel = selected[0].get("mask_path")
        if rel:
            path = sam_dir / str(rel)
            if path.exists():
                return path
    fallback = sam_dir / "masks" / f"{frame}_mask_000.png"
    return fallback if fallback.exists() else None


def depth_npz_path(depth_dir: Path, frame: str) -> Path:
    return depth_dir / f"{frame}_jpg" / "exports" / "mini_npz" / "results.npz"


def load_depth_scene(depth_dir: Path, sam_dir: Path, frame: str, image_size: tuple[int, int], min_conf: float) -> DepthScene:
    npz_path = depth_npz_path(depth_dir, frame)
    if not npz_path.exists():
        raise FileNotFoundError(f"missing depth npz: {npz_path}")
    mask_path = sam_mask_path(sam_dir, frame)
    if mask_path is None:
        raise FileNotFoundError(f"missing sam3 road mask for frame: {frame}")
    npz = np.load(npz_path, allow_pickle=True)
    depth = np.asarray(npz["depth"][0], dtype=np.float32)
    conf = np.asarray(npz["conf"][0], dtype=np.float32)
    intrinsics = np.asarray(npz["intrinsics"][0], dtype=np.float32)
    image_width, image_height = image_size
    road_mask_img = Image.open(mask_path).convert("L")
    if road_mask_img.size != image_size:
        road_mask_img = road_mask_img.resize(image_size, Image.Resampling.NEAREST)
    road_mask = np.asarray(road_mask_img) > 0
    return DepthScene(
        depth=depth,
        conf=conf,
        intrinsics=intrinsics,
        road_mask=road_mask,
        image_width=image_width,
        image_height=image_height,
        min_conf=min_conf,
    )


def sample_polyline(points: list[list[float]], n: int) -> list[list[float]]:
    if not points:
        return []
    if len(points) == 1:
        return [points[0] for _ in range(max(1, n))]
    segs = []
    total = 0.0
    cumulative = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        length = math.dist(a, b)
        segs.append((a, b, length))
        total += length
        cumulative.append(total)
    if total <= 1e-6:
        return [points[0] for _ in range(max(1, n))]
    out: list[list[float]] = []
    seg_idx = 0
    for idx in range(max(1, n)):
        target = total * idx / max(n - 1, 1)
        while seg_idx < len(segs) - 1 and cumulative[seg_idx + 1] < target:
            seg_idx += 1
        a, b, length = segs[seg_idx]
        if length <= 1e-6:
            out.append([float(a[0]), float(a[1])])
            continue
        t = max(0.0, min(1.0, (target - cumulative[seg_idx]) / length))
        out.append([float(a[0] + t * (b[0] - a[0])), float(a[1] + t * (b[1] - a[1]))])
    return out


def extension_line(
    points: list[list[float]],
    n: int,
    image_width: int,
    image_height: int,
    max_px: float,
    *,
    mode: str,
) -> dict[str, Any]:
    if len(points) < 2:
        return {"points": [], "source": mode, "reason": "not_enough_points"}
    if mode == "forward_extension":
        first = np.asarray(points[0], dtype=float)
        last = np.asarray(points[-1], dtype=float)
        anchor, other = (first, last) if float(first[1]) >= float(last[1]) else (last, first)
        direction = other - anchor
    else:
        endpoints = [
            (np.asarray(points[0], dtype=float), np.asarray(points[1], dtype=float)),
            (np.asarray(points[-1], dtype=float), np.asarray(points[-2], dtype=float)),
        ]
        anchor, neighbor = max(endpoints, key=lambda item: float(item[0][1]))
        direction = anchor - neighbor
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        return {"points": [], "source": mode, "reason": "degenerate_direction"}
    direction = direction / norm
    out: list[list[float]] = []
    for step in np.linspace(4.0, max(4.0, max_px), max(1, n)):
        p = anchor + direction * float(step)
        if 0 <= p[0] < image_width and 0 <= p[1] < image_height:
            out.append([float(p[0]), float(p[1])])
    return {
        "points": out,
        "anchor": [float(anchor[0]), float(anchor[1])],
        "direction": [float(direction[0]), float(direction[1])],
        "source": mode,
        "reason": None,
    }


def fit_lane_bev_profile(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [sample for sample in samples if sample.get("ok") and sample.get("xyz")]
    if len(valid) < 2:
        return {"ok": False, "reason": "not_enough_samples", "sample_count": len(valid)}
    xyz = np.asarray([sample["xyz"] for sample in valid], dtype=float)
    x = xyz[:, 0]
    z = xyz[:, 2]
    good = np.isfinite(x) & np.isfinite(z) & (z > 1e-6)
    if int(good.sum()) < 2:
        return {"ok": False, "reason": "invalid_bev_samples", "sample_count": int(good.sum())}
    x = x[good]
    z = z[good]
    if float(np.ptp(z)) <= 1e-6:
        return {
            "ok": True,
            "model": "constant_x",
            "x_const": float(np.median(x)),
            "z_min": float(np.min(z)),
            "z_max": float(np.max(z)),
            "sample_count": int(x.size),
        }
    slope, intercept = np.polyfit(z, x, 1)
    residual = x - (slope * z + intercept)
    return {
        "ok": True,
        "model": "linear_x_of_z",
        "slope_x_per_z": float(slope),
        "intercept_x": float(intercept),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "sample_count": int(x.size),
        "residual_median_abs": float(np.median(np.abs(residual))),
    }


def fit_slope(fit: dict[str, Any]) -> float | None:
    if not fit.get("ok"):
        return None
    if fit.get("model") == "constant_x":
        return 0.0
    if fit.get("model") == "linear_x_of_z":
        return safe_float(fit.get("slope_x_per_z"))
    return None


def fit_x_at_z(fit: dict[str, Any], z: float) -> float | None:
    if not fit.get("ok"):
        return None
    if fit.get("model") == "constant_x":
        return safe_float(fit.get("x_const"))
    if fit.get("model") == "linear_x_of_z":
        slope = safe_float(fit.get("slope_x_per_z"))
        intercept = safe_float(fit.get("intercept_x"))
        if slope is None or intercept is None:
            return None
        return float(slope) * float(z) + float(intercept)
    return None


def fit_z_span(fit: dict[str, Any]) -> float:
    z_min = safe_float(fit.get("z_min"))
    z_max = safe_float(fit.get("z_max"))
    if z_min is None or z_max is None:
        return 0.0
    return max(0.0, z_max - z_min)


def lane_bev_fit_reliability(lane: dict[str, Any], args: argparse.Namespace) -> float:
    fit = lane.get("bev_fit") or {}
    if not fit.get("ok"):
        return 0.0
    depth_count = float(lane.get("depth_count") or 0.0)
    depth_ratio = float(lane.get("depth_valid_ratio") or 0.0)
    road_ratio = float(lane.get("road_hit_ratio") or 0.0)
    z_span = fit_z_span(fit)
    residual = safe_float(fit.get("residual_median_abs"), 0.0) or 0.0
    count_score = clamp(depth_count / 36.0, 0.0, 1.0)
    depth_score = clamp(depth_ratio / 0.55, 0.0, 1.0)
    road_score = clamp(road_ratio / 0.55, 0.0, 1.0)
    span_score = clamp(z_span / max(1e-3, float(args.lane_repair_min_z_span_m)), 0.0, 1.0)
    residual_score = math.exp(-max(0.0, residual) / 0.75)
    extension_penalty = 0.85 if lane.get("used_extension") else 1.0
    score = (0.22 * count_score + 0.24 * depth_score + 0.18 * road_score + 0.28 * span_score + 0.08 * residual_score)
    return float(clamp(score * extension_penalty, 0.0, 1.0))


def median_or_default(values: list[float], default: float) -> float:
    if not values:
        return float(default)
    return float(np.median(np.asarray(values, dtype=float)))


def repair_lane_bev_profiles(lanes: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not lanes:
        return
    ordered = sorted(
        lanes,
        key=lambda lane: (
            lane_image_bottom_x(lane) if lane_image_bottom_x(lane) is not None else float(lane.get("index") or 0),
            float(lane.get("index") or 0),
        ),
    )
    order_by_id = {str(lane.get("lane_id")): idx for idx, lane in enumerate(ordered)}
    for lane in ordered:
        reliability = lane_bev_fit_reliability(lane, args)
        lane["lane_order_left_to_right"] = order_by_id[str(lane.get("lane_id"))]
        lane["bev_fit_reliability"] = reliability
        lane["effective_bev_fit"] = dict(lane.get("bev_fit") or {})
        lane["effective_bev_fit_source"] = "raw"

    references = [
        lane
        for lane in ordered
        if lane.get("bev_fit_reliability", 0.0) >= float(args.lane_repair_reference_threshold)
        and (lane.get("bev_fit") or {}).get("ok")
        and fit_z_span(lane.get("bev_fit") or {}) >= max(1.0, float(args.lane_repair_min_z_span_m))
    ]
    if not references:
        return
    z_refs = []
    slopes = []
    for lane in references:
        fit = lane.get("bev_fit") or {}
        z_min = safe_float(fit.get("z_min"))
        z_max = safe_float(fit.get("z_max"))
        if z_min is not None and z_max is not None:
            z_refs.append((z_min + z_max) * 0.5)
        slope = fit_slope(fit)
        if slope is not None:
            slopes.append(slope)
    z_ref = median_or_default(z_refs, 20.0)
    default_slope = median_or_default(slopes, 0.0)
    widths: list[float] = []
    for left_idx, left_lane in enumerate(references):
        for right_lane in references[left_idx + 1 :]:
            order_gap = abs(order_by_id[str(right_lane.get("lane_id"))] - order_by_id[str(left_lane.get("lane_id"))])
            if order_gap <= 0:
                continue
            left_x = fit_x_at_z(left_lane.get("bev_fit") or {}, z_ref)
            right_x = fit_x_at_z(right_lane.get("bev_fit") or {}, z_ref)
            if left_x is None or right_x is None:
                continue
            width = abs(right_x - left_x) / float(order_gap)
            if 1.8 <= width <= 5.8:
                widths.append(width)
    lane_width = median_or_default(widths, float(args.lane_band_default_width_m))
    z_min_values = [safe_float((lane.get("bev_fit") or {}).get("z_min")) for lane in references]
    z_max_values = [safe_float((lane.get("bev_fit") or {}).get("z_max")) for lane in references]
    repaired_z_min = min([value for value in z_min_values if value is not None], default=max(0.0, z_ref - 15.0))
    repaired_z_max = max([value for value in z_max_values if value is not None], default=z_ref + 15.0)

    for lane in ordered:
        raw_fit = lane.get("bev_fit") or {}
        raw_ok = bool(raw_fit.get("ok"))
        z_span = fit_z_span(raw_fit)
        reliability = float(lane.get("bev_fit_reliability") or 0.0)
        needs_repair = (
            not raw_ok
            or reliability < float(args.lane_repair_target_threshold)
            or z_span < max(1.0, float(args.lane_repair_min_z_span_m))
        )
        if not needs_repair:
            continue
        lane_order = order_by_id[str(lane.get("lane_id"))]
        reference = min(references, key=lambda item: abs(order_by_id[str(item.get("lane_id"))] - lane_order))
        reference_fit = reference.get("effective_bev_fit") or reference.get("bev_fit") or {}
        reference_order = order_by_id[str(reference.get("lane_id"))]
        steps = lane_order - reference_order
        reference_x = fit_x_at_z(reference_fit, z_ref)
        if reference_x is None:
            continue
        slope = fit_slope(reference_fit)
        if slope is None:
            slope = default_slope
        repaired_x = reference_x + float(steps) * lane_width
        repaired_fit = {
            "ok": True,
            "model": "linear_x_of_z",
            "slope_x_per_z": float(slope),
            "intercept_x": float(repaired_x - float(slope) * z_ref),
            "z_min": float(repaired_z_min),
            "z_max": float(repaired_z_max),
            "sample_count": int(lane.get("depth_count") or 0),
            "repair_reference_lane_id": str(reference.get("lane_id")),
            "repair_reference_order": int(reference_order),
            "repair_target_order": int(lane_order),
            "repair_order_steps": int(steps),
            "repair_lane_width_m": float(lane_width),
            "repair_z_ref_m": float(z_ref),
            "raw_reliability": float(reliability),
            "raw_z_span_m": float(z_span),
            "raw_fit": raw_fit,
        }
        lane["effective_bev_fit"] = repaired_fit
        lane["effective_bev_fit_source"] = "neighbor_prior_repair"


def lane_depth_profile(scene: DepthScene, lane: dict[str, Any], boundary_type: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    base_samples = sample_polyline(lane["points"], args.lane_samples)
    extension_attempts: list[dict[str, Any]] = []

    def collect(samples: list[list[float]], source: str) -> list[dict[str, Any]]:
        collected = []
        for x, y in samples:
            item: dict[str, Any] = {
                "x": x,
                "y": y,
                "road_x": x,
                "road_y": y,
                "road_radius": 0,
                "road_found": False,
                "sample_source": source,
            }
            if not scene.road_mask_at_image_xy(x, y):
                item["reason"] = "not_on_road_mask"
                collected.append(item)
                continue
            depth_info = scene.robust_depth_at_image_xy(x, y, radius=args.lane_depth_radius)
            item["road_found"] = True
            item.update(depth_info)
            collected.append(item)
        return collected

    samples = collect(base_samples, "centerline_exact")
    depth_count = sum(1 for sample in samples if sample.get("ok"))
    used_extension = False
    if depth_count < args.lane_min_depth_samples:
        used_extension = True
        for source in ("forward_extension", "ego_extension"):
            info = extension_line(
                lane["points"],
                args.lane_extension_samples,
                scene.image_width,
                scene.image_height,
                args.lane_extension_max_px,
                mode=source,
            )
            extension_samples = info.get("points") or []
            collected = collect(extension_samples, source)
            attempt = {key: value for key, value in info.items() if key != "points"}
            attempt["sample_count"] = len(extension_samples)
            attempt["road_hit_count"] = sum(1 for sample in collected if sample.get("road_found"))
            attempt["depth_count"] = sum(1 for sample in collected if sample.get("ok"))
            extension_attempts.append(attempt)
            samples.extend(collected)
            depth_count = sum(1 for sample in samples if sample.get("ok"))
            if depth_count >= args.lane_min_depth_samples:
                break

    depths = [float(sample["depth"]) for sample in samples if sample.get("ok")]
    xyz = [sample["xyz"] for sample in samples if sample.get("ok") and sample.get("xyz")]
    road_hits = sum(1 for sample in samples if sample.get("road_found"))
    valid_for_association = len(depths) >= args.lane_min_depth_samples
    invalid_reason = None if valid_for_association else "insufficient_strict_road_depth_samples"
    return {
        "lane_id": lane["lane_id"],
        "index": lane["index"],
        "attribute": lane.get("attribute"),
        "boundary_type": boundary_type,
        "points": lane["points"],
        "image_width": scene.image_width,
        "image_height": scene.image_height,
        "image_bottom_x": lane_image_bottom_x(lane),
        "image_top_x": lane_image_top_x(lane),
        "sampling_policy": "centerline_exact_then_forward_extension_then_ego_extension",
        "sample_count": len(samples),
        "road_hit_count": road_hits,
        "road_hit_ratio": road_hits / max(len(samples), 1),
        "depth_count": len(depths),
        "depth_valid_ratio": len(depths) / max(len(samples), 1),
        "depth_median": float(np.median(depths)) if depths else None,
        "depth_p25": float(np.percentile(depths, 25)) if depths else None,
        "depth_p75": float(np.percentile(depths, 75)) if depths else None,
        "used_extension": used_extension,
        "extension_info": {"attempts": extension_attempts} if used_extension else None,
        "valid_for_association": valid_for_association,
        "invalid_reason": invalid_reason,
        "bev_fit": fit_lane_bev_profile(samples) if valid_for_association else {"ok": False, "reason": invalid_reason, "sample_count": len(depths)},
        "samples": samples,
        "xyz": xyz if valid_for_association else [],
    }


def lane_center_x_at_z(lane: dict[str, Any], z: float) -> float | None:
    fit = lane.get("effective_bev_fit") or lane.get("bev_fit") or {}
    if not fit.get("ok"):
        return None
    return fit_x_at_z(fit, z)


def lane_bands_at_z(lanes: list[dict[str, Any]], z: float, default_width_m: float) -> list[dict[str, Any]]:
    centers = []
    for lane in lanes:
        x = lane_center_x_at_z(lane, z)
        if x is None or not math.isfinite(x):
            continue
        centers.append({"lane": lane, "lane_id": lane["lane_id"], "center_x": float(x)})
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


def sample_road_points_in_box(scene: DepthScene, box: tuple[float, float, float, float], max_points: int) -> list[dict[str, Any]]:
    x1, y1, x2, y2 = box
    h, w = scene.road_mask.shape
    x0, xh = int(max(0, math.floor(x1))), int(min(w - 1, math.ceil(x2)))
    y0, yh = int(max(0, math.floor(y1))), int(min(h - 1, math.ceil(y2)))
    if xh <= x0 or yh <= y0:
        return []
    ys, xs = np.where(scene.road_mask[y0 : yh + 1, x0 : xh + 1])
    if xs.size == 0:
        return []
    xs = xs + x0
    ys = ys + y0
    if xs.size > max_points:
        order = np.linspace(0, xs.size - 1, max_points).astype(int)
        xs = xs[order]
        ys = ys[order]
    points = []
    for x, y in zip(xs, ys):
        info = scene.robust_depth_at_image_xy(float(x), float(y))
        if not info.get("ok"):
            continue
        points.append({"x": float(x), "y": float(y), "depth": info["depth"], "xyz": info["xyz"]})
    return points


def object_depth_anchor(scene: DepthScene, obj: dict[str, Any], body_depth: dict[str, Any]) -> dict[str, Any]:
    if not body_depth.get("ok"):
        return {"ok": False, "reason": body_depth.get("reason", "invalid_body_depth")}
    x1, y1, x2, y2 = [float(v) for v in obj["bbox"]]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    z = float(body_depth["depth_median"])
    u, v = scene.image_to_depth_xy(cx, cy)
    xyz = scene.depth_to_camera(u, v, z)
    return {
        "ok": True,
        "image_xy": [float(cx), float(cy)],
        "depth_xy": [int(u), int(v)],
        "depth": z,
        "xyz": xyz,
        "bev_xz": [float(xyz[0]), float(xyz[2])],
    }


def object_quality(scene: DepthScene, obj: dict[str, Any], kind: str, body_depth: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = [float(v) for v in obj["bbox"]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    image_area = max(1.0, float(scene.image_width * scene.image_height))
    area_ratio = area / image_area
    short_side = min(width, height)
    long_side = max(width, height)
    touch_border = []
    border_px = 2.0
    if x1 <= border_px:
        touch_border.append("left")
    if y1 <= border_px:
        touch_border.append("top")
    if x2 >= scene.image_width - 1 - border_px:
        touch_border.append("right")
    if y2 >= scene.image_height - 1 - border_px:
        touch_border.append("bottom")
    near_bottom = y2 >= scene.image_height * 0.96
    depth_valid_ratio = float(body_depth.get("valid_ratio") or 0.0)
    depth_iqr = None
    if body_depth.get("ok") and body_depth.get("depth_p25") is not None and body_depth.get("depth_p75") is not None:
        depth_iqr = float(body_depth["depth_p75"]) - float(body_depth["depth_p25"])
    score = 1.0
    if kind == "sign_signal":
        score *= clamp(short_side / 42.0, 0.0, 1.0)
        score *= clamp(area_ratio / 0.0012, 0.0, 1.0)
        if touch_border:
            score *= 0.88
    elif kind == "road_marking":
        score *= clamp(short_side / 14.0, 0.0, 1.0)
        score *= clamp(area_ratio / 0.00045, 0.0, 1.0)
        if near_bottom or "bottom" in touch_border:
            score *= 0.55
    else:
        score *= clamp(short_side / 20.0, 0.0, 1.0)
    score *= clamp(depth_valid_ratio / 0.30, 0.0, 1.0) if body_depth.get("ok") else 0.25
    if depth_iqr is not None and body_depth.get("depth_median"):
        depth_median = max(1e-3, float(body_depth["depth_median"]))
        relative_iqr = depth_iqr / depth_median
        if relative_iqr > 0.45:
            score *= 0.75
    flags = []
    if kind == "sign_signal" and short_side < 30:
        flags.append("small_sign_short_side")
    if kind == "sign_signal" and area_ratio < 0.001:
        flags.append("small_sign_area")
    if kind == "road_marking" and near_bottom:
        flags.append("road_marking_near_bottom")
    if touch_border:
        flags.append("bbox_touches_border")
    return {
        "score": float(clamp(score, 0.0, 1.0)),
        "bbox_width_px": float(width),
        "bbox_height_px": float(height),
        "bbox_short_side_px": float(short_side),
        "bbox_long_side_px": float(long_side),
        "bbox_area_ratio": float(area_ratio),
        "touch_border": touch_border,
        "truncated": bool(touch_border),
        "near_bottom": bool(near_bottom),
        "depth_valid_ratio": float(depth_valid_ratio),
        "depth_iqr_m": float(depth_iqr) if depth_iqr is not None else None,
        "flags": flags,
    }


def softmax_details(details: list[dict[str, Any]], raw_scores: list[float], method: str) -> dict[str, Any]:
    total = float(sum(raw_scores))
    if total <= 0:
        return {"scores": details, "reason": "zero_total_score", "method": method}
    for item, raw in zip(details, raw_scores):
        item["score"] = float(raw / total)
    details.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    probs = np.asarray([item.get("score", 0.0) for item in details], dtype=float)
    entropy = float(-(probs * np.log(np.clip(probs, 1e-9, 1.0))).sum() / max(math.log(max(len(probs), 2)), 1e-9))
    margin = float(probs[0] - probs[1]) if len(probs) > 1 else float(probs[0])
    return {
        "scores": details,
        "top_lane_id": details[0]["lane_id"] if details else None,
        "top_score": float(probs[0]) if probs.size else 0.0,
        "top2_margin": margin,
        "entropy": entropy,
        "assignment_mode": "single_lane_default",
        "method": method,
    }


def score_road_marking_to_lane_bands(candidates: list[dict[str, Any]], lanes: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    method = "road_marking_lane_band_bev"
    if not candidates:
        return {"scores": [], "reason": "no_ground_candidates", "method": method}
    candidate_items = [item for item in candidates if item.get("xyz")]
    if not candidate_items:
        return {"scores": [], "reason": "no_candidate_xyz", "method": method}
    raw_scores = []
    details = []
    distance_scale = max(1e-3, float(args.lane_band_distance_scale_m))
    center_scale = max(1e-3, float(args.lane_band_center_scale_m))
    default_width = max(1e-3, float(args.lane_band_default_width_m))
    for lane in lanes:
        if not (lane.get("effective_bev_fit") or lane.get("bev_fit") or {}).get("ok"):
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_lane_bev_fit", "method": method})
            continue
        band_distances: list[float] = []
        center_offsets: list[float] = []
        inside_count = 0
        preview = []
        for cand in candidate_items:
            xyz = np.asarray(cand["xyz"], dtype=float)
            cx, cz = float(xyz[0]), float(xyz[2])
            bands = lane_bands_at_z(lanes, cz, default_width)
            band = next((item for item in bands if item["lane_id"] == lane["lane_id"]), None)
            if band is None:
                continue
            left = float(band["left_x"])
            right = float(band["right_x"])
            center_x = float(band["center_x"])
            inside = left <= cx <= right
            dist = 0.0 if inside else min(abs(cx - left), abs(cx - right))
            offset = abs(cx - center_x)
            band_distances.append(float(dist))
            center_offsets.append(float(offset))
            inside_count += int(inside)
            if len(preview) < 24:
                preview.append(
                    {
                        "lane_id": lane["lane_id"],
                        "x": float(cand["x"]),
                        "y": float(cand["y"]),
                        "depth": float(cand["depth"]),
                        "bev_xz": [cx, cz],
                        "band_left_x": left,
                        "band_right_x": right,
                        "band_center_x": center_x,
                        "inside_band": inside,
                        "band_distance": float(dist),
                        "center_offset": float(offset),
                    }
                )
        if not band_distances:
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_candidate_band_eval", "method": method})
            continue
        inside_fraction = inside_count / max(len(band_distances), 1)
        robust_band_distance = float(np.percentile(band_distances, 50))
        robust_center_offset = float(np.percentile(center_offsets, 50))
        band_score = math.exp(-robust_band_distance / distance_scale)
        center_score = math.exp(-robust_center_offset / center_scale)
        raw = (0.05 + inside_fraction) * band_score * (0.35 + 0.65 * center_score)
        raw_scores.append(raw)
        details.append(
            {
                "lane_id": lane["lane_id"],
                "raw_score": raw,
                "method": method,
                "inside_fraction": float(inside_fraction),
                "robust_band_distance": robust_band_distance,
                "robust_center_offset": robust_center_offset,
                "candidate_count": len(band_distances),
                "lane_bev_fit": lane.get("effective_bev_fit") or lane.get("bev_fit"),
                "lane_bev_fit_source": lane.get("effective_bev_fit_source", "raw"),
                "lane_bev_fit_reliability": lane.get("bev_fit_reliability"),
                "band_samples_preview": preview,
            }
        )
    return softmax_details(details, raw_scores, method)


def score_sign_bev_anchor_to_lanes(anchor: dict[str, Any], obj: dict[str, Any], lanes: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    method = "sign_bev_lateral_lane_band"
    if not anchor.get("ok") or not anchor.get("xyz"):
        return {"scores": [], "reason": anchor.get("reason", "invalid_sign_anchor"), "method": method}
    anchor_xyz = np.asarray(anchor["xyz"], dtype=float)
    anchor_bev = np.asarray([anchor_xyz[0], anchor_xyz[2]], dtype=float)
    raw_scores = []
    details = []
    lateral_scale = max(1e-3, float(args.sign_lateral_scale_m))
    default_width = max(1e-3, float(args.lane_band_default_width_m))
    z_extrapolation_scale = max(1e-3, float(args.sign_z_extrapolation_scale_m))
    bands = lane_bands_at_z(lanes, float(anchor_bev[1]), default_width)
    band_by_lane = {str(item["lane_id"]): item for item in bands}
    ordered_lanes = sorted(
        [lane for lane in lanes if lane_image_bottom_x(lane) is not None],
        key=lambda lane: float(lane_image_bottom_x(lane) or 0.0),
    )
    rightmost_lane_id = str(ordered_lanes[-1].get("lane_id")) if ordered_lanes else None
    label_name = str(obj.get("label_name") or "")
    bbox_value = obj.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    x1, y1, x2, y2 = [float(v) for v in bbox_value]
    center_x_norm = ((x1 + x2) * 0.5) / max(1.0, float(obj.get("image_width") or 1.0))
    quality = obj.get("object_quality") or {}
    touch_border = set(quality.get("touch_border") or [])
    rightmost_prior_active = (
        label_name in BUS_RIGHTMOST_PRIOR_LABELS
        and rightmost_lane_id is not None
        and (center_x_norm >= 0.52 or bool(touch_border & {"top", "right"}))
    )
    for lane in lanes:
        band = band_by_lane.get(str(lane["lane_id"]))
        if band is None:
            raw_scores.append(0.0)
            details.append({"lane_id": lane["lane_id"], "raw_score": 0.0, "reason": "no_lane_bev_fit", "method": method})
            continue
        left = float(band["left_x"])
        right = float(band["right_x"])
        center_x = float(band["center_x"])
        anchor_x = float(anchor_bev[0])
        inside = left <= anchor_x <= right
        band_distance = 0.0 if inside else min(abs(anchor_x - left), abs(anchor_x - right))
        center_offset = abs(anchor_x - center_x)
        center_score = math.exp(-center_offset / lateral_scale)
        band_score = 1.0 if inside else math.exp(-band_distance / lateral_scale)
        lane_quality = clamp(float(lane.get("bev_fit_reliability") or 0.0), 0.0, 1.0)
        quality_factor = 0.65 + 0.35 * lane_quality
        fit = lane.get("effective_bev_fit") or lane.get("bev_fit") or {}
        z_min = fit.get("z_min")
        z_max = fit.get("z_max")
        z_extrapolation = 0.0
        if z_min is not None and float(anchor_bev[1]) < float(z_min):
            z_extrapolation = float(z_min) - float(anchor_bev[1])
        elif z_max is not None and float(anchor_bev[1]) > float(z_max):
            z_extrapolation = float(anchor_bev[1]) - float(z_max)
        z_penalty = math.exp(-max(0.0, z_extrapolation - 5.0) / z_extrapolation_scale)
        prior_factor = 1.0
        if rightmost_prior_active and str(lane["lane_id"]) == rightmost_lane_id:
            prior_factor = max(1.0, float(args.sign_rightmost_prior))
        raw = quality_factor * band_score * center_score * z_penalty * prior_factor
        raw_scores.append(raw)
        details.append(
            {
                "lane_id": lane["lane_id"],
                "raw_score": raw,
                "method": method,
                "lane_quality": lane_quality,
                "quality_factor": quality_factor,
                "lane_depth_median": lane.get("depth_median"),
                "anchor_bev_xz": [float(anchor_bev[0]), float(anchor_bev[1])],
                "lane_center_x_at_anchor_z": center_x,
                "band_left_x": left,
                "band_right_x": right,
                "inside_band": bool(inside),
                "band_distance": float(band_distance),
                "center_offset": float(center_offset),
                "center_score": float(center_score),
                "band_score": float(band_score),
                "z_extrapolation_m": float(z_extrapolation),
                "z_penalty": float(z_penalty),
                "rightmost_prior_active": bool(rightmost_prior_active),
                "rightmost_prior_factor": float(prior_factor),
                "lane_bev_fit": fit,
                "lane_bev_fit_source": lane.get("effective_bev_fit_source", "raw"),
                "lane_bev_fit_reliability": lane_quality,
            }
        )
    return softmax_details(details, raw_scores, method)


def analyze_object(scene: DepthScene, obj: dict[str, Any], lane_profiles: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    kind = object_kind(obj.get("label_name", ""))
    body_depth = scene.robust_depth_in_bbox(obj["bbox"])
    anchor = object_depth_anchor(scene, obj, body_depth)
    quality = object_quality(scene, obj, kind, body_depth)
    obj_for_scoring = {**obj, "object_quality": quality, "image_width": scene.image_width, "image_height": scene.image_height}
    filtered = False
    filter_reason = None
    if obj.get("label_name") in ALWAYS_FILTER_LABELS:
        filtered = True
        filter_reason = "mixed_lane_signal_candidate_filtered"
    elif kind == "sign_signal":
        if not body_depth.get("ok") or float(body_depth.get("valid_ratio") or 0.0) < args.min_object_depth_valid_ratio:
            filtered = True
            filter_reason = "sign_depth_unreliable"
        elif float(body_depth["depth_median"]) > args.sign_max_depth_m:
            filtered = True
            filter_reason = "sign_depth_gt_max"

    result = {
        **obj,
        "kind": kind,
        "body_depth": body_depth,
        "object_quality": quality,
        "anchor": anchor,
        "filtered": filtered,
        "filter_reason": filter_reason,
        "sign_max_depth_m": args.sign_max_depth_m if kind == "sign_signal" else None,
    }
    if filtered:
        result["ground_candidate_count"] = 0
        result["assignment"] = {"scores": [], "reason": filter_reason}
        return result
    if kind == "sign_signal":
        result["ground_candidate_count"] = 0
        result["ground_depth_median"] = None
        result["association_input"] = "sign_bev_depth_anchor"
        result["assignment"] = score_sign_bev_anchor_to_lanes(anchor, obj_for_scoring, lane_profiles, args)
        return result

    candidates = sample_road_points_in_box(scene, tuple(obj["bbox"]), args.max_ground_candidates) if kind == "road_marking" else []
    result["ground_candidate_count"] = len(candidates)
    result["ground_depth_median"] = float(np.median([item["depth"] for item in candidates])) if candidates else None
    result["association_input"] = "road_marking_ground_candidates" if kind == "road_marking" else "unknown"
    result["ground_candidates_preview"] = candidates[:30]
    result["assignment"] = score_road_marking_to_lane_bands(candidates, lane_profiles, args)
    top_score = (result["assignment"].get("scores") or [{}])[0] if result["assignment"].get("scores") else {}
    result["road_marking_band_preview"] = top_score.get("band_samples_preview", [])
    return result


def confidence_stats(conf: Any) -> dict[str, float | int | None]:
    values = conf.astype(float).reshape(-1)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {"count": 0}
    return {
        "count": int(valid.size),
        "min": float(np.min(valid)),
        "p01": float(np.percentile(valid, 1)),
        "p10": float(np.percentile(valid, 10)),
        "median": float(np.median(valid)),
        "p90": float(np.percentile(valid, 90)),
        "p99": float(np.percentile(valid, 99)),
        "max": float(np.max(valid)),
    }


def discover_frames(args: argparse.Namespace) -> list[str]:
    stems = {path.stem for path in args.center_line_dir.glob("*.json")}
    for prompt_dir in prompt_dirs(args.inference_dir, args.prompts):
        stems.update(Path(path.stem).stem for path in prompt_dir.glob("*.json"))
    frames = sorted(stems)
    if args.frames:
        maybe_path = Path(args.frames)
        if maybe_path.exists():
            wanted = {Path(line.strip()).stem for line in maybe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        else:
            wanted = {Path(item.strip()).stem for item in args.frames.split(",") if item.strip()}
        frames = [frame for frame in frames if frame in wanted]
    if args.limit > 0:
        frames = frames[: args.limit]
    return frames


def process_frame(args: argparse.Namespace, frame: str) -> dict[str, Any]:
    center_path = args.center_line_dir / f"{frame}.json"
    image_path = image_path_for_frame(args.image_dir, frame)
    if image_path is None:
        raise FileNotFoundError(f"missing source image for frame {frame}")
    with Image.open(image_path) as image:
        image_size = image.size
    lanes = load_lanes(center_path)
    boundary_types = load_boundary_types(args.inference_dir, frame)
    objects = load_objects_for_frame(args.inference_dir, args.image_dir, frame, args.prompts)
    scene = load_depth_scene(args.depth_dir, args.sam_dir, frame, image_size, args.min_conf)
    lane_profiles = [
        lane_depth_profile(scene, lane, boundary_types.get(str(lane["lane_id"])), args)
        for lane in lanes
    ]
    repair_lane_bev_profiles(lane_profiles, args)
    object_results = [analyze_object(scene, obj, lane_profiles, args) for obj in objects]
    result = {
        "schema_version": "camera_3d_lane_association_probe/v1-adapted",
        "frame": frame,
        "ok": True,
        "image_path": str(image_path),
        "depth_npz": str(depth_npz_path(args.depth_dir, frame)),
        "sam_mask": str(sam_mask_path(args.sam_dir, frame)),
        "confidence_stats": confidence_stats(scene.conf),
        "lanes": lane_profiles,
        "objects": object_results,
        "filtered_object_count": sum(1 for obj in object_results if obj.get("filtered")),
        "active_object_count": sum(1 for obj in object_results if not obj.get("filtered")),
    }
    dump_json(args.output_dir / "frames" / f"{frame}.json", result)
    result["artifact_json"] = str(args.output_dir / "frames" / f"{frame}.json")
    return result


def validate_paths(args: argparse.Namespace) -> None:
    missing = []
    for label, path in (
        ("inference_dir", args.inference_dir),
        ("image_dir", args.image_dir),
        ("center_line_dir", args.center_line_dir),
        ("depth_dir", args.depth_dir),
        ("sam_dir", args.sam_dir),
    ):
        if not path.exists():
            missing.append(f"{label}: {path}")
    if missing:
        raise FileNotFoundError("Required path(s) not found:\n" + "\n".join(f"- {item}" for item in missing))


def main() -> int:
    args = parse_args()
    if NUMPY_IMPORT_ERROR is not None:
        raise RuntimeError("Missing dependency: numpy. Run in the remote environment or install numpy.") from NUMPY_IMPORT_ERROR
    if PIL_IMPORT_ERROR is not None:
        raise RuntimeError("Missing dependency: Pillow. Run in the remote environment or install pillow.") from PIL_IMPORT_ERROR
    for name in ("data_dir", "inference_dir", "image_dir", "center_line_dir", "depth_dir", "sam_dir", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser())
    validate_paths(args)
    frames = discover_frames(args)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)

    print(f"[info] frames={len(frames)} output={args.output_dir}")
    results: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames, 1):
        try:
            if args.dry_run:
                image_path = image_path_for_frame(args.image_dir, frame)
                lanes = load_lanes(args.center_line_dir / f"{frame}.json")
                objects = load_objects_for_frame(args.inference_dir, args.image_dir, frame, args.prompts)
                results.append({"frame": frame, "ok": True, "image_path": str(image_path), "lanes": lanes, "objects": objects})
            else:
                results.append(process_frame(args, frame))
        except Exception as exc:  # noqa: BLE001 - keep batch moving for diagnosis.
            print(f"[error] frame {frame}: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append({"frame": frame, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        if idx % 10 == 0 or idx == len(frames):
            print(f"[info] processed {idx}/{len(frames)}")

    payload = {
        "schema_version": "camera_3d_lane_association_probe/v1-adapted",
        "created_at": utc_now(),
        "data_dir": str(args.data_dir),
        "inference_dir": str(args.inference_dir),
        "output_dir": str(args.output_dir),
        "config": {
            "sign_max_depth_m": args.sign_max_depth_m,
            "min_conf": args.min_conf,
            "min_object_depth_valid_ratio": args.min_object_depth_valid_ratio,
            "lane_samples": args.lane_samples,
            "lane_sampling_policy": "centerline_exact_then_forward_extension_then_ego_extension",
            "lane_min_depth_samples": args.lane_min_depth_samples,
            "lane_depth_radius": args.lane_depth_radius,
            "lane_extension_samples": args.lane_extension_samples,
            "lane_extension_max_px": args.lane_extension_max_px,
            "sign_association_method": "sign_bev_lateral_lane_band",
            "road_marking_association_method": "road_marking_lane_band_bev",
            "lane_band_default_width_m": args.lane_band_default_width_m,
            "lane_band_distance_scale_m": args.lane_band_distance_scale_m,
            "lane_band_center_scale_m": args.lane_band_center_scale_m,
            "sign_lateral_scale_m": args.sign_lateral_scale_m,
            "lane_repair_reference_threshold": args.lane_repair_reference_threshold,
            "lane_repair_target_threshold": args.lane_repair_target_threshold,
            "lane_repair_min_z_span_m": args.lane_repair_min_z_span_m,
            "sign_rightmost_prior": args.sign_rightmost_prior,
            "sign_z_extrapolation_scale_m": args.sign_z_extrapolation_scale_m,
            "max_ground_candidates": args.max_ground_candidates,
        },
        "frame_count": len(frames),
        "ok_count": sum(1 for item in results if item.get("ok")),
        "failed_count": sum(1 for item in results if not item.get("ok")),
        "active_object_count": sum(item.get("active_object_count", len(item.get("objects", []))) for item in results if item.get("ok")),
        "filtered_object_count": sum(item.get("filtered_object_count", 0) for item in results if item.get("ok")),
        "results": results,
    }
    if not args.dry_run:
        dump_json(args.output_dir / "association_results.json", payload)
        dump_json(args.output_dir / "_summary.json", payload)
    else:
        print(f"dry_run_output_dir={args.output_dir}")
    print(
        f"[done] ok={payload['ok_count']} failed={payload['failed_count']} "
        f"active_objects={payload['active_object_count']} filtered_objects={payload['filtered_object_count']}"
    )
    return 0 if payload["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
