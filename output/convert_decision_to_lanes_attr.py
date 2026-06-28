#!/usr/bin/env python3
"""Export final lane type decisions to the downstream lane-attribute schema."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("/nas/nfs/large-model/vince/data/xd-online-las-data/test-v3")
DEFAULT_CENTER_LINE_DIR = DEFAULT_DATA_DIR / "center_line_2d"
DEFAULT_DECISION_DIR = DEFAULT_DATA_DIR / "inference" / "decision_opt"
DEFAULT_OUTPUT_FILE = DEFAULT_DATA_DIR / "output_lanes_attr.json"

LANE_TYPE_TO_ATTR = {
    "normal": 0,
    "bus": 1,
    "bicycle": 2,
    "variable": 3,
    "tidal": 4,
}
LANE_ATTRIBUTE_MAPPING = {
    "0": "普通车道",
    "1": "公交车道",
    "2": "自行车道",
    "3": "可变车道",
    "4": "潮汐车道",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert rule-based lane decisions to output_lanes_attr.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--center-line-dir", "--center_line_dir", type=Path, default=DEFAULT_CENTER_LINE_DIR)
    parser.add_argument("--decision-dir", "--decision_dir", type=Path, default=DEFAULT_DECISION_DIR)
    parser.add_argument("--output-file", "--output_file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--frames", type=str, default="", help="Comma-separated frame ids or a text file with one id per line.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any non-empty frame has no decision JSON.")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def parse_frames_arg(frames_arg: str) -> set[str]:
    if not frames_arg:
        return set()
    maybe_path = Path(frames_arg)
    if maybe_path.exists():
        return {Path(line.strip()).stem for line in maybe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return {Path(item.strip()).stem for item in frames_arg.split(",") if item.strip()}


def image_name_for_center_payload(path: Path, payload: dict[str, Any]) -> str:
    image = payload.get("image")
    if isinstance(image, str) and image:
        return Path(image).name
    return f"{path.stem}.jpg"


def normalize_lanes(lane_value: Any) -> list[dict[str, Any]]:
    if isinstance(lane_value, list):
        return [item for item in lane_value if isinstance(item, dict)]
    if isinstance(lane_value, dict):
        return [item for item in lane_value.values() if isinstance(item, dict)]
    return []


def decision_path_for_frame(decision_dir: Path, frame: str) -> Path:
    return decision_dir / "frames" / f"{frame}.json"


def load_lane_decision_map(decision_dir: Path, frame: str) -> dict[str, dict[str, Any]]:
    path = decision_path_for_frame(decision_dir, frame)
    if not path.exists():
        return {}
    payload = load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("lane_decisions") or []:
        lane_id = item.get("lane_id")
        if lane_id is None:
            continue
        out[str(lane_id)] = item
    return out


def attr_for_lane(lane_id: str, decisions: dict[str, dict[str, Any]]) -> tuple[int, str]:
    decision = decisions.get(str(lane_id)) or {}
    lane_type = str(decision.get("lane_type") or "normal")
    attr = LANE_TYPE_TO_ATTR.get(lane_type)
    if attr is None:
        return LANE_TYPE_TO_ATTR["normal"], "normal"
    return attr, lane_type


def with_attr(points: Any, attr: int) -> list[Any]:
    if not isinstance(points, list):
        return []
    out = []
    for point in points:
        if isinstance(point, dict):
            point_copy = copy.deepcopy(point)
            point_copy["Attr"] = int(attr)
            out.append(point_copy)
        else:
            out.append(copy.deepcopy(point))
    return out


def convert_lane(lane: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any], str]:
    lane_id = str(lane.get("id") or lane.get("lane_id") or "")
    if not lane_id:
        raise ValueError(f"Lane is missing id: {lane}")
    attr, lane_type = attr_for_lane(lane_id, decisions)
    return (
        lane_id,
        {
            "id": lane_id,
            "points_utm": with_attr(lane.get("points_utm"), attr),
            "Geo_points_utm": with_attr(lane.get("Geo_points_utm"), attr),
        },
        lane_type,
    )


def convert_frame(center_path: Path, center_payload: dict[str, Any], decisions: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, int]]:
    image_name = image_name_for_center_payload(center_path, center_payload)
    lanes = normalize_lanes(center_payload.get("lane"))
    if not lanes:
        return image_name, {}, {}

    lane_dict: dict[str, Any] = {}
    lane_type_counts = {name: 0 for name in LANE_TYPE_TO_ATTR}
    for lane in lanes:
        lane_id, lane_payload, lane_type = convert_lane(lane, decisions)
        lane_dict[lane_id] = lane_payload
        lane_type_counts[lane_type] = lane_type_counts.get(lane_type, 0) + 1

    payload = {
        "forward_sample_y": copy.deepcopy(center_payload.get("forward_sample_y") or []),
        "visible_geo_y_range": copy.deepcopy(center_payload.get("visible_geo_y_range") or []),
        "lane_attribute_mapping": dict(LANE_ATTRIBUTE_MAPPING),
        "lane": lane_dict,
    }
    return image_name, payload, lane_type_counts


def main() -> int:
    args = parse_args()
    args.center_line_dir = args.center_line_dir.expanduser()
    args.decision_dir = args.decision_dir.expanduser()
    args.output_file = args.output_file.expanduser()

    if args.output_file.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Output file already exists: {args.output_file}. Use --overwrite to replace it.")

    wanted = parse_frames_arg(args.frames)
    center_paths = sorted(args.center_line_dir.glob("*.json"))
    if wanted:
        center_paths = [path for path in center_paths if path.stem in wanted]
    if not center_paths:
        raise FileNotFoundError(f"No center-line JSON files found under {args.center_line_dir}")

    output: dict[str, Any] = {}
    summary = {
        "frames": 0,
        "non_empty_frames": 0,
        "lanes": 0,
        "missing_decision_frames": [],
        "lane_type_counts": {name: 0 for name in LANE_TYPE_TO_ATTR},
    }

    for center_path in center_paths:
        frame = center_path.stem
        center_payload = load_json(center_path)
        lanes = normalize_lanes(center_payload.get("lane") if isinstance(center_payload, dict) else None)
        decisions = load_lane_decision_map(args.decision_dir, frame)
        if lanes and not decisions:
            summary["missing_decision_frames"].append(frame)
            if args.strict:
                raise FileNotFoundError(f"Missing decision JSON for non-empty frame: {frame}")
        image_name, frame_payload, lane_type_counts = convert_frame(center_path, center_payload, decisions)
        output[image_name] = frame_payload
        summary["frames"] += 1
        if frame_payload:
            summary["non_empty_frames"] += 1
            summary["lanes"] += len(frame_payload.get("lane") or {})
        for lane_type, count in lane_type_counts.items():
            summary["lane_type_counts"][lane_type] = summary["lane_type_counts"].get(lane_type, 0) + int(count)

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        dump_json(args.output_file, output)
        print(json.dumps({**summary, "output_file": str(args.output_file)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
