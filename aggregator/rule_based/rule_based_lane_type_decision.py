#!/usr/bin/env python3
"""Rule-based lane type decision from VLM perception and lane associations.

The decision classes are mutually exclusive:
  tidal / variable / bus / bicycle / normal

Inputs are the per-frame association JSONs produced by:
  associator/associate_elements_to_lanes.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2"
DEFAULT_ASSOCIATION_DIR = f"{DEFAULT_DATA_DIR}/inference/association"
DEFAULT_OUTPUT_DIR = f"{DEFAULT_DATA_DIR}/inference/decision"

LANE_TYPES = ("tidal", "variable", "bus", "bicycle", "normal")
LANE_TYPE_ZH = {
    "tidal": "潮汐车道",
    "variable": "可变车道",
    "bus": "公交车道",
    "bicycle": "自行车道",
    "normal": "普通车道",
}
LANE_TYPE_INDEX = {name: idx for idx, name in enumerate(LANE_TYPES)}
SPECIAL_TYPES = ("tidal", "variable", "bus", "bicycle")
PRIORITY = ("tidal", "variable", "bus", "bicycle")

TIDAL_WORDS = {"潮", "汐"}
VARIABLE_WORDS = {"可", "变"}
BUS_WORDS = {"公", "交"}

BUS_LABELS = {"bus_icon", "bus_related_time_restriction_sign", "bus_sign"}
BICYCLE_LABELS = {"bicycle_icon", "bicycle_sign"}
INVALID_SIGNAL_LABELS = {"variable_lane_signal", "red_x"}

PAIR_WORD_RULES = {
    "tidal_text_pair": ("tidal", TIDAL_WORDS, 8.0, "road_text_pair_chao_xi"),
    "variable_text_pair": ("variable", VARIABLE_WORDS, 8.0, "road_text_pair_ke_bian"),
    "bus_text_pair": ("bus", BUS_WORDS, 8.0, "road_text_pair_gong_jiao"),
}
SINGLE_WORD_RULES = {
    "潮": ("tidal", 5.0, "road_text_chao"),
    "汐": ("tidal", 5.0, "road_text_xi"),
    "可": ("variable", 5.0, "road_text_ke"),
    "变": ("variable", 5.0, "road_text_bian"),
    "公": ("bus", 5.0, "road_text_gong"),
    "交": ("bus", 5.0, "road_text_jiao"),
}
OBJECT_RULES = {
    "bus_icon": ("bus", 5.0, "road_bus_icon"),
    "bus_related_time_restriction_sign": ("bus", 7.0, "bus_related_time_restriction_sign"),
    "bus_sign": ("bus", 7.0, "bus_sign"),
    "bicycle_icon": ("bicycle", 5.0, "road_bicycle_icon"),
    "bicycle_sign": ("bicycle", 5.0, "bicycle_sign"),
}


@dataclass
class Evidence:
    rule_id: str
    lane_type: str | None
    score_delta: float
    source: str
    label: str | None = None
    object_id: str | None = None
    association_score: float | None = None
    top2_margin: float | None = None
    accepted: bool = True
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "rule_id": self.rule_id,
            "lane_type": self.lane_type,
            "score_delta": round(float(self.score_delta), 6),
            "source": self.source,
            "accepted": self.accepted,
        }
        for key in ("label", "object_id", "association_score", "top2_margin", "reason"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.details:
            payload["details"] = self.details
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate lane type decisions with deterministic rules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--association-dir", "--association_dir", type=Path, default=Path(DEFAULT_ASSOCIATION_DIR))
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--frames", type=str, default="", help="Comma-separated frame ids or a text file with one id per line.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--min-road-score", "--min_road_score", type=float, default=0.35)
    parser.add_argument("--min-road-inside-fraction", "--min_road_inside_fraction", type=float, default=0.20)
    parser.add_argument("--min-sign-score", "--min_sign_score", type=float, default=0.50)
    parser.add_argument("--min-sign-margin", "--min_sign_margin", type=float, default=0.0)
    parser.add_argument("--tie-margin", "--tie_margin", type=float, default=1.0)
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


def parse_frames_arg(frames_arg: str) -> set[str]:
    if not frames_arg:
        return set()
    maybe_path = Path(frames_arg)
    if maybe_path.exists():
        return {Path(line.strip()).stem for line in maybe_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return {Path(item.strip()).stem for item in frames_arg.split(",") if item.strip()}


def frame_paths(association_dir: Path, frames_arg: str, limit: int) -> list[Path]:
    wanted = parse_frames_arg(frames_arg)
    frames_dir = association_dir / "frames"
    paths = sorted(frames_dir.glob("*.json")) if frames_dir.exists() else []
    if wanted:
        paths = [path for path in paths if path.stem in wanted]
    if limit > 0:
        paths = paths[:limit]
    return paths


def one_hot(lane_type: str) -> list[int]:
    values = [0 for _ in LANE_TYPES]
    values[LANE_TYPE_INDEX[lane_type]] = 1
    return values


def one_hot_dict(lane_type: str) -> dict[str, int]:
    return {name: int(name == lane_type) for name in LANE_TYPES}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def score_item_for_lane(obj: dict[str, Any], lane_id: str) -> dict[str, Any] | None:
    assignment = obj.get("assignment") or {}
    scores = assignment.get("scores") or []
    for item in scores:
        if str(item.get("lane_id")) == str(lane_id):
            return item
    return None


def top_score_item(obj: dict[str, Any]) -> dict[str, Any] | None:
    assignment = obj.get("assignment") or {}
    scores = assignment.get("scores") or []
    return scores[0] if scores else None


def accepted_top_object(obj: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str | None, dict[str, Any] | None]:
    if obj.get("filtered"):
        return False, f"filtered:{obj.get('filter_reason')}", None
    label = str(obj.get("label_name") or "")
    if label in INVALID_SIGNAL_LABELS:
        return False, "invalid_signal_label_for_lane_type", top_score_item(obj)
    score_item = top_score_item(obj)
    if score_item is None:
        return False, "missing_association_score", None
    kind = str(obj.get("kind") or "")
    score = safe_float(score_item.get("score"))
    margin = safe_float((obj.get("assignment") or {}).get("top2_margin"))
    inside_fraction = safe_float(score_item.get("inside_fraction"))
    if kind == "road_marking":
        if score >= args.min_road_score or inside_fraction >= args.min_road_inside_fraction:
            return True, None, score_item
        return False, "road_marking_association_below_threshold", score_item
    if kind == "sign_signal":
        if score >= args.min_sign_score and margin >= args.min_sign_margin:
            return True, None, score_item
        return False, "sign_association_below_threshold", score_item
    return False, "unknown_object_kind", score_item


def collect_objects_by_lane(frame_payload: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = []
    for obj in frame_payload.get("objects") or []:
        accepted, reason, score_item = accepted_top_object(obj, args)
        top_lane_id = str(score_item.get("lane_id")) if score_item and score_item.get("lane_id") is not None else None
        record = {
            "object_id": obj.get("object_id"),
            "label_name": obj.get("label_name"),
            "kind": obj.get("kind"),
            "prompt_name": obj.get("prompt_name"),
            "top_lane_id": top_lane_id,
            "top_score": safe_float(score_item.get("score")) if score_item else None,
            "top2_margin": safe_float((obj.get("assignment") or {}).get("top2_margin")),
            "accepted": accepted,
            "reason": reason,
        }
        if accepted and top_lane_id is not None:
            enriched = dict(obj)
            enriched["_accepted_score_item"] = score_item
            by_lane[top_lane_id].append(enriched)
        else:
            rejected.append(record)
    return by_lane, rejected


def boundary_value(lane: dict[str, Any], side: str) -> str:
    boundary = lane.get("boundary_type") or {}
    value = boundary.get(side)
    return str(value or "none")


def add_score(scores: dict[str, float], evidence: list[Evidence], item: Evidence) -> None:
    evidence.append(item)
    if item.accepted and item.lane_type in SPECIAL_TYPES:
        scores[str(item.lane_type)] += float(item.score_delta)


def boundary_evidence(lane: dict[str, Any], scores: dict[str, float], evidence: list[Evidence]) -> None:
    left = boundary_value(lane, "left")
    right = boundary_value(lane, "right")
    details = {"left_boundary": left, "right_boundary": right}
    if left == "double_yellow_dash" and right == "double_yellow_dash":
        add_score(
            scores,
            evidence,
            Evidence("both_double_yellow_dash", "tidal", 7.0, "boundary_type", details=details),
        )
    elif left == "double_yellow_dash" or right == "double_yellow_dash":
        evidence.append(
            Evidence(
                "single_side_double_yellow_dash_ignored",
                None,
                0.0,
                "boundary_type",
                accepted=False,
                reason="double_yellow_dash_requires_both_sides",
                details=details,
            )
        )

    if left == "zigzag_line" and right == "zigzag_line":
        add_score(
            scores,
            evidence,
            Evidence("both_zigzag_line", "variable", 7.0, "boundary_type", details=details),
        )
    elif left == "zigzag_line" or right == "zigzag_line":
        evidence.append(
            Evidence(
                "single_side_zigzag_line_ignored",
                None,
                0.0,
                "boundary_type",
                accepted=False,
                reason="zigzag_line_requires_both_sides",
                details=details,
            )
        )


def object_confidence(obj: dict[str, Any]) -> tuple[float, float]:
    score_item = obj.get("_accepted_score_item") or {}
    score = safe_float(score_item.get("score"), default=1.0)
    margin = safe_float((obj.get("assignment") or {}).get("top2_margin"))
    return score, margin


def add_object_rule(
    obj: dict[str, Any],
    lane_type: str,
    base_weight: float,
    rule_id: str,
    scores: dict[str, float],
    evidence: list[Evidence],
) -> None:
    assoc_score, margin = object_confidence(obj)
    delta = base_weight * assoc_score
    add_score(
        scores,
        evidence,
        Evidence(
            rule_id,
            lane_type,
            delta,
            str(obj.get("kind") or "object"),
            label=str(obj.get("label_name") or ""),
            object_id=str(obj.get("object_id") or ""),
            association_score=round(assoc_score, 6),
            top2_margin=round(margin, 6),
            details={"base_weight": base_weight},
        ),
    )


def text_evidence(objects: list[dict[str, Any]], scores: dict[str, float], evidence: list[Evidence]) -> set[str]:
    label_best: dict[str, dict[str, Any]] = {}
    for obj in objects:
        label = str(obj.get("label_name") or "")
        if label not in SINGLE_WORD_RULES:
            continue
        assoc_score, _margin = object_confidence(obj)
        previous = label_best.get(label)
        if previous is None or assoc_score > previous["_assoc_score"]:
            enriched = dict(obj)
            enriched["_assoc_score"] = assoc_score
            label_best[label] = enriched

    used_labels: set[str] = set()
    for _rule_id, (lane_type, labels, base_weight, reason) in PAIR_WORD_RULES.items():
        if labels.issubset(label_best.keys()):
            assoc_score = min(label_best[label]["_assoc_score"] for label in labels)
            delta = base_weight * assoc_score
            object_ids = [str(label_best[label].get("object_id") or "") for label in sorted(labels)]
            add_score(
                scores,
                evidence,
                Evidence(
                    reason,
                    lane_type,
                    delta,
                    "road_text_pair",
                    label="+".join(sorted(labels)),
                    object_id=",".join(object_ids),
                    association_score=round(assoc_score, 6),
                    details={"base_weight": base_weight, "paired_labels": sorted(labels)},
                ),
            )
            used_labels.update(labels)

    for label, obj in sorted(label_best.items()):
        if label in used_labels:
            continue
        lane_type, base_weight, rule_id = SINGLE_WORD_RULES[label]
        add_object_rule(obj, lane_type, base_weight, rule_id, scores, evidence)
    return set(label_best.keys())


def object_evidence(objects: list[dict[str, Any]], scores: dict[str, float], evidence: list[Evidence]) -> None:
    text_evidence(objects, scores, evidence)
    for obj in objects:
        label = str(obj.get("label_name") or "")
        if label in SINGLE_WORD_RULES:
            continue
        if label in INVALID_SIGNAL_LABELS:
            evidence.append(
                Evidence(
                    "invalid_signal_label_ignored",
                    None,
                    0.0,
                    str(obj.get("kind") or "object"),
                    label=label,
                    object_id=str(obj.get("object_id") or ""),
                    accepted=False,
                    reason="variable_lane_signal_and_red_x_are_invalid_for_lane_type",
                )
            )
            continue
        rule = OBJECT_RULES.get(label)
        if rule is None:
            evidence.append(
                Evidence(
                    "unsupported_label_ignored",
                    None,
                    0.0,
                    str(obj.get("kind") or "object"),
                    label=label,
                    object_id=str(obj.get("object_id") or ""),
                    accepted=False,
                    reason="label_has_no_rule",
                )
            )
            continue
        lane_type, base_weight, rule_id = rule
        add_object_rule(obj, lane_type, base_weight, rule_id, scores, evidence)


def choose_lane_type(scores: dict[str, float], tie_margin: float) -> tuple[str, str]:
    max_score = max(scores.get(name, 0.0) for name in SPECIAL_TYPES)
    if max_score <= 0.0:
        return "normal", "no_special_evidence"
    close = {name for name in SPECIAL_TYPES if max_score - scores.get(name, 0.0) <= tie_margin}
    for lane_type in PRIORITY:
        if lane_type in close:
            if len(close) > 1:
                return lane_type, f"priority_tie_break_within_{tie_margin:g}"
            return lane_type, "highest_score"
    return "normal", "fallback"


def decide_lane(
    frame: str,
    lane: dict[str, Any],
    objects: list[dict[str, Any]],
    rejected_objects: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    lane_id = str(lane.get("lane_id"))
    scores = {name: 0.0 for name in SPECIAL_TYPES}
    evidence: list[Evidence] = []
    boundary_evidence(lane, scores, evidence)
    object_evidence(objects, scores, evidence)
    lane_type, reason = choose_lane_type(scores, args.tie_margin)
    full_scores = {name: round(float(scores.get(name, 0.0)), 6) for name in SPECIAL_TYPES}
    full_scores["normal"] = 1.0 if lane_type == "normal" else 0.0
    boundary = lane.get("boundary_type") or {}
    return {
        "frame": frame,
        "lane_id": lane_id,
        "lane_index": lane.get("index"),
        "lane_type": lane_type,
        "lane_type_zh": LANE_TYPE_ZH[lane_type],
        "one_hot_order": list(LANE_TYPES),
        "one_hot": one_hot(lane_type),
        "one_hot_dict": one_hot_dict(lane_type),
        "decision_reason": reason,
        "scores": full_scores,
        "boundary_type": {
            "left": boundary.get("left"),
            "right": boundary.get("right"),
            "status": boundary.get("status"),
            "path": boundary.get("path"),
        },
        "accepted_objects": [
            {
                "object_id": obj.get("object_id"),
                "label_name": obj.get("label_name"),
                "kind": obj.get("kind"),
                "prompt_name": obj.get("prompt_name"),
                "association_score": safe_float((obj.get("_accepted_score_item") or {}).get("score")),
                "top2_margin": safe_float((obj.get("assignment") or {}).get("top2_margin")),
            }
            for obj in objects
        ],
        "evidence": [item.to_json() for item in evidence],
        "rejected_objects_for_lane": [obj for obj in rejected_objects if str(obj.get("top_lane_id")) == lane_id],
    }


def process_frame(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(path)
    frame = str(payload.get("frame") or path.stem)
    objects_by_lane, rejected = collect_objects_by_lane(payload, args)
    lane_decisions = []
    for lane in payload.get("lanes") or []:
        lane_id = str(lane.get("lane_id"))
        lane_decisions.append(decide_lane(frame, lane, objects_by_lane.get(lane_id, []), rejected, args))
    return {
        "schema_version": "rule_based_lane_type_decision/v1",
        "frame": frame,
        "ok": True,
        "source_association_json": str(path),
        "image_path": payload.get("image_path"),
        "lane_count": len(lane_decisions),
        "lane_decisions": lane_decisions,
        "rejected_object_count": len(rejected),
    }


def write_csv(path: Path, frame_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame",
        "lane_id",
        "lane_index",
        "lane_type",
        "lane_type_zh",
        "score_tidal",
        "score_variable",
        "score_bus",
        "score_bicycle",
        "left_boundary",
        "right_boundary",
        "accepted_objects",
        "evidence_rules",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for frame in frame_results:
            for lane in frame.get("lane_decisions", []):
                boundary = lane.get("boundary_type") or {}
                scores = lane.get("scores") or {}
                writer.writerow(
                    {
                        "frame": lane.get("frame"),
                        "lane_id": lane.get("lane_id"),
                        "lane_index": lane.get("lane_index"),
                        "lane_type": lane.get("lane_type"),
                        "lane_type_zh": lane.get("lane_type_zh"),
                        "score_tidal": scores.get("tidal", 0.0),
                        "score_variable": scores.get("variable", 0.0),
                        "score_bus": scores.get("bus", 0.0),
                        "score_bicycle": scores.get("bicycle", 0.0),
                        "left_boundary": boundary.get("left"),
                        "right_boundary": boundary.get("right"),
                        "accepted_objects": ";".join(str(obj.get("label_name")) for obj in lane.get("accepted_objects", [])),
                        "evidence_rules": ";".join(str(item.get("rule_id")) for item in lane.get("evidence", []) if item.get("accepted")),
                    }
                )


def validate_paths(args: argparse.Namespace) -> None:
    if not args.association_dir.exists():
        raise FileNotFoundError(f"association-dir not found: {args.association_dir}")
    frames_dir = args.association_dir / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"association frames dir not found: {frames_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"output-dir is not empty, pass --overwrite: {args.output_dir}")


def main() -> int:
    args = parse_args()
    args.association_dir = args.association_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    validate_paths(args)
    paths = frame_paths(args.association_dir, args.frames, args.limit)
    if not paths:
        raise FileNotFoundError(f"No association frame JSON files found under {args.association_dir / 'frames'}")
    print(f"[info] frames={len(paths)} output={args.output_dir}")
    if not args.dry_run:
        (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)

    frame_results = []
    failed = []
    for idx, path in enumerate(paths, 1):
        try:
            result = process_frame(path, args)
            frame_results.append(result)
            if not args.dry_run:
                dump_json(args.output_dir / "frames" / f"{result['frame']}.json", result)
        except Exception as exc:  # noqa: BLE001 - keep batch moving.
            print(f"[error] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if idx % 10 == 0 or idx == len(paths):
            print(f"[info] processed {idx}/{len(paths)}")

    lane_counter = Counter()
    lane_count = 0
    for frame in frame_results:
        for lane in frame.get("lane_decisions", []):
            lane_counter[str(lane.get("lane_type"))] += 1
            lane_count += 1
    summary = {
        "schema_version": "rule_based_lane_type_decision/v1",
        "created_at": utc_now(),
        "association_dir": str(args.association_dir),
        "output_dir": str(args.output_dir),
        "config": {
            "min_road_score": args.min_road_score,
            "min_road_inside_fraction": args.min_road_inside_fraction,
            "min_sign_score": args.min_sign_score,
            "min_sign_margin": args.min_sign_margin,
            "tie_margin": args.tie_margin,
            "priority": list(PRIORITY),
            "lane_type_order": list(LANE_TYPES),
            "invalid_signal_labels": sorted(INVALID_SIGNAL_LABELS),
            "both_sides_required": ["zigzag_line", "double_yellow_dash"],
        },
        "frame_count": len(paths),
        "ok_count": len(frame_results),
        "failed_count": len(failed),
        "lane_count": lane_count,
        "lane_type_counts": {name: int(lane_counter.get(name, 0)) for name in LANE_TYPES},
        "failed": failed,
        "results": frame_results,
    }
    if not args.dry_run:
        dump_json(args.output_dir / "rule_based_decision_results.json", summary)
        dump_json(args.output_dir / "_summary.json", summary)
        write_csv(args.output_dir / "lane_decisions.csv", frame_results)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] ok={len(frame_results)} failed={len(failed)} lanes={lane_count} counts={dict(lane_counter)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
