#!/usr/bin/env python3
"""Organize stacked dev visualizations by special lane type."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = Path("/nas/nfs/large-model/vince/data/xd-online-las-data/test-v3")
SPECIAL_TYPES = ("bicycle", "bus", "tidal", "variable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy stacked dev visualization frames into per-lane-type major folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", "--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--decision-dir", "--decision_dir", type=Path, default=None)
    parser.add_argument("--stack-dir", "--stack_dir", type=Path, default=None)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument("--frames", type=str, default="", help="Comma-separated frame ids or a text file with one id per line.")
    parser.add_argument("--copy-mode", "--copy_mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
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


def lane_type_frames(decision_dir: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    out = {lane_type: {"frames": set(), "lane_count": 0, "lanes": []} for lane_type in SPECIAL_TYPES}
    decision_paths = sorted((decision_dir / "frames").glob("*.json"))
    for path in decision_paths:
        frame = path.stem
        if wanted and frame not in wanted:
            continue
        payload = load_json(path)
        for lane in payload.get("lane_decisions") or []:
            lane_type = str(lane.get("lane_type") or "")
            if lane_type not in out:
                continue
            out[lane_type]["frames"].add(frame)
            out[lane_type]["lane_count"] += 1
            out[lane_type]["lanes"].append(
                {
                    "frame": frame,
                    "lane_id": lane.get("lane_id"),
                    "lane_index": lane.get("lane_index"),
                    "lane_type": lane_type,
                    "decision_reason": lane.get("decision_reason"),
                    "scores": lane.get("scores"),
                }
            )
    return out


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def write_category_index(output_dir: Path, lane_type: str, records: list[dict[str, Any]]) -> None:
    figures = []
    for record in records:
        rel = Path(record["output"]).relative_to(output_dir / lane_type)
        frame = html.escape(str(record.get("frame")))
        figures.append(f"<figure><a href='{html.escape(str(rel))}'><img src='{html.escape(str(rel))}'></a><figcaption>{frame}</figcaption></figure>")
    body = "".join(figures) if figures else "<p>No frames for this lane type.</p>"
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(lane_type)} Major Visualization</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #111827; padding: 10px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    figcaption {{ margin-top: 8px; color: #94a3b8; }}
    a {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <p><a href="../index.html">Back to major index</a></p>
  <h1>{html.escape(lane_type)} Major Visualization</h1>
  <p>Generated at {html.escape(utc_now())}. Frames: {len(records)}.</p>
  <div class="grid">{body}</div>
</body>
</html>
"""
    (output_dir / lane_type / "index.html").write_text(html_text, encoding="utf-8")


def write_major_index(output_dir: Path, summary: dict[str, Any]) -> None:
    cards = []
    for lane_type in SPECIAL_TYPES:
        info = summary["categories"][lane_type]
        cards.append(
            "<section>"
            f"<h2><a href='{html.escape(lane_type)}/index.html'>{html.escape(lane_type)}</a></h2>"
            f"<p>frames={info['frame_count']}, lanes={info['lane_count']}</p>"
            "</section>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Major Lane Type Visualization</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    section {{ background: #111827; padding: 16px; border-radius: 8px; }}
    a {{ color: #93c5fd; }}
  </style>
</head>
<body>
  <h1>Major Lane Type Visualization</h1>
  <p>Generated at {html.escape(utc_now())}. Source stack dir: {html.escape(summary['stack_dir'])}</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.expanduser()
    decision_dir = (args.decision_dir or args.data_dir / "inference" / "decision").expanduser()
    stack_dir = (args.stack_dir or args.data_dir / "vis" / "frames").expanduser()
    output_dir = (args.output_dir or args.data_dir / "vis" / "major").expanduser()
    wanted = parse_frames_arg(args.frames)

    if not (decision_dir / "frames").exists():
        raise FileNotFoundError(f"Decision frames directory not found: {decision_dir / 'frames'}")
    if not stack_dir.exists():
        raise FileNotFoundError(f"Stack visualization frames directory not found: {stack_dir}")
    if output_dir.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(output_dir)

    categorized = lane_type_frames(decision_dir, wanted)
    summary = {
        "schema_version": "major_dev_stack_by_lane_type/v1",
        "created_at": utc_now(),
        "decision_dir": str(decision_dir),
        "stack_dir": str(stack_dir),
        "output_dir": str(output_dir),
        "copy_mode": args.copy_mode,
        "categories": {},
    }

    for lane_type in SPECIAL_TYPES:
        category_dir = output_dir / lane_type
        records = []
        missing_frames = []
        for frame in sorted(categorized[lane_type]["frames"]):
            src = stack_dir / f"{frame}.jpg"
            dst = category_dir / f"{frame}.jpg"
            if not src.exists():
                missing_frames.append(frame)
                continue
            if not args.dry_run:
                copy_or_link(src, dst, args.copy_mode)
            records.append({"frame": frame, "source": str(src), "output": str(dst)})
        if not args.dry_run:
            category_dir.mkdir(parents=True, exist_ok=True)
            write_category_index(output_dir, lane_type, records)
            dump_json(
                category_dir / "_summary.json",
                {
                    "lane_type": lane_type,
                    "frame_count": len(records),
                    "lane_count": categorized[lane_type]["lane_count"],
                    "missing_frame_count": len(missing_frames),
                    "missing_frames": missing_frames,
                    "lanes": categorized[lane_type]["lanes"],
                    "records": records,
                },
            )
        summary["categories"][lane_type] = {
            "frame_count": len(records),
            "lane_count": categorized[lane_type]["lane_count"],
            "missing_frame_count": len(missing_frames),
            "dir": str(category_dir),
        }

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        dump_json(output_dir / "_summary.json", summary)
        write_major_index(output_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
