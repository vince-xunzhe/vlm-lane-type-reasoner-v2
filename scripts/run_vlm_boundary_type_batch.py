#!/usr/bin/env python3
"""Run boundary-type inference for each referenced center line.

The boundary-type task is different from ordinary prompt-by-image tasks: each
image can contain multiple referenced center lines. This runner expands each
center_line_2d JSON file into one inference item per lane, builds a prompt by
replacing the template placeholder with that lane's normalized image
coordinates, and runs each item in an isolated child process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_vlm_batch_inference as core  # noqa: E402


PROMPT_NAME = "prompt-classification-boundary-type"
DEFAULT_IMAGE_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/images"
DEFAULT_CENTER_LINE_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/center_line_2d"
DEFAULT_OUTPUT_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference"
DEFAULT_PROMPT_FILE = "prompt-perception/prompt-classification-boundary-type.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM boundary-type inference for every lane in center_line_2d JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--model-path", "--model_path", dest="model_path", type=Path, default=Path(core.DEFAULT_MODEL_PATH))
    parser.add_argument("--image-dir", "--image_dir", "--image-folder", "--image_folder", dest="image_dir", type=Path, default=Path(DEFAULT_IMAGE_DIR))
    parser.add_argument("--center-line-dir", "--center_line_dir", "--center-line-root", "--center_line_root", dest="center_line_dir", type=Path, default=Path(DEFAULT_CENTER_LINE_DIR))
    parser.add_argument("--prompt-file", "--prompt_file", "--prompt-path", "--prompt_path", dest="prompt_file", type=Path, default=Path(DEFAULT_PROMPT_FILE))
    parser.add_argument("--output-dir", "--output_dir", "--output-folder", "--output_folder", dest="output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image-extensions", "--image_extensions", dest="image_extensions", nargs="+", default=list(core.DEFAULT_IMAGE_EXTENSIONS))
    parser.add_argument("--limit", "--limit-per-task", "--limit_per_task", dest="limit_per_task", type=int, default=0, help="Limit center-line JSON files.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many center-line JSON files.")
    parser.add_argument("--line-offset", "--line_offset", dest="line_offset", type=int, default=0, help="Skip this many lanes per image.")
    parser.add_argument("--max-lines-per-image", "--max_lines_per_image", dest="max_lines_per_image", type=int, default=0)
    parser.add_argument("--coordinate-mode", "--coordinate_mode", dest="coordinate_mode", choices=("pixel", "normalized"), default="pixel")
    parser.add_argument("--prompt-placeholder", "--prompt_placeholder", dest="prompt_placeholder", default="[xxx]")
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=0)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty", type=float, default=1.0)
    parser.add_argument("--torch-dtype", "--torch_dtype", dest="torch_dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", "--device_map", dest="device_map", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", "--attn_implementation", dest="attn_implementation", default=None)
    parser.add_argument("--min-pixels", "--min_pixels", dest="min_pixels", type=int, default=None)
    parser.add_argument("--max-pixels", "--max_pixels", dest="max_pixels", type=int, default=None)
    parser.add_argument("--resized-height", "--resized_height", dest="resized_height", type=int, default=None)
    parser.add_argument("--resized-width", "--resized_width", dest="resized_width", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", "--fail_fast", dest="fail_fast", action="store_true")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=1)
    parser.add_argument("--log-tail-lines", "--log_tail_lines", dest="log_tail_lines", type=int, default=120)
    parser.add_argument("--summary-file", "--summary_file", dest="summary_file", type=Path, default=None)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--run-single-line", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--work-item-file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    missing = []
    for label, path in (
        ("model", args.model_path),
        ("image dir", args.image_dir),
        ("center-line dir/file", args.center_line_dir),
        ("prompt file", args.prompt_file),
    ):
        if not path.expanduser().exists():
            missing.append(f"{label}: {path}")
    if missing:
        lines = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Required path(s) not found:\n{lines}")


def center_line_files(path: Path, offset: int, limit: int) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        files = [path]
    else:
        files = sorted(item for item in path.glob("*.json") if item.is_file())
    if offset > 0:
        files = files[offset:]
    if limit > 0:
        files = files[:limit]
    return files


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_image_path(data: Any, json_path: Path, image_dir: Path, extensions: Iterable[str]) -> Path | None:
    image_value = data.get("image") if isinstance(data, dict) else None
    if isinstance(image_value, str) and image_value.strip():
        image_path = Path(image_value)
        if image_path.is_absolute() and image_path.exists():
            return image_path
        candidate = image_dir / image_path
        if candidate.exists():
            return candidate

    for suffix in core.normalize_extensions(extensions):
        candidate = image_dir / f"{json_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def image_size(image_path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def raw_lane_items(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("lane", "lanes", "center_lines", "center_line", "lines", "objects", "annotations"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def point_from_value(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        for key in ("point", "pt", "coord", "xy"):
            if key in value:
                return point_from_value(value[key])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def points_from_lane(item: Any) -> list[tuple[float, float]]:
    if isinstance(item, dict):
        for key in ("points", "polyline", "centerline", "center_line", "line", "coords"):
            value = item.get(key)
            if isinstance(value, list):
                points = [point for point in (point_from_value(raw) for raw in value) if point is not None]
                if points:
                    return points
    if isinstance(item, list):
        return [point for point in (point_from_value(raw) for raw in item) if point is not None]
    return []


def lane_id(item: Any, index: int) -> str:
    if isinstance(item, dict):
        for key in ("id", "lane_id", "line_id", "track_id", "uuid"):
            value = item.get(key)
            if value is not None:
                return str(value)
    return f"{index:03d}"


def clamp_0_1000(value: float) -> int:
    return max(0, min(1000, int(round(value))))


def normalize_points(
    points: list[tuple[float, float]],
    size: tuple[int, int],
    mode: str,
) -> list[list[int]]:
    width, height = size
    if mode == "normalized":
        return [[clamp_0_1000(x), clamp_0_1000(y)] for x, y in points]
    return [[clamp_0_1000(x / width * 1000.0), clamp_0_1000(y / height * 1000.0)] for x, y in points]


def build_prompt(template: str, coordinates: list[list[int]], placeholder: str) -> str:
    coordinate_text = json.dumps(coordinates, ensure_ascii=False, separators=(",", ":"))
    if placeholder in template:
        return template.replace(placeholder, coordinate_text)
    return f"{template.rstrip()}\nReferenced center-line normalized image coordinates: {coordinate_text}"


def output_paths(output_dir: Path, image_path: Path, line_id: str, line_index: int) -> tuple[Path, Path, Path]:
    image_slug = core.sanitize_name(image_path.stem)
    line_slug = core.sanitize_name(f"{line_index:03d}_{line_id}")
    json_path = output_dir / PROMPT_NAME / image_slug / f"{line_slug}.json"
    text_path = json_path.with_suffix(".txt")
    work_item_path = output_dir / "_boundary_work_items" / image_slug / f"{line_slug}.json"
    return json_path, text_path, work_item_path


def build_work_items(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    template = args.prompt_file.expanduser().read_text(encoding="utf-8").strip()
    work_items: list[dict[str, Any]] = []
    skipped_inputs: list[dict[str, Any]] = []
    for center_json in center_line_files(args.center_line_dir, args.offset, args.limit_per_task):
        data = load_json(center_json)
        image_path = resolve_image_path(data, center_json, args.image_dir.expanduser(), args.image_extensions)
        if image_path is None:
            skipped_inputs.append({"center_line_json": str(center_json), "reason": "missing_image"})
            continue
        size = image_size(image_path)
        lane_items = raw_lane_items(data)
        if args.line_offset > 0:
            lane_items = lane_items[args.line_offset :]
        if args.max_lines_per_image > 0:
            lane_items = lane_items[: args.max_lines_per_image]
        if not lane_items:
            skipped_inputs.append({"center_line_json": str(center_json), "image_path": str(image_path), "reason": "no_lane"})
            continue

        for local_index, item in enumerate(lane_items, args.line_offset):
            points_raw = points_from_lane(item)
            if not points_raw:
                skipped_inputs.append(
                    {
                        "center_line_json": str(center_json),
                        "image_path": str(image_path),
                        "lane_index": local_index,
                        "reason": "empty_points",
                    }
                )
                continue
            item_line_id = lane_id(item, local_index)
            points_norm = normalize_points(points_raw, size, args.coordinate_mode)
            prompt_text = build_prompt(template, points_norm, args.prompt_placeholder)
            json_path, text_path, work_item_path = output_paths(args.output_dir.expanduser(), image_path, item_line_id, local_index)
            work_items.append(
                {
                    "prompt_name": PROMPT_NAME,
                    "prompt_template_path": str(args.prompt_file.expanduser()),
                    "center_line_json_path": str(center_json),
                    "image_path": str(image_path),
                    "image_size": list(size),
                    "lane_id": item_line_id,
                    "lane_index": local_index,
                    "center_line_points_raw": [[float(x), float(y)] for x, y in points_raw],
                    "center_line_points": points_norm,
                    "coordinate_mode": args.coordinate_mode,
                    "prompt_text": prompt_text,
                    "output_json_path": str(json_path),
                    "output_text_path": str(text_path),
                    "work_item_path": str(work_item_path),
                }
            )
    return work_items, skipped_inputs


def generation_config(args: argparse.Namespace) -> core.GenerationConfig:
    return core.GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )


def vision_config(args: argparse.Namespace) -> core.VisionInputConfig:
    return core.VisionInputConfig(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
    )


def write_success(
    work_item: dict[str, Any],
    response: str,
    args: argparse.Namespace,
    elapsed_seconds: float,
) -> None:
    json_path = Path(work_item["output_json_path"])
    text_path = Path(work_item["output_text_path"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "model_path": str(args.model_path),
        "prompt_name": PROMPT_NAME,
        "prompt_template_path": work_item["prompt_template_path"],
        "center_line_json_path": work_item["center_line_json_path"],
        "image_path": work_item["image_path"],
        "image_size": work_item["image_size"],
        "lane_id": work_item["lane_id"],
        "lane_index": work_item["lane_index"],
        "center_line_points": work_item["center_line_points"],
        "center_line_points_raw": work_item["center_line_points_raw"],
        "coordinate_mode": work_item["coordinate_mode"],
        "response": response,
        "generation_config": asdict(generation_config(args)),
        "vision_input_config": asdict(vision_config(args)),
        "max_length": args.max_length,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "created_at": utc_now(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.write_text:
        text_path.write_text(response + "\n", encoding="utf-8")


def write_error(
    work_item: dict[str, Any],
    args: argparse.Namespace,
    error_type: str,
    error: str,
    returncode: int | None = None,
    child_output: str = "",
    command: list[str] | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    json_path = Path(work_item["output_json_path"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "error",
        "model_path": str(args.model_path),
        "prompt_name": PROMPT_NAME,
        "prompt_template_path": work_item["prompt_template_path"],
        "center_line_json_path": work_item["center_line_json_path"],
        "image_path": work_item["image_path"],
        "image_size": work_item["image_size"],
        "lane_id": work_item["lane_id"],
        "lane_index": work_item["lane_index"],
        "center_line_points": work_item["center_line_points"],
        "center_line_points_raw": work_item["center_line_points_raw"],
        "coordinate_mode": work_item["coordinate_mode"],
        "error_type": error_type,
        "error": error,
        "returncode": returncode,
        "generation_config": asdict(generation_config(args)),
        "vision_input_config": asdict(vision_config(args)),
        "max_length": args.max_length,
        "child_command": command,
        "child_output_tail": tail_lines(child_output, args.log_tail_lines),
        "elapsed_seconds": round(elapsed_seconds, 4) if elapsed_seconds is not None else None,
        "created_at": utc_now(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tail_lines(text: str, line_count: int) -> str:
    if line_count <= 0:
        return ""
    return "\n".join(text.splitlines()[-line_count:])


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def single_line_command(args: argparse.Namespace, work_item_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-single-line",
        "--work-item-file",
        str(work_item_path),
        "--model",
        str(args.model_path),
        "--prompt-file",
        str(args.prompt_file),
        "--image-dir",
        str(args.image_dir),
        "--center-line-dir",
        str(args.center_line_dir),
        "--output-dir",
        str(args.output_dir),
        "--max-length",
        str(args.max_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
        "--log-every",
        "0",
    ]
    cmd.extend(["--image-extensions", *args.image_extensions])
    for name, value in (
        ("--device", args.device),
        ("--attn-implementation", args.attn_implementation),
        ("--min-pixels", args.min_pixels),
        ("--max-pixels", args.max_pixels),
        ("--resized-height", args.resized_height),
        ("--resized-width", args.resized_width),
    ):
        if value is not None:
            cmd.extend([name, str(value)])
    if args.overwrite:
        cmd.append("--overwrite")
    if not args.resume:
        cmd.append("--no-resume")
    if not args.write_text:
        cmd.append("--no-write-text")
    return cmd


def run_single_line(args: argparse.Namespace) -> int:
    if args.work_item_file is None:
        raise ValueError("--work-item-file is required with --run-single-line")
    work_item = load_json(args.work_item_file.expanduser())
    args.model_path = args.model_path.expanduser()
    prompt = core.PromptSpec(
        name=PROMPT_NAME,
        path=work_item["prompt_template_path"],
        text=work_item["prompt_text"],
    )
    gen_config = generation_config(args)
    vis_config = vision_config(args)
    processor, model = core.load_processor_and_model(args)
    if args.device:
        model.to(args.device)
    input_device = args.device or core.pick_input_device(model)
    start = time.perf_counter()
    try:
        response = core.run_single_inference(
            processor=processor,
            model=model,
            prompt=prompt,
            image_path=Path(work_item["image_path"]),
            generation_config=gen_config,
            vision_config=vis_config,
            input_device=input_device,
            max_length=args.max_length,
        )
        write_success(work_item, response, args, time.perf_counter() - start)
        return 0
    except Exception as exc:  # noqa: BLE001 - persist line-level Python failures.
        write_error(
            work_item=work_item,
            args=args,
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.perf_counter() - start,
        )
        print(f"ERROR {work_item['image_path']} lane={work_item['lane_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def run_batch(args: argparse.Namespace) -> int:
    args.model_path = args.model_path.expanduser()
    args.image_dir = args.image_dir.expanduser()
    args.center_line_dir = args.center_line_dir.expanduser()
    args.prompt_file = args.prompt_file.expanduser()
    args.output_dir = args.output_dir.expanduser()
    validate_paths(args)

    work_items, skipped_inputs = build_work_items(args)
    summary_path = args.summary_file.expanduser() if args.summary_file else args.output_dir / "_boundary_type_summary.json"

    print(f"model={args.model_path}", flush=True)
    print(f"image_dir={args.image_dir}", flush=True)
    print(f"center_line_dir={args.center_line_dir}", flush=True)
    print(f"prompt_file={args.prompt_file}", flush=True)
    print(f"output_dir={args.output_dir}", flush=True)
    print(f"planned_line_inferences={len(work_items)} skipped_inputs={len(skipped_inputs)}", flush=True)

    if args.dry_run:
        if work_items:
            first = work_items[0]
            print(f"example_output={first['output_json_path']}", flush=True)
            print(f"example_center_line_points={first['center_line_points']}", flush=True)
        print(f"summary_file={summary_path}", flush=True)
        return 0

    started_at = utc_now()
    start_time = time.perf_counter()
    completed = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, Any]] = []

    for index, work_item in enumerate(work_items, 1):
        output_json = Path(work_item["output_json_path"])
        if args.resume and output_json.exists() and not args.overwrite:
            skipped += 1
            continue

        work_item_path = Path(work_item["work_item_path"])
        work_item_path.parent.mkdir(parents=True, exist_ok=True)
        work_item_path.write_text(json.dumps(work_item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.log_every > 0 and (index == 1 or index % args.log_every == 0):
            print(f"[{index}/{len(work_items)}] {work_item['image_path']} lane={work_item['lane_id']}", flush=True)

        cmd = single_line_command(args, work_item_path)
        child_start = time.perf_counter()
        try:
            completed_process = subprocess.run(
                cmd,
                check=False,
                cwd=str(Path.cwd()),
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=None if args.timeout <= 0 else args.timeout,
            )
            elapsed = time.perf_counter() - child_start
            if completed_process.returncode == 0:
                completed += 1
                continue

            failed += 1
            failure = {
                "image_path": work_item["image_path"],
                "lane_id": work_item["lane_id"],
                "lane_index": work_item["lane_index"],
                "output_path": work_item["output_json_path"],
                "returncode": completed_process.returncode,
            }
            failures.append(failure)
            if not output_json.exists():
                write_error(
                    work_item=work_item,
                    args=args,
                    error_type="ChildProcessError",
                    error=f"subprocess exited with return code {completed_process.returncode}",
                    returncode=completed_process.returncode,
                    child_output=completed_process.stdout or "",
                    command=cmd,
                    elapsed_seconds=elapsed,
                )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - child_start
            failed += 1
            output = exc.stdout or ""
            output_text = output if isinstance(output, str) else output.decode(errors="replace")
            failure = {
                "image_path": work_item["image_path"],
                "lane_id": work_item["lane_id"],
                "lane_index": work_item["lane_index"],
                "output_path": work_item["output_json_path"],
                "returncode": None,
                "error_type": "TimeoutExpired",
            }
            failures.append(failure)
            write_error(
                work_item=work_item,
                args=args,
                error_type="TimeoutExpired",
                error=f"subprocess timed out after {args.timeout} seconds",
                child_output=output_text,
                command=cmd,
                elapsed_seconds=elapsed,
            )

        if args.fail_fast or not args.continue_on_error:
            break

    summary = {
        "status": "ok" if failed == 0 and not skipped_inputs else "partial",
        "model_path": str(args.model_path),
        "image_dir": str(args.image_dir),
        "center_line_dir": str(args.center_line_dir),
        "prompt_file": str(args.prompt_file),
        "output_dir": str(args.output_dir),
        "planned_line_inferences": len(work_items),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "skipped_inputs": skipped_inputs,
        "failures": failures,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
    }
    write_summary(summary_path, summary)
    print(f"done completed={completed} skipped={skipped} failed={failed} summary={summary_path}", flush=True)
    return 0 if failed == 0 else 2


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.run_single_line:
        return run_single_line(args)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
