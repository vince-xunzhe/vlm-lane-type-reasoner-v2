#!/usr/bin/env python3
"""Robust batch image runner for prompt-by-image VLM inference.

This script is the batch entrypoint. It discovers all prompts and images, then
invokes run_vlm_batch_inference.py once per prompt-image pair. Isolating each
image in a child process keeps a native runtime abort from stopping the whole
batch, while preserving the same output layout:

    <output_dir>/<prompt_name>/<relative_image_name>.json
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
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_vlm_batch_inference as core  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM perception prompts over a batch of images with per-image process isolation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--model-path", "--model_path", dest="model_path", type=Path, default=Path(core.DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-dir", "--prompt_dir", dest="prompt_dir", type=Path, default=Path(core.DEFAULT_PROMPT_DIR))
    parser.add_argument("--prompt-file", "--prompt_file", "--prompt-path", "--prompt_path", dest="prompt_file", type=Path, default=None)
    parser.add_argument("--prompt-root", "--prompt_root", dest="prompt_root", type=Path, default=None)
    parser.add_argument(
        "--image-dir",
        "--image_dir",
        "--image-folder",
        "--image_folder",
        dest="image_dir",
        type=Path,
        default=Path(core.DEFAULT_IMAGE_DIR),
    )
    parser.add_argument("--image-file", "--image_file", "--image-path", "--image_path", dest="image_file", type=Path, default=None)
    parser.add_argument("--image-root", "--image_root", dest="image_root", type=Path, default=None)
    parser.add_argument("--output-dir", "--output_dir", "--output-folder", "--output_folder", dest="output_dir", type=Path, default=Path(core.DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--image-extensions",
        "--image_extensions",
        dest="image_extensions",
        nargs="+",
        default=list(core.DEFAULT_IMAGE_EXTENSIONS),
    )
    parser.add_argument(
        "--prompt-extensions",
        "--prompt_extensions",
        dest="prompt_extensions",
        nargs="+",
        default=list(core.DEFAULT_PROMPT_EXTENSIONS),
    )
    parser.add_argument(
        "--exclude-prompt-names",
        "--exclude_prompt_names",
        dest="exclude_prompt_names",
        nargs="+",
        default=[],
        help="Prompt folder names to skip, e.g. prompt-classification-boundary-type.",
    )
    parser.add_argument("--limit", "--limit-per-task", "--limit_per_task", dest="limit_per_task", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--max-length",
        "--max_length",
        dest="max_length",
        type=int,
        default=0,
        help="Processor truncation length. Use <=0 to keep the model/processor default.",
    )
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
    parser.add_argument("--timeout", type=int, default=0, help="Per prompt-image subprocess timeout in seconds; <=0 disables it.")
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=1)
    parser.add_argument("--log-tail-lines", "--log_tail_lines", dest="log_tail_lines", type=int, default=120)
    parser.add_argument("--summary-file", "--summary_file", dest="summary_file", type=Path, default=None)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser.parse_args()


def expand_path(path: Path | None) -> Path | None:
    return path.expanduser() if path is not None else None


def validate_paths(model_path: Path, prompt_source: Path, image_source: Path) -> None:
    missing = []
    if not model_path.exists():
        missing.append(f"model: {model_path}")
    if not prompt_source.exists():
        missing.append(f"prompt source: {prompt_source}")
    if not image_source.exists():
        missing.append(f"image source: {image_source}")
    if missing:
        lines = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Required path(s) not found:\n{lines}")


def resolve_sources(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    model_path = args.model_path.expanduser()
    prompt_source = (args.prompt_file or args.prompt_dir).expanduser()
    image_source = (args.image_file or args.image_dir).expanduser()
    prompt_root = expand_path(args.prompt_root) or (prompt_source if prompt_source.is_dir() else prompt_source.parent)
    image_root = expand_path(args.image_root) or (image_source if image_source.is_dir() else image_source.parent)
    output_dir = args.output_dir.expanduser()
    return model_path, prompt_source, prompt_root, image_source, image_root, output_dir


def discover_work(args: argparse.Namespace, prompt_source: Path, prompt_root: Path, image_source: Path) -> tuple[list[core.PromptSpec], list[Path]]:
    prompts = core.discover_prompts(
        prompt_source,
        core.normalize_extensions(args.prompt_extensions),
        prompt_root,
    )
    images = core.discover_images(
        image_source,
        core.normalize_extensions(args.image_extensions),
        limit=None,
        offset=0,
    )
    if args.offset > 0:
        images = images[args.offset :]
    if args.limit_per_task > 0:
        images = images[: args.limit_per_task]
    excluded = set(args.exclude_prompt_names)
    if excluded:
        prompts = [prompt for prompt in prompts if prompt.name not in excluded and Path(prompt.path).stem not in excluded]
    return prompts, images


def append_optional(cmd: list[str], name: str, value: Any) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def child_command(
    args: argparse.Namespace,
    model_path: Path,
    prompt: core.PromptSpec,
    prompt_root: Path,
    image_path: Path,
    image_root: Path,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_vlm_batch_inference.py"),
        "--model",
        str(model_path),
        "--prompt-file",
        prompt.path,
        "--prompt-root",
        str(prompt_root),
        "--image-file",
        str(image_path),
        "--image-root",
        str(image_root),
        "--output-dir",
        str(output_dir),
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
    append_optional(cmd, "--device", args.device)
    append_optional(cmd, "--attn-implementation", args.attn_implementation)
    append_optional(cmd, "--min-pixels", args.min_pixels)
    append_optional(cmd, "--max-pixels", args.max_pixels)
    append_optional(cmd, "--resized-height", args.resized_height)
    append_optional(cmd, "--resized-width", args.resized_width)
    if args.overwrite:
        cmd.append("--overwrite")
    if not args.resume:
        cmd.append("--no-resume")
    if not args.write_text:
        cmd.append("--no-write-text")
    return cmd


def tail_lines(text: str, line_count: int) -> str:
    if line_count <= 0:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


def write_child_error(
    json_path: Path,
    prompt: core.PromptSpec,
    image_path: Path,
    model_path: Path,
    args: argparse.Namespace,
    command: list[str],
    error_type: str,
    error: str,
    returncode: int | None,
    output: str,
    elapsed_seconds: float,
) -> None:
    generation_config = core.GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    vision_config = core.VisionInputConfig(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "error",
        "model_path": str(model_path),
        "prompt_name": prompt.name,
        "prompt_path": prompt.path,
        "image_path": str(image_path),
        "error_type": error_type,
        "error": error,
        "returncode": returncode,
        "generation_config": asdict(generation_config),
        "vision_input_config": asdict(vision_config),
        "max_length": args.max_length,
        "child_command": command,
        "child_output_tail": tail_lines(output, args.log_tail_lines),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "created_at": utc_now(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path, prompt_source, prompt_root, image_source, image_root, output_dir = resolve_sources(args)
    validate_paths(model_path, prompt_source, image_source)
    prompts, images = discover_work(args, prompt_source, prompt_root, image_source)
    if not prompts:
        raise RuntimeError(f"No prompt files found in {prompt_source}")
    if not images:
        raise RuntimeError(f"No images found in {image_source}")

    planned = len(prompts) * len(images)
    summary_path = args.summary_file.expanduser() if args.summary_file else output_dir / "_batch_summary.json"

    print(f"model={model_path}", flush=True)
    print(f"prompt_source={prompt_source} prompts={len(prompts)}", flush=True)
    print(f"image_source={image_source} images={len(images)} offset={args.offset} limit_per_task={args.limit_per_task}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"planned_inferences={planned}", flush=True)

    if args.dry_run:
        for prompt in prompts:
            example_json, _ = core.output_paths(output_dir, prompt, images[0], image_root)
            print(f"prompt={prompt.name} example_output={example_json}", flush=True)
        print(f"summary_file={summary_path}", flush=True)
        return 0

    started_at = utc_now()
    start_time = time.perf_counter()
    completed = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, Any]] = []

    for prompt in prompts:
        print(f"running_prompt={prompt.name}", flush=True)
        for image_path in images:
            index = completed + skipped + failed + 1
            json_path, _ = core.output_paths(output_dir, prompt, image_path, image_root)
            if args.resume and json_path.exists() and not args.overwrite:
                skipped += 1
                continue

            if args.log_every > 0 and (index == 1 or index % args.log_every == 0):
                print(f"[{index}/{planned}] {prompt.name} <- {image_path}", flush=True)

            cmd = child_command(args, model_path, prompt, prompt_root, image_path, image_root, output_dir)
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
                child_elapsed = time.perf_counter() - child_start
                if completed_process.returncode == 0:
                    completed += 1
                    continue

                failed += 1
                failure = {
                    "prompt_name": prompt.name,
                    "image_path": str(image_path),
                    "output_path": str(json_path),
                    "returncode": completed_process.returncode,
                }
                failures.append(failure)
                write_child_error(
                    json_path=json_path,
                    prompt=prompt,
                    image_path=image_path,
                    model_path=model_path,
                    args=args,
                    command=cmd,
                    error_type="ChildProcessError",
                    error=f"subprocess exited with return code {completed_process.returncode}",
                    returncode=completed_process.returncode,
                    output=completed_process.stdout or "",
                    elapsed_seconds=child_elapsed,
                )
            except subprocess.TimeoutExpired as exc:
                child_elapsed = time.perf_counter() - child_start
                failed += 1
                output = exc.stdout or ""
                failure = {
                    "prompt_name": prompt.name,
                    "image_path": str(image_path),
                    "output_path": str(json_path),
                    "returncode": None,
                    "error_type": "TimeoutExpired",
                }
                failures.append(failure)
                write_child_error(
                    json_path=json_path,
                    prompt=prompt,
                    image_path=image_path,
                    model_path=model_path,
                    args=args,
                    command=cmd,
                    error_type="TimeoutExpired",
                    error=f"subprocess timed out after {args.timeout} seconds",
                    returncode=None,
                    output=output if isinstance(output, str) else output.decode(errors="replace"),
                    elapsed_seconds=child_elapsed,
                )

            if args.fail_fast or not args.continue_on_error:
                break
        if (args.fail_fast or not args.continue_on_error) and failed > 0:
            break

    summary = {
        "status": "ok" if failed == 0 else "partial",
        "model_path": str(model_path),
        "prompt_source": str(prompt_source),
        "prompt_root": str(prompt_root),
        "image_source": str(image_source),
        "image_root": str(image_root),
        "output_dir": str(output_dir),
        "planned": planned,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
        "failures": failures,
    }
    write_summary(summary_path, summary)
    print(f"done completed={completed} skipped={skipped} failed={failed} summary={summary_path}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
