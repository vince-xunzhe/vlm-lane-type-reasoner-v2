#!/usr/bin/env python3
"""Run prompt-by-image VLM inference and save per-image results.

The script is designed for Qwen-style vision-language checkpoints loaded from a
local Hugging Face model directory. It reads every prompt file in a prompt
directory, runs each prompt against every image in an image directory, and writes
results under:

    <output_dir>/<prompt_name>/<relative_image_name>.json

Text sidecar files can also be written for quick inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_PATH = "/nas/nfs/large-model/vince/model/Qwen3.6-27B-PER-SFT-260529"
DEFAULT_IMAGE_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test/images"
DEFAULT_OUTPUT_DIR = "/nas/nfs/large-model/vince/data/xd-online-las-data/test/inference"
DEFAULT_PROMPT_DIR = "prompt-perception"
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEFAULT_PROMPT_EXTENSIONS = (".txt", ".md", ".json")


@dataclass(frozen=True)
class PromptSpec:
    name: str
    path: str
    text: str


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float


@dataclass(frozen=True)
class VisionInputConfig:
    min_pixels: int | None
    max_pixels: int | None
    resized_height: int | None
    resized_width: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch VLM inference for all prompt files and all images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "--model-path", "--model_path", dest="model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--image-dir",
        "--image_dir",
        "--image-folder",
        "--image_folder",
        "--image-file",
        "--image_file",
        "--image-path",
        "--image_path",
        dest="image_dir",
        default=DEFAULT_IMAGE_DIR,
        help="Image directory or a single image file.",
    )
    parser.add_argument(
        "--image-root",
        "--image_root",
        default=None,
        help="Optional directory used only for preserving relative output paths.",
    )
    parser.add_argument(
        "--prompt-dir",
        "--prompt_dir",
        "--prompt-file",
        "--prompt_file",
        "--prompt-path",
        "--prompt_path",
        dest="prompt_dir",
        default=DEFAULT_PROMPT_DIR,
        help="Prompt directory or a single prompt file.",
    )
    parser.add_argument(
        "--prompt-root",
        "--prompt_root",
        default=None,
        help="Optional directory used only for deriving prompt output folder names.",
    )
    parser.add_argument("--output-dir", "--output_dir", "--output-folder", "--output_folder", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTENSIONS),
        help="Image file extensions to include.",
    )
    parser.add_argument(
        "--prompt-extensions",
        nargs="+",
        default=list(DEFAULT_PROMPT_EXTENSIONS),
        help="Prompt file extensions to include.",
    )
    parser.add_argument(
        "--max-length",
        "--max_length",
        dest="max_length",
        type=int,
        default=0,
        help="Processor truncation length. Use <=0 to keep the model/processor default.",
    )
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0 for deterministic decoding; values >0 enable sampling.",
    )
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--torch-dtype",
        "--torch_dtype",
        dest="torch_dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        "--device_map",
        dest="device_map",
        default="auto",
        help="Transformers device_map. Use 'none' to omit the argument.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional explicit device, e.g. cuda:0. If set, model.to(device) is called after load.",
    )
    parser.add_argument(
        "--attn-implementation",
        "--attn_implementation",
        dest="attn_implementation",
        default=None,
        help="Optional Transformers attention implementation, e.g. flash_attention_2.",
    )
    parser.add_argument(
        "--min-pixels",
        "--min_pixels",
        dest="min_pixels",
        type=int,
        default=None,
        help="Optional Qwen-VL minimum image pixels for dynamic resizing.",
    )
    parser.add_argument(
        "--max-pixels",
        "--max_pixels",
        dest="max_pixels",
        type=int,
        default=None,
        help="Optional Qwen-VL maximum image pixels for dynamic resizing.",
    )
    parser.add_argument(
        "--resized-height",
        "--resized_height",
        dest="resized_height",
        type=int,
        default=None,
        help="Optional fixed resized image height passed to Qwen-VL utilities.",
    )
    parser.add_argument(
        "--resized-width",
        "--resized_width",
        dest="resized_width",
        type=int,
        default=None,
        help="Optional fixed resized image width passed to Qwen-VL utilities.",
    )
    parser.add_argument("--limit", "--limit-per-task", "--limit_per_task", dest="limit", type=int, default=None, help="Limit images per prompt.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many images before applying --limit.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument(
        "--write-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a .txt sidecar with only the model response.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered prompts/images and planned output layout.",
    )
    parser.add_argument(
        "--fail-fast",
        "--fail_fast",
        dest="fail_fast",
        action="store_true",
        help="Stop at the first image/prompt inference error.",
    )
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=1)
    return parser.parse_args()


def normalize_extensions(values: Iterable[str]) -> set[str]:
    return {value.lower() if value.startswith(".") else f".{value.lower()}" for value in values}


def sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    clean = clean.strip("._")
    return clean or "prompt"


def prompt_name(prompt_path: Path, prompt_root: Path) -> str:
    try:
        relative = prompt_path.relative_to(prompt_root)
    except ValueError:
        relative = Path(prompt_path.name)
    without_suffix = relative.with_suffix("")
    return sanitize_name("__".join(without_suffix.parts))


def read_prompt_file(path: Path) -> str:
    raw = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() != ".json":
        return raw

    payload = json.loads(raw)
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("prompt", "text", "content", "instruction"):
            value = payload.get(key)
            if isinstance(value, str):
                return value.strip()
        if "messages" in payload:
            return json.dumps(payload["messages"], ensure_ascii=False, indent=2)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def discover_prompts(prompt_dir: Path, extensions: set[str], prompt_root_override: Path | None = None) -> list[PromptSpec]:
    if prompt_dir.is_file():
        prompt_paths = [prompt_dir]
        prompt_root = prompt_root_override or prompt_dir.parent
    else:
        prompt_root = prompt_root_override or prompt_dir
        prompt_paths = sorted(
            path for path in prompt_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )

    prompts = [
        PromptSpec(name=prompt_name(path, prompt_root), path=str(path), text=read_prompt_file(path))
        for path in prompt_paths
    ]
    return [prompt for prompt in prompts if prompt.text]


def discover_images(image_dir: Path, extensions: set[str], limit: int | None, offset: int = 0) -> list[Path]:
    if image_dir.is_file():
        images = [image_dir] if image_dir.suffix.lower() in extensions else []
    else:
        images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    if offset > 0:
        images = images[offset:]
    if limit is not None and limit > 0:
        return images[:limit]
    return images


def output_paths(output_dir: Path, prompt: PromptSpec, image_path: Path, image_root: Path) -> tuple[Path, Path]:
    relative = Path(image_path.name) if image_root.is_file() else image_path.relative_to(image_root)
    relative_parent = relative.parent if str(relative.parent) != "." else Path()
    base_name = f"{relative.name}.json"
    json_path = output_dir / prompt.name / relative_parent / base_name
    text_path = json_path.with_suffix(".txt")
    return json_path, text_path


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return mapping[dtype_name]


def load_processor_and_model(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install torch, transformers, pillow, and qwen-vl-utils if needed."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": resolve_torch_dtype(torch, args.torch_dtype),
    }
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    processor = transformers.AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    candidate_class_names = (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2VLForConditionalGeneration",
        "AutoModelForCausalLM",
    )
    errors: list[str] = []
    for class_name in candidate_class_names:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is None:
            continue
        try:
            model = model_cls.from_pretrained(args.model_path, **model_kwargs)
            model.eval()
            return processor, model
        except TypeError as exc:
            retry_variants: list[dict[str, Any]] = []
            if "attn_implementation" in model_kwargs:
                retry_kwargs = dict(model_kwargs)
                retry_kwargs.pop("attn_implementation", None)
                retry_variants.append(retry_kwargs)
            if "dtype" in model_kwargs:
                retry_kwargs = dict(model_kwargs)
                retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
                retry_variants.append(retry_kwargs)

            for retry_kwargs in retry_variants:
                try:
                    model = model_cls.from_pretrained(args.model_path, **retry_kwargs)
                    model.eval()
                    return processor, model
                except Exception as retry_exc:  # noqa: BLE001 - report all candidate failures.
                    errors.append(f"{class_name}: {type(retry_exc).__name__}: {retry_exc}")
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")
        except (ValueError, ImportError, AttributeError) as exc:
            errors.append(f"{class_name}: {type(exc).__name__}: {exc}")

    joined_errors = "\n".join(f"- {error}" for error in errors)
    raise RuntimeError(f"Unable to load model from {args.model_path}. Tried:\n{joined_errors}")


def pick_input_device(model: Any) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for value in device_map.values():
            if isinstance(value, int):
                return f"cuda:{value}"
            if isinstance(value, str) and value not in {"cpu", "disk", "meta"}:
                return value
    if hasattr(model, "device"):
        return str(model.device)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def build_messages(prompt_text: str, image_path: Path, vision_config: VisionInputConfig) -> list[dict[str, Any]]:
    image_payload: dict[str, Any] = {"type": "image", "image": str(image_path)}
    image_payload.update({key: value for key, value in asdict(vision_config).items() if value is not None})

    return [
        {
            "role": "user",
            "content": [
                image_payload,
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def prepare_inputs(processor: Any, messages: list[dict[str, Any]], input_device: str, max_length: int | None) -> Any:
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        process_vision_info = None

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    processor_kwargs: dict[str, Any] = {
        "padding": True,
        "return_tensors": "pt",
    }
    if max_length is not None and max_length > 0:
        processor_kwargs.update({"truncation": True, "max_length": max_length})

    if process_vision_info is not None:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            **processor_kwargs,
        )
    else:
        from PIL import Image

        image_path = Path(messages[0]["content"][0]["image"])
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text], images=[image], **processor_kwargs)

    return inputs.to(input_device)


def run_single_inference(
    processor: Any,
    model: Any,
    prompt: PromptSpec,
    image_path: Path,
    generation_config: GenerationConfig,
    vision_config: VisionInputConfig,
    input_device: str,
    max_length: int | None,
) -> str:
    import torch

    messages = build_messages(prompt.text, image_path, vision_config)
    inputs = prepare_inputs(processor, messages, input_device, max_length)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": generation_config.max_new_tokens,
        "repetition_penalty": generation_config.repetition_penalty,
    }
    if generation_config.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": generation_config.temperature,
                "top_p": generation_config.top_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generate_kwargs["pad_token_id"] = eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generate_kwargs)

    input_token_count = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_token_count:]
    decoded = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()


def write_result(
    json_path: Path,
    text_path: Path,
    response: str,
    prompt: PromptSpec,
    image_path: Path,
    model_path: str,
    generation_config: GenerationConfig,
    vision_config: VisionInputConfig,
    max_length: int | None,
    elapsed_seconds: float,
    write_text: bool,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "model_path": model_path,
        "prompt_name": prompt.name,
        "prompt_path": prompt.path,
        "image_path": str(image_path),
        "response": response,
        "generation_config": asdict(generation_config),
        "vision_input_config": asdict(vision_config),
        "max_length": max_length,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if write_text:
        text_path.write_text(response + "\n", encoding="utf-8")


def write_error(
    json_path: Path,
    prompt: PromptSpec,
    image_path: Path,
    model_path: str,
    generation_config: GenerationConfig,
    vision_config: VisionInputConfig,
    max_length: int | None,
    error: BaseException,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "error",
        "model_path": model_path,
        "prompt_name": prompt.name,
        "prompt_path": prompt.path,
        "image_path": str(image_path),
        "error_type": type(error).__name__,
        "error": str(error),
        "generation_config": asdict(generation_config),
        "vision_input_config": asdict(vision_config),
        "max_length": max_length,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_paths(model_path: Path, prompt_dir: Path, image_dir: Path) -> None:
    missing = []
    if not model_path.exists():
        missing.append(f"model path: {model_path}")
    if not prompt_dir.exists():
        missing.append(f"prompt dir/file: {prompt_dir}")
    if not image_dir.exists():
        missing.append(f"image dir: {image_dir}")
    if missing:
        lines = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(f"Required path(s) not found:\n{lines}")


def build_vision_config(args: argparse.Namespace) -> VisionInputConfig:
    if (args.resized_height is None) != (args.resized_width is None):
        raise ValueError("--resized-height and --resized-width must be provided together.")
    return VisionInputConfig(
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        resized_height=args.resized_height,
        resized_width=args.resized_width,
    )


def main() -> int:
    args = parse_args()

    model_path = Path(args.model_path).expanduser()
    prompt_dir = Path(args.prompt_dir).expanduser()
    image_dir = Path(args.image_dir).expanduser()
    prompt_root = Path(args.prompt_root).expanduser() if args.prompt_root else None
    image_root = Path(args.image_root).expanduser() if args.image_root else image_dir
    output_dir = Path(args.output_dir).expanduser()

    validate_paths(model_path, prompt_dir, image_dir)

    prompts = discover_prompts(prompt_dir, normalize_extensions(args.prompt_extensions), prompt_root)
    images = discover_images(image_dir, normalize_extensions(args.image_extensions), args.limit, args.offset)
    if not prompts:
        raise RuntimeError(f"No prompt files found in {prompt_dir}")
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    planned_count = len(prompts) * len(images)
    print(f"model_path={model_path}")
    print(f"prompt_dir={prompt_dir} prompts={len(prompts)}")
    print(f"image_dir={image_dir} images={len(images)}")
    print(f"output_dir={output_dir}")
    print(f"planned_inferences={planned_count}")

    if args.dry_run:
        for prompt in prompts:
            example_json, _ = output_paths(output_dir, prompt, images[0], image_root)
            print(f"prompt={prompt.name} example_output={example_json}")
        return 0

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    vision_config = build_vision_config(args)

    processor, model = load_processor_and_model(args)
    if args.device:
        model.to(args.device)
    input_device = args.device or pick_input_device(model)
    print(f"input_device={input_device}")

    completed = 0
    skipped = 0
    failed = 0
    total = planned_count

    for prompt in prompts:
        print(f"running_prompt={prompt.name}")
        for image_path in images:
            json_path, text_path = output_paths(output_dir, prompt, image_path, image_root)
            if args.resume and json_path.exists() and not args.overwrite:
                skipped += 1
                continue

            index = completed + skipped + failed + 1
            if args.log_every > 0 and (index == 1 or index % args.log_every == 0):
                print(f"[{index}/{total}] {prompt.name} <- {image_path}", flush=True)
            start = time.perf_counter()
            try:
                response = run_single_inference(
                    processor=processor,
                    model=model,
                    prompt=prompt,
                    image_path=image_path,
                    generation_config=generation_config,
                    vision_config=vision_config,
                    input_device=input_device,
                    max_length=args.max_length,
                )
                elapsed = time.perf_counter() - start
                write_result(
                    json_path=json_path,
                    text_path=text_path,
                    response=response,
                    prompt=prompt,
                    image_path=image_path,
                    model_path=str(model_path),
                    generation_config=generation_config,
                    vision_config=vision_config,
                    max_length=args.max_length,
                    elapsed_seconds=elapsed,
                    write_text=args.write_text,
                )
                completed += 1
            except Exception as exc:  # noqa: BLE001 - persist per-image failures and continue.
                failed += 1
                write_error(json_path, prompt, image_path, str(model_path), generation_config, vision_config, args.max_length, exc)
                print(f"ERROR {prompt.name} <- {image_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
                if args.fail_fast:
                    raise

    print(f"done completed={completed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
