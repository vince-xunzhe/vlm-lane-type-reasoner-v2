# VLM Grounding Reasoner

Batch VLM inference for perception prompts.

Remote project path:

```bash
/nas/nfs/large-model/vince/code/vlm-grounding-reasoner-release-v1
```

Default model path:

```bash
/nas/nfs/large-model/vince/model/Qwen3.6-27B-PER-SFT-260529
```

Default sample data:

```bash
/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/images
```

Default output:

```bash
/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/inference
```

## Run

Put prompt files under `prompt-perception/`. Supported prompt extensions are
`.txt`, `.md`, and `.json`.

On the remote machine:

```bash
cd /nas/nfs/large-model/vince/code/vlm-grounding-reasoner-release-v1
bash scripts/run_xd_online_las_test.sh
```

`run_xd_online_las_test.sh` uses `scripts/run_vlm_batch_images.py`, which
runs regular prompts against every image. It skips
`prompt-classification-boundary-type`, because that task needs one inference
per referenced center line rather than one inference per image. Each
prompt-image pair is isolated in a child process, so a native runtime abort on
one image is recorded as an error JSON and the rest of the batch can continue.

Run boundary-type separately:

```bash
bash scripts/run_xd_online_las_boundary_type.sh --dry-run
bash scripts/run_xd_online_las_boundary_type.sh
```

`run_xd_online_las_boundary_type.sh` reads center-line reference JSON files
from:

```bash
/nas/nfs/large-model/vince/data/xd-online-las-data/test-v2/center_line_2d
```

For every image, it expands each `lane` entry into a separate prompt by
replacing `[xxx]` in `prompt-classification-boundary-type.txt` with the lane's
0-1000 normalized image coordinates.

Useful options:

```bash
# Preview prompt/image discovery and output layout.
bash scripts/run_xd_online_las_test.sh --dry-run

# Run only the first 10 images per prompt.
bash scripts/run_xd_online_las_test.sh --limit-per-task 10

# Skip the first 20 discovered images, then run 10 images per prompt.
bash scripts/run_xd_online_las_test.sh --offset 20 --limit-per-task 10

# Regenerate outputs that already exist.
bash scripts/run_xd_online_las_test.sh --overwrite

# Reduce visual tokens if a specific image shape hits a PPU offline-cache issue.
bash scripts/run_xd_online_las_test.sh --max-pixels 262144
```

The batch runner accepts benchmark-style aliases from
`perception_benchmark_infer.py`, including `--model`, `--output-dir`,
`--max-length`, `--max-new-tokens`, `--temperature`, `--resume/--no-resume`,
`--overwrite`, `--offset`, `--limit-per-task`, and `--log-every`.

Before running full inference on the PPU machine, sanity-check the runtime:

```bash
python scripts/check_ppu_runtime.py
```

If this reports aborts for `bfloat16` or missing offline cache files, the PPU
runtime/cache needs to be repaired before the 27B bf16 model can generate.

Output layout:

```text
<output_dir>/
  _batch_summary.json
  <prompt_name>/
    <image_name>.jpg.json
    <image_name>.jpg.txt
  prompt-classification-boundary-type/
    <image_name>/
      <lane_index>_<lane_id>.json
      <lane_index>_<lane_id>.txt
  _boundary_type_summary.json
```

Each JSON file stores the prompt path, image path, raw model response,
generation config, elapsed time, and status.
