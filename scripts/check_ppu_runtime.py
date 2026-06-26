#!/usr/bin/env python3
"""Check whether the current PPU runtime can execute basic GEMM kernels.

Native PPU runtime failures can abort the Python process. This wrapper runs each
dtype check in a subprocess so the overall diagnostic can report which dtypes
are usable instead of disappearing with SIGABRT.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap


CHECK_TEMPLATE = r"""
import torch
dtype = getattr(torch, {dtype_name!r})
print("device_count", torch.cuda.device_count())
print("device0", torch.cuda.get_device_name(0))
a = torch.randn(({m}, {k}), device="cuda:0", dtype=dtype)
b = torch.randn(({k}, {n}), device="cuda:0", dtype=dtype)
c = a @ b
torch.cuda.synchronize()
print("ok", c.shape, c.dtype, float(c.float().mean().cpu()))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated PPU GEMM checks for common inference dtypes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float16", "bfloat16"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=64)
    return parser.parse_args()


def run_check(dtype_name: str, m: int, k: int, n: int) -> int:
    code = CHECK_TEMPLATE.format(dtype_name=dtype_name, m=m, k=k, n=n)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(f"==== {dtype_name} exit={proc.returncode}")
    print(proc.stdout.rstrip())
    return proc.returncode


def main() -> int:
    args = parse_args()
    failures = 0
    for dtype_name in args.dtypes:
        if run_check(dtype_name, args.m, args.k, args.n) != 0:
            failures += 1
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
