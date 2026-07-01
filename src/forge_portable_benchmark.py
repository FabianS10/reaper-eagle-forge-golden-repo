"""Portable TinyModel benchmark for Forge ML evidence capture.

This is the safe path. It still uses PyTorch's `cuda` device type when an AMD
GPU is exposed through ROCm-enabled PyTorch, because that is how PyTorch presents
compatible accelerator backends. Forge should distinguish this compatibility API
from NVIDIA-only project assumptions such as nvidia-smi, TensorRT, CUDA base
images, or hard-coded CUDA dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Sequence

import torch


class TinyModel(torch.nn.Module):
    def __init__(self, width: int = 1024, hidden: int = 2048) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backend_label(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    if getattr(torch.version, "hip", None):
        return "rocm_via_torch_cuda_api"
    return "cuda_or_other_torch_cuda_api"


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile over an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((q / 100) * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--precision", default="fp32", choices=["fp32"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", default="manifests/benchmark_results.json")
    args = parser.parse_args()

    if args.runs < 2:
        raise ValueError("--runs must be at least 2 so the report is not a single-sample benchmark")
    if args.warmup < 1:
        raise ValueError("--warmup must be at least 1")

    torch.manual_seed(args.seed)
    device = pick_device()
    model = TinyModel().to(device).eval()
    x = torch.randn(args.batch_size, 1024, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)
        sync_if_needed(device)

        latencies = []
        for _ in range(args.runs):
            start = time.perf_counter()
            _ = model(x)
            sync_if_needed(device)
            latencies.append((time.perf_counter() - start) * 1000)

    total_seconds = sum(latencies) / 1000
    result = {
        "benchmark_name": "forge_portable_tiny_mlp",
        "capture_mode": "live_manual_run",
        "device_backend": backend_label(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname_present": bool(os.environ.get("HOSTNAME") or platform.node()),
        "batch_size": args.batch_size,
        "precision": args.precision,
        "seed": args.seed,
        "warmup_count": args.warmup,
        "timed_runs": args.runs,
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "throughput_items_per_second": (args.batch_size * args.runs) / total_seconds,
        "latencies_ms": latencies,
        "claim_status": "live_result_requires_environment_manifest_before_external performance claims",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
