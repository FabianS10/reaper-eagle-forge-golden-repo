"""Portable TinyModel benchmark for Forge ML evidence capture."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

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


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--output", default="manifests/benchmark_results.json")
    args = parser.parse_args()

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

    result = {
        "benchmark_name": "forge_portable_tiny_mlp",
        "device_type": device.type,
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "batch_size": args.batch_size,
        "precision": args.precision,
        "warmup_count": args.warmup,
        "timed_runs": args.runs,
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1],
        "throughput_items_per_second": (args.batch_size * args.runs) / (sum(latencies) / 1000),
        "latencies_ms": latencies,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
