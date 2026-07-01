"""Intentionally NVIDIA-shaped benchmark for Reaper Eagle Forge ML static analysis.

This file is not meant to be a good benchmark. It is meant to trigger Forge's
CUDA/NVIDIA lock-in and benchmark-discipline findings.
"""

import subprocess
import time

import torch

try:
    import tensorrt as trt  # noqa: F401 - intentionally vendor-specific
except Exception:
    trt = None


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    # Telemetry portability gap: this is NVIDIA-specific and should be abstracted.
    subprocess.run(["nvidia-smi"], check=False)

    # Execution blocker / portability risk: direct device movement without abstraction.
    model = TinyModel().cuda()
    x = torch.randn(32, 1024).cuda()

    # Invalid benchmark discipline: no warm-up, no synchronization, one timing sample.
    start = time.time()
    y = model(x)
    end = time.time()

    print("latency_seconds:", end - start)
    print("output_shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
