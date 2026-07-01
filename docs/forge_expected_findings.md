# Expected Forge Findings

## Execution blockers

- `CUDA_DEVICE_HARDCODED` in `src/nvidia_locked_benchmark.py`
- `TORCH_CUDA_DIRECT_DEVICE` in `src/nvidia_locked_benchmark.py`
- `TENSORRT_DEPENDENCY` in `src/nvidia_locked_benchmark.py`
- `CUDA_EXTENSION_BUILD` / `NVCC_REQUIRED` may be inferred from `requirements-bad.txt` or build files depending on scanner rules.

## Portability gaps

- `NVIDIA_SMI_DEPENDENCY` in `src/nvidia_locked_benchmark.py`
- `CUDA_DOCKER_BASE` or `NVIDIA_DOCKER_HINT` in `Dockerfile.nvidia`
- `CUPY_CUDA_DEPENDENCY` in `requirements-bad.txt`

## Benchmark discipline failures

- `NO_WARMUP_POLICY` in `src/nvidia_locked_benchmark.py`
- `NO_SYNCHRONIZATION_BEFORE_TIME` in `src/nvidia_locked_benchmark.py`
- `NO_P50_P95_LATENCY` in `src/nvidia_locked_benchmark.py`
- `NO_PRECISION_DECLARATION` in `src/nvidia_locked_benchmark.py`

## Expected narrative

The bad path should score low because the repo is NVIDIA-shaped. The portable benchmark demonstrates the intended migration path and can produce captured evidence on ROCm-enabled PyTorch.
