# Reaper Eagle Forge Golden Repo

Synthetic PyTorch repository for demonstrating **Reaper Eagle Forge ML**.

This repository is intentionally split into two paths:

1. `src/nvidia_locked_benchmark.py` — intentionally NVIDIA-shaped code for static analysis.
2. `src/forge_portable_benchmark.py` — portable benchmark path intended to run on ROCm-enabled PyTorch, NVIDIA CUDA PyTorch, or CPU fallback.

The purpose is to prove the Forge demo arc:

> The AMD environment can be viable while the repository is blocked by NVIDIA-shaped assumptions.

## What Forge should detect

The bad path intentionally contains:

- direct `.cuda()` calls;
- `torch.cuda` direct usage without context;
- `nvidia-smi` telemetry dependency;
- optional TensorRT import;
- CUDA Dockerfile assumptions;
- benchmark timing without warm-up;
- timing without GPU synchronization;
- no p50/p95 latency reporting;
- no precision declaration.

## Safe benchmark path

Run the portable benchmark:

```bash
python src/forge_portable_benchmark.py --batch-size 32 --warmup 5 --runs 30 --precision fp32 --output manifests/benchmark_results.json
```

On ROCm-enabled PyTorch, AMD GPUs may still appear through the `torch.cuda` API. This is expected. The point is not to panic at the word `cuda`; the real task is distinguishing PyTorch's compatibility API from NVIDIA-only project assumptions.

## Important safety note

This repository is for static-analysis and benchmark-demonstration purposes. Forge ML should **not execute arbitrary repository code** during Repo Scan mode. The portable benchmark is provided for manual golden-path validation only.

## Suggested demo sequence

1. Scan this repo with Forge ML.
2. Show CUDA/NVIDIA lock-in findings from `src/nvidia_locked_benchmark.py`, `requirements-bad.txt`, and `Dockerfile.nvidia`.
3. Run Forge's own Live Environment Check on the backend.
4. Manually run `src/forge_portable_benchmark.py` during the MI300X capture session to produce real benchmark evidence.
5. Upload the captured JSON/logs into the Forge Evidence Replay capsule.
