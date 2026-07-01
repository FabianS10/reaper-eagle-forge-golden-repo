# Reaper Eagle Forge Golden Repo

Synthetic PyTorch repository for demonstrating **Reaper Eagle Forge ML**.

This repository is intentionally designed as a clean demo target: it contains one safe portable benchmark path and one clearly isolated NVIDIA-shaped fixture path. Forge should be able to scan the repository, detect the portability and benchmark-discipline risks, and still show that the repo has a valid AMD/ROCm-ready remediation path.

## Demo thesis

> The AMD environment may be viable while the repository is blocked by NVIDIA-shaped assumptions, weak benchmark discipline, or incomplete evidence.

Forge should not merely ask, "does this code mention CUDA?" On ROCm-enabled PyTorch, AMD GPUs may still appear through the `torch.cuda` API. The real audit is whether the repository depends on NVIDIA-only tooling, invalid timing, missing evidence, or overclaimed performance.

## Repository layout

```text
.
├── src/
│   ├── nvidia_locked_benchmark.py      # intentionally bad static-analysis fixture
│   └── forge_portable_benchmark.py     # portable benchmark path
├── manifests/
│   ├── benchmark_results.example.json  # replay example, not a live claim
│   └── evidence_manifest.example.json  # evidence capsule example
├── requirements.txt                    # safe portable benchmark dependency file
├── requirements-bad.txt                # commented vendor-lock-in fixture, not installable
├── Dockerfile                          # safe default container path
├── Dockerfile.nvidia                   # intentionally NVIDIA-shaped fixture, not production
├── docker-compose.yml
└── SUBMISSION_SCOPE.md
```

## What Forge should detect

The intentionally bad path contains:

- direct `.cuda()` calls;
- `torch.cuda` direct usage without a portability explanation;
- `nvidia-smi` telemetry dependency;
- optional TensorRT import;
- CUDA Dockerfile assumptions;
- benchmark timing without warm-up;
- timing without GPU synchronization;
- single-sample latency reporting;
- no p50/p95 latency reporting;
- no precision declaration;
- no evidence manifest attached to the bad claim.

## Safe benchmark path

Install the safe dependency set:

```bash
pip install -r requirements.txt
```

Run the portable benchmark:

```bash
python src/forge_portable_benchmark.py \
  --batch-size 32 \
  --warmup 5 \
  --runs 30 \
  --precision fp32 \
  --output manifests/benchmark_results.json
```

Run through Docker:

```bash
docker compose up --build
```

## Important safety note

This repository is for static-analysis and benchmark-demonstration purposes. Forge ML should **not execute arbitrary repository code** during Repo Scan mode. The portable benchmark is provided for manual golden-path validation only.

The files `requirements-bad.txt`, `Dockerfile.nvidia`, and `src/nvidia_locked_benchmark.py` are fixtures. They exist so Forge can detect vendor lock-in and benchmark-theater patterns. They are not production entrypoints.

## Suggested demo sequence

1. Scan this repo with Forge ML.
2. Show CUDA/NVIDIA lock-in findings from `src/nvidia_locked_benchmark.py`, `requirements-bad.txt`, and `Dockerfile.nvidia`.
3. Show that the default repo path is containerized and portable.
4. Run Forge's own Live Environment Check on the backend.
5. Manually run `src/forge_portable_benchmark.py` during the MI300X capture session to produce real benchmark evidence.
6. Upload the captured JSON/logs into the Forge Evidence Replay capsule.

## Claim discipline

Allowed claim for this repo:

> This repository demonstrates how Forge separates a risky NVIDIA-shaped benchmark path from a safer portable benchmark path and produces an evidence-ready audit surface.

Blocked claim:

> This repository proves AMD MI300X is faster than another GPU.

That claim would require live comparative runs, controlled hardware profiles, repeated trials, synchronized timing, p50/p95 reporting, and a signed evidence manifest.
