#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-forge_mi300x_evidence}"
mkdir -p "$OUT_DIR/environment" "$OUT_DIR/benchmark" "$OUT_DIR/profiler" "$OUT_DIR/integrity"

(date -u +"%Y-%m-%dT%H:%M:%SZ") > "$OUT_DIR/captured_at_utc.txt"

rocminfo > "$OUT_DIR/environment/rocminfo.txt" 2>&1 || true
rocm-smi > "$OUT_DIR/environment/rocm_smi.txt" 2>&1 || true
amd-smi list > "$OUT_DIR/environment/amd_smi.txt" 2>&1 || true
hipcc --version > "$OUT_DIR/environment/hipcc_version.txt" 2>&1 || true
python --version > "$OUT_DIR/environment/python_version.txt" 2>&1 || true
env | grep -E "ROCM|HIP|HSA|ROCR|CUDA|PYTORCH|VLLM|RCCL" > "$OUT_DIR/environment/env_vars_redacted.txt" 2>&1 || true

python src/pytorch_rocm_smoke_test.py > "$OUT_DIR/environment/pytorch_rocm_smoke_test.txt" 2>&1 || true
python src/forge_portable_benchmark.py --batch-size 32 --warmup 5 --runs 30 --precision fp32 --output "$OUT_DIR/benchmark/benchmark_results.json" > "$OUT_DIR/benchmark/benchmark_stdout.txt" 2> "$OUT_DIR/benchmark/benchmark_stderr.txt" || true

if command -v rocprofv3 >/dev/null 2>&1; then
  rocprofv3 --stats -o "$OUT_DIR/profiler/rocprofv3" python src/forge_portable_benchmark.py --batch-size 32 --warmup 3 --runs 10 --precision fp32 --output "$OUT_DIR/benchmark/benchmark_results_profiled.json" > "$OUT_DIR/profiler/rocprofv3_stdout.txt" 2> "$OUT_DIR/profiler/rocprofv3_stderr.txt" || true
else
  echo "rocprofv3 not available" > "$OUT_DIR/profiler/rocprofv3_summary.txt"
fi

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$OUT_DIR" && find . -type f ! -path "./integrity/*" -print0 | sort -z | xargs -0 sha256sum > integrity/sha256_manifest.txt)
fi

echo "MI300X evidence captured in $OUT_DIR"
