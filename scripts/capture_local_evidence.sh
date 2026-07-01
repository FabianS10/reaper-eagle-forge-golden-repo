#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-forge_local_evidence}"
mkdir -p "$OUT_DIR/environment" "$OUT_DIR/benchmark" "$OUT_DIR/integrity"

(date -u +"%Y-%m-%dT%H:%M:%SZ") > "$OUT_DIR/captured_at_utc.txt"
python --version > "$OUT_DIR/environment/python_version.txt" 2>&1 || true
python src/pytorch_rocm_smoke_test.py > "$OUT_DIR/environment/pytorch_rocm_smoke_test.txt" 2>&1 || true
python src/forge_portable_benchmark.py --output "$OUT_DIR/benchmark/benchmark_results.json" > "$OUT_DIR/benchmark/benchmark_stdout.txt" 2> "$OUT_DIR/benchmark/benchmark_stderr.txt" || true

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$OUT_DIR" && find . -type f ! -path "./integrity/*" -print0 | sort -z | xargs -0 sha256sum > integrity/sha256_manifest.txt)
fi

echo "Evidence captured in $OUT_DIR"
