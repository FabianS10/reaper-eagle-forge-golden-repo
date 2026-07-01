FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY src ./src
COPY manifests ./manifests

CMD ["python", "src/forge_portable_benchmark.py", "--batch-size", "8", "--warmup", "2", "--runs", "5", "--output", "manifests/benchmark_results.json"]
