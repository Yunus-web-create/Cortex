FROM python:3.12-slim

# One image for both the laptop (`docker run`) and Render. It stays small, runs as a
# non-root user, and bakes the two ONNX retrieval models into the image so a cold start
# never waits on a 130 MB download.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    FASTEMBED_CACHE_PATH=/opt/models \
    CORTEX_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Embeddings + reranker, warmed into the image. Non-fatal: if Hugging Face is
# unreachable during the build the app still starts and fetches them on first use.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')" \
    || echo "model cache skipped - will download at runtime"

COPY . .

RUN mkdir -p /app/data /opt/models && useradd -m cortex && chown -R cortex:cortex /app /opt/models
USER cortex

# Render injects PORT (10000 by default); run.py honours it. EXPOSE is documentation.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD curl -fsS "http://localhost:${PORT:-8000}/api/health" || exit 1

CMD ["python", "run.py"]
