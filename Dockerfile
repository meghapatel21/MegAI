# Single image running both the FastAPI backend and the Streamlit frontend.
# On a platform that scales them independently you would usually split these
# into two containers and point the frontend at the backend with BACKEND_URL.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="MegAI"       org.opencontainers.image.description="Natural language analytics over uploaded spreadsheets, driven by a 10-agent LangGraph pipeline."       org.opencontainers.image.authors="Megha Patel <patelmegha1726@gmail.com>"       org.opencontainers.image.source="https://github.com/meghapatel21/agentic-bi-natural-language-querying"       org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code edits do not invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir azure-storage-blob

# Bake the embedding model into the image. Without this the first request after
# a scale-from-zero downloads ~50MB from HuggingFace and times out the probe.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed
RUN mkdir -p /app/.fastembed && \
    python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY app ./app
COPY ui ./ui

# Runtime assets the UI opens by path. Without these the container starts and
# serves, but with broken graphics and a silently reduced upload cap:
#   .streamlit/config.toml  - theme AND server.maxUploadSize=500. Missing, it
#                             falls back to Streamlit's 200 MB widget default,
#                             so 200-500 MB uploads are rejected before the
#                             app's own MAX_UPLOAD_BYTES check ever runs.
#   Logo/                   - header, sidebar and favicon images.
#   agentic_bi_architecture.jpeg - shown on the Architecture tab.
#   all_questions_list.txt  - the sidebar "Download All Prompts" button.
COPY .streamlit ./.streamlit
COPY Logo ./Logo
COPY agentic_bi_architecture.jpeg all_questions_list.txt ./

# Uploaded datasets land here when STORAGE_BACKEND=local. Mount a volume, or
# set STORAGE_BACKEND=azure, if you run more than one replica.
RUN mkdir -p /app/datasets /app/memories /app/rag_store
ENV DATASET_DIR=/app/datasets \
    MEMORY_DIR=/app/memories \
    RAG_DIR=/app/rag_store \
    BACKEND_URL=http://localhost:8000

# Run as a non-root user. Hugging Face Spaces runs every container as uid
# 1000, so the paths created by root above would be read-only at runtime and
# the first upload would fail with a permission error. Chowning /app covers
# the dataset, memory, RAG and fastembed-cache directories in one step.
# Good practice on Azure too - Container Apps and App Service both run
# non-root images without extra configuration.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & exec streamlit run ui/presentation_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
