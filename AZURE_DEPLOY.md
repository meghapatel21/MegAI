# Deploying MegAI to Azure

Deployment guide for **MegAI**, by Megha Patel
([GitHub](https://github.com/meghapatel21) ·
[LinkedIn](https://linkedin.com/in/meghapatel21) ·
[Email](mailto:patelmegha1726@gmail.com)).

Two shapes, depending on how much you want to run.

---

## The one decision that matters: where uploads live

`STORAGE_BACKEND=local` writes uploaded sheets as Parquet under `DATASET_DIR`.
That is correct for local development and for a **single** always-on instance.

It **breaks the moment you have two replicas**: the upload lands on instance A,
the follow-up question is load-balanced to instance B, and B returns
`404 Dataset not found`. Azure App Service and Container Apps both scale out by
default, so before you raise the replica count:

```bash
STORAGE_BACKEND=azure
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=..."
AZURE_STORAGE_CONTAINER=agentic-bi-datasets
```

and add `azure-storage-blob` to the image (the Dockerfile already installs it).
Nothing else in the code changes — `AzureBlobDatasetStore` implements the same
four methods the agents call.

---

## Option A — Container Apps (recommended)

Backend and frontend scale independently, and the backend can scale to zero.

### 1. Resource group, registry, storage

```bash
RG=agentic-bi-rg
LOC=eastus
ACR=agenticbiacr$RANDOM
STORAGE=agenticbistore$RANDOM

az group create -n $RG -l $LOC
az acr create -g $RG -n $ACR --sku Basic --admin-enabled true
az storage account create -g $RG -n $STORAGE -l $LOC --sku Standard_LRS

CONN=$(az storage account show-connection-string -g $RG -n $STORAGE -o tsv)
az storage container create --name agentic-bi-datasets --connection-string "$CONN"
```

### 2. Build and push

```bash
az acr build -r $ACR -t agentic-bi:latest .
```

### 3. Environment and backend

```bash
az containerapp env create -g $RG -n agentic-bi-env -l $LOC

az containerapp create -g $RG -n agentic-bi-api \
  --environment agentic-bi-env \
  --image $ACR.azurecr.io/agentic-bi:latest \
  --registry-server $ACR.azurecr.io \
  --target-port 8000 --ingress external \
  --min-replicas 0 --max-replicas 5 \
  --command "uvicorn" --args "app.main:app,--host,0.0.0.0,--port,8000" \
  --secrets groq-key=$GROQ_API_KEY storage-conn="$CONN" \
  --env-vars GROQ_API_KEY=secretref:groq-key \
             STORAGE_BACKEND=azure \
             AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn \
             AZURE_STORAGE_CONTAINER=agentic-bi-datasets

API_URL=https://$(az containerapp show -g $RG -n agentic-bi-api --query properties.configuration.ingress.fqdn -o tsv)
```

### 4. Frontend

```bash
az containerapp create -g $RG -n agentic-bi-ui \
  --environment agentic-bi-env \
  --image $ACR.azurecr.io/agentic-bi:latest \
  --registry-server $ACR.azurecr.io \
  --target-port 8501 --ingress external \
  --min-replicas 1 --max-replicas 3 \
  --command "streamlit" \
  --args "run,ui/presentation_app.py,--server.port,8501,--server.address,0.0.0.0,--server.headless,true" \
  --env-vars BACKEND_URL=$API_URL
```

Streamlit holds a websocket, so set **session affinity** on the UI app if you
raise its replica count:

```bash
az containerapp ingress sticky-sessions set -g $RG -n agentic-bi-ui --affinity sticky
```

---

## Option B — App Service (single container, simplest)

```bash
az appservice plan create -g $RG -n agentic-bi-plan --is-linux --sku B2

az webapp create -g $RG -p agentic-bi-plan -n agentic-bi-app \
  --deployment-container-image-name $ACR.azurecr.io/agentic-bi:latest

az webapp config appsettings set -g $RG -n agentic-bi-app --settings \
  GROQ_API_KEY=$GROQ_API_KEY \
  STORAGE_BACKEND=azure \
  AZURE_STORAGE_CONNECTION_STRING="$CONN" \
  WEBSITES_PORT=8501 \
  WEBSITES_CONTAINER_START_TIME_LIMIT=600
```

Both processes run in one container; Streamlit reaches the API on
`http://localhost:8000`, which is the Dockerfile's `BACKEND_URL` default.

---

## Settings worth reviewing

| Variable | Default | Notes |
| :-- | :-- | :-- |
| `GROQ_API_KEY` | — | Without it the agents return a data preview instead of an analysis. Store as a secret, never an app setting in plain text. |
| `STORAGE_BACKEND` | `local` | Set to `azure` before scaling past one replica. |
| `MAX_UPLOAD_BYTES` | 500 MB | Raw/compressed upload size. App Service also caps request bodies independently - very large files need Blob direct-upload instead. Streamlit's own `server.maxUploadSize` (see `.streamlit/config.toml`) must be kept at or above this or its widget rejects the file before this limit is ever checked. |
| `MAX_ZIP_UNCOMPRESSED_BYTES` | 1.5 GB | What a `.zip` is allowed to decompress to, checked while reading - not just against the archive's own (untrusted) size metadata. |
| `MAX_ZIP_MEMBERS` | 30 | Data files per zip. |
| `MAX_ROWS_PER_SHEET` | 6,000,000 | Per-sheet safety valve, checked before the row ever reaches an agent. |
| `PROFILE_SAMPLE_ROWS` | 250,000 | Above this row count the upload briefing's *descriptive* passes (column roles, cardinality, quality checks, correlations) run on a random sample instead of every row - the single biggest cost in a large upload. Row counts, totals, breakdowns and the date range stay exact either way. `0` profiles every row. |
| `MAX_ANALYSIS_ATTEMPTS` | 3 | Each retry is another LLM call — raising this raises cost and latency. |
| `MEMORY_DIR` | `./memories` | Local Qdrant path. On multi-replica deployments each replica keeps its own memory unless you point Mem0 at a shared vector store. |
| `EMBEDDING_BACKEND` | `fastembed` | Local ONNX model, free, no key. `hashing` removes the dependency entirely; `azure` uses Azure OpenAI. |
| `RAG_DIR` | `./rag_store` | Glossary + query-memory vectors. **Same multi-replica caveat as datasets** — on Blob-backed deployments mount shared storage here, or each replica learns separately. |
| `CLARIFY_ENABLED` / `PLANNER_ENABLED` / `CRITIC_ENABLED` / `INSIGHT_ENABLED` | `true` | Each is one extra LLM call per question. Disable to cut latency and stay inside Groq free-tier rate limits. |

### Staying on the free tier

Container Apps includes a standing monthly free grant (180,000 vCPU-seconds,
360,000 GiB-seconds, 2M requests) — enough for a portfolio demo at $0, separate
from the 30-day trial credit. To stay inside it:

```bash
--min-replicas 0 --max-replicas 1 --cpu 0.5 --memory 1.0Gi
```

Scale-to-zero means a cold start on the first request. The Dockerfile bakes the
embedding model into the image specifically so that cold start does not also
have to download it.

**`1.0Gi` assumes small demo files.** The default `MAX_ZIP_UNCOMPRESSED_BYTES`
(1.5 GB) is sized for a *bigger* container than the free-tier example above —
a sheet that size plus pandas' own overhead (typically 2-4x the raw bytes once
loaded as a DataFrame with an index, dtype padding, and object overhead on any
text column) will not fit in 1 GB. Pick one:

- **Staying on the free tier:** lower the limits to match — `MAX_UPLOAD_BYTES=50000000`,
  `MAX_ZIP_UNCOMPRESSED_BYTES=150000000` is a reasonable fit for a 1 GB container.
- **Supporting large files:** raise `--memory` (e.g. `4Gi`–`6Gi` for the
  1.5 GB default) — this exceeds the free grant, so it is a real cost, scaled
  to how many queries actually run against large datasets.

Either way, the ceiling for "how large a file can this app usefully analyze"
is set by container RAM, not by upload cleverness: every question runs
generated pandas against the sheet fully loaded in memory, the same as
uploading does. Zip support fixes the *transport* problem (getting a large,
compressed export there in one request) and upload no longer holding 2-3x the
file in memory at once mid-parse — it does not turn this into a big-data
engine. Multi-GB+ datasets need a different execution engine (DuckDB or Polars
lazy queries, so analysis never fully materializes the sheet) — a separate,
larger change from what zip support does here.

## Known limitations

- **Generated code has no execution timeout.** The Impact Agent blocks unsafe
  *operations*, but a pathological `merge` on a large sheet can still occupy a
  worker. Cap `MAX_ROWS_PER_SHEET` and set container CPU limits accordingly.
- **Datasets are never garbage-collected.** Add a Blob lifecycle rule (delete
  after N days) or a timer job calling `DELETE /datasets/{id}`.
- **No authentication.** Anyone who reaches the ingress can upload and query.
  Put Azure AD / Easy Auth in front of both apps before exposing them.
