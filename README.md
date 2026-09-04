# MegAI — Enterprise Natural Language Analytics

<div align="center">

  <p align="center">
    <b>A Production-Grade Generative BI Platform Powered by LangGraph & Multi-Agent Orchestration</b>
  </p>

  <p align="center">
    Built by <b>Megha Patel</b> &nbsp;·&nbsp;
    <a href="https://github.com/meghapatel21">GitHub</a> &nbsp;·&nbsp;
    <a href="https://linkedin.com/in/meghapatel21">LinkedIn</a> &nbsp;·&nbsp;
    <a href="mailto:patelmegha1726@gmail.com">Email</a>
  </p>

  [![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![LangChain](https://img.shields.io/badge/LangChain-Integration-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com/)
  [![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-FF9900)](https://langchain-ai.github.io/langgraph/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange?logo=git-lfs&logoColor=white)](https://git-lfs.github.com/)
</div>

---

## Table of Contents

- [Live Demo](#live-demo)
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [The RAG Layer](#the-rag-layer)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Installation & Setup](#installation--setup)
- [API](#api)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [License](#license)
- [Contact](#contact)

---

## Live Demo

**[agenticbi.streamlit.app](https://agenticbi.streamlit.app/)**

Upload a spreadsheet and ask a question in plain English — no setup required.
To run it on your own machine instead, see [Installation & Setup](#installation--setup).

---

## Overview

**Agentic BI** turns a spreadsheet into a conversation. Upload an Excel or CSV
file and ask questions about it in plain English — *"which department overspent
last quarter?"* — and a pipeline of AI agents profiles your columns, writes
pandas code to answer the question, audits that code for safety, runs it, and
renders the result as KPIs and charts.

There is **no database and no fixed schema**. The agents read the real column
names, dtypes, and sample values out of whatever file you upload, so the system
works on your data on the first run with no setup.

---

## System Architecture

```
  upload .xlsx / .csv                    optional: business glossary
          │                                        │
          ▼                                        ▼
   ┌──────────────┐                        ┌───────────────┐
   │ DatasetStore │ sheets → Parquet       │  Vector index │ definitions + past
   │              │ columns profiled       │  (FAISS/NumPy)│ successful analyses
   └──────┬───────┘                        └───────┬───────┘
          │ dataset_id                             │ retrieval
          ▼                                        ▼
  ┌──────────────────────────── LangGraph ─────────────────────────────┐
  │                                                                    │
  │  profile ─▶ clarify ─┬─(ambiguous)────────────────────▶ bi ─▶ END  │
  │                      │                                             │
  │                      └▶ rag ─▶ planner ─▶ analysis ─▶ impact       │
  │                                             ▲            │         │
  │                                             │            ▼         │
  │                          insight ◀─ advance ◀─ critic ◀─ execute   │
  │                             │          ▲         │          │      │
  │                             ▼          └─────────┴──────────┘      │
  │                            bi              reflexion on error      │
  └────────────────────────────────────────────────────────────────────┘
```

### The agents

| Agent | What it actually does |
| :-- | :-- |
| **1. Profile** | Loads the workbook profile, then rewrites the question against the real column names — fixing typos and expanding shorthand. |
| **2. Clarify** | Asks back when a question is genuinely ambiguous, offering concrete options built from the schema. Deliberately conservative: a sensible default beats an interruption. |
| **3. RAG** | Retrieves the organisation's certified definitions and the most similar past analyses. See below. |
| **4. Planner** | Splits compound questions into ordered steps, each able to build on the last via `prev`. Defaults to one step, because most questions need one. |
| **5. Analysis** | Generates pandas constrained to the profiled columns, grounded by what the RAG Agent retrieved. On a retry it receives the previous code *and its error*. |
| **6. Impact** | Governance gate. Audits the generated code's AST and rejects imports, file/network access, dunder escapes, and any call that would write data out. Nothing reaches `exec` unaudited. |
| **7. Execute** | Runs approved code against in-memory DataFrames in a restricted namespace with a minimal builtin surface. |
| **8. Critic** | Catches code that *ran* but did not answer — empty results, all-NaN columns, a single row where a breakdown was asked for. Can send the work back for another attempt. |
| **9. Insight** | Writes the analyst narrative over verified numbers. Cites specific figures, and is forbidden from asserting causes it cannot see. |
| **10. BI** | Assembles KPIs, chart form, provenance, and the final response. |

Three agents can send work back to Analysis through **one** reflexion edge, all
via the same `state["error"]` field: **Impact** (failed the audit), **Execute**
(the code raised), and **Critic** (the code ran but did not answer). Every loop
is bounded by an attempt budget, so the graph always terminates.

---

## The RAG layer

Two retrievers, because they answer different questions.

### 1. Business glossary — *curated, authoritative*

How your company defines "ARR" or "active customer" is a policy decision, not
something a model should re-derive per question. Upload a glossary once (Markdown,
text, CSV, or Excel) and matching definitions are injected so the generated code
follows **your** rule.

Retrieval is **hybrid**: a glossary term is a defined token, so a literal mention
is a certainty rather than a similarity — asking *"what is our ARR by tier?"*
always returns the ARR definition. Semantic search then catches the terms a user
paraphrased instead of naming:

```
"how much annual recurring revenue do we make?"   → ARR              (0.63)
"who ordered something in the last three months?" → Active Customer  (0.74)
```

### 2. Query memory — *learned, self-populating*

Every analysis the Critic accepts is embedded and stored with the code that
produced it. A later question retrieves the nearest past successes and passes
them to the Analysis Agent as worked examples, so the system gets better at a
given workbook the more it is used — with nobody curating a prompt:

```
"break down sales by territory"            → "total revenue by region"  (0.72)
"how long do agents take to close tickets" → "average resolution time
                                              per support agent"        (0.65)
```

Examples from the *same* workbook outrank examples from other files, since they
already reference the right column names. Only clean runs are stored — the agent
never learns from code the Critic rejected.

### Embeddings are pluggable

| Backend | Cost | Image size | Notes |
| :-- | :-- | :-- | :-- |
| `fastembed` *(default)* | free | ~200 MB | BAAI/bge-small-en-v1.5 on ONNX. No torch, no API key. |
| `azure` | ~$0.02/1M tokens | ~200 MB | Azure OpenAI `text-embedding-3-small`. |
| `hashing` | free | 0 | Character n-gram hashing. No downloads at all; weaker recall. Used by the test suite. |

The vector index uses FAISS when it is installed and falls back to brute-force
cosine in NumPy otherwise — at this corpus size FAISS is an optimisation, not a
requirement, so it is not a hard dependency.

---

## Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | Streamlit + Plotly | Upload, chat, glossary editor, schema view, charts, agent logs |
| **Backend** | FastAPI | `/upload`, `/ask`, `/datasets/{id}`, `/glossary`, `/health` |
| **Orchestration** | LangGraph | 10-node agent graph with conditional reflexion edges |
| **LLM** | Groq (`openai/gpt-oss-120b`) | Interpretation, planning, code generation, critique, narrative |
| **Embeddings** | fastembed (ONNX) / Azure OpenAI | Glossary + query-memory retrieval |
| **Vector search** | FAISS, NumPy fallback | Persistent index with payload filtering |
| **Compute** | pandas / NumPy | All analysis runs in memory |
| **Storage** | Parquet on disk **or** Azure Blob | Uploaded sheets, behind one swappable interface |
| **Memory** | Mem0 + Qdrant | Cross-session preferences. **Dormant by default** - needs `sentence-transformers`, which is not in `requirements.txt` because it pulls in torch. The app logs `[MEMORY] Mem0 unavailable` and runs normally without it. |
| **Safety** | Python `ast` | Static audit of generated code before execution |

---

## Key Features

### 1. Bring your own spreadsheet

Upload `.xlsx`, `.xlsm`, `.xls`, or `.csv` - or a `.zip` of several of them for
a large export. Every file becomes one or more sheets the agents can query and
join across, the same way multiple Excel tabs would. Column names are
normalised (`Revenue ($)` → `revenue`) so generated code can reference them
safely, and a zip is decompressed under a hard byte cap that holds even if the
archive's own size metadata is wrong - not just a check on paper.

### 2. Retrieval-grounded generation

Certified definitions and past successful analyses are retrieved before any code
is written, and the response shows exactly what was retrieved and why it matched.

### 3. Self-healing pipeline

Three different failure modes route back to the Analysis Agent with the reason
attached, capped by an attempt budget. When the budget is spent the BI Agent
reports the failure and the code that caused it — it does not invent a result.

### 4. Governance that actually runs

The Impact Agent blocks imports, `open`, `eval`/`exec`, `getattr`, dunder
attribute access, `to_csv`/`to_sql`/`read_*`, and anything else that would reach
outside the DataFrames. Rejected code never reaches the Execute Agent.

### 5. Honest metrics

KPIs are computed from the returned rows. The period-over-period change appears
only when the result contains an ordered time column; otherwise the card reads
`N/A`, not a placeholder percentage. When the Critic has a reservation it is
shown next to the result rather than hidden.

### 6. Transparent reasoning

Every answer ships with the pandas code that produced it, the plan it followed,
and the retrieved context it was grounded on — downloadable as `.py` and `.csv`.

## Installation & Setup

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier works)

### 1. Clone and install

```bash
git clone https://github.com/meghapatel21/agentic-bi-natural-language-querying.git
cd agentic-bi-natural-language-querying
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# then set GROQ_API_KEY in .env
```

Without a key the app still runs, but the Analysis Agent returns a preview of
your sheet instead of a generated analysis.

### 3. Run

**Terminal 1 — backend:**
```bash
uvicorn app.main:app --port 8000 --reload
```

**Terminal 2 — frontend:**
```bash
streamlit run ui/presentation_app.py
```

Open `http://localhost:8501` and upload a spreadsheet. If you do not have one
to hand, `sample_data/` ships a small workbook and a zipped export:

```
sample_data/sample_sales.xlsx   # single workbook
sample_data/sample_export.zip   # several files, each becomes a sheet
```

> The frontend falls back to running the agents in-process if the backend is
> unreachable, so a single-terminal run works too.

### Performance notes

Profiling a large sheet used to dominate upload time. Two settings control it:

| Setting | Default | Effect |
| :--- | :--- | :--- |
| `PROFILE_SAMPLE_ROWS` | `250000` | Above this row count the *descriptive* passes (column roles, cardinality, quality checks, correlations) run on a random sample. Row counts, totals, breakdowns and the date range are always computed on every row. Set `0` to profile every row. |
| `python-calamine` | not installed | Optional. `openpyxl` reads roughly 100k spreadsheet rows a second, which dominates any sizeable `.xlsx` upload; `pip install python-calamine` swaps in a much faster reader automatically. It ships a compiled extension, so it is optional rather than required. |

CSV (or a zipped CSV) avoids the Excel parser entirely and is the fastest way
to load a large export.

### 4. Run the tests

```bash
pip install pytest
pytest
```

178 tests covering the governance gate (every blocked attack class), the
dataset store, zip handling, RAG retrieval, chart-type intent, result
normalisation, and the reflexion loop. They stub the Analysis Agent, so no API
key or network access is needed.

---

## API

```http
POST /upload                          multipart  ->  { dataset_id, filename, profile }
POST /ask                             { dataset_id, question, history }
                                                 ->  { kpis, data, code, insight,
                                                       plan, definitions_used,
                                                       examples_used, reasoning }
GET  /datasets/{id}                   ->  schema + row counts
DEL  /datasets/{id}                   ->  removes the stored dataset
POST /datasets/{id}/glossary          multipart  ->  indexes a glossary file
POST /datasets/{id}/glossary/text     { text }   ->  indexes typed definitions
GET  /datasets/{id}/glossary          ->  term count
GET  /health                          ->  liveness + embedder + agent toggles
```

Example:
```bash
curl -F "file=@sales.xlsx" http://localhost:8000/upload
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json"      -d '{"dataset_id":"<id>","question":"total revenue by region"}'
```

---

## Deployment

**Live deployment:** the app runs on Streamlit Community Cloud at
[agenticbi.streamlit.app](https://agenticbi.streamlit.app/), deployed from this
repository with `ui/presentation_app.py` as the entry point and `GROQ_API_KEY`
supplied as a platform secret. Streamlit runs the frontend only; the agents
execute in-process, which the app falls back to automatically when the FastAPI
backend is unreachable.

**Azure:** see **[AZURE_DEPLOY.md](AZURE_DEPLOY.md)** for Container Apps and App
Service recipes, including the one setting that matters most:

> `STORAGE_BACKEND=local` keeps uploads on the instance's disk. With two or more
> replicas an upload can land on one instance and the follow-up question on
> another. Set `STORAGE_BACKEND=azure` before scaling out.

**Docker:**
```bash
docker build -t agentic-bi .
docker run -p 8501:8501 -p 8000:8000 --env-file .env agentic-bi
```

---

## Known limitations

- **No execution timeout.** The Impact Agent blocks unsafe *operations*, but a
  pathological join on a large sheet can still occupy a worker.
- **No authentication.** Anyone who can reach the app can upload and query. Put
  Azure AD / Easy Auth in front of it before exposing it publicly.
- **Datasets are never garbage-collected.** Add a Blob lifecycle rule or a timer
  job calling `DELETE /datasets/{id}`.
- **Generated code is LLM output.** It is audited for safety, not for analytical
  correctness — the code is shown with every answer so you can check it.
- **Query memory is shared, not per-user.** Every accepted analysis enters one
  index. Multi-tenant use needs a tenant filter on the payload (the index already
  supports payload filtering; the filter is simply not wired to tenant yet).
- **Each optional agent is another LLM call.** Clarify, Planner, Critic and
  Insight roughly triple the calls per question versus the minimum path. Turn any
  of them off with the `*_ENABLED` env vars if latency or rate limits bite.
- **The screenshots and demo GIF need retaking.** They showed the older
  SQL-backed interface and were removed in the Excel-upload rewrite.

## License

Distributed under the Apache 2.0 License. See `LICENSE` for details.

---

## Contact

Megha Patel

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/meghapatel21)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/meghapatel21)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:patelmegha1726@gmail.com)

---
