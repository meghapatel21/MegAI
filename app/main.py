from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List

from app import config
from app.billing.metering import record_usage
from app.langgraph.graph import RECURSION_LIMIT, bi_graph
from app.rag import glossary, query_memory
from app.storage.dataset_store import DatasetError, store

app = FastAPI(
    title="Agentic BI",
    description="Natural-language analytics over an uploaded spreadsheet.",
    version="2.0.0",
)

# Streamlit runs on a different origin in every deployment target we support.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    dataset_id: str
    question: str
    tenant_id: str = "default"
    user_id: str = "anonymous"
    history: List[Any] = []


class GlossaryText(BaseModel):
    text: str


@app.get("/health")
def health():
    """Liveness probe for Azure App Service / Container Apps."""
    from app.rag.embedder import get_embedder

    return {
        "status": "ok",
        "llm_configured": bool(config.GROQ_API_KEY),
        "embedder": get_embedder().name,
        "query_memory": query_memory.stats(),
        "agents": {
            "clarify": config.CLARIFY_ENABLED,
            "planner": config.PLANNER_ENABLED,
            "critic": config.CRITIC_ENABLED,
            "insight": config.INSIGHT_ENABLED,
        },
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept a spreadsheet and return a dataset_id plus its profile."""
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw) / 1e6:.1f} MB; the limit is {config.MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )

    try:
        meta = store.save(raw, file.filename)
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record_usage("default", "upload", 1)
    return meta


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    """Schema and row counts for an uploaded workbook."""
    try:
        return store.get_meta(dataset_id)
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/datasets/{dataset_id}/overview")
def get_overview(dataset_id: str):
    """Automatic briefing: metrics, breakdowns, data-quality flags, narrative.

    Computed on demand rather than at upload so a slow LLM never delays the
    upload response itself.
    """
    from app.agents import overview_agent

    try:
        return overview_agent.run(dataset_id)
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    try:
        store.delete(dataset_id)
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "deleted", "dataset_id": dataset_id}


@app.post("/datasets/{dataset_id}/glossary")
async def upload_glossary(dataset_id: str, file: UploadFile = File(...)):
    """Attach a business glossary so the agents use your definitions, not theirs."""
    try:
        store.get_meta(dataset_id)
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    raw = await file.read()
    if len(raw) > config.MAX_GLOSSARY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Glossary exceeds {config.MAX_GLOSSARY_BYTES / 1e6:.0f} MB.",
        )

    try:
        entries = glossary.parse_glossary(raw, file.filename)
        count = glossary.store_glossary(dataset_id, entries)
    except glossary.GlossaryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"dataset_id": dataset_id, "terms": count,
            "preview": [e["term"] for e in entries[:12]]}


@app.post("/datasets/{dataset_id}/glossary/text")
def upload_glossary_text(dataset_id: str, payload: GlossaryText):
    """Same as the file upload, for definitions typed into the UI."""
    try:
        store.get_meta(dataset_id)
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        entries = glossary.parse_glossary(payload.text.encode("utf-8"), "glossary.md")
        count = glossary.store_glossary(dataset_id, entries)
    except glossary.GlossaryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"dataset_id": dataset_id, "terms": count,
            "preview": [e["term"] for e in entries[:12]]}


@app.get("/datasets/{dataset_id}/glossary")
def get_glossary(dataset_id: str):
    return {"dataset_id": dataset_id, "terms": glossary.glossary_size(dataset_id)}


@app.post("/ask")
def ask(payload: AskRequest):
    try:
        state = bi_graph.invoke(
            {
                "tenant_id": payload.tenant_id,
                "user_id": payload.user_id,
                "dataset_id": payload.dataset_id,
                "question": payload.question,
                "history": payload.history,
            },
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    record_usage(payload.tenant_id, "query", 1)
    return state["response"]
