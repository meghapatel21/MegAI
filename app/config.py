import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY in ("your_groq_key", "your_groq_api_key", ""):
    GROQ_API_KEY = None

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# --- Dataset storage ---
# "local" writes to DATASET_DIR; "azure" uses Blob Storage (see storage/dataset_store.py).
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets"))
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "agentic-bi-datasets")

# --- Upload limits ---
# The whole pipeline still loads a sheet fully into memory to run generated
# pandas against it (that is how the agents work), so these bound what the
# deployment's RAM can actually carry through to query time - raising them
# without raising the container's memory allocation just moves the crash from
# upload time to the first question asked.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 500 * 1024 * 1024))  # 500 MB, raw/compressed
MAX_ROWS_PER_SHEET = int(os.getenv("MAX_ROWS_PER_SHEET", 6_000_000))
ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".csv"}

# --- Upload profiling ---
# The upload briefing is descriptive statistics (column roles, cardinality,
# quality checks, correlations). Those answers are statistically the same from
# a large random sample as from every row, but the cost is not: each one is a
# full-column pass, and there are a dozen of them. Above this many rows the
# descriptive passes run on a sample instead.
#
# Figures a reader would quote back - row counts, headline totals, category
# breakdowns, the date range - are always computed on the full sheet, because
# an approximate total is a wrong number rather than a slower one.
# Set to 0 to always profile every row.
PROFILE_SAMPLE_ROWS = int(os.getenv("PROFILE_SAMPLE_ROWS", 250_000))

# --- Zip uploads ---
# A zip lets a large CSV/Excel export travel compressed; these bound what gets
# decompressed regardless of what the zip's own (attacker-controllable) file
# size metadata claims - see storage/dataset_store.py's bounded reader.
MAX_ZIP_UNCOMPRESSED_BYTES = int(os.getenv("MAX_ZIP_UNCOMPRESSED_BYTES", 1536 * 1024 * 1024))  # 1.5 GB
MAX_ZIP_MEMBERS = int(os.getenv("MAX_ZIP_MEMBERS", 30))

# --- Agent behaviour ---
# How many times the analysis agent may retry after a failed execution (reflexion loop).
MAX_ANALYSIS_ATTEMPTS = int(os.getenv("MAX_ANALYSIS_ATTEMPTS", 3))
# Rows returned to the frontend for tables/charts.
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", 5000))

# --- Memory ---
MEMORY_DIR = os.getenv("MEMORY_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "memories", "qdrant_storage"))

# --- RAG ---
_ROOT = os.path.dirname(os.path.dirname(__file__))
RAG_DIR = os.getenv("RAG_DIR", os.path.join(_ROOT, "rag_store"))

# "fastembed" (local ONNX, free, no key), "azure" (Azure OpenAI), or
# "hashing" (dependency-free fallback used by tests and offline runs).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "fastembed")
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

RAG_QUERY_MEMORY_ENABLED = os.getenv("RAG_QUERY_MEMORY_ENABLED", "true").lower() == "true"
RAG_GLOSSARY_ENABLED = os.getenv("RAG_GLOSSARY_ENABLED", "true").lower() == "true"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 3))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", 0.35))
MAX_GLOSSARY_BYTES = int(os.getenv("MAX_GLOSSARY_BYTES", 2 * 1024 * 1024))

# --- Optional agents ---
# Each enabled agent is an extra LLM call per question.
CLARIFY_ENABLED = os.getenv("CLARIFY_ENABLED", "true").lower() == "true"
PLANNER_ENABLED = os.getenv("PLANNER_ENABLED", "true").lower() == "true"
CRITIC_ENABLED = os.getenv("CRITIC_ENABLED", "true").lower() == "true"
INSIGHT_ENABLED = os.getenv("INSIGHT_ENABLED", "true").lower() == "true"
MAX_PLAN_STEPS = int(os.getenv("MAX_PLAN_STEPS", 4))
