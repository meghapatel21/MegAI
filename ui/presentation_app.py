import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
import sys
import base64
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from ui.render_utils import render_data_results, render_overview
except ImportError:
    # Fallback if running from root
    from render_utils import render_data_results, render_overview

# Port 8000 is the FastAPI backend. The default used to be 8501, which is
# Streamlit's own port - so with no BACKEND_URL set the frontend probed
# itself, got a 200 back, and believed the API was up while actually
# falling through to the in-process path on every call.
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# Logo: a white glyph on a transparent background, so it's inlined as base64
# for in-page use (rather than served as a static path Streamlit doesn't
# expose by default).
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Logo", "icons8-hammerstein-100.png")
try:
    with open(_LOGO_PATH, "rb") as _f:
        LOGO_DATA_URI = "data:image/png;base64," + base64.b64encode(_f.read()).decode("ascii")
except FileNotFoundError:
    LOGO_DATA_URI = None

# Favicon: st.set_page_config only reliably picks up a local image when it's
# handed a PIL.Image, not a bare path string - passing the path silently fell
# back to Streamlit's own default icon. The glyph is also all-white, which
# would vanish against a browser tab's usual white/light chrome, so it's
# composited onto a brand-purple backing square first.
try:
    _logo_img = Image.open(_LOGO_PATH).convert("RGBA")
    _favicon = Image.new("RGBA", _logo_img.size, "#E50914")
    _favicon.alpha_composite(_logo_img)
    PAGE_ICON = _favicon
except FileNotFoundError:
    PAGE_ICON = "🤖"

# Page Config
st.set_page_config(
    page_title="MegAI | Enterprise Suite",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Styles */
    :root {
        --primary-gold: #FFFFFF;
        --secondary-gold: #FFFFFF;
        --accent-blue: #E50914;
        --accent-green: #FFFFFF;
        --accent-purple: #FF4757;
        --background-dark: #0A0A0A;
    }
    
    .stApp {
        background-color: var(--background-dark);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0A0A 0%, #141414 100%);
        border-right: 3px solid var(--accent-blue);
    }
    
    h1, h2, h3 {
        color: #ffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Tabs & Cards (Improved Full-Width Tabs) */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        width: 100% !important;
        display: flex !important;
    }
    .stTabs [data-baseweb="tab"] { 
        height: 55px; 
        background-color: #141414; 
        border-radius: 8px; 
        color: #FFFFFF; 
        font-weight: 700; 
        flex-grow: 1 !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #1F1F1F;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background-color: #E50914; 
        color: white; 
        border: 1px solid #FF4757;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
    }
    .metric-card { background-color: #1F1F1F; padding: 20px; border-radius: 10px; border: 1px solid #2A2A2A; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Helper: Get CST Time
def get_ist_time():
    # Production-grade CST converter (UTC - 6:00)
    utc_now = datetime.now(timezone.utc)
    cst_now = utc_now + timedelta(hours=-6)
    return cst_now.strftime("%Y-%m-%d %H:%M:%S CST")

# Session State for Logs
if "system_logs" not in st.session_state:
    st.session_state.system_logs = []

# Global Configuration
BACKEND_TIMEOUT = int(os.getenv("BACKEND_TIMEOUT", "180"))

# Severity is carried explicitly rather than guessed from the message text:
# inferring it from substrings classified "Query Success" as a success only
# because the word happened to appear in it, and would have mislabelled any
# message that merely mentioned an error.
LOG_LEVELS = ("INFO", "SUCCESS", "WARNING", "ERROR")

# Dropdown label -> level, so the filter and the feed agree on one vocabulary.
_SEVERITY_CHOICES = {
    "🔴 Errors Only": "ERROR",
    "🟢 Success Only": "SUCCESS",
    "🟡 Warnings Only": "WARNING",
    "🔵 Info Only": "INFO",
}

# The feed is session-scoped and purely informational, so it is capped rather
# than allowed to grow for the life of the tab.
MAX_LOG_ENTRIES = 500


def log_event(event_type, details, level="INFO"):
    # Normalised on the way in: a typo'd level would otherwise match no filter
    # and make the entry invisible in the feed.
    level = str(level).upper()
    if level not in LOG_LEVELS:
        level = "INFO"

    st.session_state.system_logs.insert(0, {
        "Timestamp": get_ist_time(),
        "Event": event_type,
        "Details": details,
        "Level": level,
    })
    del st.session_state.system_logs[MAX_LOG_ENTRIES:]


# --- BACKEND BRIDGE -------------------------------------------------------
# Prefers the FastAPI service. When it is unreachable (Streamlit Cloud, or a
# local run with only one terminal open) the agents are invoked in-process
# against the same dataset store, so the app stays usable either way.

def _backend_alive():
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data(ttl=10, show_spinner=False)
def _backend_status():
    """Is the API reachable? Cached briefly - this renders on every rerun
    of the diagnostics tab and the probe carries a 2s timeout."""
    return _backend_alive()


def upload_dataset(uploaded_file):
    """Send a spreadsheet to the backend. Returns (meta, error_message)."""
    raw = uploaded_file.getvalue()

    if _backend_alive():
        try:
            resp = requests.post(
                f"{BACKEND_URL}/upload",
                files={"file": (uploaded_file.name, raw)},
                timeout=120,
            )
            if resp.status_code == 200:
                log_event("Upload", f"{uploaded_file.name} accepted by backend", "SUCCESS")
                return resp.json(), None
            return None, resp.json().get("detail", resp.text)
        except requests.exceptions.RequestException as exc:
            log_event("Upload Fallback", f"Backend upload failed: {exc}", "WARNING")

    try:
        from app.storage.dataset_store import store
        meta = store.save(raw, uploaded_file.name)
        log_event("Upload", f"{uploaded_file.name} processed in-process", "SUCCESS")
        return meta, None
    except Exception as exc:
        return None, str(exc)


def fetch_overview(dataset_id):
    """Automatic briefing for a freshly uploaded file. Returns (overview, error)."""
    if _backend_alive():
        try:
            resp = requests.get(f"{BACKEND_URL}/datasets/{dataset_id}/overview", timeout=120)
            if resp.status_code == 200:
                return resp.json(), None
            return None, resp.json().get("detail", resp.text)
        except requests.exceptions.RequestException as exc:
            log_event("Overview Fallback", str(exc), "WARNING")

    try:
        from app.agents import overview_agent
        return overview_agent.run(dataset_id), None
    except Exception as exc:
        return None, str(exc)


def upload_glossary(dataset_id, uploaded_file=None, text=None):
    """Attach business definitions to a dataset. Returns (result, error)."""
    if _backend_alive():
        try:
            if uploaded_file is not None:
                resp = requests.post(
                    f"{BACKEND_URL}/datasets/{dataset_id}/glossary",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=60,
                )
            else:
                resp = requests.post(
                    f"{BACKEND_URL}/datasets/{dataset_id}/glossary/text",
                    json={"text": text or ""}, timeout=60,
                )
            if resp.status_code == 200:
                return resp.json(), None
            return None, resp.json().get("detail", resp.text)
        except requests.exceptions.RequestException as exc:
            log_event("Glossary Fallback", str(exc), "WARNING")

    try:
        from app.rag import glossary as glossary_module
        if uploaded_file is not None:
            entries = glossary_module.parse_glossary(uploaded_file.getvalue(), uploaded_file.name)
        else:
            entries = glossary_module.parse_glossary((text or "").encode("utf-8"), "glossary.md")
        count = glossary_module.store_glossary(dataset_id, entries)
        return {"terms": count, "preview": [e["term"] for e in entries[:12]]}, None
    except Exception as exc:
        return None, str(exc)


def ask_question(question, dataset_id, history):
    """Run one question through the agent graph. Returns (response, error)."""
    payload = {
        "dataset_id": dataset_id,
        "question": question,
        "tenant_id": "default",
        "user_id": st.session_state.get("user_id", "u1"),
        "history": history,
    }

    if _backend_alive():
        try:
            resp = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=BACKEND_TIMEOUT)
            if resp.status_code == 200:
                return resp.json(), None
            try:
                return None, resp.json().get("detail", resp.text)
            except ValueError:
                return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.RequestException as exc:
            log_event("Query Fallback", f"Backend unreachable: {exc}", "WARNING")

    try:
        from app.langgraph.graph import RECURSION_LIMIT, bi_graph
        state = bi_graph.invoke(payload, config={"recursion_limit": RECURSION_LIMIT})
        return state["response"], None
    except Exception as exc:
        import traceback
        log_event("Agent Error", traceback.format_exc()[:400], "ERROR")
        return None, f"{type(exc).__name__}: {exc}"

# --- SUPERIOR SIDEBAR ---
with st.sidebar:
    # 1. Project Title Card
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(10, 10, 10, 0.9) 0%, rgba(20, 20, 20, 0.9) 100%); padding: 18px; border-radius: 10px; text-align: center; border: 1px solid rgba(229, 9, 20, 0.5);'>
        <h2 style='margin: 0; font-size: 1.5rem; font-weight: 700;'>
            <img src='{LOGO_DATA_URI}' style='height: 1.3rem; vertical-align: -0.2rem; margin-right: 4px;'/>
            <span style='background: -webkit-linear-gradient(left, #E50914, #FF4757); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Meg</span>
            <span style='background: -webkit-linear-gradient(left, #FF4757, #FFFFFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>AI</span>
        </h2>
        <p style='color: #999999; font-size: 0.72rem; font-weight: 600; margin-top: 6px; letter-spacing: 1.2px;'>
            INTELLIGENT ANALYTICS PLATFORM
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Developer Profile
    st.markdown("### Developer")
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(229, 9, 20, 0.15) 0%, rgba(255, 71, 87, 0.1) 100%);
                padding: 15px; border-radius: 10px; border: 1px solid rgba(229, 9, 20, 0.35);'>
        <p style='margin: 0; color: #FFFFFF; font-weight: 700; font-size: 1rem;'>Megha Patel</p>
        <p style='margin: 2px 0 10px 0; color: #999999; font-size: 0.8rem;'>Creator &amp; Developer, MegAI</p>
        <div style='font-size: 0.85rem;'>
            <a href='https://github.com/meghapatel21' target='_blank' style='color: #CCCCCC; text-decoration: none; background-color: #141414; padding: 4px 10px; border-radius: 5px; border: 1px solid #333333;'>
                GitHub
            </a>
            &nbsp;
            <a href='https://linkedin.com/in/meghapatel21' target='_blank' style='color: #FFFFFF; text-decoration: none; background-color: #E50914; padding: 4px 10px; border-radius: 5px;'>
                LinkedIn
            </a>
            &nbsp;
            <a href='mailto:patelmegha1726@gmail.com' style='color: #CCCCCC; text-decoration: none; background-color: #141414; padding: 4px 10px; border-radius: 5px; border: 1px solid #333333;'>
                Email
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("**MegAI** lets you upload a spreadsheet and ask questions about it in plain English. A 10-agent pipeline profiles your columns, writes pandas code, audits it for safety, runs it, and turns the result into KPIs and charts.")

    st.markdown("---")

    # 3. User Resources
    st.markdown("### User Guide")
    try:
        with open("all_questions_list.txt", "rb") as file:
            st.download_button(
                label="Download All Prompts",
                data=file,
                file_name="agentic_bi_prompts.txt",
                mime="text/plain",
                help="Get the full list of 24 supported Quick Start Questions to try out!",
                width='stretch'
            )
    except FileNotFoundError:
        st.warning("Prompts file not found.")





# --- TOP BADGE (Developer Credential) ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("")  # Spacer for alignment
with col_h2:
    st.markdown("""
<div style='background: #141414; padding: 10px 16px; border-radius: 10px; border: 1px solid rgba(229, 9, 20, 0.4); text-align: center; margin-bottom: 5px;'>
    <p style='margin: 0; color: #FFFFFF; font-weight: 700; font-size: 0.85rem; line-height: 1.2;'>Megha Patel</p>
    <div style='display: flex; justify-content: center; gap: 15px; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 7px; margin-top: 7px;'>
        <a href='https://github.com/meghapatel21' target='_blank' style='color: #CCCCCC; text-decoration: none; font-size: 0.78rem; font-weight: 600; display: flex; align-items: center; gap: 5px;'>
            <img src="https://img.icons8.com/ios-filled/50/cccccc/github.png" width="14" height="14"/> GitHub
        </a>
        <a href='https://linkedin.com/in/meghapatel21' target='_blank' style='color: #CCCCCC; text-decoration: none; font-size: 0.78rem; font-weight: 600; display: flex; align-items: center; gap: 5px;'>
            <img src="https://img.icons8.com/ios-filled/50/cccccc/linkedin.png" width="14" height="14"/> LinkedIn
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO HEADER (Premium UI) ---
st.markdown(f"""
<div style='text-align: center; padding: 28px 20px; background: linear-gradient(135deg, rgba(229, 9, 20, 0.07) 0%, rgba(255, 71, 87, 0.07) 100%); border-radius: 14px; margin-bottom: 25px; border: 1px solid rgba(229, 9, 20, 0.2);'>
    <div style='display: flex; align-items: center; justify-content: center; gap: 14px; margin-bottom: 8px;'>
        <img src='{LOGO_DATA_URI}' style='height: 2.2rem;'/>
        <h1 style='color: #FFFFFF; margin: 0; font-size: 2.3rem; font-weight: 700; letter-spacing: 0.5px; line-height: 1; font-family: "Inter", sans-serif;'>
            Meg<span style='color: #E50914;'>AI</span>
        </h1>
    </div>
    <p style='font-size: 1.05rem; color: #999999; font-weight: 400; margin: 4px 0 0 0;'>
        Natural Language Analytics, Powered by Autonomous AI Agents
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Live Demo", 
    "About Project", 
    "Tech Stack", 
    "HLD & LLD", 
    "Architecture", 
    "System Logs"
])

# --- TAB 1: LIVE DEMO ---
with tab1:
    # Initialize Chat History (Must be at the top)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 1. Header Section
    # 1. Header Section
    st.markdown("""
    <div style='background: rgba(229, 9, 20, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #E50914; margin-bottom: 20px;'>
        <h3 style='margin: 0; color: #ffff;'>MegAI Analyst</h3>
        <p style='margin: 5px 0 0 0; color: #CCCCCC;'>Upload your own Excel, CSV, or zip file and ask questions about it in plain English. Driven by a <b>10-agent LangGraph pipeline</b> — no SQL, no setup, no fixed schema.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. How to Use Section
    st.markdown("""
<div style='background: #141414; padding: 20px; border-radius: 12px; border: 1px solid #2A2A2A; margin-bottom: 25px;'>
<h4 style='color: #FFFFFF; margin: 0 0 15px 0; font-weight: 700;'>How to Use This Live Demo</h4>
<div style='color: #CCCCCC; line-height: 1.8;'>
<p style='margin: 8px 0;'><span style='background: rgba(229, 9, 20, 0.25); color: #FF4757; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;'>STEP 1</span> <b>Upload an Excel, CSV, or .zip of several files</b> (up to 500 MB, 1.5 GB once a zip is unpacked). Every sheet becomes a table MegAI can query and join across - no database, no fixed schema.</p>
<p style='margin: 8px 0;'><span style='background: rgba(229, 9, 20, 0.25); color: #FF4757; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;'>STEP 2</span> Optionally upload a <b>business glossary</b> so terms like "ARR" or "active customer" follow your definition instead of a guess.</p>
<p style='margin: 8px 0;'><span style='background: rgba(229, 9, 20, 0.25); color: #FF4757; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;'>STEP 3</span> Ask a question in plain English, or pick one of the suggestions generated from your actual columns.</p>
<p style='margin: 8px 0;'><span style='background: rgba(229, 9, 20, 0.25); color: #FF4757; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;'>STEP 4</span> A pipeline of agents interprets, retrieves relevant context, plans, writes pandas, audits it for safety, runs it, and critiques the result before you see <b>KPIs, a chart, and a data table</b>.</p>
<p style='margin: 8px 0;'><span style='background: rgba(229, 9, 20, 0.25); color: #FF4757; padding: 3px 10px; border-radius: 6px; font-weight: 600; font-size: 0.85rem;'>STEP 5</span> Expand <b>"See Generated Code & Reasoning"</b> to read the exact pandas that produced your answer, or download it.</p>
</div>
<div style='margin-top: 15px; padding: 10px 14px; background: rgba(0, 0, 0, 0.3); border-radius: 8px; border-left: 3px solid #E50914;'>
<p style='color: #999999; margin: 0; font-size: 0.88rem;'><b style='color: #FF4757;'>Tip:</b> MegAI has <b>memory</b> - try follow-up questions like "Now show that for only the North region", and ask for a specific chart type ("as a pie chart") to get it directly.</p>
</div>
</div>
    """, unsafe_allow_html=True)
    
    
    
    # ------------------------------------------------------------------
    # DATA SOURCE - the uploaded workbook replaces the old fixed SQL schema
    # ------------------------------------------------------------------
    if "main_query_input" not in st.session_state:
        st.session_state.main_query_input = ""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None

    def set_query(text):
        st.session_state.main_query_input = text

    st.markdown("""
    <h4 style='color: #FFFFFF; margin: 10px 0 5px 0; font-weight: 700; font-size: 1.2rem;'>
    Step 1 — Load Your Data
    </h4>
    <hr style='border: none; height: 1px; background: rgba(229, 9, 20, 0.4); margin: 0 0 15px 0;'>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a spreadsheet",
        type=["xlsx", "xls", "xlsm", "csv", "zip"],
        help="Every sheet becomes a table the agents can query. Nothing is written back to the file. "
             "For a large export, zip it first - a .zip of one or more .xlsx/.csv files works too, "
             "and each file inside becomes its own sheet.",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        signature = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("dataset_signature") != signature:
            with st.spinner(f"Reading `{uploaded.name}` and profiling the columns..."):
                meta, err = upload_dataset(uploaded)
            if err:
                st.error(f"⚠️ Could not read that file: {err}")
                log_event("Upload Failed", err[:200], "ERROR")
            else:
                st.session_state.dataset = meta
                st.session_state.dataset_signature = signature
                st.session_state.chat_history = []
                st.session_state.glossary_terms = 0
                sheets = len(meta["profile"]["sheets"])
                rows = sum(s["row_count"] for s in meta["profile"]["sheets"])
                st.success(f"✅ **{uploaded.name}** loaded - {sheets} sheet(s), {rows:,} rows.")

                # Profile the data straight away so the user gets findings
                # before they have to think of a question.
                with st.spinner("Analysing your data..."):
                    overview, ov_err = fetch_overview(meta["dataset_id"])
                if ov_err:
                    log_event("Overview Failed", ov_err[:200], "ERROR")
                else:
                    st.session_state.overview = overview
                    log_event("Overview Ready",
                              f"{overview['sheet_count']} sheet(s), {overview['total_rows']:,} rows",
                              "SUCCESS")

    dataset = st.session_state.get("dataset")

    if not dataset:
        st.info(
            "👆 **Upload an Excel or CSV file to begin.** The agents read the real column "
            "names and types from your file, then write pandas code to answer questions about it."
        )
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 18px; border-radius: 12px;
                    border: 1px dashed rgba(255, 255, 255, 0.2); margin: 15px 0 25px 0;'>
            <p style='color: #CCCCCC; font-size: 0.85rem; margin: 0 0 10px 0; font-weight: 600;
                      text-transform: uppercase; letter-spacing: 1px;'>What works well</p>
            <ul style='color: #CCCCCC; margin: 0; line-height: 1.8; font-size: 0.9rem;'>
                <li>A header row on the first row of each sheet</li>
                <li>One record per row (transactions, customers, orders, tickets...)</li>
                <li>Multiple sheets - the agents can join across them</li>
                <li>A large export? Zip it - each file inside becomes its own sheet</li>
                <li>Up to 500 MB per upload (1.5 GB once a zip is unpacked)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        profile = dataset["profile"]

        # --- SCHEMA VIEW -------------------------------------------------
        with st.expander(f"📑 Detected schema for `{dataset['filename']}`", expanded=False):
            for sheet in profile["sheets"]:
                primary = " *(primary)*" if sheet["name"] == profile["primary_sheet"] else ""
                st.markdown(f"**`{sheet['name']}`**{primary} — {sheet['row_count']:,} rows")
                st.dataframe(
                    pd.DataFrame([{
                        "column": c["name"],
                        "type": c["dtype"],
                        "nulls": c["null_count"],
                        "sample / range": (
                            ", ".join(c["sample_values"][:5]) if "sample_values" in c
                            else (f"{c['min']:,.4g} to {c['max']:,.4g}" if "min" in c else "")
                        ),
                    } for c in sheet["columns"]]),
                    width='stretch', hide_index=True,
                )
            st.caption(
                "Column names are normalised to lowercase snake_case so generated code "
                "can reference them safely. `Revenue ($)` becomes `revenue`."
            )

        # --- BUSINESS GLOSSARY (RAG) -------------------------------------
        glossary_count = st.session_state.get("glossary_terms", 0)
        glossary_label = (
            f"📖 Business glossary — {glossary_count} term(s) loaded"
            if glossary_count else
            "📖 Business glossary — teach the agents your definitions (optional)"
        )
        with st.expander(glossary_label, expanded=False):
            st.caption(
                "Your definition of *ARR* or *active customer* is a policy decision, not "
                "something the model should guess. Definitions you add here are retrieved "
                "when a question mentions them, and the generated code follows your rule."
            )

            g_tab1, g_tab2 = st.tabs(["✍️ Type definitions", "📁 Upload a file"])

            with g_tab1:
                glossary_text = st.text_area(
                    "One definition per line",
                    height=150,
                    placeholder=(
                        "ARR: monthly_price * 12, active subscriptions only\n"
                        "Active customer: at least one order in the last 90 days\n"
                        "Churn rate = cancelled / active at period start"
                    ),
                    key="glossary_text_input",
                    label_visibility="collapsed",
                )
                if st.button("💾 Save definitions", width='stretch', key="save_glossary_text"):
                    if not glossary_text.strip():
                        st.warning("Add at least one definition first.")
                    else:
                        result, err = upload_glossary(dataset["dataset_id"], text=glossary_text)
                        if err:
                            st.error(f"Could not parse that: {err}")
                        else:
                            st.session_state.glossary_terms = result["terms"]
                            st.success(f"✅ {result['terms']} definition(s) indexed: "
                                       + ", ".join(result["preview"]))
                            log_event("Glossary Updated", f"{result['terms']} terms", "SUCCESS")

            with g_tab2:
                glossary_file = st.file_uploader(
                    "Glossary file",
                    type=["md", "txt", "csv", "xlsx", "xls"],
                    key="glossary_file",
                    label_visibility="collapsed",
                    help="Markdown/text with 'Term: definition' lines, or a sheet with term and definition columns.",
                )
                if glossary_file is not None:
                    signature = f"{glossary_file.name}:{glossary_file.size}"
                    if st.session_state.get("glossary_signature") != signature:
                        result, err = upload_glossary(dataset["dataset_id"], uploaded_file=glossary_file)
                        if err:
                            st.error(f"Could not parse that glossary: {err}")
                        else:
                            st.session_state.glossary_signature = signature
                            st.session_state.glossary_terms = result["terms"]
                            st.success(f"✅ {result['terms']} definition(s) indexed: "
                                       + ", ".join(result["preview"]))
                            log_event("Glossary Uploaded", f"{result['terms']} terms", "SUCCESS")

        # --- AUTOMATIC BRIEFING ------------------------------------------
        # Computed on upload, so findings arrive before the first question.
        overview = st.session_state.get("overview")
        if overview:
            render_overview(overview, on_question=set_query)
        else:
            st.info("Profiling this file... ask a question below in the meantime.")

        c_reset, c_info = st.columns([1, 3])

        with c_reset:
            if st.button("🗑️ Remove Dataset", width='stretch'):
                st.session_state.dataset = None
                st.session_state.dataset_signature = None
                st.session_state.chat_history = []
                st.rerun()
        with c_info:
            st.caption(
                f"Dataset `{dataset['dataset_id'][:8]}…` · "
                f"{len(profile['sheets'])} sheet(s) · queries run in memory, your file is never modified."
            )

    # Input Section (Moved to Bottom)


    # Results Section Title (Always Visible)
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style='text-align: center; padding: 10px; margin-bottom: 5px;'>
    <h2 style='color: #FFFFFF; font-weight: 700; margin: 0; font-size: 1.6rem; white-space: nowrap;'>
    MegAI <span style='color: #E50914;'>Intelligence</span>
    </h2>
</div>
<hr style='border: none; height: 1px; background: rgba(229, 9, 20, 0.4); margin: 5px 0 30px 0;'>
    """, unsafe_allow_html=True)

    # --- CHAT THREAD DISPLAY ---
    if st.session_state.chat_history:
        st.markdown("#### Conversation Thread")
        for i, turn in enumerate(st.session_state.chat_history):
            # User Message
            st.markdown(f"""
            <div style='background: rgba(229, 9, 20, 0.1); padding: 15px; border-radius: 12px; margin-bottom: 10px; border-left: 5px solid #E50914; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                <p style='color: #E50914; font-weight: 700; margin: 0 0 5px 0; font-size: 0.85rem; text-transform: uppercase;'>👤 User</p>
                <p style='color: #FFFFFF; margin: 0; line-height: 1.5;'>{turn['user']}</p>
            </div>
            """, unsafe_allow_html=True)
            # AI Summary (Using markdown for multi-line support)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 12px 12px 0 0; border-left: 5px solid #FFFFFF; margin-bottom: 0;'>
                <p style='color: #FFFFFF; font-weight: 700; margin: 0; font-size: 0.85rem; text-transform: uppercase;'>🤖 MegAI Analyst</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.05); padding: 0 15px 15px 15px; border-radius: 0 0 12px 12px; border-left: 5px solid #FFFFFF; margin-bottom: 25px;'>
                <div style='color: #E5E5E5; font-style: normal; line-height: 1.6;'>
            """, unsafe_allow_html=True)
            st.markdown(turn['ai'])
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # RENDER DATA WITHIN CHAT THREAD
            turn_data = turn.get("full_data", {})

            if turn_data.get("status") == "needs_clarification":
                # The agents asked a question back. Offer the options as buttons
                # so answering is one click rather than a retype.
                st.markdown(
                    f"<p style='color:#FF4757;font-weight:600;margin:5px 0 10px 0;'>"
                    f"❓ {turn_data.get('question', '')}</p>",
                    unsafe_allow_html=True,
                )
                option_cols = st.columns(min(len(turn_data.get("options", [])) or 1, 3))
                for opt_i, option in enumerate(turn_data.get("options", [])):
                    option_cols[opt_i % len(option_cols)].button(
                        option,
                        key=f"clarify_{i}_{opt_i}",
                        width='stretch',
                        on_click=set_query,
                        args=(option,),
                    )
                st.caption("Pick one, or rephrase the question yourself below.")
            elif "full_data" in turn:
                with st.expander(f"📊 View Analysis Details for Question {i+1}", expanded=(i == len(st.session_state.chat_history)-1)):
                    render_data_results(turn["full_data"], turn_index=i)

            # Integrated Feedback Row for THIS turn
            _, feed_col, _ = st.columns([1, 2, 1])
            with feed_col:
                fc1, fc2, fc3 = st.columns([3, 1, 1])
                with fc1:
                    st.markdown("<div style='text-align: right; padding-top: 5px; font-weight: 600; color: #CCCCCC; font-size: 0.8rem;'>Was this answer helpful?</div>", unsafe_allow_html=True)
                with fc2:
                    if st.button("👍", key=f"thumbs_up_{i}", help="Yes, this was helpful"):
                        st.toast("Thanks for the feedback!", icon="⭐")
                        log_event("Thumbs Up", f"Query: {turn['user']}", "INFO")
                with fc3:
                    if st.button("👎", key=f"thumbs_down_{i}", help="No, this needs improvement"):
                        st.toast("Noted, we'll improve!", icon="🔧")
                        log_event("Thumbs Down", f"Query: {turn['user']}", "INFO")
            
            st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

                    
        st.markdown("<hr style='border-top: 1px dashed #444; margin-bottom: 40px;'>", unsafe_allow_html=True)

    # Results display moved to chat thread
    pass

    # --- QUICK FEEDBACK SECTION (Relocated to Bottom) ---
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    .feedback-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 25px;
        padding: 15px 30px;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-radius: 100px;
        width: fit-content;
        margin: 0 auto;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .feedback-text {
        color: #E5E5E5;
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    

    



    # --- INPUT SECTION (MOVED TO BOTTOM) ---
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)
    
    # Callback function for Clear button (Complete Reset)
    def clear_input():
        st.session_state.main_query_input = ""
        st.session_state.show_result = False
        if "query_result" in st.session_state:
            del st.session_state["query_result"]
    
    # Heading for Input Section
    st.markdown("""
    <h4 style='color: #FFFFFF; margin: 0 0 10px 0; font-weight: 700; font-size: 1.15rem;'>
    Your Query — Answering to MegAI Analyst
    </h4>
    """, unsafe_allow_html=True)

    # Input bound to session state - Using text_area for larger input field
    q = st.text_area("Question Input", 
                      placeholder="Ask a question about your uploaded file, e.g. 'Total revenue by region, highest first'",
                      key="main_query_input",
                      label_visibility="collapsed",
                      height=100)

    # Colorful Button Row: Analyze Data, Enter, Clear, Download, Feedback
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)
    
    # Custom CSS for colorful buttons with distinct colors
    st.markdown("""
    <style>
    /* Analyze Data Button - Blue */
    div[data-testid="column"]:nth-child(1) button {
        background: linear-gradient(135deg, #E50914 0%, #B71C1C 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="column"]:nth-child(1) button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6) !important;
    }
    
    /* Enter Button - Green */
    div[data-testid="column"]:nth-child(2) button {
        background: linear-gradient(135deg, #FFFFFF 0%, #CCCCCC 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="column"]:nth-child(2) button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(204, 204, 204, 0.6) !important;
    }
    
    /* Clear Button - Red/Orange */
    div[data-testid="column"]:nth-child(3) button {
        background: linear-gradient(135deg, #E50914 0%, #B71C1C 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="column"]:nth-child(3) button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(183, 28, 28, 0.6) !important;
    }
    
    /* Download Button - Purple */
    div[data-testid="column"]:nth-child(4) button {
        background: linear-gradient(135deg, #CCCCCC 0%, #FFFFFF 100%) !important;
        color: #333 !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="column"]:nth-child(4) button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(204, 204, 204, 0.6) !important;
    }
    
    /* Feedback Button - Teal */
    div[data-testid="column"]:nth-child(5) button {
        background: linear-gradient(135deg, #FF4757 0%, #E50914 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="column"]:nth-child(5) button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    

    # Logic for handling query submission (Callback to safely clear input)
    def submit_query():
        q = (st.session_state.main_query_input or "").strip()
        if not q:
            st.session_state.last_error = "Please enter a question."
            return

        dataset = st.session_state.get("dataset")
        if not dataset:
            st.session_state.last_error = "Upload a spreadsheet before asking a question."
            return

        log_event("Query Initiated", f"Question: {q}", "INFO")

        # Recent turns give the agents the context for follow-ups like "now by region".
        history_context = []
        for turn in st.session_state.chat_history[-4:]:
            history_context.append(f"User: {turn['user']}")
            history_context.append(f"AI: {turn['ai'][:300]}")

        data, err = ask_question(q, dataset["dataset_id"], history_context)

        if err:
            st.session_state.last_error = err
            log_event("Query Error", err[:200], "ERROR")
            st.session_state.show_result = False
            return

        status = data.get("status")
        if status == "failed":
            log_event("Analysis Failed", str(data.get("error"))[:200], "ERROR")
        elif status == "needs_clarification":
            log_event("Clarification Requested", data.get("question", "")[:150], "INFO")
        else:
            log_event("Query Success", f"{data.get('kpis', {}).get('row_count', 0)} rows returned", "SUCCESS")
            if data.get("plan"):
                log_event("Plan Executed", " → ".join(data["plan"]), "INFO")
            if data.get("definitions_used"):
                log_event("Glossary Applied",
                          ", ".join(d["term"] for d in data["definitions_used"]), "INFO")
            if data.get("examples_used"):
                log_event("RAG Retrieval",
                          f"{len(data['examples_used'])} similar past analysis/analyses reused", "INFO")
            if data.get("warning"):
                log_event("Critic Warning", data["warning"][:200], "WARNING")

        kpis = data.get("kpis", {})
        label = kpis.get("primary_label", "Result")
        value = kpis.get("primary_val", 0)
        shown = f"${value:,.2f}" if kpis.get("is_currency") else f"{value:,.2f}".rstrip("0").rstrip(".")

        if status in ("failed", "needs_clarification"):
            ai_message = data.get("reasoning", "The analysis could not be completed.")
        else:
            insight = data.get("insight", "")
            ai_message = f"{insight}\n\n" if insight else ""
            ai_message += f"{data.get('reasoning', '')}\n\n**{label}:** {shown}"

        st.session_state.chat_history.append({"user": q, "ai": ai_message, "full_data": data})
        st.session_state.query_result = data
        st.session_state.show_result = True
        st.session_state.last_error = None
        st.session_state.main_query_input = ""


    analyze_clicked = btn_col1.button("📊 Analyze Data", width='stretch', on_click=submit_query)
    enter_clicked = btn_col2.button("✅ Enter", width='stretch', on_click=submit_query)
    
    def clear_all():
        st.session_state.main_query_input = ""
        st.session_state.chat_history = []
        st.session_state.query_result = None
        st.session_state.show_result = False
        st.session_state.last_error = None

    clear_clicked = btn_col3.button("🗑️ Clear", width='stretch', key="clear_btn", on_click=clear_all)
    download_clicked = btn_col4.button("📥 Download", width='stretch', key="download_btn")
    feedback_clicked = btn_col5.button("💬 Feedback", width='stretch', key="feedback_btn")
    
    # Clear button now uses on_click callback - no manual logic needed here
    
    # Download button logic
    if download_clicked:
        if "query_result" in st.session_state:
            result_data = st.session_state.query_result
            # Create downloadable text content
            _k = result_data.get('kpis', {})
            _src = st.session_state.get('dataset', {}).get('filename', 'N/A')
            txt_content = f"""MegAI ANALYSIS RESULT
=====================================
Source file: {_src}
Question:    {q if q else 'N/A'}

KPIs:
-----
{_k.get('primary_label', 'Result')}: {_k.get('primary_val', 'N/A')}
Rows returned: {_k.get('row_count', 'N/A')}
Change over period: {_k.get('trend_pct', 'not applicable to this result')}

Generated pandas code:
----------------------
{result_data.get('code', 'N/A')}

Agent Reasoning:
----------------
{result_data.get('reasoning', 'N/A')}

Data:
-----
{result_data.get('data', 'N/A')}
"""
            st.download_button(
                label="💾 Download Results as TXT",
                data=txt_content,
                file_name=f"query_result_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                width='stretch'
            )
        else:
            st.warning("⚠️ No results to download. Please run a query first.")
    
    # Feedback button logic
    if feedback_clicked:
        st.info("💬 **Feedback Form**\n\nPlease share your thoughts about this AI analyst experience:")
        feedback_text = st.text_area("Your feedback:", key="feedback_input", height=100)
        if st.button("📨 Submit Feedback"):
            if feedback_text:
                log_event("Feedback Submitted", feedback_text, "SUCCESS")
                st.success("✅ Thank you for your feedback!")
            else:
                st.warning("Please enter your feedback before submitting.")
    
    # Display error if it occurred in callback
    if st.session_state.get("last_error"):
        st.error(f"⚠️ {st.session_state.last_error}")
        st.session_state.last_error = None  # Clear after showing


# --- TAB 2: ABOUT ---
with tab2:
    # 1. Premium Header / Vision Section
    st.markdown("""
    <div style='background: #141414;
                padding: 28px; border-radius: 14px; border: 1px solid #2A2A2A; border-left: 3px solid #E50914; margin-bottom: 30px;'>
        <h2 style='color: #FFFFFF; margin: 0 0 10px 0; font-weight: 700;'>Project Vision &amp; Mission</h2>
        <p style='color: #CCCCCC; font-size: 1.05rem; line-height: 1.6;'>
            <b>MegAI</b> is an enterprise-grade Business Intelligence platform powered by Agentic AI.
            Users can ask business questions in plain English, and the system intelligently converts them into
            trusted, governed, explainable insights.
        </p>
        <p style='color: #999999; font-size: 0.95rem; margin-top: 10px; font-style: italic;'>
            "To democratize data access within enterprises, reducing the dependency on data analyst teams for routine reporting."
        </p>
        <hr style='border: none; height: 1px; background: #2A2A2A; margin: 20px 0;'>
        <div style='display: flex; gap: 12px; align-items: center; flex-wrap: wrap;'>
            <div style='background: rgba(229, 9, 20, 0.15); padding: 5px 14px; border-radius: 8px; color: #FF4757; font-weight: 600; font-size: 0.82rem;'>AI Analyst</div>
            <div style='background: rgba(229, 9, 20, 0.15); padding: 5px 14px; border-radius: 8px; color: #FF4757; font-weight: 600; font-size: 0.82rem;'>Long-Term Memory</div>
            <div style='background: rgba(255, 255, 255, 0.08); padding: 5px 14px; border-radius: 8px; color: #FFFFFF; font-weight: 600; font-size: 0.82rem;'>Enterprise Governance</div>
        </div>
        <p style='color: #999999; margin-top: 18px; font-size: 0.92rem;'>
            In simple words: this system replaces manual BI dashboards with an <span style='color: #FF4757; font-weight: 600;'>AI analyst</span> that remembers, learns, and scales securely.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # 2. Agent Team (Cards Style)
    st.subheader("Meet the Agent Team")
    st.markdown("<p style='color: #CCCCCC; margin-bottom: 20px;'>A coordinated swarm of specialized AI agents working together to answer your questions.</p>", unsafe_allow_html=True)

    # CSS for cards
    st.markdown("""
    <style>
    .agent-card {
        background: #141414;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2A2A2A;
        height: 100%;
        transition: border-color 0.2s;
    }
    .agent-card:hover {
        border-color: #E50914;
    }
    .agent-icon { font-size: 1.7rem; margin-bottom: 10px; display: block; }
    .agent-title { color: #fff; font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; }
    .agent-desc { color: #999999; font-size: 0.88rem; line-height: 1.5; margin-bottom: 0; }
    </style>
    """, unsafe_allow_html=True)

    # The BI Agent (10) assembles the response; see the flow below.
    st.caption("Nine agents contribute to an answer; the BI Agent assembles the response.")
    st.write("")
    col_0, col_1, col_2 = st.columns(3)
    with col_0:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">🗂️</span>
            <div class="agent-title">1. Profile Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Loads the uploaded workbook and profiles every sheet</li>
                <li>Columns, dtypes, ranges, sample values</li>
                <li>Rewrites the question against the real column names</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_1:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">❓</span>
            <div class="agent-title">2. Clarify Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Detects genuinely ambiguous questions</li>
                <li>Asks back with concrete options drawn from the schema</li>
                <li>Deliberately conservative - a default beats an interruption</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_2:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">🕸️</span>
            <div class="agent-title">3. RAG Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li><b>Glossary retrieval</b>: your certified definitions, hybrid lexical + semantic</li>
                <li><b>Query memory</b>: similar past analyses as few-shot examples</li>
                <li>Self-populating - accuracy grows with use</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    col_3, col_4, col_5 = st.columns(3)
    with col_3:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">🗺️</span>
            <div class="agent-title">4. Planner Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Splits compound questions into ordered steps</li>
                <li>Each step can build on the last via <code>prev</code></li>
                <li>Defaults to a single step - most questions need one</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_4:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">💻</span>
            <div class="agent-title">5. Analysis Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Generates pandas constrained to the profiled columns</li>
                <li>Grounded by retrieved definitions and examples</li>
                <li>On retry, receives the previous code <i>and its error</i></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_5:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">🛡️</span>
            <div class="agent-title">6. Impact Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Governance gate: audits the code's AST before it runs</li>
                <li>Blocks imports, file/network access, dunder escapes</li>
                <li>Rejects anything that would write data out</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    col_6, col_7, col_8 = st.columns(3)
    with col_6:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">⚡</span>
            <div class="agent-title">7. Execute Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Runs approved code in a restricted namespace</li>
                <li>Sheets are in-memory DataFrames - no database</li>
                <li>Runtime errors feed the reflexion edge</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_7:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">🔍</span>
            <div class="agent-title">8. Critic Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Catches code that <i>ran</i> but did not answer</li>
                <li>Empty results, all-NaN columns, missing breakdowns</li>
                <li>Can send the work back for another attempt</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_8:
        st.markdown("""
        <div class="agent-card">
            <span class="agent-icon">🧠</span>
            <div class="agent-title">9. Insight Agent</div>
            <ul class="agent-desc" style="padding-left: 1.2rem;">
                <li>Writes the analyst narrative over verified numbers</li>
                <li>Cites specific figures; never asserts causes</li>
                <li>Runs only after the Critic accepts the result</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    st.write("")

    st.markdown("---")

    # 3. End-to-End Flow (Visual Pipeline)
    st.subheader("End-to-End Execution Flow")
    st.markdown("**(UPLOAD → AGENTS → BI OUTPUT)**")

    st.markdown("""
    <div style='display: flex; flex-direction: column; gap: 10px;'>
        <div style='background: rgba(229, 9, 20, 0.1); border-left: 4px solid #E50914; padding: 15px; border-radius: 8px;'>
            <strong style='color: #E50914;'>STEP 1: UPLOAD</strong><br>
            <span style='color: #CCCCCC; font-size: 0.9rem;'>User uploads an Excel/CSV file. Each sheet is normalised and stored as Parquet; the columns are profiled. An optional business glossary is embedded into a vector index.</span>
        </div>
        <div style='text-align: center; color: #666;'>▼</div>
        <div style='background: rgba(255, 71, 87, 0.1); border-left: 4px solid #FF4757; padding: 15px; border-radius: 8px;'>
            <strong style='color: #FF4757;'>STEP 2: API &amp; ORCHESTRATOR</strong><br>
            <span style='color: #CCCCCC; font-size: 0.9rem;'>FastAPI resolves the dataset_id -&gt; LangGraph takes control of the agent swarm.</span>
        </div>
        <div style='text-align: center; color: #666;'>▼</div>
        <div style='background: rgba(255, 71, 87, 0.1); border-left: 4px solid #FF4757; padding: 15px; border-radius: 8px;'>
            <strong style='color: #FF4757;'>STEP 3: AGENT PIPELINE</strong><br>
            <span style='color: #CCCCCC; font-size: 0.9rem;'>Profile → Clarify → RAG → Planner → Analysis → Impact → Execute → Critic → Insight → BI.<br>
            Three agents can send work back to Analysis through one reflexion edge: <b>Impact</b> (code failed the audit), <b>Execute</b> (code raised), and <b>Critic</b> (code ran but did not answer). Every loop is bounded by an attempt budget.</span>
        </div>
        <div style='text-align: center; color: #666;'>▼</div>
        <div style='background: rgba(255, 255, 255, 0.1); border-left: 4px solid #FFFFFF; padding: 15px; border-radius: 8px;'>
            <strong style='color: #FFFFFF;'>STEP 4: RESPONSE &amp; LEARNING</strong><br>
            <span style='color: #CCCCCC; font-size: 0.9rem;'>Delivers KPIs, charts, the narrative insight, and the generated pandas. Analyses the Critic accepted are embedded into the query-memory index, so the next similar question retrieves them as worked examples.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



    st.markdown("---")
    
    # 5. Core Technologies & MCP
    col_tech_1, col_tech_2, col_tech_3 = st.columns(3)
    
    with col_tech_1:
        st.markdown("""
        <div style='background: #141414; padding: 20px; border-radius: 12px; border: 1px solid #1F1F1F; height: 100%;'>
            <h3 style='color: #FF4757;'>LangGraph</h3>
            <p style='color: #999999; font-size: 0.9rem;'>The 'Brain Controller'</p>
            <ul style='color: #CCCCCC; font-size: 0.9rem; margin-bottom: 0;'>
                <li>Deterministic execution</li>
                <li>Debuggable agent flows</li>
                <li>Shared state management</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_tech_2:
        st.markdown("""
        <div style='background: #141414; padding: 20px; border-radius: 12px; border: 1px solid #1F1F1F; height: 100%;'>
            <h3 style='color: #FF4757;'>Mem0 Memory</h3>
            <p style='color: #999999; font-size: 0.9rem;'>The 'Long-Term Storage'</p>
            <ul style='color: #CCCCCC; font-size: 0.9rem; margin-bottom: 0;'>
                <li>Remembers user prefs</li>
                <li>Cross-session memory</li>
                <li>Personalized BI context</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_tech_3:
        st.markdown("""
        <div style='background: #141414; padding: 20px; border-radius: 12px; border: 1px solid #1F1F1F; height: 100%;'>
            <h3 style='color: #FF4757;'>RAG Retrieval</h3>
            <p style='color: #999999; font-size: 0.9rem;'>The 'Grounding Layer'</p>
            <ul style='color: #CCCCCC; font-size: 0.9rem; margin-bottom: 0;'>
                <li>Local fastembed (ONNX) embeddings</li>
                <li>Hybrid exact-match + semantic glossary</li>
                <li>Self-populating query memory</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    
    # 6. Real Platform Capabilities (Vertical Layout)
    st.subheader("Platform Capabilities")

    # Card 1: Safety & Reliability
    st.markdown("""
    <div style='background: rgba(229, 9, 20, 0.1); border-left: 5px solid #E50914; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #E50914; margin-top: 0;'>Safety & Reliability</h3>
        <ul style='color: #CCCCCC; font-size: 0.95rem; line-height: 1.6;'>
            <li><b>AST Code Audit</b>: Every generated pandas script is parsed and audited before it runs - imports, file/network access, and dunder escapes are blocked.</li>
            <li><b>Sandboxed Execution</b>: Approved code runs in a restricted namespace against in-memory DataFrames - no database, nothing written back.</li>
            <li><b>Bounded Reflexion Loop</b>: Impact, Execute, and Critic can each send work back to Analysis with the exact failure attached, capped by <code>MAX_ANALYSIS_ATTEMPTS</code>.</li>
            <li><b>Upload Guardrails</b>: Row, file-size, and zip-decompression caps (checked while reading, not just against a zip's own metadata) prevent zip-bomb and OOM scenarios.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Card 2: RAG & Memory
    st.markdown("""
    <div style='background: rgba(255, 255, 255, 0.1); border-left: 5px solid #FFFFFF; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #FFFFFF; margin-top: 0;'>RAG & Long-Term Memory</h3>
        <ul style='color: #CCCCCC; font-size: 0.95rem; line-height: 1.6;'>
            <li><b>Business Glossary</b>: Hybrid exact-match + semantic retrieval, so "ARR" or "active customer" follows your certified definition instead of a guess.</li>
            <li><b>Self-Populating Query Memory</b>: Only Critic-verified analyses are embedded, so retrieval quality improves with real usage.</li>
            <li><b>Zero-Cost Embeddings</b>: Local ONNX inference via fastembed (BAAI/bge-small-en-v1.5) - no API key, no per-query cost.</li>
            <li><b>Cross-Session Memory</b>: Mem0, backed by a Qdrant vector store, remembers context across sessions and degrades to a safe no-op if unavailable.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Card 3: Deployment
    st.markdown("""
    <div style='background: rgba(255, 71, 87, 0.1); border-left: 5px solid #FF4757; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #FF4757; margin-top: 0;'>Deployment</h3>
        <div style='display: flex; gap: 20px; flex-wrap: wrap; margin-top: 15px;'>
            <div style='flex: 1; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;'>
                <strong style='color: #fff;'>💻 Local Development</strong>
                <ul style='color: #CCCCCC; font-size: 0.9rem; padding-left: 20px; margin-bottom: 0;'>
                    <li>Parquet datasets on local disk</li>
                    <li>FastAPI + Streamlit on one machine</li>
                    <li>In-process fallback if the API is offline</li>
                </ul>
            </div>
            <div style='flex: 1; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px;'>
                <strong style='color: #fff;'>☁️ Azure Container Apps</strong>
                <ul style='color: #CCCCCC; font-size: 0.9rem; padding-left: 20px; margin-bottom: 0;'>
                    <li>Backend & frontend scale independently</li>
                    <li>Backend scales to zero when idle</li>
                    <li>Azure Blob Storage swap for multi-replica</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("---")
    
    # 7. Deep Dive Expander (Covering actual file <-> feature mappings)
    with st.expander("📄 View Detailed Execution & File Structure (Project Explanation)", expanded=False):
        st.markdown("""
        ### Project Code Structure & Feature Mapping

        **1. USER INTERFACE**
        - `ui/presentation_app.py`: Main Streamlit app (upload, chat, KPIs, charts).
        - `ui/render_utils.py`: Chart rendering and brand-aware Plotly theming.

        **2. API LAYER**
        - `app/main.py`: FastAPI gateway - `/upload`, `/datasets/{id}/overview`, `/datasets/{id}/glossary`, `/ask`.

        **3. INTELLIGENT AGENTS** (`app/agents/`)
        - `profile_agent.py`: Profiles every sheet - columns, dtypes, ranges, samples.
        - `clarify_agent.py`: Asks back on genuinely ambiguous questions.
        - `rag_agent.py`: Retrieves glossary definitions and similar past analyses.
        - `planner_agent.py`: Splits compound questions into ordered steps.
        - `analysis_agent.py`: Generates the pandas code that answers the question.
        - `impact_agent.py`: AST safety audit before anything executes.
        - `execute_agent.py`: Runs approved code in a restricted namespace.
        - `critic_agent.py`: Validates the result actually answers the question.
        - `insight_agent.py`: Writes the analyst narrative over verified numbers.
        - `bi_agent.py`: Assembles KPIs, chart spec, and the final response.

        **4. ORCHESTRATION**
        - `app/langgraph/state.py`: Shared graph state definition.
        - `app/langgraph/graph.py`: Node wiring and the bounded reflexion edges.

        **5. RETRIEVAL & MEMORY**
        - `app/rag/glossary.py`, `app/rag/query_memory.py`, `app/rag/embedder.py`, `app/rag/vector_index.py`: Hybrid glossary + query-memory retrieval on local fastembed embeddings.
        - `app/memory/mem0_client.py`: Cross-session memory (Mem0 + Qdrant), with a safe no-op fallback.

        **6. STORAGE & CONFIG**
        - `app/storage/dataset_store.py`: Parquet persistence - local disk or Azure Blob.
        - `app/config.py`: Upload limits, model name, and feature flags in one place.
        """)
    
    st.markdown("---")

    # 8. Final Summary
    st.markdown("""
    <div style='background: linear-gradient(to right, #1F1F1F, #0A0A0A);
                padding: 25px; border-radius: 15px; text-align: center; border: 2px solid #FF4757;'>
        <h3 style='color: #fff; margin-bottom: 10px;'>SUMMARY</h3>
        <p style='color: #999999; margin-bottom: 20px;'>
            No SQL, no fixed schema, no dashboard to build - just a spreadsheet and a question.
        </p>
        <div style='display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;'>
            <span style='background: #FF4757; color: #000; padding: 5px 15px; border-radius: 5px; font-weight: bold;'>✅ 10-Agent LangGraph Pipeline</span>
            <span style='background: #FF4757; color: #000; padding: 5px 15px; border-radius: 5px; font-weight: bold;'>✅ RAG-Grounded Answers</span>
            <span style='background: #FF4757; color: #000; padding: 5px 15px; border-radius: 5px; font-weight: bold;'>✅ Audited, Sandboxed Code</span>
            <span style='background: #FF4757; color: #000; padding: 5px 15px; border-radius: 5px; font-weight: bold;'>✅ Azure-Ready</span>
        </div>
        <h2 style='color: #FF4757; margin-top: 25px;'>MegAI = NATURAL LANGUAGE ANALYSIS, END TO END</h2>
    </div>
    """, unsafe_allow_html=True)

# --- TAB 3: TECH STACK ---
# --- TAB 3: TECH STACK ---
with tab3:
    # 1. Premium Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(255, 71, 87, 0.1) 100%); 
                padding: 30px; border-radius: 15px; border-bottom: 4px solid #FF4757; margin-bottom: 25px;'>
        <h2 style='color: #FF4757; margin: 0 0 10px 0; font-weight: 800;'>The Complete Technology Stack</h2>
        <p style='color: #CCCCCC; font-size: 1.1rem; line-height: 1.6;'>
            Every layer that turns an uploaded spreadsheet into a governed, RAG-grounded answer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- DEFINERS for Reusability ---
    def tech_badge(label, color="#1F1F1F"):
        return f"<span style='display:inline-block; padding:4px 12px; margin:2px; border-radius:20px; background-color:{color}; border:1px solid #333333; font-size:0.8em; color:white;'>{label}</span>"

    # --- SECTION A: OVERVIEW (Always Visible) ---
    st.markdown("### High-Level Snapshot")
    st.markdown("A quick glance at the primary engines powering MegAI.")
    
    ov_col1, ov_col2, ov_col3 = st.columns(3)
    with ov_col1:
        with st.container(border=True):
            st.markdown("#### Frontend")
            st.markdown("Streamlit • Plotly")
            st.caption("Status: Active 🟢")
    with ov_col2:
        with st.container(border=True):
            st.markdown("#### AI Brain")
            st.markdown("LangGraph • Groq")
            st.caption("Status: Active 🟢")
    with ov_col3:
        with st.container(border=True):
            st.markdown("#### Storage & Memory")
            st.markdown("Parquet / Azure Blob • Mem0")
            st.caption("Status: Active 🟢")
            
    st.markdown("---")
    st.subheader("Detailed Breakdown (Click to Expand)")

    # --- SECTION B: Vertical Interactive Layers ---
    
    # 1. FRONTEND
    with st.expander("1. Frontend & Visualization Layer", expanded=True):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### 1. UI Framework")
            st.markdown(tech_badge("Streamlit") + tech_badge("Python"), unsafe_allow_html=True)
            st.markdown("""
            - **Interactive BI dashboards**
            - KPI cards, tables, charts
            - Dark mode support
            - Rapid UI development
            """)
        with col_f2:
            st.markdown("#### 2. Visualization")
            st.markdown(tech_badge("Plotly") + tech_badge("JSON"), unsafe_allow_html=True)
            st.markdown("""
            - **Dynamic charts** (bar, line, time-series)
            - BI-grade visualizations
            """)

    # 2. AI & BACKEND
    with st.expander("2. AI, Backend & Orchestration", expanded=False):
        c_ai1, c_ai2, c_ai3 = st.columns(3)
        with c_ai1:
            st.markdown("#### 3. Backend / API")
            st.markdown(tech_badge("FastAPI") + tech_badge("Uvicorn"), unsafe_allow_html=True)
            st.markdown("""
            - **Python 3.10+** (Core language)
            - **FastAPI**: REST API backend
              - High performance and async-ready
              - API documentation via Swagger
            - **Uvicorn**: ASGI server for FastAPI
            """)
            
            st.markdown("#### 8. Long-Term Memory")
            st.markdown(tech_badge("Mem0") + tech_badge("Qdrant"), unsafe_allow_html=True)
            st.markdown("""
            - **Mem0**: cross-session memory layer
            - Backed by a local **Qdrant** vector store
            - Degrades to a safe no-op if unavailable
            """)

        with c_ai2:
            st.markdown("#### 4. Orchestration")
            st.markdown(tech_badge("LangGraph") + tech_badge("LangChain"), unsafe_allow_html=True)
            st.markdown("""
            - **LangGraph**:
              - Agent orchestration framework
              - Deterministic state-based workflows
              - Bounded reflexion loop on failure
            - **LangChain**:
              - Groq LLM client (`langchain-groq`)
              - Prompt construction per agent
            """)

            st.markdown("#### 6. RAG System")
            st.markdown(tech_badge("fastembed") + tech_badge("FAISS"), unsafe_allow_html=True)
            st.markdown("""
            - **Vector Search**: FAISS-accelerated when installed, NumPy brute-force fallback otherwise
            - **Embeddings**:
              - fastembed (local ONNX, BAAI/bge-small-en-v1.5)
              - Azure OpenAI embeddings (optional)
            """)

        with c_ai3:
            st.markdown("#### 5. LLM")
            st.markdown(tech_badge("Groq"), unsafe_allow_html=True)
            st.markdown("""
            - **Groq** (`openai/gpt-oss-120b`):
              - pandas code generation
              - Clarification, planning & critique
              - Narrative insight writing
            """)

    # 3. DATA & SAFETY
    with st.expander("3. Data Storage & Safety", expanded=False):
        c_dat1, c_dat2 = st.columns(2)
        with c_dat1:
            st.markdown("#### 7. Storage Layer")
            st.markdown(tech_badge("Parquet") + tech_badge("Azure Blob"), unsafe_allow_html=True)
            st.markdown("""
            - **Parquet**: every uploaded sheet round-trips through pyarrow
            - **Local disk** for single-instance / development
            - **Azure Blob Storage**: swap-in backend for multi-replica deployments (`STORAGE_BACKEND=azure`)
            """)

            st.markdown("#### 10. Testing")
            st.markdown(tech_badge("pytest"), unsafe_allow_html=True)
            st.markdown("""
            - **177-test suite** covering profiling, retrieval, and end-to-end analysis correctness
            - Offline by default - no real Groq calls in CI
            """)

        with c_dat2:
            st.markdown("#### 9. Code Safety")
            st.markdown(tech_badge("AST Audit") + tech_badge("Sandboxing"), unsafe_allow_html=True)
            st.markdown("""
            - Every generated script is parsed and audited before execution
            - Blocks imports, file/network access, dunder escapes
            - Runs in a restricted namespace - no writes back to the file
            - Row, upload-size, and zip-decompression caps
            """)

    # 4. OPS & INFRA
    with st.expander("4. Ops & Deployment", expanded=False):
        c_ops1, c_ops2 = st.columns(2)
        with c_ops1:
            st.markdown("#### 11. Local Development")
            st.markdown("""
            - **FastAPI + Streamlit** run side by side
            - Streamlit falls back to invoking agents **in-process** if the API is unreachable
            - `.env` / `.env.example` for local configuration
            """)

            st.markdown("#### 13. Configuration")
            st.markdown("""
            - **python-dotenv**: environment variable management
            - All limits and feature flags centralised in `app/config.py`
            """)

        with c_ops2:
            st.markdown("#### 12. Deployment")
            st.markdown(tech_badge("Docker") + tech_badge("Azure Container Apps"), unsafe_allow_html=True)
            st.markdown("""
            - **Docker**: single image, two entry points (API and UI)
            - **Azure Container Apps**: backend and frontend scale independently, backend scales to zero
            - **Azure App Service**: single-container option for a simpler deploy
            """)

    st.markdown("---")
    # Final Summary Footer
    st.info("""
    **FINAL ARCHITECTURE SUMMARY**

    The combination of **LangGraph**, **RAG**, and a **Critic-gated reflexion loop** creates a system that is:
    1.  **Agentic**: it plans, writes, audits, and runs code - not just chat.
    2.  **Grounded**: every answer is checked against your glossary and past verified analyses.
    3.  **Self-correcting**: failures retry with the exact error attached, up to a bounded limit.
    """)

# --- TAB 4: HLD & LLD ---
with tab4:
    # 1. Premium Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(204, 204, 204, 0.1) 100%); 
                padding: 30px; border-radius: 15px; border-bottom: 4px solid #FFFFFF; margin-bottom: 25px;'>
        <h2 style='color: #FFFFFF; margin: 0 0 10px 0; font-weight: 800;'>Design & Architecture Specification</h2>
        <p style='color: #CCCCCC; font-size: 1.1rem; line-height: 1.6;'>
            Comprehensive documentation covering <b>Software Requirements (SRS)</b>, 
            <b>High-Level Design (HLD)</b>, and <b>Low-Level Design (LLD)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Helper for styled badges (Local scope to avoid errors)
    def design_badge(label, color="#141414"):
        return f"<span style='display:inline-block; padding:4px 10px; margin:2px; border-radius:15px; background-color:{color}; border:1px solid #333333; font-size:0.8em; color:white;'>{label}</span>"

    # 2. Vertical Expanders for Document Sections (Vertical Layout)
    
    # --- SECTION A: SRS ---
    with st.expander("1. Software Requirements Specification (SRS)", expanded=True):
        st.markdown("### 1. Software Requirements Specification")

        c_srs1, c_srs2 = st.columns(2)
        with c_srs1:
            with st.container(border=True):
                st.markdown("#### 1.1 Purpose")
                st.info("To define the requirements, architecture, and detailed design of the MegAI platform. The system lets a user upload a spreadsheet and ask questions about it in plain English, and returns a verified, explainable analysis - no SQL, no fixed schema.")
        with c_srs2:
            with st.container(border=True):
                st.markdown("#### 1.2 Scope")
                st.markdown("""
                - Natural language querying over uploaded Excel/CSV/zip files
                - 10-agent LangGraph pipeline with a bounded reflexion loop
                - RAG grounding via a business glossary and query memory
                - AST-audited, sandboxed code execution
                - Local or Azure Blob-backed deployment
                """)

        st.markdown("#### 1.3 Functional Requirements (FR)")
        fr_data = [
            {"ID": "FR-01", "Requirement": "System shall accept an Excel, CSV, or zip upload and profile every sheet."},
            {"ID": "FR-02", "Requirement": "System shall accept natural language business questions about the uploaded data."},
            {"ID": "FR-03", "Requirement": "System shall generate pandas code constrained to the profiled columns."},
            {"ID": "FR-04", "Requirement": "System shall audit generated code before execution and reject unsafe operations."},
            {"ID": "FR-05", "Requirement": "System shall execute approved code in a restricted, sandboxed namespace."},
            {"ID": "FR-06", "Requirement": "System shall retry failed analyses with the exact error attached, up to a bounded limit."},
            {"ID": "FR-07", "Requirement": "System shall generate KPIs, a chart, and a narrative insight from verified results."},
            {"ID": "FR-08", "Requirement": "System shall retrieve certified glossary definitions and similar past analyses via RAG."},
            {"ID": "FR-09", "Requirement": "System shall persist datasets as Parquet, locally or in Azure Blob Storage."}
        ]
        st.table(fr_data)

        c_nfr1, c_nfr2 = st.columns(2)
        with c_nfr1:
            st.markdown("#### 1.4 User Classes")
            st.markdown("- Business User asking ad-hoc questions\n- Analyst validating generated code\n- Developer operating the deployment")
        with c_nfr2:
            st.markdown("#### 1.6 Non-Functional Requirements")
            st.markdown("""
            - **Performance**: bounded by container RAM - a sheet is fully loaded for analysis
            - **Safety**: every script is AST-audited before it runs
            - **Reliability**: bounded reflexion retries, no unbounded loops
            - **Maintainability**: one agent, one file, one responsibility
            """)

    # --- SECTION B: HLD ---
    with st.expander("2. High Level Design (HLD)", expanded=False):
        st.markdown("### 2. High Level Design")

        with st.container(border=True):
            st.markdown("#### 2.1 Architectural Overview")
            st.code("User → Streamlit UI → FastAPI Backend → LangGraph Orchestrator → Agentic AI Pipeline → BI Output", language="text")

        c_hld1, c_hld2 = st.columns(2)
        with c_hld1:
            with st.container(border=True):
                st.markdown("#### 2.2 System Components")
                st.markdown("""
                - **Presentation Layer**: Streamlit
                - **API Layer**: FastAPI
                - **Orchestration**: LangGraph
                - **Agent Layer**: Profile, Clarify, RAG, Planner, Analysis, Impact, Execute, Critic, Insight, BI
                - **Memory Layer**: RAG store (glossary + query memory) + Mem0
                - **Data Layer**: Parquet, local or Azure Blob
                """)
        with c_hld2:
            with st.container(border=True):
                st.markdown("#### 2.3 Deployment Architecture")
                st.markdown(design_badge("Local Dev") + " Single process, local Parquet", unsafe_allow_html=True)
                st.markdown(design_badge("Azure Container Apps") + " Independent scaling, scale-to-zero backend", unsafe_allow_html=True)
                st.divider()
                st.markdown("#### 2.4 Data Flow")
                st.markdown("**User Question** → Agent Pipeline (with reflexion retries) → Critic Verification → BI Output → **Query Memory**")

    # --- SECTION C: LLD ---
    with st.expander("3. Low Level Design (LLD)", expanded=False):
        st.markdown("### 3. Low Level Design")

        st.markdown("#### 3.1 Presentation & API Layers")
        c_lld1, c_lld2 = st.columns(2)
        with c_lld1:
            st.markdown("**3.1 Presentation Layer**")
            st.markdown("- Streamlit chat UI\n- KPI cards, charts, code viewer\n- In-process fallback if the API is offline")
        with c_lld2:
            st.markdown("**3.2 API Layer**")
            st.markdown("- FastAPI REST endpoints (`/upload`, `/ask`, `/datasets/{id}/overview`)\n- JSON request/response\n- Stateless - dataset_id resolves the file")

        st.divider()
        st.markdown("#### 3.3 & 3.4 Orchestration & Agents (Detailed)")
        st.markdown("**3.3 Orchestration Layer**: LangGraph shared state, deterministic node order, three conditional reflexion edges back to Analysis.")

        # Agent Grid
        ag1, ag2, ag3 = st.columns(3)
        with ag1:
            st.markdown(design_badge("3.4.1 Profile Agent"), unsafe_allow_html=True)
            st.markdown("- Profiles every sheet\n- Rewrites the question against real column names")

            st.markdown(design_badge("3.4.4 Impact Agent"), unsafe_allow_html=True)
            st.markdown("- AST-audits generated code\n- Blocks imports & file/network access")

        with ag2:
            st.markdown(design_badge("3.4.2 RAG Agent"), unsafe_allow_html=True)
            st.markdown("- Retrieves glossary + past analyses\n- Adds grounded context")

            st.markdown(design_badge("3.4.5 Execute Agent"), unsafe_allow_html=True)
            st.markdown("- Runs approved code in a sandbox\n- Returns the result DataFrame")

        with ag3:
            st.markdown(design_badge("3.4.3 Analysis Agent"), unsafe_allow_html=True)
            st.markdown("- Generates pandas from the question\n- Retries with the prior error on failure")

            st.markdown(design_badge("3.4.6 Critic + BI Agent"), unsafe_allow_html=True)
            st.markdown("- Critic verifies the result answers the question\n- BI Agent assembles KPIs, chart & response")

        st.divider()
        st.markdown("#### 3.5, 3.6, 3.7 Memory, Retrieval & Storage")
        c_mg1, c_mg2, c_mg3 = st.columns(3)
        with c_mg1:
            st.markdown("**3.5 RAG Store**")
            st.markdown("- Glossary: hybrid exact + semantic\n- Query memory: only Critic-verified analyses\n- Local fastembed embeddings")
        with c_mg2:
            st.markdown("**3.6 Mem0 Memory**")
            st.markdown("- Cross-session context via Qdrant\n- Safe no-op fallback if unavailable")
        with c_mg3:
            st.markdown("**3.7 Dataset Storage**")
            st.markdown("- Parquet round-trip via pyarrow\n- Local disk or Azure Blob (`STORAGE_BACKEND`)")

    st.markdown("---")
    # Conclusion Footer
    st.success("""
    **CONCLUSION**

    The MegAI platform is a self-correcting, RAG-grounded analysis system: every generated script is audited,
    every result is critic-verified before reaching the user, and the architecture deploys the same way locally
    or on Azure Container Apps.
    """)

# --- TAB 5: ARCHITECTURE ---
with tab5:
    # 1. Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(255, 71, 87, 0.1) 0%, rgba(183, 28, 28, 0.1) 100%);
                padding: 30px; border-radius: 15px; border-bottom: 4px solid #FF4757; margin-bottom: 25px;'>
        <h2 style='color: #FF4757; margin: 0 0 10px 0; font-weight: 800;'>System Architecture &amp; Blueprint</h2>
        <p style='color: #CCCCCC; font-size: 1.1rem; line-height: 1.6;'>
            A 10-agent LangGraph pipeline that turns an uploaded spreadsheet and a plain-English question into a verified, RAG-grounded answer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Helper for badges
    def arch_badge(label, color="#141414"):
        return f"<span style='display:inline-block; padding:4px 10px; margin:2px; border-radius:15px; background-color:{color}; border:1px solid #333333; font-size:0.8em; color:white;'>{label}</span>"

    # 2. Architecture Diagram (Image)
    st.subheader("Architectural Blueprint")

    # Robust Image Finder for Cloud/Local compatibility
    img_filename = "agentic_bi_architecture.jpeg"
    possible_paths = [
        img_filename,
        os.path.join("agentic-bi-saas", img_filename),
        os.path.join(os.getcwd(), img_filename),
    ]

    img_found = None
    for p in possible_paths:
        if os.path.exists(p):
            img_found = p
            break

    if img_found:
        st.image(img_found, caption="MegAI - Agentic Data Analysis Pipeline", width='stretch')
    else:
        st.warning(f"Architecture diagram ({img_filename}) not found. Ensure it is present in the repository root.")

    # 3. Interactive Graphviz Chart
    with st.expander("Interactive Service Map (Live Diagram)", expanded=True):
        st.graphviz_chart("""
        digraph G {
            rankdir=LR;
            bgcolor="transparent";
            node [shape=box, style=filled, fillcolor="#CCCCCC", fontname="Sans-Serif", color="#999999"];
            edge [color="#999999", fontname="Sans-Serif", fontsize=10];

            User [label="User", shape=ellipse, fillcolor="#E50914", fontcolor=white];
            UI [label="Streamlit UI", fillcolor="#E50914", fontcolor=white];

            subgraph cluster_backend {
                style=dashed; color="#999999";
                label = "FastAPI + LangGraph";
                API [label="FastAPI Gateway", fillcolor="#FF4757", fontcolor=white];
                Orchestrator [label="10-Agent Pipeline\\n(Profile → ... → BI)", fillcolor="#FF4757", fontcolor=white];
            }

            subgraph cluster_data {
                style=filled; color="#1F1F1F";
                label = "Storage & Retrieval";
                Dataset [label="Dataset Store\\n(Parquet / Azure Blob)", shape=cylinder, fillcolor="#FFFFFF"];
                RagStore [label="RAG Store\\n(Glossary + Query Memory)", shape=cylinder, fillcolor="#FFFFFF"];
                Mem0 [label="Mem0\\n(Qdrant)", shape=cylinder, fillcolor="#FFFFFF"];
            }

            Groq [label="Groq LLM", shape=component, fillcolor="#141414", fontcolor=white];

            User -> UI [label="question"];
            UI -> API [label="HTTP"];
            API -> Orchestrator [label="invoke graph"];
            Orchestrator -> Dataset [label="load sheet"];
            Orchestrator -> RagStore [label="retrieve / embed"];
            Orchestrator -> Mem0 [label="context"];
            Orchestrator -> Groq [label="generate / critique"];
            Orchestrator -> API [label="KPIs, chart, code"];
            API -> UI [label="JSON response"];
        }
        """)

    # 4. Detailed Architecture Content (Vertical Layout)

    # --- SECTION A: HLD ---
    with st.expander("1. High-Level Flow & Design (HLD)", expanded=False):
        c_hld1, c_hld2 = st.columns(2)
        with c_hld1:
            st.markdown("#### 1. System Flow")
            st.code("""
User (Browser)
  |
Streamlit UI
  |
FastAPI Gateway
  |
LangGraph Orchestrator
  |
Profile -> Clarify -> RAG -> Planner -> Analysis
  -> Impact -> Execute -> Critic -> Insight -> BI
  |
Response (KPIs, Chart, Data Table, Code)
  |
Query Memory (critic-verified analyses only)
            """, language="text")

        with c_hld2:
            st.markdown("#### 2.2 System Components")
            st.markdown(f"""
            - **Presentation Layer**: Streamlit
            - **API Layer**: FastAPI Gateway
            - **Orchestration**: LangGraph (Controller)
            - **Agent Layer**: {arch_badge("Profile")} {arch_badge("Clarify")} {arch_badge("RAG")} {arch_badge("Planner")} {arch_badge("Analysis")} {arch_badge("Impact")} {arch_badge("Execute")} {arch_badge("Critic")} {arch_badge("Insight")} {arch_badge("BI")}
            - **Shared Services**: {arch_badge("RAG Store", "#FF4757")} {arch_badge("Mem0", "#FFFFFF")} {arch_badge("AST Audit", "#E50914")}
            - **Data Layer**: Parquet (local / Azure Blob)
            """, unsafe_allow_html=True)

            st.markdown("#### HLD Key Points")
            st.success("""
            - UI is separated from the backend, with an in-process fallback if the API is offline
            - Agents run in a deterministic, centrally orchestrated graph
            - Three conditional edges (Impact, Execute, Critic) form a bounded reflexion loop
            - Deploys identically on a laptop or on Azure Container Apps
            """)

    # --- SECTION B: LLD (Components) ---
    with st.expander("2. Component Details (LLD)", expanded=False):
        c_lld1, c_lld2 = st.columns(2)

        with c_lld1:
            st.markdown("#### 3.1 UI Layer (Streamlit)")
            st.markdown("- **Input**: uploaded file + natural language question\n- **Output**: KPI cards, chart, data table, generated code\n- Maintains chat history in session state")

            st.markdown("#### 3.2 API Layer (FastAPI)")
            st.markdown("- Stateless REST endpoints keyed by `dataset_id`\n- Resolves the dataset, invokes the graph\n- Routes to LangGraph")

        with c_lld2:
            st.markdown("#### 3.3 LangGraph Orchestration")
            st.markdown("**Shared State**: question, profile, plan, code, result, response")
            st.markdown("**Responsibilities**: deterministic node order, bounded retries, no unbounded loops")
            st.markdown("**Agent Execution Order**:")
            st.code("""
1. Profile   6. Impact
2. Clarify   7. Execute
3. RAG       8. Critic
4. Planner   9. Insight
5. Analysis  10. BI
            """, language="text")

        st.divider()
        st.markdown("#### 3.4 Agent Details (The Swarm)")

        # Agent Grid
        ag1, ag2, ag3 = st.columns(3)
        with ag1:
            st.info("**Profile Agent**\n\n- Profiles every sheet\n- Rewrites the question against real columns\n- Outputs schema context")
            st.info("**Impact Agent**\n\n- AST-audits generated code\n- Blocks imports, file/network access\n- Rejects writes back to the file")
            st.info("**Insight Agent**\n\n- Writes the analyst narrative\n- Cites specific figures only\n- Runs after the Critic accepts")
        with ag2:
            st.info("**Clarify Agent**\n\n- Detects genuinely ambiguous questions\n- Asks back with concrete options\n- Conservative by default")
            st.info("**Execute Agent**\n\n- Runs approved code in a sandbox\n- In-memory DataFrames, no database\n- Runtime errors feed the reflexion edge")
            st.info("**BI Agent**\n\n- Computes KPIs\n- Builds the chart spec\n- Assembles the final response")
        with ag3:
            st.info("**RAG Agent**\n\n- Retrieves glossary definitions\n- Retrieves similar past analyses\n- Hybrid exact + semantic match")
            st.info("**Critic Agent**\n\n- Checks the result actually answers\n- Catches empty / all-NaN results\n- Can send work back for another attempt")
            st.info("**Planner Agent**\n\n- Splits compound questions into steps\n- Each step can build on the last\n- Defaults to a single step")

    # --- SECTION C: Shared Services ---
    with st.expander("3. Shared Services & Deployment", expanded=False):
        c_srv1, c_srv2, c_srv3 = st.columns(3)

        with c_srv1:
            with st.container(border=True):
                st.markdown("#### 3.5 RAG Store")
                st.markdown("- Business glossary (hybrid retrieval)\n- Self-populating query memory\n- Local fastembed (ONNX) embeddings")

        with c_srv2:
            with st.container(border=True):
                st.markdown("#### 3.6 Mem0 Memory")
                st.markdown("- Cross-session context, backed by Qdrant\n- Groq LLM + HuggingFace embedder\n- Degrades to a safe no-op if unavailable")

        with c_srv3:
            with st.container(border=True):
                st.markdown("#### 3.7 Code Safety")
                st.markdown("- AST audit before every execution\n- Restricted execution namespace\n- Upload, row & zip-decompression caps")

        st.divider()
        st.markdown("#### 4. Deployment Models")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"### {arch_badge('Local Development', '#141414')}", unsafe_allow_html=True)
            st.markdown("- Parquet on local disk\n- FastAPI + Streamlit on one machine\n- In-process agent fallback")
        with d_col2:
            st.markdown(f"### {arch_badge('Azure Container Apps', '#E50914')}", unsafe_allow_html=True)
            st.markdown("- Backend & frontend scale independently\n- Backend scales to zero when idle\n- Azure Blob Storage for multi-replica")

    st.markdown("---")
    st.success("""
    **FINAL ARCHITECTURE SUMMARY**

    This architecture is **agentic** (it plans, writes, audits, and runs code), **grounded** (every answer is checked
    against your glossary and past verified analyses), and **self-correcting** (bounded retries with the exact
    failure attached) - and it deploys the same way locally or on Azure.
    """)

# --- TAB 6: SYSTEM LOGS ---
# --- TAB 6: SYSTEM LOGS ---
with tab6:
    # 1. Premium Header
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, rgba(229, 9, 20, 0.1) 100%); 
                padding: 20px; border-radius: 15px; border-bottom: 4px solid #E50914; margin-bottom: 25px;'>
        <h2 style='color: #E50914; margin: 0 0 10px 0; font-weight: 800;'>System Diagnostic & Event Logs</h2>
        <p style='color: #CCCCCC; font-size: 1.0rem; line-height: 1.5;'>
            Real-time monitoring of <b>Agent Orchestration</b>, <b>Code Generation</b>, and <b>System Events</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # When this session started - also stamps the boot log below.
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = get_ist_time()

    # Facts that are actually true and actually useful in a diagnostics
    # tab. The panel this replaces reported a random Guest_User_#### that
    # identified nothing and a hardcoded "Securely Connected" status that
    # was never checked against anything.
    with st.container(border=True):
        st.markdown("### Session Details")
        c_sess1, c_sess2, c_sess3 = st.columns(3)
        with c_sess1:
            st.markdown(f"**Session Started**\n\n`{st.session_state.session_start_time}`")
        with c_sess2:
            _ds = st.session_state.get("dataset")
            st.markdown(f"**Dataset Loaded**\n\n`{_ds['filename'] if _ds else 'none'}`")
        with c_sess3:
            if _backend_status():
                st.markdown("**Compute**\n\n<span style='color:#FFFFFF'>FastAPI backend</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("**Compute**\n\n<span style='color:#FF4757'>In-process fallback</span>",
                            unsafe_allow_html=True)

    st.markdown("---")

    logs = st.session_state.get('system_logs', [])
    
    # Inject initial connection logs if empty (Simulation of Boot Sequence)
    if not logs:
        boot_sequence = [
            ("System", "Kernel Initializing...", "INFO"),
            ("Module Load", "LangGraph Orchestrator [OK]", "SUCCESS"),
            ("Module Load", "RAG Store (Glossary + Query Memory) [OK]", "SUCCESS"),
            ("Module Load", "Groq Inference Engine [OK]", "SUCCESS"),
            ("Agent Swarm", "10 Agents Registered (Profile, Clarify, RAG, Planner, Analysis, Impact, Execute, Critic, Insight, BI)", "INFO"),
            ("System", "Session initialized. Waiting for interactions...", "INFO")
        ]
        
        # Populate in newest-first order
        for idx, (evt, det, lvl) in enumerate(reversed(boot_sequence)):
             logs.append({
                "Timestamp": st.session_state.session_start_time, 
                "Event": evt,
                "Details": det,
                "Level": lvl 
            })

    # 2. Metrics & Parsing
    total_events = len(logs)
    unique_agents = len(set([l['Event'] for l in logs])) if logs else 0
    error_count = sum(1 for l in logs if str(l.get('Level', '')).upper() == "ERROR")
    
    # Summary Metrics (Custom HTML Cards)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #FFFFFF; text-align: center;'>
            <h4 style='color: #CCCCCC; margin: 0; font-size: 0.9rem;'>TOTAL EVENTS</h4>
            <p style='color: #FFFFFF; font-size: 1.8rem; font-weight: bold; margin: 5px 0;'>{total_events}</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style='background: rgba(229, 9, 20, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #E50914; text-align: center;'>
            <h4 style='color: #CCCCCC; margin: 0; font-size: 0.9rem;'>EVENT TYPES</h4>
            <p style='color: #E50914; font-size: 1.8rem; font-weight: bold; margin: 5px 0;'>{unique_agents}</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        err_color = "#E50914" if error_count > 0 else "#999999"
        st.markdown(f"""
        <div style='background: rgba(229, 9, 20, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid {err_color}; text-align: center;'>
            <h4 style='color: #CCCCCC; margin: 0; font-size: 0.9rem;'>ERRORS DETECTED</h4>
            <p style='color: {err_color}; font-size: 1.8rem; font-weight: bold; margin: 5px 0;'>{error_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    # 3. Controls (Advanced Omni-Filters)
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    
    with f_col1:
        all_event_types = sorted(list(set([l['Event'] for l in logs]))) if logs else []
        selected_types = st.multiselect("🔍 Filter by Agent / Event Type:", all_event_types, default=all_event_types)
    
    with f_col2:
        selected_severity = st.selectbox("🚦 Severity Filter:",
                                         ["All Levels"] + list(_SEVERITY_CHOICES))
        
    with f_col3:
        search_query = st.text_input("📝 Keyword Search:", placeholder="e.g. 'SQL' or 'North'")

    # Export row
    c_act2, c_act3, c_spacer = st.columns([1.2, 1.2, 5.2])
    with c_act2:
        if logs:
            # TEXT EXPORT
            log_text = "MegAI SYSTEM LOGS\n======================\n"
            log_text += f"Export Time: {get_ist_time()}\n\n"
            for log in reversed(logs):
                log_text += f"[{log.get('Timestamp', '')}] {log.get('Event', '')}: {log.get('Details', '')}\n"
            
            st.download_button(
                label="📄 Export TXT",
                data=log_text,
                file_name=f"agentic_bi_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                width='stretch',
                key="btn_export_txt"
            )

    with c_act3:
        if logs:
            # CSV EXPORT
            df_logs = pd.DataFrame(logs)
            csv_logs = df_logs.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📊 Export CSV",
                data=csv_logs,
                file_name=f"agentic_bi_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width='stretch',
                key="btn_export_csv"
            )
            
    with st.sidebar:
        st.markdown("---")
        if st.sidebar.button("Simulate System Error", help="Add a manual error log for testing"):
            log_event("Security", "MANUAL ALERT: Simulated system error triggered by admin.", "ERROR")

    # 4. Scrollable Log Feed
    st.markdown("### Real-time Event Feed")
    
    # Remove fixed height to show ALL logs as requested
    log_container = st.container(border=True)
    with log_container:
        if not logs:
            st.info("📭 No logs generated yet. Start interacting with the Demo tab to see the Agent Swarm in action!")
        else:
            # log_event inserts at 0, so the list 'logs' is already newest-first.
            # Showing newest at the top.
            for log in logs:
                # 1. Type Filter
                type_match = log['Event'] in selected_types
                
                # 2. Severity Filter - reads the level the event was logged
                # with, rather than guessing from the wording of the message.
                details_lower = str(log.get('Details', '')).lower()
                level = str(log.get('Level', 'INFO')).upper()
                sev_match = (selected_severity == "All Levels"
                             or level == _SEVERITY_CHOICES.get(selected_severity))
                
                # 3. Keyword Filter
                kw_match = True
                if search_query:
                    kw_match = search_query.lower() in details_lower or search_query.lower() in str(log['Event']).lower()
                
                # Combined Logic
                if type_match and sev_match and kw_match:
                    # Enforce CST Formatting if missing
                    raw_time = str(log.get('Timestamp', ''))
                    if "CST" not in raw_time:
                         timestamp = f"{raw_time} CST"
                    else:
                         timestamp = raw_time
                         
                    event = log.get('Event', 'Unknown')
                    details = log.get('Details', '')
                    
                    # Style Logic - driven by the stored level
                    if level == "ERROR":
                        st.error(f"**[{timestamp}] {event}**: {details}", icon="🚨")
                    elif level == "SUCCESS":
                        st.success(f"**[{timestamp}] {event}**: {details}", icon="✅")
                    elif level == "WARNING":
                        st.warning(f"**[{timestamp}] {event}**: {details}", icon="⚠️")
                    else:
                        st.info(f"**[{timestamp}] {event}**: {details}", icon="ℹ️")

# --- FOOTER ---
st.markdown("<hr style='border: none; height: 1px; background: linear-gradient(90deg, transparent, #E50914, transparent); opacity: 0.6; margin: 40px 0 30px 0;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align: center; padding: 24px 20px 16px 20px; background: linear-gradient(180deg, rgba(20, 20, 20, 0.6) 0%, rgba(10, 10, 10, 0.3) 100%); border-radius: 14px; border: 1px solid rgba(229, 9, 20, 0.25);'>
    <p style='color: #FFFFFF; font-weight: 700; font-size: 1.1rem; margin-bottom: 6px; letter-spacing: 0.3px;'><img src='{LOGO_DATA_URI}' width='26' height='26' style='margin-right: 8px; vertical-align: middle;'> MegAI Platform</p>
    <p style='color: #999999; font-weight: 500; font-size: 0.9rem; margin-bottom: 20px;'>Built by <span style='color: #FF4757; font-weight: 700;'>Megha Patel</span></p>
    <div style='max-width: 780px; margin: 0 auto;'>
        <span style='color: #666666; font-size: 0.7rem; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; display: block; margin-bottom: 8px;'>Tech Stack:</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #E50914; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>LangGraph</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #E50914; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>LangChain</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #E50914; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>Groq</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #E50914; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>Mem0</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #FF4757; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>FAISS</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #FF4757; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>Fastembed</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #FF4757; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>Pandas</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #FF4757; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>Plotly</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #B71C1C; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>FastAPI</span>
        <span style='display: inline-block; padding: 5px 14px; margin: 3px; border-radius: 20px; background-color: #141414; border: 1px solid #B71C1C; font-size: 0.8em; color: #FFFFFF; font-weight: 500;'>Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Social links
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

with col2:
    st.markdown('<p style="text-align: center; margin: 0;"><a href="https://github.com/meghapatel21" target="_blank" style="text-decoration: none; color: #FFFFFF; font-size: 1.1rem; font-weight: 600;">GitHub</a></p>', unsafe_allow_html=True)

with col3:
    st.markdown('<p style="text-align: center; margin: 0;"><a href="mailto:patelmegha1726@gmail.com" style="text-decoration: none; color: #E50914; font-size: 1.1rem; font-weight: 600;">Email</a></p>', unsafe_allow_html=True)

with col4:
    st.markdown('<p style="text-align: center; margin: 0;"><a href="https://linkedin.com/in/meghapatel21" target="_blank" style="text-decoration: none; color: #FF4757; font-size: 1.1rem; font-weight: 600;">LinkedIn</a></p>', unsafe_allow_html=True)

# Close the visual footer
st.markdown("""
<div style='height: 10px; background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(229, 9, 20, 0.15) 100%); border-radius: 0 0 10px 10px; border-bottom: 2px solid #FFFFFF; margin-top: -10px;'></div>
""", unsafe_allow_html=True)

