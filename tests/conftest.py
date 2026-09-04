import io
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the store at a scratch directory and disable the LLM before any app
# module is imported, so tests never call Groq or touch a developer's datasets.
#
# Set (not pop!) to a placeholder config.py already treats as "no key" - if
# the var were simply removed, app.config's load_dotenv() call would just
# read a real GROQ_API_KEY straight back out of a developer's .env file
# (load_dotenv does not override a key that IS present, but happily fills in
# one that's absent), silently turning every test into a real Groq call.
os.environ["DATASET_DIR"] = tempfile.mkdtemp(prefix="agentic_bi_tests_")
os.environ["GROQ_API_KEY"] = "your_groq_api_key"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


@pytest.fixture(scope="session")
def workbook_bytes():
    """A two-sheet workbook with the messy headers real uploads have."""
    rng = np.random.default_rng(42)
    orders = pd.DataFrame({
        "Order ID": range(1, 241),
        "Order Date": np.repeat(
            pd.date_range("2025-01-01", periods=12, freq="MS").astype(str), 20
        ),
        "Region ": rng.choice(["East", "West", "North", "South"], 240),
        "Revenue ($)": rng.uniform(100, 5000, 240).round(2),
        "Customer_ID": rng.integers(1, 40, 240),
    })
    customers = pd.DataFrame({
        "Customer_ID": range(1, 41),
        "Country": rng.choice(["US", "UK", "India"], 40),
        "Tier": rng.choice(["Gold", "Silver", "Bronze"], 40),
    })

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="Q1 Orders", index=False)
        customers.to_excel(writer, sheet_name="Customers", index=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def dataset(workbook_bytes):
    from app.storage.dataset_store import store
    return store.save(workbook_bytes, "sales_report.xlsx")


@pytest.fixture
def base_state(dataset):
    question = "total revenue by region"
    return {
        "dataset_id": dataset["dataset_id"],
        "question": question,
        "corrected_question": question,
        "user_id": "test_user",
        "profile": dataset["profile"],
        "filename": dataset["filename"],
        "history": [],
        "attempts": 1,
    }
