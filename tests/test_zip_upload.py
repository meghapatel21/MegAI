"""Zip uploads: extraction, multi-file merging, and bomb protection.

Bomb-protection tests use highly compressible payloads (repeated bytes zip at
roughly 1000:1) so a small, fast-to-build fixture can still blow past a
(monkeypatched, low) limit - proving the guard works without needing an
actual gigabyte-scale file in the test suite.
"""

import io
import zipfile

import pandas as pd
import pytest

from app import config
from app.storage.dataset_store import DatasetError, parse_workbook


def _zip_of(files: dict) -> bytes:
    """files: {member_name: bytes}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _highly_compressible(size: int) -> bytes:
    return b"0" * size


# --- happy paths ------------------------------------------------------------

def test_zip_of_multiple_csvs_becomes_multiple_sheets():
    raw = _zip_of({
        "orders.csv": b"order_id,revenue\n1,100\n2,200\n",
        "customers.csv": b"customer_id,country\n1,US\n2,UK\n",
    })
    frames = parse_workbook(raw, "export.zip")
    assert set(frames) == {"orders", "customers"}
    assert list(frames["orders"].columns) == ["order_id", "revenue"]
    assert list(frames["customers"].columns) == ["customer_id", "country"]


def test_zip_of_single_xlsx_matches_uploading_that_xlsx_directly():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    xlsx_bytes = xlsx_buf.getvalue()

    direct = parse_workbook(xlsx_bytes, "report.xlsx")
    zipped = parse_workbook(_zip_of({"report.xlsx": xlsx_bytes}), "report.zip")

    assert list(direct) == list(zipped)
    pd.testing.assert_frame_equal(direct["sheet1"], zipped["sheet1"])


def test_zip_with_excel_multisheet_and_csv_together():
    df = pd.DataFrame({"x": [1], "y": [2]})
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Budget", index=False)

    raw = _zip_of({
        "finance.xlsx": xlsx_buf.getvalue(),
        "notes.csv": b"note\nhello\n",
    })
    frames = parse_workbook(raw, "bundle.zip")
    assert set(frames) == {"budget", "notes"}


def test_zip_ignores_macos_junk_entries():
    raw = _zip_of({
        "__MACOSX/._data.csv": b"junk",
        "data.csv": b"a,b\n1,2\n",
        "__MACOSX/orders/._x": b"junk",
    })
    frames = parse_workbook(raw, "export.zip")
    assert list(frames) == ["data"]


def test_zip_ignores_non_data_entries():
    raw = _zip_of({
        "readme.txt": b"about this export",
        "data.csv": b"a,b\n1,2\n",
        "cover.pdf": b"%PDF-fake",
    })
    frames = parse_workbook(raw, "export.zip")
    assert list(frames) == ["data"]


def test_zip_same_basename_in_different_folders_gets_disambiguated():
    raw = _zip_of({
        "2024/summary.csv": b"a,b\n1,2\n",
        "2025/summary.csv": b"a,b\n3,4\n",
    })
    frames = parse_workbook(raw, "export.zip")
    assert len(frames) == 2  # both kept, under distinct sheet keys


# --- rejections --------------------------------------------------------------

def test_zip_with_no_data_files_is_rejected():
    raw = _zip_of({"readme.txt": b"nothing useful here"})
    with pytest.raises(DatasetError, match="No spreadsheet"):
        parse_workbook(raw, "export.zip")


def test_corrupt_zip_is_rejected_cleanly():
    with pytest.raises(DatasetError, match="[Nn]ot a valid zip|corrupt"):
        parse_workbook(b"this is not a zip file at all", "export.zip")


def test_truncated_zip_is_rejected_cleanly():
    good = _zip_of({"data.csv": b"a,b\n1,2\n" * 100})
    truncated = good[: len(good) // 2]
    with pytest.raises(DatasetError):
        parse_workbook(truncated, "export.zip")


def test_too_many_members_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_ZIP_MEMBERS", 2)
    raw = _zip_of({f"f{i}.csv": b"a,b\n1,2\n" for i in range(5)})
    with pytest.raises(DatasetError, match="limit is 2"):
        parse_workbook(raw, "export.zip")


# --- zip bomb protection ------------------------------------------------------

def test_zip_bomb_is_rejected_without_fully_materialising_it(monkeypatch):
    """A tiny zip that claims/would-decompress to far more than the cap must
    be rejected, and rejected while still reading it - not after."""
    monkeypatch.setattr(config, "MAX_ZIP_UNCOMPRESSED_BYTES", 5 * 1024 * 1024)  # 5 MB cap
    raw = _zip_of({"bomb.csv": _highly_compressible(50 * 1024 * 1024)})  # 50 MB real payload
    assert len(raw) < 100 * 1024, "fixture should compress hard, or this test proves nothing"

    with pytest.raises(DatasetError, match="expands past"):
        parse_workbook(raw, "export.zip")


def test_zip_bomb_cap_is_a_total_across_all_members(monkeypatch):
    """Several members individually under the cap, but over it combined."""
    monkeypatch.setattr(config, "MAX_ZIP_UNCOMPRESSED_BYTES", 5 * 1024 * 1024)
    raw = _zip_of({
        f"part{i}.csv": _highly_compressible(3 * 1024 * 1024) for i in range(3)
    })  # 9 MB combined, each part alone is under the 5 MB cap

    with pytest.raises(DatasetError, match="expands past"):
        parse_workbook(raw, "export.zip")


def test_zip_within_the_cap_succeeds(monkeypatch):
    monkeypatch.setattr(config, "MAX_ZIP_UNCOMPRESSED_BYTES", 10 * 1024 * 1024)
    raw = _zip_of({"small.csv": b"a,b\n1,2\n2,3\n"})
    frames = parse_workbook(raw, "export.zip")
    assert list(frames) == ["small"]
