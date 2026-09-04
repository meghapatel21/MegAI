from collections import OrderedDict

import pandas as pd
import pytest

from app.storage.dataset_store import DatasetError, parse_workbook, store


def test_normalises_messy_column_names(dataset):
    columns = [c["name"] for c in dataset["profile"]["sheets"][0]["columns"]]
    # "Region " and "Revenue ($)" must survive as safe Python identifiers.
    assert "region" in columns
    assert "revenue" in columns
    assert all(c.isidentifier() for c in columns)


def test_sheet_names_become_identifiers(dataset):
    names = [s["name"] for s in dataset["profile"]["sheets"]]
    assert names == ["q1_orders", "customers"]
    assert dataset["profile"]["primary_sheet"] == "q1_orders"


def test_profile_describes_columns(dataset):
    columns = {c["name"]: c for c in dataset["profile"]["sheets"][0]["columns"]}
    assert columns["revenue"]["min"] < columns["revenue"]["max"]
    assert set(columns["region"]["sample_values"]) == {"East", "West", "North", "South"}
    assert dataset["profile"]["sheets"][0]["row_count"] == 240


def test_round_trips_through_parquet(dataset):
    frames = store.load(dataset["dataset_id"])
    assert list(frames) == ["q1_orders", "customers"]
    assert len(frames["q1_orders"]) == 240


def test_rejects_path_traversal():
    for bad in ["../../etc", "..", "a/b", "", "x" * 32]:
        with pytest.raises(DatasetError):
            store.get_meta(bad)


def test_rejects_unknown_extension():
    with pytest.raises(DatasetError, match="Unsupported file type"):
        parse_workbook(b"MZ\x00", "malware.exe")


def test_rejects_unreadable_file():
    with pytest.raises(DatasetError):
        parse_workbook(b"not really a spreadsheet", "broken.xlsx")


def test_reads_csv():
    frames = parse_workbook(b"Name,Total Sales\nA,10\nB,20\n", "data.csv")
    assert list(frames) == ["data"]
    assert list(frames["data"].columns) == ["name", "total_sales"]


def test_duplicate_columns_are_disambiguated():
    # "a" and "A" collide once case is folded, so the store must still hand the
    # agents unique, referenceable identifiers.
    columns = list(parse_workbook(b"a,A,a\n1,2,3\n", "dupes.csv")["dupes"].columns)
    assert len(columns) == len(set(columns)) == 3
    assert all(c.isidentifier() for c in columns)


def test_csv_sheet_key_comes_from_the_filename():
    frames = parse_workbook(b"a,b\n1,2\n", "quarterly_sales.csv")
    assert list(frames) == ["quarterly_sales"]


# --- crash-safety of save() -------------------------------------------------

def test_partial_multisheet_write_leaves_no_orphaned_dataset(monkeypatch, tmp_path):
    """A failure on sheet 2 of 2 must not leave sheet 1's parquet file
    sitting on disk under a directory get_meta()/load() can't see but that
    never gets cleaned up either."""
    from app.storage.dataset_store import LocalDatasetStore

    local_store = LocalDatasetStore(root=str(tmp_path))

    original_to_parquet = pd.DataFrame.to_parquet
    calls = {"n": 0}

    def flaky_to_parquet(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise IOError("simulated disk failure on the second sheet")
        return original_to_parquet(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", flaky_to_parquet)

    raw = b"a,b\n1,2\n"  # single-sheet CSV, but save() also writes meta.json
    # Force two "sheets" so there is a second to_parquet call to fail on.
    import app.storage.dataset_store as ds_module
    monkeypatch.setattr(
        ds_module, "parse_workbook",
        lambda raw, filename: OrderedDict([
            ("sheet_one", pd.DataFrame({"a": [1]})),
            ("sheet_two", pd.DataFrame({"a": [2]})),
        ]),
    )

    with pytest.raises(IOError, match="simulated disk failure"):
        local_store.save(raw, "two_sheets.csv")

    # No dataset directory (final or staging) should remain.
    leftover = list(tmp_path.iterdir())
    assert leftover == [], f"expected no leftover directories, found: {leftover}"


def test_successful_multisheet_save_is_visible_atomically(tmp_path):
    from app.storage.dataset_store import LocalDatasetStore

    local_store = LocalDatasetStore(root=str(tmp_path))
    raw = b"a,b\n1,2\n2,3\n"
    meta = local_store.save(raw, "ok.csv")

    assert (tmp_path / meta["dataset_id"]).is_dir()
    assert not any(p.name.startswith(".staging-") for p in tmp_path.iterdir())
    assert local_store.get_meta(meta["dataset_id"])["dataset_id"] == meta["dataset_id"]
