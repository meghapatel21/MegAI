"""Persistence for uploaded spreadsheets.

Sheets are normalised to Parquet on upload so that every later query is a cheap,
typed read instead of re-parsing the workbook. `DatasetStore` is intentionally
small: swapping local disk for Azure Blob is a matter of implementing the four
methods, not touching the agents.
"""

import importlib.util
import io
import json
import os
import re
import shutil
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd

from app import config

# Zip entries that are never data, regardless of extension: directory markers,
# and the resource-fork junk macOS adds to almost every zip it creates.
_ZIP_JUNK_PREFIXES = ("__MACOSX/",)
_ZIP_READ_CHUNK = 1024 * 1024  # 1 MB

# openpyxl is pure Python and reads roughly 100k spreadsheet rows a second,
# which dominates the upload time of any sizeable .xlsx. python-calamine is a
# drop-in pandas engine that reads the same files an order of magnitude faster.
# It is optional on purpose - it ships a compiled extension, and this app has
# to keep working on machines whose policy blocks those - so it is used when
# importable and silently skipped when not.
# find_spec rather than a try/import: the module is never actually used here,
# only detected, and an unused import is a lint failure.
_EXCEL_KWARGS = (
    {"engine": "calamine"}
    if importlib.util.find_spec("python_calamine") is not None
    else {}
)


class DatasetError(Exception):
    """Raised for anything the caller should see as a 4xx."""


def _safe_sheet_name(raw: str, taken: set) -> str:
    """Turn a sheet title into an identifier usable as a Python variable."""
    name = re.sub(r"\W+", "_", str(raw)).strip("_").lower()
    if not name or name[0].isdigit():
        name = f"sheet_{name}" if name else "sheet"
    candidate, n = name, 2
    while candidate in taken:
        candidate = f"{name}_{n}"
        n += 1
    return candidate


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names so generated code can reference them predictably."""
    seen, cols = set(), []
    for col in df.columns:
        name = re.sub(r"\W+", "_", str(col)).strip("_").lower() or "column"
        if name[0].isdigit():
            name = f"col_{name}"
        candidate, n = name, 2
        while candidate in seen:
            candidate = f"{name}_{n}"
            n += 1
        seen.add(candidate)
        cols.append(candidate)
    df = df.copy()
    df.columns = cols
    return df


def _parse_single_file(raw: bytes, filename: str) -> "OrderedDict[str, pd.DataFrame]":
    """Read one spreadsheet (not a zip) into {sheet_key: DataFrame}."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise DatasetError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )

    try:
        if ext == ".csv":
            # Named after the file, not hardcoded "data" - matters once a zip
            # can hold several CSVs that all need distinct sheet keys.
            stem = os.path.splitext(os.path.basename(filename))[0] or "data"
            frames = {stem: pd.read_csv(io.BytesIO(raw))}
        else:
            frames = pd.read_excel(io.BytesIO(raw), sheet_name=None, **_EXCEL_KWARGS)
    except Exception as exc:
        raise DatasetError(f"Could not read '{filename}': {exc}") from exc

    if not frames:
        raise DatasetError(f"'{filename}' contains no readable sheets.")

    out, taken = OrderedDict(), set()
    for title, df in frames.items():
        if df is None or df.empty:
            continue
        if len(df) > config.MAX_ROWS_PER_SHEET:
            raise DatasetError(
                f"'{title}' has {len(df):,} rows; the limit is {config.MAX_ROWS_PER_SHEET:,}."
            )
        key = _safe_sheet_name(title, taken)
        taken.add(key)
        out[key] = _clean_columns(df)

    if not out:
        raise DatasetError(f"Every sheet in '{filename}' is empty.")
    return out


def _read_zip_member_bounded(zf: zipfile.ZipFile, info: zipfile.ZipInfo, remaining: int) -> bytes:
    """Decompress one zip entry, aborting the instant it exceeds `remaining`.

    A zip entry's declared `file_size` is metadata the archive's creator
    controls, not a verified fact - a crafted archive can claim almost nothing
    and still decompress into gigabytes (a zip bomb). Reading in bounded
    chunks means the cap holds even when that metadata lies, because nothing
    beyond `remaining` bytes is ever materialised in memory to begin with.
    """
    chunks, total = [], 0
    with zf.open(info) as fh:
        while True:
            chunk = fh.read(_ZIP_READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > remaining:
                raise DatasetError(
                    f"The zip expands past the "
                    f"{config.MAX_ZIP_UNCOMPRESSED_BYTES / 1e6:.0f} MB decompressed limit "
                    f"(while reading '{info.filename}')."
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _extract_zip(raw: bytes) -> "OrderedDict[str, bytes]":
    """Pull the data files out of an uploaded zip: {member_filename: raw_bytes}."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise DatasetError(f"Not a valid zip file: {exc}") from exc

    bad = zf.testzip()
    if bad is not None:
        raise DatasetError(f"The zip is corrupt (failed integrity check on '{bad}').")

    candidates = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        if any(info.filename.startswith(p) for p in _ZIP_JUNK_PREFIXES):
            continue
        basename = os.path.basename(info.filename)
        if basename.startswith("._") or not basename:
            continue
        if os.path.splitext(basename)[1].lower() not in config.ALLOWED_EXTENSIONS:
            continue
        candidates.append(info)

    if not candidates:
        raise DatasetError(
            "No spreadsheet found inside the zip. Allowed inside a zip: "
            f"{', '.join(sorted(config.ALLOWED_EXTENSIONS))}"
        )
    if len(candidates) > config.MAX_ZIP_MEMBERS:
        raise DatasetError(
            f"The zip has {len(candidates)} data files; the limit is {config.MAX_ZIP_MEMBERS}."
        )

    out, taken, budget = OrderedDict(), set(), config.MAX_ZIP_UNCOMPRESSED_BYTES
    for info in candidates:
        data = _read_zip_member_bounded(zf, info, budget)
        budget -= len(data)

        name = os.path.basename(info.filename)
        candidate, n = name, 2
        while candidate in taken:  # two members with the same basename in different folders
            stem, ext = os.path.splitext(name)
            candidate = f"{stem}_{n}{ext}"
            n += 1
        taken.add(candidate)
        out[candidate] = data

    return out


def parse_workbook(raw: bytes, filename: str) -> "OrderedDict[str, pd.DataFrame]":
    """Read an uploaded file - spreadsheet or zip of spreadsheets - into
    {sheet_key: DataFrame}. A zip's members are merged into one dataset, each
    becoming one or more sheets, the same way multiple Excel tabs would."""
    if os.path.splitext(filename)[1].lower() != ".zip":
        return _parse_single_file(raw, filename)

    members = _extract_zip(raw)

    merged, taken = OrderedDict(), set()
    for member_name, member_bytes in members.items():
        for title, df in _parse_single_file(member_bytes, member_name).items():
            key = _safe_sheet_name(title, taken)  # dedup across files too
            taken.add(key)
            merged[key] = df

    return merged


def profile_frames(frames: "OrderedDict[str, pd.DataFrame]") -> dict:
    """Describe the workbook well enough for an LLM to write pandas against it."""
    sheets = []
    for key, df in frames.items():
        columns = []
        for col in df.columns:
            series = df[col]
            info = {
                "name": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
            }
            non_null = series.dropna()
            if pd.api.types.is_numeric_dtype(series) and not non_null.empty:
                info["min"] = float(non_null.min())
                info["max"] = float(non_null.max())
            elif not non_null.empty:
                # unique() first, then stringify: converting every value in a
                # multi-million-row column to str only to discard all but 12 of
                # them is wasted work.
                uniques = non_null.unique()
                info["unique_count"] = int(len(uniques))
                # Sample values matter more than stats for text columns: they tell
                # the model whether "East" or "east" or "EAST" is in the data.
                info["sample_values"] = [str(v) for v in uniques[:12]]
            columns.append(info)
        sheets.append({
            "name": key,
            "row_count": int(len(df)),
            "columns": columns,
        })
    return {"sheets": sheets, "primary_sheet": sheets[0]["name"] if sheets else None}


class DatasetStore(ABC):
    @abstractmethod
    def save(self, raw: bytes, filename: str) -> dict:
        """Persist an upload. Returns {dataset_id, filename, profile}."""

    @abstractmethod
    def load(self, dataset_id: str) -> "OrderedDict[str, pd.DataFrame]":
        """Return the sheets for a dataset."""

    @abstractmethod
    def get_meta(self, dataset_id: str) -> dict:
        """Return {dataset_id, filename, profile} without loading the data."""

    @abstractmethod
    def delete(self, dataset_id: str) -> None:
        ...


class LocalDatasetStore(DatasetStore):
    """Parquet-on-disk. Fine for single-instance runs and local development."""

    def __init__(self, root: str = None, cache_size: int = 4):
        self.root = root or config.DATASET_DIR
        os.makedirs(self.root, exist_ok=True)
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def _dir(self, dataset_id: str) -> str:
        # Guard against traversal - dataset_id reaches us from the client.
        if not re.fullmatch(r"[0-9a-f]{32}", dataset_id or ""):
            raise DatasetError("Invalid dataset id.")
        return os.path.join(self.root, dataset_id)

    def save(self, raw: bytes, filename: str) -> dict:
        frames = parse_workbook(raw, filename)
        dataset_id = uuid.uuid4().hex
        final_path = os.path.join(self.root, dataset_id)
        # Written under a staging name first, then renamed into place in one
        # step - a large multi-sheet zip is exactly the case most likely to
        # fail partway (a row-cap hit on sheet 4 of 6, disk pressure, a bad
        # dtype), and a bare loop over `final_path` would leave those earlier
        # sheets on disk with no meta.json, invisible to get_meta() but never
        # cleaned up either.
        staging_path = os.path.join(self.root, f".staging-{dataset_id}")
        os.makedirs(staging_path, exist_ok=True)

        try:
            for key, df in frames.items():
                df.to_parquet(os.path.join(staging_path, f"{key}.parquet"), index=False)

            meta = {
                "dataset_id": dataset_id,
                "filename": filename,
                "sheet_order": list(frames.keys()),
                "profile": profile_frames(frames),
            }
            with open(os.path.join(staging_path, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)

            os.replace(staging_path, final_path)  # atomic: staging never partially visible
        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise

        self._cache[dataset_id] = frames
        self._trim_cache()
        return meta

    def get_meta(self, dataset_id: str) -> dict:
        meta_path = os.path.join(self._dir(dataset_id), "meta.json")
        if not os.path.exists(meta_path):
            raise DatasetError("Dataset not found. Please upload the file again.")
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)

    def load(self, dataset_id: str) -> "OrderedDict[str, pd.DataFrame]":
        if dataset_id in self._cache:
            self._cache.move_to_end(dataset_id)
            return self._cache[dataset_id]

        meta = self.get_meta(dataset_id)
        path = self._dir(dataset_id)
        frames = OrderedDict()
        for key in meta["sheet_order"]:
            frames[key] = pd.read_parquet(os.path.join(path, f"{key}.parquet"))

        self._cache[dataset_id] = frames
        self._trim_cache()
        return frames

    def delete(self, dataset_id: str) -> None:
        self._cache.pop(dataset_id, None)
        shutil.rmtree(self._dir(dataset_id), ignore_errors=True)

    def _trim_cache(self) -> None:
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


class AzureBlobDatasetStore(DatasetStore):
    """Blob-backed store for multi-instance Azure deployments.

    Layout mirrors LocalDatasetStore: `<dataset_id>/meta.json` plus one Parquet
    blob per sheet, so instances behind a load balancer all see the same data.
    Requires `azure-storage-blob` and AZURE_STORAGE_CONNECTION_STRING.
    """

    def __init__(self, connection_string: str = None, container: str = None, cache_size: int = 4):
        from azure.storage.blob import BlobServiceClient  # imported lazily

        conn = connection_string or config.AZURE_STORAGE_CONNECTION_STRING
        if not conn:
            raise DatasetError("AZURE_STORAGE_CONNECTION_STRING is not set.")

        self._service = BlobServiceClient.from_connection_string(conn)
        self._container_name = container or config.AZURE_STORAGE_CONTAINER
        self._container = self._service.get_container_client(self._container_name)
        try:
            self._container.create_container()
        except Exception:
            pass  # already exists

        self._cache = OrderedDict()
        self._cache_size = cache_size

    @staticmethod
    def _check_id(dataset_id: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{32}", dataset_id or ""):
            raise DatasetError("Invalid dataset id.")
        return dataset_id

    def save(self, raw: bytes, filename: str) -> dict:
        frames = parse_workbook(raw, filename)
        dataset_id = uuid.uuid4().hex

        try:
            for key, df in frames.items():
                buf = io.BytesIO()
                df.to_parquet(buf, index=False)
                buf.seek(0)
                self._container.upload_blob(f"{dataset_id}/{key}.parquet", buf, overwrite=True)

            meta = {
                "dataset_id": dataset_id,
                "filename": filename,
                "sheet_order": list(frames.keys()),
                "profile": profile_frames(frames),
            }
            # meta.json is written last and is what get_meta()/load() key off
            # of, so nothing above this line is visible to a caller yet even
            # without the local store's rename trick.
            self._container.upload_blob(
                f"{dataset_id}/meta.json", json.dumps(meta, indent=2), overwrite=True
            )
        except Exception:
            # Blob Storage has no atomic multi-object rename, so a failure
            # midway through a multi-sheet upload leaves orphaned blobs unless
            # they are swept up here.
            for blob in self._container.list_blobs(name_starts_with=f"{dataset_id}/"):
                self._container.delete_blob(blob.name)
            raise

        self._cache[dataset_id] = frames
        self._trim_cache()
        return meta

    def get_meta(self, dataset_id: str) -> dict:
        self._check_id(dataset_id)
        try:
            blob = self._container.get_blob_client(f"{dataset_id}/meta.json")
            return json.loads(blob.download_blob().readall())
        except Exception as exc:
            raise DatasetError("Dataset not found. Please upload the file again.") from exc

    def load(self, dataset_id: str) -> "OrderedDict[str, pd.DataFrame]":
        if dataset_id in self._cache:
            self._cache.move_to_end(dataset_id)
            return self._cache[dataset_id]

        meta = self.get_meta(dataset_id)
        frames = OrderedDict()
        for key in meta["sheet_order"]:
            blob = self._container.get_blob_client(f"{dataset_id}/{key}.parquet")
            frames[key] = pd.read_parquet(io.BytesIO(blob.download_blob().readall()))

        self._cache[dataset_id] = frames
        self._trim_cache()
        return frames

    def delete(self, dataset_id: str) -> None:
        self._check_id(dataset_id)
        self._cache.pop(dataset_id, None)
        for blob in self._container.list_blobs(name_starts_with=f"{dataset_id}/"):
            self._container.delete_blob(blob.name)

    def _trim_cache(self) -> None:
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)


def _build_store() -> DatasetStore:
    if config.STORAGE_BACKEND == "azure":
        return AzureBlobDatasetStore()
    return LocalDatasetStore()


store = _build_store()
