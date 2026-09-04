"""Compatibility shims for optional native dependencies.

Some environments block a compiled extension outright - for example a Windows
Application Control (WDAC) policy refusing to load `xxhash`'s `.pyd` file,
which is not something this application can fix, only work around. Each shim
here is a last resort: it is only installed when the real package fails to
import, so a normal environment (Docker on Linux, most local machines) keeps
using the real, faster implementation untouched.
"""

import hashlib
import sys
import types


def ensure_xxhash():
    """Install a pure-Python xxhash stand-in if the real package cannot load.

    langgraph (task ids) and langsmith (replica ids) both use xxhash purely for
    deterministic hashing - nothing in this app depends on its output matching
    upstream xxhash bit-for-bit, only that the same input always produces the
    same output within one run. hashlib.blake2b satisfies that with no native
    code, so it is a safe stand-in specifically for this use.
    """
    try:
        # Bound to a name and referenced below: this has to be a real import
        # (the failure mode is a *load* error, not a missing module, so
        # importlib.util.find_spec would not detect it), and pyflakes - which
        # CI runs - does not honour "# noqa", so an unused import fails Lint.
        import xxhash
        _ = xxhash.__name__
        return  # the real package works - nothing to do
    except Exception as exc:
        reason = exc

    if "xxhash" in sys.modules:
        return  # already shimmed, or something else claimed the module name

    stub = types.ModuleType("xxhash")

    class _Digest:
        def __init__(self, data: bytes = b""):
            self._data = data if isinstance(data, (bytes, bytearray)) else str(data).encode()

        def digest(self) -> bytes:
            return hashlib.blake2b(self._data, digest_size=16).digest()

        def hexdigest(self) -> str:
            return hashlib.blake2b(self._data, digest_size=16).hexdigest()

        def intdigest(self) -> int:
            return int.from_bytes(self.digest(), "big")

    def xxh3_128(data: bytes = b"", seed: int = 0):
        return _Digest(data)

    def xxh3_128_hexdigest(data: bytes = b"", seed: int = 0) -> str:
        return _Digest(data).hexdigest()

    def xxh3_128_intdigest(data: bytes = b"", seed: int = 0) -> int:
        return _Digest(data).intdigest()

    def xxh64_hexdigest(data: bytes = b"", seed: int = 0) -> str:
        return hashlib.blake2b(
            data if isinstance(data, (bytes, bytearray)) else str(data).encode(),
            digest_size=8,
        ).hexdigest()

    def xxh32_hexdigest(data: bytes = b"", seed: int = 0) -> str:
        return hashlib.blake2b(
            data if isinstance(data, (bytes, bytearray)) else str(data).encode(),
            digest_size=4,
        ).hexdigest()

    stub.xxh3_128 = xxh3_128
    stub.xxh3_128_hexdigest = xxh3_128_hexdigest
    stub.xxh3_128_intdigest = xxh3_128_intdigest
    stub.xxh64_hexdigest = xxh64_hexdigest
    stub.xxh32_hexdigest = xxh32_hexdigest

    sys.modules["xxhash"] = stub
    print(f"[COMPAT] xxhash native module unavailable ({reason}); "
          "using a pure-Python stand-in. Hashing stays deterministic, just not "
          "bit-identical to upstream xxhash - fine for task/replica ids, which "
          "only need to be stable within a run, not portable across one.")
