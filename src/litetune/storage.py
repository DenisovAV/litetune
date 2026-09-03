"""Where a run's artifacts live, addressed by name and identified by content.

Two rules shape this module.

**A key is a name, not a path.** The interface takes strings and returns bytes;
nothing in it mentions a filesystem. That is not abstraction for its own sake --
the same pipeline has to run on a laptop and behind a hosted service, and a
`Path` in the interface would put a local filesystem in every signature the
service cannot honour. `LocalStorage` is the only backend in scope here; the
Protocol is what lets a bucket-backed one arrive without touching the runner.

**Identity is the content, never the location.** `content_hash` exists so that
the cache key of a downstream stage moves when the bytes move. Keying on the
key -- the location -- is how a dataset replaced at the same URI silently trains
the next run on stale data under a green manifest, which is the failure this
whole layer is arranged to prevent.

A missing key raises. It does not return `b""`, `None` or an empty list: an
absent artifact read as an empty one is a confident answer that nothing
established, and callers here are computing release decisions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

logger = logging.getLogger(__name__)

# Hashes are returned prefixed with the algorithm that produced them. A backend
# that can only offer a different digest (an object store's etag, say) then
# reports a visibly different string, so its values miss the cache rather than
# colliding with a sha256 and reading as a hit.
HASH_ALGORITHM = "sha256"

# Artifacts here are model files; a `.litertlm` export measured 285,577,392
# bytes. Hashing streams in chunks so that identifying one does not cost its
# size in resident memory.
_CHUNK_BYTES = 1024 * 1024

# Partial writes are renamed into place, never written in place, so that a
# crash mid-write cannot leave a truncated artifact that hashes cleanly and
# reads as complete.
_TMP_PREFIX = ".litetune-tmp-"

_RESERVED_SEGMENTS = frozenset({"", ".", ".."})


class StorageError(Exception):
    """Base for every error this module raises."""


class KeyNotFound(StorageError):
    """No object is stored under this key."""


class InvalidKey(StorageError, ValueError):
    """The key is not a well-formed name."""


class Storage(Protocol):
    """Named byte objects. Implementations must not assume a local filesystem.

    Keys are `/`-separated names, relative and without `.` or `..` segments.
    Every read of an absent key raises `KeyNotFound`.
    """

    def read_bytes(self, key: str) -> bytes:
        """The stored bytes. Raises `KeyNotFound` if the key is absent."""

    def write_bytes(self, key: str, data: bytes) -> None:
        """Store `data`, replacing anything already at `key`."""

    def read_text(self, key: str) -> str:
        """The stored bytes decoded as UTF-8."""

    def write_text(self, key: str, text: str) -> None:
        """Store `text` encoded as UTF-8."""

    def exists(self, key: str) -> bool:
        """Whether anything is stored under `key`."""

    def list(self, prefix: str = "") -> list[str]:
        """Every key beginning with `prefix`, sorted."""

    def content_hash(self, key: str) -> str:
        """`"sha256:<hex>"` over the stored bytes. Raises `KeyNotFound` if absent.

        This is what a cache key is built from, so it must describe the bytes
        and nothing else -- not the key, not an mtime, not a size.
        """

    def size(self, key: str) -> int:
        """Stored bytes, as a count. Raises `KeyNotFound` if absent.

        Not derivable from `read_bytes` at acceptable cost: the manifest records
        a size for every artifact, and `len(read_bytes(key))` would pull a
        285 MB export into memory to learn it.
        """


@runtime_checkable
class SupportsFileIngest(Protocol):
    """Optional capability: take a local file without reading it into memory.

    Deliberately outside `Storage`, because it does name a local path and the
    core interface must not. The runner checks for it with `isinstance` -- which
    is why this one is `runtime_checkable` -- and falls back to `write_bytes`,
    so a backend that cannot offer it stays usable.
    """

    def ingest(self, key: str, source: Path) -> None:
        """Store the contents of the local file `source` under `key`."""


def validate_key(key: str) -> str:
    """Check a key's shape, returning it. Raises `InvalidKey` naming the key.

    A key is a name. `../../etc/passwd` is a path, and honouring it would let a
    stage's declared artifact write outside the run's storage; a backend that
    happens to be a filesystem must not turn a naming mistake into an escape.
    """
    if not isinstance(key, str) or not key:
        raise InvalidKey(f"key must be a non-empty string, got {key!r}")
    if key != key.strip():
        raise InvalidKey(f"key {key!r} has leading or trailing whitespace")
    if key.startswith("/") or "\\" in key or "\0" in key:
        raise InvalidKey(f"key {key!r} must be a relative, '/'-separated name with no backslashes")
    bad = [segment for segment in key.split("/") if segment in _RESERVED_SEGMENTS]
    if bad:
        raise InvalidKey(
            f"key {key!r} has an empty or relative segment {bad!r}: a key is a name, not a path"
        )
    return key


def hash_bytes(data: bytes) -> str:
    """The same digest format `Storage.content_hash` returns, for in-memory bytes."""
    return f"{HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def hash_file(path: Path) -> str:
    """The same digest format, streamed from a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"{HASH_ALGORITHM}:{digest.hexdigest()}"


@dataclass(frozen=True)
class LocalStorage:
    """`Storage` over a directory. The only backend in the current scope."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    # -- internals ---------------------------------------------------------

    def _path(self, key: str) -> Path:
        return self.root / validate_key(key)

    def _existing(self, key: str) -> Path:
        path = self._path(key)
        if not path.is_file():
            raise KeyNotFound(f"no object stored under {key!r} in {self.root}")
        return path

    # -- Storage -----------------------------------------------------------

    def read_bytes(self, key: str) -> bytes:
        path = self._existing(key)
        try:
            return path.read_bytes()
        except OSError:
            logger.exception("could not read %s (key %r)", path, key)
            raise

    def write_bytes(self, key: str, data: bytes) -> None:
        self._replace(key, lambda tmp: tmp.write_bytes(data))

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, text: str) -> None:
        self.write_bytes(key, text.encode("utf-8"))

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        if not self.root.is_dir():
            # No directory means no objects. That is an answer about the store,
            # not a swallowed error: nothing was written here.
            return []
        keys = [
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and not path.name.startswith(_TMP_PREFIX)
        ]
        return sorted(key for key in keys if key.startswith(prefix))

    def content_hash(self, key: str) -> str:
        return hash_file(self._existing(key))

    def size(self, key: str) -> int:
        return self._existing(key).stat().st_size

    # -- local-only extras -------------------------------------------------

    def ingest(self, key: str, source: Path) -> None:
        """Copy a local file in, without reading it into memory. See `SupportsFileIngest`."""
        if not source.is_file():
            raise KeyNotFound(f"cannot ingest {source}: it is not a file")
        self._replace(key, lambda tmp: shutil.copyfile(source, tmp))

    def local_path(self, key: str) -> Path:
        """Where this backend keeps `key`. Local-only, and not part of `Storage`."""
        return self._path(key)

    def _replace(self, key: str, write) -> None:
        """Write through a temporary name and rename over the target.

        `os.replace` is atomic within a filesystem, so a reader either sees the
        previous object or the complete new one. A half-written artifact that
        hashes cleanly is indistinguishable from a real one downstream.
        """
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f"{_TMP_PREFIX}{uuid4().hex}"
        try:
            write(tmp)
            os.replace(tmp, path)
        except OSError:
            logger.exception("could not write key %r under %s", key, self.root)
            raise
        finally:
            tmp.unlink(missing_ok=True)


def put_file(storage: Storage, key: str, source: Path) -> None:
    """Store a local file, using the backend's streaming path when it has one."""
    if isinstance(storage, SupportsFileIngest):
        storage.ingest(key, source)
        return
    storage.write_bytes(key, source.read_bytes())
