"""Storage identifies objects by their content, and a missing one is an error.

The two failures these tests pin: a key that is really a path can write outside
the run, and an absent artifact read as empty bytes reads downstream as a real
measurement over an empty file.
"""

from pathlib import Path

import pytest

from litetune.storage import (
    InvalidKey,
    KeyNotFound,
    LocalStorage,
    hash_bytes,
    hash_file,
    put_file,
    validate_key,
)


@pytest.fixture
def store(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "store")


def test_bytes_round_trip(store):
    store.write_bytes("artifacts/model.litertlm", b"\x00\x01weights")
    assert store.read_bytes("artifacts/model.litertlm") == b"\x00\x01weights"


def test_text_round_trip(store):
    store.write_text("runs/r1/manifest.json", '{"status": "passed"}')
    assert store.read_text("runs/r1/manifest.json") == '{"status": "passed"}'


def test_missing_key_raises_rather_than_returning_empty(store):
    # An absent artifact read as b"" is a confident answer about a file that is
    # not there, which is the whole failure mode this project is arranged around.
    with pytest.raises(KeyNotFound) as exc:
        store.read_bytes("artifacts/absent.bin")
    assert "artifacts/absent.bin" in str(exc.value)


def test_missing_key_has_no_hash_and_no_size(store):
    with pytest.raises(KeyNotFound):
        store.content_hash("nope")
    with pytest.raises(KeyNotFound):
        store.size("nope")


def test_exists_is_false_for_an_absent_key(store):
    assert not store.exists("nope")
    store.write_text("nope", "x")
    assert store.exists("nope")


def test_content_hash_names_its_algorithm_and_matches_the_bytes(store):
    store.write_bytes("a", b"weights")
    assert store.content_hash("a") == hash_bytes(b"weights")
    assert store.content_hash("a").startswith("sha256:")


def test_content_hash_moves_when_the_content_does(store):
    store.write_bytes("dataset.jsonl", b"one")
    before = store.content_hash("dataset.jsonl")
    # The same key, different bytes: this is a dataset replaced at its URI, and
    # the hash is the only thing that notices.
    store.write_bytes("dataset.jsonl", b"two")
    assert store.content_hash("dataset.jsonl") != before


def test_size_reports_the_stored_bytes(store):
    store.write_bytes("a", b"0123456789")
    assert store.size("a") == 10


def test_write_replaces_rather_than_appends(store):
    store.write_text("a", "first")
    store.write_text("a", "second")
    assert store.read_text("a") == "second"


@pytest.mark.parametrize(
    "key",
    ["/absolute", "a/../../escape", "..", "a//b", "", "  spaced", "back\\slash", "a/./b"],
)
def test_a_key_is_a_name_not_a_path(store, key):
    # `../../etc/passwd` would let a stage's declared artifact write outside the
    # run's storage; a naming mistake must not become an escape.
    with pytest.raises(InvalidKey):
        validate_key(key)
    with pytest.raises(InvalidKey):
        store.write_text(key, "x")


def test_list_is_sorted_and_filtered_by_prefix(store):
    for key in ("runs/r2/manifest.json", "runs/r1/manifest.json", "cache/index.json"):
        store.write_text(key, "{}")
    assert store.list() == ["cache/index.json", "runs/r1/manifest.json", "runs/r2/manifest.json"]
    assert store.list("runs/") == ["runs/r1/manifest.json", "runs/r2/manifest.json"]


def test_list_of_an_absent_root_is_empty(tmp_path):
    assert LocalStorage(tmp_path / "never-written").list() == []


def test_list_hides_partial_writes(store, tmp_path):
    store.write_text("a", "x")
    (store.root / ".litetune-tmp-deadbeef").write_text("half a model")
    # A temporary name is a write in progress, not an artifact; listing it would
    # offer a truncated file to anything that hashes what it finds.
    assert store.list() == ["a"]


def test_put_file_stores_a_local_file(store, tmp_path):
    source = tmp_path / "model.litertlm"
    source.write_bytes(b"exported weights")
    put_file(store, "runs/r1/export/model.litertlm", source)
    assert store.read_bytes("runs/r1/export/model.litertlm") == b"exported weights"
    assert store.content_hash("runs/r1/export/model.litertlm") == hash_file(source)


def test_ingest_of_a_missing_file_raises(store, tmp_path):
    with pytest.raises(KeyNotFound):
        store.ingest("a", tmp_path / "not-there")
