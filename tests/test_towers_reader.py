"""`litetune towers`: the TFLite reader on the layouts flatc writes (trimmed and
shared vtables, an omitted subgraph index), the reader's own bounds, and the
refusals and messages `test_towers.py` leaves unpinned. Helpers and the fake
builder come from test_towers."""

from __future__ import annotations

import errno
import logging
import os
import struct
from typing import Any

import pytest
from test_towers import (  # noqa: F401 - toolchain is a fixture
    BARE_META,
    BLOCK_SIZE,
    FORMAT,
    GEMMA4_E2B_TOML,
    GEMMA4_META,
    PIECES,
    SM8850_TOML,
    WIDTH,
    builder_with,
    bundle,
    donor,
    embedder,
    flatbuffer,
    gemma4,
    offsets,
    read_bundle,
    sm8850,
    tflite,
    toolchain,
    types,
)

from litetune.cli import main
from litetune.towers import BUILDER_TYPES, TowersError, add_towers, drop_towers

# Every test here runs the rebuild against the fake builder `toolchain` installs.
pytestmark = pytest.mark.usefixtures("toolchain")

# -- the TFLite reader on layouts flatc writes ----------------------------------------


def graph(
    shape: list[int],
    sig: list[Any] | None = None,
    sub_index: int = 0,
    tensor_index: int = 0,
    writer=flatbuffer,
) -> bytes:
    """One subgraph holding one tensor of `shape`, one signature whose only output
    names tensor `tensor_index` of subgraph `sub_index`. `sig` replaces the
    SignatureDef's field list (inputs, outputs, signature_key, deprecated_tag,
    subgraph_index)."""
    tensors = [("table", [("ints", shape)])]
    subgraphs = [("table", [("tables", tensors)])]
    outputs = [("table", [("string", "o"), ("uint", tensor_index)])]
    fields = (
        sig
        if sig is not None
        else [None, ("tables", outputs), ("string", "prefill"), None, ("uint", sub_index)]
    )
    fields = [("tables", outputs) if f == "OUTPUTS" else f for f in fields]
    model = [
        None,
        None,
        ("tables", subgraphs),
        None,
        None,
        None,
        None,
        ("tables", [("table", fields)]),
    ]
    return writer(("table", model))


def add_with_embedder(tmp_path, data: bytes):
    return add_towers(
        sm8850(tmp_path, graphs={"embedder": data}),
        tmp_path / "o",
        ["vision"],
        donor(tmp_path),
        metadata_from_donor=True,
    )


@pytest.mark.parametrize(
    "sig",
    [
        # As flatc writes a subgraph_index of 0: absent, and the vtable trimmed after
        # the last field set.
        [None, "OUTPUTS", ("string", "prefill")],
        # Absent behind a zero vtable entry.
        [None, "OUTPUTS", ("string", "prefill"), None, None],
        # Trimmed to exactly the slot before subgraph_index.
        [None, "OUTPUTS", ("string", "prefill"), ("string", "tag")],
    ],
    ids=["trimmed", "zero-entry", "trimmed-at-the-slot"],
)
def test_a_signature_without_a_subgraph_index_reads_subgraph_0(tmp_path, sig):
    result = add_with_embedder(tmp_path, graph([1, 8, WIDTH], sig=sig))
    assert result.embedding_width == WIDTH


@pytest.mark.parametrize("trimmed", [True, False])
def test_a_graph_without_signature_defs_is_refused_as_having_none(tmp_path, trimmed):
    subgraphs = ("tables", [("table", [("tables", [("table", [("ints", [1, 8, WIDTH])])])])])
    model = [None, None, subgraphs] + ([] if trimmed else [None] * 5)
    with pytest.raises(TowersError, match="embedder graph: the embedder has no signature"):
        add_with_embedder(tmp_path, flatbuffer(("table", model)))


def flatbuffer_vtables_after(root: tuple[str, Any]) -> bytes:
    """`flatbuffer`, with each table's vtable after it: a negative soffset, as a
    vtable flatc shares with an earlier-built table has."""
    out = bytearray(4) + b"TFL3"

    def align() -> None:
        out.extend(bytes(-len(out) % 4))

    def emit(node: tuple[str, Any]) -> int:
        kind, value = node
        align()
        start = len(out)
        if kind == "table":
            out.extend(bytes(4))
            later = []
            for item in value:
                if item is not None and item[0] == "uint":
                    out.extend(struct.pack("<I", item[1]))
                    continue
                if item is not None:
                    later.append((len(out), item))
                out.extend(bytes(4))
            vtable = len(out)
            out.extend(struct.pack("<HH", 4 + 2 * len(value), 4 + 4 * len(value)))
            for i, item in enumerate(value):
                out.extend(struct.pack("<H", 0 if item is None else 4 + 4 * i))
            struct.pack_into("<i", out, start, start - vtable)
            for pos, item in later:
                struct.pack_into("<I", out, pos, emit(item) - pos)
            return start
        if kind == "string":
            data = value.encode()
            out.extend(struct.pack("<I", len(data)) + data + b"\0")
        elif kind == "ints":
            out.extend(struct.pack(f"<I{len(value)}i", len(value), *value))
        else:
            out.extend(struct.pack("<I", len(value)) + bytes(4 * len(value)))
            for i, child in enumerate(value):
                slot = start + 4 + 4 * i
                struct.pack_into("<I", out, slot, emit(child) - slot)
        return start

    struct.pack_into("<I", out, 0, emit(root))
    return bytes(out)


def test_a_vtable_after_its_table_is_read(tmp_path):
    data = graph([1, 8, WIDTH], writer=flatbuffer_vtables_after)
    assert struct.unpack_from("<i", data, struct.unpack_from("<I", data, 0)[0])[0] < 0
    assert add_with_embedder(tmp_path, data).embedding_width == WIDTH


def _u32(buf: bytes, pos: int) -> int:
    return struct.unpack_from("<I", buf, pos)[0]


def _patched(buf: bytes, pos: int, value: int) -> bytes:
    out = bytearray(buf)
    struct.pack_into("<I", out, pos, value)
    return bytes(out)


def test_a_vector_longer_than_the_graph_is_refused(tmp_path):
    data = embedder()
    model = _u32(data, 0)
    vtable = model - struct.unpack_from("<i", data, model)[0]
    field = model + struct.unpack_from("<H", data, vtable + 8)[0]  # Model.subgraphs
    data = _patched(data, field + _u32(data, field), 1000)
    with pytest.raises(TowersError, match="a vector of 1000 elements runs past its end"):
        add_with_embedder(tmp_path, data)


def test_a_signature_key_longer_than_the_graph_is_refused(tmp_path):
    data = embedder(WIDTH, decode=8)
    at = data.index(b"decode_embedder") - 4
    with pytest.raises(TowersError, match="a string runs past its end"):
        add_with_embedder(tmp_path, _patched(data, at, 10_000))


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (graph([1, 8, WIDTH], sub_index=1), "a signature names subgraph 1 of 1"),
        (graph([1, 8, WIDTH], tensor_index=1), "an output names tensor 1 of 1"),
    ],
)
def test_an_index_past_its_list_is_refused_not_crashed(tmp_path, data, reason):
    with pytest.raises(
        TowersError, match=f"could not be established from its embedder graph: {reason}"
    ):
        add_with_embedder(tmp_path, data)


@pytest.mark.parametrize("shape", [[280, WIDTH], [1, 1, 280, WIDTH]])
def test_an_adapters_width_is_its_last_dim_at_any_rank(tmp_path, shape):
    """The runtime takes the vision adapter's tokens from dims[size-2] at any rank
    of 2 or more (vision_executor_utils.cc:54-64)."""
    giver = donor(tmp_path, graphs={"vision_adapter": tflite(("patches", [shape]))})
    result = add_towers(
        sm8850(tmp_path), tmp_path / "o", ["vision"], giver, metadata_from_donor=True
    )
    assert result.embedding_width == WIDTH


def test_an_adapter_output_without_a_shape_is_refused_not_crashed(tmp_path):
    giver = donor(tmp_path, graphs={"audio_adapter": tflite(("audio", [[]]))})
    with pytest.raises(
        TowersError, match="audio_adapter writes could not be established: an output has no shape"
    ):
        add_towers(sm8850(tmp_path), tmp_path / "o", ["audio"], giver, metadata_from_donor=True)


def test_externalized_adapter_weights_are_not_read_as_a_graph(tmp_path):
    weights = (
        '\n[[section]]\nmodel_type = "vision_adapter"\nsection_type = "TFLiteWeights"\n'
        'data_path = "Section12_TFLiteWeights_tf_lite_vision_adapter.bin"\n'
    )
    giver = donor(tmp_path, GEMMA4_E2B_TOML + weights)
    result = add_towers(
        sm8850(tmp_path), tmp_path / "o", ["vision"], giver, metadata_from_donor=True
    )
    assert ("TFLiteWeights", "vision_adapter") in {
        (d["section_type"], d["model_type"]) for d in result.added
    }


def test_externalized_embedder_weights_are_not_a_second_embedder(tmp_path):
    toml = SM8850_TOML + (
        '\n[[section]]\nmodel_type = "embedder"\nsection_type = "TFLiteWeights"\n'
        'data_path = "Section6_TFLiteWeights_tf_lite_embedder.bin"\n'
    )
    model = bundle(tmp_path / "m.litertlm", toml, BARE_META, PIECES[:-1])
    result = add_towers(
        model, tmp_path / "o", ["vision"], donor(tmp_path), metadata_from_donor=True
    )
    assert result.embedding_width == WIDTH


# -- the header ------------------------------------------------------------------------


def test_sections_that_touch_are_not_overlapping(tmp_path):
    """The builder starts the next section where one ends when its length is a
    multiple of the block (litertlm_builder.py:1327-1333)."""
    model = gemma4(tmp_path, graphs={"per_layer_embedder": bytes(BLOCK_SIZE)})
    listed = offsets(model)
    assert any(
        a[1] == b[0] for a, b in zip(listed, listed[1:], strict=False)
    ), "the input has touching sections"
    out = tmp_path / "o.litertlm"
    drop_towers(model, out, ["vision"])
    assert "per_layer_embedder" in types(out)


def _header_end(path) -> int:
    return struct.unpack_from("<Q", path.read_bytes(), 24)[0]


def test_a_section_that_begins_one_byte_inside_the_header_is_refused(tmp_path):
    model = gemma4(tmp_path)
    header, raw = FORMAT["read"](model)
    contents = [raw[b:e] for b, e in header["sections"]]
    listed = [list(s) for s in header["sections"]]
    for _ in range(4):
        end = _header_end(model)
        if listed[0][0] == end - 1:
            break
        listed[0][0] = end - 1
        FORMAT["write"](model, header["toml"], contents, listed)
    else:
        raise AssertionError("the header's length did not settle")
    out = tmp_path / "o.litertlm"
    with pytest.raises(TowersError, match=f"section 0 spans bytes {end - 1} to"):
        drop_towers(model, out, ["vision"])
    assert not out.exists()


def test_a_header_that_lists_sections_out_of_file_order_is_read(tmp_path):
    model = gemma4(tmp_path)
    header, raw = FORMAT["read"](model)
    contents = [raw[b:e] for b, e in header["sections"]]
    order = list(range(len(contents)))
    order[2], order[3] = order[3], order[2]
    laid = [contents[i] for i in order]
    FORMAT["write"](model, header["toml"], laid)
    placed = offsets(model)
    listed: list[Any] = [None] * len(contents)
    for pos, i in enumerate(order):
        listed[i] = placed[pos]
    FORMAT["write"](model, header["toml"], laid, listed)
    _, raw = FORMAT["read"](model)
    assert [raw[b:e] for b, e in listed] == contents and listed[2][0] > listed[3][0]
    drop_towers(model, tmp_path / "o.litertlm", ["vision"])


def test_a_rebuild_written_short_is_named_as_the_rebuilt_bundle(tmp_path, monkeypatch):
    builder_with(
        monkeypatch,
        "    write(output_path, _toml(doc), contents)\n",
        "    Path(output_path).write_bytes(Path(output_path).read_bytes()[:-1])\n",
    )
    with pytest.raises(
        TowersError, match=r"the rebuilt bundle \(rebuilt.litertlm\): section \d+ spans"
    ):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])


def test_a_donor_section_unpack_leaves_out_names_the_donor(tmp_path):
    with pytest.raises(
        TowersError, match="the donor has 13 sections and litert-lm-builder unpacks 12"
    ):
        add_towers(
            sm8850(tmp_path),
            tmp_path / "o",
            ["vision"],
            donor(tmp_path, hidden=1),
            metadata_from_donor=True,
        )


# -- metadata --------------------------------------------------------------------------


def test_adding_vision_needs_no_audio_start_token(tmp_path):
    meta = GEMMA4_META.replace(
        '    start_of_audio_token {\n      token_str: "<|audio>"\n    }\n', ""
    )
    assert "start_of_audio_token" not in meta
    result = add_towers(
        sm8850(tmp_path, meta=meta), tmp_path / "o", ["vision"], donor(tmp_path, meta=meta)
    )
    assert result.metadata_source == "bundle"


def test_an_empty_end_token_message_is_refused(tmp_path):
    meta = GEMMA4_META.replace(
        "    patch_width", "    end_of_image_token {\n    }\n    patch_width"
    )
    with pytest.raises(TowersError, match="gives end_of_image_token without a token_str"):
        add_towers(
            sm8850(tmp_path, meta=meta), tmp_path / "o", ["vision"], donor(tmp_path, meta=meta)
        )


def test_a_negative_token_id_is_no_piece(tmp_path):
    meta = GEMMA4_META + "suppress_tokens {\n  ids: -1\n}\n"
    with pytest.raises(TowersError, match="id -1 is no piece in the bundle's tokenizer"):
        add_towers(
            sm8850(tmp_path, meta=meta, pieces=PIECES),
            tmp_path / "o",
            ["vision"],
            donor(tmp_path, meta=meta),
        )


def test_each_added_towers_settings_are_compared(tmp_path):
    own = GEMMA4_META.replace(
        "    max_num_patches", "    skip_mel_spectrogram_extraction: true\n    max_num_patches"
    )
    with pytest.raises(TowersError, match="skip_mel_spectrogram_extraction"):
        add_towers(sm8850(tmp_path, meta=own), tmp_path / "o", ["vision", "audio"], donor(tmp_path))


# -- the builder and the host ---------------------------------------------------------


def test_builder_types_are_0_18_0s():
    """`TfLiteModelType` in litert-lm-builder 0.18.0, litertlm_builder.py:211-236,
    without its `tf_lite_` prefix. The fake builder takes its list from
    BUILDER_TYPES, so only this pins what the real one is compared with."""
    assert BUILDER_TYPES == (
        "prefill_decode",
        "embedder",
        "per_layer_embedder",
        "aux",
        "audio_frontend",
        "audio_encoder_hw",
        "audio_adapter",
        "end_of_audio",
        "vision_encoder",
        "vision_adapter",
        "end_of_vision",
        "artisan_text_decoder",
        "mtp_drafter",
        "mtp_aux",
        "text_encoder",
    )


@pytest.mark.parametrize(
    "code", [errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK]
)
def test_each_no_link_errno_gets_the_filesystem_advice(tmp_path, monkeypatch, code):
    def no_link(src, dst):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr("litetune.towers.os.link", no_link)
    with pytest.raises(TowersError, match="hard links"):
        drop_towers(gemma4(tmp_path), tmp_path / "o", ["vision"])


def test_a_delivered_report_logs_no_delivery_error(tmp_path, caplog, capsys):
    argv = [
        "towers",
        "--model",
        str(gemma4(tmp_path)),
        "--drop",
        "vision",
        "--output",
        str(tmp_path / "o"),
        "--json",
    ]
    with caplog.at_level(logging.ERROR, logger="litetune.cli"):
        assert main(argv) == 0
    assert "could not be delivered" not in caplog.text
