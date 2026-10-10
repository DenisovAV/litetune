"""`litetune towers` on an untrusted header: what reaches litert-lm-builder's file
names and TOML is checked before `unpack` runs, and what the readers will parse
is bounded. Each is shown with test_towers' own fake builder."""

from __future__ import annotations

import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_towers import (  # noqa: F401 - toolchain is a fixture
    FAKE_BUILDER,
    FAKE_FORMAT,
    FAKE_PEEK,
    GEMMA4_E2B_TOML,
    builder_with,
    flatbuffer,
    gemma4,
    toolchain,
)

from litetune.towers import _TOWERS_SCRIPT, BUILDER_TYPES, TowersError, drop_towers

pytestmark = pytest.mark.usefixtures("toolchain")


@pytest.fixture
def script(tmp_path, monkeypatch) -> Iterator[dict[str, Any]]:
    """The towers script's functions, imported against the fake builder."""
    shim = tmp_path / "shim"
    package = shim / "litert_lm_builder"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    settings = (
        f"KNOWN_TYPES = {list(BUILDER_TYPES)!r}\nPACK_FAILS = False\nPACK_ADDS_KEY = False\n"
        "PACK_ALTERS_BYTES = False\nSHRINK_INPUT = None\n"
    )
    (package / "litertlm_builder.py").write_text(settings + FAKE_BUILDER, encoding="utf-8")
    (package / "litertlm_peek.py").write_text(FAKE_PEEK, encoding="utf-8")
    (package / "litertlm_core.py").write_text(FAKE_FORMAT, encoding="utf-8")
    monkeypatch.syspath_prepend(str(shim))
    for name in [m for m in sys.modules if m.startswith("litert_lm_builder")]:
        monkeypatch.delitem(sys.modules, name)
    ns: dict[str, Any] = {"__name__": "towers_script"}
    exec(_TOWERS_SCRIPT, ns)
    yield ns
    for name in [m for m in sys.modules if m.startswith("litert_lm_builder")]:
        del sys.modules[name]


# -- 1. a section key from the unpacked TOML is written back bare -------------------


def test_a_section_key_cannot_add_a_section_to_the_toml_pack_reads(script, tmp_path):
    key = 'x = 1\n[[section]]\nsection_type = "GenericBinaryData"\ndata_path = "/etc/hosts"\nz'
    doc = {
        "system_metadata": {"entries": []},
        "section": [{"section_type": "SP_Tokenizer", "data_path": "a.spiece", key: 1}],
    }
    # `write_toml` would write the key bare, a second section with an absolute
    # data_path after it; `check_keys` is what runs before it.
    with pytest.raises(script["Refused"]):
        script["check_keys"](doc, "the bundle")


def test_a_section_key_from_the_input_never_reaches_pack(tmp_path):
    outside = tmp_path / "outside" / "not-in-the-work-directory.bin"
    key = (
        '"x = 1\\n[[section]]\\nsection_type = \\"GenericBinaryData\\"\\n'
        f'data_path = \\"{outside}\\"\\nz" = 1\n'
    )
    anchor = 'data_path = "Section1_SP_Tokenizer.spiece"\n'
    toml = GEMMA4_E2B_TOML.replace(anchor, anchor + key)
    with pytest.raises(TowersError) as refused:
        drop_towers(gemma4(tmp_path, toml), tmp_path / "o", ["vision"])
    assert "FileNotFoundError" not in str(refused.value), refused.value


# -- 2. the header's items are not checked before `unpack` writes files by them ------


def test_a_model_type_with_a_path_in_it_is_refused_before_unpack(tmp_path, monkeypatch):
    calls = tmp_path / "unpack-calls.txt"
    log = f"    open({str(calls)!r}, 'a').write(str(litertlm_path) + '\\n')\n"
    builder_with(monkeypatch, "    header, raw = read(litertlm_path)\n", log)
    toml = GEMMA4_E2B_TOML.replace(
        'model_type = "mtp_drafter"', 'model_type = "x/../../../../escaped"'
    )
    with pytest.raises(TowersError, match="x/../../../../escaped"):
        drop_towers(gemma4(tmp_path, toml), tmp_path / "o", ["vision"])
    assert not calls.exists(), "unpack ran on: " + calls.read_text()


# -- 3. no bound on the number of sections a header may list ------------------------


def test_a_header_listing_200k_empty_sections_is_refused(script, tmp_path, monkeypatch):
    path = tmp_path / "b.litertlm"
    raw = bytearray(b"LITERTLM" + bytes(16) + struct.pack("<Q", 64) + bytes(32) + b"data")
    path.write_bytes(bytes(raw))
    size = len(raw)

    class Obj:
        def BeginOffset(self):
            return size

        def EndOffset(self):
            return size

    class Listed:
        def ObjectsLength(self):
            return 200_000

        def Objects(self, i):
            return Obj()

    class Header:
        def SectionMetadata(self):
            return Listed()

    monkeypatch.setattr(script["litertlm_peek"], "read_litertlm_header", lambda p, s: Header())
    with pytest.raises(script["Refused"]):
        script["spans"](path, "the bundle")


# -- 4. bounded parsing ---------------------------------------------------------------


def test_a_varint_longer_than_ten_bytes_is_refused(script):
    with pytest.raises(ValueError):
        script["varint"](b"\xff" * 11 + b"\x01", 0)


def shared_graph(path: Path, S: int, N: int) -> Path:
    """S signatures that share one SignatureDef, whose subgraph has N tensors that
    share one Tensor of shape [1, 8, 16]: 4 * (S + N) bytes and a little more."""
    out = bytearray(4) + b"TFL3"

    def align():
        out.extend(bytes(-len(out) % 4))

    def table(fields):
        align()
        start = len(out)
        out.extend(struct.pack("<HH", 4 + 2 * len(fields), 4 + 4 * len(fields)))
        for i, f in enumerate(fields):
            out.extend(struct.pack("<H", 0 if f is None else 4 + 4 * i))
        align()
        t = len(out)
        out.extend(struct.pack("<i", t - start))
        slots = []
        for _ in fields:
            slots.append(len(out))
            out.extend(bytes(4))
        return t, slots

    def vector(n):
        align()
        start = len(out)
        out.extend(struct.pack("<I", n) + bytes(4 * n))
        return start

    def point(slot, target):
        struct.pack_into("<I", out, slot, target - slot)

    model, mslots = table([None, None, 1, None, None, None, None, 1])
    subs = vector(1)
    point(mslots[2], subs)
    sub, sslots = table([1])
    point(subs + 4, sub)
    tensors = vector(N)
    point(sslots[0], tensors)
    tensor, tslots = table([1])
    shape = vector(3)
    struct.pack_into("<3i", out, shape + 4, 1, 8, 16)
    point(tslots[0], shape)
    for i in range(N):
        point(tensors + 4 + 4 * i, tensor)
    sigs = vector(S)
    point(mslots[7], sigs)
    sig, gslots = table([None, 1, None, None, 0])
    for i in range(S):
        point(sigs + 4 + 4 * i, sig)
    outs = vector(1)
    point(gslots[1], outs)
    tm, _ = table([None, 0])
    point(outs + 4, tm)
    struct.pack_into("<I", out, 0, model)
    path.write_bytes(bytes(out))
    return path


def test_the_tflite_reader_reads_each_subgraphs_tensors_once(script, tmp_path):
    """A reader linear in the file reads O(S + N) words; this one reads S * N."""
    S, N = 50, 2000
    path = shared_graph(tmp_path / "g.tflite", S, N)
    reads = [0]
    original = script["Flat"].at

    def counted(self, fmt, pos):
        reads[0] += 1
        return original(self, fmt, pos)

    script["Flat"].at = counted
    try:
        script["output_shapes"](path)
    except ValueError:
        pass  # refused, within the budget
    finally:
        script["Flat"].at = original
    assert reads[0] < 20 * (S + N), f"{reads[0]} reads for S={S}, N={N}"


def test_a_shape_with_more_dims_than_a_tensor_has_is_refused(script, tmp_path):
    from test_towers import tflite

    path = tmp_path / "e.tflite"
    path.write_bytes(tflite(("prefill", [[1, 8] + [2] * 1000])))
    with pytest.raises(ValueError):
        script["text_width"](path)


# -- 5. a stray key is put in the refusal as it is, control characters included --------


def test_a_stray_key_reaches_the_terminal_escaped(tmp_path):
    toml = '"\\u001b]0;owned\\u0007\\u001b[2J" = 1\n' + GEMMA4_E2B_TOML
    with pytest.raises(TowersError) as refused:
        drop_towers(gemma4(tmp_path, toml), tmp_path / "o", ["vision"])
    assert "\x1b" not in str(refused.value), repr(str(refused.value))


def test_a_tokenizer_past_the_size_cap_is_refused_unread(script, tmp_path):
    path = tmp_path / "big.spiece"
    path.write_bytes(b"\x0a\x03\x0a\x01a")
    script["pieces"].__globals__["MAX_TOKENIZER_BYTES"] = 4
    with pytest.raises(script["Refused"], match="more than 4 a tokenizer is read to"):
        script["pieces"](path)


def test_a_tokenizer_with_more_pieces_than_the_cap_is_refused(script, tmp_path):
    path = tmp_path / "many.spiece"
    path.write_bytes(b"\x0a\x00" * 3)
    script["pieces"].__globals__["MAX_PIECES"] = 2
    with pytest.raises(script["Refused"], match="holds more than 2 pieces"):
        script["pieces"](path)


def test_a_piece_with_more_fields_than_a_piece_has_is_refused(script, tmp_path):
    """One piece of many empty fields would pass the piece count and be held whole."""
    inner = b"\x22\x00" * 17
    path = tmp_path / "wide.spiece"
    path.write_bytes(b"\x0a" + bytes([len(inner)]) + inner)
    with pytest.raises(script["Refused"], match="a piece of more than 16 fields"):
        script["pieces"](path)
