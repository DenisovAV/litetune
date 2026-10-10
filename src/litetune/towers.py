"""`towers`: a bundle without the vision or audio sections it carries, or with
another bundle's.

Google's Gemma 4 E2B CPU/GPU bundle (`litert-community/gemma-4-E2B-it-litert-lm`,
`gemma-4-E2B-it.litertlm` at `b3ca0d2f`) carries its vision and audio towers as six
sections of their own beside the text model: 332,387,932 of its 2,588,147,712
bytes. Stripped of them, it answered eight greedy prompts with the same text on
litert-lm 0.18.0's CPU backend, at a peak RSS within 3 MB of the full bundle's
(MEASUREMENTS.md, *Gemma 4 E2B without its towers*). An application that only
sends text ships the difference for nothing.

**Sections are chosen by type, from an allowlist, and nothing else.** A section
belongs to a tower when its `section_type` is `TFLiteModel` or `TFLiteWeights` (a
graph and its externalized weights) *and* its `model_type` is one this module
names for that tower -- values from litert-lm-builder 0.18.0's `TfLiteModelType`
(`litertlm_builder.py:211-232`); the script refuses a builder whose list differs
from that one in either direction. A `section_type` alone would match the text
model too. Any other section is kept.

**The header is checked before anything reads by it.** `litertlm_peek` and
`unpack` read each length the header gives in one call and never compare it with
the file (`litertlm_peek.py:112-119, 300-305`): a file cut short would be copied
short, and a header that claims more than the file holds would be read in one
call of that size. The script reads the header's end and every section's offsets
itself first, and refuses a file whose header or sections do not lie inside it,
or whose sections overlap.

**What the builder cannot write back is refused, not lost.** `unpack` leaves out
of its TOML the sections whose type `pack` does not take -- TTS and ASR metadata,
`NONE`, `Deprecated` (`litertlm_peek.py:631-688`) -- and `pack` raises for a
`model_type` outside its enum; the header's section count, every type and every
TOML key are checked first. `unpack` also writes LlmMetadata as text through the
builder's own proto, which loses a field that proto does not declare -- 2 bytes
of Google's E2B metadata, `gemma4` field 13, reserved in 0.18.0 -- so the metadata
goes back in as its raw bytes, which `pack` copies unparsed.

**The result is checked by reading it back.** Every section of the new file must
hash the same, as raw bytes at its header offsets, as the section it was taken
from, and the TOML `unpack` writes for it must equal what was asked -- the system
`uuid` and `creation_timestamp` aside, which `pack` regenerates. It is put at the
output path only then, and never over a file already there.

**Adding is the other direction**, and the less certain one: LiteRT-LM does not
offer it as an operation. Checked: the donor carries the whole tower, one graph
each for its encoder, adapter and end marker; each added adapter writes as many
values per token as the bundle's embedder, read from the graphs' signatures --
LiteRT-LM copies the one into the other without comparing them
(`embedding_lookup_multi_modal.cc:120-123, 148-163`); the LlmMetadata the result
carries is Gemma 4's, names as a string the token each added tower's input starts
with, and gives every Gemma 4 media token as a string, the only form the runtime
reads (`model_data_processor_factory.cc:57-65, 191-206`); every token string and
id it names is the same piece, of the same type, at the same id in both
tokenizers, read from their SentencePiece protobufs; and the image and audio
settings the added towers were built for are the donor's. A text-only build's
metadata names no media token, so the donor's is taken only when the caller asks
(`metadata_from_donor`), whole, and refused where it would change what LiteRT-LM's
NPU executor reads or the settings of a tower the bundle keeps; the report lists
each field that changes as the builder's text form shows it, which leaves out a
field the builder's proto does not declare (Google's E2B metadata carries one,
`gemma4` field 13). Not checked: other strings in the metadata, the template
included.

**It runs in the runtime environment**, not the export one: `litert-lm==0.18.0`
requires `litert-lm-builder==0.18.0`, so `envs.RUNTIME` already carries the
builder, and the export environment would add a converter this never calls. The
scratch directory beside the output holds, at its fullest, the unpacked bundle
and the result for a drop -- about the bundle's size plus the result's -- and the
unpacked donor besides for an add.

What the model answers is not checked here; `litetune verify` does that.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from litetune import envs
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

TOWERS_SCHEMA = "litetune.towers/1"

# `TfLiteModelType` in litert-lm-builder 0.18.0 (`litertlm_builder.py:211-232`),
# without its `tf_lite_` prefix, which `unpack` leaves off and older bundles keep.
# The script refuses a builder whose list is not exactly this one.
BUILDER_TYPES: tuple[str, ...] = (
    "prefill_decode", "embedder", "per_layer_embedder", "aux",
    "audio_frontend", "audio_encoder_hw", "audio_adapter", "end_of_audio",
    "vision_encoder", "vision_adapter", "end_of_vision",
    "artisan_text_decoder", "mtp_drafter", "mtp_aux", "text_encoder",
)  # fmt: skip

TOWER_SECTIONS: dict[str, tuple[str, ...]] = {
    "vision": ("vision_encoder", "vision_adapter", "end_of_vision"),
    "audio": ("audio_frontend", "audio_encoder_hw", "audio_adapter", "end_of_audio"),
}

# A donor's tower is taken when it carries every other graph of it. Google's
# Gemma 4 E2B bundle carries no `audio_frontend` (its unpacked TOML in
# tests/test_towers.py); one is taken when there is one.
TOWER_OPTIONAL = frozenset({"audio_frontend"})
TOWER_REQUIRED: dict[str, tuple[str, ...]] = {
    tower: tuple(t for t in types if t not in TOWER_OPTIONAL)
    for tower, types in TOWER_SECTIONS.items()
}

# A rebuild unpacks, packs and unpacks again, once more for a donor.
TOWERS_TIMEOUT_S = 1800

# link(2): the filesystem cannot hard-link (EPERM on Linux, ENOTSUP or EOPNOTSUPP
# elsewhere), the two names are on different ones (EXDEV), or the file has as
# many links as it may (EMLINK). Only for these is another output path the advice.
NO_LINK_ERRNOS = frozenset(
    {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}
)


class TowersError(Exception):
    """A request `towers` will not carry out, or a rebuild it would not keep."""


_TOWERS_SCRIPT = r'''
"""Drop named tower sections from a bundle, or add them from another; leave the
result, read back and verified, in the work directory for the parent to publish.

Reads JSON {mode: "drop"|"add", artifact, work, names: [name, ...], towers: {name:
[model_type, ...]} for every tower, required: {name: [model_type, ...]}, builder_types:
[model_type, ...], donor, metadata_from_donor} from argv[1]. Writes JSON to argv[2]:
  {"builder": <version>, "sections": [...], "dropped": [...], "added": [...],
   "metadata_source": ..., "metadata_changes": [...], "metadata_raw_differs": ...,
   "tokens_checked": <int>, "embedding_width": <int>, "rebuilt": <path|null>,
   "reason": <str|null>}
"""
import copy
import hashlib
import importlib.metadata
import io
import json
import math
import mmap
import os
import re
import shutil
import struct
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: the builder itself depends on tomli
    import tomli as tomllib

from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder import litertlm_core
from litert_lm_builder import litertlm_peek

VOLATILE = {"uuid", "creation_timestamp"}
# Sections a tower's graphs and their externalized weights live in.
GRAPH_TYPES = ("TFLiteModel", "TFLiteWeights")
# The graph whose output LiteRT-LM copies into the text model's embeddings.
ADAPTER = {"vision": "vision_adapter", "audio": "audio_adapter"}
# The `Gemma4` TokenUnion naming the token a tower's input starts with, and all
# four the runtime reads, each through `GetTokenString`, which fails for one that
# is set without a `token_str` (model_data_processor_factory.cc:57-65, 191-206).
TOWER_TOKEN = {"vision": "start_of_image_token", "audio": "start_of_audio_token"}
GEMMA4_TOKENS = ("start_of_image_token", "end_of_image_token", "start_of_audio_token",
                 "end_of_audio_token")
# The `Gemma4` fields a tower's input is prepared by (llm_model_type.proto:249-305).
# Each is compared as the runtime reads it: a field left at the proto's own default
# (0, false) keeps the processor's built-in value, any other value replaces it
# (model_data_processor_factory.cc:246-262; gemma4_data_processor_config.h:39-62).
MEDIA_DEFAULTS = {"patch_width": "16", "patch_height": "16", "max_num_patches": "2520",
                  "pooling_kernel_size": "3", "merge_patches": "false",
                  "skip_mel_spectrogram_extraction": "false"}
MEDIA_FIELDS = {
    "vision": ("patch_width", "patch_height", "max_num_patches", "pooling_kernel_size",
               "merge_patches"),
    "audio": ("skip_mel_spectrogram_extraction",),
}
# LlmMetadata fields LiteRT-LM's NPU executor reads, and what it takes when one is
# unset: max_num_tokens as a static model's sequence length when above 0
# (llm_litert_npu_compiled_model_executor.cc:2350-2357), kv_cache_init_value as the
# KV cache's fill value, 0 when unset (npu/llm_litert_npu_kv_cache.cc:88-96).
NPU_FIELDS = {"max_num_tokens": "0", "kv_cache_init_value": "0"}
# SentencePiece's `ModelProto.SentencePiece.Type`; `type` is the piece's field 3,
# NORMAL when absent (sentencepiece_model.proto).
PIECE_TYPES = {1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL", 4: "USER_DEFINED", 5: "UNUSED",
               6: "BYTE"}
# A metadata value longer than this is reported by its length.
SHOWN_CHARS = 60


class Refused(Exception):
    """A reason this script will not write the bundle asked for."""


def load(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def model_type(section):
    return str(section.get("model_type", "")).lower().removeprefix("tf_lite_")


def toml_string(text):
    """A TOML basic string. Not json.dumps: that writes a character outside the
    BMP as a UTF-16 surrogate pair (`\\ud83d\\ude80`), which TOML forbids and
    tomllib rejects, so one emoji in any metadata value would sink the pack."""
    out = []
    for ch in text:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append(f"\\u{o:04X}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return toml_string(str(v))


def entry_line(e):
    """One element of an `entries` / `additional_metadata` array. Both arrays
    carry the same {key, value_type, value} shape, so both are written here."""
    return (
        f"  {{ key = {toml_value(e['key'])}, "
        f"value_type = {toml_value(e['value_type'])}, "
        f"value = {toml_value(e['value'])} }},"
    )


def write_toml(doc, path):
    """Emit TOML for the builder to read back. Only the shapes `unpack` writes."""
    lines = ["[system_metadata]", "entries = ["]
    lines += [entry_line(e) for e in doc.get("system_metadata", {}).get("entries", [])]
    lines += ["]", ""]
    for sec in doc.get("section", []):
        lines.append("[[section]]")
        for k, v in sec.items():
            if k == "additional_metadata":
                continue
            lines.append(f"{k} = {toml_value(v)}")
        meta = sec.get("additional_metadata") or []
        if meta:
            lines.append("additional_metadata = [")
            lines += [entry_line(e) for e in meta]
            lines.append("]")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def check_keys(doc, where):
    """Refuse a TOML key `write_toml` would not write back."""
    stray = sorted(set(doc) - {"system_metadata", "section"})
    stray += sorted(f"system_metadata.{k}"
                    for k in set(doc.get("system_metadata", {})) - {"entries"})
    if stray:
        raise Refused(f"litert-lm-builder's TOML for {where} has {', '.join(stray)}, which "
                      "this script does not write back")


def comparable(doc):
    """The TOML without what `pack` regenerates and without file names, which say
    nothing about content; the content is compared as raw bytes (`digest`)."""
    sm = doc.get("system_metadata", {})
    entries = [e for e in sm.get("entries", []) if e.get("key") not in VOLATILE]
    sections = copy.deepcopy(doc.get("section", []))
    for s in sections:
        s.pop("data_path", None)
    return {"system_metadata": {"entries": entries}, "section": sections}


def spans(path, where):
    """Each section's (begin, end) byte offsets, from the file's own header --
    refused unless the header and every section lie inside the file and no two
    sections overlap. The header's end is 8 bytes at
    HEADER_END_LOCATION_BYTE_OFFSET and the header runs from
    HEADER_BEGIN_BYTE_OFFSET to it (`litertlm_peek.py:112-119`)."""
    size = os.path.getsize(path)
    where = f"{where} ({Path(path).name})"
    with open(path, "rb") as f:
        f.seek(litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
        raw = f.read(8)
    if len(raw) != 8:
        raise Refused(f"{where} is {size} bytes, too short for a LiteRT-LM header")
    (header_end,) = struct.unpack("<Q", raw)
    if not litertlm_core.HEADER_BEGIN_BYTE_OFFSET <= header_end <= size:
        raise Refused(f"{where}: its header ends at byte {header_end}, outside its {size} bytes")
    try:
        listed = litertlm_peek.read_litertlm_header(str(path), io.StringIO()).SectionMetadata()
        count = listed.ObjectsLength() if listed else 0
        places = [(listed.Objects(i).BeginOffset(), listed.Objects(i).EndOffset())
                  for i in range(count)]
    except (ValueError, IndexError, struct.error) as exc:
        raise Refused(f"{where}: its header could not be read: {exc}") from None
    for i, (begin, end) in enumerate(places):
        if begin > end:
            raise Refused(f"{where}: section {i} begins at byte {begin}, after it ends at {end}")
        if begin < header_end or end > size:
            raise Refused(f"{where}: section {i} spans bytes {begin} to {end}, outside the "
                          f"{header_end} to {size} after its header")
    order = sorted(range(len(places)), key=lambda i: places[i])
    for i, j in zip(order, order[1:]):
        if places[j][0] < places[i][1]:
            raise Refused(f"{where}: sections {i} and {j} overlap")
    return places


def read_span(path, span):
    begin, end = span
    with open(path, "rb") as f:
        f.seek(begin)
        data = f.read(end - begin)
    if len(data) != end - begin:
        raise Refused(f"{path} ended before byte {end}, which its header names")
    return data


def digest(path, span):
    begin, end = span
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(begin)
        left = end - begin
        while left:
            block = f.read(min(left, 1 << 24))
            if not block:
                raise Refused(f"{path} ended before byte {end}, which its header names")
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def describe(section, span):
    return {
        "section_type": section.get("section_type"),
        "model_type": model_type(section) or None,
        "bytes": span[1] - span[0],
    }


def label(section):
    return " ".join(filter(None, (section.get("section_type"), model_type(section))))


RAW_META = "LlmMetadataProto.pb"


def unpacked(path, directory, where, places):
    """The bundle's TOML and each section's place in the file, ready for `pack`
    -- refused if the builder could not write it back whole.

    `unpack` dumps every section but puts in its TOML only the types `pack`
    takes, and `pack` raises for a `model_type` outside its enum. And `unpack`
    writes LlmMetadata as text through the builder's own proto, so a field that
    proto does not declare is gone from the text: the section's raw bytes are
    written beside it as a binary `.pb`, which `pack` copies unparsed.
    """
    lb.unpack(str(path), str(directory))
    doc = load(Path(directory) / "model.toml")
    check_keys(doc, where)
    listed = doc.setdefault("section", [])
    if len(places) != len(listed):
        raise Refused(
            f"{where} has {len(places)} sections and litert-lm-builder unpacks "
            f"{len(listed)} of them into a form it can write back; the rest would be lost "
            "(sections `unpack` leaves out: TTS or ASR metadata, NONE, Deprecated)"
        )
    for s in listed:
        if s.get("section_type") in GRAPH_TYPES:
            try:
                lb.TfLiteModelType.get_enum_from_tf_free_value(str(s.get("model_type", "")))
            except ValueError:
                raise Refused(
                    f"{where} has a {s['section_type']} section of model_type "
                    f"{s.get('model_type')!r}, which litert-lm-builder cannot write back"
                ) from None
    for s, place in zip(listed, places, strict=True):
        if s.get("section_type") == "LlmMetadata":
            raw = Path(directory) / RAW_META
            raw.write_bytes(read_span(path, place))
            s["data_path"] = str(raw)
        else:
            s["data_path"] = str(Path(directory) / os.path.basename(s["data_path"]))
    return doc, [(str(path), place) for place in places]


# -- protobuf, read by hand: the runtime environment has no sentencepiece --------

def varint(buf, i):
    shift = value = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def fields(buf):
    """(field number, wire type, value) for each field of one protobuf message."""
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = varint(buf, i)
        elif wire == 1:
            if i + 8 > len(buf):
                raise ValueError("a 64-bit field runs past the end of the message")
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            n, i = varint(buf, i)
            if i + n > len(buf):
                raise ValueError("a length-delimited field runs past the end of the message")
            value, i = buf[i:i + n], i + n
        elif wire == 5:
            if i + 4 > len(buf):
                raise ValueError("a 32-bit field runs past the end of the message")
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"protobuf wire type {wire} is not one a tokenizer uses")
        yield number, wire, value


def pieces(path):
    """A SentencePiece model's (piece, type) by id, the piece as bytes:
    `ModelProto` field 1, each piece's field 1 and field 3."""
    out = []
    try:
        for number, wire, value in fields(Path(path).read_bytes()):
            if number == 1 and wire == 2:
                piece = list(fields(value))  # all of it, so a truncated piece is refused
                text = next((v for n, w, v in piece if n == 1 and w == 2), None)
                kind = next((v for n, w, v in piece if n == 3 and w == 0), 1)
                out.append((text, kind))
    except (IndexError, ValueError) as exc:
        raise Refused(f"{Path(path).name} is not a SentencePiece model: {exc}") from None
    if not out:
        raise Refused(f"{Path(path).name} holds no SentencePiece pieces")
    return out


def shown_piece(entry):
    if entry is None:
        return "no piece"
    text, kind = entry
    shown = repr((text or b"").decode("utf-8", "replace"))
    return f"{shown} ({PIECE_TYPES.get(kind, kind)})"


ESCAPES = {"n": b"\n", "r": b"\r", "t": b"\t", "a": b"\a", "b": b"\b", "f": b"\f",
           "v": b"\v", "\\": b"\\", "'": b"'", '"': b'"', "?": b"?"}


def unescape(body):
    """A protobuf text-format string body, C-escaped as `MessageToString` writes
    it (octal bytes for non-ASCII), back to text."""
    out, i = bytearray(), 0
    while i < len(body):
        ch = body[i]
        if ch != "\\":
            out += ch.encode("utf-8")
            i += 1
            continue
        nxt = body[i + 1]
        if nxt in ESCAPES:
            out += ESCAPES[nxt]
            i += 2
        elif nxt in "01234567":
            m = re.match(r"[0-7]{1,3}", body[i + 1:])
            out.append(int(m.group(0), 8) & 0xFF)
            i += 1 + len(m.group(0))
        elif nxt in "xX":
            m = re.match(r"[0-9a-fA-F]{1,2}", body[i + 2:])
            out.append(int(m.group(0), 16))
            i += 2 + len(m.group(0))
        else:
            raise ValueError(f"unknown escape \\{nxt}")
    return out.decode("utf-8", "replace")


class Quoted(str):
    """A string field's value, as against a number or an enum name."""


TOKEN = re.compile(
    r'\s+|#[^\n]*|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\''
    r"|[{}:<>\[\],;]|[^\s{}:<>\[\],;\"']+"
)


def pbtext(text):
    """Protobuf text format to a tree: a message is a list of (name, value), a
    value a message or a string. Enough of the grammar for LlmMetadata."""
    tokens = [t for t in TOKEN.findall(text) if t.strip() and not t.startswith("#")]
    pos = 0

    def message(end):
        nonlocal pos
        out = []
        while pos < len(tokens) and tokens[pos] != end:
            name = tokens[pos]
            pos += 1
            if pos < len(tokens) and tokens[pos] == ":":
                pos += 1
            if tokens[pos] in "{<":
                close = "}" if tokens[pos] == "{" else ">"
                pos += 1
                out.append((name, message(close)))
                pos += 1
            elif tokens[pos][0] in "\"'":
                parts = []
                while pos < len(tokens) and tokens[pos][0] in "\"'":
                    parts.append(unescape(tokens[pos][1:-1]))
                    pos += 1
                out.append((name, Quoted("".join(parts))))
            else:
                out.append((name, tokens[pos]))
                pos += 1
            if pos < len(tokens) and tokens[pos] in ",;":
                pos += 1
        return out

    tree = message(None)
    if pos != len(tokens):
        raise ValueError("unbalanced braces in LlmMetadata text")
    return tree


def metadata_tree(directory, where):
    text = (Path(directory) / "LlmMetadataProto.pbtext").read_text(encoding="utf-8")
    try:
        return pbtext(text)
    except IndexError:
        reason = "it ends inside a field"
    except ValueError as exc:
        reason = str(exc)
    raise Refused(f"{where}'s LlmMetadata, as `unpack` writes it, could not be read: {reason}")


def flattened(tree, prefix=""):
    """Each scalar field's dotted path and its values, in order; a repeated or
    nested message contributes one path per field inside it."""
    out = {}
    for name, value in tree:
        path = prefix + name
        if isinstance(value, list):
            for inner, values in flattened(value, path + ".").items():
                out.setdefault(inner, []).extend(values)
        else:
            out.setdefault(path, []).append(value)
    return out


def shown_values(values):
    if values is None:
        return "unset"
    shown = []
    for v in values:
        text = json.dumps(v, ensure_ascii=False) if isinstance(v, Quoted) else v
        shown.append(text if len(text) <= SHOWN_CHARS else f"<{len(v)} characters>")
    return shown[0] if len(shown) == 1 else "[" + ", ".join(shown) + "]"


def changed_fields(old, new):
    """`path: old → new` for each field whose values differ between the two."""
    a, b = flattened(old), flattened(new)
    return [f"{path}: {shown_values(a.get(path))} → {shown_values(b.get(path))}"
            for path in sorted(a.keys() | b.keys()) if a.get(path) != b.get(path)]


def named_tokens(tree):
    """Every token string and token id the metadata names: each TokenUnion's
    `token_str`, and every `ids` of a TokenIds -- `suppress_tokens`, a TokenIds of
    its own (llm_metadata.proto:130), included."""
    strings, ids = set(), set()
    for name, value in tree:
        if isinstance(value, list):
            more_strings, more_ids = named_tokens(value)
            strings |= more_strings
            ids |= more_ids
        elif name == "token_str" and value:
            strings.add(str(value))
        elif name == "ids":
            try:
                ids.add(int(value))
            except ValueError:
                raise Refused(f"the LlmMetadata names a token id {value!r}") from None
    return strings, ids


def model_settings(tree):
    """The `llm_model_type` case the metadata sets, and that case's fields."""
    for name, value in reversed(tree):
        if name == "llm_model_type" and isinstance(value, list) and value:
            case, settings = value[0]
            return case, settings if isinstance(settings, list) else []
    return None, []


def gemma4_field(tree, field):
    """The fields of `Gemma4`'s message `field`, or None when it is not set."""
    case, settings = model_settings(tree)
    found = [v for n, v in settings if n == field and isinstance(v, list)]
    return found[-1] if case == "gemma4" and found else None


def media(tree, towers):
    """The model type and its media settings for `towers`, as set."""
    case, settings = model_settings(tree)
    wanted = {f for t in towers for f in MEDIA_FIELDS[t]}
    found = {n: str(v) for n, v in settings if n in wanted}
    return case, {
        f: MEDIA_DEFAULTS[f] if found.get(f, "0") in ("0", "false") else found[f]
        for f in sorted(wanted)
    }


def shown_media(found):
    case, settings = found
    listed = ", ".join(f"{k}: {v}" for k, v in sorted(settings.items()))
    return f"{case or 'no llm_model_type'} with {listed or 'none of them'}"


def report(sections=(), dropped=(), rebuilt=None, reason=None, **extra):
    return {"sections": list(sections), "dropped": list(dropped), "rebuilt": rebuilt,
            "reason": reason, **extra}


def is_tower(section, names, towers):
    if section.get("section_type") not in GRAPH_TYPES:
        return None
    return next((t for t in names if model_type(section) in towers[t]), None)


def only(doc, section_type, where):
    found = [s for s in doc["section"] if s.get("section_type") == section_type]
    if len(found) != 1:
        raise Refused(f"{where} has {len(found)} {section_type} sections, not one")
    return found[0]


# -- TFLite flatbuffers, read by hand: the runtime environment has no TFLite schema --

# Vtable offsets of the fields read, in TFLite's schema (checked against
# ai_edge_litert's schema_py_generated.py). The walk is model_info.cc:483-533's:
# Model.signature_defs -> SignatureDef.outputs -> TensorMap.tensor_index into
# the SubGraph.tensors of Model.subgraphs[SignatureDef.subgraph_index] -> Tensor.shape.
MODEL_SUBGRAPHS, MODEL_SIGNATURE_DEFS = 8, 18
SUBGRAPH_TENSORS = 4
SIGNATURE_OUTPUTS, SIGNATURE_KEY, SIGNATURE_SUBGRAPH = 6, 8, 12
TENSOR_MAP_TENSOR = 6
TENSOR_SHAPE = 4


class Flat:
    """Reads of one flatbuffer, each checked against its length."""

    def __init__(self, buf):
        self.buf = buf

    def at(self, fmt, pos):
        if not 0 <= pos <= len(self.buf) - struct.calcsize(fmt):
            raise ValueError(f"an offset points outside its {len(self.buf)} bytes")
        return struct.unpack_from(fmt, self.buf, pos)[0]

    def field(self, table, slot):
        """Where `table`'s field at vtable offset `slot` is, or None when unset."""
        vtable = table - self.at("<i", table)
        if slot >= self.at("<H", vtable):
            return None
        offset = self.at("<H", vtable + slot)
        return table + offset if offset else None

    def ref(self, pos):
        return pos + self.at("<I", pos)

    def vector(self, table, slot):
        """The positions of a vector field's 4-byte elements; none when unset."""
        pos = self.field(table, slot)
        if pos is None:
            return []
        start = self.ref(pos)
        count = self.at("<I", start)
        if count > (len(self.buf) - start - 4) // 4:
            raise ValueError(f"a vector of {count} elements runs past its end")
        return [start + 4 + 4 * i for i in range(count)]

    def tables(self, table, slot):
        return [self.ref(p) for p in self.vector(table, slot)]

    def ints(self, table, slot):
        return [self.at("<i", p) for p in self.vector(table, slot)]

    def uint(self, table, slot):
        # An unset scalar is its schema default, 0 for both read here.
        pos = self.field(table, slot)
        return 0 if pos is None else self.at("<I", pos)

    def string(self, table, slot):
        pos = self.field(table, slot)
        if pos is None:
            return None
        start = self.ref(pos)
        n = self.at("<I", start)
        if n > len(self.buf) - start - 4:
            raise ValueError("a string runs past its end")
        return bytes(self.buf[start + 4:start + 4 + n]).decode("utf-8", "replace")


def output_shapes(path):
    """(signature key, [shape of each output]) for every signature of the TFLite
    graph at `path`, in the order the graph lists them."""
    with open(path, "rb") as f:
        if not os.fstat(f.fileno()).st_size:
            raise ValueError("the graph is empty")
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
            fb = Flat(buf)
            model = fb.ref(0)
            subgraphs = fb.tables(model, MODEL_SUBGRAPHS)
            out = []
            for sig in fb.tables(model, MODEL_SIGNATURE_DEFS):
                index = fb.uint(sig, SIGNATURE_SUBGRAPH)
                if index >= len(subgraphs):
                    raise ValueError(f"a signature names subgraph {index} of {len(subgraphs)}")
                tensors = fb.tables(subgraphs[index], SUBGRAPH_TENSORS)
                shapes = []
                for output in fb.tables(sig, SIGNATURE_OUTPUTS):
                    tensor = fb.uint(output, TENSOR_MAP_TENSOR)
                    if tensor >= len(tensors):
                        raise ValueError(f"an output names tensor {tensor} of {len(tensors)}")
                    shapes.append(fb.ints(tensors[tensor], TENSOR_SHAPE))
                out.append((fb.string(sig, SIGNATURE_KEY), shapes))
            return out


def one_width(signatures, width_of, what):
    if not signatures:
        raise ValueError(f"{what} has no signature")
    for key, shapes in signatures:
        if not shapes:
            raise ValueError(f"{what}'s signature {key!r} has no output")
    widths = sorted({width_of(shape) for _, shapes in signatures for shape in shapes})
    if len(widths) != 1:
        raise ValueError(f"{what}'s outputs give {widths}")
    return widths[0]


def text_width(path):
    """The values per token the embedder writes: the product of the dims after
    the second of its output (embedding_lookup_text.cc:380-400), which the
    prefill input must match (:207-216) and which LiteRT-LM copies each media
    token's values in (embedding_lookup_multi_modal.cc:120-123). The signature is
    the first on CPU and GPU (litert_compiled_model_executor_utils.cc:1179-1183,
    embedding_lookup_text.cc:350-360) and `decode_embedder` on the NPU
    (npu/llm_litert_npu_embedder.cc:525-529); both are read when both are there.
    Output 0 of each, the one the runtime reads (`output_buffers_[0]`,
    embedding_lookup_text.cc:378-400)."""
    signatures = output_shapes(path)
    used = signatures[:1] + [s for s in signatures[1:] if s[0] == "decode_embedder"]
    used = [(key, shapes[:1]) for key, shapes in used]

    def width(shape):
        if len(shape) < 3:
            raise ValueError(f"an output has the shape {shape}, fewer than 3 dims")
        return math.prod(shape[2:])

    return one_width(used, width, "the embedder")


def adapter_width(path):
    """The values per token an adapter writes: the last dim of its output, in
    every signature the runtime may run it with -- the vision adapter's first, or
    the one an image's patch count selects (vision_executor_utils.cc:54-64,
    vision_litert_compiled_model_executor.cc:621-660), the audio adapter's only one
    (audio_litert_compiled_model_executor.cc:645-667, 764-769)."""

    def width(shape):
        if not shape:
            raise ValueError("an output has no shape")
        return shape[-1]

    return one_width(output_shapes(path), width, "the adapter")


def check_width(recipient, taken):
    """The values per token the bundle's embedder writes, refused unless each
    added adapter writes as many: LiteRT-LM copies an adapter's output into the
    text embeddings in steps of the embedder's width and checks only that enough
    is left (embedding_lookup_multi_modal.cc:148-163)."""
    embedders = [s for s in recipient["section"]
                 if s.get("section_type") == "TFLiteModel" and model_type(s) == "embedder"]
    if len(embedders) != 1:
        raise Refused(f"the bundle has {len(embedders)} embedder graphs, not one, so the "
                      "width its text model embeds at could not be established")
    try:
        text = text_width(embedders[0]["data_path"])
    except (OSError, ValueError) as exc:
        raise Refused("the width the bundle's text model embeds at could not be established "
                      f"from its embedder graph: {exc}") from None
    for s, _, t in taken:
        if s.get("section_type") == "TFLiteModel" and model_type(s) == ADAPTER[t]:
            try:
                width = adapter_width(s["data_path"])
            except (OSError, ValueError) as exc:
                raise Refused(f"the width the donor's {ADAPTER[t]} writes could not be "
                              f"established: {exc}") from None
            if width != text:
                raise Refused(f"the donor's {ADAPTER[t]} writes {width} values per token and the "
                              f"bundle's embedder {text}; LiteRT-LM would copy the one into "
                              "the other without comparing them")
    return text


def check_kept_metadata(tree, names, from_donor):
    """Refuse metadata whose media tokens the runtime could not read, or that
    names no start token for a tower being added: Gemma 4's processor would put
    its built-in string before the input instead (gemma4_data_processor_config.h:
    34-61, model_data_processor_factory.cc:191-206, multimodal_processor_helper.cc:
    127), and the tokenizer check reads only what the metadata names."""
    whose = "the donor's" if from_donor else "the bundle's"
    hint = "" if from_donor else (
        "; take the donor's with --metadata-from-donor, which replaces the whole LlmMetadata")
    case, _ = model_settings(tree)
    if case != "gemma4":
        raise Refused(f"{whose} LlmMetadata has llm_model_type {case or 'unset'}, not gemma4, "
                      f"and the token and media fields this command checks are Gemma 4's{hint}")
    starts = {TOWER_TOKEN[t] for t in names}
    problems = []
    for field in GEMMA4_TOKENS:
        union = gemma4_field(tree, field)
        if field in starts and not union:
            problems.append(f"names no {field}, so the runtime would put its built-in one "
                            "before the input, which the tokenizer check cannot see")
        elif union is not None and not any(n == "token_str" and v for n, v in union):
            problems.append(f"gives {field} without a token_str, the only form the runtime "
                            "reads it in")
    if problems:
        raise Refused(f"{whose} LlmMetadata " + " and ".join(problems) + hint)


def check_tokens(tree, recipient, donor):
    """Every token string and id `tree` names must be the same piece, of the same
    type, at the same id in the bundle's tokenizer, which is kept, and in the
    donor's, which its towers were built with. The type matters as much: the
    runtime puts a media token's string in the prompt as text
    (multimodal_processor_helper.cc:127) for SentencePiece to encode. Returns how
    many were compared."""
    own = pieces(only(recipient, "SP_Tokenizer", "the bundle")["data_path"])
    theirs = pieces(only(donor, "SP_Tokenizer", "the donor")["data_path"])
    own_ids = {p: i for i, (p, _) in reversed(list(enumerate(own)))}
    their_ids = {p: i for i, (p, _) in reversed(list(enumerate(theirs)))}

    def at(listed, i):
        return "not one piece" if i is None else (
            f"id {i} ({PIECE_TYPES.get(listed[i][1], listed[i][1])})")

    strings, ids = named_tokens(tree)
    for text in sorted(strings):
        piece = text.encode("utf-8")
        a, b = own_ids.get(piece), their_ids.get(piece)
        if a is None and b is None:
            raise Refused(f"neither tokenizer has {text!r} as one piece, so this check, which "
                          "compares single pieces, cannot say whether both encode it alike")
        if a is None or b is None or a != b or own[a][1] != theirs[b][1]:
            raise Refused(f"{text!r} is {at(own, a)} in the bundle's tokenizer and "
                          f"{at(theirs, b)} in the donor's")
    for i in sorted(ids):
        a = own[i] if 0 <= i < len(own) else None
        b = theirs[i] if 0 <= i < len(theirs) else None
        if a is None or a != b:
            raise Refused(f"id {i} is {shown_piece(a)} in the bundle's tokenizer and "
                          f"{shown_piece(b)} in the donor's")
    return len(strings) + len(ids)


def check_donor_metadata(own_tree, donor_tree, kept):
    """Refuse a donor's metadata that would change what the NPU executor reads, or
    the settings of a tower the bundle keeps."""
    own, theirs = flattened(own_tree), flattened(donor_tree)
    for field, unset in NPU_FIELDS.items():
        if own.get(field, [unset]) != theirs.get(field, [unset]):
            raise Refused(f"--metadata-from-donor would change {field} from "
                          f"{shown_values(own.get(field))} to "
                          f"{shown_values(theirs.get(field))}, and LiteRT-LM's NPU executor "
                          "reads it")
    for t in kept:
        mine, given = media(own_tree, [t]), media(donor_tree, [t])
        if mine != given:
            raise Refused(f"the bundle keeps its {t} tower, built for {shown_media(mine)}, and "
                          f"--metadata-from-donor would set {shown_media(given)}")


def add(spec, work):
    names, towers, required = spec["names"], spec["towers"], spec["required"]
    use_donor_meta = bool(spec["metadata_from_donor"])
    # Both headers before either is unpacked.
    own_places = spans(spec["artifact"], "the bundle")
    donor_places = spans(spec["donor"], "the donor")
    recipient_dir, donor_dir = work / "before", work / "donor"
    recipient, own_sources = unpacked(spec["artifact"], recipient_dir, "the bundle", own_places)
    donor, donor_sources = unpacked(spec["donor"], donor_dir, "the donor", donor_places)

    carried = {is_tower(s, list(towers), towers) for s in recipient["section"]} - {None}
    present = sorted(carried & set(names))
    if present:
        raise Refused("the bundle already carries " + " and ".join(present)
                      + " sections; drop them first")
    # Each donor section taken, where its bytes are, and its tower.
    tower_of = [is_tower(s, names, towers) for s in donor["section"]]
    taken = [(s, src, t)
             for s, src, t in zip(donor["section"], donor_sources, tower_of, strict=True) if t]
    for t in names:
        # Weights alone are not a graph the runtime can run.
        graphs = [model_type(s) for s, _, tower in taken
                  if tower == t and s.get("section_type") == "TFLiteModel"]
        missing = sorted(set(required[t]) - set(graphs))
        if missing:
            raise Refused(f"the donor's {t} tower lacks {', '.join(missing)}")
        twice = sorted({g for g in graphs if graphs.count(g) > 1})
        if twice:
            raise Refused(f"the donor carries more than one {', '.join(twice)} graph")
    width = check_width(recipient, taken)

    own_meta = only(recipient, "LlmMetadata", "the bundle")
    donor_meta = only(donor, "LlmMetadata", "the donor")
    own_tree = metadata_tree(recipient_dir, "the bundle")
    donor_tree = metadata_tree(donor_dir, "the donor")
    kept_tree = donor_tree if use_donor_meta else own_tree
    check_kept_metadata(kept_tree, names, use_donor_meta)
    tokens_checked = check_tokens(kept_tree, recipient, donor)
    if use_donor_meta:
        check_donor_metadata(own_tree, donor_tree, sorted(carried))
        changes = changed_fields(own_tree, donor_tree)
        raw_differs = ((recipient_dir / RAW_META).read_bytes()
                       != (donor_dir / RAW_META).read_bytes())
    else:
        # The towers were built for the donor's image and audio settings.
        mine, theirs = media(own_tree, names), media(donor_tree, names)
        if mine != theirs:
            raise Refused(f"the bundle's LlmMetadata sets {shown_media(mine)} where the donor's, "
                          f"which its towers were built for, sets {shown_media(theirs)}")
        changes, raw_differs = [], None

    # The result, and where each of its sections' bytes come from.
    composed, sources = copy.deepcopy(recipient), list(own_sources)
    if use_donor_meta:
        at = recipient["section"].index(own_meta)
        composed["section"][at] = copy.deepcopy(donor_meta)
        sources[at] = donor_sources[donor["section"].index(donor_meta)]
    added = []
    for s, src, t in taken:
        composed["section"].append(copy.deepcopy(s))
        sources.append(src)
        added.append(dict(describe(s, src[1]), tower=t))
    return rebuild(work, composed, sources, [recipient_dir, donor_dir], added=added,
                   metadata_source="donor" if use_donor_meta else "bundle",
                   metadata_changes=changes, metadata_raw_differs=raw_differs,
                   tokens_checked=tokens_checked, embedding_width=width)


def drop(spec, work):
    names, towers = spec["names"], spec["towers"]
    places = spans(spec["artifact"], "the bundle")
    before_dir = work / "before"
    before, sources = unpacked(spec["artifact"], before_dir, "the bundle", places)
    tower_of = [is_tower(s, names, towers) for s in before["section"]]
    dropped = [dict(describe(s, src[1]), tower=t)
               for s, src, t in zip(before["section"], sources, tower_of, strict=True) if t]
    absent = [t for t in names if t not in tower_of]
    if absent:
        raise Refused("the bundle carries no " + " or ".join(absent) + " section to drop")
    kept = dict(before, section=[s for s, t in zip(before["section"], tower_of, strict=True)
                                 if not t])
    return rebuild(work, kept, [src for src, t in zip(sources, tower_of, strict=True) if not t],
                   [before_dir], dropped=dropped)


def rebuild(work, doc, sources, scratch, **extra):
    """Pack `doc` and leave it for the parent only if it reads back as asked: the
    same TOML apart from what `pack` regenerates, and every section's raw bytes in
    the new file equal to the bytes it was taken from (`sources`)."""
    toml_path = work / "result.toml"
    write_toml(doc, toml_path)
    rebuilt = work / "rebuilt.litertlm"
    lb.pack(str(toml_path), str(rebuilt))
    # Nothing reads the unpacked inputs again: `sources` are the input files
    # themselves. Removed, they leave room for the read-back.
    for directory in scratch:
        shutil.rmtree(directory)
    places = spans(rebuilt, "the rebuilt bundle")
    after_dir = work / "after"
    lb.unpack(str(rebuilt), str(after_dir))
    after = load(after_dir / "model.toml")
    check_keys(after, "the rebuilt bundle")
    got, asked = comparable(after), comparable(doc)
    if got != asked:
        if got["system_metadata"] != asked["system_metadata"]:
            what = "its system metadata"
        elif len(got["section"]) != len(asked["section"]):
            what = f"{len(got['section'])} sections where {len(asked['section'])} were packed"
        else:
            i = next(i for i, (a, b) in enumerate(zip(got["section"], asked["section"]))
                     if a != b)
            what = f"section {i} ({label(doc['section'][i])})"
        raise Refused(f"the rebuild does not read back as asked: {what}")
    if len(places) != len(sources):
        raise Refused(f"the rebuilt bundle has {len(places)} sections where "
                      f"{len(sources)} were packed")
    for i, (place, (path, span)) in enumerate(zip(places, sources, strict=True)):
        if digest(rebuilt, place) != digest(path, span):
            raise Refused(f"the bytes of section {i} ({label(doc['section'][i])}) changed in "
                          "the rebuild")
    return report([describe(s, p) for s, p in zip(after["section"], places, strict=True)],
                  rebuilt=str(rebuilt), **extra)


def main(spec):
    work = Path(spec["work"])
    builder = importlib.metadata.version("litert-lm-builder")
    try:
        # The type lists were read from builder 0.18.0. A builder without one of
        # its types would leave part of a tower behind; one with a type it lacks
        # may have one these lists do not name.
        known = {m.value.lower().removeprefix("tf_lite_") for m in lb.TfLiteModelType}
        expected = set(spec["builder_types"])
        differences = []
        if expected - known:
            differences.append("has no model type " + ", ".join(sorted(expected - known)))
        if known - expected:
            differences.append("has model types 0.18.0 does not: "
                               + ", ".join(sorted(known - expected)))
        if differences:
            raise Refused(f"this litert-lm-builder ({builder}) " + " and ".join(differences)
                          + "; the tower lists were read from 0.18.0's")
        result = (add if spec["mode"] == "add" else drop)(spec, work)
    except Refused as exc:
        result = report(reason=str(exc))
    return dict(result, builder=builder)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)), encoding="utf-8")
'''


@dataclass(frozen=True)
class TowersResult:
    """What was dropped or added, what the result carries, and the file it is in."""

    operation: Literal["drop", "add"]
    model: Path
    output: Path
    towers: tuple[str, ...]
    sections: tuple[dict[str, Any], ...]
    dropped: tuple[dict[str, Any], ...]
    bytes_before: int
    bytes_after: int
    sha256: str
    # The litert-lm-builder version that wrote the file.
    builder: str
    added: tuple[dict[str, Any], ...] = ()
    donor: Path | None = None
    metadata_source: Literal["donor", "bundle"] | None = None
    metadata_changes: tuple[str, ...] = ()
    # With the donor's metadata: whether its raw bytes differ from the bundle's.
    metadata_raw_differs: bool | None = None
    tokens_checked: int | None = None
    # Values per token the bundle's embedder writes and every added adapter does.
    embedding_width: int | None = None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TOWERS_SCHEMA,
            "operation": self.operation,
            "model": str(self.model),
            "output": str(self.output),
            "towers": list(self.towers),
            "sections": list(self.sections),
            "dropped": list(self.dropped),
            "added": list(self.added),
            "donor": None if self.donor is None else str(self.donor),
            "metadata_source": self.metadata_source,
            "metadata_changes": list(self.metadata_changes),
            "metadata_raw_differs": self.metadata_raw_differs,
            "tokens_checked": self.tokens_checked,
            "embedding_width": self.embedding_width,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "sha256": self.sha256,
            "builder": self.builder,
            "notes": list(self.notes),
        }


VERIFY_NOTE = "What the model answers was not checked here: run `litetune verify` on {name}."

# Measured on one phone, and the reason `--add` exists at all.
ADD_NOTE = (
    "Measured once: a Gemma 4 E2B build for SM8850 given the towers and LlmMetadata of "
    "Google's CPU/GPU bundle answered an image and an audio turn on an SM8850 phone, its "
    "prefill_decode graph on the NPU and the towers on the CPU (MEASUREMENTS.md, *Towers "
    "grafted into an SM8850 NPU bundle*). Run it on the device it is for."
)


def _names(towers: Sequence[str], verb: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(towers))
    unknown = [t for t in names if t not in TOWER_SECTIONS]
    if not names or unknown:
        raise TowersError(
            f"name the towers to {verb} from {sorted(TOWER_SECTIONS)}"
            + (f"; {unknown} is not one" if unknown else "")
        )
    return names


def _spec(mode: Literal["drop", "add"], names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "mode": mode,
        "names": list(names),
        "towers": {t: list(types) for t, types in TOWER_SECTIONS.items()},
        "required": {t: list(TOWER_REQUIRED[t]) for t in names},
        "builder_types": list(BUILDER_TYPES),
    }


def _link_refusal(what: str, exc: OSError) -> str:
    text = f"{what} ({exc})"
    if exc.errno in NO_LINK_ERRNOS:
        text += "; give an output path on a filesystem that supports hard links"
    return text


def _probe_links(work: Path) -> None:
    """Refuse before the rebuild where `_publish` could not link its result after it."""
    probe = work / "link-probe"
    probe.touch()
    try:
        os.link(probe, work / "link-probe-2")
    except OSError as exc:
        raise TowersError(
            _link_refusal(f"could not hard-link a file in {work.parent}; nothing was rebuilt", exc)
        ) from None


def _rebuild(
    spec: dict[str, Any],
    model: Path,
    output: Path,
    *,
    env: envs.StageEnv,
    events: EventStream | None,
    auto_provision: bool,
    timeout: int,
) -> dict[str, Any]:
    """Run the script on `spec`; return its report once `output` is written."""
    if not model.is_file():
        raise FileNotFoundError(f"no bundle at {model}")
    # `lexists`: a symlink at `output`, dangling or not, is a name already taken.
    if os.path.lexists(output):
        raise TowersError(
            f"{output} already exists; name a new file -- the input is never overwritten"
        )
    if auto_provision:
        env.provision(events=events)
    if not env.ready:
        raise TowersError(f"environment {env.name!r} is not provisioned at {env.path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Beside the output, because `_publish` hard-links the result into place and
    # `os.link` needs both names on one filesystem.
    work = Path(tempfile.mkdtemp(prefix=f".{output.stem}-towers-", dir=output.parent))
    try:
        _probe_links(work)
        script = work / "towers.py"
        script.write_text(_TOWERS_SCRIPT, encoding="utf-8")
        spec_path, result = work / "spec.json", work / "result.json"
        spec_path.write_text(
            json.dumps({**spec, "artifact": str(model), "work": str(work)}), encoding="utf-8"
        )
        if events:
            events.note(f"towers: {spec['mode']} {', '.join(spec['names'])} ({model.name})")
        try:
            proc = env.run(["python", str(script), str(spec_path), str(result)], timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TowersError(f"the rebuild did not finish within {timeout}s") from None
        if proc.returncode != 0:
            stderr = proc.stderr or ""
            last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
            logger.warning("towers rebuild of %s failed:\n%s", model, stderr[-2000:])
            reading = read_returncode(proc.returncode)
            raise TowersError(f"the rebuild script {reading.describe('the bundle')}: {last}")
        if not result.is_file():
            raise TowersError("the rebuild script exited 0 without writing its report")
        report: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
        rebuilt = report.get("rebuilt")
        if report.get("reason") or not rebuilt or not Path(rebuilt).is_file():
            raise TowersError(
                f"{report.get('reason') or 'the rebuild wrote nothing'}; nothing was written"
            )
        _publish(Path(rebuilt), output)
        return report
    finally:
        leaked: list[str] = []
        shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
        if leaked:
            logger.warning("%d scratch file(s) left under %s", len(leaked), work)


def _publish(rebuilt: Path, output: Path) -> None:
    """Put the verified bundle at `output`, never over a file already there.

    A hard link fails if the name exists, which the check before the rebuild
    could not promise for the minutes the rebuild takes, and the file appears
    whole. Where the filesystem has no hard links this refuses rather than
    copies: a copy would sit at `output` half written, and nothing could remove
    it safely if writing it failed.
    """
    try:
        os.link(rebuilt, output)
    except FileExistsError:
        raise TowersError(
            f"{output} appeared while the bundle was rebuilt; nothing was written over it"
        ) from None
    except OSError as exc:
        raise TowersError(
            _link_refusal(f"could not link the rebuilt bundle to {output}", exc)
        ) from None


def drop_towers(
    model: Path,
    output: Path,
    towers: Sequence[str],
    *,
    env: envs.StageEnv = envs.RUNTIME,
    events: EventStream | None = None,
    auto_provision: bool = True,
    timeout: int = TOWERS_TIMEOUT_S,
) -> TowersResult:
    """Write `output`: `model` without the sections of `towers`.

    Raises `TowersError` for a request it refuses or a rebuild it would not keep,
    and `FileNotFoundError` when `model` is not there.
    """
    names = _names(towers, "drop")
    model, output = Path(model).absolute(), Path(output).absolute()
    report = _rebuild(
        _spec("drop", names), model, output, env=env, events=events,
        auto_provision=auto_provision, timeout=timeout,
    )  # fmt: skip
    return TowersResult(
        operation="drop",
        model=model,
        output=output,
        towers=names,
        sections=tuple(report["sections"]),
        dropped=tuple(report["dropped"]),
        bytes_before=model.stat().st_size,
        bytes_after=output.stat().st_size,
        sha256=hash_file(output),
        builder=report["builder"],
        notes=(VERIFY_NOTE.format(name=output.name),),
    )


def add_towers(
    model: Path,
    output: Path,
    towers: Sequence[str],
    donor: Path,
    *,
    metadata_from_donor: bool = False,
    env: envs.StageEnv = envs.RUNTIME,
    events: EventStream | None = None,
    auto_provision: bool = True,
    timeout: int = TOWERS_TIMEOUT_S,
) -> TowersResult:
    """Write `output`: `model` with `donor`'s sections of `towers`.

    The bundle's own sections are kept byte for byte, and so is its LlmMetadata
    unless `metadata_from_donor` takes the donor's, whole and byte for byte --
    needed when the bundle's metadata names no token for a tower's input, and
    refused where it would change `max_num_tokens`, `kv_cache_init_value` or the
    settings of a tower the bundle keeps. Each added adapter must write as many
    values per token as the bundle's embedder, and every `token_str` and token id
    the kept metadata names must be the same piece, of the same type, at the same
    id in both tokenizers. Raises `TowersError` for a request it refuses or a
    rebuild it would not keep, `FileNotFoundError` for a missing file.
    """
    names = _names(towers, "add")
    model, output = Path(model).absolute(), Path(output).absolute()
    donor = Path(donor).absolute()
    if not donor.is_file():
        raise FileNotFoundError(f"no donor bundle at {donor}")
    spec = {
        **_spec("add", names),
        "donor": str(donor),
        "metadata_from_donor": metadata_from_donor,
    }
    report = _rebuild(
        spec, model, output, env=env, events=events, auto_provision=auto_provision,
        timeout=timeout,
    )  # fmt: skip
    changes = tuple(report["metadata_changes"])
    notes = [VERIFY_NOTE.format(name=output.name), ADD_NOTE]
    if report["metadata_source"] == "donor":
        if changes:
            told = (
                "Its fields that differ from the bundle's own, as the builder's text form shows "
                f"them: {'; '.join(changes)}. A field the builder's proto does not declare is "
                "not among them."
            )
        elif report["metadata_raw_differs"]:
            told = (
                "No field the builder's text form shows differs from the bundle's own, but the "
                "raw bytes do."
            )
        else:
            told = "It is byte for byte the bundle's own."
        notes.insert(0, f"LlmMetadata was taken from the donor, whole and byte for byte. {told}")
    return TowersResult(
        operation="add",
        model=model,
        output=output,
        towers=names,
        sections=tuple(report["sections"]),
        dropped=(),
        added=tuple(report["added"]),
        donor=donor,
        metadata_source=report["metadata_source"],
        metadata_changes=changes,
        metadata_raw_differs=report["metadata_raw_differs"],
        tokens_checked=report["tokens_checked"],
        embedding_width=report["embedding_width"],
        bytes_before=model.stat().st_size,
        bytes_after=output.stat().st_size,
        sha256=hash_file(output),
        builder=report["builder"],
        notes=tuple(notes),
    )


def summarise(result: TowersResult) -> list[str]:
    lines = []
    verb = "dropped" if result.operation == "drop" else "added"
    for tower in result.towers:
        parts = [d for d in (*result.dropped, *result.added) if d.get("tower") == tower]
        size = sum(int(d["bytes"]) for d in parts)
        types = ", ".join(str(d["model_type"]) for d in parts)
        lines.append(f"{verb} {tower}: {len(parts)} sections ({types}), {size:,} bytes")
    if result.operation == "add":
        lines.append(
            f"each added adapter writes {result.embedding_width} values per token, as the "
            "bundle's embedder does"
        )
        lines.append(
            f"tokenizers agree on the {result.tokens_checked} token strings and ids the "
            "metadata names as tokens; "
            f"LlmMetadata from the {result.metadata_source}"
        )
    lines.append(
        f"{result.output}: {result.bytes_after:,} bytes (was {result.bytes_before:,}), written "
        f"by litert-lm-builder {result.builder}; every section read back byte for byte"
    )
    lines.extend(result.notes)
    return lines
