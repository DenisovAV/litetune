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
(`litertlm_builder.py`); the script refuses a builder that lacks one. A
`section_type` alone would match the text model too. Any other section is kept.

**What the builder cannot write back is refused, not lost.** `unpack` leaves out
of its TOML the section types `pack` does not take (TTS and ASR metadata), and
`pack` raises for a `model_type` outside its enum; the header's section count
and every type are checked first. `unpack` also writes LlmMetadata as text
through the builder's own proto, which loses a field that proto does not declare
-- 2 bytes of Google's E2B metadata, `gemma4` field 13, reserved in 0.18.0 -- so
the metadata goes back in as its raw bytes, which `pack` copies unparsed.

**The result is checked by reading it back.** Every section of the new file must
hash the same, as raw bytes at its header offsets, as the section it was taken
from, and the TOML `unpack` writes for it must equal what was asked -- the system
`uuid` and `creation_timestamp` aside, which `pack` regenerates. It is put at the
output path only then, and never over a file already there.

**Adding is the other direction**, and the less certain one: LiteRT-LM does not
offer it as an operation. Checked: the donor carries the whole tower, one graph
each for its encoder, adapter and end marker; the kept LlmMetadata names the
token each added tower's input starts with -- a text-only build's names none, so
the donor's metadata is taken only when the caller asks (`metadata_from_donor`),
and the fields that differ in the builder's text form of it are reported; every
`token_str` and token id that metadata names is the same piece at the same id in
both tokenizers, read from their SentencePiece protobufs; and, when the bundle
keeps its own metadata, its image and audio settings equal the donor's. Not
checked: other strings in the metadata, the template included, and whether the
donor's adapters project into the width the bundle's text model embeds at, which
only the graphs' signatures say.

**It runs in the runtime environment**, not the export one: `litert-lm==0.18.0`
requires `litert-lm-builder==0.18.0`, so `envs.RUNTIME` already carries the
builder, and the export environment would add a converter this never calls. The
scratch directory beside the output holds the unpacked input, a donor and the
result at once: three to four times the bundle's size.

What the model answers is not checked here; `litetune verify` does that.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.events import EventStream
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

TOWERS_SCHEMA = "litetune.towers/1"

# `TfLiteModelType` in litert-lm-builder 0.18.0, without its `tf_lite_` prefix,
# which `unpack` leaves off and older bundles keep.
TOWER_SECTIONS: dict[str, tuple[str, ...]] = {
    "vision": ("vision_encoder", "vision_adapter", "end_of_vision"),
    "audio": ("audio_frontend", "audio_encoder_hw", "audio_adapter", "end_of_audio"),
}

# What a donor's tower must carry to be taken: Google's Gemma 4 E2B bundles carry
# all three of each, and an `audio_frontend` is taken when there is one.
TOWER_REQUIRED: dict[str, tuple[str, ...]] = {
    "vision": ("vision_encoder", "vision_adapter", "end_of_vision"),
    "audio": ("audio_encoder_hw", "audio_adapter", "end_of_audio"),
}

# A rebuild unpacks, packs and unpacks again, once more for a donor.
TOWERS_TIMEOUT_S = 1800


class TowersError(Exception):
    """A request `towers` will not carry out, or a rebuild it would not keep."""


_TOWERS_SCRIPT = r'''
"""Drop named tower sections from a bundle, or add them from another; leave the
result, read back and verified, in the work directory for the parent to publish.

Reads JSON {mode: "drop"|"add", artifact, work, towers: {name: [model_type, ...]},
required: {name: [model_type, ...]}, names: [name, ...], donor,
metadata_from_donor} from argv[1]. Writes JSON to argv[2]:
  {"sections": [...], "dropped": [...], "added": [...], "metadata_source": ...,
   "tokens_checked": <int>, "rebuilt": <path|null>, "reason": <str|null>}
"""
import copy
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: the builder itself depends on tomli
    import tomli as tomllib

from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder import litertlm_peek

VOLATILE = {"uuid", "creation_timestamp"}
# Sections a tower's graphs and their externalized weights live in.
GRAPH_TYPES = ("TFLiteModel", "TFLiteWeights")
# The LlmMetadata field that names the token a tower's input starts with.
TOWER_TOKEN = {"vision": "start_of_image_token", "audio": "start_of_audio_token"}


class Refused(Exception):
    """A reason this script will not write the bundle asked for."""


def load(path):
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def model_type(section):
    value = str(section.get("model_type", "")).lower()
    return value[len("tf_lite_"):] if value.startswith("tf_lite_") else value


def toml_string(text):
    out = []
    for ch in text:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
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


def comparable(doc):
    """The TOML without what `pack` regenerates and without file names, which say
    nothing about content; the content is compared as raw bytes (`spans`)."""
    sm = doc.get("system_metadata", {})
    entries = [e for e in sm.get("entries", []) if e.get("key") not in VOLATILE]
    sections = copy.deepcopy(doc.get("section", []))
    for s in sections:
        s.pop("data_path", None)
    return {"system_metadata": {"entries": entries}, "section": sections}


def spans(path):
    """Each section's (begin, end) byte offsets, from the file's own header."""
    header = litertlm_peek.read_litertlm_header(str(path), io.StringIO())
    listed = header.SectionMetadata()
    count = listed.ObjectsLength() if listed else 0
    return [(listed.Objects(i).BeginOffset(), listed.Objects(i).EndOffset())
            for i in range(count)]


def read_span(path, span):
    begin, end = span
    with open(path, "rb") as f:
        f.seek(begin)
        return f.read(end - begin)


def digest(path, span):
    begin, end = span
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(begin)
        left = end - begin
        while left:
            block = f.read(min(left, 1 << 24))
            if not block:
                break
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def describe(section, span):
    return {
        "section_type": section.get("section_type"),
        "model_type": model_type(section) or None,
        "bytes": span[1] - span[0],
    }


def unpacked(path, directory, where):
    """The bundle's TOML and each section's place in the file, ready for `pack`
    -- refused if the builder could not write it back whole.

    `unpack` dumps every section but puts in its TOML only the types `pack`
    takes: a TTS or ASR metadata section is left out, and a repack would drop it
    without a word. `pack` raises for a `model_type` outside its enum. And
    `unpack` writes LlmMetadata as text through the builder's own proto, so a
    field that proto does not declare is gone from the text: the section's raw
    bytes are written beside it as a binary `.pb`, which `pack` copies unparsed.
    """
    lb.unpack(str(path), str(directory))
    doc = load(Path(directory) / "model.toml")
    places = spans(path)
    count = len(places)
    if count != len(doc.get("section", [])):
        raise Refused(
            f"{where} has {count} sections and litert-lm-builder unpacks "
            f"{len(doc.get('section', []))} of them into a form it can write back; the rest "
            "(TTS or ASR metadata) would be lost"
        )
    for s in doc.get("section", []):
        if s.get("section_type") in GRAPH_TYPES:
            try:
                lb.TfLiteModelType.get_enum_from_tf_free_value(str(s.get("model_type", "")))
            except Exception:  # noqa: BLE001 - the builder's own refusal, restated
                raise Refused(
                    f"{where} has a {s['section_type']} section of model_type "
                    f"{s.get('model_type')!r}, which litert-lm-builder cannot write back"
                ) from None
    for s, place in zip(doc["section"], places):
        if s.get("section_type") == "LlmMetadata":
            raw = Path(directory) / "LlmMetadataProto.pb"
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
    """A SentencePiece model's pieces by id, as bytes: `ModelProto` field 1, each
    piece's field 1."""
    out = []
    try:
        for number, wire, value in fields(Path(path).read_bytes()):
            if number == 1 and wire == 2:
                piece = list(fields(value))  # all of it, so a truncated piece is refused
                out.append(next((v for n, w, v in piece if n == 1 and w == 2), None))
    except (IndexError, ValueError) as exc:
        raise Refused(f"{Path(path).name} is not a SentencePiece model: {exc}") from None
    if not out:
        raise Refused(f"{Path(path).name} holds no SentencePiece pieces")
    return out


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
                out.append((name, "".join(parts)))
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


def token_unions(tree, field=None):
    """(field name, token strings, token ids) for every TokenUnion in the tree."""
    names = [n for n, _ in tree]
    if "token_str" in names or "token_ids" in names:
        strings = [v for n, v in tree if n == "token_str" and isinstance(v, str) and v]
        ids = [int(i) for n, v in tree if n == "token_ids" and isinstance(v, list)
               for m, i in v if m == "ids"]
        yield field, strings, ids
        return
    for name, value in tree:
        if isinstance(value, list):
            yield from token_unions(value, name)


def grouped(tree):
    """Each field name's values, in order."""
    out = {}
    for name, value in tree:
        out.setdefault(name, []).append(value)
    return out


def named(tree, field):
    """Whether the metadata names a token in `field`: a TokenUnion there that is
    not empty."""
    return any(f == field and (s or i) for f, s, i in token_unions(tree))


# Fields of a model type's metadata that a tower is built for.
MEDIA_FIELDS = {
    "vision": ("patch_width", "patch_height", "max_num_patches", "pooling_kernel_size",
               "merge_patches"),
    "audio": ("skip_mel_spectrogram_extraction",),
}


def media_fields(tree, names):
    """{field: value} for the media settings of the metadata's model type."""
    model = dict(tree).get("llm_model_type")
    if not isinstance(model, list) or not model:
        return {}
    _, settings = model[0]
    if not isinstance(settings, list):
        return {}
    wanted = {f for t in names for f in MEDIA_FIELDS[t]}
    return {n: v for n, v in settings if n in wanted}


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


def add(spec, work):
    names, towers, required = spec["names"], spec["towers"], spec["required"]
    recipient_dir, donor_dir = work / "before", work / "donor"
    recipient, own_sources = unpacked(spec["artifact"], recipient_dir, "the bundle")
    donor, donor_sources = unpacked(spec["donor"], donor_dir, "the donor")
    sections = [describe(s, src[1]) for s, src in zip(recipient["section"], own_sources)]

    present = sorted({is_tower(s, names, towers) for s in recipient["section"]} - {None})
    if present:
        raise Refused("the bundle already carries " + " and ".join(present)
                      + " sections; drop them first")
    taken = [s for s in donor["section"] if is_tower(s, names, towers)]
    for t in names:
        # Weights alone are not a graph the runtime can run.
        graphs = [model_type(s) for s in taken
                  if is_tower(s, names, towers) == t and s.get("section_type") == "TFLiteModel"]
        missing = sorted(set(required[t]) - set(graphs))
        if missing:
            raise Refused(f"the donor's {t} tower lacks {', '.join(missing)}")
        twice = sorted({g for g in graphs if graphs.count(g) > 1})
        if twice:
            raise Refused(f"the donor carries more than one {', '.join(twice)} graph")

    own_meta, donor_meta = only(recipient, "LlmMetadata", "the bundle"), only(
        donor, "LlmMetadata", "the donor")
    own_tree = pbtext((recipient_dir / "LlmMetadataProto.pbtext").read_text(encoding="utf-8"))
    unnamed = [t for t in names if not named(own_tree, TOWER_TOKEN[t])]
    if unnamed and not spec["metadata_from_donor"]:
        raise Refused(
            "the bundle's LlmMetadata names no " + " or ".join(TOWER_TOKEN[t] for t in unnamed)
            + ", so the runtime would not know where an input begins; take the donor's with "
            "--metadata-from-donor, which also replaces its prompt template and stop tokens")
    use_donor_meta = bool(spec["metadata_from_donor"])
    if use_donor_meta:
        tree = pbtext((donor_dir / "LlmMetadataProto.pbtext").read_text(encoding="utf-8"))
        lacking = [t for t in names if not named(tree, TOWER_TOKEN[t])]
        if lacking:
            raise Refused("the donor's LlmMetadata names no "
                          + " or ".join(TOWER_TOKEN[t] for t in lacking) + " either")
    else:
        tree = own_tree

    # Every token the kept metadata names must be the same token in the kept
    # tokenizer (the bundle's) and the one the towers were built with (the donor's).
    own_pieces = pieces(only(recipient, "SP_Tokenizer", "the bundle")["data_path"])
    donor_pieces = pieces(only(donor, "SP_Tokenizer", "the donor")["data_path"])
    unions = list(token_unions(tree))
    strings = sorted({s for _, ss, _ in unions for s in ss})
    ids = sorted({i for _, _, ii in unions for i in ii})
    for text in strings:
        piece = text.encode("utf-8")
        a = own_pieces.index(piece) if piece in own_pieces else None
        b = donor_pieces.index(piece) if piece in donor_pieces else None
        if a is None or a != b:
            raise Refused(f"{text!r} is id {a} in the bundle's tokenizer and {b} in the donor's")
    for i in ids:
        a = own_pieces[i] if 0 <= i < len(own_pieces) else None
        b = donor_pieces[i] if 0 <= i < len(donor_pieces) else None
        if a is None or a != b:
            raise Refused(f"id {i} is {a!r} in the bundle's tokenizer and {b!r} in the donor's")

    donor_tree = tree if use_donor_meta else pbtext(
        (donor_dir / "LlmMetadataProto.pbtext").read_text(encoding="utf-8"))
    if not use_donor_meta:
        # The towers were built for the donor's image and audio settings.
        mine, theirs = media_fields(own_tree, names), media_fields(donor_tree, names)
        if mine != theirs:
            raise Refused(f"the bundle's LlmMetadata sets {mine} where the donor's, which its "
                          f"towers were built for, sets {theirs}")
    # A repeated field (stop_tokens) is compared as the whole list of its values.
    mine_all, theirs_all = grouped(own_tree), grouped(donor_tree)
    changes = sorted(n for n in mine_all.keys() | theirs_all.keys()
                     if mine_all.get(n) != theirs_all.get(n)) if use_donor_meta else []

    # The result, and where each of its sections' bytes come from.
    composed, sources = copy.deepcopy(recipient), list(own_sources)
    if use_donor_meta:
        at = next(i for i, s in enumerate(composed["section"])
                  if s.get("section_type") == "LlmMetadata")
        theirs = donor["section"].index(donor_meta)
        composed["section"][at] = copy.deepcopy(donor_meta)
        sources[at] = donor_sources[theirs]
    added = []
    for s in taken:
        i = donor["section"].index(s)
        composed["section"].append(copy.deepcopy(s))
        sources.append(donor_sources[i])
        added.append(dict(describe(s, donor_sources[i][1]), tower=is_tower(s, names, towers)))
    return rebuild(work, composed, sources, sections, (), added=added,
                   metadata_source="donor" if use_donor_meta else "bundle",
                   metadata_changes=changes, tokens_checked=len(strings) + len(ids))


def drop(spec, work):
    names, towers = spec["names"], spec["towers"]
    before, sources = unpacked(spec["artifact"], work / "before", "the bundle")
    pairs = list(zip(before["section"], sources))
    sections = [describe(s, src[1]) for s, src in pairs]
    dropped = [dict(describe(s, src[1]), tower=is_tower(s, names, towers))
               for s, src in pairs if is_tower(s, names, towers)]
    absent = [t for t in names if not any(d["tower"] == t for d in dropped)]
    if absent:
        raise Refused("the bundle carries no " + " or ".join(absent) + " section to drop")
    kept = copy.deepcopy(before)
    kept_pairs = [(s, src) for s, src in pairs if not is_tower(s, names, towers)]
    kept["section"] = [copy.deepcopy(s) for s, _ in kept_pairs]
    return rebuild(work, kept, [src for _, src in kept_pairs], sections, dropped)


def rebuild(work, doc, sources, sections, dropped, **extra):
    """Pack `doc` and leave it for the parent only if it reads back as asked: the
    same TOML apart from what `pack` regenerates, and every section's raw bytes in
    the new file equal to the bytes it was taken from (`sources`)."""
    toml_path = work / "result.toml"
    write_toml(doc, toml_path)
    rebuilt = work / "rebuilt.litertlm"
    lb.pack(str(toml_path), str(rebuilt))
    after_dir = work / "after"
    lb.unpack(str(rebuilt), str(after_dir))
    after = load(after_dir / "model.toml")
    if comparable(after) != comparable(doc):
        raise Refused("the rebuild does not read back as asked")
    places = spans(rebuilt)
    if len(places) != len(sources) or [digest(rebuilt, p) for p in places] != [
        digest(path, span) for path, span in sources
    ]:
        raise Refused("a section's bytes changed in the rebuild")
    return report([describe(s, p) for s, p in zip(after["section"], places)], dropped,
                  rebuilt=str(rebuilt), **extra)


def main(spec):
    work = Path(spec["work"])
    try:
        # The tower types were read from builder 0.18.0; a builder that renamed one
        # would drop half a tower without a word.
        known = {m.value.lower().removeprefix("tf_lite_") for m in lb.TfLiteModelType}
        unknown = sorted({t for ts in spec["towers"].values() for t in ts} - known)
        if unknown:
            raise Refused(f"this litert-lm-builder has no model type {', '.join(unknown)}; "
                          "the tower lists were read from 0.18.0's")
        return (add if spec["mode"] == "add" else drop)(spec, work)
    except Refused as exc:
        return report(reason=str(exc))


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)))
'''


@dataclass(frozen=True)
class TowersResult:
    """What was dropped or added, what the result carries, and the file it is in."""

    operation: str
    model: Path
    output: Path
    towers: tuple[str, ...]
    sections: tuple[dict[str, Any], ...]
    dropped: tuple[dict[str, Any], ...]
    bytes_before: int
    bytes_after: int
    sha256: str
    added: tuple[dict[str, Any], ...] = ()
    donor: Path | None = None
    metadata_source: str | None = None
    metadata_changes: tuple[str, ...] = ()
    tokens_checked: int | None = None
    notes: tuple[str, ...] = field(default=())

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
            "tokens_checked": self.tokens_checked,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "sha256": self.sha256,
            "notes": list(self.notes),
        }


VERIFY_NOTE = "What the model answers was not checked here: run `litetune verify` on {name}."

# Measured on one phone, and the reason `--add` exists at all.
ADD_NOTE = (
    "Not checked here: whether the donor's adapters project into the width the bundle's "
    "text model embeds at, which only the graphs' signatures say. Measured once: a Gemma 4 "
    "E2B build for SM8850 given the towers and LlmMetadata of Google's CPU/GPU bundle "
    "answered an image and an audio turn on an SM8850 phone, the text model on the NPU and "
    "the towers on the CPU (MEASUREMENTS.md, *Towers grafted into an SM8850 NPU bundle*). "
    "Run it on the device it is for."
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
    if output.exists():
        raise TowersError(
            f"{output} already exists; name a new file -- the input is never overwritten"
        )
    if not output.parent.is_dir():
        raise TowersError(f"{output.parent} is not a directory")
    if auto_provision:
        env.provision(events=events)
    if not env.ready:
        raise TowersError(f"environment {env.name!r} is not provisioned at {env.path}")

    work: Path | None = None
    try:
        # Beside the output, so the script's final `os.replace` is one rename.
        work = Path(tempfile.mkdtemp(prefix=f".{output.stem}-towers-", dir=output.parent))
        script = work / "towers.py"
        script.write_text(_TOWERS_SCRIPT, encoding="utf-8")
        spec_path, result = work / "spec.json", work / "result.json"
        spec_path.write_text(
            json.dumps({**spec, "artifact": str(model), "output": str(output), "work": str(work)}),
            encoding="utf-8",
        )
        if events:
            events.note(f"towers: {spec['mode']} {', '.join(spec['names'])} ({model.name})")
        try:
            proc = env.run(["python", str(script), str(spec_path), str(result)], timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TowersError(f"the rebuild did not finish within {timeout}s") from None
        if proc.returncode != 0 or not result.is_file():
            stderr = proc.stderr or ""
            last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
            logger.warning("towers rebuild of %s failed:\n%s", model, stderr[-2000:])
            raise TowersError(f"the rebuild script exited {proc.returncode}: {last}")
        report: dict[str, Any] = json.loads(result.read_text(encoding="utf-8"))
        rebuilt = report.get("rebuilt")
        if report.get("reason") or not rebuilt or not Path(rebuilt).is_file():
            raise TowersError(
                f"{report.get('reason') or 'the rebuild wrote nothing'}; nothing was written"
            )
        _publish(Path(rebuilt), output)
        return report
    finally:
        if work is not None:
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
            f"could not link the rebuilt bundle to {output} ({exc}); give an output path on "
            "a filesystem that supports hard links"
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
    spec = {
        "mode": "drop",
        "names": list(names),
        "towers": {t: list(TOWER_SECTIONS[t]) for t in names},
    }
    report = _rebuild(
        spec, model, output, env=env, events=events, auto_provision=auto_provision,
        timeout=timeout,
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
    unless `metadata_from_donor` takes the donor's, byte for byte -- needed when the
    bundle's metadata names no token for a tower's input, and refused without it.
    Either way every `token_str` and token id the kept metadata names must be the
    same piece at the same id in both tokenizers. Raises `TowersError` for a request
    it refuses or a rebuild it would not keep, `FileNotFoundError` for a missing file.
    """
    names = _names(towers, "add")
    model, output = Path(model).absolute(), Path(output).absolute()
    donor = Path(donor).absolute()
    if not donor.is_file():
        raise FileNotFoundError(f"no donor bundle at {donor}")
    spec = {
        "mode": "add",
        "names": list(names),
        "towers": {t: list(TOWER_SECTIONS[t]) for t in names},
        "required": {t: list(TOWER_REQUIRED[t]) for t in names},
        "donor": str(donor),
        "metadata_from_donor": metadata_from_donor,
    }
    report = _rebuild(
        spec, model, output, env=env, events=events, auto_provision=auto_provision,
        timeout=timeout,
    )  # fmt: skip
    notes = [VERIFY_NOTE.format(name=output.name), ADD_NOTE]
    if report.get("metadata_source") == "donor":
        changed = ", ".join(report.get("metadata_changes") or ()) or "nothing"
        notes.insert(
            0,
            "LlmMetadata was taken from the donor, whole and byte for byte. Its fields that "
            f"differ from the bundle's own, as the builder's text form shows them: {changed}.",
        )
    return TowersResult(
        operation="add",
        model=model,
        output=output,
        towers=names,
        sections=tuple(report["sections"]),
        dropped=(),
        added=tuple(report.get("added") or ()),
        donor=donor,
        metadata_source=report.get("metadata_source"),
        metadata_changes=tuple(report.get("metadata_changes") or ()),
        tokens_checked=report.get("tokens_checked"),
        bytes_before=model.stat().st_size,
        bytes_after=output.stat().st_size,
        sha256=hash_file(output),
        notes=tuple(notes),
    )


def summarise(result: TowersResult) -> list[str]:
    lines = []
    for tower in result.towers:
        parts = [d for d in (*result.dropped, *result.added) if d.get("tower") == tower]
        size = sum(int(d["bytes"]) for d in parts)
        types = ", ".join(str(d["model_type"]) for d in parts)
        verb = "dropped" if result.operation == "drop" else "added"
        lines.append(f"{verb} {tower}: {len(parts)} sections ({types}), {size:,} bytes")
    if result.operation == "add":
        lines.append(
            f"tokenizers agree on all {result.tokens_checked} tokens the metadata names; "
            f"LlmMetadata from the {result.metadata_source}"
        )
    lines.append(
        f"{result.output}: {result.bytes_after:,} bytes (was {result.bytes_before:,}); every "
        "section read back byte for byte"
    )
    lines += list(result.notes)
    return lines
