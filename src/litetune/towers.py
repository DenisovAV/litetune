"""`towers`: a bundle without the vision or audio sections it carries, or with
another bundle's.

A Gemma 4 `.litertlm` carries its vision and audio towers as their own sections
beside the text model, and an application that only ever sends text pays for
them in download size. Measured on Google's `gemma-4-E2B-it.litertlm`
(`litert-community/gemma-4-E2B-it-litert-lm` at `b3ca0d2f`): the six tower
sections are 332,387,932 of its 2,588,147,712 bytes, and with them dropped the
bundle answered eight greedy prompts with the same text on litert-lm 0.18.0's
CPU backend, with the same peak RSS -- a text engine does not load them
(MEASUREMENTS.md, *Gemma 4 E2B without its towers*).

**Sections are chosen by type, from an allowlist, and nothing else.** A
section is dropped only when its `section_type` is `TFLiteModel` *and* its
`model_type` is one this module names for the tower asked for -- the values
`litert-lm-builder`'s `TfLiteModelType` uses at 0.18.0 (`litertlm_builder.py`).
`section_type` alone would match the text model too. A `model_type` this module
does not know is kept, never guessed at: a later builder may add one, and a
section dropped by a guess is a bundle that fails on a device.

**The result is checked by reading it back.** The new bundle is unpacked again
and must equal the input minus exactly the dropped sections -- the system
`uuid` and `creation_timestamp` aside, which `pack` regenerates, and the
`SectionN_` index in each file name, which `unpack` renumbers -- and every kept
section's bytes must hash the same. `LlmMetadata` and the tokenizer are kept as
they were: the template inside is text metadata, and without it the runtime
derives a different one.

**Adding is the other direction**, and the less certain one: a donor's tower
sections go into a bundle that has none, which LiteRT-LM does not offer as an
operation. The bundle's own sections are kept byte for byte. Two things decide
whether a tower's output lands where the text model expects it, and both are
checked: the metadata must name the token each input starts with -- a
text-only build's does not, so the donor's metadata is taken only when the
caller says so (`metadata_from_donor`), because it also replaces the prompt
template and stop tokens -- and every token string and id that metadata names
must be the same token in both tokenizers, read straight from their
SentencePiece protobufs. A third is not checked: that the donor's adapters
project into the width the bundle's text model embeds at, which only the graphs'
signatures say. The device run in MEASUREMENTS.md is what it rests on.

**It runs in the runtime environment**, not the export one: `litert-lm==0.18.0`
requires `litert-lm-builder==0.18.0`, so `envs.RUNTIME` already carries the
builder, and the export environment would add a converter this never calls.

**Refused rather than done:** a tower to drop that the bundle does not carry; a
tower to add that it already carries or the donor does not; metadata that names
no input token for an added tower, unless the donor's is asked for; tokenizers
that disagree on a token the metadata names; an output path that already exists
(the input is never overwritten); and a rebuild that does not read back as asked.
What the model answers is not checked here; `litetune verify` does that.
"""

from __future__ import annotations

import json
import logging
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

# Unpack, pack and unpack again of a 2.6 GB bundle took 15 s on an M4 Pro.
TOWERS_TIMEOUT_S = 1800


class TowersError(Exception):
    """A request `towers` will not carry out, or a rebuild it would not keep."""


_TOWERS_SCRIPT = r'''
"""Drop named TFLiteModel sections from a bundle, or add them from another;
verify by reading the result back.

Reads JSON {mode: "drop"|"add", artifact, output, work, towers: {name:
[model_type, ...]}, names: [name, ...], donor, metadata_from_donor} from
argv[1]. Writes JSON to argv[2]:
  {"sections": [...], "dropped": [...], "added": [...], "metadata_source": ...,
   "tokens_checked": <int>, "written": <bool>, "reason": <str|null>}
"""
import copy
import hashlib
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

VOLATILE = {"uuid", "creation_timestamp"}


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


def file_name(path):
    """A section's file name without the index `unpack` gives it."""
    return re.sub(r"^Section\d+_", "", os.path.basename(path))


def comparable(doc):
    sm = doc.get("system_metadata", {})
    entries = [e for e in sm.get("entries", []) if e.get("key") not in VOLATILE]
    sections = copy.deepcopy(doc.get("section", []))
    for s in sections:
        if "data_path" in s:
            s["data_path"] = file_name(s["data_path"])
    return {"system_metadata": {"entries": entries}, "section": sections}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def describe(section, directory):
    path = Path(directory) / os.path.basename(section["data_path"])
    return {
        "section_type": section.get("section_type"),
        "model_type": model_type(section) or None,
        "bytes": path.stat().st_size,
    }


def report(sections=(), dropped=(), written=False, reason=None, **extra):
    return {"sections": list(sections), "dropped": list(dropped), "written": written,
            "reason": reason, **extra}


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
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            n, i = varint(buf, i)
            value, i = buf[i:i + n], i + n
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"protobuf wire type {wire} is not one a tokenizer uses")
        yield number, wire, value


def pieces(path):
    """A SentencePiece model's pieces by id: `ModelProto` field 1, each piece's
    field 1. Read from the protobuf directly; the runtime environment has no
    sentencepiece."""
    out = []
    for number, wire, value in fields(Path(path).read_bytes()):
        if number == 1 and wire == 2:
            out.append(next((v.decode("utf-8", "replace") for n, w, v in fields(value)
                             if n == 1 and w == 2), None))
    return out


def unquote(text):
    return text.replace('\\"', '"').replace("\\\\", "\\")


def named_tokens(pbtext):
    """The token strings and ids an LlmMetadata text proto names."""
    strings = [unquote(m) for m in re.findall(r'token_str:\s*"((?:[^"\\]|\\.)*)"', pbtext)]
    ids = [int(m) for m in re.findall(r"\bids:\s*(\d+)", pbtext)]
    return strings, ids


# The LlmMetadata field that names the token a tower's input starts with.
TOWER_TOKEN = {"vision": "start_of_image_token", "audio": "start_of_audio_token"}


def add(spec, work):
    names, towers = spec["names"], spec["towers"]
    recipient_dir, donor_dir = work / "before", work / "donor"
    lb.unpack(spec["artifact"], str(recipient_dir))
    lb.unpack(spec["donor"], str(donor_dir))
    recipient = load(recipient_dir / "model.toml")
    donor = load(donor_dir / "model.toml")
    sections = [describe(s, recipient_dir) for s in recipient.get("section", [])]

    def tower_of(section):
        if section.get("section_type") != "TFLiteModel":
            return None
        return next((t for t in names if model_type(section) in towers[t]), None)

    present = sorted({tower_of(s) for s in recipient["section"] if tower_of(s)})
    if present:
        return report(sections, reason="the bundle already carries " + " and ".join(present)
                      + " sections; drop them first")
    taken = [s for s in donor["section"] if tower_of(s)]
    absent = [t for t in names if not any(tower_of(s) == t for s in taken)]
    if absent:
        return report(sections, reason="the donor carries no " + " or ".join(absent)
                      + " section to take")

    def only(doc, section_type, where):
        found = [s for s in doc["section"] if s.get("section_type") == section_type]
        if len(found) != 1:
            raise SystemExit(f"{where} has {len(found)} {section_type} sections")
        return found[0]

    own_meta = only(recipient, "LlmMetadata", "the bundle")
    donor_meta = only(donor, "LlmMetadata", "the donor")
    own_path = recipient_dir / os.path.basename(own_meta["data_path"])
    own_text = own_path.read_text(encoding="utf-8")
    unnamed = [t for t in names if TOWER_TOKEN[t] not in own_text]
    if unnamed and not spec["metadata_from_donor"]:
        return report(sections, reason=(
            "the bundle's LlmMetadata names no " + " or ".join(TOWER_TOKEN[t] for t in unnamed)
            + ", so the runtime would not know where an input begins; take the donor's with "
            "--metadata-from-donor, which also replaces its prompt template and stop tokens"))
    use_donor_meta = bool(spec["metadata_from_donor"])
    meta_dir, meta = (donor_dir, donor_meta) if use_donor_meta else (recipient_dir, own_meta)
    meta_text = (meta_dir / os.path.basename(meta["data_path"])).read_text(encoding="utf-8")

    # Every token the metadata names must be the same token in the tokenizer that
    # is kept (the bundle's) and in the one the towers were built with (the donor's).
    own_tok = only(recipient, "SP_Tokenizer", "the bundle")["data_path"]
    donor_tok = only(donor, "SP_Tokenizer", "the donor")["data_path"]
    own_pieces = pieces(recipient_dir / os.path.basename(own_tok))
    donor_pieces = pieces(donor_dir / os.path.basename(donor_tok))
    strings, ids = named_tokens(meta_text)
    for text in strings:
        a = own_pieces.index(text) if text in own_pieces else None
        b = donor_pieces.index(text) if text in donor_pieces else None
        if a is None or a != b:
            return report(sections, reason=f"{text!r} is id {a} in the bundle's tokenizer and "
                          f"{b} in the donor's")
    for i in ids:
        a = own_pieces[i] if i < len(own_pieces) else None
        b = donor_pieces[i] if i < len(donor_pieces) else None
        if a is None or a != b:
            return report(sections, reason=f"id {i} is {a!r} in the bundle's tokenizer and "
                          f"{b!r} in the donor's")

    composed = copy.deepcopy(recipient)
    for s in composed["section"]:
        s["data_path"] = str(recipient_dir / os.path.basename(s["data_path"]))
    if use_donor_meta:
        at = next(i for i, s in enumerate(composed["section"])
                  if s.get("section_type") == "LlmMetadata")
        replacement = copy.deepcopy(donor_meta)
        replacement["data_path"] = str(donor_dir / os.path.basename(donor_meta["data_path"]))
        composed["section"][at] = replacement
    added = []
    for s in taken:
        moved = copy.deepcopy(s)
        moved["data_path"] = str(donor_dir / os.path.basename(s["data_path"]))
        composed["section"].append(moved)
        added.append(dict(describe(s, donor_dir), tower=tower_of(s)))
    return rebuild(spec, work, composed, sections, (), added=added,
                   metadata_source="donor" if use_donor_meta else "bundle",
                   tokens_checked=len(strings) + len(ids))


def rebuild(spec, work, doc, sections, dropped, **extra):
    """Pack `doc`, read it back, and replace the output only if it reads back as asked."""
    toml_path = work / "result.toml"
    write_toml(doc, toml_path)
    rebuilt = work / "rebuilt.litertlm"
    lb.pack(str(toml_path), str(rebuilt))
    after_dir = work / "after"
    lb.unpack(str(rebuilt), str(after_dir))
    after = load(after_dir / "model.toml")
    if comparable(after) != comparable(doc):
        return report(sections, dropped, reason="the rebuild does not read back as asked", **extra)
    if [digest(s["data_path"]) for s in doc["section"]] != [
        digest(after_dir / os.path.basename(s["data_path"])) for s in after["section"]
    ]:
        return report(sections, dropped,
                      reason="a section's bytes changed in the rebuild", **extra)
    os.replace(rebuilt, spec["output"])
    return report([describe(s, after_dir) for s in after["section"]], dropped, written=True,
                  **extra)


def main(spec):
    work = Path(spec["work"])
    if spec["mode"] == "add":
        return add(spec, work)
    artifact = spec["artifact"]
    wanted = {t: set(spec["towers"][t]) for t in spec["names"]}

    before_dir = work / "before"
    lb.unpack(artifact, str(before_dir))
    before = load(before_dir / "model.toml")
    sections = [describe(s, before_dir) for s in before.get("section", [])]

    def tower_of(section):
        if section.get("section_type") != "TFLiteModel":
            return None
        return next((t for t, types in wanted.items() if model_type(section) in types), None)

    dropped = [
        dict(describe(s, before_dir), tower=tower_of(s))
        for s in before.get("section", [])
        if tower_of(s)
    ]
    absent = [t for t in spec["names"] if not any(d["tower"] == t for d in dropped)]
    if absent:
        return report(sections, reason="the bundle carries no " + " or ".join(absent)
                      + " section to drop")

    kept = copy.deepcopy(before)
    kept["section"] = [s for s in kept["section"] if not tower_of(s)]
    for s in kept["section"]:
        s["data_path"] = str(before_dir / os.path.basename(s["data_path"]))
    return rebuild(spec, work, kept, sections, dropped)


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
            "tokens_checked": self.tokens_checked,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "sha256": self.sha256,
            "notes": list(self.notes),
        }


VERIFY_NOTE = "What the model answers was not checked here: run `litetune verify` on {name}."

# Measured on one phone, and the reason `--add` exists at all.
ADD_NOTE = (
    "Not checked here: that the donor's adapters project into the width the bundle's text "
    "model embeds at, which only the graphs' signatures say. Measured once: Google's "
    "SM8850 Gemma 4 E2B bundle with the towers and metadata of its CPU/GPU bundle "
    "answered an image and an audio turn on an SM8850 NPU (MEASUREMENTS.md, *Towers "
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
        if not report.get("written") or not output.is_file():
            raise TowersError(
                f"{report.get('reason') or 'the rebuild wrote nothing'}; nothing was written"
            )
        return report
    finally:
        if work is not None:
            leaked: list[str] = []
            shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
            if leaked:
                logger.warning("%d scratch file(s) left under %s", len(leaked), work)


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
    """Write `output`: `model` without the sections of `towers`. Raises `TowersError`."""
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
    """Write `output`: `model` with `donor`'s sections of `towers`. Raises `TowersError`.

    The bundle's own sections are kept byte for byte, and so is its LlmMetadata
    unless `metadata_from_donor` -- needed when the bundle's metadata names no token
    for a tower's input, and refused without it. Either way every token string and
    id the kept metadata names must be the same token in both tokenizers.
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
        "donor": str(donor),
        "metadata_from_donor": metadata_from_donor,
    }
    report = _rebuild(
        spec, model, output, env=env, events=events, auto_provision=auto_provision,
        timeout=timeout,
    )  # fmt: skip
    notes = [VERIFY_NOTE.format(name=output.name), ADD_NOTE]
    if report.get("metadata_source") == "donor":
        notes.insert(
            0,
            "LlmMetadata was taken from the donor: its prompt template, stop tokens and media "
            "fields replace the bundle's.",
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
