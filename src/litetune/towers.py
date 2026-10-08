"""`towers`: a bundle without the vision or audio sections it carries.

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

**It runs in the runtime environment**, not the export one: `litert-lm==0.18.0`
requires `litert-lm-builder==0.18.0`, so `envs.RUNTIME` already carries the
builder, and the export environment would add a converter this never calls.

**Refused rather than done:** a tower named that the bundle does not carry, an
output path that already exists (the input is never overwritten), and a rebuild
that does not read back as asked. What the model answers is not checked here;
`litetune verify` does that.
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
"""Drop named TFLiteModel sections from a bundle; verify by reading it back.

Reads JSON {artifact, output, work, towers: {name: [model_type, ...]}, drop:
[name, ...]} from argv[1]. Writes JSON to argv[2]:
  {"sections": [...], "dropped": [...], "written": <bool>, "reason": <str|null>}
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


def report(sections=(), dropped=(), written=False, reason=None):
    return {"sections": list(sections), "dropped": list(dropped), "written": written,
            "reason": reason}


def main(spec):
    artifact, output, work = spec["artifact"], spec["output"], Path(spec["work"])
    wanted = {t: set(spec["towers"][t]) for t in spec["drop"]}

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
    absent = [t for t in spec["drop"] if not any(d["tower"] == t for d in dropped)]
    if absent:
        return report(sections, reason="the bundle carries no " + " or ".join(absent)
                      + " section to drop")

    kept = copy.deepcopy(before)
    kept["section"] = [s for s in kept["section"] if not tower_of(s)]
    for s in kept["section"]:
        s["data_path"] = str(before_dir / os.path.basename(s["data_path"]))
    toml_path = work / "kept.toml"
    write_toml(kept, toml_path)
    rebuilt = work / "rebuilt.litertlm"
    lb.pack(str(toml_path), str(rebuilt))

    after_dir = work / "after"
    lb.unpack(str(rebuilt), str(after_dir))
    after = load(after_dir / "model.toml")
    if comparable(after) != comparable(kept):
        return report(sections, dropped,
                      reason="the rebuild does not read back as the input minus those sections")
    if [digest(s["data_path"]) for s in kept["section"]] != [
        digest(after_dir / os.path.basename(s["data_path"])) for s in after["section"]
    ]:
        return report(sections, dropped,
                      reason="a kept section's bytes changed in the rebuild")
    os.replace(rebuilt, output)
    return report(sections, dropped, written=True)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)))
'''


@dataclass(frozen=True)
class TowersResult:
    """What was dropped, what is left, and the file it is in."""

    model: Path
    output: Path
    towers: tuple[str, ...]
    sections: tuple[dict[str, Any], ...]
    dropped: tuple[dict[str, Any], ...]
    bytes_before: int
    bytes_after: int
    sha256: str
    notes: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TOWERS_SCHEMA,
            "model": str(self.model),
            "output": str(self.output),
            "towers": list(self.towers),
            "sections": list(self.sections),
            "dropped": list(self.dropped),
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "sha256": self.sha256,
            "notes": list(self.notes),
        }


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
    names = tuple(dict.fromkeys(towers))
    unknown = [t for t in names if t not in TOWER_SECTIONS]
    if not names or unknown:
        raise TowersError(
            f"name the towers to drop from {sorted(TOWER_SECTIONS)}"
            + (f"; {unknown} is not one" if unknown else "")
        )
    model, output = Path(model).absolute(), Path(output).absolute()
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
        spec, result = work / "spec.json", work / "result.json"
        spec.write_text(
            json.dumps(
                {
                    "artifact": str(model),
                    "output": str(output),
                    "work": str(work),
                    "towers": {t: list(TOWER_SECTIONS[t]) for t in names},
                    "drop": list(names),
                }
            ),
            encoding="utf-8",
        )
        if events:
            events.note(f"towers: dropping {', '.join(names)} from {model.name}")
        try:
            proc = env.run(["python", str(script), str(spec), str(result)], timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TowersError(f"the rebuild did not finish within {timeout}s") from None
        if proc.returncode != 0 or not result.is_file():
            stderr = proc.stderr or ""
            last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
            logger.warning("towers rebuild of %s failed:\n%s", model, stderr[-2000:])
            raise TowersError(f"the rebuild script exited {proc.returncode}: {last}")
        report = json.loads(result.read_text(encoding="utf-8"))
        if not report.get("written") or not output.is_file():
            raise TowersError(
                f"{report.get('reason') or 'the rebuild wrote nothing'}; nothing was written"
            )
    finally:
        if work is not None:
            leaked: list[str] = []
            shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
            if leaked:
                logger.warning("%d scratch file(s) left under %s", len(leaked), work)

    return TowersResult(
        model=model,
        output=output,
        towers=names,
        sections=tuple(report["sections"]),
        dropped=tuple(report["dropped"]),
        bytes_before=model.stat().st_size,
        bytes_after=output.stat().st_size,
        sha256=hash_file(output),
        notes=(
            "What the model answers was not checked here: run `litetune verify` on "
            f"{output.name}.",
        ),
    )


def summarise(result: TowersResult) -> list[str]:
    lines = []
    for tower in result.towers:
        parts = [d for d in result.dropped if d.get("tower") == tower]
        size = sum(int(d["bytes"]) for d in parts)
        types = ", ".join(str(d["model_type"]) for d in parts)
        lines.append(f"dropped {tower}: {len(parts)} sections ({types}), {size:,} bytes")
    lines.append(
        f"{result.output}: {result.bytes_after:,} bytes (was {result.bytes_before:,}); every "
        "other section read back byte for byte"
    )
    lines += list(result.notes)
    return lines
