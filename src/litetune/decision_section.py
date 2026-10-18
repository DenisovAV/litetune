"""The decision a model answers, carried inside its `.litertlm` as one section.

In the 2026-10-16 design, flutter_edge_ai's `createDecider` is to read the
question, the trained options, the temperature, the abstain threshold and the
scoring form from the model file, and nothing may ship beside the model. They travel
in a `GenericBinaryData` section whose `name` item is `litetune.decision` and whose
payload is the UTF-8 JSON `verify` wrote (`decision_verify.section_payload`).

**A section, not header `system_metadata`.** LiteRT-LM reads at most 16 KiB of
header (`kLitertLmHeaderMaxSize`, `runtime/util/litert_lm_loader.h:44` at
v0.18.0) and litert-lm-builder does not check it; 151 options with
descriptions longer than 63 bytes each would not fit (computed 2026-10-16; no such
bundle was built). The LLM runtime never maps a `GenericBinaryData`
section (`litert_lm_loader.cc:189-201,369-404`). Measured 2026-10-16 on a
CLINC151 bundle: a 3,480-byte payload moved the header end from 760 to 864
bytes and grew the file by 12,024 bytes; the payload read back byte for byte,
survived an unpack and repack, and the bundle loaded and scored bit-identically
on LiteRT-LM 0.18.0 and on a nightly.

The script runs in `envs.EXPORT`, beside litetune's own repack script, whose
TOML reading and writing it uses: unpack, append the section (replacing one a
previous build wrote), pack, then check the header end, unpack the result again
and compare it with what was built, and read the payload back from the file's
own offsets before it replaces the original.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.events import EventStream

logger = logging.getLogger(__name__)

SECTION_NAME = "litetune.decision"
HEADER_MAX = 16 * 1024
SECTION_TIMEOUT_S = 1800
STDERR_TAIL_CHARS = 2000

SECTION_SCRIPT = r'''
"""Add, replace or remove the litetune.decision section. Spec: argv[1]; result JSON: argv[2]."""
import copy
import hashlib
import io
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repack as R  # litetune's repack script, written beside this one

from litert_lm_builder import litertlm_builder as lb
from litert_lm_builder import litertlm_peek


def named(section, name):
    return section.get("section_type") == "GenericBinaryData" and any(
        e.get("key") == "name" and str(e.get("value")) == name
        for e in section.get("additional_metadata") or []
    )


def header_end(path):
    with open(path, "rb") as fh:
        fh.seek(R.litertlm_core.HEADER_END_LOCATION_BYTE_OFFSET)
        return struct.unpack("<Q", fh.read(8))[0]


def section_bytes(path, name):
    """The payloads of every section named `name`, read by the header's own offsets."""
    md = litertlm_peek.read_litertlm_header(str(path), io.StringIO())
    secs = md.SectionMetadata()
    found = []
    for i in range(secs.ObjectsLength() if secs else 0):
        obj = secs.Objects(i)
        items = [litertlm_peek.kvp_to_dict(obj.Items(j)) for j in range(obj.ItemsLength())]
        if any(it.get("key") == "name" and it.get("value") == name for it in items):
            with open(path, "rb") as fh:
                fh.seek(obj.BeginOffset())
                found.append(fh.read(obj.EndOffset() - obj.BeginOffset()))
    return found


def main(spec):
    artifact, work, name = spec["artifact"], Path(spec["work"]), spec["name"]
    # No payload: take every section named `name` out and add none.
    payload = None if spec["payload"] is None else Path(spec["payload"]).read_bytes()
    cut = R.past_the_end(artifact)
    if cut is not None:
        return {"written": False, "reason": cut}
    end_before = header_end(artifact)
    before_dir = work / "before"
    lb.unpack(artifact, str(before_dir))
    before = R.load(before_dir / "model.toml")
    kept = [s for s in before.get("section", []) if not named(s, name)]
    replaced = len(before.get("section", [])) - len(kept)
    if payload is None and not replaced:
        # Nothing to take out, so the file is left as it is.
        return {"written": False, "reason": None, "replaced": 0, "header_end_before": end_before}
    for s in kept:
        s["data_path"] = str(before_dir / os.path.basename(s["data_path"]))
    added = []
    if payload is not None:
        # Named as `unpack` names it, so the round trip compares like with like.
        data = work / f"Section{len(kept)}_GenericBinaryData.bin"
        data.write_bytes(payload)
        added = [
            {
                "section_type": "GenericBinaryData",
                "data_path": str(data),
                "additional_metadata": [{"key": "name", "value_type": "String", "value": name}],
            }
        ]
    built = copy.deepcopy(before)
    built["section"] = kept + added
    toml = work / "decision.toml"
    R.write_toml(built, toml)
    rebuilt = work / Path(artifact).name
    lb.pack(str(toml), str(rebuilt))
    end = header_end(rebuilt)
    if end > spec["header_max"]:
        return {"written": False, "reason": f"the header would end at byte {end}, past "
                f"{spec['header_max']}, which LiteRT-LM does not read past"}
    after_dir = work / "after"
    lb.unpack(str(rebuilt), str(after_dir))
    after = R.load(after_dir / "model.toml")
    if R.comparable(after) != R.comparable(built):
        return {"written": False, "reason": "the rebuilt bundle unpacks to something else"}
    read = section_bytes(rebuilt, name)
    if read != ([] if payload is None else [payload]):
        return {"written": False, "reason": f"{len(read)} section(s) named {name} read back, "
                + ("after taking them out" if payload is None else "not the one payload written")}
    os.replace(rebuilt, artifact)
    return {
        "written": True,
        "reason": None,
        "header_end": end,
        "header_end_before": end_before,
        "replaced": replaced,
        "payload_bytes": None if payload is None else len(payload),
        "payload_sha256": None if payload is None else hashlib.sha256(read[0]).hexdigest(),
        "sections": len(built["section"]),
    }


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)), encoding="utf-8")
'''


@dataclass(frozen=True)
class SectionWrite:
    """What writing the section did. `reason` is set exactly when nothing was written."""

    written: bool
    reason: str | None
    report: dict[str, Any]


@dataclass(frozen=True)
class SectionRemoval:
    """What taking the section out did.

    `reason` is set exactly when the file is not known to be without one; `removed`
    counts the sections taken out, 0 for a file that had none and was left as it is.
    """

    removed: int
    reason: str | None
    report: dict[str, Any]


def payload_bytes(payload: dict[str, Any]) -> bytes:
    """The section's bytes: compact UTF-8 JSON, keys in the order `verify` wrote them.

    Strict JSON: a value that is not finite raises rather than being written as
    `NaN` or `Infinity`, which Python accepts and a JSON decoder does not.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def write_section(
    artifact: Path,
    payload: dict[str, Any],
    env: envs.StageEnv = envs.EXPORT,
    auto_provision: bool = True,
    events: EventStream | None = None,
    timeout: int = SECTION_TIMEOUT_S,
) -> SectionWrite:
    """Write the decision section into `artifact` in place, and read it back."""
    data = payload_bytes(payload)
    report, failure = _run_section_script(artifact, data, env, auto_provision, events, timeout)
    if failure is not None:
        return SectionWrite(False, failure, report)
    if not report.get("written"):
        return SectionWrite(False, str(report.get("reason") or "no reason given"), report)
    if report.get("payload_sha256") != hashlib.sha256(data).hexdigest():
        return SectionWrite(False, "the section read back is not the payload written", report)
    return SectionWrite(True, None, report)


def remove_section(
    artifact: Path,
    env: envs.StageEnv = envs.EXPORT,
    auto_provision: bool = True,
    events: EventStream | None = None,
    timeout: int = SECTION_TIMEOUT_S,
) -> SectionRemoval:
    """Take every decision section out of `artifact` in place, and check that none is left."""
    report, failure = _run_section_script(artifact, None, env, auto_provision, events, timeout)
    if failure is not None:
        return SectionRemoval(0, failure, report)
    if report.get("reason"):
        return SectionRemoval(0, str(report["reason"]), report)
    removed = report.get("replaced")
    if not isinstance(removed, int) or isinstance(removed, bool):
        return SectionRemoval(0, "the section script reported no count of sections removed", report)
    return SectionRemoval(removed, None, report)


def _run_section_script(
    artifact: Path,
    data: bytes | None,
    env: envs.StageEnv,
    auto_provision: bool,
    events: EventStream | None,
    timeout: int,
) -> tuple[dict[str, Any], str | None]:
    """Run `SECTION_SCRIPT` on `artifact`: its report, or why there is none.

    `data` is the payload to write, or `None` to take the section out.
    """
    from litetune.export import _REPACK_SCRIPT

    if auto_provision:
        try:
            env.provision(events=events)
        except Exception as exc:  # noqa: BLE001 - reported as the reason
            logger.exception("could not provision environment %r", env.name)
            return {}, f"the {env.name} environment could not be provisioned: {exc}"
    # Beside the artifact, never in the system temp: the script's last step is
    # an `os.replace` onto the artifact, which fails across filesystems (a tmpfs
    # /tmp), and the unpacks are a model's size each. As `export`'s repack.
    artifact = Path(artifact).absolute()
    with tempfile.TemporaryDirectory(
        prefix=f".{artifact.stem}-section-", dir=artifact.parent
    ) as tmp:
        work = Path(tmp)
        (work / "repack.py").write_text(_REPACK_SCRIPT, encoding="utf-8")
        script = work / "section.py"
        script.write_text(SECTION_SCRIPT, encoding="utf-8")
        if data is not None:
            (work / "payload.json").write_bytes(data)
        spec = work / "spec.json"
        result = work / "result.json"
        spec.write_text(
            json.dumps(
                {
                    "artifact": str(artifact),
                    "work": str(work),
                    "name": SECTION_NAME,
                    "payload": None if data is None else str(work / "payload.json"),
                    "header_max": HEADER_MAX,
                }
            ),
            encoding="utf-8",
        )
        try:
            proc = env.run(["python", str(script), str(spec), str(result)], timeout=timeout)
        except subprocess.TimeoutExpired as expired:
            # The child's last words, as evaluate.py keeps them on a timeout: a
            # budget alone says nothing about where the script was.
            killed = expired.stderr or ""
            if isinstance(killed, bytes):
                killed = killed.decode("utf-8", "surrogateescape")
            tail = killed[-STDERR_TAIL_CHARS:].strip()
            return {}, (
                f"no result after {timeout} s (timeout)"
                + (f"; its stderr ended: {tail}" if tail else "")
            )
        except OSError as exc:
            logger.exception("could not start the section script in environment %r", env.name)
            return {}, f"the section script could not be started: {type(exc).__name__}: {exc}"
        if proc.returncode != 0 or not result.is_file():
            last = next(
                (ln for ln in reversed((proc.stderr or "").splitlines()) if ln.strip()),
                "no stderr",
            )
            return {}, f"the section script exited {proc.returncode}: {last}"
        report = json.loads(result.read_text(encoding="utf-8"))
    return report, None


def dart_string(text: str) -> str:
    """A single-quoted Dart string literal: `$` would interpolate, `\\` and `'` would end it.

    Every character that is not printable ASCII is written as Dart's `\\u{XXXX}`, so the
    printed call is ASCII and keeps the lines `dart_call` gives it: a line separator in
    an id (U+2028, U+0085) would otherwise split it under `str.splitlines()`.
    """
    escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("$", "\\$")
    return "'" + "".join(c if " " <= c <= "~" else f"\\u{{{ord(c):X}}}" for c in escaped) + "'"


def dart_call(payload: dict[str, Any]) -> str:
    """The call an app makes: the trained options, in their trained order.

    The `none_option` is not among them: it is the section's, as the question is,
    and the 2026-10-16 design has flutter_edge_ai list it last itself.
    """
    options = ", ".join(f"DeciderOption({dart_string(o['id'])})" for o in payload["options"])
    return "final decider = await model.createDecider(\n" f"  options: [{options}],\n" ");"
