"""`embedding_vision`: a grafted EmbeddingGemma 2 text tower spliced into Google's
text-vision file, for a checkpoint tuned on images.

Google publishes EmbeddingGemma 2 with its vision tower in a file of its own, in
another repository than the text file (`VISION_REPO`, `VISION_FILE` at
`VISION_REVISION`, `VISION_BYTES` bytes, sha256 `VISION_SHA256`). Its sections,
read with litert-lm-builder 0.18.0 on 2026-11-04:
the EmbeddingMetadata, the SentencePiece tokenizer, the `embedder` and
`text_encoder` graphs and weights, which are byte for byte the sections of the text
file the graft writes into (`embedding_convert.GOOGLE_FILE`), and the
`vision_encoder`, `vision_adapter` and `end_of_vision` graphs with the weights of
the first two. The four text sections are described alike in both files' TOML
(no backend constraint, no additional metadata).

An image checkpoint trains the text tower only; the vision tower, `embed_vision`
and the token embeddings are frozen. So `convert` grafts the fine-tune into the
text file as for a text checkpoint (`embedding_convert.graft`), then `splice`
puts the graft's four text sections in place of the text-vision file's and
repacks it with litert-lm-builder. Every other section, and the TOML, stay the
text-vision file's: `vision_encoder` and `vision_adapter`, and `end_of_vision`,
whose output is a constant, the base model's token embedding of `<image|>`
(`eoi_token_id` 258882 in its config.json at the pinned revision) times
sqrt(512), the text hidden size. So the kept sections are right for a fine-tune
only while those tensors are the base's, and `frozen_changes` holds `convert`
to that: an image checkpoint whose `FROZEN_PREFIXES` tensors are not the base's
is refused before the graft.

**Read back twice.** The towers script's `rebuild` packs the TOML and checks the
packed file against what was packed: the same TOML apart from `uuid` and
`creation_timestamp`, which `pack` stamps anew, and every section's bytes equal to
the bytes it was taken from. Then this script reads the packed file again against
the two input files by role (`read_back`): its TOML is the text-vision file's
apart from those two entries, each of the four text sections holds the graft's
bytes, and every other section the text-vision file's.

**Refusals name their cause** (`EmbeddingConvertError.cause`): `base_sha256` for a
text-vision file whose sha256 is not `VISION_SHA256`; `input` for a file the
builder cannot unpack whole, one without exactly one of each text section, or a
graft whose text sections the TOML describes otherwise than the text-vision
file's; `readback` for a pack that does not read back; `run` for a script that did
not finish.

What the file embeds is not checked here: `litetune verify` measures it.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from litetune import envs
from litetune.embedding_convert import EmbeddingConvertError, publish_packed
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.towers import _TOWERS_SCRIPT

logger = logging.getLogger(__name__)

VISION_REPO = "litert-community/embeddinggemma-2-text-vision-440m-litert-lm"
VISION_FILE = "embeddinggemma-2-text-vision-440m.litertlm"
VISION_REVISION = "e301f74d5551b0c2641bd5cb4652a76239d5c5f8"
# The file at VISION_REVISION, hashed 2026-11-04 from the Hub cache.
VISION_SHA256 = "92dcbea108899e5d6e30d919b0744f90d9967e80c67a4ab5503ac16d54f62eb0"
VISION_BYTES = 387_710_976

# The sections the splice takes from the graft: each part's graph and weights.
TEXT_PARTS: tuple[str, ...] = ("embedder", "text_encoder")
TEXT_KINDS: tuple[str, ...] = ("TFLiteModel", "TFLiteWeights")

# An allowance for an unpack, a pack and a read-back of a few hundred MB, not a
# measurement.
SPLICE_TIMEOUT_S = 1800

# The tensors the image recipe freezes (`embedding_tune.ImageRecipe.frozen`), by
# the names a checkpoint `tune` saved and the base checkpoint both give them.
# The splice keeps Google's sections computed from them.
FROZEN_PREFIXES: tuple[str, ...] = (
    "vision_tower.",
    "embed_vision.",
    "language_model.embed_tokens.",
)
# A safetensors header larger than this is not one either file has.
_MAX_HEADER = 100 << 20
# Values compared at a time: a few MB of each file.
_CHUNK_VALUES = 1 << 22
_WIDTH = {"F32": 4, "BF16": 2}


def _header(path: Path, handle: BinaryIO) -> tuple[int, dict[str, tuple[str, list[int], int, int]]]:
    """A safetensors file's data offset and its tensors: name -> (dtype, shape, begin, end)."""
    size = path.stat().st_size
    raw = handle.read(8)
    if len(raw) != 8:
        raise EmbeddingConvertError("input", f"{path} is {size} bytes, too short for safetensors")
    (n,) = struct.unpack("<Q", raw)
    if n > _MAX_HEADER or 8 + n > size:
        raise EmbeddingConvertError("input", f"{path}: a {n}-byte header does not fit the file")
    try:
        header = json.loads(handle.read(n))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbeddingConvertError("input", f"{path}: its header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise EmbeddingConvertError("input", f"{path}: its header is not a JSON object")
    header.pop("__metadata__", None)
    tensors: dict[str, tuple[str, list[int], int, int]] = {}
    for name, t in header.items():
        try:
            dtype, shape = str(t["dtype"]), [int(d) for d in t["shape"]]
            begin, end = (int(o) for o in t["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingConvertError(
                "input", f"{path}: tensor {name!r} has no dtype, shape and data_offsets"
            ) from exc
        if not 0 <= begin <= end or 8 + n + end > size:
            raise EmbeddingConvertError("input", f"{path}: tensor {name!r} lies outside the file")
        tensors[name] = (dtype, shape, 8 + n + begin, 8 + n + end)
    return 8 + n, tensors


def _values(handle: BinaryIO, at: int, end: int, dtype: str) -> Iterator[bytes]:
    """A tensor's values as little-endian float32 bytes, a chunk at a time.

    BF16 is the upper half of a float32, so widening it is exact: the low two
    bytes of each value are zero.
    """
    width = _WIDTH[dtype]
    while at < end:
        handle.seek(at)
        raw = handle.read(min(end - at, _CHUNK_VALUES * width))
        if not raw:
            raise EmbeddingConvertError("input", "a safetensors file ended inside a tensor")
        at += len(raw)
        if dtype == "F32":
            yield raw
            continue
        wide = bytearray(len(raw) * 2)
        wide[2::4] = raw[0::2]
        wide[3::4] = raw[1::2]
        yield bytes(wide)


def _same(a: Iterator[bytes], b: Iterator[bytes]) -> bool:
    """Whether two streams of bytes are the same bytes, however each is chunked."""
    left, right = b"", b""
    while True:
        if not left:
            left = next(a, b"")
        if not right:
            right = next(b, b"")
        if not left or not right:
            return not left and not right
        n = min(len(left), len(right))
        if left[:n] != right[:n]:
            return False
        left, right = left[n:], right[n:]


def frozen_changes(base: Path, tuned: Path) -> list[str]:
    """The `FROZEN_PREFIXES` tensors `tuned` does not hold as `base` does, each with how.

    Compared by value as float32: the base's BF16 widened exactly, the
    fine-tune's F32 as written, either side in either type. A tensor missing
    from one side, of another shape, or of a type other than F32 and BF16 is a
    change. Raises `EmbeddingConvertError("input", ...)` for a file that is not
    safetensors, or when neither holds a tensor under the prefixes. The
    standard library only, in this process.
    """
    changes: list[str] = []
    with base.open("rb") as b_handle, tuned.open("rb") as t_handle:
        _, b_tensors = _header(base, b_handle)
        _, t_tensors = _header(tuned, t_handle)
        names = sorted(n for n in {*b_tensors, *t_tensors} if n.startswith(FROZEN_PREFIXES))
        if not names:
            raise EmbeddingConvertError(
                "input", f"neither {base} nor {tuned} holds a tensor under {list(FROZEN_PREFIXES)}"
            )
        for name in names:
            if name not in t_tensors or name not in b_tensors:
                side = "fine-tune" if name in b_tensors else "base"
                changes.append(f"{name} (not in the {side})")
                continue
            b_type, b_shape, b_begin, b_end = b_tensors[name]
            t_type, t_shape, t_begin, t_end = t_tensors[name]
            if b_shape != t_shape:
                changes.append(f"{name} (shape {t_shape}, the base's {b_shape})")
            elif b_type not in _WIDTH or t_type not in _WIDTH:
                changes.append(f"{name} ({t_type} against the base's {b_type}, not compared)")
            elif not _same(
                _values(t_handle, t_begin, t_end, t_type),
                _values(b_handle, b_begin, b_end, b_type),
            ):
                changes.append(f"{name} (other values)")
    return changes


_SPLICE_SCRIPT = r'''
"""Splice a grafted text tower into Google's text-vision EmbeddingGemma 2 file;
leave the result, packed and read back, in the work directory for the parent to
publish.

Reads JSON {grafted, vision, vision_sha256, work, parts: [...], kinds: [...]} from
argv[1]. Writes JSON to argv[2]:
  {"ok": true, "rebuilt": <path>, "bytes": ..., "sha256": ..., "sections": [...],
   "replaced": [...]}
  or {"ok": false, "cause": <str>, "reason": <str>}
"""
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

VISION_WHERE = "Google's text-vision file"
GRAFT_WHERE = "the graft"
SPLICED_WHERE = "the spliced file"


class Refused(Exception):
    """A reason this script will not write the file asked for."""

    def __init__(self, cause, reason):
        super().__init__(reason)
        self.cause = cause


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def text_sections(doc, where, parts, kinds, T):
    """(part, kind) -> the index of that section in `doc`; refused unless each is
    there exactly once."""
    out = {}
    for part in parts:
        for kind in kinds:
            hits = [i for i, s in enumerate(doc.get("section", []))
                    if s.get("section_type") == kind and T.model_type(s) == part]
            if len(hits) != 1:
                raise Refused("input", f"{where} has {len(hits)} {kind} sections for {part}, "
                              "not one")
            out[(part, kind)] = hits[0]
    return out


def described(section):
    """A section's TOML apart from its file name."""
    return {k: v for k, v in section.items() if k != "data_path"}


def read_back(rebuilt, after, vision, vision_doc, vision_places, grafted, graft_places,
              graft_text, parts, kinds, T):
    """Refused unless the packed file is the text-vision file with the graft's text
    sections in it: its TOML the text-vision file's apart from what `pack`
    regenerates, each text section's bytes the graft's, every other section's the
    text-vision file's. Each section's description, digest and source otherwise."""
    if T.comparable(after) != T.comparable(vision_doc):
        raise Refused("readback", f"{SPLICED_WHERE}'s TOML is not {VISION_WHERE}'s apart from "
                      "uuid and creation_timestamp")
    places = T.spans(rebuilt, SPLICED_WHERE)
    if len(places) != len(vision_places):
        raise Refused("readback", f"{SPLICED_WHERE} has {len(places)} sections where "
                      f"{VISION_WHERE} has {len(vision_places)}")
    role = {i: key for key, i in text_sections(after, SPLICED_WHERE, parts, kinds, T).items()}
    sections = []
    for i, place in enumerate(places):
        got = T.digest(rebuilt, place)
        google = T.digest(vision, vision_places[i])
        key = role.get(i)
        if key is None:
            want, source = google, "google_vision"
        else:
            want, source = T.digest(grafted, graft_places[graft_text[key]]), "graft"
        if got != want:
            whose = GRAFT_WHERE if key else VISION_WHERE
            raise Refused("readback", f"section {i} ({T.label(after['section'][i])}) of "
                          f"{SPLICED_WHERE} does not hold the bytes of {whose}")
        sections.append(dict(T.describe(after["section"][i], place), index=i, source=source,
                             sha256=got, changed=got != google))
    return sections


def splice(spec, T):
    work = Path(spec["work"])
    vision, grafted = spec["vision"], spec["grafted"]
    parts, kinds = spec["parts"], spec["kinds"]
    found = sha256(vision)
    if found != spec["vision_sha256"]:
        raise Refused("base_sha256", f"{Path(vision).name} has sha256 {found}, not Google's "
                      f"{spec['vision_sha256']}")
    vision_dir, graft_dir = work / "vision", work / "graft"
    try:
        vision_places = T.spans(vision, VISION_WHERE)
        graft_places = T.spans(grafted, GRAFT_WHERE)
        vision_doc, vision_sources = T.unpacked(vision, vision_dir, VISION_WHERE, vision_places)
        graft_doc, graft_sources = T.unpacked(grafted, graft_dir, GRAFT_WHERE, graft_places)
    except T.Refused as exc:
        raise Refused("input", str(exc)) from None
    vision_text = text_sections(vision_doc, VISION_WHERE, parts, kinds, T)
    graft_text = text_sections(graft_doc, GRAFT_WHERE, parts, kinds, T)
    # What the text-vision file's TOML says about a text section is kept; it must
    # say of the graft's what the graft's own TOML says.
    for (part, kind), i in vision_text.items():
        ours = described(vision_doc["section"][i])
        theirs = described(graft_doc["section"][graft_text[(part, kind)]])
        if ours != theirs:
            raise Refused("input", f"{GRAFT_WHERE}'s {kind} {part} section is described as "
                          f"{theirs}, {VISION_WHERE}'s as {ours}")
    composed = copy.deepcopy(vision_doc)
    sources = list(vision_sources)
    for key, i in vision_text.items():
        j = graft_text[key]
        composed["section"][i]["data_path"] = graft_doc["section"][j]["data_path"]
        sources[i] = graft_sources[j]
    try:
        report = T.rebuild(work, composed, sources, [vision_dir, graft_dir])
        rebuilt = report["rebuilt"]
        after = T.load(work / "after" / "model.toml")
        sections = read_back(rebuilt, after, vision, vision_doc, vision_places, grafted,
                             graft_places, graft_text, parts, kinds, T)
    except T.Refused as exc:
        raise Refused("readback", str(exc)) from None
    return {
        "ok": True, "rebuilt": rebuilt, "bytes": os.path.getsize(rebuilt),
        "sha256": sha256(rebuilt), "vision_sha256": found, "sections": sections,
        "replaced": [T.label(after["section"][s["index"]]) for s in sections
                     if s["source"] == "graft"],
    }


def main(spec):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import towers_script as T  # litetune's towers script, written beside this one

    builder = importlib.metadata.version("litert-lm-builder")
    try:
        result = splice(spec, T)
    except Refused as exc:
        result = {"ok": False, "cause": exc.cause, "reason": str(exc)}
    return dict(result, builder=builder)


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    Path(sys.argv[2]).write_text(json.dumps(main(spec)), encoding="utf-8")
'''


@dataclass(frozen=True)
class SpliceResult:
    """The text-vision file written and what the script recorded writing it."""

    output: Path
    bytes: int
    # `storage.hash_file` of `output`.
    sha256: str
    # The litert-lm-builder version that packed it.
    builder: str
    # Per section of `output`, in order: section_type, model_type, bytes, sha256,
    # `source` ("graft" or "google_vision") and whether it differs from the
    # text-vision file's section at that place.
    sections: tuple[dict[str, Any], ...]
    # The labels ("TFLiteModel embedder", ...) of the sections taken from the graft.
    replaced: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "bytes": self.bytes,
            "sha256": self.sha256,
            "builder": self.builder,
            "replaced": list(self.replaced),
            "sections": [dict(s) for s in self.sections],
        }


def splice(
    env: envs.StageEnv,
    grafted: Path,
    vision_file: Path,
    out_path: Path,
    *,
    events: EventStream | None = None,
    auto_provision: bool = True,
    timeout: int = SPLICE_TIMEOUT_S,
) -> SpliceResult:
    """Write `out_path`: Google's text-vision file with `grafted`'s text sections in it.

    `grafted` is a file `embedding_convert.graft` wrote. Raises
    `EmbeddingConvertError` for a splice it refuses, `FileNotFoundError` for a
    missing input.
    """
    inputs = [Path(p).absolute() for p in (grafted, vision_file)]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"no file at {path}")
    output = Path(out_path).absolute()
    if os.path.lexists(output):
        raise EmbeddingConvertError(
            "input", f"{output} already exists; name a new file -- nothing is written over one"
        )
    if auto_provision:
        env.provision(events=events)
    if not env.ready:
        raise EmbeddingConvertError("run", f"environment {env.name!r} {env.unready_detail}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Beside the output, for the hard link that publishes the result.
    work = Path(tempfile.mkdtemp(prefix=f".{output.stem}-splice-", dir=output.parent))
    try:
        (work / "towers_script.py").write_text(_TOWERS_SCRIPT, encoding="utf-8")
        script = work / "embedding_splice.py"
        script.write_text(_SPLICE_SCRIPT, encoding="utf-8")
        spec_path, result_path = work / "spec.json", work / "result.json"
        graft_path, vision = inputs
        spec_path.write_text(
            json.dumps(
                {
                    "grafted": str(graft_path),
                    "vision": str(vision),
                    "vision_sha256": VISION_SHA256,
                    "work": str(work),
                    "parts": list(TEXT_PARTS),
                    "kinds": list(TEXT_KINDS),
                }
            ),
            encoding="utf-8",
        )
        if events:
            events.note(f"embedding splice: {graft_path.name}'s text tower into {vision.name}")
        report = _run_script(env, script, spec_path, result_path, output, timeout)
        digest = publish_packed(Path(report["rebuilt"]), output, report["sha256"], "the splice")
    finally:
        leaked: list[str] = []
        shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
        if leaked:
            logger.warning("%d scratch file(s) left under %s", len(leaked), work)
    return SpliceResult(
        output=output,
        bytes=output.stat().st_size,
        sha256=digest,
        builder=report["builder"],
        sections=tuple(report["sections"]),
        replaced=tuple(report["replaced"]),
    )


def _run_script(
    env: envs.StageEnv,
    script: Path,
    spec_path: Path,
    result_path: Path,
    output: Path,
    timeout: int,
) -> dict[str, Any]:
    """The script's report when it says ok; `EmbeddingConvertError` otherwise."""
    try:
        proc = env.run(["python", str(script), str(spec_path), str(result_path)], timeout=timeout)
    except subprocess.TimeoutExpired:
        raise EmbeddingConvertError("run", f"the splice did not finish within {timeout}s") from None
    if proc.returncode != 0:
        stderr = proc.stderr or ""
        last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
        logger.error("embedding splice into %s failed:\n%s", output, stderr[-2000:])
        reading = read_returncode(proc.returncode)
        raise EmbeddingConvertError(
            "run", f"the splice script {reading.describe('the splice')}: {last}"
        )
    if not result_path.is_file():
        raise EmbeddingConvertError("run", "the splice script exited 0 without its report")
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.exception("the splice script's report could not be read")
        raise EmbeddingConvertError(
            "run", f"the splice script's report could not be read: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise EmbeddingConvertError(
            "run", f"the splice script reported {type(report).__name__}, not an object"
        )
    if not report.get("ok"):
        raise EmbeddingConvertError(
            str(report.get("cause") or "run"),
            f"{report.get('reason') or 'the splice wrote nothing'}; nothing was written",
        )
    return report
