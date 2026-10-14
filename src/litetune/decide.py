"""`decide`: fit an embedding decider offline, for flutter_edge_ai's `loadDecider`.

flutter_edge_ai 2.4.0 (`@experimental`) gives an embedding model a decider: options as
the centroids of their embedded inputs, optionally a softmax regression over them, and a
threshold on the highest centroid cosine below which it answers "none of these"
(`lib/src/decision/` of DenisovAV/flutter_gemma, branch `feat/decision-engine`, read at
`77f0484e`). On the phone, `createDecider` embeds every example and fits. This command does
the same on a workstation -- with as many examples per option as there are, a threshold
calibrated on held-out rows, and figures measured before anything ships -- and writes the
artifact `toArtifact()` would, which the app loads with `embedder.loadDecider(artifact)`
without embedding anything.

**The fit is the Dart fit**, ported line for line (the script below, function by function,
names its Dart source). Centroids, cross-fit thresholds and temperatures are summed in the
Dart code's order, and the golden tests hold them to what the Dart code itself computed on
the same vectors, bit for bit (`tests/fixtures/decide/`). The softmax regression is the one
place a summation order differs: its two matrix products are BLAS's, so its weights are held
to the tolerance flutter_edge_ai's own fit fingerprints allow between two platforms
(`test/decision/fit_fingerprint.dart`, 1e-9 relative).

**The embedding is the app's**: LiteRT-LM's `EmbeddingEngine` on the CPU, normalised, with
special tokens, and the text prefixed with `TaskType.classification`'s prefix -- for
descriptions, examples and requests alike, as `createDecider` embeds them
(`embedding_decider.dart` L163-L179, L233-L241). It runs in `envs.RUNTIME`, which pins
litert-lm 0.18.0 and numpy.

**The artifact is checked as the app will read it** before it is written: a port of
`loadEmbeddingDecider`'s checks (`embedding_decider.dart` L906-L1114), every
`FormatException` it raises. Its embedding-space id names the installed file the way
`deciderSpaceId` does (L66-L113): an app that installs the file under another name, or
passes no matching `embedderId`, is refused with `DeciderSpaceMismatchException`.

**The figures are the artifact's**: the `--test` rows are scored against the float32
centroids and weights read back from the file written, as a loaded decider scores them
(`EmbeddingDecider._score`, L701-L804), not against the in-memory fit.
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import logging
import math
import shutil
import struct
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from litetune import envs, jsonfile
from litetune._version import __version__
from litetune.evaluate import RUNTIME_ENGINE_SOURCE
from litetune.events import EventStream
from litetune.exits import read_returncode
from litetune.metrics import Proportion, Unavailable
from litetune.storage import hash_bytes, hash_file

logger = logging.getLogger(__name__)

DECIDE_SCHEMA = "litetune.decide/1"

# What the artifact is, as flutter_edge_ai names it (`embedding_decider.dart` L36-L42) and
# where that was read.
SOURCE = "DenisovAV/flutter_gemma feat/decision-engine 77f0484e, flutter_edge_ai 2.4.0"
ARTIFACT_FORMAT = "flutter_edge_ai.decider"
ARTIFACT_VERSION = 1
TASK_TYPE = "classification"
# `TaskType.classification.prefix` (`lib/src/embedding/task_type.dart` L47).
TASK_PREFIX = "task: classification | query: "
# The backend term of the space id: `PreferredBackend.cpu.name`. The CPU is the one backend
# this stage embeds on; another backend's vectors are another space.
BACKEND = "cpu"

METHODS = ("centroid", "linear")
CALIBRATIONS = ("crossFit", "labelled", "minSimilarity", "none")
# `DeciderCalibration.crossFit({keep = 0.95})` and `fromLabelled(..., maxError = 0.05)`
# (`decider_method.dart` L90, L130).
DEFAULT_KEEP = 0.95
DEFAULT_MAX_ERROR = 0.05
DEFAULT_EXAMPLES_IN_ARTIFACT = 30
# `linearWeightDecay` (`decider_fit.dart` L22), the script's `LINEAR_WEIGHT_DECAY`.
LINEAR_WEIGHT_DECAY = 1e-5

# flutter_edge_ai_litertlm sets the engine's max input length to the smaller of 1024 and the
# bundle's largest text signature (`litert_lm_embedding_common.dart` L43,
# `litert_lm_embedding_pass.dart` L1088-L1102).
MAX_INPUT_LENGTH = 1024
# The loader's slack on a stored centroid's norm (`embedding_decider.dart` L1073-L1080).
CENTROID_NORM_SLACK = 1e-3

ARTIFACT_NAME = "decider.json"
REPORT_NAME = "decide.json"

# Embedding is one request per input, as the app makes them; tens of thousands of rows on a
# CPU take a while, and the fit after them seconds to minutes.
DECIDE_TIMEOUT_S = 4 * 3600


class DecideError(Exception):
    """A request `decide` will not carry out, or a decider it would not write."""


class ArtifactError(DecideError):
    """An artifact flutter_edge_ai's `loadDecider` would refuse, with the reason it gives."""


# ---------------------------------------------------------------------------
# The embedding-space id
# ---------------------------------------------------------------------------


def escape(part: str) -> str:
    """`_escape` (`embedding_decider.dart` L123-L137): a control character, `%`, `+`, `;`
    or `@` as `%` and two uppercase hex digits, so two identities never spell one id.

    Dart walks UTF-16 code units and Python code points; the characters escaped are all
    single units, and every other one is written as it is either way.
    """
    out = []
    for char in part:
        unit = ord(char)
        if unit <= 0x1F or char in "%+;@":
            out.append(f"%{unit:02X}")
        else:
            out.append(char)
    return "".join(out)


def identity(*, install_name: str | None, size: int | None, embedder_id: str | None) -> str:
    """The `model=` term of `deciderSpaceId` (L71-L94): the escaped `embedder_id` when the app
    passes one, otherwise the installed file as `<escaped name>@<bytes>`."""
    if embedder_id is not None:
        return escape(embedder_id)
    if install_name is None or size is None:
        raise ValueError("an identity needs the install name and size, or an embedder id")
    return f"{escape(install_name)}@{size}"


def space_id_parts(model: str) -> tuple[str, str]:
    """What `deciderSpaceId` (L109-L112) puts before and after the dimension, for a CPU
    embedder and a decider fitted on text alone, which carries no `vision=` term."""
    return f"model={model};dim=", f";backend={BACKEND};prefix={TASK_PREFIX}"


def space_id(model: str, dimension: int) -> str:
    head, tail = space_id_parts(model)
    return f"{head}{dimension}{tail}"


# ---------------------------------------------------------------------------
# Reading the inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One row of a JSONL split: its text and its option id (None: out of scope)."""

    source: Path
    line: int
    text: str
    label: str | None

    @property
    def name(self) -> str:
        return f"{self.source.name} line {self.line}"


@dataclass(frozen=True)
class OptionSpec:
    """An option as `--options` gives it."""

    id: str
    description: str | None = None
    metadata: dict[str, Any] | None = None


# A row without a "label" key, which a test row's explicit null must not be taken for.
_NO_LABEL = object()


def _refuse_constant(name: str) -> Any:
    # Dart's `jsonDecode` has no NaN or Infinity, and Python's `json` reads both.
    raise ValueError(f"{name} is not JSON")


def _text_of(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DecideError(f"could not read {path}: {exc}") from None
    try:
        return jsonfile.decode(raw)
    except UnicodeDecodeError as exc:
        raise DecideError(f"{path} {jsonfile.UNREADABLE} ({exc})") from None


def read_rows(path: Path, *, out_of_scope: bool = False) -> tuple[Row, ...]:
    """The rows of a JSONL split: `{"text": str, "label": str}` each, or with `out_of_scope`
    also `"label": null`. Raises `DecideError` naming `path` and the line."""
    rows: list[Row] = []
    # Split on newlines alone: `str.splitlines` also splits on U+2028, which JSON allows
    # unescaped inside a string.
    for number, line in enumerate(_text_of(path).split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_refuse_constant)
        except ValueError as exc:
            raise DecideError(f"{path}:{number}: not valid JSON: {exc}") from None
        if not isinstance(obj, dict):
            raise DecideError(f"{path}:{number}: expected an object with 'text' and 'label'")
        text, label = obj.get("text"), obj.get("label", _NO_LABEL)
        if not isinstance(text, str) or not text.strip():
            raise DecideError(f"{path}:{number}: 'text' must be a non-empty string")
        if label is None and out_of_scope:
            rows.append(Row(path, number, text, None))
            continue
        if not isinstance(label, str) or not label:
            wanted = "an option id"
            if out_of_scope:
                wanted += ", or null for an out-of-scope row"
            raise DecideError(f"{path}:{number}: 'label' must be {wanted}")
        rows.append(Row(path, number, text, label))
    if not rows:
        raise DecideError(f"{path}: no rows")
    return tuple(rows)


def read_options(path: Path) -> tuple[OptionSpec, ...]:
    """`--options`: a JSON array of `{"id", "description"?, "metadata"?}`."""
    try:
        data = json.loads(_text_of(path), parse_constant=_refuse_constant)
    except ValueError as exc:
        raise DecideError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(data, list):
        raise DecideError(f"{path}: expected a JSON array of options")
    options = []
    for index, entry in enumerate(data):
        where = f"{path}: option {index}"
        if not isinstance(entry, dict):
            raise DecideError(f"{where} is not an object")
        unknown = sorted(set(entry) - {"id", "description", "metadata"})
        if unknown:
            # An `examples` key here would otherwise be read as nothing at all.
            raise DecideError(
                f"{where} has {unknown}; an option takes id, description and metadata, and "
                "its examples are the --train rows labelled with its id"
            )
        option_id, description, metadata = (
            entry.get("id"),
            entry.get("description"),
            entry.get("metadata"),
        )
        if not isinstance(option_id, str):
            raise DecideError(f"{where}: 'id' must be a string")
        if description is not None and not isinstance(description, str):
            raise DecideError(f"{where}: 'description' must be a string")
        if metadata is not None and not isinstance(metadata, dict):
            raise DecideError(f"{where}: 'metadata' must be an object")
        options.append(OptionSpec(option_id, description, metadata))
    return tuple(options)


# ---------------------------------------------------------------------------
# The request, and the refusals flutter_edge_ai would make
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecideRequest:
    train: Path
    embedder: Path
    output_dir: Path
    options: Path | None = None
    calib: Path | None = None
    test: Path | None = None
    embedder_name: str | None = None
    embedder_id: str | None = None
    method: str = "centroid"
    calibration: str = "crossFit"
    keep: float | None = None
    max_error: float | None = None
    min_similarity: float | None = None
    examples_in_artifact: int = DEFAULT_EXAMPLES_IN_ARTIFACT
    timeout_s: int = DECIDE_TIMEOUT_S
    auto_provision: bool = True


@dataclass(frozen=True)
class Plan:
    """What the script embeds and fits, in `createDecider`'s order."""

    options: tuple[OptionSpec, ...]
    train: tuple[Row, ...]
    calib: tuple[Row, ...]
    test: tuple[Row, ...]
    # The calibration as the artifact stores it: kind, then keep | maxError | threshold.
    calibration: dict[str, Any]
    # Each option's first `examples_in_artifact` train texts, in data order.
    stored_examples: tuple[tuple[str, ...], ...]

    @property
    def index(self) -> dict[str, int]:
        return {option.id: o for o, option in enumerate(self.options)}

    def inputs(self) -> dict[str, list[Any]]:
        """The option inputs, option by option: the description first, then every train row
        of the option in data order (`embedding_decider.dart` L163-L179). `rank` is the
        position within the option, which the cross-fit's folds are taken from."""
        index = self.index
        by_option: list[list[Row]] = [[] for _ in self.options]
        for row in self.train:
            by_option[index[str(row.label)]].append(row)
        texts: list[str] = []
        options: list[int] = []
        ranks: list[int] = []
        names: list[str] = []
        for o, option in enumerate(self.options):
            rank = 0
            if option.description is not None:
                texts.append(option.description)
                options.append(o)
                ranks.append(rank)
                names.append(f'option "{option.id}", its description')
                rank += 1
            for row in by_option[o]:
                texts.append(row.text)
                options.append(o)
                ranks.append(rank)
                names.append(row.name)
                rank += 1
        return {"texts": texts, "options": options, "ranks": ranks, "names": names}

    def rows(self, rows: Sequence[Row]) -> dict[str, list[Any]]:
        index = self.index
        return {
            "texts": [row.text for row in rows],
            "options": [None if row.label is None else index[row.label] for row in rows],
            "names": [row.name for row in rows],
        }


def _calibration(request: DecideRequest) -> dict[str, Any]:
    """The calibration block, refusing a flag the calibration does not read."""
    kind = request.calibration
    if kind not in CALIBRATIONS:
        raise DecideError(f"--calibration must be one of {list(CALIBRATIONS)}, got {kind!r}")
    stray = [
        flag
        for flag, value, owner in (
            ("--keep", request.keep, "crossFit"),
            ("--max-error", request.max_error, "labelled"),
            ("--calib", request.calib, "labelled"),
            ("--min-similarity", request.min_similarity, "minSimilarity"),
        )
        if value is not None and kind != owner
    ]
    if stray:
        raise DecideError(
            f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} not read by "
            f"--calibration {kind}; leave {'it' if len(stray) == 1 else 'them'} out"
        )
    if kind == "crossFit":
        keep = DEFAULT_KEEP if request.keep is None else request.keep
        # `_validateOptions` (`embedding_decider.dart` L347-L355).
        if not (0 < keep <= 1):
            raise DecideError(f"--keep must be in (0, 1], got {keep}")
        return {"kind": kind, "keep": keep}
    if kind == "labelled":
        max_error = DEFAULT_MAX_ERROR if request.max_error is None else request.max_error
        if not (0 <= max_error < 1):
            raise DecideError(f"--max-error must be in [0, 1), got {max_error}")
        if request.calib is None:
            raise DecideError(
                "--calibration labelled calibrates on held-out labelled rows: give them as "
                "--calib"
            )
        return {"kind": kind, "maxError": max_error}
    if kind == "minSimilarity":
        if request.min_similarity is None:
            raise DecideError("--calibration minSimilarity needs --min-similarity")
        if not (-1 <= request.min_similarity <= 1):
            raise DecideError(
                f"--min-similarity is a cosine and must be in [-1, 1], got {request.min_similarity}"
            )
        return {"kind": kind, "threshold": request.min_similarity}
    return {"kind": kind}


def plan(request: DecideRequest) -> Plan:
    """Read every input and refuse, by name, what `createDecider` would refuse
    (`embedding_decider.dart` L188-L224, L312-L398; `decider_option.dart` L25-L42)."""
    if request.method not in METHODS:
        raise DecideError(f"--method must be one of {list(METHODS)}, got {request.method!r}")
    if request.examples_in_artifact < 0:
        raise DecideError("--examples-in-artifact must be 0 or more")
    calibration = _calibration(request)
    train = read_rows(request.train)

    if request.options is None:
        options = tuple(OptionSpec(label) for label in dict.fromkeys(str(r.label) for r in train))
    else:
        options = read_options(request.options)
    if not options:
        raise DecideError(f"{request.options}: no options")
    seen: set[str] = set()
    for option in options:
        if not option.id:
            raise DecideError("an option id must not be empty")
        if option.id in seen:
            raise DecideError(f'two options share the id "{option.id}"')
        seen.add(option.id)
        if option.description is not None and not option.description:
            raise DecideError(
                f'option "{option.id}": an empty description must be left out instead'
            )

    counts = Counter(str(row.label) for row in train)
    stray = [row for row in train if row.label not in seen]
    if stray:
        raise DecideError(
            f"{stray[0].name}: label {stray[0].label!r} names no option in {request.options}"
            + (f" ({len(stray)} rows do)" if len(stray) > 1 else "")
        )
    for option in options:
        inputs = counts[option.id] + (option.description is not None)
        if inputs == 0:
            raise DecideError(
                f'option "{option.id}" has neither a description nor a --train row: nothing '
                "to embed"
            )
        if calibration["kind"] == "crossFit" and inputs < 2:
            raise DecideError(
                "--calibration crossFit holds each input out against the others, so every "
                f"option needs at least two inputs (description plus train rows); option "
                f'"{option.id}" has {inputs}. Add rows, or calibrate with --calibration '
                "minSimilarity, labelled or none"
            )
    if request.method == "linear" and len(options) < 2:
        raise DecideError("--method linear needs at least two options")

    calib: tuple[Row, ...] = ()
    if request.calib is not None:
        calib = read_rows(request.calib)
        option_texts = {r.text for r in train} | {
            o.description for o in options if o.description is not None
        }
        for row in calib:
            if row.label not in seen:
                raise DecideError(f"{row.name}: label {row.label!r} names no option")
            if row.text in option_texts:
                raise DecideError(
                    f"{row.name} repeats a --train row or a description; held-out rows must "
                    "be new requests, or the threshold is fitted on data the decider already "
                    "knows"
                )

    test: tuple[Row, ...] = ()
    if request.test is not None:
        test = read_rows(request.test, out_of_scope=True)
        for row in test:
            if row.label is not None and row.label not in seen:
                raise DecideError(f"{row.name}: label {row.label!r} names no option")

    stored: list[list[str]] = [[] for _ in options]
    index = {option.id: o for o, option in enumerate(options)}
    for row in train:
        texts = stored[index[str(row.label)]]
        if len(texts) < request.examples_in_artifact:
            texts.append(row.text)
    return Plan(
        options=options,
        train=train,
        calib=calib,
        test=test,
        calibration=calibration,
        stored_examples=tuple(tuple(t) for t in stored),
    )


# ---------------------------------------------------------------------------
# The loader's checks
# ---------------------------------------------------------------------------


def _is_int(value: object) -> bool:
    # A JSON `true` is a Python int, and not a Dart one.
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _type_name(value: object) -> str:
    """The Dart runtime type a JSON value decodes to, as the loader's messages name it."""
    if value is None:
        return "Null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "String"
    if isinstance(value, list):
        return "List<dynamic>"
    return "_Map<String, dynamic>"


def _field(mapping: Mapping[str, Any], key: str, kind: str) -> Any:
    """`_field<T>` (L1116-L1124): `kind` is `String`, `int`, `Map` or `List`."""
    value = mapping.get(key)
    ok = {
        "String": isinstance(value, str),
        "int": _is_int(value),
        "Map": isinstance(value, dict),
        "List": isinstance(value, list),
    }[kind]
    if not ok:
        wanted = {"Map": "Map<String, Object?>", "List": "List<Object?>"}.get(kind, kind)
        raise ArtifactError(
            f'Decider artifact field "{key}" is {_type_name(value)}, expected {wanted}.'
        )
    return value


def _optional_double(mapping: Mapping[str, Any], key: str) -> float | None:
    """`_optionalDouble` (L1126-L1133)."""
    value = mapping.get(key)
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(value):
        raise ArtifactError(f'Decider artifact field "{key}" is not a number.')
    return float(value)


def _decode_float32(encoded: object, dimension: int, option: int) -> list[float]:
    """`_decodeFloat32` (L1135-L1161)."""
    if not isinstance(encoded, str):
        raise ArtifactError(f"Option {option} has no encoded vector.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ArtifactError(f"Option {option}: a vector is not valid base64.") from None
    if len(raw) != dimension * 4:
        raise ArtifactError(
            f"Option {option}: a vector has {len(raw) // 4} values, expected {dimension}."
        )
    values = list(struct.unpack(f"<{dimension}f", raw))
    if not all(math.isfinite(v) for v in values):
        raise ArtifactError(f"Option {option}: a vector value is not finite.")
    return values


def check_artifact(artifact: object, *, expected_space_id: str, dimension: int) -> None:
    """Raise `ArtifactError` where `loadEmbeddingDecider` (L906-L1114) would refuse
    `artifact` on an embedder whose space id is `expected_space_id` and whose dimension is
    `dimension`. The checks and their messages are the loader's, in its order; a space id
    that differs is its `DeciderSpaceMismatchException`."""
    if not isinstance(artifact, dict):
        raise ArtifactError("Not a decider artifact: it is not a JSON object.")
    if artifact.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError(
            f"Not a decider artifact: format is {artifact.get('format')}, expected "
            f"{ARTIFACT_FORMAT}."
        )
    version = artifact.get("version")
    # Dart's `1.0 != 1` is false, and `true != 1` is true.
    if not (_is_number(version) and version == ARTIFACT_VERSION):
        raise ArtifactError(
            f"Decider artifact version {version}; this release reads version "
            f"{ARTIFACT_VERSION} only."
        )
    if artifact.get("kind") != "embedding":
        raise ArtifactError(
            f'A "{artifact.get("kind")}" decider artifact cannot be loaded by an EmbeddingModel.'
        )
    task = _field(artifact, "taskType", "String")
    if task != TASK_TYPE:
        raise ArtifactError(
            f'Decider artifact taskType "{task}": this release embeds options and inputs with '
            f'"{TASK_TYPE}" only.'
        )
    expected = _field(artifact, "spaceId", "String")
    images = artifact.get("imageInputs")
    if images is not None and not isinstance(images, bool):
        raise ArtifactError("Decider artifact imageInputs is not a bool.")
    if expected != expected_space_id:
        raise ArtifactError(
            "The decider was fitted in another embedding space: its vectors do not compare "
            f"with this embedder's (DeciderSpaceMismatchException: expected {expected}, actual "
            f"{expected_space_id})."
        )
    d = _field(artifact, "dimension", "int")
    if d != dimension:
        raise ArtifactError(f"Decider artifact dimension {d} does not match its space id.")
    method = _field(artifact, "method", "String")
    if method not in METHODS:
        raise ArtifactError(f'Unknown decider method "{method}".')
    calibration = _field(artifact, "calibration", "Map")
    kind = _field(calibration, "kind", "String")
    if kind not in CALIBRATIONS:
        raise ArtifactError(f'Unknown calibration kind "{kind}".')
    threshold = _optional_double(calibration, "threshold")
    temperature = _optional_double(calibration, "temperature")
    if kind != "none" and threshold is None:
        raise ArtifactError(f"A {kind} calibration needs a threshold.")
    if kind == "none" and threshold is not None:
        raise ArtifactError("A none calibration never abstains, so it has no threshold.")
    if kind == "labelled" and temperature is None:
        raise ArtifactError("A labelled calibration needs its temperature.")
    if kind != "labelled" and temperature is not None:
        raise ArtifactError(f"Only a labelled calibration has a temperature; this one is {kind}.")
    if temperature is not None and not temperature > 0:
        raise ArtifactError("A calibration temperature must be positive.")
    linear = method == "linear"

    options = _field(artifact, "options", "List")
    if not options:
        raise ArtifactError("A decider artifact needs options.")
    if linear and len(options) < 2:
        raise ArtifactError("A linear decider artifact needs at least two options.")
    ids = []
    for o, raw in enumerate(options):
        if not isinstance(raw, dict):
            raise ArtifactError(f"Option {o} of the decider artifact is not a map.")
        option_id = raw.get("id")
        if not isinstance(option_id, str):
            raise ArtifactError(f'Option {o}: "id" is {_type_name(option_id)}, not text.')
        description = raw.get("description")
        if description is not None and not isinstance(description, str):
            raise ArtifactError(
                f'Option {o}: "description" is {_type_name(description)}, not text.'
            )
        examples = raw.get("examples")
        if not isinstance(examples, list):
            raise ArtifactError(f'Option {o}: "examples" is {_type_name(examples)}, not a list.')
        for text in examples:
            if not isinstance(text, str):
                raise ArtifactError(f'Option {o}: "examples" holds a {_type_name(text)}, not text.')
        metadata = raw.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ArtifactError(f'Option {o}: "metadata" is not a map.')
        # `DeciderOption`'s own ArgumentErrors, as the loader rewraps them (L1051-L1062).
        if not option_id:
            raise ArtifactError(f"Option {o} of the decider artifact: must not be empty")
        if description is not None and not description:
            raise ArtifactError(
                f"Option {o} of the decider artifact: must not be empty; leave it out instead"
            )
        media = raw.get("mediaExamples", 0)
        if media is None:
            media = 0
        if not _is_int(media) or media < 0:
            raise ArtifactError(f"Option {o}: mediaExamples is not a count.")
        centroid = _decode_float32(raw.get("centroid"), d, o)
        norm = 0.0
        for v in centroid:
            norm += v * v
        if abs(math.sqrt(norm) - 1) > CENTROID_NORM_SLACK:
            raise ArtifactError(
                f"Option {o}: its centroid is not unit length (norm {math.sqrt(norm)})."
            )
        if linear:
            _decode_float32(raw.get("weights"), d, o)
            bias = raw.get("bias")
            if not _is_number(bias) or not math.isfinite(bias):
                raise ArtifactError(f"Option {o} has no finite bias.")
        ids.append(option_id)
    if len(set(ids)) != len(ids):
        raise ArtifactError("Two options of the artifact share an id.")


# ---------------------------------------------------------------------------
# The script: the fit, ported from decider_fit.dart and softmax_regression.dart
# ---------------------------------------------------------------------------

# Runs in `envs.RUNTIME`. Each function names the Dart it ports; `fitDecider` and the
# functions it calls are at `lib/src/decision/decider_fit.dart`, the regression and its
# optimiser at `lib/src/decision/softmax_regression.dart` (77f0484e).
_FIT_SOURCE = r'''
"""Fit a decider and write its artifact, as flutter_edge_ai's `createDecider` and
`toArtifact` would; then score held-out rows against the artifact as written.

Reads JSON from argv[1] (see `run`), writes JSON to argv[2].
"""
import base64
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# decider_fit.dart L22.
LINEAR_WEIGHT_DECAY = 1e-5
# decider_fit.dart L421-L423.
GRID_LOW, GRID_HIGH, GRID_STEPS = 1e-3, 10.0, 241
# keepThreshold's nudge (decider_fit.dart L382-L389): `(1 - 0.9) * 10` is
# 0.9999999999999998, which floors to 0 without it.
KEEP_NUDGE = 1e-9
# errorBudgetThreshold's "abstain on everything" (decider_fit.dart L494).
NO_THRESHOLD = 1.0 + 1e-9
# fitTemperature's probability floor (decider_fit.dart L446).
PROBABILITY_FLOOR = 1e-12
# The cross-fit's folds (decider_fit.dart L355-L358).
FOLDS = 5


class Refused(Exception):
    """A decider the Dart fit refuses to make, with its reason."""


def seq_sum(values):
    """`s = 0.0; for v in values: s += v`: the Dart code's order. numpy's `sum` and `dot`
    pair their terms, which is a different number; `cumsum` adds one term after another,
    and `0.0 +` makes its -0.0 the +0.0 Dart's sum starts from."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return 0.0 + float(np.cumsum(values)[-1])


def seq_row_sums(matrix):
    """`seq_sum` of each row of `matrix`."""
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0])
    return 0.0 + np.cumsum(matrix, axis=1)[:, -1]


def seq_dot(a, b):
    return seq_sum(a * b)


def kernel_products(rows, matrix):
    """`rows @ matrix.T`, each entry `x[0]*m[0] + x[1]*m[1] + ...` from 0.0, in that order.

    `DenseKernel.multiply` (dense_kernel.dart L37-L59) and `_dot` (decider_fit.dart
    L545-L551, embedding_decider.dart L889-L895) add each product, formed on its own, to an
    accumulator that starts at 0.0; Dart never fuses the two (dense_kernel.dart L1-L6), and
    neither do two numpy operations.
    """
    out = np.zeros((rows.shape[0], matrix.shape[0]))
    for j in range(rows.shape[1]):
        out += rows[:, j : j + 1] * matrix[:, j][None, :]
    return out


def unit_row(vector, dimension, what):
    """`unitRowInto` (decider_embedding.dart L129-L159)."""
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (dimension,):
        raise Refused(
            f"The embedder returned {v.size} values for {what}; its dimension is {dimension}."
        )
    if not np.all(np.isfinite(v)):
        raise Refused(
            f"The embedder returned a non-finite value for {what}; a decision is never "
            "computed from it."
        )
    norm = seq_sum(v * v)
    if norm == 0:
        raise Refused(f"The embedder returned a zero vector for {what}.")
    return v / math.sqrt(norm)


def option_centroids(x, labels, count, include=None, ids=None):
    """`optionCentroids` (decider_fit.dart L292-L335): each option's mean input,
    renormalised; rows added in order."""
    sums = np.zeros((count, x.shape[1]))
    counts = [0] * count
    for i, o in enumerate(labels):
        if include is not None and not include[i]:
            continue
        counts[o] += 1
        sums[o] += x[i]
    for o in range(count):
        if counts[o] == 0:
            raise Refused(f"option {o} has no input to take a centroid of")
        mean = sums[o] / counts[o]
        norm = seq_sum(mean * mean)
        if norm == 0:
            name = f"option {o}" if ids is None else f'option "{ids[o]}"'
            raise Refused(
                f"The inputs of {name} cancel out: their mean is the zero vector, so the "
                "option has no direction to compare a request with. Give it inputs that mean "
                "the same thing."
            )
        sums[o] = mean / math.sqrt(norm)
    return sums


def cross_fit_signal(x, labels, ranks, count, ids=None):
    """`crossFitSignal` (decider_fit.dart L340-L377): each input's best cosine to the
    centroids of the other folds; an input's fold is its rank within its option, mod 5."""
    folds = np.asarray(ranks) % FOLDS
    out = np.zeros(len(labels))
    for f in range(FOLDS):
        held = np.nonzero(folds == f)[0]
        if held.size == 0:
            continue
        centroids = option_centroids(x, labels, count, include=folds != f, ids=ids)
        out[held] = kernel_products(x[held], centroids).max(axis=1)
    return out


def dart_order(value):
    """A sort key that puts -0.0 before 0.0, as Dart's `double.compareTo` does."""
    return (value, math.copysign(1.0, value))


def keep_threshold(signal, keep):
    """`keepThreshold` (decider_fit.dart L386-L393)."""
    ordered = sorted((float(v) for v in signal), key=dart_order)
    index = min(math.floor((1 - keep) * len(ordered) + KEEP_NUDGE), len(ordered) - 1)
    return ordered[index]


def error_budget_threshold(signal, ok, budget):
    """`errorBudgetThreshold` (decider_fit.dart L480-L495): the lowest signal at which the
    rows answered -- those at or above it, rows of one signal together -- keep their error
    rate within `budget`; `1 + 1e-9` when none does."""
    values = [float(v) for v in signal]
    order = sorted(range(len(values)), key=lambda i: dart_order(values[i]), reverse=True)
    errors = 0
    last = None
    k = 0
    while k < len(order):
        value = values[order[k]]
        while k < len(order) and values[order[k]] == value:
            if not ok[order[k]]:
                errors += 1
            k += 1
        if errors / k <= budget:
            last = value
    return NO_THRESHOLD if last is None else last


def fit_temperature(scores, labels):
    """`fitTemperature` (decider_fit.dart L407-L469): of 241 log-spaced temperatures in
    [1e-3, 10], the first with the lowest mean negative log-likelihood of `labels` under
    softmax(scores / T); refused at either end of the grid."""
    if not np.all(np.isfinite(scores)):
        raise Refused(
            "A labelled row has a non-finite score; the temperature is never fitted from it."
        )
    lo, hi = math.log(GRID_LOW), math.log(GRID_HIGH)
    step = (hi - lo) / (GRID_STEPS - 1)
    labels = np.asarray(labels)
    rows = np.arange(len(labels))
    best_nll, best_step, best_temp = math.inf, -1, math.nan
    for s in range(GRID_STEPS):
        temp = math.exp(hi if s == GRID_STEPS - 1 else s * step + lo)
        v = scores / temp
        e = np.exp(v - v.max(axis=1, keepdims=True))
        q = e[rows, labels] / seq_row_sums(e)
        nll = seq_sum(-np.log(np.where(q > PROBABILITY_FLOOR, q, PROBABILITY_FLOOR))) / len(rows)
        if nll < best_nll:
            best_nll, best_step, best_temp = nll, s, temp
    if best_step in (0, GRID_STEPS - 1):
        end = (
            "1e-3: every row is answered right with room to spare, so a near-tie would be "
            "reported as certain"
            if best_step == 0
            else "10: the rows are answered so poorly that a flat distribution fits best"
        )
        raise Refused(
            "the --calib rows cannot calibrate probabilities: the best softmax temperature is "
            f"the end of the grid ({end}). Use more held-out requests, including ones the "
            "decider gets wrong or nearly wrong, or calibrate the threshold alone with "
            "--calibration crossFit or minSimilarity"
        )
    return best_temp


class SoftmaxObjective:
    """`SoftmaxObjective` (softmax_regression.dart L92-L205): the mean cross-entropy of
    softmax(x W + b) plus `decay * ||W||^2`, and its gradient. Parameters are flat: W
    (d x C, row-major), then b.

    `x @ w` and `x.T @ r` are BLAS's, whose summation order is not Dart's: the one place
    this fit is held to a tolerance rather than to the bit. Every other sum is in Dart's
    order.
    """

    def __init__(self, x, labels, classes, decay):
        self.x = x
        self.labels = np.asarray(labels)
        self.n = len(self.labels)
        self.d = x.shape[1]
        self.c = classes
        self.decay = decay
        self.rows = np.arange(self.n)

    @property
    def parameter_count(self):
        return self.d * self.c + self.c

    def evaluate(self, theta, grad):
        d, c, n = self.d, self.c, self.n
        w = theta[: d * c].reshape(d, c)
        z = self.x @ w + theta[d * c :]
        m = z.max(axis=1)
        e = np.exp(z - m[:, None])
        total = seq_row_sums(e)
        loss = seq_sum(m + np.log(total) - z[self.rows, self.labels])
        r = e / total[:, None]
        r[self.rows, self.labels] -= 1.0
        r /= n
        penalty = seq_sum(w * w)
        grad[: d * c] = (self.x.T @ r + 2 * self.decay * w).ravel()
        grad[d * c :] = 0.0 + np.cumsum(r, axis=0)[-1]
        return loss / n + self.decay * penalty


def max_abs(v):
    """torch's `v.abs().max()`, NaN when any entry is (softmax_regression.dart L626-L637)."""
    return float(np.max(np.abs(v)))


def py_max(a, b):
    """softmax_regression.dart L603."""
    return b if b > a else a


def py_min(a, b):
    """softmax_regression.dart L606."""
    return b if b < a else a


def dart_min(a, b):
    """Dart's `math.min` on doubles, NaN when either is (Python's `min` keeps `a`)."""
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a <= b else b


def cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """`_cubicInterpolate` (softmax_regression.dart L566-L600)."""
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    elif x1 <= x2:
        xmin_bound, xmax_bound = x1, x2
    else:
        xmin_bound, xmax_bound = x2, x1
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 * d1 - g1 * g2
    if d2_square >= 0:
        d2 = math.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return py_min(py_max(min_pos, xmin_bound), xmax_bound)
    return (xmin_bound + xmax_bound) / 2.0


def directional(evaluate, x, t, d):
    """`_directional` (softmax_regression.dart L400-L415): the objective at x + t d."""
    g = np.zeros(x.size)
    return evaluate(x + t * d, g), g


def strong_wolfe(evaluate, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25):
    """`_strongWolfe` (softmax_regression.dart L419-L563), torch's `_strong_wolfe`."""
    d_norm = max_abs(d)
    f_new, g_new = directional(evaluate, x, t, d)
    ls_func_evals = 1
    gtd_new = seq_dot(g_new, d)

    t_prev, f_prev, g_prev, gtd_prev = 0.0, f, g, gtd
    done = False
    ls_iter = 0
    bracket, bracket_f, bracket_g, bracket_gtd = [], [], [], []
    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket, bracket_f = [t_prev, t], [f_prev, f_new]
            bracket_g, bracket_gtd = [g_prev, g_new], [gtd_prev, gtd_new]
            break
        if abs(gtd_new) <= -c2 * gtd:
            bracket, bracket_f, bracket_g = [t], [f_new], [g_new]
            done = True
            break
        if gtd_new >= 0:
            bracket, bracket_f = [t_prev, t], [f_prev, f_new]
            bracket_g, bracket_gtd = [g_prev, g_new], [gtd_prev, gtd_new]
            break
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )
        t_prev, f_prev, g_prev, gtd_prev = tmp, f_new, g_new, gtd_new
        f_new, g_new = directional(evaluate, x, t, d)
        ls_func_evals += 1
        gtd_new = seq_dot(g_new, d)
        ls_iter += 1

    if ls_iter == max_ls:
        bracket, bracket_f, bracket_g = [0.0, t], [f, f_new], [g, g_new]

    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break
        t = cubic_interpolate(
            bracket[0], bracket_f[0], bracket_gtd[0], bracket[1], bracket_f[1], bracket_gtd[1]
        )
        hi = py_max(bracket[0], bracket[1])
        lo = py_min(bracket[0], bracket[1])
        eps = 0.1 * (hi - lo)
        if py_min(hi - t, t - lo) < eps:
            if insuf_progress or t >= hi or t <= lo:
                t = hi - eps if abs(t - hi) < abs(t - lo) else lo + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = directional(evaluate, x, t, d)
        ls_func_evals += 1
        gtd_new = seq_dot(g_new, d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos], bracket_f[high_pos] = t, f_new
            bracket_g[high_pos], bracket_gtd[high_pos] = g_new, gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos], bracket_f[high_pos] = bracket[low_pos], bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
            bracket[low_pos], bracket_f[low_pos] = t, f_new
            bracket_g[low_pos], bracket_gtd[low_pos] = g_new, gtd_new

    return bracket_f[low_pos], bracket_g[low_pos], bracket[low_pos], ls_func_evals


def minimize_lbfgs(
    evaluate,
    x,
    lr=1.0,
    max_iter=500,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=20,
):
    """`minimizeLbfgs` (softmax_regression.dart L223-L341): torch's L-BFGS with the strong
    Wolfe line search, minimising `evaluate` from `x` in place. The history is a list of at
    most `history_size` pairs, oldest first, as `_History`'s ring reads (L346-L394). Returns
    (loss, iterations, evaluations)."""
    max_evals = max_eval if max_eval is not None else max_iter * 5 // 4
    flat_grad = np.zeros(x.size)
    loss = evaluate(x, flat_grad)
    current_evals = 1
    if max_abs(flat_grad) <= tolerance_grad:
        return loss, 0, current_evals

    d = np.zeros(x.size)
    t = 0.0
    dirs, steps, ros = [], [], []
    h_diag = 1.0
    prev_flat_grad = np.zeros(x.size)
    prev_loss = loss
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        if n_iter == 1:
            d = flat_grad * -1
            h_diag = 1.0
        else:
            y = flat_grad - prev_flat_grad
            s = d * t
            ys = seq_dot(y, s)
            if ys > 1e-10:
                if len(dirs) == history_size:
                    dirs.pop(0)
                    steps.pop(0)
                    ros.pop(0)
                dirs.append(y)
                steps.append(s)
                ros.append(1.0 / ys)
                h_diag = ys / seq_dot(y, y)
            d = flat_grad * -1
            al = [0.0] * len(dirs)
            for i in range(len(dirs) - 1, -1, -1):
                al[i] = seq_dot(steps[i], d) * ros[i]
                d += -al[i] * dirs[i]
            d = d * h_diag
            for i in range(len(dirs)):
                be_i = seq_dot(dirs[i], d) * ros[i]
                d += (al[i] - be_i) * steps[i]
        prev_flat_grad = flat_grad.copy()
        prev_loss = loss

        if n_iter == 1:
            t = dart_min(1.0, 1.0 / seq_sum(np.abs(flat_grad))) * lr
        else:
            t = lr

        gtd = seq_dot(flat_grad, d)
        if gtd > -tolerance_change:
            break

        loss, flat_grad, t, ls_evals = strong_wolfe(
            evaluate,
            x,
            t,
            d,
            loss,
            flat_grad,
            gtd,
            tolerance_change=tolerance_change,
            max_ls=max_evals - current_evals,
        )
        x += t * d
        opt_cond = max_abs(flat_grad) <= tolerance_grad
        current_evals += ls_evals

        if n_iter == max_iter:
            break
        if current_evals >= max_evals:
            break
        if opt_cond:
            break
        if max_abs(d) * abs(t) <= tolerance_change:
            break
        if abs(loss - prev_loss) < tolerance_change:
            break
    return loss, n_iter, current_evals


def fit_softmax_regression(x, labels, classes, decay=LINEAR_WEIGHT_DECAY):
    """`fitSoftmaxRegression` (softmax_regression.dart L53-L88): from all zeros."""
    objective = SoftmaxObjective(x, labels, classes, decay)
    theta = np.zeros(objective.parameter_count)
    loss, iterations, evaluations = minimize_lbfgs(objective.evaluate, theta)
    split = objective.d * classes
    return {
        "weights": theta[:split].reshape(objective.d, classes),
        "bias": theta[split:].copy(),
        "loss": loss,
        "iterations": iterations,
        "evaluations": evaluations,
    }


def check_fitted(centroids, weights=None, bias=None, loss=None, threshold=None, temperature=None):
    """`checkFittedParameters` (decider_fit.dart L239-L285)."""

    def finite(values, what):
        if values is not None and not np.all(np.isfinite(values)):
            raise Refused(
                f"The decider fit produced a non-finite {what}; a decision is never computed "
                "from it."
            )

    finite(centroids, "centroid")
    norms = seq_row_sums(centroids.astype(np.float64) ** 2)
    for o, norm in enumerate(norms):
        if norm == 0:
            raise Refused(f"The decider fit produced a zero centroid (option {o}).")
    finite(weights, "linear weight")
    finite(bias, "linear bias")
    for value, what in (
        (loss, "linear loss"),
        (threshold, "threshold"),
        (temperature, "temperature"),
    ):
        if value is not None and not math.isfinite(value):
            raise Refused(
                f"The decider fit produced a non-finite {what} ({value}); a decision is never "
                "computed from it."
            )


def fit_decider(x, labels, ranks, count, ids, linear, calibration, labelled_x=None,
                labelled_labels=None):
    """`fitDecider` (decider_fit.dart L113-L234) on unit rows `x` (rows x d).
    `calibration` is the artifact's block: kind, then keep | maxError | threshold."""
    kind = calibration["kind"]
    centroids64 = option_centroids(x, labels, count, ids=ids)
    fit = fit_softmax_regression(x, labels, count) if linear else None

    threshold = temperature = signal = None
    if kind == "minSimilarity":
        threshold = calibration["threshold"]
    elif kind == "crossFit":
        signal = cross_fit_signal(x, labels, ranks, count, ids=ids)
        threshold = keep_threshold(signal, calibration["keep"])
    elif kind == "labelled":
        cosines = kernel_products(labelled_x, centroids64)
        signal = cosines.max(axis=1)
        if fit is None:
            scores = cosines
        else:
            # `_linearLogit` (L530-L543): the float64 fit, sum first, then the bias.
            scores = kernel_products(labelled_x, fit["weights"].T) + fit["bias"]
        ok = [int(np.argmax(scores[i])) == labelled_labels[i] for i in range(len(scores))]
        threshold = error_budget_threshold(signal, ok, calibration["maxError"])
        if threshold > 1:
            raise Refused(
                "no threshold keeps the error on the --calib rows within --max-error "
                f"{calibration['maxError']}: the decider answers them too poorly. Add examples "
                "per option, use --method linear, or raise --max-error"
            )
        # After the threshold, as Dart orders them (L194-L196).
        temperature = fit_temperature(scores, labelled_labels)
    elif kind != "none":
        raise ValueError(f"calibration {kind!r}")

    weights = bias = None
    if fit is not None:
        weights = fit["weights"].T.astype(np.float32)
        bias = fit["bias"].astype(np.float32)
    centroids = centroids64.astype(np.float32)
    check_fitted(
        centroids,
        weights=weights,
        bias=bias,
        loss=None if fit is None else fit["loss"],
        threshold=threshold,
        temperature=temperature,
    )
    return {
        "centroids": centroids,
        "centroids64": centroids64,
        "weights": weights,
        "bias": bias,
        "threshold": threshold,
        "temperature": temperature,
        "signal": signal,
        "linear": fit,
    }


def encode_float32(values):
    """`_encodeFloat32` (embedding_decider.dart L897-L903): little-endian float32, base64."""
    return base64.b64encode(np.asarray(values, dtype="<f4").tobytes()).decode("ascii")


def decode_float32(text):
    return np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float64)


def to_artifact(space_id, dimension, method, calibration, options, fitted):
    """`toArtifact` (embedding_decider.dart L820-L861), key for key and in its order.
    `options` are {id, description, metadata, examples} with the texts to store."""
    block = {"kind": calibration["kind"]}
    if calibration["kind"] == "crossFit":
        block["keep"] = calibration["keep"]
    if calibration["kind"] == "labelled":
        block["maxError"] = calibration["maxError"]
    if fitted["threshold"] is not None:
        block["threshold"] = fitted["threshold"]
    if fitted["temperature"] is not None:
        block["temperature"] = fitted["temperature"]
    artifact = {
        "format": "flutter_edge_ai.decider",
        "version": 1,
        "kind": "embedding",
        "spaceId": space_id,
        "dimension": dimension,
        "taskType": "classification",
        "method": method,
        "calibration": block,
    }
    if fitted["weights"] is not None:
        artifact["linear"] = {"weightDecay": LINEAR_WEIGHT_DECAY}
    stored = []
    for o, option in enumerate(options):
        entry = {"id": option["id"]}
        if option.get("description") is not None:
            entry["description"] = option["description"]
        entry["examples"] = list(option["examples"])
        if option.get("metadata") is not None:
            entry["metadata"] = json.loads(json.dumps(option["metadata"], allow_nan=False))
        entry["centroid"] = encode_float32(fitted["centroids"][o])
        if fitted["weights"] is not None:
            entry["weights"] = encode_float32(fitted["weights"][o])
            entry["bias"] = float(fitted["bias"][o])
        stored.append(entry)
    artifact["options"] = stored
    return artifact


def decisions(rows, artifact):
    """What a decider loaded from `artifact` answers for unit `rows`: each row's top option,
    highest centroid cosine and whether it abstains -- `EmbeddingDecider._score`
    (embedding_decider.dart L701-L804) over every option, from the stored float32 values."""
    options = artifact["options"]
    centroids = np.stack([decode_float32(o["centroid"]) for o in options])
    cosines = kernel_products(rows, centroids)
    similarity = cosines.max(axis=1)
    if artifact["method"] == "linear":
        weights = np.stack([decode_float32(o["weights"]) for o in options])
        bias = np.array([float(o["bias"]) for o in options])
        raw = kernel_products(rows, weights) + bias
        m = raw.max(axis=1, keepdims=True)
        log_sum = m[:, 0] + np.log(seq_row_sums(np.exp(raw - m)))
        scores = raw - log_sum[:, None]
    else:
        scores = cosines
    threshold = artifact["calibration"].get("threshold")
    if threshold is None:
        abstain = np.zeros(len(rows), dtype=bool)
    else:
        abstain = similarity < threshold
    # The first of equal scores ranks first (L746-L750).
    return scores.argmax(axis=1), similarity, abstain, scores


def measure(rows, labels, artifact):
    """Counts over held-out `rows`: `labels[i]` an option index, or None out of scope."""
    top, _, abstain, _ = decisions(rows, artifact)
    inside = [i for i, label in enumerate(labels) if label is not None]
    outside = [i for i, label in enumerate(labels) if label is None]
    answered = [i for i in inside if not abstain[i]]
    return {
        "in_scope": len(inside),
        "argmax_right": sum(1 for i in inside if top[i] == labels[i]),
        "answered": len(answered),
        "answered_wrong": sum(1 for i in answered if top[i] != labels[i]),
        "out_of_scope": len(outside),
        "rejected": sum(1 for i in outside if abstain[i]),
    }
'''

_MAIN_SOURCE = r'''

def embed_rows(engine, options, prefix, block, dimension, setting):
    """`block`'s texts embedded one request at a time, as `embedDeciderRows` sends them, each
    the task prefix and the text (flutter_edge_ai_litertlm embedding_parts.dart L28)."""
    rows = np.zeros((len(block["texts"]), dimension))
    for i, text in enumerate(block["texts"]):
        what = block["names"][i]
        try:
            response = engine.compute_embedding(prefix + text, options)
        except RuntimeError as exc:
            raise Refused(
                f"{what}: litert-lm could not embed it ({exc}). One cause is an input over "
                f"{setting} tokens, prefix and special tokens included, which is refused here "
                "rather than truncated"
            ) from exc
        rows[i] = unit_row(response.embedding, dimension, what)
    return rows


def run(spec):
    """Spec: model, cache_dir, prefix, max_input_length, space_id_head, space_id_tail,
    method, calibration, options [{id, description, metadata, examples}], inputs {texts,
    options, ranks, names}, labelled and test {texts, options, names} or null, artifact."""
    import litert_lm

    started = time.monotonic()
    with litert_lm.ModelInfo(spec["model"]) as info:
        if not info.is_embedding_model or info.embedding is None:
            raise Refused(
                f"{spec['model']} is not an embedding model (its model type is "
                f"{info.model_type.name})"
            )
        dimension = info.embedding.dimension
        signatures = info.embedding.signature_selection
    if not dimension:
        raise Refused(f"{spec['model']} declares no embedding dimension")
    lengths = sorted(n for n in (signatures or ()) if n > 0)
    if not lengths:
        raise Refused(f"{spec['model']} declares no text signature length")
    setting = min(spec["max_input_length"], lengths[-1])
    options = litert_lm.EmbeddingOptions(
        normalize=True,
        insert_special_tokens=True,
        input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
    )
    with litert_lm.EmbeddingEngine(
        spec["model"],
        backend=litert_lm.Backend.CPU(),
        cache_dir=spec["cache_dir"],
        max_input_length=setting,
    ) as engine:
        x = embed_rows(engine, options, spec["prefix"], spec["inputs"], dimension, setting)
        labelled = spec.get("labelled")
        lx = None if labelled is None else embed_rows(
            engine, options, spec["prefix"], labelled, dimension, setting
        )
        test = spec.get("test")
        tx = None if test is None else embed_rows(
            engine, options, spec["prefix"], test, dimension, setting
        )
    embedded = time.monotonic()

    ids = [option["id"] for option in spec["options"]]
    fitted = fit_decider(
        x,
        spec["inputs"]["options"],
        spec["inputs"]["ranks"],
        len(ids),
        ids,
        spec["method"] == "linear",
        spec["calibration"],
        labelled_x=lx,
        labelled_labels=None if labelled is None else labelled["options"],
    )
    artifact = to_artifact(
        spec["space_id_head"] + str(dimension) + spec["space_id_tail"],
        dimension,
        spec["method"],
        spec["calibration"],
        spec["options"],
        fitted,
    )
    Path(spec["artifact"]).write_text(json.dumps(artifact, allow_nan=False), encoding="utf-8")
    fitted_at = time.monotonic()

    # Measured from the file as written, as an app that loads it would score.
    written = json.loads(Path(spec["artifact"]).read_text(encoding="utf-8"))
    measured = None if tx is None else measure(tx, test["options"], written)
    linear = fitted["linear"]
    return {
        "dimension": dimension,
        "signatures": lengths,
        "max_input_length": setting,
        "embedded": {
            "inputs": len(x),
            "labelled": 0 if lx is None else len(lx),
            "test": 0 if tx is None else len(tx),
        },
        "fit": {
            "threshold": fitted["threshold"],
            "temperature": fitted["temperature"],
            "linear": None if linear is None else {
                "loss": linear["loss"],
                "iterations": linear["iterations"],
                "evaluations": linear["evaluations"],
            },
        },
        "test": measured,
        "seconds": {
            "embedding": embedded - started,
            "fit": fitted_at - embedded,
            "measure": time.monotonic() - fitted_at,
        },
    }


def main(spec_path, result_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result = {"runtime_version": runtime_version(), "cpu": cpu_report()}
    try:
        result.update(run(spec))
    except Refused as exc:
        result["refused"] = str(exc)
    Path(result_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''

_DECIDE_SCRIPT = _FIT_SOURCE + RUNTIME_ENGINE_SOURCE + _MAIN_SOURCE


@functools.cache
def _script() -> dict[str, Any]:
    """The script's own definitions, so the tests hold the fit the stage runs to the Dart
    goldens -- from the one source, not a copy. Needs numpy where it runs."""
    namespace: dict[str, Any] = {"__name__": "litetune_decide_script"}
    exec(compile(_DECIDE_SCRIPT, "litetune decide script", "exec"), namespace)  # noqa: S102
    return namespace


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecideResult:
    request: DecideRequest
    artifact: Path
    report_path: Path
    report: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.report


def _per_option(plan: Plan, rows: Sequence[Row]) -> dict[str, int]:
    counts = Counter(row.label for row in rows if row.label is not None)
    return {option.id: counts[option.id] for option in plan.options}


def _proportion(successes: int, n: int, missing: str) -> dict[str, Any]:
    if n == 0:
        return Unavailable(missing).as_dict()
    return Proportion.of(successes, n).as_dict()


def _measurements(measured: Mapping[str, int] | None) -> dict[str, Any]:
    if measured is None:
        return Unavailable("no --test rows were given").as_dict()
    inside, outside = measured["in_scope"], measured["out_of_scope"]
    return {
        "available": True,
        "source": f"{ARTIFACT_NAME} as written: its float32 centroids and weights read back",
        "rows": {"in_scope": inside, "out_of_scope": outside},
        # The top option against the label, whatever the threshold.
        "argmax_accuracy": _proportion(
            measured["argmax_right"], inside, "--test has no in-scope row"
        ),
        "coverage": _proportion(measured["answered"], inside, "--test has no in-scope row"),
        "error_on_answered": _proportion(
            measured["answered_wrong"],
            measured["answered"],
            "no in-scope row was answered at the threshold",
        ),
        "out_of_scope_rejected": _proportion(
            measured["rejected"], outside, '--test has no row with "label": null'
        ),
    }


def _install_rule(request: DecideRequest, name: str, size: int, sid: str) -> str:
    if request.embedder_id is not None:
        how = f"pass embedderId {request.embedder_id!r} to createDecider and loadDecider"
    else:
        how = (
            f"install the embedder under the name {name!r} ({size:,} bytes, the size this "
            "file has) and pass no embedderId"
        )
    return (
        f"The app must {how}: flutter_edge_ai's loadDecider compares the embedder's space id "
        f"with {sid!r} and refuses another with DeciderSpaceMismatchException."
    )


def _spec(plan: Plan, request: DecideRequest, model_term: str, work: Path) -> dict[str, Any]:
    head, tail = space_id_parts(model_term)
    return {
        "model": str(request.embedder.absolute()),
        "cache_dir": str(work / "cache"),
        "prefix": TASK_PREFIX,
        "max_input_length": MAX_INPUT_LENGTH,
        "space_id_head": head,
        "space_id_tail": tail,
        "method": request.method,
        "calibration": plan.calibration,
        "options": [
            {
                "id": option.id,
                "description": option.description,
                "metadata": option.metadata,
                "examples": list(plan.stored_examples[o]),
            }
            for o, option in enumerate(plan.options)
        ],
        "inputs": plan.inputs(),
        "labelled": plan.rows(plan.calib) if plan.calibration["kind"] == "labelled" else None,
        "test": plan.rows(plan.test) if plan.test else None,
        "artifact": str(work / ARTIFACT_NAME),
    }


def _run_script(
    spec: dict[str, Any], work: Path, *, env: envs.StageEnv, timeout: int
) -> dict[str, Any]:
    script = work / "decide.py"
    script.write_text(_DECIDE_SCRIPT, encoding="utf-8")
    spec_path, result_path = work / "spec.json", work / "result.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    try:
        proc = env.run(["python", str(script), str(spec_path), str(result_path)], timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DecideError(f"the decide script did not finish within {timeout}s") from None
    if proc.returncode != 0:
        stderr = proc.stderr or ""
        last = next((ln for ln in reversed(stderr.splitlines()) if ln.strip()), "no stderr")
        logger.warning("the decide script failed:\n%s", stderr[-2000:])
        reading = read_returncode(proc.returncode)
        raise DecideError(f"the decide script {reading.describe('the fit')}: {last}")
    if not result_path.is_file():
        raise DecideError("the decide script exited 0 without writing its result")
    result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    return result


def _write_new(path: Path, data: bytes) -> None:
    """Write `data` to `path`, which must not exist yet."""
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError:
        raise DecideError(f"{path} appeared while the decider was fitted; it is kept") from None


def run_decide(
    request: DecideRequest,
    *,
    events: EventStream | None = None,
    env: envs.StageEnv = envs.RUNTIME,
) -> DecideResult:
    """Fit the decider `request` describes and write `decider.json` and `decide.json` to
    `request.output_dir`. Raises `DecideError` for a request it refuses or a decider it will
    not write; nothing is written then."""
    if request.embedder_id is not None and request.embedder_name is not None:
        raise DecideError(
            "--embedder-id replaces the installed file's name in the space id; give one of "
            "--embedder-id and --embedder-name"
        )
    if request.embedder_id is not None and not request.embedder_id:
        raise DecideError("--embedder-id must not be empty")
    name = request.embedder_name if request.embedder_name is not None else request.embedder.name
    if not name or "/" in name or "\\" in name:
        raise DecideError(f"--embedder-name must be a file name, got {name!r}")
    if not request.embedder.is_file():
        raise FileNotFoundError(f"no embedder at {request.embedder}")
    plan_ = plan(request)
    output = Path(request.output_dir)
    artifact_path, report_path = output / ARTIFACT_NAME, output / REPORT_NAME
    for path in (artifact_path, report_path):
        if path.exists() or path.is_symlink():
            raise DecideError(f"{path} already exists; give another --output-dir")

    size = request.embedder.stat().st_size
    model_term = identity(install_name=name, size=size, embedder_id=request.embedder_id)
    if request.auto_provision:
        env.provision(events=events)
    if not env.ready:
        raise DecideError(f"environment {env.name!r} {env.unready_detail}")
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".decide-", dir=output))
    try:
        (work / "cache").mkdir()
        if events:
            events.note(
                f"decide: embedding {len(plan_.train)} train, {len(plan_.calib)} calib and "
                f"{len(plan_.test)} test rows with {request.embedder.name} on the CPU, then "
                f"fitting {request.method} / {request.calibration}"
            )
        result = _run_script(
            _spec(plan_, request, model_term, work), work, env=env, timeout=request.timeout_s
        )
        if result.get("refused"):
            raise DecideError(f"{result['refused']}; nothing was written")
        dimension = int(result["dimension"])
        sid = space_id(model_term, dimension)
        written = (work / ARTIFACT_NAME).read_bytes()
        try:
            artifact = json.loads(written, parse_constant=_refuse_constant)
        except ValueError as exc:
            raise ArtifactError(f"the artifact the script wrote is not JSON: {exc}") from None
        try:
            check_artifact(artifact, expected_space_id=sid, dimension=dimension)
        except ArtifactError as exc:
            raise ArtifactError(
                f"flutter_edge_ai's loadDecider would refuse the artifact written: {exc} "
                "Nothing was written."
            ) from None
        report = _report(request, plan_, result, name, size, sid, written)
        _write_new(artifact_path, written)
        _write_new(report_path, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    finally:
        leaked: list[str] = []
        shutil.rmtree(work, onerror=lambda _fn, path, _exc: leaked.append(str(path)))
        if leaked:
            logger.warning("%d scratch file(s) left under %s", len(leaked), work)
    if events:
        events.artifact(str(artifact_path), name=ARTIFACT_NAME)
    return DecideResult(request, artifact_path, report_path, report)


def _report(
    request: DecideRequest,
    plan_: Plan,
    result: Mapping[str, Any],
    name: str,
    size: int,
    sid: str,
    written: bytes,
) -> dict[str, Any]:
    fit = result["fit"]
    kind = plan_.calibration["kind"]
    calibration: dict[str, Any] = {"kind": kind}
    if kind == "crossFit":
        calibration["keep"] = plan_.calibration["keep"]
    elif kind == "labelled":
        calibration["max_error"] = plan_.calibration["maxError"]
    elif kind == "minSimilarity":
        calibration["min_similarity"] = plan_.calibration["threshold"]
    calibration["threshold"] = fit["threshold"]
    calibration["temperature"] = fit["temperature"]
    linear = fit["linear"]
    test_rows = plan_.test
    trained = {row.text for row in plan_.train}
    repeats = sum(1 for row in test_rows if row.text in trained)
    notes = [
        f"Format {ARTIFACT_FORMAT} version {ARTIFACT_VERSION}, read by flutter_edge_ai 2.4.0's "
        "EmbeddingModel.loadDecider, which is @experimental.",
        "Embedded on the CPU: a decider fitted on these vectors is refused on another "
        "backend's embedder, whose vectors are another space.",
    ]
    if plan_.calibration["kind"] == "none":
        notes.append("Calibration none never abstains: every --test row is answered.")
    if repeats:
        notes.append(f"{repeats} --test row(s) repeat a --train text.")
    return {
        "schema": DECIDE_SCHEMA,
        "litetune": __version__,
        "outcome": "written",
        "format_source": SOURCE,
        "artifact": {
            "path": str(request.output_dir / ARTIFACT_NAME),
            "bytes": len(written),
            "sha256": hash_bytes(written),
            "examples_per_option": request.examples_in_artifact,
        },
        "space_id": sid,
        "install_rule": _install_rule(request, name, size, sid),
        "embedder": {
            "path": str(request.embedder),
            "install_name": None if request.embedder_id is not None else name,
            "embedder_id": request.embedder_id,
            "bytes": size,
            "sha256": hash_file(request.embedder),
            "dimension": result["dimension"],
            "text_signatures": result["signatures"],
            "max_input_length": result["max_input_length"],
            "backend": BACKEND,
            "task_prefix": TASK_PREFIX,
        },
        "runtime": {"litert_lm": result.get("runtime_version"), "cpu": result.get("cpu")},
        "inputs": {
            "train": {
                "path": str(request.train),
                "sha256": hash_file(request.train),
                "rows": len(plan_.train),
                "per_option": _per_option(plan_, plan_.train),
            },
            "options": None if request.options is None else str(request.options),
            "calib": None
            if request.calib is None
            else {
                "path": str(request.calib),
                "sha256": hash_file(request.calib),
                "rows": len(plan_.calib),
                "per_option": _per_option(plan_, plan_.calib),
            },
            "test": None
            if request.test is None
            else {
                "path": str(request.test),
                "sha256": hash_file(request.test),
                "rows": len(test_rows),
                "per_option": _per_option(plan_, test_rows),
                "out_of_scope": sum(1 for row in test_rows if row.label is None),
            },
        },
        "options": [
            {
                "id": option.id,
                "description": option.description is not None,
                "stored_examples": len(plan_.stored_examples[o]),
            }
            for o, option in enumerate(plan_.options)
        ],
        "method": request.method,
        "linear": None
        if linear is None
        else {
            "weight_decay": LINEAR_WEIGHT_DECAY,
            "loss": linear["loss"],
            "iterations": linear["iterations"],
            "evaluations": linear["evaluations"],
        },
        "calibration": calibration,
        "measurements": _measurements(result.get("test")),
        "seconds": result.get("seconds"),
        "notes": notes,
    }


def summarise(result: DecideResult) -> list[str]:
    report = result.report
    calibration = report["calibration"]
    fitted = []
    if calibration.get("threshold") is not None:
        fitted.append(f"threshold {calibration['threshold']}")
    if calibration.get("temperature") is not None:
        fitted.append(f"temperature {calibration['temperature']}")
    inputs = report["inputs"]
    lines = [
        f"wrote {result.artifact} ({report['artifact']['bytes']:,} bytes): "
        f"{len(report['options'])} options, {report['method']}, {calibration['kind']}"
        + (f" ({', '.join(fitted)})" if fitted else ""),
        f"  {inputs['train']['rows']} train rows"
        + (f", {inputs['calib']['rows']} calib" if inputs["calib"] else "")
        + (f", {inputs['test']['rows']} test" if inputs["test"] else "")
        + f"; embedded by litert-lm {report['runtime']['litert_lm']} on the CPU",
        f"  space id: {report['space_id']}",
        f"  {report['install_rule']}",
    ]
    measured = report["measurements"]
    if measured.get("available"):
        for key, label in (
            ("argmax_accuracy", "argmax accuracy (in scope)"),
            ("coverage", "answered (in scope)"),
            ("error_on_answered", "error on answered"),
            ("out_of_scope_rejected", "out of scope rejected"),
        ):
            value = measured[key]
            if value.get("available"):
                lines.append(
                    f"  {label}: {value['value']:.4f} ±{value['ci95']:.4f} (n={value['n']})"
                )
            else:
                lines.append(f"  {label}: not measured — {value['reason']}")
    else:
        lines.append(f"  not measured — {measured['reason']}")
    lines.extend(f"  note: {note}" for note in report["notes"])
    return lines
