"""Export sweeps, with no toolchain, no network and no accelerator.

`StageEnv.run` is faked: it writes files instead of running `litert-torch`, so
every branch here -- a non-zero exit, a clean exit that produced nothing, an
environment that was never built -- is exercised in milliseconds. The fake
matches `StageEnv.run`'s contract exactly: it returns a `CompletedProcess` and
never raises on a non-zero exit, because that distinction is what the module
under test is built to record.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import (
    buildable_here,
    fifo_released_after,
    machine,
    mark_provisioned,
    record_opens,
    run_compose_script,
)

from litetune import envs, models, template_checks
from litetune.checks import Check, Outcome
from litetune.cli import summarise_convert
from litetune.events import Event, EventStream
from litetune.export import (
    _COMPOSE_SCRIPT,
    CANONICAL_OR_COMPOSED_FLAG,
    CHAT_TEMPLATE_CHECK,
    COMPOSED_TEMPLATE_FLAG,
    COMPOSED_TEMPLATE_PREFIX,
    KNOWN_RECIPES,
    MEASURED_RECIPES,
    NO_ARTIFACT,
    NOT_VERIFIED,
    PROBE_TEXT,
    TOOLCHAIN_DEFAULT_RECIPE,
    ExportRequest,
    ExportResult,
    NoRecipesRequested,
    SizeComparison,
    Uncompared,
    compare_sizes,
    export_recipe,
    parse_pip_freeze,
    requirement_names,
    run_export,
)

# `pip freeze`, not `pip show`: show reports only what the environment declares,
# and those are already pinned. The 2026-08-30 breakage lived in the transitive
# closure -- `absl-py` and `flatbuffers` below stand for it, and are exactly the
# packages a post-hoc diagnosis would need.
PIP_FREEZE = """absl-py==2.1.0
ai-edge-litert-nightly==2.3.0.dev20260925
ai-edge-quantizer-nightly==0.10.0.dev20260926
flatbuffers==24.3.25
litert-converter==0.4.0
litert-lm==0.18.0
litert-lm-builder==0.18.0
litert-torch-nightly==0.10.0.dev20260926
numpy==2.0.2
transformers==5.16.1
"""


# The chat template of Qwen/Qwen3-0.6B at c1899de, as its tokenizer carries it.
QWEN3_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen3-0.6b-c1899de.chat_template.jinja"
).read_text(encoding="utf-8")

# Kept at import, before any fixture replaces it, for the test that needs it back.
_REAL_REFUSAL = envs.StageEnv.refusal


@dataclass
class RecipeRun:
    """What the fake toolchain does for one recipe."""

    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    # (filename, bytes) written into the recipe's output directory.
    writes: tuple[tuple[str, int], ...] = (("model.litertlm", 4096),)
    raises: BaseException | None = None


@dataclass
class Call:
    argv: list[str]
    timeout: int
    kwargs: dict


@dataclass
class FakeToolchain:
    """Stands in for `envs.StageEnv.run`.

    For the GPU-activation repack it runs the real `_REPACK_SCRIPT` with the
    host interpreter and a fake `litert_lm_builder` on `PYTHONPATH`: `unpack`
    writes a `model.toml` and section files from the bundle, `pack` writes a
    bundle that carries its TOML. The script's own TOML parsing, patching and
    round-trip comparison are therefore exercised for real -- a regex-shaped
    bug in the script would show here -- while the bundle format is a fake.
    `repack_toml` sets what a pristine artifact unpacks to; the `repack_*`
    knobs make each builder step fail.

    The exporter fake embeds a chat template the way `package_model` does with
    an override: the file read in text mode (`litert_lm_builder.py:430`).
    `embedded` records it by artifact, and the fake builder's `unpack` writes it
    into the LlmMetadata section's text, where the repack script reads it back.
    `template_embedded` changes what is embedded, for an exporter that put
    something else in the bundle.
    """

    runs: dict[str, RecipeRun] = field(default_factory=dict)
    default: RecipeRun = field(default_factory=RecipeRun)
    pip_stdout: str = PIP_FREEZE
    pip_returncode: int = 0
    calls: list[Call] = field(default_factory=list)
    # The repack's knobs.
    repack_toml: str | None = None  # what a pristine artifact unpacks to
    repack_unpack_fails: bool = False
    repack_pack_fails: bool = False
    repack_pack_drifts: bool = False  # pack adds an unrequested key
    repack_pack_misplaces: bool = False  # pack puts the key on the embedder
    repack_script_returncode: int = 0  # a crashed interpreter
    # The chat template script's knobs: what the checkpoint's tokenizer carries,
    # and each way the script can fail to say.
    chat_template: object = QWEN3_TEMPLATE
    template_returncode: int | None = None  # the child's code, in place of running it
    template_stderr: str = ""
    template_no_result: bool = False  # exits 0 having written no result
    template_raises: BaseException | None = None
    template_overwrite: bytes | None = None  # written over its file once it has run
    template_result: str | None = None  # written as its result, in place of running it
    template_exit_after: int | None = None  # the code it ends with, after it has run
    template_report_edit: Callable[[dict], object] | None = None  # rewrites its report
    template_loads: list[str] = field(default_factory=list)
    transformers_version: str = "5.16.1"
    hub_files: dict[str, str] = field(default_factory=dict)  # what a Hub id's repo holds
    hub_fault: str = ""  # how the Hub download fails instead, when it does
    render_raises: str = ""  # an ImportError the fake renderer raises, as transformers can
    # The end offsets the fake bundle's header gives its sections; by default
    # one section ending at the end of the file. `repack_header_raises` makes
    # the header reader raise that message, as the real one does on a wrong magic.
    repack_section_ends: list[int] | None = None
    repack_header_raises: str = ""
    bundle_header_end: int | None = None  # what the exporter fake writes at byte 24
    # What the exporter embeds as the bundle's template, and how a test changes it.
    embedded: dict[str, str | None] = field(default_factory=dict)
    template_embedded: Callable[[str | None], str | None] | None = None
    repack_pack_rewrites_template: bool = False
    # The conversation script's knobs: its exit code, or its result, in place
    # of running it.
    conversation_returncode: int | None = None
    conversation_result: str | None = None

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(Call(argv=list(args), timeout=timeout, kwargs=dict(kwargs)))
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, self.pip_returncode, self.pip_stdout, "")
        if args[0] == "python" and str(args[1]).endswith("repack.py"):
            return self._repack(args, timeout)
        if args[0] == "python" and str(args[1]).endswith("compose_template.py"):
            return self._compose(args, timeout)
        if args[0] == "python" and str(args[1]).endswith("conversation_template.py"):
            return self._conversation(args, timeout)
        assert args[:2] == ["litert-torch", "export_hf"], args
        flags = dict(a.removeprefix("--").split("=", 1) for a in args[2:] if "=" in a)
        run = self.runs.get(flags["quantization_recipe"], self.default)
        if run.raises is not None:
            raise run.raises
        out_dir = Path(flags["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        template = None
        override = flags.get("jinja_chat_template_override")
        if override and os.path.exists(override):
            with open(override, encoding="utf-8") as handle:  # text mode, as package_model
                template = handle.read()
        if self.template_embedded is not None:
            template = self.template_embedded(template)
        for name, size in run.writes:
            body = bytearray(size)
            if self.bundle_header_end is not None and size >= 32:
                # Where a bundle names its header's end (`litertlm_core.py`: byte 24).
                body[24:32] = self.bundle_header_end.to_bytes(8, "little")
            (out_dir / name).write_bytes(bytes(body))
            self.embedded[str((out_dir / name).absolute())] = template
        return subprocess.CompletedProcess(args, run.returncode, run.stdout, run.stderr)

    @property
    def exports(self) -> list[Call]:
        return [c for c in self.calls if c.argv[0] == "litert-torch"]

    @property
    def repacks(self) -> list[Call]:
        return [c for c in self.calls if c.argv[0] == "python" and c.argv[1].endswith("repack.py")]

    @property
    def compositions(self) -> list[Call]:
        return [
            c
            for c in self.calls
            if c.argv[0] == "python" and c.argv[1].endswith("compose_template.py")
        ]

    @property
    def conversations(self) -> list[Call]:
        return [
            c
            for c in self.calls
            if c.argv[0] == "python" and c.argv[1].endswith("conversation_template.py")
        ]

    def _conversation(self, args, timeout: int) -> subprocess.CompletedProcess:
        """`export._CONVERSATION_SCRIPT`, run for real, or a canned answer."""
        if self.conversation_returncode is not None:
            return subprocess.CompletedProcess(args, self.conversation_returncode, "", "boom")
        if self.conversation_result is not None:
            Path(args[3]).write_text(self.conversation_result, encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")
        return run_compose_script(args, timeout, self.chat_template)[0]

    def _compose(self, args, timeout: int) -> subprocess.CompletedProcess:
        if self.template_raises is not None:
            raise self.template_raises
        if self.template_returncode is not None:
            return subprocess.CompletedProcess(
                args, self.template_returncode, "", self.template_stderr
            )
        if self.template_no_result:
            return subprocess.CompletedProcess(args, 0, "", self.template_stderr)
        if self.template_result is not None:
            Path(args[3]).write_text(self.template_result, encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, "", "")
        proc, loads = run_compose_script(
            args,
            timeout,
            self.chat_template,
            version=self.transformers_version,
            hub_files=self.hub_files,
            hub_fault=self.hub_fault,
            render_raises=self.render_raises,
        )
        self.template_loads.extend(loads)
        if self.template_overwrite is not None:
            out = Path(json.loads(Path(args[2]).read_text(encoding="utf-8"))["out"])
            out.write_bytes(self.template_overwrite)
        if self.template_exit_after is not None:
            return subprocess.CompletedProcess(args, self.template_exit_after, "", "")
        if self.template_report_edit is not None:
            report = json.loads(Path(args[3]).read_text(encoding="utf-8"))
            Path(args[3]).write_text(json.dumps(self.template_report_edit(report)))
        return proc

    def _repack(self, args, timeout: int) -> subprocess.CompletedProcess:
        if self.repack_script_returncode:
            return subprocess.CompletedProcess(
                args, self.repack_script_returncode, "", "Traceback: boom"
            )
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim" / "litert_lm_builder"
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "__init__.py").write_text("")
        embedded = shim.parent / "embedded.json"
        embedded.write_text(json.dumps(self.embedded), encoding="utf-8")
        (shim / "litertlm_builder.py").write_text(
            FAKE_BUILDER_MODULE.format(
                default_toml=self.repack_toml if self.repack_toml is not None else FAKE_TOML,
                unpack_fails=self.repack_unpack_fails,
                pack_fails=self.repack_pack_fails,
                pack_drifts=self.repack_pack_drifts,
                pack_misplaces=self.repack_pack_misplaces,
                pack_rewrites_template=self.repack_pack_rewrites_template,
                embedded_path=str(embedded),
            ),
            encoding="utf-8",
        )
        (shim / "litertlm_peek.py").write_text(
            f"SECTION_ENDS = {self.repack_section_ends!r}\n"
            f"HEADER_RAISES = {self.repack_header_raises!r}\n" + FAKE_PEEK_HEADER,
            encoding="utf-8",
        )
        # The real layout constant (`litertlm_core.py`): where a bundle names its header's end.
        (shim / "litertlm_core.py").write_text("HEADER_END_LOCATION_BYTE_OFFSET = 24\n")
        # The builder's own proto and protobuf's text format, which the script
        # imports to read the LlmMetadata text `unpack` wrote.
        for package, module, source in (
            (shim / "runtime" / "proto", "llm_metadata_pb2.py", FAKE_LLM_METADATA_PB2),
            (shim.parent / "google" / "protobuf", "text_format.py", FAKE_TEXT_FORMAT),
        ):
            package.mkdir(parents=True, exist_ok=True)
            for level in (package, package.parent):
                (level / "__init__.py").write_text("")
            (package / module).write_text(source, encoding="utf-8")
        return subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim.parent)},
        )


# A stand-in for the real `litert_lm_builder.litertlm_builder`, faithful in
# the three places a mutation test showed the old fake let production-breaking
# regressions through green:
#   - the bundle is self-describing: its TOML lives *inside* the bytes after the
#     `LITERTLM` magic, so the only way a test can read the new key back is if
#     the rebuilt file really replaced the original (os.replace is load-bearing);
#   - `pack` resolves each `data_path` the way the real `_resolve_path` does
#     (absolute as-is, relative against the TOML's directory) and *opens* it,
#     so a path the script forgot to absolutize raises FileNotFoundError;
#   - `pack` regenerates the system `uuid` and `creation_timestamp`, so the
#     script's exclusion of those two keys is what makes the round-trip equal.
# Knobs are formatted in as Python literals; braces that belong to the module
# are doubled for `str.format`.
FAKE_BUILDER_MODULE = """
import json
import os
import re
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TOML = {default_toml!r}
UNPACK_FAILS = {unpack_fails}
PACK_FAILS = {pack_fails}
PACK_DRIFTS = {pack_drifts}
PACK_MISPLACES = {pack_misplaces}
PACK_REWRITES_TEMPLATE = {pack_rewrites_template}
EMBEDDED = json.loads(Path({embedded_path!r}).read_text(encoding="utf-8"))
# The LlmMetadata text of each bundle `pack` wrote, by path: kept beside the
# fake rather than inside the bundle, so that a bundle's size stays the
# exporter fake's.
PACKED = os.path.join(os.path.dirname({embedded_path!r}), "packed.json")
MAGIC = b"LITERTLM"
METADATA = "LlmMetadataProto.pbtext"


def _packed():
    if not os.path.exists(PACKED):
        return {{}}
    return json.loads(Path(PACKED).read_text(encoding="utf-8"))


def _metadata_text(template):
    if template is None:
        return ""
    return "jinja_prompt_template: " + json.dumps(template) + "\\n"


def _contents_of(bundle_path):
    # (TOML, LlmMetadata text) of a bundle.
    raw = Path(bundle_path).read_bytes()
    if not raw.startswith(MAGIC):
        # A pristine artifact from the export fake: its template is the one the
        # exporter fake embedded.
        return DEFAULT_TOML, _metadata_text(EMBEDDED.get(os.path.abspath(bundle_path)))
    toml = raw[len(MAGIC):].rstrip(b"\\0").decode("utf-8")
    return toml, _packed().get(os.path.abspath(bundle_path), "")


def unpack(litertlm_path, output_dir, jinja_prompt_template_path=None):
    if UNPACK_FAILS:
        raise RuntimeError("unpack: boom")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    toml, metadata = _contents_of(litertlm_path)
    (out / "model.toml").write_text(toml, encoding="utf-8")
    for m in re.finditer(r'data_path\\s*=\\s*"([^"]+)"', toml):
        name = os.path.basename(m.group(1))
        if name == METADATA:
            (out / name).write_text(metadata, encoding="utf-8")
        else:
            (out / name).write_bytes(b"x")
    return str(out / "model.toml")


def _regenerate_system(toml):
    stamp = datetime.now(timezone.utc).isoformat()
    fresh = (
        '  {{ key = "uuid", value_type = "String", value = "%s" }},\\n'
        '  {{ key = "creation_timestamp", value_type = "String", value = "%s" }},\\n'
    ) % (_uuid.uuid4(), stamp)
    toml = re.sub(r'\\s*\\{{ key = "(uuid|creation_timestamp)"[^}}]*}},?\\n?', "", toml)
    return re.sub(r"(\\[system_metadata\\]\\nentries = \\[\\n?)", r"\\1" + fresh, toml, count=1)


def pack(toml_path, output_path, jinja_prompt_template_path=None):
    if PACK_FAILS:
        raise RuntimeError("pack: boom")
    toml_path = Path(toml_path)
    toml = toml_path.read_text(encoding="utf-8")
    metadata = []
    # Resolve and open every data_path as the real builder does; a relative
    # path the caller forgot to absolutize fails here, not silently.
    def resolve(m):
        raw = m.group(2)
        path = Path(raw) if os.path.isabs(raw) else toml_path.parent / raw
        with open(path, "rb") as handle:
            if os.path.basename(raw) == METADATA:
                metadata.append(handle.read())
        return m.group(1) + os.path.basename(raw) + m.group(3)
    toml = re.sub(r'(data_path\\s*=\\s*")([^"]+)(")', resolve, toml)
    toml = _regenerate_system(toml)
    if PACK_DRIFTS:
        toml = toml.replace("[[section]]\\n", '[[section]]\\nextra = "drift"\\n', 1)
    if PACK_MISPLACES:
        lines = toml.splitlines()
        act = [l for l in lines if "prefer_activation_type" in l and "key =" in l]
        if act:
            lines.remove(act[0])
            idx = max(i for i, l in enumerate(lines) if 'model_type = "embedder"' in l)
            lines[idx:idx + 1] = [lines[idx], "additional_metadata = [", act[0], "]"]
            toml = "\\n".join(lines)
    body = MAGIC + toml.encode("utf-8")
    if metadata:
        packed = _packed()
        text = metadata[0].decode("utf-8")
        packed[os.path.abspath(output_path)] = (
            _metadata_text("rewritten") if PACK_REWRITES_TEMPLATE else text
        )
        Path(PACKED).write_text(json.dumps(packed), encoding="utf-8")
    target = Path(output_path)
    original = target.parent.parent / target.name
    size = original.stat().st_size if original.exists() else len(body)
    target.write_bytes(body + b"\\0" * max(0, size - len(body)))
    return str(target)
"""


FAKE_TOML = """[system_metadata]
entries = []

[[section]]
section_type = "LlmMetadata"
data_path = "LlmMetadataProto.pbtext"

[[section]]
section_type = "SP_Tokenizer"
data_path = "tok.spiece"

[[section]]
model_type = "prefill_decode"
section_type = "TFLiteModel"
data_path = "prefill_decode.tflite"

[[section]]
model_type = "embedder"
section_type = "TFLiteModel"
data_path = "embedder.tflite"
"""


# `litertlm_peek.read_litertlm_header`, as far as the repack script reads it:
# each section's end offset, from `SECTION_ENDS` or else the file's size.
FAKE_PEEK_HEADER = """
import os


class _Section:
    def __init__(self, end):
        self.end = end

    def EndOffset(self):
        return self.end


class _Sections:
    def __init__(self, ends):
        self.ends = ends

    def ObjectsLength(self):
        return len(self.ends)

    def Objects(self, index):
        return _Section(self.ends[index])


class _Header:
    def __init__(self, ends):
        self.ends = ends

    def SectionMetadata(self):
        return _Sections(self.ends)


def read_litertlm_header(path, output_stream):
    if HEADER_RAISES:
        raise ValueError(HEADER_RAISES)
    return _Header(SECTION_ENDS if SECTION_ENDS is not None else [os.path.getsize(path)])
"""


# The builder's `LlmMetadata` and protobuf's `text_format.Parse`, as far as the
# repack script uses them, reading the one-field-per-line text the fake
# builder's `unpack` writes.
FAKE_LLM_METADATA_PB2 = """
class LlmMetadata:
    def __init__(self):
        self.fields = {}

    def HasField(self, name):
        return name in self.fields

    @property
    def jinja_prompt_template(self):
        return self.fields.get("jinja_prompt_template", "")
"""

FAKE_TEXT_FORMAT = """
import json


def Parse(text, message):
    for line in text.splitlines():
        if line.strip():
            name, _, value = line.partition(": ")
            message.fields[name] = json.loads(value)
    return message
"""


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    """A fake export environment: provisioning writes a marker, `run` writes files."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    buildable_here(monkeypatch)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


@pytest.fixture
def request_for(tmp_path):
    def _build(recipes=MEASURED_RECIPES, **kwargs) -> ExportRequest:
        return ExportRequest(
            model="google/functiongemma-270m-it",
            output_dir=tmp_path / "out",
            recipes=recipes,
            **kwargs,
        )

    return _build


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


# The two artifacts `functiongemma-270m-it` produced, in bytes. Recorded here
# because "the same bit width, a file 0.04% larger" is asserted in the README
# and in three docstrings, and a figure repeated four times with no derivation
# is the shape of a claim rather than a measurement.
MEASURED_DYNAMIC_BYTES = 455_759_152
MEASURED_WEIGHT_ONLY_BYTES = 455_939_600


def test_sweep_produces_one_result_per_recipe(toolchain, request_for):
    recipes = ("dynamic_wi8_afp32", "weight_only_wi8_afp32", "dynamic_wi4_afp32")
    toolchain.runs = {
        "dynamic_wi8_afp32": RecipeRun(writes=(("model.litertlm", 4096),)),
        # Larger, and the better model: size does not rank recipes. The
        # measured pair is in `test_sweep_reports_the_size_frontier`.
        "weight_only_wi8_afp32": RecipeRun(writes=(("model.litertlm", 4098),)),
        "dynamic_wi4_afp32": RecipeRun(writes=(("model.litertlm", 2048),)),
    }
    result = run_export(request_for(recipes))

    assert result.outcome is Outcome.PASSED
    assert [e.recipe for e in result.exports] == list(recipes)
    assert len(result.succeeded) == 3
    assert [e.artifact_bytes for e in result.exports] == [4096, 4098, 2048]
    # One directory per recipe: a sweep whose artifacts overwrite each other
    # has measured one thing three times.
    assert len({e.artifact.parent for e in result.exports}) == 3
    assert all(e.artifact.is_file() for e in result.exports)
    assert len(toolchain.exports) == 3


def test_sweep_reports_the_size_frontier_and_refuses_to_rank_on_it(toolchain, request_for):
    # The bytes both recipes actually produced for `functiongemma-270m-it`, so
    # the "0.04% apart" the README and three docstrings assert has a derivation
    # in the suite instead of being a number four files repeat.
    toolchain.runs = {
        "dynamic_wi8_afp32": RecipeRun(writes=(("model.litertlm", MEASURED_DYNAMIC_BYTES),)),
        "weight_only_wi8_afp32": RecipeRun(
            writes=(("model.litertlm", MEASURED_WEIGHT_ONLY_BYTES),)
        ),
    }
    result = run_export(request_for(MEASURED_RECIPES))

    comparison = result.comparison
    assert isinstance(comparison, SizeComparison)
    assert comparison.smallest == "dynamic_wi8_afp32"
    assert comparison.largest == "weight_only_wi8_afp32"
    # 0.0396%, which rounds to the 0.04% the prose states.
    assert comparison.spread_share == pytest.approx(0.000396, abs=1e-6)
    assert round(comparison.spread_share * 100, 2) == 0.04
    # Size is one axis of two, and it is not the deciding one.
    assert comparison.as_dict()["accuracy"]["available"] is False


def test_a_recipe_that_fails_does_not_hide_the_ones_that_worked(toolchain, request_for):
    recipes = ("dynamic_wi8_afp32", "weight_only_wi8_afp32", "dynamic_wi4_afp32")
    toolchain.runs = {
        "weight_only_wi8_afp32": RecipeRun(returncode=1, stderr="boom", writes=()),
        "dynamic_wi4_afp32": RecipeRun(returncode=1, stderr="boom", writes=()),
    }
    result = run_export(request_for(recipes))

    assert len(result.exports) == 3
    assert [e.recipe for e in result.succeeded] == ["dynamic_wi8_afp32"]
    assert [e.recipe for e in result.failed] == ["weight_only_wi8_afp32", "dynamic_wi4_afp32"]
    assert result.outcome is Outcome.FAILED
    # One survivor is not a comparison either.
    assert result.comparison.compared is False
    assert "1 of 3" in result.comparison.reason


# ---------------------------------------------------------------------------
# One recipe is not a comparison
# ---------------------------------------------------------------------------


def test_single_recipe_result_says_no_alternative_was_measured(toolchain, request_for):
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    assert result.outcome is Outcome.PASSED
    assert isinstance(result.comparison, Uncompared)
    assert result.comparison.compared is False
    assert "no alternative was measured" in result.comparison.reason
    assert result.as_dict()["comparison"]["compared"] is False
    # The reason has to travel with the result, not only with the object.
    assert any("no alternative was measured" in limit for limit in result.limitations)


def test_sweeping_only_the_toolchain_default_is_called_out(toolchain, request_for):
    result = run_export(request_for((TOOLCHAIN_DEFAULT_RECIPE,)))
    assert any("toolchain's own default" in limit for limit in result.limitations)


def test_no_recipes_is_refused_rather_than_defaulted(tmp_path):
    with pytest.raises(NoRecipesRequested) as excinfo:
        ExportRequest(model="m", output_dir=tmp_path, recipes=())
    # The refusal has to name the alternative, or the caller just picks the default again.
    assert TOOLCHAIN_DEFAULT_RECIPE in str(excinfo.value)


def test_recipes_are_deduplicated_so_the_sweep_measures_distinct_things(tmp_path):
    request = ExportRequest(
        model="m",
        output_dir=tmp_path,
        recipes=["dynamic_wi8_afp32", "dynamic_wi8_afp32", " weight_only_wi8_afp32 "],
    )
    assert request.recipes == ("dynamic_wi8_afp32", "weight_only_wi8_afp32")


def test_a_recipe_name_cannot_escape_the_output_directory(tmp_path):
    with pytest.raises(ValueError):
        ExportRequest(model="m", output_dir=tmp_path, recipes=("../../etc",))


def test_unknown_recipe_is_passed_through_but_flagged(toolchain, request_for):
    result = run_export(request_for(("dynamic_wi8_afp32", "experimental_wi2")))
    assert result.outcome is Outcome.PASSED
    (flagged,) = [limit for limit in result.limitations if "passed through" in limit]
    named = flagged.split(" are not among")[0]
    assert "experimental_wi2" in named
    # The measured preset in the same sweep is not swept up with it. Asserted on
    # the names the limitation flags, not on the whole sentence: it also lists
    # the measured recipes for context, and `dynamic_wi8_afp32` is one of them.
    assert "dynamic_wi8_afp32" not in named
    assert "experimental_wi2" not in KNOWN_RECIPES


# ---------------------------------------------------------------------------
# A produced file is not a successful export
# ---------------------------------------------------------------------------


def test_nothing_the_module_returns_claims_verification(toolchain, request_for):
    result = run_export(request_for(MEASURED_RECIPES))
    record = result.as_dict()

    assert result.verified is False
    assert record["verified"] is False
    assert record["unverified_reason"]
    assert all(e.verified is False for e in result.exports)
    assert all(e["verified"] is False for e in record["exports"])
    assert any("liveness is not established" in limit for limit in result.limitations)


def test_zero_exit_without_an_artifact_is_a_failure(toolchain, request_for):
    toolchain.default = RecipeRun(returncode=0, writes=())
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    export = result.exports[0]
    assert export.check.outcome is Outcome.FAILED
    assert export.returncode == 0
    assert export.artifact is None
    assert "exited zero" in export.check.detail
    assert result.outcome is Outcome.FAILED


def test_companion_files_alone_are_not_an_artifact(toolchain, request_for):
    # `--externalize_embedder` writes an embedder beside the .litertlm. Only the
    # embedder appearing means the export did not finish.
    toolchain.default = RecipeRun(returncode=0, writes=(("embedder.tflite", 1024),))
    result = run_export(request_for(("weight_only_wi8_afp32",), externalize_embedder=True))

    assert result.exports[0].check.outcome is Outcome.FAILED
    assert "wrote no .litertlm" in result.exports[0].check.detail


def test_an_artifact_from_a_previous_run_is_not_this_run_s_result(toolchain, request_for):
    request = request_for(("weight_only_wi8_afp32",))
    stale_dir = request.dir_for("weight_only_wi8_afp32")
    stale_dir.mkdir(parents=True)
    stale = stale_dir / "model.litertlm"
    stale.write_bytes(b"\0" * 4096)
    old = 1_600_000_000.0
    os.utime(stale, (old, old))
    toolchain.default = RecipeRun(returncode=0, writes=())

    result = run_export(request)

    assert result.exports[0].check.outcome is Outcome.FAILED
    assert result.exports[0].artifact is None


def test_two_artifacts_are_ambiguous_rather_than_a_guess(toolchain, request_for):
    toolchain.default = RecipeRun(writes=(("model.litertlm", 4096), ("model_v2.litertlm", 4096)))
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    assert result.exports[0].check.outcome is Outcome.FAILED
    assert "ambiguous" in result.exports[0].check.detail


def test_companions_ship_with_the_artifact(toolchain, request_for):
    toolchain.default = RecipeRun(writes=(("model.litertlm", 4096), ("embedder.tflite", 1024)))
    result = run_export(request_for(("weight_only_wi8_afp32",), externalize_embedder=True))

    export = result.exports[0]
    assert export.ok
    assert export.companions == ("embedder.tflite",)
    assert export.artifact_bytes == 4096
    assert export.shipped_bytes == 5120


def test_artifact_is_identified_by_content(toolchain, request_for):
    import hashlib

    result = run_export(request_for(("weight_only_wi8_afp32",)))
    export = result.exports[0]
    assert export.sha256 == hashlib.sha256(export.artifact.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Failures are data
# ---------------------------------------------------------------------------


def test_non_zero_exit_is_recorded_with_the_whole_of_stderr(toolchain, request_for):
    stderr = "FIRST-FRAME: AttributeError: pad_token\n" + ("noise\n" * 2000) + "LAST-FRAME\n"
    toolchain.default = RecipeRun(returncode=2, stderr=stderr, writes=())
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    export = result.exports[0]
    assert export.check.outcome is Outcome.FAILED
    assert export.attempted is True
    assert export.returncode == 2
    # A nightly-specific failure is diagnosed from the first frame as often as
    # the last, so neither end may be truncated away.
    assert "FIRST-FRAME" in export.stderr
    assert "LAST-FRAME" in export.stderr
    assert export.stderr == stderr
    assert export.as_dict()["stderr"] == stderr
    # The human-facing detail still gets the tail rather than 2000 lines.
    assert "LAST-FRAME" in export.check.detail
    assert len(export.check.detail) < 1000


def test_a_timeout_is_not_a_verdict_about_the_recipe(toolchain, request_for):
    toolchain.default = RecipeRun(raises=subprocess.TimeoutExpired(cmd="litert-torch", timeout=1))
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    export = result.exports[0]
    assert export.check.outcome is Outcome.UNCHECKED
    assert export.attempted is False
    assert result.outcome is Outcome.UNCHECKED
    assert result.failed == []


def test_a_toolchain_that_cannot_start_is_unchecked(toolchain, request_for):
    toolchain.default = RecipeRun(raises=FileNotFoundError("litert-torch: not found"))
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    assert result.exports[0].check.outcome is Outcome.UNCHECKED
    assert result.outcome is Outcome.UNCHECKED


def test_a_missing_environment_could_not_be_checked_rather_than_failed(
    monkeypatch, toolchain, request_for
):
    def refuse(self, events=None, force: bool = False):
        raise RuntimeError("could not provision environment 'export': no network")

    monkeypatch.setattr(envs.StageEnv, "provision", refuse)
    result = run_export(request_for(MEASURED_RECIPES))

    # The machine could not build the environment. That says nothing about the
    # recipes, so none of them may be reported as failing.
    assert result.outcome is Outcome.UNCHECKED
    assert result.failed == []
    assert result.exports == []
    assert result.not_attempted == tuple(MEASURED_RECIPES)
    assert toolchain.exports == []
    # And the provenance says why it is empty, rather than looking like a machine
    # on which pip happened to be quiet.
    assert "environment was not usable" in result.toolchain.unresolved_reason


def test_an_unprovisioned_environment_is_not_silently_used(toolchain, request_for):
    result = run_export(request_for(MEASURED_RECIPES, auto_provision=False))

    assert result.outcome is Outcome.UNCHECKED
    assert "not provisioned" in result.checks.first_unchecked.detail
    assert result.not_attempted == tuple(MEASURED_RECIPES)


# ---------------------------------------------------------------------------
# Provenance and the machine it ran on
# ---------------------------------------------------------------------------


def test_resolved_toolchain_versions_are_recorded(toolchain, request_for):
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    assert result.toolchain.available
    assert result.toolchain.resolved["litert-torch-nightly"] == "0.10.0.dev20260926"
    assert result.toolchain.resolved["numpy"] == "2.0.2"
    # The point of freezing: packages nobody declared are recorded too, because
    # that is where the toolchain actually moved under us.
    assert result.toolchain.resolved["flatbuffers"] == "24.3.25"
    assert result.toolchain.missing == ()
    # The declared pins travel with what they resolved to; neither implies the other.
    assert "litert-lm==0.18.0" in result.toolchain.declared
    assert result.as_dict()["toolchain"]["available"] is True


def test_unreadable_toolchain_versions_do_not_turn_a_good_export_into_a_non_result(
    toolchain, request_for
):
    toolchain.pip_stdout = ""
    toolchain.pip_returncode = 1
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    # pip is provenance, not a check: folding it into the check set would
    # report "could not check" about an artifact that plainly exists.
    assert result.outcome is Outcome.PASSED
    assert result.toolchain.available is False
    assert any("provenance is incomplete" in limit for limit in result.limitations)


def test_export_does_not_ask_for_an_accelerator(toolchain, request_for):
    run_export(request_for(("weight_only_wi8_afp32",)))

    call = toolchain.exports[0]
    # Measured: 285,577,392 bytes in 122 s on CPU. Export must stay runnable on
    # a machine with no GPU, and the recorded duration must stay comparable.
    assert call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert call.kwargs["env"]["HIP_VISIBLE_DEVICES"] == ""
    assert "--backend=gpu" not in call.argv


class _AnyTemplate:
    """Matches the template flag whatever absolute path it carries."""

    def __eq__(self, other: object) -> bool:
        # By path parts, not by a `/` suffix: the path is the platform's own,
        # and on Windows its separator is `\`.
        return (
            isinstance(other, str)
            and other.startswith("--jinja_chat_template_override=")
            and Path(other.partition("=")[2]).parts[-2:] == ("templates", "functiongemma.jinja")
        )

    def __repr__(self) -> str:
        return "--jinja_chat_template_override=<packaged functiongemma.jinja>"


_ANY_TEMPLATE = _AnyTemplate()


def test_argv_is_the_documented_command(request_for):
    request = request_for(MEASURED_RECIPES)
    assert request.argv("weight_only_wi8_afp32") == [
        "litert-torch",
        "export_hf",
        "--model=google/functiongemma-270m-it",
        f"--output_dir={request.dir_for('weight_only_wi8_afp32')}",
        "--quantization_recipe=weight_only_wi8_afp32",
        # Added by the family's rules, not by the caller: this model's
        # config.json says `gemma3_text`, which the exporter does not recognise,
        # and without the override the bundle is typed `generic_model` and the
        # runtime creates no tool-call channel.
        "--litert_lm_model_type_override=function_gemma",
        # The template the checkpoint ships with uses `macro` and `dictsort`,
        # which LiteRT-LM's MiniJinja does not support: a bundle carrying it
        # exports cleanly and then fails the native tool path. Compared by
        # suffix because the value is an absolute path into the installed
        # package, which differs per machine.
        _ANY_TEMPLATE,
        # Appended by `ExportRequest` itself, because the caller said nothing
        # about it and the default is on. Every measured artifact carries it,
        # and an export without it is a different shape -- 286 MB against
        # 457 MB for this model -- not a smaller version of the same one.
        "--externalize_embedder",
    ]
    assert request.externalize_source == "litetune default"
    externalised = request_for(MEASURED_RECIPES, externalize_embedder=True)
    # Membership, not position: the family's rules append their own flags after
    # the caller's, so asserting "last" would pin an ordering nothing promises.
    assert "--externalize_embedder" in externalised.argv("dynamic_wi8_afp32")


def test_the_export_argv_carries_no_dtype_flag(request_for):
    """`tune.py`'s limitation strings say "export.py passes no dtype to the
    exporter at all" -- pins the fact, not the prose, the same way
    `test_the_float_reference_loads_at_float32_regardless_of_spec_dtype`
    closes the matching gap on the evaluation side. `test_argv_is_the_
    documented_command` above would incidentally catch a dtype flag added to
    that one recipe's argv too, since it asserts the whole list, but nothing
    states the claim directly, and nothing covers the other measured recipe.
    """
    request = request_for(MEASURED_RECIPES)
    for recipe in MEASURED_RECIPES:
        assert not any("dtype" in flag for flag in request.argv(recipe))


def test_export_uses_the_export_environment(toolchain, request_for):
    request = request_for(("weight_only_wi8_afp32",))
    assert request.env is envs.EXPORT
    run_export(request)
    assert toolchain.exports[0].timeout == request.timeout_s


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_the_stage_reports_progress_without_claiming_verification(toolchain, request_for):
    seen: list[Event] = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    run_export(request_for(MEASURED_RECIPES), events=events)

    kinds = [e.kind for e in seen]
    assert kinds[0] == "stage_started"
    assert kinds[-1] == "stage_finished"
    artifacts = [e for e in seen if e.kind == "artifact_written"]
    assert len(artifacts) == 2
    assert all(e.data["verified"] is False for e in artifacts)
    assert seen[-1].data["verified"] is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_parse_pip_freeze_records_the_whole_resolved_set():
    # Including packages nobody declared -- that is the point of freezing.
    assert parse_pip_freeze(PIP_FREEZE) == {
        "absl-py": "2.1.0",
        "ai-edge-litert-nightly": "2.3.0.dev20260925",
        "ai-edge-quantizer-nightly": "0.10.0.dev20260926",
        "flatbuffers": "24.3.25",
        "litert-converter": "0.4.0",
        "litert-lm": "0.18.0",
        "litert-lm-builder": "0.18.0",
        "litert-torch-nightly": "0.10.0.dev20260926",
        "numpy": "2.0.2",
        "transformers": "5.16.1",
    }


def test_parse_pip_freeze_keeps_unpinnable_entries_verbatim():
    # Editable and URL installs are the hardest to reproduce, so dropping them
    # would lose exactly the lines a post-hoc diagnosis needs most.
    frozen = parse_pip_freeze("-e /src/litert-torch\nfoo @ https://example.invalid/foo.whl\n")
    assert "/src/litert-torch" in frozen["litert-torch"]
    assert frozen["foo"].startswith("foo @ https://")


def test_a_frozen_direct_url_keeps_its_hash_and_loses_its_credential():
    """pip writes the archive's hash in the fragment; it is provenance, not a secret."""
    frozen = parse_pip_freeze(
        "pkg @ https://u:p4ss@h.invalid/pkg-1-py3-none-any.whl?sig=T0KEN#sha256=ab12\n"
    )

    assert frozen["pkg"] == "pkg @ https://***@h.invalid/pkg-1-py3-none-any.whl?***#sha256=ab12"


def test_requirement_names_drop_the_pins():
    assert requirement_names(("litert-lm==0.16.1", "numpy==2.0.2")) == ("litert-lm", "numpy")
    assert requirement_names(("https://example.invalid/wheel.whl",)) == ()


def test_compare_sizes_names_the_reason_it_could_not_compare():
    assert compare_sizes(["only_one"], []).compared is False
    assert "no alternative was measured" in compare_sizes(["only_one"], []).reason


def test_an_unexpected_error_mid_export_is_could_not_check_not_a_crash(
    monkeypatch, toolchain, request_for
):
    # The backstop: anything escaping the export body means the recipe was
    # never established either way, so the stage must not report a verdict on it.
    monkeypatch.setattr(
        "litetune.export._sha256",
        lambda path: (_ for _ in ()).throw(OSError("input/output error")),
    )
    result = run_export(request_for(("weight_only_wi8_afp32",)))

    assert result.exports[0].check.outcome is Outcome.UNCHECKED
    assert result.outcome is Outcome.UNCHECKED
    assert result.failed == []


@pytest.mark.parametrize(("platform", "debian"), [("linux", ["libvulkan1"]), ("darwin", [])])
def test_the_request_names_a_debian_package_only_on_linux(tmp_path, monkeypatch, platform, debian):
    """A `--json` report from a Mac listed `libvulkan1`, a Debian name."""
    # The `sys` envs reads, which is the one `sys` -- named so it stays that.
    monkeypatch.setattr(envs.sys, "platform", platform)
    request = ExportRequest(
        model="google/functiongemma-270m-it",
        output_dir=tmp_path / "out",
        recipes=("dynamic_wi8_afp32",),
    )
    assert request.as_dict()["environment"]["system_requirements"] == debian


def test_the_result_survives_the_trip_through_a_manifest(toolchain, request_for):
    import json

    result = run_export(request_for(MEASURED_RECIPES))
    record = json.loads(json.dumps(result.as_dict()))

    assert record["schema"] == "litetune.export/1"
    assert record["verified"] is False
    assert [e["recipe"] for e in record["exports"]] == list(MEASURED_RECIPES)
    assert record["request"]["environment"]["requirements"]


def test_a_zero_byte_artifact_is_not_compared(toolchain, request_for):
    """`(high - low) / low` guarded by `if low else 0.0` reported no spread.

    A measured-looking number for a comparison that cannot be made, in the
    module that distinguishes those two everywhere else.
    """
    toolchain.runs = {
        "dynamic_wi8_afp32": RecipeRun(writes=(("model.litertlm", 0),)),
        "weight_only_wi8_afp32": RecipeRun(writes=(("model.litertlm", 4096),)),
    }

    result = run_export(request_for(MEASURED_RECIPES))

    assert isinstance(result.comparison, Uncompared)
    assert "zero-byte" in result.comparison.reason


def test_a_tokenizer_path_from_another_machine_is_repaired(tmp_path):
    """`tune` writes an absolute `vocab_file`; the checkpoint then moves.

    `export_lib.export_tokenizer` reads `tokenizer.vocab_file` and opens it
    verbatim, with no resolution against the model directory. So a checkpoint
    trained on one machine and converted on another dies with
    `FileNotFoundError: /tmp/merged/tokenizer.model` -- a path that never
    existed here -- after the model has finished loading.

    Reproduced exactly that way: a checkpoint built on a Linux worker, exported
    from a laptop. The path is a property of where the checkpoint is now, which
    is knowable here and was not knowable there.
    """
    from litetune.export import repair_vocab_file

    model = tmp_path / "merged"
    model.mkdir()
    (model / "tokenizer.model").write_bytes(b"sp")
    config = model / "tokenizer_config.json"
    config.write_text(
        json.dumps({"vocab_file": "/tmp/merged/tokenizer.model", "model_max_length": 8192}),
        encoding="utf-8",
    )

    ok, note = repair_vocab_file(model)

    assert ok
    assert note and "/tmp/merged/tokenizer.model" in note
    written = json.loads(config.read_text(encoding="utf-8"))
    assert written["vocab_file"] == str((model / "tokenizer.model").resolve())
    assert written["model_max_length"] == 8192, "the rest of the config must survive"


def test_a_vocab_file_that_resolves_is_left_alone(tmp_path):
    from litetune.export import repair_vocab_file

    model = tmp_path / "merged"
    model.mkdir()
    real = model / "tokenizer.model"
    real.write_bytes(b"sp")
    config = model / "tokenizer_config.json"
    config.write_text(json.dumps({"vocab_file": str(real)}), encoding="utf-8")

    assert repair_vocab_file(model) == (True, None)
    assert json.loads(config.read_text(encoding="utf-8"))["vocab_file"] == str(real)


def test_a_bpe_checkpoint_needs_no_repair_and_is_not_an_error(tmp_path):
    """Qwen's tokenizer is BPE: there is no `tokenizer.model` to point at."""
    from litetune.export import repair_vocab_file

    model = tmp_path / "merged"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(
        json.dumps({"vocab_file": "/tmp/gone/tokenizer.model"}), encoding="utf-8"
    )

    assert repair_vocab_file(model) == (True, None)


def _stale_vocab_checkpoint(tmp_path) -> Path:
    """A checkpoint whose `tokenizer_config.json` names a `vocab_file` elsewhere."""
    model = _checkpoint(tmp_path)
    (model / "tokenizer.model").write_bytes(b"sp")
    (model / "tokenizer_config.json").write_text(
        json.dumps({"vocab_file": "/tmp/elsewhere/tokenizer.model"}), encoding="utf-8"
    )
    return model


def _files_of(model: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(model.iterdir())}


def _convert_checkpoint(model: Path, tmp_path) -> ExportResult:
    return run_export(
        ExportRequest(
            model=str(model),
            base_model="google/gemma-3-270m-it",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
        )
    )


def test_convert_refused_on_this_machine_writes_nothing_into_the_checkpoint(
    toolchain, monkeypatch, tmp_path
):
    """Windows has no `litert-converter` wheel, and `export` says so before anything else.

    The `vocab_file` repair rewrites `tokenizer_config.json` in the checkpoint
    it was handed, and it used to run -- with the plan drawn first -- before
    provisioning refused, so a run that could never convert here still edited
    the caller's input. The same checkpoint where the environment stands is
    repaired, which is what makes "nothing was written" mean something.
    """
    monkeypatch.setattr(envs.StageEnv, "refusal", _REAL_REFUSAL)
    provisioned: list[str] = []
    monkeypatch.setattr(
        envs.StageEnv,
        "provision",
        lambda self, events=None, force=False: provisioned.append(1) or mark_provisioned(self),
    )
    model = _stale_vocab_checkpoint(tmp_path)
    before = _files_of(model)

    monkeypatch.setattr(envs.wheels, "current_host", lambda: machine("win_amd64", (3, 12)))
    result = _convert_checkpoint(model, tmp_path)

    assert _files_of(model) == before, "the checkpoint is left as it was handed in"
    assert provisioned == [], "and nothing was asked to provision"
    assert result.outcome is Outcome.UNCHECKED
    assert result.not_attempted == ("dynamic_wi8_afp32",)
    detail = result.checks.first_unchecked.detail
    assert detail.startswith("environment 'export' cannot be provisioned on win_amd64")
    assert "litert-converter==0.4.0 publishes no wheel for win_amd64" in detail

    monkeypatch.setattr(envs.wheels, "current_host", lambda: machine("linux_x86_64", (3, 12)))
    _convert_checkpoint(model, tmp_path)
    config = json.loads((model / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert provisioned == [1]
    assert config["vocab_file"] == str(
        (model / "tokenizer.model").resolve()
    ), "where the environment stands, the same checkpoint is repaired"


def _pip_fails(monkeypatch) -> None:
    """The real `provision`, with a virtualenv that builds and a pip that finds nothing."""
    monkeypatch.setattr(
        envs.venv.EnvBuilder, "create", lambda self, path: Path(path).mkdir(parents=True)
    )
    monkeypatch.setattr(
        envs,
        "_run_guarded",
        lambda *a, **k: subprocess.CompletedProcess(
            a[0], 1, "", "ERROR: No matching distribution found for litert-converter==0.4.0\n"
        ),
    )


@pytest.mark.parametrize("how", ["provision-raises", "opted-out-and-pip-fails"])
def test_a_convert_that_could_not_provision_writes_nothing_into_the_checkpoint(
    monkeypatch, tmp_path, how
):
    """The repair waits for a usable environment, on every platform.

    With `LITETUNE_ALLOW_MISSING_WHEELS=1` on Windows the refusal is a note and
    pip is asked; when pip has nothing either, the checkpoint must be as it was.
    """
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    if how == "provision-raises":
        monkeypatch.setattr(envs.wheels, "current_host", lambda: machine("linux_x86_64", (3, 12)))

        def provision(self, events=None, force=False):
            raise RuntimeError("could not provision environment 'export'")

        monkeypatch.setattr(envs.StageEnv, "provision", provision)
    else:
        monkeypatch.setattr(envs.wheels, "current_host", lambda: machine("win_amd64", (3, 12)))
        monkeypatch.setenv(envs.ALLOW_MISSING_WHEELS, "1")
        _pip_fails(monkeypatch)
    model = _stale_vocab_checkpoint(tmp_path)
    before = _files_of(model)

    result = _convert_checkpoint(model, tmp_path)

    assert _files_of(model) == before
    assert result.outcome is Outcome.UNCHECKED
    assert result.not_attempted == ("dynamic_wi8_afp32",)
    assert "could not provision" in result.checks.first_unchecked.detail


_PRIVATE_WHEEL = "https://user:s3cret@packages.invalid/pkg-1-py3-none-any.whl?sig=T0KEN#sha256=ab12"
_PRIVATE_SHOWN = "https://***@packages.invalid/pkg-1-py3-none-any.whl?***#sha256=ab12"


@pytest.mark.parametrize("ready", [False, True], ids=["unready", "ready"])
def test_a_report_carries_no_credential_from_a_requirement(
    toolchain, request_for, monkeypatch, ready
):
    """The manifest is what a user pastes into an issue.

    A direct URL is an accepted pin, and every place the report lists what an
    environment pins -- the request, the toolchain's declaration, the
    environment check -- shows it without its user information or query, and
    with its hash.
    """
    private = envs.StageEnv(name="export", requirements=(_PRIVATE_WHEEL,))
    if ready:
        mark_provisioned(private)

    result = run_export(request_for(("weight_only_wi8_afp32",), env=private, auto_provision=False))
    report = json.dumps(result.as_dict(), default=str)

    assert "s3cret" not in report and "T0KEN" not in report
    assert report.count(_PRIVATE_SHOWN) >= (3 if ready else 2), report


def test_a_pip_freeze_that_failed_is_quoted_without_credentials(monkeypatch, tmp_path):
    """What `pip freeze` printed when it reported nothing reaches the manifest."""
    from litetune.export import resolve_toolchain

    def run(self, args, timeout=3600, **kwargs):
        said = "Looking in indexes: https://u:p4ssw0rd@corp.invalid/simple?token=T0KEN"
        return subprocess.CompletedProcess(args, 1, "", said)

    monkeypatch.setattr(envs.StageEnv, "run", run)

    toolchain = resolve_toolchain(envs.StageEnv(name="x", requirements=("a==1",)))

    reason = toolchain.unresolved_reason or ""
    assert "corp.invalid" in reason
    assert "p4ssw0rd" not in reason and "T0KEN" not in reason


def test_a_ready_export_environment_is_used_on_any_machine(toolchain, request_for, monkeypatch):
    """The refusal is for an environment that would have to be built here.

    One already standing -- built from another index, or by hand -- is used,
    as `provision` uses it; refusing it on PyPI's account would refuse a
    machine that has what it needs.
    """
    from litetune.envs import StageEnv

    monkeypatch.setattr(StageEnv, "refusal", _REAL_REFUSAL)
    monkeypatch.setattr(envs.wheels, "current_host", lambda: machine("win_amd64", (3, 12)))
    mark_provisioned(envs.EXPORT)

    result = run_export(request_for(("weight_only_wi8_afp32",)))

    assert result.not_attempted == ()
    assert toolchain.exports, "the recipe ran"


def test_run_export_repairs_the_path_and_says_so(toolchain, request_for, tmp_path, monkeypatch):
    """The call site, not just the function.

    Three tests can prove `repair_vocab_file` works and none of them notice if
    nobody calls it -- and the call is what disappears in a refactor. This one
    drives `run_export` and asserts both that the repair happened and that the
    run said so: it edits the directory the caller handed in, and a tool that
    modifies your input must report it where you will read it.
    """
    from litetune.export import run_export

    model = tmp_path / "merged"
    model.mkdir()
    (model / "tokenizer.model").write_bytes(b"sp")
    config = model / "tokenizer_config.json"
    config.write_text(
        json.dumps({"vocab_file": "/tmp/elsewhere/tokenizer.model"}), encoding="utf-8"
    )

    request = request_for(("weight_only_wi8_afp32",))
    object.__setattr__(request, "model", str(model))
    result = run_export(request)

    assert json.loads(config.read_text(encoding="utf-8"))["vocab_file"] == str(
        (model / "tokenizer.model").resolve()
    )
    repaired = [line for line in result.limitations if "vocab_file" in line]
    assert repaired, "a rewrite of the caller's own directory must reach the report"
    assert "/tmp/elsewhere/tokenizer.model" in repaired[0]


def _checkpoint(tmp_path, *, model_type="gemma3_text", recorded=None):
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    if recorded is not None:
        (d / "litetune.json").write_text(json.dumps({"base_model": recorded}), encoding="utf-8")
    return d


def test_a_flag_that_contradicts_the_checkpoint_refuses_before_any_work(tmp_path):
    """`--base-model` used to win in silence, and the check said passed.

    One stale copy-pasted command line then exported the wrong family with a
    report actively vouching for it. The recorded value came from the run that
    produced the weights; the flag came from a shell. Neither is discarded.
    """
    from litetune.export import ExportRequest, run_export

    checkpoint = _checkpoint(tmp_path, recorded="google/functiongemma-270m-it")
    request = ExportRequest(
        model=str(checkpoint),
        base_model="google/gemma-3-270m-it",
        output_dir=tmp_path / "out",
        recipes=("dynamic_wi8_afp32",),
    )

    assert request.identity_conflict and "different families" in request.identity_conflict
    result = run_export(request)
    assert result.outcome is Outcome.UNCHECKED
    assert result.not_attempted == ("dynamic_wi8_afp32",)
    assert not result.exports, "nothing may be built under two answers"


def test_a_flag_that_agrees_with_the_checkpoint_is_not_a_conflict(tmp_path):
    from litetune.export import ExportRequest

    checkpoint = _checkpoint(tmp_path, recorded="google/functiongemma-270m-it")
    request = ExportRequest(
        model=str(checkpoint),
        base_model="google/functiongemma-270m-it",
        output_dir=tmp_path / "out",
        recipes=("dynamic_wi8_afp32",),
    )
    assert request.identity_conflict is None
    assert request.plan.rules is not None
    assert request.plan.rules.family == "functiongemma"


def test_the_plan_is_keyed_on_the_identity_while_argv_still_names_the_path(tmp_path):
    """Two values that are easy to swap and would swap silently.

    `plan_export` must see the identity; `litert-torch` must see the directory.
    Exchanging them produces either a plan with no rules or an export of a model
    that is not on disk, and nothing else would notice.
    """
    from litetune.export import ExportRequest

    checkpoint = _checkpoint(tmp_path, recorded="google/functiongemma-270m-it")
    request = ExportRequest(
        model=str(checkpoint),
        output_dir=tmp_path / "out",
        recipes=("dynamic_wi8_afp32",),
    )

    assert request.plan.rules is not None
    assert request.plan.rules.family == "functiongemma"
    assert f"--model={checkpoint}" in request.argv("dynamic_wi8_afp32")


def _cut_short(sidecar):
    sidecar.write_text('{"base_model": "goo', encoding="utf-8")


def _a_directory_sidecar(sidecar):
    sidecar.mkdir()


@pytest.mark.parametrize(
    ("spoil", "said"),
    [
        (_cut_short, "could not be read: JSONDecodeError: "),
        (_a_directory_sidecar, "could not be read: OSError: "),
    ],
    ids=["cut-short", "directory"],
)
def test_an_unreadable_sidecar_reaches_the_report(tmp_path, monkeypatch, spoil, said):
    """Recorded in the hint is not enough: nobody reads a hint."""
    from litetune.export import ExportRequest, run_export

    # About the sidecar, not this machine: on Windows or 3.13 `export` would be
    # refused before the sidecar is read.
    buildable_here(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    spoil(checkpoint / "litetune.json")

    result = run_export(
        ExportRequest(
            model=str(checkpoint),
            base_model="google/functiongemma-270m-it",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
            # Without this the request provisions the real export environment:
            # `pip install litert-torch-nightly ...` from PyPI, on a test that
            # only reads a sidecar. The workflow says the suite reaches no
            # network, and on 3.10 and 3.12 -- where the interpreter ceiling
            # does not refuse first -- it did.
            auto_provision=False,
        )
    )
    assert any(said in line for line in result.limitations), result.limitations


# ---------------------------------------------------------------------------
# GPU activation type
# ---------------------------------------------------------------------------


def _fake_env(toolchain, tmp_path):
    """A StageEnv whose `run` is the fake; the repack only needs `run`."""
    from litetune import envs

    return envs.StageEnv(name="export", requirements=())


def _artifact(tmp_path, size=16384) -> Path:
    path = tmp_path / "model.litertlm"
    path.write_bytes(b"\0" * size)
    return path


def _toml_in(artifact: Path) -> str:
    """The TOML inside the fake bundle (see FAKE_BUILDER_MODULE). Readable only
    from a file the fake `pack` wrote -- so only after the original was replaced."""
    raw = artifact.read_bytes()
    assert raw.startswith(b"LITERTLM"), "not a repacked fake bundle: was the original replaced?"
    return raw[len(b"LITERTLM") :].rstrip(b"\0").decode("utf-8")


def _toml_loads(text: str):
    """tomllib on 3.11+, tomli below -- the same fallback the repack script
    uses, resolved by name so mypy's ignore codes do not differ per version."""
    import importlib

    try:
        mod = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        mod = importlib.import_module("tomli")
    return mod.loads(text)


def _sections(toml: str) -> list[str]:
    return toml.split("[[section]]")[1:]


def test_the_activation_type_is_written_into_the_prefill_decode_section(toolchain, tmp_path):
    """Measured 2026-09-05: without this key a Galaxy S24 floods `<pad>` on GPU
    (14/20) and gets 3/20 tool names; with it, 0/20 and 20/20. The engine
    reports success either way, so the artifact has to carry the fix.
    """
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None)
    assert not [p for p in artifact.parent.iterdir() if p.name.startswith(".")], "work dir gone"
    assert artifact.read_bytes()[:8] == b"LITERTLM", "the rebuilt file replaced the original"
    (prefill,) = [t for t in _sections(_toml_in(artifact)) if '"prefill_decode"' in t]
    (embedder,) = [t for t in _sections(_toml_in(artifact)) if '"embedder"' in t]
    assert 'key = "prefer_activation_type"' in prefill and f'value = "{GPU_ACTIVATION}"' in prefill
    assert "prefer_activation_type" not in embedder, "the key belongs to prefill_decode only"


def test_an_artifact_given_by_a_relative_path_is_repacked_too(toolchain, tmp_path, monkeypatch):
    """`convert --output-dir artifacts/dyn` on Python 3.10 or 3.11: the work
    directory came out relative, the script wrote `data_path`s relative to the
    current directory, and the builder resolved each against the TOML's own
    directory -- `artifacts/dyn/.model-repack-X/artifacts/dyn/.model-repack-X/...`
    -- so every conversion to a relative output directory shipped a bundle
    without the GPU key.

    Only below Python 3.12: `tempfile.mkdtemp(dir="rel")` returned
    `rel/tmp...` on 3.10.19 and 3.11.14 and an absolute path on 3.12.15
    (measured 2026-10-08). The relative return is put back here so the case
    is tested on every Python litetune supports."""
    import tempfile

    from litetune import export
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    real_mkdtemp = tempfile.mkdtemp
    calls = []

    def mkdtemp_before_3_12(suffix=None, prefix=None, dir=None):
        calls.append(dir)
        made = real_mkdtemp(suffix, prefix, dir)
        return made if dir is None or os.path.isabs(dir) else os.path.relpath(made)

    monkeypatch.setattr(export.tempfile, "mkdtemp", mkdtemp_before_3_12)
    out = tmp_path / "artifacts" / "dyn"
    out.mkdir(parents=True)
    _artifact(out)
    monkeypatch.chdir(tmp_path)

    value, note = set_gpu_activation(
        Path("artifacts/dyn/model.litertlm"), _fake_env(toolchain, tmp_path)
    )

    # The simulation reached the call; without this the test would pass
    # unguarded on 3.12 if `mkdtemp` were ever bound some other way.
    assert calls, "set_gpu_activation never called tempfile.mkdtemp"
    assert (value, note) == (GPU_ACTIVATION, None)
    assert (out / "model.litertlm").read_bytes()[:8] == b"LITERTLM", "the original was replaced"
    assert not [p for p in out.iterdir() if p.name.startswith(".")], "work dir gone"


def test_a_symlinked_artifact_is_replaced_and_its_target_left_alone(toolchain, tmp_path):
    """`absolute()`, not `resolve()`: the path is made absolute without
    following a link, so the file replaced is the one the caller named and the
    link's target, which may sit outside the output directory, is untouched."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = _artifact(elsewhere)
    before = target.read_bytes()
    out = tmp_path / "out"
    out.mkdir()
    link = out / "model.litertlm"
    os.symlink(target, link)

    value, note = set_gpu_activation(link, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None)
    assert not link.is_symlink(), "the link itself is the file replaced"
    assert "prefer_activation_type" in _toml_in(link)
    assert target.read_bytes() == before, "the link's target is left alone"
    assert not [p for p in elsewhere.iterdir() if p.name.startswith(".")]


def test_a_bundle_that_already_declares_an_activation_type_is_left_alone(toolchain, tmp_path):
    from litetune.export import set_gpu_activation

    toolchain.repack_toml = FAKE_TOML.replace(
        'model_type = "prefill_decode"\n',
        'model_type = "prefill_decode"\nprefer_activation_type = "fp32_fp16"\n',
    )
    artifact = _artifact(tmp_path)
    before = artifact.read_bytes()

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value == "fp32_fp16", "what the bundle carries, not what was asked for"
    assert note and "fp32_fp16" in note and "left as written" in note
    assert artifact.read_bytes() == before


@pytest.mark.parametrize(
    "knob, expect",
    [
        ("repack_unpack_fails", "exited 1: RuntimeError: unpack: boom"),
        ("repack_pack_fails", "exited 1: RuntimeError: pack: boom"),
        ("repack_script_returncode", "the repack script exited 1: Traceback: boom"),
        ("repack_pack_drifts", "beyond the one key"),
        ("repack_pack_misplaces", "beyond the one key"),
    ],
)
def test_every_repack_failure_keeps_the_original_and_says_why(toolchain, tmp_path, knob, expect):
    """The original is a working CPU bundle. Losing it to a failed repack would
    turn a GPU limitation into no artifact at all."""
    from litetune.export import set_gpu_activation

    setattr(toolchain, knob, 1 if knob.endswith("returncode") else True)
    artifact = _artifact(tmp_path)
    before = artifact.read_bytes()

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value is None
    assert note and expect in note, note
    assert artifact.read_bytes() == before
    assert not [p for p in artifact.parent.iterdir() if p.name.startswith(".")], "work dir gone"


def test_a_toml_without_a_prefill_section_is_not_guessed_at(toolchain, tmp_path):
    from litetune.export import set_gpu_activation

    toolchain.repack_toml = FAKE_TOML.replace('model_type = "prefill_decode"', 'model_type = "aux"')
    artifact = _artifact(tmp_path)
    before = artifact.read_bytes()

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value is None
    assert note and "found 0 prefill_decode sections" in note
    assert artifact.read_bytes() == before


def test_two_prefill_sections_are_ambiguous_rather_than_a_guess(toolchain, tmp_path):
    from litetune.export import set_gpu_activation

    toolchain.repack_toml = FAKE_TOML + (
        '\n[[section]]\nmodel_type = "prefill_decode"\nsection_type = "TFLiteModel"\n'
        'data_path = "second.tflite"\n'
    )
    artifact = _artifact(tmp_path)
    before = artifact.read_bytes()

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value is None
    assert note and "found 2 prefill_decode sections" in note
    assert artifact.read_bytes() == before


def test_run_export_repacks_every_artifact_and_records_it(toolchain, request_for):
    """The call site: the recorded hash and size are of the file that ships."""
    from litetune.export import GPU_ACTIVATION, run_export

    result = run_export(request_for(MEASURED_RECIPES))

    assert [e.gpu_activation for e in result.succeeded] == [GPU_ACTIVATION, GPU_ACTIVATION]
    for export in result.succeeded:
        assert export.check.observed["gpu_activation"] == GPU_ACTIVATION
        assert f"GPU activations {GPU_ACTIVATION}" in export.check.detail
        assert export.gpu_activation_note is None
        assert export.as_dict()["gpu_activation"] == GPU_ACTIVATION
        assert export.as_dict()["gpu_activation_state"] == "set"
        assert export.sha256 == hashlib.sha256(export.artifact.read_bytes()).hexdigest()
    assert not any("could not be written" in text for text in result.limitations)


def test_a_failed_repack_is_a_limitation_on_a_passed_export(toolchain, request_for):
    """CPU-only is a smaller thing than not exported. The check stays passed; the
    console detail carries the reason; the report names the recipe."""
    from litetune.export import run_export

    toolchain.repack_unpack_fails = True

    result = run_export(request_for(("dynamic_wi8_afp32",)))

    (export,) = result.succeeded
    assert export.ok
    assert export.gpu_activation is None
    assert (
        "GPU activations not set: an app that passes none gets F16 on the GPU: "
        "the repack script exited 1: "
        "RuntimeError: unpack: boom" in export.check.detail
    )
    assert export.gpu_activation_note and "unpack: boom" in export.gpu_activation_note
    assert export.as_dict()["gpu_activation_state"] == "unset"
    assert any(
        "dynamic_wi8_afp32 (the repack script exited 1: RuntimeError: unpack: boom" in text
        and "<pad>" in text
        for text in result.limitations
    )


def test_an_exception_inside_the_repack_cannot_unmake_the_export(
    toolchain, request_for, monkeypatch
):
    """`guard` turns an escaping exception into "could not check". A repack that
    raises would then bury a perfectly good CPU export under that verdict, so
    nothing may escape it."""
    from litetune import envs
    from litetune.export import run_export

    def explode(self, args, timeout=3600, **kwargs):
        if args[0] == "python":
            raise KeyError("quantization_recipe")
        return toolchain(args, timeout, **kwargs)

    monkeypatch.setattr(envs.StageEnv, "run", explode)

    result = run_export(request_for(("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.ok, export.check.detail
    assert export.gpu_activation is None
    assert export.gpu_activation_note and "KeyError" in export.gpu_activation_note


def test_a_work_dir_that_cannot_be_created_is_a_note_not_an_unchecked_export(
    toolchain, request_for, monkeypatch
):
    """Creating the work dir happens inside the try: a permission error there
    is a note on a passed export, not an escape into `guard`."""
    import tempfile

    from litetune.export import run_export

    def refuse(*args, **kwargs):
        raise PermissionError("no")

    monkeypatch.setattr(tempfile, "mkdtemp", refuse)

    result = run_export(request_for(("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.ok, export.check.detail
    assert export.gpu_activation is None
    assert export.gpu_activation_note and "PermissionError" in export.gpu_activation_note


def test_the_repack_is_charged_against_the_export_timeout(toolchain, request_for):
    """`--timeout-s` bounds export and repack together: the repack gets what is
    left of the recipe's budget, never a fresh 300 s on top."""
    from litetune.export import REPACK_TIMEOUT_S, run_export

    run_export(request_for(("dynamic_wi8_afp32",), timeout_s=7))

    (repack,) = toolchain.repacks
    assert repack.timeout <= 7 < REPACK_TIMEOUT_S


def test_a_key_the_builder_wrote_into_additional_metadata_is_recognised(toolchain, tmp_path):
    """Real `unpack` round-trips the key as one entry of the section's
    `additional_metadata` array, not as a bare line."""
    from litetune.export import set_gpu_activation

    toolchain.repack_toml = FAKE_TOML.replace(
        '[[section]]\nmodel_type = "prefill_decode"\n',
        "[[section]]\nadditional_metadata = [\n"
        '  { key = "prefer_activation_type", value_type = "String", value = "fp32" },\n'
        ']\nmodel_type = "prefill_decode"\n',
    )
    artifact = _artifact(tmp_path)
    before = artifact.read_bytes()

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value == "fp32"
    assert note and "= fp32;" in note and "left as written" in note
    assert artifact.read_bytes() == before


def test_the_key_is_placed_by_section_not_by_line_order(toolchain, tmp_path):
    """Key order in a TOML table is free and the prefill section need not be
    first; a parser, not a regex, decides which table is which."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    toolchain.repack_toml = (
        "[system_metadata]\nentries = []\n\n"
        '[[section]]\nmodel_type = "embedder"\nsection_type = "TFLiteModel"\n'
        'data_path = "embedder.tflite"\n\n'
        '[[section]]\ndata_path = "prefill_decode.tflite"\nsection_type = "TFLiteModel"\n'
        'additional_metadata = [\n  { key = "License", value_type = "String", value = "x" },\n]\n'
        'model_type = "prefill_decode"\n'
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None)
    (embedder, prefill) = _sections(_toml_in(artifact))
    assert "prefer_activation_type" not in embedder
    assert f'value = "{GPU_ACTIVATION}"' in prefill
    assert 'key = "License"' in prefill, "existing metadata survives"


@pytest.mark.parametrize("declared", ["fp16", "fp32_fp16"])
def test_an_upstream_declaration_is_recorded_as_itself_not_as_fp32(
    toolchain, request_for, declared
):
    """`--flag=--experimental_use_mixed_precision` produces `fp32_fp16`; an
    explicit `fp16` reproduces the fault. Both are left as found, and the
    record must say what the bundle carries."""
    from litetune.export import GPU_ACTIVATION, run_export

    toolchain.repack_toml = FAKE_TOML.replace(
        'model_type = "prefill_decode"\n',
        f'model_type = "prefill_decode"\nprefer_activation_type = "{declared}"\n',
    )

    result = run_export(request_for(("dynamic_wi8_afp32",)))

    (export,) = result.succeeded
    assert export.ok
    assert export.gpu_activation == declared
    assert export.as_dict()["gpu_activation"] == declared
    assert export.as_dict()["gpu_activation_state"] == "declared_upstream"
    assert f"GPU activations {declared} (declared upstream, not {GPU_ACTIVATION})" in (
        export.check.detail
    )
    assert any(declared in text and "left as found" in text for text in result.limitations)


def test_an_empty_upstream_declaration_is_reported_as_empty_not_as_unset(toolchain, tmp_path):
    from litetune.export import set_gpu_activation

    toolchain.repack_toml = FAKE_TOML.replace(
        'model_type = "prefill_decode"\n',
        'model_type = "prefill_decode"\nprefer_activation_type = ""\n',
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value == ""
    assert note and "= (empty);" in note and "left as written" in note


def test_toml_syntax_inside_a_string_value_is_not_structure(toolchain, tmp_path):
    """The reason this is a parser and not a regex. A description containing
    `[[section]]` and a `prefer_activation_type` line must be read as text:
    the real section still gets the key, and nothing is read out of the
    string. Found by an independent review of the regex version, which
    matched inside the string."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    toolchain.repack_toml = FAKE_TOML.replace(
        'model_type = "embedder"\n',
        'model_type = "embedder"\n'
        'description = """\n[[section]]\nmodel_type = "prefill_decode"\n'
        'prefer_activation_type = "fp16"\n"""\n',
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None)
    tables = _sections(_toml_in(artifact))
    real_prefill = [
        t
        for t in tables
        if t.lstrip().startswith(("model_type", "prefer", "data", "section", "additional"))
    ]
    assert any(f'value = "{GPU_ACTIVATION}"' in t and '"prefill_decode"' in t for t in real_prefill)


def test_a_trailing_system_table_is_not_a_section(toolchain, tmp_path):
    """`unpack` may write `[system_metadata]` after the last `[[section]]`; a
    system-level key of the same name is not the prefill section's."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    toolchain.repack_toml = (
        '[[section]]\nmodel_type = "prefill_decode"\nsection_type = "TFLiteModel"\n'
        'data_path = "p.tflite"\n\n[system_metadata]\nentries = [\n'
        '  { key = "prefer_activation_type", value_type = "String", value = "fp16" },\n]\n'
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None), "the section had no key; it must be written"


def test_the_embedded_script_is_valid_python():
    """It ships as a string; a syntax error would surface only at repack time,
    as a note, on every bundle."""
    from litetune.export import _REPACK_SCRIPT

    compile(_REPACK_SCRIPT, "repack.py", "exec")


def test_an_empty_artifact_is_not_repacked(toolchain, tmp_path):
    """A zero-byte export is the sweep's size comparison's finding to make; a
    repack that turned it into a few header bytes would hide it."""
    from litetune.export import set_gpu_activation

    artifact = _artifact(tmp_path, size=0)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value is None
    assert note == "the artifact is empty; nothing to repack"
    assert artifact.stat().st_size == 0
    assert toolchain.repacks == [], "no script was run"


def test_metadata_strings_survive_the_repack_whatever_they_contain(toolchain, tmp_path):
    """An emoji in any metadata value sank the first version: `json.dumps`
    writes a non-BMP character as a UTF-16 surrogate pair, TOML forbids
    surrogates, and the builder's parser rejected the rebuilt TOML -- so the
    bundle stayed CPU-only for a reason nothing named. The script runs here
    with a real TOML parser, so this is a real round-trip. Found in review."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    nasty = 'rocket \U0001f680 quote \\" back \\\\ tab\\t triple \\"\\"\\" accent é cjk 中'
    toolchain.repack_toml = FAKE_TOML.replace(
        "[system_metadata]\nentries = []\n",
        "[system_metadata]\nentries = [\n"
        f'  {{ key = "description", value_type = "String", value = "{nasty}" }},\n]\n',
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None), note
    doc = _toml_loads(_toml_in(artifact))
    (desc,) = [e["value"] for e in doc["system_metadata"]["entries"] if e["key"] == "description"]
    assert "\U0001f680" in desc and '"""' in desc and "\u00e9" in desc, "the value came back intact"


def test_the_timeout_charges_elapsed_export_time(toolchain, request_for, monkeypatch):
    """`remaining = timeout_s - seconds`: the repack gets what the export left,
    floored at 1 so subprocess never sees timeout=0. Pinned by driving the
    clock, not by hoping `min` alone is the arithmetic."""
    import itertools

    from litetune import export as export_module
    from litetune.export import run_export

    # perf_counter is read twice per recipe: start, then after the export.
    ticks = itertools.chain([100.0, 103.0], itertools.repeat(103.0))
    monkeypatch.setattr(export_module.time, "perf_counter", lambda: next(ticks))

    run_export(request_for(("dynamic_wi8_afp32",), timeout_s=7))

    (repack,) = toolchain.repacks
    assert repack.timeout == 4, "7 s budget minus 3 s of export"


def test_an_export_that_used_its_whole_budget_still_gets_one_second(
    toolchain, request_for, monkeypatch
):
    import itertools

    from litetune import export as export_module
    from litetune.export import run_export

    ticks = itertools.chain([100.0, 120.0], itertools.repeat(120.0))
    monkeypatch.setattr(export_module.time, "perf_counter", lambda: next(ticks))

    run_export(request_for(("dynamic_wi8_afp32",), timeout_s=7))

    (repack,) = toolchain.repacks
    assert repack.timeout == 1


def test_a_weights_section_with_the_same_model_type_is_not_a_prefill_section(toolchain, tmp_path):
    """Real `peek` writes `model_type` on TFLiteWeights sections too; a bundle
    with external weights carries two `prefill_decode`-typed sections, and the
    `section_type == "TFLiteModel"` filter is what keeps the count at one."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    toolchain.repack_toml = FAKE_TOML + (
        '\n[[section]]\nmodel_type = "prefill_decode"\nsection_type = "TFLiteWeights"\n'
        'data_path = "weights.bin"\n'
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None), note
    tables = _sections(_toml_in(artifact))
    (weights,) = [t for t in tables if '"TFLiteWeights"' in t]
    assert "prefer_activation_type" not in weights


@pytest.mark.parametrize("shape", ["bare", "additional_metadata"])
def test_an_upstream_empty_declaration_in_either_toml_shape(toolchain, tmp_path, shape):
    """Real `peek` writes a null entry as `""` inside `additional_metadata`;
    a hand-written TOML may carry a bare key. Both read back as empty, not unset."""
    from litetune.export import set_gpu_activation

    if shape == "bare":
        extra = 'prefer_activation_type = ""\n'
    else:
        extra = (
            "additional_metadata = [\n"
            '  { key = "prefer_activation_type", value_type = "String", value = "" },\n]\n'
        )
    toolchain.repack_toml = FAKE_TOML.replace(
        'model_type = "prefill_decode"\n', 'model_type = "prefill_decode"\n' + extra
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert value == ""
    assert note and "= (empty);" in note


def test_non_string_metadata_values_survive_the_rebuild(toolchain, tmp_path):
    """The real format carries Int32/Bool values; `toml_value` must re-emit
    them as TOML numbers and booleans, not as quoted strings."""
    from litetune.export import GPU_ACTIVATION, set_gpu_activation

    toolchain.repack_toml = FAKE_TOML.replace(
        "[system_metadata]\nentries = []\n",
        "[system_metadata]\nentries = [\n"
        '  { key = "layers", value_type = "Int32", value = 18 },\n'
        '  { key = "quantized", value_type = "Bool", value = true },\n]\n',
    )
    artifact = _artifact(tmp_path)

    value, note = set_gpu_activation(artifact, _fake_env(toolchain, tmp_path))

    assert (value, note) == (GPU_ACTIVATION, None), note
    entries = {
        e["key"]: e["value"] for e in _toml_loads(_toml_in(artifact))["system_metadata"]["entries"]
    }
    assert entries["layers"] == 18 and entries["quantized"] is True


# ---------------------------------------------------------------------------
# Recipes litetune defines itself
# ---------------------------------------------------------------------------

OWNED = "dynamic_wi4b32_emb8_afp32"


def test_an_owned_recipe_hands_the_toolchain_its_packaged_file(request_for):
    request = request_for((OWNED,))
    argv = request.argv(OWNED)

    (flag,) = [a for a in argv if a.startswith("--quantization_recipe=")]
    path = Path(flag.split("=", 1)[1])
    assert path.name == f"{OWNED}.json"
    assert path.is_file()
    # The name, not the path, is what the caller sees on disk.
    assert request.dir_for(OWNED).name == OWNED
    assert f"--output_dir={request.dir_for(OWNED)}" in argv


@pytest.mark.parametrize("recipe", ["weight_only_wi8_afp32", "dynamic_wi4b64_afp32"])
def test_other_recipe_names_reach_the_toolchain_as_given(request_for, recipe):
    assert f"--quantization_recipe={recipe}" in request_for((recipe,)).argv(recipe)


def test_the_report_carries_the_owned_recipe_and_its_rules(toolchain, request_for):
    result = run_export(request_for(("dynamic_wi8_afp32", OWNED)))

    owned = next(e for e in result.exports if e.recipe == OWNED)
    record = owned.as_dict()
    assert owned.ok
    assert record["recipe_source"] == "litetune"
    assert sorted(rule["operation"] for rule in record["recipe_definition"]) == [
        "EMBEDDING_LOOKUP",
        "FULLY_CONNECTED",
    ]
    preset = next(e for e in result.exports if e.recipe == "dynamic_wi8_afp32").as_dict()
    assert preset["recipe_source"] == "toolchain"
    assert preset["recipe_definition"] is None
    # What ran is the file, under the recipe's own directory.
    (owned_call,) = [
        c.argv for c in toolchain.exports if any(a.endswith(f"{OWNED}.json") for a in c.argv)
    ]
    assert any(
        Path(a.partition("=")[2]).name == OWNED for a in owned_call if a.startswith("--output_dir=")
    )


def test_an_owned_recipe_is_not_flagged_as_passed_through(toolchain, request_for):
    result = run_export(request_for((OWNED, "dynamic_wi4b64_afp32")))

    passed_through = [text for text in result.limitations if "passed through" in text]
    assert len(passed_through) == 1
    named = passed_through[0].split(" are not among")[0]
    assert "dynamic_wi4b64_afp32" in named
    assert OWNED not in named
    # And it now appears on the other side of that sentence: among the recipes
    # litetune has measured, which is what the README's table says of it.
    assert OWNED in passed_through[0]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "missing"),
        ("not json", "could not be read"),
        ('{"operation": "FULLY_CONNECTED"}', "not a list of quantization rules"),
        # Each conjunct of the shape check earns its own case: an empty list and
        # a rule with no operation both validated while only the isinstance was
        # pinned, and either would hand the toolchain a file that quantizes
        # nothing while the report vouched for the rules it did not apply.
        ("[]", "not a list of quantization rules"),
        ('[{"regex": ".*"}]', "not a list of quantization rules"),
    ],
)
def test_an_owned_recipe_that_cannot_be_read_is_not_exported_in_anothers_place(
    toolchain, request_for, tmp_path, monkeypatch, content, reason
):
    from dataclasses import replace

    from litetune.recipes import DEFINED_RECIPES

    target = tmp_path / "recipes" / f"{OWNED}.json"
    if content is not None:
        target.parent.mkdir()
        target.write_text(content, encoding="utf-8")
    monkeypatch.setitem(DEFINED_RECIPES, OWNED, replace(DEFINED_RECIPES[OWNED], path=target))

    result = run_export(request_for((OWNED, "dynamic_wi8_afp32")))

    owned = next(e for e in result.exports if e.recipe == OWNED)
    assert owned.check.outcome is Outcome.UNCHECKED
    assert OWNED in owned.check.detail
    assert reason in owned.check.detail
    assert not owned.attempted
    exported = [c.argv for c in toolchain.exports]
    assert not any(f"--quantization_recipe={target}" in argv for argv in exported)
    assert any("--quantization_recipe=dynamic_wi8_afp32" in argv for argv in exported)
    assert next(e for e in result.exports if e.recipe == "dynamic_wi8_afp32").ok


# ---------------------------------------------------------------------------
# The checkpoint's own chat template, behind the text-parts block
# ---------------------------------------------------------------------------

QWEN3 = "Qwen/Qwen3-0.6B"
QWEN25_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "qwen2.5-0.5b-instruct-7ae5576.chat_template.jinja"
).read_text(encoding="utf-8")
_BLOCK = Path(models.packaged_template(models.TEXT_PARTS_BLOCK)).read_bytes()
# A template written for parts: it reads the first part's text and nothing else.
PARTS_NATIVE = "{%- for m in messages %}{{ m.content[0].text }}{% endfor %}"
# One that reads both.
TOLERANT = (
    "{%- for m in messages %}{% if m.content is string %}{{ m.content }}"
    "{% else %}{{ m.content[0].text }}{% endif %}{% endfor %}"
)


@pytest.fixture
def qwen_request(tmp_path):
    def _build(model=QWEN3, recipes=MEASURED_RECIPES, **kwargs) -> ExportRequest:
        return ExportRequest(model=model, output_dir=tmp_path / "out", recipes=recipes, **kwargs)

    return _build


def _template_flags(argv: Sequence[str]) -> list[str]:
    return [a for a in argv if a.startswith(f"{models.TEMPLATE_FLAG}=")]


def _chat_template_check(result):
    (check,) = [c for c in result.checks.checks if c.name == CHAT_TEMPLATE_CHECK]
    return check


def _composed_files(directory: Path) -> list[str]:
    """Every file a composition left in the output directory, by name."""
    return sorted(p.name for p in directory.glob(f"{COMPOSED_TEMPLATE_PREFIX}*"))


@pytest.mark.parametrize(
    ("model", "carried", "canonical"),
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", QWEN25_TEMPLATE, None),
        # A fine-tune whose name still contains the size: claimed by the qwen-3
        # rule, and its own template, which LiteRT-LM's does not render like,
        # is the one composed.
        (
            "someorg/Qwen3-0.6B-sql-assistant",
            "{%- for m in messages %}{{ m.content }}{% endfor %}",
            models.QWEN3_CANONICAL_TEMPLATE,
        ),
    ],
    ids=["qwen2.5", "fine-tune"],
)
def test_a_qwen_export_carries_its_checkpoints_template_behind_the_block(
    toolchain, qwen_request, tmp_path, model, carried, canonical
):
    """What every recipe's exporter is handed in place of the marker: one file,
    written once for the sweep and named for its content, holding the block and
    then the template the checkpoint's own tokenizer carries -- and what each
    bundle is read back to carry."""
    toolchain.chat_template = carried
    request = qwen_request(model)
    result = run_export(request)

    assert result.outcome is Outcome.PASSED
    composed_bytes = _BLOCK + carried.encode("utf-8")
    digest = hashlib.sha256(composed_bytes).hexdigest()
    composed = tmp_path / "out" / f"{COMPOSED_TEMPLATE_PREFIX}{digest[:12]}.jinja"
    assert composed.read_bytes() == composed_bytes
    assert _composed_files(tmp_path / "out") == [composed.name], "no partial file is left"
    check = _chat_template_check(result)
    assert check.outcome is Outcome.PASSED
    sha256 = hashlib.sha256(carried.encode("utf-8")).hexdigest()
    assert check.observed["template_sha256"] == sha256
    assert check.observed["template_source"] == "tokenizer"
    assert check.observed["composed"] == str(composed)
    assert str(composed) in check.detail and sha256[:12] in check.detail
    # The record names the file and both digests.
    assert result.chat_template is not None
    assert result.as_dict()["chat_template"] == {
        "path": str(composed),
        "template_sha256": sha256,
        "composed_sha256": digest,
        "source": "composed",
        "canonical": canonical,
    }
    # Read once, from what the exporter is given as --model, in the stage's time.
    (composition,) = toolchain.compositions
    assert composition.timeout == request.timeout_s
    assert toolchain.template_loads == [model]
    assert len(toolchain.exports) == 2
    for call in toolchain.exports:
        assert _template_flags(call.argv) == [f"{models.TEMPLATE_FLAG}={composed}"]
    for export in result.exports:
        assert f"{models.TEMPLATE_FLAG}={composed}" in export.argv
        assert export.check.observed["chat_template_sha256"] == digest
    assert not any(models.COMPOSED_TEMPLATE in a for c in toolchain.calls for a in c.argv)


def test_the_composed_file_is_the_block_and_the_template_byte_for_byte(
    toolchain, qwen_request, tmp_path
):
    """Nothing between the two and nothing translated: a non-ASCII character
    and the missing final newline reach the file, and the bundle, as they were."""
    toolchain.chat_template = "{#- é -#}\n{%- for m in messages %}{{ m.content }}\n{% endfor %}"
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    assert result.outcome is Outcome.PASSED
    assert result.chat_template is not None
    written = result.chat_template.path.read_bytes()
    assert written == _BLOCK + toolchain.chat_template.encode("utf-8")
    observed = _chat_template_check(result).observed
    assert observed["template_characters"] == len(toolchain.chat_template)
    assert observed["composed_sha256"] == hashlib.sha256(written).hexdigest()
    (artifact,) = toolchain.embedded
    assert toolchain.embedded[artifact] == written.decode("utf-8")


def test_named_templates_compose_the_one_named_default(toolchain, qwen_request, tmp_path):
    """transformers keeps several named templates as a dict and, given no
    tools, applies "default" (`get_chat_template`)."""
    toolchain.chat_template = {
        "tool_use": "{{ 'tools' }}",
        "default": "{{ messages[0].content }}",
    }
    result = run_export(qwen_request())

    assert result.outcome is Outcome.PASSED
    assert result.chat_template is not None
    assert result.chat_template.path.read_bytes() == _BLOCK + b"{{ messages[0].content }}"
    assert _chat_template_check(result).observed["template_names"] == ["default", "tool_use"]


@pytest.mark.parametrize(
    ("carried", "says"),
    [
        (None, "the tokenizer carries no chat template"),
        ("", "the tokenizer's chat template is empty (a str)"),
        # What transformers makes of `"chat_template": []` (`__init__`, :1079-1082).
        ({}, "the tokenizer's chat template is empty (a dict)"),
        (
            {"tool_use": "{{ 'a' }}", "rag": "{{ 'b' }}"},
            "holds 2 named templates (rag, tool_use) and none named 'default'",
        ),
        ({"default": "", "tool_use": "{{ 'a' }}"}, "chat template named 'default' is empty"),
        (" \n\t", "is 3 characters of whitespace"),
        ("{{ messages[0].content }}\r\n", "contains a carriage return"),
        # The block is not for a template that reads parts itself.
        (PARTS_NATIVE, "given one user message as a string, does not print its text"),
        (
            "{{ raise_exception('no') }}",
            "raises on one user message given as a string (TemplateError: no) -- an error of "
            "the template's own",
        ),
        # One that reads a name the block defines prints differently behind it.
        (
            "{{ messages[0].content }}{% if litetune is defined %}!{% endif %}",
            "this template does not render that way",
        ),
    ],
    ids=[
        "none",
        "empty",
        "empty-dict",
        "unnamed",
        "empty-default",
        "blank",
        "carriage-return",
        "parts-native",
        "raises",
        "renders-differently",
    ],
)
def test_a_checkpoint_with_no_template_to_compose_refuses_and_exports_nothing(
    toolchain, qwen_request, tmp_path, carried, says
):
    """A fact about the checkpoint, so `failed`; never an export without the
    block, and never one with the marker in it."""
    toolchain.chat_template = carried
    result = run_export(qwen_request())

    check = _chat_template_check(result)
    assert check.outcome is Outcome.FAILED
    assert says in check.detail
    # The advice is a command line `convert` accepts as typed, and it names the
    # block a caller's own template would need in front.
    assert f"pass --flag={models.TEMPLATE_FLAG}=<file>" in check.detail
    assert models.packaged_template(models.TEXT_PARTS_BLOCK) in check.detail
    assert result.outcome is Outcome.FAILED
    assert toolchain.exports == []
    assert result.exports == [] and result.failed == []
    assert result.not_attempted == tuple(MEASURED_RECIPES)
    assert result.chat_template is None
    assert any(text.startswith("none of") and says in text for text in result.limitations)
    assert _composed_files(tmp_path / "out") == []


def test_the_probe_quotes_what_the_template_rendered(toolchain, qwen_request):
    toolchain.chat_template = "{{ messages[0].content }}{% if litetune is defined %}!{% endif %}"
    detail = _chat_template_check(run_export(qwen_request())).detail
    assert f"printed '{PROBE_TEXT}'" in detail
    assert f"printed '{PROBE_TEXT}!'" in detail
    assert "the checkpoint's side through transformers' apply_chat_template" in detail
    assert "Jinja2 configured as LiteRT-LM's MiniJinja" in detail
    assert "not through LiteRT-LM itself" in detail


@pytest.mark.parametrize(
    ("carried", "alone"), [(QWEN3_TEMPLATE, False), (TOLERANT, True)], ids=["qwen3", "tolerant"]
)
def test_a_template_that_reads_both_passes_and_says_the_block_was_not_needed(
    toolchain, qwen_request, carried, alone
):
    """The block is harmless in front of a template that already reads a list
    of parts; the record says whether it needed it."""
    toolchain.chat_template = carried
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    assert result.outcome is Outcome.PASSED
    assert _chat_template_check(result).observed["renders_parts_without_the_block"] is alone


def test_with_no_template_the_refusal_says_what_the_exporter_would_embed(toolchain, qwen_request):
    """`export_lib.py` tries `chat_template.json`; with none, the exporter embeds
    no template and LiteRT-LM 0.18.0 makes its own default."""
    toolchain.chat_template = None
    detail = _chat_template_check(run_export(qwen_request())).detail
    assert "carries no chat template, and its repository has no chat_template.json" in detail
    assert "the exporter would embed no template" in detail
    assert "LiteRT-LM 0.18.0 would put its own default for the model type" in detail
    assert "empty template" not in detail


def test_a_hub_id_with_no_template_falls_back_to_its_chat_template_json(toolchain, qwen_request):
    """As `export_lib.py:390-402` does for a repo id: `hf_hub_download` of
    chat_template.json, and its "chat_template"."""
    toolchain.chat_template = None
    toolchain.hub_files = {"chat_template.json": json.dumps({"chat_template": TOLERANT})}
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    assert result.outcome is Outcome.PASSED
    assert result.chat_template is not None
    assert result.chat_template.path.read_bytes() == _BLOCK + TOLERANT.encode("utf-8")
    observed = _chat_template_check(result).observed
    assert observed["template_source"] == "chat_template.json"
    assert observed["chat_template_json"]["outcome"] == "found"


def test_a_local_checkpoint_with_no_template_falls_back_to_its_own_chat_template_json(
    toolchain, tmp_path
):
    """For a path the exporter reads `<path>/chat_template.json` and never the
    Hub, which a local checkpoint's fake hub would have answered otherwise."""
    checkpoint = tmp_path / "Qwen3-0.6B"
    checkpoint.mkdir()
    (checkpoint / "chat_template.json").write_text(
        json.dumps({"chat_template": TOLERANT}), encoding="utf-8"
    )
    toolchain.chat_template = None
    toolchain.hub_files = {"chat_template.json": json.dumps({"chat_template": "{{ 'hub' }}"})}
    result = run_export(
        ExportRequest(
            model=str(checkpoint), output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",)
        )
    )

    assert result.outcome is Outcome.PASSED
    assert result.chat_template is not None
    assert result.chat_template.path.read_bytes() == _BLOCK + TOLERANT.encode("utf-8")
    observed = _chat_template_check(result).observed
    assert observed["chat_template_json"]["path"] == str(checkpoint / "chat_template.json")


def test_a_local_checkpoint_without_one_says_which_file_it_looked_for(toolchain, tmp_path):
    checkpoint = tmp_path / "Qwen3-0.6B"
    checkpoint.mkdir()
    toolchain.chat_template = None
    result = run_export(
        ExportRequest(
            model=str(checkpoint), output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",)
        )
    )
    detail = _chat_template_check(result).detail
    assert (
        f"carries no chat template, and there is no {checkpoint / 'chat_template.json'}" in detail
    )
    assert "repository" not in detail


@pytest.mark.parametrize(
    ("hub_json", "says"),
    [
        ('{"chat_template": null}', "and the chat_template in"),
        ('{"other": "x"}', "has no chat_template key"),
        ("{not json", "could not be used (JSONDecodeError"),
    ],
    ids=["null", "no-key", "not-json"],
)
def test_what_the_chat_template_json_holds_is_said_as_it_is(
    toolchain, qwen_request, hub_json, says
):
    """A present null is not an absent key, and a file the exporter could not
    use is a fact about the checkpoint, as a missing one is."""
    toolchain.chat_template = None
    toolchain.hub_files = {"chat_template.json": hub_json}
    check = _chat_template_check(run_export(qwen_request()))
    assert check.outcome is Outcome.FAILED
    assert says in check.detail
    if hub_json.startswith('{"chat_template": null'):
        assert "is null" in check.detail and "no chat_template key" not in check.detail


def test_a_template_that_is_not_text_is_refused(toolchain, qwen_request):
    """Reachable through chat_template.json, whose value the exporter assigns as
    it finds it (`export_lib.py:401-402`); transformers turns a tokenizer's own
    list into named templates when it loads it (`tokenization_utils_base.py:1079-1082`)."""
    toolchain.chat_template = None
    toolchain.hub_files = {"chat_template.json": json.dumps({"chat_template": ["a", "b"]})}
    check = _chat_template_check(run_export(qwen_request()))
    assert check.outcome is Outcome.FAILED
    assert "the chat_template in" in check.detail and "is a list, not text" in check.detail


def test_an_empty_template_from_chat_template_json_is_named_as_from_there(toolchain, qwen_request):
    toolchain.chat_template = None
    toolchain.hub_files = {"chat_template.json": json.dumps({"chat_template": ""})}
    check = _chat_template_check(run_export(qwen_request()))
    assert check.outcome is Outcome.FAILED
    assert check.observed["template_source"] == "chat_template.json"
    assert "the chat_template in" in check.detail and "is empty (a str)" in check.detail
    assert "the tokenizer's chat template" not in check.detail


def test_an_empty_string_template_falls_back_as_the_exporter_does(toolchain, qwen_request):
    """`export_lib.py:390` tests `not tokenizer.chat_template`: "" falls back too."""
    toolchain.chat_template = ""
    toolchain.hub_files = {"chat_template.json": json.dumps({"chat_template": TOLERANT})}
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))
    assert result.outcome is Outcome.PASSED
    assert _chat_template_check(result).observed["template_source"] == "chat_template.json"


def test_a_directory_at_chat_template_json_is_the_checkpoints_whatever_open_raises(
    tmp_path, monkeypatch
):
    """On Windows, opening a directory is not an `IsADirectoryError`.

    The first windows-latest run read a directory there as "unreachable" --
    the machine's -- because `open` raised something else. The script asks
    whether it is a directory instead, so the answer does not depend on the
    platform's error; here `open` is made to raise as it did there.
    """
    import builtins
    import types

    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    script: dict = {"__name__": "compose_template"}
    exec(compile(_COMPOSE_SCRIPT, "compose_template.py", "exec"), script)
    checkpoint = tmp_path / "model"
    (checkpoint / "chat_template.json").mkdir(parents=True)

    def refuses(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(args[0]))

    monkeypatch.setattr(builtins, "open", refuses)
    template, lookup = script["chat_template_json"](str(checkpoint))

    assert template is None
    assert lookup["outcome"] == "unusable"
    assert lookup["error"].startswith("IsADirectoryError")


@pytest.mark.parametrize(
    ("make", "says"),
    [
        (lambda path: path.mkdir(), "could not be used (IsADirectoryError"),
        (
            lambda path: path.write_bytes(b'{"chat_template": "\xe9"}'),
            "could not be used (UnicodeDecodeError",
        ),
    ],
    ids=["a-directory", "not-utf-8"],
)
def test_what_a_local_checkpoint_holds_at_chat_template_json_is_its_own(
    toolchain, tmp_path, make, says
):
    """The exporter reads the path and goes on with no template whatever it finds
    there (`export_lib.py:396-404`): a fact about the checkpoint, not the machine."""
    checkpoint = tmp_path / "Qwen3-0.6B"
    checkpoint.mkdir()
    make(checkpoint / "chat_template.json")
    toolchain.chat_template = None
    result = run_export(
        ExportRequest(
            model=str(checkpoint), output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",)
        )
    )
    check = _chat_template_check(result)
    assert check.outcome is Outcome.FAILED
    assert says in check.detail


# `chmod(0)` is what makes the file unreadable, and on Windows it only sets the
# read-only flag (`os.chmod`, Python's documentation): a POSIX permission bit.
@pytest.mark.posix
def test_a_local_chat_template_json_the_machine_will_not_open_is_could_not_check(
    toolchain, tmp_path
):
    """Not a missing file and not the checkpoint's: the file system would not let it be read."""
    checkpoint = tmp_path / "Qwen3-0.6B"
    checkpoint.mkdir()
    locked = checkpoint / "chat_template.json"
    locked.write_text("{}", encoding="utf-8")
    locked.chmod(0)
    toolchain.chat_template = None
    try:
        result = run_export(
            ExportRequest(
                model=str(checkpoint), output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",)
            )
        )
    finally:
        locked.chmod(0o644)
    check = _chat_template_check(result)
    assert check.outcome is Outcome.UNCHECKED
    assert "could not be looked up (PermissionError" in check.detail


# -- the probe: each condition on its own ------------------------------------


def test_a_block_that_does_not_read_parts_is_caught_by_the_probe(
    monkeypatch, toolchain, qwen_request, tmp_path
):
    """The composed rendering is given parts: with a block that does nothing,
    Qwen3's template behind it does not render them as it renders the string."""
    noop = tmp_path / "noop.jinja"
    noop.write_text("{#- a block that does nothing -#}", encoding="utf-8")
    real = models.packaged_template
    monkeypatch.setattr(
        models,
        "packaged_template",
        lambda name: str(noop) if name == models.TEXT_PARTS_BLOCK else real(name),
    )
    toolchain.chat_template = QWEN25_TEMPLATE
    check = _chat_template_check(run_export(qwen_request("Qwen/Qwen2.5-0.5B-Instruct")))
    assert check.outcome is Outcome.FAILED
    assert "this template does not render that way" in check.detail


@pytest.mark.parametrize(
    ("carried", "says"),
    [
        # Output, but not the user's text: a template reading another field.
        ("{%- for m in messages %}<|user|>{{ m.text }}{% endfor %}", "does not print its text"),
        # A difference of whitespace only is a difference.
        (
            "{{ messages[0].content }}{% if litetune is defined %} {% endif %}",
            "this template does not render that way",
        ),
        # Rendered with the generation prompt, as the runtime renders it.
        (
            "{{ messages[0].content }}"
            "{% if add_generation_prompt and litetune is defined %}!{% endif %}",
            "this template does not render that way",
        ),
        # With the tokenizer's special tokens, as transformers passes them.
        (
            "{% if eos_token is defined %}{{ litetune is defined }}{% endif %}"
            "{{ messages[0].content }}",
            "this template does not render that way",
        ),
    ],
    ids=["other-text", "whitespace", "generation-prompt", "special-tokens"],
)
def test_the_probe_holds_each_condition(toolchain, qwen_request, carried, says):
    toolchain.chat_template = carried
    check = _chat_template_check(run_export(qwen_request()))
    assert check.outcome is Outcome.FAILED
    assert says in check.detail


@pytest.mark.parametrize("base_model", [None, QWEN3])
def test_the_template_is_read_from_the_checkpoint_converted_not_its_base(
    toolchain, tmp_path, base_model
):
    """The checkpoint's own: a tuned model's tokenizer is the one the exporter
    loads, whatever `--base-model` or its sidecar names."""
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}),
        encoding="utf-8",
    )
    (checkpoint / "litetune.json").write_text(json.dumps({"base_model": QWEN3}), encoding="utf-8")
    result = run_export(
        ExportRequest(
            model=str(checkpoint),
            base_model=base_model,
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
        )
    )

    assert result.outcome is Outcome.PASSED
    assert toolchain.template_loads == [str(checkpoint)]


def test_the_transformers_version_is_the_one_the_script_saw(toolchain, qwen_request):
    toolchain.transformers_version = "9.9.9"
    check = _chat_template_check(run_export(qwen_request(recipes=("dynamic_wi8_afp32",))))
    assert check.observed["transformers"] == "9.9.9"
    assert "transformers 9.9.9 reads it" in check.detail


@pytest.mark.parametrize(
    ("knobs", "says"),
    [
        (
            {
                "template_returncode": 1,
                "template_stderr": "Traceback (most recent call last):\n"
                "OSError: Qwen/Qwen3-0.6B does not appear to have a file named tokenizer.json\n",
            },
            "exited 1: OSError: Qwen/Qwen3-0.6B does not appear to have a file named",
        ),
        # The signal's name is POSIX's: Windows has no SIGKILL to name.
        pytest.param(
            {"template_returncode": -9},
            "chat template was killed by SIGKILL",
            marks=pytest.mark.posix,
        ),
        # Killed after its result and file were written: still no answer.
        pytest.param(
            {"template_exit_after": -9},
            "chat template was killed by SIGKILL",
            marks=pytest.mark.posix,
        ),
        ({"template_no_result": True}, "exited 0 and wrote no result"),
        ({"template_raises": subprocess.TimeoutExpired("python", 5)}, "gave no result after"),
        ({"template_result": "{not json"}, "wrote a result that is not JSON"),
        ({"template_result": "[]"}, "wrote list, not a report"),
        ({"template_result": '{"found": "maybe"}'}, "reported 'maybe', which is not one of"),
        ({"template_overwrite": b"{{ messages }}"}, "does not hold the text-parts block"),
        # Each half of the read-back on its own: the block, then another template ...
        (
            {"template_overwrite": _BLOCK + b"{{ 'not the template it reported' }}"},
            "does not hold the text-parts block",
        ),
        # ... and the template, behind a block with one byte changed: the first,
        (
            {"template_overwrite": b"X" + _BLOCK[1:] + QWEN3_TEMPLATE.encode("utf-8")},
            "does not hold the text-parts block",
        ),
        # ... and a later one.
        (
            {
                "template_overwrite": _BLOCK[:1]
                + b"X" * (len(_BLOCK) - 1)
                + QWEN3_TEMPLATE.encode("utf-8")
            },
            "does not hold the text-parts block",
        ),
        # The environment's faults, not facts about the checkpoint: a jinja2
        # transformers cannot render with ...
        (
            {"render_raises": "apply_chat_template requires jinja2>=3.1.0 to be installed"},
            "the probe could not render in the export environment (ImportError",
        ),
        # ... a Hub that cannot be reached for the chat_template.json fallback ...
        (
            {"chat_template": None, "hub_fault": "LocalEntryNotFoundError"},
            "could not be looked up (LocalEntryNotFoundError",
        ),
        # ... or that drops the connection.
        (
            {"chat_template": None, "hub_fault": "Connection reset by peer"},
            "could not be looked up (ConnectionResetError: Connection reset by peer)",
        ),
        (
            {"template_report_edit": lambda report: report | {"renders": "no"}},
            "renders this litetune cannot read",
        ),
    ],
)
def test_a_script_that_cannot_say_refuses_with_the_reason(
    toolchain, qwen_request, tmp_path, knobs, says
):
    """Nothing here is a fact about the checkpoint, so `could not check`; and no
    recipe ran, so none is recorded as failed."""
    for name, value in knobs.items():
        setattr(toolchain, name, value)
    result = run_export(qwen_request())

    check = _chat_template_check(result)
    assert check.outcome is Outcome.UNCHECKED
    assert says in check.detail
    assert result.outcome is Outcome.UNCHECKED
    assert toolchain.exports == []
    assert result.failed == []
    assert result.not_attempted == tuple(MEASURED_RECIPES)
    assert any(text.startswith("none of") and says in text for text in result.limitations)
    assert _composed_files(tmp_path / "out") == []


def test_a_file_another_run_left_is_never_read_or_removed(toolchain, qwen_request, tmp_path):
    """This run writes a name of its own: a file under the content-addressed
    name, or any other, is neither read back as this run's nor removed when
    this run is refused."""
    out = tmp_path / "out"
    out.mkdir()
    others = [
        out / f"{COMPOSED_TEMPLATE_PREFIX}000000000000.jinja",
        out / f"{COMPOSED_TEMPLATE_PREFIX}abcdef.partial",
    ]
    for other in others:
        other.write_bytes(_BLOCK + QWEN3_TEMPLATE.encode("utf-8"))
    toolchain.template_result = json.dumps(
        {
            "transformers": "5.16.1",
            "found": "text",
            "sha256": hashlib.sha256(QWEN3_TEMPLATE.encode("utf-8")).hexdigest(),
            "characters": len(QWEN3_TEMPLATE),
        }
    )
    result = run_export(qwen_request())

    assert _chat_template_check(result).outcome is Outcome.UNCHECKED
    assert toolchain.exports == []
    assert _composed_files(out) == sorted(p.name for p in others)


def test_an_error_after_the_script_wrote_leaves_no_file(
    monkeypatch, toolchain, qwen_request, tmp_path
):
    """The `finally` is what removes this run's file when something it does not
    catch escapes; `guard` records the check as unchecked."""

    def raises(*args):
        raise RuntimeError("probe reader broke")

    monkeypatch.setattr("litetune.export._choose_template", raises)
    result = run_export(qwen_request())

    check = _chat_template_check(result)
    assert check.outcome is Outcome.UNCHECKED
    assert "probe reader broke" in check.detail
    assert toolchain.exports == []
    assert _composed_files(tmp_path / "out") == []


def test_a_composition_that_passes_with_no_file_exports_nothing(
    monkeypatch, toolchain, qwen_request
):
    """The sweep runs on one condition: a passed check *and* a file. Either
    alone is could-not-check, never a passed stage with nothing exported."""
    monkeypatch.setattr(
        "litetune.export.compose_chat_template",
        lambda request: (Check.passed(CHAT_TEMPLATE_CHECK, "said it passed"), None),
    )
    result = run_export(qwen_request())

    assert _chat_template_check(result).outcome is Outcome.UNCHECKED
    assert result.outcome is Outcome.UNCHECKED
    assert toolchain.exports == []
    assert result.not_attempted == tuple(MEASURED_RECIPES)


def test_a_refusal_is_reported_as_events(toolchain, qwen_request):
    toolchain.chat_template = None
    seen: list[Event] = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    run_export(qwen_request(), events=events)

    checks = [e for e in seen if e.kind == "check" and e.data["name"] == CHAT_TEMPLATE_CHECK]
    assert [e.data["outcome"] for e in checks] == ["failed"]
    assert seen[-1].kind == "stage_finished"
    assert seen[-1].data["attempted"] == 0


# ---------------------------------------------------------------------------
# LiteRT-LM's Qwen 3 template, and whether each bundle holds a conversation
# ---------------------------------------------------------------------------

LITERT_QWEN3 = models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE)
# Qwen3's template behind a line of its own: a fine-tune whose single turns
# LiteRT-LM's template does not render, and whose replies still go through
# Qwen3's think block.
QWEN3_WITH_A_PREAMBLE = "{{- 'Be brief.\\n' }}" + QWEN3_TEMPLATE
# Qwen3's template with one byte more, which renders every case as it does:
# what tells it from Qwen's own is its bytes alone.
QWEN3_ONE_BYTE_ON = QWEN3_TEMPLATE + "\n"
_NOT_RECORDED = object()


def _one_message(result: ExportResult) -> list[str]:
    """The report's limitations that say a bundle answers one message."""
    return [t for t in result.limitations if template_checks.LIMITATION_PREFIX in t]


@pytest.fixture
def trained(tmp_path):
    """A Qwen3 0.6B checkpoint `tune` wrote, recording how it trained.

    `record` is what its `litetune.json` holds besides the base model -- by
    default what a `runtime_rendered` run without declarations records -- or
    `_NOT_RECORDED` for a directory with no `litetune.json` at all, or raw text.
    """

    # Named for the model, so that a checkpoint without a record is still
    # claimed by the qwen-3 rule, by its path.
    def _build(record: object = None, name: str = "Qwen3-0.6B-tuned") -> str:
        checkpoint = tmp_path / name
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
            encoding="utf-8",
        )
        if record is _NOT_RECORDED:
            return str(checkpoint)
        sidecar = checkpoint / models.PROVENANCE_NAME
        if isinstance(record, str):
            sidecar.write_text(record, encoding="utf-8")
            return str(checkpoint)
        fields = {"prompt_mode": "runtime_rendered", "declarations_sha256": None}
        fields.update(record if isinstance(record, dict) else {})
        sidecar.write_text(json.dumps({"base_model": QWEN3, **fields}), encoding="utf-8")
        return str(checkpoint)

    return _build


def test_a_qwen3_checkpoint_gets_litert_lms_template_and_holds_a_conversation(
    toolchain, qwen_request, trained, tmp_path
):
    """Trained runtime_rendered without declarations, from a template that renders
    single turns as LiteRT-LM's does, in every context: the bundle carries
    LiteRT-LM's, which holds a conversation, and the report says nothing about
    one message."""
    toolchain.chat_template = QWEN3_TEMPLATE
    result = run_export(qwen_request(trained()))

    assert result.outcome is Outcome.PASSED
    template = result.chat_template
    assert template is not None
    assert (template.source, template.canonical) == ("canonical", models.QWEN3_CANONICAL_TEMPLATE)
    assert template.path.read_bytes() == LITERT_QWEN3.encode("utf-8")
    digest = hashlib.sha256(LITERT_QWEN3.encode("utf-8")).hexdigest()
    assert template.path.name == f"{COMPOSED_TEMPLATE_PREFIX}{digest[:12]}.jinja"
    assert _composed_files(tmp_path / "out") == [template.path.name], "no partial file is left"
    check = _chat_template_check(result)
    assert check.outcome is Outcome.PASSED
    assert check.observed["chosen"] == "canonical"
    assert check.observed["canonical_withheld"] is None
    assert (
        "records that it was trained runtime_rendered without declarations"
        in (check.observed["canonical_offered"])
    )
    assert check.observed["canonical_differences"] == []
    assert f"LiteRT-LM's canonical template {models.QWEN3_CANONICAL_TEMPLATE}" in check.detail
    assert "each of 96 single-turn renderings" in check.detail
    assert len(toolchain.exports) == 2
    for call in toolchain.exports:
        assert _template_flags(call.argv) == [f"{models.TEMPLATE_FLAG}={template.path}"]
    assert set(toolchain.embedded.values()) == {LITERT_QWEN3}
    for export in result.exports:
        assert export.conversation is not None
        assert export.conversation.holds is True
        dumped = export.as_dict()["conversation"]
        assert dumped["holds"] is True
        assert dumped["contexts"] == [context.name for context in template_checks.CONTEXTS]
    # Once for the sweep: both bundles carry the one template.
    assert len(toolchain.conversations) == 1
    assert _one_message(result) == []
    assert result.as_dict()["chat_template"]["source"] == "canonical"
    summary = summarise_convert(result)
    assert any(
        f"LiteRT-LM's {models.QWEN3_CANONICAL_TEMPLATE}, which renders single turns" in line
        for line in summary
    )


@pytest.mark.parametrize(
    ("record", "carried", "says"),
    [
        (
            _NOT_RECORDED,
            QWEN3_ONE_BYTE_ON,
            f"carries no {models.PROVENANCE_NAME}, so nothing records that it was",
        ),
        ({"prompt_mode": "prerendered"}, QWEN3_TEMPLATE, "records that it was trained prerendered"),
        (
            {"declarations_sha256": "ab" * 32},
            QWEN3_TEMPLATE,
            "records that it was trained with tool declarations (sha256 abababababab)",
        ),
        ({"prompt_mode": None}, QWEN3_TEMPLATE, "records no prompt_mode"),
        ({"prompt_mode": "rendered-somehow"}, QWEN3_TEMPLATE, "could not be read (ValueError"),
        ("{not json", QWEN3_TEMPLATE, "could not be read (JSONDecodeError"),
    ],
    ids=["no-record", "prerendered", "declarations", "no-mode", "unknown-mode", "unreadable"],
)
def test_a_qwen3_checkpoint_whose_training_the_cases_do_not_cover_keeps_its_own(
    toolchain, qwen_request, trained, record, carried, says
):
    """The single-turn cases stand for what training rendered only where the
    record shows training rendered one user message through the template, or
    where there is no record and the template is Qwen's own. Anything else --
    Qwen's own template, trained another way; or no record and a template a
    byte away from Qwen's -- keeps the checkpoint's own, which answers one
    message, and the report says so."""
    toolchain.chat_template = carried
    result = run_export(qwen_request(trained(record)))

    assert result.outcome is Outcome.PASSED
    template = result.chat_template
    assert template is not None and template.source == "composed"
    assert template.path.read_bytes() == _BLOCK + carried.encode("utf-8")
    check = _chat_template_check(result)
    assert check.observed["canonical_offered"] is None
    assert says in check.observed["canonical_withheld"]
    assert f"LiteRT-LM's canonical template {models.QWEN3_CANONICAL_TEMPLATE} was not offered" in (
        check.detail
    )
    assert says in check.detail
    assert "canonical_differences" not in check.observed
    (limitation,) = _one_message(result)
    assert "google-ai-edge/LiteRT-LM#3443" in limitation


@pytest.mark.parametrize("checkpoint", ["hub", "local"])
def test_the_base_model_as_published_gets_litert_lms_template(
    toolchain, qwen_request, trained, checkpoint
):
    """flutter_gemma #659 converted `Qwen/Qwen3-0.6B` itself: no litetune.json,
    and the template Qwen ships, byte for byte -- Qwen 3's own chat format, which
    LiteRT-LM publishes its template for. That bundle holds a conversation."""
    toolchain.chat_template = QWEN3_TEMPLATE
    model = QWEN3 if checkpoint == "hub" else trained(_NOT_RECORDED)
    result = run_export(qwen_request(model))

    assert result.outcome is Outcome.PASSED
    template = result.chat_template
    assert template is not None and template.source == "canonical"
    assert template.path.read_bytes() == LITERT_QWEN3.encode("utf-8")
    check = _chat_template_check(result)
    assert check.observed["canonical_withheld"] is None
    offered = check.observed["canonical_offered"]
    assert f"{model} carries no {models.PROVENANCE_NAME}" in offered
    assert "Qwen/Qwen3-0.6B's own at c1899de289a04d12100db370d81485cdf75e47ca" in offered
    assert "byte for byte (sha256 a55ee1b16601)" in offered
    assert check.observed["canonical_differences"] == []
    assert f"offered because {offered}" in check.detail
    assert all(e.conversation and e.conversation.holds is True for e in result.exports)
    assert _one_message(result) == []


def _a_directory(sidecar: Path) -> None:
    sidecar.mkdir()


def _a_link_going_nowhere(sidecar: Path) -> None:
    sidecar.symlink_to(sidecar.parent / "never-written.json")


def _not_json(sidecar: Path) -> None:
    sidecar.write_text("{not json", encoding="utf-8")


def _not_an_object(sidecar: Path) -> None:
    sidecar.write_text("[]", encoding="utf-8")


def _withheld_for_an_unreadable_record(result: ExportResult) -> str:
    """Why LiteRT-LM's template was withheld, asserting that the bundle kept the
    checkpoint's own and says it answers one message."""
    assert result.outcome is Outcome.PASSED
    template = result.chat_template
    assert template is not None and template.source == "composed"
    assert template.path.read_bytes() == _BLOCK + QWEN3_TEMPLATE.encode("utf-8")
    check = _chat_template_check(result)
    assert check.observed["canonical_offered"] is None
    withheld = check.observed["canonical_withheld"]
    assert "could not be read (" in withheld
    assert f"carries no {models.PROVENANCE_NAME}" not in withheld
    (limitation,) = _one_message(result)
    assert "google-ai-edge/LiteRT-LM#3443" in limitation
    return withheld


@pytest.mark.parametrize(
    ("spoil", "raised"),
    [
        # Refused before it is opened, on every platform.
        (_a_directory, ("OSError: ", "is not a regular file: it is a directory")),
        (_a_link_going_nowhere, ("FileNotFoundError: ",)),
        (_not_json, ("JSONDecodeError: ",)),
        (_not_an_object, ("ValueError: ",)),
    ],
    ids=["directory", "dangling-symlink", "malformed-json", "not-an-object"],
)
def test_a_record_that_is_there_and_cannot_be_read_is_not_an_absent_one(
    toolchain, qwen_request, trained, spoil, raised
):
    """Only a checkpoint with no litetune.json at all is offered LiteRT-LM's
    template on its template's bytes. A litetune.json that is there and cannot
    be read keeps the checkpoint's own -- here Qwen's template, byte for byte,
    which the base model with no record is offered LiteRT-LM's on."""
    toolchain.chat_template = QWEN3_TEMPLATE
    model = trained(_NOT_RECORDED)
    spoil(Path(model) / models.PROVENANCE_NAME)
    withheld = _withheld_for_an_unreadable_record(run_export(qwen_request(model)))
    assert all(said in withheld for said in raised), withheld


# What a reader that opened it would read: a record that offers LiteRT-LM's
# template, so that reading it fails the test as surely as waiting would.
_OFFERING_RECORD = json.dumps({"base_model": QWEN3, "prompt_mode": "runtime_rendered"}).encode()


# Untyped, so that `mypy --platform win32` does not look for `os.mkfifo`.
def _a_fifo(sidecar):
    os.mkfifo(sidecar)


def _a_link_to_dev_null(sidecar):
    sidecar.symlink_to("/dev/null")


@pytest.mark.posix
@pytest.mark.parametrize(
    ("make", "kind"),
    [
        (_a_fifo, "a FIFO"),
        (_a_link_to_dev_null, "a character device"),
    ],
    ids=["fifo", "link-to-dev-null"],
)
def test_a_record_that_is_not_a_regular_file_is_refused_unopened(
    toolchain, qwen_request, trained, monkeypatch, make, kind
):
    """A FIFO named litetune.json would hold the open until something wrote to
    it, and a device need not end. Neither is opened: refused for what it is."""
    toolchain.chat_template = QWEN3_TEMPLATE
    model = trained(_NOT_RECORDED)
    sidecar = Path(model) / models.PROVENANCE_NAME
    make(sidecar)
    opened = record_opens(monkeypatch)
    if make is _a_fifo:
        with fifo_released_after(sidecar, 5, _OFFERING_RECORD) as released:
            result = run_export(qwen_request(model))
        assert released == [], "convert waited on the FIFO"
    else:
        # No watchdog: opening the device for writing would not tell a reader.
        result = run_export(qwen_request(model))
    assert not {sidecar, sidecar.resolve()} & set(opened)
    withheld = _withheld_for_an_unreadable_record(result)
    assert f"(OSError: {sidecar} is not a regular file: it is {kind})" in withheld


@pytest.mark.posix
def test_a_record_that_cannot_be_opened_is_not_an_absent_one(toolchain, qwen_request, trained):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root opens a file whatever its permission bits")
    toolchain.chat_template = QWEN3_TEMPLATE
    model = trained()
    sidecar = Path(model) / models.PROVENANCE_NAME
    sidecar.chmod(0)
    try:
        result = run_export(qwen_request(model))
    finally:
        sidecar.chmod(0o600)
    assert "(PermissionError: " in _withheld_for_an_unreadable_record(result)


def test_a_base_model_whose_single_turns_differ_is_not_given_litert_lms_template(
    toolchain, qwen_request, monkeypatch
):
    """Qwen's own bytes offer the template; the renderings still decide."""
    pinned = {hashlib.sha256(QWEN3_WITH_A_PREAMBLE.encode("utf-8")).hexdigest(): "a test's"}
    monkeypatch.setitem(models.CANONICAL_UPSTREAM, models.QWEN3_CANONICAL_TEMPLATE, pinned)
    toolchain.chat_template = QWEN3_WITH_A_PREAMBLE
    result = run_export(qwen_request(QWEN3))

    assert result.chat_template is not None and result.chat_template.source == "composed"
    check = _chat_template_check(result)
    assert check.observed["canonical_offered"] is not None
    assert len(check.observed["canonical_differences"]) == 96
    assert _one_message(result)


def test_a_qwen3_fine_tune_whose_single_turns_differ_keeps_its_own_and_answers_one_message(
    toolchain, qwen_request, trained
):
    """LiteRT-LM's template does not render this checkpoint's single turns, so the
    bundle keeps the checkpoint's own behind the block -- whose think block breaks
    a conversation, which the report says, naming both recipes."""
    toolchain.chat_template = QWEN3_WITH_A_PREAMBLE
    result = run_export(qwen_request(trained()))

    assert result.outcome is Outcome.PASSED
    template = result.chat_template
    assert template is not None and template.source == "composed"
    assert template.path.read_bytes() == _BLOCK + QWEN3_WITH_A_PREAMBLE.encode("utf-8")
    check = _chat_template_check(result)
    assert check.observed["chosen"] == "composed"
    assert check.observed["canonical_withheld"] is None
    assert len(check.observed["canonical_differences"]) == 96
    assert (
        f"LiteRT-LM's canonical template {models.QWEN3_CANONICAL_TEMPLATE} was not chosen: "
        "96 of 96 single-turn renderings differ"
    ) in check.detail
    for export in result.exports:
        assert export.conversation is not None and export.conversation.holds is False
        assert export.conversation.breaks
        assert export.as_dict()["conversation"]["holds"] is False
    (limitation,) = _one_message(result)
    assert limitation.startswith(
        f"{', '.join(MEASURED_RECIPES)}: {template_checks.LIMITATION_PREFIX}"
    )
    assert "google-ai-edge/LiteRT-LM#3443" in limitation
    assert "72 of the 72 cases break it" in limitation
    assert f"  note: {limitation}" in summarise_convert(result)
    assert limitation in result.as_dict()["limitations"]


@pytest.mark.parametrize(
    ("record", "carried", "holds"),
    [
        (None, QWEN3_TEMPLATE, True),
        (None, QWEN3_WITH_A_PREAMBLE, False),
        ({"prompt_mode": "prerendered"}, QWEN3_TEMPLATE, False),
        (_NOT_RECORDED, QWEN3_TEMPLATE, True),
        (_NOT_RECORDED, QWEN3_ONE_BYTE_ON, False),
        (_NOT_RECORDED, "{%- for m in messages %}{{ m.content }}{% endfor %}", True),
    ],
    ids=[
        "qwen3",
        "qwen3-preamble",
        "qwen3-prerendered",
        "base",
        "base-one-byte-on",
        "fine-tune",
    ],
)
def test_the_report_says_a_bundle_answers_one_message_exactly_when_it_does(
    toolchain, qwen_request, trained, record, carried, holds
):
    toolchain.chat_template = carried
    result = run_export(qwen_request(trained(record)))

    assert result.outcome is Outcome.PASSED
    assert [e.conversation.holds if e.conversation else None for e in result.exports] == [
        holds,
        holds,
    ]
    assert bool(_one_message(result)) is not holds


def test_qwen25_keeps_its_own_template_which_holds_a_conversation(toolchain, qwen_request):
    toolchain.chat_template = QWEN25_TEMPLATE
    result = run_export(qwen_request("Qwen/Qwen2.5-0.5B-Instruct"))

    assert result.outcome is Outcome.PASSED
    assert result.chat_template is not None and result.chat_template.source == "composed"
    assert "canonical_withheld" not in _chat_template_check(result).observed
    assert all(e.conversation and e.conversation.holds is True for e in result.exports)
    assert _one_message(result) == []


# A filter the renderer does not render as MiniJinja does, and a chained
# comparison, which MiniJinja 2.14 reads as nested ones: `1 == 2 == false` is
# false in Jinja2 and true on the device.
_UNVOUCHED_CONSTRUCTS = [
    ("{{ 'x' | title }}", "uses the filter `title`"),
    (
        "{% if 1 == 2 == false %}!{% endif %}",
        "uses a chained comparison, which MiniJinja 2.14 reads as nested ones",
    ),
]


@pytest.mark.parametrize(("construct", "named"), _UNVOUCHED_CONSTRUCTS)
def test_a_composed_template_the_renderer_does_not_vouch_for_exports_nothing(
    toolchain, qwen_request, construct, named
):
    """`could not check`, never a pass: a checkpoint template using a construct
    the renderer does not render as MiniJinja does."""
    toolchain.chat_template = "{{ messages[0].content }}" + construct
    result = run_export(qwen_request("Qwen/Qwen2.5-0.5B-Instruct"))

    check = _chat_template_check(result)
    assert check.outcome is Outcome.UNCHECKED
    assert named in check.detail and "was not checked" in check.detail
    assert toolchain.exports == []


def test_every_family_is_checked_and_functiongemmas_template_holds(toolchain, request_for):
    """No template is written at export for FunctionGemma; the one its bundles
    carry, the packaged one, is read back and checked all the same."""
    result = run_export(request_for())

    assert result.outcome is Outcome.PASSED
    assert toolchain.compositions == []
    assert len(toolchain.conversations) == 1
    packaged = Path(models.functiongemma_template()).read_bytes()
    for export in result.exports:
        assert export.conversation is not None and export.conversation.holds is True
        assert export.conversation.template_sha256 == hashlib.sha256(packaged).hexdigest()
    assert _one_message(result) == []


def _own_template(tmp_path: Path, text: str) -> str:
    path = tmp_path / "own.jinja"
    path.write_text(text, encoding="utf-8")
    return f"--jinja_chat_template_override={path}"


def test_a_template_on_the_single_turn_protocol_is_not_applicable(toolchain, tmp_path):
    """A template of the caller's own, which LiteRT-LM renders through its
    single-turn protocol: the full-history invariant does not apply, and the
    report says that -- not that the bundle answers one message."""
    template = "{%- if is_appending_to_prefill %}{% endif %}{{ messages[-1].content[0].text }}"
    result = run_export(
        ExportRequest(
            model="acme/some-model",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
            extra_flags=(_own_template(tmp_path, template),),
        )
    )

    assert result.outcome is Outcome.PASSED
    assert toolchain.conversations == []
    (export,) = result.exports
    assert export.conversation is not None
    assert export.conversation.holds is None and not export.conversation.applicable
    assert _one_message(result) == []
    assert any(
        "the conversation check does not apply to this bundle" in text
        and "is_appending_to_prefill" in text
        for text in result.limitations
    )


@pytest.mark.parametrize(("construct", "named"), _UNVOUCHED_CONSTRUCTS)
def test_a_bundled_template_the_renderer_does_not_vouch_for_is_not_checked(
    toolchain, tmp_path, construct, named
):
    template = "{%- for m in messages %}{{ m.content[0].text }};" + construct + "{% endfor %}"
    result = run_export(
        ExportRequest(
            model="acme/some-model",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
            extra_flags=(_own_template(tmp_path, template),),
        )
    )

    assert result.outcome is Outcome.PASSED
    (export,) = result.exports
    assert export.conversation is not None and export.conversation.holds is None
    assert _one_message(result) == []
    assert any(
        "was not checked" in t and named.removeprefix("uses ") in t for t in result.limitations
    )


def test_a_bundle_carrying_no_template_is_said_not_to_be_checked(toolchain, tmp_path):
    """Gemma 3 gets no override, and this fake exporter embeds nothing without one."""
    result = run_export(
        ExportRequest(
            model="google/gemma-3-270m-it",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
        )
    )

    assert result.outcome is Outcome.PASSED
    assert toolchain.conversations == []
    (export,) = result.exports
    assert export.conversation is not None and export.conversation.holds is None
    assert _one_message(result) == []
    assert any(
        "dynamic_wi8_afp32: whether this bundle holds a LiteRT-LM conversation past its first "
        "message was not checked: it carries no jinja_prompt_template" in text
        for text in result.limitations
    )


@pytest.mark.parametrize(
    ("knobs", "says"),
    [
        ({"conversation_returncode": 1}, "the script rendering it exited 1: boom"),
        ({"conversation_result": "{not json"}, "wrote a result that is not JSON"),
        ({"conversation_result": "[]"}, "the script rendering it wrote []"),
        ({"conversation_result": '{"renders": []}'}, "unvouched constructs this litetune cannot"),
        (
            {"conversation_result": '{"unvouched": [], "renders": []}'},
            "the conversations is not 12 renderings",
        ),
        ({"repack_script_returncode": 1}, "its chat template could not be read back"),
    ],
)
def test_a_conversation_that_could_not_be_checked_is_said_so_and_exported(
    toolchain, request_for, knobs, says
):
    """Never a verdict either way, and never a reason to refuse the bundle."""
    for name, value in knobs.items():
        setattr(toolchain, name, value)
    result = run_export(request_for())

    assert result.outcome is Outcome.PASSED
    assert all(e.conversation and e.conversation.holds is None for e in result.exports)
    assert _one_message(result) == []
    (text,) = [t for t in result.limitations if "was not checked" in t]
    assert says in text
    assert text.startswith(", ".join(MEASURED_RECIPES))


def test_a_conversation_check_that_raises_is_said_not_to_be_checked(
    toolchain, request_for, monkeypatch
):
    def raises(*args, **kwargs):
        raise RuntimeError("renderer broke")

    monkeypatch.setattr("litetune.export.check_conversation", raises)
    result = run_export(request_for())

    assert result.outcome is Outcome.PASSED
    assert all(e.conversation and e.conversation.holds is None for e in result.exports)
    assert any("was not checked" in t and "renderer broke" in t for t in result.limitations)


def test_a_packaged_canonical_template_that_is_not_the_pinned_one_exports_nothing(
    toolchain, qwen_request, trained, monkeypatch
):
    monkeypatch.setitem(models.CANONICAL_TEMPLATES, models.QWEN3_CANONICAL_TEMPLATE, "0" * 64)
    result = run_export(qwen_request(trained()))

    check = _chat_template_check(result)
    assert check.outcome is Outcome.UNCHECKED
    assert "the packaged canonical template could not be used" in check.detail
    assert toolchain.compositions == [] and toolchain.exports == []
    assert result.not_attempted == tuple(MEASURED_RECIPES)


@pytest.mark.parametrize(
    ("embedded", "says", "outcome"),
    [
        (lambda text: text[:-1], "carries a jinja_prompt_template with sha256", Outcome.FAILED),
        (lambda text: None, "carries no jinja_prompt_template", Outcome.FAILED),
        # The exporter reads the override in text mode: a CR it folded is a
        # template other than the one composed.
        (lambda text: text.replace("\n", "\r\n", 1), "with sha256", Outcome.FAILED),
        # Equality, not length and not equality modulo trailing whitespace.
        (
            lambda text: text[:-1] + ("X" if text[-1] != "X" else "Y"),
            "with sha256",
            Outcome.FAILED,
        ),
        (lambda text: text + "\n", "with sha256", Outcome.FAILED),
    ],
    ids=["truncated", "none", "translated", "same-length", "trailing-newline"],
)
def test_a_bundle_that_does_not_carry_the_composed_template_fails_that_recipe(
    toolchain, qwen_request, tmp_path, embedded, says, outcome
):
    """The invariant: no recipe passes unless its bundle carries exactly the
    composed text, whatever the exporter was handed. The refused bundle is moved
    out of the place a usable one goes, and the detail says where."""
    toolchain.template_embedded = embedded
    result = run_export(qwen_request())

    assert result.outcome is outcome
    assert len(result.exports) == 2
    for export in result.exports:
        assert export.check.outcome is outcome
        assert says in export.check.detail
        assert export.artifact is None, "not handed on as an artifact"
        recipe_dir = tmp_path / "out" / export.recipe
        assert not (recipe_dir / "model.litertlm").exists()
        aside = recipe_dir / "model.litertlm.refused"
        assert aside.is_file()
        assert f"moved to {aside}" in export.check.detail
        assert export.check.observed["refused_bundle"] == str(aside)
    assert result.artifacts == []


def test_a_bundle_that_cannot_be_moved_aside_says_where_it_is(
    monkeypatch, toolchain, qwen_request, tmp_path
):
    import litetune.export as export_module

    real = export_module.os.replace

    def refuse(src, dst):
        if str(dst).endswith(".refused"):
            raise PermissionError("read-only")
        return real(src, dst)

    monkeypatch.setattr(export_module.os, "replace", refuse)
    toolchain.template_embedded = lambda text: None
    (export,) = run_export(qwen_request(recipes=("dynamic_wi8_afp32",))).exports

    assert export.check.outcome is Outcome.FAILED
    bundle = tmp_path / "out" / "dynamic_wi8_afp32" / "model.litertlm"
    assert f"could not be moved aside (PermissionError: read-only) and is still at {bundle}" in (
        export.check.detail
    )
    assert export.check.observed["refused_bundle"] is None


@pytest.mark.parametrize("model", [QWEN3, "google/functiongemma-270m-it"])
def test_a_bundle_cut_short_fails_whatever_the_family(toolchain, tmp_path, model):
    """`unpack` reads a short section without raising, so a cut bundle would
    read back whole and repack into a shorter, well-formed one: the header's
    section ends are compared with the file's size first."""
    toolchain.repack_section_ends = [1024, 8192]
    result = run_export(
        ExportRequest(model=model, output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",))
    )

    (export,) = result.exports
    assert export.check.outcome is Outcome.FAILED
    assert "is not whole: section 1 of 2 ends at byte 8,192, past the end" in export.check.detail
    assert export.artifact is None
    assert (tmp_path / "out" / "dynamic_wi8_afp32" / "model.litertlm.refused").is_file()
    assert result.outcome is Outcome.FAILED


@pytest.mark.parametrize("model", [QWEN3, "google/functiongemma-270m-it"])
@pytest.mark.parametrize(
    ("knobs", "size", "says"),
    [
        (
            {"repack_header_raises": "Invalid magic number: b'xxxxxxxx'"},
            4096,
            "its header cannot be read (ValueError: Invalid magic number",
        ),
        (
            {"bundle_header_end": 1 << 20},
            4096,
            "its header ends at byte 1,048,576, past the end of the 4,096-byte file",
        ),
        ({}, 20, "the 20-byte file is cut short inside its header"),
    ],
    ids=["bad-magic", "header-past-the-end", "cut-inside-the-header"],
)
def test_a_bundle_whose_header_is_unsound_fails_whatever_the_family(
    toolchain, tmp_path, model, knobs, size, says
):
    """The builder writes the header last (`litertlm_builder.py` `_build_sections`),
    so a bundle cut short or never finished may have none to read."""
    for name, value in knobs.items():
        setattr(toolchain, name, value)
    toolchain.default = RecipeRun(writes=(("model.litertlm", size),))
    result = run_export(
        ExportRequest(model=model, output_dir=tmp_path / "out", recipes=("dynamic_wi8_afp32",))
    )
    (export,) = result.exports
    assert export.check.outcome is Outcome.FAILED
    assert says in export.check.detail
    assert (tmp_path / "out" / "dynamic_wi8_afp32" / "model.litertlm.refused").is_file()
    assert NOT_VERIFIED not in result.limitations, "no artifact for it to be about"


def test_a_bundle_whose_sections_end_at_its_size_is_whole(toolchain, qwen_request):
    toolchain.repack_section_ends = [1024, 4096]
    (export,) = run_export(qwen_request(recipes=("dynamic_wi8_afp32",))).exports
    assert export.check.outcome is Outcome.PASSED


def test_a_bundle_whose_template_cannot_be_read_back_is_could_not_check(toolchain, qwen_request):
    """The builder's `unpack` failed before it read anything: no reading, which
    says nothing about the bundle."""
    toolchain.repack_unpack_fails = True
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.check.outcome is Outcome.UNCHECKED
    assert "could not be read back" in export.check.detail
    assert "unpack: boom" in export.check.detail


def test_a_bundle_with_no_llm_metadata_section_fails(toolchain, qwen_request):
    toolchain.repack_toml = FAKE_TOML.replace(
        '[[section]]\nsection_type = "LlmMetadata"\ndata_path = "LlmMetadataProto.pbtext"\n\n',
        "",
    )
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.check.outcome is Outcome.FAILED
    assert "0 LlmMetadata sections" in export.check.detail


def test_two_llm_metadata_sections_fail(toolchain, qwen_request):
    """The script and the parent each insist on exactly one."""
    section = '[[section]]\nsection_type = "LlmMetadata"\ndata_path = "LlmMetadataProto.pbtext"\n\n'
    toolchain.repack_toml = FAKE_TOML.replace(section, section + section)
    (export,) = run_export(qwen_request(recipes=("dynamic_wi8_afp32",))).exports
    assert export.check.outcome is Outcome.FAILED
    assert "2 LlmMetadata sections" in export.check.detail


def test_an_empty_bundle_carries_no_template_section(toolchain, qwen_request):
    toolchain.default = RecipeRun(writes=(("model.litertlm", 0),))
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.check.outcome is Outcome.FAILED
    assert "is empty (0 bytes), so it carries no template" in export.check.detail


def test_a_repack_that_fails_after_reading_keeps_the_reading(toolchain, qwen_request):
    """The template is written out before `pack` runs, so a pack that fails
    leaves the original -- whose template was read -- and a passed recipe
    with the GPU key's note."""
    toolchain.repack_pack_fails = True
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.check.outcome is Outcome.PASSED
    assert export.gpu_activation is None
    assert "pack: boom" in (export.gpu_activation_note or "")


def test_a_rebuild_that_would_change_the_template_is_refused(toolchain, qwen_request):
    """The repack keeps the original rather than ship a template other than
    the one it read, so the reading still describes the bundle."""
    toolchain.repack_pack_rewrites_template = True
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.check.outcome is Outcome.PASSED
    assert export.gpu_activation is None
    assert "the rebuild changed the bundle's chat template" in (export.gpu_activation_note or "")


def test_a_callers_own_template_is_passed_through_and_nothing_is_composed(
    toolchain, qwen_request, tmp_path
):
    """`RequiredFlag.satisfied_by`: the caller's value wins, and then there is
    nothing to compose or read back -- the plan says so in a note."""
    own = tmp_path / "mine.jinja"
    own.write_text("{{ messages }}", encoding="utf-8")
    result = run_export(qwen_request(extra_flags=(f"{models.TEMPLATE_FLAG}={own}",)))

    assert result.outcome is Outcome.PASSED
    assert toolchain.compositions == []
    assert not any(c.name == CHAT_TEMPLATE_CHECK for c in result.checks.checks)
    for call in toolchain.exports:
        assert _template_flags(call.argv) == [f"{models.TEMPLATE_FLAG}={own}"]
    assert _composed_files(tmp_path / "out") == []
    assert any("litetune put no text-parts block" in text for text in result.limitations)


def test_a_recipe_that_raises_records_the_argv_it_was_given(monkeypatch, toolchain, qwen_request):
    """The sweep's backstop rebuilds the argv for its record, and that one too
    carries the composed file rather than raising on the marker."""
    monkeypatch.setattr(
        "litetune.export._sha256",
        lambda path: (_ for _ in ()).throw(OSError("input/output error")),
    )
    result = run_export(qwen_request(recipes=("dynamic_wi8_afp32",)))

    (export,) = result.exports
    assert export.check.outcome is Outcome.UNCHECKED
    assert result.chat_template is not None
    path = result.chat_template.path
    assert _template_flags(export.argv) == [f"{models.TEMPLATE_FLAG}={path}"]


@pytest.mark.parametrize(
    ("flag", "source", "carried"),
    [
        ("--noexternalize_embedder", "caller opted out", False),
        ("--externalize-embedder=False", "caller opted out", False),
        ("--externalize_embedder", "caller", True),
    ],
)
def test_externalize_embedder_is_read_off_the_parsed_flags(request_for, flag, source, carried):
    """A caller's own setting, in any spelling Fire reads, is theirs: the default
    is added only where nothing set it."""
    request = request_for(MEASURED_RECIPES, extra_flags=(flag,))
    argv = request.argv("dynamic_wi8_afp32")
    assert request.externalize_source == source
    assert request.as_dict()["externalize_embedder"] is carried
    assert flag in argv
    assert ("--externalize_embedder" in argv) is (flag == "--externalize_embedder")


def test_the_cli_opt_out_is_the_callers_setting(request_for):
    """`convert --no-externalize-embedder` enters the plan's table as the caller's
    own setting, so the exporter gets the opt-out and nothing litetune adds."""
    request = request_for(MEASURED_RECIPES, externalize_embedder=False)
    argv = request.argv("dynamic_wi8_afp32")
    assert request.externalize_source == "caller opted out"
    assert request.as_dict()["externalize_embedder"] is False
    assert "--noexternalize_embedder" in argv
    assert "--externalize_embedder" not in argv


GEMMA4 = "google/gemma-4-E2B-it"


@pytest.mark.parametrize(
    ("kwargs", "flags"),
    [
        ({"externalize_embedder": False}, ()),
        ({}, ("--noexternalize_embedder",)),
        ({}, ("--externalize-embedder=False",)),
        ({}, ("--externalize_embedder", "--externalize_embedder=0")),
    ],
)
def test_gemma4_refuses_an_embedder_opt_out(tmp_path, kwargs, flags):
    """Gemma 4 does not export without `--externalize_embedder`: a caller's
    opt-out, from `convert`'s own option or a `--flag=` Fire reads as False, is
    refused with that reason instead of being overridden by the rule."""
    with pytest.raises(models.FlagRefused, match="this family needs --externalize_embedder") as exc:
        ExportRequest(
            model=GEMMA4,
            output_dir=tmp_path / "out",
            recipes=MEASURED_RECIPES,
            extra_flags=flags,
            **kwargs,
        )
    assert "External embedder is required for Gemma4" in str(exc.value)


@pytest.mark.parametrize(
    ("kwargs", "flags", "source"),
    [
        ({}, (), "family rule"),
        ({"externalize_embedder": True}, (), "caller"),
        ({}, ("--externalize-embedder",), "caller"),
    ],
)
def test_gemma4_names_who_set_the_embedder_flag(tmp_path, kwargs, flags, source):
    """The flag Gemma 4's rule adds is reported as the rule's, not the caller's,
    and the exporter gets it exactly once."""
    request = ExportRequest(
        model=GEMMA4,
        output_dir=tmp_path / "out",
        recipes=MEASURED_RECIPES,
        extra_flags=flags,
        **kwargs,
    )
    argv = request.argv("dynamic_wi8_afp32")
    assert request.externalize_source == source
    assert request.as_dict()["externalize_embedder"] is True
    assert sum(models.flag_key(a) == "externalize_embedder" for a in argv) == 1
    assert ("--externalize_embedder" in request.plan.added) is (source == "family rule")


def test_the_argv_sets_each_keyword_once_and_none_of_litetunes_three(request_for):
    """`ExportRequest.argv` checks what the plan promises."""
    request = request_for(MEASURED_RECIPES)
    duplicated = (*request.extra_flags, "--externalize-embedder")
    object.__setattr__(request, "extra_flags", duplicated)
    with pytest.raises(ValueError, match="more than once"):
        request.argv("dynamic_wi8_afp32")
    object.__setattr__(request, "extra_flags", (*duplicated[:-1], "--model=other"))
    with pytest.raises(ValueError, match="Use --model"):
        request.argv("dynamic_wi8_afp32")


def test_a_refusal_before_export_claims_no_artifact(toolchain, qwen_request):
    toolchain.chat_template = None
    result = run_export(qwen_request())
    assert result.artifacts == []
    assert NOT_VERIFIED not in result.limitations
    assert result.as_dict()["unverified_reason"] == NO_ARTIFACT


def test_an_export_with_an_artifact_says_it_is_not_verified(toolchain, request_for):
    result = run_export(request_for(("dynamic_wi8_afp32",)))
    assert result.artifacts
    assert result.limitations[0] == NOT_VERIFIED
    assert result.as_dict()["unverified_reason"] == NOT_VERIFIED


def test_a_composing_plan_given_no_template_is_recorded_not_raised(qwen_request):
    """`export_recipe` called directly, as `test_exits` does: there is no
    command line to give the exporter, and that is a record, not an error."""
    export = export_recipe(qwen_request(), "dynamic_wi8_afp32")
    assert export.check.outcome is Outcome.UNCHECKED
    assert "none was composed" in export.check.detail
    assert not export.attempted


def test_a_family_with_a_packaged_template_composes_nothing(toolchain, request_for):
    result = run_export(request_for(("dynamic_wi8_afp32",)))

    assert result.outcome is Outcome.PASSED
    assert toolchain.compositions == []
    assert not any(c.name == CHAT_TEMPLATE_CHECK for c in result.checks.checks)
    assert result.chat_template is None


@pytest.mark.parametrize(
    ("model", "marker", "canonical"),
    [
        (QWEN3, CANONICAL_OR_COMPOSED_FLAG, models.QWEN3_CANONICAL_TEMPLATE),
        ("Qwen/Qwen2.5-0.5B-Instruct", COMPOSED_TEMPLATE_FLAG, None),
    ],
)
def test_the_marker_never_reaches_an_exporter_argv(qwen_request, model, marker, canonical):
    """It describes a file, and the exporter would look it up as a repository."""
    request = qwen_request(model)
    assert request.composes_chat_template
    assert marker in request.extra_flags
    assert request.canonical_template == canonical
    with pytest.raises(ValueError, match="and none was given"):
        request.argv("dynamic_wi8_afp32")
    composed = Path("/composed/t.jinja")
    argv = request.argv("dynamic_wi8_afp32", composed)
    # The path as this platform spells it: `\composed\t.jinja` on Windows.
    assert _template_flags(argv) == [f"{models.TEMPLATE_FLAG}={composed}"]
    assert not any(m in a for m in models.WRITTEN_AT_EXPORT for a in argv)


def test_a_marker_in_any_other_shape_is_refused_too(qwen_request):
    request = qwen_request()
    flags = (*request.extra_flags, f"--other={models.COMPOSED_TEMPLATE}")
    object.__setattr__(request, "extra_flags", flags)
    with pytest.raises(ValueError, match="would hand the exporter the marker"):
        request.argv("dynamic_wi8_afp32", Path("/composed/t.jinja"))


def test_the_composed_path_is_absolute_for_a_relative_output_dir(tmp_path, monkeypatch):
    """So that the argv and the report name the file however they are read."""
    monkeypatch.chdir(tmp_path)
    request = ExportRequest(model=QWEN3, output_dir=Path("rel"), recipes=("dynamic_wi8_afp32",))
    path = request.composed_template_path("ab" * 32)
    assert path.is_absolute()
    assert path == Path.cwd() / "rel" / f"{COMPOSED_TEMPLATE_PREFIX}{'ab' * 6}.jinja"


def test_no_recipe_name_can_be_the_composed_file(tmp_path):
    """A recipe names a directory beside the composed file; the prefix is one
    the recipe-name rule refuses."""
    name = f"{COMPOSED_TEMPLATE_PREFIX}{'ab' * 6}.jinja"
    with pytest.raises(ValueError, match="not a usable recipe name"):
        ExportRequest(model=QWEN3, output_dir=tmp_path, recipes=(name,))


def test_the_chat_template_script_is_python_that_imports_nothing_from_litetune():
    """It runs in the export environment, which does not install litetune."""
    import re

    compile(_COMPOSE_SCRIPT, "compose_template.py", "exec")
    assert not re.search(r"^\s*(?:import|from)\s+litetune\b", _COMPOSE_SCRIPT, re.MULTILINE)


def test_the_script_decides_local_or_hub_as_the_exporter_does(tmp_path):
    """`get_model_path_type`, export_hf/core/utils.py:110-139."""
    model_path_type = _exec_script(_COMPOSE_SCRIPT)["model_path_type"]
    assert model_path_type(str(tmp_path)) == "local"
    assert model_path_type("./nowhere") == "local_not_found"
    assert model_path_type("Qwen/Qwen3-0.6B") == "repo_id"
    assert model_path_type("gpt2") == "repo_id"
    assert model_path_type("a/b/c") == "unknown"


def _exec_script(source: str) -> dict:
    """The script's functions, without running its `__main__` (or importing
    transformers, which a stub stands in for)."""
    import types

    stub = types.ModuleType("transformers")
    namespace: dict = {"__name__": "compose_script_under_test"}
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = stub
    try:
        exec(compile(source, "compose_template.py", "exec"), namespace)  # noqa: S102
    finally:
        if saved is None:
            del sys.modules["transformers"]
        else:
            sys.modules["transformers"] = saved
    return namespace


def test_requirement_names_skip_a_pin_whose_marker_is_false_here():
    """`resolve_toolchain` reads what is installed; a pin for another Python is
    not installed here and is not reported missing."""
    other = "3.99" if sys.version_info[:2] != (3, 99) else "3.98"
    names = requirement_names(
        [f'onlythere==1.0; python_version == "{other}"', "here==2.0", "pkg @ https://h/p.whl"]
    )
    # A named direct URL keeps its name, which `pip show` answers to.
    assert names == ("here", "pkg")


def test_requirement_names_read_a_pin_as_pip_does():
    """Leading whitespace is no part of a name to pip (`packaging.Requirement`)."""
    assert requirement_names([" pkg==1.0", "other[extra]==2.0"]) == ("pkg", "other")


# ---------------------------------------------------------------------------
# Decision checkpoints
# ---------------------------------------------------------------------------


def _decision_checkpoint(tmp_path) -> Path:
    from litetune.decisions import decision_from

    asked = decision_from({"question": "Which?", "options": [{"id": "a"}, {"id": "b"}]}, "q")
    d = tmp_path / "model"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "gemma3_text"}), encoding="utf-8")
    (d / "litetune.json").write_text(
        json.dumps(
            {
                "base_model": "google/gemma-3-270m-it",
                "decision": asked.as_dict(),
                "decision_sha256": asked.digest(),
            }
        ),
        encoding="utf-8",
    )
    return d


def test_a_decision_checkpoint_is_exported_with_the_small_prefill_signature(tmp_path):
    from litetune.export import ExportRequest

    checkpoint = _decision_checkpoint(tmp_path)

    request = ExportRequest(
        model=str(checkpoint), output_dir=tmp_path / "out", recipes=["dynamic_wi8_afp32"]
    )

    assert "--prefill_lengths=8,128" in request.extra_flags
    assert "--prefill_lengths=8,128" in request.argv("dynamic_wi8_afp32")
    record = request.as_dict()["decision"]
    assert record["prefill_lengths"] == "--prefill_lengths=8,128"
    assert record["prefill_lengths_source"] == "decision checkpoint"
    assert record["decision_sha256"].startswith("sha256:")


def test_the_callers_prefill_lengths_are_kept_and_said_to_be_theirs(tmp_path):
    from litetune.export import ExportRequest

    checkpoint = _decision_checkpoint(tmp_path)

    request = ExportRequest(
        model=str(checkpoint),
        output_dir=tmp_path / "out",
        recipes=["dynamic_wi8_afp32"],
        extra_flags=["--prefill-lengths=16,128"],
    )

    assert not any(f.startswith("--prefill_lengths=8,128") for f in request.extra_flags)
    assert request.as_dict()["decision"]["prefill_lengths_source"] == "caller"


def test_a_checkpoint_without_a_decision_gets_no_prefill_flag(tmp_path):
    from litetune.export import ExportRequest

    checkpoint = _checkpoint(tmp_path, recorded="google/gemma-3-270m-it")

    request = ExportRequest(
        model=str(checkpoint), output_dir=tmp_path / "out", recipes=["dynamic_wi8_afp32"]
    )

    assert not any("prefill" in f for f in request.extra_flags)
    assert request.as_dict()["decision"] is None
