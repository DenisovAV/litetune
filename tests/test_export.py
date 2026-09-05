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
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from litetune import envs
from litetune.checks import Outcome
from litetune.events import Event, EventStream
from litetune.export import (
    KNOWN_RECIPES,
    MEASURED_RECIPES,
    TOOLCHAIN_DEFAULT_RECIPE,
    ExportRequest,
    NoRecipesRequested,
    SizeComparison,
    Uncompared,
    compare_sizes,
    parse_pip_freeze,
    requirement_names,
    run_export,
)

# `pip freeze`, not `pip show`: show reports only what the environment declares,
# and those are already pinned. The 2026-08-30 breakage lived in the transitive
# closure -- `absl-py` and `flatbuffers` below stand for it, and are exactly the
# packages a post-hoc diagnosis would need.
PIP_FREEZE = """absl-py==2.1.0
flatbuffers==24.3.25
litert-lm==0.16.1
litert-lm-builder==0.16.1
litert-torch-nightly==0.10.0.dev20260826
numpy==2.0.2
"""


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

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(Call(argv=list(args), timeout=timeout, kwargs=dict(kwargs)))
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, self.pip_returncode, self.pip_stdout, "")
        if args[0] == "python" and str(args[1]).endswith("repack.py"):
            return self._repack(args, timeout)
        assert args[:2] == ["litert-torch", "export_hf"], args
        flags = dict(a.removeprefix("--").split("=", 1) for a in args[2:] if "=" in a)
        run = self.runs.get(flags["quantization_recipe"], self.default)
        if run.raises is not None:
            raise run.raises
        out_dir = Path(flags["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, size in run.writes:
            (out_dir / name).write_bytes(b"\0" * size)
        return subprocess.CompletedProcess(args, run.returncode, run.stdout, run.stderr)

    @property
    def exports(self) -> list[Call]:
        return [c for c in self.calls if c.argv[0] == "litert-torch"]

    @property
    def repacks(self) -> list[Call]:
        return [c for c in self.calls if c.argv[0] == "python"]

    def _repack(self, args, timeout: int) -> subprocess.CompletedProcess:
        if self.repack_script_returncode:
            return subprocess.CompletedProcess(
                args, self.repack_script_returncode, "", "Traceback: boom"
            )
        script, spec, result = str(args[1]), str(args[2]), str(args[3])
        shim = Path(spec).parent / "shim" / "litert_lm_builder"
        shim.mkdir(parents=True, exist_ok=True)
        (shim / "__init__.py").write_text("")
        (shim / "litertlm_builder.py").write_text(
            FAKE_BUILDER_MODULE.format(
                default_toml=self.repack_toml if self.repack_toml is not None else FAKE_TOML,
                unpack_fails=self.repack_unpack_fails,
                pack_fails=self.repack_pack_fails,
                pack_drifts=self.repack_pack_drifts,
                pack_misplaces=self.repack_pack_misplaces,
            ),
            encoding="utf-8",
        )
        return subprocess.run(
            [sys.executable, script, spec, result],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(shim.parent)},
        )


# A stand-in for the real `litert_lm_builder.litertlm_builder`. A fake bundle
# is opaque bytes of the caller's size; the TOML that describes it lives in a
# sidecar `<bundle>.toml` (a real bundle stores it in a header, which is why
# the size does not grow with one key). `unpack` writes that TOML plus empty
# section files; `pack` reads a TOML, writes a bundle of the original's size
# and the sidecar beside it. Knobs are formatted in as Python literals;
# braces that belong to the module are doubled for `str.format`.
FAKE_BUILDER_MODULE = """
import os
import re
from pathlib import Path

DEFAULT_TOML = {default_toml!r}
UNPACK_FAILS = {unpack_fails}
PACK_FAILS = {pack_fails}
PACK_DRIFTS = {pack_drifts}
PACK_MISPLACES = {pack_misplaces}


def _sidecar(bundle_path):
    return Path(str(bundle_path) + ".toml")


def _toml_of(bundle_path):
    side = _sidecar(bundle_path)
    return side.read_text(encoding="utf-8") if side.exists() else DEFAULT_TOML


def unpack(litertlm_path, output_dir, jinja_prompt_template_path=None):
    if UNPACK_FAILS:
        raise RuntimeError("unpack: boom")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    toml = _toml_of(litertlm_path)
    (out / "model.toml").write_text(toml, encoding="utf-8")
    for m in re.finditer(r'data_path\\s*=\\s*"([^"]+)"', toml):
        (out / os.path.basename(m.group(1))).write_bytes(b"x")
    return str(out / "model.toml")


def pack(toml_path, output_path, jinja_prompt_template_path=None):
    if PACK_FAILS:
        raise RuntimeError("pack: boom")
    toml = Path(toml_path).read_text(encoding="utf-8")
    # Paths were made absolute for the build; store them by name, as unpack does.
    toml = re.sub(
        r'(data_path\\s*=\\s*")([^"]+)(")',
        lambda m: m.group(1) + os.path.basename(m.group(2)) + m.group(3),
        toml,
    )
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
    target = Path(output_path)
    original = target.parent.parent / target.name
    size = original.stat().st_size if original.exists() else 4096
    target.write_bytes(b"LITERTLM" + b"\\0" * max(0, size - 8))
    _sidecar(target).write_text(toml, encoding="utf-8")
    if original.exists():
        _sidecar(original).write_text(toml, encoding="utf-8")
    return str(target)
"""


FAKE_TOML = """[system_metadata]
entries = []

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


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    """A fake export environment: provisioning writes a marker, `run` writes files."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".litetune-ready").write_text(self.identity)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
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
    assert any("experimental_wi2" in limit for limit in result.limitations)
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
    assert result.toolchain.resolved["litert-torch-nightly"] == "0.10.0.dev20260826"
    assert result.toolchain.resolved["numpy"] == "2.0.2"
    # The point of freezing: packages nobody declared are recorded too, because
    # that is where the toolchain actually moved under us.
    assert result.toolchain.resolved["flatbuffers"] == "24.3.25"
    assert result.toolchain.missing == ()
    # The declared pins travel with what they resolved to; neither implies the other.
    assert "litert-lm==0.16.1" in result.toolchain.declared
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
        return (
            isinstance(other, str)
            and other.startswith("--jinja_chat_template_override=")
            and other.endswith("templates/functiongemma.jinja")
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
        "flatbuffers": "24.3.25",
        "litert-lm": "0.16.1",
        "litert-lm-builder": "0.16.1",
        "litert-torch-nightly": "0.10.0.dev20260826",
        "numpy": "2.0.2",
    }


def test_parse_pip_freeze_keeps_unpinnable_entries_verbatim():
    # Editable and URL installs are the hardest to reproduce, so dropping them
    # would lose exactly the lines a post-hoc diagnosis needs most.
    frozen = parse_pip_freeze("-e /src/litert-torch\nfoo @ https://example.invalid/foo.whl\n")
    assert "/src/litert-torch" in frozen["litert-torch"]
    assert frozen["foo"].startswith("foo @ https://")


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


def test_an_unreadable_sidecar_reaches_the_report(tmp_path):
    """Recorded in the hint is not enough: nobody reads a hint."""
    from litetune.export import ExportRequest, run_export

    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "litetune.json").write_text('{"base_model": "goo', encoding="utf-8")

    result = run_export(
        ExportRequest(
            model=str(checkpoint),
            base_model="google/functiongemma-270m-it",
            output_dir=tmp_path / "out",
            recipes=("dynamic_wi8_afp32",),
        )
    )
    assert any("could not be read" in line for line in result.limitations)


# ---------------------------------------------------------------------------
# GPU activation type
# ---------------------------------------------------------------------------


def _fake_env(toolchain, tmp_path):
    """A StageEnv whose `run` is the fake; the repack only needs `run`."""
    from litetune import envs

    return envs.StageEnv(name="export", python_ceiling=(3, 12), requirements=())


def _artifact(tmp_path, size=4096) -> Path:
    path = tmp_path / "model.litertlm"
    path.write_bytes(b"\0" * size)
    return path


def _toml_in(artifact: Path) -> str:
    """What the fake builder recorded for this bundle (see FAKE_BUILDER_MODULE)."""
    return Path(str(artifact) + ".toml").read_text(encoding="utf-8")


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
    (prefill,) = [t for t in _sections(_toml_in(artifact)) if '"prefill_decode"' in t]
    (embedder,) = [t for t in _sections(_toml_in(artifact)) if '"embedder"' in t]
    assert 'key = "prefer_activation_type"' in prefill and f'value = "{GPU_ACTIVATION}"' in prefill
    assert "prefer_activation_type" not in embedder, "the key belongs to prefill_decode only"


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
        "CPU-only (GPU activations not set): the repack script exited 1: "
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
