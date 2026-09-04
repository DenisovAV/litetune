"""The command line.

`verify` has to work standalone -- one artifact, one reference, one JSONL, no
job spec and no run directory -- so these tests drive `main()` exactly as a user
would, with only the two backends faked. Every other subcommand is driven the
same way: no network, no accelerator, no toolchain.
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from conftest import FakeBackend, correct_texts, labelled_rows

from litetune import envs
from litetune import verify as verify_module
from litetune.cli import build_parser, main, measurements_from_verify, summarise
from litetune.evaluate import PromptMode
from litetune.verify import BackendPair, Status


@pytest.fixture
def fake_backends(monkeypatch):
    """Install canned backends behind the CLI's own construction path."""

    def install(candidate_texts, reference_texts):
        def build(request):
            return BackendPair(
                candidate=FakeBackend(texts=candidate_texts),
                reference=FakeBackend(model=request.reference, texts=reference_texts),
            )

        monkeypatch.setattr(verify_module, "build_backends", build)

    return install


def test_the_contract_takes_its_stop_token_from_the_training_record(tmp_path):
    """The bundle should declare the terminator the model was trained on.

    `Contract.stop_tokens` defaulted to empty and was populated only by hand.
    An empty declaration reads to a runtime as "this model has no stop token",
    and a runtime that waits for one the model never emits does not stop -- the
    same failure as training the wrong terminator, arriving from the other end.
    """
    metrics = tmp_path / "train.json"
    metrics.write_text(
        json.dumps(
            {"turn_terminator": {"ids": [106], "source": "chat_template", "text": "<end_of_turn>"}}
        ),
        encoding="utf-8",
    )

    (tmp_path / "m.litertlm").write_text("{}", encoding="utf-8")
    (tmp_path / "d.json").write_text("[]", encoding="utf-8")

    # Through the CLI, not by re-reading the file this test just wrote. The
    # first version asserted that `json.loads` works: deleting the whole
    # `--train-metrics` block from `cli.py` left the suite green.
    main(
        [
            "bundle",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            str(tmp_path / "m.litertlm"),
            "--declarations",
            str(tmp_path / "d.json"),
            "--prompt-mode",
            "prerendered",
            "--base-model",
            "g",
            "--base-model-revision",
            "r",
            "--train-metrics",
            str(metrics),
        ]
    )

    contract = json.loads((tmp_path / "out" / "contract.json").read_text(encoding="utf-8"))
    assert contract["stop_tokens"] == ["<end_of_turn>"]
    assert any("chat_template" in note for note in contract["notes"])


def test_an_explicit_stop_token_still_wins(tmp_path):
    """The record is a default, not an override of the operator.

    The first version asserted `args.stop_token == ["<custom>"]` -- that
    `action="append"` works. Inverting the precedence rule in `cli.py` so the
    record overrides the operator, the exact opposite of this test's name, left
    the suite green.
    """
    metrics = tmp_path / "train.json"
    metrics.write_text(
        json.dumps({"turn_terminator": {"text": "<end_of_turn>", "source": "chat_template"}}),
        encoding="utf-8",
    )
    (tmp_path / "m.litertlm").write_text("{}", encoding="utf-8")
    (tmp_path / "d.json").write_text("[]", encoding="utf-8")

    main(
        [
            "bundle",
            "--output-dir",
            str(tmp_path / "out"),
            "--model",
            str(tmp_path / "m.litertlm"),
            "--declarations",
            str(tmp_path / "d.json"),
            "--prompt-mode",
            "prerendered",
            "--base-model",
            "g",
            "--base-model-revision",
            "r",
            "--train-metrics",
            str(metrics),
            "--stop-token",
            "<custom>",
        ]
    )

    contract = json.loads((tmp_path / "out" / "contract.json").read_text(encoding="utf-8"))
    assert contract["stop_tokens"] == ["<custom>"]


def test_max_tokens_reaches_the_request_it_names():
    """The generation limit was a spec field nothing read.

    `EvalSpec.max_tokens` was validated and hashed into the run's identity while
    `DecodeConfig` stayed at its default, so two runs declaring different limits
    were counted as different experiments and behaved identically. A setting
    that changes the bookkeeping and not the behaviour is worse than a missing
    one: the missing one is visible.
    """

    from litetune.cli import build_parser

    args = build_parser().parse_args(
        [
            "verify",
            "--model",
            "m.litertlm",
            "--reference",
            "r",
            "--data",
            "d.jsonl",
            "--max-tokens",
            "512",
        ]
    )

    # Through `run_verify`, not by re-doing `dataclasses.replace` here. The
    # first version asserted that `replace` works: changing `_verify` to pass
    # `decode=GREEDY` and discard `--max-tokens` entirely left the suite green.
    seen: dict[str, int] = {}

    def capture(request, **kwargs):
        seen["max_tokens"] = request.decode.max_tokens
        raise SystemExit(0)

    import litetune.cli as cli_module

    original = cli_module.run_verify
    cli_module.run_verify = capture
    try:
        with pytest.raises(SystemExit):
            cli_module._verify(args)
    finally:
        cli_module.run_verify = original

    assert seen["max_tokens"] == 512


def test_without_the_flag_the_default_limit_is_unchanged():
    from litetune.cli import build_parser
    from litetune.evaluate import GREEDY

    args = build_parser().parse_args(
        ["verify", "--model", "m.litertlm", "--reference", "r", "--data", "d.jsonl"]
    )

    assert args.max_tokens is None
    assert GREEDY.max_tokens == 256


def test_verify_runs_standalone_and_prints_a_result(tmp_path, capsys, write_split, fake_backends):
    rows = labelled_rows(16)
    fake_backends(correct_texts(rows), correct_texts(rows))
    code = main(
        [
            "verify",
            "--model",
            str(tmp_path / "model.litertlm"),
            "--reference",
            "org/reference",
            "--data",
            str(write_split(rows)),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "status: passed" in captured.out
    # Progress belongs to stderr; the result owns stdout.
    assert "verify" in captured.err


def test_json_writes_the_manifest_to_stdout(tmp_path, capsys, write_split, fake_backends):
    rows = labelled_rows(16)
    fake_backends(correct_texts(rows), correct_texts(rows))
    main(
        [
            "verify",
            "--model",
            str(tmp_path / "model.litertlm"),
            "--reference",
            "org/reference",
            "--data",
            str(write_split(rows)),
            "--json",
        ]
    )
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["schema"] == "litetune.verify/1"
    assert manifest["status"] == "passed"


def test_limit_is_passed_through(tmp_path, capsys, write_split, fake_backends):
    rows = labelled_rows(40)
    fake_backends(correct_texts(rows)[:5], correct_texts(rows)[:5])
    main(
        [
            "verify",
            "--model",
            str(tmp_path / "model.litertlm"),
            "--reference",
            "org/reference",
            "--data",
            str(write_split(rows)),
            "--limit",
            "5",
            "--json",
        ]
    )
    assert json.loads(capsys.readouterr().out)["data"]["n"] == 5


def test_unmeasured_does_not_exit_zero(tmp_path, capsys, write_split, fake_backends):
    # A release gate must not read "we never measured quality" as success.
    rows = [{"prompt": f"do {i}"} for i in range(8)]
    fake_backends(["call:a{x:<escape>1<escape>}"], ["call:a{x:<escape>1<escape>}"])
    code = main(
        [
            "verify",
            "--model",
            str(tmp_path / "model.litertlm"),
            "--reference",
            "org/reference",
            "--data",
            str(write_split(rows)),
        ]
    )
    assert code == 3
    assert "not measured" in capsys.readouterr().out


def test_an_unexpected_crash_makes_no_claim_about_the_model(
    tmp_path, capsys, write_split, monkeypatch
):
    def explode(*args, **kwargs):
        raise RuntimeError("something the harness never anticipated")

    monkeypatch.setattr("litetune.cli.run_verify", explode)
    code = main(
        [
            "verify",
            "--model",
            str(tmp_path / "model.litertlm"),
            "--reference",
            "org/reference",
            "--data",
            str(write_split(labelled_rows(2))),
        ]
    )
    captured = capsys.readouterr()
    assert code == 4
    assert "no claim is made about the model" in captured.err
    assert captured.out == ""


def test_model_and_reference_and_data_are_all_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["verify", "--model", "m.litertlm"])


def test_summary_never_prints_a_bare_number():
    manifest = {
        "status": "passed",
        "quality": {
            "available": True,
            "candidate": {
                "exact_match": {"value": 0.8906, "ci95": 0.0284, "n": 640},
                "name_accuracy": {"value": 0.995, "ci95": 0.0055, "n": 640},
                "argument_accuracy": {"available": True, "value": 0.8448, "ci95": 0.0281},
            },
            "reference": {
                "exact_match": {"value": 0.9094, "ci95": 0.026, "n": 640},
                "name_accuracy": {"value": 0.9953, "ci95": 0.0053, "n": 640},
                "argument_accuracy": {"available": True, "value": 0.8744, "ci95": 0.0258},
            },
        },
        "attribution": {
            "conversion_cost": {"available": True, "value": 0.0297, "ci95": 0.017, "resolved": True}
        },
        "limitations": ["measured on the cpu backend"],
    }
    text = "\n".join(summarise(manifest))
    assert "0.8906 ±0.0284 (n=640)" in text
    assert "operation 0.9950" in text
    assert "conversion_cost: +0.0297 ±0.0170" in text


def test_summary_says_when_a_difference_is_unresolved():
    manifest = {
        "status": "passed",
        "quality": {"available": False, "reason": "no labelled held-out data"},
        "attribution": {
            "conversion_cost": {
                "available": True,
                "value": 0.0297,
                "ci95": 0.0385,
                "resolved": False,
            }
        },
        "limitations": [],
    }
    text = "\n".join(summarise(manifest))
    assert "unresolved at this sample size" in text


def test_summary_of_a_non_passing_run_says_the_model_is_not_verified():
    text = "\n".join(
        summarise({"status": Status.FAILED_SMOKE.value, "quality": {"available": False}})
    )
    assert "not verified" in text


# ---------------------------------------------------------------------------
# The other subcommands
# ---------------------------------------------------------------------------


@dataclass
class FakeToolchain:
    """Stands in for `envs.StageEnv.run`: writes files instead of converting."""

    calls: list[list[str]] = field(default_factory=list)
    pip_stdout: str = "transformers==5.5.0\n"

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, 0, self.pip_stdout, "")
        flags = dict(a.removeprefix("--").split("=", 1) for a in args[2:] if "=" in a)
        out_dir = Path(flags["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.litertlm").write_bytes(b"\0" * 4096)
        return subprocess.CompletedProcess(args, 0, "", "")

    @property
    def exports(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "litert-torch"]


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeToolchain:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".litetune-ready").write_text(self.identity)
        return self.path

    fake = FakeToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


# -- convert ----------------------------------------------------------------


def test_convert_sweeps_the_recipes_it_was_given(toolchain, tmp_path, capsys):
    code = main(
        [
            "convert",
            "--model",
            "google/functiongemma-270m-it",
            "--output-dir",
            str(tmp_path / "out"),
            "--recipe",
            "dynamic_wi8_afp32",
            "--recipe",
            "weight_only_wi8_afp32",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert [c[2] for c in toolchain.exports] == ["--model=google/functiongemma-270m-it"] * 2
    assert "dynamic_wi8_afp32" in captured.out
    # Export establishes that a file exists and nothing else.
    assert "produced, not verified" in captured.out
    assert "export" in captured.err


def test_convert_without_a_recipe_refuses_rather_than_choosing_one(tmp_path, capsys):
    code = main(["convert", "--model", "org/m", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 4
    # The refusal carries the measurement that justifies it.
    assert "0.024 exact match" in captured.err
    assert captured.out == ""


def test_convert_refuses_the_model_type_override(tmp_path, capsys):
    code = main(
        [
            "convert",
            "--model",
            "google/gemma-4-E2B-it",
            "--output-dir",
            str(tmp_path),
            "--recipe",
            "dynamic_wi4b32_afp32",
            "--flag=--litert_lm_model_type_override=gemma4",
        ]
    )
    captured = capsys.readouterr()
    assert code == 4
    assert "refuses" in captured.err
    assert "generic_model" in captured.err
    assert captured.out == ""


def test_convert_auto_corrects_a_gemma4_export_and_prints_the_reason(toolchain, tmp_path, capsys):
    code = main(
        [
            "convert",
            "--model",
            "google/gemma-4-E2B-it",
            "--output-dir",
            str(tmp_path / "out"),
            "--recipe",
            "dynamic_wi4b32_afp32",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    argv = toolchain.exports[0]
    assert "--externalize_embedder" in argv
    assert "--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm" in argv
    # Not "added --externalize_embedder": it is on by default now, so the caller
    # supplies it and the rule has nothing to add. It is still in the argv --
    # asserted above -- and the reason it is required lives in `--help` and in
    # `models.py` rather than in this run's output.
    assert "added --jinja_chat_template_override" in captured.out
    # The ceiling is stated, not implied.
    assert "NOT equivalent to Google's published" in captured.out
    # And the reason for the flag the rule still adds reached the user.
    assert "MiniJinja" in captured.err or "chat_template" in captured.err


def test_convert_will_not_guess_a_gemma4_template_and_exits_could_not_check(
    toolchain, tmp_path, capsys
):
    code = main(
        [
            "convert",
            "--model",
            "google/gemma-4-it",
            "--output-dir",
            str(tmp_path / "out"),
            "--recipe",
            "dynamic_wi4b32_afp32",
        ]
    )
    captured = capsys.readouterr()
    assert code == 4
    assert toolchain.exports == []
    assert "not attempted" in captured.out


def test_convert_json_writes_the_report_to_stdout(toolchain, tmp_path, capsys):
    main(
        [
            "convert",
            "--model",
            "google/functiongemma-270m-it",
            "--output-dir",
            str(tmp_path / "out"),
            "--recipe",
            "dynamic_wi8_afp32",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "litetune.export/1"
    assert report["verified"] is False
    # The rules the export ran under are part of the record, not a detail: this
    # family needs a model-type override the caller did not pass, and a report
    # that did not name it could not explain the resulting bundle.
    assert report["model_rules"]["family"] == "functiongemma"


# -- tune -------------------------------------------------------------------


def test_tune_parses_and_dispatches(monkeypatch, tmp_path, capsys):
    from litetune import cli
    from litetune.checks import Check, CheckSet
    from litetune.tune import TuneResult

    seen = {}

    def fake_run_tune(request, events=None):
        seen["request"] = request
        checks = CheckSet(name="train")
        checks.add(Check.passed("training run", "faked"))
        return TuneResult(request=request, checks=checks)

    monkeypatch.setattr(cli, "run_tune", fake_run_tune)
    code = main(
        [
            "tune",
            "--model",
            "google/functiongemma-270m-it",
            "--data",
            str(tmp_path / "train.jsonl"),
            "--output-dir",
            str(tmp_path / "run"),
            "--method",
            "lora",
            "--prompt-mode",
            "prerendered",
            "--epochs",
            "2",
            "--lora-rank",
            "8",
        ]
    )
    request = seen["request"]
    assert code == 0
    assert request.method == "lora"
    assert request.epochs == 2.0
    assert request.lora_rank == 8
    assert request.prompt_mode is PromptMode.PRERENDERED
    # The rate is the method's, and the report says which.
    assert request.rate == 2e-4
    assert "default for the method" in capsys.readouterr().out


def test_tune_requires_the_prompt_mode_to_be_stated(tmp_path):
    # There is no defensible default: the two conventions are mutually
    # exclusive and the wrong one is a fluent wrong answer.
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "tune",
                "--model",
                "m",
                "--data",
                str(tmp_path / "d.jsonl"),
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_tune_refuses_an_unknown_method(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "tune",
                "--model",
                "m",
                "--data",
                str(tmp_path / "d.jsonl"),
                "--output-dir",
                str(tmp_path),
                "--prompt-mode",
                "prerendered",
                "--method",
                "distillation",
            ]
        )


# -- prepare ----------------------------------------------------------------


def _dataset(tmp_path: Path, n: int = 12) -> Path:
    path = tmp_path / "raw.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "prompt": f"set the background to colour {i}",
                    "target": {"name": "change_background_color", "args": {"color": f"c{i}"}},
                }
            )
            for i in range(n)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_prepare_writes_a_content_keyed_split(tmp_path, capsys):
    code = main(
        [
            "prepare",
            "--data",
            str(_dataset(tmp_path)),
            "--output-dir",
            str(tmp_path / "out"),
            "--context-length",
            "1024",
        ]
    )
    captured = capsys.readouterr()
    assert (tmp_path / "out" / "train.jsonl").is_file()
    assert (tmp_path / "out" / "heldout.jsonl").is_file()
    assert "12 rows" in captured.out
    # No tokenizer was supplied, so no example was measured against the context
    # window: that check did not run, and the stage says so rather than passing.
    assert code == 4
    assert "token lengths: not measured" in captured.out


def test_prepare_json_carries_the_spec_fragment(tmp_path, capsys):
    main(
        [
            "prepare",
            "--data",
            str(_dataset(tmp_path)),
            "--output-dir",
            str(tmp_path / "out"),
            "--context-length",
            "1024",
            "--heldout-size",
            "4",
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["heldout"]["n"] == 4
    assert report["spec_fragment"]["eval"]["heldout_content_sha256"]


def test_prepare_on_a_file_that_is_not_there_makes_no_claim(tmp_path, capsys):
    code = main(
        [
            "prepare",
            "--data",
            str(tmp_path / "absent.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--context-length",
            "1024",
        ]
    )
    assert code == 4
    assert "no claim is made about the model" in capsys.readouterr().err


# -- bundle -----------------------------------------------------------------


@pytest.fixture
def deliverable(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"weights")
    declarations = tmp_path / "tools.json"
    declarations.write_text(json.dumps([{"name": "change_background_color"}]), encoding="utf-8")
    return model, declarations


def _bundle_argv(tmp_path, model, declarations, *extra):
    return [
        "bundle",
        "--output-dir",
        str(tmp_path / "bundle"),
        "--model",
        str(model),
        "--declarations",
        str(declarations),
        "--prompt-mode",
        "prerendered",
        "--base-model",
        "google/functiongemma-270m-it",
        "--base-model-revision",
        "a" * 40,
        *extra,
    ]


def test_bundle_assembles_the_four_members(tmp_path, deliverable, capsys):
    model, declarations = deliverable
    code = main(_bundle_argv(tmp_path, model, declarations))
    out = capsys.readouterr().out
    bundle_dir = tmp_path / "bundle"
    assert (bundle_dir / "contract.json").is_file()
    assert (bundle_dir / "declarations.json").is_file()
    assert (bundle_dir / "report.json").is_file()
    assert (bundle_dir / "manifest.json").is_file()
    # No status was carried in, so the run is inconclusive: bundling re-measures
    # nothing and must not decide its own verdict.
    assert code == 2
    assert "prompt mode prerendered" in out
    assert "measurements not made" in out


def test_bundle_requires_a_prompt_mode(tmp_path, deliverable):
    model, declarations = deliverable
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "bundle",
                "--output-dir",
                str(tmp_path / "bundle"),
                "--model",
                str(model),
                "--declarations",
                str(declarations),
                "--base-model",
                "org/base",
                "--base-model-revision",
                "a" * 40,
            ]
        )


def test_bundle_carries_a_verify_manifest_into_the_report(tmp_path, deliverable, capsys):
    model, declarations = deliverable
    manifest = tmp_path / "verify.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "passed",
                "reference": {"role": "float_twin"},
                "measurements": {
                    "candidate": {"label": "candidate"},
                    "reference": {"label": "reference"},
                },
                "attribution": {"conversion_cost": {"available": True, "value": 0.03}},
                "limitations": ["measured on the cpu backend"],
            }
        ),
        encoding="utf-8",
    )
    code = main(
        _bundle_argv(tmp_path, model, declarations, "--verify-manifest", str(manifest), "--json")
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["status"] == "passed"
    assert sorted(report["measurements"]) == ["tuned_converted", "tuned_float"]
    # The third point was never taken, and the bundle says which one and why.
    assert [m["point"] for m in report["measurements_not_made"]] == ["base_float"]
    assert "measured on the cpu backend" in report["limitations"]


def test_a_reference_that_was_the_untuned_base_is_filed_under_the_right_point():
    points = measurements_from_verify(
        {
            "reference": {"role": "untuned_base"},
            "measurements": {"candidate": {}, "reference": {}},
        }
    )
    assert sorted(points) == ["base_float", "tuned_converted"]


def test_a_reference_that_was_never_run_is_not_recorded_as_a_measurement():
    points = measurements_from_verify(
        {
            "reference": {"role": "float_twin"},
            "measurements": {
                "candidate": {},
                "reference": {"available": False, "reason": "no labelled examples"},
            },
        }
    )
    assert sorted(points) == ["tuned_converted"]


def test_bundle_of_a_gemma4_model_carries_the_recorded_ceiling(tmp_path, deliverable, capsys):
    model, declarations = deliverable
    main(
        [
            "bundle",
            "--output-dir",
            str(tmp_path / "bundle"),
            "--model",
            str(model),
            "--declarations",
            str(declarations),
            "--prompt-mode",
            "runtime_rendered",
            "--base-model",
            "google/gemma-4-E2B-it",
            "--base-model-revision",
            "b" * 40,
        ]
    )
    assert "NOT equivalent to Google's published" in capsys.readouterr().out


def test_verify_prompt_mode_reaches_the_manifest(tmp_path, capsys, write_split, fake_backends):
    rows = labelled_rows(16)
    fake_backends(correct_texts(rows), correct_texts(rows))
    main(
        [
            "verify",
            "--model",
            str(tmp_path / "model.litertlm"),
            "--reference",
            "org/reference",
            "--data",
            str(write_split(rows)),
            "--prompt-mode",
            "prerendered",
            "--json",
        ]
    )
    decision = json.loads(capsys.readouterr().out)["harness"]["prompt_mode_decision"]
    assert decision == {
        "prompt_mode": "prerendered",
        "source": "declared",
        "evidence": "the caller declared this mode; no inference was made",
        "marker_share": None,
        "ambiguous": False,
    }


# ---------------------------------------------------------------------------
# Boundaries: a value the run cannot honour must be refused, not rounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--limit", "--max-tokens"])
@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_a_count_below_one_is_refused_not_rounded(flag, value):
    """`--limit 0` used to evaluate one example and report the request as 0.

    The slice check ran after the append, so the manifest recorded a limit the
    run did not use. `--max-tokens 0` was worse: falsy, so it was silently
    replaced by the default.
    """
    from litetune.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["verify", "--model", "m", "--reference", "r", "--data", "d", flag, value]
        )


def test_a_rendering_fault_cannot_change_the_verdict(capsys):
    """A KeyError in the summary used to turn `passed` into exit 4.

    Exit 4 is "could not check". A formatting bug reporting it about a run that
    completed is the exact confusion this tool exists to prevent, arriving from
    inside the tool.
    """
    from litetune.cli import summarise

    # Manifests this function did not build: available with no value, and empty.
    assert summarise({"attribution": {"cost": {"available": True}}})
    assert summarise({})


# ---------------------------------------------------------------------------
# Output is not the verdict — and a guard that never fires is worse than none
# ---------------------------------------------------------------------------


def test_a_closed_pipe_does_not_change_the_verdict(monkeypatch):
    """`litetune verify --json | head`. The reader went away on purpose.

    The first version of this guard flushed nothing, so `print`'s buffer meant
    the pipe was only touched at interpreter exit, outside the try -- the fix
    never fired for a payload smaller than a pipe buffer, which is every real
    manifest, and the process exited 120: a code in neither of this tool's
    vocabularies.
    """
    import io

    from litetune import cli

    class ClosedPipe(io.StringIO):
        def write(self, *args):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    monkeypatch.setattr(cli, "_BROKEN_PIPE", False)
    monkeypatch.setattr(sys, "stdout", ClosedPipe())

    delivered = cli._report({"status": "passed"}, lambda: ["line"], as_json=True)

    assert delivered is True
    assert cli._BROKEN_PIPE is True


def test_a_write_that_fails_loses_the_deliverable(monkeypatch):
    """With `--json` the output *is* the result. A lost one is not a pass."""
    import io

    from litetune import cli

    class FullDisk(io.StringIO):
        def write(self, *args):
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", FullDisk())

    assert cli._report({"status": "passed"}, lambda: ["line"], as_json=True) is False
    # The human summary is a rendering of a verdict the exit code already
    # carries, so losing it changes nothing.
    assert cli._report({"status": "passed"}, lambda: ["line"], as_json=False) is True


def test_a_rendering_fault_still_delivers_the_payload(capsys):
    """The summary is a rendering of the verdict, not the verdict."""
    from litetune import cli

    def boom():
        raise KeyError("value")

    assert cli._report({"status": "passed"}, boom, as_json=False) is True
    assert "summary unavailable" in capsys.readouterr().out


def test_the_degraded_summary_reads_the_key_the_payload_uses(capsys):
    """`prepare`, `tune` and `convert` key on `outcome`, not `status`."""
    from litetune import cli

    def boom():
        raise KeyError("value")

    cli._report({"outcome": "passed"}, boom, as_json=False)

    assert "status: passed" in capsys.readouterr().out


def test_the_help_advertises_only_codes_the_command_can_return():
    """Documentation contradicted by the code is the failure this tool is about.

    `bundle` was grouped with the stage commands and told the reader it returns
    0, 1 or 4 — while its default, fully successful invocation returns 2, which
    `test_bundle_writes_a_manifest_and_a_report` asserts and explains.
    """
    import contextlib
    import io

    from litetune.cli import build_parser

    def help_for(command: str) -> str:
        buffer = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buffer):
            build_parser().parse_args([command, "--help"])
        return buffer.getvalue()

    # Five codes: it carries a status it did not produce.
    assert "2 inconclusive" in help_for("bundle")
    assert "2 inconclusive" in help_for("verify")
    # Three: these map an outcome and cannot reach 2 or 3.
    for stage in ("prepare", "tune", "convert"):
        assert "2 inconclusive" not in help_for(stage)
        assert "3 nothing" not in help_for(stage)


def test_main_is_reusable_in_one_process(tmp_path):
    """A sticky module flag made the second call redirect the caller's stdout.

    `_BROKEN_PIPE` was set once and never cleared, so every later `main()` in
    the same process replaced stdout — and under `contextlib.redirect_stdout`,
    which has no descriptor, the attempt raised and turned a good verdict into
    exit 4.
    """
    import contextlib
    import io

    from litetune import cli

    (tmp_path / "m.litertlm").write_text("{}", encoding="utf-8")
    (tmp_path / "d.json").write_text("[]", encoding="utf-8")

    def bundle_into(name: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()):
            return main(
                [
                    "bundle",
                    "--output-dir",
                    str(tmp_path / name),
                    "--model",
                    str(tmp_path / "m.litertlm"),
                    "--declarations",
                    str(tmp_path / "d.json"),
                    "--prompt-mode",
                    "prerendered",
                    "--base-model",
                    "g",
                    "--base-model-revision",
                    "r",
                ]
            )

    # Set it first: the flag is what a previous command's closed pipe leaves
    # behind, and a test that never sets it cannot notice that the reset is
    # gone. Removing `_BROKEN_PIPE = False` from `_run` used to leave this
    # green.
    cli._BROKEN_PIPE = True

    assert bundle_into("one") == 2
    assert cli._BROKEN_PIPE is False, "a stale flag silences the next command's output"
    assert bundle_into("two") == 2

    caller = io.StringIO()
    with contextlib.redirect_stdout(caller):
        print("the caller's own output")
    assert "the caller's own output" in caller.getvalue()


def test_every_status_a_bundle_can_carry_has_an_exit_code():
    """`_bundle` maps `RunStatus` through `Status`, and only one direction is
    guarded.

    `test_manifest` asserts `Status` is a subset of `RunStatus`; `manifest.py`
    invites growth in the other direction. One new `RunStatus` member and a
    completed bundle raises into the catch-all as exit 4.
    """
    from litetune.manifest import RunStatus
    from litetune.verify import EXIT_CODES, Status

    for status in RunStatus:
        assert EXIT_CODES[Status(status.value)] in {0, 1, 2, 3, 4}


def test_every_outcome_a_stage_can_report_has_an_exit_code():
    from litetune.checks import Outcome
    from litetune.cli import OUTCOME_EXIT_CODES

    for outcome in Outcome:
        assert OUTCOME_EXIT_CODES[outcome] in {0, 1, 4}


def _run_with_a_dead_pipe(argv: list[str], *, break_stderr: bool = False) -> tuple[int, bytes]:
    """Run `argv` with one output stream whose reader is already gone.

    The read end is closed *before* the child starts, so the first write fails
    deterministically. Spawning `head` and waiting for it does not: a payload
    smaller than the pipe buffer is accepted whole, `head` exits afterwards, and
    nothing ever breaks — which is how the first version of this test passed
    against code that had the bug, testing nothing.
    """
    import os
    import subprocess

    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        if break_stderr:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=write_end)
        else:
            process = subprocess.Popen(argv, stdout=write_end, stderr=subprocess.PIPE)
    finally:
        os.close(write_end)
    live = process.stdout if break_stderr else process.stderr
    captured = b""
    if live is not None:
        captured = live.read() or b""
        live.close()
    process.wait(timeout=60)
    return process.returncode, captured


def test_help_to_a_closed_pipe_is_not_a_failure():
    """`litetune verify --help | head` exited 120.

    argparse prints and then raises `SystemExit`, so the write is still in the
    buffer when the guard around `parse_args` would see it; the pipe only breaks
    during the interpreter's own exit flush, where `Py_FinalizeEx` sets status
    120 — a code in neither of this tool's vocabularies, for a request that was
    answered. Flushing inside the guard is what makes it observable.
    """
    import sys as _sys

    for command in (["--help"], ["verify", "--help"], ["bundle", "--help"]):
        returncode, stderr = _run_with_a_dead_pipe(
            [_sys.executable, "-m", "litetune.cli", *command]
        )

        assert returncode == 0, f"{command} exited {returncode}"
        assert b"BrokenPipeError" not in stderr


def test_a_closed_stderr_does_not_turn_a_failure_into_a_pass(tmp_path):
    """The single thing this tool exists to prevent, arriving from inside it.

    The event stream writes progress to stderr. With `2>&1 | head` the reader
    closes it mid-run, and the handler that caught the resulting
    `BrokenPipeError` returned 0 — so `bundle` on a missing model went from
    exit 4 to exit 0. A closed progress channel means the command was
    interrupted, not that it succeeded.
    """
    import subprocess
    import sys as _sys

    (tmp_path / "declarations.json").write_text("[]", encoding="utf-8")

    def bundle_into(name: str) -> list[str]:
        return [
            _sys.executable,
            "-m",
            "litetune.cli",
            "bundle",
            "--output-dir",
            str(tmp_path / name),
            "--model",
            str(tmp_path / "absent.litertlm"),
            "--declarations",
            str(tmp_path / "declarations.json"),
            "--prompt-mode",
            "prerendered",
            "--base-model",
            "g",
            "--base-model-revision",
            "r",
        ]

    intact = subprocess.run(bundle_into("a"), capture_output=True, timeout=120)

    broken, _ = _run_with_a_dead_pipe(bundle_into("b"), break_stderr=True)

    assert intact.returncode == 4
    assert broken == intact.returncode, f"exited {broken}"


def test_a_usage_error_to_a_closed_stderr_keeps_its_code():
    """argparse writes usage errors to stderr; flushing only stdout missed them.

    Left to the interpreter's exit flush, the process exits 120 instead of
    argparse's 2.
    """
    import sys as _sys

    returncode, _ = _run_with_a_dead_pipe(
        [_sys.executable, "-m", "litetune.cli", "not-a-command"], break_stderr=True
    )

    # Not `in (0, 2)`: the first version of this assertion admitted the answer
    # its own name rules out. 4, not argparse's 2, because a usage error is a
    # request litetune will not run -- and 2 is `inconclusive`, a statement
    # about a measurement.
    assert returncode == 4, f"exited {returncode}"


def test_a_refusal_to_a_closed_stderr_keeps_its_code():
    """`litetune convert --model m --output-dir o 2>&1 | head`.

    Forgetting `--recipe` is the refusal `convert`'s own help describes, and its
    message is printed by `main` — *after* the guard that was supposed to cover
    stderr, which sits inside the `try` and cannot see the two handlers that
    follow it. So the flagship refusal still exited 120, the code the previous
    round set out to eliminate.
    """
    import subprocess
    import sys as _sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        argv = [
            _sys.executable,
            "-m",
            "litetune.cli",
            "convert",
            "--model",
            "org/nothing",
            "--output-dir",
            f"{tmp}/out",
        ]
        intact = subprocess.run(argv, capture_output=True, timeout=120)
        broken, _ = _run_with_a_dead_pipe(argv, break_stderr=True)

    assert intact.returncode == 4
    assert broken == intact.returncode, f"exited {broken}"


def test_a_usage_error_does_not_borrow_a_verdict_code():
    """argparse exits 2; here 2 means `inconclusive`.

    So a typo used to produce "the interval does not resolve the threshold" —
    a claim about a measurement that never ran. A malformed command line is a
    request litetune will not run, which is what 4 means everywhere else.
    """
    import subprocess
    import sys as _sys

    for argv in (["not-a-command"], ["verify"], ["prepare", "--bogus"]):
        finished = subprocess.run(
            [_sys.executable, "-m", "litetune.cli", *argv], capture_output=True, timeout=60
        )
        assert finished.returncode == 4, f"{argv} exited {finished.returncode}"

    helped = subprocess.run(
        [_sys.executable, "-m", "litetune.cli", "--help"], capture_output=True, timeout=60
    )
    assert helped.returncode == 0


def _stop_token_argv(tmp_path, *extra):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"artifact")
    declarations = tmp_path / "tools.json"
    declarations.write_text("[]", encoding="utf-8")
    return [
        "bundle",
        "--output-dir",
        str(tmp_path / "bundle"),
        "--model",
        str(model),
        "--declarations",
        str(declarations),
        "--prompt-mode",
        "prerendered",
        "--base-model",
        "google/functiongemma-270m-it",
        "--base-model-revision",
        "1234567890abcdef1234567890abcdef12345678",
        *extra,
    ]


def _stop_token_contract(tmp_path):
    return json.loads((tmp_path / "bundle" / "contract.json").read_text(encoding="utf-8"))


def test_the_family_stop_token_supplements_the_recorded_one(tmp_path):
    """Both, in that order: what the model emits, then where the app takes over."""
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps({"turn_terminator": {"text": "<end_of_turn>", "source": "chat template"}}),
        encoding="utf-8",
    )
    main(_stop_token_argv(tmp_path, "--train-metrics", str(metrics)))

    contract = _stop_token_contract(tmp_path)
    assert contract["stop_tokens"] == ["<end_of_turn>", "<start_function_response>"]
    notes = " ".join(contract["notes"])
    assert "litetune added the stop token(s) <start_function_response>" in notes
    assert "recorded terminator" in notes, "the trained one must stay attributed to the run"


def test_a_family_stop_token_alone_says_the_trained_one_is_unrecorded(tmp_path):
    """Supplementing is not substituting.

    With no run and no `--stop-token`, the family's terminator would be the only
    one declared -- a bundle that tells the runtime where the application takes
    over but not where the turn ends. That is worse than declaring nothing,
    because it looks specified, so it is said out loud.
    """
    main(_stop_token_argv(tmp_path))

    contract = _stop_token_contract(tmp_path)
    assert contract["stop_tokens"] == ["<start_function_response>"]
    notes = " ".join(contract["notes"])
    assert "the terminator the model was actually trained to emit is unrecorded" in notes


def test_an_unrecorded_wire_convention_is_a_named_limitation(tmp_path):
    """A consumer that has to guess is told it has to guess."""
    main(_stop_token_argv(tmp_path))
    report = json.loads((tmp_path / "bundle" / "report.json").read_text(encoding="utf-8"))
    limitation = [line for line in report["limitations"] if "--wire-convention" in line]
    assert limitation, "an unrecorded convention must be reported, not silently omitted"
    assert "0.019-0.036" in limitation[0], "the cost of guessing wrong belongs in the text"


def test_a_recorded_wire_convention_raises_no_limitation(tmp_path):
    main(_stop_token_argv(tmp_path, "--wire-convention", "template_dictsort"))
    contract = _stop_token_contract(tmp_path)
    report = json.loads((tmp_path / "bundle" / "report.json").read_text(encoding="utf-8"))
    assert contract["wire_convention"] == "template_dictsort"
    assert not [line for line in report["limitations"] if "--wire-convention" in line]


def test_version_is_a_flag_and_not_an_abbreviation_of_verbose(capsys):
    """`litetune --version` is the first thing a new user types.

    It used to match `--verbose` through argparse's prefix abbreviation: the
    flag silently switched on debug logging and the user was then told they had
    not named a command. Declaring `--version` answers the question and removes
    the collision.
    """
    from litetune._version import __version__

    # `main` converts argparse's SystemExit into a return code -- the same path
    # that keeps `litetune --help | head` from exiting 120 on a closed pipe.
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"litetune {__version__}"


def test_verbose_still_needs_a_command(capsys):
    """`--verbose` modifies a run; it does not constitute one."""
    assert main(["--verbose"]) == 4
    assert "required: command" in capsys.readouterr().err
