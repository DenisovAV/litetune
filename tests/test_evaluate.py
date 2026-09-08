"""The evaluator and its two backends.

Nothing here starts a process or loads a model: `StageEnv.run` is monkeypatched
at the boundary, which is the only place either backend touches the outside
world.

The backend tests are mostly about one distinction -- a process that ran and
failed against a process that never ran. Collapsing the two is what produced
eight confident negatives during the measurement work, so each has its own test.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeBackend, call_text, fake_torch, labelled_rows

from litetune import envs
from litetune.evaluate import (
    UNKNOWN_BACKEND,
    DataError,
    DecodeConfig,
    HuggingFaceBackend,
    LiteRtLmBackend,
    PromptMode,
    device_mismatch,
    evaluate,
    harness_mismatch,
    load_split,
    strip_runtime_noise,
)
from litetune.metrics import score_exact_text, trim_terminator

# -- the split --------------------------------------------------------------


def test_load_split_reads_prompts_and_targets(write_split):
    path = write_split(labelled_rows(3))
    split = load_split(path)
    assert split.n == 3
    assert len(split.labelled) == 3
    assert split.examples[0].target is not None
    assert split.examples[0].target.name == "change_background_color"


def test_examples_without_a_target_are_kept_but_unlabelled(write_split):
    path = write_split([{"prompt": "do a thing"}])
    split = load_split(path)
    assert split.n == 1
    assert split.labelled == ()


def test_limit_takes_the_first_n(write_split):
    path = write_split(labelled_rows(10))
    assert load_split(path, limit=4).n == 4


def test_split_identity_follows_content_not_location(write_split):
    a = load_split(write_split(labelled_rows(3), name="a.jsonl"))
    b = load_split(write_split(labelled_rows(3), name="b.jsonl"))
    assert a.id == b.id
    assert a.source != b.source


def test_a_limited_split_is_a_different_sample(write_split):
    # A 4-example slice is not evidence about the 10 it came from, so it must
    # not share their identity.
    path = write_split(labelled_rows(10))
    assert load_split(path).id != load_split(path, limit=4).id


def test_malformed_line_names_itself(write_split, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prompt": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(DataError) as exc:
        load_split(path)
    assert ":2:" in str(exc.value)


def test_row_without_a_prompt_is_refused(write_split):
    with pytest.raises(DataError):
        load_split(write_split([{"target": {"name": "x", "args": {}}}]))


# -- output cleaning --------------------------------------------------------


def test_runtime_log_lines_are_not_model_output():
    stdout = (
        "I0830 12:00:00.123456 12 engine.cc:42] loading\n"
        "call:open{app:<escape>maps<escape>}\n"
        "Prefill speed: 120 tok/s\n"
    )
    assert strip_runtime_noise(stdout) == "call:open{app:<escape>maps<escape>}"


# -- litert-lm --------------------------------------------------------------


def _litertlm(tmp_path: Path, **kwargs) -> LiteRtLmBackend:
    return LiteRtLmBackend(model=tmp_path / "model.litertlm", auto_provision=False, **kwargs)


def test_argv_pins_the_prompt_construction_mode(tmp_path):
    backend = _litertlm(tmp_path)
    argv = backend.argv("hello")
    assert argv[:2] == ["litert-lm", "run"]
    assert "--backend=cpu" in argv
    assert "--no-template" in argv
    assert "--prompt=hello" in argv
    # --no-template forces the runtime's tool list to null, so the prompt must
    # arrive already rendered. The mode has to say so.
    assert backend.prompt_mode is PromptMode.PRERENDERED


def test_one_process_per_prompt(monkeypatch, tmp_path):
    seen = []

    def fake_run(self, args, timeout=3600, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="call:a{}", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gens = _litertlm(tmp_path).generate(["one", "two"])
    assert len(seen) == 2
    assert [g.text for g in gens] == ["call:a{}", "call:a{}"]
    assert all(g.ok for g in gens)


def test_a_missing_binary_is_not_a_model_failure(monkeypatch, tmp_path):
    def fake_run(self, args, timeout=3600, **kwargs):
        raise FileNotFoundError("litert-lm")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is not None
    assert not gen.ran
    assert gen.returncode is None


def test_a_timeout_is_recorded_as_unperformed(monkeypatch, tmp_path):
    def fake_run(self, args, timeout=3600, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is not None
    assert "timeout" in gen.harness_error


def test_a_missing_system_library_is_unperformed_not_failed(monkeypatch, tmp_path):
    # litert-lm links vulkan even for the CPU backend; without it every
    # invocation dies in under a second and looks exactly like a dead model.
    def fake_run(self, args, timeout=3600, **kwargs):
        return subprocess.CompletedProcess(
            args, 127, stdout="", stderr="error while loading shared libraries: libvulkan.so.1"
        )

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is not None
    assert not gen.ran


def test_an_ordinary_non_zero_exit_is_a_real_observation(monkeypatch, tmp_path):
    def fake_run(self, args, timeout=3600, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="decode failed at token 4")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gen = _litertlm(tmp_path).generate(["one"])[0]
    assert gen.harness_error is None
    assert gen.ran and not gen.ok


def test_backend_reports_which_engine_produced_the_numbers(tmp_path):
    described = _litertlm(tmp_path).describe()
    assert described["engine"] == "litert-lm"
    assert described["backend"] == "cpu"
    assert "litert-lm==0.16.1" in described["requirements"]


# -- transformers -----------------------------------------------------------


def test_hugging_face_backend_runs_the_whole_split_in_one_process(monkeypatch):
    calls = []

    def fake_run(self, args, timeout=3600, **kwargs):
        calls.append(args)
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            "\n".join(
                json.dumps({"index": i, "text": call_text("a", i=str(i))})
                for i in range(len(spec["prompts"]))
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gens = HuggingFaceBackend(model="org/model", auto_provision=False).generate(["a", "b", "c"])
    assert len(calls) == 1
    assert [g.ok for g in gens] == [True, True, True]
    assert gens[2].text == call_text("a", i="2")


def test_a_generation_script_that_dies_reports_unperformed(monkeypatch):
    def fake_run(self, args, timeout=3600, **kwargs):
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="ModuleNotFoundError: torch")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    gens = HuggingFaceBackend(model="org/model", auto_provision=False).generate(["a"])
    assert gens[0].harness_error is not None
    assert not gens[0].ran


def test_a_reference_generation_reaches_scoring_with_its_terminator_intact(tmp_path):
    """The HF script keeps the terminator on purpose; nothing between the
    results file and the scorer may strip it, or liveness goes blind to
    leakage. A transport mutation stripping `<eos>` in `_read_results`
    survived the whole suite before this test existed.
    """
    results = tmp_path / "r.jsonl"
    results.write_text('{"index": 0, "text": "label_3<end_of_turn>\\n<eos>"}\n', encoding="utf-8")
    texts = HuggingFaceBackend(model="org/m", auto_provision=False)._read_results(results)
    assert texts[0] == "label_3<end_of_turn>\n<eos>"
    assert trim_terminator(texts[0]) == "label_3"
    assert score_exact_text(["label_3"], [texts[0]]).exact_match.value == 1.0


def test_turning_on_the_chat_template_changes_the_measured_mode():
    plain = HuggingFaceBackend(model="m", auto_provision=False)
    templated = HuggingFaceBackend(model="m", auto_provision=False, runtime_rendered=True)
    assert plain.prompt_mode is PromptMode.PRERENDERED
    assert templated.prompt_mode is PromptMode.RUNTIME_RENDERED


def test_hugging_face_backend_reports_an_unknown_device_before_it_has_run():
    # Not "cpu": a manifest read before `generate()` ran must not claim a
    # device nothing measured yet. "unknown" and not a second word for it --
    # `verify.py` already prints the `unknown` fallback for the
    # same state when the key is missing entirely.
    described = HuggingFaceBackend(model="org/m", auto_provision=False).describe()
    assert described["backend"] == UNKNOWN_BACKEND == "unknown"


def test_the_two_backends_say_which_vocabulary_their_backend_field_is_in(tmp_path):
    """One key, two answers to two different questions.

    `backend` is a torch device on the reference side and the flag passed to
    `litert-lm` on the candidate side. They overlap at exactly one string,
    "cpu", where they mean different things -- so each `describe()` says which
    question it answered.
    """
    reference = HuggingFaceBackend(model="org/m", auto_provision=False).describe()
    candidate = _litertlm(tmp_path).describe()
    assert reference["backend_vocabulary"] == "torch device"
    assert candidate["backend_vocabulary"] == "litert-lm --backend flag"
    assert reference["backend_vocabulary"] != candidate["backend_vocabulary"]


def test_hugging_face_backend_reports_the_device_it_actually_used(monkeypatch):
    """Used to hardcode "cpu" unconditionally, which made a laptop's manifest
    and a GPU box's byte-identical in the one field meant to tell them apart.
    """

    def fake_provision(self, events=None, force=False):
        # The marker file, not just the directory: `_ensure_env` gates the
        # probe on `env.ready`, which is exactly this file's existence.
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".litetune-ready").write_text(self.identity)
        return self.path

    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(
        envs,
        "resolve_device",
        lambda env, timeout=30, events=None: envs.DeviceProbe(device="cuda", detail="fake"),
    )

    def fake_run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        Path(spec["run_report"]).write_text(json.dumps({"device": "cuda"}), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)

    backend = HuggingFaceBackend(model="org/model")
    backend.generate(["a"])

    assert backend.describe()["backend"] == "cuda"


def _ready_env() -> None:
    """Make `envs.TRAIN` look provisioned without provisioning anything.

    `_ensure_env` gates the device probe on `env.ready`, which is the marker
    file and nothing else.
    """
    envs.TRAIN.path.mkdir(parents=True, exist_ok=True)
    (envs.TRAIN.path / ".litetune-ready").write_text(envs.TRAIN.identity)


def _generating_env(monkeypatch, *, probe: str | None, script_device: str | None) -> list:
    """A stage environment that answers the probe and then generates.

    Returns the argv of every call, so a test can assert that the probe
    happened -- or did not.
    """
    calls: list = []

    def fake_run(self, args, timeout=3600, **kwargs):
        calls.append(list(args))
        if args[1] == "-c":
            if probe is None:
                return subprocess.CompletedProcess(args, 1, "", "no torch")
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps({"device": probe, "cuda_build": None, "device_count": 0}),
                "",
            )
        spec = json.loads(Path(args[2]).read_text())
        Path(spec["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        if script_device is not None:
            Path(spec["run_report"]).write_text(
                json.dumps({"device": script_device}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    return calls


def test_the_probe_runs_over_a_ready_environment_nobody_asked_to_provision(monkeypatch):
    """The probe provisions nothing, so readiness is its real precondition.

    Gated on `auto_provision` instead, a caller that manages the environment's
    lifecycle itself -- a library caller, and every backend in this file --
    reported `describe()["backend"] == "unknown"` for a run whose device was
    perfectly knowable. No CLI path reaches that state: `verify` has no
    `--no-provision` flag.
    """
    _ready_env()
    calls = _generating_env(monkeypatch, probe="cuda", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert [c[1] for c in calls][0] == "-c"
    assert backend.describe()["backend"] == "cuda"


def test_no_probe_against_an_environment_that_is_not_there(monkeypatch):
    """An environment with no marker is one nothing can be asked of."""
    calls = _generating_env(monkeypatch, probe="cuda", script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert all(c[1] != "-c" for c in calls)
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_the_script_reports_the_device_it_used_and_that_answer_wins(monkeypatch):
    """The probe is a prediction; the run report is the observation.

    They differ whenever the script took its own fallback. `tune.py` has had
    the observation all along through `metrics.device`; the reference side
    predicted "cpu" and published it as though it had watched.
    """
    _ready_env()
    _generating_env(monkeypatch, probe="cpu", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert backend.device == "cuda"
    assert backend.describe()["backend"] == "cuda"
    # Both, not only the winner. A manifest that records "cuda" and nothing
    # else cannot show that the prediction was wrong, and the logger line that
    # says so does not travel with the measurement.
    assert backend.probed_device == "cpu"
    assert backend.describe()["backend_probed"] == "cpu"


def test_a_run_that_used_what_it_was_told_records_the_two_as_equal(monkeypatch):
    _ready_env()
    _generating_env(monkeypatch, probe="cuda", script_device="cuda")

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    described = backend.describe()
    assert described["backend"] == described["backend_probed"] == "cuda"


def test_a_script_that_wrote_no_report_leaves_the_prediction_standing(monkeypatch):
    """A run that died before writing it, or an older script that never did.
    The prediction is still the best thing known, and `None` would be worse."""
    _ready_env()
    _generating_env(monkeypatch, probe="cuda", script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    backend.generate(["a"])

    assert backend.device == "cuda"


def test_a_run_report_that_is_not_valid_utf8_leaves_the_prediction_standing(tmp_path):
    """Same family as the blocker: `_read_results` a few lines below already
    catches `UnicodeDecodeError` on a damaged results file; `_read_run_report`
    used to let it escape, which would have aborted verification after the
    generations that report was only ever supposed to annotate had already
    been produced."""
    report = tmp_path / "run.json"
    report.write_bytes(b"\xff\xfe{not json")
    device = HuggingFaceBackend(model="org/m", auto_provision=False)._read_run_report(report)
    assert device is None


def test_a_failed_probe_does_not_erase_a_device_already_known(monkeypatch):
    """Reuse. The previous run's answer is a better record than `None`, and
    `None` would claim the device was never established when it was."""
    _ready_env()
    _generating_env(monkeypatch, probe=None, script_device=None)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False, device="cuda")
    backend.generate(["a"])

    assert backend.device == "cuda"


def test_a_timeout_clears_the_device_it_never_confirmed(monkeypatch):
    """The probe answered, but the generation subprocess it precedes never
    finished: nothing ran, so `describe()` must not report the probe's
    prediction as though the run had confirmed it."""
    _ready_env()

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"device": "cuda", "cuda_build": None, "device_count": 1}), ""
            )
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    gens = backend.generate(["a"])

    assert gens[0].harness_error is not None
    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_a_script_that_could_not_start_clears_the_device_it_never_confirmed(monkeypatch):
    _ready_env()

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"device": "cuda", "cuda_build": None, "device_count": 1}), ""
            )
        raise FileNotFoundError("python")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(model="org/model", auto_provision=False)
    gens = backend.generate(["a"])

    assert gens[0].harness_error is not None
    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND


def test_a_failed_probe_does_not_force_a_stale_device_onto_the_child(monkeypatch):
    """The do-not-erase rule above protects the *report*; it must not also
    hand the *child* a directive its own probe could not vouch for. Before
    this was fixed, `spec["device"]` read `self.device` -- the previous run's
    "cuda" -- even though this call's probe could not answer, which would make
    `model.to("cuda")` fail on a box where CUDA had since gone away."""
    _ready_env()
    specs: list[dict] = []

    def fake_run(self, args, timeout=3600, **kwargs):
        if args[1] == "-c":
            # No torch: the probe cannot answer this call.
            return subprocess.CompletedProcess(args, 1, "", "no torch")
        specs.append(json.loads(Path(args[2]).read_text()))
        Path(json.loads(Path(args[2]).read_text())["out"]).write_text(
            json.dumps({"index": 0, "text": call_text("a")}) + "\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)

    backend = HuggingFaceBackend(model="org/model", auto_provision=False, device="cuda")
    backend.generate(["a"])

    assert backend.device == "cuda", "the report still remembers the previous answer"
    assert len(specs) == 1
    assert (
        specs[0]["device"] is None
    ), "a probe that could not answer must not be forced onto the child"


def test_a_blocked_environment_clears_the_device(monkeypatch):
    """Nothing ran, so nothing has a device. The opposite of the case above:
    there is no result here for a stale answer to describe."""

    def explode(self, events=None, force=False):
        raise RuntimeError("no interpreter")

    monkeypatch.setattr(envs.StageEnv, "provision", explode)

    backend = HuggingFaceBackend(model="org/model", device="cuda")
    gens = backend.generate(["a"])

    assert backend.device is None
    assert backend.describe()["backend"] == UNKNOWN_BACKEND
    assert gens[0].harness_error is not None


# -- comparing two points that ran on different hardware ---------------------


def _point(label: str, backend: str, engine: str = "transformers"):
    from litetune.evaluate import GREEDY, MeasurementPoint

    return MeasurementPoint(
        label=label,
        model_ref="org/m",
        backend=engine,
        prompt_mode=PromptMode.PRERENDERED,
        decode=GREEDY,
        split_id="s",
        engine={"engine": engine, "backend": backend},
        decode_enforced=True,
    )


def test_two_points_on_different_hardware_are_annotated_not_refused():
    """`build_backends` pins the candidate to litert-lm's CPU backend and lets
    the reference resolve its own device, so on a GPU box the two differ and
    the conversion cost carries a hardware difference. Refusing would leave
    such a machine unable to verify at all; the number is kept and told what
    is in it.
    """
    note = device_mismatch(
        _point("candidate", "cpu", engine="litert-lm"), _point("reference", "cuda")
    )
    assert note is not None
    assert "cpu" in note and "cuda" in note
    assert "candidate" in note and "reference" in note
    # Still comparable: this is a limitation, not a refusal.
    assert (
        harness_mismatch(
            _point("candidate", "cpu", engine="litert-lm"), _point("reference", "cuda")
        )
        is None
    )


def test_two_points_on_the_same_device_are_not_annotated():
    assert (
        device_mismatch(_point("candidate", "cpu", engine="litert-lm"), _point("reference", "cpu"))
        is None
    )


def test_an_unestablished_device_is_not_a_difference():
    """Nothing was established to differ from, and `describe()` already says
    so in the same field."""
    assert (
        device_mismatch(
            _point("candidate", "cpu", engine="litert-lm"),
            _point("reference", UNKNOWN_BACKEND),
        )
        is None
    )


# -- the evaluator ----------------------------------------------------------


def test_measurement_point_records_what_produced_it(write_split):
    split = load_split(write_split(labelled_rows(2)))
    point = evaluate(FakeBackend(texts=["call:a{}"]), split, label="candidate")
    assert point.label == "candidate"
    assert point.split_id == split.id
    assert point.prompt_mode is PromptMode.PRERENDERED
    assert point.n == 2


def test_a_backend_that_drops_a_prompt_is_a_contract_violation(write_split):
    class ShortBackend(FakeBackend):
        def generate(self, prompts, events=None):
            return super().generate(prompts[:-1])

    split = load_split(write_split(labelled_rows(3)))
    with pytest.raises(ValueError):
        evaluate(ShortBackend(texts=["call:a{}"]), split, label="candidate")


def test_comparison_across_prompt_modes_is_refused(write_split):
    # --no-template forces the tool list to null, so the two modes differ by the
    # whole declaration block: their difference measures the mode, not the model.
    split = load_split(write_split(labelled_rows(2)))
    a = evaluate(FakeBackend(texts=["call:a{}"]), split, label="a")
    b = evaluate(
        FakeBackend(texts=["call:a{}"], prompt_mode=PromptMode.RUNTIME_RENDERED), split, label="b"
    )
    reason = harness_mismatch(a, b)
    assert reason is not None
    assert "mode" in reason


def test_comparison_across_splits_is_refused(write_split):
    a = evaluate(FakeBackend(texts=["call:a{}"]), load_split(write_split(labelled_rows(2))), "a")
    b = evaluate(
        FakeBackend(texts=["call:a{}"]),
        load_split(write_split(labelled_rows(2)[:1] + [{"prompt": "other"}], name="o.jsonl")),
        "b",
    )
    assert harness_mismatch(a, b) is not None


def test_comparison_across_decoding_is_refused(write_split):
    split = load_split(write_split(labelled_rows(2)))
    a = evaluate(FakeBackend(texts=["call:a{}"]), split, "a")
    b = evaluate(FakeBackend(texts=["call:a{}"], decode=DecodeConfig(max_tokens=8)), split, "b")
    assert harness_mismatch(a, b) is not None


def test_equivalent_measurements_are_comparable(write_split):
    split = load_split(write_split(labelled_rows(2)))
    a = evaluate(FakeBackend(texts=["call:a{}"]), split, "a")
    b = evaluate(FakeBackend(model="other", texts=["call:b{}"]), split, "b")
    assert harness_mismatch(a, b) is None


def test_attention_implementation_is_threaded_into_the_generate_script():
    """A float reference on sdpa against a model trained on eager is a harness
    difference wearing a conversion cost's clothes. The reference notebooks warn
    about this mismatch three separate times."""
    from litetune.evaluate import _HF_GENERATE_SCRIPT, HuggingFaceBackend

    assert 'attn_implementation=spec["attn_implementation"]' in _HF_GENERATE_SCRIPT
    assert HuggingFaceBackend(model="m").attn_implementation == "eager"


def test_a_script_that_failed_after_writing_results_is_not_recorded_as_clean():
    """Results plus a non-zero exit used to produce `returncode=0, stderr=""`.

    The generation did happen, so it keeps its text and stays scoreable — but
    erasing the process failure meant a reference side could be reported healthy
    while its environment had died on the way out.
    """
    import subprocess

    from litetune.evaluate import HuggingFaceBackend

    backend = HuggingFaceBackend(model="org/m", declared_prompt_mode=PromptMode.PRERENDERED)
    proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="teardown blew up")

    out = backend._assemble(["p0", "p1"], {0: "a", 1: "b"}, proc)

    assert [g.text for g in out] == ["a", "b"]
    # Typed, not buried in a message: `returncode` stays 0 so the generation is
    # still scoreable, and the batch's exit is its own field so a consumer can
    # see it without parsing prose.
    assert all(g.batch_returncode == 1 for g in out)
    assert all(g.ok for g in out)


def test_a_failed_batch_is_counted_where_something_reads_it():
    """A typed field nothing reads is the erasure it replaced, with a type.

    `batch_returncode` carried the evidence and no production caller looked at
    it: liveness sees `.ok`, and the manifest did not serialise it. The count
    is what `verify` turns into a limitation.
    """
    import subprocess

    from litetune.evaluate import GREEDY, HuggingFaceBackend, MeasurementPoint

    backend = HuggingFaceBackend(model="org/m", declared_prompt_mode=PromptMode.PRERENDERED)
    proc = subprocess.CompletedProcess(args=[], returncode=3, stdout="", stderr="died")
    generations = backend._assemble(["p0", "p1"], {0: "a", 1: "b"}, proc)

    point = MeasurementPoint(
        label="candidate",
        model_ref="m",
        backend="fake",
        prompt_mode=PromptMode.PRERENDERED,
        decode=GREEDY,
        split_id="s",
        engine={},
        generations=tuple(generations),
        decode_enforced=True,
    )

    assert point.batch_failures == 2
    assert point.as_dict()["generations"]["from_a_failed_batch"] == 2


def test_the_reference_decoder_keeps_the_terminator_scoring_trims():
    """The premise the whole normalisation rests on, pinned in the module that
    makes it. `skip_special_tokens=False` is deliberate -- the liveness tier
    needs to see leakage -- and it is why every reference generation carries a
    marker scoring has to remove. Flip it and the leakage check goes blind
    while the trimming becomes dead code, with nothing else noticing: the whole
    suite stayed green through exactly that mutation.
    """
    from litetune.evaluate import _HF_GENERATE_SCRIPT

    # The call, not the word. An earlier version of this test searched for the
    # bare string and passed against a flipped call, because the comment above
    # it explaining the choice contains the same text.
    assert "skip_special_tokens=False)" in _HF_GENERATE_SCRIPT
    assert "skip_special_tokens=True" not in _HF_GENERATE_SCRIPT


class _GenerationStopped(RuntimeError):
    """Stands in for a reference run that dies once the model is placed.

    The OOM killer is the realistic cause -- this project has been killed at
    32 GiB more than once -- but any death after placement and before the last
    completion has the same shape, and the script's run report exists to
    survive it.
    """


def _run_hf_generate_script(
    tmp_path: Path,
    monkeypatch,
    *,
    dtype: str = "bfloat16",
    cuda: bool = False,
    given=None,
    stop_generation: bool = False,
) -> dict:
    """Runs `_HF_GENERATE_SCRIPT`'s real `main()` against faked `torch` and
    `transformers`, and returns what the fakes captured.

    `torch` and `transformers` are faked at the import boundary -- the same
    pattern `test_tune.py`'s `script_namespace` fixture uses to test the
    training script's body without a real torch. `cuda` and `given` are the
    two things that decide where the script places the model: `cuda` is what
    the fake `torch.cuda.is_available()` answers, `given` is `spec["device"]`,
    what the parent already resolved. `FakeInputIds` and `FakeEncoding` track
    their own device and hand back a new object when `.to()` actually moves
    them, the way a real tensor does -- a double that returns `self`
    unconditionally cannot tell a dropped `.to()` call from one that ran.
    """
    import sys
    import types

    from litetune.evaluate import _HF_GENERATE_SCRIPT

    captured: dict = {}

    class FakeModel:
        def eval(self) -> "FakeModel":
            return self

        def to(self, device: str) -> "FakeModel":
            # The script places the model before generating. Recorded rather
            # than ignored so a reader can see this double is standing in for
            # a real move, not silently swallowing one.
            captured["model_device"] = device
            return self

        def generate(self, **kwargs: object) -> list[list[int]]:
            captured["generate_input_ids_device"] = getattr(kwargs["input_ids"], "device", None)
            if stop_generation:
                raise _GenerationStopped("the run stopped after the model was placed")
            return [[0, 1, 2, 3]]

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model: str, torch_dtype: object, attn_implementation: str) -> FakeModel:
            captured["torch_dtype"] = torch_dtype
            return FakeModel()

    class FakeInputIds(list):
        device: str = "cpu"

        @property
        def shape(self) -> tuple[int, int]:
            return (1, len(self[0]))

        def to(self, device: str) -> "FakeInputIds":
            if device == self.device:
                return self
            moved = FakeInputIds(self)
            moved.device = device
            return moved

    class FakeEncoding(dict):
        """What the tokenizer hands back, which the script places on a device.

        A plain dict has no `.to`, and the script calls it -- so a double that
        is a bare dict makes this test fail for a reason that has nothing to do
        with the dtype or device it exists to pin.
        """

        device: str = "cpu"

        def to(self, device: str) -> "FakeEncoding":
            if device == self.device:
                return self
            moved = FakeEncoding(
                {k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()}
            )
            moved.device = device
            return moved

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text: str, return_tensors: str = "pt") -> "FakeEncoding":
            return FakeEncoding({"input_ids": FakeInputIds([[10, 11, 12]])})

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            return "generated"

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str) -> FakeTokenizer:
            return FakeTokenizer()

    # `Any`: a real module has none of these attributes declared, and this
    # double exists to be assigned onto dynamically -- same as the untyped
    # test bodies elsewhere in this suite that `check_untyped_defs = false`
    # leaves unchecked, made explicit here because this function is typed.
    fake_torch_module: Any = types.ModuleType("torch")
    fake_torch_module.float32 = "sentinel:float32"
    fake_torch_module.bfloat16 = "sentinel:bfloat16"
    fake_torch_module.no_grad = __import__("contextlib").nullcontext
    # `fake_torch` from `conftest.py`: the one shape a fake `torch.cuda` takes
    # in this suite, reused rather than a fourth hand-rolled `is_available`.
    fake_torch_module.cuda = fake_torch(cuda=cuda).cuda

    fake_transformers: Any = types.ModuleType("transformers")
    fake_transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    fake_transformers.AutoTokenizer = FakeAutoTokenizer

    monkeypatch.setitem(sys.modules, "torch", fake_torch_module)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "out.jsonl"
    report_path = tmp_path / "run.json"
    spec_path.write_text(
        json.dumps(
            {
                "model": "org/m",
                "prompts": ["hi"],
                "max_tokens": 8,
                "runtime_rendered": False,
                "attn_implementation": "eager",
                "dtype": dtype,
                "device": given,
                "out": str(out_path),
                "run_report": str(report_path),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["generate.py", str(spec_path)])

    # `exec` on this module's own `_HF_GENERATE_SCRIPT` constant, not on
    # external input.
    namespace: dict = {"__name__": "litetune_generate_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "generate_script.py", "exec"), namespace)
    if stop_generation:
        with pytest.raises(_GenerationStopped):
            namespace["main"]()
    else:
        namespace["main"]()
    # The run report is the script's structured answer about its own device,
    # and it is what the backend reads back -- so the fixture returns it
    # beside what the torch doubles captured. Whether it exists at all is
    # reported separately, because a run that stopped early is exactly the
    # case where the answer is worth having and the file might be missing.
    captured["run_report_written"] = report_path.exists()
    if report_path.exists():
        captured["run_report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return captured


def test_the_run_report_is_written_before_any_generation_runs(tmp_path, monkeypatch):
    """Pins the *ordering*, which the comment beside the write claims and no
    other test enforces.

    Moving that write to after the generation loop -- present, otherwise
    identical -- passes every other test in this suite, because they all read
    the report from a run that finished. Deleting the write is caught; writing
    it late is not. The difference only shows on a run that stops part-way,
    and that is the run whose device is hardest to recover afterwards: the
    parent falls back to its own prediction, or to `unknown`, for a reference
    that had demonstrably reached a device and started work.

    So the model is placed, the first `generate` raises, and the report must
    already be on disk with the right answer in it.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=True, stop_generation=True)

    assert captured["model_device"] == "cuda"
    assert captured["run_report_written"], (
        "the run report must exist once the model is placed, not only once "
        "generation has finished"
    )
    assert captured["run_report"]["device"] == "cuda"


def test_the_float_reference_loads_at_float32_regardless_of_spec_dtype(tmp_path, monkeypatch):
    """S7: pins the *fact*, not the string. `tune.py`'s comments and
    limitations quote the prose "evaluate.py's float reference loads at an
    unconditional float32 regardless of the training dtype" -- three tests
    assert that prose, and all three survive changing the hardcoded
    `torch_dtype=torch.float32` to `torch_dtype=getattr(torch,
    spec.get("dtype", "float32"))`, because nothing writes a "dtype" key into
    the spec `HuggingFaceBackend` builds, so the mutation is inert against
    every fixture that goes through the backend.

    This runs the script's actual `main()` against a spec that carries a
    "dtype" key anyway (the backend does not have to write one for this to
    matter -- only the script's own indifference to it does), and asserts the
    dtype the fake model loader actually received. No GPU and no `given`
    keeps this test about the dtype: the device choice is the tests below's
    to make.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, dtype="bfloat16")
    assert captured["torch_dtype"] == "sentinel:float32"


def test_the_reference_script_generates_on_cuda_when_there_is_one():
    """Falls back to asking torch only when the parent could not answer."""
    from litetune.evaluate import _HF_GENERATE_SCRIPT

    namespace: dict = {"__name__": "litetune_hf_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf_generate.py", "exec"), namespace)

    assert namespace["generation_device"](fake_torch(cuda=True)) == "cuda"
    assert namespace["generation_device"](fake_torch(cuda=False)) == "cpu"


def test_the_reference_script_moves_everything_to_the_device_it_resolves(tmp_path, monkeypatch):
    """Kills three mutants at once, all invisible against a cuda-less fixture:
    `device = generation_device(...)` hardcoded to `"cpu"`, `model.to(device)`
    dropped or aimed at `"meta"`, and `.to(device)` dropped off the encoding
    before `model.generate(**enc)`. Each leaves the model and its inputs on
    different devices, which only shows up once the fake `torch` can say
    there is a GPU.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=True)
    assert captured["model_device"] == "cuda"
    assert captured["generate_input_ids_device"] == "cuda"


def test_the_reference_script_writes_its_device_where_the_parent_can_read_it(tmp_path, monkeypatch):
    """Structured, not only printed to stderr -- `_assemble` discards stderr
    on a clean exit, which is the path that matters."""
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=True)
    assert captured["run_report"] == {"device": "cuda"}


def test_the_reference_script_prefers_what_the_parent_already_resolved(tmp_path, monkeypatch):
    """`given` (`spec["device"]`, what `HuggingFaceBackend` asked
    `envs.resolve_device` before this script started) wins over asking torch
    -- not a second, possibly-disagreeing guess made inside the subprocess.
    """
    captured = _run_hf_generate_script(tmp_path, monkeypatch, cuda=False, given="cuda")
    assert captured["model_device"] == "cuda"
