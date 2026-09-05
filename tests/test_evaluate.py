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

import pytest
from conftest import FakeBackend, call_text, labelled_rows

from litetune import envs
from litetune.evaluate import (
    DataError,
    DecodeConfig,
    HuggingFaceBackend,
    LiteRtLmBackend,
    PromptMode,
    evaluate,
    harness_mismatch,
    load_split,
    strip_runtime_noise,
)

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


def test_turning_on_the_chat_template_changes_the_measured_mode():
    plain = HuggingFaceBackend(model="m", auto_provision=False)
    templated = HuggingFaceBackend(model="m", auto_provision=False, runtime_rendered=True)
    assert plain.prompt_mode is PromptMode.PRERENDERED
    assert templated.prompt_mode is PromptMode.RUNTIME_RENDERED


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


def test_the_reference_script_generates_on_cuda_when_there_is_one():
    """The float reference used to stay on the CPU whatever the machine had --
    the one stage of a GPU run left behind. Same rule as training."""
    from litetune.evaluate import _HF_GENERATE_SCRIPT

    namespace: dict = {"__name__": "litetune_hf_script_under_test"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf_generate.py", "exec"), namespace)

    class Cuda:
        def __init__(self, available):
            self.available = available

        def is_available(self):
            return self.available

    class Torch:
        def __init__(self, available):
            self.cuda = Cuda(available)

    assert namespace["generation_device"](Torch(True)) == "cuda"
    assert namespace["generation_device"](Torch(False)) == "cpu"
