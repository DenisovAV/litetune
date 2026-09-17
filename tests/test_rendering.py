"""Whether the runtime and the reference put the same prompt tokens in front of the model.

No runtime and no transformers in this process. The comparison is a pure
function; the two scripts are `exec`'d against fake `litert_lm` and
`transformers` modules, the way `test_tune.py` runs the training script's body;
and `verify` is driven with a fake observer.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeBackend, correct_texts, labelled_rows

from litetune import envs
from litetune.prompt_mode import PromptMode
from litetune.rendering import (
    _REFERENCE_SCRIPT,
    _RUNTIME_SCRIPT,
    RENDERING_CHECK,
    RenderingComparison,
    RenderingProbe,
    RenderingProbeError,
    compare_renderings,
)
from litetune.verify import BackendPair, Status, VerifyRequest, build_backends, run_verify

PROMPTS = ["what is 2+2?", "classify: it was fine"]


def rows(ids_by_prompt, prefill=None, rendered="rendered"):
    return [
        {
            "index": i,
            "ids": ids,
            "rendered": rendered,
            "prefill_tokens": None if prefill is None else prefill[i],
        }
        for i, ids in enumerate(ids_by_prompt)
    ]


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def test_identical_ids_and_prefill_counts_agree():
    comparison = compare_renderings(
        PROMPTS,
        rows([[1, 2, 3], [4, 5]], prefill=[3, None]),
        rows([[1, 2, 3], [4, 5]]),
        prefill_sent=1,
    )

    assert comparison.agrees
    check = comparison.check()
    assert check.name == RENDERING_CHECK
    assert check.outcome.value == "passed"
    record = comparison.as_dict()
    assert record["prompts_compared"] == 2
    assert record["prefill_sampled"] == [{"index": 0, "prefill_tokens": 3, "reference_tokens": 3}]
    assert record["mismatches"] == 0


def test_differing_ids_name_the_prompt_the_counts_and_where_they_split():
    comparison = compare_renderings(
        PROMPTS,
        rows([[1, 2, 3], [4, 9, 5, 6]], rendered="...<|im_start|>assistant\n"),
        rows([[1, 2, 3], [4, 5, 6]], rendered="...<|im_start|>assistant\n<think>"),
        prefill_sent=0,
    )

    assert not comparison.agrees
    (mismatch,) = comparison.mismatches
    assert (mismatch.index, mismatch.kind) == (1, "ids")
    assert (mismatch.runtime_tokens, mismatch.reference_tokens) == (4, 3)
    assert mismatch.first_difference == 1
    assert mismatch.reference_tail.endswith("<think>")
    check = comparison.check()
    assert check.outcome.value == "failed"
    assert "1 of 2 prompts differ" in check.detail
    assert "runtime renders 4 tokens and the reference 3" in check.detail
    assert "position 1" in check.detail


def test_ids_of_the_same_length_are_still_compared_id_by_id():
    comparison = compare_renderings(
        PROMPTS[:1], rows([[4, 9, 6]]), rows([[4, 5, 6]]), prefill_sent=0
    )
    (mismatch,) = comparison.mismatches
    assert (mismatch.runtime_tokens, mismatch.reference_tokens) == (3, 3)
    assert mismatch.first_difference == 1


def test_one_list_that_is_a_prefix_of_the_other_splits_where_the_shorter_ends():
    # The shape an extra BOS, or a missing generation prompt, takes.
    comparison = compare_renderings(
        PROMPTS[:1], rows([[2, 2, 7]]), rows([[2, 2, 7, 8]]), prefill_sent=0
    )
    assert comparison.mismatches[0].first_difference == 3


def test_a_prefill_count_the_rendering_does_not_explain_is_a_mismatch():
    # The rendered ids agree, and the runtime still prefilled one more token when
    # it sent the prompt: a BOS the session prepends outside the rendered text.
    comparison = compare_renderings(
        PROMPTS[:1], rows([[2, 7, 8]], prefill=[4]), rows([[2, 7, 8]]), prefill_sent=1
    )

    (mismatch,) = comparison.mismatches
    assert mismatch.kind == "prefill"
    assert mismatch.prefill_tokens == 4
    assert "prefilled 4 tokens" in comparison.check().detail


@pytest.mark.parametrize("side", ["runtime", "reference"])
def test_a_script_that_did_not_cover_every_prompt_is_not_a_comparison(side):
    full, short = rows([[1], [2]]), rows([[1]])
    runtime, reference = (short, full) if side == "runtime" else (full, short)
    with pytest.raises(RenderingProbeError, match=f"the {side} rendering script returned 1"):
        compare_renderings(PROMPTS, runtime, reference, prefill_sent=0)


# ---------------------------------------------------------------------------
# The runtime's script, against a fake `litert_lm`
# ---------------------------------------------------------------------------


class FakeEngine:
    """One id per character, `<bos>` read as literal text the way the tokenizer reads it."""

    def __init__(self, bos: int | None, prefill_extra: int = 0):
        self.bos_token_id = bos
        self.prefill_extra = prefill_extra
        self.sent: list[str] = []
        self.rendered: list[str] = []
        # Every `create_conversation` call, by the arguments it was given. What
        # the runtime is asked for is the thing under test once declarations are
        # in play, and a rendered string cannot show whether they were passed.
        self.conversations: list[dict] = []

    def tokenize(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def detokenize(self, ids: list[int]) -> str:
        return "<bos>" if ids == [self.bos_token_id] else "".join(chr(i) for i in ids)


def _exec(source: str, name: str) -> dict:
    namespace: dict = {"__name__": name}
    exec(compile(source, f"{name}.py", "exec"), namespace)
    return namespace


def test_prefill_ids_turn_a_leading_bos_string_into_the_bos_id():
    prefill_ids = _exec(_RUNTIME_SCRIPT, "runtime_script")["prefill_ids"]
    engine = FakeEngine(bos=2)

    assert prefill_ids(engine, "<bos>hi") == [2, ord("h"), ord("i")]
    # Only a leading one, the way `StringToProcessedInputText` checks.
    assert prefill_ids(engine, "hi<bos>") == [ord(c) for c in "hi<bos>"]


def test_prefill_ids_leave_text_alone_for_a_model_without_a_bos():
    prefill_ids = _exec(_RUNTIME_SCRIPT, "runtime_script")["prefill_ids"]
    assert prefill_ids(FakeEngine(bos=None), "<bos>hi") == [ord(c) for c in "<bos>hi"]


def test_a_first_turn_starts_with_the_bos_the_session_prepends():
    runtime_ids = _exec(_RUNTIME_SCRIPT, "runtime_script")["runtime_ids"]

    # gemma-3-270m: the rendering carries no <bos>; the session adds it.
    assert runtime_ids(FakeEngine(bos=2), "hi") == [2, ord("h"), ord("i")]
    # A rendering that brings its own gets two, as the runtime would give it.
    assert runtime_ids(FakeEngine(bos=2), "<bos>hi") == [2, 2, ord("h"), ord("i")]
    # Qwen3: no BOS at all, nothing added.
    assert runtime_ids(FakeEngine(bos=None), "hi") == [ord("h"), ord("i")]


def _fake_litert_lm(engine: FakeEngine) -> Any:
    module: Any = types.ModuleType("litert_lm")

    class Conversation:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
            self.prefill: int | None = None
            engine.conversations.append(kwargs)

        def __enter__(self) -> Conversation:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def render_message_to_string(self, prompt: str) -> str:
            engine.rendered.append(prompt)
            return f"[{prompt}]"

        def send_message(self, prompt: str) -> dict:
            engine.sent.append(prompt)
            self.prefill = len(prompt) + 2 + engine.prefill_extra
            return {"content": []}

        def get_benchmark_info(self) -> Any:
            return types.SimpleNamespace(last_prefill_token_count=self.prefill)

    class Engine:
        def __init__(self, model: str, backend: object, enable_benchmark: bool):
            module.opened = (model, enable_benchmark)

        def __enter__(self) -> FakeEngine:
            engine.create_conversation = Conversation  # type: ignore[attr-defined]
            return engine

        def __exit__(self, *exc: object) -> None:
            return None

    module.Engine = Engine

    class Tool:
        """The base `create_conversation` requires: it takes `Tool` instances or
        callables, never raw JSON, and asks each for `get_tool_description()`."""

        def get_tool_description(self) -> dict:
            raise NotImplementedError

        def execute(self, param: Any) -> Any:
            raise NotImplementedError

    module.Tool = Tool
    module.Backend = types.SimpleNamespace(CPU=lambda: "cpu")
    module.SamplerConfig = lambda **kwargs: kwargs
    module.LogSeverity = types.SimpleNamespace(ERROR="error")
    module.set_min_log_severity = lambda level: None
    return module


def _run_script(source, tmp_path, monkeypatch, modules, spec) -> list[dict]:
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    out = tmp_path / "rows.json"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({**spec, "out": str(out)}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["render.py", str(spec_path)])
    assert _exec(source, "render_script_under_test")["main"]() == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_runtime_script_renders_every_prompt_and_sends_only_the_sample(tmp_path, monkeypatch):
    engine = FakeEngine(bos=2)
    prompts = ["a", "bb", "ccc"]

    written = _run_script(
        _RUNTIME_SCRIPT,
        tmp_path,
        monkeypatch,
        {"litert_lm": _fake_litert_lm(engine)},
        {"model": "m.litertlm", "prompts": prompts, "prefill_sample": 2},
    )

    assert engine.rendered == prompts
    assert engine.sent == ["a", "bb"]
    assert [row["prefill_tokens"] for row in written] == [3, 4, None]
    assert written[1]["rendered"] == "[bb]"
    assert written[1]["ids"] == [2, ord("["), ord("b"), ord("b"), ord("]")]


TOOLS = [{"type": "function", "function": {"name": "open_app"}}]


def test_the_runtime_script_declares_the_tools_it_was_given(tmp_path, monkeypatch):
    """The runtime renders the declaration turn itself, from the tools it is
    handed. Nothing else in this check would notice their absence: the ids would
    simply agree on a prompt neither side carries declarations in.
    """
    engine = FakeEngine(bos=2)

    _run_script(
        _RUNTIME_SCRIPT,
        tmp_path,
        monkeypatch,
        {"litert_lm": _fake_litert_lm(engine)},
        {"model": "m.litertlm", "prompts": ["hi"], "prefill_sample": 1, "tools": TOOLS},
    )

    assert engine.conversations, "no conversation was created at all"
    for call in engine.conversations:
        assert [t.get_tool_description() for t in call["tools"]] == TOOLS
        # The prefill sample sends a message for real; executing a declared tool
        # to measure a prompt is not something a check may do.
        assert call["automatic_tool_calling"] is False


def test_without_declarations_the_conversation_is_the_one_it_always_made(tmp_path, monkeypatch):
    """The no-declarations path must not acquire arguments. A runtime whose
    Python API has no `Tool` at all -- an older one, or a fake in a suite that
    never needed it -- still has to render."""
    engine = FakeEngine(bos=2)

    _run_script(
        _RUNTIME_SCRIPT,
        tmp_path,
        monkeypatch,
        {"litert_lm": _fake_litert_lm(engine)},
        {"model": "m.litertlm", "prompts": ["hi"], "prefill_sample": 1},
    )

    assert engine.conversations
    for call in engine.conversations:
        assert "tools" not in call
        assert "automatic_tool_calling" not in call


def test_both_sides_are_asked_about_the_same_declarations(monkeypatch, tmp_path):
    """The check compares what two renderers produce. Asking them different
    questions would compare two different prompts and report the difference as a
    rendering fault."""
    seen: list[tuple[str, dict]] = []

    def run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        seen.append((self.name, spec))
        Path(spec["out"]).write_text(json.dumps(rows([[1, 2]])), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    probe = _probe(monkeypatch, tmp_path, run)
    probe.declarations = TOOLS

    probe.observe(["hi"])

    assert [name for name, _ in seen] == [envs.RUNTIME.name, envs.TRAIN.name]
    assert seen[0][1]["tools"] == TOOLS
    assert seen[1][1]["tools"] == TOOLS


@pytest.mark.parametrize(
    "sample, sent",
    [(0, []), (2, ["a", "bb"]), (-1, ["a", "bb"]), (100, ["a", "bb", "ccc"])],
)
def test_the_script_sends_what_the_probe_counts_for_every_sample(
    sample, sent, tmp_path, monkeypatch
):
    """`RenderingProbe.observe` records `len(prompts[: prefill_sample])` as the
    number sent, and never sees the script's own `rows[: prefill_sample]`. The
    two are the same expression over the same length only as long as this holds
    for every value the public field can take -- a zero, a negative, and a
    sample larger than the split."""
    engine = FakeEngine(bos=2)
    prompts = ["a", "bb", "ccc"]

    _run_script(
        _RUNTIME_SCRIPT,
        tmp_path,
        monkeypatch,
        {"litert_lm": _fake_litert_lm(engine)},
        {"model": "m.litertlm", "prompts": prompts, "prefill_sample": sample},
    )

    assert engine.sent == sent
    assert len(engine.sent) == len(prompts[:sample])


# ---------------------------------------------------------------------------
# The reference's script, against a fake `transformers`
# ---------------------------------------------------------------------------


def _fake_transformers() -> Any:
    class Tokenizer:
        bos_token_id = 2

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
            ids = [2 if word == "<bos>" else 10 + n for n, word in enumerate(text.split())]
            return {"input_ids": [2, *ids] if add_special_tokens else ids}

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=False, tools=None
        ):
            # Without tools this renders exactly what it rendered before they
            # were an input, word for word, so the test below still pins the
            # same ids.
            declarations = "".join(f"decl:{t['function']['name']} " for t in (tools or []))
            return "<bos> " + declarations + "user " + messages[0]["content"] + " model"

    module: Any = types.ModuleType("transformers")
    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda model: Tokenizer())
    return module


def test_the_reference_script_gives_the_ids_the_reference_generates_from(tmp_path, monkeypatch):
    written = _run_script(
        _REFERENCE_SCRIPT,
        tmp_path,
        monkeypatch,
        {"transformers": _fake_transformers()},
        {"model": "org/reference", "prompts": ["hi"]},
    )

    # One BOS: the template's. The same ids the generation script is given --
    # `test_evaluate.py` pins that side with the same tokenizer shape.
    assert written == [{"index": 0, "rendered": "<bos> user hi model", "ids": [2, 11, 12, 13]}]


def test_the_reference_script_renders_the_declarations_the_spec_carries(tmp_path, monkeypatch):
    """The spec is how the declarations reach this side, and the shared source
    renders them here exactly as it does in training.

    Without this, a script that stopped passing them would leave both sides
    agreeing on a prompt no serving caller sends, and every other test in this
    suite would stay green: the others render `render_prompt` directly or supply
    no tools at all.
    """
    written = _run_script(
        _REFERENCE_SCRIPT,
        tmp_path,
        monkeypatch,
        {"transformers": _fake_transformers()},
        {
            "model": "org/reference",
            "prompts": ["hi"],
            "tools": [{"type": "function", "function": {"name": "open_app"}}],
        },
    )

    assert "decl:open_app" in written[0]["rendered"]
    # The declaration is part of what the model is measured on, not a label
    # beside it: one more rendered word is one more id.
    assert written[0]["ids"] == [2, 11, 12, 13, 14]


# ---------------------------------------------------------------------------
# The probe, against a fake environment
# ---------------------------------------------------------------------------


def _probe(monkeypatch, tmp_path, run) -> RenderingProbe:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(envs.StageEnv, "run", run)
    return RenderingProbe(model=tmp_path / "m.litertlm", reference="org/ref", auto_provision=False)


def test_the_probe_runs_one_script_per_environment_and_compares(monkeypatch, tmp_path):
    seen: list[tuple[str, dict]] = []

    def run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        seen.append((self.name, spec))
        written = rows([[1, 2]], prefill=[2] if "prefill_sample" in spec else None)
        Path(spec["out"]).write_text(json.dumps(written), encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    comparison = _probe(monkeypatch, tmp_path, run).observe(["hi"])

    assert comparison.agrees
    assert [name for name, _ in seen] == [envs.RUNTIME.name, envs.TRAIN.name]
    assert seen[0][1]["prefill_sample"] == 8
    assert seen[1][1]["model"] == "org/ref"


def test_a_script_that_fails_says_where_and_why(monkeypatch, tmp_path):
    def run(self, args, timeout=3600, **kwargs):
        return subprocess.CompletedProcess(
            args, 1, "", "AttributeError: no render_message_to_string"
        )

    with pytest.raises(RenderingProbeError, match="exited 1: AttributeError"):
        _probe(monkeypatch, tmp_path, run).observe(["hi"])


def test_a_script_that_exits_non_zero_after_writing_is_not_believed(monkeypatch, tmp_path):
    def run(self, args, timeout=3600, **kwargs):
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        Path(spec["out"]).write_text(json.dumps(rows([[1]])), encoding="utf-8")
        return subprocess.CompletedProcess(args, 3, "", "died on the way out")

    with pytest.raises(RenderingProbeError, match="exited 3"):
        _probe(monkeypatch, tmp_path, run).observe(["hi"])


def test_a_script_that_hangs_is_reported_not_waited_on(monkeypatch, tmp_path):
    def run(self, args, timeout=3600, **kwargs):
        raise subprocess.TimeoutExpired(args, timeout)

    with pytest.raises(RenderingProbeError, match="no result after"):
        _probe(monkeypatch, tmp_path, run).observe(["hi"])


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class FakeObserver:
    def __init__(self, runtime_ids=None, prefill=None, raises: Exception | None = None):
        self.runtime_ids = runtime_ids
        self.prefill = prefill
        self.raises = raises
        self.calls: list[list[str]] = []

    def observe(self, prompts, events=None) -> RenderingComparison:
        self.calls.append(list(prompts))
        if self.raises is not None:
            raise self.raises
        reference = [[i, 100] for i in range(len(prompts))]
        runtime = self.runtime_ids(reference) if self.runtime_ids else reference
        prefill = [None] * len(prompts)
        if self.prefill is not None:
            prefill[0] = self.prefill
        # This observer sends exactly the prompt it answers for, and none when
        # it answers for none. Saying so is the point of the argument.
        return compare_renderings(
            prompts,
            rows(runtime, prefill=prefill),
            rows(reference),
            prefill_sent=1 if self.prefill is not None else 0,
        )


def _verify(write_split, observer, prompt_mode=PromptMode.RUNTIME_RENDERED):
    labelled = labelled_rows(8)
    return run_verify(
        VerifyRequest(
            model=Path("m.litertlm"),
            reference="org/reference",
            data=write_split(labelled),
            prompt_mode=prompt_mode,
        ),
        backends=BackendPair(
            candidate=FakeBackend(texts=correct_texts(labelled), prompt_mode=prompt_mode),
            reference=FakeBackend(
                model="org/reference", texts=correct_texts(labelled), prompt_mode=prompt_mode
            ),
            rendering=observer,
        ),
    )


def _rendering_check(result):
    return next(c for c in result.manifest["checks"] if c["name"] == RENDERING_CHECK)


def test_agreeing_renderings_are_recorded_and_the_measurement_goes_on(write_split):
    observer = FakeObserver(prefill=2)
    result = _verify(write_split, observer)

    assert result.status is Status.PASSED
    assert len(observer.calls[0]) == 8
    assert _rendering_check(result)["outcome"] == "passed"
    record = result.manifest["harness"]["rendering_check"]
    assert record["applied"] is True
    assert record["prompts_compared"] == 8
    assert record["prefill_sampled"] == [{"index": 0, "prefill_tokens": 2, "reference_tokens": 2}]


def test_differing_ids_end_the_run_before_anything_is_generated(write_split):
    def extra_bos(reference):
        return [[2, *ids] for ids in reference]

    result = _verify(write_split, FakeObserver(runtime_ids=extra_bos))

    assert result.status is Status.FAILED_HARNESS
    assert result.exit_code == 4
    check = _rendering_check(result)
    assert check["outcome"] == "failed"
    assert "8 of 8 prompts differ" in check["detail"]
    assert "position 0" in check["detail"]
    assert "candidate" not in result.manifest["measurements"]
    assert result.manifest["attribution"] == {}


def test_a_differing_prefill_count_ends_the_run_too(write_split):
    result = _verify(write_split, FakeObserver(prefill=3))

    assert result.status is Status.FAILED_HARNESS
    assert "prefilled 3 tokens" in _rendering_check(result)["detail"]
    assert "candidate" not in result.manifest["measurements"]


def test_a_check_that_could_not_run_is_not_a_pass(write_split):
    result = _verify(write_split, FakeObserver(raises=RenderingProbeError("runtime died")))

    assert result.status is Status.FAILED_HARNESS
    assert _rendering_check(result)["outcome"] == "could_not_check"
    assert "candidate" not in result.manifest["measurements"]


def test_a_prerendered_measurement_does_not_run_the_check(write_split):
    observer = FakeObserver(runtime_ids=lambda reference: [[] for _ in reference])
    result = _verify(write_split, observer, prompt_mode=PromptMode.PRERENDERED)

    assert result.status is Status.PASSED
    assert observer.calls == []
    record = result.manifest["harness"]["rendering_check"]
    assert record["applied"] is False
    assert record["reason"].startswith("prerendered")


def test_backends_without_an_observer_say_the_check_was_not_run(write_split):
    result = _verify(write_split, None)

    record = result.manifest["harness"]["rendering_check"]
    assert record["applied"] is False
    assert any("without a rendering observer" in text for text in result.manifest["limitations"])


def test_the_real_backends_carry_the_real_probe(tmp_path):
    pair = build_backends(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=tmp_path / "d.jsonl",
            prompt_mode=PromptMode.RUNTIME_RENDERED,
        )
    )
    assert isinstance(pair.rendering, RenderingProbe)
    assert pair.rendering.model == tmp_path / "m.litertlm"
    assert pair.rendering.reference == "org/reference"


def test_a_run_that_reported_no_prefill_count_is_not_a_pass():
    """Equal ids are half the check; the prefill count is the half that catches
    tokens added outside the rendered text, which is how the session BOS was
    found. If the runtime returned no count for any prompt it was sent, that
    half did not run and the check must not vouch for it."""
    comparison = compare_renderings(
        PROMPTS,
        rows([[1, 2, 3], [4, 5, 6]]),
        rows([[1, 2, 3], [4, 5, 6]]),
        prefill_sent=2,
    )

    assert comparison.agrees
    check = comparison.check()
    assert check.outcome.value == "could_not_check"
    assert "no prefill count" in check.detail
    assert check.observed["prompts_compared"] == 2


def _answers_every_prompt_without_a_prefill_count(self, args, timeout=3600, **kwargs):
    """A runtime that renders every prompt and reports no prefill count at all."""
    spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
    written = rows([[1, 2, 3] for _ in spec["prompts"]])
    Path(spec["out"]).write_text(json.dumps(written), encoding="utf-8")
    return subprocess.CompletedProcess(args, 0, "", "")


def test_the_count_in_the_message_is_the_prompts_the_runtime_was_sent(monkeypatch, tmp_path):
    """The runtime script prefills `rows[: prefill_sample]`, so a split shorter
    than the sample sends fewer prompts than the sample names. Reporting the
    sample instead would put a number of sends that never happened into the
    message and into the manifest -- the defect this check exists to catch,
    committed by the check itself."""
    comparison = _probe(
        monkeypatch, tmp_path, _answers_every_prompt_without_a_prefill_count
    ).observe(["hi"])

    assert comparison.prefill_sent == 1
    check = comparison.check()
    assert check.outcome.value == "could_not_check"
    assert "any of the 1 prompts it was sent" in check.detail
    assert check.observed["prefill_sent"] == 1


def test_asking_for_no_prefill_sample_is_not_a_failed_check(monkeypatch, tmp_path):
    """Zero is a caller's choice, not a runtime that went quiet.

    `RenderingProbe.prefill_sample` is public and slicing by 0 sends nothing,
    so treating that as an unanswered comparison would fail a run whose ids
    agree over every prompt.
    """
    probe = _probe(monkeypatch, tmp_path, _answers_every_prompt_without_a_prefill_count)
    probe.prefill_sample = 0

    check = probe.observe(PROMPTS).check()

    assert check.outcome.value == "passed"
    assert check.observed["prefill_sent"] == 0


def test_a_negative_prefill_sample_reports_what_the_slice_sent(monkeypatch, tmp_path):
    """`prefill_sample` is public and unvalidated, and `rows[:-1]` sends every
    prompt but the last. Following the slice rather than the number keeps the
    recorded count from going negative or naming sends that did not happen."""
    probe = _probe(monkeypatch, tmp_path, _answers_every_prompt_without_a_prefill_count)
    probe.prefill_sample = -1

    comparison = probe.observe(PROMPTS)

    assert comparison.prefill_sent == 1
    # The message is where a wrong number would be read, so pin it there too.
    check = comparison.check()
    assert check.outcome.value == "could_not_check"
    assert "any of the 1 prompts it was sent" in check.detail


def test_the_comparison_will_not_guess_what_the_runtime_was_sent():
    """`prefill_sent` has no default, and that is a contract rather than a style.

    A default of zero lets a caller that never established the number record
    "none were sent" in the manifest, and zero also switches off the guard at
    the top of `check()` -- so an observer whose runtime answered nothing would
    read as passed, which is the one thing that guard exists to prevent. Nothing
    else in the suite fails if the default comes back, because every call site
    passes the argument; this test is what holds the signature.
    """
    with pytest.raises(TypeError, match="prefill_sent"):
        compare_renderings(PROMPTS, rows([[1], [2]]), rows([[1], [2]]))


def test_a_partly_answered_sample_does_not_call_the_answers_the_sends():
    """Eight prompts sent, three answered: the passing message must not put the
    word "sent" on the number that answered.

    The five that were sent and said nothing are the half of the check that
    catches tokens added outside the rendered text, and a reader comparing the
    detail against `prefill_sent` in the same record would otherwise get two
    different answers to how many the runtime got.
    """
    prompts = [f"prompt {i}" for i in range(10)]
    ids = [[1, 2, 3] for _ in prompts]
    answered = [3, 3, 3] + [None] * 7

    comparison = compare_renderings(prompts, rows(ids, prefill=answered), rows(ids), prefill_sent=8)

    check = comparison.check()
    assert check.outcome.value == "passed"
    assert "on 3 of the 8 prompts it was sent" in check.detail
    assert check.observed["prefill_sent"] == 8
    assert len(check.observed["prefill_sampled"]) == 3
