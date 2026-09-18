"""Which calling convention a measurement is taken in, and where that came from.

`--no-template` is narrow. It routes the runtime to `create_session()` instead
of `create_conversation()`, bypassing the chat template, the `<|turn>model`
anchor, tool handling and channel extraction, and it is right only when the
caller built the whole prompt including control tokens. That is a property of
how the checkpoint was *trained*, not of its family, so it is decided by `tune`,
carried by the bundle contract, and only inferred -- visibly -- for an artifact
litetune did not produce.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from conftest import FakeBackend, correct_texts, labelled_rows

from litetune import envs
from litetune.bundle import Contract
from litetune.declarations import canonical_text, read_declarations
from litetune.evaluate import HuggingFaceBackend, LiteRtLmBackend
from litetune.prompt_mode import (
    RENDERING_SOURCE,
    TURN_MARKERS,
    PromptMode,
    PromptModeConflict,
    marker_share,
    parse_prompt_mode,
    prompt_evidence,
    resolve_prompt_mode,
)
from litetune.storage import hash_file
from litetune.verify import (
    DECLARATIONS_CHECK,
    EXIT_CODES,
    BackendPair,
    Status,
    VerifyRequest,
    build_backends,
    run_verify,
)

RENDERED = "<start_of_turn>user\nset the background to red<end_of_turn>\n<start_of_turn>model\n"
BARE = "set the background to red"


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------


def test_a_declared_mode_wins_over_everything():
    decision = resolve_prompt_mode(
        [RENDERED] * 4,
        declared=PromptMode.RUNTIME_RENDERED,
        contract=PromptMode.PRERENDERED,
    )
    assert decision.mode is PromptMode.RUNTIME_RENDERED
    assert decision.source == "declared"
    assert not decision.inferred


def test_the_contract_wins_over_the_prompts():
    # The contract is where `tune`'s decision was written down; the prompts are
    # a heuristic. The recorded fact beats the guess.
    decision = resolve_prompt_mode([BARE] * 4, contract=PromptMode.PRERENDERED)
    assert decision.mode is PromptMode.PRERENDERED
    assert decision.source == "contract"


def test_prompts_that_already_carry_control_tokens_are_prerendered():
    decision = resolve_prompt_mode([RENDERED] * 10)
    assert decision.mode is PromptMode.PRERENDERED
    assert decision.source == "inferred"
    assert "<start_of_turn>" in decision.evidence
    assert "double-wrap" in decision.evidence


def test_bare_prompts_need_the_runtime_to_render_them():
    decision = resolve_prompt_mode([BARE] * 10)
    assert decision.mode is PromptMode.RUNTIME_RENDERED
    assert decision.source == "inferred"
    assert not decision.ambiguous


def test_a_split_that_mixes_the_two_conventions_says_so():
    decision = resolve_prompt_mode([RENDERED] * 5 + [BARE] * 5)
    assert decision.ambiguous
    assert "mixes the two conventions" in decision.evidence
    # Something has to run: the prompts are used as they are, which transforms
    # nothing and leaves the evidence in the record.
    assert decision.mode is PromptMode.PRERENDERED


def test_marker_share_names_what_it_saw():
    share, seen = marker_share([RENDERED, BARE])
    assert share == 0.5
    assert seen == ("<start_of_turn>",)


@pytest.mark.parametrize(
    ("rendered", "bare", "mode"),
    [
        (0, 10, PromptMode.RUNTIME_RENDERED),
        (1, 9, PromptMode.RUNTIME_RENDERED),
        (5, 5, None),
        (9, 1, PromptMode.PRERENDERED),
        (10, 0, PromptMode.PRERENDERED),
    ],
)
def test_both_bounds_belong_to_the_mode_they_name(rendered, bare, mode):
    evidence = prompt_evidence([RENDERED] * rendered + [BARE] * bare)
    assert evidence.mode is mode
    assert evidence.count == 10


def test_one_rendered_prompt_in_ten_is_bare_text_to_verify_too():
    # `share <= 1.0 - 0.9` put this split in the mixed band: 1.0 - 0.9 is
    # 0.09999999999999998, and 1/10 is not below it.
    decision = resolve_prompt_mode([RENDERED] + [BARE] * 9)
    assert decision.mode is PromptMode.RUNTIME_RENDERED
    assert not decision.ambiguous
    assert decision.markers == ("<start_of_turn>",)


# ---------------------------------------------------------------------------
# The backends
# ---------------------------------------------------------------------------


def test_the_runtime_only_gets_no_template_for_a_prerendered_prompt(tmp_path):
    model = tmp_path / "model.litertlm"
    prerendered = LiteRtLmBackend(
        model=model, auto_provision=False, declared_prompt_mode=PromptMode.PRERENDERED
    )
    templated = LiteRtLmBackend(
        model=model, auto_provision=False, declared_prompt_mode=PromptMode.RUNTIME_RENDERED
    )
    assert "--no-template" in prerendered.argv("hi")
    assert "--no-template" not in templated.argv("hi")
    assert templated.describe()["template_flag"] is None
    assert templated.prompt_mode is PromptMode.RUNTIME_RENDERED


def test_a_backend_records_whether_the_mode_was_declared(tmp_path):
    model = tmp_path / "model.litertlm"
    undeclared = LiteRtLmBackend(model=model, auto_provision=False)
    declared = LiteRtLmBackend(
        model=model, auto_provision=False, declared_prompt_mode=PromptMode.PRERENDERED
    )
    assert undeclared.describe()["prompt_mode_declared"] is False
    assert declared.describe()["prompt_mode_declared"] is True


def test_a_declared_mode_reaches_the_reference_generation_script(monkeypatch):
    seen: list[dict] = []

    def fake_run(self, args, timeout=3600, **kwargs):
        seen.append(json.loads(Path(args[2]).read_text()))
        return subprocess.CompletedProcess(args, 1, "", "")

    monkeypatch.setattr(envs.StageEnv, "run", fake_run)
    backend = HuggingFaceBackend(
        model="org/model",
        auto_provision=False,
        # The low-level switch says no template; the declared mode says
        # otherwise, and the declared mode is the one that was decided.
        runtime_rendered=False,
        declared_prompt_mode=PromptMode.RUNTIME_RENDERED,
    )
    backend.generate(["a"])
    assert seen[0]["runtime_rendered"] is True
    assert backend.prompt_mode is PromptMode.RUNTIME_RENDERED


def test_both_sides_are_built_from_one_decision(tmp_path):
    pair = build_backends(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=tmp_path / "d.jsonl",
            prompt_mode=PromptMode.RUNTIME_RENDERED,
        )
    )
    assert pair.candidate.prompt_mode is PromptMode.RUNTIME_RENDERED
    assert pair.reference.prompt_mode is PromptMode.RUNTIME_RENDERED
    assert "--no-template" not in pair.candidate.argv("hi")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _pair(rows):
    return BackendPair(
        candidate=FakeBackend(texts=correct_texts(rows)),
        reference=FakeBackend(model="org/reference", texts=correct_texts(rows)),
    )


def _contract(tmp_path: Path, mode: PromptMode) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            Contract(
                prompt_mode=mode,
                established_against={"litert-lm": "0.16.1"},
                base_model="org/base",
                base_model_revision="a" * 40,
            ).as_dict()
        ),
        encoding="utf-8",
    )
    return path


def test_verify_reads_the_mode_out_of_the_bundle_contract(tmp_path, write_split):
    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=write_split(rows),
            contract=_contract(tmp_path, PromptMode.RUNTIME_RENDERED),
        ),
        backends=_pair(rows),
    )
    decision = result.manifest["harness"]["prompt_mode_decision"]
    assert decision["source"] == "contract"
    assert decision["prompt_mode"] == "runtime_rendered"
    # A recorded mode is not a guess, so nothing is flagged.
    assert not any("was not declared" in text for text in result.manifest["limitations"])


def test_an_explicit_mode_beats_the_contract(tmp_path, write_split):
    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=write_split(rows),
            contract=_contract(tmp_path, PromptMode.RUNTIME_RENDERED),
            prompt_mode=PromptMode.PRERENDERED,
        ),
        backends=_pair(rows),
    )
    decision = result.manifest["harness"]["prompt_mode_decision"]
    assert decision["source"] == "declared"
    assert decision["prompt_mode"] == "prerendered"


def test_a_contract_that_cannot_be_read_is_not_quietly_replaced_by_a_guess(tmp_path, write_split):
    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=write_split(rows),
            contract=tmp_path / "absent.json",
        ),
        backends=_pair(rows),
    )
    assert result.status is Status.FAILED_HARNESS
    assert result.manifest["checks"][-1]["outcome"] == "could_not_check"


def test_an_inferred_mode_travels_with_the_measurement(tmp_path, write_split):
    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm", reference="org/reference", data=write_split(rows)
        ),
        backends=_pair(rows),
    )
    decision = result.manifest["harness"]["prompt_mode_decision"]
    assert decision["source"] == "inferred"
    # These prompts are bare text, so the runtime has to render its own
    # template: the flag was never general.
    assert decision["prompt_mode"] == "runtime_rendered"
    note = next(text for text in result.manifest["limitations"] if "was not declared" in text)
    assert "--no-template" in note
    assert "create_session()" in note


def test_a_supplied_backend_that_ignores_the_resolved_mode_is_recorded(tmp_path, write_split):
    # Only reachable when a caller injects their own backends: the manifest
    # carries the resolved mode and the measured one, and says they differ
    # rather than letting the two fields disagree in silence.
    rows = labelled_rows(8)
    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference="org/reference",
            data=write_split(rows),
            contract=_contract(tmp_path, PromptMode.RUNTIME_RENDERED),
        ),
        backends=_pair(rows),  # both fakes measure prerendered
    )
    assert result.manifest["harness"]["prompt_mode"] == "prerendered"
    assert result.manifest["harness"]["prompt_mode_decision"]["prompt_mode"] == "runtime_rendered"
    assert any("did not take the resolved mode" in text for text in result.manifest["limitations"])


# ---------------------------------------------------------------------------
# The record `tune` leaves beside the checkpoint
# ---------------------------------------------------------------------------


def test_the_training_record_wins_and_says_where_it_came_from():
    decision = resolve_prompt_mode([BARE] * 10, recorded=PromptMode.PRERENDERED)
    assert decision.mode is PromptMode.PRERENDERED
    assert decision.source == "checkpoint"
    assert "litetune.json" in decision.evidence


def test_a_declared_mode_or_contract_that_agrees_with_the_record_is_the_record():
    decision = resolve_prompt_mode(
        [BARE] * 10,
        declared=PromptMode.PRERENDERED,
        contract=PromptMode.PRERENDERED,
        recorded=PromptMode.PRERENDERED,
    )
    assert decision.source == "checkpoint"


@pytest.mark.parametrize("which", ["declared", "contract"])
def test_a_value_that_contradicts_the_record_raises_naming_both(which):
    with pytest.raises(PromptModeConflict) as raised:
        resolve_prompt_mode(
            [BARE] * 10, recorded=PromptMode.PRERENDERED, **{which: PromptMode.RUNTIME_RENDERED}
        )
    assert "runtime_rendered" in str(raised.value)
    assert "prerendered" in str(raised.value)


def _checkpoint(tmp_path: Path, record: dict | str) -> str:
    path = tmp_path / "tuned-model"
    path.mkdir()
    text = record if isinstance(record, str) else json.dumps(record)
    (path / "litetune.json").write_text(text, encoding="utf-8")
    return str(path)


def _verify(tmp_path, write_split, reference, **kwargs):
    rows = labelled_rows(8)  # bare text: inference alone would say runtime_rendered
    return run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm", reference=reference, data=write_split(rows), **kwargs
        ),
        backends=_pair(rows),
    )


def test_verify_uses_the_mode_training_recorded_beside_the_reference(tmp_path, write_split):
    reference = _checkpoint(
        tmp_path,
        {
            "base_model": "org/base",
            "prompt_mode": "prerendered",
            "prompt_mode_decision": {"prompt_mode": "prerendered", "source": "overridden"},
        },
    )
    result = _verify(tmp_path, write_split, reference)

    decision = result.manifest["harness"]["prompt_mode_decision"]
    assert decision["source"] == "checkpoint"
    assert decision["prompt_mode"] == "prerendered"
    assert not any("was not declared" in text for text in result.manifest["limitations"])


def test_a_record_that_predates_the_decision_field_is_still_the_record(tmp_path, write_split):
    # The keys `tune` wrote into litetune.json before it recorded how it decided.
    reference = _checkpoint(
        tmp_path,
        {
            "base_model": "org/base",
            "base_model_revision": None,
            "prompt_mode": "prerendered",
            "turn_terminator": {"ids": [1], "source": "tokenizer_eos", "text": "<eos>"},
            "sentencepiece": None,
        },
    )
    result = _verify(tmp_path, write_split, reference)

    assert result.manifest["harness"]["prompt_mode_decision"]["source"] == "checkpoint"


def test_verify_refuses_a_declared_mode_the_training_record_contradicts(tmp_path, write_split):
    reference = _checkpoint(tmp_path, {"base_model": "org/base", "prompt_mode": "prerendered"})
    result = _verify(tmp_path, write_split, reference, prompt_mode=PromptMode.RUNTIME_RENDERED)

    assert result.status is Status.FAILED_HARNESS
    last = result.manifest["checks"][-1]
    assert last["outcome"] == "could_not_check"
    assert "the declared mode is runtime_rendered" in last["detail"]
    assert "trained prerendered" in last["detail"]
    assert "candidate" not in result.manifest["measurements"]


def test_verify_refuses_a_contract_the_training_record_contradicts(tmp_path, write_split):
    reference = _checkpoint(tmp_path, {"base_model": "org/base", "prompt_mode": "prerendered"})
    result = _verify(
        tmp_path,
        write_split,
        reference,
        contract=_contract(tmp_path, PromptMode.RUNTIME_RENDERED),
    )

    assert result.status is Status.FAILED_HARNESS
    assert "the bundle contract is runtime_rendered" in result.manifest["checks"][-1]["detail"]
    assert "candidate" not in result.manifest["measurements"]


def test_a_directory_without_a_record_keeps_the_contract_then_the_prompts(tmp_path, write_split):
    plain = tmp_path / "foreign-model"
    plain.mkdir()
    inferred = _verify(tmp_path, write_split, str(plain))
    assert inferred.manifest["harness"]["prompt_mode_decision"]["source"] == "inferred"


def test_a_record_without_a_mode_keeps_the_contract(tmp_path, write_split):
    reference = _checkpoint(tmp_path, {"base_model": "org/base"})
    result = _verify(
        tmp_path, write_split, reference, contract=_contract(tmp_path, PromptMode.PRERENDERED)
    )
    assert result.manifest["harness"]["prompt_mode_decision"]["source"] == "contract"


@pytest.mark.parametrize("text", ['{"prompt_mode": ', "[1]", '{"prompt_mode": "templated"}'])
def test_a_record_that_cannot_be_read_is_not_replaced_by_a_guess(tmp_path, write_split, text):
    result = _verify(tmp_path, write_split, _checkpoint(tmp_path, text))

    assert result.status is Status.FAILED_HARNESS
    assert result.manifest["checks"][-1]["outcome"] == "could_not_check"


# ---------------------------------------------------------------------------
# One rendering, one parser
# ---------------------------------------------------------------------------


def test_training_the_reference_and_the_rendering_check_render_from_one_source():
    # Training learns from this text and the rendering check compares the
    # reference's ids with the runtime's: a copy in any one script renders a
    # prompt the others may not.
    from litetune.evaluate import _HF_GENERATE_SCRIPT
    from litetune.rendering import _REFERENCE_SCRIPT
    from litetune.tune import _TRAIN_SCRIPT

    for script in (_TRAIN_SCRIPT, _HF_GENERATE_SCRIPT, _REFERENCE_SCRIPT):
        assert script.count(RENDERING_SOURCE) == 1
        assert script.count("def render_prompt(") == 1


def test_every_generated_script_is_valid_python():
    """Counting copies of the source does not prove any of them parses.

    A docstring edited into `RENDERING_SOURCE` at the wrong depth once produced
    seven failures in tests about dtype and device -- every test that execs a
    script -- and none of them said the script would not compile. This says it.
    """
    from litetune.evaluate import _HF_GENERATE_SCRIPT
    from litetune.rendering import _REFERENCE_SCRIPT
    from litetune.tune import _TRAIN_SCRIPT

    for name, script in [
        ("train", _TRAIN_SCRIPT),
        ("generate", _HF_GENERATE_SCRIPT),
        ("reference", _REFERENCE_SCRIPT),
    ]:
        compile(script, f"{name}_script.py", "exec")


def _render_prompt():
    """`render_prompt` out of the shared source, the way every script gets it."""
    namespace: dict = {}
    exec(compile(RENDERING_SOURCE, "rendering_source.py", "exec"), namespace)
    return namespace["render_prompt"]


class _RecordingTokenizer:
    """A tokenizer double that records exactly how the template was called."""

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        tools = kwargs.get("tools")
        declarations = "".join(
            f"<start_function_declaration>declaration:{t['function']['name']}"
            f"<end_function_declaration>"
            for t in (tools or [])
        )
        return f"{declarations}<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n"


def test_declarations_reach_the_prompt_the_model_is_trained_on():
    """The runtime renders a developer turn carrying the declarations, and this
    is the source both the training prompt and the reference's prompt come
    from. Without them the two sides agree on a prompt no serving caller sends.
    """
    tok = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "open_app", "description": "d"}}]

    text, add_special = _render_prompt()(tok, "open maps", True, tools)

    assert "declaration:open_app" in text
    assert tok.calls[0]["tools"] == tools
    assert add_special is False


def test_without_declarations_the_template_is_not_even_asked_about_tools():
    """Byte-identical is not "renders the same thing anyway".

    Four tokenizer doubles in this suite declare `apply_chat_template` without a
    `tools` parameter, and real templates branch on it. Passing `tools=None` to
    find out whether it changes nothing is not the same as not passing it, so
    the no-declarations path must not mention the keyword at all.
    """
    tok = _RecordingTokenizer()

    with_none = _render_prompt()(tok, "open maps", True)
    with_empty = _render_prompt()(tok, "open maps", True, [])

    assert "tools" not in tok.calls[0]
    assert "tools" not in tok.calls[1]
    assert with_none == with_empty
    assert "declaration:" not in with_none[0]


def test_a_prerendered_prompt_is_untouched_by_declarations():
    """The mode decides whether a template runs at all. `prerendered` means the
    caller built the whole prompt, so there is nothing for declarations to be
    rendered into."""
    tok = _RecordingTokenizer()
    tools = [{"type": "function", "function": {"name": "open_app", "description": "d"}}]

    text, add_special = _render_prompt()(tok, "already rendered", False, tools)

    assert text == "already rendered"
    assert add_special is True
    assert tok.calls == []


@pytest.mark.parametrize("raw", ["prerendered", PromptMode.RUNTIME_RENDERED])
def test_a_recorded_mode_reads_back_as_itself(raw):
    assert parse_prompt_mode(raw, "the record") is PromptMode(raw)


@pytest.mark.parametrize("raw", ["templated", "", 1, None])
def test_an_unknown_recorded_mode_is_refused_naming_the_record_and_the_modes(raw):
    with pytest.raises(ValueError) as exc:
        parse_prompt_mode(raw, "run/litetune.json")

    message = str(exc.value)
    assert message.startswith(f"run/litetune.json records prompt_mode {raw!r}")
    assert all(mode.value in message for mode in PromptMode)


@pytest.mark.parametrize("marker", TURN_MARKERS)
def test_every_control_token_the_list_carries_marks_a_prompt_as_rendered(marker):
    """Each entry earns its place, or a family stops being recognised.

    One literal covered this before, so dropping any other marker from the tuple
    left the suite green while a Qwen split scored 0% rendered and trained
    double-wrapped.
    """
    decision = resolve_prompt_mode([f"{marker}user\nhi"] * 10)

    assert decision.mode is PromptMode.PRERENDERED
    assert decision.markers == (marker,)


def test_the_markers_reach_the_record_a_reader_contradicts_it_with():
    # Asserted on the serialised form, not the attribute: the record is what
    # `litetune.json`, the tune metrics and the verify manifest carry.
    record = resolve_prompt_mode([RENDERED] * 10).as_dict()

    assert record["markers"] == ["<start_of_turn>"]
    assert record["source"] == "inferred"


def test_a_sidecar_that_exists_and_cannot_be_read_raises(tmp_path):
    """The docstring promises this, and `is_file()` quietly broke the promise.

    A mode the checkpoint wrote down must not be replaced by one inferred from
    the prompts because the file could not be looked at.
    """
    from litetune.verify import recorded_prompt_mode

    reference = _checkpoint(tmp_path, {"prompt_mode": "prerendered"})
    sidecar = Path(reference) / "litetune.json"
    sidecar.unlink()
    sidecar.mkdir()  # exists, and reading it is an OSError that is not "absent"

    with pytest.raises(OSError):
        recorded_prompt_mode(reference)


def test_a_sidecar_that_is_a_dangling_symlink_raises(tmp_path):
    """A link is an entry: something recorded a mode here and the link stopped
    reaching it. Reading it raises `FileNotFoundError`, the same exception a
    sidecar that was never there raises, and the two are not the same
    statement -- one is "no record", the other is a record that cannot be read.
    """
    from litetune.verify import recorded_prompt_mode

    reference = _checkpoint(tmp_path, {"prompt_mode": "prerendered"})
    sidecar = Path(reference) / "litetune.json"
    sidecar.unlink()
    sidecar.symlink_to(tmp_path / "never-written.json")

    with pytest.raises(OSError):
        recorded_prompt_mode(reference)


def test_a_reference_directory_that_is_a_broken_link_raises(tmp_path):
    """The broken link can be the directory rather than the sidecar.

    Reading `<link>/litetune.json` raises `FileNotFoundError`, and the sidecar's
    own `is_symlink` is false because what is missing is its parent -- so
    nothing about the failure tells it apart from a checkpoint that never
    recorded a mode, unless the reference itself is looked at.
    """
    from litetune.verify import recorded_prompt_mode

    reference = tmp_path / "link-to-nowhere"
    reference.symlink_to(tmp_path / "never-created")

    with pytest.raises(OSError):
        recorded_prompt_mode(str(reference))


def test_a_healthy_symlinked_reference_with_no_sidecar_is_no_record(tmp_path):
    """The companion to the broken-link test, and the one that pins the
    difference between them.

    `models/current -> models/run-42` is an ordinary layout, and a checkpoint
    that never recorded a mode is an ordinary checkpoint. Asking only whether
    the reference is a link refuses both cases alike, and this run would end as
    a harness failure instead of falling through to the contract or the prompts.
    """
    from litetune.verify import recorded_prompt_mode

    real = tmp_path / "run-42"
    real.mkdir()
    link = tmp_path / "current"
    link.symlink_to(real)

    assert recorded_prompt_mode(str(link)) is None


def test_a_reference_with_no_sidecar_is_still_no_record(tmp_path):
    from litetune.verify import recorded_prompt_mode

    empty = tmp_path / "plain-checkpoint"
    empty.mkdir()

    assert recorded_prompt_mode(str(empty)) is None
    assert recorded_prompt_mode("Qwen/Qwen3-0.6B") is None


def _verify_with(tmp_path, write_split, reference, declarations, rows=None):
    """`run_verify` with backends this test keeps a handle on.

    `_verify` above builds its pair inside, which is enough when the question is
    what the manifest says. Here the question is whether anything ran at all, so
    the fakes have to be visible to the assertions.
    """
    rows = rows if rows is not None else labelled_rows(8)
    candidate = FakeBackend(texts=correct_texts(rows))
    reference_backend = FakeBackend(model="org/reference", texts=correct_texts(rows))
    result = run_verify(
        VerifyRequest(
            model=tmp_path / "m.litertlm",
            reference=reference,
            data=write_split(rows),
            declarations=declarations,
        ),
        backends=BackendPair(candidate=candidate, reference=reference_backend),
    )
    return result, candidate, reference_backend


def test_declarations_that_disagree_with_the_checkpoints_record_are_refused(tmp_path, write_split):
    """A model measured against a different tool list than it learned is measured
    on another task, and the number that comes out of that reads as a conversion
    cost. The refusal names both digests, because "they differ" without saying
    which is which leaves the reader to guess what to fix."""
    from litetune.storage import hash_file

    trained = tmp_path / "trained.json"
    trained.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d"}}]',
        encoding="utf-8",
    )
    measured = tmp_path / "measured.json"
    measured.write_text(
        '[{"type": "function", "function": {"name": "set_timer", "description": "d"}}]',
        encoding="utf-8",
    )
    reference = _checkpoint(
        tmp_path, {"prompt_mode": "prerendered", "declarations_sha256": hash_file(trained)}
    )

    result, candidate, reference_backend = _verify_with(tmp_path, write_split, reference, measured)

    assert result.status is Status.FAILED_HARNESS
    assert EXIT_CODES[result.status] == 4
    refusal = next(c for c in result.manifest["checks"] if c["name"] == DECLARATIONS_CHECK)
    # In `prerendered` both are the files' bytes: the digest `tune` recorded
    # for this mode, and the one the refusal compares it with.
    assert hash_file(measured) in refusal["detail"]
    assert hash_file(trained) in refusal["detail"]
    # Refused before either side was asked for anything.
    assert candidate.prompts_seen == []
    assert reference_backend.prompts_seen == []


def test_a_checkpoint_that_learned_declarations_is_not_measured_without_them(tmp_path, write_split):
    """Found in review: the check ran only when the flag was given, so a
    checkpoint that recorded declarations and a run that forgot them was measured
    on a prompt without the tool list it learned -- 0 of 5 on both sides when
    measured -- and passed with no word about it. Only where the runtime renders
    the declarations: in `prerendered` the prompts carry them already."""
    trained = tmp_path / "trained.json"
    trained.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d"}}]',
        encoding="utf-8",
    )
    reference = _checkpoint(
        tmp_path,
        {"prompt_mode": "runtime_rendered", "declarations_sha256": read_declarations(trained)[1]},
    )

    result, candidate, reference_backend = _verify_with(tmp_path, write_split, reference, None)

    assert result.status is Status.FAILED_HARNESS
    refusal = next(c for c in result.manifest["checks"] if c["name"] == DECLARATIONS_CHECK)
    assert "Pass --declarations" in refusal["detail"]
    assert candidate.prompts_seen == []
    assert reference_backend.prompts_seen == []


def test_a_prerendered_checkpoint_is_measured_without_declarations_it_recorded(
    tmp_path, write_split
):
    """Found in review: the refusal fired in `prerendered` too, with a false
    reason -- there the prompts already carry the tool list and the flag reaches
    no backend -- and the README's own walkthrough led into it."""
    trained = tmp_path / "trained.json"
    trained.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d"}}]',
        encoding="utf-8",
    )
    reference = _checkpoint(
        tmp_path, {"prompt_mode": "prerendered", "declarations_sha256": hash_file(trained)}
    )

    result, candidate, _ = _verify_with(tmp_path, write_split, reference, None)

    assert result.status is not Status.FAILED_HARNESS
    assert candidate.prompts_seen != []


def test_a_record_of_the_files_bytes_still_matches_that_file(tmp_path, write_split):
    """Checkpoints trained before the digest named the tool list recorded the
    digest of the file's bytes, and are still the same model on the same list."""
    from litetune.storage import hash_file

    decls = tmp_path / "declarations.json"
    decls.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d"}}]',
        encoding="utf-8",
    )
    reference = _checkpoint(
        tmp_path, {"prompt_mode": "prerendered", "declarations_sha256": hash_file(decls)}
    )

    result, candidate, _ = _verify_with(tmp_path, write_split, reference, decls)

    assert result.status is not Status.FAILED_HARNESS
    assert candidate.prompts_seen != []
    # Found in review: the manifest recorded the tool list's digest where the
    # checkpoint, `tune` and the bundle's contract record the file's bytes.
    assert result.manifest["harness"]["declarations_sha256"] == hash_file(decls)
    assert hash_file(decls) != read_declarations(decls)[1]


def test_a_prerendered_record_matches_only_the_same_bytes(tmp_path, write_split):
    """Found in review: a record of a file whose bytes were already the list's
    canonical text -- one taken from a runtime_rendered bundle -- matched the
    same tools reformatted, because the list's digest was accepted in either
    mode. In `prerendered` the application renders the file as it is."""
    from litetune.declarations import canonical_text

    entries = [{"type": "function", "function": {"name": "send_email", "description": "d"}}]
    trained = tmp_path / "trained.json"
    trained.write_text(canonical_text(read_declarations_of(tmp_path, entries)), encoding="utf-8")
    assert hash_file(trained) == read_declarations(trained)[1]
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(entries), encoding="utf-8")
    reference = _checkpoint(
        tmp_path, {"prompt_mode": "prerendered", "declarations_sha256": hash_file(trained)}
    )

    result, candidate, _ = _verify_with(tmp_path, write_split, reference, reformatted)

    assert result.status is Status.FAILED_HARNESS
    assert candidate.prompts_seen == []


def read_declarations_of(tmp_path, entries):
    source = tmp_path / "entries.json"
    source.write_text(json.dumps(entries), encoding="utf-8")
    return read_declarations(source)[0]


def test_the_declarations_a_bundle_shipped_match_the_record_of_the_file_trained_on(
    tmp_path, write_split
):
    """A `runtime_rendered` bundle ships the list in declared order, and the
    README tells an application to send it as shipped. Its bytes differ from the
    file `tune` read; its tool list does not, and that is what is compared."""
    trained = tmp_path / "trained.json"
    trained.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d",'
        ' "parameters": {"type": "object", "properties": {'
        '"to": {"type": "string", "description": "t"},'
        ' "body": {"type": "string", "description": "b"}}}}}]',
        encoding="utf-8",
    )
    parsed, recorded = read_declarations(trained)
    shipped = tmp_path / "shipped.json"
    shipped.write_text(canonical_text(parsed), encoding="utf-8")
    assert shipped.read_bytes() != trained.read_bytes()
    # `runtime_rendered`, as the docstring says: the fixture said `prerendered`,
    # where the list's digest was accepted too until review found it should not.
    reference = _checkpoint(
        tmp_path, {"prompt_mode": "runtime_rendered", "declarations_sha256": recorded}
    )

    result, _, _ = _verify_with(tmp_path, write_split, reference, shipped)

    assert result.status is not Status.FAILED_HARNESS
    assert result.manifest["harness"]["declarations_sha256"] == recorded


def test_declarations_that_match_the_record_are_measured_and_recorded(tmp_path, write_split):
    decls = tmp_path / "declarations.json"
    decls.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d"}}]',
        encoding="utf-8",
    )
    recorded = read_declarations(decls)[1]
    reference = _checkpoint(
        tmp_path, {"prompt_mode": "runtime_rendered", "declarations_sha256": recorded}
    )

    result, candidate, _ = _verify_with(tmp_path, write_split, reference, decls)

    assert result.status is not Status.FAILED_HARNESS
    assert result.manifest["harness"]["declarations_sha256"] == recorded
    assert candidate.prompts_seen != []


def test_a_checkpoint_that_recorded_no_declarations_is_not_a_disagreement(tmp_path, write_split):
    """`None` is the absence of something to disagree with, not a mismatch. Every
    checkpoint trained before declarations were an input records nothing here,
    and refusing those would refuse every run that predates this change."""
    decls = tmp_path / "declarations.json"
    decls.write_text(
        '[{"type": "function", "function": {"name": "send_email", "description": "d"}}]',
        encoding="utf-8",
    )
    reference = _checkpoint(tmp_path, {"prompt_mode": "prerendered"})

    result, candidate, _ = _verify_with(tmp_path, write_split, reference, decls)

    assert result.status is not Status.FAILED_HARNESS
    assert candidate.prompts_seen != []
