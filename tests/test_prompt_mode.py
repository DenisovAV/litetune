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

from conftest import FakeBackend, correct_texts, labelled_rows

from litetune import envs
from litetune.bundle import Contract
from litetune.evaluate import (
    HuggingFaceBackend,
    LiteRtLmBackend,
    PromptMode,
    marker_share,
    resolve_prompt_mode,
)
from litetune.verify import BackendPair, Status, VerifyRequest, build_backends, run_verify

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
