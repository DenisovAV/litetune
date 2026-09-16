"""Which prompt mode a model is trained and measured in, decided once and read back.

`--no-template` is narrow, and it was carried around as though it were
general. Its own help says "the input should include all control tokens for
the model expected", and what it actually does is route the runtime to
`create_session()` instead of `create_conversation()`, bypassing the chat
template, the `<|turn>model` anchor, tool handling and channel extraction. It
is correct when the caller built the whole prompt including control tokens --
the FunctionGemma case, where training used a hand-rendered wire format -- and
wrong by default.

The mode is not a property of the model. The same FunctionGemma trained
through `apply_chat_template` would need the opposite flag, so it cannot be
looked up by family (see `litetune.models`). It is decided by how the prompt
was built at training time:

    hand-rendered wire format with control tokens -> --no-template
    apply_chat_template / native runtime tools     -> no flag

`tune` records that decision beside the checkpoint, `bundle.Contract.prompt_mode`
carries it, and `verify` reads it back from either. Only when there is neither
-- a foreign artifact, which is this tool's primary entry point -- is it
inferred, and then the inference and its evidence are reported so a user can
contradict them.

This module holds what every stage shares; each stage keeps its own policy.
`tune.decide_prompt_mode` infers the mode from the training prompts and refuses
only what it cannot read -- a split that mixes the two conventions, or a declared
mode the prompts contradict without `--force-prompt-mode`. `verify` measures
under `resolve_prompt_mode`, and `bundle` takes the mode the training run
recorded. How a prompt becomes tokenizer input is `RENDERING_SOURCE`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PromptMode(str, Enum):
    """How the text that reached the model was constructed."""

    # The prompt is used verbatim: the application rendered any declarations
    # into it and the runtime's own template is disabled (`--no-template`).
    PRERENDERED = "prerendered"
    # The runtime applies its own chat template and renders declarations.
    RUNTIME_RENDERED = "runtime_rendered"


# Control tokens that only appear in a prompt somebody already rendered. Bare
# user text does not contain them.
TURN_MARKERS = (
    "<start_of_turn>",
    "<|turn>",
    "<|im_start|>",
    "<|start_header_id|>",
    "<start_of_image>",
)

# At or above this share of prompts carrying a marker the split is pre-rendered;
# at or below `_MARKER_BARE_SHARE` it is bare. In between the split is
# inconsistent, which is reported rather than smoothed over. Two constants rather
# than `1.0 - _MARKER_CONFIDENT_SHARE`, which is 0.09999999999999998: a split with
# exactly one rendered prompt in ten fell into the inconsistent band.
_MARKER_CONFIDENT_SHARE = 0.9
_MARKER_BARE_SHARE = 0.1


@dataclass(frozen=True)
class PromptModeDecision:
    """Which mode a measurement runs in, where that came from, and on what evidence.

    `source` is `checkpoint` (`tune` recorded it beside the reference),
    `declared` (the caller said so), `contract` (the bundle that shipped the
    model said so) or `inferred` (none of those existed and the prompts were
    inspected). Only the last one is a guess, and it says so in every
    report it reaches. `tune` adds `overridden`: a declared mode its training
    prompts contradict, kept on purpose. `markers` names the control tokens the
    prompts carried.
    """

    mode: PromptMode
    source: str
    evidence: str
    marker_share: float | None = None
    ambiguous: bool = False
    markers: tuple[str, ...] = ()

    @property
    def inferred(self) -> bool:
        return self.source == "inferred"

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_mode": self.mode.value,
            "source": self.source,
            "evidence": self.evidence,
            "marker_share": self.marker_share,
            "ambiguous": self.ambiguous,
            "markers": list(self.markers),
        }


def marker_share(prompts: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    """Share of prompts already carrying a control token, and which ones were seen."""
    if not prompts:
        return 0.0, ()
    seen: list[str] = []
    hits = 0
    for prompt in prompts:
        found = [marker for marker in TURN_MARKERS if marker in prompt]
        if found:
            hits += 1
            seen.extend(m for m in found if m not in seen)
    return hits / len(prompts), tuple(seen)


@dataclass(frozen=True)
class PromptEvidence:
    """What a set of prompts says about how they were built, before anyone decides.

    `mode` is `None` when the prompts disagree with each other. `verify` then
    measures them verbatim and says so; `tune` refuses to train on them unless a
    mode is declared.
    """

    share: float
    markers: tuple[str, ...]
    count: int

    @property
    def mode(self) -> PromptMode | None:
        if self.share >= _MARKER_CONFIDENT_SHARE:
            return PromptMode.PRERENDERED
        if self.share <= _MARKER_BARE_SHARE:
            return PromptMode.RUNTIME_RENDERED
        return None


def prompt_evidence(prompts: Sequence[str]) -> PromptEvidence:
    """The one classification `verify` and `tune` both use, so they cannot drift apart."""
    share, seen = marker_share(prompts)
    return PromptEvidence(share=share, markers=seen, count=len(prompts))


class PromptModeConflict(ValueError):
    """A declared mode or a contract disagrees with the mode the checkpoint trained under."""


def resolve_prompt_mode(
    prompts: Sequence[str],
    declared: PromptMode | None = None,
    contract: PromptMode | None = None,
    recorded: PromptMode | None = None,
) -> PromptModeDecision:
    """Decide the mode this measurement runs in. The training record always wins.

    `recorded` is the mode `tune` wrote beside the reference checkpoint. It is
    what the weights learned, so a declared mode or a contract that disagrees
    with it raises `PromptModeConflict` rather than measure the checkpoint on a
    prompt it never saw. Without a record, precedence is declared, then the
    contract the artifact shipped with, then the prompts themselves. The last is
    a heuristic and is labelled as one: a prompt that already contains
    `<start_of_turn>` was rendered by whoever wrote the split, and templating it
    again double-wraps it.
    """
    if recorded is not None:
        for name, other in (("the declared mode", declared), ("the bundle contract", contract)):
            if other is not None and other is not recorded:
                raise PromptModeConflict(
                    f"{name} is {other.value}, but the reference checkpoint records that it was "
                    f"trained {recorded.value}. Measuring it {other.value} scores it on a prompt "
                    "it never learned; leave out the value that disagrees, or check that "
                    "--reference and --contract belong to the same run"
                )
        return PromptModeDecision(
            mode=recorded,
            source="checkpoint",
            evidence=(
                "read from the litetune.json `tune` wrote beside the reference checkpoint, which "
                "records the mode it trained under"
            ),
        )
    if declared is not None:
        return PromptModeDecision(
            mode=declared,
            source="declared",
            evidence="the caller declared this mode; no inference was made",
        )
    if contract is not None:
        return PromptModeDecision(
            mode=contract,
            source="contract",
            evidence=(
                "read from the bundle contract that shipped with this model, which records the "
                "convention the checkpoint was trained for"
            ),
        )

    evidence = prompt_evidence(prompts)
    share, seen = evidence.share, evidence.markers
    found = ", ".join(seen) if seen else "none"
    if evidence.mode is PromptMode.PRERENDERED:
        return PromptModeDecision(
            mode=PromptMode.PRERENDERED,
            source="inferred",
            evidence=(
                f"{share:.0%} of the held-out prompts already contain control tokens ({found}), so "
                "they were rendered before they reached this tool; applying a chat template to "
                "them would double-wrap them"
            ),
            marker_share=share,
            markers=seen,
        )
    if evidence.mode is PromptMode.RUNTIME_RENDERED:
        return PromptModeDecision(
            mode=PromptMode.RUNTIME_RENDERED,
            source="inferred",
            evidence=(
                f"{share:.0%} of the held-out prompts contain control tokens, so they are bare "
                "text and the runtime has to render its own template around them"
            ),
            marker_share=share,
            markers=seen,
        )
    # Neither shape. Something has to run, so the prompts are used as they are --
    # the option that transforms nothing and leaves the evidence in the record --
    # and the inconsistency is stated rather than hidden.
    return PromptModeDecision(
        mode=PromptMode.PRERENDERED,
        source="inferred",
        evidence=(
            f"{share:.0%} of the held-out prompts contain control tokens ({found}) and the rest do "
            "not: the split mixes the two conventions, so no single mode is right for all of it. "
            "The prompts were used verbatim; declare the mode explicitly to remove the guess"
        ),
        marker_share=share,
        ambiguous=True,
        markers=seen,
    )


def parse_prompt_mode(raw: object, where: str) -> PromptMode:
    """A prompt mode read back from a record. `where` names the record in the error.

    Raises `ValueError` for a value that is not a mode. Each caller turns that
    into its own refusal, and none falls back: a mode a record wrote down is
    never replaced by a default or an inference.
    """
    try:
        return PromptMode(raw)
    except ValueError:
        raise ValueError(
            f"{where} records prompt_mode {raw!r}, which is not a known mode; expected one of "
            f"{[mode.value for mode in PromptMode]}"
        ) from None


# The one function that turns a prompt into the text a tokenizer receives, in
# every script that tokenizes one: training (`tune`), the float reference's
# generation (`evaluate`) and the reference side of the rendering check
# (`rendering`). Pasted into each as source. Training learns from this text and
# the rendering check compares the reference's ids with the runtime's, so a
# hand-written copy in any of them agrees with the others only until one of
# them is edited.
RENDERING_SOURCE = r'''
def render_prompt(tok, prompt, runtime_rendered):
    """The text the tokenizer receives, and whether it may add special tokens.

    Two mutually exclusive conventions, and a model learns whichever one it was
    trained against. prerendered: the prompt is used verbatim and the tokenizer
    supplies the BOS. runtime_rendered: the chat template renders the turn the
    way the serving runtime will and has already emitted every special token the
    turn needs, BOS included, so the tokenizer must not add them again:
    transformers' chat-templating guide says to pass add_special_tokens=False
    after apply_chat_template(tokenize=False). Without it google/gemma-3-270m-it
    gave 15 ids with two leading <bos> where the template renders 14; in
    training the second BOS shifts every position by one, invisibly in the loss.
    """
    if not runtime_rendered:
        return prompt, True
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return text, False
'''
