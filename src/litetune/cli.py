"""The `litetune` command line.

Every subcommand runs standalone. `verify` is the one the product is built on --
a `.litertlm` file, a float reference and a held-out split are enough, and
nothing about a job spec, a run directory or a previous stage is assumed, so it
works on artifacts litetune did not produce -- but `convert`, `tune`, `prepare`
and `bundle` are the same shape: arguments mirror the stage's request dataclass,
the stage is called, and its own result is printed.

Progress goes to stderr as events; the result owns stdout. The exit code
distinguishes "the model failed" from "the run could not tell you", because a
release gate that collapses the two is the failure this whole tool exists to
prevent -- see `verify.EXIT_CODES`, which every subcommand here reuses rather
than inventing a second vocabulary.

A request litetune refuses -- a forbidden export flag, a sweep with no recipes,
a bundle with no prompt-rendering mode -- exits with the "could not check" code
and prints the reason. Refusing is not a verdict about a model, so it must not
share an exit code with one.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

from litetune import envs, models
from litetune._version import __version__
from litetune.bundle import (
    BundleError,
    BundleRequest,
    BundleResult,
    Contract,
    WireConvention,
    build_bundle,
    versions_from,
)
from litetune.checks import Outcome
from litetune.envs import cached_environments, env_cache_root, remove_cached
from litetune.evaluate import GREEDY, DataError, PromptMode
from litetune.events import EventStream, TerminalRenderer
from litetune.export import (
    GPU_ACTIVATION,
    MEASURED_RECIPES,
    ExportRequest,
    ExportResult,
    NoRecipesRequested,
    RecipeExport,
    SizeComparison,
    run_export,
)
from litetune.manifest import RunStatus
from litetune.metrics import SCORERS
from litetune.prepare import (
    HuggingFaceTokenCounter,
    LengthStats,
    PrepareError,
    PrepareRequest,
    PrepareResult,
    prepare,
)
from litetune.spec import DTYPES, SpecError, mutable_ref_refusal, weak_revision_limitations
from litetune.tune import METHODS, TuneError, TuneRequest, TuneResult, run_tune, write_report
from litetune.verify import EXIT_CODES, ReferenceRole, Status, VerifyRequest, run_verify

logger = logging.getLogger(__name__)

# The stage modules answer in `checks.Outcome`, `verify` answers in
# `verify.Status`, and the shell only understands integers. These are the same
# integers `verify` uses: a stage that failed exits like a verification that
# failed, and a stage that could not be performed exits like a verification that
# could not be performed. The Status *names* are not reused -- `failed_gate`
# would be a lie in an export report -- only the numbers.
OUTCOME_EXIT_CODES: dict[Outcome, int] = {
    Outcome.PASSED: EXIT_CODES[Status.PASSED],
    Outcome.FAILED: EXIT_CODES[Status.FAILED_GATE],
    Outcome.UNCHECKED: EXIT_CODES[Status.FAILED_HARNESS],
}


# Requests litetune will not run. Every one of them is a statement about the
# request, never about a model, so they share the "could not check" exit code
# with an unexpected crash rather than the "failed" one.
class BundleInputError(BundleError):
    """A file this command was handed cannot be read as what it claims to be.

    A `BundleError` so it lands in `REFUSALS` and exits `could not check`: a
    malformed manifest says nothing about the model. Before this it raised a
    bare `JSONDecodeError` into `main`'s catch-all and produced "litetune could
    not complete the run", which names neither the file nor the fault.
    """


REFUSALS = (
    models.FlagRefused,
    NoRecipesRequested,
    BundleError,
    TuneError,
    PrepareError,
    DataError,
    SpecError,
    FileNotFoundError,
)

# `verify` can return all five; the stage commands map an outcome through
# `OUTCOME_EXIT_CODES` and can only ever return 0, 1 or 4. Advertising codes a
# command cannot produce is the same overclaim in miniature, so they get
# different text.
EXIT_CODE_HELP = (
    "Exit codes: 0 passed, 1 failed, 2 inconclusive, 3 nothing was established "
    "(no labelled data, or a difference that cannot be attributed), 4 could not check "
    "(including a request litetune refused to run)."
)
STAGE_EXIT_CODE_HELP = (
    "Exit codes: 0 passed, 1 failed, 4 could not check "
    "(including a request litetune refused to run)."
)
# `bundle` carries a verdict it did not produce, so it can return any of the
# five: with no `--status` and no `--verify-manifest` it is `inconclusive` (2),
# because bundling re-measures nothing and must not decide its own verdict.
BUNDLE_EXIT_CODE_HELP = (
    "Exit codes: the status this bundle carries -- 0 passed, 1 failed, "
    "2 inconclusive (the default: nothing was carried in, and bundling decides "
    "nothing), 3 nothing was established, 4 could not check (including a request "
    "litetune refused to run)."
)


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


def _positive(text: str) -> int:
    """An integer argument that must be at least 1.

    `--limit 0` used to evaluate one example -- the slice check ran after the
    append -- and `--max-tokens 0` was silently replaced by the default. Both
    reported the requested value in the manifest, so the record disagreed with
    the run. Refusing at the boundary keeps them equal.
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


class _Parser(argparse.ArgumentParser):
    """An argument parser whose usage errors do not collide with a verdict.

    argparse exits 2 for a malformed command line. For `verify` this tool
    defines 2 as `inconclusive` -- "the interval does not resolve the
    threshold" -- so a typo produced a statement about a measurement that never
    ran. A usage error is a request litetune will not run, which is what 4
    means everywhere else in this vocabulary.
    """

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_CODES[Status.ERROR], f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="litetune",
        description=(
            "Fine-tune a small model, convert it to run on a phone, and know what the "
            "conversion cost you."
        ),
    )
    # A real `--version`, because argparse's prefix matching gave it away.
    # `litetune --version` matched `--verbose` as an abbreviation, so a user
    # checking which version they had silently turned on debug logging and was
    # then told they had not named a command. Declaring the flag both answers
    # the question and removes the collision.
    parser.add_argument(
        "--version",
        action="version",
        version=f"litetune {__version__}",
        help="print the version and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "log at debug level. A refused request prints its reason either way; this also "
            "shows the traceback behind it"
        ),
    )
    sub = parser.add_subparsers(parser_class=_Parser, dest="command", required=True)
    _add_verify(sub)
    _add_prepare(sub)
    _add_tune(sub)
    _add_convert(sub)
    _add_bundle(sub)

    env = sub.add_parser(
        "env",
        help="show or clear the environments the stages provisioned",
        description=(
            "Stage environments are cached under a path nothing else prints, and their "
            "sizes differ by an order of magnitude: the export toolchain is over a "
            "gigabyte, the runtime a tenth of that. A failed provision also leaves a "
            "directory that looks like a working one from the outside."
        ),
    )
    env.add_argument(
        "--clean",
        action="store_true",
        help="remove every cached environment. The next stage that needs one rebuilds it",
    )
    return parser


def _add_verify(sub) -> None:
    verify = sub.add_parser(
        "verify",
        help="measure what conversion cost, on held-out data",
        description=(
            "Evaluate a converted .litertlm model against a float reference on held-out "
            f"data. {EXIT_CODE_HELP}"
        ),
    )
    verify.add_argument("--model", required=True, type=Path, help="path to the .litertlm artifact")
    verify.add_argument(
        "--reference",
        required=True,
        help="HF id or path of the float model to compare against",
    )
    verify.add_argument(
        "--reference-role",
        choices=[role.value for role in ReferenceRole],
        default=ReferenceRole.FLOAT_TWIN.value,
        help=(
            "what the reference is. float_twin (default): the same weights before conversion, "
            "so the difference is the conversion cost. untuned_base: a different checkpoint, so "
            "the difference confounds training with conversion and neither is attributed"
        ),
    )
    verify.add_argument("--data", required=True, type=Path, help="held-out JSONL")
    verify.add_argument(
        "--limit",
        type=_positive,
        help="use only the first N examples",
    )
    verify.add_argument(
        "--scorer",
        choices=sorted(SCORERS),
        default="tool-call",
        help=(
            "what counts as correct. tool-call (default): the parsed call's operation name "
            "and every argument value match the target. exact-text: the generation equals "
            "the target text once whitespace is collapsed — for any task with one right "
            "answer and no structure inside it. Everything after scoring is task-agnostic, "
            "so this is the only flag a different task has to change"
        ),
    )
    verify.add_argument(
        "--max-conversion-cost",
        type=float,
        help=(
            "fail if conversion costs more than this much exact match. A threshold finer than "
            "the split's confidence interval returns inconclusive, not a verdict"
        ),
    )
    verify.add_argument(
        "--prompt-mode",
        choices=[mode.value for mode in PromptMode],
        help=(
            "how the prompt reaching the model is built. prerendered: the prompt already contains "
            "the declarations and every control token, and the runtime is told not to template it "
            "(--no-template). runtime_rendered: the runtime applies its own chat template. "
            "Without this litetune reads --contract, and without that it infers the mode from the "
            "prompts and says so"
        ),
    )
    verify.add_argument(
        "--contract",
        type=Path,
        help=(
            "path to the bundle contract.json this model shipped with; its prompt_mode is the "
            "mode the checkpoint was trained for"
        ),
    )
    verify.add_argument(
        "--max-tokens",
        type=_positive,
        help=(
            "generation limit for the reference side (default 256). litetune passes no "
            "decoding flags to the device side, so this bounds the reference only; the "
            "manifest names that asymmetry and counts how many runtime generations end "
            "without a terminator. The pinned litert-lm does accept --top-k, --top-p, "
            "--temperature and --seed; wiring them through would close the gap"
        ),
    )
    verify.add_argument("--json", action="store_true", help="write the manifest to stdout")


def _add_prepare(sub) -> None:
    prep = sub.add_parser(
        "prepare",
        help="split a dataset and report every reason it might not be measurable",
        description=(
            "Read a raw JSONL dataset, profile it, and write a content-keyed train/held-out "
            f"split. Runs before anyone books a GPU. {STAGE_EXIT_CODE_HELP}"
        ),
    )
    prep.add_argument("--data", required=True, type=Path, help="raw JSONL dataset")
    prep.add_argument("--output-dir", required=True, type=Path, help="where the splits are written")
    prep.add_argument(
        "--context-length",
        required=True,
        type=int,
        help=(
            "the binding context window: pass the smaller of export.context_length and "
            "train.max_seq_length. Rows that do not fit are named, never truncated"
        ),
    )
    prep.add_argument("--seed", type=int, default=0, help="split seed (with the file's content)")
    prep.add_argument("--heldout-fraction", type=float, default=0.2)
    prep.add_argument("--heldout-size", type=int, help="exact number of held-out rows")
    prep.add_argument(
        "--min-heldout-examples",
        type=int,
        default=PrepareRequest.min_heldout_examples,
        help="below this the split is flagged as too small to read differences off",
    )
    prep.add_argument(
        "--tokenizer",
        help=(
            "HF id whose tokenizer measures example lengths. Without it no example is measured "
            "against the context window and that check reports could_not_check"
        ),
    )
    prep.add_argument("--tokenizer-revision", help="revision of --tokenizer")
    prep.add_argument("--json", action="store_true", help="write the report to stdout")


def _add_tune(sub) -> None:
    tune = sub.add_parser(
        "tune",
        help="supervised fine-tuning, with the loss on the completion only",
        description=(
            "Fine-tune inside the pinned training environment. Training completing is not "
            f"evidence: a run that scored nine times worse than its base finished cleanly with a "
            f"lower loss. {STAGE_EXIT_CODE_HELP}"
        ),
    )
    tune.add_argument("--model", required=True, help="HF id or path of the base checkpoint")
    tune.add_argument("--data", required=True, type=Path, help="training JSONL")
    tune.add_argument("--output-dir", required=True, type=Path)
    tune.add_argument("--method", choices=list(METHODS), default="full")
    tune.add_argument("--revision", help="pin the base checkpoint's revision")
    tune.add_argument(
        "--prompt-mode",
        required=True,
        choices=[mode.value for mode in PromptMode],
        help=(
            "how this run builds its prompts, which is the convention the checkpoint will expect "
            "for the rest of its life. prerendered: the prompt carries the declarations and every "
            "control token, and the serving runtime must not template it. runtime_rendered: the "
            "prompt is bare text and the runtime applies its own chat template. Required, because "
            "the two are mutually exclusive and calling a model in the wrong one produces a "
            "fluent wrong answer rather than an error"
        ),
    )
    tune.add_argument(
        "--learning-rate",
        type=float,
        help="default is per method: 1e-5 for full, 2e-4 for lora",
    )
    tune.add_argument("--epochs", type=float, default=1.0)
    tune.add_argument("--batch-size", type=int, default=8)
    tune.add_argument("--max-seq-length", type=int, default=1024)
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--lora-rank", type=int, default=16)
    tune.add_argument("--lora-alpha", type=int, default=32)
    tune.add_argument("--lora-dropout", type=float, default=0.05)
    # `choices`, not just a default: spec.py:87 removed float16 from `DTYPES`
    # because a 270M model's loss goes to NaN in it while bfloat16 holds. The
    # spec file was closed and this line was not, so the path everyone actually
    # uses stayed open to the one value the design refuses.
    tune.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    tune.add_argument("--attn-implementation", default="eager")
    tune.add_argument("--timeout-s", type=int, default=TuneRequest.timeout_s)
    tune.add_argument(
        "--no-provision",
        action="store_true",
        help=(
            "do not build the pinned training environment; use it as it is. Without it the first "
            "run installs torch and transformers, which takes minutes and needs the network"
        ),
    )
    tune.add_argument("--json", action="store_true", help="write the report to stdout")


def _add_convert(sub) -> None:
    convert = sub.add_parser(
        "convert",
        help="checkpoint to .litertlm, one artifact per candidate recipe",
        description=(
            "Sweep quantization recipes and report what each one produced. A produced file is "
            "not a verified model: run `litetune verify` on each artifact against held-out data. "
            "Each bundle is then repacked with prefer_activation_type=fp32 on its prefill/decode "
            "section: without it the Android GPU backend computes in F16 and floods <pad> while "
            "reporting success (measured on one Snapdragon Galaxy S24, 3/20 vs 20/20 tool names "
            "on 20 rows). The repack changes one metadata string and no bytes of the model; a "
            "bundle that could not be repacked is kept, named in the report, and is CPU-only. "
            "--json records what each bundle carries as exports[].gpu_activation. "
            f"{STAGE_EXIT_CODE_HELP}"
        ),
    )
    convert.add_argument("--model", required=True, help="HF id or path of the checkpoint")
    convert.add_argument("--output-dir", required=True, type=Path, help="one directory per recipe")
    convert.add_argument(
        "--train-metrics",
        type=Path,
        help=(
            "the `metrics.json` a `tune` run wrote. `--model` is a directory after training, "
            "and a directory carries no identity — so without this the per-family export "
            "flags have nothing to key on and are silently not applied, which is how a "
            "FunctionGemma bundle ends up typed generic_model with no tool-call channel. "
            "Pass `--base-model` instead if you did not train here"
        ),
    )
    convert.add_argument(
        "--base-model",
        help=(
            "what this checkpoint was fine-tuned from, when `--model` is a local directory "
            "and there is no `--train-metrics` to read it out of"
        ),
    )
    convert.add_argument(
        "--recipe",
        action="append",
        metavar="NAME",
        help=(
            "a quantization recipe to sweep; repeat it. There is no default: the toolchain's own "
            f"choice cost 0.024 exact match. A defensible minimum is {list(MEASURED_RECIPES)}"
        ),
    )
    convert.add_argument(
        "--externalize-embedder",
        action=argparse.BooleanOptionalAction,
        # `None`, not `True`: the default belongs to `ExportRequest`, and a
        # `True` here reached `plan_export` looking exactly like a flag the user
        # had typed -- so `report.json` credited them with it and the "added by
        # litetune, required by gemma-4" line was never printed.
        default=None,
        help=(
            "write the tied embedding as its own section (default: on). Required for Gemma 4, "
            "and every artifact this project has measured was exported with it -- on "
            "Gemma-3-270M the embedding is over 60%% of the parameters, so the same model is "
            "286 MB without it and 457 MB with it, and two exports differing in it cannot be "
            "compared on .litertlm size at all. Defaulting it off meant a plain `convert` "
            "produced an artifact structurally unlike every size in the catalogue. "
            "`--no-externalize-embedder` turns it off"
        ),
    )
    convert.add_argument(
        "--flag",
        action="append",
        metavar="FLAG",
        default=[],
        help=(
            "an extra flag passed to the exporter verbatim; repeat it. Write it attached, "
            "--flag=--some_exporter_flag=value, so argparse does not read it as one of ours. "
            "Flags this model family requires are added automatically and named in the report. "
            "--flag=--experimental_use_mixed_precision pre-empts the fp32 repack: it writes "
            "prefer_activation_type=fp32_fp16 and also runs a graph pass, so the bundle is no "
            "longer the one the CPU figures describe; the report records it as fp32_fp16"
        ),
    )
    convert.add_argument("--timeout-s", type=int, default=ExportRequest.timeout_s)
    convert.add_argument(
        "--no-provision",
        action="store_true",
        help=(
            "do not build the pinned export environment; use it as it is. Without it the first "
            "run installs the toolchain, which takes minutes and needs the network"
        ),
    )
    convert.add_argument("--json", action="store_true", help="write the report to stdout")


def _add_bundle(sub) -> None:
    bundle = sub.add_parser(
        "bundle",
        help="assemble the deliverable: model + declarations + contract + report",
        description=(
            "A model file on its own is not a deliverable. This writes the model, the tool "
            "declarations, the calling contract and the report into one directory. "
            f"{BUNDLE_EXIT_CODE_HELP}"
        ),
    )
    bundle.add_argument("--output-dir", required=True, type=Path)
    bundle.add_argument(
        "--model", required=True, type=Path, help=".litertlm file or checkpoint dir"
    )
    bundle.add_argument("--declarations", required=True, type=Path, help="tool declarations JSON")
    bundle.add_argument(
        "--prompt-mode",
        required=True,
        choices=[mode.value for mode in PromptMode],
        help=(
            "how this model must be called. Required and never defaulted: a runtime cannot infer "
            "it, both conventions are in the field, and the wrong one is a fluent wrong answer"
        ),
    )
    bundle.add_argument("--base-model", required=True, help="the checkpoint this started from")
    bundle.add_argument(
        "--base-model-revision",
        required=True,
        help="its revision; a bundle whose starting weights are unrecorded cannot be reproduced",
    )
    bundle.add_argument(
        "--wire-convention",
        choices=[w.value for w in WireConvention],
        help=(
            "the order this model's tool declarations were rendered in. Two conventions are in "
            "the field for the same model -- flutter_gemma renders properties in declaration "
            "order, the jinja template inside the .litertlm sorts them -- and they disagree for "
            "every declaration with more than one property. Measured twice on the base "
            "checkpoint, choosing wrong costs a resolved 0.019-0.036 exact match, and the "
            "failure is a wrong argument value rather than a reordered one. Left unset the "
            "bundle records that it does not know, which is the honest answer and not a default"
        ),
    )
    bundle.add_argument("--context-length", type=int)
    bundle.add_argument(
        "--adapter",
        type=Path,
        help=(
            "the LoRA weights this run produced (`<tune output>/adapter`). "
            "`merge_and_unload()` is one-way, so an adapter left outside the bundle is "
            "gone the first time the training directory is cleaned up"
        ),
    )
    bundle.add_argument("--stop-token", action="append", default=[], metavar="TOKEN")
    bundle.add_argument(
        "--train-metrics",
        type=Path,
        help=(
            "training metrics JSON. Its recorded turn terminator becomes the contract's "
            "stop token unless --stop-token says otherwise, so the bundle declares what the "
            "model was trained to emit rather than what someone remembered to type"
        ),
    )
    bundle.add_argument("--note", action="append", default=[], help="a note for the contract")
    bundle.add_argument(
        "--status",
        choices=[status.value for status in RunStatus],
        help=(
            "the run's status, carried in rather than decided here: bundling re-measures nothing. "
            "Read from --verify-manifest when one is given, otherwise inconclusive"
        ),
    )
    bundle.add_argument(
        "--verify-manifest",
        type=Path,
        help=(
            "a `litetune verify --json` manifest whose measurements and status "
            "this bundle carries"
        ),
    )
    bundle.add_argument("--json", action="store_true", help="write the report to stdout")


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _stream() -> EventStream:
    """Progress on stderr, rendered for a person. Nothing here prints directly."""
    events = EventStream(stream=sys.stderr, echo_json=False)
    events.subscribe(TerminalRenderer(sys.stderr))
    return events


# Set when stdout's reader closed the pipe. `main` acts on it; nothing else does.
_BROKEN_PIPE = False


def _write(payload: dict[str, Any], lines: Sequence[str], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    for line in lines:
        print(line)


def _report(payload: dict[str, Any], render: Callable[[], Sequence[str]], as_json: bool) -> bool:
    """Write the result. Never raises; returns whether it was delivered.

    Three faults used to reach `main`'s catch-all and come back as exit 4 --
    "could not check" -- for runs that had completed:

    * a `KeyError` in a summariser, because the summary was evaluated as an
      argument to `_write` and so ran before the exit code was returned;
    * `litetune verify --json | head`, which closes the pipe;
    * any write failure at all.

    None says anything about the model. The distinctions that do matter:

    * a **closed pipe** is the reader's choice (`| head`), so it is absorbed and
      the verdict stands;
    * a **failed write** with `--json` loses the deliverable -- the manifest
      *is* the output there -- so the caller is told, and exits accordingly.

    The first version flushed nothing, so `print`'s buffer meant the pipe was
    only touched at interpreter exit, outside the guard: the fix never fired for
    a payload under a pipe buffer, which is every real manifest. `flush()` here
    is what makes the guard reachable.
    """
    try:
        lines = render()
    except Exception:  # noqa: BLE001 - a rendering fault must not restate the verdict
        logger.exception("could not render the summary; the payload is unaffected")
        lines = [
            f"status: {payload.get('status') or payload.get('outcome') or 'unknown'}",
            "  (summary unavailable)",
        ]
    try:
        _write(payload, lines, as_json)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader went away on purpose. Record it so `main` can silence the
        # interpreter's own exit-time flush -- that belongs at process scope,
        # not in a helper a library caller may have imported.
        global _BROKEN_PIPE
        _BROKEN_PIPE = True
        return True
    except OSError:
        logger.exception("could not write the result to stdout")
        # Only `--json` loses something by this. The human summary is a
        # rendering of a verdict the exit code already carries; the manifest is
        # the deliverable, and a run whose manifest never landed has not been
        # reported. The docstring claimed this distinction before the code made
        # it, which is the same overclaim this tool exists to catch.
        return not as_json
    return True


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _verify(args: argparse.Namespace) -> int:
    events = _stream()
    request = VerifyRequest(
        model=args.model,
        reference=args.reference,
        data=args.data,
        limit=args.limit,
        reference_role=ReferenceRole(args.reference_role),
        scorer=args.scorer,
        max_conversion_cost=args.max_conversion_cost,
        prompt_mode=PromptMode(args.prompt_mode) if args.prompt_mode else None,
        contract=args.contract,
        # `is not None`, not truthiness: `--max-tokens 0` is a request this
        # cannot honour, and silently substituting the default would report a
        # limit the run did not use.
        decode=(
            replace(GREEDY, max_tokens=args.max_tokens) if args.max_tokens is not None else GREEDY
        ),
    )
    result = run_verify(request, events=events)
    # The exit code is the verdict; the summary is a rendering of it. Computing
    # the summary as an argument to `_write` put a KeyError in a formatter on
    # the path to `main`'s catch-all, where it became exit 4 -- "could not
    # check" -- for a run that had passed. Those two must never be confused,
    # which is the whole subject of this tool.
    code = result.exit_code
    delivered = _report(result.manifest, lambda: summarise(result.manifest), args.json)
    return code if delivered else EXIT_CODES[Status.ERROR]


def _mapping(value: object) -> dict:
    """A mapping, whatever arrived.

    `summarise` reads a document it did not build -- a manifest from an older
    version, a hand-edited file, a different tool. A list where an object was
    expected used to raise `AttributeError` through to `main`'s catch-all and
    come back as exit 4.
    """
    return value if isinstance(value, dict) else {}


def _num(value: object, *, signed: bool = False) -> str:
    """Format a number that a manifest may not carry.

    Total by construction: `summarise` reads a document it did not build, and a
    missing key here used to raise through to `main`'s catch-all and change the
    run's exit code.
    """
    if not isinstance(value, int | float) or isinstance(value, bool):
        return "?"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def summarise(manifest: dict) -> list[str]:
    """The human-readable result. Every number printed carries n and its interval."""
    status = manifest.get("status")
    lines = [f"status: {status}"]

    quality = _mapping(manifest.get("quality"))
    if quality.get("available"):
        for side in ("candidate", "reference"):
            block = _mapping(quality.get(side))
            exact = _mapping(block.get("exact_match"))
            lines.append(
                f"  {side:<9}  exact match {_num(exact.get('value'))} "
                f"±{_num(exact.get('ci95'))} (n={exact.get('n', '?')})"
            )
            name = _mapping(block.get("name_accuracy"))
            args_ = _mapping(block.get("argument_accuracy"))
            arg_text = (
                f"{_num(args_.get('value'))} ±{_num(args_.get('ci95'))}"
                if args_.get("available")
                else f"unavailable — {args_.get('reason')}"
            )
            lines.append(f"  {'':<9}  operation {_num(name.get('value'))}, arguments {arg_text}")
    else:
        lines.append(f"  quality: not measured — {quality.get('reason')}")

    for name, value in _mapping(manifest.get("attribution")).items():
        if not isinstance(value, dict):
            continue
        if value.get("available"):
            resolved = "" if value.get("resolved") else "  (unresolved at this sample size)"
            lines.append(
                f"  {name}: {_num(value.get('value'), signed=True)} "
                f"±{_num(value.get('ci95'))}{resolved}"
            )
        else:
            lines.append(f"  {name}: unavailable — {value.get('reason')}")

    decision = _mapping(_mapping(manifest.get("harness")).get("prompt_mode_decision"))
    if decision:
        lines.append(
            f"  prompt mode: {decision.get('prompt_mode', '?')} ({decision.get('source', '?')})"
        )

    for limitation in manifest.get("limitations", []):
        lines.append(f"  note: {limitation}")
    if status != Status.PASSED.value:
        lines.append("  this model is not verified; see the manifest for what was and was not run")
    return lines


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def _prepare(args: argparse.Namespace) -> int:
    events = _stream()
    counter = (
        HuggingFaceTokenCounter(model=args.tokenizer, revision=args.tokenizer_revision)
        if args.tokenizer
        else None
    )
    request = PrepareRequest(
        data=args.data,
        output_dir=args.output_dir,
        context_length=args.context_length,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
        heldout_size=args.heldout_size,
        min_heldout_examples=args.min_heldout_examples,
        tokens=counter,
    )
    result = prepare(request, events=events)
    delivered = _report(result.as_dict(), lambda: summarise_prepare(result), args.json)
    if not delivered:
        # The report is the deliverable when `--json` is on, and a run whose
        # result could not be written has not been reported.
        return EXIT_CODES[Status.ERROR]
    return OUTCOME_EXIT_CODES[result.outcome]


def summarise_prepare(result: PrepareResult) -> list[str]:
    lines = [f"outcome: {result.outcome.value}"]
    if result.train is not None and result.heldout is not None:
        lines.append(
            f"  {result.n_rows} rows → {result.train.n} train, {result.heldout.n} held out "
            f"(content {result.content_sha256.split(':')[-1][:16]}, seed {result.request.seed})"
        )
        lines.append(f"  {result.train.path}")
        lines.append(f"  {result.heldout.path}")
    else:
        lines.append(f"  {result.n_rows} rows read; no split was written")
    if isinstance(result.lengths, LengthStats):
        lines.append(
            f"  longest example {result.lengths.total['max']} tokens against a "
            f"{result.lengths.context_length}-token window; expected supervised fraction "
            f"{result.lengths.expected_supervised_fraction:.4f}"
        )
    else:
        lines.append(f"  token lengths: not measured — {result.lengths.reason}")
    for profile in result.unscoreable:
        lines.append(f"  unscoreable: {profile.tool}.{profile.argument}")
    for skipped in result.skipped:
        lines.append(f"  skipped {skipped.name}: {skipped.reason}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    return lines


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------


def _tune(args: argparse.Namespace) -> int:
    events = _stream()
    request = TuneRequest(
        model=args.model,
        data=args.data,
        output_dir=args.output_dir,
        method=args.method,
        revision=args.revision,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        prompt_mode=PromptMode(args.prompt_mode),
        timeout_s=args.timeout_s,
        auto_provision=not args.no_provision,
    )
    result = run_tune(request, events=events)
    report = write_report(result)
    events.artifact(str(report), name="tune.json")
    delivered = _report(result.as_dict(), lambda: summarise_tune(result), args.json)
    if not delivered:
        # The report is the deliverable when `--json` is on, and a run whose
        # result could not be written has not been reported.
        return EXIT_CODES[Status.ERROR]
    return OUTCOME_EXIT_CODES[result.outcome]


def summarise_tune(result: TuneResult) -> list[str]:
    request = result.request
    source = "default for the method" if request.rate_is_default else "declared"
    lines = [
        f"outcome: {result.outcome.value}",
        f"  {request.method} fine-tune at learning rate {request.rate:g} ({source}), "
        f"prompt mode {result.prompt_mode.value}",
    ]
    metrics = result.metrics
    if metrics is not None:
        fraction = metrics.supervised_token_fraction
        if fraction is not None:
            lines.append(
                f"  loss computed on {fraction:.4f} of tokens "
                f"({metrics.supervised_tokens} of {metrics.total_tokens})"
            )
        if metrics.final_loss is not None:
            lines.append(
                f"  final loss {metrics.final_loss:.4f} over {len(metrics.epochs)} epoch(s)"
            )
    if result.model_dir is not None:
        lines.append(f"  checkpoint {result.model_dir}")
    if result.adapter_dir is not None:
        lines.append(f"  adapter {result.adapter_dir}")
    for check in result.checks.checks:
        if check.outcome is not Outcome.PASSED:
            lines.append(f"  {check.outcome.value}: {check.name} — {check.detail}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    return lines


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


def _human(size: int) -> str:
    scaled = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if scaled < 1024 or unit == "GB":
            return f"{scaled:.0f} {unit}" if unit == "B" else f"{scaled:.1f} {unit}"
        scaled /= 1024.0
    return f"{scaled:.1f} GB"


def _env(args: argparse.Namespace) -> int:
    entries = cached_environments()
    if not entries:
        print(f"no cached environments under {env_cache_root()}")
        return EXIT_CODES[Status.PASSED]

    total = sum(e.bytes for e in entries)
    if not args.clean:
        print(f"{env_cache_root()}  —  {_human(total)} in {len(entries)} environment(s)")
        for entry in entries:
            # "Not for this interpreter" rather than "unused": the identity is a
            # hash of the pins *and* the running Python, so an environment built
            # by another version looks foreign from here and is not junk. Saying
            # "unclaimed" would invite deleting the ones that work.
            belongs = (
                entry.stage or f"not for python{sys.version_info.major}.{sys.version_info.minor}"
            )
            state = "ready" if entry.ready else "incomplete"
            print(f"  {_human(entry.bytes):>9}  {state:<10} {belongs:<28} {entry.path.name}")
        print("\nRe-run with --clean to remove them; the next stage that needs one rebuilds it.")
        return EXIT_CODES[Status.PASSED]

    freed, failures = remove_cached(entries)
    print(f"removed {len(entries) - len(failures)} environment(s), freed {_human(freed)}")
    for failure in failures:
        print(f"  could not remove {failure}")
    return EXIT_CODES[Status.PASSED] if not failures else EXIT_CODES[Status.ERROR]


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def _read_base_model(metrics: Path | None) -> str | None:
    """The model a `tune` run started from, out of the file it wrote.

    `convert` needs the name because a trained checkpoint is a directory, and
    the per-family export flags key on the name. Reading it from the run that
    produced the checkpoint beats asking the caller to retype it, which is how
    the two come to disagree.
    """
    if metrics is None:
        return None
    try:
        recorded = json.loads(metrics.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{metrics} is not valid JSON: {exc}") from None
    except OSError as exc:
        raise DataError(f"could not read {metrics}: {exc}") from None
    if not isinstance(recorded, dict):
        raise DataError(f"{metrics} holds {type(recorded).__name__}, not training metrics")
    base = recorded.get("base_model")
    if base is None:
        # The caller typed --train-metrics; answering None sends them back to
        # guessing from the path, and the refusal they would then get tells them
        # to pass the flag they just passed.
        raise DataError(
            f"{metrics} records no 'base_model'. Runs from before litetune 0.1.3 did not "
            "write one; pass --base-model <id> instead"
        )
    if not isinstance(base, str) or not base.strip():
        raise DataError(f"{metrics} records a 'base_model' that is not a name: {base!r}")
    return base


def _convert(args: argparse.Namespace) -> int:
    events = _stream()
    # Explicit beats recorded: a caller who names the base model has a reason,
    # and the recorded one is only ever a convenience for the common case.
    base_model = args.base_model or _read_base_model(args.train_metrics)
    request = ExportRequest(
        model=args.model,
        base_model=base_model,
        output_dir=args.output_dir,
        # Not defaulted here either: `ExportRequest` refuses an empty sweep with
        # the measurement that makes the refusal worth reading.
        recipes=args.recipe or (),
        externalize_embedder=args.externalize_embedder,
        extra_flags=tuple(args.flag),
        timeout_s=args.timeout_s,
        auto_provision=not args.no_provision,
    )
    result = run_export(request, events=events)
    delivered = _report(result.as_dict(), lambda: summarise_convert(result), args.json)
    if not delivered:
        # The report is the deliverable when `--json` is on, and a run whose
        # result could not be written has not been reported.
        return EXIT_CODES[Status.ERROR]
    return OUTCOME_EXIT_CODES[result.outcome]


def _artifact_line(export: RecipeExport) -> str:
    """One produced artifact. Companion files are named, not folded into one number.

    `--externalize_embedder` writes the embedding as its own section *inside* the
    `.litertlm`, not beside it: the same 270M model measures 285,577,392 bytes
    without the flag and 455,759,152 with it, both as one file. So this normally
    prints one number and the companion clause never fires.

    It is kept anyway, and is not dead code: `shipped_bytes` is the sum of
    everything the recipe produced and `artifact_bytes` is the model file alone,
    so a toolchain version that starts emitting a sidecar shows up here as a
    named difference instead of silently making every recorded size wrong.
    """
    beside = (export.shipped_bytes or 0) - (export.artifact_bytes or 0)
    companions = f" (+{beside:,} bytes beside it)" if beside > 0 else ""
    # Named on the line, not only in the JSON: a bundle without it is CPU-only
    # and looks identical to one with it from every other field here.
    if export.gpu_activation is None:
        gpu = ", CPU-only (GPU activations not set)"
    elif export.gpu_activation == GPU_ACTIVATION:
        gpu = f", GPU activations {export.gpu_activation}"
    else:
        # An upstream declaration reads the same shape as the good case unless
        # it is marked: `fp16` is the fault itself.
        gpu = f", GPU activations {export.gpu_activation} (declared upstream, not {GPU_ACTIVATION})"
    return (
        f"  {export.recipe}: {export.artifact_bytes:,} bytes{companions} "
        f"in {export.seconds:.1f}s{gpu} — {export.artifact}"
    )


def summarise_convert(result: ExportResult) -> list[str]:
    plan = result.request.plan
    lines = [f"outcome: {result.outcome.value}"]
    if plan.rules is not None:
        lines.append(f"  model family: {plan.rules.family}")
    for flag in plan.added:
        lines.append(
            f"  added {flag} (required by {plan.rules.family if plan.rules else 'litetune'})"
        )
    for export in result.exports:
        if export.ok and export.artifact is not None:
            lines.append(_artifact_line(export))
        else:
            lines.append(f"  {export.recipe}: {export.check.outcome.value} — {export.check.detail}")
    if result.not_attempted:
        lines.append(f"  not attempted: {', '.join(result.not_attempted)}")
    if isinstance(result.comparison, SizeComparison):
        lines.append(
            f"  size spread {result.comparison.spread_share:.4%} between "
            f"{result.comparison.smallest} and {result.comparison.largest}; accuracy is not "
            "observable at export time"
        )
    for text in result.recommendations:
        lines.append(f"  recommendation: {text}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    if result.artifacts:
        # Never omitted when there is an artifact: a file of the right size that
        # loads without error is the documented shape of a bad conversion.
        lines.append("  produced, not verified: run `litetune verify` on each artifact")
    return lines


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


# What `verify` calls its two measurement points, and what a bundle calls them.
# The reference's name depends on what it was: against the float twin it is the
# tuned model in float, against an untuned base it is the base. Getting this
# wrong would file a measurement under a point it is not.
_REFERENCE_POINT: dict[str, str] = {
    ReferenceRole.FLOAT_TWIN.value: "tuned_float",
    ReferenceRole.UNTUNED_BASE.value: "base_float",
}


def measurements_from_verify(manifest: dict[str, Any]) -> dict[str, Any]:
    """Map a verify manifest's measurement points onto the bundle's names."""
    # `or {}` keeps a non-empty list, and `.get` on a list raises through to
    # `main`'s catch-all as exit 4 -- "could not check" about a model, for a
    # malformed file. The manifest is a document this function did not build.
    measured = _mapping(manifest.get("measurements"))
    role = _mapping(manifest.get("reference")).get("role", ReferenceRole.FLOAT_TWIN.value)
    points: dict[str, Any] = {}
    candidate = measured.get("candidate")
    if isinstance(candidate, dict) and candidate.get("available") is not False:
        points["tuned_converted"] = candidate
    reference = measured.get("reference")
    if isinstance(reference, dict) and reference.get("available") is not False:
        name = _REFERENCE_POINT.get(role)
        if name is not None:
            points[name] = reference
    return points


def _bundle(args: argparse.Namespace) -> int:
    events = _stream()

    # The same refusal a spec file gets. `--base-model-revision main` was
    # accepted here while `base_model.revision: main` was refused there, so the
    # bundle -- the artifact whose whole purpose is to record what it was built
    # from -- was the lenient path.
    refusal = mutable_ref_refusal(args.base_model_revision, "--base-model-revision")
    if refusal:
        raise BundleInputError(refusal)

    manifest: dict[str, Any] = {}
    if args.verify_manifest is not None:
        try:
            manifest = json.loads(args.verify_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BundleInputError(f"{args.verify_manifest} is not valid JSON: {exc}") from None
        except OSError as exc:
            raise BundleInputError(f"could not read {args.verify_manifest}: {exc}") from None
        if not isinstance(manifest, dict):
            raise BundleInputError(
                f"{args.verify_manifest} holds {type(manifest).__name__}, not a manifest object"
            )

    # A status string this version does not know is a fact about the file, not
    # about the model, so it lands in `REFUSALS` and exits "could not check"
    # rather than as an anonymous ValueError from `main`'s catch-all.
    try:
        status = (
            RunStatus(args.status)
            if args.status
            else RunStatus(manifest["status"])
            if manifest.get("status")
            else RunStatus.INCONCLUSIVE
        )
    except ValueError as exc:
        raise BundleInputError(f"unrecognised run status: {exc}") from None
    # A stop token typed from memory and one taken from the run that produced
    # the model are different claims. Training records which terminator it
    # supervised; preferring it here means an empty declaration is a choice
    # rather than an oversight -- and a bundle whose runtime waits for a token
    # the model was never trained to emit does not stop.
    stop_tokens = tuple(args.stop_token)
    terminator_note = None
    if not stop_tokens and args.train_metrics is not None:
        try:
            recorded = json.loads(args.train_metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BundleInputError(f"{args.train_metrics} is not valid JSON: {exc}") from None
        except OSError as exc:
            raise BundleInputError(f"could not read {args.train_metrics}: {exc}") from None
        if not isinstance(recorded, dict):
            raise BundleInputError(
                f"{args.train_metrics} holds {type(recorded).__name__}, not training metrics"
            )
        term = (recorded.get("turn_terminator") or {}).get("text")
        if term:
            stop_tokens = (term,)
            terminator_note = (
                "stop token taken from the training run's recorded terminator "
                f"({(recorded.get('turn_terminator') or {}).get('source')})"
            )

    # Terminators the family needs that the training run could not record. Added
    # after the recorded one and never instead of it: the run observed what the
    # model was trained to emit, and this observes what the serving convention
    # additionally requires it to stop at. Both are named in the notes, because
    # "the bundle declares two stop tokens" and "litetune added one of them" are
    # different claims and the report is where a reader tells them apart.
    #
    # This completes the *contract*, not the artifact: the `.litertlm` already
    # carries the same terminator as a token id. A consumer reading contract.json
    # instead of parsing the bundle's protobuf was the one being short-changed.
    family_stops, stop_reason = models.stop_tokens_for(args.base_model)
    added_stops = tuple(t for t in family_stops if t not in stop_tokens)
    unrecorded_primary = added_stops and not stop_tokens
    if added_stops:
        stop_tokens = stop_tokens + added_stops
        terminator_notes = tuple(n for n in (terminator_note,) if n) + (
            f"litetune added the stop token(s) {', '.join(added_stops)} for "
            f"{args.base_model}: {stop_reason}",
        )
        if unrecorded_primary:
            # The family token supplements the trained terminator; it must not
            # stand in for it. A bundle declaring only "stop where the app takes
            # over" and not "stop at the end of the turn" is worse than one
            # declaring nothing, because it looks specified.
            terminator_notes += (
                "the stop token(s) above are the only ones declared: no training run was "
                "supplied (--train-metrics) and none was named (--stop-token), so the "
                "terminator the model was actually trained to emit is unrecorded. Declaring "
                "only the family's token tells the runtime where the application takes over "
                "and not where the turn ends",
            )
    else:
        terminator_notes = tuple(n for n in (terminator_note,) if n)

    contract = Contract(
        prompt_mode=PromptMode(args.prompt_mode),
        wire_convention=(WireConvention(args.wire_convention) if args.wire_convention else None),
        # The runtime's pins, because which prompt a runtime renders is a
        # property of that runtime's release. `export.resolve_toolchain` reads
        # the resolved closure and is better; a run that produced one should
        # pass it in rather than declaring it here.
        established_against=versions_from(envs.RUNTIME),
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        context_length=args.context_length,
        stop_tokens=stop_tokens,
        notes=tuple(args.note) + terminator_notes,
    )
    request = BundleRequest(
        output_dir=args.output_dir,
        model=args.model,
        adapter=args.adapter,
        declarations=args.declarations,
        contract=contract,
        status=status,
        measurements=measurements_from_verify(manifest),
        attribution=manifest.get("attribution") or {},
        limitations=tuple(manifest.get("limitations") or ())
        + tuple(models.limitations_for(args.base_model))
        # Only when there are declarations to order. A model called without
        # tools cannot be affected by this, and a limitation that does not
        # apply teaches readers to skim them.
        + (
            (
                "--wire-convention was not given, so this bundle does not record which property "
                "order its declarations were rendered in. The two conventions in the field "
                "disagree for every declaration with more than one property, and on the base "
                "checkpoint the wrong one costs a resolved 0.019-0.036 exact match -- as a wrong "
                "argument value, not a reordered one. A consumer has to guess.",
            )
            if args.wire_convention is None
            else ()
        )
        # A tag can be moved. Recorded rather than refused: it still pins
        # something, and the bundle is where a reader finds out how much.
        + tuple(weak_revision_limitations(args.base_model_revision, "--base-model-revision")),
    )
    result = build_bundle(request, events=events)
    delivered = _report(result.as_dict(), lambda: summarise_bundle(result), args.json)
    if not delivered:
        return EXIT_CODES[Status.ERROR]
    return EXIT_CODES[Status(result.status.value)]


def summarise_bundle(result: BundleResult) -> list[str]:
    contract = result.request.contract
    lines = [
        f"status: {result.status.value}",
        f"  {len(result.members)} file(s) in {result.request.output_dir}",
        f"  contract: prompt mode {contract.prompt_mode.value}, base {contract.base_model}"
        f"@{contract.base_model_revision}",
    ]
    for check in result.checks.checks:
        if check.outcome is not Outcome.PASSED:
            lines.append(f"  {check.outcome.value}: {check.name} — {check.detail}")
    if result.missing_measurements:
        lines.append(f"  measurements not made: {', '.join(result.missing_measurements)}")
    for text in result.limitations:
        lines.append(f"  note: {text}")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


HANDLERS = {
    "verify": _verify,
    "prepare": _prepare,
    "tune": _tune,
    "convert": _convert,
    "bundle": _bundle,
    "env": _env,
}


def main(argv: list[str] | None = None) -> int:
    """Every exit code this tool produces comes from here.

    Which means every path out of it has to survive a reader that closed the
    pipe. Three attempts got this wrong in three different places -- the
    result write, argparse's own output, and the refusal messages printed by
    the handlers below -- each time leaving `Py_FinalizeEx` to turn the unread
    buffer into status 120, or returning 0 for a request that failed. The
    structure here is: decide the code, then write, then make sure nothing is
    left in a buffer aimed at a dead pipe.
    """
    # argparse writes usage errors straight to stderr, and whether that raises
    # here or at interpreter exit depends on the version -- 3.11 wraps the write
    # in `except OSError: pass`, 3.10 does not. Both paths are covered below;
    # what matters is that neither returns 0.
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        # `--help` and usage errors. argparse printed and exited; the write is
        # buffered, so a closed pipe does not raise here -- it raises during the
        # interpreter's exit flush. Flushing inside the guard is what makes it
        # observable, and `exc.code` is what argparse decided: 0 for help, 2 for
        # a usage error. Returning 0 for both reported a malformed command line
        # as success.
        if not _flush_streams():
            _silence_streams()
            # Not 2: that is `inconclusive`, a claim about a measurement, and
        # `_Parser` exists to stop this command emitting it for a request
        # it would not run.
        return exc.code if isinstance(exc.code, int) else EXIT_CODES[Status.ERROR]
    except BrokenPipeError:
        # The parser could not report. A usage error is still a usage error, and
        # returning 0 here reported a malformed command line as success.
        _silence_streams()
        return EXIT_CODES[Status.ERROR]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    code = _dispatch(args)
    if not _flush_streams():
        # Anything still buffered is aimed at a reader that is gone. Silence
        # both streams so the interpreter's own flush cannot raise after this
        # function has already decided the answer.
        _silence_streams()
    return code


def _flush_streams() -> bool:
    """Flush stdout and stderr. False if a reader had closed one of them."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (BrokenPipeError, ValueError):
            return False
        except OSError:
            return False
    return True


def _dispatch(args: argparse.Namespace) -> int:
    """Run the command and convert every failure into a code."""
    try:
        return _run(args)
    except BrokenPipeError:
        # `2>&1 | head` while the run is still going: the event stream writes
        # progress to stderr, so this aborts the command partway. There is no
        # verdict yet, and returning 0 turned a failing bundle into a passing
        # one -- the single thing this tool exists to prevent.
        #
        # A closed pipe *after* the result is written is a different case and
        # never reaches here: `_report` absorbs it and the verdict stands.
        logger.debug("stderr closed mid-run", exc_info=True)
        return EXIT_CODES[Status.ERROR]
    except REFUSALS as exc:
        # litetune will not run this request, and said why. Not a verdict about
        # any model, so it shares the "could not check" code and not the
        # "failed" one. The reason is printed rather than logged: these messages
        # exist to be read by the person who typed the command. `_say` because
        # this print is itself on a stream that may be gone, and a refusal that
        # cannot be printed is still a refusal.
        logger.debug("litetune %s refused the request", args.command, exc_info=True)
        _say(f"litetune {args.command} did not run: {exc}")
        _say("no claim is made about the model")
        return EXIT_CODES[Status.ERROR]
    except Exception:  # noqa: BLE001 - top-level boundary: a crash must not read as a verdict
        logger.exception("litetune %s could not run", args.command)
        _say(
            f"status: {Status.ERROR.value} — litetune could not complete the run; "
            "no claim is made about the model"
        )
        return EXIT_CODES[Status.ERROR]


def _say(message: str) -> None:
    """Write to stderr, and do not let a closed pipe replace the verdict."""
    try:
        print(message, file=sys.stderr)
    except (BrokenPipeError, OSError, ValueError):
        logger.debug("could not write to stderr: %s", message)


def _run(args: argparse.Namespace) -> int:
    """Dispatch, then stop a closed pipe from rewriting the verdict.

    When the reader goes away (`| head`), the interpreter's exit-time flush
    raises `BrokenPipeError` and `Py_FinalizeEx` sets exit status **120** -- a
    code in neither of this tool's vocabularies, for a run that finished. The
    only way to prevent that is to leave nothing in the buffer pointed at a
    dead pipe.

    Here rather than in `_report` for one reason only: at this point the command
    has finished and its output is written, so replacing stdout cannot discard
    anything the run still needed.

    It is **not** a smaller trap for a library caller -- `main` is one frame up,
    and the descriptor is not restored. It cannot be: the whole purpose is to
    leave nothing pointed at a dead pipe when the interpreter flushes at exit,
    and putting the original back would undo exactly that. A caller that keeps
    running after `main()` returns and writes to a stream whose reader had gone
    away will find its output discarded. That is why `README.md` says there is
    no library API yet and to treat everything below `litetune.<module>` as
    private. An earlier version of this docstring claimed the descriptor was
    restored; it never was.
    """
    global _BROKEN_PIPE
    _BROKEN_PIPE = False
    try:
        code = HANDLERS[args.command](args)
    finally:
        if _BROKEN_PIPE:
            _silence_streams()
    return code


def _silence_streams() -> None:
    """Point stdout and stderr at devnull. Both, because either can be the pipe.

    `_report` only ever covered stdout, but the event stream writes progress and
    errors to stderr, so `litetune tune ... 2>&1 | head` broke on the channel
    nothing was guarding.
    """
    for stream in (sys.stdout, sys.stderr):
        _silence(stream)


def _silence(stream: Any) -> None:
    """Redirect one stream, keeping fd-backed and replaced ones apart.

    `stream.fileno()` raises for anything not backed by a descriptor -- which
    `contextlib.redirect_stdout` installs, and which is the obvious way to drive
    this entry point in-process. Raising there turned a good verdict into exit 4.
    """
    try:
        fileno = stream.fileno()
    except Exception:  # noqa: BLE001 - a replaced stream has no descriptor to fix
        if stream is sys.stdout:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - closed at exit
        else:
            sys.stderr = open(os.devnull, "w")  # noqa: SIM115 - closed at exit
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, fileno)
    finally:
        os.close(devnull)


if __name__ == "__main__":
    sys.exit(main())
