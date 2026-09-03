"""Fine-tuning, with no torch, no accelerator and no model load in this process.

`StageEnv.run` is faked: it reads the config the stage wrote and produces the
files a real run would, so every branch -- a non-zero exit, a clean exit with no
checkpoint, a run whose loss was never masked -- is exercised in milliseconds.
The fake matches `StageEnv.run`'s contract exactly: it returns a
`CompletedProcess` and never raises on a non-zero exit, because "the subprocess
failed" is data this module has to record rather than an exception.

The generated training script is exercised too, without torch: its masking and
batching functions are module-level and import nothing, so they are `exec`'d
against a fake tokenizer. That matters more than the wrapper -- the mask is the
one thing in this tool that failed silently and cost nine times the base score.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from litetune import envs
from litetune.checks import Outcome
from litetune.evaluate import PromptMode
from litetune.events import EventStream
from litetune.tune import (
    _TRAIN_SCRIPT,
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_DTYPE,
    ENV_CHECK,
    EXPECTED_SUPERVISED_FRACTION,
    LEARNING_RATES,
    MASKING_CHECK,
    MERGE_CHECK,
    TRAINING_CHECK,
    TrainingMetrics,
    TuneError,
    TuneRequest,
    masking_check,
    run_tune,
    write_report,
)

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# A fake training environment
# ---------------------------------------------------------------------------


@dataclass
class Call:
    argv: list[str]
    timeout: int


@dataclass
class FakeTrainer:
    """Stands in for `envs.StageEnv.run`, doing what the real script would do."""

    returncode: int = 0
    stderr: str = ""
    stdout: str = ""
    supervised_tokens: int = 24
    total_tokens: int = 350
    epoch_losses: tuple[float, ...] = (1.45,)
    write_model: bool = True
    write_adapter: bool = True
    write_metrics: bool = True
    drop_fraction: bool = False
    raises: BaseException | None = None
    calls: list[Call] = field(default_factory=list)
    configs: list[dict] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(Call(argv=list(args), timeout=timeout))
        if self.raises is not None:
            raise self.raises
        config = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.configs.append(config)

        if self.write_model:
            model_dir = Path(config["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.safetensors").write_bytes(b"\0" * 32)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
        if self.write_adapter and config.get("adapter_dir"):
            adapter = Path(config["adapter_dir"])
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 8)
        if self.write_metrics:
            fraction = self.supervised_tokens / self.total_tokens if self.total_tokens else None
            payload = {
                "method": config["method"],
                "learning_rate": config["learning_rate"],
                "dtype": config["dtype"],
                "attn_implementation": config["attn_implementation"],
                "prompt_mode": config["prompt_mode"],
                "n_examples": 40,
                "supervised_tokens": self.supervised_tokens,
                "total_tokens": self.total_tokens,
                "masked_tokens": self.total_tokens - self.supervised_tokens,
                "supervised_token_fraction": fraction,
                "trainable_parameters": 1234,
                "base_parameters": 270_000_000,
                "epochs": [
                    {"epoch": i + 1, "portion": 1.0, "loss": loss, "steps": 5}
                    for i, loss in enumerate(self.epoch_losses)
                ],
                "model_dir": config["model_dir"],
                "adapter_dir": config.get("adapter_dir"),
            }
            if self.drop_fraction:
                payload.pop("supervised_token_fraction")
            Path(config["metrics_out"]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def trainer(monkeypatch, tmp_path) -> FakeTrainer:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / ".litetune-ready").write_text(self.identity)
        return self.path

    fake = FakeTrainer()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


@pytest.fixture
def train_data(tmp_path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": f"set the background to colour swatch{i}",
                    "completion": (
                        f"call:change_background_color{{color:<escape>swatch{i}<escape>}}"
                    ),
                    "source_line": i + 1,
                }
            )
            + "\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def request_for(tmp_path, train_data):
    def _build(**kwargs) -> TuneRequest:
        params = {
            "model": "google/functiongemma-270m-it",
            "data": train_data,
            "output_dir": tmp_path / "tuned",
            # Stated, not defaulted: `TuneRequest` refuses to guess it.
            "prompt_mode": PromptMode.PRERENDERED,
        }
        params.update(kwargs)
        return TuneRequest(**params)

    return _build


def check_named(result, name):
    return next(c for c in result.checks.checks if c.name == name)


# ---------------------------------------------------------------------------
# The parent process never touches the training stack
# ---------------------------------------------------------------------------


def test_the_parent_process_imports_neither_torch_nor_transformers(trainer, request_for):
    run_tune(request_for())

    # The entire reason the per-stage environments exist: torch/transformers and
    # litert-torch/numpy<2.1 cannot share an interpreter, so this one loads
    # neither.
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert "peft" not in sys.modules


def test_the_run_happens_inside_the_train_environment(trainer, request_for):
    request = request_for()
    run_tune(request)

    assert len(trainer.calls) == 1
    argv = trainer.calls[0].argv
    assert argv[0] == "python"
    assert Path(argv[1]).name == "train_script.py"
    assert trainer.calls[0].timeout == request.timeout_s
    assert request.env is envs.TRAIN


# ---------------------------------------------------------------------------
# Learning rates are per method
# ---------------------------------------------------------------------------


def test_each_method_gets_its_own_default_learning_rate(trainer, request_for):
    run_tune(request_for(method="full"))
    run_tune(request_for(method="lora", output_dir=request_for().output_dir / "lora"))

    assert trainer.configs[0]["learning_rate"] == LEARNING_RATES["full"] == 1e-5
    assert trainer.configs[1]["learning_rate"] == LEARNING_RATES["lora"] == 2e-4
    # A single shared rate starves one of the two, and a comparison run that way
    # measures the rate rather than the method.
    assert LEARNING_RATES["lora"] / LEARNING_RATES["full"] == 20.0


def test_a_declared_learning_rate_is_used_and_marked_as_declared(trainer, request_for):
    request = request_for(method="lora", learning_rate=5e-5)
    result = run_tune(request)

    assert trainer.configs[0]["learning_rate"] == 5e-5
    assert request.rate_is_default is False
    assert result.as_dict()["request"]["learning_rate_source"] == "declared"


def test_an_unknown_method_is_refused(tmp_path, train_data):
    with pytest.raises(TuneError):
        TuneRequest(
            model="m",
            data=train_data,
            output_dir=tmp_path,
            method="qlora",
            prompt_mode=PromptMode.PRERENDERED,
        )


# ---------------------------------------------------------------------------
# The mask is the finding
# ---------------------------------------------------------------------------


def test_a_supervised_fraction_near_one_is_reported_as_masking_not_applied(trainer, request_for):
    # The measured failure: without masking the tool declarations are ~330 of
    # ~350 tokens, the run reaches a *lower* loss (0.50 against 1.45), and it
    # scores 0.0625 against a 0.5625 base -- nine times worse than not training.
    trainer.supervised_tokens = 350
    trainer.total_tokens = 350
    trainer.epoch_losses = (0.50,)

    result = run_tune(request_for(method="lora"))

    mask = check_named(result, MASKING_CHECK)
    assert mask.outcome is Outcome.FAILED
    assert "0.0625" in mask.detail and "0.5625" in mask.detail
    assert mask.observed["supervised_token_fraction"] == 1.0
    assert result.outcome is Outcome.FAILED
    # The training check itself passed: the process exited zero and wrote a
    # checkpoint. That is exactly why the fraction has to be a separate,
    # reported number rather than an assumption.
    assert check_named(result, TRAINING_CHECK).outcome is Outcome.PASSED
    assert any("expected to be worse" in text for text in result.limitations)


def test_a_masked_run_reports_the_fraction_it_actually_trained_on(trainer, request_for):
    trainer.supervised_tokens = 24
    trainer.total_tokens = 350

    result = run_tune(request_for())

    mask = check_named(result, MASKING_CHECK)
    assert mask.outcome is Outcome.PASSED
    assert mask.observed["supervised_token_fraction"] == pytest.approx(0.0686, abs=1e-4)
    # ~0.07 is what a correctly masked run looks like on this data shape.
    assert abs(mask.observed["supervised_token_fraction"] - EXPECTED_SUPERVISED_FRACTION) < 0.02
    assert result.outcome is Outcome.PASSED


def test_a_run_that_masked_nothing_at_all_fails_even_below_the_threshold():
    metrics = TrainingMetrics(
        n_examples=10,
        supervised_tokens=90,
        total_tokens=100,
        masked_tokens=0,
        supervised_token_fraction=0.9,
        epochs=(),
    )
    check = masking_check(metrics, "unused")

    # 0.9 is below MASKING_NOT_APPLIED_ABOVE, but zero masked tokens is
    # conclusive on its own.
    assert check.outcome is Outcome.FAILED


def test_a_missing_fraction_is_could_not_check_not_a_pass(trainer, request_for):
    trainer.drop_fraction = True

    result = run_tune(request_for())

    mask = check_named(result, MASKING_CHECK)
    assert mask.outcome is Outcome.UNCHECKED
    assert "0.0625" in mask.detail
    assert result.outcome is Outcome.UNCHECKED


def test_no_metrics_file_leaves_the_mask_unobserved(trainer, request_for):
    trainer.write_metrics = False

    result = run_tune(request_for())

    assert result.metrics is None
    assert check_named(result, MASKING_CHECK).outcome is Outcome.UNCHECKED
    assert any("reported no metrics" in text for text in result.limitations)


def test_a_stale_metrics_file_is_not_read_as_this_run(trainer, request_for):
    request = request_for()
    request.output_dir.mkdir(parents=True, exist_ok=True)
    (request.output_dir / "metrics.json").write_text(
        json.dumps({"supervised_token_fraction": 0.07}), encoding="utf-8"
    )
    trainer.write_metrics = False

    result = run_tune(request)

    # A previous attempt's numbers read as this attempt's is the same failure as
    # an export inheriting a stale artifact.
    assert result.metrics is None
    assert check_named(result, MASKING_CHECK).outcome is Outcome.UNCHECKED


# ---------------------------------------------------------------------------
# The generated script's masking, exercised without torch
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """One id per whitespace word, plus an optional leading BOS.

    No `apply_chat_template`: this stands in for the prerendered path, where
    the terminator falls back to `eos_token_id`.
    """

    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        ids = [1000 + i for i, _ in enumerate(text.split())]
        return {"input_ids": ([2] if add_special_tokens else []) + ids}

    def decode(self, ids) -> str:
        return " ".join(f"<{i}>" for i in ids)


class FakeTemplateTokenizer(FakeTokenizer):
    """A tokenizer whose chat template closes the turn with something other
    than `eos_token_id` -- the case that made models fail to stop."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<start>{m['role']}\n{m['content']}<end_of_turn>\n" for m in messages)
        return out + ("<start>model\n" if add_generation_prompt else "")


@pytest.fixture(scope="module")
def script_namespace() -> dict:
    """The training script's module-level functions, without running `main`.

    `main` is where torch is imported, and it only runs under
    `__name__ == "__main__"`, so the masking logic can be tested directly.
    """
    namespace: dict = {"__name__": "litetune_train_script_under_test"}
    exec(compile(_TRAIN_SCRIPT, "train_script.py", "exec"), namespace)
    return namespace


def test_the_terminator_comes_from_the_chat_template_when_there_is_one(script_namespace):
    """A model must be trained to emit the token its runtime waits for.

    Gemma closes an assistant turn with `<end_of_turn>` while `eos_token_id` is
    a different token. Training the second while the runtime waits for the
    first produces a model that never stops -- observed as
    34 of 40 generations emitting more than one tool call, the same call
    repeated up to eight times. The score barely moves; the device fires every
    call.
    """
    turn_terminator = script_namespace["turn_terminator"]

    ids, source = turn_terminator(FakeTemplateTokenizer(), runtime_rendered=True)

    assert source == "chat_template"
    assert ids and ids != [FakeTokenizer.eos_token_id]


def test_the_terminator_falls_back_to_eos_without_a_template(script_namespace):
    turn_terminator = script_namespace["turn_terminator"]

    assert turn_terminator(FakeTokenizer(), runtime_rendered=False) == (
        [FakeTokenizer.eos_token_id],
        "tokenizer_eos",
    )


def test_the_terminator_is_supervised_not_merely_appended(script_namespace):
    """Appending it to the input while masking it would teach nothing."""
    build_examples = script_namespace["build_examples"]
    rows = [{"prompt": "a b", "completion": "x", "source_line": 1}]

    examples, _, _, terminator = build_examples(FakeTokenizer(), rows, 64, False)

    input_ids, labels = examples[0]
    assert input_ids[-len(terminator["ids"]) :] == terminator["ids"]
    assert labels[-len(terminator["ids"]) :] == terminator["ids"]


def test_which_terminator_was_used_is_recorded(script_namespace):
    """A terminator chosen silently is one nobody can check."""
    build_examples = script_namespace["build_examples"]
    rows = [{"prompt": "a", "completion": "x", "source_line": 1}]

    _, _, _, terminator = build_examples(FakeTokenizer(), rows, 64, False)

    assert terminator["source"] == "tokenizer_eos"
    assert terminator["ids"] == [FakeTokenizer.eos_token_id]
    assert terminator["text"]


def test_the_script_masks_every_prompt_token(script_namespace):
    build_examples = script_namespace["build_examples"]
    rows = [{"prompt": "a b c d e", "completion": "x y", "source_line": 1}]

    examples, supervised, total, terminator = build_examples(FakeTokenizer(), rows, 64, False)

    input_ids, labels = examples[0]
    assert len(input_ids) == len(labels)
    # 1 BOS + 5 prompt words masked; 2 completion words + EOS supervised.
    assert labels[:6] == [IGNORE_INDEX] * 6
    assert labels[6:] == input_ids[6:]
    assert supervised == 3
    assert total == 9
    assert supervised / total < 0.4


def test_the_script_reports_a_fraction_that_matches_the_declaration_ratio(script_namespace):
    build_examples = script_namespace["build_examples"]
    # The real shape: ~330 declaration tokens against a ~25-token answer.
    rows = [
        {
            "prompt": " ".join(["decl"] * 330),
            "completion": " ".join(["ans"] * 24),
            "source_line": 1,
        }
    ]

    _, supervised, total, _term = build_examples(FakeTokenizer(), rows, 1024, False)

    assert supervised / total == pytest.approx(EXPECTED_SUPERVISED_FRACTION, abs=0.01)


def test_the_script_refuses_to_truncate_and_names_the_row(script_namespace):
    build_examples = script_namespace["build_examples"]
    rows = [{"prompt": " ".join(["w"] * 200), "completion": "answer", "source_line": 42}]

    with pytest.raises(ValueError) as exc:
        build_examples(FakeTokenizer(), rows, 32, False)

    assert "42" in str(exc.value)
    assert "Truncating" in str(exc.value)


def test_prerendered_training_uses_the_prompt_verbatim(script_namespace):
    render_prompt = script_namespace["render_prompt"]

    text, add_special = render_prompt(FakeTokenizer(), "make it blue", False)

    assert text == "make it blue"
    # No template applied, so the tokenizer supplies the BOS.
    assert add_special is True


def test_runtime_rendered_training_applies_the_chat_template_once(script_namespace):
    class TemplatingTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "<bos><start_of_turn>user\n" + messages[0]["content"] + "<end_of_turn>\n"

    render_prompt = script_namespace["render_prompt"]

    text, add_special = render_prompt(TemplatingTokenizer(), "make it blue", True)

    assert text.startswith("<bos>") and "make it blue" in text
    # The template already emitted the BOS. A second one shifts every position
    # by one and is invisible in the loss.
    assert add_special is False


def test_the_trained_prompt_mode_reaches_the_script_and_the_report(trainer, request_for):
    result = run_tune(request_for(prompt_mode=PromptMode.RUNTIME_RENDERED))

    assert trainer.configs[0]["prompt_mode"] == "runtime_rendered"
    # This is the field `bundle.Contract` refuses to default, so the stage that
    # decided it has to publish it.
    assert result.prompt_mode is PromptMode.RUNTIME_RENDERED
    assert result.as_dict()["prompt_mode"] == "runtime_rendered"


def test_a_prompt_mode_that_is_not_one_of_the_two_is_refused(tmp_path, train_data):
    with pytest.raises(TuneError):
        TuneRequest(model="m", data=train_data, output_dir=tmp_path, prompt_mode="runtime_rendered")


def test_the_script_masks_padding_out_of_the_loss(script_namespace):
    class FakeTensorLib:
        long = "long"

        @staticmethod
        def tensor(values, dtype=None):
            return values

    batches = script_namespace["batches"]
    examples = [([1, 2, 3], [IGNORE_INDEX, 2, 3]), ([4, 5], [IGNORE_INDEX, 5])]

    ((input_ids, attention, labels),) = list(batches(examples, 2, 0, FakeTensorLib))

    assert input_ids == [[1, 2, 3], [4, 5, 0]]
    assert attention == [[1, 1, 1], [1, 1, 0]]
    # Padding contributes no gradient either, or the shorter rows in a batch
    # would teach the model to emit pad tokens.
    assert labels == [[IGNORE_INDEX, 2, 3], [IGNORE_INDEX, 5, IGNORE_INDEX]]


# ---------------------------------------------------------------------------
# The adapter is an artifact
# ---------------------------------------------------------------------------


def test_lora_retains_the_adapter_beside_the_merged_checkpoint(trainer, request_for):
    request = request_for(method="lora")
    result = run_tune(request)

    assert result.adapter_dir is not None and result.adapter_dir.is_dir()
    assert any(result.adapter_dir.iterdir())
    assert result.model_dir == request.model_dir and result.model_dir.is_dir()
    # Distinct directories: a merged checkpoint cannot be un-merged, so the
    # adapter is the only recoverable form of what was learned.
    assert result.adapter_dir != result.model_dir
    assert check_named(result, MERGE_CHECK).outcome is Outcome.PASSED


def test_a_lora_run_that_kept_no_adapter_fails(trainer, request_for):
    trainer.write_adapter = False

    result = run_tune(request_for(method="lora"))

    merge = check_named(result, MERGE_CHECK)
    assert merge.outcome is Outcome.FAILED
    assert "cannot be un-merged" in merge.detail
    assert result.outcome is Outcome.FAILED


def test_a_full_fine_tune_has_no_adapter_and_says_so(trainer, request_for):
    result = run_tune(request_for(method="full"))

    assert result.request.adapter_dir is None
    assert result.adapter_dir is None
    assert check_named(result, MERGE_CHECK).outcome is Outcome.PASSED


# ---------------------------------------------------------------------------
# A non-zero exit is data
# ---------------------------------------------------------------------------


def test_a_non_zero_exit_is_recorded_not_raised(trainer, request_for):
    trainer.returncode = 1
    trainer.stderr = "CUDA out of memory"
    trainer.write_model = False

    result = run_tune(request_for())

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.FAILED
    assert "exited 1" in check.detail and "CUDA out of memory" in check.detail
    assert result.returncode == 1
    assert result.model_dir is None


def test_exit_zero_with_no_checkpoint_is_a_failure(trainer, request_for):
    trainer.write_model = False

    result = run_tune(request_for())

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.FAILED
    assert "exited zero but wrote no checkpoint" in check.detail


def test_a_timeout_is_could_not_check_not_a_failure(trainer, request_for):
    trainer.raises = subprocess.TimeoutExpired(cmd="python", timeout=10)

    result = run_tune(request_for())

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    assert "timeout" in check.detail
    assert result.outcome is Outcome.UNCHECKED


def test_an_unprovisioned_environment_is_could_not_check(monkeypatch, request_for, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(
        envs.StageEnv, "provision", lambda self, events=None, force=False: self.path
    )

    def never_called(*args, **kwargs):
        raise AssertionError("the training script must not run without an environment")

    monkeypatch.setattr(envs.StageEnv, "run", never_called)

    result = run_tune(request_for())

    check = check_named(result, ENV_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    assert result.outcome is Outcome.UNCHECKED
    assert any("not attempted" in text for text in result.limitations)


def test_a_missing_training_split_is_a_failure_not_a_crash(trainer, request_for, tmp_path):
    result = run_tune(request_for(data=tmp_path / "absent.jsonl"))

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.FAILED
    assert "does not exist" in check.detail
    assert trainer.calls == []


# ---------------------------------------------------------------------------
# Events, dtype, and the refusal to claim verification
# ---------------------------------------------------------------------------


def test_per_epoch_metrics_reach_the_event_stream(trainer, request_for, capsys):
    trainer.epoch_losses = (1.45, 0.98, 0.71)
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    run_tune(request_for(epochs=3), events=events)

    assert capsys.readouterr().out == ""
    losses = [e.data for e in seen if e.kind == "metric" and e.data.get("name") == "train.loss"]
    assert [d["value"] for d in losses] == [1.45, 0.98, 0.71]
    assert [d["epoch"] for d in losses] == [1, 2, 3]
    fraction = next(e.data for e in seen if e.data.get("name") == "supervised_token_fraction")
    assert fraction["expected"] == EXPECTED_SUPERVISED_FRACTION


def test_the_dtype_and_attention_the_export_path_uses_are_passed_through(trainer, request_for):
    run_tune(request_for())

    config = trainer.configs[0]
    # Mismatching either against what export and evaluation load produces output
    # that is fluent, wrong, and passes every label-free check.
    assert config["dtype"] == DEFAULT_DTYPE == "bfloat16"
    assert config["attn_implementation"] == DEFAULT_ATTN_IMPLEMENTATION == "eager"


def test_a_different_dtype_is_allowed_and_named_as_a_limitation(trainer, request_for):
    result = run_tune(request_for(dtype="float32"))

    assert trainer.configs[0]["dtype"] == "float32"
    assert any("fluent, wrong" in text for text in result.limitations)


def test_a_completed_run_is_never_reported_as_verified(trainer, request_for):
    result = run_tune(request_for())

    assert result.outcome is Outcome.PASSED
    assert result.verified is False
    record = result.as_dict()
    assert record["verified"] is False
    assert "0.0625" in record["unverified_reason"]
    # Structural: `verified` is a property, so no code path can set it True.
    with pytest.raises(AttributeError):
        result.verified = True


def test_the_recorded_mode_is_what_a_bundle_contract_takes(trainer, request_for):
    from litetune.bundle import Contract, versions_from

    result = run_tune(request_for())

    # The handoff `bundle.Contract` exists to force: the stage that decided the
    # convention hands it over, rather than a human retyping it into a config.
    contract = Contract(
        prompt_mode=result.prompt_mode,
        established_against=versions_from(envs.TRAIN, envs.RUNTIME),
        base_model=result.request.model,
        base_model_revision="0123456789abcdef0123456789abcdef01234567",
    )
    assert contract.prompt_mode is result.prompt_mode
    assert contract.as_dict()["established_against"]["transformers"] == "5.16.1"


def test_a_report_is_written_for_a_failed_run(trainer, request_for):
    trainer.returncode = 1
    trainer.write_model = False

    result = run_tune(request_for())
    path = write_report(result)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["outcome"] == "failed"
    assert record["returncode"] == 1
    assert record["verified"] is False


# ---------------------------------------------------------------------------
# The whole script, against stub modules -- still no torch in this process
# ---------------------------------------------------------------------------
#
# `FakeTrainer` writes the metrics payload it thinks the script writes. Nothing
# above checks those two agree, and a field renamed on one side would leave
# `read_metrics` silently reporting no supervised-token fraction -- which this
# module treats as `could not check` rather than as a failure. So the real
# script is executed once, end to end, against stubs that satisfy only the
# handful of calls it makes.

_STUB_TORCH = """
bfloat16 = "bfloat16"
float32 = "float32"
long = "long"


def manual_seed(seed):
    return seed


def tensor(values, dtype=None):
    return values


class _AdamW:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.steps = 0

    def step(self):
        self.steps += 1

    def zero_grad(self, set_to_none=False):
        pass


class optim:
    AdamW = _AdamW
"""

_STUB_TRANSFORMERS = """
import json
import os
from pathlib import Path


def _log(record):
    path = os.environ["LITETUNE_STUB_LOG"]
    with open(path, "a", encoding="utf-8") as sink:
        sink.write(json.dumps(record) + "\\n")


class _Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        ids = [1000 + i for i, _ in enumerate(text.split())]
        return {"input_ids": ([2] if add_special_tokens else []) + ids}

    def decode(self, ids):
        return " ".join(f"<{i}>" for i in ids)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model, **kwargs):
        return _Tokenizer()


class _Loss:
    def __init__(self, value):
        self.value = value

    def backward(self):
        pass

    def detach(self):
        return self

    def __float__(self):
        return self.value


class _Output:
    def __init__(self, loss):
        self.loss = loss


class _Config:
    use_cache = True


class _Parameter:
    def __init__(self, count):
        self._count = count
        self.requires_grad = True

    def numel(self):
        return self._count


class _Model:
    def __init__(self, tag):
        self.tag = tag
        self.config = _Config()

    def parameters(self):
        return [_Parameter(1000)]

    def train(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        _log({"event": "forward", "rows": len(labels), "width": len(labels[0])})
        return _Output(_Loss(1.45))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / (self.tag + ".json")).write_text("{}", encoding="utf-8")
        _log({"event": "save", "tag": self.tag, "path": str(path)})

    def merge_and_unload(self):
        _log({"event": "merge"})
        return _Model("merged")


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model, **kwargs):
        _log({"event": "load", "model": model, "kwargs": {k: str(v) for k, v in kwargs.items()}})
        return _Model("full")
"""

_STUB_PEFT = """
class LoraConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def get_peft_model(model, config):
    model.tag = "adapter"
    model.lora = config
    return model
"""


@pytest.fixture
def stub_env(tmp_path):
    """A directory of stub torch/transformers/peft modules, and a call log."""
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text(_STUB_TORCH, encoding="utf-8")
    (stubs / "transformers.py").write_text(_STUB_TRANSFORMERS, encoding="utf-8")
    (stubs / "peft.py").write_text(_STUB_PEFT, encoding="utf-8")
    log = tmp_path / "stub.log"
    return stubs, log


def run_real_script(request: TuneRequest, stub_env) -> subprocess.CompletedProcess:
    stubs, log = stub_env
    request.output_dir.mkdir(parents=True, exist_ok=True)
    script = request.output_dir / "train_script.py"
    script.write_text(_TRAIN_SCRIPT, encoding="utf-8")
    config = request.output_dir / "train_config.json"
    config.write_text(
        json.dumps(request.config(request.output_dir / "metrics.json")), encoding="utf-8"
    )
    return subprocess.run(
        [sys.executable, str(script), str(config)],
        capture_output=True,
        text=True,
        env={"PATH": "", "PYTHONPATH": str(stubs), "LITETUNE_STUB_LOG": str(log)},
    )


def stub_log(stub_env) -> list[dict]:
    _, log = stub_env
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_the_real_script_writes_metrics_this_module_can_read(request_for, stub_env):
    from litetune.tune import read_metrics

    request = request_for(epochs=2, batch_size=8)
    proc = run_real_script(request, stub_env)
    assert proc.returncode == 0, proc.stderr

    # The contract that `FakeTrainer` cannot check: the script's own output,
    # parsed by the parser this module ships.
    metrics = read_metrics(request.output_dir / "metrics.json")
    assert metrics.n_examples == 40
    assert metrics.total_tokens == metrics.supervised_tokens + metrics.masked_tokens
    assert metrics.supervised_token_fraction == pytest.approx(
        metrics.supervised_tokens / metrics.total_tokens
    )
    # Prompts are six words, completions one -- masked, and visibly so.
    assert metrics.supervised_token_fraction < 0.4
    assert [e.epoch for e in metrics.epochs] == [1, 2]
    assert metrics.final_loss == pytest.approx(1.45)


def test_the_real_script_loads_the_dtype_and_attention_it_was_given(request_for, stub_env):
    request = request_for()
    assert run_real_script(request, stub_env).returncode == 0

    load = next(entry for entry in stub_log(stub_env) if entry["event"] == "load")
    assert load["kwargs"]["dtype"] == DEFAULT_DTYPE
    assert load["kwargs"]["attn_implementation"] == DEFAULT_ATTN_IMPLEMENTATION


def test_the_real_script_saves_the_adapter_before_merging(request_for, stub_env):
    request = request_for(method="lora")
    assert run_real_script(request, stub_env).returncode == 0

    events = [e["event"] for e in stub_log(stub_env)]
    saves = [e for e in stub_log(stub_env) if e["event"] == "save"]
    # Order matters: a merged checkpoint cannot be un-merged, so the adapter has
    # to be written first and kept.
    assert events.index("merge") > events.index("save")
    assert [s["tag"] for s in saves] == ["adapter", "merged"]
    assert (request.adapter_dir / "adapter.json").is_file()
    assert (request.model_dir / "merged.json").is_file()
    assert (request.model_dir / "tokenizer.json").is_file()


def test_the_real_script_refuses_an_over_length_row(request_for, stub_env, tmp_path):
    long_row = tmp_path / "long.jsonl"
    long_row.write_text(
        json.dumps({"prompt": " ".join(["w"] * 200), "completion": "answer", "source_line": 9})
        + "\n",
        encoding="utf-8",
    )
    request = request_for(data=long_row, max_seq_length=32)

    proc = run_real_script(request, stub_env)

    assert proc.returncode != 0
    assert "source line 9" in proc.stderr
    assert not (request.output_dir / "metrics.json").exists()


# ---------------------------------------------------------------------------
# The three stages compose
# ---------------------------------------------------------------------------


def test_prepare_feeds_tune_feeds_bundle(trainer, tmp_path):
    """The handoffs, end to end: split -> checkpoint -> deliverable."""
    from litetune.bundle import BundleRequest, Contract, build_bundle, versions_from
    from litetune.manifest import RunStatus
    from litetune.prepare import PrepareRequest, prepare

    data = tmp_path / "raw.jsonl"
    data.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": f"set the background to colour swatch{i}",
                    "target": {
                        "name": "change_background_color",
                        "args": {"color": f"swatch{i}"},
                    },
                }
            )
            + "\n"
            for i in range(400)
        ),
        encoding="utf-8",
    )

    class WordCounter:
        name = "fake"

        def describe(self):
            return {"tokenizer": "fake"}

        def count(self, texts):
            return [len(text.split()) for text in texts]

    prepared = prepare(
        PrepareRequest(
            data=data,
            output_dir=tmp_path / "prepared",
            context_length=1024,
            tokens=WordCounter(),
        )
    )
    assert prepared.outcome is Outcome.PASSED

    tuned = run_tune(
        TuneRequest(
            model="google/functiongemma-270m-it",
            data=prepared.train.path,
            output_dir=tmp_path / "tuned",
            method="lora",
            prompt_mode=PromptMode.PRERENDERED,
        )
    )
    assert tuned.outcome is Outcome.PASSED
    assert tuned.model_dir is not None

    declarations = tmp_path / "tools.json"
    declarations.write_text(json.dumps([{"name": "change_background_color"}]), encoding="utf-8")

    bundled = build_bundle(
        BundleRequest(
            output_dir=tmp_path / "bundle",
            model=tuned.model_dir,
            declarations=declarations,
            # The mode is carried from the stage that decided it, never retyped.
            contract=Contract(
                prompt_mode=tuned.prompt_mode,
                established_against=versions_from(envs.TRAIN, envs.RUNTIME),
                base_model=tuned.request.model,
                base_model_revision="0123456789abcdef0123456789abcdef01234567",
            ),
            status=RunStatus.UNMEASURED,
            limitations=list(prepared.limitations) + list(tuned.limitations),
        )
    )

    assert bundled.complete is True
    assert (bundled.request.output_dir / "model" / "model.safetensors").is_file()
    # Nothing was measured, and the bundle says so in three places rather than
    # reporting a pass.
    assert bundled.status is RunStatus.UNMEASURED
    assert bundled.missing_measurements == ["base_float", "tuned_float", "tuned_converted"]
    assert bundled.verified is False
    # The upstream limitations travel with the deliverable: a small held-out
    # split and an untrained-is-not-verified warning are properties of the
    # artifact, not of the run that has already finished.
    assert any("below" in text for text in bundled.limitations)
    assert any("0.0625" in text for text in bundled.limitations)


def test_fractional_epochs_are_scheduled_rather_than_rounded_away(script_namespace):
    # `spec.TrainSpec.epochs` is a float. Rounding 1.5 down to 1 would train for
    # two thirds of what the spec says while the manifest records the spec's
    # figure -- a run nobody could reproduce from its own record.
    epoch_schedule = script_namespace["epoch_schedule"]

    assert epoch_schedule(2.0, 40, 8) == [(1, 1.0, 5), (2, 1.0, 5)]
    # 40 examples in batches of 8 is 5 steps an epoch, so the 0.6 tail is 3.
    assert epoch_schedule(1.6, 40, 8) == [(1, 1.0, 5), (2, pytest.approx(0.6), 3)]
    assert epoch_schedule(0.4, 40, 8) == [(1, pytest.approx(0.4), 2)]
    # A schedule with no steps in it is refused rather than reported as a
    # training run that happened.
    with pytest.raises(ValueError):
        epoch_schedule(0.0, 40, 8)


def test_omitting_the_prompt_mode_is_refused(tmp_path, train_data):
    """The property `field(kw_only=True)` exists to guarantee.

    Reverting it to `= PromptMode.PRERENDERED` used to pass the whole suite:
    every call site already passed it by keyword, so nothing tested that
    omitting it fails. Which convention the prompt was built under cannot be
    guessed, and guessing wrong trains the model on a prompt the runtime never
    sends.
    """
    with pytest.raises(TypeError, match="prompt_mode"):
        TuneRequest(model="m", data=train_data, output_dir=tmp_path)


def test_the_prompt_mode_is_keyword_only(tmp_path, train_data):
    """Positionally it would land where `timeout_s` reads in the class body."""
    with pytest.raises(TypeError):
        TuneRequest("m", train_data, tmp_path, PromptMode.PRERENDERED)


def test_the_sentencepiece_model_is_carried_back_beside_the_checkpoint(tmp_path, script_namespace):
    """transformers 5.x stopped writing `tokenizer.model`, and nothing failed.

    The exporter's SentencePiece branch tests for that file and for a
    `vocab_file` the tokenizer classes no longer expose, so without the
    carry-back every bundle gets an HF tokenizer section instead of
    `SP_Tokenizer`. LiteRT-LM's FST-constrained decoding is SentencePiece-only,
    so the artifact still runs, still scores the same, and still passes every
    liveness check -- it just cannot do constrained tool-calling. This is the
    "a capability is missing while every check passes" shape, and the only way
    to catch it is to assert on the file.
    """
    import types

    carry_back_sentencepiece = script_namespace["carry_back_sentencepiece"]

    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.model").write_bytes(b"sp model bytes")
    out = tmp_path / "checkpoint"
    out.mkdir()
    (out / "tokenizer_config.json").write_text('{"model_max_length": 8192}', encoding="utf-8")

    tok = types.SimpleNamespace(name_or_path=str(base))
    outcome = carry_back_sentencepiece(tok, out, "google/functiongemma-270m-it", None)

    assert outcome == "carried back"
    assert (out / "tokenizer.model").read_bytes() == b"sp model bytes"
    config = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    # An absolute path: the exporter does not resolve a bare filename from here.
    assert config["vocab_file"] == str((out / "tokenizer.model").resolve())
    assert config["model_max_length"] == 8192, "the rest of the config must survive"


def test_a_model_without_a_sentencepiece_tokenizer_is_reported_not_failed(
    tmp_path, script_namespace
):
    """Qwen's tokenizer is BPE. Absent is a fact about the model, not an error."""
    import types

    carry_back_sentencepiece = script_namespace["carry_back_sentencepiece"]

    base = tmp_path / "base"
    base.mkdir()
    out = tmp_path / "checkpoint"
    out.mkdir()

    tok = types.SimpleNamespace(name_or_path=str(base))
    outcome = carry_back_sentencepiece(tok, out, "Qwen/Qwen3-0.6B", "deadbeef")

    assert not (out / "tokenizer.model").exists()
    assert outcome.startswith("unavailable:"), outcome
