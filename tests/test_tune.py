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
from conftest import fake_torch, mark_provisioned

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
    device: str | None = None
    raises: BaseException | None = None
    calls: list[Call] = field(default_factory=list)
    configs: list[dict] = field(default_factory=list)

    # -- the device probe ---------------------------------------------------
    # `run_tune` makes two calls into the environment, and they are different
    # commands with different contracts: `python -c <source>` for
    # `envs.resolve_device`, then `python <script> <config>` for the training
    # run. The fake has to answer both. Until it did, every probe here fell
    # into the training branch, tried to read the probe's *source code* as a
    # config path, raised `FileNotFoundError`, and `resolve_device` swallowed
    # that into "could not answer" -- so every test in this file ran against a
    # failed probe and could not tell one from a working one.
    #
    # `probe_device`/`probe_cuda_build`/`probe_device_count` compose the JSON
    # line a real probe prints. `probe_stdout` overrides them verbatim, which
    # is how a banner, an empty answer or an unparseable one are expressed;
    # `probe_returncode`, `probe_stderr` and `probe_raises` cover the ways a
    # probe fails to produce one at all.
    probe_device: str | None = "cpu"
    probe_cuda_build: str | None = None
    probe_device_count: int = 0
    probe_stdout: str | None = None
    probe_returncode: int = 0
    probe_stderr: str = ""
    probe_raises: BaseException | None = None

    @staticmethod
    def is_probe(args) -> bool:
        return len(args) >= 2 and args[1] == "-c"

    def probe_result(self, args) -> subprocess.CompletedProcess:
        if self.probe_raises is not None:
            raise self.probe_raises
        stdout = self.probe_stdout
        if stdout is None:
            stdout = (
                json.dumps(
                    {
                        "device": self.probe_device,
                        "cuda_build": self.probe_cuda_build,
                        "device_count": self.probe_device_count,
                    }
                )
                + "\n"
            )
        return subprocess.CompletedProcess(args, self.probe_returncode, stdout, self.probe_stderr)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(Call(argv=list(args), timeout=timeout))
        if self.is_probe(args):
            # Ahead of `raises`, which describes the training run: a test that
            # makes training time out is not also asking the probe to.
            return self.probe_result(args)
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
            if self.device is not None:
                payload["device"] = self.device
            if self.drop_fraction:
                payload.pop("supervised_token_fraction")
            Path(config["metrics_out"]).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def trainer(monkeypatch, tmp_path) -> FakeTrainer:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
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
    """Two calls now, both in the same environment, in this order.

    The device probe goes first and deliberately: it is what lets the run say
    where it is about to train before the training happens, rather than after.
    It runs in the train environment because that is the torch whose answer
    matters -- asking any other interpreter would answer about a machine this
    run is not using.
    """
    request = request_for()
    run_tune(request)

    assert len(trainer.calls) == 2
    probe, training = trainer.calls
    assert probe.argv[:2] == ["python", "-c"]
    assert "cuda.is_available" in probe.argv[2]
    assert probe.timeout < training.timeout
    assert training.argv[0] == "python"
    assert Path(training.argv[1]).name == "train_script.py"
    assert training.timeout == request.timeout_s
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


def test_which_terminator_was_used_is_recorded_from_the_chat_template(script_namespace):
    """Distinct from the test above on purpose: that one drives the fallback
    path, where `terminator_source` genuinely is "tokenizer_eos" -- a
    `build_examples` that hardcoded that string into the metrics dict instead
    of using the value `turn_terminator` returned would still pass it. This
    drives the chat-template path, where the two values differ, so the
    metrics dict has to be carrying the real one.
    """
    build_examples = script_namespace["build_examples"]
    rows = [{"prompt": "a", "completion": "x", "source_line": 1}]

    _, _, _, terminator = build_examples(FakeTemplateTokenizer(), rows, 64, True)

    assert terminator["source"] == "chat_template"


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
    trainer.stderr = "ValueError: the training split is empty"
    trainer.write_model = False

    result = run_tune(request_for())

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.FAILED
    assert "exited 1" in check.detail and "the training split is empty" in check.detail
    assert result.returncode == 1
    assert result.model_dir is None


# ---------------------------------------------------------------------------
# ... but an accelerator or host-capacity failure is not a verdict about the run
# ---------------------------------------------------------------------------
#
# This stderr used to be the fixture for the test above, and the outcome it
# pinned was `failed`: "training exited 1: torch.OutOfMemoryError: CUDA out of
# memory" recorded as a statement about the method and the data. `-9` was
# already handled -- the killed branch reads it through `litetune.exits` -- but
# an allocation that fails inside torch raises, and an uncaught exception exits
# 1 like any other. Unreachable while the run was pinned to the CPU; reachable
# now that it is not.
#
# CPU training is a first-class destination here, not an edge case, and the
# host's own allocator raises the identical shape of exception -- `RuntimeError:
# DefaultCPUAllocator: not enough memory` -- for the identical reason. Judging
# it as a failed run while a GPU allocation failure on the same box gets a
# shrug was an asymmetry this list creates, not the toolchain.


@pytest.mark.parametrize(
    "stderr",
    [
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "torch.cuda.OutOfMemoryError: CUDA out of memory",
        # Each alternative in `_GPU_FAILURE_RE` on its own, so that deleting
        # one is a failing test rather than a case another alternative happens
        # to catch: the exception class without the message, and the message
        # without the class.
        "torch.OutOfMemoryError: out of memory",
        "RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB",
        "RuntimeError: CUDA error: no kernel image is available for execution on the device",
        "RuntimeError: CUDA error: invalid device ordinal",
        "RuntimeError: CUDA error: no CUDA-capable device is detected",
        "RuntimeError: CUDA error: out of memory",
        "RuntimeError: CUDA driver version is insufficient for the CUDA runtime version",
        "RuntimeError: HIP out of memory. Tried to allocate 2.00 GiB",
        # The host allocator, not the device: exact wording, not a bare "not
        # enough memory" that would also match a message this list was never
        # meant to speak for.
        "RuntimeError: DefaultCPUAllocator: not enough memory: you tried to allocate "
        "402653184 bytes.",
        "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling "
        "`cublasCreate(handle)`",
        "RuntimeError: cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
    ],
)
def test_an_accelerator_failure_is_could_not_check_not_a_failed_method(
    trainer, request_for, stderr
):
    trainer.returncode = 1
    trainer.stderr = stderr
    trainer.write_model = False

    result = run_tune(request_for())

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.UNCHECKED, check.detail
    # The verdict is withheld *and* the reason travels: a `could_not_check`
    # with no stderr in it is unactionable, and the matched shape is what
    # says which of these it was.
    assert "says nothing about the method or the data" in check.detail
    assert stderr[-40:] in check.detail
    assert check.observed["matched"] in stderr
    assert result.outcome is Outcome.UNCHECKED
    assert result.model_dir is None


def test_an_accelerator_failure_records_which_device_failed(trainer, request_for):
    """The device in `observed` is read by a human deciding what to do next,
    and until this test nothing pinned it: replacing it with `None` passed
    every other test in the suite.

    It matters because the advice differs. An accelerator failure on a box the
    probe resolved to `cuda` means the GPU could not carry this run -- fewer
    tokens, a smaller batch, or more VRAM. The same message with no device
    behind it means something else happened, and a reader who cannot tell them
    apart cannot act on either.
    """
    trainer.probe_device = "cuda"
    trainer.returncode = 1
    trainer.stderr = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    trainer.write_model = False

    result = run_tune(request_for())

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.UNCHECKED, check.detail
    assert check.observed["device"] == "cuda"


@pytest.mark.parametrize("method", ["full", "lora"])
def test_an_accelerator_failure_produces_the_same_report_for_either_method(
    trainer, request_for, method, tmp_path
):
    """It returns where the killed branch returns, and for the same reason.

    Falling through instead reaches `_merge_check`, which on a LoRA run with
    no adapter records a *failed* merge -- "the learned delta is
    unrecoverable" -- while a full fine-tune's merge check short-circuits to
    passed. One machine event, two different reports, chosen by `--method`.
    """
    trainer.returncode = 1
    trainer.stderr = "torch.OutOfMemoryError: CUDA out of memory"
    trainer.write_model = False
    trainer.write_adapter = False

    result = run_tune(request_for(method=method, output_dir=tmp_path / method))

    assert [c.name for c in result.checks.checks] == [ENV_CHECK, TRAINING_CHECK, MASKING_CHECK]
    assert not [c for c in result.checks.checks if c.outcome is Outcome.FAILED]
    assert result.outcome is Outcome.UNCHECKED


@pytest.mark.parametrize(
    "stderr",
    [
        "RuntimeError: expected scalar type Float but found BFloat16",
        # The one that decides the shape of `_GPU_FAILURE_RE`. A device-side
        # assert is the GPU face of an out-of-range index -- a token id past
        # the embedding table -- and the identical defect on the CPU raises
        # `IndexError` and is a failed run. Matching the `CUDA error:` prefix
        # would make the same data bug a verdict on a laptop and a shrug on a
        # GPU box, with "says nothing about the method or the data" attached
        # to the one case where it says exactly that.
        "RuntimeError: CUDA error: device-side assert triggered",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
    ],
)
def test_an_ordinary_failure_is_still_a_failure(trainer, request_for, stderr):
    """The other side of the branch. `_GPU_FAILURE_RE` widened to match
    everything would make every training failure unreportable, which is the
    opposite error and just as bad.
    """
    trainer.returncode = 1
    trainer.stderr = stderr
    trainer.write_model = False

    result = run_tune(request_for())

    assert check_named(result, TRAINING_CHECK).outcome is Outcome.FAILED
    assert result.outcome is Outcome.FAILED


def test_an_accelerator_message_on_a_clean_exit_changes_nothing(trainer, request_for):
    """Consulted only on a non-zero exit. A run that recovered from an OOM,
    retried at a smaller batch and exited zero with the message still in its
    stderr is a run that finished -- reading stderr first would turn it into
    a non-result.
    """
    trainer.returncode = 0
    trainer.stderr = "torch.OutOfMemoryError: CUDA out of memory (caught, retried)"

    result = run_tune(request_for())

    assert check_named(result, TRAINING_CHECK).outcome is Outcome.PASSED
    assert result.model_dir is not None


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
    """A departure from the default is recorded -- and nothing more.

    `--dtype float32` must not be told it is "not the pair the export and
    evaluation paths use": evaluate.py's float reference always loads at
    float32 regardless of training dtype, so it is the one value that already
    matches it. It must not be told which dtype the published numbers were
    taken at either, in either direction: MEASUREMENTS.md gives a dtype for
    exactly one run -- the second-family banking77 one, trained in float32 --
    and says nothing about the headline table's. Attention still defaults to
    eager here, so its own, separate limitation must not fire.
    """
    result = run_tune(request_for(dtype="float32"))

    assert trainer.configs[0]["dtype"] == "float32"
    departure = [text for text in result.limitations if "trains with dtype 'float32'" in text]
    assert len(departure) == 1, result.limitations
    assert "MEASUREMENTS.md" in departure[0]
    assert "every published number" not in departure[0]
    assert not any("fluent, wrong" in text for text in result.limitations)


def test_a_different_attention_implementation_is_named_as_a_limitation(trainer, request_for):
    result = run_tune(request_for(attn_implementation="sdpa"))

    assert trainer.configs[0]["attn_implementation"] == "sdpa"
    assert any("fluent, wrong" in text for text in result.limitations)


def test_training_at_the_default_dtype_states_what_is_and_is_not_established(trainer, request_for):
    """The two corrections to the limitation this branch shipped: the reason
    bfloat16 beats float16 *is* in the repo (spec.py's overflow, cli.py's
    NaN), so "not established" must be scoped to bfloat16 versus float32
    only; and export.py passing no dtype is not the same claim as export not
    loading in bfloat16, which nothing here establishes.
    """
    result = run_tune(request_for())

    text = next(t for t in result.limitations if "optimiser's moments are bfloat16" in t)
    assert "float16 overflows" in text
    assert "NaN in float16" in text
    assert "bfloat16 versus float32" in text
    assert "export passes no dtype" in text
    assert "the export and evaluation paths do not load in it" not in text


def test_the_default_dtype_comment_does_not_claim_export_loads_in_bfloat16():
    """A pure comment, not a runtime value -- pinned by reading the module's
    own source, the way `test_manifest.py` pins `pyproject.toml`'s text.
    `DEFAULT_DTYPE`'s comment used to claim the export and evaluation paths
    use bfloat16 the same way they use eager attention; only the attention
    half is true (evaluate.py:653 passes it through; evaluate.py:652 is an
    unconditional `torch_dtype=torch.float32`).
    """
    import inspect

    import litetune.tune as tune_module

    source = inspect.getsource(tune_module)
    assert "bfloat16 and eager attention, because the export and evaluation paths use" not in source
    assert "Eager attention, because the export and evaluation paths use it too" in source


def test_the_tune_request_docstring_does_not_claim_dtype_has_a_pair():
    """R6's sixth copy of the claim: `help(TuneRequest)` used to say `dtype`
    and `attn_implementation` "default to the pair the export and evaluation
    paths use". Only `attn_implementation` has a pair to match -- evaluate.py
    loads the float reference at an unconditional `float32` regardless of
    training dtype, and export.py passes no dtype to the exporter at all.
    """
    import inspect

    from litetune.tune import TuneRequest

    doc = inspect.getdoc(TuneRequest) or ""
    assert "default to the pair the export and evaluation paths use" not in doc
    assert "such pair to default to" in doc


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
import json
import os

bfloat16 = "bfloat16"
float32 = "float32"
long = "long"


def _log(record):
    path = os.environ["LITETUNE_STUB_LOG"]
    with open(path, "a", encoding="utf-8") as sink:
        sink.write(json.dumps(record) + "\\n")


def manual_seed(seed):
    return seed


class _Tensor(list):
    # A list that tracks its own device and answers `.to(device)` the way a
    # tensor does: with itself when the device is unchanged, with a new
    # object when it is not. Returning `self` unconditionally made a dropped
    # `.to()` call in the training script indistinguishable from one that ran.
    def __init__(self, values, device="cpu"):
        super().__init__(values)
        self.device = device

    def to(self, device):
        if device == self.device:
            return self
        return _Tensor(list(self), device=device)


def tensor(values, dtype=None):
    return _Tensor(values)


class _Cuda:
    @staticmethod
    def is_available():
        # A flag read from the environment, not a hardcoded `False`: this
        # stub runs in its own subprocess and cannot share `conftest.py`'s
        # `fake_torch`, but it answers the same question the same way -- a
        # boolean that can say "there is a GPU", which `run_real_script`'s
        # `cuda` parameter sets before the subprocess starts.
        return os.environ.get("LITETUNE_STUB_CUDA") == "1"


cuda = _Cuda()


class _AdamW:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.steps = 0
        # Logged so a test can assert the model moved to its device *before*
        # this ran: parameters an optimiser is built against, then moved
        # afterwards, silently end up split across two devices.
        _log({"event": "optimiser_init", "n_params": len(self.params)})

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

    def to(self, device):
        _log({"event": "to", "device": device})
        return self

    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        _log({
            "event": "forward",
            "rows": len(labels),
            "width": len(labels[0]),
            # Where each of the three actually arrived, not just that `.to()`
            # was called on it -- the doubles hand back a new object only
            # when the device changed, so a dropped `.to()` call shows up
            # here as "cpu" while the model sits on something else.
            "input_ids_device": getattr(input_ids, "device", None),
            "attention_mask_device": getattr(attention_mask, "device", None),
            "labels_device": getattr(labels, "device", None),
        })
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


def run_real_script(
    request: TuneRequest, stub_env, cuda: bool = False
) -> subprocess.CompletedProcess:
    """Runs `_TRAIN_SCRIPT` for real, against the stub modules `stub_env` wrote.

    `cuda` sets the one flag the stub `torch.cuda.is_available()` reads. The
    stub lives in its own subprocess and cannot share `conftest.py`'s
    `fake_torch`, but this is the same shape: a boolean parameter, not a
    hardcoded `False` unable to express "there is a GPU".
    """
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
        env={
            "PATH": "",
            "PYTHONPATH": str(stubs),
            "LITETUNE_STUB_LOG": str(log),
            "LITETUNE_STUB_CUDA": "1" if cuda else "0",
        },
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


def test_the_real_script_puts_the_model_on_a_device_and_records_it(request_for, stub_env):
    from litetune.tune import read_metrics

    request = request_for()
    assert run_real_script(request, stub_env).returncode == 0

    moved = [e for e in stub_log(stub_env) if e["event"] == "to"]
    assert [m["device"] for m in moved] == ["cpu"]
    assert read_metrics(request.output_dir / "metrics.json").device == "cpu"


def test_the_real_script_trains_on_cuda_when_the_stub_says_there_is_one(request_for, stub_env):
    """Kills four mutants at once, all invisible against the cuda-less stub
    above: the hardcoded `"cpu"` `training_device` could fall back to,
    `model.to(device)` dropped or aimed at `"meta"`, `.to(device)` dropped off
    `input_ids`, `attention_mask` or `labels` before the forward pass, and the
    metrics payload's `"device"` key replaced with a literal `"cpu"`. Every
    one of them happens to still say "cpu" against a cuda-less fixture --
    only one that can say "there is a GPU" tells them apart.
    """
    from litetune.tune import read_metrics

    request = request_for()
    assert run_real_script(request, stub_env, cuda=True).returncode == 0

    log = stub_log(stub_env)
    moved = [e for e in log if e["event"] == "to"]
    assert [m["device"] for m in moved] == ["cuda"]

    forward = next(e for e in log if e["event"] == "forward")
    assert forward["input_ids_device"] == "cuda"
    assert forward["attention_mask_device"] == "cuda"
    assert forward["labels_device"] == "cuda"

    assert read_metrics(request.output_dir / "metrics.json").device == "cuda"


def test_the_real_script_moves_the_model_before_building_the_optimiser(request_for, stub_env):
    """An optimiser built before the model is moved is constructed against
    parameters that then move out from under it -- silently, since nothing
    about the run fails. `test_..._puts_the_model_on_a_device...` already
    kills a move that is dropped outright; this is the one mutant that
    survives even then, by moving the model back to front.
    """
    request = request_for()
    assert run_real_script(request, stub_env).returncode == 0

    events = [e["event"] for e in stub_log(stub_env)]
    assert events.index("to") < events.index("optimiser_init")


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


def test_the_recorded_terminator_reaches_bundles_stop_tokens(request_for, stub_env, tmp_path):
    """The path README publishes: "`tune` records it at `turn_terminator.text`
    in `metrics.json`, and `bundle` carries it into `contract.json`'s
    `stop_tokens`". Every other test of either end feeds a hand-written
    `metrics.json` fixture, so the two ends have only ever been shown to
    agree with each other, not with what `tune` actually writes. This test
    joins the hops: the real training script (via `run_real_script`, not
    `FakeTrainer`'s canned payload) writes the real `metrics.json`, and that
    exact file is handed to `bundle` through `--train-metrics`.
    """
    from litetune.cli import main

    request = request_for()
    proc = run_real_script(request, stub_env)
    assert proc.returncode == 0, proc.stderr

    metrics_path = request.output_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    terminator_text = metrics["turn_terminator"]["text"]
    # A real, computed decode -- not a placeholder a mutant renaming the key
    # could still satisfy by accident.
    assert terminator_text == "<99>"

    model_file = tmp_path / "m.litertlm"
    model_file.write_text("{}", encoding="utf-8")
    declarations = tmp_path / "d.json"
    declarations.write_text("[]", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    main(
        [
            "bundle",
            "--output-dir",
            str(bundle_dir),
            "--model",
            str(model_file),
            "--declarations",
            str(declarations),
            "--prompt-mode",
            "prerendered",
            # A family with no extra stop tokens of its own, so the assertion
            # below is exactly the recorded terminator and nothing added.
            "--base-model",
            "google/gemma-3-270m-it",
            "--base-model-revision",
            "0123456789abcdef0123456789abcdef01234567",
            "--train-metrics",
            str(metrics_path),
        ]
    )

    contract = json.loads((bundle_dir / "contract.json").read_text(encoding="utf-8"))
    assert contract["stop_tokens"] == [terminator_text]


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


# ---------------------------------------------------------------------------
# Where the run happens, and that it says so
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# What the probe answered, and where the answer goes
# ---------------------------------------------------------------------------
#
# Three destinations, all from one call: the config the training script is
# handed, the report the run writes, and the pre-run note. Each of them used to
# be unpinned -- `device = None` in `run_tune`, a literal `"device": "cpu"` in
# `TuneRequest.config`, and deleting the probe outright all survived the whole
# suite, because the fake could not answer the probe at all and every test ran
# against a failure.


def notes(seen) -> list[str]:
    return [e.data.get("message", "") for e in seen if e.kind == "note"]


def run_with_events(request):
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)
    return run_tune(request, events=events), seen


def test_the_probes_answer_reaches_the_script_and_the_report(trainer, request_for, tmp_path):
    """`cuda`, so that every hardcoded `"cpu"` on the way is visible.

    The script is *told* the device rather than left to decide: that is the
    whole point of asking before the run instead of reading `metrics.device`
    after it.
    """
    trainer.probe_device = "cuda"
    request = request_for()

    result = run_tune(request)

    assert result.device == "cuda"
    assert trainer.configs[0]["device"] == "cuda"
    assert result.as_dict()["request"]["device"] == "cuda"
    written = json.loads((request.output_dir / "train_config.json").read_text(encoding="utf-8"))
    assert written["device"] == "cuda"


def test_a_cpu_answer_reaches_the_script_too(trainer, request_for):
    """The other value, so that `"device": "cuda"` hardcoded anywhere on the
    path is as visible as `"cpu"` is."""
    trainer.probe_device = "cpu"
    run_tune(request_for())
    assert trainer.configs[0]["device"] == "cpu"


def test_a_probe_that_cannot_answer_is_a_limitation_not_a_device(trainer, request_for):
    """`None` is not "cpu", and it must not be silent either. `logging` alone
    reaches nothing the report carries, so a run whose device was never
    established says so where the rest of the run's caveats are."""
    trainer.probe_returncode = 1
    trainer.probe_stderr = "ModuleNotFoundError: No module named 'torch'"

    result = run_tune(request_for())

    assert result.device is None
    assert trainer.configs[0]["device"] is None
    assert any(
        "device probe" in text and "was not established" in text for text in result.limitations
    ), result.limitations
    assert any("No module named" in text for text in result.limitations), result.limitations


def test_a_killed_probe_is_not_reported_as_an_exit_status(trainer, request_for):
    """`-9` is a signal, not a status the probe chose. See `litetune.exits`."""
    trainer.probe_returncode = -9

    result = run_tune(request_for())

    assert result.device is None
    limitation = next(text for text in result.limitations if "device probe" in text)
    assert "exited -9" not in limitation
    assert "SIGKILL" in limitation


def test_a_probe_that_times_out_is_not_a_device(trainer, request_for):
    trainer.probe_raises = subprocess.TimeoutExpired(cmd="python", timeout=30)

    result = run_tune(request_for())

    assert result.device is None
    assert any("TimeoutExpired" in text for text in result.limitations), result.limitations


def test_a_banner_before_the_answer_does_not_destroy_it(trainer, request_for):
    """A stage environment is free to print on startup. The answer is the last
    line the probe writes, and reading the whole of stdout threw away a good
    answer because something else spoke first."""
    trainer.probe_stdout = (
        "warning: overriding a pinned dependency\n"
        '{"device": "cuda", "cuda_build": "12.4", "device_count": 1}\n'
    )

    result = run_tune(request_for())

    assert result.device == "cuda"
    assert not any("device probe" in text for text in result.limitations)


def test_a_torch_that_cannot_reach_its_gpu_is_not_the_same_as_no_gpu(trainer, request_for):
    """`is_available()` answers False for a CPU-only wheel, a container with no
    `--gpus`, and a driver too old -- all of which are "torch cannot reach a
    GPU here", not "this machine has none". The run still happens on the CPU;
    what changes is that the report says which observation it was.
    """
    trainer.probe_device = "cpu"
    trainer.probe_cuda_build = "12.4"
    trainer.probe_device_count = 0

    result = run_tune(request_for())

    assert result.device == "cpu"
    assert any(
        "CUDA 12.4 build" in text and "not the same observation" in text
        for text in result.limitations
    ), result.limitations


def test_a_cpu_only_wheel_reporting_cpu_carries_no_such_limitation(trainer, request_for):
    """The other half: a wheel built without CUDA answering "cpu" is an
    ordinary CPU machine and has nothing to explain."""
    trainer.probe_device = "cpu"
    trainer.probe_cuda_build = None

    result = run_tune(request_for())

    assert not any("not the same observation" in text for text in result.limitations)


def test_the_probe_runs_even_without_provisioning(trainer, request_for, tmp_path):
    """`litetune tune --no-provision` over an environment that is already there
    is a run whose device is knowable. The probe provisions nothing -- gating it
    on `auto_provision` made such a run report no device at all.
    """
    mark_provisioned(envs.TRAIN)
    trainer.probe_device = "cuda"

    result = run_tune(request_for(auto_provision=False))

    assert result.device == "cuda"
    assert check_named(result, TRAINING_CHECK).outcome is Outcome.PASSED


# ---------------------------------------------------------------------------
# The pre-run note, and the two details that quote the same hint
# ---------------------------------------------------------------------------


def test_a_cpu_probe_warns_before_the_wait_not_after_it(trainer, request_for):
    trainer.probe_device = "cpu"

    _, seen = run_with_events(request_for(dtype="bfloat16"))

    warnings = [m for m in notes(seen) if "bfloat16 on the CPU" in m]
    assert len(warnings) == 1
    assert warnings[0].startswith("training will run bfloat16 on the CPU")
    assert "--dtype float32" in warnings[0]


def test_an_unanswered_probe_still_warns_and_says_it_is_a_maybe(trainer, request_for):
    """The case this gate used to lose. `None` is not "cuda", and the script's
    own fallback resolves to cpu on every machine without a reachable GPU --
    so the run most likely to spend six hours is the one that was told
    nothing. "may", not "will": the device is genuinely not established.
    """
    trainer.probe_raises = OSError("no interpreter")

    _, seen = run_with_events(request_for(dtype="bfloat16"))

    warnings = [m for m in notes(seen) if "bfloat16 on the CPU" in m]
    assert len(warnings) == 1
    assert warnings[0].startswith("training may run bfloat16 on the CPU")
    assert "not established yet" in warnings[0]


def test_a_cuda_probe_warns_about_nothing(trainer, request_for):
    trainer.probe_device = "cuda"

    _, seen = run_with_events(request_for(dtype="bfloat16"))

    assert not [m for m in notes(seen) if "bfloat16 on the CPU" in m]


def test_a_float32_run_is_not_told_to_take_the_advice_it_already_took(trainer, request_for):
    trainer.probe_device = "cpu"

    _, seen = run_with_events(request_for(dtype="float32"))

    assert not [m for m in notes(seen) if "bfloat16 on the CPU" in m]


def test_a_timeout_carries_the_hint_even_when_the_probe_said_nothing(trainer, request_for):
    """This ending returns before any metrics file is opened, so the pre-run
    probe is the only thing that knows anything about the device -- and a
    six-hour non-result is exactly the ending that needs the hint."""
    trainer.probe_raises = OSError("no interpreter")
    trainer.raises = subprocess.TimeoutExpired(cmd="python", timeout=10)

    result = run_tune(request_for(dtype="bfloat16"))

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    assert "--dtype float32" in check.detail
    # And it must not turn into a claim about a CPU nobody observed.
    assert "its device was never established" in check.detail
    assert "it was running bfloat16 on the CPU" not in check.detail


def test_a_timeout_on_an_observed_cpu_says_so_plainly(trainer, request_for):
    trainer.probe_device = "cpu"
    trainer.raises = subprocess.TimeoutExpired(cmd="python", timeout=10)

    detail = check_named(run_tune(request_for(dtype="bfloat16")), TRAINING_CHECK).detail

    assert "it was running bfloat16 on the CPU" in detail
    assert "never established" not in detail


def test_a_timeout_on_cuda_carries_no_dtype_hint(trainer, request_for):
    trainer.probe_device = "cuda"
    trainer.raises = subprocess.TimeoutExpired(cmd="python", timeout=10)

    result = run_tune(request_for(dtype="bfloat16"))

    assert "--dtype float32" not in check_named(result, TRAINING_CHECK).detail


def test_a_killed_run_prefers_what_the_script_reported_over_the_prediction(trainer, request_for):
    """The probe said cuda; the script fell back and reported cpu. What ran is
    what the detail is about.

    `-15` (SIGTERM), not `-9`: SIGKILL carries its own hint -- see
    `test_a_sigkill_does_not_carry_the_speed_hint_too` -- and this test is
    about the confirmed-over-predicted wording, not about that one.
    """
    trainer.probe_device = "cuda"
    trainer.device = "cpu"
    trainer.returncode = -15

    result = run_tune(request_for(dtype="bfloat16"))

    check = check_named(result, TRAINING_CHECK)
    assert check.outcome is Outcome.UNCHECKED
    assert "--dtype float32" in check.detail
    assert "it ran bfloat16 on the CPU" in check.detail


def test_a_killed_run_that_established_no_device_says_that_instead(trainer, request_for):
    trainer.probe_raises = OSError("no interpreter")
    trainer.write_metrics = False
    trainer.returncode = -15

    detail = check_named(run_tune(request_for(dtype="bfloat16")), TRAINING_CHECK).detail

    assert "--dtype float32" in detail
    assert "its device was never established" in detail
    assert "it ran bfloat16 on the CPU" not in detail


def test_a_killed_run_with_only_a_prediction_does_not_claim_it_ran(trainer, request_for):
    """The probe predicted the CPU; the run was killed before `metrics.json`
    -- and so `metrics.device` -- ever existed to confirm it. "it ran" is a
    past tense nothing here established; the wording says a prediction, not
    an observation.
    """
    trainer.probe_device = "cpu"
    trainer.write_metrics = False
    trainer.returncode = -15

    detail = check_named(run_tune(request_for(dtype="bfloat16")), TRAINING_CHECK).detail

    assert "--dtype float32" in detail
    assert "it ran bfloat16 on the CPU" not in detail
    assert "predicted the CPU" in detail
    assert "killed before its own device was confirmed" in detail


def test_a_sigkill_does_not_carry_the_speed_hint_too(trainer, request_for):
    """`reading.describe` already named the near-certain cause of a SIGKILL --
    the machine's out-of-memory killer -- and a speed hint stacked next to it
    reads as a second, contradictory story: a box killed while loading the
    checkpoint, seconds in, never ran a single matmul, slow or otherwise. Same
    device shape as the confirmed-cpu test above, `-9` instead of `-15`.
    """
    trainer.probe_device = "cuda"
    trainer.device = "cpu"
    trainer.returncode = -9

    detail = check_named(run_tune(request_for(dtype="bfloat16")), TRAINING_CHECK).detail

    assert "out-of-memory killer" in detail
    assert "--dtype float32" not in detail
    assert "it ran bfloat16 on the CPU" not in detail


def test_a_killed_run_on_cuda_carries_no_dtype_hint(trainer, request_for):
    trainer.probe_device = "cuda"
    trainer.device = "cuda"
    trainer.returncode = -9

    result = run_tune(request_for(dtype="bfloat16"))

    assert "--dtype float32" not in check_named(result, TRAINING_CHECK).detail


def test_the_script_trains_on_cuda_when_there_is_one(script_namespace):
    """The fallback, reached when the parent has no answer to give: a probe
    that could not run, or an environment nobody probed. Without it a GPU box
    would train on the CPU because a sub-second probe failed."""
    training_device = script_namespace["training_device"]
    assert training_device(fake_torch(cuda=True)) == "cuda"
    assert training_device(fake_torch(cuda=False)) == "cpu"


def test_the_script_takes_the_parents_answer_over_its_own(script_namespace):
    """`given` wins, and the fake torch is set to disagree so that it has to.

    Against a torch that says "no GPU", `if given is not None` reduced to `if
    False` still answers "cpu" and nothing notices -- which is how the trust
    branch stayed unpinned on this side while `evaluate.py`'s had a test.
    """
    training_device = script_namespace["training_device"]
    assert training_device(fake_torch(cuda=False), given="cuda") == "cuda"
    assert training_device(fake_torch(cuda=True), given="cpu") == "cpu"


def test_the_device_is_recorded_in_the_metrics():
    from litetune.tune import TrainingMetrics

    payload = {
        "n_examples": 1,
        "supervised_tokens": 1,
        "total_tokens": 2,
        "masked_tokens": 1,
        "supervised_token_fraction": 0.5,
        "epochs": [],
        "device": "cuda",
    }
    assert TrainingMetrics.from_dict(payload).device == "cuda"
    assert TrainingMetrics.from_dict(payload).as_dict()["device"] == "cuda"
    # Older metrics files have no such field; absent is absent, not "cpu".
    del payload["device"]
    assert TrainingMetrics.from_dict(payload).device is None


def test_bfloat16_on_the_cpu_is_named_as_a_limitation(trainer, request_for):
    """bfloat16 is the default because a fine-tune in float16 overflows where
    bfloat16 does not, and a 270M model's loss goes to NaN in float16 while
    bfloat16 holds -- not because it matches export or evaluation, which
    `TuneRequest`'s own docstring and `BFLOAT16_CPU_HINT`'s comment both
    disclaim. On a CPU it is also, on the one machine measured, a
    single-threaded matmul: a 300-step LoRA run sat on one core for 52
    minutes without finishing, and the same run in float32 took 307 s on ten
    threads. The default stands regardless, but a CPU run is told what it is
    paying and which flag buys it back."""
    trainer.device = "cpu"
    result = run_tune(request_for(dtype="bfloat16"))
    assert any("--dtype float32" in text for text in result.limitations)


def test_a_cuda_run_carries_no_dtype_warning(trainer, request_for):
    trainer.device = "cuda"
    result = run_tune(request_for(dtype="bfloat16"))
    assert not any("--dtype float32" in text for text in result.limitations)


def test_a_cpu_run_at_float32_carries_no_dtype_warning(trainer, request_for):
    """The other half of the condition: `device == "cpu"` alone is not enough
    to fire the hint. A run that already took the advice -- `--dtype
    float32` on the CPU -- must not be told to take it again."""
    trainer.device = "cpu"
    result = run_tune(request_for(dtype="float32"))
    assert not any("trained bfloat16 on the CPU" in text for text in result.limitations)


# ---------------------------------------------------------------------------
# Five endings, and whether each still tells the device story
# ---------------------------------------------------------------------------
#
# The note and the limitation both read `result.metrics.device`, and used to
# sit at the very end of `run_tune`, after the killed early return -- so on
# that one ending they never ran. Two of the five endings moved, not one: the
# killed return, and the accelerator-failure return added alongside it, which
# sits after this block too. The timeout and the blocked-start returns both
# return before a metrics file is ever opened, and the fifth ending, the
# bottom of the function, was already downstream. The fixture below makes a
# killed run that wrote metrics, which is a narrow case in production -- the
# real script writes `metrics.json` last, after the checkpoint -- but it is
# the case the placement exists for, and the other four are here so that a
# regression in any of them is a failing test rather than a silence.


def test_the_device_story_appears_on_a_clean_run(trainer, request_for):
    trainer.device = "cpu"
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = run_tune(request_for(), events=events)

    assert check_named(result, TRAINING_CHECK).outcome is Outcome.PASSED
    assert result.metrics is not None and result.metrics.device == "cpu"
    device_notes = [
        e for e in seen if e.kind == "note" and e.data.get("message", "").startswith("trained on")
    ]
    assert len(device_notes) == 1
    # The message and the typed `device` field both have to carry the real
    # answer: a note that prints "trained on cpu" while `device=None` (or the
    # reverse) is still wrong, just wrong in a way string-matching the
    # message alone would miss.
    assert device_notes[0].data["message"] == "trained on cpu"
    assert device_notes[0].data["device"] == "cpu"
    assert any("trained bfloat16 on the CPU" in text for text in result.limitations)


def test_the_device_story_appears_when_training_is_killed(trainer, request_for):
    """`-9` returns before the bottom of `run_tune` -- but the metrics file a
    killed process wrote (if it got that far) was already on disk, and the
    story about it must not be skipped along with the verdict."""
    trainer.returncode = -9
    trainer.device = "cpu"
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = run_tune(request_for(), events=events)

    assert check_named(result, TRAINING_CHECK).outcome is Outcome.UNCHECKED
    assert result.metrics is not None and result.metrics.device == "cpu"
    device_notes = [
        e for e in seen if e.kind == "note" and e.data.get("message", "").startswith("trained on")
    ]
    assert len(device_notes) == 1
    # The message and the typed `device` field both have to carry the real
    # answer: a note that prints "trained on cpu" while `device=None` (or the
    # reverse) is still wrong, just wrong in a way string-matching the
    # message alone would miss.
    assert device_notes[0].data["message"] == "trained on cpu"
    assert device_notes[0].data["device"] == "cpu"
    assert any("trained bfloat16 on the CPU" in text for text in result.limitations)


def test_the_device_story_appears_on_an_accelerator_failure(trainer, request_for):
    """The other ending that moved alongside the killed one: `_GPU_FAILURE_RE`
    also returns before the bottom of `run_tune`, and it sits after the
    device-story block for the same reason the killed branch does."""
    trainer.returncode = 1
    trainer.stderr = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    trainer.write_model = False
    trainer.device = "cpu"
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = run_tune(request_for(), events=events)

    assert check_named(result, TRAINING_CHECK).outcome is Outcome.UNCHECKED
    assert result.metrics is not None and result.metrics.device == "cpu"
    device_notes = [
        e for e in seen if e.kind == "note" and e.data.get("message", "").startswith("trained on")
    ]
    assert len(device_notes) == 1
    assert device_notes[0].data["message"] == "trained on cpu"
    assert device_notes[0].data["device"] == "cpu"
    assert any("trained bfloat16 on the CPU" in text for text in result.limitations)


def test_the_device_story_appears_when_training_exits_nonzero(trainer, request_for):
    trainer.returncode = 1
    trainer.device = "cpu"
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = run_tune(request_for(), events=events)

    assert check_named(result, TRAINING_CHECK).outcome is Outcome.FAILED
    assert result.metrics is not None and result.metrics.device == "cpu"
    device_notes = [
        e for e in seen if e.kind == "note" and e.data.get("message", "").startswith("trained on")
    ]
    assert len(device_notes) == 1
    # The message and the typed `device` field both have to carry the real
    # answer: a note that prints "trained on cpu" while `device=None` (or the
    # reverse) is still wrong, just wrong in a way string-matching the
    # message alone would miss.
    assert device_notes[0].data["message"] == "trained on cpu"
    assert device_notes[0].data["device"] == "cpu"
    assert any("trained bfloat16 on the CPU" in text for text in result.limitations)


def test_the_device_story_is_silent_with_no_metrics_file(trainer, request_for):
    trainer.write_metrics = False
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = run_tune(request_for(), events=events)

    assert result.metrics is None
    assert not any(e.kind == "note" and "trained on" in e.data.get("message", "") for e in seen)
    assert not any("trained bfloat16 on the CPU" in text for text in result.limitations)


def test_the_device_story_is_silent_when_metrics_predate_the_field(trainer, request_for):
    """`trainer.device` is left at its default `None`: the payload
    `FakeTrainer` writes then carries no "device" key at all -- indistinguishable
    from a real `metrics.json` written before the field existed."""
    seen: list = []
    events = EventStream(echo_json=False)
    events.subscribe(seen.append)

    result = run_tune(request_for(), events=events)

    assert result.metrics is not None
    assert result.metrics.device is None
    assert not any(e.kind == "note" and "trained on" in e.data.get("message", "") for e in seen)
    assert not any("trained bfloat16 on the CPU" in text for text in result.limitations)
