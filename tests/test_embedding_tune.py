"""`tune` for google/embeddinggemma-2, with no torch and no model in this process.

`StageEnv.run` is faked by `FakeEmbedToolchain`, this file's own: it answers
the device probe and, for the training script, reads the spec the stage wrote and
writes what a real run would -- a checkpoint directory and the metrics -- or
refuses, exits non-zero, times out. The script's helpers import nothing and are
`exec`'d here against plain values.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import machine, mark_provisioned, probe_answer

from litetune import envs, models, wheels
from litetune.checks import Outcome
from litetune.cli import main
from litetune.decisions import Decision, Option
from litetune.embedding_tune import (
    _EMBED_TRAIN_SCRIPT,
    DATA_CHECK,
    DECIDER_PREFIX,
    DIRECTION,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_TYPE,
    EMBEDDING_REVISION,
    MODEL_CHECK,
    PROMPT_NAME,
    RECIPE,
    REFUSED_EXIT,
    EmbeddingRecord,
    EmbeddingTuneError,
    EmbeddingTuneRequest,
    LabelledText,
    labels_for,
    read_embedding_record,
    run_embedding_tune,
    write_embedding_record,
)
from litetune.tune import DEVICE_CHECK, ENV_CHECK, TRAINING_CHECK

# ---------------------------------------------------------------------------
# A fake EMBED_TRAIN environment
# ---------------------------------------------------------------------------


@dataclass
class FakeEmbedToolchain:
    """Stands in for `envs.StageEnv.run`: the device probe, then the training script."""

    probe_device: str | None = "cpu"
    probe_fields: dict[str, Any] = field(default_factory=dict)
    returncode: int = 0
    stderr: str = ""
    write_model: bool = True
    write_metrics: bool = True
    refuse_with: str | None = None
    raises: BaseException | None = None
    head_accuracy: float = 0.75
    metrics_overrides: dict[str, Any] = field(default_factory=dict)
    specs: list[dict[str, Any]] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    child_envs: list[dict[str, str] | None] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if len(args) >= 2 and args[1] == "-c":
            stdout = probe_answer(self.probe_device, **self.probe_fields) + "\n"
            return subprocess.CompletedProcess(args, 0, stdout, "")
        self.child_envs.append(kwargs.get("env"))
        if self.raises is not None:
            raise self.raises
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        out = Path(spec["metrics_out"])
        if self.refuse_with is not None:
            out.write_text(json.dumps({"refused": self.refuse_with}), encoding="utf-8")
            return subprocess.CompletedProcess(args, REFUSED_EXIT, "", self.refuse_with)
        if self.write_model:
            model_dir = Path(spec["model_dir"])
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "config.json").write_text(
                json.dumps(
                    {"model_type": EMBEDDING_MODEL_TYPE, "architectures": ["EmbeddingGemma2Model"]}
                ),
                encoding="utf-8",
            )
            (model_dir / "model.safetensors").write_bytes(b"\0" * 16)
        if self.write_metrics:
            train = Path(spec["train_rows"]).read_text(encoding="utf-8").splitlines()
            heldout = Path(spec["heldout_rows"]).read_text(encoding="utf-8").splitlines()
            batch, epochs = spec["recipe"]["batch_size"], spec["recipe"]["epochs"]
            steps = epochs * -(-len(train) // batch)
            metrics = {
                "base_model": spec["model"],
                "base_model_revision": spec["revision"],
                "prefix": spec["prefix"],
                "labels": spec["labels"],
                "rows": {"train": len(train), "heldout": len(heldout)},
                "steps_done": steps,
                "steps_planned": steps,
                "train_seconds": 12.5,
                "device": spec["device"] or "cpu",
                "dtype": spec["recipe"]["dtype"],
                "head_heldout_accuracy": self.head_accuracy,
                "mps_environment": None,
            }
            metrics.update(self.metrics_overrides)
            out.write_text(json.dumps(metrics), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, "", self.stderr)


@pytest.fixture
def toolchain(monkeypatch, tmp_path) -> FakeEmbedToolchain:
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeEmbedToolchain()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


# ---------------------------------------------------------------------------
# A decision split, as `prepare` writes one
# ---------------------------------------------------------------------------

OPTIONS = ("refund", "card_lost", "balance")


def _decision(kind: str = "choice", none_option: str | None = None) -> Decision:
    if kind == "noul":
        return Decision("noul", "Is it urgent?", (Option("no"), Option("yes")))
    return Decision(
        "choice",
        "What does the customer want?",
        tuple(Option(o) for o in OPTIONS),
        none_option,
    )


def _write_split(path: Path, decision: Decision, rows: list[tuple[str, str]]) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": decision.render(state),
                    "completion": label,
                    "target": label,
                    "source_line": i + 1,
                },
                sort_keys=True,
            )
            + "\n"
            for i, (state, label) in enumerate(rows)
        ),
        encoding="utf-8",
    )
    return path


def _rows(n: int, labels: tuple[str, ...] = OPTIONS) -> list[tuple[str, str]]:
    return [(f"customer message {i}", labels[i % len(labels)]) for i in range(n)]


@dataclass
class Split:
    dir: Path
    train: Path
    heldout: Path
    decision: Path


def _split(
    tmp_path: Path,
    decision: Decision | None = None,
    train: list[tuple[str, str]] | None = None,
    heldout: list[tuple[str, str]] | None = None,
) -> Split:
    decision = decision or _decision()
    train = train if train is not None else _rows(20)
    heldout = heldout if heldout is not None else _rows(6)
    root = tmp_path / "split"
    root.mkdir(exist_ok=True)
    decision_path = root / "decision.json"
    decision_path.write_text(decision.canonical_text(), encoding="utf-8")
    return Split(
        root,
        _write_split(root / "train.jsonl", decision, train),
        _write_split(root / "heldout.jsonl", decision, heldout),
        decision_path,
    )


@pytest.fixture
def split(tmp_path) -> Split:
    return _split(tmp_path)


def _request(split: Split, tmp_path: Path, **kwargs: Any) -> EmbeddingTuneRequest:
    params: dict[str, Any] = {
        "data": split.train,
        "heldout": split.heldout,
        "decision": split.decision,
        "output_dir": tmp_path / "tuned",
    }
    params.update(kwargs)
    return EmbeddingTuneRequest(**params)


def _check(result, name):
    return next(c for c in result.checks.checks if c.name == name)


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


def test_a_run_writes_the_embedding_record_beside_the_checkpoint(toolchain, split, tmp_path):
    result = run_embedding_tune(_request(split, tmp_path))

    assert result.outcome is Outcome.PASSED, result.checks.as_dict()
    assert result.verified is False
    record = read_embedding_record(tmp_path / "tuned" / "model")
    assert record is not None
    assert record == result.record
    data = json.loads((tmp_path / "tuned" / "model" / "litetune.json").read_text())
    assert data["direction"] == "embedding"
    assert data["base_model"] == EMBEDDING_MODEL
    assert data["base_model_revision"] == EMBEDDING_REVISION
    assert data["prefix"] == "task: sentence similarity | query: "
    assert data["prompt_name"] == "SentenceSimilarity"
    assert data["labels"] == sorted(OPTIONS)
    assert data["rows"] == {"train": 20, "heldout": 6}
    # ceil(20 / 8) = 3 steps an epoch, two epochs.
    assert (data["steps_done"], data["steps_planned"]) == (6, 6)
    assert data["seconds"] == 12.5
    assert (data["device"], data["dtype"]) == ("cpu", "float32")
    assert data["head_heldout_accuracy"] == 0.75
    assert data["recipe"] == RECIPE.as_dict()
    assert data["decision"] == _decision().as_dict()
    assert data["decision_sha256"] == _decision().digest()


def test_the_report_is_written_and_names_the_direction(toolchain, split, tmp_path):
    from litetune.embedding_tune import write_report

    result = run_embedding_tune(_request(split, tmp_path))
    report = json.loads(write_report(result).read_text())
    assert report["direction"] == "embedding"
    assert report["outcome"] == "passed"
    assert report["verified"] is False
    assert report["request"]["environment"]["name"] == "embed-train"
    assert report["record"]["labels"] == sorted(OPTIONS)


def test_the_spec_carries_the_prefix_the_model_and_the_measured_recipe(toolchain, split, tmp_path):
    run_embedding_tune(_request(split, tmp_path))

    (spec,) = toolchain.specs
    assert spec["model"] == "google/embeddinggemma-2"
    assert spec["revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b"
    assert spec["model_type"] == "embedding_gemma2"
    assert spec["prompt_name"] == "SentenceSimilarity"
    assert spec["prefix"] == "task: sentence similarity | query: "
    assert spec["labels"] == ["balance", "card_lost", "refund"]
    recipe = spec["recipe"]
    assert recipe["learning_rate"] == 2e-5
    assert recipe["head_learning_rate"] == 1e-3
    assert recipe["weight_decay"] == 0.0
    assert recipe["batch_size"] == 8
    assert recipe["epochs"] == 2
    assert recipe["warmup_fraction"] == 0.1
    assert recipe["seed"] == 0
    assert recipe["max_seq_length"] == 128
    assert (recipe["embedding_dim"], recipe["head_hidden"]) == (768, 256)
    assert recipe["dtype"] == "float32"
    assert spec["model_dir"] == str(tmp_path / "tuned" / "model")


def test_the_script_is_given_the_states_not_the_rendered_prompts(toolchain, split, tmp_path):
    run_embedding_tune(_request(split, tmp_path))

    (spec,) = toolchain.specs
    rows = [json.loads(line) for line in Path(spec["train_rows"]).read_text().splitlines()]
    assert rows[0] == {"text": "customer message 0", "label": "refund"}
    assert len(rows) == 20
    assert all("Question:" not in r["text"] for r in rows)
    heldout = [json.loads(line) for line in Path(spec["heldout_rows"]).read_text().splitlines()]
    assert len(heldout) == 6


def test_the_run_happens_in_embed_train(toolchain, split, tmp_path):
    seen: list[str] = []
    original = envs.StageEnv.run

    def spy(self, args, timeout: int = 3600, **kwargs):
        seen.append(self.name)
        return original(args, timeout=timeout, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(envs.StageEnv, "run", spy)
        run_embedding_tune(_request(split, tmp_path))
    assert seen == ["embed-train", "embed-train"]
    assert toolchain.calls[1][0] == "python"
    assert Path(toolchain.calls[1][1]).name == "embed_train_script.py"


def test_the_parent_imports_no_training_stack(toolchain, split, tmp_path):
    run_embedding_tune(_request(split, tmp_path))
    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_a_none_option_is_refused_before_the_environment(toolchain, tmp_path):
    decision = _decision(none_option="other")
    s = _split(
        tmp_path,
        decision,
        train=_rows(20, (*OPTIONS, "other")),
        heldout=_rows(4, (*OPTIONS, "other")),
    )
    with pytest.raises(EmbeddingTuneError, match="none_option \\('other'\\)"):
        run_embedding_tune(_request(s, tmp_path))
    assert toolchain.calls == []


# ---------------------------------------------------------------------------
# Endings
# ---------------------------------------------------------------------------


def test_a_model_the_script_refuses_is_a_failed_model_check(toolchain, split, tmp_path):
    toolchain.refuse_with = "config.json declares model_type 'gemma3_text'"
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    check = _check(result, MODEL_CHECK)
    assert "gemma3_text" in check.detail
    assert read_embedding_record(tmp_path / "tuned" / "model") is None


def test_a_non_zero_exit_is_a_failure_and_writes_no_record(toolchain, split, tmp_path):
    toolchain.returncode = 1
    toolchain.stderr = "Traceback: ValueError: boom"
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert "boom" in _check(result, TRAINING_CHECK).detail
    assert result.record is None
    assert not (tmp_path / "tuned" / "model" / "litetune.json").exists()


def test_an_out_of_memory_ending_is_could_not_check(toolchain, split, tmp_path):
    toolchain.returncode = 1
    toolchain.stderr = "torch.OutOfMemoryError: CUDA out of memory"
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.UNCHECKED


def test_a_killed_run_is_could_not_check(toolchain, split, tmp_path):
    toolchain.returncode = -9
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.UNCHECKED


def test_a_timeout_is_could_not_check(toolchain, split, tmp_path):
    toolchain.raises = subprocess.TimeoutExpired(cmd="python", timeout=5)
    result = run_embedding_tune(_request(split, tmp_path, timeout_s=5))
    assert result.outcome is Outcome.UNCHECKED
    assert "timeout" in _check(result, TRAINING_CHECK).detail


def test_exit_zero_without_a_checkpoint_is_a_failure(toolchain, split, tmp_path):
    toolchain.write_model = False
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert "no checkpoint" in _check(result, TRAINING_CHECK).detail


def test_exit_zero_without_metrics_is_a_failure(toolchain, split, tmp_path):
    toolchain.write_metrics = False
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert "no readable metrics" in _check(result, TRAINING_CHECK).detail


def test_metrics_with_other_labels_than_given_are_a_failure(toolchain, split, tmp_path):
    toolchain.metrics_overrides = {"labels": ["refund", "card_lost", "balance"]}
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED
    assert result.record is None


def test_a_stale_metrics_file_is_not_read_as_this_run(toolchain, split, tmp_path):
    out = tmp_path / "tuned"
    out.mkdir()
    (out / "embed_metrics.json").write_text(json.dumps({"steps_done": 99}))
    toolchain.write_metrics = False
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.FAILED


def test_an_unprovisioned_environment_is_could_not_check(monkeypatch, split, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def never(*args, **kwargs):
        raise AssertionError("nothing may run in an environment that is not there")

    monkeypatch.setattr(envs.StageEnv, "run", never)
    result = run_embedding_tune(_request(split, tmp_path, auto_provision=False))
    assert result.outcome is Outcome.UNCHECKED
    assert _check(result, ENV_CHECK).outcome is Outcome.UNCHECKED
    assert _check(result, DATA_CHECK).outcome is Outcome.PASSED


# ---------------------------------------------------------------------------
# The device
# ---------------------------------------------------------------------------


def test_the_probed_device_reaches_the_script_and_the_record(toolchain, split, tmp_path):
    toolchain.probe_device = "cuda"
    toolchain.probe_fields = {"cuda_build": "12.8", "device_count": 1}
    result = run_embedding_tune(_request(split, tmp_path))
    assert toolchain.specs[0]["device"] == "cuda"
    assert result.device == "cuda"
    assert result.record is not None and result.record.device == "cuda"


def test_litetune_device_cpu_overrides_the_probe(toolchain, split, tmp_path, monkeypatch):
    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    toolchain.probe_device = "cuda"
    toolchain.probe_fields = {"cuda_build": "12.8", "device_count": 1}
    result = run_embedding_tune(_request(split, tmp_path))
    assert toolchain.specs[0]["device"] == "cpu"
    assert any("LITETUNE_DEVICE=cpu" in t for t in result.limitations)


def test_an_unknown_litetune_device_is_refused_before_the_environment(
    toolchain, split, tmp_path, monkeypatch
):
    monkeypatch.setenv("LITETUNE_DEVICE", "gpu")
    result = run_embedding_tune(_request(split, tmp_path))
    assert result.outcome is Outcome.UNCHECKED
    assert _check(result, DEVICE_CHECK).outcome is Outcome.UNCHECKED
    assert toolchain.calls == []


def test_an_mps_run_gets_the_memory_policy_in_its_environment(toolchain, split, tmp_path):
    from litetune import devices

    toolchain.probe_device = "mps"
    result = run_embedding_tune(_request(split, tmp_path))
    assert toolchain.specs[0]["device"] == "mps"
    assert result.mps_memory is not None
    (child_env,) = toolchain.child_envs
    assert child_env is not None and devices.HIGH_WATERMARK in child_env


def test_an_unanswered_probe_leaves_the_device_to_the_script(toolchain, split, tmp_path):
    toolchain.probe_device = None
    toolchain.probe_fields = {}
    result = run_embedding_tune(_request(split, tmp_path))
    assert toolchain.specs[0]["device"] is None
    assert any("not established" in t for t in result.limitations)


# ---------------------------------------------------------------------------
# Refusals of the request
# ---------------------------------------------------------------------------


def test_a_model_other_than_embeddinggemma_2_is_refused(split, tmp_path):
    with pytest.raises(EmbeddingTuneError, match="google/embeddinggemma-2 only"):
        _request(split, tmp_path, model="google/embeddinggemma-300m")


def test_another_revision_is_refused(split, tmp_path):
    with pytest.raises(EmbeddingTuneError, match="914f7f89"):
        _request(split, tmp_path, revision="main")


def test_a_question_that_is_not_choice_is_refused(toolchain, tmp_path):
    decision = _decision("noul")
    s = _split(tmp_path, decision, train=_rows(10, ("no", "yes")), heldout=_rows(4, ("no", "yes")))
    with pytest.raises(EmbeddingTuneError, match="noul question"):
        run_embedding_tune(_request(s, tmp_path))
    assert toolchain.calls == []


def test_an_option_with_no_training_row_is_refused(toolchain, tmp_path):
    s = _split(tmp_path, train=_rows(10, ("refund", "card_lost")))
    with pytest.raises(EmbeddingTuneError, match="'balance'"):
        run_embedding_tune(_request(s, tmp_path))
    assert toolchain.calls == []


def test_fewer_than_two_labels_is_refused():
    one = Decision("choice", "Which?", (Option("only"),))
    with pytest.raises(EmbeddingTuneError, match="at least two"):
        labels_for(one, [LabelledText("x", "only")])


def test_a_row_not_rendered_with_the_decision_is_refused(toolchain, split, tmp_path):
    split.train.write_text(
        json.dumps({"prompt": "bare text", "completion": "refund", "source_line": 1}) + "\n"
    )
    with pytest.raises(EmbeddingTuneError, match=r"train\.jsonl:1"):
        run_embedding_tune(_request(split, tmp_path))


def test_a_missing_heldout_split_is_refused(toolchain, split, tmp_path):
    split.heldout.unlink()
    with pytest.raises(EmbeddingTuneError, match="held-out split"):
        run_embedding_tune(_request(split, tmp_path))


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _record(**overrides: Any) -> EmbeddingRecord:
    fields: dict[str, Any] = {
        "base_model": EMBEDDING_MODEL,
        "base_model_revision": EMBEDDING_REVISION,
        "prompt_name": PROMPT_NAME,
        "prefix": DECIDER_PREFIX,
        "labels": ("a", "b"),
        "rows": {"train": 10, "heldout": 4},
        "steps_done": 4,
        "steps_planned": 4,
        "seconds": 1.5,
        "device": "cpu",
        "dtype": "float32",
        "head_heldout_accuracy": 0.5,
        "recipe": RECIPE.as_dict(),
        "decision": _decision().as_dict(),
        "decision_sha256": _decision().digest(),
    }
    fields.update(overrides)
    return EmbeddingRecord(**fields)


def test_the_record_round_trips(tmp_path):
    record = _record()
    write_embedding_record(tmp_path, record)
    assert read_embedding_record(tmp_path) == record
    assert json.loads((tmp_path / "litetune.json").read_text())["direction"] == DIRECTION


def test_no_record_and_an_llm_record_are_not_embedding_records(tmp_path):
    assert read_embedding_record(tmp_path) is None
    (tmp_path / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m"}))
    assert read_embedding_record(tmp_path) is None


@pytest.mark.parametrize(
    ("key", "value"),
    [("labels", "a,b"), ("steps_done", True), ("prefix", None), ("rows", [10, 4])],
)
def test_a_malformed_embedding_record_raises_naming_the_field(tmp_path, key, value):
    data = _record().as_dict()
    data[key] = value
    (tmp_path / "litetune.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=key):
        read_embedding_record(tmp_path)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _argv(split: Split, tmp_path: Path, *extra: str, model: str = EMBEDDING_MODEL) -> list[str]:
    return [
        "tune",
        "--model",
        model,
        "--data",
        str(split.train),
        "--output-dir",
        str(tmp_path / "run"),
        "--decision",
        str(split.decision),
        *extra,
    ]


@pytest.fixture
def dispatched(monkeypatch):
    """Which of the two stages `litetune tune` called, and with what."""
    from litetune import cli
    from litetune.checks import Check, CheckSet
    from litetune.embedding_tune import EmbeddingTuneResult
    from litetune.tune import TuneResult

    seen: dict[str, Any] = {}

    def fake_embedding(request, events=None):
        seen["embedding"] = request
        checks = CheckSet(name="train")
        checks.add(Check.passed("training run", "faked"))
        return EmbeddingTuneResult(request=request, checks=checks)

    def fake_llm(request, events=None):
        seen["llm"] = request
        checks = CheckSet(name="train")
        checks.add(Check.passed("training run", "faked"))
        return TuneResult(request=request, checks=checks)

    monkeypatch.setattr(cli, "run_embedding_tune", fake_embedding)
    monkeypatch.setattr(cli, "run_tune", fake_llm)
    return seen


def test_the_embedding_model_id_goes_to_the_embedding_path(dispatched, split, tmp_path):
    assert main(_argv(split, tmp_path)) == 0
    assert "llm" not in dispatched
    request = dispatched["embedding"]
    assert request.data == split.train
    # `prepare` writes the held-out split beside the training one.
    assert request.heldout == split.heldout
    assert request.decision == split.decision
    assert json.loads((tmp_path / "run" / "tune.json").read_text())["direction"] == "embedding"


def test_an_explicit_heldout_is_used(dispatched, split, tmp_path):
    other = tmp_path / "other.jsonl"
    assert main(_argv(split, tmp_path, "--heldout", str(other))) == 0
    assert dispatched["embedding"].heldout == other


def test_any_other_model_goes_to_the_llm_path_with_its_defaults(dispatched, split, tmp_path):
    assert main(_argv(split, tmp_path, model="google/gemma-3-270m-it")) == 0
    assert "embedding" not in dispatched
    request = dispatched["llm"]
    assert (request.method, request.epochs, request.batch_size) == ("full", 1.0, 8)
    assert (request.max_seq_length, request.seed) == (1024, 0)
    assert (request.lora_rank, request.lora_alpha, request.lora_dropout) == (16, 32, 0.05)
    assert request.attn_implementation == "eager"


def test_a_local_embeddinggemma_path_is_not_the_embedding_path(dispatched, split, tmp_path):
    """Only the exact id: a directory is whatever its own record says, and `tune` takes ids."""
    main(_argv(split, tmp_path, model=str(tmp_path / "embeddinggemma-2")))
    assert "embedding" not in dispatched


@pytest.mark.parametrize(
    "extra",
    [
        ["--method", "lora"],
        ["--method", "full"],
        ["--lora-rank", "8"],
        ["--lora-alpha", "16"],
        ["--lora-dropout", "0.1"],
        ["--learning-rate", "1e-4"],
        ["--epochs", "1"],
        ["--batch-size", "4"],
        ["--micro-batch-size", "2"],
        ["--max-seq-length", "256"],
        ["--seed", "1"],
        ["--attn-implementation", "sdpa"],
        ["--prompt-mode", "runtime_rendered"],
        ["--force-prompt-mode"],
        ["--declarations", "tools.json"],
    ],
)
def test_an_llm_flag_is_refused_by_name_not_ignored(dispatched, split, tmp_path, capsys, extra):
    code = main(_argv(split, tmp_path, *extra))
    assert code == 4
    assert dispatched == {}
    err = capsys.readouterr().err
    assert extra[0] in err
    assert "measured embedding recipe" in err


def test_bfloat16_is_refused_and_float32_accepted(dispatched, split, tmp_path, capsys):
    assert main(_argv(split, tmp_path, "--dtype", "bfloat16")) == 4
    assert "bfloat16 was not measured" in capsys.readouterr().err
    assert main(_argv(split, tmp_path, "--dtype", "float32")) == 0


def test_no_decision_is_refused(dispatched, split, tmp_path, capsys):
    argv = _argv(split, tmp_path)
    argv = argv[: argv.index("--decision")]
    assert main(argv) == 4
    assert "needs --decision" in capsys.readouterr().err
    assert dispatched == {}


def test_another_revision_is_refused_on_the_command_line(dispatched, split, tmp_path, capsys):
    assert main(_argv(split, tmp_path, "--revision", "main")) == 4
    assert EMBEDDING_REVISION in capsys.readouterr().err
    assert main(_argv(split, tmp_path, "--revision", EMBEDDING_REVISION)) == 0


def test_heldout_is_refused_for_an_llm(dispatched, split, tmp_path, capsys):
    argv = _argv(split, tmp_path, "--heldout", "h.jsonl", model="google/gemma-3-270m-it")
    assert main(argv) == 4
    assert "--heldout" in capsys.readouterr().err
    assert dispatched == {}


def test_a_request_refused_by_the_stage_exits_could_not_check(toolchain, split, tmp_path, capsys):
    split.heldout.unlink()
    assert main(_argv(split, tmp_path)) == 4
    assert "held-out split" in capsys.readouterr().err


def test_a_whole_run_through_the_command_line(toolchain, split, tmp_path):
    assert main(_argv(split, tmp_path)) == 0
    record = read_embedding_record(tmp_path / "run" / "model")
    assert record is not None and record.labels == ("balance", "card_lost", "refund")


# ---------------------------------------------------------------------------
# The script's helpers, run here
# ---------------------------------------------------------------------------


@pytest.fixture
def script() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "embed_train_script"}
    exec(compile(_EMBED_TRAIN_SCRIPT, "embed_train_script.py", "exec"), namespace)
    return namespace


def test_the_script_refuses_a_model_type_it_was_not_measured_on(script):
    refusal = script["model_type_refusal"]
    assert refusal({"model_type": "embedding_gemma2"}, "embedding_gemma2") is None
    assert "gemma3_text" in refusal({"model_type": "gemma3_text"}, "embedding_gemma2")
    assert "None" in refusal([], "embedding_gemma2")


def test_the_script_refuses_a_prompt_that_is_not_the_deciders(script):
    refusal = script["prefix_refusal"]
    prompts = {"SentenceSimilarity": DECIDER_PREFIX, "Classification": "task: classification | "}
    assert refusal(prompts, PROMPT_NAME, DECIDER_PREFIX) is None
    assert refusal({}, PROMPT_NAME, DECIDER_PREFIX) is not None
    assert refusal({PROMPT_NAME: "task: classification | query: "}, PROMPT_NAME, DECIDER_PREFIX)


def test_the_schedule_is_train_s5s(script):
    plan, factor = script["plan"], script["lr_factor"]
    # 100 rows at batch 8: 13 steps an epoch, 26 in two; warmup int(2.6) = 2.
    assert plan(100, 8, 2, 0.1) == (26, 2)
    # A run too short for a tenth to be a step still warms up for one.
    assert plan(4, 8, 2, 0.1) == (2, 1)
    steps, warm = 26, 2
    assert [factor(s, warm, steps) for s in range(3)] == [0.5, 1.0, 1.0]
    assert factor(25, warm, steps) == pytest.approx(1 / 24)
    assert factor(26, warm, steps) == 0.0


def test_the_script_batches_every_row_once_an_epoch(script):
    batches = script["epoch_batches"](10, 4, lambda order: order.reverse())
    assert batches == [[9, 8, 7, 6], [5, 4, 3, 2], [1, 0]]


def test_the_script_reports_accuracy_to_four_places(script):
    assert script["accuracy"](["a", "b", "a"], ["a", "b", "b"]) == 0.6667


def test_the_script_takes_the_parents_device(script):
    class NoCuda:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    assert script["training_device"](NoCuda, "mps") == "mps"
    assert script["training_device"](NoCuda, None) == "cpu"


# ---------------------------------------------------------------------------
# Neighbours: the family rules and the environment
# ---------------------------------------------------------------------------


def test_embeddinggemma_2_is_no_llm_family():
    assert models.identify(EMBEDDING_MODEL) is None


def test_a_tuned_embeddinggemma_checkpoint_is_no_llm_family(tmp_path):
    """The directory `tune` writes: EG2's config.json and an embedding record."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "embedding_gemma2",
                "architectures": ["EmbeddingGemma2Model"],
                "text_config": {"model_type": "embedding_gemma2_text"},
                "vision_config": {"model_type": "gemma4_vision"},
            }
        )
    )
    write_embedding_record(tmp_path, _record())
    assert models.identify(str(tmp_path)) is None


def test_embed_train_installs_where_its_pins_have_wheels():
    pins = envs.EMBED_TRAIN.requirements
    assert wheels.gaps(pins, machine("linux_x86_64", (3, 11))) == []
    assert wheels.gaps(pins, machine("macosx_arm64", (3, 12))) == []
    blocked = [g.release.requirement for g in wheels.gaps(pins, machine("macosx_x86_64", (3, 12)))]
    assert "torch==2.9.1" in blocked
