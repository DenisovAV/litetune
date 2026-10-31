"""`verify` for a decision checkpoint, with both scoring scripts faked.

`FakeScoringEnv` stands in for a stage environment: it reads the spec the stage
wrote, recognises which script it was handed, and writes the rows and report
that script would, from scores this file decides. No model and no runtime runs.
The splits are the ones `prepare` writes, so the rows reaching `verify` are the
contract's prompts.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from litetune import calibration, devices, envs
from litetune.decision_section import payload_bytes
from litetune.decision_verify import (
    AGREEMENT_CHECK,
    CALIBRATION_CHECK,
    MPS_ROW_BATCH,
    PROBE_ROWS,
    PROMPT_CHECK,
    REFERENCE_CHECK,
    REFERENCE_ON_MPS,
    ROWS_CHECK,
    RULE_CHECK,
    RUNTIME_CHECK,
    TARGETS_CHECK,
    TWIN_CHECK,
    TWIN_TOLERANCE,
    TYPE_LESS_MEASURED,
    DecisionVerify,
    run_script,
    settle,
    summed,
    unique_lengths,
    verify_decision,
)
from litetune.decisions import NOUL_IDS, decision_from
from litetune.events import EventStream
from litetune.prepare import PrepareRequest, prepare

TERM = "<end_of_turn>"
OPTIONS = ("joy", "anger", "fear")


def tokens(option: str) -> list[int]:
    """A character-level stand-in for the tokenizer: one id per character, then the terminator."""
    return [ord(c) for c in option] + [999]


def label_of(prompt: str) -> str | None:
    found = re.match(r"text \d+ about (\w+)", prompt)
    return found.group(1) if found else None


def default_scores(index: int, prompt: str, option: str, ids: list[int]) -> list[float]:
    """The right option scores near 0, the rest well below; how far varies with the prompt.

    A function of the prompt, never of `index`: the two scripts number the same
    row differently, as the real ones do, and a real model sees only the prompt.
    Every tenth in-scope row is a near miss: the first option scores just above
    the right one. Rows that are all right with room to spare fit a temperature
    at the bottom of the grid, which calibration refuses, as it would a real
    model's.
    """
    label = label_of(prompt)
    right = label or "none_of_these"
    number = int(re.search(r"\d+", prompt).group())  # type: ignore[union-attr]
    first = 0.0 if option == right else -2.0 - (number % 5) - 0.1 * len(option)
    if label is not None and number % 10 == 9 and option != right:
        first = 0.3 if option == OPTIONS[0] and right != OPTIONS[0] else first
        if right == OPTIONS[0] and option == OPTIONS[1]:
            first = 0.3
    return [first] + [-0.01] * (len(ids) - 1)


BOS = 2

# What each family's recorded end-of-turn ids decode to: gemma-3-270m-it's
# (E5's record, below), and ChatML's and Granite's, with made-up ids.
VOCAB = {106: TERM, 107: "\n", 151645: "<|im_end|>", 100257: "<|end_of_text|>"}


def prompt_ids(prompt: str) -> list[int]:
    """A stand-in for the runtime's tokenisation of a rendered prompt, without the BOS."""
    return [ord(c) % 1000 + 3 for c in prompt]


def near_miss(prompt: str) -> bool:
    """Whether `default_scores` answers this row wrong: every tenth in-scope row."""
    number = int(re.search(r"\d+", prompt).group())  # type: ignore[union-attr]
    return label_of(prompt) is not None and number % 10 == 9


def expected_accuracy(rows: list[dict[str, Any]]) -> float:
    """`default_scores`'s accuracy on in-scope rows: right except on its near misses."""
    in_scope = [r for r in rows if r["target"] != "none_of_these"]
    return sum(not near_miss(r["prompt"]) for r in in_scope) / len(in_scope)


def _reading(time_: int) -> dict[str, Any]:
    return {
        "platform": "darwin",
        "looked": True,
        "gpu_client": "pid 1, python3",
        "gpu_time": time_,
        "gpu_clients": {"0x1": time_},
    }


GPU_WORKED = {"at_engine": _reading(10), "after_generation": _reading(500)}
GPU_IDLE = {"at_engine": _reading(10), "after_generation": _reading(10)}


@dataclass
class FakeScoringEnv:
    """A stage environment that answers either scoring script with chosen scores."""

    name: str
    scores: Callable[[int, str, str, list[int]], list[float]] = default_scores
    returncode: int = 0
    report_extra: dict[str, Any] = field(default_factory=dict)
    skip_rows: set[int] = field(default_factory=set)
    token_of: Callable[[str], list[int]] = tokens
    # Options whose unique prefix "does not round-trip": scored in full, as the
    # runtime script does.
    full_for: set[str] = field(default_factory=set)
    # How the runtime tokenises a row's prompt. The reference returns
    # `ref_prefix` and then `prompt_ids`: [BOS], as a chat template that writes
    # the BOS does (Gemma 3), or [] for one that writes none (ChatML).
    prompt_ids_of: Callable[[str], list[int]] = prompt_ids
    ref_prefix: list[int] = field(default_factory=lambda: [BOS])
    # What the runtime reports its session puts first; None reports nothing.
    session_prefix: list[int] | None = field(default_factory=lambda: [BOS])
    # The reference's tokenizer, for the end of turn: what each recorded id decodes to.
    vocab: dict[int, str] = field(default_factory=lambda: dict(VOCAB))
    specs: list[dict[str, Any]] = field(default_factory=list)
    provisioned: int = 0
    # A GPU run's kernel readings, and which (engine run, row) pairs it scores
    # wrongly: (1, 3) is the first engine run's row 3.
    device: dict[str, Any] | None = None
    # A reading for one engine run only, by its number; the rest use `device`.
    device_for_run: dict[int, dict[str, Any]] = field(default_factory=dict)
    glitch: set[tuple[int, int]] = field(default_factory=set)
    # Rows an engine run, by its number, leaves out.
    skip_for_run: dict[int, set[int]] = field(default_factory=dict)
    engine_runs: int = 0
    # The `env` each run was given, in order.
    child_envs: list[Any] = field(default_factory=list)
    ready: bool = True
    path: Path = Path("/nonexistent")
    # A reference run on this device exits with this code and stderr.
    stops_on: dict[str, tuple[int, str]] = field(default_factory=dict)

    def provision(self, events=None, force: bool = False) -> Path:
        self.provisioned += 1
        return Path("/nonexistent")

    def run(self, argv, timeout=None, env=None) -> subprocess.CompletedProcess:
        script = Path(argv[1]).read_text(encoding="utf-8")
        spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        self.child_envs.append(env)
        engine = "litert_lm" in script
        if not engine and spec.get("device") in self.stops_on:
            code, said = self.stops_on[spec["device"]]
            stop = {"stopped_by": said, "device": spec["device"], "mps_environment": None}
            Path(spec["report"]).write_text(json.dumps(stop), encoding="utf-8")
            return subprocess.CompletedProcess(argv, code, "", said)
        cont = [self.token_of(o) for o in spec["options"]]
        ks = unique_lengths(cont)
        report: dict[str, Any] = {"cont_ids": cont}
        if engine:
            self.engine_runs += 1
            if spec.get("backend") == "gpu":
                default = self.device if self.device is not None else GPU_WORKED
                report["device"] = self.device_for_run.get(self.engine_runs, default)
            full = spec["targets"] == "full"
            lengths = [
                len(c) if full or o in self.full_for else k
                for o, c, k in zip(spec["options"], cont, ks, strict=True)
            ]
            report.update(
                {
                    "runtime": {"litert-lm": "0.18.0"},
                    "cpu": "fake CPU",
                    "threads": spec["threads"],
                    "mode": "once" if spec["mode"] == "auto" else spec["mode"],
                    "method": spec["method"],
                    "selfcheck": {"verdict": "dropped", "comparisons": 18, "max_abs_diff": 0.0},
                    "k": ks,
                    "scored_lengths": lengths,
                    "mismatches": [{"option": o} for o in spec["options"] if o in self.full_for],
                }
            )
            if self.session_prefix is not None:
                report["session_prefix"] = self.session_prefix
        else:
            if env is not None:
                # As `environment_seen` reads them in the child.
                report["mps_environment"] = {n: env.get(n) for n in devices.MPS_VARIABLES}
            report.update(
                {
                    "device": "cpu",
                    "bos_id": BOS,
                    "cache_variant": "position_ids",
                    "terminator": self.vocab[spec["turn_ids"][0]],
                }
            )
        report.update(self.report_extra)
        with open(spec["out"], "w", encoding="utf-8") as out:
            for row in spec["rows"]:
                if row["index"] in self.skip_rows:
                    continue
                if engine and row["index"] in self.skip_for_run.get(self.engine_runs, set()):
                    continue
                per = [
                    self.scores(row["index"], row["prompt"], o, c)
                    for o, c in zip(spec["options"], cont, strict=True)
                ]
                if engine:
                    per = [p[:n] for p, n in zip(per, lengths, strict=True)]
                    if (self.engine_runs, row["index"]) in self.glitch:
                        # Wrong scores: every option moved, differently for each run and
                        # each row, as the real wrong scores were never the same twice.
                        shift = 3.0 * self.engine_runs + 0.001 * row["index"]
                        per = [[v - shift for v in p] for p in per]
                rec = {"index": row["index"], "token_scores": per}
                rec["ms" if engine else "seconds"] = 10.0
                if engine:
                    rec["prompt_ids"] = self.prompt_ids_of(row["prompt"])
                elif row["index"] in spec.get("ids_for", []):
                    rec["prompt_ids"] = [*self.ref_prefix, *prompt_ids(row["prompt"])]
                out.write(json.dumps(rec) + "\n")
        Path(spec["report"]).write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.returncode, "", "fake stderr")


@pytest.fixture(autouse=True)
def _probe_answers_cpu(monkeypatch):
    """The device probe answers the CPU off a Mac, unless a test answers otherwise.

    `FakeScoringEnv` runs scoring scripts, not the probe's `python -c`.
    """
    monkeypatch.setattr(
        envs,
        "resolve_device",
        lambda env, events=None: envs.DeviceProbe(
            device="cpu", detail="this environment's torch reports cpu", source="probe"
        ),
    )


@pytest.fixture
def split(tmp_path):
    """`prepare`'s decision split: three emotions and an out-of-scope option."""

    def _make(
        with_none: bool = True, per_option: int = 30, oos: int = 40, kind: str = "choice"
    ) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "question": "Which emotion?",
            "options": [{"id": o} for o in OPTIONS],
        }
        options: tuple[str, ...] = OPTIONS
        if kind == "noul":
            obj = {"type": "noul", "question": "Is it joyful?"}
            options = NOUL_IDS
        elif kind == "score":
            obj = {"type": "score", "question": "How joyful?", "options": ["low", "high"]}
            options = ("0", "1")
        if with_none:
            obj["none_option"] = "none_of_these"
        questions = tmp_path / "questions.json"
        questions.write_text(json.dumps(obj), encoding="utf-8")
        rows: list[dict[str, Any]] = [
            {"prompt": f"text {i} about {o}", "label": o}
            for o in options
            for i in range(per_option)
        ]
        if with_none:
            rows += [{"prompt": f"weather {i}", "label": None} for i in range(oos)]
        data = tmp_path / "rows.jsonl"
        data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        result = prepare(
            PrepareRequest(
                data=data,
                output_dir=tmp_path / "prepared",
                context_length=4096,
                questions=questions,
                calibration_fraction=0.3,
            )
        )
        assert result.calibration is not None and result.heldout is not None
        assert result.decision is not None
        decision = result.decision.decision
        record = {
            "base_model": "google/gemma-3-270m-it",
            "decision": decision.as_dict(),
            "decision_sha256": decision.digest(),
            # As `tune` records it: E5's gemma-3-270m-it run wrote exactly this.
            "turn_terminator": {
                "ids": [106, 107],
                "source": "chat_template",
                "text": "<end_of_turn>\n",
                "call": None,
            },
        }
        return {
            "decision": decision,
            "record": record,
            "heldout": result.heldout.path,
            "calibration": result.calibration.path,
        }

    return _make


def job_for(split_: dict[str, Any], tmp_path, **kwargs) -> DecisionVerify:
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    params: dict[str, Any] = {
        "model": model,
        "reference": str(tmp_path / "checkpoint"),
        "heldout": split_["heldout"],
        "calibration": split_["calibration"],
        "train_env": FakeScoringEnv("train"),
        "runtime_env": FakeScoringEnv("runtime"),
        # The record's google/gemma-3-270m-it, as `run_verify` looks it up.
        "family": "gemma-3-text",
    }
    params.update(kwargs)
    return DecisionVerify(**params)


def run(job: DecisionVerify, record) -> tuple[str, dict[str, Any]]:
    return verify_decision(job, record, EventStream(echo_json=False))


def check(fields: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in fields["checks"]["checks"] if c["name"] == name)


# ---------------------------------------------------------------------------


def test_a_decision_that_scores_alike_on_both_sides_passes(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed", fields["checks"]
    assert fields["scorer"]["name"] == "option-scoring"
    fit = fields["decision"]["fit"]
    assert fit["rule"] == calibration.C2 and fit["abstain_below"] is not None
    payload = fields["decision"]["section"]["payload"]
    assert payload == {
        "contract_version": 1,
        "type": "choice",
        "question": "Which emotion?",
        "options": [{"id": o} for o in OPTIONS],
        "none_option": "none_of_these",
        "scoring": "unique_prefix",
        "method": "rewind",
        "terminator": TERM,
        "scored_lengths": fields["decision"]["scored_lengths"],
        "temperature": fit["temperature"],
        "abstain_rule": calibration.C2,
        "abstain_below": fit["abstain_below"],
    }
    assert payload["scored_lengths"] == unique_lengths([tokens(o) for o in s["decision"].ids])
    assert fields["decision"]["section"]["bytes"] == len(payload_bytes(payload))
    assert fields["quality"]["argmax_agreement"] == 1.0
    held = [json.loads(line) for line in s["heldout"].read_text().splitlines()]
    assert fields["measurements"]["reference"]["accuracy"] == expected_accuracy(held)
    assert fields["cost"]["mode"] == "once"
    assert fields["cost"]["scored_tokens_per_decision"] == sum(
        unique_lengths([tokens(o) for o in s["decision"].ids])
    )
    assert any("--float-twin" in text for text in fields["limitations"])


def test_the_reference_scores_every_calibration_and_heldout_row_and_the_runtime_a_subset(
    split, tmp_path
):
    s = split()
    job = job_for(s, tmp_path, engine_rows=5)

    run(job, s["record"])

    calib_n = sum(1 for _ in s["calibration"].read_text().splitlines())
    held_n = sum(1 for _ in s["heldout"].read_text().splitlines())
    ref_spec = job.train_env.specs[0]
    eng_spec = job.runtime_env.specs[0]
    assert len(ref_spec["rows"]) == calib_n + held_n
    assert ref_spec["model"] == job.reference
    assert ref_spec["options"] == list(s["decision"].ids)
    assert (ref_spec["turn_ids"], ref_spec["turn_text"]) == ([106, 107], "<end_of_turn>\n")
    assert "terminator" not in ref_spec and eng_spec["terminator"] == TERM
    assert len(eng_spec["rows"]) == 5
    assert eng_spec["mode"] == "auto" and eng_spec["targets"] == "unique"
    held = [json.loads(line)["prompt"] for line in s["heldout"].read_text().splitlines()]
    # Spread through the file, not its first rows: a split keeps the source's
    # order, and a source grouped by label would give them to one option.
    picked = [((2 * i + 1) * held_n) // 10 for i in range(5)]
    assert [r["prompt"] for r in eng_spec["rows"]] == [held[i] for i in picked]
    assert ref_spec["ids_for"] == [calib_n + i for i in picked]


def test_a_runtime_that_answers_differently_is_refused(split, tmp_path):
    s = split()

    def swapped(index, prompt, option, ids):
        right = label_of(prompt) or "none_of_these"
        wrong = "anger" if right == "joy" else "joy"
        return [0.0 if option == wrong else -5.0] + [-0.01] * (len(ids) - 1)

    job = job_for(s, tmp_path, runtime_env=FakeScoringEnv("runtime", scores=swapped))

    status, fields = run(job, s["record"])

    assert status == "failed_gate"
    assert check(fields, AGREEMENT_CHECK)["outcome"] == "failed"


def test_a_self_check_that_cannot_decide_is_a_harness_failure(split, tmp_path):
    s = split()
    eng = FakeScoringEnv(
        "runtime",
        returncode=3,
        report_extra={"error": "the self-check could not tell the engine apart: mixed"},
        skip_rows=set(range(10_000)),
    )
    job = job_for(s, tmp_path, runtime_env=eng)

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    detail = check(fields, RUNTIME_CHECK)["detail"]
    assert "could not tell the engine apart: mixed" in detail


def test_options_tokenised_differently_on_the_two_sides_are_a_harness_failure(split, tmp_path):
    s = split()
    eng = FakeScoringEnv("runtime", token_of=lambda o: [ord(c) + 1 for c in o] + [999])
    job = job_for(s, tmp_path, runtime_env=eng)

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert check(fields, TARGETS_CHECK)["outcome"] == "failed"


def test_rows_the_reference_left_unscored_are_a_harness_failure(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path, train_env=FakeScoringEnv("train", skip_rows={2}))

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "1 of" in check(fields, REFERENCE_CHECK)["detail"]


def test_a_decision_checkpoint_without_calibration_rows_is_refused(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path, calibration=None), s["record"])

    assert status == "failed_harness"
    assert "--calibration" in check(fields, CALIBRATION_CHECK)["detail"]


def test_calibration_rows_for_a_checkpoint_that_records_no_decision_are_refused(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path), {"base_model": "google/gemma-3-270m-it"})

    assert status == "failed_harness"
    assert "records no decision" in check(fields, ROWS_CHECK)["detail"]


def test_a_decision_record_whose_digest_is_not_its_own_is_refused(split, tmp_path):
    s = split()
    record = dict(s["record"], decision_sha256="sha256:" + "0" * 64)

    status, fields = run(job_for(s, tmp_path), record)

    assert status == "failed_harness"
    assert "not its own" in check(fields, ROWS_CHECK)["detail"]


@pytest.fixture
def on_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


def _engine_specs(job):
    return [s for s in job.runtime_env.specs if "terminator" in s and "turn_ids" not in s]


def _candidate_specs(job):
    """The engine runs that scored the candidate's rows (unique targets), in order."""
    return [s for s in _engine_specs(job) if s["targets"] == "unique"]


NOT_LOOKED = {
    "at_engine": {"platform": "darwin", "looked": False},
    "after_generation": {"platform": "darwin", "looked": False},
}


def test_on_macos_the_gpu_scores_every_row_twice_and_passes_when_they_agree(
    split, tmp_path, on_macos
):
    s = split()
    job = job_for(s, tmp_path, backend="gpu")

    status, fields = run(job, s["record"])

    assert status == "passed", fields["checks"]
    first, second = _candidate_specs(job)
    assert first["backend"] == second["backend"] == "gpu"
    # The repeat takes the mode the first run's self-check chose, and the rows
    # in reverse order.
    assert (first["mode"], second["mode"]) == ("auto", "once")
    assert [r["index"] for r in second["rows"]] == [r["index"] for r in first["rows"]][::-1]
    assert fields["harness"]["gpu_repeat"] == {
        "runs": 2,
        "rows_rescored": 0,
        "engine_rows": [],
        "observed": [True, True],
    }
    engine = fields["measurements"]["candidate"]["engine"]
    assert engine["backend"] == "gpu" and engine["backend_observed"] is True
    assert engine["threads"] is None
    assert fields["cost"]["backend"] == "gpu"
    assert fields["cost"]["threads"] is None and fields["cost"]["cpu"] is None
    said = [t for t in fields["limitations"] if "third run" in t]
    assert len(said) == 1
    assert said[0].endswith(f"0 of {len(first['rows'])} rows needed the third run")
    assert "float twin was checked on the CPU backend" in said[0]
    assert not any("did not establish" in t for t in fields["limitations"])


def test_a_row_the_two_gpu_runs_disagree_on_is_settled_by_a_third(split, tmp_path, on_macos):
    s = split()
    clean_status, clean = run(job_for(s, tmp_path, backend="gpu"), s["record"])
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.glitch = {(1, 3)}

    status, fields = run(job, s["record"])

    assert clean_status == status == "passed"
    candidate = _candidate_specs(job)
    assert len(candidate) == 3
    n = len(candidate[0]["rows"])
    # Row 3, five times, under indices of their own.
    copies = [r["index"] for r in candidate[2]["rows"]]
    assert sorted(copies) == [n * c + 3 for c in range(1, 6)]
    assert candidate[2]["mode"] == "once"
    assert len({r["prompt"] for r in candidate[2]["rows"]}) == 1
    assert fields["harness"]["gpu_repeat"] == {
        "runs": 3,
        "rows_rescored": 1,
        "engine_rows": [3],
        "observed": [True, True, True],
    }
    # The glitch is gone from what is measured: the clean run's numbers exactly.
    assert fields["quality"] == clean["quality"]
    assert any("1 of " in t and "third run" in t for t in fields["limitations"])


def test_the_third_run_shuffles_its_copies(split, tmp_path, on_macos):
    s = split()
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.glitch = {(1, 3), (1, 5)}

    status, _ = run(job, s["record"])

    assert status == "passed"
    third = _candidate_specs(job)[2]
    n = len(_candidate_specs(job)[0]["rows"])
    built = [n * c + i for c in range(1, 6) for i in (3, 5)]
    shuffled = [r["index"] for r in third["rows"]]
    assert sorted(shuffled) == sorted(built) and shuffled != built


def _engine_row_count(s, tmp_path) -> int:
    probe = job_for(s, tmp_path)
    run(probe, s["record"])
    return len(_candidate_specs(probe)[0]["rows"])


def test_a_row_hit_in_both_runs_is_settled_by_the_copies_in_the_third(split, tmp_path, on_macos):
    s = split()
    n = _engine_row_count(s, tmp_path)
    clean_status, clean = run(job_for(s, tmp_path, backend="gpu"), s["record"])
    job = job_for(s, tmp_path, backend="gpu")
    # Hit in the first two runs and in two of the third run's five copies.
    job.runtime_env.glitch = {(1, 3), (2, 3), (3, n + 3), (3, 2 * n + 3)}

    status, fields = run(job, s["record"])

    assert clean_status == status == "passed"
    assert len(_candidate_specs(job)) == 3
    assert fields["harness"]["gpu_repeat"]["engine_rows"] == [3]
    assert fields["quality"] == clean["quality"]


def test_no_settled_score_in_three_runs_is_a_harness_failure(split, tmp_path, on_macos):
    s = split()
    n = _engine_row_count(s, tmp_path)
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.glitch = {(1, 3), (2, 3)} | {(3, n * c + 3) for c in range(1, 6)}

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    detail = check(fields, RUNTIME_CHECK)["detail"]
    assert "1 rows got no settled score in three GPU runs" in detail
    assert "of the 7 readings of each" in detail
    assert len(_candidate_specs(job)) == 3


def test_a_gpu_run_the_kernel_shows_idle_is_refused(split, tmp_path, on_macos):
    s = split()
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.device = GPU_IDLE

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "the first run did no GPU work" in check(fields, RUNTIME_CHECK)["detail"]


@pytest.mark.parametrize(
    ("run_no", "glitch", "word"),
    [(1, set(), "first"), (2, set(), "second"), (3, {(1, 3)}, "third")],
)
def test_a_repeat_the_kernel_shows_idle_is_refused(split, tmp_path, on_macos, run_no, glitch, word):
    """Each GPU run is held to the kernel's reading, not only the first: an idle
    repeat's scores would otherwise outvote the GPU's."""
    s = split()
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.glitch = glitch
    job.runtime_env.device_for_run = {run_no: GPU_IDLE}

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert f"the {word} run did no GPU work" in check(fields, RUNTIME_CHECK)["detail"]


@pytest.mark.parametrize(
    ("skips", "glitch", "word"),
    [({2: {3}}, set(), "second"), ({3: None}, {(1, 3)}, "third")],
)
def test_a_repeat_that_leaves_rows_unscored_is_refused(
    split, tmp_path, on_macos, skips, glitch, word
):
    s = split()
    n = _engine_row_count(s, tmp_path)
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.glitch = glitch
    # The third run: one of row 3's copies.
    job.runtime_env.skip_for_run = {k: v if v is not None else {n + 3} for k, v in skips.items()}

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert f"the {word} GPU run left rows unscored" in check(fields, RUNTIME_CHECK)["detail"]


@pytest.mark.parametrize(
    ("unseen", "glitch", "said"),
    [
        ({1: NOT_LOOKED}, set(), "the first run"),
        ({2: NOT_LOOKED}, set(), "the second run"),
        ({1: NOT_LOOKED, 3: NOT_LOOKED}, {(1, 3)}, "the first and third run"),
    ],
)
def test_a_gpu_run_the_kernel_did_not_settle_is_recorded_as_asked_for(
    split, tmp_path, on_macos, unseen, glitch, said
):
    s = split()
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.glitch = glitch
    job.runtime_env.device_for_run = unseen

    status, fields = run(job, s["record"])

    assert status == "passed"
    flagged = [t for t in fields["limitations"] if "did not establish" in t]
    assert len(flagged) == 1 and f"GPU work in {said}" in flagged[0]
    assert "not as observed" in flagged[0]


def test_a_first_gpu_run_by_another_method_is_refused_before_the_repeats(split, tmp_path, on_macos):
    s = split()
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.report_extra = {"method": "other"}

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "by method 'other'" in check(fields, RUNTIME_CHECK)["detail"]
    assert len(_candidate_specs(job)) == 1


@pytest.mark.parametrize(
    ("declared", "said"),
    [
        ({"bundle_activation": "fp32"}, None),
        ({"bundle_activation": None}, "declares none, and the runtime's default on a GPU is F16"),
        ({"bundle_activation": "fp16"}, "declares 'fp16'"),
        ({"bundle_activation": None, "bundle_activation_error": "OSError: x"}, "could not read"),
    ],
)
def test_a_gpu_run_says_what_the_bundle_by_itself_declares(
    split, tmp_path, on_macos, declared, said
):
    s = split()
    job = job_for(s, tmp_path, backend="gpu")
    job.runtime_env.report_extra = declared

    status, fields = run(job, s["record"])

    assert status == "passed"
    found = [t for t in fields["limitations"] if "litetune passed fp32 activations" in t]
    if said is None:
        assert found == []
    else:
        assert len(found) == 1 and said in found[0]


def test_the_gpu_with_an_engine_per_option_is_refused_before_scoring(split, tmp_path, on_macos):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(
        s,
        tmp_path,
        backend="gpu",
        engine_per_prompt_reason="state survives a rewind",
        family="qwen-3.5",
        float_twin=twin,
    )

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    detail = check(fields, RUNTIME_CHECK)["detail"]
    assert "an Engine of its own (method fresh_engine)" in detail and "--backend cpu" in detail
    assert _engine_specs(job) == []


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_the_gpu_backend_off_macos_is_refused_before_scoring(
    split, tmp_path, monkeypatch, platform
):
    monkeypatch.setattr(sys, "platform", platform)
    s = split()
    job = job_for(s, tmp_path, backend="gpu")

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    detail = check(fields, RUNTIME_CHECK)["detail"]
    assert "measured on macOS only" in detail and platform in detail
    assert _engine_specs(job) == []


def test_a_backend_other_than_cpu_or_gpu_is_refused(split, tmp_path):
    s = split()
    status, fields = run(job_for(s, tmp_path, backend="npu"), s["record"])
    assert status == "failed_harness"


def test_the_cpu_scores_once_and_records_the_cpu(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed"
    assert [x["backend"] for x in _candidate_specs(job)] == ["cpu"]
    assert "gpu_repeat" not in fields["harness"]
    engine = fields["measurements"]["candidate"]["engine"]
    assert engine["backend"] == "cpu" and engine["backend_observed"] is False
    assert engine["threads"] == job.threads
    assert fields["cost"]["backend"] == "cpu" and fields["cost"]["threads"] == job.threads


def test_settle_keeps_the_scores_most_readings_gave():
    w, r, z = [1.0], [2.0], [9.0]
    # Two readings alike and no other as often.
    assert settle([{1: w}, {1: r}, {1: r}], [1]) == ({1: r}, [])
    # The most readings, not the first value to recur.
    assert settle([{1: w}, {1: r}, {1: w}, {1: r}, {1: r}], [1]) == ({1: r}, [])
    # A tie, whatever the order, settles nothing.
    assert settle([{1: w}, {1: r}, {1: w}, {1: r}, {1: z}], [1]) == ({}, [1])
    assert settle([{1: r}, {1: w}, {1: r}, {1: w}, {1: z}], [1]) == ({}, [1])
    # No two alike, or one reading only.
    assert settle([{1: w}, {1: r}, {1: z}], [1]) == ({}, [1])
    assert settle([{1: w}], [1]) == ({}, [1])
    assert settle([{0: w}, {0: w}], [0]) == ({0: w}, [])


def test_a_family_that_needs_an_engine_per_prompt_is_scored_on_fresh_engines(split, tmp_path):
    """Option scoring rewinds within one session, which is the mechanism an
    Engine per prompt exists to avoid; such a family is scored on an Engine per
    option instead, and the section says so."""
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(
        s,
        tmp_path,
        engine_per_prompt_reason="state survives a rewind",
        family="qwen-3.5",
        float_twin=twin,
    )

    status, fields = run(job, s["record"])

    assert status == "passed", fields["checks"]
    assert [spec["method"] for spec in job.runtime_env.specs] == ["fresh_engine"] * 2
    assert fields["decision"]["section"]["payload"]["method"] == "fresh_engine"
    assert fields["cost"]["method"] == "fresh_engine"
    assert "for every option: an Engine load" in fields["cost"]["measures"]
    said = [t for t in fields["limitations"] if "state survives a rewind" in t]
    assert len(said) == 1 and "an Engine load per option per decision" in said[0]


def test_without_an_engine_per_prompt_the_section_says_rewind(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed"
    assert [spec["method"] for spec in job.runtime_env.specs] == ["rewind"]
    assert fields["cost"]["method"] == "rewind"
    assert not any("an Engine of its own" in t for t in fields["limitations"])


def test_a_runtime_that_scored_by_another_method_is_refused(split, tmp_path):
    s = split()
    eng = FakeScoringEnv("runtime", report_extra={"method": "rewind"})
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(
        s,
        tmp_path,
        runtime_env=eng,
        engine_per_prompt_reason="state survives a rewind",
        family="qwen-3.5",
        float_twin=twin,
    )

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "'fresh_engine' was asked for" in failed(fields, RUNTIME_CHECK)["detail"]


def test_fresh_engines_get_time_for_every_load():
    from litetune.decision_verify import (
        ENGINE_LOAD_TIMEOUT_S,
        ENGINE_ROW_TIMEOUT_S,
        SELFCHECK_LOADS,
        STARTUP_TIMEOUT_S,
        _engine_timeout,
    )

    assert _engine_timeout(10, 6, "rewind") == STARTUP_TIMEOUT_S + 10 * ENGINE_ROW_TIMEOUT_S
    assert _engine_timeout(10, 6, "fresh_engine") == (
        STARTUP_TIMEOUT_S
        + SELFCHECK_LOADS * ENGINE_LOAD_TIMEOUT_S
        + 10 * (ENGINE_ROW_TIMEOUT_S + 6 * ENGINE_LOAD_TIMEOUT_S)
    )


def test_rows_rendered_with_another_decision_are_refused_by_line(split, tmp_path):
    s = split()
    other = decision_from({"question": "Which mood?", "options": [{"id": "joy"}, {"id": "x"}]}, "q")
    bad = tmp_path / "other.jsonl"
    bad.write_text(
        json.dumps({"prompt": other.render("text 1 about joy"), "target": "joy"}) + "\n",
        encoding="utf-8",
    )

    status, fields = run(job_for(s, tmp_path, heldout=bad), s["record"])

    assert status == "failed_harness"
    assert f"{bad}:1: the prompt is not a state rendered" in check(fields, ROWS_CHECK)["detail"]


def test_a_float_twin_within_the_tolerance_passes(split, tmp_path):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(s, tmp_path, float_twin=twin)

    status, fields = run(job, s["record"])

    assert status == "passed"
    assert check(fields, TWIN_CHECK)["outcome"] == "passed"
    twin_spec = job.runtime_env.specs[-1]
    assert twin_spec["model"] == str(twin)
    assert twin_spec["targets"] == "full" and twin_spec["mode"] == "once"
    assert not any("--float-twin" in text for text in fields["limitations"])


def test_a_float_twin_that_drifts_is_a_harness_failure(split, tmp_path):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    calls = {"n": 0}

    def drifting(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        return [base[0] + 0.01, *base[1:]] if calls["n"] else base

    eng = FakeScoringEnv("runtime", scores=drifting)
    real_run = eng.run

    def counting(argv, timeout=None, env=None):
        proc = real_run(argv, timeout, env)
        calls["n"] += 1
        return proc

    eng.run = counting
    job = job_for(s, tmp_path, float_twin=twin, runtime_env=eng)

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert check(fields, TWIN_CHECK)["outcome"] == "failed"
    assert "1.00e-02" in check(fields, TWIN_CHECK)["detail"]


def _drifting_twin(by):
    calls = {"n": 0}

    def drifting(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        return [base[0] + by, *base[1:]] if calls["n"] else base

    eng = FakeScoringEnv("runtime", scores=drifting)
    real_run = eng.run

    def counting(argv, timeout=None, env=None):
        proc = real_run(argv, timeout, env)
        calls["n"] += 1
        return proc

    eng.run = counting
    return eng


@pytest.mark.parametrize(("drift", "outcome"), [(0.02, "passed"), (0.06, "failed")])
def test_a_family_tolerance_holds_the_twin_to_its_own_bound_and_says_why(
    split, tmp_path, drift, outcome
):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(
        s,
        tmp_path,
        float_twin=twin,
        runtime_env=_drifting_twin(drift),
        twin_tolerance=5e-2,
        twin_tolerance_reason="measured so",
    )

    _, fields = run(job, s["record"])

    assert check(fields, TWIN_CHECK)["outcome"] == outcome
    assert "refused above 0.05" in check(fields, TWIN_CHECK)["detail"]
    said = [t for t in fields["limitations"] if "the float twin is held to 0.05" in t]
    assert len(said) == 1 and said[0].endswith("measured so")


def test_the_default_tolerance_says_nothing_about_a_tolerance(split, tmp_path):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")

    _, fields = run(job_for(s, tmp_path, float_twin=twin, family="qwen-3"), s["record"])

    assert not any("the float twin is held to" in t for t in fields["limitations"])


def test_another_runtime_without_a_twin_cannot_be_called_correct(split, tmp_path):
    s = split()
    eng = FakeScoringEnv("runtime", report_extra={"runtime": {"litert-lm": "0.19.0"}})

    status, fields = run(job_for(s, tmp_path, runtime_env=eng), s["record"])

    assert status == "inconclusive"
    assert check(fields, TWIN_CHECK)["outcome"] == "could_not_check"


def test_out_of_scope_rows_the_threshold_lets_through_on_heldout_fail_the_rule(split, tmp_path):
    """Calibration's out-of-scope rows are told apart; the held-out ones look in scope."""
    s = split()
    held = {json.loads(line)["prompt"] for line in s["heldout"].read_text().splitlines()}

    def leaky(index, prompt, option, ids):
        if prompt in held and label_of(prompt) is None:
            return [0.0 if option == "joy" else -9.0] + [-0.01] * (len(ids) - 1)
        return default_scores(index, prompt, option, ids)

    job = job_for(
        s,
        tmp_path,
        train_env=FakeScoringEnv("train", scores=leaky),
        runtime_env=FakeScoringEnv("runtime", scores=leaky),
    )

    status, fields = run(job, s["record"])

    assert status == "failed_gate"
    assert check(fields, RULE_CHECK)["outcome"] == "failed"


@pytest.mark.parametrize(
    ("kind", "text"),
    [
        (
            "noul",
            "this is a noul question; option scoring has trained and measured Choice on several "
            "tasks, and noul on two",
        ),
        (
            "score",
            "this is a score question; option scoring has trained and measured Choice on several "
            "tasks, and score on one only",
        ),
    ],
)
def test_a_question_type_measured_on_fewer_tasks_says_how_many(split, tmp_path, kind, text):
    """Spelled out: MEASUREMENTS.md's decisions section is what this sentence claims."""
    s = split(with_none=False, kind=kind)
    assert s["decision"].type == kind

    _, fields = run(job_for(s, tmp_path), s["record"])

    assert text in fields["limitations"]


def test_a_choice_question_carries_no_type_limitation(split, tmp_path):
    s = split()

    _, fields = run(job_for(s, tmp_path), s["record"])

    # Every wording so far: "Only Choice questions ...", "this is a choice question ...".
    phrase = TYPE_LESS_MEASURED.split("{kind}")[1]
    assert phrase.strip()
    assert not any(
        phrase in text or "choice question" in text.lower() for text in fields["limitations"]
    )


def test_without_out_of_scope_rows_the_error_budget_is_used_and_said(split, tmp_path):
    s = split(with_none=False)

    status, fields = run(job_for(s, tmp_path), s["record"])

    assert fields["decision"]["fit"]["rule"] == calibration.C1
    assert calibration.C1_NOTE in fields["limitations"]
    assert status in ("passed", "inconclusive")
    assert check(fields, CALIBRATION_CHECK)["outcome"] == "passed"


def test_the_kl_and_the_paired_difference_are_reported(split, tmp_path):
    s = split()

    def softer(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        return [base[0] * 0.9, *base[1:]]

    _, fields = run(
        job_for(s, tmp_path, runtime_env=FakeScoringEnv("runtime", scores=softer)), s["record"]
    )

    quality = fields["quality"]
    # The fake's classes separate so cleanly that the fitted temperature makes
    # both distributions almost one-hot; the KL is reported, not large.
    assert math.isfinite(quality["mean_kl"]) and quality["mean_kl"] > -1e-12
    assert quality["max_kl"] >= quality["mean_kl"]
    assert quality["accuracy_difference"]["method"] == "paired (McNemar)"


# ---------------------------------------------------------------------------
# The runtime script itself, against a fake `litert_lm`
# ---------------------------------------------------------------------------

FAKE_LITERT_LM = r'''
"""A pure-Python stand-in for litert_lm 0.18.0's session API, enough for ENGINE_SCRIPT.

Tokens are special markers, words, or single characters. Scoring is a fixed
function of the whole context the session reads. With FAKE_DROPS=1 the session
reads a prompt prefilled once without its second-to-last token, as LiteRT-LM
0.18.0 does (#3561), and prefilling its last token again restores it; without
it the session reads what was prefilled, as an engine with the fix does; with
FAKE_DROPS=alternate it drops on every other pair of scorings, an engine nobody
has measured. Scoring an option advances the session past it until a rewind,
and a rewind needs a saved checkpoint, so the order of the calls matters.

FAKE_TEMPLATE picks the bundle's template: Gemma 3's (the default) or ChatML
("chatml", as Qwen's). FAKE_BOS picks its start token: "<bos>" (the default),
or "none" for a bundle with none, as Qwen's. A session reads the BOS first on
its first prefill, as LiteRT-LM's does. With FAKE_LOADS set, every Engine
appends a line to that file, so a test can count loads.
"""
import os
import re
import zlib

_SPECIAL = r"<start_of_turn>|<end_of_turn>|<bos>|<\|im_start\|>|<\|im_end\|>"
_TEMPLATES = {
    "gemma": ("<start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
    "chatml": ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"),
    # A template that prints the BOS string itself, as MiniCPM5's own does.
    "gemma_bos": ("<bos><start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
    # Made up for the self-check's pair: its second id is its second-to-last,
    # and its third, put there, runs into the last as one word.
    "tricky": ("<start_of_turn>:a\n", "<end_of_turn>\n<start_of_turn>:b"),
}
_WORDS = {}
_SCORINGS = 0


def _ids(text, bos_special=True):
    """One id per piece, the same in every process: crc32 of the piece.

    `Engine.tokenize` reads the BOS string as text, as LiteRT-LM's tokenizer
    does (`rendering.prefill_ids`); a session's prefill reads it as the BOS.
    """
    out = []
    special = _SPECIAL if bos_special else _SPECIAL.replace("|<bos>", "")
    for piece in re.findall(special + r"|[A-Za-z_]+|.", text, flags=re.S):
        token = zlib.crc32(piece.encode("utf-8")) % 1_000_003 + 1
        _WORDS[token] = piece
        out.append(token)
    return out


def true_score(context, token):
    return -((hash(tuple(context)) + 31 * token) % 997) / 100.0


class LogSeverity:
    ERROR = 3


def set_min_log_severity(level):
    pass


class SamplerConfig:
    def __init__(self, **kw):
        pass


class Backend:
    @staticmethod
    def CPU(thread_count=1):
        return ("cpu", thread_count)


class _CFunc:
    def __call__(self, ptr, name):
        return 0


class _Save:
    def __call__(self, ptr, name):
        ptr.saved = True
        return 0


class _Rewind:
    def __call__(self, ptr, name):
        if not ptr.saved:
            return -1
        ptr.pending = []
        return 0


class _Lib:
    litert_lm_session_save_checkpoint = _Save()
    litert_lm_session_rewind_to_checkpoint = _Rewind()
    litert_lm_get_last_error_message = _CFunc()


class _Result:
    def __init__(self, scores):
        self.token_scores = [scores]


class _Session:
    def __init__(self, bos=None):
        self.bos = bos
        self.tokens = [] if bos is None else [bos]
        self.prefills = 0
        self._lib = _Lib()
        self._ptr = self
        self.pending = []
        self.saved = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run_prefill(self, parts):
        for part in parts:
            bos_text = None if self.bos is None else _WORDS[self.bos]
            if bos_text and part.startswith(bos_text):
                self.tokens += [self.bos] + _ids(part[len(bos_text) :], False)
            else:
                self.tokens += _ids(part, False)
        self.prefills += 1

    def _context(self):
        global _SCORINGS
        _SCORINGS += 1
        t = self.tokens
        drops = os.environ.get("FAKE_DROPS")
        if drops == "alternate":
            if ((_SCORINGS - 1) // 2) % 2:
                return t
        elif drops != "1":
            return t
        if self.prefills == 1 and len(t) >= 2:
            return t[:-2] + t[-1:]
        if self.prefills >= 2 and len(t) >= 2 and t[-1] == t[-2]:
            return t[:-1]
        return t

    def run_text_scoring(self, texts, store_token_lengths=False):
        ctx, target = list(self._context()) + self.pending, _ids(texts[0])
        self.pending = self.pending + target
        return _Result([true_score(ctx + target[:i], target[i]) for i in range(len(target))])


class _Conversation:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def render_message_to_string(self, prompt):
        before, after = _TEMPLATES[os.environ.get("FAKE_TEMPLATE", "gemma")]
        return before + prompt + after


class Engine:
    def __init__(self, path, backend=None, cache_dir=None):
        self.path = path
        bos = os.environ.get("FAKE_BOS", "<bos>")
        self.bos_token_id = None if bos == "none" else _ids(bos)[0]
        if os.environ.get("FAKE_LOADS"):
            with open(os.environ["FAKE_LOADS"], "a") as fh:
                fh.write("load\n")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def tokenize(self, text):
        return _ids(text, False)

    def detokenize(self, ids):
        return "".join(_WORDS[i] for i in ids)

    def create_session(self, apply_prompt_template=True, sampler_config=None):
        return _Session(self.bos_token_id)

    def create_conversation(self, sampler_config=None, max_output_tokens=None):
        return _Conversation()
'''


def _run_engine_script(
    tmp_path,
    drops: bool | str,
    prompts,
    options,
    mode="auto",
    fake_source=FAKE_LITERT_LM,
    fake_env: dict[str, str] | None = None,
    term: str = TERM,
    method: str = "rewind",
    script_tail: str = "",
    **spec_extra: Any,
):
    import os
    import sys

    from litetune.decision_scripts import ENGINE_SCRIPT

    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "litert_lm.py").write_text(fake_source, encoding="utf-8")
    script = tmp_path / "engine.py"
    source = ENGINE_SCRIPT
    if script_tail:
        # Definitions that replace the script's own, ahead of its entry point.
        cut = source.rindex('if __name__ == "__main__":')
        source = source[:cut] + script_tail + "\n\n" + source[cut:]
    script.write_text(source, encoding="utf-8")
    spec = tmp_path / "spec.json"
    out, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    spec.write_text(
        json.dumps(
            {
                "model": "m.litertlm",
                "rows": [{"index": i, "prompt": p} for i, p in enumerate(prompts)],
                "options": list(options),
                "terminator": term,
                "threads": 2,
                "targets": "unique",
                "mode": mode,
                "method": method,
                "out": str(out),
                "report": str(report),
                **spec_extra,
            }
        ),
        encoding="utf-8",
    )
    drops_value = drops if isinstance(drops, str) else "1" if drops else "0"
    env = dict(os.environ, PYTHONPATH=str(fake), FAKE_DROPS=drops_value, **(fake_env or {}))
    proc = subprocess.run(
        [sys.executable, str(script), str(spec)], env=env, capture_output=True, text=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    return proc, rows, json.loads(report.read_text())


def _fake_module(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fake_litert_lm", tmp_path / "fake" / "litert_lm.py"
    )
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    return fake


def _expected(tmp_path, prompts, options, template="gemma", bos="<bos>", term=TERM):
    """Each option's unique-prefix token scores on the BOS and the whole rendered prompt."""
    fake = _fake_module(tmp_path)
    cont = [fake._ids(o + term) for o in options]
    ks = unique_lengths(cont)
    before, after = fake._TEMPLATES[template]
    first = [] if bos == "none" else fake._ids(bos)
    out = []
    for p in prompts:
        ctx = first + fake._ids(before + p + after)
        out.append(
            [
                [fake.true_score(ctx + c[:i], c[i]) for i in range(k)]
                for c, k in zip(cont, ks, strict=True)
            ]
        )
    return out


@pytest.mark.parametrize(("drops", "mode"), [(True, "once"), (False, "plain")])
def test_the_self_check_picks_the_mode_that_scores_the_whole_prompt(tmp_path, drops, mode):
    prompts = ["text 1 about joy", "text 22 about fear"]
    options = ("joy", "anger", "fear", "joyful")

    proc, rows, report = _run_engine_script(tmp_path, drops, prompts, options)

    assert proc.returncode == 0, proc.stderr
    assert report["mode"] == mode
    assert report["selfcheck"]["verdict"] == ("dropped" if drops else "not dropped")
    assert report["selfcheck"]["comparisons"] > 0
    assert [r["token_scores"] for r in rows] == _expected(tmp_path, prompts, options)


@pytest.mark.parametrize(("drops", "wrong"), [(True, "plain"), (False, "once")])
def test_the_other_mode_is_wrong_on_each_engine(tmp_path, drops, wrong):
    """What the self-check is for: forcing the other mode reads another context."""
    prompts = ["text 1 about joy"]
    options = ("joy", "anger", "fear")

    proc, rows, _ = _run_engine_script(tmp_path, drops, prompts, options, mode=wrong)

    assert proc.returncode == 0, proc.stderr
    assert [r["token_scores"] for r in rows] != _expected(tmp_path, prompts, options)


def test_the_runtime_script_scores_unique_prefixes(tmp_path):
    proc, rows, report = _run_engine_script(
        tmp_path, True, ["text 1 about joy"], ("joy", "joyful", "anger")
    )

    assert proc.returncode == 0, proc.stderr
    assert report["k"] == unique_lengths(report["cont_ids"])
    assert [len(t) for t in rows[0]["token_scores"]] == report["scored_lengths"]
    assert report["mismatches"] == []


# ---------------------------------------------------------------------------
# Through `run_verify` and the CLI
# ---------------------------------------------------------------------------


def test_run_verify_measures_a_decision_checkpoint_by_option_scoring(split, tmp_path, monkeypatch):
    from litetune import envs
    from litetune.cli import summarise
    from litetune.verify import MANIFEST_SCHEMA, Status, VerifyRequest, run_verify

    s = split()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "litetune.json").write_text(json.dumps(s["record"]), encoding="utf-8")
    # As `tune` leaves it: the family is read from the record's base model.
    (checkpoint / "config.json").write_text('{"model_type": "gemma3_text"}', encoding="utf-8")
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    fake = FakeScoringEnv("any")
    monkeypatch.setattr(
        envs.StageEnv, "run", lambda self, argv, timeout=None, env=None: fake.run(argv, timeout)
    )
    monkeypatch.setattr(envs.StageEnv, "provision", lambda self, events=None, force=False: None)

    result = run_verify(
        VerifyRequest(
            model=model,
            reference=str(checkpoint),
            data=s["heldout"],
            calibration=s["calibration"],
            engine_rows=7,
        )
    )

    assert result.status is Status.PASSED
    assert result.exit_code == 0
    manifest = result.manifest
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["status"] == "passed"
    assert manifest["model"]["sha256"]
    assert manifest["scorer"]["name"] == "option-scoring"
    assert manifest["quality"]["rows"] == 7
    lines = summarise(manifest)
    assert lines[0] == "status: passed"
    assert any(line.startswith("  decision: choice, 4 options, temperature ") for line in lines)
    assert any(line.startswith("  cost: ") and "mode once" in line for line in lines)


def test_a_checkpoint_with_no_decision_and_no_calibration_is_verified_as_before(tmp_path):
    from litetune.verify import _decision_record

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "litetune.json").write_text(
        json.dumps({"base_model": "google/gemma-3-270m-it", "decision": None}), encoding="utf-8"
    )

    assert _decision_record(str(checkpoint)) == (None, None)
    assert _decision_record("google/gemma-3-270m-it") == (None, None)


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------


def test_an_option_is_scored_on_its_unique_prefix_only():
    from litetune.decision_verify import summed

    assert summed([[-1.0, -2.0, -3.0], [-4.0, -5.0]], [1, 2]) == [-1.0, -9.0]


def test_accuracy_is_over_in_scope_rows_and_out_of_scope_rows_are_counted_apart():
    from litetune.decision_verify import answers, outcome_metrics

    fit = calibration.Fit(1.0, calibration.C2, 0.0, 3, 1, 0.0, 0.0)
    # Three in-scope rows answered right; one out-of-scope row answered as "joy".
    scores = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    metrics = outcome_metrics(answers(scores, 1.0), [0, 1, 0, 2], fit, 2, "choice")

    assert metrics["accuracy"] == 1.0
    assert metrics["in_scope_rows"] == 3
    assert metrics["out_of_scope_rows"] == 1
    assert metrics["out_of_scope_rejected"] == 0.0


def test_a_heldout_row_whose_target_is_not_an_option_is_refused(split, tmp_path):
    s = split()
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"prompt": s["decision"].render("text 1 about joy"), "target": "love"}) + "\n",
        encoding="utf-8",
    )

    status, fields = run(job_for(s, tmp_path, heldout=bad), s["record"])

    assert status == "failed_harness"
    assert "the target 'love' is not one of" in check(fields, ROWS_CHECK)["detail"]


def test_the_reference_scores_of_heldout_rows_are_the_ones_measured(split, tmp_path):
    """The reference scores calibration rows first, held-out rows after them."""
    s = split()
    job = job_for(s, tmp_path)

    _, fields = run(job, s["record"])

    held = [json.loads(line) for line in s["heldout"].read_text().splitlines()]
    in_scope = [r for r in held if r["target"] != "none_of_these"]
    assert fields["measurements"]["reference"]["rows"] == len(held)
    assert fields["measurements"]["reference"]["in_scope_rows"] == len(in_scope)
    assert fields["measurements"]["reference"]["accuracy"] == expected_accuracy(held)


# ---------------------------------------------------------------------------
# What the review found
# ---------------------------------------------------------------------------


def test_the_scorers_get_the_first_token_of_the_recorded_end_of_turn(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed"
    assert job.runtime_env.specs[0]["terminator"] == "<end_of_turn>"
    assert fields["decision"]["section"]["payload"]["terminator"] == "<end_of_turn>"


@pytest.mark.parametrize(
    ("ids", "text", "first"),
    [
        ([151645, 107], "<|im_end|>\n", "<|im_end|>"),
        ([100257, 107], "<|end_of_text|>\n", "<|end_of_text|>"),
    ],
    ids=["ChatML", "Granite"],
)
def test_another_familys_end_of_turn_is_scored_with_its_own_first_token(
    split, tmp_path, ids, text, first
):
    s = split()
    record = dict(
        s["record"], turn_terminator={"ids": ids, "source": "chat_template", "text": text}
    )
    job = job_for(s, tmp_path)

    status, fields = run(job, record)

    assert status == "passed", fields["checks"]
    assert job.train_env.specs[0]["turn_ids"] == ids
    assert job.runtime_env.specs[0]["terminator"] == first
    assert fields["decision"]["section"]["payload"]["terminator"] == first


@pytest.mark.parametrize(
    ("recorded", "said"),
    [
        (None, "records no end of turn"),
        ("<end_of_turn>\n", "records no end of turn"),
        (
            {"ids": [1], "source": "tokenizer_eos", "text": "<eos>"},
            "from 'tokenizer_eos', not from its chat template",
        ),
        ({"ids": [], "source": "chat_template", "text": ""}, "has no ids or no text"),
    ],
    ids=["none", "a string", "the tokenizer's EOS", "empty"],
)
def test_an_end_of_turn_not_from_the_template_is_refused_before_scoring(
    split, tmp_path, recorded, said
):
    s = split()
    record = dict(s["record"], turn_terminator=recorded)
    job = job_for(s, tmp_path)

    status, fields = run(job, record)

    assert status == "failed_harness"
    assert said in failed(fields, ROWS_CHECK)["detail"]
    assert job.train_env.specs == [] and job.runtime_env.specs == []


def test_a_reference_whose_terminator_does_not_begin_the_recorded_turn_is_refused(split, tmp_path):
    s = split()
    job = job_for(
        s, tmp_path, train_env=FakeScoringEnv("train", vocab={**VOCAB, 106: "<|im_end|>"})
    )

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "which it does not begin with" in failed(fields, REFERENCE_CHECK)["detail"]
    assert job.train_env.specs and job.runtime_env.specs == []


def test_options_scored_in_full_on_the_runtime_are_scored_in_full_on_both_sides(split, tmp_path):
    s = split()

    def tail_heavy(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        # The tail decides: summed over the unique prefix only, "anger" would win
        # every row; summed in full, only its own.
        if option == "anger":
            tail = 0.0 if label_of(prompt) == "anger" else -30.0
            return [0.5, *([tail] * (len(ids) - 1))]
        return base

    eng = FakeScoringEnv("runtime", scores=tail_heavy, full_for={"anger"})
    job = job_for(
        s, tmp_path, train_env=FakeScoringEnv("train", scores=tail_heavy), runtime_env=eng
    )

    status, fields = run(job, s["record"])

    assert "scored_lengths" in fields["decision"], fields["checks"]
    lengths = fields["decision"]["scored_lengths"]
    anger = s["decision"].ids.index("anger")
    assert lengths[anger] == len(tokens("anger"))
    assert fields["quality"]["argmax_agreement"] == 1.0
    assert "scored in full on both sides" in check(fields, TARGETS_CHECK)["detail"]
    assert status == "passed"


def test_a_runtime_that_abstains_where_the_reference_answers_is_refused(split, tmp_path):
    s = split(with_none=False)

    def unsure(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        return [base[0] * 0.05, *base[1:]]

    eng = FakeScoringEnv("runtime", scores=unsure)

    status, fields = run(job_for(s, tmp_path, runtime_env=eng), s["record"])

    assert fields["quality"]["argmax_agreement"] == 1.0
    assert fields["quality"]["answer_agreement"] < 0.95
    assert status == "failed_gate"
    assert check(fields, AGREEMENT_CHECK)["outcome"] == "failed"


def test_calibrating_on_the_heldout_rows_is_refused(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path, calibration=s["heldout"]), s["record"])

    assert status == "failed_harness"
    assert "the same rows" in check(fields, CALIBRATION_CHECK)["detail"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"reference_role": "untuned_base"}, "--reference-role"), ({"limit": 10}, "--limit")],
)
def test_what_the_decision_path_does_not_do_is_refused_not_ignored(
    split, tmp_path, kwargs, message
):
    s = split()

    status, fields = run(job_for(s, tmp_path, **kwargs), s["record"])

    assert status == "failed_harness"
    assert message in check(fields, ROWS_CHECK)["detail"]


def test_an_unreadable_record_is_named_as_unreadable(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path, record_error="ValueError: not JSON"), None)

    assert status == "failed_harness"
    assert "could not be read (ValueError: not JSON)" in check(fields, ROWS_CHECK)["detail"]


def test_a_reference_score_that_is_not_finite_is_refused(split, tmp_path):
    s = split()

    def nan_once(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        # Every "joy" row: some are surely among the rows the reference scores.
        return [float("nan"), *base[1:]] if label_of(prompt) == "joy" else base

    job = job_for(s, tmp_path, train_env=FakeScoringEnv("train", scores=nan_once))

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "not finite" in check(fields, REFERENCE_CHECK)["detail"]


def test_a_script_that_writes_no_report_is_a_harness_failure(split, tmp_path):
    s = split()

    class Silent(FakeScoringEnv):
        def run(self, argv, timeout=None, env=None):
            proc = super().run(argv, timeout, env)
            Path(json.loads(Path(argv[2]).read_text())["report"]).unlink()
            return proc

    status, fields = run(job_for(s, tmp_path, train_env=Silent("train")), s["record"])

    assert status == "failed_harness"
    assert "wrote no report" in check(fields, REFERENCE_CHECK)["detail"]


def test_a_well_trained_none_option_ships_a_threshold_json_can_hold(split, tmp_path):
    """Every out-of-scope calibration row already picks the none option: the rule's -inf."""
    from litetune.decision_section import payload_bytes

    s = split()

    status, fields = run(job_for(s, tmp_path), s["record"])

    payload = fields["decision"]["section"]["payload"]
    assert payload["abstain_below"] == 0.0
    assert json.loads(payload_bytes(payload).decode("utf-8"), parse_constant=None) == payload
    assert status == "passed"


def test_no_runtime_rows_is_refused_rather_than_divided_by(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path, engine_rows=0), s["record"])

    assert status == "failed_harness"
    assert "engine_rows is 0" in check(fields, RUNTIME_CHECK)["detail"]


def test_plain_on_the_measured_runtime_without_a_twin_is_not_called_correct(split, tmp_path):
    s = split()
    eng = FakeScoringEnv("runtime", report_extra={"mode": "plain"})

    status, fields = run(job_for(s, tmp_path, runtime_env=eng), s["record"])

    assert status == "inconclusive"
    assert "the measured method is 'once'" in check(fields, TWIN_CHECK)["detail"]


def test_the_reference_checks_its_longest_row_against_the_uncached_scores(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path)

    run(job, s["record"])

    spec = job.train_env.specs[0]
    longest = max(spec["rows"], key=lambda r: len(r["prompt"]))
    assert spec["check_indices"] == [longest["index"]]


# ---------------------------------------------------------------------------
# What the review of 2026-10-17 found
# ---------------------------------------------------------------------------


def failed(fields: dict[str, Any], name: str) -> dict[str, Any]:
    """The failed check of that name: a refusal can follow a passed one of the same name."""
    return next(
        c for c in fields["checks"]["checks"] if c["name"] == name and c["outcome"] == "failed"
    )


@pytest.mark.parametrize(
    ("total", "count", "picked"),
    [(3, 2, [0, 2]), (10, 5, [1, 3, 5, 7, 9]), (4, 4, [0, 1, 2, 3]), (2, 5, [0, 1]), (7, 1, [3])],
)
def test_the_runtime_rows_are_spread_to_the_end_of_the_split(total, count, picked):
    from litetune.decision_verify import spread

    assert spread(total, count) == picked


def test_summed_refuses_an_option_with_fewer_token_scores_than_its_length():
    with pytest.raises(ValueError, match="option 0 has 1 token score"):
        summed([[-1.0], [-2.0, -3.0]], [2, 2])


def test_a_reference_that_scored_fewer_tokens_than_the_runtime_is_refused(split, tmp_path):
    s = split()

    def short_anger(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        return base[:1] if option == "anger" else base

    job = job_for(
        s,
        tmp_path,
        train_env=FakeScoringEnv("train", scores=short_anger),
        runtime_env=FakeScoringEnv("runtime", full_for={"anger"}),
    )

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "too few tokens" in failed(fields, REFERENCE_CHECK)["detail"]


@dataclass
class ExitingEnv:
    """A stage environment whose script writes nothing: it exits, times out, or never starts."""

    returncode: int = 1
    stderr: str = ""
    raises: BaseException | None = None
    name: str = "runtime"

    def provision(self, events=None, force: bool = False) -> None:
        return None

    def run(self, argv, timeout=None, env=None) -> subprocess.CompletedProcess:
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode, "", self.stderr)


def test_a_script_that_dies_before_its_report_says_why():
    said = "Traceback (most recent call last):\nRuntimeError: Failed to create LiteRT-LM engine"
    got = run_script(ExitingEnv(stderr=said), "pass", {}, timeout=5, auto_provision=False)

    assert got.failure is not None and got.failure.startswith("the script exited 1:")
    assert "Failed to create LiteRT-LM engine" in got.failure


def test_a_script_that_exits_cleanly_without_a_report_says_so_with_its_stderr():
    env = ExitingEnv(returncode=0, stderr="a warning on the way out")
    got = run_script(env, "pass", {}, timeout=5, auto_provision=False)

    assert got.failure == "the script wrote no report: a warning on the way out"


def test_a_timeout_keeps_the_end_of_the_scripts_stderr():
    expired = subprocess.TimeoutExpired(
        ["python"], 5, stderr=b"RuntimeError: litert_lm_session_rewind_to_checkpoint returned 3"
    )
    got = run_script(ExitingEnv(raises=expired), "pass", {}, timeout=5, auto_provision=False)

    assert got.failure is not None and got.failure.startswith("no result after 5 s (timeout): ")
    assert "rewind_to_checkpoint returned 3" in got.failure


def test_a_script_that_cannot_start_is_a_failure_not_an_exception():
    missing = FileNotFoundError(2, "No such file or directory", "python")
    got = run_script(ExitingEnv(raises=missing), "pass", {}, timeout=5, auto_provision=False)

    assert got.failure is not None
    assert got.failure.startswith("the script could not be started: FileNotFoundError")


@pytest.mark.parametrize("which", ["model", "float_twin"])
def test_a_missing_model_or_twin_is_refused_before_the_reference_runs(split, tmp_path, which):
    s = split()
    job = job_for(s, tmp_path, **{which: tmp_path / "nowhere.litertlm"})

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "is not a file" in failed(fields, RUNTIME_CHECK)["detail"]
    assert job.train_env.specs == [], "the reference never ran"


def test_both_sides_scoring_the_same_prompts_is_a_passed_check(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path), s["record"])

    assert status == "passed", fields["checks"]
    assert check(fields, PROMPT_CHECK)["outcome"] == "passed"


@pytest.mark.parametrize(
    "tokenised",
    [
        lambda p: [BOS, *prompt_ids(p)],
        lambda p: [],
        lambda p: prompt_ids(p + " "),
    ],
    ids=["a second BOS", "an empty render", "another text"],
)
def test_a_runtime_that_scores_another_prompt_is_refused(split, tmp_path, tokenised):
    s = split()
    eng = FakeScoringEnv("runtime", prompt_ids_of=tokenised)

    status, fields = run(job_for(s, tmp_path, runtime_env=eng), s["record"])

    assert status == "failed_harness"
    assert "is not the reference's" in failed(fields, PROMPT_CHECK)["detail"]


def test_the_runtime_script_refuses_a_prompt_rendered_as_empty_text(tmp_path):
    empty = FAKE_LITERT_LM.replace("return before + prompt + after", 'return ""')
    assert empty != FAKE_LITERT_LM

    # In a mode given, not picked: the self-check renders through the same call.
    proc, rows, _ = _run_engine_script(
        tmp_path, True, ["text 1 about joy"], OPTIONS, mode="once", fake_source=empty
    )

    assert proc.returncode == 0, proc.stderr
    assert [r.get("error") for r in rows] == ["the runtime rendered the prompt as empty text"]


def test_the_runtime_script_returns_each_rows_prompt_ids(tmp_path):
    import importlib.util

    proc, rows, _ = _run_engine_script(tmp_path, True, ["text 1 about joy"], OPTIONS)

    spec = importlib.util.spec_from_file_location("fake_ids", tmp_path / "fake" / "litert_lm.py")
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    rendered = "<start_of_turn>user\ntext 1 about joy<end_of_turn>\n<start_of_turn>model\n"
    assert proc.returncode == 0, proc.stderr
    assert rows[0]["prompt_ids"] == fake._ids(rendered)


def test_a_threshold_that_answers_no_in_scope_row_fails_the_rule(split, tmp_path):
    s = split()

    def confident_outside(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        if label_of(prompt) is None:
            # Out of scope, read as "joy" by more than any in-scope row leads.
            return [0.0 if option == "joy" else -9.0] + base[1:]
        return base

    job = job_for(
        s,
        tmp_path,
        train_env=FakeScoringEnv("train", scores=confident_outside),
        runtime_env=FakeScoringEnv("runtime", scores=confident_outside),
    )

    status, fields = run(job, s["record"])

    assert fields["decision"]["fit"]["abstain_below"] <= 1.0
    assert fields["measurements"]["reference"]["answered"] == 0
    assert "would answer nothing" in failed(fields, RULE_CHECK)["detail"]
    assert status == "failed_gate"


def test_fewer_than_ten_out_of_scope_calibration_rows_are_said_to_set_the_threshold(
    split, tmp_path
):
    s = split(oos=12)

    status, fields = run(job_for(s, tmp_path), s["record"])

    out_of_scope = fields["decision"]["fit"]["out_of_scope_rows"]
    assert 0 < out_of_scope < 10
    assert any("most confidently misread one" in text for text in fields["limitations"])


def test_enough_out_of_scope_calibration_rows_are_not_called_thin(split, tmp_path):
    s = split(oos=80)

    _, fields = run(job_for(s, tmp_path), s["record"])

    assert fields["decision"]["fit"]["out_of_scope_rows"] >= 10
    assert not any("most confidently misread one" in text for text in fields["limitations"])


def test_a_recorded_decision_without_its_digest_is_refused(split, tmp_path):
    s = split()
    record = {k: v for k, v in s["record"].items() if k != "decision_sha256"}

    status, fields = run(job_for(s, tmp_path), record)

    assert status == "failed_harness"
    assert "a record without its digest" in failed(fields, ROWS_CHECK)["detail"]


def test_the_reference_runs_where_litetune_device_says(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path, device="cpu")

    run(job, s["record"])

    assert job.train_env.specs[0]["device"] == "cpu"


def test_a_heldout_error_clearly_over_the_budget_fails_the_rule(split, tmp_path):
    s = split(with_none=False)
    held = {json.loads(line)["prompt"] for line in s["heldout"].read_text().splitlines()}

    def wrong_on_heldout(index, prompt, option, ids):
        if prompt in held:
            right = label_of(prompt)
            wrong = "anger" if right == "joy" else "joy"
            return [0.0 if option == wrong else -9.0] + [-0.01] * (len(ids) - 1)
        return default_scores(index, prompt, option, ids)

    job = job_for(
        s,
        tmp_path,
        train_env=FakeScoringEnv("train", scores=wrong_on_heldout),
        runtime_env=FakeScoringEnv("runtime", scores=wrong_on_heldout),
    )

    status, fields = run(job, s["record"])

    assert fields["decision"]["fit"]["rule"] == calibration.C1
    assert failed(fields, RULE_CHECK)["outcome"] == "failed"
    assert status == "failed_gate"


def test_a_failed_gate_outranks_an_unchecked_twin(split, tmp_path):
    s = split()

    def swapped(index, prompt, option, ids):
        right = label_of(prompt) or "none_of_these"
        wrong = "anger" if right == "joy" else "joy"
        return [0.0 if option == wrong else -5.0] + [-0.01] * (len(ids) - 1)

    eng = FakeScoringEnv(
        "runtime", scores=swapped, report_extra={"runtime": {"litert-lm": "0.19.0"}}
    )

    status, fields = run(job_for(s, tmp_path, runtime_env=eng), s["record"])

    assert check(fields, AGREEMENT_CHECK)["outcome"] == "failed"
    assert check(fields, TWIN_CHECK)["outcome"] == "could_not_check"
    assert status == "failed_gate"


def test_scored_lengths_shorter_than_the_unique_prefix_are_refused(split, tmp_path):
    s = split()
    eng = FakeScoringEnv("runtime", report_extra={"scored_lengths": [0, 0, 0, 0]})

    status, fields = run(job_for(s, tmp_path, runtime_env=eng), s["record"])

    assert status == "failed_harness"
    assert "unusable scored lengths" in failed(fields, TARGETS_CHECK)["detail"]


def test_a_self_check_that_sees_both_behaviours_refuses(tmp_path):
    proc, rows, report = _run_engine_script(
        tmp_path, "alternate", ["text 1 about joy"], ("joy", "anger")
    )

    assert proc.returncode == 3, (proc.stdout, proc.stderr, report)
    assert report["selfcheck"]["verdict"] == "mixed"
    assert rows == []


# -- through run_verify ---------------------------------------------------------------


def _checkpoint(tmp_path, record) -> Path:
    cp = tmp_path / "checkpoint"
    cp.mkdir(exist_ok=True)
    (cp / "litetune.json").write_text(json.dumps(record), encoding="utf-8")
    return cp


@pytest.mark.parametrize("flag", ["calibration", "float_twin"])
def test_decision_flags_for_a_checkpoint_without_a_decision_are_refused(split, tmp_path, flag):
    from litetune.verify import VerifyRequest, run_verify

    s = split()
    cp = _checkpoint(tmp_path, {"base_model": "google/gemma-3-270m-it"})
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    value = s["calibration"] if flag == "calibration" else tmp_path / "twin.litertlm"

    result = run_verify(
        VerifyRequest(model=model, reference=str(cp), data=s["heldout"], **{flag: value})
    )

    assert result.manifest["scorer"]["name"] == "option-scoring"
    assert result.status.value == "failed_harness"
    details = [c["detail"] for c in result.manifest["checks"]["checks"]]
    assert any("records no decision" in d for d in details)


def test_run_verify_hands_every_decision_field_to_the_decision_path(tmp_path, monkeypatch):
    import litetune.decision_verify as dv
    from litetune.verify import ReferenceRole, VerifyRequest, run_verify

    seen = {}

    def capture(job, record, events):
        seen["job"] = job
        return "failed_harness", {"scorer": {"name": "option-scoring"}, "limitations": []}

    monkeypatch.setattr(dv, "verify_decision", capture)
    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    cp = _checkpoint(tmp_path, {"decision": {"question": "Q?", "options": [{"id": "a"}]}})
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")

    run_verify(
        VerifyRequest(
            model=model,
            reference=str(cp),
            data=tmp_path / "heldout.jsonl",
            calibration=tmp_path / "calibration.jsonl",
            float_twin=tmp_path / "twin.litertlm",
            engine_rows=7,
            threads=8,
            backend="gpu",
            reference_role=ReferenceRole("untuned_base"),
            limit=10,
        )
    )

    job = seen["job"]
    assert job.float_twin == tmp_path / "twin.litertlm"
    assert (job.engine_rows, job.threads, job.backend) == (7, 8, "gpu")
    assert (job.reference_role, job.limit) == ("untuned_base", 10)
    assert (job.device, job.device_error, job.not_applicable) == ("cpu", None, ())


def test_verify_flags_a_decision_does_not_use_are_refused_by_name(split, tmp_path):
    from litetune.verify import VerifyRequest, run_verify

    s = split()
    cp = _checkpoint(tmp_path, s["record"])
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")

    result = run_verify(
        VerifyRequest(
            model=model,
            reference=str(cp),
            data=s["heldout"],
            calibration=s["calibration"],
            scorer="exact-text",
            max_conversion_cost=0.0,
        )
    )

    assert result.status.value == "failed_harness"
    detail = next(
        c["detail"] for c in result.manifest["checks"]["checks"] if c["outcome"] == "failed"
    )
    assert "--scorer, --max-conversion-cost do(es) not apply" in detail


def test_a_litetune_device_litetune_does_not_accept_stops_a_decision_run(
    split, tmp_path, monkeypatch
):
    from litetune.verify import VerifyRequest, run_verify

    s = split()
    cp = _checkpoint(tmp_path, s["record"])
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    monkeypatch.setenv("LITETUNE_DEVICE", "gpu")

    result = run_verify(
        VerifyRequest(
            model=model, reference=str(cp), data=s["heldout"], calibration=s["calibration"]
        )
    )

    assert result.status.value == "failed_harness"
    details = [c["detail"] for c in result.manifest["checks"]["checks"]]
    assert any("verification was not attempted" in d and "LITETUNE_DEVICE" in d for d in details)


@pytest.mark.parametrize(
    ("base", "needs"), [("Qwen/Qwen3.5-0.8B", True), ("google/gemma-3-270m-it", False)]
)
def test_run_verify_hands_the_rules_engine_per_prompt_reason_over(
    tmp_path, monkeypatch, base, needs
):
    import litetune.decision_verify as dv
    from litetune.verify import VerifyRequest, run_verify

    seen = {}

    def capture(job, record, events):
        seen["job"] = job
        return "failed_harness", {"scorer": {"name": "option-scoring"}, "limitations": []}

    monkeypatch.setattr(dv, "verify_decision", capture)
    monkeypatch.setenv("LITETUNE_DEVICE", "cpu")
    cp = _checkpoint(
        tmp_path,
        {"base_model": base, "decision": {"question": "Q?", "options": [{"id": "a"}]}},
    )
    config = {"model_type": "qwen3_5"} if needs else {"model_type": "gemma3_text"}
    (cp / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    run_verify(VerifyRequest(model=model, reference=str(cp), data=tmp_path / "heldout.jsonl"))
    assert bool(seen["job"].engine_per_prompt_reason) is needs
    if needs:
        assert seen["job"].twin_tolerance == 5e-2
        assert "2.65e-02" in seen["job"].twin_tolerance_reason
    else:
        assert seen["job"].twin_tolerance == TWIN_TOLERANCE
        assert seen["job"].twin_tolerance_reason == ""
    assert seen["job"].family == ("qwen-3.5" if needs else "gemma-3-text")


# -- the reference script's end of turn, with torch and transformers faked -----------

FAKE_TORCH = """
__version__ = "fake"
float32 = "float32"


class _Flags:
    allow_tf32 = True


class backends:
    class cuda:
        matmul = _Flags()

    cudnn = _Flags()


class cuda:
    @staticmethod
    def is_available():
        return False


def no_grad():
    return lambda f: f


def tanh(x):
    import math

    return math.tanh(x)
"""

FAKE_TRANSFORMERS = """
import json
import os

__version__ = "fake"
# One id per piece; a piece is a special token or one character.
VOCAB = json.loads(os.environ["FAKE_VOCAB"])
IDS = {v: int(k) for k, v in VOCAB.items()}


class _Tokenizer:
    pad_token_id = 0
    bos_token_id = None

    def __call__(self, text, add_special_tokens=True):
        out, i = [], 0
        while i < len(text):
            longest = sorted(IDS, key=len, reverse=True)
            piece = next((p for p in longest if text.startswith(p, i)), None)
            if piece is None:
                piece = text[i]
                IDS.setdefault(piece, 1000 + ord(piece))
            out.append(IDS[piece])
            i += len(piece)
        return {"input_ids": out}

    def decode(self, ids):
        names = {v: k for k, v in IDS.items()}
        return "".join(VOCAB.get(str(i)) or names[i] for i in ids)


class AutoTokenizer:
    @staticmethod
    def from_pretrained(name, **kw):
        return _Tokenizer()


class _Model:
    class config:
        final_logit_softcapping = None

    def to(self, device):
        return self

    def eval(self):
        return self


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(name, **kw):
        return _Model()
"""


def _run_reference_script(tmp_path, vocab, turn_ids, turn_text):
    import os
    import sys

    from litetune.decision_scripts import REFERENCE_SCRIPT

    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
    (fake / "transformers.py").write_text(FAKE_TRANSFORMERS, encoding="utf-8")
    script = tmp_path / "reference.py"
    script.write_text(REFERENCE_SCRIPT, encoding="utf-8")
    report = tmp_path / "report.json"
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "model": "checkpoint",
                "rows": [],
                "options": ["joy", "anger"],
                "turn_ids": turn_ids,
                "turn_text": turn_text,
                "device": "cpu",
                "out": str(tmp_path / "out.jsonl"),
                "report": str(report),
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(fake), FAKE_VOCAB=json.dumps(vocab))
    proc = subprocess.run(
        [sys.executable, str(script), str(spec)], env=env, capture_output=True, text=True
    )
    return proc, json.loads(report.read_text()) if report.exists() else None


@pytest.mark.parametrize(
    ("vocab", "ids", "text", "first"),
    [
        ({"106": "<end_of_turn>", "107": "\n"}, [106, 107], "<end_of_turn>\n", "<end_of_turn>"),
        ({"151645": "<|im_end|>", "198": "\n"}, [151645, 198], "<|im_end|>\n", "<|im_end|>"),
    ],
    ids=["Gemma 3", "ChatML"],
)
def test_the_reference_script_reports_the_first_token_of_the_recorded_turn(
    tmp_path, vocab, ids, text, first
):
    proc, report = _run_reference_script(tmp_path, vocab, ids, text)

    assert proc.returncode == 0, proc.stderr
    assert report["terminator"] == first
    assert all(c[-1] == ids[0] for c in report["cont_ids"])


def test_the_reference_script_refuses_a_first_token_that_does_not_tokenise_back(tmp_path):
    # Id 5 decodes to text the tokenizer reads as id 6.
    proc, report = _run_reference_script(tmp_path, {"5": "a", "6": "a"}, [5], "a")

    assert proc.returncode != 0
    assert "not that one token" in proc.stderr
    # What a stopped run still tells the parent, and only that.
    assert report is not None
    assert set(report) == {"stopped_by", "device", "mps_environment", "mps_memory_sampled_max"}
    assert report["stopped_by"].startswith("SystemExit: the end of turn's first token 5")
    assert report["device"] == "cpu" and report["mps_memory_sampled_max"] is None


# -- what the session puts before the prompt ----------------------------------------


@pytest.mark.parametrize(
    ("ref_prefix", "session_prefix"),
    [([BOS], [BOS]), ([], [])],
    ids=["Gemma 3: the template's BOS, the session's BOS", "ChatML: no BOS on either side"],
)
def test_the_prompt_check_adds_what_the_session_puts_first(
    split, tmp_path, ref_prefix, session_prefix
):
    s = split()
    job = job_for(
        s,
        tmp_path,
        train_env=FakeScoringEnv("train", ref_prefix=ref_prefix),
        runtime_env=FakeScoringEnv("runtime", session_prefix=session_prefix),
    )

    status, fields = run(job, s["record"])

    assert status == "passed", fields["checks"]
    assert check(fields, PROMPT_CHECK)["outcome"] == "passed"
    assert fields["decision"]["session_prefix"] == session_prefix


@pytest.mark.parametrize(
    ("ref_prefix", "session_prefix"),
    [([BOS], []), ([], [7])],
    ids=[
        "the template prints a BOS the session does not add (MiniCPM5's shape)",
        "the session adds a BOS the template does not print (Granite's shape)",
    ],
)
def test_a_bos_on_one_side_only_is_refused(split, tmp_path, ref_prefix, session_prefix):
    s = split()
    job = job_for(
        s,
        tmp_path,
        train_env=FakeScoringEnv("train", ref_prefix=ref_prefix),
        runtime_env=FakeScoringEnv("runtime", session_prefix=session_prefix),
    )

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    detail = failed(fields, PROMPT_CHECK)["detail"]
    assert "is not the reference's" in detail and f"session prefix {session_prefix}" in detail


def test_a_runtime_that_does_not_say_what_its_session_puts_first_is_refused(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path, runtime_env=FakeScoringEnv("runtime", session_prefix=None))

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    assert "did not report the ids its session puts" in failed(fields, PROMPT_CHECK)["detail"]


@pytest.mark.parametrize(
    ("template", "bos"), [("gemma", "<bos>"), ("chatml", "none")], ids=["Gemma 3", "ChatML"]
)
def test_the_runtime_script_reports_its_session_prefix_and_scores_after_it(tmp_path, template, bos):
    prompts = ["text 1 about joy", "text 22 about fear"]
    options = ("joy", "anger", "fear")
    term = "<end_of_turn>" if template == "gemma" else "<|im_end|>"

    proc, rows, report = _run_engine_script(
        tmp_path,
        True,
        prompts,
        options,
        fake_env={"FAKE_TEMPLATE": template, "FAKE_BOS": bos},
        term=term,
    )

    assert proc.returncode == 0, proc.stderr
    fake = _fake_module(tmp_path)
    assert report["session_prefix"] == ([] if bos == "none" else fake._ids(bos))
    before, after = fake._TEMPLATES[template]
    assert rows[0]["prompt_ids"] == fake._ids(before + prompts[0] + after)
    assert [r["token_scores"] for r in rows] == _expected(
        tmp_path, prompts, options, template=template, bos=bos, term=term
    )


# -- the self-check's pair, from the bundle's own template ---------------------------


def test_on_gemma_3_the_self_check_pair_is_the_measured_one(tmp_path):
    proc, _, report = _run_engine_script(tmp_path, True, ["text 1 about joy"], OPTIONS)

    assert proc.returncode == 0, proc.stderr
    assert report["selfcheck"]["pair"] == [
        "<start_of_turn>user\nAnswer with one word.<end_of_turn>\n<start_of_turn>model\n",
        "<start_of_turn>user\nAnswer with one word.<end_of_turn>\n<start_of_turn>user\n",
    ]


@pytest.mark.parametrize(("drops", "mode"), [(True, "once"), (False, "plain")])
def test_on_chatml_the_self_check_renders_its_own_pair_and_picks_the_mode(tmp_path, drops, mode):
    prompts = ["text 1 about joy", "text 22 about fear"]
    options = ("joy", "anger", "fear")
    env = {"FAKE_TEMPLATE": "chatml", "FAKE_BOS": "none"}

    proc, rows, report = _run_engine_script(
        tmp_path, drops, prompts, options, fake_env=env, term="<|im_end|>"
    )

    assert proc.returncode == 0, proc.stderr
    a, b = report["selfcheck"]["pair"]
    assert a == "<|im_start|>user\nAnswer with one word.<|im_end|>\n<|im_start|>assistant\n"
    assert b == "<|im_start|>user\nAnswer with one word.<|im_end|>\n<|im_start|>user\n"
    assert report["selfcheck"]["comparisons"] > 0 and report["mode"] == mode
    assert [r["token_scores"] for r in rows] == _expected(
        tmp_path, prompts, options, template="chatml", bos="none", term="<|im_end|>"
    )


def test_a_rendering_with_no_pair_is_no_comparisons_and_a_harness_failure(tmp_path):
    # A template that renders the message as one token cannot differ at its second-to-last.
    one = FAKE_LITERT_LM.replace(
        "return before + prompt + after",
        'return "x" if prompt == "Answer with one word." else before + prompt + after',
    )
    assert one != FAKE_LITERT_LM

    proc, rows, report = _run_engine_script(
        tmp_path, True, ["text 1 about joy"], OPTIONS, fake_source=one
    )

    assert proc.returncode == 3
    assert report["selfcheck"]["verdict"] == "no comparisons"
    assert report["selfcheck"]["pair"] is None and rows == []


# -- a family the method was not measured on -----------------------------------------


@pytest.mark.parametrize("family", ["qwen-3", "functiongemma", None])
def test_another_family_without_a_float_twin_is_refused_before_anything_runs(
    split, tmp_path, family
):
    s = split()
    job = job_for(s, tmp_path, family=family)

    status, fields = run(job, s["record"])

    assert status == "failed_harness"
    detail = failed(fields, TWIN_CHECK)["detail"]
    assert "without a float twin on gemma-3-text only" in detail and "--float-twin" in detail
    assert job.train_env.specs == [] and job.runtime_env.specs == []


def test_another_family_with_a_float_twin_is_measured_and_says_it_is_unmeasured(split, tmp_path):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(s, tmp_path, family="gemma-4-e4b", float_twin=twin)

    status, fields = run(job, s["record"])

    assert status == "passed", fields["checks"]
    assert check(fields, TWIN_CHECK)["outcome"] == "passed"
    assert any(
        "have not been trained or measured on gemma-4-e4b" in t for t in fields["limitations"]
    )


def test_a_family_measured_on_one_task_says_which_models_and_which_task(split, tmp_path):
    s = split()
    twin = tmp_path / "twin.litertlm"
    twin.write_bytes(b"float")
    job = job_for(s, tmp_path, family="qwen-3", float_twin=twin)

    status, fields = run(job, s["record"])

    assert status == "passed", fields["checks"]
    said = [t for t in fields["limitations"] if "on qwen-3 were trained and verified" in t]
    assert len(said) == 1
    assert "Qwen/Qwen3-0.6B and Qwen/Qwen3-1.7B" in said[0] and "six options" in said[0]
    assert not any("have not been trained or measured on" in t for t in fields["limitations"])


def test_the_measured_family_says_nothing_about_being_unmeasured(split, tmp_path):
    s = split()

    _, fields = run(job_for(s, tmp_path), s["record"])

    assert not any("have not been trained or measured on" in t for t in fields["limitations"])


# -- the runtime script on fresh engines ---------------------------------------------


@pytest.mark.parametrize(("drops", "mode"), [(True, "once"), (False, "plain")])
def test_fresh_engine_loads_an_engine_per_option_and_never_rewinds(tmp_path, drops, mode):
    prompts = ["text 1 about joy", "text 22 about fear"]
    options = ("joy", "anger", "fear")
    loads = tmp_path / "loads.txt"
    # A fake whose rewind fails the run: fresh_engine must never call it.
    no_rewind = FAKE_LITERT_LM.replace(
        "        if not ptr.saved:\n            return -1",
        "        raise AssertionError('rewound')",
    )
    assert no_rewind != FAKE_LITERT_LM

    proc, rows, report = _run_engine_script(
        tmp_path,
        drops,
        prompts,
        options,
        fake_source=no_rewind,
        fake_env={"FAKE_LOADS": str(loads)},
        method="fresh_engine",
    )

    assert proc.returncode == 0, proc.stderr
    assert report["method"] == "fresh_engine" and report["mode"] == mode
    comparisons = report["selfcheck"]["comparisons"]
    # One to render and tokenise, two per self-check comparison, one per option per row.
    want = 1 + 2 * comparisons + len(options) * len(prompts)
    assert report["engine_loads"] == want == len(loads.read_text().splitlines())
    assert [r["token_scores"] for r in rows] == _expected(tmp_path, prompts, options)


def test_rewind_loads_one_engine(tmp_path):
    loads = tmp_path / "loads.txt"

    proc, _, report = _run_engine_script(
        tmp_path, True, ["text 1 about joy"], OPTIONS, fake_env={"FAKE_LOADS": str(loads)}
    )

    assert proc.returncode == 0, proc.stderr
    assert report["engine_loads"] == 1 == len(loads.read_text().splitlines())


def test_a_rendering_that_starts_with_the_bos_string_is_read_as_prefill_reads_it(tmp_path):
    """The runtime's tokenizer reads the BOS string as text, and prefill as the
    BOS: the ids returned are prefill's, so the prompt check sees the second BOS
    the session would read."""
    proc, rows, report = _run_engine_script(
        tmp_path, True, ["text 1 about joy"], OPTIONS, fake_env={"FAKE_TEMPLATE": "gemma_bos"}
    )

    assert proc.returncode == 0, proc.stderr
    bos = report["session_prefix"]
    assert len(bos) == 1 and rows[0]["prompt_ids"][:1] == bos
    assert rows[0]["prompt_ids"] == bos + _fake_module(tmp_path)._ids(
        "<start_of_turn>user\ntext 1 about joy<end_of_turn>\n<start_of_turn>model\n"
    )


def test_the_self_check_pair_skips_ids_that_change_nothing_or_do_not_round_trip(tmp_path):
    proc, _, report = _run_engine_script(
        tmp_path, True, ["text 1 about joy"], OPTIONS, fake_env={"FAKE_TEMPLATE": "tricky"}
    )

    assert proc.returncode == 0, proc.stderr
    # Not ":" (the id already there), not "a" ("ab" is one word): the newline.
    head = "<start_of_turn>:a\nAnswer with one word.<end_of_turn>\n<start_of_turn>"
    assert report["selfcheck"]["pair"] == [head + ":b", head + "\nb"]


_HEAD_DRIVER = """
import json, runpy, sys, types
ns = runpy.run_path(sys.argv[1], run_name="reference_under_test")
def cfg(**kw):
    text = kw.pop("text", None)
    c = types.SimpleNamespace(**kw)
    if text is not None:
        c.get_text_config = lambda: types.SimpleNamespace(**text)
    return c
cases = {
    "granite": cfg(logits_scaling=10.0),
    "gemma3": cfg(final_logit_softcapping=None),
    "gemma2": cfg(final_logit_softcapping=30.0),
    "gemma4": cfg(text={"final_logit_softcapping": 30.0}),
    "plain": cfg(),
}
out = {k: ns["head_transform"](v) for k, v in cases.items()}
out["granite_value"] = ns["transformed"](25.0, 10.0, None)
out["capped_value"] = ns["transformed"](60.0, None, 30.0)
print(json.dumps(out))
"""


def test_the_reference_scores_through_each_familys_own_logit_transform(tmp_path):
    """lm_head's output is what the scoring reads, and the bundle's head changes
    it: Granite divides by logits_scaling, a top-level final_logit_softcapping
    soft-caps. Gemma 4's text-config cap is in its forward but not in its bundle,
    so the reference leaves it out; with it a Gemma 4 reference disagrees with its
    own float twin, as without the division a Granite one does."""
    import math
    import os
    import sys

    from litetune.decision_scripts import REFERENCE_SCRIPT

    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "torch.py").write_text(FAKE_TORCH, encoding="utf-8")
    (fake / "transformers.py").write_text(FAKE_TRANSFORMERS, encoding="utf-8")
    script = tmp_path / "reference.py"
    script.write_text(REFERENCE_SCRIPT, encoding="utf-8")
    driver = tmp_path / "driver.py"
    driver.write_text(_HEAD_DRIVER, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(fake), FAKE_VOCAB="{}")
    proc = subprocess.run(
        [sys.executable, str(driver), str(script)], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["granite"] == [10.0, None]
    assert out["gemma3"] == [None, None]
    assert out["gemma2"] == [None, 30.0]
    assert out["gemma4"] == [None, None]
    assert out["plain"] == [None, None]
    assert out["granite_value"] == 2.5
    assert out["capped_value"] == math.tanh(2.0) * 30.0


def test_every_head_the_reference_scores_through_goes_through_the_transform():
    """The scoring reads lm_head's output in one place, `head_logp`; the other paths
    read the model's own forward, and a variant trial against `head_logp` decides
    whether they are used. So the one place must apply the transform, and must
    take it from the model's config."""
    from litetune.decision_scripts import REFERENCE_SCRIPT

    assert REFERENCE_SCRIPT.count("model.lm_head(") == 1
    assert "transformed(model.lm_head(hidden).float(), scale, cap)" in REFERENCE_SCRIPT
    assert "scale, cap = head_transform(model.config)" in REFERENCE_SCRIPT


def test_the_twin_rows_are_scored_alone_by_the_reference(split, tmp_path):
    """The float twin is held to 1e-3 nats against the reference; on its rows the
    reference scores each row alone with no cache, so batching is not in it."""
    s = split()
    job = job_for(s, tmp_path)

    status, _ = run(job, s["record"])

    assert status == "passed"
    spec = job.train_env.specs[0]
    assert spec["alone_for"] == spec["ids_for"][:PROBE_ROWS]
    assert len(spec["alone_for"]) == PROBE_ROWS


# -- the reference on a real model, rows in batches ----------------------------------
#
# These run the reference script in the training environment, on a small random
# Llama and a word-level tokenizer built here; without that environment they are
# skipped. Every row's prompt has a length of its own, so every batch is padded.
# CI runs them in a job of their own (`LITETUNE_TEST_TRAIN_PYTHON`).

_TINY_MODEL = r"""
import json, sys
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

out = sys.argv[1]
words = ["<pad>", "<unk>", "<s>", "<eot>", "[U]", "[A]", "joy", "anger", "fear", "surprise"]
words += [f"w{i}" for i in range(60)]
inner = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
inner.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tok = PreTrainedTokenizerFast(
    tokenizer_object=inner, pad_token="<pad>", unk_token="<unk>", bos_token="<s>",
    additional_special_tokens=["<eot>"],
)
tok.chat_template = (
    "<s>{% for m in messages %}{% if m['role'] == 'user' %}[U] {{ m['content'] }} <eot> "
    "{% else %}[A] {{ m['content'] }} <eot> {% endif %}{% endfor %}"
    "{% if add_generation_prompt %}[A] {% endif %}"
)
tok.save_pretrained(out)
torch.manual_seed(0)
config = LlamaConfig(
    vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    pad_token_id=0, bos_token_id=2, eos_token_id=3, initializer_range=0.3,
)
LlamaForCausalLM(config).save_pretrained(out)
print(json.dumps({"eot": tok.convert_tokens_to_ids("<eot>")}))
"""


def _real_train_python():
    # Read at import, before conftest points LITETUNE_ENV_DIR at a temporary
    # directory for every test. CI names an interpreter with torch and
    # transformers instead, and there a missing one is an error, not a skip.
    from litetune import envs

    named = os.environ.get("LITETUNE_TEST_TRAIN_PYTHON")
    if named:
        return named
    return str(envs.TRAIN.python) if envs.TRAIN.ready else None


_TRAIN_PYTHON = _real_train_python()


def _in_train_env(args, child_env=None):
    if _TRAIN_PYTHON is None:
        pytest.skip("the training environment is not provisioned")
    env = {**os.environ, **child_env} if child_env else None
    return subprocess.run(
        [_TRAIN_PYTHON, *args], capture_output=True, text=True, timeout=600, env=env
    )


def _words(i):
    # Longest first, so that only sorting puts rows of a length together: up
    # to row 7 three words fewer per row, then one fewer: sorted into batches
    # of eight, rows 0 to 7 are padded by up to 21 tokens, the others by 7.
    k = 23 - i
    return k + 1 if k < 16 else 17 + 3 * (k - 16)


def _shortest_alike(i):
    # Real data's shape: the eight shortest rows of one length, not padded at
    # all, the rest as above.
    return 2 if i >= 16 else _words(i)


def _padded_middle(i):
    # The longest rows, 0 to 7, all of one length; the most padded batch is rows
    # 8 to 15 (10 to 31 words); the shortest, 16 to 23, are 1 to 8 words.
    return 40 if i < 8 else 10 + 3 * (i - 8) if i < 16 else i - 15


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """One small random Llama for the module, and its one-at-a-time scores."""
    work = tmp_path_factory.mktemp("tiny")
    make = work / "make.py"
    make.write_text(_TINY_MODEL, encoding="utf-8")
    made = _in_train_env([str(make), str(work / "model")])
    assert made.returncode == 0, made.stderr
    return {"work": work, "alone": {}}


def _run(tiny, script, row_batch, words=_words, extra=None, ok=True, child_env=None):
    work = tiny["work"]
    tag = f"b{row_batch}-{abs(hash((script, words.__name__, json.dumps(extra)))) % 10**10}"
    spec = work / f"spec-{tag}.json"
    out, report = work / f"out-{tag}.jsonl", work / f"report-{tag}.json"
    spec.write_text(
        json.dumps(
            {
                "model": str(work / "model"),
                "rows": [
                    {"index": i, "prompt": " ".join(f"w{j}" for j in range(words(i)))}
                    for i in range(24)
                ],
                "options": ["joy", "anger", "fear surprise"],
                "turn_ids": [3],
                "turn_text": "<eot>",
                "device": "cpu",
                "row_batch": row_batch,
                **(extra or {}),
                "out": str(out),
                "report": str(report),
            }
        ),
        encoding="utf-8",
    )
    path = work / f"reference-{tag}.py"
    path.write_text(script, encoding="utf-8")
    proc = _in_train_env([str(path), str(spec)], child_env)
    if not ok:
        return proc
    assert proc.returncode == 0, proc.stderr
    rows = {r["index"]: r for r in map(json.loads, out.read_text().splitlines())}
    return rows, json.loads(report.read_text())


def _alone(tiny, words=_words, extra=None):
    """The unchanged script, one row at a time: what batching must reproduce."""
    from litetune.decision_scripts import REFERENCE_SCRIPT

    key = (words.__name__, json.dumps(extra))
    if key not in tiny["alone"]:
        tiny["alone"][key] = _run(tiny, REFERENCE_SCRIPT, 1, words, extra)
    return tiny["alone"][key]


def _largest_difference(a, b):
    assert a.keys() == b.keys()
    return max(
        abs(x - y)
        for i in a
        for f, g in zip(a[i]["token_scores"], b[i]["token_scores"], strict=True)
        for x, y in zip(f, g, strict=True)
    )


def _replaced(old, new):
    from litetune.decision_scripts import REFERENCE_SCRIPT

    assert REFERENCE_SCRIPT.count(old) == 1, old
    return REFERENCE_SCRIPT.replace(old, new)


_VARIANTS = 'VARIANTS = ("cache_position+position_ids", "position_ids", "plain")'
_SCORE_BATCH = "    def score_batch(prompts_ids, how):\n"
_SCORE_BATCH_END = "            del lp\n        return res\n"
_GOOD_POSITIONS = "position_ids=(mask.cumsum(-1) - 1).clamp(min=0)"
# Positions counted from the first column: wrong on every padded row.
_DRIFTING = _GOOD_POSITIONS.replace(
    "(mask.cumsum(-1) - 1).clamp(min=0)", "torch.arange(width, device=device).expand(b, -1)"
)


def test_rows_scored_in_batches_match_rows_scored_one_at_a_time(tiny):
    from litetune.decision_scripts import REFERENCE_SCRIPT

    extra = {"ids_for": [5, 20], "check_indices": [10]}
    alone, alone_report = _alone(tiny, extra=extra)
    batched, report = _run(tiny, REFERENCE_SCRIPT, 8, extra=extra)

    assert alone_report["row_batch"] == 1 and report["row_batch"] == 8
    assert report["batches"] == 3
    assert report["batch_tokens_held"] > 100
    assert report["batch_trial"] is None
    assert report["most_padding_checked"] == 21
    assert _largest_difference(alone, batched) < 1e-5
    # The first batch is the most padded, rows 0 to 7, checked row by row; it
    # holds the first three rows and the longest, row 0. Then row 10, asked for.
    checked = sorted(i for i, r in batched.items() if "check_max_abs_token" in r)
    assert checked == [*range(8), 10]
    # The prompt ids of the rows asked for, and of no other.
    assert sorted(i for i, r in batched.items() if "prompt_ids" in r) == [5, 20]
    for i in (5, 20):
        assert batched[i]["prompt_ids"] == alone[i]["prompt_ids"]
        assert len(batched[i]["prompt_ids"]) == batched[i]["prompt_len"] == _words(i) + 4
    # Every option scored, each token of it: "fear surprise" is two words.
    assert [len(f) for f in batched[0]["token_scores"]] == [2, 2, 3]


def test_rows_scored_on_mps_match_rows_scored_on_the_cpu(tiny):
    from litetune.decision_scripts import REFERENCE_SCRIPT

    has_mps = _in_train_env(["-c", "import torch; print(torch.backends.mps.is_available())"])
    if has_mps.stdout.strip() != "True":
        pytest.skip("the training environment's torch has no MPS")
    cpu, _ = _alone(tiny)
    on_mps, report = _run(
        tiny, REFERENCE_SCRIPT, 8, extra={"device": "mps"}, child_env={devices.LOW_WATERMARK: "0.5"}
    )

    assert report["device"] == "mps" and report["row_batch"] == 8
    assert _largest_difference(cpu, on_mps) < 1e-4
    sampled = report["mps_memory_sampled_max"]
    # The model loaded, then per batch its prompt forward, each chunk and its end.
    assert report["batches"] == 3
    assert sampled["samples"] > 1 + 3 * report["batches"]
    assert sampled["sampled_at"] == [
        "after loading the model",
        "after each batch's prompt forward",
        "at each chunk's log-probabilities, its cache copy alive",
        "after each batch",
    ]
    assert sampled["current_allocated_bytes"] > 0
    assert set(report["mps_environment"]) == set(devices.MPS_VARIABLES)
    assert report["mps_environment"][devices.LOW_WATERMARK] == "0.5"


def test_rows_scored_alone_on_the_cpu_sample_no_mps_memory(tiny):
    _, report = _alone(tiny)

    assert report["device"] == "cpu"
    assert report["mps_memory_sampled_max"] is None


def test_the_rows_asked_for_alone_are_recorded_as_scored_with_no_cache(tiny):
    from litetune.decision_scripts import REFERENCE_SCRIPT

    # With no way of passing the cache every row is scored alone, uncached.
    uncached, _ = _run(tiny, _replaced(_VARIANTS, "VARIANTS = ()"), 8)
    batched, report = _run(tiny, REFERENCE_SCRIPT, 8, extra={"alone_for": [20]})

    assert report["row_batch"] == 8
    assert batched[20]["token_scores"] == uncached[20]["token_scores"]
    assert "check_max_abs_token" in batched[20]
    assert batched[21]["token_scores"] != uncached[21]["token_scores"]


def test_a_way_of_passing_the_cache_without_positions_never_batches(tiny):
    rows, report = _run(tiny, _replaced(_VARIANTS, 'VARIANTS = ("plain",)'), 8)

    assert report["cache_variant"] == "plain"
    assert report["row_batch"] == 1 and report["batches"] == 24
    assert report["batch_trial"] is None and report["most_padding_checked"] is None
    assert report["batch_tokens_held"] is None
    assert _largest_difference(_alone(tiny)[0], rows) < 1e-5


def test_a_first_batch_that_drifts_is_scored_again_one_row_at_a_time(tiny):
    # The eight shortest rows are of one length, as in real data: a trial on
    # them would see no padding and pass, and the drift would stop the run.
    drifting = _replaced(_GOOD_POSITIONS, _DRIFTING)
    rows, report = _run(tiny, drifting, 8, words=_shortest_alike)

    assert report["row_batch"] == 1 and report["batches"] == 24
    assert report["batch_trial"]["row_batch"] == 8
    assert report["batch_trial"]["max_abs_token"] > 3e-3
    assert report["most_padding_checked"] is None
    assert _largest_difference(_alone(tiny, words=_shortest_alike)[0], rows) < 1e-5


def test_a_first_batch_that_raises_is_scored_again_one_row_at_a_time(tiny):
    raising = _replaced(_SCORE_BATCH, _SCORE_BATCH + '        raise RuntimeError("no room")\n')
    rows, report = _run(tiny, raising, 8)

    assert report["row_batch"] == 1
    assert report["batch_trial"] == {"row_batch": 8, "error": "RuntimeError: no room"}
    assert _largest_difference(_alone(tiny)[0], rows) < 1e-5


def test_a_first_batch_with_a_nan_is_scored_again_one_row_at_a_time(tiny):
    # NaN on the options' later tokens only, after finite first ones.
    later = "lp = torch.log_softmax(model(**kw).logits.float(), dim=-1)"
    rows, report = _run(tiny, _replaced(later, later + ' * float("nan")'), 8)

    assert report["row_batch"] == 1
    # As a string: JSON has no NaN, and the manifest is read by strict parsers.
    assert report["batch_trial"] == {"row_batch": 8, "max_abs_token": "nan"}
    assert _largest_difference(_alone(tiny)[0], rows) < 1e-5


_ON_THE_SECOND_BATCH = '        globals()["calls"] = globals().get("calls", 0) + 1\n'


def test_a_batch_after_the_first_that_raises_stops_the_run(tiny):
    second = _ON_THE_SECOND_BATCH + (
        '        if globals()["calls"] == 2:\n            raise RuntimeError("second batch")\n'
    )
    proc = _run(tiny, _replaced(_SCORE_BATCH, _SCORE_BATCH + second), 8, ok=False)

    assert proc.returncode != 0
    assert "RuntimeError: second batch" in proc.stderr


def test_a_checked_row_that_drifts_after_the_first_batch_stops_the_run(tiny):
    # The second batch scored is rows 16 to 23; row 20 is checked.
    wrong = _ON_THE_SECOND_BATCH + (
        '        if globals()["calls"] == 2:\n'
        "            res = [[[v + 1.0 for v in f] for f in r] for r in res]\n"
    )
    script = _replaced(_SCORE_BATCH_END, "            del lp\n" + wrong + "        return res\n")
    proc = _run(tiny, script, 8, extra={"check_indices": [20]}, ok=False)

    assert proc.returncode != 0
    assert "row 20: scored in a batch and alone, the scores differ by" in proc.stderr


def test_a_token_budget_makes_smaller_batches_with_the_same_scores(tiny):
    from litetune.decision_scripts import REFERENCE_SCRIPT

    rows, report = _run(tiny, REFERENCE_SCRIPT, 8, extra={"batch_tokens": 100})

    # Prompts of 5 to 42 tokens: eight fit 100 tokens only among the shortest.
    assert report["row_batch"] == 8 and report["batches"] > 3
    assert 0 < report["batch_tokens_held"] <= 100
    assert report["batch_trial"] is None
    assert _largest_difference(_alone(tiny)[0], rows) < 1e-5


def test_the_first_batch_is_the_most_padded_and_the_longest_row_is_checked(tiny):
    from litetune.decision_scripts import REFERENCE_SCRIPT

    rows, report = _run(tiny, REFERENCE_SCRIPT, 8, words=_padded_middle)

    assert report["row_batch"] == 8 and report["batches"] == 3
    assert report["most_padding_checked"] == 21
    # The first batch, rows 8 to 15, row by row; the first three rows; and the
    # longest by tokens, row 7, the last of rows of one length once sorted.
    checked = sorted(i for i, r in rows.items() if "check_max_abs_token" in r)
    assert checked == [0, 1, 2, 7, *range(8, 16)]
    assert _largest_difference(_alone(tiny, words=_padded_middle)[0], rows) < 1e-5


def test_the_engine_script_asks_for_the_gpu_as_the_text_path_does_and_reads_the_kernel():
    """The option-scoring script's GPU branch: the backend the spec names, fp32
    activations through the shared `engine_kwargs`, and the kernel's two readings
    around the rows with the engine open -- what `gpu_observed` takes as GPU work."""
    from litetune.decision_scripts import ENGINE_SCRIPT

    assert 'on_gpu = spec.get("backend", "cpu") == "gpu"' in ENGINE_SCRIPT
    assert 'engine_kwargs({"backend": "gpu", "activation_data_type": "fp32"})' in ENGINE_SCRIPT
    assert "def engine_kwargs" in ENGINE_SCRIPT and "def device_report" in ENGINE_SCRIPT
    at_engine = ENGINE_SCRIPT.index('device["at_engine"] = device_report()')
    rows = ENGINE_SCRIPT.index('for row in spec["rows"]:')
    after = ENGINE_SCRIPT.index('device["after_generation"] = device_report()')
    assert at_engine < rows < after
    # Both readings only on the GPU.
    assert ENGINE_SCRIPT.count("if on_gpu:") >= 3
    compile(ENGINE_SCRIPT, "engine.py", "exec")


def test_the_engine_script_on_the_gpu_opens_its_engine_so_and_reads_the_kernel(tmp_path):
    """The GPU branch run, not read: the engine gets the backend `engine_kwargs`
    builds and no weight cache, and both kernel readings and the bundle's own
    activation key reach the report."""
    kwargs_file = tmp_path / "engine-kwargs.jsonl"
    recording = FAKE_LITERT_LM.replace(
        "    def __init__(self, path, backend=None, cache_dir=None):\n        self.path = path\n",
        "    def __init__(self, path, backend=None, cache_dir=None):\n        self.path = path\n"
        "        with open(os.environ['FAKE_ENGINE_KW'], 'a') as fh:\n"
        "            seen = {'backend': str(backend), 'cache_dir': cache_dir}\n"
        "            fh.write(json.dumps(seen) + '\\n')\n",
    )
    assert recording != FAKE_LITERT_LM
    if "import json" not in recording:
        recording = "import json\n" + recording
    stubs = (
        "def engine_kwargs(spec):\n"
        "    assert spec == {'backend': 'gpu', 'activation_data_type': 'fp32'}, spec\n"
        "    return {'backend': 'GPU-STUB'}\n"
        "READINGS = iter(['at engine', 'after'])\n"
        "def device_report():\n"
        "    return {'reading': next(READINGS)}\n"
    )
    proc, rows, report = _run_engine_script(
        tmp_path,
        True,
        ["text 1 about joy"],
        OPTIONS,
        fake_source=recording,
        fake_env={"FAKE_ENGINE_KW": str(kwargs_file)},
        backend="gpu",
        script_tail=stubs,
    )

    assert proc.returncode == 0, proc.stderr
    assert report["backend"] == "gpu"
    assert report["device"] == {
        "at_engine": {"reading": "at engine"},
        "after_generation": {"reading": "after"},
    }
    # No builder in the stand-in's environment: said, not passed off as "none".
    assert report["bundle_activation"] is None and report["bundle_activation_error"]
    loads = [json.loads(line) for line in kwargs_file.read_text().splitlines()]
    assert loads and all(x == {"backend": "GPU-STUB", "cache_dir": ":nocache"} for x in loads)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# The reference on Apple's GPU


def mps_probe(**fields: Any) -> envs.DeviceProbe:
    values: dict[str, Any] = {
        "detail": "this environment's torch reports mps",
        "source": "probe",
        "mps_recommended_max_memory": 16 * 2**30,
        "os": "Darwin",
        "os_version": "15.0",
        "os_version_source": "platform.mac_ver()",
        "machine": "arm64",
    }
    values.update(fields)
    return envs.DeviceProbe(device="mps", **values)


@pytest.fixture
def probed(monkeypatch):
    """`envs.resolve_device` answering `probe`, and the environments it was asked about."""
    asked: list[str] = []

    def use(probe: envs.DeviceProbe) -> list[str]:
        def resolve(env, events=None):
            asked.append(env.name)
            return probe

        monkeypatch.setattr(envs, "resolve_device", resolve)
        return asked

    return use


def reference_spec(job: DecisionVerify) -> dict[str, Any]:
    assert isinstance(job.train_env, FakeScoringEnv)
    (spec,) = job.train_env.specs
    return spec


def test_a_reference_on_a_mac_with_mps_runs_there_under_the_memory_policy(split, tmp_path, probed):
    s = split()
    asked = probed(mps_probe())
    train = FakeScoringEnv("train", report_extra={"device": "mps"})
    job = job_for(s, tmp_path, train_env=train)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    assert asked == ["train"]
    spec = reference_spec(job)
    assert spec["device"] == "mps"
    assert spec["row_batch"] == MPS_ROW_BATCH == 64
    policy = devices.prepare_mps(mps_probe(), os.environ)
    assert train.child_envs == [policy.child_env]
    assert fields["harness"]["reference_mps_memory"] == policy.as_dict()
    assert REFERENCE_ON_MPS in fields["limitations"]
    # It saw the policy it was given, so no mismatch is reported.
    assert not [m for m in fields["limitations"] if "reference script" in m]
    # The runtime is not on the reference's device or policy.
    assert job.runtime_env.child_envs and all(e is None for e in job.runtime_env.child_envs)


def test_the_environment_is_provisioned_before_its_torch_is_asked(split, tmp_path, monkeypatch):
    s = split()
    train = FakeScoringEnv("train")
    order: list[str] = []
    original = train.provision

    def provision(events=None, force=False):
        order.append("provision")
        return original(events=events, force=force)

    def resolve(env, events=None):
        order.append("probe")
        return envs.DeviceProbe(device="cpu", detail="cpu")

    monkeypatch.setattr(train, "provision", provision)
    monkeypatch.setattr(envs, "resolve_device", resolve)

    run(job_for(s, tmp_path, train_env=train), s["record"])

    assert order == ["provision", "probe"]
    assert train.provisioned == 1


def test_an_environment_that_cannot_be_provisioned_refuses_before_any_probe(
    split, tmp_path, monkeypatch, probed
):
    s = split()
    asked = probed(mps_probe())
    train = FakeScoringEnv("train")

    def broken(events=None, force=False):
        raise RuntimeError("no torch wheel")

    monkeypatch.setattr(train, "provision", broken)

    status, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    assert status == "failed_harness", fields
    detail = check(fields, REFERENCE_CHECK)["detail"]
    assert "train environment could not be provisioned: no torch wheel" in detail
    assert asked == [] and train.specs == []


def test_a_mac_whose_memory_policy_is_refused_runs_the_reference_on_the_cpu(
    split, tmp_path, probed
):
    s = split()
    probed(mps_probe(mps_recommended_max_memory=None))
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    spec = reference_spec(job)
    assert spec["device"] == "cpu" and "row_batch" not in spec
    assert job.train_env.child_envs == [None]
    moved = [m for m in fields["limitations"] if m.startswith("the float reference was not run")]
    assert len(moved) == 1 and moved[0].startswith("the float reference was not run on MPS: ")
    assert "reference_mps_memory" not in fields["harness"]
    assert REFERENCE_ON_MPS not in fields["limitations"]


def test_the_memory_policys_own_limitations_reach_the_report(split, tmp_path, probed, monkeypatch):
    s = split()
    probed(mps_probe())
    # The operator's own: unsupported operations may then run on the CPU.
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    train = FakeScoringEnv("train", report_extra={"device": "mps"})

    status, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    assert status == "passed", fields
    policy = devices.prepare_mps(mps_probe(), os.environ)
    assert policy.limitations
    for text in policy.limitations:
        assert text in fields["limitations"]


def test_the_probe_is_recorded(split, tmp_path, probed):
    s = split()
    probe = mps_probe()
    probed(probe)
    train = FakeScoringEnv("train", report_extra={"device": "mps"})

    _, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    assert fields["harness"]["reference_device_probe"] == {**probe.as_dict(), "attempted": True}


def test_litetune_device_cpu_keeps_the_reference_off_mps_and_says_so(split, tmp_path, probed):
    s = split()
    probed(mps_probe())
    # `run_verify` reads `LITETUNE_DEVICE` into the job.
    job = job_for(s, tmp_path, device="cpu")

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    assert reference_spec(job)["device"] == "cpu"
    assert job.train_env.child_envs == [None]
    assert "reference_mps_memory" not in fields["harness"]
    said = [m for m in fields["limitations"] if m.startswith(f"{devices.DEVICE_VARIABLE}=cpu")]
    assert len(said) == 1 and "the probe found: this environment's torch reports mps" in said[0]


@pytest.mark.parametrize(
    ("found", "given"), [("cuda", "auto"), ("cpu", "cpu")], ids=["cuda", "cpu"]
)
def test_off_a_mac_the_reference_runs_where_the_probe_found(split, tmp_path, probed, found, given):
    s = split()
    probed(envs.DeviceProbe(device=found, detail=f"torch reports {found}", source="probe"))
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    spec = reference_spec(job)
    assert spec["device"] == given and "row_batch" not in spec
    assert job.train_env.child_envs == [None]
    assert "reference_mps_memory" not in fields["harness"]
    assert not [m for m in fields["limitations"] if "torch reports" in m]


def test_a_mac_whose_torch_has_no_mps_says_why_the_reference_is_on_the_cpu(split, tmp_path, probed):
    s = split()
    detail = "this environment's torch reports cpu: mps is_built True, is_available False"
    probed(envs.DeviceProbe(device="cpu", detail=detail, source="probe", os="Darwin"))
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    assert reference_spec(job)["device"] == "cpu"
    assert detail in fields["limitations"]


def test_a_probe_that_could_not_answer_leaves_the_device_to_the_script_and_says_so(
    split, tmp_path, probed
):
    s = split()
    probed(envs.DeviceProbe(device=None, detail="the device probe could not answer: boom"))
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    assert reference_spec(job)["device"] == "auto"
    said = [m for m in fields["limitations"] if m.startswith("the device probe could not answer")]
    assert len(said) == 1 and "chose its device itself" in said[0]


def test_an_environment_that_is_not_provisioned_is_not_asked(split, tmp_path, probed):
    s = split()
    asked = probed(mps_probe())
    train = FakeScoringEnv("train", ready=False)
    job = job_for(s, tmp_path, train_env=train, auto_provision=False)

    status, fields = run(job, s["record"])

    assert asked == [] and train.provisioned == 0
    assert reference_spec(job)["device"] == "auto"
    assert fields["harness"]["reference_device_probe"]["device"] is None
    assert fields["harness"]["reference_device_probe"]["attempted"] is False
    assert any(m.startswith("no device probe was attempted") for m in fields["limitations"])


def test_a_reference_that_did_not_see_the_mps_policy_says_so(split, tmp_path, probed):
    s = split()
    probed(mps_probe())
    seen = {name: None for name in devices.MPS_VARIABLES}
    train = FakeScoringEnv("train", report_extra={"device": "mps", "mps_environment": seen})

    status, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    policy = devices.prepare_mps(mps_probe(), os.environ)
    expected = devices.environment_mismatch(policy, seen, "reference script")
    assert expected is not None
    assert expected in fields["limitations"]


MPS_OOM = (
    "torch.OutOfMemoryError: MPS backend out of memory (MPS allocated: 9.10 GiB, other "
    "allocations: 1.2 GiB, max allowed: 10.0 GiB). Tried to allocate 1.5 GiB on private pool."
)
MPS_NO_OP = (
    "NotImplementedError: The operator 'aten::foo' is not currently implemented for the MPS "
    "device."
)


@pytest.mark.parametrize(
    ("said", "ending"), [(MPS_OOM, "out of memory"), (MPS_NO_OP, "a device fact")]
)
def test_a_reference_that_stops_on_mps_is_scored_again_on_the_cpu(
    split, tmp_path, probed, said, ending
):
    s = split()
    probed(mps_probe())
    train = FakeScoringEnv("train", stops_on={"mps": (1, said)})
    job = job_for(s, tmp_path, train_env=train)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    assert [spec["device"] for spec in train.specs] == ["mps", "cpu"]
    assert "row_batch" not in train.specs[1]
    policy = devices.prepare_mps(mps_probe(), os.environ)
    assert train.child_envs == [policy.child_env, None]
    assert fields["harness"]["reference"]["device"] == "cpu"
    first = fields["harness"]["reference_on_mps"]
    assert said in first["failure"] and first["report"]["device"] == "mps"
    again = [m for m in fields["limitations"] if m.startswith("the float reference stopped")]
    assert len(again) == 1
    assert again[0].startswith(f"the float reference stopped on MPS ({ending}) and was scored")
    if ending == "out of memory":
        assert devices.mps_oom_advice(policy) in again[0]
    else:
        assert "ran out of memory" not in again[0]
    assert REFERENCE_ON_MPS not in fields["limitations"]


def test_a_reference_that_fails_on_mps_for_another_reason_is_refused_without_memory_advice(
    split, tmp_path, probed
):
    s = split()
    probed(mps_probe())
    said = "OSError: checkpoint not found"
    train = FakeScoringEnv("train", stops_on={"mps": (1, said)})

    status, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    assert status == "failed_harness", fields
    assert [spec["device"] for spec in train.specs] == ["mps"]
    detail = check(fields, REFERENCE_CHECK)["detail"]
    assert said in detail and "out of memory" not in detail
    assert "reference_on_mps" not in fields["harness"]


def test_the_operators_own_mps_settings_reach_the_reference(split, tmp_path, probed, monkeypatch):
    s = split()
    probed(mps_probe())
    monkeypatch.setenv(devices.HIGH_WATERMARK, "0.0")
    monkeypatch.setenv(devices.LOW_WATERMARK, "0.0")
    train = FakeScoringEnv("train", report_extra={"device": "mps"})

    status, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    assert status == "passed", fields
    variables = fields["harness"]["reference_mps_memory"]["variables"]
    assert variables[devices.HIGH_WATERMARK] == {"value": "0.0", "set_by": "user"}


def test_a_low_watermark_above_the_high_one_puts_the_reference_on_the_cpu(
    split, tmp_path, probed, monkeypatch
):
    s = split()
    probed(mps_probe())
    monkeypatch.setenv(devices.HIGH_WATERMARK, "0.5")
    monkeypatch.setenv(devices.LOW_WATERMARK, "0.9")
    job = job_for(s, tmp_path)

    status, fields = run(job, s["record"])

    assert status == "passed", fields
    assert reference_spec(job)["device"] == "cpu"
    assert any(
        m.startswith("the float reference was not run on MPS: ") for m in fields["limitations"]
    )


def test_a_reference_asked_for_mps_that_reports_the_cpu_is_not_called_mps(split, tmp_path, probed):
    s = split()
    probed(mps_probe())

    status, fields = run(job_for(s, tmp_path), s["record"])

    assert status == "passed", fields
    assert fields["harness"]["reference"]["device"] == "cpu"
    assert REFERENCE_ON_MPS not in fields["limitations"]


def test_a_broken_memory_policy_is_not_taken_for_a_refused_one(
    split, tmp_path, probed, monkeypatch
):
    s = split()
    probed(mps_probe())

    def broken(probe, environ, sysctl=None):
        raise ValueError("a bug")

    monkeypatch.setattr(devices, "prepare_mps", broken)

    with pytest.raises(ValueError, match="a bug"):
        run(job_for(s, tmp_path), s["record"])


def test_a_reference_that_fails_off_a_mac_gets_no_mps_advice(split, tmp_path):
    s = split()
    train = FakeScoringEnv("train", stops_on={"cpu": (1, "OSError: checkpoint not found")})

    status, fields = run(job_for(s, tmp_path, train_env=train), s["record"])

    assert status == "failed_harness", fields
    detail = check(fields, REFERENCE_CHECK)["detail"]
    assert "OSError: checkpoint not found" in detail
    assert "MPS" not in detail and devices.DEVICE_VARIABLE not in detail
