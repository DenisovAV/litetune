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
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from litetune import calibration
from litetune.decision_verify import (
    AGREEMENT_CHECK,
    CALIBRATION_CHECK,
    REFERENCE_CHECK,
    ROWS_CHECK,
    RULE_CHECK,
    RUNTIME_CHECK,
    TARGETS_CHECK,
    TWIN_CHECK,
    DecisionVerify,
    section_bytes,
    unique_lengths,
    verify_decision,
)
from litetune.decisions import decision_from
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
    """
    right = label_of(prompt) or "none_of_these"
    number = int(re.search(r"\d+", prompt).group())  # type: ignore[union-attr]
    first = 0.0 if option == right else -2.0 - (number % 5) - 0.1 * len(option)
    return [first] + [-0.01] * (len(ids) - 1)


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
    specs: list[dict[str, Any]] = field(default_factory=list)
    provisioned: int = 0

    def provision(self, events=None, force: bool = False) -> Path:
        self.provisioned += 1
        return Path("/nonexistent")

    def run(self, argv, timeout=None, env=None) -> subprocess.CompletedProcess:
        script = Path(argv[1]).read_text(encoding="utf-8")
        spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        engine = "litert_lm" in script
        cont = [self.token_of(o) for o in spec["options"]]
        ks = unique_lengths(cont)
        report: dict[str, Any] = {"cont_ids": cont}
        if engine:
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
                    "selfcheck": {"verdict": "dropped", "comparisons": 18, "max_abs_diff": 0.0},
                    "k": ks,
                    "scored_lengths": lengths,
                    "mismatches": [{"option": o} for o in spec["options"] if o in self.full_for],
                }
            )
        else:
            report.update({"device": "cpu", "cache_variant": "position_ids"})
        report.update(self.report_extra)
        with open(spec["out"], "w", encoding="utf-8") as out:
            for row in spec["rows"]:
                if row["index"] in self.skip_rows:
                    continue
                per = [
                    self.scores(row["index"], row["prompt"], o, c)
                    for o, c in zip(spec["options"], cont, strict=True)
                ]
                if engine:
                    per = [p[:n] for p, n in zip(per, lengths, strict=True)]
                rec = {"index": row["index"], "token_scores": per}
                rec["ms" if engine else "seconds"] = 10.0
                out.write(json.dumps(rec) + "\n")
        Path(spec["report"]).write_text(json.dumps(report), encoding="utf-8")
        return subprocess.CompletedProcess(argv, self.returncode, "", "fake stderr")


@pytest.fixture
def split(tmp_path):
    """`prepare`'s decision split: three emotions and an out-of-scope option."""

    def _make(with_none: bool = True, per_option: int = 30, oos: int = 40) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "question": "Which emotion?",
            "options": [{"id": o} for o in OPTIONS],
        }
        if with_none:
            obj["none_option"] = "none_of_these"
        questions = tmp_path / "questions.json"
        questions.write_text(json.dumps(obj), encoding="utf-8")
        rows: list[dict[str, Any]] = [
            {"prompt": f"text {i} about {o}", "label": o}
            for o in OPTIONS
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
        "temperature": fit["temperature"],
        "abstain_below": fit["abstain_below"],
    }
    assert fields["decision"]["section"]["bytes"] == len(section_bytes(payload))
    assert fields["quality"]["argmax_agreement"] == 1.0
    assert fields["measurements"]["reference"]["accuracy"] == 1.0
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
    assert ref_spec["terminator"] == TERM
    assert len(eng_spec["rows"]) == 5
    assert eng_spec["mode"] == "auto" and eng_spec["targets"] == "unique"
    held = [json.loads(line)["prompt"] for line in s["heldout"].read_text().splitlines()]
    assert [r["prompt"] for r in eng_spec["rows"]] == held[:5]


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


def test_the_gpu_backend_is_refused(split, tmp_path):
    s = split()

    status, fields = run(job_for(s, tmp_path, backend="gpu"), s["record"])

    assert status == "failed_harness"
    assert "CPU backend only" in check(fields, RUNTIME_CHECK)["detail"]


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
function of the context the session reads. With FAKE_DROPS=1 the session reads
a prompt prefilled once without its second-to-last token, as LiteRT-LM 0.18.0
does (#3561), and prefilling its last token again restores it; without it the
session reads what was prefilled, as an engine with the fix does.
"""
import os
import re
import zlib

_SPECIAL = r"<start_of_turn>|<end_of_turn>|<bos>"
_WORDS = {}


def _ids(text):
    """One id per piece, the same in every process: crc32 of the piece."""
    out = []
    for piece in re.findall(_SPECIAL + r"|[A-Za-z_]+|.", text, flags=re.S):
        token = zlib.crc32(piece.encode("utf-8")) % 1_000_003 + 1
        _WORDS[token] = piece
        out.append(token)
    return out


def true_score(context, token):
    return -((hash(tuple(context[-3:])) + 31 * token) % 997) / 100.0


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


class _Lib:
    litert_lm_session_save_checkpoint = _CFunc()
    litert_lm_session_rewind_to_checkpoint = _CFunc()
    litert_lm_get_last_error_message = _CFunc()


class _Result:
    def __init__(self, scores):
        self.token_scores = [scores]


class _Session:
    def __init__(self):
        self.tokens = []
        self.prefills = 0
        self._lib = _Lib()
        self._ptr = 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run_prefill(self, parts):
        for part in parts:
            self.tokens += _ids(part)
        self.prefills += 1

    def _context(self):
        t = self.tokens
        if os.environ.get("FAKE_DROPS") != "1":
            return t
        if self.prefills == 1 and len(t) >= 2:
            return t[:-2] + t[-1:]
        if self.prefills >= 2 and len(t) >= 2 and t[-1] == t[-2]:
            return t[:-1]
        return t

    def run_text_scoring(self, texts, store_token_lengths=False):
        ctx, target = list(self._context()), _ids(texts[0])
        return _Result([true_score(ctx + target[:i], target[i]) for i in range(len(target))])


class _Conversation:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def render_message_to_string(self, prompt):
        return "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n"


class Engine:
    def __init__(self, path, backend=None, cache_dir=None):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def tokenize(self, text):
        return _ids(text)

    def detokenize(self, ids):
        return "".join(_WORDS[i] for i in ids)

    def create_session(self, apply_prompt_template=True, sampler_config=None):
        return _Session()

    def create_conversation(self, sampler_config=None, max_output_tokens=None):
        return _Conversation()
'''


def _run_engine_script(tmp_path, drops: bool, prompts, options, mode="auto"):
    import os
    import sys

    from litetune.decision_scripts import ENGINE_SCRIPT

    fake = tmp_path / "fake"
    fake.mkdir(exist_ok=True)
    (fake / "litert_lm.py").write_text(FAKE_LITERT_LM, encoding="utf-8")
    script = tmp_path / "engine.py"
    script.write_text(ENGINE_SCRIPT, encoding="utf-8")
    spec = tmp_path / "spec.json"
    out, report = tmp_path / "out.jsonl", tmp_path / "report.json"
    spec.write_text(
        json.dumps(
            {
                "model": "m.litertlm",
                "rows": [{"index": i, "prompt": p} for i, p in enumerate(prompts)],
                "options": list(options),
                "terminator": TERM,
                "threads": 2,
                "targets": "unique",
                "mode": mode,
                "out": str(out),
                "report": str(report),
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(fake), FAKE_DROPS="1" if drops else "0")
    proc = subprocess.run(
        [sys.executable, str(script), str(spec)], env=env, capture_output=True, text=True
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    return proc, rows, json.loads(report.read_text())


def _expected(tmp_path, prompts, options):
    """Each option's unique-prefix token scores on the whole rendered prompt."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fake_litert_lm", tmp_path / "fake" / "litert_lm.py"
    )
    fake = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fake)
    cont = [fake._ids(o + TERM) for o in options]
    ks = unique_lengths(cont)
    out = []
    for p in prompts:
        ctx = fake._ids("<start_of_turn>user\n" + p + "<end_of_turn>\n<start_of_turn>model\n")
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
    assert fields["measurements"]["reference"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# What the review found
# ---------------------------------------------------------------------------


def test_the_scorers_get_the_first_token_of_the_recorded_end_of_turn(split, tmp_path):
    s = split()
    job = job_for(s, tmp_path)

    status, _ = run(job, s["record"])

    assert status == "passed"
    assert job.train_env.specs[0]["terminator"] == "<end_of_turn>"
    assert job.runtime_env.specs[0]["terminator"] == "<end_of_turn>"


def test_a_checkpoint_whose_turn_ends_otherwise_is_refused(split, tmp_path):
    s = split()
    record = dict(s["record"], turn_terminator={"text": "<|im_end|>\n", "ids": [1]})

    status, fields = run(job_for(s, tmp_path), record)

    assert status == "failed_harness"
    failed = [c for c in fields["checks"]["checks"] if c["outcome"] == "failed"]
    assert len(failed) == 1 and "'<|im_end|>\\n'" in failed[0]["detail"]


def test_options_scored_in_full_on_the_runtime_are_scored_in_full_on_both_sides(split, tmp_path):
    s = split()

    def tail_heavy(index, prompt, option, ids):
        base = default_scores(index, prompt, option, ids)
        # The tail decides: summed over the unique prefix only, "anger" would win.
        if option == "anger":
            return [0.5, *([-30.0] * (len(ids) - 1))]
        return base

    eng = FakeScoringEnv("runtime", scores=tail_heavy, full_for={"anger"})
    job = job_for(
        s, tmp_path, train_env=FakeScoringEnv("train", scores=tail_heavy), runtime_env=eng
    )

    status, fields = run(job, s["record"])

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
