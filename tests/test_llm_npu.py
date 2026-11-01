"""`convert --npu` for an LLM checkpoint (`litetune.llm_npu`).

The script that runs inside `envs.NPU_LLM` imports only the standard library at its top level,
so its decision functions -- the dead-KV retype, the calibration plan, the exporter flags --
run here for real on synthetic inputs. The run itself is driven with a stand-in for
`StageEnv.run` that writes the answer and the staged bundle the script would.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_embedding_export import _convert

from litetune import envs, llm_npu, wheels
from litetune.checks import Outcome
from litetune.events import EventStream
from litetune.llm_npu import (
    BLOCKED,
    DECISION_REFUSAL,
    DEFAULT_CACHE_LENGTH,
    FAMILIES,
    LLM_NPU_SCHEMA,
    REPORT_NAME,
    LlmNpuError,
    LlmNpuRequest,
    kv_types_problem,
    max_prompt_tokens,
    npu_target_of,
    read_calibration_rows,
    request_problem,
    run_llm_npu_export,
)
from litetune.storage import hash_file

# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def _problem(socs=("SM8750",), cache=DEFAULT_CACHE_LENGTH, family="qwen-3", decision=False):
    return request_problem(socs, cache, family=family, model="m", decision=decision)


def test_a_measured_family_with_a_known_soc_is_accepted():
    for family in FAMILIES:
        assert _problem(family=family) is None


@pytest.mark.parametrize("soc", ["SM8450", "QCS8275", "sm8750"])
def test_a_soc_outside_the_four_is_refused(soc):
    assert soc in _problem(socs=(soc,))


def test_a_repeated_or_missing_soc_is_refused():
    assert "twice" in _problem(socs=("SM8750", "SM8750"))
    assert "no SoC" in _problem(socs=())


def test_a_decision_checkpoint_is_refused_with_the_rollback_line():
    problem = _problem(decision=True)
    assert DECISION_REFUSAL in problem
    assert "llm_litert_npu_compiled_model_executor.cc:2139" in problem
    assert "rolling back one token" in problem


@pytest.mark.parametrize("cache", [128, 255, 1000])
def test_a_cache_length_that_leaves_no_prompt_or_is_off_the_prefill_grid_is_refused(cache):
    assert "--cache-length" in _problem(cache=cache)


def test_the_default_cache_length_is_the_measured_one():
    assert DEFAULT_CACHE_LENGTH == 1024
    assert max_prompt_tokens(1024) == 896


@pytest.mark.parametrize("family", sorted(BLOCKED))
def test_a_blocked_family_is_refused_with_its_reason(family):
    assert BLOCKED[family] in _problem(family=family)


@pytest.mark.parametrize("family", [None, "granite-4.1", "gemma3-text-unidentified"])
def test_an_unmeasured_family_is_refused(family):
    assert "unmeasured family is refused" in _problem(family=family)


def test_the_environment_is_export_plus_the_qualcomm_sdk_on_linux_x86_64():
    env = envs.NPU_LLM
    assert set(envs.EXPORT.requirements) < set(env.requirements)
    assert "ai-edge-litert-sdk-qualcomm==2.3.0" in env.requirements
    assert env.install_as == envs.NPU_QUALCOMM.install_as
    assert env.installed_files == envs.NPU_QUALCOMM.installed_files
    assert {"libc++1", "libatomic1"} <= set(env.system_requirements)
    tags = ("manylinux_2_27_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64")
    linux = wheels.Host.of((*tags, "linux_x86_64"), (3, 12))
    assert wheels.gaps(env.requirements, linux) == []
    assert env.unprovisionable(linux) is None
    mac = wheels.Host.of(("macosx_14_0_arm64",), (3, 12))
    assert env.unprovisionable(mac) is not None
    assert env.identity != envs.EXPORT.identity and env.path != envs.NPU_QUALCOMM.path


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------


def _rows(path: Path, n: int = 16, words: int = 5) -> Path:
    lines = [
        json.dumps({"prompt": " ".join([f"p{i}"] * words), "completion": f"label{i}"})
        for i in range(n)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_calibration_rows_need_a_prompt_and_a_completion(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "x"}) + "\n")
    with pytest.raises(LlmNpuError, match="string prompt and completion"):
        read_calibration_rows(bad)


def test_too_few_calibration_rows_are_refused(tmp_path):
    with pytest.raises(LlmNpuError, match="at least 8"):
        read_calibration_rows(_rows(tmp_path / "few.jsonl", n=3))


# ---------------------------------------------------------------------------
# The script's decisions, run for real
# ---------------------------------------------------------------------------


@pytest.fixture
def script() -> dict[str, Any]:
    namespace: dict[str, Any] = {"__name__": "npu_llm_script"}
    exec(compile(llm_npu._NPU_LLM_SCRIPT, "npu_llm.py", "exec"), namespace)
    return namespace


FLOAT32, INT16 = 0, 7
TYPE_NAMES = {FLOAT32: "FLOAT32", INT16: "INT16"}


def _tensor(type_, shape, quant=None):
    return SimpleNamespace(type=type_, shape=list(shape), quantization=quant)


def _io(name, index):
    return SimpleNamespace(name=name.encode(), tensorIndex=index)


def _graph(dead_shape=(1, 8, 128, 1024)):
    """prefill_128 reads kv_cache_k_0 and not kv_cache_k_1 (the last layer's); decode reads
    both. Quantizer output as observed: the unread prefill input left FLOAT32."""
    prefill = SimpleNamespace(
        tensors=[
            _tensor(INT16, (1, 8, 128, 1024), quant={"scale": [0.5]}),
            _tensor(FLOAT32, dead_shape),
            _tensor(FLOAT32, (1, 128, 64)),
        ],
        operators=[SimpleNamespace(inputs=[0, 2, -1])],
    )
    decode = SimpleNamespace(
        tensors=[
            _tensor(INT16, (1, 8, 128, 1024), quant={"scale": [0.5]}),
            _tensor(INT16, (1, 8, 128, 1024), quant={"scale": [0.25]}),
        ],
        operators=[SimpleNamespace(inputs=[0, 1])],
    )
    return SimpleNamespace(
        subgraphs=[prefill, decode],
        signatureDefs=[
            SimpleNamespace(
                signatureKey=b"prefill_128",
                subgraphIndex=0,
                inputs=[_io("kv_cache_k_0", 0), _io("kv_cache_k_1", 1), _io("embeddings", 2)],
            ),
            SimpleNamespace(
                signatureKey=b"decode",
                subgraphIndex=1,
                inputs=[_io("kv_cache_k_0", 0), _io("kv_cache_k_1", 1)],
            ),
        ],
    )


def test_a_dead_prefill_kv_input_takes_its_decode_namesakes_type_and_quantization(script):
    model = _graph()
    before = script["kv_type_counts"](model, TYPE_NAMES)
    assert before == {"prefill_128": {"INT16": 1, "FLOAT32": 1}, "decode": {"INT16": 2}}
    changed = script["retype_dead_kv"](model, TYPE_NAMES, lambda q: dict(q))
    assert changed == [
        {"signature": "prefill_128", "input": "kv_cache_k_1", "from": "FLOAT32", "to": "INT16"}
    ]
    dead = model.subgraphs[0].tensors[1]
    assert dead.type == INT16 and dead.quantization == {"scale": [0.25]}
    assert dead.quantization is not model.subgraphs[1].tensors[1].quantization
    assert script["kv_type_counts"](model, TYPE_NAMES)["prefill_128"] == {"INT16": 2}
    assert kv_types_problem(script["kv_type_counts"](model, TYPE_NAMES)) is None


def test_a_kv_input_that_is_read_is_left_as_it_is(script):
    model = _graph()
    model.subgraphs[0].operators.append(SimpleNamespace(inputs=[1]))
    assert script["retype_dead_kv"](model, TYPE_NAMES, dict) == []
    assert model.subgraphs[0].tensors[1].type == FLOAT32


def test_a_dead_input_whose_shape_differs_from_decode_is_refused(script):
    with pytest.raises(script["Refused"], match="shape"):
        script["retype_dead_kv"](_graph(dead_shape=(1, 8, 128, 512)), TYPE_NAMES, dict)


def test_a_graph_without_a_decode_signature_is_refused(script):
    model = _graph()
    model.signatureDefs = model.signatureDefs[:1]
    with pytest.raises(script["Refused"], match="no decode signature"):
        script["retype_dead_kv"](model, TYPE_NAMES, dict)


@pytest.mark.parametrize(
    "kv_types, expected",
    [
        ({"prefill_128": {"INT16": 56}, "decode": {"INT16": 56}}, None),
        ({"prefill_128": {"INT16": 54, "FLOAT32": 2}, "decode": {"INT16": 56}}, "FLOAT32"),
        ({}, "not read"),
        ({"decode": {}}, "no KV-cache inputs"),
    ],
)
def test_only_int16_kv_inputs_everywhere_pass(kv_types, expected):
    problem = kv_types_problem(kv_types)
    assert (problem is None) if expected is None else (expected in problem)


def test_long_targets_reach_the_longest_prompt_served(script):
    targets = script["long_targets"](896, 32)
    assert len(targets) == 32 and targets[0] == 448 and targets[-1] == 896
    assert targets == sorted(targets)
    assert script["long_targets"](896, 1) == [896] and script["long_targets"](896, 0) == []


def _words(text: str) -> int:
    return len(text.split())


def test_the_calibration_plan_has_a_short_half_and_a_long_half_up_to_the_limit(script):
    rows = [{"prompt": f"q{i} " * 4, "completion": f"a{i}"} for i in range(40)]
    prompts, summary = script["calibration_plan"](rows, lambda t: t, _words, 200, 8, 8)
    assert summary["short"]["count"] == 8 and summary["long"]["count"] == 8
    assert summary["count"] == len(prompts) == 16
    assert summary["short"]["max_tokens"] == 4
    # Each long prompt is grown example by example (5 words each) until its target.
    assert summary["long"]["min_tokens"] >= 100
    assert 195 <= summary["longest_tokens"] <= 200 == summary["max_prompt_tokens"]
    assert all(_words(p) <= 200 for p in prompts)
    again, same = script["calibration_plan"](rows, lambda t: t, _words, 200, 8, 8)
    assert again == prompts and same["sha256"] == summary["sha256"]


def test_a_training_prompt_longer_than_the_limit_is_refused(script):
    rows = [{"prompt": "w " * 300, "completion": "x"}] + [
        {"prompt": "w", "completion": "x"} for _ in range(10)
    ]
    with pytest.raises(script["Refused"], match="raise --cache-length"):
        script["calibration_plan"](rows, lambda t: t, _words, 200, 4, 4)


def test_exporter_flags_become_keyword_arguments(script):
    got = script["exporter_kwargs"](
        [
            "--jinja_chat_template_override=/t/x.jinja",
            "--externalize_embedder",
            "--litert_lm_model_type_override=gemma3",
        ]
    )
    assert got == {
        "jinja_chat_template_override": "/t/x.jinja",
        "externalize_embedder": True,
        "litert_lm_model_type_override": "gemma3",
    }
    assert script["exporter_kwargs"](["--noexternalize_embedder"]) == {
        "externalize_embedder": False
    }
    with pytest.raises(script["Refused"]):
        script["exporter_kwargs"](["externalize_embedder"])


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _checkpoint(tmp_path: Path, base="google/gemma-3-270m-it", **record) -> Path:
    model = tmp_path / "tuned"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "gemma3_text"}))
    (model / "litetune.json").write_text(
        json.dumps({"base_model": base, "prompt_mode": "runtime_rendered", **record})
    )
    return model


CALIBRATION = {
    "count": 64,
    "short": {"count": 32, "min_tokens": 20, "max_tokens": 60},
    "long": {"count": 32, "min_tokens": 448, "max_tokens": 890},
    "shortest_tokens": 20,
    "longest_tokens": 890,
    "max_prompt_tokens": 896,
    "sha256": "ab" * 32,
}
KV_OK = {"prefill_128": {"INT16": 36}, "decode": {"INT16": 36}}


@dataclass
class FakeRun:
    """`StageEnv.run` on the NPU script: writes the answer and a staged bundle per SoC."""

    kv_types: dict[str, dict[str, int]] = field(default_factory=lambda: dict(KV_OK))
    answer: dict[str, Any] | None = None
    returncode: int = 0
    specs: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 0, **kwargs) -> subprocess.CompletedProcess:
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        answer = self.answer
        if answer is None:
            socs = {}
            for soc in spec["socs"]:
                packed = Path(spec["work"]) / f"{soc}.litertlm"
                packed.write_bytes(f"npu bundle {soc}".encode())
                socs[soc] = {"packed": str(packed), "kv_types": self.kv_types}
            answer = {
                "ok": True,
                "calibration": CALIBRATION,
                "socs": socs,
                "versions": {"litert-torch-nightly": "0.10.0.dev20260926"},
                "stage_s": {"stage1": 1.0},
                "dead_kv": {"changed": [{"input": "kv_cache_k_17"}]},
            }
        Path(spec["answer"]).write_text(json.dumps(answer), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, "", "")


@pytest.fixture
def ready_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(envs.StageEnv, "ready", property(lambda self: True))
    monkeypatch.setattr(envs.StageEnv, "refusal", lambda self: None)
    return envs.NPU_LLM


def _run(tmp_path, model, socs=("SM8750",), **kwargs) -> llm_npu.LlmNpuResult:
    request = LlmNpuRequest(
        model=str(model),
        output_dir=tmp_path / "out",
        socs=tuple(socs),
        calibration_data=_rows(tmp_path / "train.jsonl"),
        auto_provision=False,
        **kwargs,
    )
    return run_llm_npu_export(request, EventStream(echo_json=False), host="")


def test_a_bundle_is_published_with_its_report(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    result = _run(tmp_path, _checkpoint(tmp_path), socs=("SM8750", "SM8650"))
    assert result.outcome is Outcome.PASSED, result.as_dict()["checks"]
    spec = fake.specs[0]
    assert spec["socs"] == ["SM8750", "SM8650"] and spec["max_prompt_tokens"] == 896
    assert spec["prompt_mode"] == "runtime_rendered" and spec["prefill_length"] == 128
    assert "--litert_lm_model_type_override=gemma3" in spec["flags"]
    for soc in ("SM8750", "SM8650"):
        model = tmp_path / "out" / f"npu-{soc}" / "model.litertlm"
        report = json.loads((model.parent / REPORT_NAME).read_text())
        assert model.read_bytes() == f"npu bundle {soc}".encode()
        assert report["schema"] == LLM_NPU_SCHEMA and report["kind"] == "npu"
        assert report["generation_only"] is True and report["verified"] is False
        assert report["max_prompt_tokens"] == 890 and report["calibration"] == CALIBRATION
        assert f"sha256:{report['npu']['sha256']}" == hash_file(model)
        assert npu_target_of(model) == soc
    as_dict = result.as_dict()
    assert as_dict["npu"][0]["device_run"] == llm_npu.DEVICE_RUNS["SM8750"]
    assert as_dict["npu"][1]["device_run"] is None
    assert any("longer than 890 tokens" in t for t in as_dict["limitations"])
    assert not list((tmp_path / "out").glob(".litetune-npu-llm-*"))


def test_a_compiled_bundle_with_a_float_kv_input_is_not_published(monkeypatch, tmp_path, ready_env):
    monkeypatch.setattr(
        envs.StageEnv,
        "run",
        FakeRun(kv_types={"prefill_128": {"INT16": 34, "FLOAT32": 2}, "decode": {"INT16": 36}}),
    )
    result = _run(tmp_path, _checkpoint(tmp_path))
    assert result.outcome is Outcome.FAILED
    (check,) = [c for c in result.checks.checks if c.name == llm_npu.NPU_COMPILE_CHECK]
    assert "FLOAT32" in check.detail and check.detail.startswith("compiled:")
    assert not (tmp_path / "out" / "npu-SM8750").exists()


def test_a_script_refusal_keeps_its_cause(monkeypatch, tmp_path, ready_env):
    refused = {"ok": False, "cause": "data", "reason": "raise --cache-length"}
    monkeypatch.setattr(envs.StageEnv, "run", FakeRun(answer=refused, returncode=3))
    result = _run(tmp_path, _checkpoint(tmp_path))
    (check,) = [c for c in result.checks.checks if c.name == llm_npu.NPU_COMPILE_CHECK]
    assert check.detail == "data: raise --cache-length"


def test_a_decision_checkpoint_is_refused_before_anything_runs(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    model = _checkpoint(tmp_path, decision={"options": ["a", "b"]})
    result = _run(tmp_path, model)
    assert result.outcome is Outcome.FAILED and fake.specs == []
    assert "rolling back one token" in result.checks.checks[0].detail


def test_an_unmeasured_family_is_refused_before_anything_runs(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    result = _run(tmp_path, _checkpoint(tmp_path, base="ibm-granite/granite-4.1-3b"))
    assert result.outcome is Outcome.FAILED and fake.specs == []
    assert "unmeasured family" in result.checks.checks[0].detail


def test_a_checkpoint_without_a_prompt_mode_is_refused(monkeypatch, tmp_path, ready_env):
    model = _checkpoint(tmp_path)
    (model / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m-it"}))
    result = _run(tmp_path, model)
    assert "records no prompt mode" in result.checks.checks[0].detail


def test_an_existing_output_is_refused(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    (tmp_path / "out" / "npu-SM8750").mkdir(parents=True)
    result = _run(tmp_path, _checkpoint(tmp_path))
    assert "already exists" in result.checks.checks[0].detail and fake.specs == []


def test_off_linux_x86_64_it_is_refused(tmp_path):
    request = LlmNpuRequest(
        model=str(tmp_path),
        output_dir=tmp_path / "out",
        socs=("SM8750",),
        calibration_data=tmp_path / "x.jsonl",
        auto_provision=False,
    )
    result = run_llm_npu_export(request, EventStream(echo_json=False), host="Linux x86-64 only")
    assert result.outcome is Outcome.FAILED
    assert result.checks.checks[0].detail == "Linux x86-64 only"


def test_npu_target_of_reads_only_its_own_report_for_this_file(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    report = {
        "schema": LLM_NPU_SCHEMA,
        "kind": "npu",
        "npu": {"soc": "SM8750", "sha256": hash_file(model).removeprefix("sha256:")},
    }
    (tmp_path / REPORT_NAME).write_text(json.dumps(report))
    assert npu_target_of(model) == "SM8750"
    model.write_bytes(b"another bundle")
    assert npu_target_of(model) is None
    (tmp_path / REPORT_NAME).write_text(json.dumps({**report, "schema": "other"}))
    assert npu_target_of(model) is None


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_llm_npu_without_calibration_data_is_refused(tmp_path, capsys):
    model = _checkpoint(tmp_path)
    assert _convert(model, tmp_path / "out", "--npu", "SM8750") == 4
    assert "--calibration-data" in capsys.readouterr().err


def test_llm_npu_refuses_the_recipe_flags(tmp_path, capsys):
    model = _checkpoint(tmp_path)
    data = _rows(tmp_path / "train.jsonl")
    args = ("--npu", "SM8750", "--calibration-data", str(data), "--recipe", "dynamic_wi8_afp32")
    assert _convert(model, tmp_path / "out", *args) == 4
    assert "--recipe" in capsys.readouterr().err


def test_calibration_flags_without_npu_are_refused(tmp_path, capsys):
    model = _checkpoint(tmp_path)
    assert _convert(model, tmp_path / "out", "--cache-length", "1024") == 4
    assert "--calibration-data and --cache-length are for --npu" in capsys.readouterr().err


def test_verify_refuses_an_llm_npu_bundle(tmp_path):
    from litetune.verify import Status, VerifyRequest, run_verify

    model = tmp_path / "npu-SM8750" / "model.litertlm"
    model.parent.mkdir()
    model.write_bytes(b"bundle")
    report = {
        "schema": LLM_NPU_SCHEMA,
        "kind": "npu",
        "npu": {"soc": "SM8750", "sha256": hash_file(model).removeprefix("sha256:")},
    }
    (model.parent / REPORT_NAME).write_text(json.dumps(report))
    data = _rows(tmp_path / "held.jsonl")
    result = run_verify(
        VerifyRequest(model=model, reference=str(_checkpoint(tmp_path)), data=data),
        EventStream(echo_json=False),
    )
    assert result.status is Status.FAILED_HARNESS
    assert "Qualcomm SM8750 NPU" in json.dumps(result.manifest)
