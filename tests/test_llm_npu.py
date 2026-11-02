"""`convert --npu` for an LLM checkpoint (`litetune.llm_npu`).

The script that runs inside `envs.NPU_LLM` imports only the standard library at its top level,
so its decision functions -- the dead-KV retype, the KV quantization check, the calibration
plan, the exporter flags, the compiled-graph check -- run here for real on synthetic inputs.
The run itself is driven with a stand-in for `StageEnv.run` (`FakeRun`) that writes the answer
and the staged bundles the script would, and answers the chat-template script as the other
toolchain fakes do.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import run_compose_script
from test_embedding_export import _convert

from litetune import cli, envs, llm_npu, wheels
from litetune.checks import Check, Outcome
from litetune.events import EventStream
from litetune.llm_npu import (
    BLOCKED,
    DECISION_REFUSAL,
    DEFAULT_CACHE_LENGTH,
    FAMILIES,
    LLM_NPU_SCHEMA,
    MAX_CACHE_LENGTH,
    MIN_CACHE_LENGTH,
    NOT_VERIFIED,
    REPORT_NAME,
    RUN_REPORT_NAME,
    LlmNpuError,
    LlmNpuRequest,
    kv_types_problem,
    npu_target_of,
    prompt_limit_tokens,
    read_calibration_rows,
    request_problem,
    run_llm_npu_export,
    summarise,
)
from litetune.storage import hash_file

GEMMA = "google/gemma-3-270m-it"
QWEN25 = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def _problem(
    socs=("SM8750",),
    cache=DEFAULT_CACHE_LENGTH,
    family="qwen-3",
    identity="Qwen/Qwen3-0.6B",
    decision=False,
):
    return request_problem(socs, cache, family=family, identity=identity, decision=decision)


@pytest.mark.parametrize(
    "family, identity",
    [
        ("gemma-3-text", "google/gemma-3-270m-it"),
        ("qwen-3", "Qwen/Qwen3-0.6B"),
        ("qwen-2.5", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("minicpm5", "openbmb/MiniCPM5-1B"),
    ],
)
def test_a_measured_family_at_its_measured_size_is_accepted(family, identity):
    assert family in FAMILIES
    assert _problem(family=family, identity=identity) is None


@pytest.mark.parametrize(
    "family, identity",
    [
        ("gemma-3-text", "google/gemma-3-1b-it"),
        ("qwen-3", "Qwen/Qwen3-1.7B"),
        ("qwen-3", "Qwen/Qwen3-4B"),
        ("qwen-2.5", "Qwen/Qwen2.5-1.5B-Instruct"),
    ],
)
def test_another_size_of_a_measured_family_is_refused(family, identity):
    assert "measured here at" in _problem(family=family, identity=identity)


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


@pytest.mark.parametrize(
    "cache, ok",
    [
        (128, False),
        (255, False),
        (256, True),
        (1000, False),
        (1024, True),
        (4096, True),
        (4224, False),
    ],
)
def test_the_cache_length_bounds(cache, ok):
    problem = _problem(cache=cache)
    assert (problem is None) if ok else ("--cache-length" in problem)


def test_the_defaults_are_the_measured_ones():
    assert DEFAULT_CACHE_LENGTH == 1024 and MIN_CACHE_LENGTH == 256 and MAX_CACHE_LENGTH == 4096
    assert prompt_limit_tokens(1024) == 896
    assert llm_npu.CALIBRATION_DECODE_STEPS == llm_npu.REPLY_TOKENS == 128


@pytest.mark.parametrize("family", sorted(BLOCKED))
def test_a_blocked_family_is_refused_with_its_reason(family):
    assert BLOCKED[family] in _problem(family=family)


def test_functiongemma_is_refused():
    assert "functiongemma" in BLOCKED and "functiongemma" not in FAMILIES


@pytest.mark.parametrize("family", [None, "granite-4.1", "gemma3-text-unidentified"])
def test_an_unmeasured_family_is_refused(family):
    assert "unmeasured family is refused" in _problem(family=family)


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def test_the_environment_is_export_plus_the_qualcomm_sdk_on_linux_x86_64():
    env = envs.NPU_LLM
    assert set(envs.EXPORT.requirements) < set(env.requirements)
    assert {"ai-edge-litert-sdk-qualcomm==2.3.0", "flatbuffers==25.12.19"} <= set(env.requirements)
    assert any(r.startswith("numpy==") for r in env.requirements)
    assert env.install_as == envs.NPU_QUALCOMM.install_as
    assert env.installed_files == envs.NPU_QUALCOMM.installed_files
    assert env.provision_variables == (("SKIP_SDK_DOWNLOAD", ""),)
    assert env.pip_options == ("--no-cache-dir",)
    assert {"libvulkan1", "libc++1", "libatomic1"} == set(env.system_requirements)
    tags = ("manylinux_2_27_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64")
    linux = wheels.Host.of((*tags, "linux_x86_64"), (3, 12))
    assert wheels.gaps(env.requirements, linux) == []
    assert env.unprovisionable(linux) is None
    mac = wheels.Host.of(("macosx_14_0_arm64",), (3, 12))
    assert env.unprovisionable(mac) is not None
    assert env.identity != envs.EXPORT.identity and env.path != envs.NPU_QUALCOMM.path


def test_the_npu_llm_environment_is_claimed_in_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    envs.NPU_LLM.path.mkdir(parents=True)
    (entry,) = envs.cached_environments()
    assert entry.stage == envs.NPU_LLM.name


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
    with pytest.raises(LlmNpuError, match="string prompt and completion") as raised:
        read_calibration_rows(bad)
    assert raised.value.cause == "data"


def test_calibration_rows_skip_blank_lines_and_need_eight(tmp_path):
    path = _rows(tmp_path / "eight.jsonl", n=8)
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")
    assert len(read_calibration_rows(path)) == 8
    with pytest.raises(LlmNpuError, match="at least 8"):
        read_calibration_rows(_rows(tmp_path / "seven.jsonl", n=7))


def test_unreadable_calibration_data_is_refused(tmp_path):
    with pytest.raises(LlmNpuError, match="calibration-data") as missing:
        read_calibration_rows(tmp_path / "missing.jsonl")
    assert missing.value.cause == "input"
    binary = tmp_path / "binary.jsonl"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(LlmNpuError, match="not UTF-8") as undecodable:
        read_calibration_rows(binary)
    assert undecodable.value.cause == "data"
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n")
    with pytest.raises(LlmNpuError, match="broken.jsonl:1") as bad:
        read_calibration_rows(broken)
    assert bad.value.cause == "data"
    with pytest.raises(LlmNpuError) as few:
        read_calibration_rows(_rows(tmp_path / "seven.jsonl", n=7))
    assert few.value.cause == "input"


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


def _quant(scale, zero=0):
    return SimpleNamespace(scale=[scale], zeroPoint=[zero])


def _copy(quant):
    return SimpleNamespace(**vars(quant)) if quant is not None else None


def _tensor(type_, shape, quant=None):
    return SimpleNamespace(type=type_, shape=list(shape), quantization=quant)


def _io(name, index):
    return SimpleNamespace(name=name.encode(), tensorIndex=index)


def _graph(dead_shape=(1, 8, 128, 1024)):
    """prefill_128 reads kv_cache_k_0 and not kv_cache_k_1 (the last layer's); decode reads
    both. Quantizer output as observed: the unread prefill input left FLOAT32."""
    prefill = SimpleNamespace(
        tensors=[
            _tensor(INT16, (1, 8, 128, 1024), quant=_quant(0.5)),
            _tensor(FLOAT32, dead_shape),
            _tensor(FLOAT32, (1, 128, 64)),
            _tensor(FLOAT32, (1, 128, 64)),
            _tensor(FLOAT32, (1, 8, 128, 1024)),
            _tensor(INT16, (1, 8, 128, 1024), quant=_quant(0.25)),
        ],
        operators=[SimpleNamespace(inputs=[0, 2, -1])],
        outputs=[3],
    )
    decode = SimpleNamespace(
        tensors=[
            _tensor(INT16, (1, 8, 128, 1024), quant=_quant(0.5)),
            _tensor(INT16, (1, 8, 128, 1024), quant=_quant(0.25)),
        ],
        operators=[SimpleNamespace(inputs=[0, 1])],
        outputs=[],
    )
    return SimpleNamespace(
        subgraphs=[prefill, decode],
        signatureDefs=[
            SimpleNamespace(
                signatureKey=b"prefill_128",
                subgraphIndex=0,
                inputs=[_io("kv_cache_k_0", 0), _io("kv_cache_k_1", 1), _io("embeddings", 2)],
                outputs=[_io("kv_slice_k_0", 3)],
            ),
            SimpleNamespace(
                signatureKey=b"decode",
                subgraphIndex=1,
                inputs=[_io("kv_cache_k_0", 0), _io("kv_cache_k_1", 1)],
                outputs=[],
            ),
        ],
    )


def test_a_dead_prefill_kv_input_takes_its_decode_namesakes_type_and_quantization(script):
    model = _graph()
    before = script["kv_type_counts"](model, TYPE_NAMES)
    assert before == {"prefill_128": {"INT16": 1, "FLOAT32": 1}, "decode": {"INT16": 2}}
    changed = script["retype_dead_kv"](model, TYPE_NAMES, _copy)
    assert changed == [
        {"signature": "prefill_128", "input": "kv_cache_k_1", "from": "FLOAT32", "to": "INT16"}
    ]
    dead = model.subgraphs[0].tensors[1]
    assert dead.type == INT16 and dead.quantization.scale == [0.25]
    assert dead.quantization is not model.subgraphs[1].tensors[1].quantization
    assert script["kv_type_counts"](model, TYPE_NAMES)["prefill_128"] == {"INT16": 2}
    assert kv_types_problem(script["kv_type_counts"](model, TYPE_NAMES)) is None
    assert script["kv_quantization_problem"]([model], TYPE_NAMES) is None


def test_a_kv_input_that_is_read_is_left_as_it_is(script):
    model = _graph()
    model.subgraphs[0].operators.append(SimpleNamespace(inputs=[1]))
    assert script["retype_dead_kv"](model, TYPE_NAMES, _copy) == []
    assert model.subgraphs[0].tensors[1].type == FLOAT32


def test_a_kv_input_handed_out_as_an_output_is_not_dead(script):
    model = _graph()
    model.subgraphs[0].outputs = [1]
    assert script["retype_dead_kv"](model, TYPE_NAMES, _copy) == []
    model = _graph()
    model.subgraphs[0].outputs = None
    model.signatureDefs[0].outputs = [_io("kv_cache_k_1", 1)]
    assert script["retype_dead_kv"](model, TYPE_NAMES, _copy) == []


def test_an_op_without_inputs_does_not_break_the_retype(script):
    model = _graph()
    model.subgraphs[0].operators.append(SimpleNamespace(inputs=None))
    assert len(script["retype_dead_kv"](model, TYPE_NAMES, _copy)) == 1


def test_the_retype_leaves_what_it_has_no_reason_to_touch(script):
    model = _graph()
    sd = model.signatureDefs[0]
    # The dead FLOAT32 kv_cache_k_1 replaced by: a dead non-KV input, a dead KV input with no
    # decode twin, and a dead KV input that is INT16 already.
    sd.inputs = [
        _io("kv_cache_k_0", 0),
        _io("other_input", 1),
        _io("kv_cache_v_9", 4),
        _io("kv_cache_k_1", 5),
    ]
    assert script["retype_dead_kv"](model, TYPE_NAMES, _copy) == []
    assert model.subgraphs[0].tensors[1].type == FLOAT32
    assert model.subgraphs[0].tensors[4].type == FLOAT32


def test_a_dead_input_whose_shape_differs_from_decode_is_refused(script):
    with pytest.raises(script["Refused"], match="shape"):
        script["retype_dead_kv"](_graph(dead_shape=(1, 8, 128, 512)), TYPE_NAMES, _copy)


def test_a_graph_without_a_decode_signature_is_refused(script):
    model = _graph()
    model.signatureDefs = model.signatureDefs[:1]
    with pytest.raises(script["Refused"], match="no decode signature"):
        script["retype_dead_kv"](model, TYPE_NAMES, _copy)


def _one_input_graph(name, tensor, key=b"decode_cache_update"):
    return SimpleNamespace(
        subgraphs=[SimpleNamespace(tensors=[tensor], operators=[], outputs=[])],
        signatureDefs=[
            SimpleNamespace(signatureKey=key, subgraphIndex=0, inputs=[_io(name, 0)], outputs=[])
        ],
    )


def test_kv_inputs_must_agree_across_graphs(script):
    model = _graph()
    script["retype_dead_kv"](model, TYPE_NAMES, _copy)
    aux = _one_input_graph("kv_cache_k_0", _tensor(INT16, (1,), quant=_quant(0.5)))
    assert script["kv_quantization_problem"]([model, aux], TYPE_NAMES) is None
    aux.subgraphs[0].tensors[0].quantization = _quant(0.125)
    assert "kv_cache_k_0" in script["kv_quantization_problem"]([model, aux], TYPE_NAMES)
    floating = _one_input_graph("kv_cache_v_7", _tensor(FLOAT32, (1,)))
    assert "not INT16" in script["kv_quantization_problem"]([floating], TYPE_NAMES)


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


def _stamp(soc):
    return b"Qualcomm".ljust(125, b"\0") + soc.encode().ljust(125, b"\0")


def test_the_compiled_signatures_are_the_two_graphs_and_the_two_masks(script):
    assert script["compiled_signatures"](128) == (
        ["prefill_128", "decode"],
        ["prefill_mask_128", "decode_mask"],
    )


def test_a_compiled_graph_must_be_one_dispatch_op_per_signature_and_stamped(script):
    check, refused = script["check_compiled"], script["Refused"]
    sigs = ["prefill_128", "decode"]
    good = {
        "ops": {"prefill_128": {"DISPATCH_OP": 1}, "decode": {"DISPATCH_OP": 1}},
        "stamp": _stamp("SM8750"),
    }
    assert check(good, "SM8750", sigs) == ["Qualcomm", "SM8750"]
    for ops in (
        {"prefill_128": {"DISPATCH_OP": 1}, "decode": {"DISPATCH_OP": 2}},
        {"prefill_128": {"DISPATCH_OP": 1}, "decode": {"DISPATCH_OP": 1, "ADD": 1}},
        {"prefill_128": {"DISPATCH_OP": 1}, "decode": {}},
    ):
        with pytest.raises(refused, match="not one DISPATCH_OP"):
            check({**good, "ops": ops}, "SM8750", sigs)
    with pytest.raises(refused, match="signatures are"):
        check({**good, "ops": {"decode": {"DISPATCH_OP": 1}}}, "SM8750", sigs)
    extra = {**good["ops"], "prefill_256": {"DISPATCH_OP": 1}}
    with pytest.raises(refused, match="signatures are"):
        check({**good, "ops": extra}, "SM8750", sigs)
    assert check({**good, "ops": extra}, "SM8750", sigs, only=False) == ["Qualcomm", "SM8750"]
    with pytest.raises(refused, match="LiteRtStamp is None"):
        check({**good, "stamp": None}, "SM8750", sigs)
    with pytest.raises(refused, match="stamped"):
        check(good, "SM8650", sigs)


AUX_BEFORE = {
    "prefill_cache_update_128": {"DYNAMIC_UPDATE_SLICE": 4},
    "decode_cache_update": {"DYNAMIC_UPDATE_SLICE": 4},
    "prefill_mask_128": {"LESS": 1, "SELECT_V2": 1},
    "decode_mask": {"LESS": 1, "SELECT_V2": 1},
}


def _aux(**ops):
    compiled = {
        **AUX_BEFORE,
        "prefill_mask_128": {"DISPATCH_OP": 1},
        "decode_mask": {"DISPATCH_OP": 1},
    }
    return {"ops": {k: v for k, v in {**compiled, **ops}.items() if v is not None},
            "stamp": _stamp("SM8750")}


def test_the_aux_graph_needs_its_masks_dispatched_and_keeps_its_cache_updates(script):
    check, refused = script["check_compiled_aux"], script["Refused"]
    assert check(_aux(), "SM8750", 128, AUX_BEFORE) == ["Qualcomm", "SM8750"]
    with pytest.raises(refused, match="stamped"):
        check(_aux(), "SM8650", 128, AUX_BEFORE)
    with pytest.raises(refused, match="decode_mask"):
        check(_aux(decode_mask={"LESS": 1}), "SM8750", 128, AUX_BEFORE)
    with pytest.raises(refused, match="decode_cache_update"):
        check(_aux(decode_cache_update={"DISPATCH_OP": 1}), "SM8750", 128, AUX_BEFORE)


@pytest.mark.parametrize(
    "ops",
    [
        {"decode_cache_update": None},
        {"prefill_mask_256": {"DISPATCH_OP": 1}},
    ],
)
def test_the_aux_graph_keeps_stage_3s_signatures_exactly(script, ops):
    with pytest.raises(script["Refused"], match="stage 3's"):
        script["check_compiled_aux"](_aux(**ops), "SM8750", 128, AUX_BEFORE)


def test_an_aux_graph_without_the_masks_is_refused(script):
    before = {k: v for k, v in AUX_BEFORE.items() if "mask" not in k}
    compiled = {"ops": dict(before), "stamp": _stamp("SM8750")}
    with pytest.raises(script["Refused"], match="signatures are"):
        script["check_compiled_aux"](compiled, "SM8750", 128, before)


def test_decode_summary_says_how_far_the_calibration_replies_ran(script):
    record = [{"steps": 128, "position": 1000}, {"steps": 7, "position": 60},
              {"steps": 40, "position": 200}]
    assert script["decode_summary"](record, 128) == {
        "count": 3, "cap": 128, "min_steps": 7, "median_steps": 40, "max_steps": 128,
        "full": 1, "longest_position": 1000,
    }
    record.append({"steps": 9, "position": 70})
    assert script["decode_summary"](record, 128)["median_steps"] == 24.5
    assert script["decode_summary"]([], 128) is None


def test_recording_decode_counts_each_prompts_decode_steps(script):
    class Executor:
        def sample_text(self, prompt, max_sample_step=None):
            state = SimpleNamespace(time_step=len(prompt))
            for _ in range(max_sample_step):
                state = self.decode_step(state)
            return [prompt]

        def decode_step(self, state):
            return SimpleNamespace(time_step=state.time_step + 1)

    record = []
    script["recording_decode"](Executor, record)
    assert Executor().sample_text("abcd", max_sample_step=3) == ["abcd"]
    Executor().sample_text("ab", max_sample_step=0)
    assert record == [{"steps": 3, "position": 7}, {"steps": 0, "position": 0}]


def _sections(tmp_path, name, **files):
    out = {}
    for key, content in files.items():
        if key in ("chat_template", "stop_token_ids"):
            out[key] = content
            continue
        path = tmp_path / f"{name}-{key}"
        path.write_bytes(content)
        out[key] = str(path)
    return out


def test_the_compiled_metadata_must_be_stage_3s(script, tmp_path):
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    compare, refused = script["compare_metadata"], script["Refused"]
    # The unpacker reads the stop tokens into a set.
    files = {"LlmMetadataProto": b"meta", "HF_Tokenizer": b"tok", "chat_template": "{{x}}",
             "stop_token_ids": {106, 1}}
    reference = _sections(tmp_path, "ref", **files)
    got = compare(_sections(tmp_path, "npu", **files), reference, sha)
    assert got["LlmMetadataProto"] == hashlib.sha256(b"meta").hexdigest()
    assert got["stop_token_ids"] == [1, 106] and "SP_Tokenizer" not in got
    json.dumps(got)
    with pytest.raises(refused, match="HF_Tokenizer"):
        compare(_sections(tmp_path, "a", **{**files, "HF_Tokenizer": b"other"}), reference, sha)
    with pytest.raises(refused, match="stop_token_ids"):
        compare(_sections(tmp_path, "b", **{**files, "stop_token_ids": {1}}), reference, sha)
    with pytest.raises(refused, match="chat_template"):
        compare(_sections(tmp_path, "c", **{**files, "chat_template": None}), reference, sha)
    no_tokenizer = {k: v for k, v in reference.items() if k != "HF_Tokenizer"}
    with pytest.raises(refused, match="no tokenizer"):
        compare(reference, no_tokenizer, sha)
    no_metadata = {k: v for k, v in reference.items() if k != "LlmMetadataProto"}
    with pytest.raises(refused, match="no LlmMetadata"):
        compare(reference, no_metadata, sha)


def test_long_targets_reach_the_prompt_limit(script):
    targets = script["long_targets"](896, 32)
    assert len(targets) == 32 and targets[0] == 448 and targets[-1] == 896
    assert targets == sorted(targets)
    assert script["long_targets"](896, 1) == [896] and script["long_targets"](896, 0) == []


def _words(text: str) -> int:
    return len(text.split())


def _render(text: str) -> str:
    return f"<u> {text} </u>"


def _example_rows(n=40):
    return [{"prompt": f"q{i} " * 4, "completion": f"a{i}"} for i in range(n)]


def test_the_calibration_plan_renders_both_halves_and_reaches_the_limit(script):
    rows = _example_rows()
    prompts, summary = script["calibration_plan"](rows, _render, _words, 200, 8, 8, 20)
    assert all(p.startswith("<u> ") and p.endswith(" </u>") for p in prompts)
    shorts, longs = prompts[:8], prompts[8:]
    assert all(" a" not in p for p in shorts)
    assert all(any(f" a{i}" in p for i in range(40)) for p in longs)
    lengths = [_words(p) for p in longs]
    assert lengths == sorted(lengths) and len(longs) == summary["long"]["count"] == 8
    assert 95 <= summary["long"]["min_tokens"] <= 110
    assert summary["short"] == {"count": 8, "min_tokens": 6, "max_tokens": 6}
    assert 190 <= summary["longest_tokens"] <= 200 == summary["prompt_limit_tokens"]
    assert summary["long"]["targets"] == [100, 200] and summary["long"]["reached"] >= 7
    joined = "".join(p + "\n" for p in prompts).encode()
    assert summary["sha256"] == hashlib.sha256(joined).hexdigest()
    again, same = script["calibration_plan"](rows, _render, _words, 200, 8, 8, 20)
    assert again == prompts and same["sha256"] == summary["sha256"]


def test_an_example_too_long_to_pack_is_skipped_not_the_end_of_packing(script):
    _, clean = script["calibration_plan"](_example_rows(), _render, _words, 200, 8, 8, 20)
    rows = _example_rows()
    rows[3]["completion"] = "x " * 500
    _, summary = script["calibration_plan"](rows, _render, _words, 200, 8, 8, 20)
    assert summary["longest_tokens"] >= 180
    assert summary["long"]["reached"] == clean["long"]["reached"]


def test_calibration_does_not_format_the_rendered_prompts_again(script):
    assert script["CALIBRATE_KWARGS"] == {"enable_formatting": False}


def test_a_calibration_set_that_cannot_reach_the_limit_is_refused(script):
    rows = [{"prompt": "w " * 4, "completion": "x " * 300} for _ in range(12)]
    with pytest.raises(script["Refused"], match="must reach within 128"):
        script["calibration_plan"](rows, _render, _words, 896, 4, 4, 128)


def test_a_training_prompt_longer_than_the_limit_is_refused(script):
    rows = [{"prompt": "w " * 300, "completion": "x"}] + [
        {"prompt": "w", "completion": "x"} for _ in range(10)
    ]
    with pytest.raises(script["Refused"], match="raise --cache-length"):
        script["calibration_plan"](rows, _render, _words, 200, 4, 4, 20)


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
    assert script["exporter_kwargs"](["--nothing"]) == {"nothing": True}
    with pytest.raises(script["Refused"]):
        script["exporter_kwargs"](["externalize_embedder"])


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _safetensors(path: Path, shapes: dict[str, list[int]]) -> Path:
    """A safetensors file's header alone: what `checkpoint_parameters` reads."""
    header = json.dumps(
        {"__metadata__": {"format": "pt"}}
        | {k: {"dtype": "F32", "shape": v, "data_offsets": [0, 0]} for k, v in shapes.items()}
    ).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header)
    return path


def _checkpoint(
    tmp_path: Path, base=GEMMA, model_type="gemma3_text", shapes=None, **record
) -> Path:
    model = tmp_path / "tuned"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    if shapes is not False:
        family = {"gemma3_text": "gemma-3-text", "qwen2": "qwen-2.5"}.get(model_type)
        default = {"w": [llm_npu.MEASURED_PARAMETERS[family]]} if family else {"w": [1]}
        _safetensors(model / "model.safetensors", shapes or default)
    (model / "litetune.json").write_text(
        json.dumps({"base_model": base, "prompt_mode": "runtime_rendered", **record})
    )
    return model


CALIBRATION = {
    "count": 64,
    "short": {"count": 32, "min_tokens": 20, "max_tokens": 60},
    "long": {
        "count": 32,
        "min_tokens": 448,
        "max_tokens": 890,
        "targets": [448, 896],
        "reached": 31,
    },
    "shortest_tokens": 20,
    "longest_tokens": 890,
    "prompt_limit_tokens": 896,
    "sha256": "ab" * 32,
    "decode": {
        "count": 64, "cap": 128, "min_steps": 128, "median_steps": 128, "max_steps": 128,
        "full": 64, "longest_position": 1018,
    },
}
KV_OK = {"prefill_128": {"INT16": 36}, "decode": {"INT16": 36}}


def _soc_entry(packed: Path, soc: str, kv_types=None) -> dict[str, Any]:
    return {
        "packed": str(packed),
        "kv_types": kv_types or dict(KV_OK),
        "ops": {"prefill_128": {"DISPATCH_OP": 1}, "decode": {"DISPATCH_OP": 1}},
        "stamp": ["Qualcomm", soc],
        "qairt": "2.50.0.260828",
        "metadata": {"LlmMetadataProto": "cd" * 32},
    }


def _good_answer(work: Path, socs=("SM8750",)) -> dict[str, Any]:
    entries = {}
    for soc in socs:
        packed = work / f"{soc}.litertlm"
        packed.write_bytes(f"npu bundle {soc}".encode())
        entries[soc] = _soc_entry(packed, soc)
    return {
        "ok": True,
        "calibration": CALIBRATION,
        "socs": entries,
        "versions": {"litert-torch-nightly": "0.10.0.dev20260926"},
        "stage_s": {"stage1": 1.0},
        "dead_kv": {"changed": [{"input": "kv_cache_k_17"}]},
    }


@dataclass
class FakeRun:
    """`StageEnv.run`: answers the NPU script with `make(spec)` (a good answer by default),
    and the chat-template script as the other toolchain fakes do."""

    make: Any = None
    returncode: int = 0
    raises: BaseException | None = None
    chat_template: object = "{%- for m in messages %}{{ m.content }}{% endfor %}"
    specs: list[dict[str, Any]] = field(default_factory=list)
    timeouts: list[int] = field(default_factory=list)

    def __call__(self, args, timeout: int = 0, **kwargs) -> subprocess.CompletedProcess:
        if str(args[1]).endswith(("compose_template.py", "conversation_template.py")):
            return run_compose_script(args, timeout, self.chat_template)[0]
        assert str(args[1]).endswith("npu_llm.py"), args
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        self.timeouts.append(timeout)
        if self.raises is not None:
            raise self.raises
        answer = (self.make or (lambda s: _good_answer(Path(s["work"]), s["socs"])))(spec)
        if isinstance(answer, str):
            Path(spec["answer"]).write_text(answer, encoding="utf-8")
        elif answer is not None:
            Path(spec["answer"]).write_text(json.dumps(answer), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, "", "Traceback: boom")


def _with_kv(kv_types):
    def make(spec):
        answer = _good_answer(Path(spec["work"]), spec["socs"])
        for entry in answer["socs"].values():
            entry["kv_types"] = kv_types
        return answer

    return make


@pytest.fixture
def ready_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(envs.StageEnv, "ready", property(lambda self: True))
    monkeypatch.setattr(envs.StageEnv, "refusal", lambda self: None)
    return envs.NPU_LLM


def _run(tmp_path, model, socs=("SM8750",), fake=None, monkeypatch=None, **kwargs):
    if fake is not None:
        monkeypatch.setattr(envs.StageEnv, "run", fake)
    kwargs.setdefault("auto_provision", False)
    request = LlmNpuRequest(
        model=str(model),
        output_dir=tmp_path / "out",
        socs=tuple(socs),
        calibration_data=kwargs.pop("data", None) or _rows(tmp_path / "train.jsonl"),
        **kwargs,
    )
    return run_llm_npu_export(request, EventStream(echo_json=False), host="")


def _check(result, name):
    (check,) = [c for c in result.checks.checks if c.name == name]
    return check


def test_a_bundle_is_published_with_its_report(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    result = _run(tmp_path, _checkpoint(tmp_path), ("SM8750", "SM8650"), fake, monkeypatch)
    assert result.outcome is Outcome.PASSED, result.as_dict()["checks"]
    spec = fake.specs[0]
    assert spec["flags"] == ["--litert_lm_model_type_override=gemma3", "--externalize_embedder"]
    assert spec["chat_template"] is None
    assert spec["socs"] == ["SM8750", "SM8650"] and spec["prompt_limit_tokens"] == 896
    assert spec["prompt_mode"] == "runtime_rendered" and spec["prefill_length"] == 128
    assert spec["short"] == spec["long"] == 32 and spec["decode_steps"] == 128
    assert spec["model"] == str(tmp_path / "tuned")
    assert fake.timeouts == [llm_npu.NPU_LLM_TIMEOUT_S]
    for soc in ("SM8750", "SM8650"):
        model = tmp_path / "out" / f"npu-{soc}" / "model.litertlm"
        report = json.loads((model.parent / REPORT_NAME).read_text())
        assert model.read_bytes() == f"npu bundle {soc}".encode()
        assert report["schema"] == LLM_NPU_SCHEMA and report["kind"] == "npu"
        assert report["generation_only"] is True and report["verified"] is False
        assert report["longest_calibrated_tokens"] == 890 and report["calibration"] == CALIBRATION
        assert report["family_measured"] == FAMILIES["gemma-3-text"]
        assert report["dead_kv"] == {"changed": [{"input": "kv_cache_k_17"}]}
        assert report["request"]["cache_length"] == 1024
        assert NOT_VERIFIED in report["limitations"]
        assert any("longer than 890 tokens" in t for t in report["limitations"])
        assert report["npu"]["stamp"] == ["Qualcomm", soc] and report["npu"]["qairt"]
        assert f"sha256:{report['npu']['sha256']}" == hash_file(model)
        assert npu_target_of(model) == soc
    sm8650 = json.loads((tmp_path / "out" / "npu-SM8650" / REPORT_NAME).read_text())
    assert any(t.startswith("SM8650: compiled, never run") for t in sm8650["limitations"])
    assert sm8650["npu"]["device_run"] is None
    run_report = json.loads((tmp_path / "out" / RUN_REPORT_NAME).read_text())
    assert run_report["outcome"] == "passed" and len(run_report["npu"]) == 2
    assert {c["name"] for c in run_report["checks"]["checks"]} >= {"npu compile", "npu publish"}
    assert not list((tmp_path / "out").glob(".litetune-npu-llm-*"))
    lines = summarise(result)
    assert any("calibrated on 64 prompts of 20-890 tokens" in line for line in lines)


def test_a_composed_template_reaches_the_exporter_as_a_file(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    model = _checkpoint(tmp_path, base=QWEN25, model_type="qwen2")
    result = _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.PASSED, result.as_dict()["checks"]
    spec = fake.specs[0]
    (override,) = [f for f in spec["flags"] if f.startswith("--jinja_chat_template_override=")]
    path = Path(override.split("=", 1)[1])
    assert path.is_file() and "<" not in str(path) and spec["chat_template"] == str(path)
    assert result.as_dict()["chat_template"]["path"] == str(path)


def test_no_composed_template_compiles_nothing(monkeypatch, tmp_path, ready_env):
    import litetune.export as export

    fake = FakeRun()
    monkeypatch.setattr(
        export,
        "compose_chat_template",
        lambda request: (Check.unchecked("chat template", "no"), None),
    )
    model = _checkpoint(tmp_path, base=QWEN25, model_type="qwen2")
    result = _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert "no NPU bundle was compiled: no chat template" in result.limitations


def test_the_cache_length_reaches_the_script_and_an_unmeasured_one_is_a_limitation(
    monkeypatch, tmp_path, ready_env
):
    fake = FakeRun()
    result = _run(
        tmp_path,
        _checkpoint(tmp_path),
        fake=fake,
        monkeypatch=monkeypatch,
        cache_length=512,
        timeout_s=77,
    )
    assert fake.specs[0]["cache_length"] == 512 and fake.specs[0]["prompt_limit_tokens"] == 384
    assert fake.timeouts == [77]
    assert any("cache_length 512 has not been measured" in t for t in result.limitations)


def test_a_prerendered_checkpoint_hands_its_mode_to_the_script(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    model = _checkpoint(tmp_path)
    record = json.loads((model / "litetune.json").read_text())
    (model / "litetune.json").write_text(json.dumps({**record, "prompt_mode": "prerendered"}))
    _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert fake.specs[0]["prompt_mode"] == "prerendered"


def test_a_compiled_bundle_with_a_float_kv_input_is_not_published(monkeypatch, tmp_path, ready_env):
    kv = {"prefill_128": {"INT16": 34, "FLOAT32": 2}, "decode": {"INT16": 36}}
    result = _run(
        tmp_path, _checkpoint(tmp_path), fake=FakeRun(make=_with_kv(kv)), monkeypatch=monkeypatch
    )
    assert result.outcome is Outcome.FAILED
    check = _check(result, llm_npu.NPU_COMPILE_CHECK)
    assert "FLOAT32" in check.detail and check.detail.startswith("compiled:")
    assert not (tmp_path / "out" / "npu-SM8750").exists()


@pytest.mark.parametrize(
    "cause, outcome",
    [
        ("data", Outcome.FAILED),
        ("partitioned", Outcome.FAILED),
        ("run", Outcome.UNCHECKED),
        ("script", Outcome.UNCHECKED),
    ],
)
def test_a_script_refusal_keeps_its_cause_and_its_grade(
    monkeypatch, tmp_path, ready_env, cause, outcome
):
    fake = FakeRun(make=lambda s: {"ok": False, "cause": cause, "reason": "why"}, returncode=3)
    result = _run(tmp_path, _checkpoint(tmp_path), fake=fake, monkeypatch=monkeypatch)
    check = _check(result, llm_npu.NPU_COMPILE_CHECK)
    assert check.detail == f"{cause}: why" and check.outcome is outcome


def _without(key):
    def make(spec):
        answer = _good_answer(Path(spec["work"]), spec["socs"])
        del answer[key]
        return answer

    return make


def _soc_replaced(entry_for):
    def make(spec):
        answer = _good_answer(Path(spec["work"]), spec["socs"])
        answer["socs"] = {"SM8750": entry_for(Path(spec["work"]))}
        return answer

    return make


@pytest.mark.parametrize(
    "fake, said",
    [
        (FakeRun(make=lambda s: None), "the export exited 0"),
        (FakeRun(make=lambda s: "{"), "could not be read"),
        (FakeRun(make=lambda s: "[1]"), "not an object"),
        (FakeRun(raises=subprocess.TimeoutExpired("python", 5)), "did not finish"),
        (FakeRun(raises=OSError("no python")), "could not be started"),
        (FakeRun(returncode=1), "answered ok but"),
        (FakeRun(make=_without("versions")), "has no versions"),
        (FakeRun(make=lambda s: _good_answer(Path(s["work"]), ("SM8650",))), "names SoCs"),
        (FakeRun(make=_soc_replaced(lambda w: {"packed": str(w / "gone")})), "has no kv_types"),
        (FakeRun(make=_soc_replaced(lambda w: _soc_entry(w / "gone", "SM8750"))), "not there"),
    ],
)
def test_every_answer_guard_is_a_run_failure(monkeypatch, tmp_path, ready_env, fake, said):
    result = _run(tmp_path, _checkpoint(tmp_path), fake=fake, monkeypatch=monkeypatch)
    check = _check(result, llm_npu.NPU_COMPILE_CHECK)
    assert check.outcome is Outcome.UNCHECKED and check.detail.startswith("run:")
    assert said in check.detail
    assert not (tmp_path / "out" / "npu-SM8750").exists()


def _link_failing_for(target_part, error):
    real_link = os.link

    def link(src, dst):
        if target_part in str(dst):
            raise error
        return real_link(src, dst)

    return link


def test_a_failed_link_publishes_what_reached_the_disk(monkeypatch, tmp_path, ready_env):
    monkeypatch.setattr(llm_npu.os, "link", _link_failing_for("npu-SM8650", OSError("disk full")))
    result = _run(tmp_path, _checkpoint(tmp_path), ("SM8750", "SM8650"), FakeRun(), monkeypatch)
    assert [b.soc for b in result.bundles] == ["SM8750"]
    assert result.partial == [str(tmp_path / "out" / "npu-SM8650")]
    publish = _check(result, llm_npu.NPU_PUBLISH_CHECK)
    assert publish.outcome is Outcome.UNCHECKED and "published 1 of 2" in publish.detail
    assert any("Run the rest with another --output-dir" in t for t in result.limitations)
    run_report = json.loads((tmp_path / "out" / RUN_REPORT_NAME).read_text())
    assert run_report["partial"] == result.partial


def test_a_report_link_failing_after_the_model_leaves_a_named_partial(
    monkeypatch, tmp_path, ready_env
):
    error = OSError("disk full")
    monkeypatch.setattr(llm_npu.os, "link", _link_failing_for("SM8750/convert.json", error))
    result = _run(tmp_path, _checkpoint(tmp_path), fake=FakeRun(), monkeypatch=monkeypatch)
    assert result.bundles == [] and result.partial == [str(tmp_path / "out" / "npu-SM8750")]
    assert _check(result, llm_npu.NPU_PUBLISH_CHECK).outcome is Outcome.UNCHECKED


def test_an_interrupt_while_publishing_is_recorded_and_raised(monkeypatch, tmp_path, ready_env):
    monkeypatch.setattr(llm_npu.os, "link", _link_failing_for("npu-SM8650", KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _checkpoint(tmp_path), ("SM8750", "SM8650"), FakeRun(), monkeypatch)
    run_report = json.loads((tmp_path / "out" / RUN_REPORT_NAME).read_text())
    assert [n["soc"] for n in run_report["npu"]] == ["SM8750"]
    assert run_report["partial"] == [str(tmp_path / "out" / "npu-SM8650")]


@pytest.mark.parametrize("existing", ["npu-SM8650", RUN_REPORT_NAME])
def test_every_output_is_checked_before_anything_runs(monkeypatch, tmp_path, ready_env, existing):
    fake = FakeRun()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / existing).mkdir()
    result = _run(tmp_path, _checkpoint(tmp_path), ("SM8750", "SM8650"), fake, monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert "already exists" in result.checks.checks[0].detail


@pytest.mark.parametrize(
    "record, said",
    [
        ({"decision": {"options": ["a", "b"]}}, "rolling back one token"),
        ({"decision": ["a"]}, "cannot be read"),
        ({"declarations_sha256": "ab" * 32}, "tool declarations"),
    ],
)
def test_a_decision_or_tool_checkpoint_is_refused_before_anything_runs(
    monkeypatch, tmp_path, ready_env, record, said
):
    fake = FakeRun()
    result = _run(tmp_path, _checkpoint(tmp_path, **record), fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert said in result.checks.checks[0].detail


@pytest.mark.parametrize("base", ["ibm-granite/granite-4.1-3b", "Qwen/Qwen3-1.7B"])
def test_an_unmeasured_family_or_size_is_refused_before_anything_runs(
    monkeypatch, tmp_path, ready_env, base
):
    # No safetensors to count: the size is read from the name.
    fake = FakeRun()
    model = _checkpoint(tmp_path, base=base, shapes=False)
    result = _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []


@pytest.mark.parametrize("offset", [-1, 1])
def test_a_count_outside_the_band_is_refused(monkeypatch, tmp_path, ready_env, offset):
    # Another size's weights under a 270m name: refused by what they are.
    low, high = llm_npu.parameter_band("gemma-3-text")
    count = low - 1 if offset < 0 else high + 1
    fake = FakeRun()
    model = _checkpoint(tmp_path, shapes={"a": [count // 2], "b": [count - count // 2]})
    result = _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert f"{count:,} parameters" in result.checks.checks[0].detail


@pytest.mark.parametrize("edge", ["low", "high"])
def test_a_count_at_the_bands_edge_is_accepted_and_reported(
    monkeypatch, tmp_path, ready_env, edge
):
    low, high = llm_npu.parameter_band("gemma-3-text")
    count = low if edge == "low" else high
    model = _checkpoint(tmp_path, shapes={"a": [count]})
    result = _run(tmp_path, model, fake=FakeRun(), monkeypatch=monkeypatch)
    assert result.outcome is Outcome.PASSED
    assert result.as_dict()["parameters"] == count


def test_the_band_is_one_percent_around_each_measured_count():
    for family, measured in llm_npu.MEASURED_PARAMETERS.items():
        low, high = llm_npu.parameter_band(family)
        assert low <= measured * 0.99 + 1 and high >= measured * 1.01 - 1
        assert high - low <= measured * 0.02 + 2


def test_a_stored_output_layer_does_not_count(tmp_path):
    # Qwen3-0.6B's own file: a tied lm_head stored anyway.
    _safetensors(tmp_path / "model.safetensors",
                 {"model.layers": [596_049_920], "lm_head.weight": [151_936, 1_024]})
    assert llm_npu.checkpoint_parameters(tmp_path) == llm_npu.MEASURED_PARAMETERS["qwen-3"]
    assert llm_npu.family_problem("qwen-3", "Qwen/Qwen3-0.6B", 596_049_920) is None


@pytest.mark.parametrize(
    "header",
    [b"[]", b'"x"', b'{"w": []}', b'{"w": {"shape": "640"}}', b'{"w": {"shape": [2, -1]}}',
     b'{"w": {"shape": [true]}}', b'{"w": {}}'],
)
def test_a_header_that_is_not_safetensors_is_could_not_check(
    monkeypatch, tmp_path, ready_env, header
):
    model = _checkpoint(tmp_path)
    (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    fake = FakeRun()
    result = _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert "safetensors header cannot be read" in result.checks.checks[0].detail
    assert not any("no safetensors" in t for t in result.limitations)


def test_a_checkpoint_without_safetensors_is_sized_by_name_and_says_so(
    monkeypatch, tmp_path, ready_env
):
    model = _checkpoint(tmp_path, shapes=False)
    result = _run(tmp_path, model, fake=FakeRun(), monkeypatch=monkeypatch)
    assert result.outcome is Outcome.PASSED and result.as_dict()["parameters"] is None
    assert any("no safetensors" in t and GEMMA in t for t in result.limitations)


def test_an_unreadable_safetensors_header_is_could_not_check(monkeypatch, tmp_path, ready_env):
    model = _checkpoint(tmp_path)
    (model / "model.safetensors").write_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00{nope")
    fake = FakeRun()
    result = _run(tmp_path, model, fake=fake, monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert "safetensors header cannot be read" in result.checks.checks[0].detail


def test_checkpoint_parameters_sums_every_shard(tmp_path):
    _safetensors(tmp_path / "model-00001-of-00002.safetensors", {"a": [3, 4]})
    _safetensors(tmp_path / "model-00002-of-00002.safetensors", {"b": [5], "c": []})
    assert llm_npu.checkpoint_parameters(tmp_path) == 18
    assert llm_npu.checkpoint_parameters(tmp_path / "missing") is None
    (tmp_path / "huge.safetensors").write_bytes((200 << 20).to_bytes(8, "little"))
    with pytest.raises(ValueError, match="huge.safetensors.*-byte header"):
        llm_npu.checkpoint_parameters(tmp_path)


def test_malformed_calibration_data_fails_and_too_few_rows_cannot_check(
    monkeypatch, tmp_path, ready_env
):
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json\n")
    result = _run(tmp_path, _checkpoint(tmp_path), fake=FakeRun(), monkeypatch=monkeypatch,
                  data=broken)
    assert result.outcome is Outcome.FAILED
    assert _check(result, llm_npu.NPU_DATA_CHECK).outcome is Outcome.FAILED
    few = _rows(tmp_path / "few.jsonl", n=7)
    assert not (tmp_path / "out").exists()
    model = tmp_path / "tuned"
    result = _run(tmp_path, model, fake=FakeRun(), monkeypatch=monkeypatch, data=few)
    assert result.outcome is Outcome.UNCHECKED
    missing = _run(tmp_path, model, fake=FakeRun(), monkeypatch=monkeypatch,
                   data=tmp_path / "missing.jsonl")
    assert missing.outcome is Outcome.UNCHECKED


def test_a_checkpoint_without_a_prompt_mode_is_refused(monkeypatch, tmp_path, ready_env):
    model = _checkpoint(tmp_path)
    (model / "litetune.json").write_text(json.dumps({"base_model": GEMMA}))
    result = _run(tmp_path, model, fake=FakeRun(), monkeypatch=monkeypatch)
    assert result.outcome is Outcome.UNCHECKED
    assert "records no prompt mode" in result.checks.checks[0].detail


def test_an_identity_conflict_is_refused(monkeypatch, tmp_path, ready_env):
    fake = FakeRun()
    result = _run(
        tmp_path, _checkpoint(tmp_path), fake=fake, monkeypatch=monkeypatch, base_model=QWEN25
    )
    assert result.outcome is Outcome.UNCHECKED and fake.specs == []
    assert "different families" in result.checks.checks[0].detail


def test_off_linux_x86_64_it_cannot_check(tmp_path):
    request = LlmNpuRequest(
        model=str(tmp_path),
        output_dir=tmp_path / "out",
        socs=("SM8750",),
        calibration_data=tmp_path / "x.jsonl",
        auto_provision=False,
    )
    result = run_llm_npu_export(request, EventStream(echo_json=False), host="Linux x86-64 only")
    assert result.outcome is Outcome.UNCHECKED
    assert result.checks.checks[0].detail == "Linux x86-64 only"


def test_an_environment_that_fails_to_provision_cannot_check(monkeypatch, tmp_path, ready_env):
    def boom(self, events=None, force=False):
        raise RuntimeError("pip failed")

    monkeypatch.setattr(envs.StageEnv, "provision", boom)
    fake = FakeRun()
    result = _run(
        tmp_path, _checkpoint(tmp_path), fake=fake, monkeypatch=monkeypatch, auto_provision=True
    )
    assert _check(result, llm_npu.NPU_ENV_CHECK).outcome is Outcome.UNCHECKED
    assert fake.specs == []


def test_an_environment_that_is_not_ready_cannot_check(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    monkeypatch.setattr(envs.StageEnv, "refusal", lambda self: None)
    fake = FakeRun()
    result = _run(tmp_path, _checkpoint(tmp_path), fake=fake, monkeypatch=monkeypatch)
    assert _check(result, llm_npu.NPU_ENV_CHECK).outcome is Outcome.UNCHECKED
    assert fake.specs == []


def test_published_whole_needs_the_model_and_a_report_naming_it(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"bundle")
    sha = hash_file(model).removeprefix("sha256:")
    (tmp_path / REPORT_NAME).write_text(json.dumps({"npu": {"sha256": sha}}))
    assert llm_npu._published_whole(model)
    (tmp_path / REPORT_NAME).write_text(json.dumps({"npu": {"sha256": "00" * 32}}))
    assert not llm_npu._published_whole(model)
    (tmp_path / REPORT_NAME).write_text(json.dumps({"npu": {"sha256": sha}}))
    model.unlink()
    assert not llm_npu._published_whole(model)


def test_npu_target_of_reads_only_its_own_report_for_this_file(tmp_path):
    model = tmp_path / "model.litertlm"
    assert npu_target_of(model) is None
    model.write_bytes(b"bundle")
    report = {
        "schema": LLM_NPU_SCHEMA,
        "kind": "npu",
        "npu": {"soc": "SM8750", "sha256": hash_file(model).removeprefix("sha256:")},
    }
    (tmp_path / REPORT_NAME).write_text(json.dumps(report))
    assert npu_target_of(model) == "SM8750"
    for broken in (
        {**report, "kind": "graft"},
        {**report, "npu": {**report["npu"], "soc": 7}},
        {**report, "schema": "other"},
    ):
        (tmp_path / REPORT_NAME).write_text(json.dumps(broken))
        assert npu_target_of(model) is None
    (tmp_path / REPORT_NAME).write_text(json.dumps(report))
    model.write_bytes(b"another bundle")
    assert npu_target_of(model) is None


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_llm_npu_without_calibration_data_is_refused(tmp_path, capsys):
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", "--npu", "SM8750") == 4
    assert "--calibration-data" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [
        ("--recipe", "dynamic_wi8_afp32"),
        ("--flag=--prefill_lengths=128",),
        ("--externalize-embedder",),
        ("--weights", "int8"),
        ("--base-litertlm", "x.litertlm"),
    ],
)
def test_llm_npu_refuses_the_flags_it_does_not_read(tmp_path, capsys, extra):
    data = _rows(tmp_path / "train.jsonl")
    args = ("--npu", "SM8750", "--calibration-data", str(data), *extra)
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", *args) == 4
    assert extra[0].split("=")[0] in capsys.readouterr().err


def test_calibration_flags_without_npu_are_refused(tmp_path, capsys):
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", "--cache-length", "1024") == 4
    assert "--calibration-data and --cache-length are for --npu" in capsys.readouterr().err


@pytest.mark.parametrize(
    "outcome, code", [(Outcome.PASSED, 0), (Outcome.FAILED, 1), (Outcome.UNCHECKED, 4)]
)
def test_the_command_line_maps_onto_the_request(monkeypatch, tmp_path, outcome, code):
    seen = {}

    def fake_run(request, events=None):
        seen["request"] = request
        result = llm_npu.LlmNpuResult(request=request)
        result.checks.add(Check(llm_npu.NPU_REQUEST_CHECK, outcome, "x"))
        return result

    monkeypatch.setattr(cli, "run_llm_npu_export", fake_run)
    data = _rows(tmp_path / "train.jsonl")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"base_model": GEMMA}))
    args = (
        "--npu", "SM8750", "--npu", "SM8650", "--calibration-data", str(data),
        "--cache-length", "512", "--timeout-s", "77", "--no-provision",
        "--train-metrics", str(metrics),
    )  # fmt: skip
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", *args) == code
    request = seen["request"]
    assert request.socs == ("SM8750", "SM8650") and request.cache_length == 512
    assert request.timeout_s == 77 and request.auto_provision is False
    assert request.base_model == GEMMA and request.calibration_data == data


def test_verify_refuses_an_llm_npu_bundle_before_routing_by_the_reference(tmp_path):
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
    reference = _checkpoint(tmp_path, decision={"options": ["a", "b"]})
    result = run_verify(
        VerifyRequest(model=model, reference=str(reference), data=data),
        EventStream(echo_json=False),
    )
    assert result.status is Status.FAILED_HARNESS
    assert "Qualcomm SM8750 NPU" in json.dumps(result.manifest)


def _calibration_replaced(decode):
    def make(spec):
        answer = _good_answer(Path(spec["work"]), spec["socs"])
        answer["calibration"] = {**CALIBRATION, "decode": decode}
        if decode is None:
            del answer["calibration"]["decode"]
        return answer

    return make


def test_replies_short_of_the_decode_cap_are_a_limitation_in_every_report(
    monkeypatch, tmp_path, ready_env
):
    short = {**CALIBRATION["decode"], "median_steps": 9, "max_steps": 128, "full": 3}
    result = _run(
        tmp_path, _checkpoint(tmp_path), fake=FakeRun(make=_calibration_replaced(short)),
        monkeypatch=monkeypatch,
    )
    assert result.outcome is Outcome.PASSED
    report = json.loads((tmp_path / "out" / "npu-SM8750" / REPORT_NAME).read_text())
    (line,) = [t for t in report["limitations"] if "decode steps" in t]
    assert "9 decode steps at the median" in line and "3 of 64" in line
    assert report["calibration"]["decode"]["full"] == 3


def test_replies_that_all_reach_the_cap_need_no_decode_limitation(
    monkeypatch, tmp_path, ready_env
):
    result = _run(tmp_path, _checkpoint(tmp_path), fake=FakeRun(), monkeypatch=monkeypatch)
    assert not any("decode steps" in t for t in result.limitations)


def test_unrecorded_decode_steps_are_a_limitation(monkeypatch, tmp_path, ready_env):
    result = _run(
        tmp_path, _checkpoint(tmp_path), fake=FakeRun(make=_calibration_replaced(None)),
        monkeypatch=monkeypatch,
    )
    assert any("was not recorded" in t for t in result.limitations)


def test_a_bundle_naming_no_qairt_build_fails(monkeypatch, tmp_path, ready_env):
    def entry(work):
        packed = work / "SM8750.litertlm"
        packed.write_bytes(b"npu bundle")
        return {**_soc_entry(packed, "SM8750"), "qairt": None}

    result = _run(
        tmp_path, _checkpoint(tmp_path), fake=FakeRun(make=_soc_replaced(entry)),
        monkeypatch=monkeypatch,
    )
    assert result.outcome is Outcome.FAILED
    assert "no QAIRT build" in _check(result, llm_npu.NPU_COMPILE_CHECK).detail
    assert not (tmp_path / "out" / "npu-SM8750").exists()


def test_the_aux_graphs_ops_reach_the_report(monkeypatch, tmp_path, ready_env):
    aux = {"ops": {"prefill_mask_128": {"DISPATCH_OP": 1}, "decode_mask": {"DISPATCH_OP": 1}}}

    def entry(work):
        packed = work / "SM8750.litertlm"
        packed.write_bytes(b"npu bundle")
        return {**_soc_entry(packed, "SM8750"), "aux": aux}

    _run(tmp_path, _checkpoint(tmp_path), fake=FakeRun(make=_soc_replaced(entry)),
         monkeypatch=monkeypatch)
    report = json.loads((tmp_path / "out" / "npu-SM8750" / REPORT_NAME).read_text())
    assert report["npu"]["aux_ops_per_signature"] == aux["ops"]


def _canonical_template(name):
    jinja2 = pytest.importorskip("jinja2")
    source = (Path(llm_npu.__file__).parent / "templates" / name).read_text(encoding="utf-8")
    template = jinja2.Environment().from_string(source)

    def apply(messages, tokenize=False, add_generation_prompt=True):
        return template.render(messages=messages, add_generation_prompt=add_generation_prompt)

    return apply


@pytest.mark.parametrize(
    "name", ["litert-lm-v0.18.0-qwen3.jinja", "litert-lm-v0.18.0-minicpm5-litetune.jinja"]
)
def test_the_bundle_template_gets_text_parts_and_would_drop_a_string(script, name):
    apply = _canonical_template(name)
    text = "Classify the customer's banking message."
    with_parts = script["make_render"](apply, "runtime_rendered", parts=True)
    assert text in with_parts(text) and script["render_problem"](with_parts, text) is None
    # Given a string, the template iterates its characters and renders an empty turn.
    with_string = script["make_render"](apply, "runtime_rendered", parts=False)
    assert text not in with_string(text)
    assert "without its text" in script["render_problem"](with_string, text)


def test_the_checkpoints_own_template_gets_a_string_and_prerendered_is_left_alone(script):
    seen = []

    def apply(messages, tokenize, add_generation_prompt):
        seen.append(messages[0]["content"])
        return f"<user>{messages[0]['content']}</user>"

    assert script["make_render"](apply, "runtime_rendered", parts=False)("hi") == "<user>hi</user>"
    assert seen == ["hi"]
    assert script["make_render"](apply, "prerendered", parts=True)("<raw>") == "<raw>"
    assert seen == ["hi"]


def test_packing_ends_and_refuses_when_the_template_drops_the_text(script):
    calls = []

    def count(text):
        calls.append(text)
        return 12  # an empty turn: the same length whatever is packed

    with pytest.raises(script["Refused"], match="could not be packed that far"):
        script["calibration_plan"](_example_rows(), lambda t: "<empty turn>", count, 200, 8, 8, 20)
    assert len(calls) < 10_000


def test_every_family_carries_what_its_replies_held_to_on_a_device():
    assert set(llm_npu.REPLY_MEASURED) == set(FAMILIES)
    assert "classification-length answers only" in llm_npu.REPLY_MEASURED["gemma-3-text"]


def test_the_measured_reply_length_reaches_every_socs_report(monkeypatch, tmp_path, ready_env):
    result = _run(tmp_path, _checkpoint(tmp_path), ("SM8750", "SM8650"), FakeRun(), monkeypatch)
    assert result.outcome is Outcome.PASSED
    for soc in ("SM8750", "SM8650"):
        report = json.loads((tmp_path / "out" / f"npu-{soc}" / REPORT_NAME).read_text())
        assert llm_npu.REPLY_MEASURED["gemma-3-text"] in report["limitations"]
