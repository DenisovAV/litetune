# ruff: noqa: F811 -- the fixtures imported from test_embedding_export are parameters here
"""`convert --npu`: the refusals, the request, the compile's answer as `compile_for_npu`
reads it, the NPU script's own decisions, and the convert path around them.

Nothing is compiled here: the compiler runs on Linux x86-64 only and downloads QAIRT.
`FakeNpuRun` stands in for `StageEnv.run` on the NPU script and writes the answer the
script writes; `FakeCompile` stands in for `compile_for_npu` in the convert path. The
script itself (`_NPU_SCRIPT`) is run with stand-ins for the modules it imports
(`script`), and its decision functions are called for real.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from test_embedding_export import _checkpoint, _convert, grafter, hub  # noqa: F401

from litetune import embedding_export, embedding_npu, envs, wheels
from litetune.checks import Outcome
from litetune.embedding_export import (
    NPU_NOT_VERIFIED,
    REPORT_NAME,
    EmbeddingExportError,
    EmbeddingExportRequest,
)
from litetune.embedding_npu import (
    ANSWER_KEYS,
    DEVICE_RUNS,
    NPU_SIGNATURES,
    NPU_SOCS,
    NPU_TIMEOUT_S,
    STAMP_BYTES,
    EmbeddingNpuError,
    NpuResult,
    compile_for_npu,
    host_problem,
    output_for,
    request_problem,
)
from litetune.embedding_verify import npu_target_of

# ---------------------------------------------------------------------------
# What is refused before anything runs
# ---------------------------------------------------------------------------


def test_int4_is_refused_with_the_measured_reason():
    problem = request_problem(("SM8750",), "int4")
    assert problem is not None and "--weights int8" in problem and "194 partitions" in problem


def test_int8_with_known_socs_is_accepted():
    assert request_problem(("SM8750", "SM8650"), "int8") is None
    assert request_problem((), "int4") is None


@pytest.mark.parametrize("soc", ["SM9999", "SM8450", "SM8475", "SM8350", "SM8845"])
def test_an_unknown_or_out_of_scope_soc_is_refused(soc):
    assert "one of" in (request_problem((soc,), "int8") or "")


def test_a_repeated_soc_is_refused():
    assert "twice" in (request_problem(("SM8750", "SM8750"), "int8") or "")


def test_the_socs_are_the_four_google_publishes_files_for():
    assert NPU_SOCS == ("SM8550", "SM8650", "SM8750", "SM8850")
    assert set(DEVICE_RUNS) == {"SM8750"}


@pytest.mark.parametrize(
    "system, machine, ok",
    [
        ("linux", "x86_64", True),
        ("linux", "AMD64", True),
        ("linux", "aarch64", False),
        ("darwin", "arm64", False),
        ("win32", "AMD64", False),
    ],
)
def test_only_linux_x86_64_compiles(system, machine, ok):
    assert (host_problem(system, machine) is None) is ok


@pytest.mark.parametrize("system, ok", [("linux", True), ("darwin", False)])
def test_the_host_is_read_from_this_interpreter_by_default(monkeypatch, system, ok):
    monkeypatch.setattr(sys, "platform", system)
    monkeypatch.setattr(embedding_npu.platform, "machine", lambda: "x86_64")
    assert (host_problem() is None) is ok


def test_the_request_refuses_int4_with_npu(tmp_path):
    with pytest.raises(EmbeddingExportError, match="--weights int8"):
        EmbeddingExportRequest(model=tmp_path, output_dir=tmp_path, weights="int4", npu=("SM8750",))


def test_the_request_refuses_a_zero_npu_timeout(tmp_path):
    with pytest.raises(EmbeddingExportError, match="npu_timeout_s"):
        EmbeddingExportRequest(
            model=tmp_path, output_dir=tmp_path, weights="int8", npu=("SM8750",), npu_timeout_s=0
        )


def test_the_signatures_are_google_npu_files_four():
    assert NPU_SIGNATURES == ("encoder_1x128", "encoder_1x256", "encoder_1x512", "encoder_1x1024")


def test_the_npu_script_compiles_as_python():
    compile(embedding_npu._NPU_SCRIPT, "npu.py", "exec")


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def test_the_qualcomm_sdk_is_a_source_build_on_linux_x86_64_only():
    sdk = wheels.PUBLISHED["ai-edge-litert-sdk-qualcomm==2.3.0"]
    assert sdk.sdist and sdk.source_build
    linux = wheels.Host.of(("manylinux_2_27_x86_64", "linux_x86_64"), (3, 12))
    assert wheels.gaps(envs.NPU_QUALCOMM.requirements, linux) == []
    for elsewhere in (("macosx_14_0_arm64",), ("manylinux_2_27_aarch64", "linux_aarch64")):
        host = wheels.Host.of(elsewhere, (3, 12))
        assert sdk.requirement in [
            g.release.requirement for g in wheels.gaps([sdk.requirement], host)
        ]


def test_a_sdist_only_pin_without_a_reason_is_still_a_gap():
    sdk = wheels.PUBLISHED["ai-edge-litert-sdk-qualcomm==2.3.0"]
    table = dict(wheels.PUBLISHED) | {sdk.requirement: replace(sdk, source_build="")}
    linux = wheels.Host.of(("manylinux_2_27_x86_64", "linux_x86_64"), (3, 12))
    found = wheels.gaps([sdk.requirement], linux, table)
    assert [g.release.requirement for g in found] == [sdk.requirement]


def test_the_sdk_is_installed_from_its_exact_file_and_only_it_from_source():
    env = envs.NPU_QUALCOMM
    (sdk,) = [r for r in env.pip_requirements if r.startswith("ai-edge-litert-sdk-qualcomm")]
    assert sdk.endswith(
        f"ai_edge_litert_sdk_qualcomm-2.3.0.tar.gz#sha256={envs.NPU_SDK_SDIST_SHA256}"
    )
    assert "ai-edge-litert-sdk-qualcomm==2.3.0" in env.requirements
    assert {
        "--no-cache-dir",
        "--only-binary=:all:",
        "--no-binary=ai-edge-litert-sdk-qualcomm",
    } <= set(env.pip_options)
    assert dict(env.provision_variables) == {"SKIP_SDK_DOWNLOAD": ""}
    assert any(p.endswith("/libQnnHtp.so") for p in env.installed_files)
    assert any(p.endswith("/libQnnSystem.so") for p in env.installed_files)


def test_the_install_fields_change_the_identity_and_leave_others_alone():
    bare = envs.StageEnv(name="npu-qualcomm", requirements=envs.NPU_QUALCOMM.requirements)
    assert bare.identity != envs.NPU_QUALCOMM.identity
    assert (
        envs.StageEnv(name="x", requirements=("a==1",)).identity
        == envs.StageEnv(name="x", requirements=("a==1",), installed_files=()).identity
    )


def test_the_npu_environment_is_claimed_in_the_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    envs.NPU_QUALCOMM.path.mkdir(parents=True)
    (entry,) = envs.cached_environments()
    assert entry.stage == envs.NPU_QUALCOMM.name


def test_a_provision_whose_files_are_missing_is_not_ready(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    env = envs.StageEnv(
        name="probe",
        requirements=("a==1",),
        installed_files=("lib/x/libQnnHtp.so",),
        provision_variables=(("SKIP_SDK_DOWNLOAD", ""),),
        pip_options=("--no-cache-dir",),
    )
    seen: dict[str, Any] = {}

    class Builder:
        def __init__(self, **kwargs):
            pass

        def create(self, path):
            (Path(path) / "bin").mkdir(parents=True)
            env.python.parent.mkdir(parents=True, exist_ok=True)
            env.python.write_text("")

    def fake_run(cmd, timeout, overrides=None):
        seen["cmd"], seen["overrides"] = cmd, overrides
        return subprocess.CompletedProcess(cmd, 0, "Skipping SDK download...", "")

    monkeypatch.setattr(envs, "_StageEnvBuilder", Builder)
    monkeypatch.setattr(envs, "_run_guarded", fake_run)
    monkeypatch.setattr(
        envs, "_host_verdict", lambda e: types.SimpleNamespace(refused=False, reason=None)
    )
    with pytest.raises(RuntimeError, match="lib/x/libQnnHtp.so matched nothing"):
        env.provision()
    assert not env.ready
    assert "--no-cache-dir" in seen["cmd"] and seen["overrides"] == {"SKIP_SDK_DOWNLOAD": ""}


def test_a_failed_provision_names_the_source_build():
    message = envs.NPU_QUALCOMM._provisioning_failed("ERROR: boom", platform="linux")
    assert "built from source" in message and "QAIRT 2.50.0.260828" in message


# ---------------------------------------------------------------------------
# compile_for_npu reading the script's answer
# ---------------------------------------------------------------------------


def _ok(**overrides: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "ok": True,
        "bytes": 8,
        "sha256": "ab" * 32,
        "compile_s": 233.1,
        "encoder_bytes": 169590480,
        "encoder_sha256": "cd" * 32,
        "partitions": [[0, 1779, 1779, 1]] * 4,
        "ops_per_signature": {k: {"DISPATCH_OP": 1} for k in NPU_SIGNATURES},
        "stamp": ["Qualcomm", "SM8750"],
        "qnn_context": {
            "bytecode_offset": 3792,
            "bytecode_size": 169586688,
            "to_weights": 1.241,
            "gate": "at most 2x the encoder's weight bytes",
        },
        "signatures": {},
        "qairt": "2.50.0.260828221209",
        "compiler": {"ai-edge-litert": "2.3.0"},
    }
    answer.update(overrides)
    return answer


@dataclass
class FakeNpuRun:
    """`StageEnv.run` on the NPU script: writes `answer` (and the packed file it names)."""

    answer: dict[str, Any] | str | None
    returncode: int = 0
    raises: BaseException | None = None
    write_packed: bool = True
    specs: list[dict[str, Any]] = field(default_factory=list)
    timeouts: list[int] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        self.timeouts.append(timeout)
        if self.raises is not None:
            raise self.raises
        if isinstance(self.answer, str):
            Path(spec["answer"]).write_text(self.answer, encoding="utf-8")
        elif self.answer is not None:
            answer = dict(self.answer)
            if answer.get("ok"):
                packed = Path(spec["work"]) / "packed.litertlm"
                if self.write_packed:
                    packed.write_bytes(b"npu file")
                answer.setdefault("packed", str(packed))
            Path(spec["answer"]).write_text(json.dumps(answer), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, "", "Traceback: boom")


@pytest.fixture
def npu_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    return envs.NPU_QUALCOMM


def _compile(tmp_path, npu_env, soc="SM8750", **kwargs):
    out = output_for(tmp_path / "out", soc)
    return out, compile_for_npu(
        npu_env, tmp_path / "g.litertlm", soc, out, auto_provision=False, **kwargs
    )


def test_a_compiled_file_is_moved_to_its_output(monkeypatch, tmp_path, npu_env):
    fake = FakeNpuRun(_ok())
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    out, got = _compile(tmp_path, npu_env, timeout=77)
    assert out.read_bytes() == b"npu file"
    assert got.soc == "SM8750" and got.encoder_bytes == 169590480
    assert got.as_dict()["device_run"] == DEVICE_RUNS["SM8750"]
    spec = fake.specs[0]
    assert spec["soc"] == "SM8750" and spec["signatures"] == list(NPU_SIGNATURES)
    assert spec["stamp_bytes"] == STAMP_BYTES == 250
    assert spec["system_requirements"] == ["libc++1", "libatomic1"]
    assert Path(spec["work"]).parent == out.parent, "the working directory is beside the output"
    assert fake.timeouts == [77]
    assert out == tmp_path / "out" / "int8-npu-SM8750" / "model.litertlm"


def test_the_soc_asked_for_is_the_one_the_script_is_given(monkeypatch, tmp_path, npu_env):
    fake = FakeNpuRun(_ok())
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    _compile(tmp_path, npu_env, soc="SM8650")
    assert fake.specs[0]["soc"] == "SM8650"


def test_every_answer_key_is_recorded(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(_ok()))
    _, got = _compile(tmp_path, npu_env)
    recorded = got.as_dict()
    for key in ("partitions", "ops_per_signature", "stamp", "qnn_context", "qairt", "compiler"):
        assert recorded[key] == _ok()[key], key
    assert recorded["encoder_sha256"] == "cd" * 32


def test_an_ok_answer_from_a_process_that_died_is_not_trusted(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(_ok(), returncode=-9))
    with pytest.raises(EmbeddingNpuError, match="answered ok but") as exc:
        _compile(tmp_path, npu_env)
    assert exc.value.cause == "run"
    assert not output_for(tmp_path / "out", "SM8750").exists()


@pytest.mark.parametrize("key", ANSWER_KEYS[1:])
def test_an_answer_missing_a_key_moves_nothing(monkeypatch, tmp_path, npu_env, key):
    answer = _ok()
    del answer[key]
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(answer))
    with pytest.raises(EmbeddingNpuError, match=key) as exc:
        _compile(tmp_path, npu_env)
    assert exc.value.cause == "run"
    assert not output_for(tmp_path / "out", "SM8750").exists()


def test_an_answer_with_a_bad_value_moves_nothing(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(_ok(bytes="many")))
    with pytest.raises(EmbeddingNpuError, match="could not be read") as exc:
        _compile(tmp_path, npu_env)
    assert exc.value.cause == "run"
    assert not output_for(tmp_path / "out", "SM8750").exists()


@pytest.mark.parametrize("raw", ["not json", "[1, 2]"])
def test_an_unreadable_answer_is_a_run_failure(monkeypatch, tmp_path, npu_env, raw):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(raw))
    with pytest.raises(EmbeddingNpuError) as exc:
        _compile(tmp_path, npu_env)
    assert exc.value.cause == "run"


def test_a_packed_file_that_is_not_there_is_a_run_failure(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(_ok(), write_packed=False))
    with pytest.raises(EmbeddingNpuError, match="not there") as exc:
        _compile(tmp_path, npu_env)
    assert exc.value.cause == "run"


def test_a_refusal_keeps_its_cause(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(
        envs.StageEnv, "run", FakeNpuRun({"ok": False, "cause": "partitioned", "reason": "194"}, 3)
    )
    with pytest.raises(EmbeddingNpuError) as exc:
        _compile(tmp_path, npu_env, soc="SM8650")
    assert exc.value.cause == "partitioned" and not output_for(tmp_path / "out", "SM8650").exists()


def test_a_script_that_died_is_a_run_failure(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(None, 1))
    with pytest.raises(EmbeddingNpuError) as exc:
        _compile(tmp_path, npu_env)
    assert exc.value.cause == "run" and "boom" in str(exc.value)


@pytest.mark.parametrize(
    "raised, said",
    [
        (subprocess.TimeoutExpired("python", 5), "within 5s"),
        (OSError("no python"), "could not be started"),
    ],
)
def test_a_compile_that_cannot_run_is_a_run_failure(monkeypatch, tmp_path, npu_env, raised, said):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(_ok(), raises=raised))
    with pytest.raises(EmbeddingNpuError, match=said) as exc:
        _compile(tmp_path, npu_env, timeout=5)
    assert exc.value.cause == "run"


def test_an_existing_output_is_not_written_over(tmp_path, npu_env):
    out = output_for(tmp_path, "SM8750")
    out.parent.mkdir(parents=True)
    out.write_bytes(b"keep")
    with pytest.raises(EmbeddingNpuError, match="already exists"):
        compile_for_npu(npu_env, tmp_path / "g", "SM8750", out, auto_provision=False)
    assert out.read_bytes() == b"keep"


# ---------------------------------------------------------------------------
# The NPU script, run with stand-ins for what it imports
# ---------------------------------------------------------------------------


class _Buffer:
    def __init__(self):
        self.data = None


def _stub_modules(sdk_libs: Path | None) -> dict[str, types.ModuleType]:
    def module(name: str, **attrs: Any) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        return m

    schema = module(
        "ai_edge_litert.schema_py_generated",
        BufferT=_Buffer,
        BuiltinOptions2=types.SimpleNamespace(StableHLOCompositeOptions=7),
        BuiltinOptions=types.SimpleNamespace(WhileOptions=101, IfOptions=102, CallOnceOptions=103),
    )
    common = module(
        "ai_edge_litert.aot.core.common",
        get_resource=lambda p: Path("/opt/litert") / p,
        construct_ld_library_path=lambda: "/opt/litert/libs",
    )
    core = module("ai_edge_litert.aot.core", common=common)
    aot = module("ai_edge_litert.aot", core=core)
    litert = module("ai_edge_litert", schema_py_generated=schema, aot=aot)
    flex = module("flatbuffers.flexbuffers", Loads=lambda raw: {})
    builder = module("litert_lm_builder.litertlm_builder")
    header = module("litert_lm_builder.litertlm_header_schema_py_generated")
    return {
        "flatbuffers": module("flatbuffers", flexbuffers=flex, Builder=None),
        "flatbuffers.flexbuffers": flex,
        "ai_edge_litert_sdk_qualcomm": module(
            "ai_edge_litert_sdk_qualcomm", path_to_sdk_libs=lambda: sdk_libs
        ),
        "ai_edge_litert": litert,
        "ai_edge_litert.schema_py_generated": schema,
        "ai_edge_litert.aot": aot,
        "ai_edge_litert.aot.core": core,
        "ai_edge_litert.aot.core.common": common,
        "litert_lm_builder": module(
            "litert_lm_builder",
            litertlm_builder=builder,
            litertlm_header_schema_py_generated=header,
        ),
        "litert_lm_builder.litertlm_builder": builder,
        "litert_lm_builder.litertlm_header_schema_py_generated": header,
    }


@pytest.fixture
def script(monkeypatch, tmp_path) -> dict[str, Any]:
    """`_NPU_SCRIPT`'s namespace, `main` not run."""
    for name, stub in _stub_modules(tmp_path / "sdk").items():
        monkeypatch.setitem(sys.modules, name, stub)
    namespace: dict[str, Any] = {"__name__": "npu_script"}
    exec(compile(embedding_npu._NPU_SCRIPT, "npu.py", "exec"), namespace)
    return namespace


SOC = "SM8750"
WEIGHTS = 136_708_096
IO = {
    k: {"inputs": [["embeddings", [1, 128, 512], 0]], "outputs": [["encodings", [1, 768], 0]]}
    for k in NPU_SIGNATURES
}


def _compiled(**overrides: Any) -> dict[str, Any]:
    compiled: dict[str, Any] = {
        "ops": {k: {"DISPATCH_OP": 1} for k in NPU_SIGNATURES},
        "io": json.loads(json.dumps(IO)),
        "dispatch": [
            {
                "signature": k,
                "name": f"qnn_partition_{i}",
                "bytecode_offset": 3792,
                "bytecode_size": 169_586_688,
            }
            for i, k in enumerate(NPU_SIGNATURES)
        ],
        "stamp": b"Qualcomm".ljust(125, b"\0") + SOC.encode().ljust(125, b"\0"),
        "bytes": 169_590_480,
        "qairt": "2.50.0.260828221209",
    }
    compiled.update(overrides)
    return compiled


def _check(script, compiled, stats=None, weight_bytes=WEIGHTS):
    return script["check_compiled"](
        compiled,
        IO,
        list(NPU_SIGNATURES),
        SOC,
        STAMP_BYTES,
        [[0, 1779, 1779, 1]] * 4 if stats is None else stats,
        ["libc++1", "libatomic1"],
        weight_bytes,
    )


def test_a_whole_shared_stamped_compile_passes(script):
    facts = _check(script, _compiled())
    assert facts["stamp"] == ["Qualcomm", SOC]
    context = facts["qnn_context"]
    assert context["bytecode_size"] == 169_586_688
    assert context["functions"] == [f"qnn_partition_{i}" for i in range(4)]
    assert context["to_weights"] == 1.241 and context["encoder_weight_bytes"] == WEIGHTS
    assert "1.24x" in context["measured"] and "4.29x" in context["measured"]
    assert facts["signatures"] == IO and facts["qairt"] == "2.50.0.260828221209"


def _dispatch_with(**changes: Any) -> list[dict[str, Any]]:
    d = _compiled()["dispatch"]
    d[1] = {**d[1], **changes}
    return d


@pytest.mark.parametrize(
    "overrides, stats, cause, said",
    [
        ({}, [], "run", "libc++1 and libatomic1"),
        ({}, [[0, 1, 1, 1]] * 3, "run", "3 of 4"),
        (
            {"ops": {**_compiled()["ops"], "encoder_1x128": {"DISPATCH_OP": 194, "9": 218}}},
            None,
            "partitioned",
            "DISPATCH_OP",
        ),
        (
            {"ops": {k: {"DISPATCH_OP": 1} for k in NPU_SIGNATURES[:3]}},
            None,
            "partitioned",
            "DISPATCH_OP",
        ),
        (
            {"dispatch": _dispatch_with(bytecode_offset=9999)},
            None,
            "dispatch",
            "2 bytecode ranges",
        ),
        (
            {"dispatch": _dispatch_with(name="qnn_partition_0")},
            None,
            "dispatch",
            "own function",
        ),
        ({"dispatch": _dispatch_with(name=None)}, None, "dispatch", "own function"),
        ({"bytes": 1000}, None, "dispatch", "not inside"),
        ({"stamp": b"Qualcomm".ljust(249, b"\0")}, None, "stamp", "249 bytes"),
        ({"stamp": None}, None, "stamp", "None bytes"),
        ({"stamp": b"\xff" * 250}, None, "stamp", "not ASCII"),
        (
            {"stamp": b"Qualcomm".ljust(125, b"\0") + b"SM8650".ljust(125, b"\0")},
            None,
            "stamp",
            "stamped for",
        ),
        (
            {"io": {**IO, "encoder_1x256": {"inputs": [], "outputs": []}}},
            None,
            "signature",
            "encoder_1x256",
        ),
    ],
)
def test_a_compile_that_is_not_what_it_must_be_is_refused(script, overrides, stats, cause, said):
    with pytest.raises(script["Refused"]) as exc:
        _check(script, _compiled(**overrides), stats=stats)
    assert exc.value.cause == cause and said in str(exc.value)


def test_a_compile_with_a_signature_more_than_asked_is_refused(script):
    ops = {**_compiled()["ops"], "encoder_1x2048": {"9": 218, "DISPATCH_OP": 3}}
    with pytest.raises(script["Refused"]) as exc:
        _check(script, _compiled(ops=ops))
    assert exc.value.cause == "partitioned"


def test_a_context_holding_unshared_weights_is_refused(script):
    unshared = [{**d, "bytecode_size": 586_875_872} for d in _compiled()["dispatch"]]
    with pytest.raises(script["Refused"], match="look unshared") as exc:
        _check(script, _compiled(dispatch=unshared, bytes=586_879_664))
    assert exc.value.cause == "context_size"
    # The measured shared context passes at the same weights; the bound is 2x.
    assert _check(script, _compiled(), weight_bytes=169_586_688 // 2)["qnn_context"]
    with pytest.raises(script["Refused"]):
        _check(script, _compiled(), weight_bytes=169_586_688 // 2 - 1)


def test_the_sdk_libraries_are_required(script, tmp_path):
    assert "did not happen" in script["sdk_problem"](None)
    libs = tmp_path / "libs"
    libs.mkdir()
    (libs / "libQnnHtp.so").write_bytes(b"")
    assert "libQnnSystem.so" in script["sdk_problem"](libs)
    (libs / "libQnnSystem.so").write_bytes(b"")
    assert script["sdk_problem"](libs) is None


def test_main_refuses_a_missing_sdk_as_a_run_failure(script, tmp_path):
    with pytest.raises(script["Refused"]) as exc:
        script["main"]({"work": str(tmp_path), "grafted": str(tmp_path / "none")})
    assert exc.value.cause == "run" and "did not complete" in str(exc.value)


@dataclass
class FakePlugin:
    returncode: int = 0
    stdout: str = (
        "Partitioned subgraph<0>, selected 1779 ops, from a total of 1779 ops. "
        "resulted in 1 partitions.\n" * 4
    )
    write_out: bool = True
    calls: list[dict[str, Any]] = field(default_factory=list)

    def run(self, cmd, env, **kwargs):
        self.calls.append({"cmd": cmd, "env": env})
        if self.write_out:
            Path(next(a for a in cmd if a.startswith("--o=")).removeprefix("--o=")).write_bytes(
                b"tfl"
            )
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, None)


def _apply(script, monkeypatch, tmp_path, plugin: FakePlugin):
    monkeypatch.setitem(
        script, "subprocess", types.SimpleNamespace(run=plugin.run, PIPE=-1, STDOUT=-2)
    )
    return script["apply_plugin"](
        "in.tflite", str(tmp_path / "out.tflite"), SOC, str(tmp_path / "log"), tmp_path / "sdk"
    )


def test_the_compiler_is_asked_for_the_soc_with_weight_sharing(script, monkeypatch, tmp_path):
    plugin = FakePlugin()
    seconds, stats = _apply(script, monkeypatch, tmp_path, plugin)
    cmd, env = plugin.calls[0]["cmd"], plugin.calls[0]["env"]
    assert "--qualcomm_enable_weight_sharing=true" in cmd
    assert f"--soc_model={SOC}" in cmd and "--soc_manufacturer=Qualcomm" in cmd
    assert "--model=in.tflite" in cmd and "--cmd=apply" in cmd
    assert env["LD_LIBRARY_PATH"] == f"{tmp_path / 'sdk'}{os.pathsep}/opt/litert/libs"
    assert stats == [[0, 1779, 1779, 1]] * 4
    assert (tmp_path / "log").read_text() == plugin.stdout


@pytest.mark.parametrize(
    "returncode, write_out, cause",
    [(-9, True, "run"), (-11, False, "run"), (1, True, "compile"), (0, False, "compile")],
)
def test_a_compiler_that_failed_is_refused_by_how(
    script, monkeypatch, tmp_path, returncode, write_out, cause
):
    with pytest.raises(script["Refused"]) as exc:
        _apply(
            script, monkeypatch, tmp_path, FakePlugin(returncode=returncode, write_out=write_out)
        )
    assert exc.value.cause == cause


def test_an_exception_in_the_script_is_its_own_cause(script):
    def broken(spec):
        raise KeyError("bytecode_offset")

    answer = script["answer_for"]({}, broken)
    assert answer["ok"] is False and answer["cause"] == "script"
    assert answer["reason"].startswith("KeyError") and "Traceback" in answer["reason"]


def test_a_refusal_in_the_script_is_answered_with_its_cause(script):
    def refuses(spec):
        raise script["Refused"]("stamp", "wrong")

    assert script["answer_for"]({}, refuses) == {"ok": False, "cause": "stamp", "reason": "wrong"}
    assert script["answer_for"]({}, lambda spec: {"ok": True}) == {"ok": True}


# cut_model on a model of plain objects shaped like the ModelT it is given


def _tensor(name: str, ext: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(name=name, externalBuffer=ext, buffer=0)


def _composite(index: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        builtinOptions2Type=7,
        builtinOptions2=types.SimpleNamespace(decompositionSubgraphIndex=index),
        builtinOptionsType=0,
    )


def _model(**overrides: Any) -> types.SimpleNamespace:
    # Subgraphs: 0 = encoder_1x8192 (dropped), 1 = encoder_1x128 (kept), 2 = a decomposition
    # only 0 reaches, 3 = one 1 reaches.
    sub = [
        types.SimpleNamespace(tensors=[_tensor("w8192", 1)], operators=[_composite(2)]),
        types.SimpleNamespace(
            tensors=[_tensor("w", 1), _tensor("w_again", 2), _tensor("x")],
            operators=[_composite(3)],
        ),
        types.SimpleNamespace(tensors=[], operators=[]),
        types.SimpleNamespace(tensors=[_tensor("norm", 3)], operators=[]),
    ]
    model = types.SimpleNamespace(
        subgraphs=sub,
        signatureDefs=[
            types.SimpleNamespace(signatureKey=b"encoder_1x8192", subgraphIndex=0),
            types.SimpleNamespace(signatureKey=b"encoder_1x128", subgraphIndex=1),
        ],
        buffers=[_Buffer()],
        externalBuffers=[
            types.SimpleNamespace(id=1, group=1, offset=0, length=4),
            types.SimpleNamespace(id=2, group=1, offset=0, length=4),
            types.SimpleNamespace(id=3, group=1, offset=4, length=2),
        ],
        externalBufferGroups=[
            types.SimpleNamespace(name=b""),
            types.SimpleNamespace(name=b"tflite_weights"),
        ],
    )
    model.__dict__.update(overrides)
    return model


def test_the_cut_keeps_the_signature_and_what_it_reaches(script):
    m = script["cut_model"](_model(), b"\x01\x02\x03\x04\x05\x06", ["encoder_1x128"])
    assert [d.signatureKey for d in m.signatureDefs] == [b"encoder_1x128"]
    assert m.signatureDefs[0].subgraphIndex == 0
    assert len(m.subgraphs) == 2, "the kept signature and the decomposition it reaches"
    assert m.subgraphs[0].operators[0].builtinOptions2.decompositionSubgraphIndex == 1
    w, w_again, x = m.subgraphs[0].tensors
    assert w.buffer == w_again.buffer == 1, "one buffer for one external range"
    assert x.buffer == 0 and x.externalBuffer == 0
    assert bytes(m.buffers[1].data) == b"\x01\x02\x03\x04"
    (norm,) = m.subgraphs[1].tensors
    assert bytes(m.buffers[norm.buffer].data) == b"\x05\x06"
    assert len(m.buffers) == 3
    assert m.externalBuffers is None and m.externalBufferGroups is None
    assert all(t.externalBuffer == 0 for sg in m.subgraphs for t in sg.tensors)


def test_the_cut_refuses_what_it_cannot_carry(script):
    Refused = script["Refused"]
    with pytest.raises(Refused, match="no signature"):
        script["cut_model"](_model(), b"\0" * 6, ["encoder_1x256"])
    with pytest.raises(Refused, match="past the weights"):
        script["cut_model"](_model(), b"\0" * 5, ["encoder_1x128"])
    other = _model(
        externalBufferGroups=[
            types.SimpleNamespace(name=b""),
            types.SimpleNamespace(name=b"elsewhere"),
        ]
    )
    with pytest.raises(Refused, match="outside tflite_weights"):
        script["cut_model"](other, b"\0" * 6, ["encoder_1x128"])
    looped = _model()
    looped.subgraphs[1].operators.append(
        types.SimpleNamespace(builtinOptions2Type=0, builtinOptionsType=101)
    )
    with pytest.raises(Refused, match="control-flow") as exc:
        script["cut_model"](looped, b"\0" * 6, ["encoder_1x128"])
    assert exc.value.cause == "input"


# ---------------------------------------------------------------------------
# convert --npu
# ---------------------------------------------------------------------------


@dataclass
class FakeCompile:
    causes: dict[str, str] = field(default_factory=dict)
    raises: dict[str, BaseException] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, env, grafted, soc, output, *, timeout, **kwargs) -> NpuResult:
        self.calls.append(
            {"grafted": Path(grafted), "soc": soc, "output": Path(output), "timeout": timeout}
        )
        if soc in self.raises:
            raise self.raises[soc]
        if soc in self.causes:
            raise EmbeddingNpuError(self.causes[soc], f"refused for {self.causes[soc]}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"npu " + soc.encode())
        return NpuResult(
            soc,
            output,
            7,
            hashlib.sha256(b"npu " + soc.encode()).hexdigest(),
            230.0,
            169_000_000,
            "ef" * 32,
            _ok(stamp=["Qualcomm", soc]),
        )


@pytest.fixture
def compiler(monkeypatch) -> FakeCompile:
    fake = FakeCompile()
    monkeypatch.setattr(embedding_export, "compile_for_npu", fake)
    monkeypatch.setattr(embedding_export, "host_problem", lambda: None)
    monkeypatch.setattr(envs.StageEnv, "refusal", lambda self: None)
    return fake


def _report(out: Path, where: str = "int8") -> dict[str, Any]:
    return json.loads((out / where / REPORT_NAME).read_text(encoding="utf-8"))


def _npu_args(*socs: str) -> list[str]:
    return ["--weights", "int8", *[a for soc in socs for a in ("--npu", soc)]]


def test_npu_compiles_and_publishes_each_soc_with_its_report(hub, grafter, compiler, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) == 0
    assert [c["soc"] for c in compiler.calls] == ["SM8750", "SM8650"]
    assert all(c["grafted"] == out / "int8" / "model.litertlm" for c in compiler.calls)
    assert all(
        c["output"].parent.parent.name.startswith(".litetune-npu-stage-") for c in compiler.calls
    )
    assert all(c["timeout"] == NPU_TIMEOUT_S for c in compiler.calls)
    assert not list(out.glob(".litetune-npu-stage-*")), "the staging directory is gone"
    report = _report(out)
    assert report["npu_published"] is True
    assert [n["soc"] for n in report["npu"]] == ["SM8750", "SM8650"]
    assert [n["device_run"] for n in report["npu"]] == [DEVICE_RUNS["SM8750"], None]
    assert report["request"]["npu"] == ["SM8750", "SM8650"]
    assert NPU_NOT_VERIFIED in report["limitations"]
    names = [c["name"] for c in report["checks"]["checks"]]
    assert {"npu environment", "npu SM8750", "npu SM8650"} <= set(names)
    for soc in ("SM8750", "SM8650"):
        npu_file = out / f"int8-npu-{soc}" / "model.litertlm"
        assert npu_file.read_bytes() == f"npu {soc}".encode()
        sibling = _report(out, f"int8-npu-{soc}")
        assert sibling["kind"] == "npu" and sibling["artifact"] == str(npu_file)
        assert sibling["npu"]["sha256"] == hashlib.sha256(npu_file.read_bytes()).hexdigest()
        assert sibling["npu"]["stamp"] == ["Qualcomm", soc]
        assert sibling["signatures"] == list(NPU_SIGNATURES)
        assert sibling["source_graft"]["output"] == str(out / "int8" / "model.litertlm")
        assert sibling["checkpoint"] == report["checkpoint"]
        assert sibling["limitations"] == [NPU_NOT_VERIFIED]
        assert npu_target_of(npu_file) == soc


def test_timeout_s_bounds_each_npu_compile(hub, grafter, compiler, tmp_path):  # noqa: F811
    assert (
        _convert(
            _checkpoint(tmp_path), tmp_path / "out", *_npu_args("SM8750"), "--timeout-s", "123"
        )
        == 0
    )
    assert compiler.calls[0]["timeout"] == 123


def test_one_soc_failing_publishes_none(hub, grafter, compiler, tmp_path):  # noqa: F811
    compiler.causes["SM8650"] = "stamp"
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) != 0
    report = _report(out)
    assert report["outcome"] == Outcome.FAILED.value and report["npu_published"] is False
    assert report["npu"] == []
    assert not (out / "int8-npu-SM8750").exists() and not (out / "int8-npu-SM8650").exists()
    assert not list(out.glob(".litetune-npu-stage-*"))
    assert any("no NPU file was published: SM8650" in t for t in report["limitations"])
    assert (out / "int8" / "model.litertlm").is_file(), "the CPU graft is still written"


def test_a_run_failure_is_could_not_check(hub, grafter, compiler, tmp_path):  # noqa: F811
    compiler.causes["SM8750"] = "run"
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750")) != 0
    report = _report(out)
    assert report["outcome"] == Outcome.UNCHECKED.value
    (npu,) = [c for c in report["checks"]["checks"] if c["name"] == "npu SM8750"]
    assert npu["outcome"] == Outcome.UNCHECKED.value


def test_an_error_in_one_compile_is_that_socs_and_the_next_still_runs(
    hub, grafter, compiler, tmp_path
):  # noqa: F811
    compiler.raises["SM8750"] = OSError("disk full")
    out = tmp_path / "out"
    _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650"))
    assert [c["soc"] for c in compiler.calls] == ["SM8750", "SM8650"]
    checks = {c["name"]: c for c in _report(out)["checks"]["checks"]}
    assert checks["npu SM8750"]["outcome"] == Outcome.UNCHECKED.value
    assert "OSError: disk full" in checks["npu SM8750"]["detail"]
    assert checks["npu SM8650"]["outcome"] == Outcome.PASSED.value
    assert checks["graft"]["outcome"] == Outcome.PASSED.value


def test_an_npu_environment_that_is_not_ready_compiles_nothing(
    hub, grafter, compiler, monkeypatch, tmp_path
):  # noqa: F811
    real_ready = envs.StageEnv.ready
    monkeypatch.setattr(
        envs.StageEnv,
        "ready",
        property(
            lambda self: False if self.name == envs.NPU_QUALCOMM.name else real_ready.fget(self)
        ),
    )
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750")) != 0
    assert compiler.calls == []
    report = _report(out)
    (env_check,) = [c for c in report["checks"]["checks"] if c["name"] == "npu environment"]
    assert env_check["outcome"] == Outcome.UNCHECKED.value
    assert any(t.startswith("no NPU file was compiled") for t in report["limitations"])


def test_an_existing_npu_output_is_refused_before_the_graft(hub, grafter, compiler, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    existing = output_for(out, "SM8650")
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep")
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) == 4
    assert grafter.calls == [] and compiler.calls == []
    assert existing.read_bytes() == b"keep"


def test_an_npu_environment_this_host_cannot_build_is_refused_first(
    hub, grafter, compiler, monkeypatch, tmp_path, capsys
):  # noqa: F811
    monkeypatch.setattr(envs.StageEnv, "refusal", lambda self: "needs glibc 2.27")
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", *_npu_args("SM8750")) == 4
    assert grafter.calls == [] and "glibc 2.27" in capsys.readouterr().err


def test_npu_with_int4_is_refused_on_the_command_line(hub, grafter, compiler, tmp_path, capsys):  # noqa: F811
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", "--npu", "SM8750") == 4
    assert grafter.calls == [] and compiler.calls == []
    assert "--weights int8" in capsys.readouterr().err


def test_npu_off_linux_x86_64_is_refused(hub, grafter, monkeypatch, tmp_path, capsys):  # noqa: F811
    monkeypatch.setattr(embedding_export, "host_problem", lambda: "Linux x86-64 only")
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", *_npu_args("SM8750")) == 4
    assert grafter.calls == []
    assert "Linux x86-64 only" in capsys.readouterr().err


def test_npu_for_an_llm_checkpoint_takes_the_llm_path(tmp_path, capsys):
    """An LLM checkpoint's --npu is llm_npu's (tests/test_llm_npu.py), which needs its data."""
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m-it"}))
    assert _convert(llm, tmp_path / "out", "--npu", "SM8750") == 4
    assert "--npu needs --calibration-data" in capsys.readouterr().err


def test_an_npu_report_of_another_file_names_no_target(tmp_path):
    model = tmp_path / "model.litertlm"
    model.write_bytes(b"this one")
    report = {
        "schema": embedding_export.EMBEDDING_CONVERT_SCHEMA,
        "kind": "npu",
        "npu": {"soc": "SM8750", "sha256": "00" * 32},
    }
    (tmp_path / REPORT_NAME).write_text(json.dumps(report))
    assert npu_target_of(model) is None
    report["npu"]["sha256"] = hashlib.sha256(b"this one").hexdigest()
    (tmp_path / REPORT_NAME).write_text(json.dumps(report))
    assert npu_target_of(model) == "SM8750"
    report["kind"] = "graft"
    (tmp_path / REPORT_NAME).write_text(json.dumps(report))
    assert npu_target_of(model) is None


# ---------------------------------------------------------------------------
# Publication per SoC, readiness, and whose failure a cause is
# ---------------------------------------------------------------------------


def test_each_published_soc_has_its_report_inside_its_directory(hub, grafter, compiler, tmp_path):
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) == 0
    for soc in ("SM8750", "SM8650"):
        assert sorted(p.name for p in (out / f"int8-npu-{soc}").iterdir()) == [
            REPORT_NAME,
            "model.litertlm",
        ]


def test_an_existing_npu_directory_is_refused_before_the_graft(hub, grafter, compiler, tmp_path):
    out = tmp_path / "out"
    (out / "int8-npu-SM8750").mkdir(parents=True)
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750")) == 4
    assert grafter.calls == [] and compiler.calls == []


@pytest.mark.parametrize(
    "cause, verdict", [("script", Outcome.UNCHECKED), ("compiled_file", Outcome.FAILED)]
)
def test_a_script_error_is_could_not_check_and_a_bad_compiled_file_is_a_failure(
    hub, grafter, compiler, tmp_path, cause, verdict
):
    compiler.causes["SM8750"] = cause
    out = tmp_path / "out"
    _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750"))
    (npu,) = [c for c in _report(out)["checks"]["checks"] if c["name"] == "npu SM8750"]
    assert npu["outcome"] == verdict.value


def test_unreadable_compiled_bytes_are_the_compiled_files(script, monkeypatch):
    def broken(path):
        raise ValueError("unpack_from requires a buffer of at least 4 bytes")

    monkeypatch.setitem(script, "describe", broken)
    with pytest.raises(script["Refused"]) as exc:
        script["read_compiled"]("encoder_npu.tflite")
    assert exc.value.cause == "compiled_file" and "ValueError" in str(exc.value)


def test_an_installed_file_removed_after_provisioning_makes_the_environment_unready(
    monkeypatch, tmp_path
):
    from conftest import mark_provisioned

    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    env = envs.NPU_QUALCOMM
    mark_provisioned(env)
    assert env.ready
    (lib,) = env.path.glob(env.installed_files[0])
    lib.unlink()
    assert not env.ready
    assert envs.StageEnv(name="plain", requirements=("a==1",)).missing_installed_files() == []


def test_an_installed_file_that_is_a_directory_does_not_count(monkeypatch, tmp_path):
    from conftest import mark_provisioned

    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    env = envs.NPU_QUALCOMM
    mark_provisioned(env)
    (lib,) = env.path.glob(env.installed_files[0])
    lib.unlink()
    lib.mkdir()
    assert not env.ready
    assert env.missing_installed_files() == [env.installed_files[0]]


def _rename_failing(real, n: int, raised: BaseException):
    """`real`, raising `raised` on its `n`-th call (1-based)."""
    calls = {"n": 0}

    def rename(source, target):
        calls["n"] += 1
        if calls["n"] == n:
            raise raised
        real(source, target)

    return rename


def _publish_check(out: Path) -> dict[str, Any]:
    (check,) = [c for c in _report(out)["checks"]["checks"] if c["name"] == "npu files"]
    return check


def test_a_target_created_during_the_compile_is_left_untouched(
    hub, grafter, compiler, monkeypatch, tmp_path
):
    out = tmp_path / "out"
    foreign = out / "int8-npu-SM8650"
    real_call = compiler.__call__

    def compile_and_meddle(env, grafted, soc, output, **kwargs):
        if soc == "SM8650":
            foreign.mkdir(parents=True)
            (foreign / "theirs.txt").write_text("keep")
        return real_call(env, grafted, soc, output, **kwargs)

    monkeypatch.setattr(embedding_export, "compile_for_npu", compile_and_meddle)
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) != 0
    assert sorted(p.name for p in foreign.iterdir()) == ["theirs.txt"], "nothing replaced"
    assert (foreign / "theirs.txt").read_text() == "keep"
    report = _report(out)
    assert [n["soc"] for n in report["npu"]] == ["SM8750"] and report["npu_published"] is False
    check = _publish_check(out)
    assert check["outcome"] == Outcome.UNCHECKED.value
    assert "published 1 of 2" in check["detail"] and "FileExistsError" in check["detail"]
    assert (
        "written in part" not in check["detail"]
    ), "a directory this run did not create is not named partial"
    assert any("Run the rest with another --output-dir" in t for t in report["limitations"])


def test_an_interrupt_after_the_first_soc_is_reported_from_the_disk(
    hub, grafter, compiler, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        embedding_export,
        "_place",
        _rename_failing(embedding_export._place, 3, KeyboardInterrupt()),
    )
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650"))
    first = out / "int8-npu-SM8750"
    assert sorted(p.name for p in first.iterdir()) == [REPORT_NAME, "model.litertlm"]
    report = _report(out)
    assert [n["soc"] for n in report["npu"]] == ["SM8750"] and report["npu_published"] is False
    assert "KeyboardInterrupt" in _publish_check(out)["detail"]
    # The second SoC's directory was created and holds nothing of it: named, not removed.
    assert (out / "int8-npu-SM8650").is_dir()
    assert str(out / "int8-npu-SM8650") in _publish_check(out)["detail"]


def test_a_failure_moving_the_second_socs_report_leaves_both_and_names_the_partial(
    hub, grafter, compiler, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        embedding_export,
        "_place",
        _rename_failing(embedding_export._place, 4, OSError("no space")),
    )
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) != 0
    assert sorted(p.name for p in (out / "int8-npu-SM8750").iterdir()) == [
        REPORT_NAME,
        "model.litertlm",
    ]
    second = out / "int8-npu-SM8650"
    assert sorted(p.name for p in second.iterdir()) == ["model.litertlm"], "not deleted"
    report = _report(out)
    assert [n["soc"] for n in report["npu"]] == ["SM8750"] and report["npu_published"] is False
    check = _publish_check(out)
    assert "published 1 of 2" in check["detail"] and str(second) in check["detail"]
    assert "OSError: no space" in check["detail"]


def test_a_model_that_does_not_match_its_report_is_not_counted(
    hub, grafter, compiler, monkeypatch, tmp_path
):
    real = embedding_export._place

    def rename(source, target):
        real(source, target)
        if target.name == "model.litertlm" and target.parent.name == "int8-npu-SM8650":
            target.write_bytes(b"other bytes")

    monkeypatch.setattr(embedding_export, "_place", rename)
    out = tmp_path / "out"
    _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650"))
    report = _report(out)
    assert [n["soc"] for n in report["npu"]] == ["SM8750"]
    assert str(out / "int8-npu-SM8650") in _publish_check(out)["detail"]
    assert (out / "int8-npu-SM8650" / "model.litertlm").read_bytes() == b"other bytes"


def test_a_whole_publication_passes_its_check(hub, grafter, compiler, tmp_path):
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) == 0
    assert _publish_check(out)["outcome"] == Outcome.PASSED.value
    assert _report(out)["npu_published"] is True


def test_nothing_is_published_when_a_soc_did_not_compile(hub, grafter, compiler, tmp_path):
    compiler.causes["SM8650"] = "stamp"
    out = tmp_path / "out"
    _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650"))
    assert not (out / "int8-npu-SM8750").exists() and not (out / "int8-npu-SM8650").exists()
    assert not list(out.glob(".litetune-npu-stage-*"))


def test_a_file_that_appears_in_the_new_directory_is_not_replaced(
    hub, grafter, compiler, monkeypatch, tmp_path
):
    real = embedding_export._mkdir

    def mkdir_and_meddle(target):
        real(target)
        if target.name == "int8-npu-SM8650":
            (target / "model.litertlm").write_bytes(b"someone else's")

    monkeypatch.setattr(embedding_export, "_mkdir", mkdir_and_meddle)
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650")) != 0
    second = out / "int8-npu-SM8650"
    assert (second / "model.litertlm").read_bytes() == b"someone else's", "not replaced"
    report = _report(out)
    assert [n["soc"] for n in report["npu"]] == ["SM8750"]
    check = _publish_check(out)
    assert "FileExistsError" in check["detail"] and str(second) in check["detail"]


def test_an_interrupt_right_after_the_mkdir_still_names_the_directory(
    hub, grafter, compiler, monkeypatch, tmp_path
):
    real = embedding_export._mkdir

    def mkdir_then_interrupt(target):
        real(target)
        if target.name == "int8-npu-SM8650":
            raise KeyboardInterrupt

    monkeypatch.setattr(embedding_export, "_mkdir", mkdir_then_interrupt)
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        _convert(_checkpoint(tmp_path), out, *_npu_args("SM8750", "SM8650"))
    check = _publish_check(out)
    assert "written in part" in check["detail"] and str(out / "int8-npu-SM8650") in check["detail"]


def test_the_staged_copy_is_gone_once_a_file_is_placed(tmp_path):
    source, target = tmp_path / "a", tmp_path / "b"
    source.write_bytes(b"x")
    embedding_export._place(source, target)
    assert target.read_bytes() == b"x" and not source.exists()
    source.write_bytes(b"y")
    with pytest.raises(FileExistsError):
        embedding_export._place(source, target)
    assert target.read_bytes() == b"x" and source.exists()
