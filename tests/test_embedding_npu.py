"""`convert --npu`: the refusals, the request, the step after the graft, and the
compile's answer as `compile_for_npu` reads it.

Nothing is compiled here: the compiler runs on Linux x86-64 only and downloads QAIRT.
`FakeNpuRun` stands in for `StageEnv.run` on the NPU script and writes the answer the
script writes; `FakeCompile` stands in for `compile_for_npu` in the convert path.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from test_embedding_export import _checkpoint, _convert, grafter, hub  # noqa: F401

from litetune import embedding_export, embedding_npu, envs
from litetune.checks import Outcome
from litetune.embedding_export import REPORT_NAME, EmbeddingExportError, EmbeddingExportRequest
from litetune.embedding_npu import (
    MEASURED_ON_DEVICE,
    NPU_SIGNATURES,
    NPU_SOCS,
    EmbeddingNpuError,
    NpuResult,
    compile_for_npu,
    host_problem,
    output_for,
    request_problem,
)

# ---------------------------------------------------------------------------
# What is refused before anything runs
# ---------------------------------------------------------------------------


def test_int4_is_refused_with_the_measured_reason():
    problem = request_problem(("SM8750",), "int4")
    assert problem is not None and "--weights int8" in problem and "194 partitions" in problem


def test_int8_with_known_socs_is_accepted():
    assert request_problem(("SM8750", "SM8650"), "int8") is None
    assert request_problem((), "int4") is None


def test_an_unknown_or_repeated_soc_is_refused():
    assert "one of" in (request_problem(("SM9999",), "int8") or "")
    assert "twice" in (request_problem(("SM8750", "SM8750"), "int8") or "")


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


def test_the_request_refuses_int4_with_npu(tmp_path):
    with pytest.raises(EmbeddingExportError, match="--weights int8"):
        EmbeddingExportRequest(model=tmp_path, output_dir=tmp_path, weights="int4", npu=("SM8750",))


def test_the_signatures_are_google_npu_files_four():
    assert NPU_SIGNATURES == ("encoder_1x128", "encoder_1x256", "encoder_1x512", "encoder_1x1024")
    assert set(MEASURED_ON_DEVICE) <= set(NPU_SOCS)


def test_the_npu_script_compiles_as_python():
    compile(embedding_npu._NPU_SCRIPT, "npu.py", "exec")


# ---------------------------------------------------------------------------
# compile_for_npu reading the script's answer
# ---------------------------------------------------------------------------


@dataclass
class FakeNpuRun:
    """`StageEnv.run` on the NPU script: writes `answer` (and the packed file it names)."""

    answer: dict[str, Any] | None
    returncode: int = 0
    specs: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, args, timeout: int = 3600, **kwargs) -> subprocess.CompletedProcess:
        spec = json.loads(Path(args[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        if self.answer is not None:
            answer = dict(self.answer)
            if answer.get("ok"):
                packed = Path(spec["work"]) / "packed.litertlm"
                packed.write_bytes(b"npu file")
                answer["packed"] = str(packed)
            Path(spec["answer"]).write_text(json.dumps(answer), encoding="utf-8")
        return subprocess.CompletedProcess(args, self.returncode, "", "Traceback: boom")


OK = {
    "ok": True,
    "bytes": 8,
    "sha256": "ab" * 32,
    "compile_s": 233.1,
    "encoder_bytes": 169590480,
    "compiler": {"ai-edge-litert": "2.3.0"},
}


@pytest.fixture
def npu_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))
    return envs.NPU_QUALCOMM


def test_a_compiled_file_is_moved_to_its_output(monkeypatch, tmp_path, npu_env):
    fake = FakeNpuRun(OK)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    out = output_for(tmp_path / "out", "SM8750")
    got = compile_for_npu(npu_env, tmp_path / "g.litertlm", "SM8750", out, auto_provision=False)
    assert out.read_bytes() == b"npu file"
    assert got.soc == "SM8750" and got.encoder_bytes == 169590480
    assert got.as_dict()["measured_on_device"] is True
    assert fake.specs[0]["signatures"] == list(NPU_SIGNATURES)
    assert out == tmp_path / "out" / "int8-npu-SM8750" / "model.litertlm"


def test_a_partitioned_compile_is_refused_with_its_cause(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(
        envs.StageEnv, "run", FakeNpuRun({"ok": False, "cause": "partitioned", "reason": "194"}, 3)
    )
    out = output_for(tmp_path, "SM8650")
    with pytest.raises(EmbeddingNpuError) as exc:
        compile_for_npu(npu_env, tmp_path / "g", "SM8650", out, auto_provision=False)
    assert exc.value.cause == "partitioned" and not out.exists()


def test_a_script_that_died_is_a_run_failure(monkeypatch, tmp_path, npu_env):
    monkeypatch.setattr(envs.StageEnv, "run", FakeNpuRun(None, 1))
    with pytest.raises(EmbeddingNpuError) as exc:
        compile_for_npu(
            npu_env, tmp_path / "g", "SM8750", output_for(tmp_path, "SM8750"), auto_provision=False
        )
    assert exc.value.cause == "run" and "boom" in str(exc.value)


def test_an_existing_output_is_not_written_over(tmp_path, npu_env):
    out = output_for(tmp_path, "SM8750")
    out.parent.mkdir(parents=True)
    out.write_bytes(b"keep")
    with pytest.raises(EmbeddingNpuError, match="already exists"):
        compile_for_npu(npu_env, tmp_path / "g", "SM8750", out, auto_provision=False)
    assert out.read_bytes() == b"keep"


# ---------------------------------------------------------------------------
# convert --npu
# ---------------------------------------------------------------------------


@dataclass
class FakeCompile:
    cause: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, env, grafted, soc, output, **kwargs) -> NpuResult:
        self.calls.append({"grafted": Path(grafted), "soc": soc, "output": Path(output)})
        if self.cause is not None:
            raise EmbeddingNpuError(self.cause, f"refused for {self.cause}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"npu " + soc.encode())
        return NpuResult(soc, output, 7, "cd" * 32, 230.0, 169_000_000, {"compiler": {}})


@pytest.fixture
def compiler(monkeypatch) -> FakeCompile:
    fake = FakeCompile()
    monkeypatch.setattr(embedding_export, "compile_for_npu", fake)
    monkeypatch.setattr(embedding_export, "host_problem", lambda: None)
    return fake


def _report(out: Path) -> dict[str, Any]:
    return json.loads((out / "int8" / REPORT_NAME).read_text(encoding="utf-8"))


def test_npu_compiles_the_int8_graft_for_each_soc(hub, grafter, compiler, tmp_path):  # noqa: F811
    out = tmp_path / "out"
    model = _checkpoint(tmp_path)
    assert _convert(model, out, "--weights", "int8", "--npu", "SM8750", "--npu", "SM8650") == 0
    assert [c["soc"] for c in compiler.calls] == ["SM8750", "SM8650"]
    assert compiler.calls[0]["grafted"] == out / "int8" / "model.litertlm"
    assert (out / "int8-npu-SM8750" / "model.litertlm").read_bytes() == b"npu SM8750"
    report = _report(out)
    assert [n["soc"] for n in report["npu"]] == ["SM8750", "SM8650"]
    assert [n["measured_on_device"] for n in report["npu"]] == [True, False]
    assert report["request"]["npu"] == ["SM8750", "SM8650"]
    names = [c["name"] for c in report["checks"]["checks"]]
    assert "npu SM8750" in names and "npu SM8650" in names


def test_a_failed_compile_fails_the_convert(hub, grafter, monkeypatch, tmp_path):  # noqa: F811
    fake = FakeCompile(cause="stamp")
    monkeypatch.setattr(embedding_export, "compile_for_npu", fake)
    monkeypatch.setattr(embedding_export, "host_problem", lambda: None)
    out = tmp_path / "out"
    assert _convert(_checkpoint(tmp_path), out, "--weights", "int8", "--npu", "SM8750") != 0
    report = _report(out)
    assert report["outcome"] == Outcome.FAILED.value
    # The CPU graft is still written and named.
    assert (out / "int8" / "model.litertlm").is_file()


def test_npu_with_int4_is_refused_on_the_command_line(hub, grafter, compiler, tmp_path, capsys):  # noqa: F811
    assert _convert(_checkpoint(tmp_path), tmp_path / "out", "--npu", "SM8750") == 4
    assert grafter.calls == [] and compiler.calls == []
    assert "--weights int8" in capsys.readouterr().err


def test_npu_off_linux_x86_64_is_refused(hub, grafter, monkeypatch, tmp_path, capsys):  # noqa: F811
    monkeypatch.setattr(embedding_export, "host_problem", lambda: "Linux x86-64 only")
    assert (
        _convert(_checkpoint(tmp_path), tmp_path / "out", "--weights", "int8", "--npu", "SM8750")
        == 4
    )
    assert grafter.calls == []
    assert "Linux x86-64 only" in capsys.readouterr().err


def test_npu_for_an_llm_checkpoint_is_refused(tmp_path, capsys):
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "litetune.json").write_text(json.dumps({"base_model": "google/gemma-3-270m-it"}))
    assert _convert(llm, tmp_path / "out", "--npu", "SM8750") == 4
    assert "--npu" in capsys.readouterr().err


def test_the_qualcomm_sdk_is_a_source_build_not_a_gap():
    from litetune import wheels

    sdk = wheels.PUBLISHED["ai-edge-litert-sdk-qualcomm==2.3.0"]
    assert sdk.sdist and sdk.source_build
    linux = wheels.Host.of(("manylinux_2_27_x86_64", "linux_x86_64"), (3, 12))
    assert wheels.gaps(envs.NPU_QUALCOMM.requirements, linux) == []


def test_a_sdist_only_pin_without_a_reason_is_still_a_gap():
    from dataclasses import replace

    from litetune import wheels

    sdk = wheels.PUBLISHED["ai-edge-litert-sdk-qualcomm==2.3.0"]
    table = dict(wheels.PUBLISHED) | {sdk.requirement: replace(sdk, source_build="")}
    linux = wheels.Host.of(("manylinux_2_27_x86_64", "linux_x86_64"), (3, 12))
    assert [g.release.requirement for g in wheels.gaps([sdk.requirement], linux, table)] == [
        sdk.requirement
    ]
