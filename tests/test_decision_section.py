"""The decision section script, run for real against a toy `litert_lm_builder`.

The toy builder writes a file with the real layout's one fact the script relies
on -- the header's end as a little-endian u64 at byte 24 -- then a JSON header
naming each section's type, items and byte offsets, then the payloads. `unpack`
writes a `model.toml` and one file per section as the real one does; `pack`
reads them back by `data_path`. So the script's own reads -- the header end,
the payload by its offsets, the round trip -- run against bytes it cannot
fake.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from litetune import decision_section
from litetune.decision_section import (
    SECTION_NAME,
    dart_call,
    dart_string,
    payload_bytes,
    write_section,
)

TOY_BUILDER = r"""
import json
import os
import struct
import tomllib
from pathlib import Path

MAGIC = b"LITERTLM"


def _entry(key, value):
    return f'  {{ key = {json.dumps(key)}, value_type = "String", value = {json.dumps(value)} }},'


def _read(path):
    raw = Path(path).read_bytes()
    end = struct.unpack("<Q", raw[24:32])[0]
    header = json.loads(raw[32:end].decode("utf-8"))
    return raw, header


def unpack(litertlm_path, output_dir, jinja_prompt_template_path=None):
    raw, header = _read(litertlm_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["[system_metadata]", "entries = ["]
    for e in header["system"]:
        lines.append(_entry(e["key"], e["value"]))
    lines += ["]", ""]
    for i, s in enumerate(header["sections"]):
        name = f"Section{i}_{s['type']}.bin"
        (out / name).write_bytes(raw[s["begin"]:s["end"]])
        lines += ["[[section]]", f'section_type = "{s["type"]}"', f'data_path = "{name}"']
        if s["items"]:
            lines.append("additional_metadata = [")
            for it in s["items"]:
                lines.append(_entry(it["key"], it["value"]))
            lines.append("]")
        lines.append("")
    (out / "model.toml").write_text("\n".join(lines), encoding="utf-8")
    return str(out / "model.toml")


def pack(toml_path, output_path, jinja_prompt_template_path=None):
    toml_path = Path(toml_path)
    doc = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    system = [e for e in doc.get("system_metadata", {}).get("entries", [])
              if e["key"] not in ("uuid", "creation_timestamp")]
    system += [
        {"key": "uuid", "value": os.urandom(4).hex()},
        {"key": "creation_timestamp", "value": "now"},
    ]
    payloads, sections = [], []
    for s in doc.get("section", []):
        p = Path(s["data_path"])
        p = p if p.is_absolute() else toml_path.parent / p
        payloads.append(p.read_bytes())
        items = [{"key": e["key"], "value": e["value"]} for e in s.get("additional_metadata", [])]
        sections.append({"type": s["section_type"], "items": items})
    # Offsets depend on the header's length, which depends on the offsets' digits: settle it.
    end = 32
    for _ in range(5):
        offset = end
        for s, data in zip(sections, payloads):
            s["begin"], s["end"] = offset, offset + len(data)
            offset += len(data)
        body = json.dumps({"system": system, "sections": sections}).encode("utf-8")
        end = 32 + len(body)
    Path(output_path).write_bytes(
        MAGIC + b"\0" * 16 + struct.pack("<Q", end) + body + b"".join(payloads)
    )
    return str(output_path)
"""

TOY_PEEK = r"""
import json
import struct
from pathlib import Path


class _Item:
    def __init__(self, d):
        self.d = d


class _Object:
    def __init__(self, s):
        self.s = s

    def BeginOffset(self):
        return self.s["begin"]

    def EndOffset(self):
        return self.s["end"]

    def ItemsLength(self):
        return len(self.s["items"])

    def Items(self, j):
        return _Item(self.s["items"][j])


class _Sections:
    def __init__(self, sections):
        self.sections = sections

    def ObjectsLength(self):
        return len(self.sections)

    def Objects(self, i):
        return _Object(self.sections[i])


class _Header:
    def __init__(self, header):
        self.header = header

    def SectionMetadata(self):
        return _Sections(self.header["sections"])


def read_litertlm_header(path, stream):
    raw = Path(path).read_bytes()
    end = struct.unpack("<Q", raw[24:32])[0]
    return _Header(json.loads(raw[32:end].decode("utf-8")))


def kvp_to_dict(item):
    return {"key": item.d["key"], "value": item.d["value"], "value_type": "String"}
"""


@pytest.fixture
def toy(tmp_path):
    """The toy builder on a path, and a bundle it packed with two ordinary sections."""
    shim = tmp_path / "shim"
    package = shim / "litert_lm_builder"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "litertlm_builder.py").write_text(TOY_BUILDER, encoding="utf-8")
    (package / "litertlm_peek.py").write_text(TOY_PEEK, encoding="utf-8")
    (package / "litertlm_core.py").write_text("HEADER_END_LOCATION_BYTE_OFFSET = 24\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta.pbtext").write_bytes(b'jinja_prompt_template: "t"\n')
    (source / "model.tflite").write_bytes(b"\x01" * 4096)
    (source / "model.toml").write_text(
        "[system_metadata]\nentries = []\n\n"
        '[[section]]\nsection_type = "LlmMetadata"\ndata_path = "meta.pbtext"\n\n'
        '[[section]]\nsection_type = "TFLiteModel"\ndata_path = "model.tflite"\n'
        "additional_metadata = [\n"
        '  { key = "model_type", value_type = "String", value = "prefill_decode" },\n]\n',
        encoding="utf-8",
    )
    bundle = tmp_path / "model.litertlm"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from litert_lm_builder import litertlm_builder as lb; "
            "lb.pack(sys.argv[1], sys.argv[2])",
            str(source / "model.toml"),
            str(bundle),
        ],
        env={**os.environ, "PYTHONPATH": str(shim)},
        check=True,
    )
    return shim, bundle


class ShimEnv:
    """A stage environment that runs the command with the toy builder importable."""

    name = "export"

    def __init__(self, shim: Path):
        self.shim = shim

    def provision(self, events=None, force: bool = False):
        return None

    def run(self, argv, timeout=None, env=None):
        return subprocess.run(
            [sys.executable, *argv[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(self.shim)},
        )


def sections_of(bundle: Path) -> list[dict]:
    raw = bundle.read_bytes()
    end = struct.unpack("<Q", raw[24:32])[0]
    header = json.loads(raw[32:end])
    for s in header["sections"]:
        s["payload"] = raw[s["begin"] : s["end"]]
    return header["sections"]


PAYLOAD = {
    "contract_version": 1,
    "type": "choice",
    "question": "Which emotion?",
    "options": [{"id": "joy"}, {"id": "anger", "description": "cross"}],
    "none_option": "none_of_these",
    "scoring": "unique_prefix",
    "temperature": 1.37,
    "abstain_below": 0.78,
}


def test_the_section_is_written_and_reads_back_byte_for_byte(toy):
    shim, bundle = toy
    before = sections_of(bundle)

    written = write_section(bundle, PAYLOAD, env=ShimEnv(shim))

    assert written.written, written.reason
    after = sections_of(bundle)
    assert [s["payload"] for s in after[:2]] == [s["payload"] for s in before]
    assert after[2]["type"] == "GenericBinaryData"
    assert after[2]["items"] == [{"key": "name", "value": SECTION_NAME}]
    assert after[2]["payload"] == payload_bytes(PAYLOAD)
    assert json.loads(after[2]["payload"]) == PAYLOAD
    assert written.report["replaced"] == 0
    assert written.report["header_end"] > written.report["header_end_before"]


def test_a_second_build_replaces_the_section_rather_than_adding_one(toy):
    shim, bundle = toy
    write_section(bundle, PAYLOAD, env=ShimEnv(shim))
    changed = dict(PAYLOAD, temperature=2.0)

    written = write_section(bundle, changed, env=ShimEnv(shim))

    assert written.written, written.reason
    named = [s for s in sections_of(bundle) if s["type"] == "GenericBinaryData"]
    assert len(named) == 1
    assert json.loads(named[0]["payload"])["temperature"] == 2.0
    assert written.report["replaced"] == 1


def test_a_header_past_the_runtimes_limit_is_refused_and_the_file_left_alone(toy, monkeypatch):
    shim, bundle = toy
    original = bundle.read_bytes()
    monkeypatch.setattr(decision_section, "HEADER_MAX", 64)

    written = write_section(bundle, PAYLOAD, env=ShimEnv(shim))

    assert not written.written
    assert "LiteRT-LM does not read past" in str(written.reason)
    assert bundle.read_bytes() == original


def test_a_cut_bundle_is_refused(toy):
    shim, bundle = toy
    bundle.write_bytes(bundle.read_bytes()[:-100])

    written = write_section(bundle, PAYLOAD, env=ShimEnv(shim))

    assert not written.written
    assert "past the end" in str(written.reason)


def test_a_failing_script_is_reported_with_its_last_line(toy, tmp_path):
    _, bundle = toy

    written = write_section(bundle, PAYLOAD, env=ShimEnv(tmp_path / "nowhere"))

    assert not written.written
    assert "exited 1" in str(written.reason)
    assert "litert_lm_builder" in str(written.reason)


def test_the_payload_is_compact_utf8_json_in_the_order_written():
    payload = {"b": "é", "a": 1}
    assert payload_bytes(payload) == '{"b":"é","a":1}'.encode()


def test_the_dart_call_lists_the_trained_options_in_order_without_the_none_option():
    assert dart_call(PAYLOAD) == (
        "final decider = await model.createDecider(\n"
        "  options: [DeciderOption('joy'), DeciderOption('anger')],\n"
        ");"
    )


@pytest.mark.parametrize(
    ("text", "literal"),
    [("joy", "'joy'"), ("it's", "'it\\'s'"), ("$x", "'\\$x'"), ("a\\b", "'a\\\\b'")],
)
def test_ids_are_written_as_dart_literals_that_mean_themselves(text, literal):
    assert dart_string(text) == literal


# ---------------------------------------------------------------------------
# Through `build_bundle` and the CLI
# ---------------------------------------------------------------------------

REVISION = "0123456789abcdef0123456789abcdef01234567"


def a_contract():
    from litetune.bundle import Contract
    from litetune.prompt_mode import PromptMode

    return Contract(
        prompt_mode=PromptMode.RUNTIME_RENDERED,
        established_against={"litert-lm": "0.18.0"},
        base_model="google/gemma-3-270m-it",
        base_model_revision=REVISION,
    )


def test_a_decision_bundle_carries_the_section_in_its_copy_and_says_how_to_call_it(toy, tmp_path):
    from litetune.bundle import BundleRequest, build_bundle

    shim, model = toy
    original = model.read_bytes()

    result = build_bundle(
        BundleRequest(
            output_dir=tmp_path / "bundle",
            model=model,
            declarations=None,
            contract=a_contract(),
            decision=PAYLOAD,
            section_env=ShimEnv(shim),
        )
    )

    section = next(c for c in result.checks.checks if c.name == "decision section")
    assert section.outcome.value == "passed", section.detail
    assert model.read_bytes() == original, "the caller's file is never rewritten"
    shipped = tmp_path / "bundle" / model.name
    named = [s for s in sections_of(shipped) if s["type"] == "GenericBinaryData"]
    assert [json.loads(s["payload"]) for s in named] == [PAYLOAD]
    contract = json.loads((tmp_path / "bundle" / "contract.json").read_text())
    assert contract["decision"]["section"] == SECTION_NAME
    assert contract["decision"]["payload"] == PAYLOAD
    assert contract["decision"]["create_decider"] == dart_call(PAYLOAD)
    assert contract["declarations_sha256"] is None
    assert not (tmp_path / "bundle" / "declarations.json").exists()
    report = json.loads((tmp_path / "bundle" / "report.json").read_text())
    assert report["use"] == {"dart": dart_call(PAYLOAD)}
    member = next(m for m in result.members if m.name == model.name)
    assert member.content_sha256 != f"sha256:{__import__('hashlib').sha256(original).hexdigest()}"


def test_a_decision_bundle_of_a_checkpoint_directory_is_refused(toy, tmp_path):
    from litetune.bundle import BundleRequest, build_bundle

    shim, _ = toy
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")

    result = build_bundle(
        BundleRequest(
            output_dir=tmp_path / "bundle",
            model=checkpoint,
            declarations=None,
            contract=a_contract(),
            decision=PAYLOAD,
            section_env=ShimEnv(shim),
        )
    )

    section = next(c for c in result.checks.checks if c.name == "decision section")
    assert section.outcome.value == "failed"
    assert "is not a .litertlm" in section.detail
    assert result.status.value == "failed_harness"


@pytest.mark.parametrize(
    ("declarations", "decision", "message"),
    [(None, None, "or, for a decision model"), ("tools.json", PAYLOAD, "not both")],
)
def test_a_bundle_carries_declarations_or_a_decision(tmp_path, declarations, decision, message):
    from litetune.bundle import BundleError, BundleRequest

    with pytest.raises(BundleError, match=message):
        BundleRequest(
            output_dir=tmp_path / "bundle",
            model=tmp_path / "m.litertlm",
            declarations=tmp_path / declarations if declarations else None,
            contract=a_contract(),
            decision=decision,
        )


def _decision_manifest(tmp_path, model_sha256: str | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "litetune.verify/1",
                "status": "passed",
                "model": {"sha256": model_sha256} if model_sha256 else {},
                "scorer": {"name": "option-scoring"},
                "decision": {"section": {"name": SECTION_NAME, "payload": PAYLOAD}},
                "measurements": {
                    "candidate": {
                        "accuracy": 0.9,
                        "engine": {
                            "engine": "litert-lm",
                            "runtime_version": "0.18.0",
                            "cpu": {"model": "a CPU"},
                        },
                    },
                    "reference": {"accuracy": 0.91},
                },
                "limitations": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _bundle_argv(tmp_path, model, manifest, *extra):
    return [
        "bundle",
        "--output-dir",
        str(tmp_path / "bundle"),
        "--model",
        str(model),
        "--verify-manifest",
        str(manifest),
        "--base-model",
        "google/gemma-3-270m-it",
        "--base-model-revision",
        REVISION,
        "--prompt-mode",
        "runtime_rendered",
        *extra,
    ]


def test_the_cli_bundles_a_decision_model_without_declarations(toy, tmp_path, monkeypatch, capsys):
    from litetune import envs
    from litetune.cli import main

    shim, model = toy
    runner = ShimEnv(shim)
    monkeypatch.setattr(
        envs.StageEnv, "run", lambda self, argv, timeout=None, env=None: runner.run(argv, timeout)
    )
    monkeypatch.setattr(envs.StageEnv, "provision", lambda self, events=None, force=False: None)

    measured = hashlib.sha256(model.read_bytes()).hexdigest()
    code = main(_bundle_argv(tmp_path, model, _decision_manifest(tmp_path, measured)))

    out = capsys.readouterr().out
    assert code == 0, out
    assert "  use it with:" in out
    assert "    options: [DeciderOption('joy'), DeciderOption('anger')]," in out
    assert "--wire-convention was not given" not in out


def test_the_cli_refuses_declarations_beside_a_decision_manifest(toy, tmp_path, capsys):
    from litetune.cli import main

    _, model = toy
    tools = tmp_path / "tools.json"
    tools.write_text("[]")

    code = main(
        _bundle_argv(tmp_path, model, _decision_manifest(tmp_path), "--declarations", str(tools))
    )

    assert code == 4
    assert "drop --declarations" in capsys.readouterr().err


def test_the_cli_still_requires_declarations_for_any_other_model(toy, tmp_path, capsys):
    from litetune.cli import main

    _, model = toy
    manifest = tmp_path / "plain.json"
    manifest.write_text(json.dumps({"status": "passed", "scorer": {"name": "tool-call"}}))

    code = main(_bundle_argv(tmp_path, model, manifest))

    assert code == 4
    assert "--declarations is required" in capsys.readouterr().err


class ResultEnv:
    """A stage environment whose section script 'succeeds' with a result of the test's choosing."""

    name = "export"

    def __init__(self, result: dict):
        self.result = result

    def provision(self, events=None, force: bool = False):
        return None

    def run(self, argv, timeout=None, env=None):
        Path(argv[3]).write_text(json.dumps(self.result), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_a_section_whose_digest_is_not_the_payloads_is_not_taken_as_written(tmp_path):
    env = ResultEnv({"written": True, "reason": None, "payload_sha256": "0" * 64})

    written = write_section(tmp_path / "m.litertlm", PAYLOAD, env=env)

    assert not written.written
    assert "not the payload written" in str(written.reason)


TOY_CORRUPTING = TOY_BUILDER.replace(
    "        payloads.append(p.read_bytes())",
    "        data = p.read_bytes()\n"
    "        if s['section_type'] == 'GenericBinaryData':\n"
    "            data = data[:-1] + b'!'\n"
    "        payloads.append(data)",
)


def test_a_payload_the_builder_changed_is_refused_by_the_read_back(toy):
    shim, bundle = toy
    (shim / "litert_lm_builder" / "litertlm_builder.py").write_text(TOY_CORRUPTING)
    original = bundle.read_bytes()

    written = write_section(bundle, PAYLOAD, env=ShimEnv(shim))

    assert not written.written
    assert f"named {SECTION_NAME} read back" in str(written.reason)
    assert bundle.read_bytes() == original


def test_a_payload_that_is_not_finite_is_refused_rather_than_written_as_infinity():
    with pytest.raises(ValueError):
        payload_bytes({"abstain_below": float("-inf")})


def test_a_decision_bundle_of_another_file_than_the_verified_one_is_refused(toy, tmp_path):
    from litetune.bundle import BundleRequest, build_bundle

    shim, model = toy

    result = build_bundle(
        BundleRequest(
            output_dir=tmp_path / "bundle",
            model=model,
            declarations=None,
            contract=a_contract(),
            decision=PAYLOAD,
            section_env=ShimEnv(shim),
            verified_sha256="sha256:" + "0" * 64,
        )
    )

    section = next(c for c in result.checks.checks if c.name == "decision section")
    assert section.outcome.value == "failed"
    assert "is not the file verify measured" in section.detail
    report = json.loads((tmp_path / "bundle" / "report.json").read_text())
    assert report["use"] is None, "no call is printed for a file without the section"


def test_the_section_is_built_beside_the_artifact_not_in_the_system_temp(toy):
    """The script ends with an os.replace onto the artifact, which crosses no filesystem."""
    shim, bundle = toy
    seen = {}

    class Recording(ShimEnv):
        def run(self, argv, timeout=None, env=None):
            seen["work"] = Path(json.loads(Path(argv[2]).read_text())["work"])
            return super().run(argv, timeout, env)

    written = write_section(bundle, PAYLOAD, env=Recording(shim))

    assert written.written, written.reason
    assert seen["work"].parent == bundle.parent.absolute()
    assert seen["work"].name.startswith(".model-section-")
    assert not seen["work"].exists(), "the work directory is removed afterwards"


def test_the_cli_refuses_to_put_a_decision_into_a_file_verify_did_not_measure(
    toy, tmp_path, monkeypatch, capsys
):
    from litetune import envs
    from litetune.cli import main

    shim, model = toy
    runner = ShimEnv(shim)
    monkeypatch.setattr(
        envs.StageEnv, "run", lambda self, argv, timeout=None, env=None: runner.run(argv, timeout)
    )
    monkeypatch.setattr(envs.StageEnv, "provision", lambda self, events=None, force=False: None)

    code = main(_bundle_argv(tmp_path, model, _decision_manifest(tmp_path, "ab" * 32)))

    out = capsys.readouterr().out
    assert code == 4, out
    assert "is not the file verify measured" in out
    assert "use it with:" not in out
