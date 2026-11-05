"""`verify` on held-out rows that carry a picture.

The measurement itself runs on `conftest.FakeBackend`s; the backends' specs and
the driver scripts' picture handling are checked one by one, the scripts exec'd
against stubs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeBackend

from litetune import envs
from litetune.checks import Outcome
from litetune.evaluate import (
    _HF_GENERATE_SCRIPT,
    DataError,
    HuggingFaceBackend,
    LiteRtLmBackend,
    _litertlm_script,
    load_split,
)
from litetune.llm_images import PROCESSOR_FILES, RECORD_KEY, TEMPLATE_FILE, VISION_TOKENS
from litetune.prompt_mode import PromptMode
from litetune.rendering import (
    _REFERENCE_SCRIPT,
    _RUNTIME_SCRIPT,
    RENDERING_CHECK,
    RenderingComparison,
    RenderingProbe,
)
from litetune.verify import (
    PICTURES_CHECK,
    BackendPair,
    PictureInputs,
    Status,
    VerifyRequest,
    build_backends,
    run_verify,
)

TEMPLATE = "{{ bos_token }}{# the bundle's #}"
LABELS = ("healthy", "bean_rust")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split(tmp_path: Path, n: int = 12, pictures: bool = True) -> Path:
    split = tmp_path / "split"
    (split / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        row: dict[str, Any] = {"prompt": "which disease is this", "target": LABELS[i % 2]}
        if pictures:
            data = f"picture {i}".encode()
            (split / "images" / f"{_sha(data)}.jpg").write_bytes(data)
            row |= {"image": f"images/{_sha(data)}.jpg", "image_sha256": _sha(data)}
        rows.append(row)
    path = split / "heldout.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _checkpoint(tmp_path: Path, mode: str | None = "runtime_rendered", **record: Any) -> Path:
    model = tmp_path / "tuned" / "model"
    model.mkdir(parents=True, exist_ok=True)
    for name in PROCESSOR_FILES:
        (model / name).write_text(name, encoding="utf-8")
    (model / TEMPLATE_FILE).write_bytes(TEMPLATE.encode())
    body = {
        "vision_tokens": VISION_TOKENS,
        "template_sha256": _sha(TEMPLATE.encode()),
        "processor_sha256": {n: _sha(n.encode()) for n in PROCESSOR_FILES},
        **record,
    }
    sidecar: dict[str, Any] = {"base_model": "google/gemma-4-E2B-it", RECORD_KEY: body}
    if mode is not None:
        sidecar["prompt_mode"] = mode
    (model / "litetune.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return model


def _request(tmp_path: Path, reference: Path | str | None = None, **kwargs: Any) -> VerifyRequest:
    params: dict[str, Any] = {
        "model": tmp_path / "out" / "model.litertlm",
        "reference": str(reference if reference is not None else _checkpoint(tmp_path)),
        "data": _split(tmp_path),
        "scorer": "exact-text",
    }
    params.update(kwargs)
    return VerifyRequest(**params)


def _pair() -> BackendPair:
    texts = [LABELS[i % 2] for i in range(12)]
    mode = PromptMode.RUNTIME_RENDERED
    return BackendPair(
        candidate=FakeBackend(texts=texts, prompt_mode=mode, decode_enforced=False),
        reference=FakeBackend(texts=texts, prompt_mode=mode),
    )


@dataclass(frozen=True)
class _Seen:
    outcome: Outcome
    detail: str


def _check(result, name) -> _Seen:
    found = next(c for c in result.manifest["checks"] if c["name"] == name)
    return _Seen(Outcome(found["outcome"]), found["detail"])


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_a_split_with_pictures_carries_each_rows_picture_into_its_id(tmp_path):
    with_pictures = load_split(_split(tmp_path))
    assert with_pictures.images is not None
    assert [e.image.sha256 for e in with_pictures.examples] == [
        _sha(f"picture {i}".encode()) for i in range(12)
    ]
    text = load_split(_split(tmp_path / "text", pictures=False))
    assert text.images is None
    # The same prompts and targets are another sample once the pictures differ.
    assert with_pictures.id != text.id


def test_a_limited_split_keeps_the_first_rows_pictures(tmp_path):
    split = load_split(_split(tmp_path), limit=3)
    assert split.n == 3 and split.images is not None and len(split.images) == 3


def test_a_changed_picture_is_refused_by_the_split(tmp_path):
    path = _split(tmp_path)
    next((path.parent / "images").iterdir()).write_bytes(b"swapped")
    with pytest.raises(DataError, match="changed after `prepare` wrote the split"):
        load_split(path)


def test_a_missing_picture_is_refused_by_the_split(tmp_path):
    path = _split(tmp_path)
    next((path.parent / "images").iterdir()).unlink()
    with pytest.raises(DataError, match="does not exist"):
        load_split(path)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_pictures_are_measured_with_what_the_checkpoint_recorded(tmp_path):
    result = run_verify(_request(tmp_path), backends=_pair())
    assert result.status is not Status.FAILED_HARNESS, result.manifest["checks"]
    accepted = _check(result, PICTURES_CHECK)
    assert accepted.outcome is Outcome.PASSED
    pictures = result.manifest["harness"]["pictures"]
    assert pictures["pictures"] == 12 and pictures["vision_tokens"] == VISION_TOKENS
    assert pictures["template_sha256"] == _sha(TEMPLATE.encode())
    assert any("the held-out rows carry pictures" in t for t in result.manifest["limitations"])


@pytest.mark.parametrize(
    ("kwargs", "said"),
    [
        ({"backend": "gpu"}, "--backend gpu was asked"),
        ({"scorer": "tool-call"}, "--scorer tool-call"),
    ],
)
def test_what_pictures_cannot_be_measured_with_is_refused(tmp_path, kwargs, said):
    result = run_verify(_request(tmp_path, **kwargs), backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    refused = _check(result, PICTURES_CHECK)
    assert refused.outcome is Outcome.FAILED and said in refused.detail


def test_a_reference_that_records_no_pictures_is_refused(tmp_path):
    result = run_verify(_request(tmp_path, reference="google/gemma-4-E2B-it"), backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    assert "records no training on pictures" in _check(result, PICTURES_CHECK).detail


def test_a_prerendered_measurement_of_pictures_is_refused(tmp_path):
    reference = _checkpoint(tmp_path, mode=None)
    request = _request(tmp_path, reference=reference, prompt_mode=PromptMode.PRERENDERED)
    result = run_verify(request, backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    assert "the prompt mode is prerendered" in _check(result, PICTURES_CHECK).detail


def test_declarations_with_pictures_are_refused(tmp_path):
    tools = tmp_path / "tools.json"
    tools.write_text("[]", encoding="utf-8")
    result = run_verify(_request(tmp_path, declarations=tools), backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    assert "declarations were given" in _check(result, PICTURES_CHECK).detail


def test_a_processor_file_that_changed_is_refused(tmp_path):
    reference = _checkpoint(tmp_path)
    (reference / "processor_config.json").write_text('{"max_soft_tokens": 140}', encoding="utf-8")
    result = run_verify(_request(tmp_path, reference=reference), backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    assert "processor_config.json hashes to" in _check(result, PICTURES_CHECK).detail


def test_a_template_that_changed_is_refused(tmp_path):
    reference = _checkpoint(tmp_path)
    (reference / TEMPLATE_FILE).write_text("{{ other }}", encoding="utf-8")
    result = run_verify(_request(tmp_path, reference=reference), backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    assert "the template changed after `tune`" in _check(result, PICTURES_CHECK).detail


def test_an_unreadable_picture_record_is_refused(tmp_path):
    reference = _checkpoint(tmp_path, vision_tokens=70)
    result = run_verify(_request(tmp_path, reference=reference), backends=_pair())
    assert result.status is Status.FAILED_HARNESS
    assert "vision_tokens" in _check(result, PICTURES_CHECK).detail


def test_a_split_whose_picture_changed_is_an_error_before_anything_runs(tmp_path):
    request = _request(tmp_path)
    next((Path(request.data).parent / "images").iterdir()).write_bytes(b"swapped")
    pair = _pair()
    result = run_verify(request, backends=pair)
    assert result.status is Status.ERROR
    assert pair.candidate.prompts_seen == [] and pair.reference.prompts_seen == []


def test_the_real_backends_are_given_the_pictures(tmp_path):
    request = _request(tmp_path, prompt_mode=PromptMode.RUNTIME_RENDERED)
    pictures = PictureInputs(
        images=({"path": "/p/a.jpg", "sha256": "a" * 64},),
        template=TEMPLATE,
        vision_tokens=VISION_TOKENS,
        template_sha256=_sha(TEMPLATE.encode()),
    )
    pair = build_backends(request, None, pictures)
    assert isinstance(pair.candidate, LiteRtLmBackend)
    assert pair.candidate.images == pictures.images
    assert isinstance(pair.reference, HuggingFaceBackend)
    assert pair.reference.env is envs.TRAIN_VISION
    assert pair.reference.images == pictures.images
    assert pair.reference.template == TEMPLATE
    assert pair.reference.vision_tokens == VISION_TOKENS
    assert isinstance(pair.rendering, RenderingProbe)
    assert pair.rendering.images == pictures.images
    assert pair.rendering.template == TEMPLATE
    assert pair.rendering.reference_env is envs.TRAIN_VISION
    # And a text measurement is built as it was.
    text = build_backends(request)
    assert text.reference.env is envs.TRAIN and text.candidate.images is None
    assert text.rendering.images is None and text.rendering.reference_env is envs.TRAIN


# ---------------------------------------------------------------------------
# The backends' specs
# ---------------------------------------------------------------------------


class SpecEnv:
    """Records the spec a backend wrote and answers with a failed run."""

    name = "fake"
    requirements: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.specs: list[dict] = []

    def system_requirements_on(self) -> tuple[str, ...]:
        return ()

    def run(self, argv, timeout=None, env=None):
        self.specs.append(json.loads(Path(argv[2]).read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(argv, 1, "", "boom")


PICTURES = ({"path": "/p/a.jpg", "sha256": "a" * 64}, {"path": "/p/b.jpg", "sha256": "b" * 64})


def test_the_candidate_is_sent_each_picture_on_a_cpu_vision_backend():
    env = SpecEnv()
    backend = LiteRtLmBackend(
        model=Path("m.litertlm"),
        env=env,  # type: ignore[arg-type]
        auto_provision=False,
        declared_prompt_mode=PromptMode.RUNTIME_RENDERED,
        images=PICTURES,
    )
    backend.generate(["one", "two"])
    (spec,) = env.specs
    assert spec["images"] == list(PICTURES)
    assert spec["vision_backend"] == "cpu"
    assert backend.describe()["pictures"] == 2
    assert backend.describe()["vision_backend"] == "cpu"


def test_a_text_candidate_has_no_vision_backend():
    env = SpecEnv()
    LiteRtLmBackend(model=Path("m"), env=env, auto_provision=False).generate(["one"])  # type: ignore[arg-type]
    assert "images" not in env.specs[0] and "vision_backend" not in env.specs[0]


def test_a_candidate_given_the_wrong_number_of_pictures_runs_nothing():
    env = SpecEnv()
    backend = LiteRtLmBackend(model=Path("m"), env=env, auto_provision=False, images=PICTURES[:1])  # type: ignore[arg-type]
    generations = backend.generate(["one", "two"])
    assert env.specs == []
    assert all("1 pictures were given for 2 prompts" in g.harness_error for g in generations)


def test_the_reference_is_given_the_pictures_the_template_and_the_budget(monkeypatch):
    env = SpecEnv()
    monkeypatch.setattr(HuggingFaceBackend, "_ensure_env", lambda self, events: None)
    backend = HuggingFaceBackend(
        model="m",
        env=env,  # type: ignore[arg-type]
        declared_prompt_mode=PromptMode.RUNTIME_RENDERED,
        images=PICTURES,
        template=TEMPLATE,
        vision_tokens=VISION_TOKENS,
    )
    backend.generate(["one", "two"])
    (spec,) = env.specs
    assert spec["images"] == list(PICTURES)
    assert spec["template"] == TEMPLATE
    assert spec["vision_tokens"] == VISION_TOKENS
    described = backend.describe()
    assert described["pictures"] == 2 and described["template_sha256"] == _sha(TEMPLATE.encode())


@pytest.mark.parametrize(
    "kwargs",
    [{"images": PICTURES[:1]}, {"template": None}, {"vision_tokens": None}],
)
def test_a_reference_missing_what_pictures_need_runs_nothing(monkeypatch, kwargs):
    env = SpecEnv()
    monkeypatch.setattr(HuggingFaceBackend, "_ensure_env", lambda self, events: None)
    params = {"images": PICTURES, "template": TEMPLATE, "vision_tokens": VISION_TOKENS} | kwargs
    generations = HuggingFaceBackend(model="m", env=env, **params).generate(["one", "two"])  # type: ignore[arg-type]
    assert env.specs == []
    assert all(g.harness_error and "needs its own" in g.harness_error for g in generations)


# ---------------------------------------------------------------------------
# The driver scripts
# ---------------------------------------------------------------------------


def test_the_runtime_is_sent_the_apps_message(tmp_path):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"jpeg bytes")
    message = _litertlm_script()["picture_message"](
        "which disease", {"path": str(picture), "sha256": _sha(b"jpeg bytes")}
    )
    assert message == {
        "role": "user",
        "content": [
            {"type": "image", "blob": base64.b64encode(b"jpeg bytes").decode("ascii")},
            {"type": "text", "text": "which disease"},
        ],
    }
    with pytest.raises(ValueError, match="is not the one the split names"):
        _litertlm_script()["picture_message"]("q", {"path": str(picture), "sha256": "0" * 64})


def test_a_row_with_a_picture_goes_to_the_conversation_as_that_message(tmp_path):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"jpeg bytes")
    sent: list[Any] = []

    class Conversation:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def send_message_async(self, message):
            sent.append(message)
            return iter([{"content": [{"type": "text", "text": "healthy"}]}])

    class Engine:
        def create_conversation(self, sampler_config=None):
            return Conversation()

    out = tmp_path / "rows.jsonl"
    with out.open("w", encoding="utf-8") as sink:
        _litertlm_script()["score_one"](
            Engine(), True, 0, "q", sink, {"path": str(picture), "sha256": _sha(b"jpeg bytes")}
        )
    assert sent[0]["content"][0]["type"] == "image"
    assert json.loads(out.read_text()) == {"index": 0, "text": "healthy"}


def test_the_engine_gets_a_vision_backend_only_for_pictures(monkeypatch):
    interfaces = types.ModuleType("litert_lm.interfaces")
    interfaces.CPU = lambda thread_count=None: ("CPU", thread_count)  # type: ignore[attr-defined]
    interfaces.GPU = lambda: ("GPU",)  # type: ignore[attr-defined]
    package = types.ModuleType("litert_lm")
    package.interfaces = interfaces  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litert_lm", package)
    monkeypatch.setitem(sys.modules, "litert_lm.interfaces", interfaces)
    kwargs = _litertlm_script()["engine_kwargs"]
    assert kwargs({"backend": "cpu", "vision_backend": "cpu"})["vision_backend"] == ("CPU", None)
    assert "vision_backend" not in kwargs({"backend": "cpu"})


def test_the_reference_renders_a_picture_row_with_the_bundle_template(tmp_path, monkeypatch):
    calls: list[dict] = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            calls.append({"messages": messages, **kwargs})
            return "<bos>rendered"

    class Processor:
        def __call__(self, **kwargs):
            calls.append({"processor": kwargs})
            return {"input_ids": [[2, 3]]}

    class Picture:
        def load(self):
            return None

        def convert(self, mode):
            calls.append({"convert": mode})
            return self

    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(open=lambda handle: Picture())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "PIL", pil)
    namespace: dict[str, Any] = {"__name__": "hf_script"}
    exec(compile(_HF_GENERATE_SCRIPT, "hf", "exec"), namespace)  # noqa: S102
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"jpeg")
    namespace["picture_inputs"](
        Tokenizer(), Processor(), "q", {"path": str(picture), "sha256": _sha(b"jpeg")}, TEMPLATE
    )
    assert calls[1]["messages"] == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "q"}]}
    ]
    assert calls[1]["chat_template"] == TEMPLATE
    assert calls[1]["add_generation_prompt"] is True
    assert calls[2]["processor"]["text"] == "<bos>rendered"
    assert calls[2]["processor"]["add_special_tokens"] is False
    with pytest.raises(ValueError, match="is not the one the split names"):
        namespace["picture_inputs"](
            Tokenizer(), Processor(), "q", {"path": str(picture), "sha256": "0" * 64}, TEMPLATE
        )


# ---------------------------------------------------------------------------
# The rendering check
# ---------------------------------------------------------------------------


def test_a_bundle_without_a_vision_encoder_fails_the_rendering_check():
    check = RenderingComparison(compared=0, vision_sections=("embedder", "prefill_decode")).check()
    assert check.name == RENDERING_CHECK and check.outcome is Outcome.FAILED
    assert "the bundle has no vision encoder" in check.detail


def test_a_bundle_with_one_is_judged_on_its_ids():
    sections = ("vision_encoder", "vision_adapter", "end_of_vision")
    check = RenderingComparison(compared=3, vision_sections=sections).check()
    assert check.outcome is Outcome.PASSED
    assert RenderingComparison(compared=3).as_dict()["vision_sections"] is None


class ScriptEnv:
    """Runs a rendering script's spec through a canned answer, per environment."""

    def __init__(self, name: str, rows: list[dict], sections: list[str] | None = None):
        self.name = name
        self.rows = rows
        self.sections = sections
        self.specs: list[dict] = []

    def provision(self, events=None):
        return None

    def run(self, argv, timeout=None):
        spec = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        self.specs.append(spec)
        Path(spec["out"]).write_text(json.dumps(self.rows), encoding="utf-8")
        if "sections_out" in spec and self.sections is not None:
            Path(spec["sections_out"]).write_text(json.dumps(self.sections), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _probe(runtime: ScriptEnv, reference: ScriptEnv, **kwargs) -> RenderingProbe:
    return RenderingProbe(
        model=Path("m.litertlm"),
        reference="ref",
        runtime_env=runtime,  # type: ignore[arg-type]
        reference_env=reference,  # type: ignore[arg-type]
        **kwargs,
    )


def test_the_probe_renders_pictures_without_a_prefill_count_and_with_the_template():
    rows = [{"index": 0, "rendered": "r", "ids": [2, 9]}]
    runtime = ScriptEnv("runtime", rows, ["vision_encoder", "vision_adapter", "end_of_vision"])
    reference = ScriptEnv("reference", rows)
    comparison = _probe(runtime, reference, images=PICTURES[:1], template=TEMPLATE).observe(["q"])
    assert comparison.check().outcome is Outcome.PASSED
    assert runtime.specs[0]["images"] == list(PICTURES[:1])
    assert runtime.specs[0]["prefill_sample"] == 0
    assert reference.specs[0]["template"] == TEMPLATE
    assert comparison.vision_sections == ("vision_encoder", "vision_adapter", "end_of_vision")


def test_the_probe_stops_at_a_bundle_without_a_vision_encoder():
    runtime = ScriptEnv("runtime", [], ["embedder", "prefill_decode"])
    reference = ScriptEnv("reference", [])
    comparison = _probe(runtime, reference, images=PICTURES[:1], template=TEMPLATE).observe(["q"])
    assert comparison.check().outcome is Outcome.FAILED
    assert reference.specs == []


def test_a_text_probe_asks_for_neither():
    rows = [{"index": 0, "rendered": "r", "ids": [2, 9], "prefill_tokens": 2}]
    runtime, reference = ScriptEnv("runtime", rows), ScriptEnv("reference", rows)
    _probe(runtime, reference).observe(["q"])
    assert "images" not in runtime.specs[0] and "sections_out" not in runtime.specs[0]
    assert "template" not in reference.specs[0]


def test_the_runtime_script_reads_the_sections_before_any_engine(tmp_path, monkeypatch):
    built: list[Any] = []
    lm = types.ModuleType("litert_lm")
    lm.Engine = lambda *a, **k: built.append(k)  # type: ignore[attr-defined]
    lm.Backend = types.SimpleNamespace(CPU=lambda **k: "cpu")  # type: ignore[attr-defined]
    lm.SamplerConfig = lambda **k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litert_lm", lm)
    namespace: dict[str, Any] = {"__name__": "render_script"}
    exec(compile(_RUNTIME_SCRIPT, "render", "exec"), namespace)  # noqa: S102
    namespace["section_types"] = lambda path: ["embedder", "prefill_decode"]
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "model": "m",
                "prompts": ["q"],
                "prefill_sample": 0,
                "images": list(PICTURES[:1]),
                "sections_out": str(tmp_path / "sections.json"),
                "out": str(tmp_path / "rows.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["render", str(spec)])
    assert namespace["main"]() == 0
    assert built == []
    assert json.loads((tmp_path / "rows.json").read_text()) == []
    assert json.loads((tmp_path / "sections.json").read_text()) == ["embedder", "prefill_decode"]


def test_the_reference_script_renders_pictures_with_the_template(tmp_path, monkeypatch):
    seen: list[dict] = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append({"messages": messages, **kwargs})
            return "<bos>x"

        def __call__(self, text, add_special_tokens=True):
            seen.append({"add_special_tokens": add_special_tokens})
            return {"input_ids": [2, 7]}

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda m: Tokenizer())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    namespace: dict[str, Any] = {"__name__": "reference_script"}
    exec(compile(_REFERENCE_SCRIPT, "reference", "exec"), namespace)  # noqa: S102
    spec = tmp_path / "spec.json"
    out = tmp_path / "rows.json"
    spec.write_text(
        json.dumps({"model": "m", "prompts": ["q"], "template": TEMPLATE, "out": str(out)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["reference", str(spec)])
    assert namespace["main"]() == 0
    assert seen[0]["chat_template"] == TEMPLATE
    assert seen[0]["messages"][0]["content"][0] == {"type": "image"}
    assert seen[1] == {"add_special_tokens": False}
    assert json.loads(out.read_text()) == [{"index": 0, "rendered": "<bos>x", "ids": [2, 7]}]
