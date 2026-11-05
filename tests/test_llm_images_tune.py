"""`tune` on Gemma 4 E2B rows that carry a picture: the parent's choices and the real script.

The parent is driven with `test_tune`'s `FakeTrainer`. The script runs for real
against stubs of torch, transformers, peft, PIL and huggingface_hub written here:
the processor stub expands the template's image placeholder into
`LITETUNE_STUB_SOFT` soft tokens between the two markers, and the hub stub serves
a fixed bundle template from a snapshot directory, as `hf_hub_download` does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import mark_provisioned
from test_tune import (
    _STUB_PEFT,
    _STUB_TORCH,
    FakeTrainer,
    check_named,
    run_real_script,
    stub_log,
)

from litetune import envs
from litetune.checks import Outcome
from litetune.llm_images import (
    FROZEN_PREFIXES,
    PROCESSOR_FILES,
    TEMPLATE_FILE,
    VISION_TOKENS,
    image_record,
)
from litetune.prompt_mode import PromptMode, PromptModeDecision
from litetune.tune import (
    FROZEN_CHECK,
    IMAGE_MICRO_BATCH_SOURCE,
    IMAGES_CHECK,
    TRAINING_CHECK,
    TuneRequest,
    decide_micro_batch_size,
    run_tune,
)

MODEL = "google/gemma-4-E2B-it"
TEMPLATE_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
BUNDLE_TEMPLATE = "{{ bos_token }}<|turn>user\n{# the bundle's #}"
REVISION = "b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1"


def _split(tmp_path: Path, n: int = 8, prompt: str = "which disease is this") -> Path:
    split = tmp_path / "split"
    images = split / "images"
    images.mkdir(parents=True)
    lines = []
    for i in range(n):
        data = f"picture {i}".encode()
        sha = hashlib.sha256(data).hexdigest()
        (images / f"{sha}.jpg").write_bytes(data)
        lines.append(
            json.dumps(
                {
                    "prompt": prompt,
                    "completion": "bean_rust" if i % 2 else "healthy",
                    "target": None,
                    "image": f"images/{sha}.jpg",
                    "image_sha256": sha,
                    "source_line": i + 1,
                }
            )
        )
    (split / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return split / "train.jsonl"


def _request(tmp_path: Path, **kwargs: Any) -> TuneRequest:
    params: dict[str, Any] = {
        "model": MODEL,
        "data": kwargs.pop("data", None) or _split(tmp_path),
        "output_dir": tmp_path / "tuned",
        "method": "lora",
    }
    params.update(kwargs)
    return TuneRequest(**params)


# ---------------------------------------------------------------------------
# The parent
# ---------------------------------------------------------------------------


@pytest.fixture
def trainer(monkeypatch, tmp_path) -> FakeTrainer:
    """`test_tune`'s fake training environment, every stage environment answering."""
    monkeypatch.setenv("LITETUNE_ENV_DIR", str(tmp_path / "envs"))

    def fake_provision(self, events=None, force: bool = False) -> Path:
        mark_provisioned(self)
        return self.path

    fake = FakeTrainer()
    monkeypatch.setattr(envs.StageEnv, "provision", fake_provision)
    monkeypatch.setattr(envs.StageEnv, "run", fake)
    return fake


def test_a_split_with_pictures_trains_in_train_vision_with_the_picture_spec(trainer, tmp_path):
    result = run_tune(_request(tmp_path))
    assert check_named(result, IMAGES_CHECK).outcome is Outcome.PASSED
    assert result.request.env is envs.TRAIN_VISION
    assert result.request.trains_pictures
    (config,) = trainer.configs
    assert config["images"] == {
        "vision_tokens": VISION_TOKENS,
        "template_repo": TEMPLATE_REPO,
        "template_file": TEMPLATE_FILE,
        "processor_files": list(PROCESSOR_FILES),
        "frozen_prefixes": list(FROZEN_PREFIXES),
    }
    assert config["prompt_mode"] == "runtime_rendered"
    assert config["lora_container"] == "language_model"
    # One example per forward unless the caller said otherwise.
    assert config["micro_batch_size"] == 1
    assert config["micro_batch_size_source"] == IMAGE_MICRO_BATCH_SOURCE
    report = result.as_dict()
    assert report["request"]["environment"]["name"] == envs.TRAIN_VISION.name
    assert report["request"]["micro_batch_size"] == 1


def test_a_text_split_keeps_train(trainer, tmp_path):
    data = tmp_path / "text.jsonl"
    data.write_text(
        "".join(json.dumps({"prompt": f"q {i}", "completion": "a"}) + "\n" for i in range(8)),
        encoding="utf-8",
    )
    result = run_tune(_request(tmp_path, data=data))
    assert result.request.env is envs.TRAIN
    assert trainer.configs[0]["images"] is None
    assert not any(c.name == IMAGES_CHECK for c in result.checks.checks)


def test_a_declared_micro_batch_is_kept_for_pictures():
    assert decide_micro_batch_size(4, 8, "cuda", pictures=True).size == 4
    assert decide_micro_batch_size(None, 8, "cuda", pictures=True).size == 1
    assert decide_micro_batch_size(None, 8, "cuda").size == 8


@pytest.mark.parametrize(
    ("kwargs", "said"),
    [
        ({"model": "google/gemma-3-270m-it"}, "is not google/gemma-4-E2B-it"),
        ({"model": "google/gemma-4-E4B-it"}, "is not google/gemma-4-E2B-it"),
        ({"method": "full"}, "--method full was asked"),
        (
            {"prompt_mode": PromptMode.PRERENDERED, "force_prompt_mode": True},
            "the prompt mode is prerendered",
        ),
    ],
)
def test_what_cannot_train_pictures_is_refused_before_the_environment(
    trainer, tmp_path, kwargs, said
):
    result = run_tune(_request(tmp_path, **kwargs))
    refused = check_named(result, IMAGES_CHECK)
    assert refused.outcome is Outcome.FAILED
    assert said in refused.detail
    assert trainer.calls == []


def test_declarations_with_pictures_are_refused(trainer, tmp_path):
    tools = tmp_path / "tools.json"
    tools.write_text("[]", encoding="utf-8")
    result = run_tune(_request(tmp_path, declarations=tools))
    assert "declarations were given" in check_named(result, IMAGES_CHECK).detail
    assert trainer.calls == []


def test_a_decision_with_pictures_is_refused(trainer, tmp_path):
    from litetune.tune import _refuse_pictures_here

    request = _request(tmp_path, decision=tmp_path / "decision.json")
    refused = _refuse_pictures_here(request, PromptMode.RUNTIME_RENDERED)
    assert refused is not None and "a decision was given" in refused.detail


def test_a_family_with_no_template_repository_is_refused(monkeypatch, tmp_path):
    from litetune import tune

    monkeypatch.setattr(tune, "bundle_template_repo", lambda model: None)
    refused = tune._refuse_pictures_here(_request(tmp_path), PromptMode.RUNTIME_RENDERED)
    assert refused is not None and "names no chat template repository" in refused.detail


def test_a_picture_changed_since_prepare_is_refused_before_the_environment(trainer, tmp_path):
    data = _split(tmp_path)
    picture = next((data.parent / "images").iterdir())
    picture.write_bytes(b"another picture")
    result = run_tune(_request(tmp_path, data=data))
    refused = check_named(result, IMAGES_CHECK)
    assert refused.outcome is Outcome.FAILED
    assert "changed after `prepare` wrote the split" in refused.detail
    assert trainer.calls == []


def _frozen(checked: int, changed=(), missing=()) -> dict:
    return {
        "images": {
            "frozen": {"checked": checked, "changed": list(changed), "missing": list(missing)}
        }
    }


def test_an_unchanged_merge_passes_the_frozen_check(trainer, tmp_path):
    trainer.metrics_extra = _frozen(1413)
    result = run_tune(_request(tmp_path))
    frozen = check_named(result, FROZEN_CHECK)
    assert frozen.outcome is Outcome.PASSED
    assert "all 1413 tensors" in frozen.detail


@pytest.mark.parametrize(
    ("metrics", "outcome", "said"),
    [
        (_frozen(1413, changed=["model.vision_tower.x"]), Outcome.FAILED, "1 changed"),
        (_frozen(1413, missing=["model.embed_vision.y"]), Outcome.FAILED, "1 are missing"),
        (_frozen(0), Outcome.FAILED, "none of the names"),
        ({}, Outcome.UNCHECKED, "reported no comparison"),
    ],
)
def test_the_frozen_check_says_what_the_merge_did(trainer, tmp_path, metrics, outcome, said):
    trainer.metrics_extra = metrics
    result = run_tune(_request(tmp_path))
    frozen = check_named(result, FROZEN_CHECK)
    assert frozen.outcome is outcome
    assert said in frozen.detail
    assert result.outcome is not Outcome.PASSED


def test_a_text_run_has_no_frozen_check(trainer, tmp_path):
    data = tmp_path / "text.jsonl"
    data.write_text(
        "".join(json.dumps({"prompt": f"q {i}", "completion": "a"}) + "\n" for i in range(8)),
        encoding="utf-8",
    )
    result = run_tune(_request(tmp_path, data=data))
    assert check_named(result, TRAINING_CHECK).outcome is Outcome.PASSED
    assert not any(c.name == FROZEN_CHECK for c in result.checks.checks)


# ---------------------------------------------------------------------------
# The real script, against stubs
# ---------------------------------------------------------------------------

_STUB_TORCH_IMAGES = (
    _STUB_TORCH
    + '''
uint8 = "uint8"


def _shape(self):
    return (len(self), len(self[0]) if self and isinstance(self[0], list) else 0)


_Tensor.shape = property(_shape)
_Tensor.is_floating_point = lambda self: getattr(self, "floating", False)


def cat(tensors, dim=0):
    out = _Tensor([row for t in tensors for row in t])
    out.floating = any(getattr(t, "floating", False) for t in tensors)
    return out


class _Weight:
    """A state_dict tensor: hashed by the bytes its content names."""

    def __init__(self, content):
        self.content = content

    def detach(self):
        return self

    def to(self, device):
        return self

    def contiguous(self):
        return self

    def reshape(self, *shape):
        return self

    def view(self, dtype):
        return self

    def numpy(self):
        return self.content.encode()
'''
)

_STUB_TRANSFORMERS_IMAGES = """
import json
import os
import re
from pathlib import Path

import torch

SPECIAL = {"<bos>": 2, "<|image>": 255999, "<|image|>": 258880, "<image|>": 258882,
           "<turn|>": 106}


def _log(record):
    with open(os.environ["LITETUNE_STUB_LOG"], "a", encoding="utf-8") as sink:
        sink.write(json.dumps(record) + "\\n")


def _ids(text):
    out = []
    for piece in re.split(r"(<bos>|<\\|image>|<\\|image\\|>|<image\\|>|<turn\\|>)", text):
        if piece in SPECIAL:
            out.append(SPECIAL[piece])
        else:
            out.extend(1000 + len(word) for word in piece.split())
    return out


class _Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": ([2] if add_special_tokens else []) + _ids(text)}

    def apply_chat_template(self, messages, chat_template=None, tokenize=False,
                            add_generation_prompt=False, tools=None):
        _log({"event": "chat_template", "template": chat_template, "messages": messages})
        out = "<bos>user "
        content = messages[0]["content"]
        if isinstance(content, str):
            out += content
        else:
            for part in content:
                out += "\\n\\n<|image|>\\n\\n" if part["type"] == "image" else part["text"]
        out += "<turn|>\\nmodel "
        if len(messages) > 1:
            out += messages[1]["content"] + "<turn|>\\n"
        return out

    def decode(self, ids):
        return " ".join(f"<{i}>" for i in ids)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (Path(path) / name).write_text(name, encoding="utf-8")


class Gemma4ImageProcessor:
    max_soft_tokens = 140


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.image_processor = Gemma4ImageProcessor()

    def __call__(self, text=None, images=None, return_tensors=None, add_special_tokens=True):
        soft = int(os.environ.get("LITETUNE_STUB_SOFT", "4"))
        _log({"event": "processor", "text": text, "images": len(images),
              "max_soft_tokens": self.image_processor.max_soft_tokens,
              "add_special_tokens": add_special_tokens})
        expanded = text.replace("<|image|>", "<|image>" + "<|image|>" * soft + "<image|>")
        ids = _ids(expanded)
        pixels = torch._Tensor([[0.5] * 3])
        pixels.floating = True
        return {
            "input_ids": torch._Tensor([torch._Tensor(ids)]),
            "attention_mask": torch._Tensor([torch._Tensor([1] * len(ids))]),
            "pixel_values": pixels,
            "image_position_ids": torch._Tensor([[0, 0]]),
            "mm_token_type_ids": torch._Tensor(
                [torch._Tensor([1 if i == 258880 else 0 for i in ids])]
            ),
        }

    def save_pretrained(self, path):
        (Path(path) / "processor_config.json").write_text(
            json.dumps({"max_soft_tokens": self.image_processor.max_soft_tokens}),
            encoding="utf-8",
        )


class AutoProcessor:
    @staticmethod
    def from_pretrained(model, **kwargs):
        _log({"event": "load_processor", "model": model})
        return _Processor()


class AutoTokenizer:
    @staticmethod
    def from_pretrained(model, **kwargs):
        raise AssertionError("an image run takes the processor's tokenizer")


class _Logits:
    def __init__(self, rows, positions):
        self.rows = rows
        self.positions = positions
        self.shape = (rows, positions, 7)

    def float(self):
        return self

    def reshape(self, *shape):
        return self


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _Config:
    use_cache = True


class _Parameter:
    requires_grad = True

    def numel(self):
        return 1000


def _state(tag):
    names = [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        "model.embed_vision.embedding_projection.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
    ]
    out = {name: torch._Weight(name) for name in names}
    if tag == "merged" and os.environ.get("LITETUNE_STUB_CHANGE_FROZEN"):
        out[names[1]] = torch._Weight("changed")
    if tag == "merged" and os.environ.get("LITETUNE_STUB_DROP_FROZEN"):
        del out[names[0]]
    return out


class _Model:
    def __init__(self, tag):
        self.tag = tag
        self.config = _Config()

    def parameters(self):
        return [_Parameter()]

    def train(self):
        return self

    def to(self, device):
        return self

    def gradient_checkpointing_enable(self):
        pass

    def state_dict(self):
        return _state(self.tag)

    def __call__(self, input_ids=None, attention_mask=None, logits_to_keep=None, **pictures):
        _log({"event": "forward", "rows": len(input_ids), "width": len(input_ids[0]),
              "kept": list(logits_to_keep),
              "pictures": {k: {"device": getattr(v, "device", None), "value": list(v)}
                           for k, v in pictures.items()}})
        return _Output(_Logits(len(input_ids), len(logits_to_keep)))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / (self.tag + ".json")).write_text("{}", encoding="utf-8")

    def merge_and_unload(self):
        return _Model("merged")


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(model, **kwargs):
        return _Model("full")
"""

_STUB_PIL = """
class _Picture:
    def load(self):
        return None

    def convert(self, mode):
        return self


class Image:
    @staticmethod
    def open(handle):
        handle.read()
        return _Picture()
"""

_STUB_HUB = """
import os
from pathlib import Path


def hf_hub_download(repo_id=None, filename=None, revision=None):
    if filename != "chat_template.jinja":
        raise FileNotFoundError(filename)
    with open(os.environ["LITETUNE_STUB_LOG"], "a", encoding="utf-8") as sink:
        import json
        sink.write(json.dumps({"event": "hub", "repo_id": repo_id, "revision": revision}) + "\\n")
    return os.environ["LITETUNE_STUB_TEMPLATE"]
"""


@pytest.fixture
def image_stubs(tmp_path):
    stubs = tmp_path / "stubs"
    (stubs / "PIL").mkdir(parents=True)
    (stubs / "torch.py").write_text(_STUB_TORCH_IMAGES, encoding="utf-8")
    (stubs / "transformers.py").write_text(_STUB_TRANSFORMERS_IMAGES, encoding="utf-8")
    (stubs / "peft.py").write_text(_STUB_PEFT, encoding="utf-8")
    (stubs / "PIL" / "__init__.py").write_text(_STUB_PIL, encoding="utf-8")
    (stubs / "huggingface_hub.py").write_text(_STUB_HUB, encoding="utf-8")
    snapshot = tmp_path / "hub" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "chat_template.jinja").write_bytes(BUNDLE_TEMPLATE.encode())
    return stubs, tmp_path / "stub.log"


def _run(tmp_path, image_stubs, extra_env: dict[str, str] | None = None, **kwargs: Any):
    request = _request(tmp_path, trains_pictures=True, env=envs.TRAIN_VISION, **kwargs)
    decision = PromptModeDecision(PromptMode.RUNTIME_RENDERED, "inferred", "bare prompts")
    template = tmp_path / "hub" / "snapshots" / REVISION / "chat_template.jinja"
    proc = run_real_script(
        request,
        image_stubs,
        decision=decision,
        lora_container="language_model",
        extra_env={"LITETUNE_STUB_TEMPLATE": str(template), **(extra_env or {})},
    )
    return request, proc


def test_the_script_trains_each_picture_through_the_bundle_template(tmp_path, image_stubs):
    request, proc = _run(tmp_path, image_stubs)
    assert proc.returncode == 0, proc.stderr
    log = stub_log(image_stubs)

    # The template is the one the export fetches, by repository and no revision.
    (hub,) = [e for e in log if e["event"] == "hub"]
    assert hub == {"event": "hub", "repo_id": TEMPLATE_REPO, "revision": None}
    rendered = [e for e in log if e["event"] == "chat_template"]
    assert rendered and all(e["template"] == BUNDLE_TEMPLATE for e in rendered)
    # Every row is one user turn: the picture, then the text.
    rows = [e for e in rendered if not isinstance(e["messages"][0]["content"], str)]
    assert len(rows) == 8
    assert rows[0]["messages"][0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "which disease is this"},
    ]
    processed = [e for e in log if e["event"] == "processor"]
    assert len(processed) == 8
    # At the bundle's budget, not the processor's own, and no second BOS.
    assert all(e["max_soft_tokens"] == VISION_TOKENS for e in processed)
    assert all(e["add_special_tokens"] is False for e in processed)

    # One example per forward, each given its picture's inputs.
    forwards = [e for e in log if e["event"] == "forward"]
    assert forwards and all(e["rows"] == 1 for e in forwards)
    pictures = forwards[0]["pictures"]
    assert set(pictures) == {"pixel_values", "image_position_ids", "mm_token_type_ids"}
    # Token types run the row's whole width: the soft tokens 1, the answer 0.
    types = pictures["mm_token_type_ids"]["value"][0]
    assert len(types) == forwards[0]["width"]
    assert sum(types) == 4 and types[-1] == 0
    # The pixels reach the model in the training dtype.
    assert pictures["pixel_values"]["device"] in ("cpu", "bfloat16")

    # The loss is the answer and the terminator, never the picture or the prompt.
    # The stub tokenizes a word as 1000 + its length, and `<turn|>` as 106: the
    # answer, then what the bundle template puts after it.
    targets = [e["targets"] for e in log if e["event"] == "cross_entropy"]
    assert len(targets) == len(forwards)
    assert {tuple(t[0]) for t in targets} == {
        (1000 + len("healthy"), 106),
        (1000 + len("bean_rust"), 106),
    }


def test_the_checkpoint_keeps_the_template_and_records_what_verify_checks(tmp_path, image_stubs):
    request, proc = _run(tmp_path, image_stubs)
    assert proc.returncode == 0, proc.stderr
    model = request.model_dir
    assert (model / TEMPLATE_FILE).read_bytes() == BUNDLE_TEMPLATE.encode()
    record = image_record(model)
    assert record is not None
    assert record["template_sha256"] == hashlib.sha256(BUNDLE_TEMPLATE.encode()).hexdigest()
    assert record["template_revision"] == REVISION
    assert record["template_repo"] == TEMPLATE_REPO
    assert record["image_processor"] == "Gemma4ImageProcessor"
    assert record["soft_tokens"] == {"min": 4, "max": 4, "rows": 8}
    for name in PROCESSOR_FILES:
        assert (
            record["processor_sha256"][name]
            == hashlib.sha256((model / name).read_bytes()).hexdigest()
        )
    # The processor saved at the budget it trained at.
    assert json.loads((model / "processor_config.json").read_text())["max_soft_tokens"] == 280
    assert record["frozen"] == {"checked": 3, "changed": [], "missing": []}
    metrics = json.loads((request.output_dir / "metrics.json").read_text())
    assert metrics["images"] == record


@pytest.mark.parametrize(
    ("switch", "field", "name"),
    [
        (
            "LITETUNE_STUB_CHANGE_FROZEN",
            "changed",
            "model.embed_vision.embedding_projection.weight",
        ),
        (
            "LITETUNE_STUB_DROP_FROZEN",
            "missing",
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
        ),
    ],
)
def test_the_script_names_a_frozen_tensor_the_merge_touched(
    tmp_path, image_stubs, switch, field, name
):
    request, proc = _run(tmp_path, image_stubs, extra_env={switch: "1"})
    assert proc.returncode == 0, proc.stderr
    frozen = json.loads((request.output_dir / "metrics.json").read_text())["images"]["frozen"]
    assert frozen[field] == [name]


def test_a_picture_changed_before_the_script_reads_it_stops_training(tmp_path, image_stubs):
    data = _split(tmp_path)
    request = _request(tmp_path, data=data, trains_pictures=True)
    next((data.parent / "images").iterdir()).write_bytes(b"swapped")
    template = tmp_path / "hub" / "snapshots" / REVISION / "chat_template.jinja"
    proc = run_real_script(
        request,
        image_stubs,
        decision=PromptModeDecision(PromptMode.RUNTIME_RENDERED, "inferred", "bare"),
        lora_container="language_model",
        extra_env={"LITETUNE_STUB_TEMPLATE": str(template)},
    )
    assert proc.returncode != 0
    assert "changed after prepare wrote the split" in proc.stderr


@pytest.mark.parametrize("soft", ["0", "281"])
def test_a_picture_outside_the_budget_stops_training(tmp_path, image_stubs, soft):
    _, proc = _run(tmp_path, image_stubs, extra_env={"LITETUNE_STUB_SOFT": soft})
    assert proc.returncode != 0
    assert f"became {soft} image tokens" in proc.stderr


def test_a_row_too_long_with_its_picture_stops_training(tmp_path, image_stubs):
    _, proc = _run(tmp_path, image_stubs, max_seq_length=12)
    assert proc.returncode != 0
    assert "tokens with its picture, over max_seq_length 12" in proc.stderr
