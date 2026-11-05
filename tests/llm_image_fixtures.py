"""What the picture tests share: a checkpoint `tune` could have written, and the pinned template.

The template sha256 `llm_images` pins is that of the real bundle template, which
these tests do not carry; `pin_template` points the pin at the test's own.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from litetune import llm_images, tune
from litetune.llm_images import (
    FROZEN_PREFIXES,
    PROCESSOR_FILES,
    RECORD_KEY,
    TEMPLATE_FILE,
    VISION_TOKENS,
    processor_digests,
)

TEMPLATE = "{{ bos_token }}{# the bundle's template #}"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_ok() -> dict[str, Any]:
    """A comparison that found every frozen prefix and nothing changed."""
    return {
        "checked": len(FROZEN_PREFIXES),
        "per_prefix": {prefix: 1 for prefix in FROZEN_PREFIXES},
        "changed": [],
        "missing": [],
    }


def pin_template(monkeypatch, template: str = TEMPLATE) -> str:
    digest = sha(template.encode())
    monkeypatch.setattr(llm_images, "TEMPLATE_SHA256", digest)
    monkeypatch.setattr(tune, "TEMPLATE_SHA256", digest)
    return digest


def write_checkpoint(
    model: Path,
    template: str = TEMPLATE,
    mode: str | None = "runtime_rendered",
    trained_template: str = TEMPLATE,
    **record: Any,
) -> Path:
    """A checkpoint directory with processor files, the kept template and its record."""
    model.mkdir(parents=True, exist_ok=True)
    for name in PROCESSOR_FILES:
        content = {"name": name} if name == "tokenizer_config.json" else name
        text = json.dumps(content) if isinstance(content, dict) else content
        (model / name).write_text(text, encoding="utf-8")
    (model / TEMPLATE_FILE).write_bytes(template.encode())
    (model / "config.json").write_text("{}", encoding="utf-8")
    body = {
        "vision_tokens": VISION_TOKENS,
        "template_sha256": sha(trained_template.encode()),
        "processor_sha256": processor_digests(model),
        "frozen": frozen_ok(),
        **record,
    }
    sidecar: dict[str, Any] = {"base_model": "google/gemma-4-E2B-it", RECORD_KEY: body}
    if mode is not None:
        sidecar["prompt_mode"] = mode
    (model / "litetune.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return model
