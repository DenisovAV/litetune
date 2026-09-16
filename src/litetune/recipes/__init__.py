"""Quantization recipes litetune defines itself.

Every other recipe name is the export toolchain's own preset, and litetune hands
it to `litert-torch export_hf --quantization_recipe` as given. A recipe defined
here is a quantizer recipe file shipped next to this module. Its name is what a
caller passes to `--recipe` and what the output directory, the check and the
report carry; its file is what the toolchain receives -- the pinned exporter
reads a `--quantization_recipe` ending in `.json` as a recipe file
(`export_lib.quantize_model`, litert-torch-nightly 0.10.0.dev20260826).

This is the one place that knows which names are litetune's: `export` asks it
what to pass the toolchain and what to record, and `cli` what to offer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

# What a report's `recipe_source` says about where the rules of a recipe came from.
LITETUNE = "litetune"
TOOLCHAIN = "toolchain"

_HERE = Path(str(files(__name__)))


class RecipeDefinitionError(ValueError):
    """A recipe litetune defines could not be read from the installed package."""


@dataclass(frozen=True)
class DefinedRecipe:
    """A recipe file litetune ships, and the one line `--help` says about it."""

    name: str
    describes: str
    path: Path


# `dynamic_wi4b32_emb8_afp32`: fully connected weights int4 in blocks of 32
# with OCTAV, embedding lookup int8 channelwise, float activations -- the
# layout Google's quantization guide gives a decoder ("int4 blockwise-32 +
# OCTAV, embeddings int8. Never channelwise for a decoder"). On the banking77
# checkpoints both channelwise 4-bit presets failed their gates, and the
# block-wise preset, which keeps embeddings at 4 bits, cost gemma-3-270m
# +0.3483 ±0.0503.
DEFINED_RECIPES: dict[str, DefinedRecipe] = {
    recipe.name: recipe
    for recipe in (
        DefinedRecipe(
            name="dynamic_wi4b32_emb8_afp32",
            describes="int4 weights in blocks of 32 with int8 embeddings",
            path=_HERE / "dynamic_wi4b32_emb8_afp32.json",
        ),
    )
}


def source_of(recipe: str) -> str:
    """`litetune` for a recipe defined here, `toolchain` for a name passed through as given."""
    return LITETUNE if recipe in DEFINED_RECIPES else TOOLCHAIN


def toolchain_argument(recipe: str) -> str:
    """What `--quantization_recipe` receives: the packaged file, or the name as given."""
    defined = DEFINED_RECIPES.get(recipe)
    return str(defined.path) if defined is not None else recipe


def definition_of(recipe: str) -> list[dict[str, Any]] | None:
    """The rules a defined recipe hands the toolchain; None for a toolchain name.

    Raises `RecipeDefinitionError` when the file of a defined recipe is missing,
    unreadable, or not a list of quantization rules.
    """
    defined = DEFINED_RECIPES.get(recipe)
    if defined is None:
        return None
    path = defined.path
    try:
        rules = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RecipeDefinitionError(
            f"the definition of {recipe} is missing from this installation: {path}"
        ) from None
    except (OSError, ValueError) as exc:
        raise RecipeDefinitionError(
            f"the definition of {recipe} at {path} could not be read: {exc}"
        ) from None
    if not (
        isinstance(rules, list)
        and rules
        and all(isinstance(rule, dict) and "operation" in rule for rule in rules)
    ):
        raise RecipeDefinitionError(
            f"the definition of {recipe} at {path} is not a list of quantization rules"
        )
    return rules
