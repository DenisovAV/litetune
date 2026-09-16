"""Recipes litetune defines itself, shipped as quantizer recipe files.

The file is what the export toolchain loads, so it is tested as the file: read
through the installed package's resources, not the source tree, and held to
the rules Google's quantization guide gives a decoder -- "int4 blockwise-32 +
OCTAV, embeddings int8" -- rather than to whatever the file happens to say.
"""

from __future__ import annotations

import json
from importlib.resources import files

from litetune.recipes import (
    DEFINED_RECIPES,
    LITETUNE,
    TOOLCHAIN,
    definition_of,
    source_of,
    toolchain_argument,
)

OWNED = "dynamic_wi4b32_emb8_afp32"


def _rules() -> list[dict]:
    text = (files("litetune.recipes") / f"{OWNED}.json").read_text(encoding="utf-8")
    rules = json.loads(text)
    assert isinstance(rules, list)
    return rules


def _rule(operation: str) -> dict:
    (rule,) = [r for r in _rules() if r["operation"] == operation]
    return rule


def test_the_recipe_has_one_rule_for_linear_layers_and_one_for_embeddings():
    assert sorted(r["operation"] for r in _rules()) == ["EMBEDDING_LOOKUP", "FULLY_CONNECTED"]
    assert all(r["regex"] == ".*" for r in _rules())


def test_linear_layers_are_4_bit_block_32_with_octav():
    rule = _rule("FULLY_CONNECTED")
    weights = rule["op_config"]["weight_tensor_config"]
    assert rule["algorithm_key"] == "OCTAV"
    assert weights["num_bits"] == 4
    assert weights["granularity"] == "BLOCKWISE_32"
    # LiteRT's dynamic integer kernels require symmetric weights.
    assert weights["symmetric"] is True
    assert rule["op_config"]["compute_precision"] == "INTEGER"
    assert rule["op_config"]["explicit_dequantize"] is False


def test_embeddings_stay_8_bit_channelwise():
    rule = _rule("EMBEDDING_LOOKUP")
    weights = rule["op_config"]["weight_tensor_config"]
    assert weights["num_bits"] == 8
    assert weights["granularity"] == "CHANNELWISE"
    assert weights["symmetric"] is True
    assert rule["op_config"]["compute_precision"] == "INTEGER"


def test_every_defined_recipe_hands_the_toolchain_its_own_packaged_file():
    assert OWNED in DEFINED_RECIPES
    for name, recipe in DEFINED_RECIPES.items():
        assert recipe.name == name
        assert recipe.describes
        packaged = files("litetune.recipes") / f"{name}.json"
        assert toolchain_argument(name) == str(recipe.path) == str(packaged)
        assert source_of(name) == LITETUNE
        assert definition_of(name) == json.loads(packaged.read_text(encoding="utf-8"))


def test_a_toolchain_preset_passes_through_with_no_definition():
    preset = "dynamic_wi8_afp32"
    assert source_of(preset) == TOOLCHAIN
    assert toolchain_argument(preset) == preset
    assert definition_of(preset) is None
