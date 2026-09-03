"""The spec's rules, each of which is a failure that was observed rather than imagined.

A mutable base revision resolves to different weights on different days; a
dataset keyed by its URI trains on stale data under a green manifest; a gate
threshold inside a measurement key turns "tighten the bar" into hours of
regenerated output. Every test here breaks if one of those is refactored away.
"""

import pytest
from stage_fakes import DATA_A, DATA_B, HELDOUT_B, make_spec, spec_mapping

from litetune.spec import (
    IN_PROCESS,
    STAGE_SECTIONS,
    Spec,
    SpecError,
    load_spec,
)

PINNED = "0123456789abcdef0123456789abcdef01234567"


def key(spec: Spec, stage: str, **kwargs) -> str:
    """A stage's cache key with fixed inputs, so only the spec varies."""
    inputs = kwargs.pop("inputs", {"checkpoint": "sha256:" + "e" * 64})
    env = kwargs.pop("env", spec.env_identity_for(stage))
    return spec.hash_for(stage, inputs, env, **kwargs)


# -- loading and validation -------------------------------------------------


def test_a_valid_spec_loads(tmp_path):
    import yaml

    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(spec_mapping()), encoding="utf-8")
    spec = load_spec(path)
    assert spec.base_model.revision == PINNED
    assert spec.export.recipes == ("dynamic_wi8_afp32", "weight_only_wi8_afp32")
    assert spec.source == str(path)


def test_invalid_yaml_names_the_source(tmp_path):
    with pytest.raises(SpecError) as exc:
        Spec.from_yaml("base_model: [unclosed\n", source="job.yaml")
    assert "job.yaml" in str(exc.value)


def test_a_missing_revision_is_refused_by_name():
    with pytest.raises(SpecError) as exc:
        make_spec(base_model={"revision": None})
    assert "base_model.revision" in str(exc.value)


def test_an_empty_revision_key_is_the_same_as_a_missing_one():
    # `revision:` with nothing after it is what a hand-written YAML produces.
    with pytest.raises(SpecError) as exc:
        Spec.from_mapping(spec_mapping() | {"base_model": {"id": "x", "revision": None}})
    assert "base_model.revision" in str(exc.value)


@pytest.mark.parametrize("ref", ["main", "master", "HEAD", "Main", "latest", "refs/heads/dev"])
def test_a_mutable_ref_is_not_a_revision(ref):
    # The same unchanged definition produced a working export on 2026-08-26 and
    # AttributeError: pad_token on 2026-08-30. A branch name moves the same way.
    with pytest.raises(SpecError) as exc:
        make_spec(base_model={"revision": ref})
    assert "base_model.revision" in str(exc.value)
    assert ref in str(exc.value)


def test_a_tag_loads_but_is_recorded_as_a_weaker_pin():
    spec = make_spec(base_model={"revision": "v1.2.0"})
    assert spec.base_model.revision == "v1.2.0"
    assert any("not a 40-character commit sha" in text for text in spec.limitations)


def test_a_commit_sha_carries_no_pinning_limitation():
    assert not any("commit sha" in text for text in make_spec().limitations)


def test_a_missing_dataset_content_hash_is_refused_by_name():
    with pytest.raises(SpecError) as exc:
        make_spec(dataset={"content_sha256": None})
    assert "dataset.content_sha256" in str(exc.value)


@pytest.mark.parametrize("bad", ["", "not-a-hash", "abc123", 12345])
def test_a_content_hash_that_is_not_one_is_refused(bad):
    with pytest.raises(SpecError) as exc:
        make_spec(dataset={"content_sha256": bad})
    assert "dataset.content_sha256" in str(exc.value)


def test_a_typo_is_an_error_not_a_silently_ignored_field():
    # `content_sha_256` read as an unknown extra would key the cache on a
    # dataset identity nobody supplied.
    with pytest.raises(SpecError) as exc:
        Spec.from_mapping(spec_mapping(dataset={"uri": "./d.jsonl", "content_sha_256": DATA_A}))
    assert "content_sha_256" in str(exc.value)


def test_an_unknown_top_level_section_is_refused():
    with pytest.raises(SpecError) as exc:
        Spec.from_mapping(spec_mapping(quantization={"recipe": "x"}))
    assert "quantization" in str(exc.value)


@pytest.mark.parametrize(
    "section", ["base_model", "dataset", "train", "export", "eval", "toolchain"]
)
def test_a_missing_required_section_is_named(section):
    with pytest.raises(SpecError) as exc:
        make_spec(**{section: None})
    assert section in str(exc.value)


def test_gates_may_be_absent_and_the_run_says_it_is_not_judging():
    spec = make_spec(gates=None)
    assert spec.gates.max_conversion_cost is None
    assert any("no gates are declared" in text for text in spec.limitations)


# -- export -----------------------------------------------------------------


def test_recipes_must_be_a_list_even_for_one_recipe():
    with pytest.raises(SpecError) as exc:
        make_spec(export={"recipes": "dynamic_wi8_afp32"})
    assert "export.recipes" in str(exc.value)


def test_duplicate_recipes_are_refused():
    with pytest.raises(SpecError) as exc:
        make_spec(export={"recipes": ["dynamic_wi8_afp32", "dynamic_wi8_afp32"]})
    assert "export.recipes" in str(exc.value)


def test_a_one_recipe_sweep_is_recorded_as_producing_no_frontier():
    spec = make_spec(export={"recipes": ["dynamic_wi8_afp32"]})
    assert any("one-recipe sweep" in text for text in spec.limitations)


def test_context_length_is_required_because_it_is_baked_into_the_artifact():
    with pytest.raises(SpecError) as exc:
        make_spec(export={"recipes": ["dynamic_wi8_afp32"], "context_length": None})
    assert "export.context_length" in str(exc.value)


def test_a_recipe_name_that_could_escape_a_path_is_refused():
    with pytest.raises(SpecError) as exc:
        make_spec(export={"recipes": ["../../etc/passwd"]})
    assert "export.recipes" in str(exc.value)


# -- train ------------------------------------------------------------------


def test_adapter_settings_under_a_full_run_are_refused_rather_than_ignored():
    # A run believed to be an adapter run that was silently a full one is not
    # something the manifest could ever show afterwards.
    with pytest.raises(SpecError) as exc:
        make_spec(train={"mode": "full", "lora_rank": 8})
    assert "train.lora_rank" in str(exc.value)


def test_an_adapter_run_carries_the_measured_warning():
    spec = make_spec(train={"mode": "lora", "lora_rank": 8})
    assert any("0.0625" in text for text in spec.limitations)


def test_a_boolean_is_not_an_integer():
    with pytest.raises(SpecError) as exc:
        make_spec(train={"mode": "full", "batch_size": True})
    assert "train.batch_size" in str(exc.value)


# -- toolchain --------------------------------------------------------------


def test_an_unpinned_toolchain_requirement_is_refused_by_field():
    with pytest.raises(SpecError) as exc:
        make_spec(toolchain={"export": ["litert-lm>=0.16.1"]})
    assert "toolchain.export" in str(exc.value)


def test_a_missing_toolchain_environment_is_named():
    with pytest.raises(SpecError) as exc:
        make_spec(toolchain={"train": "default", "export": "default", "runtime": None})
    assert "toolchain.runtime" in str(exc.value)


def test_default_resolves_to_the_shipped_pins_and_is_recorded():
    spec = make_spec()
    from litetune import envs

    assert spec.toolchain.env("export").requirements == envs.EXPORT.requirements
    assert spec.environments()["export"] == envs.EXPORT.identity


# -- slicing ----------------------------------------------------------------


def test_the_export_slice_carries_only_what_export_depends_on():
    slice_ = make_spec().slice_for("export")
    assert set(slice_) == {"base_model", "export"}


@pytest.mark.parametrize("stage", ["train", "merge", "export", "measure"])
def test_no_measurement_stage_sees_a_threshold(stage):
    assert "gates" not in make_spec().slice_for(stage)


def test_a_measurement_stage_asking_for_gates_is_refused():
    with pytest.raises(SpecError) as exc:
        make_spec().slice_for("export", sections=("base_model", "gates"))
    assert "gates" in str(exc.value)
    assert "judgement" in str(exc.value)


def test_the_judge_stage_is_the_one_that_sees_gates():
    assert "gates" in make_spec().slice_for("judge")


def test_the_toolchain_is_in_no_slice_because_it_enters_by_resolved_identity():
    for stage in STAGE_SECTIONS:
        assert "toolchain" not in make_spec().slice_for(stage)
    with pytest.raises(SpecError) as exc:
        make_spec().slice_for("export", sections=("toolchain",))
    assert "resolved identity" in str(exc.value)


def test_an_unknown_stage_has_no_slice_rather_than_an_empty_one():
    # An empty slice would make every spec change a cache hit.
    with pytest.raises(SpecError) as exc:
        make_spec().slice_for("polish")
    assert "polish" in str(exc.value)


def test_the_dataset_slice_carries_content_and_not_location():
    identity = make_spec().slice_for("train")["dataset"]
    assert identity["content_sha256"] == DATA_A
    assert "uri" not in identity


# -- hashing ----------------------------------------------------------------


def test_moving_a_gate_does_not_move_a_measurement_key():
    # Tightening a bar must re-judge recorded metrics, not re-run generation.
    loose = make_spec(gates={"max_conversion_cost": 0.05})
    tight = make_spec(gates={"max_conversion_cost": 0.001})
    for stage in ("train", "merge", "export", "measure"):
        assert key(loose, stage) == key(tight, stage), stage


def test_moving_a_gate_does_move_the_judge_key():
    loose = make_spec(gates={"max_conversion_cost": 0.05})
    tight = make_spec(gates={"max_conversion_cost": 0.001})
    assert key(loose, "judge") != key(tight, "judge")


def test_replacing_the_dataset_content_moves_the_training_key():
    assert key(make_spec(), "train") != key(make_spec(dataset={"content_sha256": DATA_B}), "train")


def test_moving_the_dataset_file_does_not_move_any_key():
    # Same bytes at a new URI is the same data. Re-training for a path change
    # costs hours and establishes nothing.
    moved = make_spec(dataset={"uri": "s3://elsewhere/mobile-actions.jsonl"})
    for stage in ("train", "merge", "export", "measure"):
        assert key(make_spec(), stage) == key(moved, stage), stage


def test_replacing_the_heldout_content_moves_the_measurement_key():
    other = make_spec(eval={"heldout_content_sha256": HELDOUT_B})
    assert key(make_spec(), "measure") != key(other, "measure")


def test_moving_the_heldout_file_does_not_move_the_measurement_key():
    moved = make_spec(eval={"heldout_uri": "/mnt/other/heldout.jsonl"})
    assert key(make_spec(), "measure") == key(moved, "measure")


def test_a_smaller_slice_of_the_same_file_is_a_different_measurement():
    # At n=64 a recipe comparison read as 0.172; at n=640 it was 0.024. The two
    # are not the same measurement and must not share a key.
    assert key(make_spec(), "measure") != key(make_spec(eval={"limit": 64}), "measure")


def test_changing_the_environment_moves_the_key():
    pinned_later = make_spec(
        toolchain={"export": ["litert-torch-nightly==0.10.0.dev20260830", "numpy==2.0.2"]}
    )
    assert key(make_spec(), "export") != key(pinned_later, "export")


def test_a_training_pin_does_not_move_the_export_key():
    # The export stage does not care which torch trained the checkpoint; the
    # checkpoint reaches it as an input hash.
    other_train = make_spec(toolchain={"train": ["torch==2.6.0", "transformers==4.57.3"]})
    assert key(make_spec(), "export") == key(other_train, "export")
    assert key(make_spec(), "train") != key(other_train, "train")


def test_a_different_input_moves_the_key():
    spec = make_spec()
    assert key(spec, "export", inputs={"checkpoint": "sha256:" + "1" * 64}) != key(
        spec, "export", inputs={"checkpoint": "sha256:" + "2" * 64}
    )


def test_an_input_with_no_content_hash_is_refused_by_name():
    with pytest.raises(SpecError) as exc:
        make_spec().hash_for("export", {"checkpoint": None}, "env-1")
    assert "checkpoint" in str(exc.value)


def test_an_empty_environment_identity_is_refused():
    with pytest.raises(SpecError):
        make_spec().hash_for("export", {}, "")


def test_an_in_process_stage_still_has_an_identity():
    spec = make_spec()
    assert spec.env_identity_for("judge") == IN_PROCESS
    assert IN_PROCESS.startswith("in-process:")


def test_keys_are_stable_across_equal_specs():
    assert key(make_spec(), "export") == key(make_spec(), "export")


def test_the_spec_hash_covers_everything_including_the_uri():
    # The whole-spec hash identifies the job for the manifest; unlike a cache
    # key it records where the data came from too.
    assert make_spec().hash != make_spec(dataset={"uri": "s3://elsewhere.jsonl"}).hash
