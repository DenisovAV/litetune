"""The two properties a bundle's chat template must have, on the templates litetune handles.

Single-turn identity and the conversation prefix invariant (`template_checks`),
checked on Qwen3 0.6B's and Qwen2.5 0.5B Instruct's own templates, the composed
templates litetune builds from them, LiteRT-LM v0.18.0's Qwen 3 and Qwen 2.5
templates and the FunctionGemma template litetune packages; and the renderer
`convert` checks them with, held to MiniJinja 2.14.0 and to LiteRT-LM's own
goldens. Every template comes from a fixture or the package, with its
provenance in `fixtures/README.md` or its header: nothing is fetched.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from conftest import run_compose_script
from minijinja_runtime import MINIJINJA_EDITS, minijinja_render

from litetune import export, models, template_checks
from litetune.template_checks import (
    CONVERSATIONS,
    FORMS,
    LIMITATION_PREFIX,
    PROBE_TEXT,
    SINGLE_TURNS,
    ConversationCheck,
    Rendering,
    conversation_breaks,
    conversation_limitation,
    single_turn_differences,
    unchecked_limitation,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_bytes().decode("utf-8")


def _packaged(name: str) -> str:
    return Path(models.packaged_template(name)).read_bytes().decode("utf-8")


QWEN3 = _fixture("qwen3-0.6b-c1899de.chat_template.jinja")
QWEN25 = _fixture("qwen2.5-0.5b-instruct-7ae5576.chat_template.jinja")
LITERT_QWEN25 = _fixture("litert-lm-v0.18.0-qwen2_5.chat_template.jinja")
BLOCK = _packaged(models.TEXT_PARTS_BLOCK)
LITERT_QWEN3 = models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE)
FUNCTIONGEMMA = _packaged("functiongemma.jinja")

# Every template a check here renders for a bundle, by what it is.
BUNDLED = {
    "qwen3 composed": BLOCK + QWEN3,
    "qwen2.5 composed": BLOCK + QWEN25,
    "qwen3 litert-lm": LITERT_QWEN3,
    "qwen2.5 litert-lm": LITERT_QWEN25,
    "functiongemma packaged": FUNCTIONGEMMA,
    # The checkpoints' own, which LiteRT-LM 0.18.0 cannot use alone.
    "qwen3 alone": QWEN3,
    "qwen2.5 alone": QWEN25,
}


@functools.cache
def _script(name: str) -> dict[str, Any]:
    """A script `export` writes for the export environment, loaded here."""
    source = {
        "renderer": template_checks.RENDERER_SOURCE,
        "conversation": export._CONVERSATION_SCRIPT,
    }[name]
    namespace: dict[str, Any] = {"__name__": name}
    exec(compile(source, f"{name}.py", "exec"), namespace)
    return namespace


def _render(template: str, messages: list, gen: bool = True, **context: Any) -> dict:
    return _script("renderer")["runtime_render"](template, messages, gen, **context)


def _breaks(template: str) -> list[template_checks.Break]:
    """What the conversation script renders for `template`, read as `convert` reads it."""
    spec = {"template": template, "conversations": template_checks.conversation_spec()}
    report = _script("conversation")["main"](spec)
    return conversation_breaks(report["renders"])


def _minijinja(template: str, messages: list, gen: bool, **context: Any) -> dict:
    try:
        return {"text": minijinja_render(template, messages, gen, **context)}
    except Exception as error:  # noqa: BLE001 -- MiniJinja's own error type
        return {"error": str(error)}


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------


def _every_rendering() -> list[tuple[list, bool]]:
    """Each message list and generation-prompt flag the checks render a bundle with."""
    renderings = [(case.as_parts(form), True) for case in SINGLE_TURNS for form in FORMS]
    for case in CONVERSATIONS:
        renderings += [(case.history(), False), ([*case.history(), case.following()], True)]
    return renderings


@pytest.mark.parametrize("name", sorted(BUNDLED))
def test_the_renderer_renders_every_case_as_minijinja_does(name):
    """`convert` renders with Jinja2 configured as LiteRT-LM configures MiniJinja;
    on every template and case it checks, the two agree byte for byte, or both raise."""
    template = BUNDLED[name]
    for messages, gen in _every_rendering():
        ours, theirs = _render(template, messages, gen), _minijinja(template, messages, gen)
        assert ("text" in ours) == ("text" in theirs), (messages, ours, theirs)
        assert ours.get("text") == theirs.get("text"), messages


def test_the_renderer_rewrites_as_the_runtime_does():
    """The same `EditTemplateForMinijinja` list as the MiniJinja harness, in order."""
    assert _script("renderer")["MINIJINJA_EDITS"] == MINIJINJA_EDITS


@pytest.mark.parametrize(
    ("source", "printed"),
    [
        # A Python method MiniJinja has none of, unless the runtime rewrote it.
        ("{{ 'a'.upper() }}", None),
        ("{{ 'abc'.startswith('a') }}", "true"),
        ("[{{ 'abc'.foo }}]", "[]"),
        ("{{ 'a' is sequence }}|{{ [1] is sequence }}", "false|true"),
        ("{{ true }}|{{ none }}|{{ [1, 'a'] }}|{{ {'a': 1} }}", 'true|none|[1, "a"]|{"a": 1}'),
        ("{{ 1 is startingwith '1' }}", "true"),
        ("{{ {'a': [1, 'é']} | tojson }}", '{"a": [1, "é"]}'),
        ("{{ 'xxaxx' | trim('x') }}|{{ 'a,b' | split(',') | last }}", "a|b"),
        ("{{ true is number }}", "false"),
        ("{{ 'kept' }}\n", "kept\n"),
    ],
)
def test_the_renderer_reads_what_jinja2_reads_otherwise_as_minijinja_does(source, printed):
    ours, theirs = _render(source, [], True), _minijinja(source, [], True)
    assert ours.get("text") == printed
    assert theirs.get("text") == printed


def test_the_renderers_none_is_the_runtimes_is_undefined():
    """`create_env` replaces MiniJinja's `none` test with `value.is_undefined()`
    (`minijinja_template.rs`), so a null the runtime hands over is not none and a
    missing key is. The Python binding cannot show this: it hands a test `None`
    for both."""
    messages = [{"role": "user", "content": None}]
    source = "{{ messages[0].content is none }}|{{ messages[0].missing is none }}"
    assert _render(source, messages, True) == {"text": "false|true"}


def test_the_renderer_refuses_a_break_as_the_runtime_build_does():
    """LiteRT-LM builds MiniJinja 2.14.0 without `loop_controls` (`Cargo.toml`
    names `json` and `preserve_order` besides the defaults), so `{% break %}` is
    a syntax error on the device -- the Python binding, built with it, accepts it."""
    source = "{% for x in [1, 2] %}{{ x }}{% break %}{% endfor %}"
    assert "error" in _render(source, [], True)
    assert _minijinja(source, [], True) == {"text": "1"}


GOLDENS = FIXTURES / "litert-lm-v0.18.0-qwen3-testdata"
# `git ls-tree v0.18.0 models/qwen3/testdata` in google-ai-edge/LiteRT-LM.
GOLDEN_BLOBS = {
    "golden/system_instruction-thinking.txt": "975391d9ee4a633dce8e89a75b50ab00a362dac7",
    "golden/system_instruction.txt": "0b8d85d3778c63f21888b90fe8a95db35ac6322b",
    "golden/tool_response_ending-thinking.txt": "1deefe597a82cd9671df1ded0968a6e5c1f850c2",
    "golden/tool_response_ending.txt": "61b32a2d5dc9b61f47c8f89da3ba47c2f4dd88a9",
    "golden/tools-thinking.txt": "759fe508507fae8d76ab0438724e942ac111a887",
    "golden/tools.txt": "e6d9bb9e3092999e9791f7cc788c3ed5d76452b0",
    "input/system_instruction.json": "285a4f11ae9af1095b48e9636f1c3c769fa168a2",
    "input/tool_response_ending.json": "37e51dad2f47c6ed166df16387fe0e4b65e89781",
    "input/tools.json": "eee9d6b3f0fc2e608e62baa510bff055e7392b69",
}


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_each_litert_lm_fixture_is_the_file_at_v0_18_0():
    assert sorted(str(p.relative_to(GOLDENS)) for p in GOLDENS.rglob("*") if p.is_file()) == sorted(
        GOLDEN_BLOBS
    )
    for name, blob in GOLDEN_BLOBS.items():
        assert _git_blob((GOLDENS / name).read_bytes()) == blob, name
    qwen25 = (FIXTURES / "litert-lm-v0.18.0-qwen2_5.chat_template.jinja").read_bytes()
    assert _git_blob(qwen25) == "154f74aaeac0427d24bb789270fd060f90028a0b"


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("name", ["system_instruction", "tool_response_ending", "tools"])
def test_the_renderer_reproduces_litert_lms_goldens_for_its_qwen3_template(name, thinking):
    """What LiteRT-LM's own `chat_template_test_runner.cc` compares MiniJinja's
    rendering of `models/qwen3/chat_template.jinja` with -- a multi-turn
    conversation, tool declarations, a tool response -- rendered here from the
    packaged template, its provenance comment included."""
    data = json.loads((GOLDENS / "input" / f"{name}.json").read_bytes())
    suffix = "-thinking" if thinking else ""
    golden = (GOLDENS / "golden" / f"{name}{suffix}.txt").read_bytes().decode("utf-8")
    context: dict[str, Any] = {"enable_thinking": thinking}
    if "tools" in data:
        context["tools"] = data["tools"]
    gen = data.get("add_generation_prompt", True)
    assert _render(LITERT_QWEN3, data["messages"], gen, **context) == {"text": golden}
    assert _minijinja(LITERT_QWEN3, data["messages"], gen, **context) == {"text": golden}


# ---------------------------------------------------------------------------
# The packaged canonical template
# ---------------------------------------------------------------------------


def test_the_packaged_qwen3_template_is_litert_lms_byte_for_byte():
    raw = Path(models.packaged_template(models.QWEN3_CANONICAL_TEMPLATE)).read_bytes()
    header, found, body = raw.partition(models.CANONICAL_HEADER_END.encode("utf-8"))
    assert found
    sha256 = "4315c96833e8b2a21b878d70e8d2f6415416d5e23ed8ee0ae0a111a540d844f7"
    assert hashlib.sha256(body).hexdigest() == sha256
    assert models.CANONICAL_TEMPLATES[models.QWEN3_CANONICAL_TEMPLATE] == sha256
    assert _git_blob(body) == "72501d10e25fd5382a2a60ecf03f8b25acd8de4e"
    said = header.decode("utf-8")
    for fact in (
        "https://github.com/google-ai-edge/LiteRT-LM",
        "tag v0.18.0",
        "b2f686e2ed4718fb84ec398a61dd59ca0f0aff27",
        "models/qwen3/chat_template.jinja",
        "72501d10e25fd5382a2a60ecf03f8b25acd8de4e",
        sha256,
        "Apache-2.0",
    ):
        assert fact in said, fact
    # The runtime rewrites nothing in it, and it prints nothing.
    assert _script("renderer")["edit_for_minijinja"](said) == said
    assert _render(said + "-#}", [], True) == {"text": ""}
    assert models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE) == raw.decode("utf-8")


@pytest.mark.parametrize(
    "damage",
    [
        lambda text: text + " ",
        lambda text: text.replace("{#- litetune:", "{# litetune:", 1),
        lambda text: text.replace(models.CANONICAL_HEADER_END, "#}\n", 1),
    ],
    ids=["body", "header", "header-end"],
)
def test_a_packaged_canonical_template_that_is_not_the_pinned_one_is_refused(
    monkeypatch, tmp_path, damage
):
    copy = tmp_path / models.QWEN3_CANONICAL_TEMPLATE
    copy.write_bytes(damage(LITERT_QWEN3).encode("utf-8"))
    monkeypatch.setattr(models, "packaged_template", lambda name: str(tmp_path / name))
    with pytest.raises(ValueError, match="is not the upstream template its sha256 is pinned to"):
        models.canonical_template_text(models.QWEN3_CANONICAL_TEMPLATE)


def test_every_rule_offering_a_canonical_template_names_a_pinned_one():
    for rules in models.RULES:
        markers = {r.value for r in rules.required_flags if r.name == models.TEMPLATE_FLAG}
        offers = models.CANONICAL_OR_COMPOSED_TEMPLATE in markers
        assert offers == (rules.canonical_template is not None), rules.family
        if rules.canonical_template is not None:
            assert rules.canonical_template in models.CANONICAL_TEMPLATES
            models.canonical_template_text(rules.canonical_template)


# ---------------------------------------------------------------------------
# Single-turn identity
# ---------------------------------------------------------------------------


def _compose(tmp_path: Path, checkpoint: str, canonical: str | None) -> dict:
    """`export._COMPOSE_SCRIPT` on a checkpoint carrying `checkpoint`: the
    checkpoint's side through transformers -- the conftest fake renders as
    5.16.1 does -- and each candidate through the runtime's renderer."""
    work = tmp_path / hashlib.sha256((checkpoint + str(canonical)).encode()).hexdigest()[:8]
    work.mkdir()
    script, spec, result = work / "compose_template.py", work / "spec.json", work / "result.json"
    script.write_text(export._COMPOSE_SCRIPT, encoding="utf-8")
    spec.write_text(
        json.dumps(
            {
                "model": "m",
                "block": BLOCK,
                "out": str(work / "composed.jinja"),
                "canonical": canonical,
                "single_turns": template_checks.single_turn_spec(),
            }
        ),
        encoding="utf-8",
    )
    proc, _ = run_compose_script(["python", str(script), str(spec), str(result)], 60, checkpoint)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(result.read_text(encoding="utf-8"))
    assert report["found"] == "text"
    return report["renders"]


def test_litert_lms_qwen3_template_renders_single_turns_as_qwen3s_own(tmp_path):
    """The condition the qwen-3 rule packs it on, on the checkpoint it was run on."""
    renders = _compose(tmp_path, QWEN3, LITERT_QWEN3)
    assert single_turn_differences(renders["reference"], renders["candidates"]["canonical"]) == []
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []
    choice = export._choose_template("m", renders, models.QWEN3_CANONICAL_TEMPLATE)
    assert choice.source == "canonical"
    assert choice.refusal is None


def test_litert_lms_qwen25_template_does_not_render_qwen25s_default_system_prompt(tmp_path):
    """Why the qwen-2.5 rule offers no canonical template (the comment beside it):
    with no system message the checkpoint's template prints its default one and
    LiteRT-LM's prints none -- 8 of the 16 renderings, exactly those."""
    renders = _compose(tmp_path, QWEN25, LITERT_QWEN25)
    differences = single_turn_differences(renders["reference"], renders["candidates"]["canonical"])
    assert len(differences) == 8
    assert {d.case.system for d in differences} == {None}
    default = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    assert all(default in (d.reference.text or "") for d in differences)
    assert all(default not in (d.candidate.text or "") for d in differences)
    assert single_turn_differences(renders["reference"], renders["candidates"]["composed"]) == []
    choice = export._choose_template("m", renders, "litert-qwen2.5")
    assert choice.source == "composed"
    assert len(choice.canonical_differences) == 8


def test_the_cases_cover_a_system_message_and_two_parts():
    assert len(SINGLE_TURNS) == 8
    assert {case.system is None for case in SINGLE_TURNS} == {True, False}
    assert SINGLE_TURNS[0] == template_checks.SingleTurn(PROBE_TEXT)
    two = SINGLE_TURNS[1].as_parts("two_parts")
    assert [len(m["content"]) for m in two] == [2, 2]
    assert "".join(p["text"] for p in two[1]["content"]) == PROBE_TEXT
    assert SINGLE_TURNS[1].as_strings()[0] == {"role": "system", "content": SINGLE_TURNS[1].system}


def _same(count: int = len(SINGLE_TURNS), text: str = "x") -> tuple[list, list]:
    reference = [{"text": f"{text}{i}"} for i in range(count)]
    candidate = [{form: {"text": f"{text}{i}"} for form in FORMS} for i in range(count)]
    return reference, candidate


def test_identical_renderings_are_no_difference():
    assert single_turn_differences(*_same()) == []


@pytest.mark.parametrize("form", FORMS)
def test_one_differing_rendering_is_one_difference(form):
    reference, candidate = _same()
    candidate[3][form] = {"text": "other"}
    (difference,) = single_turn_differences(reference, candidate)
    assert difference.case == SINGLE_TURNS[3] and difference.form == form
    assert "printed 'x3'" in difference.describe() and "printed 'other'" in difference.describe()


def test_a_candidate_that_raises_differs_and_two_that_raise_do_not():
    reference, candidate = _same()
    candidate[2]["one_part"] = {"error": "TemplateError: boom"}
    (difference,) = single_turn_differences(reference, candidate)
    assert "the bundle's raised TemplateError: boom" in difference.describe()
    reference[2] = {"error": "TemplateError: boom"}
    candidate[2]["two_parts"] = {"error": "TemplateError: also"}
    assert single_turn_differences(reference, candidate) == []
    candidate[2]["two_parts"] = {"text": "x2"}
    (difference,) = single_turn_differences(reference, candidate)
    assert difference.form == "two_parts"


@pytest.mark.parametrize(
    "spoil",
    [
        lambda r, c: (r[:-1], c),
        lambda r, c: (r, c[:-1]),
        lambda r, c: (r, [*c[:-1], "text"]),
        lambda r, c: (r, [*c[:-1], {"one_part": {"text": "x"}}]),
        lambda r, c: ([*r[:-1], {"txt": "x"}], c),
        lambda r, c: ("renders", c),
    ],
)
def test_renderings_this_cannot_read_are_refused(spoil):
    with pytest.raises(ValueError):
        single_turn_differences(*spoil(*_same()))


# ---------------------------------------------------------------------------
# The conversation prefix invariant
# ---------------------------------------------------------------------------


def test_qwen3s_own_template_breaks_every_conversation():
    """The defect: behind the block, which it needs on 0.18.0, Qwen3's template
    renders the latest reply with a think block and an earlier one without --
    for a plain reply, one carrying a think block and one a thought channel
    took, over two turns and three, with a system message or none."""
    breaks = _breaks(BLOCK + QWEN3)
    assert [b.case for b in breaks] == list(CONVERSATIONS)
    plain = breaks[0]
    assert plain.case.reply.name == "a plain reply" and plain.case.turns == 2
    assert plain.previous.text is not None and plain.following.text is not None
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\ncard_lost" in plain.previous.text
    assert "<|im_start|>assistant\ncard_lost<|im_end|>" in plain.following.text
    assert "from character" in plain.describe()


@pytest.mark.parametrize(
    "name",
    ["qwen3 litert-lm", "qwen2.5 composed", "qwen2.5 litert-lm", "functiongemma packaged"],
)
def test_these_templates_hold_every_conversation(name):
    assert _breaks(BUNDLED[name]) == []


# Templates that break only where one kind of case reaches them: what each case
# is there to catch.
_LAST_REPLY_MARKED_FROM_TURN_THREE = (
    "{%- for m in messages %}{{ m.role }}:{{ m.content[0].text }}"
    "{% if m.role == 'assistant' and loop.index0 >= 3 and loop.last %}*{% endif %};"
    "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
)
_LAST_THINK_BLOCK_BRACKETED = (
    "{%- for m in messages %}{% set t = m.content[0].text %}"
    "{% if '<think>' in t and loop.last %}[{{ t }}]{% else %}{{ t }}{% endif %};{% endfor %}"
)
_LAST_REASONING_SHOWN = (
    "{%- for m in messages %}{{ m.content[0].text }}"
    "{% if m.reasoning_content and loop.last %}<{{ m.reasoning_content }}>{% endif %};"
    "{% endfor %}"
)


@pytest.mark.parametrize(
    ("template", "reached"),
    [
        (_LAST_REPLY_MARKED_FROM_TURN_THREE, lambda case: case.turns == 3),
        (_LAST_THINK_BLOCK_BRACKETED, lambda case: case.reply.name.endswith("a think block")),
        (_LAST_REASONING_SHOWN, lambda case: "thought channel" in case.reply.name),
    ],
    ids=["three-turns", "think-block", "thought-channel"],
)
def test_each_kind_of_case_catches_what_only_it_reaches(template, reached):
    breaks = _breaks(template)
    assert breaks, "a case is missing"
    assert [b.case for b in breaks] == [case for case in CONVERSATIONS if reached(case)]


def test_a_template_that_raises_on_a_reply_breaks_with_the_error():
    breaks = _breaks(
        "{%- for m in messages %}{% if m.role == 'assistant' %}"
        "{{ raise_exception('no replies') }}{% endif %}{% endfor %}"
    )
    assert len(breaks) == len(CONVERSATIONS)
    said = breaks[0].describe()
    assert "the template raises on the history ending in the reply" in said
    assert "no replies" in said


def test_a_template_that_grows_by_appending_holds():
    assert _breaks("{%- for m in messages %}{{ m.content[0].text }};{% endfor %}") == []


def test_the_conversations_cover_both_reply_shapes_and_three_turns():
    assert len(CONVERSATIONS) == 12
    assert {case.turns for case in CONVERSATIONS} == {2, 3}
    assert {case.system is None for case in CONVERSATIONS} == {True, False}
    three = next(c for c in CONVERSATIONS if c.turns == 3 and c.system)
    roles = [m["role"] for m in three.history()]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert three.following()["role"] == "user"
    think, channel = (r.message for r in template_checks.REPLIES[1:])
    assert "<think>" in think["content"][0]["text"]
    assert "<think>" not in channel["content"][0]["text"]
    assert channel["reasoning_content"] == channel["channels"]["thought"]


def _rendered(previous: dict, following: dict) -> list[dict]:
    return [{"previous": previous, "next": following} for _ in CONVERSATIONS]


@pytest.mark.parametrize(
    ("previous", "following", "breaks"),
    [
        ({"text": "ab"}, {"text": "abc"}, False),
        ({"text": "ab"}, {"text": "ab"}, False),
        ({"text": "ab"}, {"text": "xab"}, True),
        ({"text": "abc"}, {"text": "ab"}, True),
        ({"error": "E"}, {"text": "ab"}, True),
        ({"text": "ab"}, {"error": "E"}, True),
    ],
)
def test_the_invariant_is_a_byte_prefix(previous, following, breaks):
    assert bool(conversation_breaks(_rendered(previous, following))) is breaks


@pytest.mark.parametrize(
    "entries",
    ["renders", [], _rendered({"text": "a"}, {"text": "a"})[:-1], [{"previous": {"text": "a"}}]],
)
def test_conversation_renderings_this_cannot_read_are_refused(entries):
    with pytest.raises(ValueError):
        conversation_breaks(entries)


def test_the_limitation_names_the_recipes_the_runtime_and_the_first_break():
    check = ConversationCheck.of(BLOCK + QWEN3, _breaks(BLOCK + QWEN3), "the renderer")
    assert check.holds is False
    assert check.template_sha256 == hashlib.sha256((BLOCK + QWEN3).encode()).hexdigest()
    text = conversation_limitation(["dynamic_wi8_afp32", "weight_only_wi8_afp32"], check)
    assert text.startswith(f"dynamic_wi8_afp32, weight_only_wi8_afp32: {LIMITATION_PREFIX}")
    for fact in (
        "The new rendered template string does not start with the previous rendered template",
        "Conversation::GetSingleTurnTextFromFullHistory",
        "google-ai-edge/LiteRT-LM#3443",
        "12 of the 12 conversations break it",
        check.breaks[0],
    ):
        assert fact in text
    held = ConversationCheck.of(LITERT_QWEN3, [], "the renderer")
    assert held.holds is True and held.breaks == ()
    assert held.as_dict() == {
        "holds": True,
        "detail": held.detail,
        "template_sha256": hashlib.sha256(LITERT_QWEN3.encode()).hexdigest(),
        "breaks": [],
    }
    assert "was not checked: why" in unchecked_limitation(["r"], "why")
    assert LIMITATION_PREFIX not in unchecked_limitation(["r"], "why")


@pytest.mark.parametrize(
    ("entry", "reads"),
    [
        ({"text": "a"}, Rendering(text="a")),
        ({"error": "E"}, Rendering(error="E")),
        ({"error": "E", "environment": True}, Rendering(error="E", environment=True)),
    ],
)
def test_a_rendering_reads_as_written(entry, reads):
    assert Rendering.read(entry) == reads


@pytest.mark.parametrize("entry", [None, "a", {"text": 1}, {"error": None}, {}])
def test_a_rendering_in_another_shape_is_refused(entry):
    with pytest.raises(ValueError):
        Rendering.read(entry)
