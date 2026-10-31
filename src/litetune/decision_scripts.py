"""The two scripts that score a decision's options: the float reference, and the runtime.

Each is source text, written to a file and run inside its stage environment
with a `spec.json`, as `evaluate.py` runs its generation scripts. Neither
imports litetune: the environments do not have it. Both write one JSON line per
row, flushed as it is scored, and a report.

**The reference** (`REFERENCE_SCRIPT`, `envs.TRAIN`) scores as the
option-scoring measurements of 2026-10-15 and 2026-10-16 did, one row at a time
there and in batches here: float32, eager attention, TF32 off, on CUDA when
there is one and `LITETUNE_DEVICE` allows it, on Apple's GPU when `verify`
found MPS and gave the run `evaluate`'s memory policy, otherwise on the CPU.
Each row's
prompt is rendered as `tune` trains it, by the same
`prompt_mode.RENDERING_SOURCE`, pasted in, and every option is scored as its id
followed by the turn terminator, teacher-forced, every token's log-probability
kept. On the first row each way of passing the cache is checked
against scoring with no cache, and the first within `check_tol` is used; when
none is, every row is scored with no cache.

Rows of a similar length go up to `row_batch` at a time: one forward over
their prompts, padded on the left and given their positions, keeps the cache,
and the (row, option) pairs go in chunks over copies of their row's part of it.
The cache each forward is given is at most `batch_tokens` tokens, prompts
padded, and a row longer than that is scored alone; the batch's own prompt
cache, and a copy of it per chunk, are alive beside it, and eager attention
grows with the square of the prompt's length on top.

The first batch scored is the most padded one, and every row of it is checked
against the row scored alone with no cache. A way of passing the cache that
takes no positions scores one row at a time, as does a first batch that raises
or differs by more than `check_tol`. After it, the first three rows, the
longest and the rows in `check_indices` are checked, and a difference above
`check_tol`, a NaN or a raise stops the run. The rows in `alone_for` are
recorded as scored alone with no cache, whatever the batch gave. MEASUREMENTS.md
has what batching changed in time and in the scores.

**The runtime** (`ENGINE_SCRIPT`, `envs.RUNTIME`) scores through LiteRT-LM's
session API on the CPU, or on macOS the GPU, as the flutter_edge_ai design of
2026-10-16 has its decider do. It renders with
`Conversation.render_message_to_string`, prefills the prompt, saves a
checkpoint, and per option scores and rewinds. LiteRT-LM 0.18.0 scores right
after a prefill without the prompt's second-to-last token (#3561): mode `once`
prefills the last token again, once per decision, which gives the reference's
scores on 0.18.0; mode `plain` does not, which is right on an engine with the
fix and wrong on 0.18.0. The self-check picks between them: two prompts that
differ only in their second-to-last token score identically on 0.18.0 and
differently with the fix. Measured 2026-10-16: 0.0 in all 42 comparisons on
0.18.0, at least 0.135 on a nightly with the fix; each mode within 8.2e-4 nats
of PyTorch on a float32 bundle, and wrong on the other engine. Those
measurements were Gemma 3's; the self-check's pair is rendered by the bundle's
own template, which on Gemma 3 is the pair measured.

That is method `rewind`. Method `fresh_engine`, for a family whose rule gives
every prompt an Engine of its own (`models.ModelRules.engine_per_prompt_reason`),
scores each option on a new Engine and session with no checkpoint and no
rewind, the self-check's prompts too; mode `once` then prefills the last token
again in each.

Targets are each option's shortest unique prefix: the first k ids of
`id + terminator`, k the smallest no other option shares. A truncated target
goes to the runtime as text and must tokenise back to the same ids; one that
does not is scored in full and listed.

Both scripts return the prompt's token ids for the rows the runtime scores, so
that `verify` can check the two sides scored the same prompt.
"""

from __future__ import annotations

from litetune import devices
from litetune.evaluate import RUNTIME_ENGINE_SOURCE
from litetune.prompt_mode import RENDERING_SOURCE
from litetune.rendering import SESSION_IDS_SOURCE

REFERENCE_SCRIPT = (
    r'''
"""Float32 per-token option scores for each row. Spec: argv[1]. See litetune.decision_scripts."""
import copy
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANTS = ("cache_position+position_ids", "position_ids", "plain")
# Where the run reads torch's MPS counters on mps (`sample_mps_memory`).
MPS_SAMPLE_POINTS = ("after each batch",)
'''
    + RENDERING_SOURCE
    + devices.MPS_SCRIPT_SOURCE
    + r"""

def head_transform(config):
    # What the bundle's head does to lm_head's output, which the scoring below
    # calls directly: Granite divides by `logits_scaling`
    # (transformers 5.16.1 modeling_granite.py:497), and a config with
    # `final_logit_softcapping` at its top soft-caps. Gemma 4's forward soft-caps
    # with the one on its text config (modeling_gemma4.py:2587), but its exported
    # bundle does not: on 2026-10-20 a gemma-4-E2B float twin was 5.5e-05 nats from
    # the uncapped head and 3.13e+01 from the capped one. The reference follows the
    # bundle, and the cached paths, which read the capped forward, fail their trial.
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    scale = getattr(config, "logits_scaling", None) or getattr(text, "logits_scaling", None)
    cap = getattr(config, "final_logit_softcapping", None)
    return scale, cap


def transformed(logits, scale, cap):
    if scale:
        logits = logits / scale
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    wanted = spec.get("device", "auto")
    device = "cuda" if wanted == "auto" and torch.cuda.is_available() else "cpu"
    # `verify` asks for mps only when its probe found it and the memory
    # policy was set (`devices.prepare_mps`).
    device = "mps" if wanted == "mps" else device
    mps_samples = new_mps_samples(MPS_SAMPLE_POINTS) if device == "mps" else None
    kwargs = {"revision": spec["revision"]} if spec.get("revision") else {}
    tok = AutoTokenizer.from_pretrained(spec["model"], **kwargs)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            spec["model"], dtype=torch.float32, attn_implementation="eager", **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            spec["model"], torch_dtype=torch.float32, attn_implementation="eager", **kwargs
        )
    model = model.to(device).eval()
    scale, cap = head_transform(model.config)
    # The first token of the end of turn tune recorded, as this tokenizer
    # writes it; the runtime is handed the text and must tokenise it alike.
    first = spec["turn_ids"][0]
    term = tok.decode([first])
    term_ids = tok(term, add_special_tokens=False)["input_ids"]
    if term_ids != [first] or not spec["turn_text"].startswith(term):
        raise SystemExit(
            f"the end of turn's first token {first} decodes to {term!r}, which tokenises to "
            f"{term_ids} and begins {spec['turn_text']!r}: not that one token"
        )
    conts = [tok(o, add_special_tokens=False)["input_ids"] + term_ids for o in spec["options"]]
    if len({tuple(c) for c in conts}) != len(conts):
        raise SystemExit("two options tokenise to the same ids")
    pad = tok.pad_token_id if tok.pad_token_id is not None else term_ids[0]
    chunk = int(spec.get("chunk", 64))
    # The most tokens of cache a forward is given, prompts padded: on the
    # emotion prompts measured (53 to 112 tokens) that takes all 96 pairs of a
    # 16-row, 6-option batch at once, and long prompts take fewer rows and
    # pairs, where a fixed count would multiply the memory.
    batch_tokens = int(spec.get("batch_tokens", 8192))
    tol = float(spec.get("check_tol", 3e-3))
    state = {"variant": None, "batch_tokens_held": 0}

    def render(prompt):
        # As `tune` trained it: a decision is trained runtime_rendered.
        text, add_special = render_prompt(tok, prompt, True)
        return tok(text, add_special_tokens=add_special)["input_ids"]

    def head_logp(hidden):
        logits = transformed(model.lm_head(hidden).float(), scale, cap)
        return torch.log_softmax(logits, dim=-1)

    def expand(cache, n):
        c = copy.deepcopy(cache)
        if hasattr(c, "batch_repeat_interleave"):
            c.batch_repeat_interleave(n)
            return c
        for layer in c.layers:
            layer.keys = layer.keys.repeat_interleave(n, dim=0)
            layer.values = layer.values.repeat_interleave(n, dim=0)
        return c

    def prompt_forward(**kw):
        try:
            return model(**kw, use_cache=True, logits_to_keep=1)
        except TypeError:
            return model(**kw, use_cache=True)

    def cont_forward(inp, cache, p, length, n, how):
        pos = torch.arange(p, p + length, device=device)
        kw = {
            "input_ids": inp,
            "past_key_values": cache,
            "use_cache": True,
            "attention_mask": torch.ones((n, p + length), dtype=torch.long, device=device),
        }
        if how in ("cache_position+position_ids", "position_ids"):
            kw["position_ids"] = pos.unsqueeze(0).expand(n, -1)
        if how == "cache_position+position_ids":
            kw["cache_position"] = pos
        return model(**kw)

    @torch.no_grad()
    def score_full(prompt_ids, batch=16):
        p = len(prompt_ids)
        res = []
        for s in range(0, len(conts), batch):
            part = conts[s : s + batch]
            seqs = [prompt_ids + c for c in part]
            width = max(len(x) for x in seqs)
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, x in enumerate(seqs):
                ids[i, : len(x)] = torch.tensor(x)
                mask[i, : len(x)] = 1
            h = model.model(
                input_ids=ids.to(device), attention_mask=mask.to(device)
            ).last_hidden_state
            for i, c in enumerate(part):
                lp = head_logp(h[i, p - 1 : p - 1 + len(c)])
                res.append([lp[t, c[t]].item() for t in range(len(c))])
        return res

    @torch.no_grad()
    def score_cached(prompt_ids, how):
        if how is None:
            return score_full(prompt_ids)
        p = len(prompt_ids)
        out = prompt_forward(input_ids=torch.tensor([prompt_ids], device=device))
        first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        cache = out.past_key_values
        res = []
        # As many options as `chunk`, and no more than `batch_tokens` of cache.
        step = max(1, min(chunk, batch_tokens // (p + max(len(c) for c in conts))))
        for s in range(0, len(conts), step):
            part = conts[s : s + step]
            n = len(part)
            length = max(len(c) for c in part)
            inp = torch.full((n, length), pad, dtype=torch.long, device=device)
            for j, c in enumerate(part):
                inp[j, : len(c)] = torch.tensor(c, device=device)
            o2 = cont_forward(inp, expand(cache, n), p, length, n, how)
            lp = torch.log_softmax(o2.logits.float(), dim=-1)
            for j, c in enumerate(part):
                rest = [lp[j, t - 1, c[t]].item() for t in range(1, len(c))]
                res.append([first[c[0]].item(), *rest])
            del o2, lp
        return res

    def select(cache, owners):
        # The cache rows the continuations extend, one per (row, option) pair,
        # by the selection beam search makes. On a copy: a layer may reorder in
        # place, and every chunk starts from the prompts' cache.
        c = copy.deepcopy(cache)
        c.reorder_cache(torch.tensor(owners, device=device))
        return c

    @torch.no_grad()
    def score_batch(prompts_ids, how):
        # Several rows in one forward: prompts padded on the left so that every
        # one ends in the last column, each row's positions counted from its
        # first real token; then the (row, option) pairs in chunks over copies
        # of their row's cache. An option's last token is not fed: no score
        # reads the logits after it.
        b = len(prompts_ids)
        width = max(len(x) for x in prompts_ids)
        ids = torch.full((b, width), pad, dtype=torch.long, device=device)
        mask = torch.zeros((b, width), dtype=torch.long, device=device)
        for i, x in enumerate(prompts_ids):
            ids[i, width - len(x) :] = torch.tensor(x, device=device)
            mask[i, width - len(x) :] = 1
        state["batch_tokens_held"] = max(state["batch_tokens_held"], b * width)
        out = prompt_forward(
            input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp(min=0)
        )
        first = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
        cache = out.past_key_values
        lens = mask.sum(-1)
        res = [[[h] for h in row] for row in first[:, [c[0] for c in conts]].tolist()]
        pairs = [(i, j) for i in range(b) for j in range(len(conts))]
        step = max(1, batch_tokens // (width + max(len(c) for c in conts)))
        for s in range(0, len(pairs), step):
            part = pairs[s : s + step]
            n = len(part)
            length = max(len(conts[j]) for _, j in part) - 1
            if length == 0:
                continue
            owners = [i for i, _ in part]
            held = n * (width + length)
            state["batch_tokens_held"] = max(state["batch_tokens_held"], held)
            inp = torch.full((n, length), pad, dtype=torch.long, device=device)
            for k, (_, j) in enumerate(part):
                c = conts[j][:-1]
                inp[k, : len(c)] = torch.tensor(c, device=device)
            pos = torch.arange(length, device=device)
            kw = {
                "input_ids": inp,
                "past_key_values": select(cache, owners),
                "use_cache": True,
                "attention_mask": torch.cat(
                    [mask[owners], torch.ones((n, length), dtype=torch.long, device=device)], dim=1
                ),
                "position_ids": lens[owners].unsqueeze(1) + pos.unsqueeze(0),
            }
            if how == "cache_position+position_ids":
                kw["cache_position"] = width + pos
            lp = torch.log_softmax(model(**kw).logits.float(), dim=-1)
            at = [
                (k, t - 1, conts[j][t])
                for k, (_, j) in enumerate(part)
                for t in range(1, len(conts[j]))
            ]
            vals = iter(lp[tuple(torch.tensor(at, device=device).T)].tolist())
            for i, j in part:
                res[i][j].extend(next(vals) for _ in range(1, len(conts[j])))
            del lp
        return res

    def nan_max(values):
        # NaN when any value is: `max` would keep or drop it by its place.
        return float("nan") if any(v != v for v in values) else max(values)

    def worst(a, b):
        pairs = zip(a, b, strict=True)
        return nan_max([abs(x - y) for f, c in pairs for x, y in zip(f, c, strict=True)])

    started = time.time()
    trials = {}
    ids_for = set(spec.get("ids_for", []))
    alone_for = set(spec.get("alone_for", []))
    check = set(spec.get("check_indices", []))
    batch = int(spec.get("row_batch", 16))
    rows = spec["rows"]
    rendered = {r["index"]: render(r["prompt"]) for r in rows}
    # Rows of a length together, so that a batch pads little; records still
    # carry their index.
    first3 = {r["index"] for r in rows[:3]}
    rows = sorted(rows, key=lambda r: len(rendered[r["index"]]))
    # The longest by tokens, which is what a cache past a sliding window sees.
    check |= {rows[-1]["index"]} if rows else set()

    def gap_of(group):
        return len(rendered[group[-1]["index"]]) - len(rendered[group[0]["index"]])

    def groups_of(most):
        # Consecutive rows, at most `most` of them and at most `batch_tokens`
        # tokens once padded to the longest; a row longer than that goes alone.
        groups, part = [], []
        for r in rows:
            n = len(rendered[r["index"]])
            if part and (len(part) == most or (len(part) + 1) * n > batch_tokens):
                groups.append(part)
                part = []
            part.append(r)
        groups = [*groups, part] if part else groups
        # The most padded batch first, the largest of those if several: the
        # trial below is where batching is allowed to fall back, so it has to
        # be the batch padding could break. The shortest rows are padded least.
        if groups:
            lead = max(range(len(groups)), key=lambda i: (gap_of(groups[i]), len(groups[i])))
            groups.insert(0, groups.pop(lead))
        return groups

    with open(spec["out"], "w", encoding="utf-8") as out:
        if rows:
            prompt_ids0 = rendered[spec["rows"][0]["index"]]
            full0 = score_full(prompt_ids0)
            for how in VARIANTS:
                try:
                    d = worst(full0, score_cached(prompt_ids0, how))
                except Exception as exc:  # noqa: BLE001 - recorded; the next one is tried
                    d = f"{type(exc).__name__}: {str(exc)[:200]}"
                trials[how] = d
                if isinstance(d, float) and d <= tol:
                    state["variant"] = how
                    break
        # Padding on the left needs the positions passed; without them, or
        # with no cache at all, a row is scored alone.
        batched = batch > 1 and state["variant"] in ("cache_position+position_ids", "position_ids")
        groups = groups_of(batch if batched else 1)
        trial_gap = gap_of(groups[0]) if batched and groups else None
        batch_trial = None
        g = 0
        while g < len(groups):
            part = groups[g]
            t0 = time.perf_counter()
            pids = [rendered[r["index"]] for r in part]
            if batched:
                try:
                    pers = score_batch(pids, state["variant"])
                except Exception as exc:  # noqa: BLE001 - recorded; scored one at a time
                    if g:
                        raise
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"
                    batch_trial = {"row_batch": batch, "error": error}
                    batched, groups = False, groups_of(1)
                    continue
            else:
                pers = [score_cached(pids[0], state["variant"])]
            recs = []
            for row, prompt_ids, per in zip(part, pids, pers, strict=True):
                rec = {"index": row["index"], "prompt_len": len(prompt_ids), "token_scores": per}
                if row["index"] in ids_for:
                    rec["prompt_ids"] = prompt_ids
                # Every row of the first batch, then the first three, the
                # longest and the asked-for: each against the row scored alone
                # with no cache.
                idx = row["index"]
                if g == 0 or idx in first3 or idx in check or idx in alone_for:
                    full = score_full(prompt_ids)
                    rec["check_max_abs_token"] = worst(full, per)
                    if idx in alone_for:
                        rec["token_scores"] = full
                recs.append(rec)
            # `not <=`, so that a NaN is a difference too.
            bad = [r for r in recs if not r.get("check_max_abs_token", 0.0) <= tol]
            if bad and g == 0 and batched:
                # Rows scored together drift from rows scored alone: score
                # one at a time, as before batching. JSON has no NaN.
                drift = nan_max([r["check_max_abs_token"] for r in bad])
                drift = drift if drift == drift else "nan"
                batch_trial = {"row_batch": batch, "max_abs_token": drift}
                batched, groups = False, groups_of(1)
                continue
            if bad:
                how_scored = "in a batch and alone" if batched else "cached and uncached"
                raise SystemExit(
                    f"row {bad[0]['index']}: scored {how_scored}, the scores differ by "
                    f"{bad[0]['check_max_abs_token']}"
                )
            each = round((time.perf_counter() - t0) / len(part), 4)
            for rec in recs:
                rec["seconds"] = each
                out.write(json.dumps(rec) + "\n")
            out.flush()
            if mps_samples is not None:
                sample_mps_memory(torch, mps_samples)
            g += 1
    Path(spec["report"]).write_text(
        json.dumps(
            {
                "device": device,
                "terminator": term,
                "bos_id": tok.bos_token_id,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cont_ids": conts,
                "final_logit_softcapping": cap,
                "logits_scaling": scale,
                "cache_variant": state["variant"] or "no_cache",
                "row_batch": batch if batched else 1,
                # The groups scored; one per row after a fallback.
                "batches": len(groups),
                "batch_trial": batch_trial,
                # The padding, in tokens, of the first batch, which was checked
                # row by row.
                "most_padding_checked": trial_gap if batched else None,
                # The most tokens of cache, prompts padded, a batched forward
                # was given, a first batch that fell back included. The prompt
                # cache and its copy are alive beside it.
                "batch_tokens_held": state["batch_tokens_held"] or None,
                "variant_trials": trials,
                "seconds": round(time.time() - started, 1),
                "mps_environment": environment_seen(),
                "mps_memory_sampled_max": finished_mps_samples(mps_samples),
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
)


ENGINE_SCRIPT = (
    r'''
"""Per-token option scores through LiteRT-LM's session API. Spec: argv[1].

See litetune.decision_scripts.
"""
import ctypes
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import litert_lm as L

# The self-check's message, rendered by the bundle's own template.
SELFCHECK_MESSAGE = "Answer with one word."
'''
    + SESSION_IDS_SOURCE
    + RUNTIME_ENGINE_SOURCE
    + r'''


def last_error(lib):
    try:
        f = lib.litert_lm_get_last_error_message
        f.restype = ctypes.c_char_p
        m = f()
        return m.decode("utf-8", "replace") if m else None
    except Exception as exc:  # noqa: BLE001
        return f"(no error message: {exc})"


def c_call(session, name, *args):
    """save_checkpoint / rewind_to_checkpoint, which the Python package does not wrap."""
    f = getattr(session._lib, name)
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    rc = f(session._ptr, *args)
    if rc != 0:
        raise RuntimeError(f"{name} returned {rc}: {last_error(session._lib)}")


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        L.set_min_log_severity(L.LogSeverity.ERROR)
    except AttributeError:
        pass
    greedy = L.SamplerConfig(top_k=1, top_p=1.0, temperature=1.0, seed=0)
    term = spec["terminator"]
    options = spec["options"]
    method = spec["method"]
    if method not in ("rewind", "fresh_engine"):
        raise SystemExit(f"unknown method {method!r}")
    report = {
        "runtime": {},
        "python": platform.python_version(),
        "threads": spec["threads"],
        "method": method,
        "engine_loads": 0,
    }
    for dist in ("litert-lm", "litert-lm-api", "litert-lm-nightly", "litert-lm-api-nightly"):
        try:
            report["runtime"][dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        report["cpu"] = next(
            line.split(":", 1)[1].strip()
            for line in open("/proc/cpuinfo")
            if line.startswith("model name")
        )
    except Exception:  # noqa: BLE001
        report["cpu"] = None

    def write_report():
        Path(spec["report"]).write_text(json.dumps(report), encoding="utf-8")

    def session(engine):
        return engine.create_session(apply_prompt_template=False, sampler_config=greedy)

    def score(s, text):
        r = s.run_text_scoring([text], store_token_lengths=True)
        return [float(v) for v in r.token_scores[0]] if getattr(r, "token_scores", None) else None

    def render(engine, prompt):
        with engine.create_conversation(sampler_config=greedy, max_output_tokens=1) as conv:
            rendered = conv.render_message_to_string(prompt)
        # 0.18.0 returns "" when the render fails in C, and "" tokenises to no
        # ids, which every check below would pass.
        if not rendered:
            raise RuntimeError("the runtime rendered the prompt as empty text")
        # As prefill reads the text; the session puts `session_prefix` first.
        ids = prefill_ids(engine, rendered)
        last = engine.detokenize(ids[-1:])
        if not rendered.endswith(last) or list(engine.tokenize(last)) != ids[-1:]:
            raise RuntimeError(f"the last prompt token {ids[-1]} does not round-trip as {last!r}")
        return rendered, ids, last

    on_gpu = spec.get("backend", "cpu") == "gpu"
    report["backend"] = "gpu" if on_gpu else "cpu"
    if on_gpu:
        # As the text path asks for the GPU: fp32 activations (export.GPU_ACTIVATION),
        # and what the bundle alone declares, which an app without the override gets.
        kw = engine_kwargs({"backend": "gpu", "activation_data_type": "fp32"})
        try:
            report["bundle_activation"] = bundle_activation(spec["model"])
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not end a run
            report["bundle_activation"] = None
            report["bundle_activation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        kw = {"backend": L.Backend.CPU(thread_count=int(spec["threads"]))}
    kw["cache_dir"] = ":nocache"

    def open_engine():
        report["engine_loads"] += 1
        return L.Engine(spec["model"], **kw)

    def decide(engine, mode, rendered, last, texts):
        out = []
        t0 = time.perf_counter()
        if method == "fresh_engine":
            # An Engine and a session of their own for every option, and no
            # rewind: for a family whose state a rewind does not reset
            # (`models.ModelRules.engine_per_prompt_reason`). The prompt, then
            # in mode `once` its last token again, then the option, as E7's
            # `fresh` and `fresh_nofix` (a session per option on one Engine)
            # did, each within 6e-4 nats of PyTorch on Gemma 3 on the engine
            # whose self-check verdict picks it (2026-10-16). `engine` only
            # renders and tokenises; it never prefills, so it holds no state.
            for text in texts:
                with open_engine() as fresh, session(fresh) as s:
                    s.run_prefill([rendered])
                    if mode == "once":
                        s.run_prefill([last])
                    out.append(score(s, text))
            return out, (time.perf_counter() - t0) * 1000
        with session(engine) as s:
            s.run_prefill([rendered])
            c_call(s, "litert_lm_session_save_checkpoint", b"p")
            # Once per decision, after the checkpoint. Rewinding resets only the
            # step, and the cell this prefill wrote stays. Measured 2026-10-16:
            # this sequence gave every option the scores of a re-prefill before
            # each one, bit for bit on 3,584 tokens of a float32 bundle and to
            # the six decimals stored on 108,300 int8 tokens, and within 8.2e-4
            # nats of PyTorch on that float32 bundle.
            if mode == "once":
                s.run_prefill([last])
            for text in texts:
                out.append(score(s, text))
                c_call(s, "litert_lm_session_rewind_to_checkpoint", b"p")
        return out, (time.perf_counter() - t0) * 1000

    def selfcheck_pair(engine):
        """Two prompts of one length that differ only in their second-to-last token, or None.

        The first is the self-check's message as the bundle's template renders
        it. The second puts another id of that rendering in place of its
        second-to-last, the first one, from the second id on, that gives text
        which tokenises back to exactly those ids. On Gemma 3 that is `user` for
        `model`: the pair the flutter_edge_ai design fixed and 2026-10-16
        measured. On another template it is whatever its rendering holds, and
        the gap it shows is not measured; only a gap of exactly 0.0 is read as
        dropped.
        """
        with engine.create_conversation(sampler_config=greedy, max_output_tokens=1) as conv:
            a = conv.render_message_to_string(SELFCHECK_MESSAGE)
        ia = list(engine.tokenize(a)) if a else []
        if len(ia) < 2:
            return None
        for other in dict.fromkeys(ia[1:] + ia[:1]):
            if other == ia[-2]:
                continue
            ib = ia[:-2] + [other] + ia[-1:]
            b = engine.detokenize(ib)
            if list(engine.tokenize(b)) == ib:
                return a, b
        return None

    def one_token(engine, prompt, z):
        if method == "fresh_engine":
            # On an Engine of its own, as every option is scored then.
            with open_engine() as fresh, session(fresh) as s:
                s.run_prefill([prompt])
                per = score(s, z)
        else:
            with session(engine) as s:
                s.run_prefill([prompt])
                per = score(s, z)
        if per is None or len(per) != 1:
            raise RuntimeError(f"the one-token target {z!r} scored {per}")
        return per[0]

    t0 = time.perf_counter()
    with open_engine() as engine:
        report["init_seconds"] = round(time.perf_counter() - t0, 3)
        # On the GPU, the kernel's GPU time for this process now and after the
        # rows, with the engine open: what evaluate.gpu_observed reads (macOS).
        device = {}
        if on_gpu:
            device["at_engine"] = device_report()
        report["device"] = device
        # What every session here reads before its prompt's ids, as a first turn.
        report["session_prefix"] = session_prefix(engine)
        cont_ids = [list(engine.tokenize(o + term)) for o in options]
        report["cont_ids"] = cont_ids

        # -- the self-check -------------------------------------------------------
        mode = spec["mode"]
        if mode == "auto":
            zs = []
            for j in range(min(3, len(options))):
                z = engine.detokenize([cont_ids[j][0]])
                if list(engine.tokenize(z)) == [cont_ids[j][0]]:
                    zs.append(z)
            # One pair, as the flutter_edge_ai design has it run: measured
            # 2026-10-16 on Gemma 3, its gap was 0.0 on 0.18.0 and at least
            # 0.135 with the fix, where a task's own prompts gave a gap as small
            # as 2.1e-5 with a near-certain target. The two must never pick
            # different modes.
            diffs, checked = [], []
            pair = selfcheck_pair(engine)
            if pair is not None:
                a, b = pair
                for z in zs:
                    for _ in range(2):
                        d = abs(one_token(engine, a, z) - one_token(engine, b, z))
                        diffs.append(d)
                checked.append(len(engine.tokenize(a)))
            verdict = (
                "no comparisons" if not diffs
                else "dropped" if max(diffs) == 0.0
                else "not dropped" if min(diffs) > 0.0
                else "mixed"
            )
            report["selfcheck"] = {
                "verdict": verdict,
                "comparisons": len(diffs),
                "max_abs_diff": max(diffs) if diffs else None,
                "min_abs_diff": min(diffs) if diffs else None,
                "prompt_lengths": checked,
                "pair": list(pair) if pair else None,
            }
            if verdict not in ("dropped", "not dropped"):
                report["error"] = f"the self-check could not tell the engine apart: {verdict}"
                write_report()
                return 3
            mode = "once" if verdict == "dropped" else "plain"
        report["mode"] = mode

        # -- the targets ---------------------------------------------------------
        ks = []
        for i, ids in enumerate(cont_ids):
            k = 1
            while any(j != i and other[:k] == ids[:k] for j, other in enumerate(cont_ids)):
                k += 1
                if k > len(ids):
                    report["error"] = f"option {options[i]!r} is a prefix of another option"
                    write_report()
                    return 3
            ks.append(k)
        texts, want, mismatches = [], [], []
        for i, (o, ids, k) in enumerate(zip(options, cont_ids, ks)):
            if spec["targets"] == "full" or k == len(ids):
                texts.append(o + term)
                want.append(ids)
                continue
            text = engine.detokenize(ids[:k])
            if list(engine.tokenize(text)) == ids[:k]:
                texts.append(text)
                want.append(ids[:k])
            else:
                mismatches.append({"option": o, "k": k, "text": text})
                texts.append(o + term)
                want.append(ids)
        report.update({"k": ks, "scored_lengths": [len(w) for w in want], "mismatches": mismatches})
        write_report()

        # -- the rows -------------------------------------------------------------
        with open(spec["out"], "w", encoding="utf-8") as out:
            for row in spec["rows"]:
                try:
                    rendered, ids, last = render(engine, row["prompt"])
                    per, ms = decide(engine, mode, rendered, last, texts)
                except RuntimeError as exc:
                    out.write(json.dumps({"index": row["index"], "error": str(exc)}) + "\n")
                    out.flush()
                    continue
                bad = [
                    options[k] for k, x in enumerate(per)
                    if x is None or len(x) != len(want[k]) or not all(math.isfinite(v) for v in x)
                ]
                rec = {"index": row["index"], "prompt_ids": ids, "ms": round(ms, 1)}
                if bad:
                    rec["error"] = f"no usable scores for {bad[:5]}"
                else:
                    rec["token_scores"] = per
                out.write(json.dumps(rec) + "\n")
                out.flush()
        if on_gpu:
            device["after_generation"] = device_report()
    write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
)
