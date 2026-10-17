"""The two scripts that score a decision's options: the float reference, and the runtime.

Each is source text, written to a file and run inside its stage environment
with a `spec.json`, as `evaluate.py` runs its generation scripts. Neither
imports litetune: the environments do not have it. Both write one JSON line per
row, flushed as it is scored, and a report.

**The reference** (`REFERENCE_SCRIPT`, `envs.TRAIN`) is the scorer the
option-scoring measurements of 2026-10-15 and 2026-10-16 ran: float32, eager
attention, TF32 off, on the GPU when there is one. Each row's prompt is
rendered as `tune` trains it, and every option is scored as its id followed by
the turn terminator, teacher-forced, every token's log-probability kept. One
forward over the prompt keeps its cache, and the options go in batches over
copies of it; on the first rows each option is also scored with no cache, and
the run stops if any token differs by more than `check_tol`.

**The runtime** (`ENGINE_SCRIPT`, `envs.RUNTIME`) scores through LiteRT-LM's
session API on the CPU, the way flutter_edge_ai's decider does. It renders with
`Conversation.render_message_to_string`, prefills the prompt, saves a
checkpoint, and per option scores and rewinds. LiteRT-LM 0.18.0 scores right
after a prefill without the prompt's second-to-last token (#3561): mode `once`
prefills the last token again, once per decision, which gives the reference's
scores on 0.18.0; mode `plain` does not, which is right on an engine with the
fix and wrong on 0.18.0. The self-check picks between them: two prompts that
differ only in their second-to-last token score identically on 0.18.0 and
differently with the fix. Measured 2026-10-16: 0.0 in all 42 comparisons on
0.18.0, at least 0.135 on a nightly with the fix; each mode within 8.2e-4 nats
of PyTorch on a float32 bundle, and wrong on the other engine.

Targets are each option's shortest unique prefix: the first k ids of
`id + terminator`, k the smallest no other option shares. A truncated target
goes to the runtime as text and must tokenise back to the same ids; one that
does not is scored in full and listed.
"""

from __future__ import annotations

REFERENCE_SCRIPT = r'''
"""Float32 per-token option scores for each row. Spec: argv[1]. See litetune.decision_scripts."""
import copy
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VARIANTS = ("cache_position+position_ids", "position_ids", "plain")


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    cap = getattr(model.config, "final_logit_softcapping", None)
    term = spec["terminator"]
    term_ids = tok(term, add_special_tokens=False)["input_ids"]
    if len(term_ids) != 1:
        raise SystemExit(f"the terminator {term!r} is {term_ids}, not one token")
    conts = [tok(o, add_special_tokens=False)["input_ids"] + term_ids for o in spec["options"]]
    if len({tuple(c) for c in conts}) != len(conts):
        raise SystemExit("two options tokenise to the same ids")
    pad = tok.pad_token_id if tok.pad_token_id is not None else term_ids[0]
    chunk = int(spec.get("chunk", 64))
    tol = float(spec.get("check_tol", 3e-3))
    state = {"variant": None}

    def render(prompt):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return tok(text, add_special_tokens=False)["input_ids"]

    def head_logp(hidden):
        logits = model.lm_head(hidden).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
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

    def prompt_forward(ids):
        try:
            return model(input_ids=ids, use_cache=True, logits_to_keep=1)
        except TypeError:
            return model(input_ids=ids, use_cache=True)

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
        out = prompt_forward(torch.tensor([prompt_ids], device=device))
        first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        cache = out.past_key_values
        res = []
        for s in range(0, len(conts), chunk):
            part = conts[s : s + chunk]
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

    def worst(a, b):
        return max(abs(x - y) for f, c in zip(a, b) for x, y in zip(f, c))

    started = time.time()
    trials = {}
    with open(spec["out"], "w", encoding="utf-8") as out:
        for n_done, row in enumerate(spec["rows"]):
            t0 = time.perf_counter()
            prompt_ids = render(row["prompt"])
            if n_done == 0:
                full0 = score_full(prompt_ids)
                for how in VARIANTS:
                    try:
                        d = worst(full0, score_cached(prompt_ids, how))
                    except Exception as exc:  # noqa: BLE001 - recorded; the next one is tried
                        d = f"{type(exc).__name__}: {str(exc)[:200]}"
                    trials[how] = d
                    if isinstance(d, float) and d <= tol:
                        state["variant"] = how
                        break
            per = score_cached(prompt_ids, state["variant"])
            rec = {"index": row["index"], "prompt_len": len(prompt_ids), "token_scores": per}
            if row["index"] in set(spec.get("check_indices", [])) or n_done < 3:
                d = worst(score_full(prompt_ids), per)
                rec["check_max_abs_token"] = d
                if d > tol:
                    raise SystemExit(
                        f"row {row['index']}: cached and uncached scores differ by {d}"
                    )
            rec["seconds"] = round(time.perf_counter() - t0, 4)
            out.write(json.dumps(rec) + "\n")
            out.flush()
    Path(spec["report"]).write_text(
        json.dumps(
            {
                "device": device,
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cont_ids": conts,
                "final_logit_softcapping": cap,
                "cache_variant": state["variant"] or "no_cache",
                "variant_trials": trials,
                "seconds": round(time.time() - started, 1),
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


ENGINE_SCRIPT = r'''
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

FIXED = "<start_of_turn>user\nAnswer with one word.<end_of_turn>\n<start_of_turn>model\n"
MODEL_TURN = "<start_of_turn>model\n"


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
    report = {"runtime": {}, "python": platform.python_version(), "threads": spec["threads"]}
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
        ids = list(engine.tokenize(rendered))
        last = engine.detokenize(ids[-1:])
        if not rendered.endswith(last) or list(engine.tokenize(last)) != ids[-1:]:
            raise RuntimeError(f"the last prompt token {ids[-1]} does not round-trip as {last!r}")
        return rendered, ids, last

    def decide(engine, mode, rendered, last, texts):
        out = []
        t0 = time.perf_counter()
        with session(engine) as s:
            s.run_prefill([rendered])
            c_call(s, "litert_lm_session_save_checkpoint", b"p")
            # Once per decision, after the checkpoint. Rewinding resets only the
            # step, and the cell this prefill wrote stays: measured 2026-10-16,
            # this sequence gave every option the scores of a re-prefill before
            # each one, bit for bit over 108,300 tokens, and within 8.2e-4 nats
            # of PyTorch on a float32 bundle.
            if mode == "once":
                s.run_prefill([last])
            for text in texts:
                out.append(score(s, text))
                c_call(s, "litert_lm_session_rewind_to_checkpoint", b"p")
        return out, (time.perf_counter() - t0) * 1000

    def one_token(engine, prompt, z):
        with session(engine) as s:
            s.run_prefill([prompt])
            per = score(s, z)
        if per is None or len(per) != 1:
            raise RuntimeError(f"the one-token target {z!r} scored {per}")
        return per[0]

    t0 = time.perf_counter()
    kw = {"backend": L.Backend.CPU(thread_count=int(spec["threads"])), "cache_dir": ":nocache"}
    with L.Engine(spec["model"], **kw) as engine:
        report["init_seconds"] = round(time.perf_counter() - t0, 3)
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
            # The fixed pair only, as flutter_edge_ai runs it: measured 2026-10-16,
            # its gap was 0.0 on 0.18.0 and at least 0.135 with the fix, where a
            # task's own prompts gave a gap as small as 2.1e-5 with a near-certain
            # target. The two must never pick different modes.
            pairs = [FIXED]
            diffs, checked = [], []
            for a in pairs:
                if not a.endswith(MODEL_TURN):
                    continue
                b = a[: -len("model\n")] + "user\n"
                ia, ib = list(engine.tokenize(a)), list(engine.tokenize(b))
                differing = [i for i in range(min(len(ia), len(ib))) if ia[i] != ib[i]]
                if len(ia) != len(ib) or differing != [len(ia) - 2]:
                    continue
                for z in zs:
                    for _ in range(2):
                        d = abs(one_token(engine, a, z) - one_token(engine, b, z))
                        diffs.append(d)
                checked.append(len(ia))
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
                rendered, ids, last = render(engine, row["prompt"])
                try:
                    per, ms = decide(engine, mode, rendered, last, texts)
                except RuntimeError as exc:
                    out.write(json.dumps({"index": row["index"], "error": str(exc)}) + "\n")
                    out.flush()
                    continue
                bad = [
                    options[k] for k, x in enumerate(per)
                    if x is None or len(x) != len(want[k]) or not all(math.isfinite(v) for v in x)
                ]
                rec = {"index": row["index"], "prompt_len": len(ids), "ms": round(ms, 1)}
                if bad:
                    rec["error"] = f"no usable scores for {bad[:5]}"
                else:
                    rec["token_scores"] = per
                out.write(json.dumps(rec) + "\n")
                out.flush()
    write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
