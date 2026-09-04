# litetune

**Fine-tune a small model for tool-calling, convert it to run on a phone, and
know what the conversion cost you.**

Getting from a Hugging Face checkpoint to a model that works inside your app is
a long road — LoRA, merge, export to `.litertlm`, bundle metadata — and a
mistake at any step produces a file of the right size that loads without error
and is broken while every check stays green. litetune walks that road and knows
the traps on it.

The output is a `.litertlm` bundle: what LiteRT-LM loads, and what the
`flutter_gemma` plugin runs on Android and iOS. The scope is **function
calling** — your data is prompts and the tool calls they should produce, and
"correct" means the tool name and every argument value.

- **prepare** — split the data, and reject rows that cannot be scored
- **tune** — LoRA or full fine-tuning, with the wiring the export needs
- **convert** — checkpoint to `.litertlm`, across quantization recipes
- **verify** — measure what the conversion cost, before you ship
- **bundle** — package the artifact with what was measured about it

Everything runs on CPU, which is workable at 270M and the first thing you will
want to change above about 1B. Bring your own checkpoint and skip the first two
steps, or bring a `.litertlm` and its float checkpoint and run only `verify`.

> **Alpha.** Measured end to end on `google/functiongemma-270m-it` only. Gemma 3,
> Gemma 4 and Qwen3.5 export — litetune carries their required flags — but no
> quality number has been established for them. Try it on yours and open an issue.

---

## Install

```bash
pip install litetune
```

Or through Homebrew, which brings its own Python 3.12:

```bash
brew install DenisovAV/tap/litetune
```

**Linux or macOS**, Python 3.10–3.12. On Linux you also need `libvulkan1` —
`litert-lm` `dlopen()`s a Vulkan-linked library even for the CPU backend, and
without it every invocation, `--help` included, dies in under a second:

```bash
sudo apt-get install -y libvulkan1     # Debian/Ubuntu
```

macOS needs nothing extra; Colab works out of the box; Windows is untried.

Python 3.13 runs `tune`, `prepare` and `bundle` but not `convert` or `verify`:
each stage builds its own environment from the interpreter you launched, and
`numpy==2.0.2` — pinned by the export toolchain — stops publishing wheels after
3.12. Past a ceiling the command refuses and names the pin that set it.

First run of `tune`, `convert` or `verify` builds that environment and pulls
`torch`: several gigabytes and a few minutes, cached afterwards.

---

## Your data

One JSON object per line. Scoring rows need a `prompt` and a `target`; training
rows add the `completion` text the model should produce:

```json
{"prompt": "set an alarm for 7", "target": {"name": "set_alarm", "args": {"hour": "7"}}}
```

`prepare` splits one raw file into `train.jsonl` and `heldout.jsonl` and rejects
what it cannot score: malformed JSON, and rows with no `prompt`. Given
`--tokenizer` it also reports the token-length distribution, so a row too long
for the sequence limit fails before you rent a GPU rather than after.

The held-out half is never trained on. Scoring a model on rows it was fitted to
measures memorisation rather than whether it answers new inputs. The split is
derived from the file's content hash, so re-running `prepare` puts the same rows
on the same side.

---

## Quick start

```bash
litetune prepare --data raw.jsonl --output-dir data --context-length 1024

litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \
              --output-dir tuned --prompt-mode prerendered --method lora

litetune convert --model tuned/model --output-dir artifacts \
                 --recipe dynamic_wi8_afp32 --recipe weight_only_wi8_afp32
```

That is a shippable `.litertlm` per recipe, under `artifacts/<recipe>/`. It is
also the part most tooling makes you assemble by hand — see
[What it knows](#what-it-knows-that-a-shell-script-does-not) for what those
three commands do that calling the exporter yourself does not.

Two more steps measure the result and package it:

```bash
# convert names the artifact; look it up rather than construct it
litetune verify --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --reference tuned/model --data data/heldout.jsonl \
                --json > manifest.json

litetune bundle --output-dir bundle \
                --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --declarations tools.json --prompt-mode prerendered \
                --base-model google/functiongemma-270m-it \
                --base-model-revision <commit-sha> \
                --adapter tuned/adapter \
                --train-metrics tuned/metrics.json \
                --verify-manifest manifest.json
```

`--reference` is the **float twin**: the same weights before conversion. That is
what makes the difference between the two the conversion cost rather than a
mixture of that and whatever training did. Point it at a different checkpoint —
an untuned base, say — and pass `--reference-role untuned_base`, and both the
training gain and the conversion cost come back unavailable, because one number
cannot separate two effects.

### Recipes

litetune knows four, and has measured two:

| recipe | |
|---|---|
| `dynamic_wi8_afp32` | the toolchain's default; its own docstring warns quality "may suffer" |
| `weight_only_wi8_afp32` | dequantizes before compute, so slower by an unmeasured amount |
| `dynamic_wi4_afp32` | 4-bit, unmeasured here |
| `weight_only_wi4_afp32` | 4-bit, unmeasured here |

`--recipe` has no default. A sweep of one is not a comparison.

### Other flags that decide something

| Flag | Why it matters |
|---|---|
| `--prompt-mode` | No default. `prerendered` means your app renders the tool declarations into the prompt and the runtime must not template again; `runtime_rendered` is the opposite. Must be the **same** value in `tune` and `bundle` — the wrong one produces a fluent wrong answer, not an error. |
| `--adapter` | For a LoRA run, pass `<tune output>/adapter`, from outside `--output-dir`. Without it the bundle carries only the merged weights. |
| `--base-model-revision` | Takes a commit sha. `main` and other moving refs are refused: they resolve to different weights on different days while the bundle reads identically. |
| `--wire-convention` | Which property order your tool declarations were rendered in. Optional; unset is recorded as unknown rather than guessed. See [MEASUREMENTS.md](MEASUREMENTS.md). |

---

## What it knows that a shell script does not

Each of these was paid for once, by an artifact that looked fine and was not.

**Export flags keyed on model identity.** They are in no documentation, and
`config.json` does not contain enough to derive them:

| family | flags litetune adds |
|---|---|
| `functiongemma` | `--litert_lm_model_type_override=function_gemma` |
| `gemma-3-text` | `--litert_lm_model_type_override=gemma3` |
| `gemma-4-e2b` | `--externalize_embedder`, `--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm` |

Without the first, FunctionGemma exports as a generic model — its `config.json`
says `gemma3_text`, which the exporter does not recognise, so it falls through a
silent catch-all. The runtime then builds no tool-call channel at all. An app
that parses the response text sees nothing wrong; an app that passes tools
natively receives no calls. The export succeeds, the file is the right size,
every liveness check is green.

**The terminator comes from the chat template, not from `eos_token_id`.** They
are not always the same token, and a model trained to emit the wrong one never
closes its turn — on a device it emits call after call, and a consumer that
delimits the reply by the turn marker cannot find the end of one.

**The adapter is saved before the merge.** A merged checkpoint cannot be
un-merged, so the rank-16 delta is the only form you can re-apply to a different
base, inspect, or ship on its own.

**`tokenizer.model` is carried back.** `transformers` 5.x stopped writing it and
the tokenizer classes stopped exposing `vocab_file`, so the exporter's
SentencePiece branch never fires and the bundle silently gets an HF tokenizer
section — losing FST-constrained decoding, which is SentencePiece-only. `tune`
copies the file back and records in `metrics.json` whether it managed to.

**Minimum `transformers` per family.** Gemma 4 and Qwen3.5 fail at tokenizer
load on every 4.x release, and Gemma 4 needs 5.5.0 for `AutoConfig` to recognise
the architecture. litetune refuses with the version rather than letting you find
out from an `AttributeError`.

**One environment per stage.** The training stack and the export toolchain pin
incompatible dependencies and cannot share an interpreter.

---

## Results

`functiongemma-270m-it`, LoRA on `google/mobile-actions`, scored on 640 examples
the model never trained on:

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base | 0.7266 | — | — |
| Fine-tuned | 0.9172 | 0.9016 | 0.9047 |
| Cost of conversion | — | +0.0156 *(within noise)* | +0.0125 |

Your gain from fine-tuning depends on your data. What this table is here to show
is the last row: conversion cost something, it was small, and one of the two
figures is not distinguishable from zero at this sample size.

The two artifacts are 0.04% apart in bytes. Nothing in file size, exit code or
logs separates them — running both against held-out data is the only thing that
does.

**[MEASUREMENTS.md](MEASUREMENTS.md)** has the intervals, three runs of the same
configuration and what they disagree about, and which published claims were
withdrawn after re-measurement.

---

## Limitations

**Known to be broken**

- **Passing tools natively fails.** Handing the runtime a tool list — `litert-lm
  --preset`, or the Kotlin `ConversationConfig.tools` — raises
  `litert_lm_conversation_send_message_stream failed` on a bundle built here.
  The cause is not known. Only the prompt-rendered path, which is what
  `flutter_gemma` uses, is supported and measured.
- **Peak memory is not bounded.** Training this model on 8,693 examples was
  OOM-killed at 32 GiB more than once. There is no preflight check; a death with
  no Python traceback is probably this.

**Limits on the numbers**

- **Measured on one model.** `functiongemma-270m-it`. Other families export but
  have no quality figure.
- **Measurement runs on CPU; your users run on a phone.** The GPU backend is
  reported to score below CPU on identical artifacts, and litetune has not
  measured that gap, so every number here is an optimistic estimate.
- **Two prompt renderings are in the field** for the same model, and they
  disagree for every declaration with more than one property. Costly on a base
  checkpoint, near-free after fine-tuning; `contract.json` records which you
  used. See [MEASUREMENTS.md](MEASUREMENTS.md).

**Not built yet**

- **Decoding parameters reach only one side.** litetune passes none to the
  device, so the reference is held to an explicit token limit while the device
  runs to the runtime's own. The manifest says so and counts unterminated
  generations.
- **Evaluation is slower than it needs to be** — one subprocess per prompt. A
  persistent `litert-lm serve` client is worth roughly thirtyfold.
- **`.litertlm` only**, no library API, and no single `run` command.

---

## Exit codes

The verdict is the exit code; the printed summary renders it. A run that
completed and a run that could not be judged must never be confused, which is
why there are five and not two.

| | `verify` | `prepare` / `tune` / `convert` | `bundle` |
|---|---|---|---|
| **0** | passed | passed | passed |
| **1** | scored below the threshold you set | failed | failed |
| **2** | inconclusive: the measurement cannot tell | — | **the default** |
| **3** | nothing established: no labelled data, or a difference that cannot be attributed | — | carried in |
| **4** | could not check — a harness fault, a bad command line, or a refused request | same | same |

**4 is not a failure of the model.** It means no answer could be obtained: a
missing shared library, a malformed input, a killed process — or a command line
litetune would not run. That last one is why a usage error exits 4 and not
argparse's usual 2: here 2 means *inconclusive*, which is a claim about a
measurement, and a typo should not produce one.

`bundle` carries a verdict rather than producing one, so it returns whatever
`--status` or `--verify-manifest` gave it. With neither it returns 2: bundling
re-measures nothing.

Wiring `|| exit 1` on anything non-zero throws all of this away.

---

## Contributing

Issues and pull requests welcome, particularly measurements on models other than
the one above — that is the gap this alpha most needs closed.

Run the checks with `pytest`, `ruff check`, `ruff format --check` and `mypy src`.

## License

Apache 2.0.
