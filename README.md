# litetune

**Fine-tune a small model, convert it to run on a phone, and know what the
conversion cost you.**

Getting from a Hugging Face checkpoint to a model that works inside your app is
a long road — LoRA, merge, export to `.litertlm`, bundle metadata — and a
mistake at any step produces a file of the right size that loads without error
and is broken while every check stays green. litetune walks that road and knows
the traps on it.

The output is a `.litertlm` bundle: what LiteRT-LM loads — natively on
Android, iOS, macOS, Linux and Windows, with GPU acceleration on each and
NPU on Snapdragon and Intel. The `flutter_gemma` plugin runs it through the
LiteRT-LM C API on all five;
Google's [AI Edge Gallery](https://github.com/google-ai-edge/gallery) loads
the file directly if you only want to try it on a device. Web exists as a
text-only preview that supports neither function calling nor LoRA, so a
tuned tool-calling model is native-only for now.

Any task shaped as prompt → completion works. What "correct" means is the one
thing you choose: `--scorer tool-call` for function calling, where the operation
name and every argument value must match, or `--scorer exact-text` where there
is one right string. Everything after scoring — the paired comparison, the
intervals, whether a difference resolves, the exit code — reads only whether
each example was right, so it does not know or care which task you brought.

- **prepare** — split the data, and reject rows that cannot be scored
- **tune** — LoRA or full fine-tuning, with the wiring the export needs
- **convert** — checkpoint to `.litertlm`, across quantization recipes
- **verify** — measure what the conversion cost, before you ship
- **bundle** — package the artifact with what was measured about it

`litetune env` shows the environments the stages cached, and `--clean` removes
them.

Everything runs on CPU, which is workable at 270M and the first thing you will
want to change above about 1B. Bring your own checkpoint and skip the first two
steps, or bring a `.litertlm` and its float checkpoint and run only `verify`.

> **Alpha.** Measured end to end on two models: `google/functiongemma-270m-it`
> with the tool-call scorer, and `google/gemma-3-270m-it` with `exact-text` on a
> 77-way intent task — both on CPU, both in [MEASUREMENTS.md](MEASUREMENTS.md).
> Gemma 4 and Qwen3.5 export — litetune carries their required flags — but no
> quality number has been established for them. Try it on yours and open an
> issue.

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

Each of `tune`, `convert` and `verify` builds its environment on first use and
caches it. Measured on macOS:

| stage | pulls | size |
|---|---|---|
| `verify` | both of the below | **~740 MB** |
| `tune` | `torch`, `transformers`, `peft` | 588 MB |
| `convert` | the `litert-torch` export toolchain | **1.6 GB** |
| `bundle`, `prepare` | nothing | — |

On Linux the training environment is larger: the `torch` wheel pulls its CUDA
dependencies, several hundred megabytes each.

`litetune env` shows what is on disk and `litetune env --clean` removes it; the
next stage that needs one rebuilds it. Worth knowing because a provision that
died halfway leaves a directory that looks like a working one from the outside,
and because the cache key includes the interpreter — running litetune under two
Pythons builds two sets.

---

## Your data

One JSON object per line. Scoring rows need a `prompt` and a `target`; training
rows add the `completion` text the model should produce, or let `prepare` derive
it from the target.

**The shape of the target is how you declare the task.** An object with a `name`
is a tool call:

```json
{"prompt": "set an alarm for 7", "target": {"name": "set_alarm", "args": {"hour": "7"}}}
```

A bare string is the answer itself:

```json
{"prompt": "classify the sentiment: it was fine", "target": "neutral"}
```

Two shapes rather than a target plus a `--target-kind`, because those two could
disagree and a shape cannot disagree with itself. Match it with `--scorer` when
you get to `verify`.

`prepare` splits one raw file into `train.jsonl` and `heldout.jsonl` and rejects
what it cannot score: malformed JSON, and rows with no `prompt`. Given
`--tokenizer` it also reports the token-length distribution, so a row too long
for the sequence limit fails before you rent a GPU rather than after.

The held-out half is never trained on. Scoring a model on rows it was fitted to
measures memorisation rather than whether it answers new inputs. The split is
derived from the file's content hash, so re-running `prepare` puts the same rows
on the same side.

---

## From your data to a shippable bundle

Five commands, in order. Each is separate because each fails differently, and a
single `run` would hide which one you are in.

```bash
# 1. Split, and reject rows that cannot be scored. Seconds.
#    Without --tokenizer it cannot measure token lengths, so it splits the file
#    and exits 4 — "could not check" — rather than implying the rows all fit.
litetune prepare --data raw.jsonl --output-dir data --context-length 1024 \
                 --tokenizer google/functiongemma-270m-it

# 2. Fine-tune. On CPU, so size your expectations accordingly.
litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \
              --output-dir tuned --prompt-mode prerendered --method lora

# 3. Convert, sweeping recipes rather than trusting a default.
litetune convert --model tuned/model --output-dir artifacts \
                 --recipe dynamic_wi8_afp32 --recipe weight_only_wi8_afp32

# 4. Measure what the conversion cost, against the float twin.
#    `convert` names the artifact; look the filename up rather than build it.
litetune verify --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --reference tuned/model --data data/heldout.jsonl \
                --json > manifest.json

# 5. Package the artifact with what was measured about it.
litetune bundle --output-dir bundle \
                --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --declarations tools.json --prompt-mode prerendered \
                --base-model google/functiongemma-270m-it \
                --base-model-revision <commit-sha> \
                --adapter tuned/adapter \
                --train-metrics tuned/metrics.json \
                --verify-manifest manifest.json
```

**Step 3 already gives you something shippable** — one `.litertlm` per recipe,
under `artifacts/<recipe>/`. Steps 1–3 are also the part most tooling makes you
assemble by hand; see [What it knows](#what-it-knows-that-a-shell-script-does-not)
for what they do beyond calling the exporter yourself.

**Steps 4 and 5 are what makes it trustworthy.** `--reference` is the **float
twin**: the same weights before conversion. That is what makes the difference
between the two the conversion cost rather than a mixture of that and whatever
training did. Point it at a different checkpoint — an untuned base, say — and
pass `--reference-role untuned_base`, and both the training gain and the
conversion cost come back unavailable, because one number cannot separate two
effects.

If you already have a `.litertlm` and the checkpoint it came from, step 4 runs on
its own.

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
| `--scorer` | What counts as correct, on `verify`. `tool-call` (default) or `exact-text`. It has to match the shape of your targets; nothing else in the pipeline changes. The manifest records which one ran, because two manifests scored differently are not comparable. |
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

**A trained checkpoint is told what it came from.** `tune` writes
`litetune.json` beside the model, and `convert` reads it. This is not
bookkeeping: the per-family export flags key on the model's *name*, a directory
has none, and `transformers` 5.x deletes `_name_or_path` from `config.json` on
save — so without it a checkpoint this tool produced would export with none of
the flags its family requires, successfully and silently. `config.json` cannot
stand in: FunctionGemma and Gemma 3 270M/1B all declare `model_type:
gemma3_text` and need different values. A checkpoint from elsewhere says so with
`--base-model` or `--train-metrics`; one that says nothing is refused rather than
guessed at, because guessing wrong ships a bundle with no tool-call channel that
passes every check.

**The prompt template has to be one the device can execute.** FunctionGemma's
own template uses `macro` and `dictsort`; LiteRT-LM renders with MiniJinja,
which supports neither. A bundle carrying it exports cleanly, is the right size,
passes every liveness check, and still answers a plain text prompt — then
fails the native tool-call path, where LiteRT-LM routes the call through the
chat template, with `litert_lm_conversation_send_message_stream failed`, which
is the whole error the caller gets. The split is in the runtime, so every
consumer sees it: the `flutter_gemma` plugin, the AI Edge Gallery, or the SDK
used directly. litetune ships a template the runtime can run and passes it on
export. Measured on the same checkpoint: with the override the runtime answers
`[tool_call] set_alarm{hour:7}`; without it, `INTERNAL: Failed to apply
template`.

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

`gemma-3-270m-it`, LoRA on `mteb/banking77` (77-way intent, `--scorer
exact-text`), scored on 600 examples the model never trained on:

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | *refused* | — | — |
| Fine-tuned | 0.6933 | 0.6767 | 0.6917 |
| Cost of conversion | — | +0.0167 *(within noise)* | +0.0017 *(within noise)* |

A different family, a different scorer, the same last row. The base row says
*refused* because it was: asked for one intent label, the untuned model repeats
itself on 571 of 600 prompts, and `verify` stops at the liveness tier rather
than scoring a model that never answered. This run is also what found the
`exact-text` terminator bug fixed in 0.1.5 — see
[MEASUREMENTS.md](MEASUREMENTS.md).

**[MEASUREMENTS.md](MEASUREMENTS.md)** has the intervals, three runs of the same
configuration and what they disagree about, and which published claims were
withdrawn after re-measurement.

---

## Limitations

**Known to be broken**

- **The model does not always stop on the device.** 8 of 640 responses ran to
  the token limit — two carried 350 and 351 identical calls — and 13.4% carried
  more than one call, against 5% on the cloud CPU. The bundle names
  `<end_of_turn>` and `<start_function_response>` as stop tokens, so whatever
  stops the float path is not stopping this one. Not diagnosed.
- **Peak memory is not bounded.** Training this model on 8,693 examples was
  OOM-killed at 32 GiB more than once. There is no preflight check; a death with
  no Python traceback is probably this.

**Limits on the numbers**

- **Measured on two models.** `functiongemma-270m-it` end to end with the
  tool-call scorer, and `gemma-3-270m-it` with `exact-text`. Gemma 4 and Qwen 3.5
  export but have no quality figure.
- **The turn-terminator vocabulary is a static list.** `exact-text` scoring and
  the liveness checks both trim against a fixed set of strings, recorded
  verbatim at `harness.terminators` in every verify manifest. A family whose
  chat template closes a turn with a marker outside that list has its reference
  scored at zero, which is a fact about the vocabulary rather than the model —
  so `verify` refuses the comparison rather than reporting the difference as a
  conversion cost. That is the safe direction, not a fix: the run still cannot
  be measured until the vocabulary knows the marker. Resolving it from the
  bundle contract instead of hardcoding is a follow-up. To find your own
  model's marker before then, `tune` records it at `turn_terminator.text` in
  `metrics.json` and `bundle` carries it into `contract.json`'s `stop_tokens`.

- **bfloat16 on a CPU may train on one core.** The default dtype matches export
  and evaluation. On the one CPU measured (Apple M-series, torch 2.5.1) it ran a
  300-step LoRA on a single core for 52 minutes without finishing; the same run
  with `--dtype float32` took 307 s on ten threads. `tune` says so when it sees
  a CPU run at bfloat16, and records the dtype mismatch if you switch.
- **Measurement runs on CPU; your users run on a phone.** On one Snapdragon
  Galaxy S24 (`SC-51E`), the `dynamic_wi8_afp32` bundle on the device's CPU
  scored 0.8703 ±0.026 on the 640 held-out rows against 0.8906 for the cloud
  CPU run that produced it (run A in [MEASUREMENTS.md](MEASUREMENTS.md); runs
  B and C scored 0.9016 and 0.8969, both just outside that interval). So the
  reference number predicted the phone to within about 0.03. One device, one
  recipe.
- **The GPU number is 20 rows.** Same device, same bundle, GPU backend: 20/20
  tool names and 15/20 exact (CPU: 20/20, 14/20) at 1.8× the CPU speed — with
  `prefer_activation_type = fp32` in the bundle. Without it the GPU text
  executor computes in F16 and returns `<pad>` floods and invented tool names
  (3/20), while the engine reports success. `convert` writes that key into
  every bundle it produces that does not already declare one; `--json` records
  what each carries as `exports[].gpu_activation`, and a bundle that could not
  be repacked is named in the limitations and is CPU-only. litetune cannot
  drive a phone GPU from a laptop, so a device run is a separate job.
- **Two prompt renderings are in the field** for the same model, and they
  disagree for every declaration with more than one property. Costly on a base
  checkpoint, near-free after fine-tuning; `contract.json` records which you
  used. See [MEASUREMENTS.md](MEASUREMENTS.md).

**Not built yet**

- **Nothing here configures the app that loads the bundle.** An app targeting
  API 31+ must declare `<uses-native-library android:name="libOpenCL.so"
  android:required="false"/>` (plus the `-pixel`/`-car` names the loader also
  tries) or the runtime reports "Can not find OpenCL library on this device":
  in that case it is the missing declaration, not the device — though the
  same string covers devices with no public OpenCL at all. Set an output cap
  and a `maxNumTokens`: the bundle's KV cache is 4096 and a run that does not
  stop fills it (13 of 20 GPU rows ran to 3,500 `<pad>` tokens and 53 s each
  before the fp32 fix; capping at 1024 cut that to 6 s by cutting the garbage,
  not by fixing it). And give `EngineConfig.cacheDir` a writable directory.
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
