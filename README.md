# litetune

**Fine-tune a small model, convert it to run on a phone, and know what the
conversion cost you.**

Getting from a Hugging Face checkpoint to a model that works inside your app is
a long road — LoRA, merge, export to `.litertlm`, bundle metadata — and a
mistake at any step produces a file of the right size that loads without error
and is broken while every check stays green. litetune walks that road and knows
the traps on it.

The output is a `.litertlm` bundle: what LiteRT-LM loads — natively on
Android, iOS, macOS, Linux and Windows, with GPU acceleration on each. The
`flutter_gemma` plugin runs it through the LiteRT-LM C API on all five;
Google's [AI Edge Gallery](https://github.com/google-ai-edge/gallery) loads
the file directly if you only want to try it on a device. Web exists as a
text-only preview that supports neither function calling nor LoRA, so a
tuned tool-calling model is native-only for now. The same model also runs on
a Snapdragon 8 Elite NPU — hand-compiled through litert-torch's `npu_export`
stages at the `cache_length` this project measured while isolating
[LiteRT-LM#3508](https://github.com/google-ai-edge/LiteRT-LM/issues/3508) —
and, for single-chunk prompts, on an Intel Lunar Lake NPU. The recipe, the
numbers and the conditions are under [Limitations](#limitations); the compile
is not in `convert`.

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

`tune` and `verify`'s float reference resolve the device once per run and use
CUDA if `torch.cuda.is_available()` reports one, CPU otherwise; `convert`
always runs on CPU, deliberately — export is a pure format conversion, and
pinning it keeps the measured export time reproducible on a runner with no
GPU at all. There is no `--device` flag; to force CPU on `tune` or `verify`
regardless of what the box has, set `CUDA_VISIBLE_DEVICES=""` in the shell
you launch it from — the stage subprocess inherits the environment. CPU alone
is workable at 270M and the first thing you will want to change above about
1B. Bring your own checkpoint and skip the first two steps, or bring a
`.litertlm` and its float checkpoint and run only `verify`.

> **Alpha.** Measured end to end on five models: `google/functiongemma-270m-it`
> with the tool-call scorer, and `google/gemma-3-270m-it`,
> `google/gemma-3-1b-it`, `Qwen/Qwen3-0.6B` and `Qwen/Qwen2.5-0.5B-Instruct`
> with `exact-text` on the same 77-way intent task — every conversion scored on
> CPU, two of them also on a phone's CPU and GPU, all in
> [MEASUREMENTS.md](MEASUREMENTS.md).
> Qwen3.5 exports and needs no flags from litetune, only a `transformers`
> floor. Gemma 4 exports once you name the variant — `E2B` or `E4B` — because
> the chat template override is per-variant; a bare `gemma-4` is refused
> rather than guessed at, and the refusal names the flag to pass if you want
> to choose the template yourself. Qwen3.5 has no quality number. Gemma 4 has a
> conversion cost measured on base weights — two conversions compared against
> the float reference, no training gain because nothing was fine-tuned — also
> in [MEASUREMENTS.md](MEASUREMENTS.md). Try it on yours and open an issue.

---

## Install

```bash
pip install litetune
```

Or through Homebrew, which brings its own Python 3.12:

```bash
brew install DenisovAV/tap/litetune
```

**Linux, macOS or Windows**, Python 3.10–3.12 — with `convert` needing Linux
x86_64 or an Apple Silicon Mac, for the reason below. On Linux you also need
`libvulkan1` —
`litert-lm` `dlopen()`s a Vulkan-linked library even for the CPU backend, and
without it every invocation, `--help` included, dies in under a second:

```bash
sudo apt-get install -y libvulkan1     # Debian/Ubuntu
```

macOS needs nothing extra; Colab works out of the box.

**Windows runs everything but `convert`.** `prepare`, `tune`, `verify` and
`bundle` install and run on Windows x64: every pin in those environments has a
wheel that installs there — `win_amd64` for the ones with native code,
`litert-lm-api` included, and `py3-none-any` for the rest. `convert` cannot,
because [`litert-converter`](https://pypi.org/project/litert-converter/) — the
MLIR converter `litert-torch` depends on — has never published a Windows wheel
or a source distribution, so the install fails while pip is still resolving.
Upstream tracks it as
[litert-torch#968](https://github.com/google-ai-edge/litert-torch/issues/968),
where a collaborator says Windows is planned and gives no date. Google's
[verified platforms](https://developers.google.com/edge/litert/cli/troubleshooting)
page says the same: "Windows: `litert compile` and `litert convert` not
supported yet." The same gap covers ARM Linux and Intel macOS. There are wheels
for macOS on Apple Silicon, though upstream's README still names Linux as its
operating system, and this project has only ever converted on Linux.

Two ways round it. Run everything under WSL2 with Ubuntu — install
`pythonX.Y-dev`, or the build stops on a missing `libpython3.x.so`
([litert-torch#74](https://github.com/google-ai-edge/litert-torch/issues/74)),
and keep the checkout and `~/.cache/litetune` on the Linux filesystem rather
than under `/mnt/c`. Or
convert on any Linux machine and bring the `.litertlm` back: `verify` takes it
with the float checkpoint it came from and asks nothing of the export
environment.

`tune` on Windows trains on the CPU unless you say otherwise, and that is
PyTorch's packaging rather than litetune's: every CUDA dependency of
`torch==2.5.1` is marked `platform_system == "Linux"`, so the wheel PyPI serves
Windows is the CPU build. `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124`
reaches the stage install and gets you the CUDA one — after `litetune env
--clean`, because an environment's identity is a hash of the interpreter and
the pins, which an index does not change, so a cached CPU environment would be
reused in silence. Either way the run records what it got: `device` and
`cuda_build` are in the training metrics.

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
{"prompt": "<start_of_turn>developer\nYou are a model that can do function calling with the following functions\n<start_function_declaration>declaration:set_alarm{description:<escape>Sets an alarm<escape>,parameters:{properties:{hour:{description:<escape>Hour of the alarm<escape>,type:<escape>STRING<escape>}},required:[<escape>hour<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>\n<end_of_turn>\n<start_of_turn>user\nset an alarm for 7\n<end_of_turn>\n<start_of_turn>model\n", "target": {"name": "set_alarm", "args": {"hour": "7"}}}
```

A bare string is the answer itself:

```json
{"prompt": "classify the sentiment: it was fine", "target": "neutral"}
```

Two shapes rather than a target plus a `--target-kind`, because those two could
disagree and a shape cannot disagree with itself. Match it with `--scorer` when
you get to `verify`.

**The prompt is exactly what your application will send the model.** An
application that renders the tool declarations and every turn marker into the
prompt itself sends a prompt the runtime must not template again, and `tune`
trains it `prerendered`. The sentiment row is bare text for a runtime that
applies the model's own chat template, so it trains `runtime_rendered` — as does
a tool call whose declarations the runtime renders, below in
[Tool calling through the runtime](#tool-calling-through-the-runtime). `tune`
tells the two apart by the control tokens in the prompts, and refuses a file
that mixes them unless you declare which one it is.

`prepare` splits one raw file into `train.jsonl` and `heldout.jsonl` and rejects
what it cannot score: malformed JSON, and rows with no `prompt`. Given
`--tokenizer` it also reports the token-length distribution, so a row too long
for the sequence limit fails before you rent a GPU rather than after.

The held-out rows — a fifth of the file by default, `--heldout-fraction` or
`--heldout-size` to change it — are never trained on. Scoring a model on rows it was fitted to
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

# 2. Fine-tune. Runs on CUDA if the box has one, otherwise CPU. Declare the
#    prompt mode rather than leave it to be read off the prompts: these carry
#    FunctionGemma's control tokens, so prerendered. It is recorded beside the
#    checkpoint, where steps 4 and 5 take it from. On a CPU add --dtype float32:
#    bfloat16 runs single-threaded there (see the flag table below).
litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \
              --output-dir tuned --method lora --prompt-mode prerendered

# 3. Convert, sweeping recipes rather than trusting a default.
litetune convert --model tuned/model --output-dir artifacts \
                 --recipe dynamic_wi8_afp32 --recipe weight_only_wi8_afp32

# 4. Measure what the conversion cost, against the float twin.
#    `convert` names the artifact; look the filename up rather than build it.
#    The prompt mode is read from the record step 2 left beside tuned/model,
#    and a --prompt-mode that disagrees with it is refused.
litetune verify --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --reference tuned/model --data data/heldout.jsonl \
                --json > manifest.json

# 5. Package the artifact with what was measured about it.
litetune bundle --output-dir bundle \
                --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --declarations tools.json \
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

The recipes are
[AI Edge Quantizer](https://github.com/google-ai-edge/ai-edge-quantizer)'s,
applied by `litert-torch export_hf` during `convert`, and `--recipe` passes a
name litetune does not know straight through to it. litetune defines exactly one
of its own, shipped as a quantizer recipe file inside the package; what it adds
to the rest is a measurement of what each costs on your task:

| recipe | |
|---|---|
| `dynamic_wi8_afp32` | the toolchain's default; its own docstring warns quality "may suffer" |
| `weight_only_wi8_afp32` | dequantizes before compute, so slower by an unmeasured amount |
| `dynamic_wi4_afp32` | 4-bit channelwise. Refused on three of the four models measured — a leaked `<bos>` on gemma-3-270m, degenerate repetition on Qwen3 and, past the gate, on 65 of 600 rows on gemma-3-1b. On Qwen2.5-0.5B it ran all 600 and cost +0.2567, a third of the model's accuracy |
| `weight_only_wi4_afp32` | 4-bit channelwise, dequantised before compute. Refused on all four, and always the same way: prompts that never finished. On Qwen2.5-0.5B the gate opened and one generation of 600 still timed out, so there is no score |
| `dynamic_wi4b32_afp32` | 4-bit in blocks of 32. Reached a score on all four; cost +0.0350 on a tuned Qwen3-0.6B, +0.0767 on a tuned Qwen2.5-0.5B, +0.0883 on a tuned gemma-3-1b and +0.3483 on a tuned gemma-3-270m |
| `dynamic_wi4b32_emb8_afp32` | litetune's own: those weights with int8 embeddings. +0.0550, +0.0917, +0.0883 and +0.3200 on the same four |

`--recipe` has no default. A sweep of one is not a comparison. **At four bits,
every model measured here lost accuracy the sample resolves, and how much does
not follow from its family or its parameter count: 3.50 points on a 0.6B
Qwen3, 7.67 on a 0.5B Qwen2.5, 8.83 on a 1B Gemma 3 and 34.83 on the 270M** — see
[MEASUREMENTS.md](MEASUREMENTS.md) for the intervals, the refusals and what
those numbers do not establish.

### Other flags that decide something

| Flag | Why it matters |
|---|---|
| `--prompt-mode` | Optional, never defaulted. `prerendered` means the prompt already carries its control tokens — your app renders the tool declarations into it — and the runtime must not template it again; `runtime_rendered` means the prompt is bare text and the runtime applies the model's chat template. Without the flag `tune` reads the mode off the training prompts (control tokens in at least 90% of them: `prerendered`; in at most 10%: `runtime_rendered`) and refuses a split in between. A declared mode the prompts contradict is refused unless you add `--force-prompt-mode`. `tune` records the mode beside the checkpoint; `verify` reads it through `--reference` and `bundle` through `--train-metrics`, and each refuses a different value — the wrong mode produces a fluent wrong answer, not an error. |
| `--adapter` | For a LoRA run, pass `<tune output>/adapter`, from outside `--output-dir`. Without it the bundle carries only the merged weights. |
| `--dtype` | Training precision for `tune`. Default `bfloat16`. On the one CPU measured, bfloat16 matmuls ran single-threaded, and `--dtype float32` trains on every core instead of one. It is not a mismatch with the rest of the pipeline — export passes no dtype at all, and the float reference always loads at float32 whatever this flag says. What it changes is comparability with a particular published run: [MEASUREMENTS.md](MEASUREMENTS.md) records the banking77 runs' dtype — bfloat16, trained on a GPU where this flag's reason does not apply — and says nothing about the headline table's, so the report records yours. |
| `--base-model-revision` | Takes a commit sha. `main` and other moving refs are refused: they resolve to different weights on different days while the bundle reads identically. |
| `--scorer` | What counts as correct, on `verify`. `tool-call` (default) or `exact-text`. It has to match the shape of your targets; nothing else in the pipeline changes. The manifest records which one ran, because two manifests scored differently are not comparable. |
| `--wire-convention` | Which property order your tool declarations were rendered in. Optional; unset is recorded as unknown rather than guessed. It applies to prompts your application renders. When the runtime renders the declarations, litetune settles the order itself — see [Tool calling through the runtime](#tool-calling-through-the-runtime). See [MEASUREMENTS.md](MEASUREMENTS.md). |

### Tool calling through the runtime

The walkthrough above renders the declarations into every prompt itself. The
other way is to let the runtime do it: bare prompts, and the declarations passed
as a file to `create_conversation(tools=...)`. Then they are an input to every
stage, not only to `bundle`:

```bash
litetune prepare --data raw.jsonl --output-dir data --context-length 1024 \
                 --tokenizer google/functiongemma-270m-it \
                 --base-model google/functiongemma-270m-it --declarations tools.json

litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \
              --output-dir tuned --method lora \
              --prompt-mode runtime_rendered --declarations tools.json

# convert as in step 3, then:
litetune verify --model artifacts/<recipe>/<name>.litertlm \
                --reference tuned/model --data data/heldout.jsonl \
                --declarations tools.json --json > manifest.json
```

`tools.json` is a list of OpenAI function objects, the shape `bundle` takes.

**The order of its keys is settled for you.** The runtime prints a declaration's
keys in the order it is given; FunctionGemma's own chat template sorts them with
`dictsort`, which ignores case. So litetune sorts every mapping in the file the
same way when it reads it, and the training prompt and the runtime then render
the same tokens — the rendering check compares them on every `verify`.

It refuses what the two still render differently, and what a call could not
carry back:

- any schema key the template does not print — among them `nullable`,
  `default`, `format`, `minimum`, `additionalProperties`, a function-level
  `strict`, and `enum` on anything but a string;
- a tool or property with no `description`;
- a property named `description`, `type`, `properties`, `required` or
  `nullable`, and two properties equal but for case;
- an empty collection, and an object property with no properties;
- a type written other than as one of the seven JSON Schema names, in lowercase
  or in capitals;
- a tool or argument name the runtime's call parser cannot read
  (`[a-zA-Z_][a-zA-Z0-9_.-]*`, and not `call`, `true`, `false`, `null` or an
  exponent like `e5` or `e-3`, which its lexer reads as other tokens);
- a key given twice in one object, a tool declared twice, and text that is not
  valid Unicode;
- a description, an enum value or a property name holding `<escape>`, a
  declaration marker or a turn marker, which would end the declaration it is
  written in.

Some are capability you give up — `nullable`, the reserved names, `enum` on a
number, and OpenAI's strict mode; the rest you fix by writing the file
differently. Whatever you remove, remove from what your application sends too:
the runtime renders what it is given. `google/mobile-actions` meets one of them
itself — its tools with no arguments carry `"properties": {}`. The disagreement
is Google's:
[LiteRT-LM#3638](https://github.com/google-ai-edge/LiteRT-LM/issues/3638).

**A call is trained the way the runtime reads one**: inside
`<start_function_call>` and `<end_function_call>`, strings between `<escape>`
markers, numbers, booleans and null bare, and the arguments in the same order as
the declarations — `call:set_alarm{hour:7,label:<escape>wake<escape>}`. Only what
the runtime reads back is trained: a string holding an escape, the end-of-call
marker or one of FunctionGemma's stop tokens (`<end_of_turn>`,
`<start_function_response>`, `<eos>`), NaN or infinity, an integer a double
cannot hold exactly and a name outside its grammar are refused with the row
named, and so is a target whose arguments contradict its declaration.
The order is not cosmetic: with constrained decoding on, the runtime enforces
the declared property order, and an argument out of it is dropped from the
call. So a `runtime_rendered` bundle ships its declarations in the order the
model learned, and its contract's `declarations_sha256` names that list.

**Who serves it this way.** LiteRT-LM's Python API, as
`create_conversation(tools=...)` handed each entry of the bundle's
`declarations.json` whole, `{"type": "function", ...}` — which is how `verify`
asks. A Python function handed as a tool gets a schema the binding writes
itself, and does not render what was trained. Constrained decoding is off
unless you pass a `ConstrainedDecodingConfig` that enables it. Automatic tool
calling is on unless you turn it off: with it on, the binding runs the tools
itself and loops until the model answers in prose, so you are never handed the
call; `verify` measures with it off.

On Kotlin, an `OpenApiTool` returning each entry's `function` object from the
bundle's `declarations.json` — not the whole `{"type": "function", ...}` entry,
which it refuses — registered in the file's order. Parse the file into a JSON
object that keeps its keys' order and hand it on; a data class serialised back
out can reorder the keys. The reflection-based `@Tool` path writes its own keys,
order and `nullable`, so it renders what was trained only for a tool declared
with no `parameters` at all. Constrained decoding there is
`ExperimentalFlags.enableConversationConstrainedDecoding`, off by default and
global to the process, read when a conversation is created. Automatic tool
calling is on by default in Kotlin too; turn it off to be handed the call. The
runtime reads every number in a call as a double, so an integer argument
arrives as `7.0`: read it as a number and convert it.

flutter_gemma 1.8.3 does not use this path for FunctionGemma: it renders the
declarations in Dart, and the `flutter_gemma_litertlm` engine (1.6.4) hands the
runtime tools only for Gemma 4 (`lib/src/ffi/ffi_inference_model.dart`). A model
trained here is not served by it the way it was measured.

**`verify` picks the path from the model, not from a flag.** With declarations
and a family whose runtime renders them, it asks the runtime for a structured
call instead of reading text, twice: with constrained decoding off — the
runtime's default, and what the reference is compared with — and on, which is
what an application that enables it gets. How far the two differ, and which way,
is reported. A prompt the runtime gives no reply to is scored as a wrong answer
and counted apart by reason: its call parser rejecting what the model wrote, or
a prompt reaching the bundle's token limit. Any other reason, on any prompt,
leaves that mode unmeasured — with the grammar off the run ends as a harness
failure, with it on the mode is reported as not measured and the grammar-off
number stands — and anything else going wrong ends the run. The reference's
text is read the way the runtime reads a reply, only between the call markers,
and a reply with more than one call is a wrong answer on both sides.

**What it refuses.** Structured targets in `runtime_rendered` without
`--declarations`: their descriptions and types are your application's contract
and cannot be read off the targets. `--declarations` for a `runtime_rendered`
split of a family whose runtime litetune does not record as rendering them. A
structured target for a family whose call format litetune has not measured:
supply each row's `completion` instead. In `tune`, a row with a target and no
completion — run `prepare`, which writes it — and a call row whose completion
the runtime would not read as exactly its target's call, with the target's
types, and nothing after it: no call markers, as in a split prepared by 0.1.6
or earlier, a number written as a string, text after the call, or a stop token
inside a string. Drop such a completion so `prepare` renders it. A call to a
tool the declarations do not offer, and a call in a row whose target is text.
Marked calls in `runtime_rendered` for a local checkpoint whose `config.json`
says `gemma3_text`, which is FunctionGemma and Gemma 3 alike: say which it is in
its `litetune.json`. `verify` without `--declarations`
for a `runtime_rendered` checkpoint that recorded some, and `--scorer
exact-text` on the tool path. In `prerendered` the declarations file's bytes are
the record, so the file `tune` read has to reach `verify` and `bundle`
unchanged. `prepare` without `--base-model` still renders FunctionGemma's
format, and the report says that it assumed it.

---

## What it knows that a shell script does not

Each of these was paid for once, by an artifact that looked fine and was not.

**Export flags keyed on model identity.** They are in no documentation, and
`config.json` does not contain enough to derive them:

| family | flags litetune adds |
|---|---|
| `functiongemma` | `--litert_lm_model_type_override=function_gemma`, `--jinja_chat_template_override=<litetune's own template>` |
| `gemma-3-text` | `--litert_lm_model_type_override=gemma3` |
| `gemma-4-e2b` | `--externalize_embedder`, `--jinja_chat_template_override=litert-community/gemma-4-E2B-it-litert-lm` |
| `gemma-4-e4b` | `--externalize_embedder`, `--jinja_chat_template_override=litert-community/gemma-4-E4B-it-litert-lm` |

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
is the whole error the Python binding raises. The split is in the runtime, so
every consumer that hands it tools sees it, whatever it is written in. litetune
ships a template the runtime can run and passes it on export. Measured on the
same checkpoint: with the override the runtime answers
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

**`metrics.json` records which device trained the checkpoint.** `tune`
resolves the device once, before training starts, and writes it to `device` as
`"cuda"` or `"cpu"` — the field is absent only in a `metrics.json` written by a
version of litetune that predates it, never a guessed value. It is the durable
answer to where a given checkpoint was trained.

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
| Fine-tuned | 0.6717 | 0.6717 | 0.6683 |
| Cost of conversion | — | +0.0000 *(within noise)* | +0.0033 *(within noise)* |

A different family, a different scorer, and this time neither conversion figure
clears its interval — the dynamic recipe lands on the same score as its float
twin while still disagreeing with it on 34 of 600 prompts. The base row says
*refused* because it was: asked for one intent label, the untuned model
returned nothing at all on 53 of 600 prompts, and `verify` stops at the liveness
tier rather than scoring a model that never answered. An earlier run of this
same pair, measured in the other prompt mode, is what found the `exact-text`
terminator bug fixed in 0.1.5 — see [MEASUREMENTS.md](MEASUREMENTS.md).

`Qwen3-0.6B`, LoRA on the same 2,400 rows, scored on the same 600:

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | *not scored* | — | — |
| Fine-tuned | 0.6983 | 0.6917 | 0.6817 |
| Cost of conversion | — | +0.0067 *(within noise)* | +0.0167 |

The first family measured here that litetune had no rule for. It exported with
no flag from litetune, and the rule it has now records that none is needed. The
weight-only figure clears its interval here where the dynamic one does not, and
where neither of Gemma 3's did — on 12 disagreements out of 600. Training and
the float reference ran on a GPU and the converted models on a CPU, so this cost
carries a hardware difference as well as a conversion one — as the Gemma 3 one
does too; every manifest in both runs records it — see
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
- **A Gemma 3 270M NPU bundle needs `cache_length` ≤ 896 at `prefill_128` on
  SM8750 and SM8850; above that it keeps only the first prefill chunk.**
  Google's published 270M
  and 1B `sm8750` bundles on a Galaxy S25, its 270M `sm8850` bundle on a
  Galaxy S26 and every litetune build at `cache_length` 1024 lose everything
  after the first 128-token chunk: a 245-token prompt with the question at the
  end is answered as if the question were not there, and a tool prompt is
  garbage — reported and bisected from this project as
  [LiteRT-LM#3508](https://github.com/google-ai-edge/LiteRT-LM/issues/3508).
  On SM8750 the loss tracks the compiled prefill graph's attention mask
  crossing about 1.0 MiB — `128 × (cache_length + 128) × 4 B × 2` for a
  `prefill_128` export, the size defect in
  [litert-torch#1184](https://github.com/google-ai-edge/litert-torch/issues/1184).
  Every bundle that failed in the issue is above that line; the one published
  bundle that works on SM8850 (1B, 1.375 MiB) is above it too, so the rule is
  a measured boundary for the 270M graphs on both SoCs, not yet a traced
  cause. **The boundary is per model, not a number to carry across.** Reading
  the SRQ graphs, the operand of the failing add is the int16 mask tiled over
  the query heads inside the compiled prefill graph —
  `2 B × num_attention_heads × prefill × (cache_length + prefill)` — which for
  Gemma 3's four heads is the same number as the count above over the bundle's
  two fp32 mask inputs, and for a sixteen-head model is not: Qwen3-0.6B is
  4.0 MiB at `cache_length` 896, still above the line
  ([LiteRT-LM#3508, 2026-09-11](https://github.com/google-ai-edge/LiteRT-LM/issues/3508)).
  At 896 the mask is exactly 1.0 MiB and all five chunks of a tool
  prompt survive; 768, further below the line, keeps the context too (checked
  on two-chunk prompts). So the FunctionGemma NPU bundle is
  `prefill_lengths = [128]`, `cache_length` 896: about 640 tokens of
  declarations plus request, and about 250 for the reply.
- **A Qualcomm NPU bundle loads only in an app whose QAIRT is at least as new
  as the compiler's.** The public `ai-edge-litert-sdk-qualcomm` 2.2.0 writes
  QAIRT 2.47 context binaries. An app whose QNN libraries are QAIRT 2.44
  (what the `flutter_gemma` example app loaded on 2026-09-07) fails with
  `Failed to create engine`; the logcat line is `Context binary (2.47.0) is
  newer than the current SDK (2.44.0)`. The `native-v0.16.0` runtime tarball
  from the same project carries 2.47 and loads Google's 2.44-built bundles as
  well as ours; it is the runtime behind the S25 numbers in this file. The
  Maven `litertlm-android` 0.16.1 AAR cannot reach the
  Qualcomm NPU with any public dispatch library
  ([LiteRT#6889](https://github.com/google-ai-edge/LiteRT/issues/6889)).
- **An Intel NPU keeps only the first prefill chunk, and not for the Qualcomm
  reason.** The `dynamic_wi8_afp32` graph of the same export (the OpenVINO
  compiler rejects the static-int8 stage that follows it), compiled for Lunar
  Lake with `ai-edge-litert-sdk-intel` 2.2.0 and run on a Core Ultra 7 258V
  through the LiteRT-LM C API, answers single-chunk prompts 20/20 and loses
  every multi-chunk prompt regardless of `cache_length` — byte-identical
  output at 896 and 1024. So the mask-size rule is Qualcomm's alone; Intel's
  cause is not found. As of 2026-09-08 there was no Google Gemma 3 Intel
  bundle to compare against.
- **Gemma 4 cannot be built for an NPU with the public exporter.** On the
  transformers this tool pins (5.16.1; ≥ 5.14 removes `global_head_dim`) the
  split-cache NPU export in litert-torch-nightly 0.10.0.dev20260826 fails
  before the graph, and a bundle built on transformers 5.13.1 loads on the
  Snapdragon 8 Elite and returns garbage: Google's published Qualcomm Gemma 4
  bundle carries a per-layer-embedder layout the public exporter does not
  produce, and the runtime's NPU path expects that layout. The regular
  CPU/GPU export of Gemma 4 completes on the same pin, and a CPU one now has a
  conversion-cost figure — see [MEASUREMENTS.md](MEASUREMENTS.md).

**Limits on the numbers**

- **Measured on six models, five of them fine-tuned here.**
  `functiongemma-270m-it` with the tool-call scorer, and `gemma-3-270m-it`,
  `gemma-3-1b-it`, `Qwen3-0.6B` and `Qwen2.5-0.5B-Instruct` with `exact-text`,
  each with a conversion
  cost against its own float twin. Only FunctionGemma also has a training gain:
  no banking77 run has an untuned base figure to subtract — Gemma 3 270M's base
  was run and refused to score, the 1B's and Qwen2.5's scored 0.0000 on both
  sides, and Qwen3's never reached the base step at all — and every one of those manifests
  records the gain as unavailable. `gemma-4-E2B-it` was not
  fine-tuned at all: base weights, two conversions of them compared against the
  float reference, so that run has a conversion cost and no training gain.
  Qwen3.5 exports but has no quality figure, and a Gemma 4 without its variant
  is still refused unless you supply the template override yourself.
- **The turn-terminator vocabulary is a static list.** `exact-text` scoring and
  the liveness checks both trim against a fixed set of strings, recorded
  verbatim at `harness.terminators` in every verify manifest. A family whose
  chat template closes a turn with a marker outside that list would otherwise
  have its reference score at zero, which is a fact about the vocabulary
  rather than the model — so `verify` stops before scoring either side and
  reports a harness failure instead of a conversion cost. That is the safe
  direction, not a fix: the run still cannot be measured until the vocabulary
  knows the marker. Gemma 4 is the case that proved it — it closes a turn with
  `<turn|>`, all 600 reference generations ended there, and the comparison was
  refused until the vocabulary learned the marker. Resolving it from the bundle contract instead of
  hardcoding is a follow-up. To find your own
  model's marker before then, `tune` records it at `turn_terminator.text` in
  `metrics.json` and `bundle` carries it into `contract.json`'s `stop_tokens`.

- **In `runtime_rendered`, `verify` refuses to compare two sides that were shown
  different prompts.** Before generating anything it renders every held-out
  prompt through the runtime's own conversation path and through the
  reference's chat template, and compares the token ids; on the first 8, or on
  all of them if the split is shorter, it also compares the prefill count the
  runtime reports when the prompt is actually sent. Any difference is a harness failure (exit 4) with the prompt, both
  counts and the first differing position at `harness.rendering_check`, not a
  conversion cost. On every tuned export of both families all 600 banking77
  prompts matched, and on the `gemma-3-270m-it` base export too; a rendering that adds
  an empty `<think></think>` is refused on every prompt, which is constructed
  in the tests rather than seen in a run. Reasoning is removed from both sides before scoring,
  through the last `[/thought]` or `</think>`, and counted per side at
  `measurements.<side>.reasoning_removed`, including generations that never
  closed it.
- **The candidate is pinned to CPU; the reference is not, and your users run
  on a phone.** On one Snapdragon
  Galaxy S24 (`SC-51E`), the `dynamic_wi8_afp32` bundle on the device's CPU
  scored 0.8703 ±0.026 on the 640 held-out rows against 0.8906 for the cloud
  CPU run that produced it (run A in [MEASUREMENTS.md](MEASUREMENTS.md); runs
  B and C scored 0.9016 and 0.8969, both just outside that interval). So the
  reference number predicted the phone to within about 0.03. One device, one
  recipe.
- **On a GPU box, the reference and the candidate can run on different
  hardware, and it is recorded rather than refused.** `build_backends` pins
  the candidate to `litert-lm`'s CPU backend and lets the reference resolve
  its own device; where the reference lands on `cuda`, the two sides differ
  in hardware as well as in conversion. `harness.device_mismatch` in the
  manifest names both devices and both engines, and the same text is carried
  into the run's limitations, so the conversion-cost number does not silently
  carry a hardware difference too. Refusing the comparison instead would
  leave such a machine unable to verify at all.
- **The GPU number is 20 rows.** Same device, same bundle, GPU backend: 20/20
  tool names and 15/20 exact (CPU: 20/20, 14/20) at 1.8× the CPU speed — with
  `prefer_activation_type = fp32` in the bundle. Without it the GPU text
  executor computes in F16 and returns `<pad>` floods and invented tool names
  (3/20), while the engine reports success. `convert` writes that key into
  every bundle it produces that does not already declare one; `--json` records
  what each carries as `exports[].gpu_activation`, and a bundle that could not
  be repacked is named in the limitations and is CPU-only. litetune cannot
  drive a phone GPU from a laptop, so a device run is a separate job.
- **The NPU number is 20 rows on one SoC.** The same
  `functiongemma-270m-it` (base weights, not the fine-tune) through
  litert-torch's `npu_export` stages for a Snapdragon 8 Elite (`SM8750`,
  Galaxy S25), on the phone's NPU through the LiteRT-LM C API, greedy,
  `cache_length` 896. Five-chunk tool prompts of 579–635 tokens produced a
  parseable call on 13/20 rows, the right tool name on 12/20 and an exact
  match on 3/20, at a median 441 ms per prompt including that prefill. The
  same static-int8 graph on the CPU interpreter scored 20/20, 18/20 and 12/20,
  so the nine exact rows the NPU loses are its own decode: one to syntax the
  runtime's parser rejects (a stray `}`), one to a prose reply instead of a
  call, one to the wrong tool named first of two calls, six to a wrong or
  missing argument in an otherwise well-formed call.
  Not diagnosed beyond that; not yet measured on the fine-tuned weights. The
  full table across prompt sets is in [MEASUREMENTS.md](MEASUREMENTS.md).
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
- **`convert` does not compile for an NPU.** The NPU bundle described under
  *Known to be broken* was built by hand: the four functions in litert-torch's
  `generative/export_hf/experimental/npu_export/stages.py` — `npu_export`,
  `npu_calibrate` on 64 tool prompts disjoint from the scored set,
  `npu_quantize`, `npu_compile` — with the compile on Linux x86_64 through
  `ai-edge-litert-sdk-qualcomm` 2.2.0, one compile per SoC, `cache_length`
  896. The device outputs came from a C-API harness that is not in this
  repository and were scored outside `verify` (litetune's call parser plus a
  short comparison script); `verify` has no device mode. litert-torch's
  documented one-step `export-hf --aot_backend=qualcomm` is not this path: its
  float bundle of `gemma-3-270m-it` loads on the HTP and is garbage on every
  row, single-chunk included. Until the
  stages are in `convert` and the fine-tuned figure exists, "NPU" here means
  the four NPU bullets under *Known to be broken*, not a flag.
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

Issues and pull requests welcome, particularly measurements on models other
than the four above — that is the gap this alpha most needs closed.

Run the checks with `pytest`, `ruff check`, `ruff format --check` and `mypy src`.

`scripts/ci-local.sh` runs what CI runs, on the platform CI runs it on: the
three interpreters in Linux containers, then the wheel job. It needs podman or
`ENGINE=docker`, and writes nothing to your working tree. Worth the minute
before sending anything that touches processes, signals or paths — a green
suite on macOS and a red CI run happened on the same commit twice, because
`os.waitid` is absent from CPython there before 3.13 and the same test took a
different branch on each.

## License

Apache 2.0.
