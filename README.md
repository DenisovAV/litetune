# litetune

**Fine-tune, convert and verify small models for on-device. Know what the
conversion cost you.**

Three things, and only the last one is unusual:

- **tune** — supervised fine-tuning, optional (bring your own checkpoint and skip it)
- **convert** — checkpoint to `.litertlm`, across quantization recipes
- **verify** — measure what conversion cost, before you ship

Converting a model to run on a phone changes its answers. Most tooling will tell
you the conversion succeeded. This tells you what it cost.

> **Alpha.** Measured end to end on one model: `functiongemma-270m-it` on
> Google's `mobile-actions` dataset. Other models are not claimed — the code is
> not specific to this one, but nothing else has been measured, and this project
> exists to keep those two statements apart. Try it on yours and tell us what
> broke.

## What the numbers look like

`functiongemma-270m-it`, LoRA fine-tuned on `mobile-actions`, measured on 640
held-out single-call examples. Exact match means the tool name **and** every
argument value.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | 0.7266 ±0.0345 | — | — |
| Fine-tuned | **0.9172** ±0.0214 | 0.9016 ±0.0231 | 0.9047 ±0.0228 |
| Cost of conversion | — | +0.0156 ±0.0162 *(unresolved)* | **+0.0125** ±0.0123 |

Fine-tuning gained **+0.1906 ±0.0357**, measured paired: only the examples where
the two models disagree carry signal, and 136 of 640 did.

Pairing is not a refinement for its own sake. `weight_only_wi8_afp32` costs
**+0.0125**, and the estimate is the same either way — what changes is what can
be said about it. Paired, the interval is **±0.0123**, just clear of zero, so
the effect is resolved. Unpaired it is **±0.0312**, which straddles zero and
would be reported as noise. Both sides answered the same 640 prompts; treating
them as independent samples throws away exactly the information that settles the
question.

### The base figure is measured under a rendering the base did not learn

**Read the base column with this in mind.** The declaration properties in every
prompt above are rendered in *declaration order*, which is what the reference
consumer (`flutter_gemma`) emits. The jinja template inside the `.litertlm` we
ship renders them with `dictsort` — measured, eight occurrences, including on
declaration properties. On this dataset the two orders disagree for **100% of
rows**, so one bundle presents two different prompts depending on which path a
consumer takes.

Which order the weights prefer was argued rather than measured until it was
measured. Same greedy decode, same parser, one variable:

| | n | declaration order | `dictsort` | paired difference | discordant pairs |
|---|---|---|---|---|---|
| Base, held-out | 640 | 0.7266 | **0.7625** | −0.0359 ±0.0219 | 51 |
| Base, disjoint sample | 1280 | 0.7602 | **0.7789** | −0.0187 ±0.0099 | 42 |
| **Fine-tuned**, held-out | 640 | 0.9109 | 0.9141 | −0.0031 ±0.0087 *(unresolved)* | **8** |

**The base cares and the fine-tuned model does not.** For the base both runs
resolve, both favour `dictsort`, the intervals overlap at [−0.0286, −0.0140],
and the discordant pairs run three to one. After fine-tuning the same comparison
collapses: 51 discordant pairs become 8, and the difference is 0.0031 against an
interval of ±0.0087 — unresolved not for want of data but for want of an effect.

The mechanism is visible in the failures. It is not a parsing artifact — argument
dicts compare without regard to key order, so a reordered call scores the same.
What moves is *which argument the model extracts*: given a declaration in the
order it did not learn, the base returns `email` where the target wanted
`phone_number`. It had learned "the Nth property is X". Fine-tuning teaches it to
read the name instead, and the position stops mattering.

Two consequences for the numbers above:

- **The base column is understated**, because it was measured in declaration
  order. Under the order the weights prefer it is 0.7625.
- **The fine-tuning gain depends on which base you compare against**, and both
  are now measured rather than inferred: **+0.1843** entirely in declaration
  order, **+0.1516** entirely in `dictsort`. The published +0.19 sits above both
  because it pairs a tuned model measured at its best against a base measured at
  its worst.

`contract.json` records which convention a bundle was built under, so a consumer
is not guessing. On the evidence above that matters for a base or lightly-tuned
checkpoint and is close to free for a fully fine-tuned one — but "close to free"
is a measurement on one model and one dataset, not a property of the method.

### What three runs of the same thing disagree about

The same recipe sweep, run three times on the same data with the same code. B
and C are the identical shipping configuration; A differs only in the bundle's
declared model type, which the text-parsing measurement cannot observe:

| | run A | run B | run C |
|---|---|---|---|
| Base, float | 0.7266 | 0.7266 | 0.7266 |
| Fine-tuned, float | 0.9094 | 0.9172 | 0.9062 |
| `dynamic_wi8_afp32` | 0.8906 — **resolved** | 0.9016 — *unresolved* | 0.8969 — *unresolved* |
| `weight_only_wi8_afp32` | 0.9141 — *unresolved* | 0.9047 — **resolved** | 0.9000 — *unresolved* |
| Gap between recipes | 0.0235 | 0.0031 | 0.0031 |

**The recipes swap places, and which cost resolves moves with them.** An earlier
draft of this file read "the two recipes differ by 0.0234 on the same weights at
the same bit width" and drew a conclusion from it. Two further runs put that gap
at 0.0031 and 0.0031. The gap was noise, and a single run had presented it as a
finding.

Across all three, the conversion cost resolves in **2 of 6** recipe-runs -- and
B and C, which are the same configuration end to end, still disagree about
`weight_only`.

So what is actually established is narrower than one run suggests, and worth
separating:

- **The base figure is 0.7266**, identical to four decimal places in all three,
  and reproduced across a rebuilt container image, a `transformers` major version
  change, new batching code and a rewritten parser.
- **Fine-tuning gains about +0.18**, resolved in every run, on 132-136 discordant
  pairs. The spread across runs is 0.0109 — an order below the effect.
- **Conversion costs something small** — 0.0063 to 0.0187 across runs — and
  whether 640 examples *resolve* it is close to a coin flip. At this effect size
  the method is at its limit, and one run's verdict should not be quoted as the
  answer.

That last line is the tool working, not failing. A single accuracy number would
have shown none of this; three points and a paired interval show exactly where
the evidence stops. If you need to separate two recipes this close, you need
more held-out examples than 640 — and `verify` will keep saying "unresolved"
until you have them, rather than picking a winner.

Nothing in the file size, the exit code, or the logs separates those two
artifacts: they are 455,759,152 and 455,939,600 bytes, 0.04% apart. Running both
against held-out data is the only thing that does.

Three points rather than two, because only the differences mean anything. A
single accuracy figure for a converted model cannot distinguish a good
conversion of a bad model from a bad conversion of a good one.

**One `verify` run measures two of them, not three.** It compares the converted
model against one reference, and which reference you name decides which
difference you get:

| `--reference-role` | reference is | you get | you do not get |
|---|---|---|---|
| `float_twin` (default) | the checkpoint the artifact was converted from | conversion cost | training gain |
| `untuned_base` | the model before you trained it | *neither* — the difference confounds them | both |

The table above therefore comes from two runs, not one. `verify` reports the
missing figure as `unavailable` with the reason, rather than deriving it from
the two points it has — deriving it is the mistake the third point exists to
prevent. Composing the three into a single command is the first thing on the
list after this alpha; see *Not wired yet*.

## Where it runs

| | |
|---|---|
| **Linux** | `litert-lm` `dlopen()`s a Vulkan-linked library even for the CPU backend, so the `libvulkan1` system package is required. The export and runtime toolchains are published for Linux only. |
| **Python 3.10–3.13, and 3.10–3.12 for `convert` and `verify`** | Stage environments are built from the interpreter running litetune, so each one's ceiling is set by whichever of its pins stops publishing wheels first: `numpy==2.0.2` (cp312) for `convert` and `verify`, `torch==2.5.1` (cp313) for `tune` and for `prepare --tokenizer`. `bundle` provisions nothing and runs anywhere. Past a ceiling the command refuses with a message naming the pin that set it, rather than letting pip spend minutes failing to build from source. |
| **CPU only, all five stages** | Not a recommendation — a limitation. The generated training script places nothing on a device, `convert` sets `CUDA_VISIBLE_DEVICES=""` so an export cannot depend on an accelerator, and both sides of `verify` run the CPU backend. A GPU that is present will not be used. For a 270M model that is workable; above that it is the first thing to fix. |

**Colab works** — it is Linux and its default interpreter is in range. One cell
up front:

```
!apt-get -qq install -y libvulkan1
!pip install -q litetune
```

Google's own notebooks for this workflow tell you to **restart the runtime**
between steps, because the training and export dependency sets cannot share one
interpreter. litetune provisions one environment per stage and dispatches into
them, so there is nothing to restart. The first stage that needs an environment
builds it, which takes a few minutes; later runs reuse it.

## From a dataset to a bundle

Five commands. Each is its own subcommand because each has its own failure mode,
and a single `run` would hide which one you are in.

```bash
# 1. Split, and refuse the data that cannot be measured. Seconds.
litetune prepare --data raw.jsonl --output-dir data \
                 --context-length 1024 --tokenizer google/functiongemma-270m-it

# 2. Fine-tune. CPU, so size your expectations accordingly.
litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \
              --output-dir tuned --prompt-mode prerendered --method lora

# 3. Convert, sweeping the recipes rather than trusting a default.
litetune convert --model tuned/model --output-dir artifacts \
                 --recipe dynamic_wi8_afp32 --recipe weight_only_wi8_afp32

# 4. Measure what the conversion cost, against the float twin.
litetune verify --model artifacts/weight_only_wi8_afp32/model.litertlm \
                --reference tuned/model --data data/heldout.jsonl --json > manifest.json

# 5. Package the artifact with what was measured about it.
litetune bundle --output-dir bundle --model artifacts/weight_only_wi8_afp32/model.litertlm \
                --declarations tools.json --prompt-mode prerendered \
                --base-model google/functiongemma-270m-it --base-model-revision <sha> \
                --adapter tuned/adapter \
                --train-metrics tuned/metrics.json --verify-manifest manifest.json
```

`--adapter` matters for a LoRA run and is silent without it. `tune` merges the
adapter into the checkpoint, and a merged checkpoint cannot be un-merged: the
rank-16 delta is the only form you can re-apply to a different base, inspect, or
ship on its own, and it exists only at `<tune output>/adapter`. Name a path
outside `--output-dir` — an adapter already inside the bundle directory is
refused rather than copied over itself.

`--base-model-revision` takes a commit sha. `main` and other moving refs are
refused: they resolve to different weights on different days while the bundle
reads identically. A tag is accepted and recorded as a weaker pin than it looks.

`--recipe` has no default on purpose: a sweep of one is not a comparison, and
picking a recipe without measuring the alternative is the thing this tool exists
to stop. `--prompt-mode` has no default either — the two conventions are
mutually exclusive, and a model trained under one cannot be served under the
other.

Step 4 is the one to run first if you already have an artifact: see below.

Two things the flags do not say. `--prompt-mode` must carry the **same** value in
`tune` and `bundle` — the two conventions are mutually exclusive and the wrong
one produces a fluent wrong answer, not an error. And `convert` writes one
`.litertlm` per recipe under `artifacts/<recipe>/` with a name the toolchain
chooses, so step 4 needs you to look it up rather than construct it.

## The cheapest way to try it

Run `verify` alone, on a model you already have. Nothing is retrained and no
data leaves your machine:

```
litetune verify --model ./your-model.litertlm \
                --reference ./the-checkpoint-it-came-from \
                --data held-out.jsonl
```

`--reference` is the **float twin** — the same weights before conversion. That
is what makes the difference between the two the conversion cost, rather than a
mixture of that and whatever training did.

If you point it at a *different* checkpoint — an untuned base, say — pass
`--reference-role untuned_base`. Then both the training gain and the conversion
cost are reported as unavailable, because one number cannot separate two
effects. That is deliberate: the tool would rather report nothing than report a
figure whose meaning depends on which of two things you happened to change.

### What the data file looks like

One JSON object per line:

```json
{"prompt": "set an alarm for 7", "target": {"name": "set_alarm", "args": {"hour": "7"}}}
```

For training rows, add a `completion` field with the text the model should
produce. `litetune prepare` builds both splits from a raw file and reports the
token-length distribution, so a row too long for the sequence limit fails before
a GPU is rented rather than after.

### What you need installed

Python 3.10–3.12 on Linux (3.13 works for everything except `convert` and
`verify`), and the `libvulkan1` system package — `litert-lm`
`dlopen()`s a Vulkan-linked library even for the CPU backend, and without it
every invocation, `--help` included, dies in under a second. The tool detects
this and reports `could not check` rather than blaming the model. **macOS is not
supported for the runtime stages**; the export and evaluation toolchains are
Linux-only.

`litetune verify` provisions **two** environments on first use — the runtime for
the device side and the training stack for the float reference. The second pulls
`torch==2.5.1`, which is a 906 MB wheel plus its CUDA dependencies: several
gigabytes, and minutes. Subsequent runs reuse them. Unlike `tune` and `convert`,
`verify` has no `--no-provision` escape hatch.

## What it does

```
tools.json + data.jsonl + base model
  → train (full or adapter-based)
  → merge
  → export to .litertlm, one artifact per candidate quantization recipe
  → measure: base float, tuned float, tuned on-device
  → bundle: model + declarations + contract + report
```

Each stage is its own CLI subcommand. They are wired by hand; there is no
single `run` command yet, because the config format that would drive one is
implemented but not connected to the CLI (see *Not wired yet* below).

Four properties distinguish this from a shell script calling the same tools.

**Measurement is attributed, not aggregated.** Three points instead of two, so
"the fine-tune gained X" and "the conversion cost Y" are separate numbers. A
single net figure cannot tell you which stage to fix.

**The quantization recipe is an output.** The default recipe carries a warning
in its own toolchain's docstring about quality suffering "due to the on-the-fly
quantization", and it costs measurable accuracy. So `convert` sweeps the
candidates you name instead of accepting a default — and reports only what is
observable at export time, which is size and duration. It says so explicitly:
the comparison carries `accuracy: unavailable`, because ranking recipes needs
held-out data and that is `verify`'s job. There is no accuracy-versus-size
frontier at export time, and a tool that printed one would be inventing it.

**Checks have three outcomes, not two.** `passed`, `failed`, and **`could not
check`**. A check that did not run must never report a verdict. This sounds
pedantic until it happens: during the work that produced this tool, ten separate
checks reported a confident result when they had in fact not executed. Among
them: a binary missing from the platform, an interactive prompt with no terminal
attached, a download refused for a licence that had not been accepted, a
toolchain that changed overnight, a process killed by the out-of-memory killer
and read as a rejection, and a report from a previous run read as the current
one. Each was indistinguishable from a real answer.

**Every number carries its sample size and interval.** At 64 held-out examples,
the recipe comparison above appeared to be worth 0.172. At 640, it is 0.024.
Three conclusions drawn at the smaller size were overturned at the larger
one — each plausible, each with a mechanism, each wrong. A gate whose threshold
is finer than its interval returns `inconclusive`
rather than a verdict, and a difference whose interval straddles zero is
reported as unresolved rather than as zero.

## Why a liveness check is not enough

The tool runs a cheap label-free tier first: did the process exit cleanly, is
the output non-empty and decodable, is it free of padding-token leakage and
degenerate repetition, does it differ from the base model's output.

That tier is necessary and it is not sufficient. A LoRA run that scored 0.0625
against a 0.5625 base — nine times worse than doing nothing — passed every one
of those checks, including divergence from base. It was alive, fluent, and
wrong. Only held-out measurement caught it.

So: liveness is never reported as verification, and a run with no labelled data
is reported as *unmeasured*, not as *verified*.

## Design notes

**Runs locally, hosts later.** No cloud account is required and no cloud service
is assumed.

**Stages run in separate environments.** The training and export dependency sets
are mutually incompatible — `torch`/`transformers` against
`litert-torch`/`numpy<2.1`. Google's own notebooks handle this by instructing
the reader to restart the runtime between steps. Here the CLI provisions one
environment per stage and dispatches into them, so you install one package and
never see the conflict.

**Toolchain versions are pinned, deliberately.** An unpinned nightly produced a
working export on 2026-08-26 and `AttributeError: pad_token` on 2026-08-30 from
an unchanged install command. Pinning the environment's *identity* is not enough
when its *definition* resolves differently over time.

**The model is trained to emit the terminator its runtime waits for.** Derived
from the model's own chat template rather than assumed, recorded in the training
report, and carried into the bundle's contract. A model trained on one
terminator while the runtime waits for another does not stop — which shows up as
extra tool calls firing on the device, not as a lower score.

## Known limitations

**One bundle, two prompt renderings.** The plugin renders declaration properties
in declaration order; the jinja template inside the same `.litertlm` renders them
with `dictsort`, and they disagree for 100% of `mobile-actions` rows. On the base
checkpoint the wrong order costs a resolved 0.019–0.036; after fine-tuning the
difference is unresolved and near zero, so this is a hazard for base and
lightly-tuned models rather than a defect in a shipped one. `contract.json`
records the convention, but litetune does not render declarations itself, cannot
choose the order for you, and `verify` does not yet refuse to compare two points
rendered differently.

**The SentencePiece tokenizer is restored by hand, and that is load-bearing.**
`transformers` 5.x `save_pretrained` no longer writes `tokenizer.model`, and the
tokenizer classes no longer expose `vocab_file`. The exporter's SentencePiece
branch tests for exactly those, so without intervention every bundle silently
gets an HF tokenizer section instead of `SP_Tokenizer` — and LiteRT-LM's
FST-constrained decoding is SentencePiece-only. The artifact still runs, still
scores the same, and still passes every liveness check; it just cannot do
constrained tool-calling. `tune` copies the file back and records in
`metrics.json` whether it managed to. Check that field: a model whose tokenizer
is BPE (Qwen) has no such file at all, which is fine and reported as such.

**Peak memory is not bounded and 32 GiB was not enough.** Training a 270M model
on 8,693 examples was killed by the OOM killer at 32 GiB more than once before
the batching was rewritten to a token budget. There is no preflight check and no
documented figure — if the process dies without a Python traceback, that is what
happened.

**Measurement runs on CPU; your users run on a phone.** Published reports show
the GPU backend scoring materially worse than CPU on identical artifacts. Every
number here is therefore an optimistic estimate of on-device behaviour, and the
report says which backend and engine version produced it. Closing that gap needs
real hardware and is not done.

**Decoding parameters reach only one side.** The pinned `litert-lm` CLI takes no
decoding flags, so the reference side is held to an explicit token limit while
the device side runs to the runtime's own. The manifest names this and counts
how many device generations end without a terminator — the measure of how far
the difference could reach.

**The speed cost of `weight_only` is unmeasured.** It dequantizes weights before
compute, so it is slower — by how much is open, and job wall-time is not the
instrument (roughly 90% of it is process startup).

**Evaluation is slower than it needs to be.** One subprocess per prompt against
the runtime CLI. A persistent `litert-lm serve` client is worth roughly a
thirtyfold reduction and is not implemented.

**An untyped bundle loses its tool-call channel, and the exporter produces one
by default.** LiteRT-LM picks a data processor from the bundle's declared model
type; the generic one has no code fences, so the runtime never creates the
tool-call channel and `CreateConstraint` returns `Unimplemented`, which the
caller swallows — constrained decoding turns itself off with no diagnostic. A
consumer that parses the response text itself, as `flutter_gemma` does, is
unaffected; one that passes tools natively receives no calls at all.

The exporter chooses that type by matching `config.json`'s `model_type` against
a fixed list, with a silent catch-all. FunctionGemma declares `gemma3_text`, and
so does plain Gemma 3 — neither string is on the list, so both exported as
`generic_model` until litetune learned to pass
`--litert_lm_model_type_override`. It now does, per family, because the two
share one `model_type` and need different values, so the config cannot
distinguish them. Verified against Google's own published artifact for the same
model: same declared type, byte-identical metadata.

**The exception is Gemma 4**, where litetune *refuses* that flag: passing it
leaves `generic_model` set anyway and additionally skips the Gemma 4 metadata
builder, so it is worse than not passing it. Gemma 4 bundles are therefore still
untyped, and this limitation stands for them.

**Export targets `.litertlm` only.**

**Not wired yet.** A declarative job-spec format (`spec.py`) and a stage runner
with content-addressed caching (`runner.py`) are implemented and tested but not
reachable from the CLI. They are what a composed three-point command would be
built on. Connecting them is a design decision rather than wiring — it fixes the
shape of the config format users would then depend on — and it is deferred until
the CLI surface has been used by someone other than its author.

**No library API yet.** Every stage is `run_x(Request, events=...) -> Result`
and the results are dataclasses, so the shape is there; but `verify` returns its
manifest as a plain `dict` with no reader and no schema check, and the package
exports only its vocabulary (`Check`, `CheckSet`, `Outcome`, `Proportion`,
`Difference`, `Unavailable`). Treat
anything below `litetune.<module>` as private for now.

## Contributing

Alpha software with one measured model. The most useful thing you can send is a
run that went wrong: what you ran, what it said, and what you expected. Issues
and pull requests welcome.

## Exit codes

The verdict is the exit code; the printed summary is a rendering of it. A run
that completed and a run that could not be judged must never be confused, which
is why there are five and not two.

| | `verify` | `prepare` / `tune` / `convert` | `bundle` |
|---|---|---|---|
| **0** | passed | passed | passed |
| **1** | failed a gate, or failed a label-free check | failed | failed |
| **2** | inconclusive: the interval does not resolve the threshold | — | **the default** |
| **3** | nothing was established: no labelled data, or a difference that cannot be attributed | — | carried in |
| **4** | could not check — a harness fault, a malformed command line, or a request litetune refused to run | same | same |

`bundle` carries a verdict rather than producing one, so it returns whatever
`--status` or `--verify-manifest` gave it. With neither, it returns **2**:
bundling re-measures nothing, and a command that decided its own verdict would
be asserting something it never checked.

**4 is not a failure of the model.** It means the tool could not obtain an
answer: a missing shared library, a malformed input file, a process killed
before it finished — or a command line it would not run. That last one is why
a usage error exits 4 rather than argparse's usual 2: here 2 means
*inconclusive*, which is a statement about a measurement, and a typo should not
produce one. Wiring `|| exit 1` on anything non-zero conflates all of it.

## License

Apache-2.0. See [LICENSE](LICENSE).
