# What was measured, and what it established

Numbers for `litetune`. Most of what follows is `functiongemma-270m-it` LoRA-tuned
on `google/mobile-actions`, scored on 640 held-out single-call examples; exact
match means the tool name **and** every argument value. The last section is a
second family and the second scorer.

This file exists so the README can be a usage guide. It is the longer story:
what reproduced, what did not, and which published claims were withdrawn.

## The headline numbers

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

Two consequences for the headline table:

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

**On a phone.** The run-A `dynamic_wi8_afp32` bundle, repacked with
`prefer_activation_type = fp32` (see README, Limitations), on one Snapdragon
Galaxy S24 (`SC-51E`, Android 16), native tool path, greedy, 640 rows:

| | device CPU | reference (run A) |
|---|---|---|
| exact match | **0.8703 ±0.0260** | 0.8906 |
| tool name | 0.9812 | — |
| produced a call | 634/640 | — |
| per prompt, median | 3.3 s (2.3 s cold, 3.2 s warm: the phone throttles) | — |

Difference −0.020 against run A, inside the device interval; −0.031 and −0.027
against runs B and C, just outside it. The same bundle on the device's GPU,
20 rows: 20/20 tool names and 15/20 exact with the key, 3/20 and 2/20 without
it (`<pad>` floods; the engine reports success either way). GPU per prompt
1.4 s with the key against 2.5 s on the device CPU for the same file.

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

The headline table therefore comes from two runs, not one. `verify` reports the
missing figure as `unavailable` with the reason, rather than deriving it from
the two points it has — deriving it is the mistake the third point exists to
prevent. Composing the three into a single command is the first thing on the
list after this alpha; the README's *Limitations* records what is not wired.

## A second family, and the second scorer

`google/gemma-3-270m-it` @ `ac82b4e8`, LoRA r16/α32, lr 2e-4, one epoch over
2,400 rows of `mteb/banking77` — 77-way intent classification, target is
`label_text` — scored on 600 held-out rows with `--scorer exact-text`, prompt
mode `prerendered`, litert-lm 0.16.1 CPU backend on an M-series Mac.

| | float twin | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| exact match | 0.6933 ±0.0369 | 0.6767 ±0.0374 | 0.6917 ±0.0370 |
| Cost of conversion | — | +0.0167 ±0.0201 *(unresolved, 38 discordant)* | +0.0017 ±0.0127 *(unresolved, 15 discordant)* |

**Both costs are unresolved, and that is the finding.** At n=600 the intervals
straddle zero, so this run establishes that neither quantisation moved accuracy
by more than about two points — not that either is free. The functiongemma table
above resolves a cost of +0.0125 at n=640 because its discordant count is 136;
here it is 38 and 15. Discordant pairs, not sample size, are what buys
resolution, and a task the two artifacts almost never disagree on cannot settle
a small difference no matter how many rows it has.

**What it established beyond the numbers.** The family rule resolved
`model_type: gemma3_text` to Gemma 3 rather than FunctionGemma on a real export
— the two share that string, and until this run the disambiguation had only ever
run in a unit test.

**What it cost to get.** The first run scored the float reference **0.0000**.
The transformers reference decodes with `skip_special_tokens=False`, deliberately,
so the liveness tier can see special-token leakage; every generation it returned
ended in `<eos>`; and `exact-text` forgave whitespace and nothing else. `verify`
reported a *resolved* conversion cost of −0.6767 across 406 discordant pairs — a
confident number about a difference between two decoders. The `tool-call` scorer
had hidden this on every earlier run, because its parser stops at the closing
brace and never sees a trailing marker. The figures above are from the fixed
scorer; the defect is why `harness.terminators` is now recorded on every
manifest.

**What the terminator actually is, measured rather than assumed.** Two different
markers are in play here, and an earlier draft of this branch confused them.

The **base** checkpoint stops on its chat template's own close: run on 20 of
these held-out prompts at `max_new_tokens=64`, all 20 closed their turn and all
20 ended in `<end_of_turn>` alone, for an unterminated share of 0.000.
`generation_config.eos_token_id` is `[1, 106]` — `<eos>` and `<end_of_turn>` —
so generation halts at the close and never reaches the tokenizer's eos.
`functiongemma-270m-it` has the same shape, with `<start_function_response>`
added to the set.

The **tuned** model, and therefore the float twin this table compares against,
ends on `<eos>` instead. These prompts are `prerendered`, so `tune` does not
probe the chat template at all and takes the tokenizer's eos: `tune.metrics.json`
records `turn_terminator: {ids: [1], source: "tokenizer_eos", text: "<eos>"}`,
training appended exactly that to every completion, and `contract.json` carries
`stop_tokens: ["<eos>"]`. Both markers are in the scorer's vocabulary, which is
why neither run was mis-scored — but they are not the same marker, and which one
a given artifact emits depends on how it was trained, not on its family.

An earlier draft asserted instead that Gemma emits `<end_of_turn>\n<eos>` — two
markers, newline between — in its docstrings, its tests and its commit message.
It does not, for either supported family. The stacked shape belongs to a family whose
eos set excludes its own template close, which is why the trimmer strips
whitespace between removals; that case is constructed in the tests and labelled
as such.

**What it did not establish.** No untuned-base run, so training gain is
unattributed and the prepare stage's headroom slices are empty. CPU only — no
GPU or NPU figure for this family. Trained in float32 rather than the bfloat16
default because bfloat16 on this CPU runs on a single core.
