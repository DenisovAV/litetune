# What was measured, and what it established

Numbers for `litetune`. Everything up to the last section is
`functiongemma-270m-it` LoRA-tuned on `google/mobile-actions`, scored on 640
held-out single-call examples; exact match means the tool name **and** every
argument value. The last section is the first run of a second family and the
second scorer.

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

Everything above is one model and one scorer. This section is the first run of
anything else: `google/gemma-3-270m-it` (commit `ac82b4e8`), LoRA r16 on 2,400
rows of `mteb/banking77` — a 77-way intent task where the target is the label
string — scored with `--scorer exact-text` on 600 held-out rows, on CPU.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | *refused* | — | — |
| Fine-tuned | **0.6933** ±0.0369 | 0.6767 ±0.0374 | 0.6917 ±0.0370 |
| Cost of conversion | — | +0.0167 ±0.0201 *(unresolved, 38 discordant)* | +0.0017 ±0.0127 *(unresolved, 15 discordant)* |

Both costs are inside their intervals at n=600. That is the same shape as the
FunctionGemma table: the conversion cost something, it was small, and this
sample cannot tell it from zero. `weight_only` is again the closer of the two,
by half the estimate and half the discordant count.

### What the run established that the table does not show

**The family rule was tested on the case it exists for.** `gemma-3-270m-it` and
`functiongemma-270m-it` both declare `model_type: gemma3_text`; the export
needs different overrides for each. `convert` resolved this checkpoint to
`gemma-3-text` and added `--litert_lm_model_type_override=gemma3` — the first
time that disambiguation ran on a real export rather than in a unit test.

**The exact-text scorer was wrong before this run, and the run is what found
it.** The transformers reference backend decodes with `skip_special_tokens=False`
so the liveness tier can see leakage, and every generation it returns ends in
`<eos>`. Under `exact-text` — whitespace forgiven, nothing else — that scored
the fine-tuned float reference at **0.0000** on all 600 rows while the runtime
side scored 0.68, and `verify` reported a "conversion cost" of −0.68. It did
flag the reference-at-zero as not being evidence; the number was still wrong,
and `tool-call` had hidden the asymmetry on every earlier run because its
parser ignores trailing markers. Fixed in 0.1.5, though not the way the first
attempt tried: trimming both sides scored a model that dropped a closing tag on
every row at 1.0000, so the rule became "the generation must still contain
every terminator the target itself ends with, and anything beyond that is the
decoder's convention". The table above is from the fixed scorer, and the
numbers are unchanged by the rule that replaced it -- these targets are bare
labels carrying no marker, the case in which the two rules are the same
function.

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

**bfloat16 on this CPU trains on one core.** The default dtype, chosen to match
export and evaluation, ran a 300-step LoRA on a single core for 52 minutes
without finishing; the same run with `--dtype float32` took 307 s on ten
threads (peak 6.35 GB). The table is from the float32 checkpoint, and
`tune` records the dtype mismatch as a limitation. From 0.1.5 a CPU run is
told this at the time.

### Limitations carried by these manifests

- Measured on litert-lm's CPU backend; the GPU backend is reported to score
  below CPU on identical artifacts, so this is an optimistic estimate of
  on-device behaviour.
- Decoding parameters were passed to transformers but not to litert-lm, which
  used the pinned runtime's defaults; both are greedy.
- The untuned base has no score, so the training gain is not attributed and
  `prepare` could not identify slices where the base already scores at ceiling.
  Not for want of running it: see below.
- Training ran in float32 while export and evaluation use bfloat16, for the
  reason above.

### The base model could not be scored at all

The untuned `gemma-3-270m-it` was converted and run over the same 600 prompts.
`verify` refused to score it: 571 of 600 generations repeat themselves above the
0.50 threshold, the worst at 0.9995. Asked to answer with one intent label, the
base does not emit a label and stop — it runs on until the token limit, saying
the same thing over and over. The run ends at `failed_smoke` and exit 1, before
the quality tier.

That refusal is the point. An exact-match score against those generations would
have been a number — near zero — and it would have read as "the base is bad at
this task". What actually happened is that the base never answered the question
in the shape the task requires, which is a different claim, and the one that
explains why fine-tuning moved so much. A liveness tier that scored it anyway
would have turned "this model does not do the task" into "this model does the
task badly".

It also means the training gain here is unattributable in principle, not just
unmeasured: there is no base figure to subtract, and manufacturing one from a
degenerate run would be the mistake `attribution` exists to refuse.
