# What was measured, and what it established

Numbers for `litetune`. The first sections are `functiongemma-270m-it`
LoRA-tuned on `google/mobile-actions`, scored on 640 held-out single-call
examples; exact match means the tool name **and** every argument value. Each
section after them is another family: `gemma-3-270m-it` with the second scorer,
then `gemma-4-E2B-it` converted from its base weights.

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

### The same model on an NPU

`functiongemma-270m-it` with its *base* weights (the fine-tuned checkpoint has
not been through this yet), through litert-torch's `npu_export` stages — the
`dynamic_wi8_afp32` export, calibration on 64 tool prompts disjoint from the
20 scored, static int8, Qualcomm compile with `ai-edge-litert-sdk-qualcomm`
2.2.0 (QAIRT 2.47), litert-torch-nightly 0.10.0.dev20260826 — for `SM8750`,
run on a Galaxy S25 (Android 16) through the LiteRT-LM C API of the
`native-v0.16.0` runtime tarball (a LiteRT-LM v0.16.1 tree) on the NPU,
greedy, 20 rows per set, 2026-09-08. Each median is the upper of the two
middle values at n=20.

| prompt set | tokens | prefill chunks | `cache_length` 1024 (earlier builds, similar sets) | `cache_length` 896 |
|---|---|---|---|---|
| plain question | 17–73 | 1 | coherent | 20/20 coherent, median 389 ms |
| filler, question last | 208–250 | 2 | garbage | 20/20 address it, median 465 ms |
| tools + request | 579–635 | 5 | 0/20 parsed | parsed 13/20, tool name 12/20, exact 3/20, median 441 ms |

Same 20 tool prompts, CPU interpreter: the `dynamic_wi8_afp32` stage-1 graph
20/20 parsed, 19/20 name, 12/20 exact; the static-int8 stage-3 graph 20/20,
18/20, 12/20. So at 20 rows quantization shows no cost beyond one tool-name
row, and the nine exact rows the NPU loses are its own decode: one to syntax
the runtime's parser rejects (a stray `}`), one to a prose reply instead of a
call, one to the wrong tool named first of two calls, six to a wrong or
missing argument in an otherwise well-formed call. On
SM8750 the 1024→896 difference tracks the compiled prefill graph's attention
mask crossing ~1.0 MiB (litert-torch#1184; the rule is stated under Known to
be broken in the README); LiteRT-LM#3508 has the full table across two SoCs,
Google's bundles included, and one published bundle that works above the
line. One SoC, base weights, 20 rows: a status, not a figure.

## A second family, and the second scorer

Everything above is one model and one scorer. This section is the first run of
anything else: `google/gemma-3-270m-it` @ `ac82b4e8`, LoRA r16/α32, lr 2e-4,
one epoch over 2,400 rows of `mteb/banking77` — 77-way intent classification,
target is `label_text` — scored on 600 held-out rows (split
`0c7505b2b6f69ab1`) with `--scorer exact-text` and a 256-token limit, prompt
mode `runtime_rendered`, litert-lm 0.16.1's CPU backend against a float twin on
one A100-SXM4-40GB. Training ran in bfloat16 on that GPU. Everything not named
here was left at its default.

**The prompt mode is what changed.** An earlier version of this section measured
the same checkpoint `prerendered` — the runtime told not to apply the model's
chat template, to prompts that are bare user text — and reported 0.6933, 0.6767
and 0.6917. Those numbers are withdrawn, not adjusted: they describe a model
answering a prompt no caller sends. Here `tune` was given no `--prompt-mode`,
inferred `runtime_rendered` because 0% of the 2,400 training prompts carry a
control token, and recorded that beside the checkpoint; `verify` read the mode
back from that record rather than inferring it again. Before generating
anything it rendered all 600 held-out prompts through the runtime's own
conversation path and through the reference's chat template: identical token
ids on all 600, and the runtime's prefill count equal to the reference's on the
eight it also sent.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | *refused* | — | — |
| Fine-tuned | **0.6717** ±0.0376 | 0.6717 ±0.0376 | 0.6683 ±0.0377 |
| Cost of conversion | — | +0.0000 ±0.0190 *(unresolved, 34 discordant)* | +0.0033 ±0.0131 *(unresolved, 16 discordant)* |

**Both costs are unresolved, and that is the finding.** At n=600 each interval
contains zero, so this run establishes only that neither quantisation moved
accuracy far — the dynamic estimate's own interval reaches +0.0190, so "far"
here is up to about two points, not that either recipe is free. That recipe
also shows why a score is not an answer: it lands on the same 0.6717 as the
float twin and still disagrees with it on 34 of 600 prompts, in both
directions. What resolves a paired difference is neither the row count nor the
number of disagreements but the imbalance between the two directions of
disagreement, set against the square root of how many there are: more
disagreement widens the interval as well as feeding the estimate. The
FunctionGemma conversion cost above clears its own interval at n=640 on 16
discordant pairs, and by a hair — +0.0125 against ±0.0123. Here there are 34
and 16, and neither imbalance is large enough for the interval it has to buy.

### What the run established that the table does not show

**The family rule was tested on the case it exists for.** `gemma-3-270m-it` and
`functiongemma-270m-it` both declare `model_type: gemma3_text`; the export
needs different overrides for each. `convert` resolved this checkpoint to
`gemma-3-text` and added `--litert_lm_model_type_override=gemma3` — the first
time that disambiguation ran on a real export rather than in a unit test.

**The exact-text scorer was wrong before this pair was first measured, and this
pair is what found it.** The transformers reference backend decodes with
`skip_special_tokens=False` so the liveness tier can see leakage, and every
generation it returns ends in a terminator. Under `exact-text` — whitespace
forgiven, nothing else — that scored the fine-tuned float reference at
**0.0000** on all 600 rows while the runtime side scored 0.6767, and `verify`
reported a *resolved* "conversion cost" of −0.6767 across 406 discordant
pairs — the imbalance the section above describes, at its limit. It did flag
the reference-at-zero as not being evidence; the number was still wrong, and
`tool-call` had hidden the asymmetry on every earlier run because its parser
ignores trailing markers. Fixed in 0.1.5, though not the way the first attempt
tried: trimming both sides scored a model that dropped a closing tag on every
row at 1.0000, so the rule became "the generation must still contain every
terminator the target ends with, counting repeats and in that order, and any
terminator beyond that is ignored". That 0.6767 belongs to the withdrawn
`prerendered` run; the table above is from the re-measurement under the fixed
scorer, and the rule that replaced it changes neither — these targets are bare
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

The **tuned** model ends on whichever marker the mode it trained under put
there, and that is the second thing the re-measurement changed. In
`runtime_rendered`, `tune` renders the chat template around every prompt and
takes the terminator from that template: `metrics.json` records
`turn_terminator: {ids: [106, 107], source: "chat_template", text:
"<end_of_turn>\n"}`, training appended exactly that to every completion, and
`contract.json` carries it into `stop_tokens`. The earlier `prerendered` run of
the same checkpoint took the tokenizer's eos instead — `{ids: [1], source:
"tokenizer_eos", text: "<eos>"}` — because a prerendered prompt is not
templated, so there is no template close to read. Both markers are in the
scorer's vocabulary, recorded verbatim at `harness.terminators` in every verify
manifest, which is why neither run was mis-scored — but they are not the same
marker, and which one a given artifact emits depends on how it was trained, not
on its family.

An earlier draft asserted instead that Gemma emits `<end_of_turn>\n<eos>` — two
markers, newline between — in its docstrings, its tests and its commit message.
It does not, for either supported family. The stacked shape belongs to a family whose
eos set excludes its own template close, which is why the trimmer strips
whitespace between removals; that case is constructed in the tests and labelled
as such.

**What it did not establish.** No untuned-base score, and not for want of
running it: the base was converted and measured in this same mode, and refused
at the liveness tier — asked for one intent label it returned nothing at all on
53 of 600 prompts, so `verify` stopped before scoring either side. Training gain
is therefore unattributed, and `prepare` could not identify slices where the
base already scores at ceiling. The candidate ran on CPU only — no GPU or NPU
figure for this family. Training ran in bfloat16 on the A100; the single-core
bfloat16 behaviour that made an earlier run of this pair train in float32 is a
property of that Mac's CPU and says nothing about this one.

## A third family, and the marker the vocabulary did not know

The two sections above measure models this project fine-tuned. This one
measures neither: it compares two *conversions of the same base weights* --
`google/gemma-4-E2B-it`, untouched -- because `models.py` records that a Google
engineer recommends two recipes "to remain the model quality" and, in the same
sentence, that litetune had measured neither.

600 held-out rows of `mteb/banking77` with all 77 labels listed in the prompt,
`--scorer exact-text`, 256-token limit, prompt mode `runtime_rendered`,
reference role `float_twin`. The labels are in the prompt because these are base
weights that never saw this task and cannot guess a snake_case label out of 77;
without them the reference sits on the floor and nothing can be attributed. The
reference is `google/gemma-4-E2B-it` at float32 on one A100-SXM4-40GB, greedy;
both candidates run on litert-lm 0.16.1's CPU backend in the same container, 12
vCPU. Export environment `transformers==5.17.0`, `numpy==2.0.2`.

| | exact match | cost of conversion | discordant |
|---|---|---|---|
| float reference | **0.5983** | — | — |
| Google's published `gemma-4-E2B-it.litertlm` | 0.5717 | +0.0267 ±0.0277 *(unresolved)* | 72 of 600 |
| `dynamic_wi4b32_afp32`, exported here | 0.5483 | **+0.0500** ±0.0244 *(resolved)* | 56 of 600 |

**What it establishes.** An export made here cost something measurable on this
task. Google's published artifact's cost did not resolve at this sample size.

**What it does not.** That either is worse than the other. Each was measured
against the float reference and not against the other, and the intervals
overlap: [-0.001, 0.054] against [0.026, 0.074]. Nor are the two the same kind
of object -- Google's comes from a quantized-safetensors path litert-torch does
not support, so `float_twin` is exact for the export here and looser for
theirs. One run.

**The resolved difference rests on fewer disagreements than the unresolved
one** -- 56 against 72 -- which is the arithmetic the second family's section
describes, seen from the other side.

### What this run established that the table does not show

**One of the two recommended recipes does not build at all.**
`dynamic_wi4c_hr_afp32`, the first one named, fails inside the quantizer:
`hadamard_rotation.py` calls `ndarray.reshape(..., copy=False)`, a keyword that
arrives in NumPy 2.1, while the export environment pins `numpy==2.0.2` because
the toolchain requires it. `convert` exits 1 and reports `export: failed`. The
recipe is unreachable on this pin -- not measured and found wanting.

**The static terminator vocabulary refused this family, exactly as README said
it would.** Gemma 4 closes a turn with `<turn|>`; the vocabulary knew
`<end_of_turn>`. All 600 reference generations ended in a marker scoring did not
recognise, so `verify` stopped before scoring either side. That is the safe
direction and it is also useless until the vocabulary learns the marker. It has
since: `terminators_trimmed` then read 600 of 600, one marker each. The table
above is from the re-run.

### Limitations carried by these manifests

Two, paraphrased — the second drops a measured clause and both carry a note:

- Measured on litert-lm's CPU backend. The manifests from this run carry the
  older wording, which said published reports put the GPU backend materially
  below CPU on identical artifacts, so the number was an optimistic estimate of
  on-device behaviour. That was true of bundles without the GPU activation key
  and not of bundles with it: on the one
  device measured, a repacked bundle at `prefer_activation_type = fp32` scored
  15/20 against the CPU's 14/20 at 1.8× the speed, which at n=20 is not a
  resolved difference — see the table in `export.py`. `convert` attempts that
  repack on every bundle, and names the ones where it could not, or where a
  different value was already declared; one already declaring `fp32` needs no
  warning and gets none. The limitation was corrected after this run, and cut
  back: a manifest produced today says only that litert-lm's GPU backend is a
  different executor and the CPU number does not predict it, and points at
  README's limitations section. Figures belong in the documents, where the
  conditions that make them readable sit beside them, and not in a per-run
  limitation that would carry another run's numbers into every manifest.
- Decoding parameters were passed to transformers but not to litert-lm, which
  used the pinned runtime's defaults; both are greedy, and the token limit is
  unverified on the runtime side. In the withdrawn `prerendered` run that limit
  is what the base model ran into; in the re-measurement it returned nothing at
  all instead — see the last section.

And one fact about the re-measured run, which fired no limitation:

- Training ran in bfloat16, the default, on the same A100 that held the float
  reference. Export passes no dtype and the float reference loads at float32,
  so that is not a mismatch with either.

### The base model could not be scored at all

The untuned `gemma-3-270m-it` was converted and run over the same 600 prompts in
both prompt modes, and `verify` refused to score it both times — for different
reasons, which is itself worth recording. `prerendered`: 571 of 600 generations
repeat themselves above the 0.50 threshold, the worst at 0.9995, running on
until the token limit. `runtime_rendered`, the re-measurement: 53 of 600
generations are empty after decoding, a share of 0.0883 against a threshold of
0.0000. Either way the run ends at `failed_smoke`, before the quality tier:
asked to answer with one intent label, the base does not emit a label and
stop.

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
