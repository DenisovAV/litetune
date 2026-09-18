# What was measured, and what it established

Numbers for `litetune`. The first sections are `functiongemma-270m-it`
LoRA-tuned on `google/mobile-actions`, scored on 640 held-out single-call
examples; exact match means the tool name **and** every argument value. Each
section after them is another family: `gemma-3-270m-it` with the second scorer,
`gemma-4-E2B-it` converted from its base weights, and `Qwen3-0.6B`, the first
that is not a Gemma. The last returns to `functiongemma-270m-it`, measured
through the runtime's tool path the way an application calls it.

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

**Which path this applies to.** Everything in this section is about prompts an
application renders itself — `prerendered`, the path these runs took. Through
the runtime's tool path the runtime renders the declarations from JSON in the
order it is handed, and litetune hands it the file sorted the way the template's
`dictsort` sorts — the `dictsort` column below. There the rendering check found
the training prompt and the runtime's identical by token ids on all 640 prompts
of the run [below](#the-same-model-through-the-runtimes-tool-path).

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
53 of 600 prompts, so `verify` stopped before scoring either side, as *The base
model could not be scored at all* below describes. Training gain is therefore
unattributed, and `prepare` could not identify slices where the base already
scores at ceiling. The candidate ran on CPU only — no GPU or NPU figure for this
family. Training ran in bfloat16 on the A100; the single-core bfloat16 behaviour
that made an earlier run of this pair train in float32 is a property of that
Mac's CPU and says nothing about this one.

### Limitations carried by these manifests

Four, paraphrased:

- Measured on litert-lm's CPU backend. The wording these manifests carry is the
  corrected one: litert-lm's GPU backend is a different executor and this number
  does not predict it, pointing at README's limitations section rather than
  quoting figures. The earlier wording, which this run predates, said published
  reports put the GPU backend materially below CPU on identical artifacts. That
  was true of bundles without the GPU activation key and not of bundles with it:
  on the one device measured, a repacked bundle at `prefer_activation_type =
  fp32` scored 15/20 against the CPU's 14/20 at 1.8× the speed, which at n=20 is
  not a resolved difference — see the table in `export.py`. `convert` attempts
  that repack on every bundle, and names the ones where it could not, or where a
  different value was already declared; one already declaring `fp32` needs no
  warning and gets none. Figures belong in the documents, where the conditions
  that make them readable sit beside them, and not in a per-run limitation that
  would carry another run's numbers into every manifest.
- The candidate ran on CPU and the reference on CUDA, so the difference carries
  a hardware difference as well as a conversion one. Training ran in bfloat16 on
  that same A100; export passes no dtype and the float reference loads at
  float32, so the dtype mismatches neither side — the device does, and this is
  the limitation that records it.
- Decoding parameters were passed to transformers but not to litert-lm, which
  used the pinned runtime's defaults; both are greedy, and the token limit is
  unverified on the runtime side. In the withdrawn `prerendered` run that limit
  is what the base model ran into; in the re-measurement it returned nothing at
  all instead — see the subsection after this one.
- The sample does not resolve the difference: +0.0000 lies inside ±0.0190 at
  n=600, on 34 of 600 discordant examples.

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

## A fourth family, and the first that is not Gemma

Every section above measures a Gemma: FunctionGemma and Gemma 3 fine-tuned
here, Gemma 4 converted from its base weights. This one is `Qwen/Qwen3-0.6B`
@ `c1899de2`, the first model measured here that litetune had no per-model
rule for — `convert` said "no per-model rules for this checkpoint" on the
tuned checkpoint and on the base.

The task, the rows and the scorer are the second family's: LoRA r16/α32, lr
2e-4, one epoch, batch 8, over the same 2,400 rows of `mteb/banking77`, scored
on the same 600 held-out rows — both runs' verify manifests record split
`0c7505b2b6f69ab1`, and their prepare reports the same `heldout.content_sha256`
— with `--scorer exact-text`, a 256-token limit and prompt mode
`runtime_rendered`. `tune` was given no `--prompt-mode` and inferred it: 0% of
the 2,400 training prompts carry a control token, so they are bare text and
training renders the model's own chat template around them the way the runtime
will. `verify` read that record back from beside the checkpoint, and before
generating anything it found the runtime and the reference rendering identical
token ids on all 600 prompts. What differs from the second family is recorded:
`max_seq_length` 256 rather than 160. Training and the float reference ran on
one A100-SXM4-40GB in bfloat16, and both candidates on litert-lm 0.16.1's CPU
backend in the same container, 12 vCPU. litetune 0.1.6; training environment `torch==2.5.1`,
`transformers==5.16.1`, `peft==0.20.0`; export environment
`litert-torch-nightly==0.10.0.dev20260826`, `litert-lm==0.16.1`,
`litert-lm-builder==0.16.1`, `numpy==2.0.2`.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | *not scored* | — | — |
| Fine-tuned | **0.6983** ±0.0367 | 0.6917 ±0.0370 | 0.6817 ±0.0373 |
| Cost of conversion | — | +0.0067 ±0.0146 *(unresolved, 20 discordant)* | **+0.0167** ±0.0113 *(resolved, 12 discordant)* |

**One cost is resolved and one is not**, where both of the second family's were
unresolved — and the resolved one rests on fewer disagreements, not more.
`weight_only_wi8_afp32` costs +0.0167 with an interval of ±0.0113 on 12
disagreements out of 600, eleven of which fall the same way; `dynamic_wi8_afp32`
costs +0.0067 inside ±0.0146 on 20, split twelve against eight. That is the
arithmetic the headline section describes, seen once more: what buys an interval
is the imbalance between the two directions, not the count.

Across the two families the manifests support one observation and no ranking.
The fine-tuned Qwen3 scores higher on these rows than the fine-tuned Gemma 3
did, 0.6983 against 0.6717, and its conversions disagree with their float model
on 20 and 12 rows against Gemma 3's 34 and 16. litetune computes no interval for
a difference between two models, so that is not a resolved difference, and the
conversion costs are not ranked across families: both runs' references ran on a
different device from their candidates, so both carry that difference.

### What this run established that the table does not show

**The family needs no rule, and now has one that says so.** Both recipes
exported with no flag from litetune, and the tuned checkpoint's `config.json`
names `model_type: qwen3`. That is
the model-type trap `models.py` describes, seen from the other side: `qwen3` is
on the exporter's own list, so the config's `model_type` selects the right type
with no override. `models.py` records it as `qwen-3`, scoped to 0.6B, the one
size run.

**The terminator is the template's close, and it is also the tokenizer's
eos.** `generation_config.eos_token_id` is `[151645, 151643]` — `<|im_end|>`
and `<|endoftext|>` — and `<|im_end|>` closes a turn in the chat template.
Training in `runtime_rendered` takes the terminator from that template, so
`tune` recorded `turn_terminator: {ids: [151645, 198], source: "chat_template",
text: "<|im_end|>\n"}` — the marker and the newline the template writes after
it — and `contract.json` carries that into `stop_tokens`. On the reference,
`terminators_trimmed` read 600 of 600, one marker each. The second family's
section describes Gemma 3, where the tokenizer's eos and the template's close
are two different markers; here they are the same one.

**The runtime side returns no terminator.** On litert-lm `terminators_trimmed`
read 0 of 600 and the manifest records "600 of 600 litert-lm generations end
without a terminator", for both recipes, while liveness found no empty,
leaking or degenerate generation among the 600 and every one exited zero. The
generations stopped; the runtime hands back the text without its stop token.
The unterminated-share check reads only the transformers reference, whose
decoder is asked to keep special tokens.

**Measuring the second recipe took several times as long.** Timed in the
earlier run of these same bundles and not re-timed here: the first time each
bundle ran, litert-lm wrote an XNNPACK cache beside it, 601,946,856 bytes
for `dynamic_wi8_afp32` and 2,385,932,008 for `weight_only_wi8_afp32`, three
times that bundle's 771,404,928. On the same machine the 600-prompt verify,
float reference included and a five-prompt verify run before it, took 27
minutes for `dynamic_wi8_afp32` and 2 hours 10 minutes for
`weight_only_wi8_afp32`. Nothing measured whether the cache is the reason.

**What it did not establish.** No untuned-base score. This run never reached
the base step: the guard ahead of it stopped the container and recorded
"untuned Qwen3-0.6B reasons without a token limit on the CLI candidate". The
earlier run met the same thing from the other side: a five-prompt `verify`
given 900 seconds that did not finish. So training gain is unattributed here
too. One run.
Conversions measured on CPU in this run. The same artifacts were later run on a
phone, on both its CPU and its GPU — see *What four bits cost*. No NPU figure
for this family.

### Limitations carried by these manifests

Four on `dynamic_wi8_afp32`, three on `weight_only_wi8_afp32`: the last is
absent there because that cost does resolve.

- Measured on litert-lm's CPU backend; its GPU backend is a different executor
  and this number does not predict it.
- The candidate ran on CPU and the reference on CUDA, so the difference carries
  a hardware difference as well as a conversion one. Reported rather than
  refused: litetune cannot pin the runtime side to a device, and a refusal
  would leave such a machine unable to verify at all.
- Decoding was passed to transformers (`max_tokens` 256, greedy) but not to
  litert-lm, which uses the pinned runtime's defaults; both are greedy, and the
  token limit is unverified on the runtime side. 600 of 600 runtime generations
  end without a terminator, which the subsection above accounts for.
- On `dynamic_wi8_afp32` only: the sample does not resolve that difference.
  `weight_only_wi8_afp32` carries no such line, and its table entry above says
  why — +0.0167 ±0.0113, resolved on 12 discordant.

## What four bits cost

Everything above is 8-bit, except the Gemma 4 table, whose `dynamic_wi4b32_afp32`
row is a block-wise four-bit export of base weights rather than a tuned
checkpoint. This section converts two tuned checkpoints four more
ways: the `gemma-3-270m-it` run of the second family, and `Qwen/Qwen3-0.6B` @
`c1899de2` trained the same way — LoRA r16/α32, lr 2e-4, one epoch, bfloat16,
over the same 2,400 banking77 rows. Both are scored on the same 600 held-out
rows of split `0c7505b2b6f69ab1`, `--scorer exact-text`, 256-token limit, prompt
mode `runtime_rendered`, litert-lm 0.16.1's CPU backend against each model's own
float twin. Sizes are the `.litertlm` the export wrote.

| Qwen3-0.6B | bytes | exact match | cost of conversion | discordant |
|---|---|---|---|---|
| float twin | — | **0.6983** ±0.0367 | — | — |
| `dynamic_wi4_afp32` | 395,310,000 | *refused at the gate* | — | — |
| `weight_only_wi4_afp32` | 395,621,504 | *refused at the gate* | — | — |
| `dynamic_wi4b32_afp32` | 428,978,304 | 0.6633 ±0.0378 | **+0.0350** ±0.0247 *(resolved)* | 57 of 600 |
| `dynamic_wi4b32_emb8_afp32` | 500,691,888 | 0.6433 ±0.0383 | **+0.0550** ±0.0267 *(resolved)* | 67 of 600 |

| gemma-3-270m-it | bytes | exact match | cost of conversion | discordant |
|---|---|---|---|---|
| float twin | — | **0.6717** ±0.0376 | — | — |
| `dynamic_wi4_afp32` | 237,851,952 | *refused at the gate* | — | — |
| `weight_only_wi4_afp32` | 238,032,400 | *refused at the gate* | — | — |
| `dynamic_wi4b32_afp32` | 252,925,440 | 0.3233 ±0.0374 | **+0.3483** ±0.0503 *(resolved)* | 237 of 600 |
| `dynamic_wi4b32_emb8_afp32` | 332,617,008 | 0.3517 ±0.0382 | **+0.3200** ±0.0482 *(resolved)* | 218 of 600 |

**Channelwise four bits never reached a score.** The measurement harness gates a
bundle on five prompts before spending an hour on 600 — litetune has no gate of
its own — and both channelwise recipes failed that gate on both models: under
`dynamic_wi4_afp32` Qwen3 repeated itself on one of five prompts at a ratio of
0.9937 and gemma leaked `<bos>`; under `weight_only_wi4_afp32` Qwen3 did not
finish one prompt within 300 s, and gemma hit litetune's own 300 s per-prompt
limit on each of the first two, after which the gate's 900 s budget expired. No
manifest was written for that last one, so how many of the five it would have
finished is not known. The rendering check passed on all four
bundles, so neither refusal is a prompt the two sides disagreed about.

**Both recipes collapse, which points at the weights rather than the integer
kernels.** `weight_only_wi4_afp32` dequantises before compute and failed on the
same checkpoints as `dynamic_wi4_afp32` — on a different check, but neither
produced an answer — so integer activations are not what separates a working
bundle from a broken one here. A five-prompt diagnostic through every bundle of each checkpoint says
the same from the other side: at 8 bits both models emit the reference's label
and stop, the terminator scoring at or above −0.001 in log-probability; at 4 bits
channelwise, Qwen3 matches one of five labels exactly and on another emits the
right label and then does not stop — its terminator scores −0.64 to −4.17 — while
gemma produces no label at all, scoring the reference answer at −32 to −107
against −0.03 to −0.47 at 8 bits. Those last figures are the diagnostic's
`score_total` with the terminator token's own log-probability taken off, which is
why they are not the raw totals in `*__diag-int4.json`.

**Blocks of 32 fix the breakage and still cost accuracy.** A scale per 32 weights
instead of one per output channel passes the gate on both models and scores all
600 rows. That costs the tuned Qwen3-0.6B 3.5 points of exact match and the tuned
gemma-3-270m 34.8. Both intervals clear zero: these are differences this sample
settles, not noise.

**int8 embeddings do not rescue gemma-3-270m.** gemma holds 168M of its 268M
parameters in embeddings — a 262,144-token vocabulary at a hidden size of 640,
against the `base_parameters` its own tune metrics record — and
`dynamic_wi4b32_afp32` rounds them to four bits with everything else: its
embedder section compresses 640.01 MiB to 90.01 MiB, a ratio of 0.14 computed
from the convert manifest, against the 0.26 the converter prints for the 8-bit
run. litetune's own
`dynamic_wi4b32_emb8_afp32` keeps embeddings at int8 with OCTAV clipping on the
4-bit linear weights, which is the layout Google's quantization guide gives a
decoder. gemma still loses 32.0 points.

**What this does not establish.** That either block-wise recipe is better than
the other. Each was paired against its float twin and never against the other,
and the intervals overlap on both models — with int8 embeddings Qwen3 reads worse
(+0.0550 against +0.0350) and gemma better (+0.3200 against +0.3483), and neither
difference is one this design can settle. Nor does it say anything about models
larger than these two, or about 4-bit weights produced some other way, such as by
quantization-aware training. One run each, one runtime.

**On a phone, the backend changes the answer.** The Qwen3-0.6B artifacts above
were run on a Galaxy S24 (`SC-51E`, SM8650, Android 36) through Firebase Test
Lab, 600 rows per cell. The phone scores nothing: it records generations, and the
host scores them with the same `exact-text` scorer and the same paired interval
`verify` uses.

| bundle | backend | exact match | cost against the float reference | discordant |
|---|---|---|---|---|
| `dynamic_wi8_afp32` | CPU | 0.6817 ±0.0373 | **+0.0167** ±0.0146 *(resolved)* | 20 of 600 |
| `dynamic_wi8_afp32` | GPU | 0.6500 ±0.0382 | **+0.0483** ±0.0259 *(resolved)* | 63 of 600 |
| `dynamic_wi4b32_emb8_afp32` | CPU | 0.6333 ±0.0386 | **+0.0650** ±0.0255 *(resolved)* | 61 of 600 |
| `dynamic_wi4b32_emb8_afp32` | GPU | 0.6550 ±0.0380 | **+0.0433** ±0.0249 *(resolved)* | 58 of 600 |

Paired over the same prompts, the GPU backend costs **+0.0317 ±0.0233** against
the CPU one on the 8-bit bundle and **−0.0217 ±0.0142** on the mixed int4 bundle.
Both resolve, and they point in opposite directions: which backend is better
depends on the recipe. README's limitation — that litert-lm's GPU backend is a
different executor and a CPU number does not predict it — has a size here.

What the phone run does not establish: the difference between it and the cloud
CPU number for the same file (0.6817 against 0.6917) is two numbers, not a paired
test, because the cloud candidate's per-row generations were never shipped. One
phone model, one run per cell, greedy decoding.

## The same model through the runtime's tool path

`functiongemma-270m-it` LoRA-tuned on the 6434 single-call rows of
`google/mobile-actions` (5794 trained, 640 held out), in `runtime_rendered`
with the dataset's seven tool declarations, converted `dynamic_wi8_afp32`, and
measured the way an application calls it: `create_conversation(tools=...)`, the
runtime rendering the declarations and parsing the call itself. Both sides on
CPU. One run, 2026-09-17.

| | exact match | tool name | arguments | refused by the runtime |
|---|---|---|---|---|
| Float reference | 0.9250 ±0.0204 | | | |
| Tool path, runtime grammar off | 0.9125 ±0.0219 | 1.0000 | 0.9125 | 0 |
| Tool path, runtime grammar on | 0.9125 ±0.0219 | 1.0000 | 0.9125 | 0 |
| Cost of conversion | **+0.0125** ±0.0087 *(resolved, 8 discordant)* | | | |
| Effect of the grammar | 0.0000 *(no discordant row)* | | | |

The rendering check compared all 640 prompts, declarations included, and found
identical token ids on both sides. Every loss is in the arguments: the model
picked the right tool on all 640.

### What it took to get there

The first two runs of this pipeline failed in ways no unit test and no earlier
measurement could see, because the text scorer finds `call:` anywhere:

- **No call at all.** Trained completions lacked the `<start_function_call>`
  and `<end_function_call>` markers the runtime's parser looks for, and ended
  with the text turn's `<end_of_turn>`. Through the tool path the model
  returned no call on 5 of 5 prompts and nothing was refused.
- **A grammar that cost 0.1750.** With the markers in place, the same pipeline
  scored 0.9172 ±0.0214 with the runtime's grammar off and 0.7422 ±0.0339 with
  it on, against a reference of 0.9234. All 112 rows that differed had lost an
  argument — `send_email.body` on every row that carried one — and no row was
  helped by the grammar. The grammar enforces the declared property order, the
  declarations were sorted, and the model had been trained in the dataset's own
  argument order: an argument out of place is illegal, so the call closed
  without it. A 20-row probe confirmed it: `body` kept on 20 of 20 with the
  properties declared in the order the model writes them, on 0 of 20 with them
  sorted, and on 1 of 20 with four times the token budget. With arguments
  trained in the declared order, the grammar's effect is the zero above.

### Limitations carried by these numbers

- One run, one recipe, one dataset. The dataset's arguments are all strings, so
  no number went through the runtime here. Its parser returns every number as a
  double (`fc_parser.rs`, v0.16.1), and litetune compares numbers by value
  since — this run could not have shown the difference.
- The prompt is one user turn with the developer turn's date lines moved into
  it: litetune trains no system message. The 3220 rows with two or three calls
  were left out; every target here is one call.
- An application that enables constrained decoding with declarations whose
  properties are in another order than litetune trained against meets the
  grammar problem above; with it off, the runtime's default, the order changes
  only the prompt. A `runtime_rendered` bundle ships its declarations in the
  order the model learned; an application that builds its own list has to keep
  that order.
- flutter_gemma 1.8.3 renders FunctionGemma's declarations in Dart and does
  not pass them to the runtime, so this is not how it serves this model.
- No row was refused by the runtime in either mode. Since this run, a row the
  runtime gives no reply to is scored as a wrong answer rather than left out,
  which changes none of these numbers.
- The untuned base could not be measured through the tool path: converted
  from its Hub id it carries no SentencePiece tokenizer, and litert-lm 0.16.1
  refuses constrained decoding without one. So there is no training gain here.
- Without declarations in the prompt the text path measures nothing useful for
  this model: the reference itself degenerated on 50 of 640 generations.
