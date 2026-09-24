# What was measured, and what it established

Numbers for `litetune`. The first sections are `functiongemma-270m-it`
LoRA-tuned on `google/mobile-actions`, scored on 640 held-out single-call
examples; exact match means the tool name **and** every argument value. Each
section after them is another family: `gemma-3-270m-it` with the second scorer,
`gemma-4-E2B-it`, converted from its base weights and then fine-tuned,
`Qwen3-0.6B`, the first that is not a Gemma, `gemma-3-1b-it`, the other size one export rule claims, and
`Qwen2.5-0.5B-Instruct`, the one checkpoint here whose channelwise four-bit
export produced a score. The last returns to `functiongemma-270m-it`, measured
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

The two sections above measure models this project fine-tuned. This one opens
with neither: it compares two *conversions of the same base weights* --
`google/gemma-4-E2B-it`, untouched -- because `models.py` records that a Google
engineer recommends two recipes "to remain the model quality" and, in the same
sentence, that litetune had measured neither. The two subsections after it do
fine-tune that checkpoint, and the second converts and verifies one of those
runs.

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

### What the projection set is worth, on this family

Scoping says *where* a LoRA run may adapt. It does not say *which* projections,
and there `models.py` made a choice it could not defend with a number: litetune
names seven, while peft 0.20.0's own mapping for `gemma4` is
`.*language_model\..*\.(q_proj|v_proj)` -- two -- and Google's fine-tuning guide
passes no `target_modules` at all so that default applies. So both were run.

`google/gemma-4-E2B-it` fine-tuned on 2,400 rows of `mteb/banking77`, scored on
the 600 held-out rows of split `0c7505b2b6f69ab1` with `--scorer exact-text`,
prompt mode `runtime_rendered`, greedy, on one A100-SXM4-40GB. One base, one
split, one set of hyper-parameters; the projection set is the only difference,
and everything not named here was left at litetune's default.

The run passed no `--revision`, so it took whatever `main` pointed at that day:
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, read back afterwards from the
snapshot directory and `refs/main` in the cache rather than recorded by the run
itself. Both arms downloaded once into the same cache, so they share it; a
later run on this family should pin it.

Both arms ran on the branch that added `lora_container`, not on a release. No
released litetune could have produced arm A: up to 0.1.8 `tune` hands peft the
bare projection list, and on this checkpoint that stops in `get_peft_model` --
the vision and audio projections are `Gemma4ClippableLinear` wrappers and peft
0.20.0 dispatches on a bare `nn.Linear`. Arm B reproduces peft's own `gemma4`
default, which is a container regex for the same reason.

| | exact match | trainable parameters |
|---|---|---|
| seven projections, as shipped | **0.7883** ±0.0327 | 24,158,208 |
| `q_proj` and `v_proj`, peft's own default | 0.5483 ±0.0398 | 2,678,784 |

Paired, the difference is **+0.2400 ±0.0457, resolved**, on 196 of 600
discordant. The interval would have to be more than five times wider to contain zero.

**What it establishes.** On this task the projection set matters more than the
model does. The five checkpoints measured on this split span 0.6717 to 0.7883 --
11.7 points between the smallest checkpoint and the largest -- and changing
which projections a LoRA run adapts moved one checkpoint 24. Taking peft's
default would have left 5.1 billion parameters scoring 0.5483, below every
other family here.

**What it does not.** That seven is the right number, that a third set would not
do better, or that any of this holds on another task or another family. It is
two points on one task.

**Not the kind of number the table above carries.** Those are converted
artifacts against a float twin. These are two float checkpoints against each
other with no conversion between them, so the difference is training and
nothing else. Arm A's checkpoint was then converted and verified, which is the
section below.

### End to end, on the seven-projection checkpoint

Arm A is `google/gemma-4-E2B-it` at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
-- the commit read back from the cache after the run rather than pinned by it,
as the section above records. Its tuned weights through `convert` and `verify`:
two eight-bit recipes, the same 600 held-out rows of split `0c7505b2b6f69ab1`, `--scorer exact-text`,
prompt mode `runtime_rendered`, greedy. Candidate on litert-lm's CPU backend in
the same container as the tuning run, 12 vCPU; reference on the A100 through
transformers.
litetune 0.1.8; runtime `litert-lm==0.16.1`, `numpy==2.0.2`; reference
`torch==2.5.1`, `transformers==5.16.1`, `peft==0.20.0`,
`sentencepiece==0.2.0`.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Fine-tuned, seven projections | **0.7883** ±0.0327 | 0.7667 ±0.0338 | 0.7700 ±0.0337 |
| Cost of conversion | — | **+0.0217** ±0.0142 *(resolved, 19 discordant)* | **+0.0183** ±0.0108 *(resolved, 11 discordant)* |

Sizes are 5,071,853,520 and 5,072,115,888 bytes.

**Both eight-bit costs resolve, which no earlier section here has had happen
together.** Each recipe has resolved before on its own: `dynamic_wi8_afp32` in
run A of the FunctionGemma triple, on both phone backends for Qwen3-0.6B, and
on the tool path; `weight_only_wi8_afp32` in FunctionGemma's run B, on
Qwen3-0.6B and on gemma-3-1b. What has not happened before is both in one
run -- the closest is the FunctionGemma triple, where run A resolved one
recipe and run B the other.

**What that does not establish.** This checkpoint is roughly five times the
largest measured before it, and it is tempting to read the resolution off the
size. Nothing here separates size from the rest: one run, a different family,
a different litetune version. `dynamic_wi8_afp32` resolved on a 0.6B bundle on
a phone, so resolution is not gated on size -- and what buys an interval is
the imbalance among the discordant pairs, which this file says three times
over.

**Limitations carried by these two numbers.** The candidate ran on
litert-lm's CPU backend and the reference on cuda, so each cost carries a
hardware difference as well as a conversion one; `verify` reports that rather
than refusing it. Decoding was passed explicitly to transformers and not to
litert-lm, which used the pinned runtime's defaults; both are greedy. And
`weight_only_wi8_afp32` shipped with no `prefer_activation_type` -- its repack
did not finish inside the repack ceiling (`REPACK_TIMEOUT_S`, 300 s, or less if
the recipe's export timeout had less left), and the original was kept. That
ceiling is documented as a bound on a stalled tool rather than a budget,
chosen when a real repack of a 455 MB bundle finished in seconds; a 5 GB
bundle is the case it was not sized against. That does not
touch the number above, which was measured on CPU, but the bundle as shipped
is not the one to hand a GPU.

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

## The same family, four times the size

`google/gemma-3-1b-it` @ `dcc83ea8`, the other size `models.py` scopes the
`gemma-3-text` family to, and until this run the one of the two that had never
been measured. The rule already required
`--litert_lm_model_type_override=gemma3` for it, on the strength of the 270M;
this run is the first time the flag was exercised on the 1B, and `convert`
added it for the same reason — `config.json` declares `model_type:
gemma3_text`, which the exporter does not recognise.

The task, the rows and the scorer are the second family's, and so is
`max_seq_length` 160: LoRA α32, lr 2e-4, one epoch, batch 8, bfloat16 over the
same 2,400 rows of `mteb/banking77`, scored on the same 600 held-out rows of
split `0c7505b2b6f69ab1`, `--scorer exact-text`, a 256-token decode limit,
prompt mode `runtime_rendered`. `tune` was given no `--prompt-mode` and
inferred it: 0% of the 2,400 training prompts carry a control token. `verify`
read that record back from beside the checkpoint and, before generating
anything, found the runtime and the reference rendering identical token ids on
all 600 prompts, with the runtime's prefill count equal to the reference's on
the 8 it was sent. 13,045,760 of 999,885,952 parameters trained, final loss
0.3985, the loss computed on 0.1605 of tokens. The model is gated, so it ran
offline from a local copy of that revision. Training and the float reference
ran on one A100-SXM4-40GB, both candidates on litert-lm 0.16.1's CPU backend
in the same container, 12 vCPU. litetune 0.1.7; training environment
`torch==2.5.1`, `transformers==5.16.1`, `peft==0.20.0`,
`sentencepiece==0.2.0`; runtime `litert-lm==0.16.1`, `numpy==2.0.2`.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | **0.0000** | 0.0000 | — |
| Fine-tuned | **0.7533** ±0.0345 | 0.7450 ±0.0349 | 0.7367 ±0.0352 |
| Cost of conversion | — | +0.0083 ±0.0157 *(unresolved, 23 discordant)* | **+0.0167** ±0.0122 *(resolved, 14 discordant)* |

The two costs fall the same way as the fourth family's and for the same
arithmetic: `weight_only_wi8_afp32` resolves on 14 disagreements while
`dynamic_wi8_afp32` does not on 23. Sizes are 1,331,074,352 and 1,331,336,720
bytes.

### What this run established that the table does not show

**The untuned base answers nothing, and this time it was scored rather than
refused.** Both sides of the base comparison — the converted base and the
untuned float it came from — score **0.0000 on all 600 rows**. `verify` reports
`unmeasured` and exits 3, and the reason its manifest gives is not the zero: the
comparison names the untuned base as its reference, so conversion and training
are confounded in it and both attribution fields are unavailable. The zero is
what is left when that is said — two scores of nothing to subtract from each
other. The second family reached the same dead end from the other
side, where the base never reached the quality tier — refused for empty output
in the run of record, and for repetition in the withdrawn `prerendered` one. Two
sizes of one family, two ways of establishing nothing about training gain.

**The terminator is the template's close, as it is for the 270M.**
`tune` recorded `turn_terminator: {ids: [106, 107], source: "chat_template",
text: "<end_of_turn>\n"}`, and on the reference `terminators_trimmed` read 600
of 600, one marker each. On litert-lm it read 0 of 600, which is what the
fourth family's section describes: the runtime hands back the text without its
stop token.

**It scores higher than either model measured before it on these rows** —
0.7533 against Qwen3-0.6B's 0.6983 and gemma-3-270m's 0.6717 — and that is an
observation, not a resolved difference: litetune computes no interval for a
difference between two models, and all three references ran on a different
device from their candidates.

## A fifth family, and the first where channelwise four bits answered

`Qwen/Qwen2.5-0.5B-Instruct` @ `7ae55760`, 494,032,768 parameters, and a
checkpoint litetune held no rule for when it ran — as Qwen3-0.6B was when *it*
ran. `models.identify` returned None for it, so every `convert` in this run
printed the unknown-family note — that litetune's rules were paid for one
family at a time and a family it has not met is one whose required flags are
simply unknown. The bundle is typed correctly regardless, and not by
litetune: `config.json` says `model_type: "qwen2"`, and the switch in
`litert_lm_builder.py` — which lives in the export environment rather than in
this repository, read at `litert_torch/generative/export_hf/core/` in
litert-torch 0.10.0.dev20260826 — has `case 'qwen2' | 'qwen2p5'`. It did not
run offline: this run logged `HF_HUB_OFFLINE=unset` where both Gemma 3 runs
logged `HF_HUB_OFFLINE=1`. Whether the weights then came over the network or
off a cache already on the instance is not something these manifests record,
for this run or for those.

The task, the rows and the scorer are the second family's, and `max_seq_length`
256 is the fourth family's: LoRA α32, lr 2e-4, one epoch, bfloat16 over the same
2,400 rows of `mteb/banking77`, scored on the same 600 held-out rows of split
`0c7505b2b6f69ab1`, `--scorer exact-text`, a 256-token decode limit, prompt mode
`runtime_rendered`. `tune` was given no `--prompt-mode` and inferred it: 0% of
the 2,400 training prompts carry a control token. `verify` read that record back
from beside the checkpoint and, before generating anything, found the runtime
and the reference rendering identical token ids on all 600 prompts, with the
runtime's prefill count equal to the reference's on the 8 it was sent — on all
seven verifies this run produced, the untuned base included. 8,798,208 of 494,032,768 parameters trained, final
loss 0.4472, the loss computed on 0.0946 of tokens. Training and the float
reference ran on one A100-SXM4-40GB, every candidate on litert-lm 0.16.1's CPU
backend in the same container, 12 vCPU. litetune 0.1.8; training environment
`torch==2.5.1`, `transformers==5.16.1`, `peft==0.20.0`,
`sentencepiece==0.2.0`; runtime `litert-lm==0.16.1`, `numpy==2.0.2`.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | **0.0000** | 0.0000 | — |
| Fine-tuned | **0.7700** ±0.0337 | 0.7600 ±0.0342 | 0.7600 ±0.0342 |
| Cost of conversion | — | +0.0100 ±0.0131 *(unresolved, 16 discordant)* | +0.0100 ±0.0103 *(unresolved, 10 discordant)* |

Sizes are 647,312,304 and 647,492,736 bytes.

### What this run established that the table does not show

**Neither eight-bit cost resolves here, and on the two models before it one
did.** Qwen3-0.6B and gemma-3-1b both had `weight_only_wi8_afp32` resolve at
+0.0167 while `dynamic_wi8_afp32` did not; this model puts both inside their
intervals. The reason is in the discordant column rather than in the accuracy:
conversion changed the answer on 16 rows under one recipe and 10 under the
other, and a paired test on 600 examples cannot separate a difference that small
from none. That is a statement about this sample, not a finding that the
conversion is free.

**It scores higher than any model measured before it on these rows** — 0.7700
against gemma-3-1b's 0.7533, Qwen3-0.6B's 0.6983 and gemma-3-270m's 0.6717,
from the second smallest of the four. As with the 1B, this is an
observation and not a resolved difference: litetune computes no interval for a
difference between two models, and every reference here ran on a different
device from its candidate.

**The untuned base establishes nothing, as on the 1B.** Both
sides score **0.0000 on all 600 rows**, `verify` reports `unmeasured` and exits
3, and its manifest gives the same reason as the 1B's: the reference is the
untuned base, so training and conversion are confounded and both attribution
fields are unavailable. Of the five models fine-tuned and measured on banking77, not
one has a recorded training gain — two scored zero on both sides, one was refused for
empty output, one was never scored, and Gemma 4's tuned run records no base figure at
all. FunctionGemma, at the top of this file, is the only model
here with one at all — on another task, with another scorer, and with three
readings of the figure that section sets against each other.

**The terminator is the template's close, as in the three banking77 runs of
record before it.** `tune`
recorded `turn_terminator: {ids: [151645, 198], source: "chat_template", text:
"<|im_end|>\n"}`, and on the reference `terminators_trimmed` read 600 of 600,
one marker each. On litert-lm it read 0 of 600 — the runtime hands back the text
without its stop token, which the fourth family's section describes.

## What four bits cost

Everything above is 8-bit, except the Gemma 4 table, whose `dynamic_wi4b32_afp32`
row is a block-wise four-bit export of base weights rather than a tuned
checkpoint. This section converts four tuned checkpoints four more
ways: the `gemma-3-270m-it` run of the second family, `Qwen/Qwen3-0.6B` @
`c1899de2`, the `gemma-3-1b-it` run, and the `Qwen2.5-0.5B-Instruct` run above,
each trained the same way — LoRA α32, lr 2e-4, one epoch, bfloat16, over the
same 2,400 banking77 rows. All are
scored on the same 600 held-out rows of split `0c7505b2b6f69ab1`, `--scorer exact-text`, 256-token limit, prompt
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

| gemma-3-1b-it | bytes | exact match | cost of conversion | discordant |
|---|---|---|---|---|
| float twin | — | **0.7533** ±0.0345 | — | — |
| `dynamic_wi4_afp32` | 680,203,568 | *refused after scoring* | — | — |
| `weight_only_wi4_afp32` | 680,465,936 | *refused at the gate* | — | — |
| `dynamic_wi4b32_afp32` | 741,578,240 | 0.6650 ±0.0378 | **+0.0883** ±0.0287 *(resolved)* | 77 of 600 |
| `dynamic_wi4b32_emb8_afp32` | 879,990,064 | 0.6650 ±0.0378 | **+0.0883** ±0.0290 *(resolved)* | 79 of 600 |

| Qwen2.5-0.5B-Instruct | bytes | exact match | cost of conversion | discordant |
|---|---|---|---|---|
| float twin | — | **0.7700** ±0.0337 | — | — |
| `dynamic_wi4_afp32` | 332,256,176 | 0.5133 ±0.0400 | **+0.2567** ±0.0436 *(resolved)* | 178 of 600 |
| `weight_only_wi4_afp32` | 332,436,608 | *one generation of 600 never finished* | — | — |
| `dynamic_wi4b32_afp32` | 359,203,968 | 0.6933 ±0.0369 | **+0.0767** ±0.0257 *(resolved)* | 62 of 600 |
| `dynamic_wi4b32_emb8_afp32` | 422,409,136 | 0.6783 ±0.0374 | **+0.0917** ±0.0271 *(resolved)* | 69 of 600 |

**Channelwise four bits reached a score on one model of four.** On the other
three the harness never got that far. It gates a bundle on five prompts before
spending an hour on 600 — litetune has no gate of its own — and both channelwise
recipes failed that gate on gemma-3-270m and on Qwen3-0.6B: under
`dynamic_wi4_afp32` Qwen3 repeated itself on one of five prompts at a ratio of
0.9937 and gemma leaked `<bos>`; under `weight_only_wi4_afp32` Qwen3 did not
finish one prompt within 300 s, and gemma hit litetune's own 300 s per-prompt
limit on each of the first two, after which the gate's 900 s budget expired. No
manifest was written for that last one, so how many of the five it would have
finished is not known. The rendering check passed on all four
bundles, so neither refusal is a prompt the two sides disagreed about.

**On the 1B the gate opened and the run failed anyway**, which is the same
verdict arrived at one stage later. `dynamic_wi4_afp32` passed its five prompts
and then repeated itself on **65 of 600**, worst ratio 0.9978, so liveness
refused before any score was computed; `weight_only_wi4_afp32` did not finish
one of its five gate prompts within 300 s, as Qwen3's had not. So a five-prompt
gate is a cheap way to catch a bundle that is broken on every row, and no way
to catch one that is broken on one row in ten — a limitation of the harness's
gate, not of `verify`, which is what found it.

**On Qwen2.5-0.5B both channelwise recipes cleared the gate, and one of them
answered.** `dynamic_wi4_afp32` ran all 600 rows: 0.5133 against the float
twin's 0.7700, a cost of **+0.2567** that resolves on 178 changed rows. That is
the first channelwise figure in this file, and it is a third of the model's
accuracy — the recipe is not rehabilitated by having finally produced a number.
`weight_only_wi4_afp32` got further here than on any earlier model and still
produced nothing: its gate opened, 599 of 600 generations ran, one exceeded the
300 s per-prompt limit, and `verify` returned `failed_harness` and exit 4 rather
than scoring the rows that did finish. Across four models the same two recipes
have now produced a refusal at the gate, a refusal past the gate, a run that
generated 599 of 600 and was refused for the one it did not, and one resolved
cost — which is four ways of learning that what these recipes do is a property
of the checkpoint they are given.

**Dequantising before compute rescued nothing, on any of the four.**
`weight_only_wi4_afp32` does exactly that, and it is the one recipe in this file
that has never produced a score: it failed the gate on gemma-3-270m, Qwen3-0.6B
and gemma-3-1b, and on Qwen2.5-0.5B it cleared the gate and then lost a single
generation to the timeout. `dynamic_wi4_afp32`, which keeps activations in
integers, is the one that eventually answered. So integer activations are not
what separates a working bundle from a broken one here — if anything the
evidence runs the other way, and on three of the four checkpoints neither recipe
produced an answer at all. A five-prompt diagnostic through both int8 and both channelwise bundles of
gemma-3-270m and Qwen3-0.6B — the two block-wise recipes were never in it —
says the same from the other side: at 8 bits both models emit the reference's label
and stop, the terminator scoring at or above −0.001 in log-probability; at 4 bits
channelwise, Qwen3 matches one of five labels exactly and on another emits the
right label and then does not stop — its terminator scores −0.64 to −4.17 — while
gemma produces no label at all, scoring the reference answer at −32 to −107
against −0.03 to −0.47 at 8 bits. Those last figures are the diagnostic's
`score_total` with the terminator token's own log-probability taken off, which is
why they are not the raw totals in `*__diag-int4.json`.

**Blocks of 32 fix the breakage and still cost accuracy.** A scale per 32 weights
instead of one per output channel passes the gate on all four models and scores
all 600 rows. That costs the tuned Qwen3-0.6B 3.50 points of exact match, the
tuned Qwen2.5-0.5B 7.67, the tuned gemma-3-1b 8.83, and the tuned gemma-3-270m
34.83. Every interval clears zero: these are differences this sample settles,
not noise.

**Whatever the 270M's collapse is, it is not a property of Gemma 3.** The 1B is
the same family, the same export rule, the same template and the same run recipe
as the 270M, and four bits cost it 8.83 points where they cost the 270M 34.83.
So the family does not predict the cost, which is the one thing this pair
settles. It does not settle that size does, and the fourth model argues against
it: at 0.49B Qwen2.5 pays 7.67 points where Qwen3 at 0.6B pays 3.50 and
gemma-3-1b at 1.00B pays 8.83, so the four costs do not order by parameter count
in either direction. Only the 270M's 34.83 stands apart, and it is the smallest.
Parameter count, the share of the model that is embeddings, the architecture and
the headroom a higher float score leaves all vary together across these four,
and nothing in this design separates them.

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

On the 1B the two block-wise recipes reach **the same exact match**, 0.6650,
each paired against the float twin and neither against the other, and they reach
it from different rows — 77 discordant against 79 — for 138 MB more on disk. The
1B holds 302M of its 1.00B parameters in embeddings, the same 262,144-token
vocabulary at a hidden size of 1,152, a smaller share of the model than the
270M's. What that buys at four bits is not visible in these two scores.

**What this does not establish.** That either block-wise recipe is better than
the other. Each was paired against its float twin and never against the other,
and the intervals overlap on all four models — with int8 embeddings Qwen3 reads
worse (+0.0550 against +0.0350), Qwen2.5 worse as well (+0.0917 against
+0.0767), gemma-3-270m better (+0.3200 against +0.3483) and gemma-3-1b
identically (+0.0883 either way), and none of those differences is one this
design can settle. Nor does it say anything about models larger than these
four, or about 4-bit weights produced some other way, such as by
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
CPU. One run, 2026-09-17; a second, every stage from `prepare` on with the code
that reads a reply the way the runtime does, gave the same number in every cell
on 2026-09-19.

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
- No row was refused by the runtime in either mode, in either run. A row the
  runtime gives no reply to is scored as a wrong answer when its parser refused
  it or the prompt reached the token limit, and leaves the mode unmeasured for
  any other reason. The second run read every reply whole: none carried two
  calls, in either mode or in the reference.
- The first run read the reference with the text path's parser, which takes the
  first call anywhere in the text. It is now read as the runtime reads a reply,
  only between the call markers and one call to a pair. The second run kept the
  reference's texts and read them both ways: 0.9250 either way, and no row reads
  differently.
- The untuned base could not be measured through the tool path: converted
  from its Hub id it carries no SentencePiece tokenizer, and litert-lm 0.16.1
  refuses constrained decoding without one. So there is no training gain here.
- Without declarations in the prompt the text path measures nothing useful for
  this model: the reference itself degenerated on 50 of 640 generations.
