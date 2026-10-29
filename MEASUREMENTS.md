# What was measured, and what it established

Numbers for `litetune`. The first sections are `functiongemma-270m-it`
LoRA-tuned on `google/mobile-actions`, scored on 640 held-out single-call
examples; exact match means the tool name **and** every argument value. Each
section after them is another family: `gemma-3-270m-it` with the second scorer,
`gemma-4-E2B-it`, converted from its base weights and then fine-tuned,
`Qwen3-0.6B`, the first that is not a Gemma, `gemma-3-1b-it`, the other size one export rule claims, and
`Qwen2.5-0.5B-Instruct`, the one checkpoint here on which channelwise four bits
produced a score. Then `functiongemma-270m-it` again, measured through the
runtime's tool path the way an application calls it, and last
`gemma-3-270m-it` trained to choose one of an application's options rather
than to write an answer.

This file exists so the README can be a usage guide. It is the longer story:
what reproduced, what did not, and which published claims were withdrawn.

## Which runtime a number was taken on

Every litetune from 0.1.0a1 to 0.1.9 pins the same toolchain:
`litert-lm==0.16.1` for the candidate, `litert-torch-nightly==0.10.0.dev20260826`
for conversion (with `litert-lm-builder==0.16.1` from 0.1.4), and `torch==2.5.1`,
`transformers==5.16.1`, `peft==0.20.0` for training and the float reference. The
sections below name the runtime they ran on where they record one; the phone
sections ran a runtime of their own.

litetune 0.2.0 – 0.3.0 pin `litert-lm==0.17.1`, and that runtime does not give
0.16.1's answers. Nor does one CPU give another's. Measured 2026-09-27 on
Linux x86-64, CPU backend, through litetune's own driver scripts, each version
in its own environment and each run on its own copy of the bundles, with no
weight cache from another run beside it. The bundles were litert-community's
`gemma3-270m-it-q8` and `gemma3-1b-it-int4` and a FunctionGemma bundle this
project converted and keyed `fp32`; each of twelve prompts was prefilled and
one fixed sentence scored after it with `run_text_scoring`, which no sampler or
detokenizer touches -- the model's arithmetic and nothing else.

| Per-token scores identical, of 36 prompt-bundle scorings | same runtime | 0.16.1 against 0.17.1 |
|---|---|---|
| same CPU model, another run or machine | **36** | **1** |
| Intel Xeon @ 2.20GHz against AMD EPYC 7B12 | **0** | 0 |

Six builds took part, each on a machine drawn from a pool of 32-vCPU ones: four
landed on the Intel model and two on the AMD one. The two that recorded their
CPU features showed AVX2 and FMA and none of AVX-512 or AMX. Two builds did not
record their CPU and are placed with a model because their scores equal that
model's recorded builds' to the bit -- an inference, not a record. Within a CPU
model every run reproduced every
score to the bit, across machines and, on 0.17.1 on the Intel model, with 1, 4,
8, 16 or 32 threads alike; across the two models not one scoring did, by as much
as 1.52 nats on a single token, and across the two runtimes one did, by as much
as 1.33.
So a litert-lm number belongs to a runtime version *and* a CPU, and two compare
only when both match. The model name is not enough to say two CPUs match: the
machine the tool-path section's 0.17.1 runs used reports the same
`Intel(R) Xeon(R) CPU @ 2.20GHz` as the Intel builds above but has AVX-512,
which the one Intel build that recorded its features did not, so a CPU is its
name and its features together. From 0.2.0 a `verify` manifest records
the CPU its candidate ran on beside the runtime, wherever the driver could read
it. The sections below
name their runtime and, apart from the 0.17.1 runs, not
their CPU, and those taken on machines drawn from a pool may have landed on
different CPU models.

The same holds for what litetune scores. litetune's driver scripts on one
machine -- twelve prompts in both prompt modes on the three bundles,
and forty held-out tool-path rows with the grammar off and on -- gave the same
output on 152 of 152 when 0.16.1 ran twice, and on 47 of 152 between 0.16.1 and
0.17.1. That pair was taken on the AMD model, by the same inference; a second
0.17.1 run, whose scores place it with the Intel model, matched the first on 43 of 152, which is the CPU
and not nondeterminism. What in the runtime changed the arithmetic is not
isolated. Between the tags LiteRT-LM moves its LiteRT dependency three times, across two
XNNPACK updates and changes to int8 and dynamically quantised fully-connected
kernels; its own call parser, prompt rendering and the binding calls litetune
makes read the same, apart from integers the parser now returns as integers.

Neither version is shown to be the more accurate: that needs both scored
against the float model on one CPU, and it is not done here -- the banking77
re-measurement below scores eight bundles against it on both, but not on a CPU
the two runs are shown to share. The FunctionGemma bundle in
these runs is an early conversion that answers most tool-path rows with several
calls, so its exact match says nothing about quality and is not quoted. On the
Intel model a scoring process, three bundles in turn, peaked at 1.07 to 1.14 GiB
resident, whatever the thread count.

One behaviour changed that a number can hide. Where a bundle carries no
SentencePiece tokenizer, v0.16.1 refused to create a conversation with the
grammar on; v0.17.1 logs a warning and decodes without the grammar
(`gemma3_data_processor.cc`, `function_gemma_data_processor.cc`). litetune reads
the bundle's tokenizer sections first and refuses that mode itself, so a
grammar-on number is never unconstrained output.

So two numbers compare only when they name the same runtime. The sections below
keep their 0.16.1 numbers; one taken on 0.17.1 goes beside them and never in
place of them.

Re-measured on 0.17.1 so far: the four banking77 eight-bit pairs, below; the
four-bit recipes on the same four checkpoints, in *Four bits on litert-lm
0.17.1*; the tuned Gemma 4 E2B's two eight-bit bundles, in *Gemma 4 E2B on
litert-lm 0.17.1*; and FunctionGemma's tool path, in *The same checkpoint on
litert-lm 0.17.1*. Each of those was taken again on litert-lm 0.18.0, the runtime
0.4.0 – 0.7.1 pin, in *The 0.17.1 re-measurements again on litert-lm 0.18.0*, after the
banking77 sections. The rest has not been; among it the headline FunctionGemma
runs, the conversions of Gemma 4's base weights and the runs on untuned bases
are `verify` numbers taken on 0.16.1, and the phone runs ran on the device.

### The banking77 sections on litert-lm 0.17.1

The four banking77 eight-bit pairs were verified again on 2026-09-28 with
litetune at commit 0744a1e (unreleased; it reports itself as 0.1.9) on
litert-lm 0.17.1. Nothing was retrained or re-converted: each bundle's SHA-256
equals the one `convert` recorded when 0.1.6, 0.1.7 or 0.1.8 built it, and the
reference checkpoint, the 600 held-out rows of split `0c7505b2b6f69ab1`, the
scorer, the prompt mode and the declared decoding are each section's. `main`'s
`convert` pins a newer conversion stack, and a bundle it builds is not measured
here. The candidate's CPU is recorded, `Intel(R) Xeon(R) CPU @ 2.20GHz` with
AVX-512 and 12 vCPU, and the reference ran on cuda.

Three things changed between each section's two runs, not one: the runtime; the
transport, which is now litert-lm's Python API in one process per split rather
than `litert-lm run` per prompt (#50), and which gave byte-identical text in 36
of 36 comparisons on 0.16.1; and possibly the CPU, which the 0.16.1 runs did
not record. So a difference between the two runs is not attributed to the
runtime.

Cost of conversion, float minus candidate, with its discordant rows:

| | `dynamic_wi8_afp32`, 0.16.1 | `dynamic_wi8_afp32`, 0.17.1 | `weight_only_wi8_afp32`, 0.16.1 | `weight_only_wi8_afp32`, 0.17.1 |
|---|---|---|---|---|
| gemma-3-270m | +0.0000 (34) | −0.0067 (30) | +0.0033 (16) | −0.0033 (12) |
| Qwen3-0.6B | +0.0067 (20) | +0.0000 (10) | **+0.0167** (12, resolved) | +0.0050 (5) |
| gemma-3-1b | +0.0083 (23) | +0.0100 (12) | **+0.0167** (14, resolved) | +0.0000 (4) |
| Qwen2.5-0.5B | +0.0100 (16) | +0.0000 (8) | +0.0100 (10) | −0.0017 (3) |

Two costs resolved on 0.16.1 and none does on 0.17.1. That is two runs'
verdicts, not a measured change in a cost, and neither run is withdrawn:
litetune computes no interval between two runs, and the manifests keep scores,
not rows, so the paired test between the two candidates cannot be computed.
Every candidate is discordant with its reference on fewer rows on 0.17.1, eight
of eight; that too is an observation without an interval, and the reference
was re-run as well. Each section has its pair in full. The tuned Gemma 4 E2B's
pair, on the same split, was re-measured separately, in *Gemma 4 E2B on
litert-lm 0.17.1*.

### The 0.17.1 re-measurements again on litert-lm 0.18.0

Verified again from 2026-10-08 to 2026-10-09 on litert-lm 0.18.0 with the code of pull
request #67 before it was merged: commit c7ee2da (the runs recorded 5ddb043, the same
tree before a rebase), and for the Qwen bundles converted again and verified, 02844df,
which includes the packaged Qwen templates (both report 0.3.0). They ran on the machine
type the 0.17.1 runs used: every manifest that measured a candidate records it on
`Intel(R) Xeon(R) CPU @ 2.20GHz` with AVX-512 and 12 vCPU, and the reference on cuda.
Nothing was retrained.

Two kinds of bundle were run. The bundles the 0.17.1 runs measured, unchanged -- each
that has a manifest on both runtimes carries the SHA-256 the 0.17.1 manifest records --
so the runtime is the one thing that moved. And, for FunctionGemma and the two Qwen
sections, bundles converted again from the same checkpoints by that code (c7ee2da, and
02844df for the Qwen bundles), with
`litert-torch-nightly` 0.10.0.dev20260926, `ai-edge-quantizer-nightly`
0.10.0.dev20260926 and litert-lm-builder 0.18.0, where the bundles they are compared
with were converted in September with 0.10.0.dev20260826, 0.10.0.dev20260914 and 0.16.1:
for those the conversion stack moved as well as the runtime, and for Qwen the template.

**Every number that reached the quality tier is the 0.17.1 number again**: the same
candidate and reference exact match and the same cost with the same count of
discordant rows, to four places. The manifests keep scores, not rows, so the rows
themselves are not shown to be the same; equal totals and equal discordant counts on
600 or 640 rows are what is claimed.

| model, recipe | reference | 0.17.1 | 0.18.0, the same bundle | 0.18.0, converted again |
|---|---|---|---|---|
| gemma-3-270m, `weight_only_wi8_afp32` | 0.6717 | 0.6750, −0.0033 (12) | 0.6750, −0.0033 (12) |  |
| gemma-3-270m, `dynamic_wi8_afp32` | 0.6717 | 0.6783, −0.0067 (30) | 0.6783, −0.0067 (30) |  |
| gemma-3-1b, `weight_only_wi8_afp32` | 0.7533 | 0.7533, 0.0000 (4) | 0.7533, 0.0000 (4) |  |
| gemma-3-1b, `dynamic_wi8_afp32` | 0.7533 | 0.7433, +0.0100 (12) | 0.7433, +0.0100 (12) |  |
| qwen2.5-0.5b, `weight_only_wi8_afp32` | 0.7700 | 0.7717, −0.0017 (3) | not verified: the first send raised | 0.7717, −0.0017 (3) |
| qwen2.5-0.5b, `dynamic_wi8_afp32` | 0.7700 | 0.7700, 0.0000 (8) | not verified: the first send raised | 0.7700, 0.0000 (8) |
| qwen3-0.6b, `weight_only_wi8_afp32` | 0.6983 | 0.6933, +0.0050 (5) | refused: the render check failed | 0.6933, +0.0050 (5) |
| qwen3-0.6b, `dynamic_wi8_afp32` | 0.6983 | 0.6983, 0.0000 (10) | refused: the render check failed | 0.6983, 0.0000 (10) |
| gemma-3-270m, `dynamic_wi4b32_emb8_afp32` | 0.6717 | 0.2650, +0.4067 (262, resolved) | 0.2650, +0.4067 (262, resolved) |  |
| gemma-3-270m, `dynamic_wi4b32_afp32` | 0.6717 | 0.3067, +0.3650 (251, resolved) | 0.3067, +0.3650 (251, resolved) |  |
| gemma-3-1b, `dynamic_wi4b32_emb8_afp32` | 0.7533 | 0.6750, +0.0783 (75, resolved) | 0.6750, +0.0783 (75, resolved) |  |
| gemma-3-1b, `dynamic_wi4b32_afp32` | 0.7533 | 0.6700, +0.0833 (82, resolved) | 0.6700, +0.0833 (82, resolved) |  |
| qwen2.5-0.5b, `dynamic_wi4b32_emb8_afp32` | 0.7700 | 0.6867, +0.0833 (62, resolved) | not verified: the first send raised | 0.6867, +0.0833 (62, resolved) |
| qwen2.5-0.5b, `dynamic_wi4b32_afp32` | 0.7700 | 0.6933, +0.0767 (60, resolved) | not verified: the first send raised | 0.6933, +0.0767 (60, resolved) |
| qwen2.5-0.5b, `weight_only_wi4_afp32` | 0.7700 | 0.5200, +0.2500 (174, resolved) | not verified: the first send raised | 0.5200, +0.2500 (174, resolved) |
| qwen2.5-0.5b, `dynamic_wi4_afp32` | 0.7700 | 0.5017, +0.2683 (191, resolved) | not verified: the first send raised | 0.5017, +0.2683 (191, resolved) |
| qwen3-0.6b, `dynamic_wi4b32_emb8_afp32` | 0.6983 | 0.6483, +0.0500 (56, resolved) | refused: the render check failed | 0.6483, +0.0500 (56, resolved) |
| qwen3-0.6b, `dynamic_wi4b32_afp32` | 0.6983 | 0.6700, +0.0283 (55, resolved) | refused: the render check failed | 0.6700, +0.0283 (55, resolved) |
| gemma-4-e2b, `weight_only_wi8_afp32` | 0.7883 | 0.7833, +0.0050 (3) | 0.7833, +0.0050 (3) | |
| gemma-4-e2b, `dynamic_wi8_afp32` | 0.7883 | 0.7800, +0.0083 (11) | 0.7800, +0.0083 (11) | |

Each cell is the candidate's exact match on banking77's 600 test rows and the
conversion cost, with the count of discordant rows. The four-bit recipes refused on
0.17.1 were refused on 0.18.0 for the same reason and with the same counts. At the
five-prompt gate: gemma-3-270m `dynamic_wi4_afp32` (repeated itself on 5 of 5), Qwen3
`dynamic_wi4_afp32`, converted again (2 of 5), and `weight_only_wi4_afp32` on
gemma-3-270m, gemma-3-1b and Qwen3, converted again (five prompts not done in 900 s).
Past the gate: gemma-3-1b `dynamic_wi4_afp32` (repeated itself on 65 of 600).

**The old Qwen bundles do not survive 0.18.0**, and the two fail differently. From
v0.18.0 the runtime hands a bundle's template every message's content as a list of parts
(LiteRT-LM #3544); both Qwen checkpoints' templates read it as a string. `verify`
refused every Qwen3 bundle -- converted in September by litetune a few unreleased
commits after 0.1.6, with litert-lm-builder 0.16.1 -- at its rendering check, the
eight-bit ones on all 600 prompts (*600 of 600 prompts differ; prompt 0: the runtime
renders 8 tokens and the reference 43*) and the four-bit ones on all five prompts of
their gate. The runtime's rendering it recorded ends
`<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n` -- the user's text gone, and no
error -- where on 0.17.1 the same bundles rendered the reference's ids for all 600
prompts. Through each Qwen2.5 bundle, eight-bit and four-bit, `verify` stopped at the
same check: the runtime's first send raised `litert_lm_conversation_send_message
failed`. The bundles converted again carry the checkpoint's own template behind a
block that turns a list of text parts back into the string: `templates/qwen3-0.6b.jinja`
and `qwen2.5-0.5b-instruct.jinja` as 02844df packaged them, which differ from the two
files litetune 0.4.0 packaged in Jinja comments only. Every one of those
verified in full rendered the same token ids as the reference for all 600 prompts and
scored what the 0.17.1 bundle scored. Up to 0.3.0 neither Qwen rule adds a template
(`models.py` at v0.3.0), so a bundle litetune 0.3.0 or earlier converted without a
template passed by the caller carries the checkpoint's template alone, as these did. **A
Qwen3 0.6B or Qwen2.5 0.5B Instruct bundle converted by litetune 0.3.0 or earlier
without a template of the caller's has to be converted again for an application on
LiteRT-LM 0.18.0**, as do some litetune 0.4.0 converted (the README's *From LiteRT-LM
0.18.0 a template is handed a list of parts* says which); what was measured is the
checkpoints at the revisions the templates name. litetune after 0.4.0 no longer
packages those two files: at export it puts the same block in front of whatever chat
template the converted checkpoint's own tokenizer carries. On 2026-10-11 the four
eight-bit bundles converted from the two tuned checkpoints by the litetune that does
this -- its code before it was rebased onto the stage-environment change (#70) --
each carried that composed file as their `jinja_prompt_template`, read back from the
bundle by `convert` itself, and the template `convert` read from each checkpoint had
the sha256 of the checkpoint template inside 02844df's file for it (a55ee1b1… for
Qwen3, cd8e9439… for Qwen2.5). Against 02844df's bundles their template differs only
in Jinja comments, and the rest of their metadata only in the order of the stop
tokens, which the exporter builds from a Python set (`litert_lm_builder.py:216-254`)
and which already differed between two bundles 02844df converted. Verified on
litert-lm 0.18.0, each rendered the reference's token ids for all 600 prompts and
scored what 02844df's bundle of the same recipe scored: Qwen2.5 0.7700, 0.0000 (8)
and 0.7717, −0.0017 (3), Qwen3 0.6983, 0.0000 (10) and 0.6933, +0.0050 (5), for
`dynamic_wi8_afp32` and `weight_only_wi8_afp32`; the same four results came back at
each of the branch's five commits.
Which checkpoints get the block is the qwen-3 and qwen-2.5 patterns' to say
(`models.py`); one they do not claim gets none, and whether its template reads a list
is what verify's rendering check shows. From 0.4.3 the qwen-3 pattern hands a checkpoint
LiteRT-LM's own Qwen 3 template instead, where it is offered (below).

**Every Qwen3 bundle up to 0.4.2 answers one message of a LiteRT-LM conversation.** LiteRT-LM
renders the whole history on every message and requires the new rendering to start with the
previous one (`Conversation::GetSingleTurnTextFromFullHistory`; LiteRT-LM #3443). Qwen3's own
template renders the last assistant turn with an empty `<think>` pair and drops the pair once
the next user message arrives, so the second message fails with `INTERNAL: The new rendered
template string does not start with the previous rendered template string`. Up to 0.3.0 a Qwen3
bundle carries that template alone (`models.py` at v0.3.0), and from 0.4.0 behind the block.
Measured on 2026-10-13 on Linux x86_64 on litert-lm 0.18.0's CPU backend, `dynamic_wi8_afp32`:
the tuned Qwen3 checkpoint converted by litetune 0.4.2 answered the first message and failed the
second, in both conversations tried; `Qwen/Qwen3-0.6B` as published, converted by the main branch
after #77, did the same with a reply cut at 64 tokens and with one of 512. Converted by #78's code,
both carry LiteRT-LM v0.18.0's `models/qwen3/chat_template.jinja` and answered every message sent
(three in one conversation; "Paris", then "Berlin" for the base model at 512 tokens, its reasoning
in the `thought` channel). The tuned bundle verified at 0.6983 for candidate and reference, cost
0.0000 (10), as before, with the reference's token ids for all 600 prompts. The Qwen2.5 and
Gemma 3 bundles of the same run held a three-message conversation. litetune 0.4.3 checks each
bundle's template for this after export and says in the report when one answers one message.

FunctionGemma's tool path, both grammar modes, 640 rows: the bundle converted on
2026-09-19 and the same checkpoint converted again at c7ee2da -- litert-lm-builder
0.18.0, the GPU activation key written, a relative `--output-dir` on Python 3.11 --
both score 0.9109 ±0.0221 against 0.9250 for the float reference, cost **+0.0141**
±0.0092 *(resolved, 9 discordant)*, as on 0.17.1.

**litert-lm-builder 0.18.0 reads and writes back a bundle the 0.17.1 runs measured.**
The gemma-3-270m `dynamic_wi8_afp32` bundle converted on 2026-09-15, unpacked and
packed again by builder 0.18.0, came out at the same 455,759,152 bytes. `litert-lm-peek`
lists every section of it at the same offsets with the same metadata; what differs is
the uuid and creation timestamp, which `pack` writes new, and the file's LiteRT-LM
version, 1.6.0 before and 1.7.0 after. Verified on 0.18.0, the repacked bundle scores
0.6783, −0.0067 (30), the original's numbers.

**On a phone the GPU key is still needed.** The September probe again on a Galaxy S24
(`SC-51E`) through Firebase Test Lab with `litertlm-android` 0.18.0, the same 20
native-tool-path FunctionGemma prompts and the same two bundles:

| bundle, backend | runtime | `<pad>` in the text | no text and no call | right tool | first call exact |
|---|---|---|---|---|---|
| without the key, GPU | 0.16.1 (2026-09-05) | 14 of 20 | 0 | 3 | 2 |
| without the key, GPU | **0.18.0** | 0 | **8 of 20** | **9** | **6** |
| `fp32` key, GPU | 0.16.1 | 0 | 0 | 20 | 15 |
| `fp32` key, GPU | **0.18.0** | 0 | 0 | **20** | **15** |
| `fp32` key, CPU | 0.16.1 | 0 | 0 | 20 | 14 |
| `fp32` key, CPU | **0.18.0** | 0 | 0 | **20** | **14** |

Without the key the F16 failure no longer prints `<pad>`: eight rows come back with no
text and no call after 6.4 to 7.2 s each, where 13 of September's 14 flood rows took
14.5 to 19.1 s. Why the rows now come back empty is not established. Both runs without
the key also end two rows in a send error, where the runtime could not parse the call
the model wrote; no column of the table counts them. With the key the totals are
September's on both backends, though 4 of 20 GPU rows and 5 of 20 CPU rows return
different calls. *First call exact* compares a reply's first call only; comparing every
call, the `fp32` GPU rows are 13 of 20 exact on 0.18.0 and 14 in September.

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

**Which CPU each run's candidate ran on was not recorded**, and "the same
configuration end to end" did not include it. On litert-lm the CPU model changes
the arithmetic about as much as the runtime version does (see *Which runtime a
number was taken on*). The fine-tuned float scores differ between the three runs
as well, so the spread above is not the CPU's alone; how much of the converted
numbers' spread is the CPU's cannot be separated after the fact.

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

On litert-lm 0.16.1. The same bundles on 0.17.1 are [at the end of this
section](#gemma-3-270m-on-litert-lm-0171).

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

- The candidate ran on litert-lm's CPU backend. The wording these manifests carry is the
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

### gemma-3-270m on litert-lm 0.17.1

The same bundles and rows, verified again as [The banking77 sections on
litert-lm 0.17.1](#the-banking77-sections-on-litert-lm-0171) describes. The two
runs differ in runtime, transport and possibly CPU, so a difference between the
rows is not attributed to the runtime.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| 0.16.1 run | **0.6717** ±0.0376 | 0.6717 ±0.0376 | 0.6683 ±0.0377 |
| 0.17.1 run | **0.6717** ±0.0376 | 0.6783 ±0.0374 | 0.6750 ±0.0375 |
| Cost of conversion, 0.17.1 run | — | −0.0067 ±0.0179 *(unresolved, 30 discordant)* | −0.0033 ±0.0113 *(unresolved, 12 discordant)* |

Neither cost resolves in either run. The float reference is the same to four
places; the discordant rows are 30 and 12, against 34 and 16.

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
model does. The seven checkpoints measured on this split span 0.6717 to 0.7883 --
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

On litert-lm 0.16.1. The same bundles on 0.17.1 are [at the end of this
section](#gemma-4-e2b-on-litert-lm-0171). Sizes are 5,071,853,520 and
5,072,115,888 bytes.

**On 0.16.1 both eight-bit costs resolve, which no earlier section here has had
happen together.** Each recipe has resolved before on its own:
`dynamic_wi8_afp32` in run A of the FunctionGemma triple, on both phone backends
for Qwen3-0.6B, and on the tool path; `weight_only_wi8_afp32` in FunctionGemma's
run B, on Qwen3-0.6B and on gemma-3-1b (on 0.16.1; on 0.17.1 neither does). What
has not happened before is both in one run -- the closest is the FunctionGemma
triple, where run A resolved one recipe and run B the other.

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
touch the number above, whose candidate ran on CPU, but the bundle as shipped
is not the one to hand a GPU.

### Gemma 4 E2B on litert-lm 0.17.1

The two bundles above, each SHA-256 equal to the one `convert` recorded on
2026-09-22, verified again on 2026-09-29 with litetune 0.2.0 (commit dfb4544) on
litert-lm 0.17.1, against the same checkpoint on the same 600 rows, and
otherwise as [The banking77 sections on litert-lm
0.17.1](#the-banking77-sections-on-litert-lm-0171) describes. Both manifests
record the candidate on `Intel(R) Xeon(R) CPU @ 2.20GHz` with AVX-512 and 12
vCPU, and the reference on cuda. The two runs differ in runtime, transport and
possibly CPU, so a difference between them is not attributed to the runtime.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| 0.16.1 run | **0.7883** ±0.0327 | 0.7667 ±0.0338 | 0.7700 ±0.0337 |
| 0.17.1 run | **0.7883** ±0.0327 | 0.7800 ±0.0331 | 0.7833 ±0.0330 |
| Cost of conversion, 0.17.1 run | — | +0.0083 ±0.0108 *(unresolved, 11 discordant)* | +0.0050 ±0.0057 *(unresolved, 3 discordant)* |

Both costs resolved on 0.16.1 and neither does here: +0.0217 on 19 discordant
rows, sixteen one way, and +0.0183 on 11, all eleven one way, there; +0.0083 on
11, eight one way, and +0.0050 on 3, all three one way, here. That is two runs'
verdicts, not a measured change in a cost, and neither run is withdrawn. The
float reference is the same to four places. The conversions of the base weights
in this family's first table were not re-measured.

### Gemma 4 E2B without its towers

Google's `gemma-4-E2B-it.litertlm` (`litert-community/gemma-4-E2B-it-litert-lm`
at `b3ca0d2f`, sha256 `181938105e0eefd1…`), measured with litert-lm 0.18.0's
Python API on the CPU backend, on an AMD EPYC 7B12. Nothing here is
a tuned model or a litetune export; it is the bundle Google ships, and the
question is only what dropping its towers changes.

Its six tower sections -- `audio_encoder_hw`, `audio_adapter`, `end_of_audio`,
`vision_encoder`, `vision_adapter`, `end_of_vision` -- are 332,387,932 of its
2,588,147,712 bytes, 12.8 %. `litetune towers --drop vision --drop audio`, with pull
request #68's code at its commit *Answer the full review of `litetune towers`*
(2026-10-10; f14542a before a rebase that left `towers.py` unchanged), wrote
2,255,690,576 bytes with every kept section byte for byte, LlmMetadata included, in
22.5 s; `litert-lm describe` lists text as its only input modality, on CPU and GPU,
and still reports the metadata's `Max Vision Token Budget: 280`. A first measurement
on 2026-10-09 used a script of its own that compared the metadata in the text form
`unpack` writes, which had lost 2 bytes of it (`gemma4` field 13, which 0.18.0's
proto reserves); its file had the same length and gave the same answers.

Eight prompts, greedy, at most 128 output tokens, a fresh conversation each, each
bundle in its own process:

| | answers equal to the full bundle's | engine load | peak RSS |
|---|---|---|---|
| full bundle | — | 3.88 s | 1,620.5 MB |
| what `towers --drop` wrote | **8 of 8** | 3.89 s | 1,620.8 MB |

On this CPU backend the saving is download and storage, not memory.

An image (a red 256×256 PNG, "What colour is this image? Answer in one word.")
and an audio clip (one second of a 440 Hz tone, "Describe this sound in one
sentence."), engine opened without and with the matching backend:

| bundle | input | backend | result |
|---|---|---|---|
| full (2026-10-09) | image | — | the send raises `litert_lm_conversation_send_message failed` |
| full (2026-10-09) | image | vision | "Red" |
| full (2026-10-09) | audio | — | the send raises the same error |
| full (2026-10-09) | audio | audio | "The sound is a high-pitched, sustained tone with a slightly wavering quality." |
| `towers --drop` output | image | — | the send raises the same error |
| `towers --drop` output | image | vision | the engine is not created: `Failed to create LiteRT-LM engine` |
| `towers --drop` output | audio | — | the send raises the same error |
| `towers --drop` output | audio | audio | the engine is created, the conversation is not: `Failed to create conversation` |

The full bundle answering both is the control. Without its towers every media
input raised an error and none produced text; where it raised depends on the
modality and on how the engine was opened -- an application that opens the engine
with a vision backend cannot load the bundle at all, even for text. Only the
Python binding on one x86-64 CPU was observed -- not the Kotlin, Swift or C
bindings, and not a phone.

### Towers grafted into an SM8850 NPU bundle

A text-only Gemma 4 E2B build compiled for the SM8850 NPU, not published in
`litert-community/gemma-4-E2B-it-litert-lm`, carries an `aux` section, the
embedders, an NPU `prefill_decode`, a tokenizer, and LlmMetadata that names no
model type and no media token. Its tokenizer gives every text, image and audio
token the same id as the CPU/GPU bundle's, and differs from id 258,884 on, where
the CPU/GPU one has `<|video|>`.

`litetune towers --add vision --add audio --metadata-from-donor`, with pull request
#68's code at its commit *Answer the full review of `litetune towers`* (2026-10-10;
f14542a before a rebase that left `towers.py` unchanged), with the CPU/GPU bundle
above as the donor, wrote the build's own sections, the donor's LlmMetadata (all
12,192 bytes) and its six tower sections, each byte for byte, in 24.7 s. Each added
adapter writes 1,536 values per token, as the build's embedder does; the 8 token
strings and ids the donor's metadata names as tokens are the same pieces at the same
ids in both tokenizers; and the report lists the fields the donor's metadata changes
as the builder's text form shows them, 21 of them: the channels, the Jinja prompt
template and the older `prompt_templates`, the `gemma4` model type and its fields,
the stop tokens. Without `--metadata-from-donor` it refused, because the build's own
metadata names no model type.

Run on 2026-10-10 on a Galaxy S26 (`Build.SOC_MODEL` SM8850) through Firebase Test
Lab, through LiteRT-LM v0.18.0's C API (the `libLiteRtLm.so` of flutter_gemma's
`native-v0.18.0-a` release), its `prefill_decode` graph on the NPU and the
embedders, the `aux` graphs and the towers on the CPU, greedy. The route is
flutter_gemma's: its `libLiteRtLm.so` and Qualcomm dispatch library, with QNN
2.51.0; the Maven `litertlm-android` AAR's `Backend.NPU` was not tried. The harness
on the phone records the model file's length, not its hash. Engine load below is
timed to the end of the first prompt's reply, as the harness times it:

| input | result | engine load and first reply |
|---|---|---|
| eight text prompts, engine opened without media backends | **8 of 8 identical** to the build's own, as received (2026-10-09) | 1.33 s |
| the red PNG, "What colour is this image? Answer in one word." | **"Red"** | 8.25 s, the first reply being this one, one engine with both media backends |
| the 440 Hz tone, "Describe this sound in one sentence." | **"The sound is a high-pitched, sustained tone that fades slowly."** | |
| "What is the capital of Portugal? Answer in one sentence." | "The capital of Portugal is Lisbon." | |
| the red PNG and the tone in one turn, "What colour is the image, and what does the sound sound like? Answer in one sentence." | "The image is red, and the sound is not audible." | 8.36 s |

The turn with both inputs is answered for the image and not the tone, as Google's
own bundles answer it through the same runtime and harness: Google's unmodified
CPU/GPU bundle on the same phone model, text and towers on the CPU, answered it word
for word the same, from the same 314 prefill tokens, and Google's own SM8750 NPU
bundle on a Galaxy S25 answered "The image is red, and there is no sound in the
image."

In each run's logcat the Qualcomm dispatch library is loaded, the QNN backend is
`Htp(2)`, `libQnnHtpV81Skel.so` is opened over FastRPC, four `Found qnn graph` lines
name the compiled graphs, and LiteRT-LM logs `Detected NPU prefill size` and runs
`LiteRT NPU Compiled Model`; the towers' XNNPACK caches appear only in the media
runs. LiteRT-LM logs no error. Every run also logs 292 error-priority lines that
stop nothing: QNN logs 280, memory handles it could not deregister, while the engine
is deleted after the last reply -- as Google's own SM8750 bundle does on its phone
-- and FastRPC logs 12 while it opens the DSP session, before the skel loads. The
same composition built on 2026-10-09 with litert-lm-builder 0.18.0 directly, and one
with the towers of Google's own SM8750 NPU bundle instead, gave the same three media
answers word for word. One image, one tone and eight text prompts on one phone: this
shows the grafted bundle loads and answers both modalities on the device it was
built for, and that its text answers did not change. It is not a measurement of
image or audio quality. It ran with Qualcomm's QNN runtime 2.51.0
(`com.qualcomm.qti:qnn-runtime` on Maven Central), with a logcat warning, `Qnn
backend library version 5.51.0 is used. The version LiteRT using is 5.50.0.`; with
the QAIRT 2.47.0 libraries the engine was not created on the same phone, though the
bundle's context binary is 2.47.0.

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

On litert-lm 0.16.1. The same bundles on 0.17.1 are [at the end of this
section](#qwen3-06b-on-litert-lm-0171).

**On 0.16.1 one cost is resolved and one is not**, where both of the second
family's were unresolved — and the resolved one rests on fewer disagreements,
not more. `weight_only_wi8_afp32` costs +0.0167 with an interval of ±0.0113 on
12 disagreements out of 600, eleven of which fall the same way;
`dynamic_wi8_afp32` costs +0.0067 inside ±0.0146 on 20, split twelve against
eight. That is the arithmetic the headline section describes, seen once more:
what buys an interval is the imbalance between the two directions, not the
count.

Across the two families the manifests support one observation and no ranking.
The fine-tuned Qwen3 scores higher on these rows than the fine-tuned Gemma 3
did, 0.6983 against 0.6717, and its conversions disagree with their float model
on 20 and 12 rows against Gemma 3's 34 and 16. litetune computes no interval for
a difference between two models, so that is not a resolved difference, and the
conversion costs are not ranked across families: both runs' references ran on a
different device from their candidates, so both carry that difference.

### What this run established that the table does not show

**The exporter needs no flag for the family.** Both recipes exported with no flag
from litetune, and the tuned checkpoint's `config.json` names `model_type: qwen3`.
That is the model-type trap `models.py` describes, seen from the other side: `qwen3`
is on the exporter's own list, so the config's `model_type` selects the right type
with no override. `models.py` records it as `qwen-3`, scoped then to 0.6B, the one size
run (it has also claimed 1.7B since its template was found to be 0.6B's byte for byte); from LiteRT-LM 0.18.0 that rule puts a block in front of the checkpoint's own
template, for the runtime rather than the exporter (*The 0.17.1 re-measurements again on
litert-lm 0.18.0*).

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
The converted candidates ran on CPU in this run. The same artifacts were later run on a
phone, on both its CPU and its GPU — see *What four bits cost*. No NPU figure
for this family.

### Limitations carried by these manifests

Four on `dynamic_wi8_afp32`, three on `weight_only_wi8_afp32`: the last is
absent there because that cost resolves on 0.16.1.

- The candidate ran on litert-lm's CPU backend; its GPU backend is a different executor
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

### Qwen3-0.6B on litert-lm 0.17.1

The same bundles and rows, verified again as [The banking77 sections on
litert-lm 0.17.1](#the-banking77-sections-on-litert-lm-0171) describes. The two
runs differ in runtime, transport and possibly CPU, so a difference between the
rows is not attributed to the runtime.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| 0.16.1 run | **0.6983** ±0.0367 | 0.6917 ±0.0370 | 0.6817 ±0.0373 |
| 0.17.1 run | **0.6983** ±0.0367 | 0.6983 ±0.0367 | 0.6933 ±0.0369 |
| Cost of conversion, 0.17.1 run | — | +0.0000 ±0.0103 *(unresolved, 10 discordant)* | +0.0050 ±0.0073 *(unresolved, 5 discordant)* |

The `weight_only_wi8_afp32` cost resolved on 0.16.1 and does not here: +0.0167
on 12 discordant rows, eleven one way, there; +0.0050 on 5, four one way, here.
That is two runs' verdicts, not a measured change in the cost, and neither run
is withdrawn. `dynamic_wi8_afp32` is unresolved in both, on 10 discordant rows
against 20.

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

On litert-lm 0.16.1. The same bundles on 0.17.1 are [at the end of this
section](#gemma-3-1b-on-litert-lm-0171).

The two costs fall the same way as the fourth family's and for the same
arithmetic: on 0.16.1 `weight_only_wi8_afp32` resolves on 14 disagreements while
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

### gemma-3-1b on litert-lm 0.17.1

The same bundles and rows, verified again as [The banking77 sections on
litert-lm 0.17.1](#the-banking77-sections-on-litert-lm-0171) describes. The two
runs differ in runtime, transport and possibly CPU, so a difference between the
rows is not attributed to the runtime.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| 0.16.1 run | **0.7533** ±0.0345 | 0.7450 ±0.0349 | 0.7367 ±0.0352 |
| 0.17.1 run | **0.7533** ±0.0345 | 0.7433 ±0.0350 | 0.7533 ±0.0345 |
| Cost of conversion, 0.17.1 run | — | +0.0100 ±0.0113 *(unresolved, 12 discordant)* | +0.0000 ±0.0065 *(unresolved, 4 discordant)* |

The `weight_only_wi8_afp32` cost resolved on 0.16.1 and does not here: +0.0167
on 14 discordant rows, twelve one way, there; +0.0000 on 4, two each way, here.
That is two runs' verdicts, not a measured change in the cost, and neither run
is withdrawn. `dynamic_wi8_afp32` is unresolved in both, on 12 discordant rows
against 23.

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

The float reference in every Qwen2.5 run here, this section's and the
re-measurements on 0.17.1 and 0.18.0, decoded with `repetition_penalty` 1.1:
litetune up to 0.4.1 left the checkpoint's `generation_config` in force, and
transformers merges it into a greedy `generate` call. litetune 0.4.2
decodes the reference greedily without it. On 2026-10-12, on litert-lm 0.18.0,
the `dynamic_wi8_afp32` bundle verified against the same tuned checkpoint with
the reference of 0.4.1 and with that of 0.4.2 gave the same figures both
times: candidate 0.7700, reference 0.7700, cost 0.0000 on 8 discordant rows,
which are the 0.18.0 figures published above. The candidate never decoded with a
penalty: LiteRT-LM applies none unless asked, and litetune does not ask.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base model | **0.0000** | 0.0000 | — |
| Fine-tuned | **0.7700** ±0.0337 | 0.7600 ±0.0342 | 0.7600 ±0.0342 |
| Cost of conversion | — | +0.0100 ±0.0131 *(unresolved, 16 discordant)* | +0.0100 ±0.0103 *(unresolved, 10 discordant)* |

On litert-lm 0.16.1. The same bundles on 0.17.1 are [at the end of this
section](#qwen25-05b-on-litert-lm-0171).

Sizes are 647,312,304 and 647,492,736 bytes.

### What this run established that the table does not show

**Neither eight-bit cost resolves here, and on the same runtime one did on each
of the two models before it.** On litert-lm 0.16.1 Qwen3-0.6B and gemma-3-1b
both had `weight_only_wi8_afp32` resolve at +0.0167 while `dynamic_wi8_afp32`
did not; re-measured on 0.17.1, neither resolves. This model puts both inside
their intervals. The reason is in the discordant column rather than in the
accuracy: the candidate was discordant with the reference on 16 rows under one
recipe and 10 under the other, and a paired test on 600 examples cannot separate
a difference that small from none. That is a statement about this sample, not a
finding that the conversion is free.

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
fields are unavailable. Of the five models fine-tuned and measured on banking77 up to
this run, not one has a recorded training gain — two scored zero on both sides, one was refused for
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

### Qwen2.5-0.5B on litert-lm 0.17.1

The same bundles and rows, verified again as [The banking77 sections on
litert-lm 0.17.1](#the-banking77-sections-on-litert-lm-0171) describes. The two
runs differ in runtime, transport and possibly CPU, so a difference between the
rows is not attributed to the runtime.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| 0.16.1 run | **0.7700** ±0.0337 | 0.7600 ±0.0342 | 0.7600 ±0.0342 |
| 0.17.1 run | **0.7700** ±0.0337 | 0.7700 ±0.0337 | 0.7717 ±0.0336 |
| Cost of conversion, 0.17.1 run | — | +0.0000 ±0.0092 *(unresolved, 8 discordant)* | −0.0017 ±0.0057 *(unresolved, 3 discordant)* |

Neither cost resolves in either run; the discordant rows are 8 and 3, against
16 and 10.

## The two Qwen families, one size up

`Qwen/Qwen3-1.7B` @ `70d244cc`, 1,720,574,976 parameters, and
`Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798`, 1,543,714,304 parameters, run on
2026-10-19 with the code that first claims them in the `qwen-3` and `qwen-2.5`
rules, before it was merged: commit 1e42ebb, which reports 0.5.1. Each ships its smaller size's chat template
byte for byte -- `a55ee1b1…` and `cd8e9439…`, the digests `convert` read back from
the checkpoints -- and declares the same `model_type` and architecture, which is
what the rules were widened on. This run is what they then did.

The task, the rows and the scorer are the fifth family's: LoRA α32, rank 16, lr
2e-4, one epoch, batch 8, bfloat16, `max_seq_length` 256, the same 2,400 rows of
`mteb/banking77` and the same 600 held-out rows of split `0c7505b2b6f69ab1`,
`--scorer exact-text`, prompt mode `runtime_rendered`, which `tune` inferred and
`verify` read back from beside the checkpoint. On all four verifies the runtime and
the reference rendered identical token ids on all 600 prompts, with the runtime's
prefill count equal to the reference's on the 8 it was sent. Trained parameters
17,432,576 (Qwen3) and 18,464,768 (Qwen2.5), final loss 0.6226 and 0.3756.
Training and the float reference ran on one A100-SXM4-40GB, every candidate on
litert-lm 0.18.0's CPU backend on `Intel(R) Xeon(R) CPU @ 2.20GHz`, 12 vCPU.
Training environment `torch==2.5.1`, `transformers==5.16.1`, `peft==0.20.0`;
export environment `litert-torch-nightly` 0.10.0.dev20260926,
`ai-edge-quantizer-nightly` 0.10.0.dev20260926, litert-lm-builder 0.18.0.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Qwen3-1.7B, fine-tuned | **0.7400** ±0.0351 | 0.7383 ±0.0352 | 0.7383 ±0.0352 |
| Cost of conversion | — | +0.0017 ±0.0098 *(unresolved, 9 discordant)* | +0.0017 ±0.0086 *(unresolved, 7 discordant)* |
| Qwen2.5-1.5B-Instruct, fine-tuned | **0.7683** ±0.0338 | 0.7717 ±0.0336 | 0.7733 ±0.0335 |
| Cost of conversion | — | −0.0033 ±0.0113 *(unresolved, 12 discordant)* | −0.0050 ±0.0073 *(unresolved, 5 discordant)* |

On litert-lm 0.18.0. Sizes are 2,056,778,672 and 2,057,090,176 bytes for Qwen3,
1,802,859,440 and 1,803,072,640 for Qwen2.5. The untuned base was not scored.

**The template each rule chose is the one it chooses for the smaller size.** For
Qwen3-1.7B `convert` packed LiteRT-LM v0.18.0's canonical Qwen 3 template
(`chat_template.source` `canonical`), offered because `tune`'s record says the
checkpoint was trained `runtime_rendered` without declarations, and chosen after
its 96 single-turn renderings matched the checkpoint's own. For
Qwen2.5-1.5B-Instruct it composed the checkpoint's own template behind the
text-parts block (`composed`). Every bundle passed the conversation check after
export (72 cases, each a prefix of the next), and the bundles read back as
`llm_model_type` `qwen3` and `qwen2p5`, with no type override.

**Both hold a LiteRT-LM conversation.** On litert-lm-api 0.18.0's CPU backend each
`dynamic_wi8_afp32` bundle answered three messages in one `Conversation` and two
held-out prompts in another, every send returning a reply -- the second message is
where a Qwen3 bundle failed before #78.

**Neither model is shown to be better or worse than its smaller size.** Qwen3-1.7B's
float 0.7400 and Qwen2.5-1.5B-Instruct's 0.7683 sit beside Qwen3-0.6B's 0.6983 and
Qwen2.5-0.5B-Instruct's 0.7700 on these rows, and litetune computes no interval for a
difference between two models. Neither run scored its untuned base, so neither records
a training gain, as none of the five banking77 models before them does. No conversion cost resolves, as for the smaller sizes
on 0.17.1 and 0.18.0.

## Three more families: Granite 4.1, MiniCPM5 and Qwen 3.5

`ibm-granite/granite-4.1-3b` @ `c0650403`, 3,402,836,480 parameters,
`openbmb/MiniCPM5-1B` @ `87179e5c`, 1,080,632,832, `Qwen/Qwen3.5-0.8B` @
`2fc06364`, 752,393,024, and `Qwen/Qwen3.5-2B` @ `15852e8c`, 1,881,825,088, run
on 2026-10-20 with the code that first claims them in the `granite-4.1`,
`minicpm5` and `qwen-3.5` rules, before it was merged. Qwen3.5-0.8B and granite
ran every stage at 382925e; MiniCPM5 and Qwen3.5-2B at 1d6dee8, which adds the
start-token check below. Both report 0.5.1. From 1d6dee8 to the branch's head
only the decision path changed, and `rendering.py`'s runtime-ids script moved
into a shared function with the same body.

The task, the rows, the recipe and the environments are the previous section's:
the same 2,400 training and 600 held-out rows of split `0c7505b2b6f69ab1`,
`--scorer exact-text`, prompt mode `runtime_rendered`. On all eight verifies the
runtime and the reference rendered identical token ids on all 600 prompts.
Trained parameters 31,129,600 (granite), 11,206,656 (MiniCPM5), 6,389,760 and
10,911,744 (Qwen3.5); final loss 0.3506, 0.4297, 0.4143 and 0.3502.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| granite-4.1-3b, fine-tuned | **0.7800** ±0.0331 | 0.7883 ±0.0327 | 0.7783 ±0.0332 |
| Cost of conversion | — | −0.0083 ±0.0135 *(unresolved, 17 discordant)* | +0.0017 ±0.0057 *(unresolved, 3 discordant)* |
| MiniCPM5-1B, fine-tuned | **0.7167** ±0.0361 | 0.7067 ±0.0364 | 0.7167 ±0.0361 |
| Cost of conversion | — | +0.0100 ±0.0122 *(unresolved, 14 discordant)* | +0.0000 ±0.0092 *(unresolved, 8 discordant)* |
| Qwen3.5-0.8B, fine-tuned | **0.7800** ±0.0331 | 0.7767 ±0.0333 | 0.7917 ±0.0325 |
| Cost of conversion | — | +0.0033 ±0.0113 *(unresolved, 12 discordant)* | **−0.0117** ±0.0098 *(resolved, 9 discordant)* |
| Qwen3.5-2B, fine-tuned | **0.8000** ±0.0320 | 0.7833 ±0.0330 | 0.7967 ±0.0322 |
| Cost of conversion | — | **+0.0167** ±0.0122 *(resolved, 14 discordant)* | +0.0033 ±0.0046 *(unresolved, 2 discordant)* |

On litert-lm 0.18.0. Sizes are 3,693,186,992 and 3,693,645,952 bytes for
granite, 1,098,211,248 and 1,098,440,832 for MiniCPM5, 1,036,844,976 and
1,037,303,936 for Qwen3.5-0.8B, 2,424,750,064 and 2,425,209,040 for Qwen3.5-2B.
The untuned bases were not scored.

**Two costs resolve, in opposite directions.** Qwen3.5-0.8B's `weight_only`
file scored above its float reference, on 9 discordant rows; Qwen3.5-2B's
`dynamic` file scored below its own, on 14. Each is one run's paired interval,
and neither says which way the next checkpoint will go.

**Each rule's template holds a conversation.** granite composes the checkpoint's
own template behind the text-parts block (`composed`). MiniCPM5 and both Qwen 3.5
sizes pack LiteRT-LM v0.18.0's canonical template with litetune's edits
(`litert-lm-v0.18.0-minicpm5-litetune.jinja`,
`litert-lm-v0.18.0-qwen3_5-litetune.jinja`), each chosen after its single-turn
renderings matched the checkpoint's. Every bundle passed the conversation check
after export, and on litert-lm-api 0.18.0's CPU backend each `dynamic_wi8_afp32`
bundle answered three messages in one `Conversation` and two held-out prompts in
another.

**MiniCPM5's first convert refused, and was right to.** The checkpoint's
template prints `<s>`; the runtime hands a template an empty `bos_token` and
puts the bundle's start token in front of what it renders. At 382925e `convert`
compared single turns without that start token and could not check them. The
litetune template now prints `{{- bos_token -}}`, and the single-turn check
predicts the start token the exporter will write, renders with it in front, and
reads it back from the bundle afterwards (`<s>` for MiniCPM5).

**A shared Engine carries Qwen 3.5's state from one prompt into the next**
(LiteRT-LM#3165). The first 50 held-out prompts through `verify`'s own driver
script on the `dynamic` Qwen3.5-0.8B file, against a process per prompt, which
shares nothing:

| | Qwen3.5-0.8B, prompts answered otherwise | Qwen3-0.6B, the control |
|---|---|---|
| one Engine for every prompt | 22 of 50 (2,342 s) | 0 of 50 (69 s) |
| the same, prompts in reverse order | 29 of 50 (2,881 s) | 0 of 50 (52 s) |
| an Engine per prompt | **0 of 50** (88 s) | 0 of 50 (81 s) |

The `qwen-3.5` rule therefore has `verify` build an Engine for every prompt
(0.79-0.92 s each on this CPU), and the eight Qwen 3.5 numbers above were taken
that way, which is also the faster of the two here. This measures `verify`'s
driver, not an app.

**LoRA reaches a quarter of Qwen 3.5's attention.** Six of its 24 layers are
`full_attention` and carry `self_attn.{q,k,v,o}_proj`; the other 18 are
`linear_attention`, whose projections are not among the seven names `tune`
targets. The adapter covers every layer's MLP and those six layers' attention:
96 modules on both sizes.

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
| `dynamic_wi4_afp32` | 680,203,568 | *refused past the gate* | — | — |
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

The tables and the paragraphs below, down to *On a phone*, are litert-lm
0.16.1's. The same bundles on 0.17.1 are [at the end of this
section](#four-bits-on-litert-lm-0171).

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
twin's 0.7700, a cost of **+0.2567** that resolves on 178 discordant rows. That
is the first channelwise figure in this file, and it is a third of the model's
accuracy — the recipe is not rehabilitated by having finally produced a number.
`weight_only_wi4_afp32` got further here than on any earlier model and still
produced nothing: its gate opened, 599 of 600 generations ran, one exceeded the
300 s per-prompt limit, and `verify` returned `failed_harness` and exit 4 rather
than scoring the rows that did finish. On 0.16.1, across four models, the same
two recipes produced a refusal at the gate, a refusal past the gate, a run that
generated 599 of 600 and was refused for the one it did not, and one resolved
cost — which is four ways of learning that what these recipes do is a property
of the checkpoint they are given.

**On 0.16.1, dequantising before compute rescued nothing, on any of the four.**
`weight_only_wi4_afp32` does exactly that, and it is the one recipe in this file
that never produced a score on 0.16.1 (on 0.17.1 it did, on Qwen2.5-0.5B,
below): it failed the gate on gemma-3-270m, Qwen3-0.6B and gemma-3-1b, and on
Qwen2.5-0.5B it cleared the gate and then lost a single generation to the
timeout. `dynamic_wi4_afp32`, which keeps activations in integers, is the one
that eventually answered. So integer activations are not what separates a
working bundle from a broken one here — if anything the evidence runs the other
way, and on three of the four checkpoints neither recipe produced an answer at
all. A five-prompt diagnostic through both int8 and both channelwise bundles of
gemma-3-270m and Qwen3-0.6B — the two block-wise recipes were never in it — says
the same from the other side: at 8 bits both models emit the reference's label
and stop, the terminator scoring at or above −0.001 in log-probability; at 4
bits channelwise, Qwen3 matches one of five labels exactly and on another emits
the right label and then does not stop — its terminator scores −0.64 to −4.17 —
while gemma produces no label at all, scoring the reference answer at −32 to
−107 against −0.03 to −0.47 at 8 bits. Those last figures are the diagnostic's
`score_total` with the terminator token's own log-probability taken off, which
is why they are not the raw totals in `*__diag-int4.json`.

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
identically (+0.0883 either way), all on 0.16.1, and none of those differences
is one this design can settle. Nor does it say anything about models larger than
these four, or about 4-bit weights produced some other way, such as by
quantization-aware training. One run each on 0.16.1; the 0.17.1 run below is not
paired with it.

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

### Four bits on litert-lm 0.17.1

The same sixteen bundles, verified again from 2026-09-28 to 2026-09-29 with
litetune at commit 0744a1e on litert-lm 0.17.1, behind the same five-prompt gate
with its 900 s budget, and otherwise as [The banking77 sections on litert-lm
0.17.1](#the-banking77-sections-on-litert-lm-0171) describes for the eight-bit
pairs. Thirteen of the sixteen wrote a manifest, and each SHA-256 there equals
the one `convert` recorded when 0.1.6, 0.1.7 or 0.1.8 built the bundle; the
three whose gate ran out of time wrote none. Every manifest records the
candidate on `Intel(R) Xeon(R) CPU @ 2.20GHz` with AVX-512 and 12 vCPU, and
every one that reached the reference records it on cuda. The two runs differ in runtime, transport and possibly CPU,
so a difference between them is not attributed to the runtime. The transport
matters here in one more way: on 0.16.1 litetune cut each prompt at 300 s, and
since #50 a split shares 300 s per prompt, so one slow prompt no longer refuses
a run by itself; it can still use up the rest of the split's budget.

| | 0.16.1 | 0.17.1: exact match | 0.17.1: cost of conversion |
|---|---|---|---|
| Qwen3-0.6B, `dynamic_wi4_afp32` | refused at the gate | *refused at the gate*: repeated itself on 2 of 5 | — |
| Qwen3-0.6B, `weight_only_wi4_afp32` | refused at the gate | *refused at the gate*: five prompts not done in 900 s | — |
| Qwen3-0.6B, `dynamic_wi4b32_afp32` | **+0.0350** (57) | 0.6700 ±0.0376 | **+0.0283** ±0.0242 *(resolved, 55 discordant)* |
| Qwen3-0.6B, `dynamic_wi4b32_emb8_afp32` | **+0.0550** (67) | 0.6483 ±0.0382 | **+0.0500** ±0.0244 *(resolved, 56 discordant)* |
| gemma-3-270m, `dynamic_wi4_afp32` | refused at the gate | *refused at the gate*: repeated itself on 5 of 5 | — |
| gemma-3-270m, `weight_only_wi4_afp32` | refused at the gate | *refused at the gate*: five prompts not done in 900 s | — |
| gemma-3-270m, `dynamic_wi4b32_afp32` | **+0.3483** (237) | 0.3067 ±0.0369 | **+0.3650** ±0.0518 *(resolved, 251 discordant)* |
| gemma-3-270m, `dynamic_wi4b32_emb8_afp32` | **+0.3200** (218) | 0.2650 ±0.0353 | **+0.4067** ±0.0529 *(resolved, 262 discordant)* |
| gemma-3-1b, `dynamic_wi4_afp32` | refused past the gate | *refused past the gate*: repeated itself on 65 of 600 | — |
| gemma-3-1b, `weight_only_wi4_afp32` | refused at the gate | *refused at the gate*: five prompts not done in 900 s | — |
| gemma-3-1b, `dynamic_wi4b32_afp32` | **+0.0883** (77) | 0.6700 ±0.0376 | **+0.0833** ±0.0296 *(resolved, 82 discordant)* |
| gemma-3-1b, `dynamic_wi4b32_emb8_afp32` | **+0.0883** (79) | 0.6750 ±0.0375 | **+0.0783** ±0.0283 *(resolved, 75 discordant)* |
| Qwen2.5-0.5B, `dynamic_wi4_afp32` | **+0.2567** (178) | 0.5017 ±0.0400 | **+0.2683** ±0.0451 *(resolved, 191 discordant)* |
| Qwen2.5-0.5B, `weight_only_wi4_afp32` | one generation never finished | 0.5200 ±0.0400 | **+0.2500** ±0.0431 *(resolved, 174 discordant)* |
| Qwen2.5-0.5B, `dynamic_wi4b32_afp32` | **+0.0767** (62) | 0.6933 ±0.0369 | **+0.0767** ±0.0253 *(resolved, 60 discordant)* |
| Qwen2.5-0.5B, `dynamic_wi4b32_emb8_afp32` | **+0.0917** (69) | 0.6867 ±0.0371 | **+0.0833** ±0.0257 *(resolved, 62 discordant)* |

The 0.16.1 column is each cost with its discordant rows, from the tables above;
every one of them resolved. The float references are the same to four places as
there.

Every block-wise cost resolves again, and every recipe that refused on 0.16.1
refuses again but one: **`weight_only_wi4_afp32` on Qwen2.5-0.5B finished all
600 generations and costs +0.2500**, a third of the model's accuracy, beside
`dynamic_wi4_afp32`'s +0.2683, each paired against the float twin and not
against the other. On 0.16.1 it was refused because one prompt exceeded the 300
s per-prompt limit, which the 0.17.1 transport no longer enforces per prompt,
only across the split, so this is not the runtime finishing what 0.16.1 could
not; whether any of the 600 took longer than 300 s is not recorded.

The refusals otherwise read as follows. Qwen3's and gemma-3-1b's
`weight_only_wi4_afp32` gates, which on 0.16.1 ran four of five prompts and cut
one at 300 s, wrote nothing within the gate's 900 s budget, as gemma-3-270m's
did in both runs. gemma-3-270m's `dynamic_wi4_afp32` was refused for repetition
on all five gate prompts, worst ratio 0.9945, where on 0.16.1 it was refused for
a leaked `<bos>`; liveness stops at the first check that fails and the leak
check runs first, so whether it also repeated on 0.16.1 is not known. Qwen3's
repeated itself on two of five, worst 0.9829, against one on 0.16.1, and
gemma-3-1b's on 65 of 600 in both runs.

On gemma-3-270m the two block-wise costs reversed order: with int8 embeddings
+0.3200 against +0.3483 on 0.16.1, and +0.4067 against +0.3650 here. Within
each run both were paired against the float twin and never against each other,
so neither run orders them, and which recipe costs the 270M less is not
established. The int8-embeddings bundle scored 0.2650 here against 0.3517 on
0.16.1, 52 rows fewer against an unchanged float score; the rows were not kept,
so that is not a paired test, and it is not attributed.

## The same model through the runtime's tool path

`functiongemma-270m-it` LoRA-tuned on the 6434 single-call rows of
`google/mobile-actions` (5794 trained, 640 held out), in `runtime_rendered`
with the dataset's seven tool declarations, converted `dynamic_wi8_afp32`, and
measured the way an application calls it: `create_conversation(tools=...)`, the
runtime rendering the declarations and parsing the call itself. The candidate
on litert-lm's CPU backend, the float reference on cuda -- as every run of this
section's manifests records. One run, 2026-09-17; a second, every stage from `prepare` on with the code
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

### The same checkpoint on litert-lm 0.17.1

The runtime flutter_gemma ships from `flutter_gemma_litertlm` 1.8.0, measured
2026-09-27 by litetune at commit 9af0a90 (unreleased; it reports itself as
0.1.9) with the checkpoint, the held-out split and the bundle the
2026-09-19 run produced, so nothing was retrained: one A100-SXM4-40GB with 12
vCPU as then, reference on cuda as then. The 0.16.1 column is that run's. The
CPU the 0.17.1 runs computed on is recorded -- `Intel(R) Xeon(R) CPU @ 2.20GHz`,
with AVX-512 -- and the 0.16.1 run's is not, so the two columns are not shown
to share a CPU and, by the rule above, compare only loosely.

| | 0.16.1 | 0.17.1, the same bundle | 0.17.1, re-converted |
|---|---|---|---|
| Float reference | 0.9250 | 0.9250 | 0.9250 |
| Tool path, grammar off | 0.9125 ±0.0219 | 0.9109 ±0.0221 | 0.9109 ±0.0221 |
| Tool path, grammar on | 0.9125 ±0.0219 | 0.9109 ±0.0221 | 0.9109 ±0.0221 |
| Cost of conversion | **+0.0125** ±0.0087 *(8 discordant)* | **+0.0141** ±0.0092 *(9 discordant)* | **+0.0141** ±0.0092 *(9 discordant)* |
| Effect of the grammar | 0.0000 | 0.0000 | 0.0000 |

583 of 640 right against 584, and the conversion cost still resolves. How much
of that one-row net difference is the runtime and how much the CPU is not
established, since the 0.16.1 run did not record its CPU. The
re-converted bundle -- `litert-torch-nightly==0.10.0.dev20260926` with its
quantizer, LiteRT and converter pinned -- is the same size as the old one with a
different SHA-256, and gave the same score in both modes, with the same number
of rows discordant against the reference; the manifests keep scores, not rows,
so whether they are the same rows is not recorded. The tool name was right on
all 640 in every run; the rendering check matched all 640.

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
- Flutter Edge AI passes FunctionGemma `.litertlm` declarations to LiteRT-LM
  as structured data and reads the runtime's parsed tool calls, so this is the
  same native tool path it uses for this model.
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

## Decisions: a model that chooses among options

`google/gemma-3-270m-it` at `ac82b4e8`, trained to answer a question about a
text with one of a fixed list of options and scored the way flutter_edge_ai's
proposed `createDecider` is to use it: every option's log-probability through
the runtime, with no generation (the README's *Decisions* section). Four
tasks, each one question over its own options. Every row of each dataset was
used, split by `prepare` (seed 0) into the sizes of the dataset's own splits:

| task | question | dataset | train / calibration / held-out |
|---|---|---|---|
| emotion | `choice`, six options | `dair-ai/emotion`, config `split`, at `cab853a1` | 16,000 / 2,000 / 2,000 |
| irony | `noul` | `cardiffnlp/tweet_eval`, config `irony`, at `b3a375ba` | 2,862 / 955 / 784 |
| SST-2 | `noul` | `SetFit/sst5` at `e51bdcd8`, without its neutral rows | 6,920 / 872 / 1,821 |
| SST-5 | `score`, five levels | `SetFit/sst5` at `e51bdcd8` | 8,544 / 1,101 / 2,210 |

SST-2 is SST-5 without its 2,242 neutral rows: `very negative` and `negative`
answer `no`, `positive` and `very positive` answer `yes`, and the question is
"Is the text positive?".

Each ran `tune --method lora --epochs 2 --prompt-mode runtime_rendered
--max-seq-length 2048 --micro-batch-size 4` (LoRA rank 16 on the seven
projections, bfloat16), `convert --recipe dynamic_wi8_afp32 --recipe none`,
which added `--prefill_lengths=8,128`, `verify --engine-rows 300 --threads 4`
with the `none` file as `--float-twin`, and `bundle`. Emotion was prepared,
tuned and its eight-bit file converted on #82's branch at `d0833f1`; its float
twin was converted, and it was verified and bundled, at `b49aaab`, the commit
that merged #82. Irony and SST-5 ran every stage at `b49aaab`. SST-2 ran every
stage at `46199f4`, litetune 0.5.0, which also converted, verified and bundled
SST-5's checkpoint again: every score `verify` took, on all three sides,
equalled `b49aaab`'s bit for bit.

The candidate is the eight-bit file on litert-lm 0.18.0's CPU backend, on four
threads of an Intel Xeon @ 2.20GHz, scoring 300 held-out rows spread through
the split. The reference is the float checkpoint in PyTorch on an NVIDIA A100,
scoring every held-out row. Agreement below is therefore between a runtime on a
CPU and a reference on a GPU.

| | emotion | irony | SST-2 | SST-5 |
|---|---|---|---|---|
| verify | passed | **failed_gate** | passed | passed |
| reference accuracy, every held-out row | 0.9335 | 0.6901 | 0.9055 | 0.5330 |
| runtime's top option is the reference's, of 300 rows | **1.0000** | **0.9367** | **0.9767** | **0.9567** |
| runtime's answer is the reference's, abstentions included | 0.9933 | 0.9367 | 0.9767 | 0.9567 |
| runtime accuracy on its 300 rows (reference on the same rows) | 0.9533 (0.9533) | 0.6767 (0.7000) | 0.8867 (0.8700) | 0.5267 (0.5167) |
| ms per decision, median | 370.1 | 234.8 | 235.4 | 394.9 |
| float twin against the reference, largest per-token difference on 4 rows | 2.96e-05 nats | 3.30e-04 nats | 6.25e-05 nats | 2.97e-05 nats |

- `verify` refused irony's eight-bit bundle: 0.9367 is under its 0.95 floor.
  The 19 rows where the runtime chose otherwise were near-ties for the
  reference: its best option led its second by at most 0.637 nats on them
  (median 0.162), against a median of 1.426 on the 281 rows where the two
  agreed. On those 300 rows the runtime's accuracy differed from the
  reference's by −0.0233 ± 0.0285, which does not resolve a difference.
  Trained for longer, irony passed (below).
- SST-2, the other `noul` task, passed, and its reference was further from ties:
  its best option led its second by less than 0.5 nats on 7 of the 300 rows,
  where irony's did on 60, and by a median of 4.911 nats on the 293 rows where
  the two agreed. Six of the 7 rows where the runtime chose otherwise had
  margins of 0.011 to 0.866 nats. The seventh did not: the reference led by
  2.248 nats, and the runtime moved one option's score by 1.674. On the 300 rows
  the runtime's accuracy differed from the reference's by +0.0167 ± 0.0173,
  which does not resolve a difference either.
- Emotion's threshold abstains: the reference answered 0.9805 of the held-out
  rows, and was wrong on 0.0571 of those it answered. A `noul` or `score`
  question never abstains.
- `bundle` wrote emotion's `litetune.decision` section. For irony, `b49aaab`
  bundled the file without one, as it should, but reported `failed_harness`
  rather than verify's `failed_gate`; #83 carries verify's status.

### Irony trained for longer

Irony was trained again for four and for six epochs, from the same split and
with every other flag as above, and each checkpoint was converted, verified
and bundled at `46199f4`. `tune.py` and `decision_scripts.py` are the same at
`b49aaab` and `46199f4`, and both runs' first two epochs had the two-epoch
run's mean training losses, bit for bit. Both bundles passed, on the same 300
rows:

| irony as `noul` | 2 epochs | 4 epochs | 6 epochs |
|---|---|---|---|
| verify | **failed_gate** | passed | passed |
| runtime's top option is the reference's, of 300 rows | 0.9367 | 0.9667 | 0.9500 |
| reference's best option led its second by under 0.5 nats, of the 300 rows | 60 | 17 | 19 |
| reference accuracy, 784 held-out rows | 0.6901 | 0.6671 | 0.7117 |
| reference answered `yes`, of the 784 rows (the labels: 0.4949) | 0.2334 | 0.7666 | 0.4592 |

- The six-epoch bundle passed on the floor: the runtime agreed with the
  reference on 285 of the 300 rows, and one row fewer would have failed the
  gate.
- Neither longer run's reference accuracy differed from the two-epoch
  reference's by a resolved amount: −0.0230 ± 0.0511 at four epochs and
  +0.0217 ± 0.0365 at six, paired on the same 784 rows. The answers moved more
  than the accuracy did: the four-epoch reference answered otherwise than the
  two-epoch one on 418 of the rows, and was right on 200 of them.

### Every family on emotion

The emotion task again, from the same split, on one or two models of every
family litetune has a rule for, each from its Hub revision through the same
five stages with the flags above: prepare, tune, convert, verify with the
`none` file as `--float-twin`, bundle. Every family but Gemma 3 text needs that
twin to be verified. Each model ran at `d686851` but Gemma 4, whose reference
this run showed to be wrong and which ran again at `eab4cf0` (below); the two
commits differ in nothing else a decision executes. The candidate and the
reference are as above: the eight-bit file on four threads of the same CPU
model, the float checkpoint on an A100.

| model @ revision | verify | reference accuracy, 2,000 rows | runtime accuracy, 300 rows | runtime's top option is the reference's | runtime's answer is the reference's | mean KL | ms per decision, median | float twin, largest per-token difference |
|---|---|---|---|---|---|---|---|---|
| gemma-3-270m-it @ ac82b4e8 | passed | 0.9335 | 0.9533 | 1.0000 | 0.9933 | 0.0007 | 415.8 | 2.96e-05 |
| gemma-3-1b-it @ dcc83ea8 | **failed_gate** | 0.9305 | 0.9467 | 0.9933 | 0.9900 | 0.0005 | 935.6 | 7.63e-05 |
| functiongemma-270m-it @ 39eccb09 | passed | 0.9305 | 0.9567 | 0.9933 | 0.9833 | 0.0014 | 371.8 | 1.79e-04 |
| gemma-4-E2B-it @ 3e22461f | passed | 0.9405 | 0.9600 | 0.9967 | 0.9933 | 0.0005 | 1069.8 | 5.50e-05 |
| Qwen3-0.6B @ c1899de2 | passed | 0.9380 | 0.9467 | 0.9900 | 0.9833 | 0.0013 | 1497.8 | 3.80e-04 |
| Qwen3-1.7B @ 70d244cc | passed | 0.9395 | 0.9567 | 0.9967 | 0.9900 | 0.0006 | 2084.6 | 1.05e-04 |
| Qwen2.5-0.5B-Instruct @ 7ae55760 | passed | 0.9330 | 0.9467 | 1.0000 | 0.9933 | 0.0004 | 550.5 | 5.96e-05 |
| Qwen2.5-1.5B-Instruct @ 989aa798 | passed | 0.9335 | 0.9533 | 0.9900 | 0.9867 | 0.0007 | 1332.8 | 2.43e-04 |
| MiniCPM5-1B @ 87179e5c | passed | 0.9355 | 0.9600 | 1.0000 | 0.9867 | 0.0006 | 921.5 | 1.23e-04 |
| granite-4.1-3b @ c0650403 | passed | 0.9325 | 0.9567 | 0.9967 | 0.9933 | 0.0003 | 2836.4 | 7.35e-05 |
| Qwen3.5-0.8B @ 2fc06364 | passed | 0.9295 | 0.9433 | 0.9967 | 0.9867 | 0.0005 | 39766.0 | 2.65e-02 |

- Every family but Qwen 3.5 had its options scored by rewinding one session,
  the method measured on Gemma 3, and every one of their twins was within
  3.80e-04 nats of the reference, under the 1e-3 the check allows.
- gemma-3-1b was refused by the abstain rule, not by the runtime: the
  reference was wrong on 0.0629 of the 1,971 held-out rows it answered, a 95 %
  lower bound of 0.0541 against the 0.05 the threshold was fitted to keep. The
  runtime agreed with it as closely as every other model's did.
- Gemma 4's forward soft-caps lm_head's output at 30, with the
  `final_logit_softcapping` of its text config; its exported bundle does not.
  A reference that applied the cap differed from the float twin by up to
  3.13e+01 nats per token and was refused; without it, by 5.50e-05. The
  reference follows the bundle. Greedy text is the same either way, since the
  cap keeps the order of the logits; sampled text is not.
- The seven other models a first run measured at an earlier commit, `e59d7bc`,
  gave the same status, accuracies, agreements, mean KL and twin difference
  again at `d686851`, field for field. ms per decision moved by up to 15 %
  (gemma-3-1b: 816.5, then 935.6).
- Qwen3.5-0.8B's row is its verify at `39ba2c2`, which holds this family's
  twin to 5e-02 nats per token and says so in the manifest. At `d686851`,
  under the 1e-3 every other family is held to, the same checkpoint was
  refused by the twin alone, with the same accuracies, agreements and mean KL. On
  the twin's rows PyTorch in float32 was within 2.7e-05 of a float64
  reference and the bundle 2.65e-02 from it, so the difference is the exported
  bundle's own arithmetic. Each option is scored on an
  Engine of its own (`fresh_engine`, LiteRT-LM#3165): this verify loaded an
  Engine 1,813 times, and a decision took a median of 39,766.0 ms.

### The three passing bundles on a phone

The bundles verify passed for emotion, SST-2 and SST-5 above, the same files,
scored on a Galaxy S24 (`SC-51E`, SM8650, Android 16) through Firebase Test
Lab, on its CPU (four threads) and on its GPU. The phone ran verify's runtime
scoring, ported step for step to LiteRT-LM 0.18.0's C API (`libLiteRtLm.so`
from flutter_gemma's native-v0.18.0-a build), on the same 300 held-out rows
and the same prompt ids, and its scores were compared with the server
runtime's and the reference's with litetune 0.5.1's own functions. The port is
not part of litetune, and a decider in an app may score otherwise. The server
column is the verify that bundled each file; SST-5's is litetune 0.5.0's, whose
median of 390.9 ms differs from the 394.9 of `b49aaab`'s run above with the
same scores.

| | emotion | SST-2 | SST-5 |
|---|---|---|---|
| runtime's top option is the reference's: server / phone CPU / phone GPU | 1.0000 / 1.0000 / 0.9967 | 0.9767 / 0.9867 / 0.9967 | 0.9567 / **0.9433** / **0.9433** |
| phone's top option is the server's, of 300: CPU / GPU | 300 / 299 | 295 / 294 | 286 / 284 |
| ms per decision, median: server / CPU / GPU | 370.1 / 437.3 / 697.2 | 235.4 / 291.0 / 405.3 | 390.9 / 499.7 / 356.4 |
| engine init: CPU / GPU | 0.831 s / 5.171 s | 0.957 s / 4.451 s | 1.024 s / 4.117 s |
| peak RSS of the test process: CPU / GPU | 1,135,532 / 680,180 kB | 1,146,528 / 676,756 kB | 1,141,700 / 681,624 kB |

- No phone option score equalled the server's but one, and none of the
  paired accuracy differences, phone against server or against the reference,
  resolved: every 95 % interval contains 0. The self-check picked `once` on
  both backends for every bundle, as on the server, and a repeat session gave
  bit-identical scores on all 300 rows.
- SST-5 is under verify's 0.95 floor on both backends, and its disagreements
  are near-ties. Its reference's best level led its second by under 0.2 nats
  on 54 of the 300 rows, where emotion's did on 2 and SST-2's on 5. Every row
  on which the server or the phone's CPU chose otherwise than the reference
  was among those 54 (13 and 17 rows), and 14 of the GPU's 17; its other three
  had margins of 0.2813, 0.3059 and 0.6728 nats. On the 246 rows with a margin
  of at least 0.2 nats the server and the phone's CPU agreed with the
  reference on all, the GPU on 0.9878.
- The GPU was faster than the CPU on SST-5 and slower on emotion and SST-2.
  A repeat of the GPU emotion session took a median of 435.8 ms, with the same
  scores; why the first was slower was not measured. Engine init includes
  writing a weight cache: every session was a fresh install.
- Not measured: thermal state and clock frequencies, GPU memory, another phone
  model, the NPU, and reading the section out of the file on the phone.

### The reference, sixteen rows at a time

`verify`'s float reference used to score a decision one row at a time: one
forward over the prompt, one over its options. It now scores up to sixteen
rows of a similar length together (`decision_scripts`). The most padded batch
goes first and every row of it is checked against the row scored alone with
no cache; if it differs or raises, the reference falls back to one row at a
time. The rows the float twin is compared on are recorded as scored alone.

The emotion decision of "Every family on emotion" was retrained with litetune
0.7.1 (gemma-3-270m-it, the same splits and flags) and verified on a Mac with an
M5 Pro, on the CPU, with `--calibration --float-twin --engine-rows 300
--threads 4` and 4,000 reference rows. The Mac was running other work, so CPU
time is the comparable figure; the wall times are given with the load average.

| | one row at a time | sixteen at a time |
|---|---|---|
| CPU time of the whole `verify`, user + system | 2,623 s | 1,175 s |
| reference stage, wall | 1,044 s | 1,257 s at load 30 to 50 (156 s for the first version at load 4) |
| whole `verify`, wall | 18 min 27 s | 25 min 21 s at load 30 to 50 (3 min 23 s for the first version at load 4) |
| runtime's top option / answer is the reference's | 0.9967 / 0.9933 | 0.9967 / 0.9933 |
| temperature, abstain below | 1.2838772, 0.5074026 | 1.2838769, 0.5074019 |
| mean KL | 3.9384e-04 | 3.9383e-04 |
| float twin, largest per-token difference | 5.15e-05 | 4.58e-05 |

The same script before and after batching, on the first 100 held-out rows of
the same task for other families' base checkpoints, measured on the first
batched version, which chunked a fixed 96 pairs; over these prompts the
committed version chunks about as many. The times are approximately
milliseconds per row: the rows the script checked against uncached scoring are
left out, and a batched row carries a share of its batch's checks. "Largest
difference" is the largest per-token difference between the old scores and the
new.

| checkpoint | one at a time | sixteen at a time | largest difference | top option changed |
|---|---|---|---|---|
| gemma-3-270m-it, tuned (300 rows) | 184.1 | 59.0 at load 45 | 4.2e-05 | 0 rows |
| Qwen2.5-0.5B-Instruct | 466.8 | 272.4 | 9.0e-05 | 0 rows |
| Qwen3-0.6B | 606.4 | 358.7 | 1.4e-04 | 0 rows |
| MiniCPM5-1B | 716.0 | 368.0 | 7.4e-05 | 0 rows |
| granite-4.1-3b | 3,582.8 | 818.1 | 4.9e-04 | 0 rows |
| Qwen3.5-0.8B | 2,969.5 | 4,941.2 | 0 | 0 rows |

- The gemma row is the committed version; the others are the first version.
- On Qwen3.5-0.8B no way of passing the cache matched uncached scoring on the
  first row, so it is scored one row at a time, uncached, before and after,
  and the scores were bit-identical. Its two times are the same code path
  under different load.
- No first batch fell back to one row at a time.
- On small random Gemma 3 (a sliding window shorter than the prompts), Granite
  and Qwen3 models, batched scores were within 4e-06 of one row at a time.
- gemma-4-E2B was not run here. In its verify for "Every family on emotion"
  (on CUDA) every way of passing the cache missed uncached scoring by 34.8
  nats, because its forward soft-caps the logits and its bundle does not, so
  it was scored with no cache. The script does not batch without a cache, so
  by the code it scores one row at a time as before; that is not measured.
- On MPS, an experiment script (not wired in) scored the 300 gemma rows at
  29.5 ms a row in batches of 16 and 14.7 ms in batches of 128, within
  8.3e-05 of the CPU; the reference runs on CUDA or the CPU.
- Not measured: FunctionGemma and gemma-4-E2B, whose checkpoints are gated, and CUDA.

### Limitations carried by the decision numbers

- One training run per task and epoch count, on one model, one recipe and the
  CPU backend of one CPU model; on the other families, one task, emotion. One
  phone, for three of the tasks, above.
- The temperature, and for emotion the threshold, were fitted on the
  reference's calibration scores and are applied to the runtime's.
- The runtime scored 300 of each task's held-out rows. The reference scored
  every one of them: 2,000, 784, 1,821 and 2,210.

## EmbeddingGemma 2: a decider fine-tuned and grafted into Google's file

`tune --model google/embeddinggemma-2` fine-tunes the text tower for one
decision, and `convert` writes the result into Google's own
`embeddinggemma-2-text-270m.litertlm`
(litert-community/embeddinggemma-2-text-270m-litert-lm, revision `9be6e8b9`,
164,626,432 bytes) in place of Google's weights: we found no published path
from a PyTorch checkpoint to an EmbeddingGemma 2 `.litertlm` as of October 2026
(litert-torch's `embedding_gemma` example converts the first EmbeddingGemma to
a bare `.tflite`, and its Hugging Face export has no embedding task). The
application uses
the file through flutter_edge_ai's `EmbeddingModel.createDecider`, which
embeds every input behind `task: sentence similarity | query: ` and answers
with the option whose centroid of examples is nearest. Every number below was
taken on a 32-vCPU Linux machine, Intel Xeon @ 2.20GHz, with LiteRT-LM 0.18.0's
`EmbeddingEngine` on the CPU backend against sentence-transformers 6.1.0 in
float32 (torch 2.9.1, transformers 5.19.0). The measurements were driven by
scripts outside this repository (`graft_s2.py`, `train_s5.py`, `emb_s5.py`,
`eval_s5.py`); litetune's own code is what the last subsection measures.

### The graft

Everything in this subsection was embedded behind
`task: classification | query: `, the prefix the first experiments used, not
the decider's; the next subsection uses the decider's.

- Google's file holds the text tower as two TFLite sections, an embedder (a
  262,144×512 table) and a text encoder with six signatures, one per input
  length. Its 219 two-dimensional weights are int4, one scale per output row,
  symmetric, and map one to one by name onto the Hugging Face checkpoint
  (`google/embeddinggemma-2` at `914f7f89`). One of them carries a folded
  constant of 1/512. The RMSNorm weights are inline float32 constants, each
  shared by the six signatures and placed by tracing it back through the
  graph: 170 of them take the checkpoint's norms, and two belong to a value
  norm that has no weight in the checkpoint and are left as Google wrote them.
- Google's weights read out and written back by the graft reproduce Google's
  file to a mean cosine of 0.9993 (all int4) and 0.9987 (an int8 encoder with
  the embedder in int4) on 200 texts, so the writer loses nothing beyond
  re-rounding.
- The base checkpoint grafted in int8 is 0.9975 (mean cosine) from
  sentence-transformers; Google's own file is 0.9690. Grafted in int4 without
  quantization-aware training it is 0.9451 (that run kept Google's norms;
  with the checkpoint's own, 0.9382): four bits cost the base model's vectors
  more than Google's file does.
- Fine-tuned on emotion (six labels, 16,000 rows; a Linear-GELU-Linear head on
  the sentence embedding, cross-entropy + Brier, the head discarded) the
  centroid and linear accuracies on 2,000 test rows were:

  | embeddings | centroid | linear |
  |---|---|---|
  | Google's file | 0.6035 | 0.7435 |
  | fine-tuned, float32 | 0.9325 | 0.9295 |
  | fine-tuned, int8 graft (300,089,344 bytes) | 0.9330 | 0.9300 |
  | fine-tuned, int4 graft, no QAT (164,626,432 bytes) | 0.9330 | 0.9340 |
  | QAT, int4 graft | 0.9335 | 0.9325 |

  Quantization-aware training added nothing measurable there. int4 still moves
  the vectors: mean cosine 0.960, minimum 0.687, to the float32 fine-tune.
- A soft-margin triplet loss tried first collapsed: its loss settled at ln 2
  and its linear probe answered the largest label.
- Of the 173 inline float32 constants of a norm's length in the text encoder's
  1,458 subgraphs, 172 are those norms; the remaining one is read by a
  multiplication inside the composites' decompositions and runs geometrically
  from 1.06e-6 to 1.0, a table of inverse frequencies, not a norm.
- Four engine processes sharing one model file failed with XNNPack's "could not
  verify buffer list mapping": the runtime writes its weight cache beside the
  model unless given a directory. `verify` gives each run a directory of its
  own.

### Many labels

The same recipe with the decider's prefix on every text, on banking77 (77
labels, 10,003 training and 3,080 test rows) and CLINC150's in-scope intents
(150 labels, 15,000 and 4,500), accuracy on the test rows; the k-shot centroid
is fitted on k rows per label, five seeded draws, the same rows for every
embedding:

| banking77 | all rows | k=5 mean | k=10 mean | linear | cosine to float32, mean / min |
|---|---|---|---|---|---|
| Google's file | 0.9110 | 0.8801 | 0.8951 | 0.9364 | |
| fine-tuned, float32 | 0.9341 | 0.9322 | 0.9345 | 0.9367 | |
| int8 | 0.9341 | 0.9311 | 0.9343 | 0.9377 | 0.9993 / 0.9923 |
| int4 | 0.9321 | 0.9294 | 0.9318 | 0.9344 | 0.9693 / 0.8612 |

| CLINC150 | all rows | k=5 mean | k=10 mean | linear | cosine to float32, mean / min |
|---|---|---|---|---|---|
| Google's file | 0.9487 | 0.9138 | 0.9372 | 0.9660 | |
| fine-tuned, float32 | 0.9680 | 0.9682 | 0.9676 | 0.9693 | |
| int8 | 0.9684 | 0.9679 | 0.9674 | 0.9696 | 0.9994 / 0.9957 |
| int4 | 0.9662 | 0.9666 | 0.9659 | 0.9693 | 0.9754 / 0.8876 |

- int4 cost 0.0027 (banking77) and 0.0017 (CLINC150) of the 10-shot centroid's
  accuracy, int8 0.0002 on both; no interval was computed for these (litetune's
  `verify` computes a paired one). So `convert` writes int4 by default.
- An int4 encoder with an int8 embedder (231,735,296 bytes) scored within
  0.0007 of int4 on the 10-shot centroid on both (within 0.0013 on every
  decider), and is not offered.
- The fine-tune gains most where the decider has few examples: +0.0521 and
  +0.0544 at k=5, +0.0394 and +0.0304 at k=10. With a linear layer on every
  training row the gain is 0.0003 and 0.0033.
- int4 moves the vectors (minimum cosine 0.86 to 0.89), so a decider
  threshold set on one file may not carry over to another. Carrying one over
  was not measured (the next subsections fit each file's own), nor were tasks
  that read distances directly, retrieval among them.
- Training took 2,880 s (banking77) and 3,189 s (CLINC150) on the 32 vCPUs,
  in float32. Nothing was trained on a GPU.

### litetune end to end

litetune's own stages, `prepare` → `tune` → `convert` (int8 and int4) →
`verify --with-base`, on banking77 through `prepare`'s split: 9,158 training
and 2,617 held-out rows (so not the official split the previous subsection
used, and not the same numbers).
All six stages exited 0. `tune` took 2,347 s.

| embeddings | all rows | k=5 mean | k=10 mean | linear | cosine to float32, mean / min |
|---|---|---|---|---|---|
| Google's file | 0.9270 | 0.8936 | 0.9137 | 0.9534 | |
| fine-tuned, float32 | 0.9316 | 0.9300 | 0.9312 | 0.9389 | |
| int8 file | 0.9308 | 0.9299 | 0.9314 | 0.9392 | 0.9993 / 0.9869 |
| int4 file | 0.9289 | 0.9260 | 0.9276 | 0.9400 | 0.9688 / 0.7768 |

- Both files passed `verify`'s gate. The conversion cost on the 10-shot centroid
  decider was -0.0002 for int8 (per draw 0 to -0.0011, intervals up to
  ±0.0021) and +0.0037 for int4 (per draw +0.0019 to +0.0050, intervals about
  ±0.004). Three of the five int4 draws resolve the cost as above zero; every
  draw's upper bound is under 0.01.
- The files' gain over Google's at k=10 was +0.0177 (int8) and +0.0138
  (int4), the float32 fine-tune's +0.0175. With a linear layer on every training row Google's file scored
  higher than the fine-tune, 0.9534 against 0.9389.
- `verify` embedded the file and Google's side by side, each at 4 threads, on
  11,775 texts: 47 minutes for int8 and 35 for int4 from start to exit, where
  the previous run, one side after another, took 76 and 62. Per text the
  int4 file took a median 135.8 ms and Google's 134.2 ms; the int8 file 196.3
  ms beside Google's 134.3 ms.
- The int4 file's 10-shot draws equal the previous run's, draw for draw. The
  int8 file's 10-shot mean moved from 0.9318 to 0.9314; between the two runs
  int8 codes changed to stop at -127, as TFLite's quantization specification
  has them, instead of -128, and nothing else in the int8 path changed that
  the int4 path did not share.

### CLINC150, from the published package

The same stages run from litetune 0.7.0 as installed from PyPI, on CLINC150's
in-scope intents (150 options) through `prepare`'s split: 13,650 training and
3,900 held-out rows. All six stages exited 0; `tune` took 3,241 s.

| embeddings | all rows | k=5 mean | k=10 mean | linear | cosine to float32, mean / min |
|---|---|---|---|---|---|
| Google's file | 0.9592 | 0.9298 | 0.9493 | 0.9787 | |
| fine-tuned, float32 | 0.9792 | 0.9796 | 0.9791 | 0.9803 | |
| int8 file | 0.9790 | 0.9796 | 0.9792 | 0.9808 | 0.9994 / 0.9913 |
| int4 file | 0.9764 | 0.9770 | 0.9765 | 0.9787 | 0.9749 / 0.8574 |

- Both files passed: a conversion cost of -0.0001 (int8) and +0.0026 (int4)
  on the 10-shot centroid decider, every draw's upper bound under 0.01. The
  files' gain over Google's at k=10 was +0.0299 (int8) and +0.0272 (int4).

### Abstention, as the app calibrates it

The app's `createDecider` abstains by default: `EmbeddingDeciderCalibration.crossFit`
(keep 0.95) fits a threshold on the options' own examples and refuses a request
whose best cosine is below it. `verify` now fits it the same way on each draw's
k examples. Measured again on the files of the two runs above (the same rows,
five draws), at k=10; every held-out row is in scope:

| banking77 | answered | right when answered | answered and right | right, answering every row |
|---|---|---|---|---|
| Google's file | 0.9558 | 0.9268 | 0.8857 | 0.9137 |
| fine-tuned, float32 | 0.9068 | 0.9702 | 0.8798 | 0.9312 |
| int8 file | 0.9086 | 0.9698 | 0.8812 | 0.9314 |
| int4 file | 0.9094 | 0.9660 | 0.8784 | 0.9276 |

| CLINC150 | answered | right when answered | answered and right | right, answering every row |
|---|---|---|---|---|
| Google's file | 0.9581 | 0.9594 | 0.9192 | 0.9493 |
| fine-tuned, float32 | 0.9082 | 0.9948 | 0.9035 | 0.9791 |
| int8 file | 0.9081 | 0.9948 | 0.9034 | 0.9792 |
| int4 file | 0.9061 | 0.9942 | 0.9008 | 0.9765 |

- With the fine-tune the decider answers fewer requests and is right on more
  of those it answers: 0.97 against 0.93 on banking77, 0.99 against 0.96 on
  CLINC150. Of all requests, the share answered and right is about the same
  on banking77 (0.878–0.886) and lower with the fine-tune on CLINC150
  (0.901–0.904 against 0.919).
- Google's file answers 0.956 and 0.958 of the held-out requests, close to the
  0.95 the calibration keeps; the fine-tune answers about 0.91 on both. The
  examples the threshold is fitted on are training rows, which the fine-tune
  was trained on, and the threshold sits high against the scores of the
  held-out requests; that is consistent with the examples scoring higher
  against their centroids than requests the fine-tune has not seen, which the
  next section tests. In absolute terms the fine-tune's thresholds were lower than
  Google's on banking77 (about 0.82 against 0.89) and about the same on
  CLINC150 (0.88–0.89 against 0.88).
- At k=5 the pattern is the same on both tasks.
- Out-of-scope requests were not in the rows, so how many of those the
  threshold refuses is not measured here; the next section measures it on
  CLINC150. No option had a description, which the app would add as one more
  input and which changes the folds. `verify` reports these numbers and does
  not gate on them.

### Which calibration, with a fine-tuned file

Besides `none`, the app offers three calibrations: `crossFit` with a chosen keep,
`fromLabelled` (a threshold fitted on labelled requests to an error budget),
and `minSimilarity` (a fixed threshold; the app documents 0.875 for Google's
int4 file). Measured on the int8 files and Google's file of the two runs
above, embedded again, at k=10 examples from the training rows, five draws.
`fromLabelled` was fitted on the run's calibration split (rows never trained
on, all in scope: the app accepts only labels that are options) with a 0.05
budget. On CLINC150, out-of-scope requests are its 1,000 test `oos` rows.
This embedding run's numbers for the default `crossFit` differ from the
section above by up to 0.002.

| banking77 | calibration | answered | right when answered | answered and right |
|---|---|---|---|---|
| fine-tuned | crossFit 0.95 | 0.9067 | 0.9701 | 0.8796 |
| fine-tuned | crossFit 0.98 | 0.9541 | 0.9557 | 0.9118 |
| fine-tuned | crossFit 0.99 | 0.9744 | 0.9471 | 0.9228 |
| fine-tuned | minSimilarity 0.875 | 0.8389 | 0.9802 | 0.8223 |
| fine-tuned | fromLabelled 0.05 | 0.9695 | 0.9499 | 0.9209 |
| Google's file | crossFit 0.95 | 0.9570 | 0.9263 | 0.8864 |
| Google's file | crossFit 0.98 | 0.9814 | 0.9194 | 0.9023 |
| Google's file | minSimilarity 0.875 | 0.9880 | 0.9175 | 0.9065 |
| Google's file | fromLabelled 0.05 | 0.8029 | 0.9545 | 0.7663 |

| CLINC150 | calibration | answered | right when answered | answered and right | out of scope refused |
|---|---|---|---|---|---|
| fine-tuned | crossFit 0.95 | 0.9083 | 0.9949 | 0.9036 | 0.9714 |
| fine-tuned | crossFit 0.98 | 0.9470 | 0.9918 | 0.9392 | 0.9500 |
| fine-tuned | crossFit 0.99 | 0.9644 | 0.9894 | 0.9542 | 0.9210 |
| fine-tuned | minSimilarity 0.875 | 0.9274 | 0.9934 | 0.9213 | 0.9668 |
| fine-tuned | fromLabelled 0.05 | 1.0000 | 0.9791 | 0.9791 | 0.0988 |
| Google's file | crossFit 0.95 | 0.9579 | 0.9588 | 0.9185 | 0.9062 |
| Google's file | crossFit 0.98 | 0.9841 | 0.9542 | 0.9390 | 0.8250 |
| Google's file | minSimilarity 0.875 | 0.9754 | 0.9560 | 0.9325 | 0.8676 |
| Google's file | fromLabelled 0.05 | 0.9939 | 0.9512 | 0.9454 | 0.5140 |

- With examples the fine-tune was not trained on (six per option from the
  calibration split, the most its smallest option has), the default
  `crossFit` 0.95 answered 0.9505 of banking77 and 0.9433 of CLINC150, against
  0.9067 and 0.9083 with training examples; Google's file answered 0.9537 and
  0.9649. The examples differ in k as well as in source, but the result is
  what the training-row explanation above predicts.
- The fine-tuned file at `crossFit` 0.98 against Google's file at the default:
  on CLINC150, answered and right 0.9392 against 0.9185, out of scope refused
  0.9500 against 0.9062; on banking77, answered and right 0.9118 against
  0.8864. Raising keep on Google's file to 0.98 refused 0.8250 of out-of-scope
  requests.
- `fromLabelled` on in-scope requests alone refused 0.0988 (fine-tuned) and
  0.5140 (Google's file) of out-of-scope requests: when the decider errs less
  than the budget on in-scope requests, the threshold it fits can be as low as
  answering everything.
- `minSimilarity` 0.875 answered 0.8389 of banking77 with the fine-tuned file:
  a threshold set for Google's file does not carry over.
- Out-of-scope requests were measured on CLINC150 only, at k=10 only, without
  option descriptions.

## The CPU and the GPU on a Mac

Each path was run through the script `verify` runs for it, unchanged except for
the backend, on an Apple M5 Pro with litert-lm 0.18.0 from PyPI: once or twice on
the CPU and three times on the GPU, four for option scoring. For each GPU run the kernel's GPU time for
the process grew during the run. The text and option-scoring runs used the
emotion bundle from *The three passing bundles on a phone* and its 300 rows.
The tool runs used `google/functiongemma-270m-it` converted by litetune 0.7.1
(`dynamic_wi8_afp32`) and 20 single-turn requests, each with its own
declarations: constrained decoding off, 128 tokens at most. The EmbeddingGemma 2
runs used the banking77 int8 file from *litetune end to end*.

| path | the CPU against the GPU | the GPU, run against run |
|---|---|---|
| text, one-word answers (300) | the same text on 300 | identical |
| text, longer answers (20, median 225 characters) | the same text on 3 | identical |
| the same 20 through litert-lm's own CLI | the same text on 3 | identical |
| tools (20) | the same reply on 15 | identical |
| option scoring (300) | the same top option on 299 or 300 | 1 to 3 rows per run with another prompt's scores, up to 13.8 nats away |
| EmbeddingGemma 2 vectors (300) | cosine median 0.990460, minimum 0.975487 | identical |

- On every backend, the CLI and `verify`'s text script gave the same text as
  each other on all 20 longer answers. The difference between the CPU and the
  GPU is litert-lm's, not litetune's. The longer GPU answers usually part from
  the CPU's within the first sentence and then read as different text.
- On the five tool requests where the CPU and the GPU differed, the GPU:
  - dropped a "." from an e-mail address and a "," from a place;
  - added a word to a message body;
  - lengthened a map query;
  - on one request, called the tool where the CPU asked for the date.

  Where both called a tool, it was the same function.
- The CPU repeated its option scores bit for bit; the GPU did not. Its bad rows
  fell at different places in each of four runs, and the top option changed on
  1 of 1,200 decisions. Scoring each option on an Engine of its own, with no
  rewind, gave six rows more than 5 nats from the CPU's scores. The rewind is
  therefore not the cause. Over the 300 decisions, the GPU took a median of 69.55 ms
  per decision and the CPU 179.8 ms.
- flutter_edge_ai's `createDecider`, on its own build of the runtime, on the
  same Mac's GPU repeated its scores bit for bit. It ran four times on the same
  emotion file and 300 rows, and four times each on three other decision
  bundles (300, 60 and 60 rows). The bad rows come from the path through
  litert-lm's Python package, which `verify` runs, and not from the GPU. Which
  difference between the two paths causes them is not established. Option
  scoring stays refused on the GPU in `verify`, which cannot measure it
  repeatably through that package.
- `verify --backend gpu` on the EmbeddingGemma 2 file passed the same gate as on
  the CPU, both on this Mac:

  | | CPU | GPU |
  |---|---|---|
  | candidate, 10-shot | 0.9317 | 0.9311 |
  | reference | 0.9312 | 0.9312 |
  | per-draw conversion cost | -0.0011 to +0.0004 | -0.0011 to +0.0023 |
  | cosine to the reference, mean / minimum | 0.99928 / 0.98909 | 0.9969 / 0.944174 |
  | median per text | 43.43 ms | 11.71 ms |
  | verify, wall time | 12 min 33 s | 6 min 21 s |

  The Python EmbeddingEngine takes no activation data type, so the GPU computes
  in the precision litert-lm picks.
- One Mac and one runtime version. The phone in *The three passing bundles on a
  phone* repeated its GPU option scores bit for bit.
