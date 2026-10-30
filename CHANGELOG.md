# Changelog

Newest first. Full notes for each release are on
[GitHub](https://github.com/DenisovAV/litetune/releases).

## 0.7.2 — 2026-10-30
- `verify --backend gpu` runs on macOS for an EmbeddingGemma 2 file and for a decision bundle, whose option scoring is repeated until the scores settle; refused elsewhere (#98).
- `verify` scores a decision's float reference sixteen rows at a time, the most padded batch first, checked row by row, falling back to one row at a time if it differs (#97).

## 0.7.1 — 2026-10-29
- `tune` computes the logits and the loss only where a token is supervised, and trains a model of 400 million parameters or more with gradient checkpointing, so gemma-4-E2B-it fits a 40 GB GPU on long rows; no flag, the loss is unchanged (#95).
- `tune` and `verify` comments and reports no longer say the training attention reaches evaluation or export; a CUDA out-of-memory ending names the run's micro-batch and length (#95).
- `verify` reports the app's default abstention for an EmbeddingGemma 2 decider, and MEASUREMENTS.md which calibration suits a fine-tuned file (#94).

## 0.7.0 — 2026-10-27
- `tune`, `convert` and `verify` fine-tune EmbeddingGemma 2 for one decision and write it into Google's own `.litertlm`, for flutter_edge_ai's `EmbeddingModel.createDecider`; measured on banking77 end to end (#92).
- An MPS run on a Mac starts at warn memory pressure and on 2 GiB available, and says so in its report (#91).
- MEASUREMENTS.md reports the three passing decision bundles on a Galaxy S24, CPU and GPU (#90).

## 0.6.0 — 2026-10-22
- `tune`, `convert` and `verify` take Qwen3-1.7B and Qwen2.5-1.5B-Instruct, measured on banking77 (#88).
- Three new families: Granite 4.1, MiniCPM5 and Qwen 3.5, each measured on banking77; `verify` gives a Qwen 3.5 bundle an Engine per prompt, since a shared one leaks state between prompts (#88).
- A template's single turns are checked with the start token the runtime prepends, so a template that prints its own BOS is converted rather than refused (#88).
- Decisions are accepted for every family with a rule, and measured on emotion on one or two models of each; `verify` needs `--float-twin` for every family but Gemma 3 text (#88).
- A decision's states and options are refused when they hold any supported family's control text, or one of the tokenizer's own special tokens when `prepare` has it (#88).
- The `litetune.decision` section records how the runtime scored the options, `rewind` or `fresh_engine` (#88).

## 0.5.1 — 2026-10-19
- `prepare` and `verify` say a `noul` question has been measured on two tasks, not one, and MEASUREMENTS.md adds SST-2 as `noul` and irony trained for four and six epochs, which verify passed where two epochs were refused (#85).
- `bundle` no longer says it records nothing that was measured beside the measurements `--verify-manifest` put in `report.json`: the limitation is now about the synthesised `manifest.json`, which lists no stages, and the bundle no longer claims a run's record it was not given (#86).

## 0.5.0 — 2026-10-18
- `prepare --questions`, `tune --decision`, `verify` and `bundle` train, measure and package a model that answers with one of the options an app passes, for flutter_edge_ai's proposed `InferenceModel.createDecider`: one `.litertlm` that carries the question, the options and the calibration in a `litetune.decision` section, measured on `google/gemma-3-270m-it` only (#82).
- `convert` exports a checkpoint that records a decision with `--prefill_lengths=8,128` unless the caller sets it (#82).
- `bundle` writes a decision's section only when verify passed or was inconclusive; with any other status it ships the file without one, taking out a section an earlier bundle left, and exits with verify's status (#82, #83).
- `convert` reports a `litetune.json` beside the checkpoint that is there but cannot be read -- a directory, a dangling link, a FIFO or a device -- as a limitation, where it took it for absent (#80).

## 0.4.3 — 2026-10-14
- **Convert a Qwen3 bundle again if an application holds a conversation with it: every Qwen3 bundle earlier litetune converted without a template of the caller's carries Qwen's own chat template, which LiteRT-LM's conversation cannot render past the first message, so the second message fails (#78).**
- `convert` checks that each bundle's chat template renders single turns as the checkpoint was trained and holds a LiteRT-LM conversation, says in its report when one does not, and gives Qwen3 LiteRT-LM's own Qwen 3 template (#78).
- On Windows each stage runs in a job object: a timeout, Ctrl-C, Ctrl-Break or a forced kill of litetune stops everything the stage started, and Ctrl-Break exits 4 (#77).
- On Windows a stage that closes its pipes and then hangs keeps its output, and on Python 3.10 a usage error with stderr closed exits 4 rather than 120 (#77).

## 0.4.2 — 2026-10-12
- A stage whose pinned packages publish no wheel for this platform and Python is refused before pip, naming the package: `convert` on Windows, a 32-bit or ARM64 Python on Windows, Intel macOS, macOS before 12 and older glibc; `LITETUNE_ALLOW_MISSING_WHEELS=1` lets pip try anyway, and litetune now depends on `packaging` (#74).
- `tune` runs on Python 3.13 on Linux x86_64, with `sentencepiece` 0.2.1 there and 0.2.0 on 3.10 – 3.12 as before (#74).
- On Windows, JSON written by PowerShell is read, console output and help no longer raise on characters the code page lacks, `convert` touches no checkpoint before its environment stands, and the test suite runs in CI (#74).
- `verify`'s float reference decodes greedily without the checkpoint's own `generation_config` penalties, and names any it left out; the Qwen2.5 references measured here applied `repetition_penalty` 1.1, and re-measured without it their figures are unchanged (#73).

## 0.4.1 — 2026-10-12
- **A Qwen3 0.6B or Qwen2.5 0.5B Instruct bundle litetune 0.4.0 converted from a checkpoint it did not recognise -- another organisation's copy, or a fine-tune under its own name -- has no block that reads LiteRT-LM 0.18.0's parts: convert it again; one converted from Qwen's own repositories at their current revisions, or what `tune` made from them, is unaffected (#71).**
- `convert` puts the text-parts block in front of the checkpoint's own chat template instead of a packaged copy, and reads the template back from every bundle it writes (#71).
- Exporter flags passed with `--flag=` are read as the exporter reads them and forms it would read differently are refused; Gemma 4 refuses any model-type override and an embedder opt-out (#71).
- Stage environments link the interpreter instead of copying it, so a Python that loads libpython relative to itself, such as uv 0.10.6's CPython on macOS, can provision them (#70).

## 0.4.0 — 2026-10-11
- **Convert a Qwen3 0.6B or Qwen2.5 0.5B Instruct bundle again for an app on LiteRT-LM 0.18.0, as flutter_edge_ai_litertlm 1.10.0 ships: 0.18.0 hands the template a list of parts, which their checkpoint templates do not read, and the user's text is lost or every send fails (#67).**
- **Convert again any bundle `convert` wrote to a relative `--output-dir` on Python 3.10 or 3.11: it has no GPU activation key, which its report named, and on a GPU backend it answers wrong (#67).**
- `verify` runs the candidate on litert-lm 0.18.0 and `convert` packs with litert-lm-builder 0.18.0; every published number re-measured on it equals its 0.17.1 number, Qwen's on bundles converted again (#67).
- The Qwen3 0.6B and Qwen2.5 0.5B Instruct rules pack the checkpoint's own chat template behind a block that reads 0.18.0's parts, and `verify` names a rendering that came back empty or without the prompt (#67).
- `litetune towers` drops a Gemma 4 bundle's vision and audio towers, or adds them from another bundle (#68).
- The site and README say which device, backend and runtime each number was measured on (#65), and FunctionGemma's measurement notes describe Flutter Edge AI's native tool path (#66).

## 0.3.0 — 2026-10-03
- **Re-run a timed-out `verify` only if you rely on its reference-backend provenance: 0.1.5 through 0.2.1 could record `unknown` after the float-reference model had already reached a device; generated answers and scores are unaffected (#63).**
- `tune` and `verify`'s float reference can use Apple GPUs through Metal/MPS with unified-memory safeguards; `tune --micro-batch-size` exposes gradient accumulation, and `LITETUNE_DEVICE=cpu` forces CPU execution (#63).

## 0.2.1 — 2026-09-29
- The four-bit recipes on the four banking77 checkpoints are re-measured on litert-lm 0.17.1, beside their 0.16.1 numbers (#60).
- The tuned Gemma 4 E2B's eight-bit pair is re-measured on litert-lm 0.17.1, beside its 0.16.1 numbers (#61).
- The site's Qwen2.5-0.5B card no longer calls its channelwise four-bit scores usable (#60).

## 0.2.0 — 2026-09-28
- **Re-run `verify` for any model you ship on litert-lm 0.17.1, as flutter_gemma_litertlm 1.8.0 does — 0.1.x measured on 0.16.1, and the two runtimes do not give the same answers (#55).**
- `verify` runs the candidate on litert-lm 0.17.1, and `convert` pins litert-torch-nightly 0.10.0.dev20260926 with litert-lm-builder 0.17.1 and the quantizer, LiteRT and converter packages it requires (#55).
- Every `verify` manifest records the litert-lm version its candidate ran on and the host's CPU (only on the CPU backend is that what computed the answers), and says so when it could not read either (#55).
- `bundle` records in `established_against` the runtime and CPU its verify manifest recorded, with the CPU's features where the manifest has them (#55, #56).
- The tool path refuses grammar-on decoding for a bundle without a SentencePiece tokenizer, which litert-lm 0.17.1 would decode without the grammar (#55).
- `tune` accepts a tool call whose integer argument fits in 64 bits but not in a double, as litert-lm 0.17.1 returns it exact (#55).
- FunctionGemma's tool path and the four banking77 eight-bit pairs are re-measured on litert-lm 0.17.1, beside their 0.16.1 numbers (#56, #57).
- The published numbers say which runtime they were taken on; the four-bit recipes and Gemma 4 are not yet re-measured (#55, #58).

## 0.1.9 — 2026-09-26
- `verify --backend gpu` measures the converted model on litert-lm's GPU backend, passing fp32 activations itself and naming it when the bundle alone would get something else (#53).
- On macOS a GPU run is recorded as measured on the GPU only when the kernel shows its process doing GPU work; elsewhere, and when the reading is not conclusive, the manifest says the GPU was asked for (#53).
- Every manifest says whether the candidate's backend was observed or only asked for (#52).
- `verify` generates a whole split in one litert-lm process through the runtime's Python API rather than one `litert-lm run` per prompt; the split's budget is still the per-prompt timeout times its prompts, so one hung prompt can use the rest's share (#50).
- A LoRA run on Gemma 4 adapts the text tower only, where before it stopped in peft on the vision and audio projections, and `metrics.json` records what peft matched (#48).
- Gemma 4 E2B is measured end to end on banking77, and litetune's seven LoRA projections against peft's default two (#48).
- Qwen2.5-0.5B-Instruct is a checked family and is measured on banking77 (#47).
- The site renders MEASUREMENTS.md, and each measured model's card links to the section that measured it (#49).
- Each measured model's card says what the model is and what it is for (#49, #51).

## 0.1.8 — 2026-09-20
- **Re-run any `verify` whose host does not default to UTF-8 — Windows always, POSIX with a locale like `en_US.ISO8859-1`** — the runtime's generations were decoded in the host's encoding, so a non-ASCII answer scored as a wrong one (#44).
- **Re-make any bundle or split written on Windows** — text writes turned LF into CRLF, so a bundle's declarations did not hash to its own contract and a split's `content_sha256` differed from the same rows elsewhere (#44).
- **Re-run any stage that crashed on Windows and was reported as a failed model** — an exit code carrying a terminating NTSTATUS is now read as a kill, which is `could not check` (#44).
- A generation whose bytes did not decode is a harness error, not a scored answer; U+FFFD from the model still scores (#44).
- Stage subprocesses are given UTF-8 on both sides of the pipe, and a host value that would decide it is replaced and said (#44).
- `prepare`, `tune`, `verify` and `bundle` run on Windows; `convert` cannot, and the refusal now names `litert-converter` and what to do instead (#45).
- A failed install names the distribution pip could not find rather than printing its trace (#45).
- `gemma-3-1b-it` on banking77, measured: four bits cost it 8.83 points against the 270M's 34.83 (#43).

## 0.1.7 — 2026-09-19
- **Re-prepare and re-train any FunctionGemma tool-call split made with 0.1.6 or earlier** — its calls carry no call markers, so the runtime's tool path returns no call (#41).
- **Re-run any `runtime_rendered` `verify` against a reference whose chat template writes `<bos>` itself, such as Gemma 3 and FunctionGemma** — the reference was prompted with two (#40).
- **Re-train `runtime_rendered` any checkpoint trained `prerendered` on bare prompts that an app serves through its chat template** — neither training nor `verify` saw that template (#40).
- **Breaking:** `tune` refuses a `--prompt-mode` its training prompts contradict; `--force-prompt-mode` overrides and records it (#40).
- `--prompt-mode` is optional: `tune` reads it off the prompts, `verify` and `bundle` from the training record, and each refuses a value that disagrees (#40).
- `verify` in `runtime_rendered` checks that both sides get the same token ids before it generates (#40).
- Reasoning blocks are removed from both sides before scoring (#40).
- Qwen3-0.6B is a checked family, `dynamic_wi4b32_emb8_afp32` is litetune's own recipe, and both are measured on banking77 (#40).
- Tool declarations are an input to `prepare`, `tune`, `verify` and `bundle`; shapes the runtime and the chat template render differently are refused (#41).
- `verify` measures a model whose runtime renders its declarations through the runtime's tool path, grammar off and on; a reply the runtime cannot parse is a wrong answer (#41).
- `tune` refuses a completion the runtime would not read back as its call, and a call to a tool the split does not declare (#41).
- `bundle` ships declarations in the order the model learned, and refuses a set the training run did not record (#41).
- Scoring ends a Gemma 4 turn at `<turn|>` and `<|tool_response>`, so the comparisons 0.1.6 refused now run, and bundles list both as stop tokens (#37).
- The README's NPU mask boundary is per model, not a formula to carry to another (#38).

## 0.1.6 — 2026-09-13
- **Re-run any `tune`, `convert` or `verify` done with `PYTHONPATH` set** — it outranked the stage's pinned packages while the manifest named the pin (#33).
- **Run `litetune env --clean` if `PIP_TARGET`, `PIP_PREFIX` or `PIP_ROOT` was set when a stage first provisioned** — pip installed elsewhere and the environment was marked ready (#33).
- A timeout, Ctrl-C, `kill` or a dropped connection stops everything the stage spawned (#33, #34).
- A stage's last output is no longer cut off at one pipe buffer (#34).
- `pip install` during provisioning runs under the same guard (#34).
- Ready means the marker is a file and the interpreter exists (#33, #34).

## 0.1.5 — 2026-09-08
- **Re-run anything scored with `--scorer exact-text` on 0.1.4 or earlier** — the reference ended in a turn marker and scored 0.0000 (#18).
- **Re-run any `tune` done on a CUDA machine** — it trained on the CPU (#19).
- An accelerator failure is recorded as a fact about the machine, not a failed fine-tune (#19).
- `gemma-3-270m-it` on banking77, measured (#22).

## 0.1.4 — 2026-09-05
- `convert` writes `prefer_activation_type = fp32`; without it the GPU floods `<pad>`.
- `convert --json` records each bundle's `exports[].gpu_activation`.

## 0.1.3 — 2026-09-04
- The native tool path works; 0.1.2 bundles carried a template LiteRT-LM cannot execute.
- `tune` writes `litetune.json`, so `convert` applies that family's export flags.
- An unidentified `gemma3_text` checkpoint is refused, not guessed.
- `--base-model` and `--train-metrics` identify a checkpoint trained elsewhere.

## 0.1.2 — 2026-09-04
- `litetune env` lists the cached stage environments; `--clean` removes them.
- The README's disk figures are measured rather than guessed.

## 0.1.1 — 2026-09-04
- `verify --scorer` chooses what correct means: `tool-call` or `exact-text`.

## 0.1.0 — 2026-09-04
- First release: `prepare`, `tune`, `convert`, `verify`, `bundle`; every check passed, failed or could-not-check, and the exit code is the verdict.
- Measured on `functiongemma-270m-it` with `google/mobile-actions`.
