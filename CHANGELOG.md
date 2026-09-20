# Changelog

Newest first. Full notes for each release are on
[GitHub](https://github.com/DenisovAV/litetune/releases).

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
