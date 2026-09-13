# Changelog

Newest first. Full notes for each release are on
[GitHub](https://github.com/DenisovAV/litetune/releases).

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
