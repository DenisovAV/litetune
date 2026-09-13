# Changelog

Every release, newest first, dated when it was published (UTC). A version that
makes an earlier result wrong says so first, and says what to do about it. The
full notes for each release are on
[GitHub](https://github.com/DenisovAV/litetune/releases).

## 0.1.5 — 2026-09-08

- **`--scorer exact-text` on 0.1.4 or earlier did not measure the conversion
  cost. Re-run it.** The reference generation ended in a turn marker that
  `exact-text` did not forgive, so the reference scored 0.0000 and `verify`
  reported a resolved cost of −0.6767. `--scorer tool-call` was not affected.
  Scoring now requires every terminator the target ends with, and the
  vocabulary it trims against is recorded at `harness.terminators`.
- **`tune` on a CUDA machine trained on the CPU. Re-run it.** Neither
  generated script chose a device, and the reference backend reported a
  hardcoded `"cpu"`. The device is now resolved once before the run and
  recorded on both sides, or left as `unknown` when it cannot be established.
  Verified on an NVIDIA L4.
- An accelerator failure such as CUDA out-of-memory is recorded as a fact about
  the machine, not as a failed fine-tune.
- A second measured model family: `gemma-3-270m-it` on banking77, both
  conversion costs unresolved at 600 rows.

[Release notes](https://github.com/DenisovAV/litetune/releases/tag/v0.1.5)

## 0.1.4 — 2026-09-05

- `convert` writes `prefer_activation_type = fp32` into every bundle and checks
  the rebuild by unpacking it again. Without it, on a Galaxy S24 (Adreno 750)
  the tuned FunctionGemma bundle flooded `<pad>` on GPU in 14 of 20 rows while
  the engine reported success; with it, 0 of 20, at 1.8× the CPU speed.
- `convert --json` records each bundle's `exports[].gpu_activation`. A bundle
  that could not be repacked is kept and named as CPU-only.

[Release notes](https://github.com/DenisovAV/litetune/releases/tag/v0.1.4)

## 0.1.3 — 2026-09-04

- The native tool path works. Bundles built by 0.1.2 carried a prompt template
  LiteRT-LM cannot execute, so handing the runtime a tool list failed; `convert`
  now passes one that it can.
- `tune` writes `litetune.json` beside the checkpoint, so `convert` knows which
  model it came from and applies that family's export flags.
- An unidentified `gemma3_text` checkpoint is refused rather than guessed at.
- `--base-model` and `--train-metrics` identify a checkpoint trained elsewhere.

[Release notes](https://github.com/DenisovAV/litetune/releases/tag/v0.1.3)

## 0.1.2 — 2026-09-04

- `litetune env` lists the environments the stages cached, largest first, and
  `litetune env --clean` removes them.
- The README's disk figures, which had never been measured, are measured:
  `verify` ~740 MB, `tune` 588 MB, `convert` 1.6 GB on macOS.

[Release notes](https://github.com/DenisovAV/litetune/releases/tag/v0.1.2)

## 0.1.1 — 2026-09-04

- `verify --scorer` chooses what correct means: `tool-call` (the default) or
  `exact-text`. The shape of the target declares the task.

[Release notes](https://github.com/DenisovAV/litetune/releases/tag/v0.1.1)

## 0.1.0 — 2026-09-04

- First release: `prepare`, `tune`, `convert`, `verify` and `bundle`, with
  every check passed, failed or could-not-check, and the exit code as the
  verdict.
- Measured on `functiongemma-270m-it` with `google/mobile-actions`.

[Release notes](https://github.com/DenisovAV/litetune/releases/tag/v0.1.0)
