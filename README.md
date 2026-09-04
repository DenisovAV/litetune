# litetune

**Fine-tune, convert and verify small models for on-device. Know what the
conversion cost you.**

- **tune** — supervised fine-tuning, optional (bring your own checkpoint and skip it)
- **convert** — checkpoint to `.litertlm`, across quantization recipes
- **verify** — measure what conversion cost, before you ship

Converting a model to run on a phone changes its answers. Most tooling tells you
the conversion succeeded. This tells you what it cost.

> **Alpha.** Tested end to end on `google/functiongemma-270m-it` only. Other
> models are not claimed to work. Try it on yours and open an issue.

---

## Install

```bash
pip install git+https://github.com/DenisovAV/litetune
```

Python 3.10–3.12. On Linux you also need the `libvulkan1` system package:

```bash
sudo apt-get install -y libvulkan1     # Debian/Ubuntu
```

macOS needs nothing extra. Colab works out of the box. See
[Requirements](#requirements) for why Vulkan is needed even on CPU, and what
runs on 3.13.

---

## Quick start

**If you already have a `.litertlm`**, this is the whole tool in one command.
Nothing is retrained, no data leaves your machine:

```bash
litetune verify --model ./your-model.litertlm \
                --reference ./the-checkpoint-it-came-from \
                --data held-out.jsonl
```

`--reference` is the **float twin** — the same weights before conversion. That is
what makes the difference between them the conversion cost, rather than a
mixture of that and whatever training did.

Pointing it at a different checkpoint (an untuned base, say) needs
`--reference-role untuned_base`, and then both the training gain and the
conversion cost are reported as unavailable: one number cannot separate two
effects.

First run provisions two environments and pulls `torch` — several gigabytes and
a few minutes. Later runs reuse them.

---

## The full pipeline

Five commands, run in order. Each is a separate subcommand because each fails
differently, and a single `run` would hide which one you are in.

```bash
# 1. Split the data, and refuse what cannot be measured. Seconds.
litetune prepare --data raw.jsonl --output-dir data \
                 --context-length 1024 --tokenizer google/functiongemma-270m-it

# 2. Fine-tune. Runs on CPU, so size your expectations accordingly.
litetune tune --model google/functiongemma-270m-it --data data/train.jsonl \
              --output-dir tuned --prompt-mode prerendered --method lora

# 3. Convert, sweeping recipes rather than trusting a default.
litetune convert --model tuned/model --output-dir artifacts \
                 --recipe dynamic_wi8_afp32 --recipe weight_only_wi8_afp32

# 4. Measure what the conversion cost, against the float twin.
litetune verify --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --reference tuned/model --data data/heldout.jsonl \
                --json > manifest.json

# 5. Package the artifact with what was measured about it.
litetune bundle --output-dir bundle \
                --model artifacts/weight_only_wi8_afp32/<name>.litertlm \
                --declarations tools.json --prompt-mode prerendered \
                --base-model google/functiongemma-270m-it \
                --base-model-revision <commit-sha> \
                --adapter tuned/adapter \
                --train-metrics tuned/metrics.json \
                --verify-manifest manifest.json
```

`convert` names the artifact itself, one per recipe under `artifacts/<recipe>/`,
so step 4 needs you to look the filename up rather than construct it.

### Flags worth knowing about

| Flag | Why it matters |
|---|---|
| `--prompt-mode` | No default. `prerendered` means your app renders the tool declarations into the prompt and the runtime must not template again; `runtime_rendered` is the opposite. Must be the **same** value in `tune` and `bundle` — the wrong one produces a fluent wrong answer, not an error. |
| `--recipe` | No default. A sweep of one is not a comparison, and picking a recipe without measuring the alternative is what this tool exists to stop. |
| `--adapter` | For a LoRA run, pass `<tune output>/adapter`. `tune` merges the adapter into the checkpoint and a merged checkpoint cannot be un-merged, so this is the only form you can re-apply to a different base or ship separately. Point it outside `--output-dir`. |
| `--base-model-revision` | Takes a commit sha. `main` and other moving refs are refused — they resolve to different weights on different days while the bundle reads identically. A tag is accepted and recorded as a weaker pin than it looks. |
| `--wire-convention` | Which property order your tool declarations were rendered in. Optional; unset is recorded as unknown rather than guessed. See [MEASUREMENTS.md](MEASUREMENTS.md). |

### Your data file

One JSON object per line:

```json
{"prompt": "set an alarm for 7", "target": {"name": "set_alarm", "args": {"hour": "7"}}}
```

Training rows add a `completion` field with the text the model should produce.

`prepare` splits one raw file into `train.jsonl` and `heldout.jsonl`. The
held-out half is never trained on — scoring a model on rows it was fitted to
measures memorisation, not whether it answers new inputs, and on a small model
the two can differ enormously. `prepare` splits by content hash rather than
position, so re-running it puts the same rows on the same side, and it reports
the token-length distribution so a row too long for the sequence limit fails
before you rent a GPU rather than after.

---

## Exit codes

The verdict is the exit code; the printed summary renders it. A run that
completed and a run that could not be judged must never be confused, which is
why there are five and not two.

| | `verify` | `prepare` / `tune` / `convert` | `bundle` |
|---|---|---|---|
| **0** | passed | passed | passed |
| **1** | failed a gate, or a label-free check | failed | failed |
| **2** | inconclusive: the interval does not resolve the threshold | — | **the default** |
| **3** | nothing established: no labelled data, or a difference that cannot be attributed | — | carried in |
| **4** | could not check — a harness fault, a bad command line, or a refused request | same | same |

**4 is not a failure of the model.** It means no answer could be obtained: a
missing shared library, a malformed input, a killed process — or a command line
litetune would not run. That last one is why a usage error exits 4 and not
argparse's usual 2: here 2 means *inconclusive*, which is a claim about a
measurement, and a typo should not produce one.

`bundle` carries a verdict rather than producing one, so it returns whatever
`--status` or `--verify-manifest` gave it. With neither it returns 2: bundling
re-measures nothing.

Wiring `|| exit 1` on anything non-zero throws all of this away.

---

## Requirements

| | |
|---|---|
| **Linux or macOS** | On Linux, `litert-lm` `dlopen()`s a Vulkan-linked library even for the CPU backend, so `libvulkan1` is required — without it every invocation, `--help` included, dies in under a second. macOS needs no such package. Verified on Apple silicon: `litetune convert` provisioned its own environment and produced a 455,759,152-byte `.litertlm` — the same byte count as the Linux runs — and `litert-lm` answered a real tool-calling prompt with the correct call. Windows is untried. |
| **Python 3.10–3.13** | Stage environments are built from the interpreter running litetune, so each has its own ceiling: `numpy==2.0.2` (3.12) for `convert` and `verify`, `torch==2.5.1` (3.13) for `tune` and `prepare --tokenizer`. `bundle` provisions nothing. Past a ceiling the command refuses and names the pin that set it. |
| **CPU only** | All five stages. `convert` sets `CUDA_VISIBLE_DEVICES=""` so an export cannot depend on an accelerator. Workable for a 270M model; above that it is the first thing to fix. |
| **Disk** | Several GB for the provisioned environments, cached between runs. |

---

## Results

`functiongemma-270m-it`, LoRA on `google/mobile-actions`, scored on **640
examples the model never trained on** — the single-call rows of the dataset's
`eval` split. Exact match means the tool name **and** every argument value.

| | float | `dynamic_wi8_afp32` | `weight_only_wi8_afp32` |
|---|---|---|---|
| Base | 0.7266 | — | — |
| Fine-tuned | **0.9172** | 0.9016 | 0.9047 |
| Cost of conversion | — | +0.0156 *(unresolved)* | **+0.0125** |

Fine-tuning gained **+0.19**; conversion cost between 0.006 and 0.019 depending
on the run. Nothing in file size, exit code or logs separates the two artifacts —
they are 0.04% apart in bytes. Running both against held-out data is the only
thing that does.

**[MEASUREMENTS.md](MEASUREMENTS.md)** has the full story: three runs of the same
configuration and what they disagree about, why the intervals are paired, and
which published claims were withdrawn after re-measurement.

---

## Limitations

- **Measured on one model.** `functiongemma-270m-it` only. The code is not
  specific to it, but nothing else has been measured.
- **Measurement runs on CPU; your users run on a phone.** Published reports put
  the GPU backend materially below CPU on identical artifacts, so every number
  here is an optimistic estimate.
- **Peak memory is not bounded.** Training this model on 8,693 examples was
  OOM-killed at 32 GiB more than once. There is no preflight check; a death with
  no Python traceback is probably this.
- **The SentencePiece tokenizer is restored by hand.** `transformers` 5.x stopped
  writing `tokenizer.model`, and without it a bundle silently gets an HF
  tokenizer section and loses FST-constrained decoding. `tune` copies it back and
  records whether it managed to — check that field.
- **Two prompt renderings are in the field** for the same model, and they
  disagree for every declaration with more than one property. Costly on a base
  checkpoint, near-free after fine-tuning; `contract.json` records which you
  used. See [MEASUREMENTS.md](MEASUREMENTS.md).
- **Decoding parameters reach only one side.** litetune passes none to the
  device side, so the reference is held to an explicit token limit while the
  device runs to the runtime's own. The manifest says so and counts unterminated
  generations. This is now litetune's gap rather than the toolchain's: the pinned
  `litert-lm` does accept `--top-k`, `--top-p`, `--temperature` and `--seed`, and
  they are not yet wired through.
- **Passing tools natively is not supported.** litetune measures the path where
  the application renders declarations into the prompt and parses calls out of
  the response — what `flutter_gemma` does, and what every number here was
  produced with. Handing the runtime a tool list instead (`litert-lm --preset`,
  or the Kotlin `ConversationConfig.tools`) fails on a bundle built here with
  `litert_lm_conversation_send_message_stream failed`. Reproduced on macOS; the
  cause is not yet known, and no number in this README depends on that path.
- **Evaluation is slower than it needs to be** — one subprocess per prompt. A
  persistent `litert-lm serve` client is worth roughly thirtyfold and is not
  implemented.
- **`.litertlm` only**, and there is no library API or single `run` command yet;
  the config format that would drive one is implemented but not wired to the CLI.

---

## Contributing

Issues and pull requests welcome, particularly measurements on models other than
the one above — that is the gap this alpha most needs closed.

Run the checks with `pytest`, `ruff check`, `ruff format --check` and `mypy src`.

## License

Apache 2.0.
