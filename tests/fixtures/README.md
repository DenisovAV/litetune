Not written here. Each `.jinja` file is a checkpoint's `chat_template` from its
`tokenizer_config.json`, byte for byte, with no header of its own: the tests pin
its sha256, so a header would change the thing they check. litetune 0.4.0
packaged both behind its text-parts block, and these are the parts of those
files after the block.

| file | repository | revision | licence |
|---|---|---|---|
| `qwen3-0.6b-c1899de.chat_template.jinja` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 |
| `qwen2.5-0.5b-instruct-7ae5576.chat_template.jinja` | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 |

From LiteRT-LM, `https://github.com/google-ai-edge/LiteRT-LM` at tag `v0.18.0`
(commit `b2f686e2ed4718fb84ec398a61dd59ca0f0aff27`), Apache-2.0, each file byte
for byte; the git blob is the one `git ls-tree v0.18.0` names for the path, and
the tests check it.

| file | path at v0.18.0 | git blob |
|---|---|---|
| `litert-lm-v0.18.0-qwen2_5.chat_template.jinja` | `models/qwen2_5/chat_template.jinja` | `154f74aaeac0427d24bb789270fd060f90028a0b` |
| `litert-lm-v0.18.0-qwen3-testdata/` | `models/qwen3/testdata/` (three inputs, six goldens) | one per file, in `test_template_checks.py` |

The goldens are what LiteRT-LM's own `chat_template_test_runner.cc` compares
`PromptTemplate::Apply` -- MiniJinja, after `EditTemplateForMinijinja` -- with,
for `models/qwen3/chat_template.jinja`: `<name>.txt` with `enable_thinking`
false, `<name>-thinking.txt` with it true. The template itself is packaged, as
`src/litetune/templates/litert-lm-v0.18.0-qwen3.jinja`.
