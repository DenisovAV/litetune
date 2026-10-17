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

`decision-contract-v1.json` holds the golden strings of the decision prompt
contract (`src/litetune/decisions.py`), which litetune's renderer tests against
and flutter_edge_ai's proposed renderer is to test against too. Each case also
carries `decision`, the section-shaped decision its `questions` read as, with
every option spelled out; the test checks it against `decision_from`. Three
cases are rows the option-scoring measurement of 2026-10-15 trained or scored,
rendered here from the dataset's own text and checked equal
to the rows it rendered; the emotion, Noul and Score cases are constructed, and
were never trained.

| case | dataset | revision | licence |
|---|---|---|---|
| `abcd` (state, option ids and descriptions) | `asappresearch/abcd` (GitHub), `data/abcd_v1.1.json.gz` | `6b8700ce67c6b37b062dd7a60abc76d7ef832a97` | MIT |
| `clinc151 in scope`, `clinc151 out of scope` (states and intent ids) | `clinc/clinc_oos` (Hugging Face), config `plus` | `155b9c710419136e17307b80d0a13e68cd46b4ec` | CC BY 3.0 (Larson et al., 2019) |
