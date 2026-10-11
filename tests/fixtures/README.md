Not written here. Each `.jinja` file is a checkpoint's `chat_template` from its
`tokenizer_config.json`, byte for byte, with no header of its own: the tests pin
its sha256, so a header would change the thing they check. litetune 0.4.0
packaged both behind its text-parts block, and these are the parts of those
files after the block.

| file | repository | revision | licence |
|---|---|---|---|
| `qwen3-0.6b-c1899de.chat_template.jinja` | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 |
| `qwen2.5-0.5b-instruct-7ae5576.chat_template.jinja` | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | Apache-2.0 |
