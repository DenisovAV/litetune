"""`verify` for an embedding checkpoint: the decider the application builds, on both sides.

`litetune verify --reference <checkpoint>` lands here when the checkpoint's
`litetune.json` is an embedding record (`embedding_tune`). flutter_edge_ai's
`EmbeddingModel.createDecider` embeds every input behind `DECIDER_PREFIX` and
answers with the option whose centroid -- the mean of its examples' vectors,
L2-normalised -- is nearest by cosine. So that is what is measured, the way
the measurement this path rests on measured it
(MEASUREMENTS.md, "EmbeddingGemma 2: many labels"):

- **the reference**, the fine-tune in float32 through sentence-transformers on
  the CPU, prompt `PROMPT_NAME`, normalised (`REFERENCE_SCRIPT`, in
  `envs.EMBED_TRAIN`);
- **the candidate**, the `.litertlm`, through LiteRT-LM's `EmbeddingEngine` on
  the CPU backend, `DECIDER_PREFIX` + the text, normalised (`ENGINE_SCRIPT`, in
  `envs.RUNTIME`);
- optionally **Google's file**, measured as the candidate is, for the gain.

Every train and held-out state is embedded once per side. The deciders
(`EVAL_SCRIPT`, in `envs.EMBED_TRAIN`, which has numpy and torch): the centroid
of every training row; the centroid of `k` rows per option, `k` in `KS`, drawn
`DRAWS` times with seeds 0.. -- the same rows for every side, so the sides are
paired; and a linear layer (LBFGS, cross-entropy, weight decay `LINEAR_DECAY`).

**The gate.** The conversion cost is the reference's accuracy minus the
candidate's on the `GATED_K`-shot centroid decider, on the same held-out rows,
with `metrics.paired_difference`'s CI95 -- once per draw. It passes when every
draw's interval lies at or under the threshold, fails when any draw's lies
entirely above it, and is inconclusive otherwise, as `verify._gate` reads one
interval. Per draw, not pooled: the draws share rows and vectors, so pooling
them would count each row `DRAWS` times.

The engine hazard: LiteRT-LM writes the XNNPack weight cache beside the model
unless it is given a cache directory, and processes writing one cache path at
once failed with "could not verify buffer list mapping"
(MEASUREMENTS.md, "EmbeddingGemma 2: many labels"). Here each engine run
gets its own cache directory under this run's temporary directory
(`cache_dir`), so nothing is written beside the user's file or Google's, and no
two runs -- of this verify or of another one -- share a cache path.

The candidate and Google's file are embedded side by side, one process each
(`ENGINES_SCRIPT`), when they are two files and the machine has the CPUs for
both at `threads` each; otherwise one after another. On a 32-vCPU Intel Xeon @
2.20GHz, LiteRT-LM 0.18.0 at 4 threads, 11,775 texts, `verify` took 76 and 62
minutes (int8, int4) one side after another and 47 and 35 side by side
(MEASUREMENTS.md, "EmbeddingGemma 2: litetune end to end"). The side-by-side
processes are started by one child of
`StageEnv.run`, not by threads here: `envs`' signal guard works on the main
thread only, and a grandchild stays in the child's session and job object, so
`envs` stops it with the child.

Only the candidate decides the verdict. When Google's file cannot be embedded,
that is a limitation and the gain is not reported; the candidate is still
gated.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litetune import envs
from litetune.checks import Check, CheckSet, Outcome
from litetune.decisions import Decision, DecisionError, decision_from
from litetune.embedding_export import (
    EMBEDDING_CONVERT_SCHEMA,
    GOOGLE_HUB_FILE,
    REPORT_NAME,
    DownloadFailed,
    check_google_file,
    download,
)
from litetune.embedding_tune import (
    DECIDER_PREFIX,
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    PROMPT_NAME,
    EmbeddingRecord,
    EmbeddingTuneError,
    LabelledText,
    states_of,
)
from litetune.evaluate import RUNTIME_ENGINE_SOURCE
from litetune.events import EventStream
from litetune.metrics import Difference, paired_difference
from litetune.prepare import PrepareError, read_rows
from litetune.storage import hash_file

logger = logging.getLogger(__name__)

SCORER = "embedding-centroid"
SCORER_MEANS = (
    "flutter_edge_ai's EmbeddingModel.createDecider's centroid answer with no abstention: the "
    "option whose centroid -- the mean of its examples' embeddings, L2-normalised -- has the "
    "highest cosine to the input's embedding, every input answered; every text embedded behind "
    "the decider's prefix"
)

# eval_s5.py's `KS` and `DRAWS`; the gate reads the larger k.
KS = (5, 10)
DRAWS = 5
GATED_K = 10
# eval_s5.py's linear layer.
LINEAR_DECAY = 1e-5
# The spike's criterion (the design, "Gate").
DEFAULT_MAX_CONVERSION_COST = 0.01
DEFAULT_THREADS = 4
REFERENCE_BATCH = 64
# flutter_edge_ai's input cap (litert_lm_embedding_pass.dart, "The input cap is
# set explicitly to 1024 tokens").
MAX_INPUT_TOKENS = 1024
# Only these runs stop the others when they fail: Google's file is not gated.
STOP_ON = ("candidate",)

# Allowances, not measurements: a start-up, then a budget per text.
STARTUP_TIMEOUT_S = 900
ROW_TIMEOUT_S = 2
EVAL_TIMEOUT_S = 3600
# How long a terminated engine run gets to exit before it is killed.
STOP_TIMEOUT_S = 30

RECORD_CHECK = "the reference is an embedding checkpoint"
ROWS_CHECK = "rows are the decision's states"
BASE_CHECK = "Google's file"
REFERENCE_CHECK = "reference embeddings"
CANDIDATE_CHECK = "candidate embeddings"
BASE_EMBED_CHECK = "Google's file embeddings"
EVAL_CHECK = "decider accuracy"
GATE_CHECK = "conversion cost within threshold"

CPU_ONLY = (
    "measured on the CPU backend of LiteRT-LM's EmbeddingEngine, {threads} threads, from the "
    "litert-lm package on PyPI on this host's CPU -- not the application's own build of the "
    "runtime on a device's CPU; the GPU and the NPU are not measured"
)
CALIBRATION_UNMEASURED = (
    "the application's default calibration, EmbeddingDeciderCalibration.crossFit keeping 0.95 "
    "(flutter_edge_ai lib/src/decision/decider_method.dart:99, "
    "lib/src/embedding/embedding_model.dart:181-182), abstains below a cosine threshold; "
    "abstention is not measured here, every input is answered. A similarity threshold picked on "
    "another file does not carry over to this one: int4 moves the cosines (MEASUREMENTS.md, "
    '"EmbeddingGemma 2: many labels")'
)
FITTED_ON_EXAMPLES = (
    "the application's decider is fitted on the examples it is given at run time. Here it is "
    "fitted on {ks} training rows per option ({draws} draws each) and on every training row; "
    "an application given other examples gets another accuracy"
)
DESCRIPTIONS_UNUSED = (
    "the options' descriptions are not embedded here, nor were they trained on; an application "
    "that passes them to the decider is not measured"
)
LINEAR_NOT_GATED = (
    f"the linear decider (LBFGS, cross-entropy, weight decay {LINEAR_DECAY:g}) is fitted on every "
    "training row, and is reported and not gated"
)
BASE_FAILED = (
    "Google's file could not be embedded, so the gain over it is not reported; the candidate is "
    "gated as without it: {reason}"
)
NOT_TIED = "the file is not tied to the reference checkpoint: {reason}"
NO_BASE = (
    "the gain over Google's file was not measured; pass --base-litertlm or --with-base to measure "
    "it on these rows"
)
SIDE_BY_SIDE = (
    "the candidate and Google's file were embedded at the same time, each on its own {threads} "
    "threads; each file's time per text was measured with the other running"
)
REFERENCE_ON_CPU = (
    "the reference ran on the CPU in float32 through sentence-transformers, whatever "
    "LITETUNE_DEVICE says"
)


@dataclass
class EmbeddingVerify:
    """Everything one embedding verification needs from `verify`'s request."""

    model: Path
    reference: str
    heldout: Path
    # Where the decider's examples come from; `None` is train.jsonl beside `heldout`.
    train: Path | None = None
    base_litertlm: Path | None = None
    with_base: bool = False
    threads: int = DEFAULT_THREADS
    max_conversion_cost: float | None = None
    backend: str = "cpu"
    reference_role: str = "float_twin"
    limit: int | None = None
    # The other verify flags given that do not apply here, by their CLI names.
    not_applicable: tuple[str, ...] = ()
    train_env: envs.StageEnv = envs.EMBED_TRAIN
    runtime_env: envs.StageEnv = envs.RUNTIME
    export_env: envs.StageEnv = envs.EXPORT
    auto_provision: bool = True

    @property
    def train_path(self) -> Path:
        return self.train if self.train is not None else Path(self.heldout).parent / "train.jsonl"

    @property
    def threshold(self) -> float:
        if self.max_conversion_cost is None:
            return DEFAULT_MAX_CONVERSION_COST
        return self.max_conversion_cost


# ---------------------------------------------------------------------------
# The scripts
# ---------------------------------------------------------------------------

REFERENCE_SCRIPT = r'''
"""Embed texts with a fine-tuned EmbeddingGemma 2: sentence-transformers, float32, CPU.

Reads JSON {model, prompt_name, prefix, texts, threads, batch_size, out, report}
from argv[1]. Writes the L2-normalised vectors, float32, one row per text, to
spec["out"] (numpy.save) and JSON to spec["report"]. A checkpoint whose prompt
is not the decider's prefix, or a vector with a non-finite value or a zero norm,
is {"error": ...} and exit 3. As the measurement's sentence-transformers side
(MEASUREMENTS.md, "EmbeddingGemma 2: many labels").

The report's "tokens" counts each text's tokens with the prompt and the
tokenizer's specials, through the model's tokenizer without truncation, and how
many exceed `max_seq_length`: sentence-transformers cuts a text there, so such a
row's reference vector is of a shorter text than the one the engine embeds. A
text of exactly `max_seq_length` tokens fits, as prepare's
`measure_embedded_lengths` counts it (`n > limit` is over).
"""
import importlib.metadata
import json
import sys
import time
from pathlib import Path

REFUSED_EXIT = 3


def unusable_rows(np, v):
    """Indices of the rows with a non-finite value or a zero (or overflowing) norm."""
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.linalg.norm(v, axis=1)
    return np.flatnonzero(~np.isfinite(v).all(axis=1) | ~(np.isfinite(norms) & (norms > 0)))


def token_counts(st, prompt, texts, batch_size):
    """Each text's token count behind `prompt`, with the tokenizer's specials and
    without truncation: `st.tokenize` cuts at max_seq_length, so its count could
    never show a text longer than that."""
    counts = []
    for start in range(0, len(texts), batch_size):
        batch = [prompt + text for text in texts[start : start + batch_size]]
        ids = st.tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        counts.extend(len(row) for row in ids)
    return counts


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    report = Path(spec["report"])

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(spec["threads"])
    started = time.time()
    st = SentenceTransformer(
        spec["model"],
        device="cpu",
        model_kwargs={"dtype": torch.float32},
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    found = st.prompts.get(spec["prompt_name"])
    if found != spec["prefix"]:
        report.write_text(json.dumps({"error": (
            f"the checkpoint's prompt {spec['prompt_name']!r} is {found!r}, not the decider's "
            f"prefix {spec['prefix']!r}")}), encoding="utf-8")
        return REFUSED_EXIT
    texts = json.loads(Path(spec["texts"]).read_text(encoding="utf-8"))
    counts = token_counts(st, found, texts, spec["batch_size"])
    limit = st.max_seq_length
    raw = st.encode(
        texts,
        prompt_name=spec["prompt_name"],
        batch_size=spec["batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float64)
    bad = unusable_rows(np, raw)
    if len(bad):
        report.write_text(json.dumps({"error": (
            f"{len(bad)} of {raw.shape[0]} reference vectors have a non-finite value or a zero "
            f"norm (rows {bad[:10].tolist()})")}), encoding="utf-8")
        return REFUSED_EXIT
    v = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    np.save(spec["out"], v.astype(np.float32))
    report.write_text(json.dumps({
        "rows": int(v.shape[0]),
        "dim": int(v.shape[1]),
        "seconds": round(time.time() - started, 1),
        "device": "cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
        "max_seq_length": limit,
        "tokens": {
            "max": max(counts) if counts else None,
            "over_max_seq_length": sum(1 for n in counts if n > limit),
        },
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "transformers", "sentence-transformers", "numpy")},
    }), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

ENGINE_SCRIPT = (
    r'''
"""Embed texts with a .litertlm through LiteRT-LM's EmbeddingEngine on the CPU backend.

Reads JSON {model, prefix, texts, threads, cache_dir, max_input_length, out,
report} from argv[1]. The engine's XNNPack weight cache goes to
spec["cache_dir"], not beside the model. The settings flutter_edge_ai's
litert_lm_embedding_pass.dart pins are set here as the 0.18.0 wrapper names
them: max_input_length, normalize, insert_special_tokens; an input past
max_input_length is an error (InputOverflowStrategy.ERROR) where the app
truncates and refuses the truncated response, a getter the PyPI wrapper does
not have. Writes the L2-normalised vectors, float32, one row per text, to
spec["out"] (numpy.save) and JSON to spec["report"], with what was set. A text
the engine refuses, or a vector with a non-finite value or a zero norm, is
{"error": ...} and exit 3. As the measurement's engine side
(MEASUREMENTS.md, "EmbeddingGemma 2: many labels").
"""
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REFUSED_EXIT = 3


def unusable_rows(v):
    """Indices of the rows with a non-finite value or a zero (or overflowing) norm."""
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.linalg.norm(v, axis=1)
    return np.flatnonzero(~np.isfinite(v).all(axis=1) | ~(np.isfinite(norms) & (norms > 0)))

'''
    + RUNTIME_ENGINE_SOURCE
    + r"""

def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    report = Path(spec["report"])
    import litert_lm

    texts = json.loads(Path(spec["texts"]).read_text(encoding="utf-8"))
    settings = {
        "cache_dir": spec["cache_dir"],
        "max_input_length": spec["max_input_length"],
        "normalize": True,
        "insert_special_tokens": True,
        "input_overflow_strategy": "ERROR",
    }
    started = time.time()
    out, times = [], []
    backend = litert_lm.Backend.CPU(thread_count=spec["threads"])
    with litert_lm.EmbeddingEngine(
        spec["model"],
        backend=backend,
        cache_dir=spec["cache_dir"],
        max_input_length=spec["max_input_length"],
    ) as engine:
        options = litert_lm.EmbeddingOptions(
            normalize=True,
            insert_special_tokens=True,
            input_overflow_strategy=litert_lm.InputOverflowStrategy.ERROR,
        )
        for i, text in enumerate(texts):
            t1 = time.perf_counter()
            try:
                answer = engine.compute_embedding(spec["prefix"] + text, options=options)
            except RuntimeError as exc:
                report.write_text(json.dumps({"error": (
                    f"the engine refused text {i} ({len(text)} characters): {exc}; an input "
                    f"longer than {spec['max_input_length']} tokens is refused here")}),
                    encoding="utf-8")
                return REFUSED_EXIT
            out.append(list(answer.embedding))
            times.append(time.perf_counter() - t1)
    raw = np.array(out, dtype=np.float64)
    bad = unusable_rows(raw)
    if len(bad):
        report.write_text(json.dumps({"error": (
            f"{len(bad)} of {raw.shape[0]} engine vectors have a non-finite value or a zero "
            f"norm (rows {bad[:10].tolist()})")}), encoding="utf-8")
        return REFUSED_EXIT
    v = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    np.save(spec["out"], v.astype(np.float32))
    runtime = {}
    for dist in ("litert-lm", "litert-lm-api"):
        try:
            runtime[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            pass
    report.write_text(json.dumps({
        "rows": int(v.shape[0]),
        "dim": int(v.shape[1]),
        "seconds": round(time.time() - started, 1),
        "ms_median": round(1000 * statistics.median(times), 2) if times else None,
        "backend": "cpu",
        "threads": spec["threads"],
        "settings": settings,
        "runtime": runtime,
        "cpu": cpu_report(),
        "python": platform.python_version(),
    }), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""
)

ENGINES_SCRIPT = r'''
"""Run the engine script once per file: side by side, or one after another.

Reads JSON {script, runs: [{name, spec, stderr}], parallel, stop_on, stop_s,
report} from argv[1]. Starts `python script spec` for each run, its stderr to
the run's file; side by side when `parallel`, else in order. A run named in
`stop_on` that exits non-zero stops the rest: a run still going is terminated
and one not yet started is not started, and both report returncode null. Any
other run's failure is only reported. A run stopped here, or still going when
this script ends for any reason, is terminated, given `stop_s` seconds, then
killed. Writes {runs: {name: {returncode, seconds}}, parallel} to
spec["report"] and exits 0; reading each run's own report is the caller's.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

POLL_S = 1.0


def start(script, run):
    err = open(run["stderr"], "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, script, run["spec"]], stdout=subprocess.DEVNULL, stderr=err
    )
    return proc, err, time.time()


def stop(proc, err, stop_s):
    """Terminate one run, wait `stop_s` seconds for it, then kill it."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=stop_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        err.close()


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    runs = spec["runs"]
    stop_on = set(spec["stop_on"])
    results = {run["name"]: {"returncode": None, "seconds": None} for run in runs}
    waiting = list(runs)
    live = {}
    stopping = False
    try:
        while waiting or live:
            while waiting and not stopping and (spec["parallel"] or not live):
                run = waiting.pop(0)
                live[run["name"]] = start(spec["script"], run)
            if stopping:
                waiting.clear()
            for name in list(live):
                proc, err, began = live[name]
                code = proc.poll()
                if code is None:
                    continue
                err.close()
                del live[name]
                results[name] = {"returncode": code, "seconds": round(time.time() - began, 1)}
                if code != 0 and name in stop_on:
                    stopping = True
                    while live:
                        stop(*live.popitem()[1][:2], spec["stop_s"])
                    break
            if live:
                time.sleep(POLL_S)
    finally:
        while live:
            stop(*live.popitem()[1][:2], spec["stop_s"])
    Path(spec["report"]).write_text(
        json.dumps({"runs": results, "parallel": bool(spec["parallel"])}), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# Its helpers above `main` use numpy alone (the linear layer is handed torch),
# so the tests run them in this process.
EVAL_SCRIPT = r'''
"""The deciders on each side's vectors, as the measurement computed them
(MEASUREMENTS.md, "EmbeddingGemma 2: many labels").

Reads JSON {vectors: {name: path}, train_labels, heldout_labels, n_labels, ks,
draws, decay, pairs, report} from argv[1]. Each vectors file holds the train
rows, then the held-out rows. Writes to spec["report"]: per side the accuracy
of every decider and, per held-out row, whether each centroid decider was
right; and the per-row cosine between each pair of sides on the held-out rows.
A side with too few or too many vectors, or with a vector that has a
non-finite value or a zero norm, is {"error": ...} and exit 3.
"""
import json
import sys
from pathlib import Path

import numpy as np


def unusable_rows(v):
    """Indices of the rows with a non-finite value or a zero (or overflowing) norm."""
    with np.errstate(invalid="ignore", over="ignore"):
        norms = np.linalg.norm(v, axis=1)
    return np.flatnonzero(~np.isfinite(v).all(axis=1) | ~(np.isfinite(norms) & (norms > 0)))


def kshot_picks(ytr, n_labels, k, draw):
    """k training rows per label, drawn as eval_s5.py draws them: one generator
    seeded `draw` per label, without replacement."""
    return np.concatenate([
        np.random.default_rng(draw).choice(np.flatnonzero(ytr == c), k, replace=False)
        for c in range(n_labels)
    ])


def centroid_correct(xtr, ytr, xte, yte, n_labels):
    """Per held-out row, whether the nearest normalised label mean is its label."""
    c = np.stack([xtr[ytr == k].mean(0) for k in range(n_labels)])
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return (xte @ c.T).argmax(1) == yte


def linear_correct(torch, xtr, ytr, xte, yte, n_labels, decay):
    torch.manual_seed(0)
    X, Y = torch.tensor(xtr), torch.tensor(ytr)
    W = torch.zeros(X.shape[1], n_labels, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_labels, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=500, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, Y) + decay * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return (torch.tensor(xte) @ W + b).numpy().argmax(1) == yte


def evaluate(v, ytr, yte, n_labels, ks, draws, linear=None):
    """(accuracies, per-row correctness) of every decider on one side's vectors."""
    ntr = len(ytr)
    xtr, xte = v[:ntr], v[ntr:]
    acc, correct = {}, {}
    hit = centroid_correct(xtr, ytr, xte, yte, n_labels)
    acc["centroid"] = round(float(np.mean(hit)), 4)
    correct["centroid"] = hit.astype(int).tolist()
    if linear is not None:
        acc["linear"] = round(float(np.mean(linear(xtr, ytr, xte, yte))), 4)
    for k in ks:
        hits = []
        for d in range(draws):
            idx = kshot_picks(ytr, n_labels, k, d)
            hits.append(centroid_correct(xtr[idx], ytr[idx], xte, yte, n_labels))
        accs = [float(np.mean(h)) for h in hits]
        acc[f"centroid_k{k}"] = {"mean": round(float(np.mean(accs)), 4), "min": round(min(accs), 4),
                                 "draws": [round(x, 4) for x in accs]}
        correct[f"centroid_k{k}"] = [h.astype(int).tolist() for h in hits]
    return acc, correct


def cosine_rows(a, b):
    cs = np.sum(a * b, 1)
    return {"mean": round(float(cs.mean()), 6), "min": round(float(cs.min()), 6),
            "rows": int(cs.shape[0])}


def main():
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    import torch

    ytr = np.array(spec["train_labels"])
    yte = np.array(spec["heldout_labels"])
    ntr, n = len(ytr), spec["n_labels"]
    vectors = {}
    for name, path in spec["vectors"].items():
        v = np.load(path).astype(np.float64)
        if v.shape[0] != ntr + len(yte):
            Path(spec["report"]).write_text(json.dumps({"error": (
                f"{name}: {v.shape[0]} vectors for {ntr + len(yte)} rows")}), encoding="utf-8")
            return 3
        bad = unusable_rows(v)
        if len(bad):
            Path(spec["report"]).write_text(json.dumps({"error": (
                f"{name}: {len(bad)} vectors have a non-finite value or a zero norm "
                f"(rows {bad[:10].tolist()})")}), encoding="utf-8")
            return 3
        vectors[name] = v
    out = {"accuracy": {}, "correct": {}, "cosine": {},
           "versions": {"numpy": np.__version__, "torch": torch.__version__}}

    def linear(xtr, ytr_, xte, yte_):
        return linear_correct(torch, xtr, ytr_, xte, yte_, n, spec["decay"])

    for name, v in vectors.items():
        out["accuracy"][name], out["correct"][name] = evaluate(
            v, ytr, yte, n, spec["ks"], spec["draws"], linear)
    for a, b in spec["pairs"]:
        out["cosine"][f"{a}|{b}"] = cosine_rows(vectors[a][ntr:], vectors[b][ntr:])
    Path(spec["report"]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


@dataclass
class ScriptRun:
    """One script's report, or why there is none."""

    report: dict[str, Any] = field(default_factory=dict)
    failure: str | None = None


def _tail(text: str | None, limit: int = 2000) -> str:
    return (text or "").strip()[-limit:]


def run_script(
    env: envs.StageEnv,
    source: str,
    spec: dict[str, Any],
    work: Path,
    name: str,
    timeout: int,
) -> ScriptRun:
    """Write `source` and `spec` under `work`, run it in `env`, read back its report."""
    run = ScriptRun()
    script, spec_path, report = work / f"{name}.py", work / f"{name}.json", work / f"{name}.out"
    script.write_text(source, encoding="utf-8")
    spec_path.write_text(json.dumps({**spec, "report": str(report)}), encoding="utf-8")
    try:
        proc = env.run(["python", str(script), str(spec_path)], timeout=timeout)
    except subprocess.TimeoutExpired:
        run.failure = f"no result after {timeout} s (timeout)"
        return run
    except OSError as exc:
        run.failure = f"the script could not be started: {type(exc).__name__}: {exc}"
        return run
    return read_run(report, proc.returncode, proc.stderr)


def read_run(report: Path, returncode: int, stderr: str | None) -> ScriptRun:
    """One finished script's report, or why there is none."""
    run = ScriptRun()
    if report.is_file():
        try:
            run.report = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            run.failure = f"the script's report could not be read: {exc}"
            return run
    if returncode != 0:
        said = run.report.get("error") or _tail(stderr) or "no stderr"
        run.failure = f"the script exited {returncode}: {said}"
    elif not report.is_file():
        run.failure = "the script wrote no report" + (f": {_tail(stderr)}" if _tail(stderr) else "")
    return run


STOPPED = "the launcher reports no exit for this file: it was stopped or never started"


def usable_cpus() -> int | None:
    """The CPUs this process may use: `os.process_cpu_count` where it exists
    (3.13+), else the affinity mask's size where there is one, else
    `os.cpu_count`. `None` when none of them can say."""
    counter = getattr(os, "process_cpu_count", None)
    if counter is not None:
        return counter()
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count()


def side_by_side(files: Sequence[Path], threads: int, cpus: int | None) -> bool:
    """Whether the engine files can be embedded at once: two of them, and the CPUs for both.

    Each run has its own XNNPack cache directory (`run_engines`), so two runs
    never write one cache path -- the hazard measured as "could not verify
    buffer list mapping" in an earlier measurement
    (MEASUREMENTS.md, "EmbeddingGemma 2: many labels"). A machine
    without `threads` CPUs per file would have the runs compete, and each
    file's time per text is reported.
    """
    if len(files) < 2:
        return False
    return (cpus or 1) >= threads * len(files)


def run_engines(
    env: envs.StageEnv,
    sides: Sequence[tuple[str, Path]],
    common: dict[str, Any],
    work: Path,
    parallel: bool,
    timeout: int,
) -> tuple[dict[str, ScriptRun], str | None]:
    """Embed each (name, file) with `ENGINE_SCRIPT` through one `ENGINES_SCRIPT` child.

    Each side's engine gets its own XNNPack cache directory, `work/cache-<name>`,
    created here, so no cache is written beside the file and no two runs share
    one. Only a failure of a side in `STOP_ON` stops the others.

    Returns each side's run and, when the launcher itself gave no result, why.
    A side the launcher stopped or never started has the failure `STOPPED`.
    """
    script = work / "engine.py"
    script.write_text(ENGINE_SCRIPT, encoding="utf-8")
    runs, reports = [], {}
    for name, model in sides:
        spec_path, report = work / f"{name}.json", work / f"{name}.out"
        cache = work / f"cache-{name}"
        cache.mkdir()
        spec_path.write_text(
            json.dumps(
                {
                    **common,
                    "model": str(model),
                    "cache_dir": str(cache),
                    "max_input_length": MAX_INPUT_TOKENS,
                    "out": str(work / f"{name}.npy"),
                    "report": str(report),
                }
            ),
            encoding="utf-8",
        )
        runs.append({"name": name, "spec": str(spec_path), "stderr": str(work / f"{name}.err")})
        reports[name] = report
    launched = run_script(
        env,
        ENGINES_SCRIPT,
        {
            "script": str(script),
            "runs": runs,
            "parallel": parallel,
            "stop_on": list(STOP_ON),
            "stop_s": STOP_TIMEOUT_S,
        },
        work,
        "engines",
        timeout,
    )
    if launched.failure is not None:
        return {}, launched.failure
    out: dict[str, ScriptRun] = {}
    for name, _ in sides:
        entry = _mapping(_mapping(launched.report.get("runs")).get(name))
        code = entry.get("returncode")
        if not isinstance(code, int):
            out[name] = ScriptRun(failure=STOPPED)
            continue
        err = work / f"{name}.err"
        stderr = err.read_text(encoding="utf-8", errors="replace") if err.is_file() else None
        out[name] = read_run(reports[name], code, stderr)
    return out, None


# ---------------------------------------------------------------------------
# The arithmetic the host does: pairing, and the gate
# ---------------------------------------------------------------------------


def _bools(values: Sequence[Any]) -> list[bool]:
    return [bool(x) for x in values]


def kshot_costs(correct: dict[str, Any], a: str, b: str, k: int) -> list[Difference]:
    """`a` minus `b` on the k-shot centroid decider, one paired difference per draw."""
    left, right = correct[a][f"centroid_k{k}"], correct[b][f"centroid_k{k}"]
    if len(left) != len(right) or not left:
        raise ValueError(f"{a} has {len(left)} draws of the {k}-shot decider, {b} {len(right)}")
    return [paired_difference(_bools(x), _bools(y)) for x, y in zip(left, right, strict=True)]


def centroid_cost(correct: dict[str, Any], a: str, b: str) -> Difference:
    """`a` minus `b` on the centroid of every training row, paired on the held-out rows."""
    return paired_difference(_bools(correct[a]["centroid"]), _bools(correct[b]["centroid"]))


def gate(costs: Sequence[Difference], threshold: float) -> tuple[str, Check]:
    """`verify._gate`'s reading of an interval against a threshold, over every draw.

    Failed when any draw's interval lies entirely above the threshold; passed
    when every draw's lies at or under it; inconclusive otherwise. Returns a
    `verify.Status` value and the check.
    """
    if not costs:
        raise ValueError("no draws to gate")
    observed = {
        "threshold": threshold,
        "draws": [{"value": d.value, "ci95": d.ci95} for d in costs],
    }
    shown = ", ".join(f"{d.value:+.4f} ±{d.ci95:.4f}" for d in costs)
    above = [i for i, d in enumerate(costs) if d.value - d.ci95 > threshold]
    if above:
        return "failed_gate", Check.failed(
            GATE_CHECK,
            f"the conversion cost on the {GATED_K}-shot centroid decider lies entirely above "
            f"{threshold:.4f} in draw(s) {above} ({shown})",
            observed=observed,
        )
    if all(d.value + d.ci95 <= threshold for d in costs):
        return "passed", Check.passed(
            GATE_CHECK,
            f"the conversion cost on the {GATED_K}-shot centroid decider is within "
            f"{threshold:.4f} in every draw ({shown})",
            observed=observed,
        )
    return "inconclusive", Check.unchecked(
        GATE_CHECK,
        f"at least one draw's interval straddles the threshold {threshold:.4f} ({shown}); these "
        "rows cannot resolve the gate",
        observed=observed,
    )


def mean_value(costs: Sequence[Difference]) -> float:
    return sum(d.value for d in costs) / len(costs)


def labels_short_of(train: Sequence[LabelledText], labels: Sequence[str], k: int) -> list[str]:
    """The labels with fewer than `k` training rows: a k-shot draw cannot be made for them."""
    counts = dict.fromkeys(labels, 0)
    for row in train:
        counts[row.label] = counts.get(row.label, 0) + 1
    return [label for label in labels if counts[label] < k]


def truncated(run: ScriptRun, texts: int) -> str | None:
    """Why the reference cannot stand for the engine on these texts, when its report says so.

    `REFERENCE_SCRIPT` counts the texts whose tokens, with the prompt and the
    tokenizer's specials, exceed the checkpoint's `max_seq_length`, as
    `prepare.measure_embedded_lengths` counts them: sentence-transformers cuts
    those, the engine (`MAX_INPUT_TOKENS`) does not, so the two sides would embed
    different texts. A run that failed is left to its own check.
    """
    if run.failure is not None:
        return None
    tokens = _mapping(run.report.get("tokens"))
    over = tokens.get("over_max_seq_length")
    if not isinstance(over, int) or isinstance(over, bool):
        return "the reference script reports no token counts, so truncation cannot be ruled out"
    if over:
        return (
            f"{over} of {texts} texts exceed the reference's max_seq_length of "
            f"{run.report.get('max_seq_length')} tokens with the prompt (longest "
            f"{tokens.get('max')}); sentence-transformers cuts them and the engine does not, so "
            "the two sides would embed different texts. `litetune prepare --tokenizer` measures "
            "this before training"
        )
    return None


@dataclass(frozen=True)
class ConvertFacts:
    """What `convert.json` records about the file it wrote: `None` where it records nothing."""

    weights: str | None
    # `storage.hash_file` of the tuned checkpoint's model.safetensors.
    checkpoint_sha256: str | None
    decision_sha256: str | None


def convert_facts(model: Path) -> tuple[ConvertFacts | None, str]:
    """What `convert` recorded for `model`, and where that came from (or why nothing).

    Read from `convert.json` beside the file, and only when that report names
    this file's sha256.
    """
    report_path = Path(model).parent / REPORT_NAME
    if not report_path.is_file():
        return None, f"no {REPORT_NAME} beside {model}"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", report_path, exc)
        return None, f"{report_path} could not be read: {exc}"
    if not isinstance(report, dict):
        return None, f"{report_path} is not a JSON object"
    graft = report.get("graft")
    if report.get("schema") != EMBEDDING_CONVERT_SCHEMA or not isinstance(graft, dict):
        return None, f"{report_path} is not an embedding convert report with a graft"
    if graft.get("sha256") != hash_file(model):
        return None, f"{report_path} describes another file than {model}"
    checkpoint = _mapping(report.get("checkpoint"))

    def text(value: object) -> str | None:
        return value if isinstance(value, str) else None

    return (
        ConvertFacts(
            weights=text(graft.get("weights")),
            checkpoint_sha256=text(checkpoint.get("sha256")),
            decision_sha256=text(checkpoint.get("decision_sha256")),
        ),
        str(report_path),
    )


def weights_of(model: Path) -> tuple[str | None, str]:
    """The weight mode `convert` recorded for `model`, and where that came from."""
    facts, source = convert_facts(model)
    return (facts.weights if facts is not None else None), source


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class _State:
    checks: CheckSet = field(default_factory=lambda: CheckSet(name="verify:embedding"))
    limitations: list[str] = field(default_factory=list)

    def add(self, check: Check, events: EventStream) -> Check:
        self.checks.add(check)
        events.check(check)
        return check

    def limitation(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)


def _record_problem(record: EmbeddingRecord) -> str | None:
    if (record.base_model, record.base_model_revision) != (EMBEDDING_MODEL, EMBEDDING_REVISION):
        return (
            f"the checkpoint records {record.base_model} at {record.base_model_revision}, not "
            f"{EMBEDDING_MODEL} at {EMBEDDING_REVISION}"
        )
    if record.prompt_name != PROMPT_NAME or record.prefix != DECIDER_PREFIX:
        return (
            f"the checkpoint was trained with prompt {record.prompt_name!r} = {record.prefix!r}, "
            f"not {PROMPT_NAME!r} = {DECIDER_PREFIX!r}, the decider's prefix"
        )
    return None


def _read_states(path: Path, decision: Decision, what: str) -> list[LabelledText]:
    if not Path(path).is_file():
        raise EmbeddingTuneError(f"the {what} split {path} does not exist")
    try:
        rows = read_rows(Path(path))
    except PrepareError as exc:
        raise EmbeddingTuneError(str(exc)) from exc
    states = states_of(rows, decision, Path(path))
    if not states:
        raise EmbeddingTuneError(f"the {what} split {path} has no rows")
    return states


def verify_embedding(
    job: EmbeddingVerify, record: EmbeddingRecord, events: EventStream
) -> tuple[str, dict[str, Any]]:
    """Measure an embedding checkpoint and its file. Returns (status, manifest fields).

    The status is one of `verify.Status`'s values; `verify` wraps the fields in
    its manifest. Whatever is wrong with the request, the files, the rows or a
    script's run comes back as a failed check and a status, not as an exception.
    """
    state = _State()
    threshold = job.threshold
    fields: dict[str, Any] = {
        "scorer": {"name": SCORER, "means": SCORER_MEANS},
        "gates": [],
    }

    def finish(status: str) -> tuple[str, dict[str, Any]]:
        fields["checks"] = state.checks.as_dict()
        fields["limitations"] = state.limitations
        return status, fields

    def refuse(name: str, detail: str, observed: Any = None) -> tuple[str, dict[str, Any]]:
        state.add(Check.failed(name, detail, observed=observed), events)
        return finish("failed_harness")

    # -- the request -------------------------------------------------------------
    if (problem := _record_problem(record)) is not None:
        return refuse(RECORD_CHECK, problem)
    if job.reference_role != "float_twin":
        return refuse(
            RECORD_CHECK,
            f"--reference-role {job.reference_role}: an embedding checkpoint is measured against "
            "itself in float32, the role float_twin",
        )
    if job.limit is not None:
        return refuse(
            RECORD_CHECK,
            "--limit does not apply to an embedding checkpoint: every train and held-out row is "
            "embedded, because the decider is fitted on the training rows",
        )
    if job.backend != "cpu":
        return refuse(
            CANDIDATE_CHECK,
            f"--backend {job.backend}: the embedding path is measured on the CPU backend only",
        )
    if job.not_applicable:
        return refuse(
            RECORD_CHECK,
            f"{', '.join(job.not_applicable)} do(es) not apply to an embedding checkpoint, which "
            "is measured by the centroid decider against its own float32 reference; refused "
            "rather than ignored",
        )
    if not 0.0 <= threshold < 1.0:
        return refuse(GATE_CHECK, f"--max-conversion-cost {threshold} is not in [0, 1)")
    try:
        decision = decision_from(record.decision, "the reference's litetune.json")
    except DecisionError as exc:
        return refuse(RECORD_CHECK, str(exc))
    if record.decision_sha256 != decision.digest():
        return refuse(
            RECORD_CHECK,
            f"the reference's litetune.json records a decision whose digest is not its own: "
            f"{record.decision_sha256} against {decision.digest()}",
        )
    if decision.type != "choice" or decision.none_option is not None:
        return refuse(
            RECORD_CHECK,
            "the embedding decider is measured for a choice question without a none_option; "
            f"this decision is a {decision.type} question"
            + (f" with none_option {decision.none_option!r}" if decision.none_option else ""),
        )
    labels = list(record.labels)
    if sorted(labels) != sorted(decision.ids):
        return refuse(
            RECORD_CHECK,
            f"the checkpoint's labels {labels[:6]} are not the decision's options",
        )
    if not Path(job.model).is_file():
        return refuse(CANDIDATE_CHECK, f"--model {job.model}: no such file")
    facts, weights_source = convert_facts(job.model)
    weights = facts.weights if facts is not None else None
    if facts is None:
        state.limitation(NOT_TIED.format(reason=weights_source))
    else:
        if facts.decision_sha256 is not None and facts.decision_sha256 != record.decision_sha256:
            return refuse(
                RECORD_CHECK,
                f"{weights_source} says the file was converted from a checkpoint of decision "
                f"{facts.decision_sha256}; the reference's litetune.json records "
                f"{record.decision_sha256}",
            )
        safetensors = Path(job.reference) / "model.safetensors"
        if facts.checkpoint_sha256 is None:
            state.limitation(
                NOT_TIED.format(reason=f"{weights_source} records no checkpoint sha256")
            )
        elif not safetensors.is_file():
            state.limitation(NOT_TIED.format(reason=f"there is no {safetensors} to compare"))
        elif (found := hash_file(safetensors)) != facts.checkpoint_sha256:
            return refuse(
                RECORD_CHECK,
                f"{weights_source} says the file was grafted from a model.safetensors of "
                f"{facts.checkpoint_sha256}; {safetensors} is {found}: the file was not "
                "converted from this reference",
            )
        if facts.decision_sha256 is None:
            state.limitation(NOT_TIED.format(reason=f"{weights_source} records no decision sha256"))
    state.add(
        Check.passed(
            RECORD_CHECK,
            f"{EMBEDDING_MODEL} fine-tune, {len(labels)} options, prefix {record.prefix!r}",
        ),
        events,
    )
    fields["embedding"] = {
        "prefix": record.prefix,
        "prompt_name": record.prompt_name,
        "weights": weights,
        "weights_source": weights_source,
        # What convert.json says the file was grafted from; None when it says nothing.
        "checkpoint_sha256": facts.checkpoint_sha256 if facts is not None else None,
        "labels": labels,
        "decision_sha256": record.decision_sha256,
        "ks": list(KS),
        "draws": DRAWS,
        "gated_on": f"centroid_k{GATED_K}",
        "threads": job.threads,
    }
    if weights is None:
        state.limitation(f"the weight mode of the file is not known: {weights_source}")

    # -- the rows ----------------------------------------------------------------
    try:
        train = _read_states(job.train_path, decision, "training")
        heldout = _read_states(job.heldout, decision, "held-out")
    except EmbeddingTuneError as exc:
        return refuse(ROWS_CHECK, str(exc))
    short = labels_short_of(train, labels, max(KS))
    if short:
        return refuse(
            ROWS_CHECK,
            f"{len(short)} option(s) have fewer than {max(KS)} training rows "
            f"({', '.join(map(repr, short[:12]))}); the {max(KS)}-shot decider the gate reads "
            "cannot be drawn",
        )
    train_sha256, heldout_sha256 = hash_file(job.train_path), hash_file(job.heldout)
    if train_sha256 == heldout_sha256:
        return refuse(
            ROWS_CHECK,
            f"the training split {job.train_path} and the held-out split {job.heldout} are the "
            f"same content ({train_sha256}); the decider would be measured on its own examples",
        )
    if record.train_sha256 is not None and record.train_sha256 != train_sha256:
        state.limitation(
            f"the training split here is {train_sha256}; the one tune read was "
            f"{record.train_sha256}"
        )
    index = {label: i for i, label in enumerate(labels)}
    fields["data"] = {
        "train": {"path": str(job.train_path), "rows": len(train), "sha256": train_sha256},
        "heldout": {"path": str(job.heldout), "rows": len(heldout), "sha256": heldout_sha256},
    }
    for what, rows in (("train", train), ("heldout", heldout)):
        trained = record.rows.get(what)
        if trained is not None and trained != len(rows):
            state.limitation(
                f"the {what} split here has {len(rows)} rows; the one tune recorded had {trained}"
            )
    state.add(
        Check.passed(ROWS_CHECK, f"{len(train)} training and {len(heldout)} held-out states"),
        events,
    )

    # -- Google's file, when asked for ---------------------------------------------
    google: Path | None = None
    if job.base_litertlm is not None or job.with_base:
        if job.base_litertlm is not None:
            google = Path(job.base_litertlm)
            if not google.is_file():
                return refuse(BASE_CHECK, f"--base-litertlm {google}: no such file")
        else:
            try:
                if job.auto_provision:
                    job.export_env.provision(events=events)
                google = download(job.export_env, [GOOGLE_HUB_FILE])[0]
            except (DownloadFailed, RuntimeError, OSError) as exc:
                logger.exception("could not fetch Google's file")
                return refuse(BASE_CHECK, f"Google's file could not be fetched: {exc}")
        if (problem := check_google_file(google)) is not None:
            return refuse(BASE_CHECK, problem)
        state.add(Check.passed(BASE_CHECK, f"{google}, Google's file by its sha256"), events)
    else:
        state.limitation(NO_BASE)

    # -- the embeddings, one process per file ---------------------------------------
    texts = [row.text for row in (*train, *heldout)]
    budget = STARTUP_TIMEOUT_S + ROW_TIMEOUT_S * len(texts)
    with tempfile.TemporaryDirectory(prefix="litetune-embedding-") as tmp:
        work = Path(tmp)
        texts_path = work / "texts.json"
        texts_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        common = {"texts": str(texts_path), "threads": job.threads, "prefix": record.prefix}
        engine_sides: list[tuple[str, Path]] = [("candidate", Path(job.model))]
        if google is not None:
            engine_sides.append(("base", google))
        parallel = side_by_side([f for _, f in engine_sides], job.threads, usable_cpus())
        fields["embedding"]["engines_side_by_side"] = parallel

        for env, check_name in (
            (job.train_env, REFERENCE_CHECK),
            (job.runtime_env, CANDIDATE_CHECK),
        ):
            if not job.auto_provision:
                continue
            try:
                env.provision(events=events)
            except Exception as exc:  # noqa: BLE001 - a failed environment is a harness failure
                logger.exception("could not provision environment %r", env.name)
                return refuse(
                    check_name, f"the {env.name} environment could not be provisioned: {exc}"
                )

        vectors: dict[str, str] = {}
        harness: dict[str, Any] = {}

        def accept(name: str, check_name: str, run: ScriptRun) -> str | None:
            """Record one side's vectors; the failure, when they are not there."""
            out = work / f"{name}.npy"
            if run.failure is not None:
                return run.failure
            if run.report.get("rows") != len(texts):
                return f"the script reports {run.report.get('rows')} vectors for {len(texts)} texts"
            if not out.is_file():
                return f"the script reports {len(texts)} vectors but wrote no {out.name}"
            harness[name] = run.report
            vectors[name] = str(out)
            state.add(
                Check.passed(check_name, f"{len(texts)} vectors of {run.report.get('dim')} values"),
                events,
            )
            return None

        events.note(f"embedding {len(texts)} texts: reference", env=job.train_env.name)
        reference_run = run_script(
            job.train_env,
            REFERENCE_SCRIPT,
            {
                **common,
                "model": job.reference,
                "prompt_name": record.prompt_name,
                "batch_size": REFERENCE_BATCH,
                "out": str(work / "reference.npy"),
            },
            work,
            "reference",
            budget,
        )
        if (problem := truncated(reference_run, len(texts))) is not None:
            return refuse(REFERENCE_CHECK, problem)
        if (problem := accept("reference", REFERENCE_CHECK, reference_run)) is not None:
            return refuse(REFERENCE_CHECK, problem)

        names = " and ".join(name for name, _ in engine_sides)
        how = ", side by side" if parallel else ""
        events.note(f"embedding {len(texts)} texts: {names}{how}", env=job.runtime_env.name)
        engine_runs, launch_failure = run_engines(
            job.runtime_env,
            engine_sides,
            common,
            work,
            parallel,
            budget if parallel else budget * len(engine_sides),
        )
        if launch_failure is not None:
            return refuse(CANDIDATE_CHECK, launch_failure)
        # The candidate first: when it failed, Google's file was stopped
        # because of it, and the failure to report is the candidate's.
        if (problem := accept("candidate", CANDIDATE_CHECK, engine_runs["candidate"])) is not None:
            return refuse(CANDIDATE_CHECK, problem)
        with_base = google is not None
        if with_base and (problem := accept("base", BASE_EMBED_CHECK, engine_runs["base"])):
            state.add(Check.unchecked(BASE_EMBED_CHECK, problem), events)
            state.limitation(BASE_FAILED.format(reason=problem))
            with_base = False
        fields["harness"] = harness

        pairs = [["reference", "candidate"]] + ([["candidate", "base"]] if with_base else [])
        evaluated = run_script(
            job.train_env,
            EVAL_SCRIPT,
            {
                "vectors": vectors,
                "train_labels": [index[row.label] for row in train],
                "heldout_labels": [index[row.label] for row in heldout],
                "n_labels": len(labels),
                "ks": list(KS),
                "draws": DRAWS,
                "decay": LINEAR_DECAY,
                "pairs": pairs,
            },
            work,
            "evaluate",
            EVAL_TIMEOUT_S,
        )
    if evaluated.failure is not None:
        return refuse(EVAL_CHECK, evaluated.failure)
    report = evaluated.report
    harness["evaluation"] = report.get("versions")
    try:
        accuracy = report["accuracy"]
        correct = report["correct"]
        k_costs = kshot_costs(correct, "reference", "candidate", GATED_K)
        all_cost = centroid_cost(correct, "reference", "candidate")
    except (KeyError, TypeError, ValueError) as exc:
        logger.exception("the evaluation's report is incomplete")
        return refuse(EVAL_CHECK, f"the evaluation's report is incomplete: {exc}")
    state.add(
        Check.passed(
            EVAL_CHECK,
            f"reference {accuracy['reference'].get(f'centroid_k{GATED_K}')}, candidate "
            f"{accuracy['candidate'].get(f'centroid_k{GATED_K}')} on the {GATED_K}-shot decider",
        ),
        events,
    )
    fields["measurements"] = {name: {"available": True, **accuracy[name]} for name in accuracy}
    candidate_report = harness.get("candidate") or {}
    fields["measurements"]["candidate"]["engine"] = {
        "engine": "litert-lm",
        "api": "EmbeddingEngine",
        "backend": "cpu",
        "runtime_version": (candidate_report.get("runtime") or {}).get("litert-lm"),
        # `evaluate.cpu_report`, run by the engine script on the CPU it computed on.
        "cpu": _mapping(candidate_report.get("cpu")),
        "threads": job.threads,
        "settings": _mapping(candidate_report.get("settings")),
        "ms_per_text_median": candidate_report.get("ms_median"),
    }
    fields["quality"] = {
        "available": True,
        "cosine": report.get("cosine", {}),
        "conversion_cost": {
            "measures": "reference minus candidate accuracy, paired on the held-out rows",
            f"centroid_k{GATED_K}": {
                "mean": round(mean_value(k_costs), 6),
                "draws": [d.as_dict() for d in k_costs],
            },
            "centroid": all_cost.as_dict(),
        },
    }
    if with_base:
        try:
            gain_k = kshot_costs(correct, "candidate", "base", GATED_K)
            gain_all = centroid_cost(correct, "candidate", "base")
        except (KeyError, TypeError, ValueError) as exc:
            logger.exception("the evaluation's report has no base side")
            return refuse(EVAL_CHECK, f"the evaluation's report has no base side: {exc}")
        fields["quality"]["gain_over_base"] = {
            "measures": "candidate minus Google's file, paired on the held-out rows; not gated",
            f"centroid_k{GATED_K}": {
                "mean": round(mean_value(gain_k), 6),
                "draws": [d.as_dict() for d in gain_k],
            },
            "centroid": gain_all.as_dict(),
        }

    # -- the gate --------------------------------------------------------------------
    status, check = gate(k_costs, threshold)
    state.add(check, events)
    fields["gates"].append(
        {"name": "conversion_cost", "outcome": check.outcome.value, "detail": check.detail}
        | {"threshold": threshold, "mean": round(mean_value(k_costs), 6)}
    )
    state.limitation(CPU_ONLY.format(threads=job.threads))
    if parallel:
        state.limitation(SIDE_BY_SIDE.format(threads=job.threads))
    state.limitation(FITTED_ON_EXAMPLES.format(ks=" and ".join(f"k={k}" for k in KS), draws=DRAWS))
    state.limitation(REFERENCE_ON_CPU)
    state.limitation(LINEAR_NOT_GATED)
    state.limitation(DESCRIPTIONS_UNUSED)
    state.limitation(CALIBRATION_UNMEASURED)
    if job.max_conversion_cost is None:
        state.limitation(
            f"no --max-conversion-cost was given; the gate used {DEFAULT_MAX_CONVERSION_COST}, "
            "the embedding path's default"
        )
    return finish(status)


def _mapping(value: object) -> dict[str, Any]:
    """A mapping, whatever arrived: `summarise` reads a document it did not build."""
    return value if isinstance(value, dict) else {}


def summarise(manifest: dict[str, Any]) -> list[str]:
    """An embedding checkpoint's result: accuracy on each side, the cost, the gate."""
    lines = [f"status: {manifest.get('status')}"]
    embedding = _mapping(manifest.get("embedding"))
    if embedding:
        lines.append(
            f"  embedding decider, {len(embedding.get('labels') or [])} options, weights "
            f"{embedding.get('weights') or 'unknown'}, prefix {embedding.get('prefix')!r}"
        )
    measured = _mapping(manifest.get("measurements"))
    key = f"centroid_k{GATED_K}"
    for name in ("reference", "candidate", "base"):
        point = _mapping(measured.get(name))
        if not point:
            continue
        k = _mapping(point.get(key))
        lines.append(
            f"  {name}: centroid {point.get('centroid')}, {key} mean {k.get('mean')} "
            f"(min {k.get('min')}), linear {point.get('linear')}"
        )
    cosine = _mapping(
        _mapping(_mapping(manifest.get("quality")).get("cosine")).get("reference|candidate")
    )
    if cosine:
        lines.append(
            f"  cosine reference|candidate: mean {cosine.get('mean')}, min {cosine.get('min')}"
        )
    checks = manifest.get("checks")
    for check in checks if isinstance(checks, list) else []:
        check = _mapping(check)
        if check and check.get("outcome") != Outcome.PASSED.value:
            lines.append(f"  {check.get('outcome')}: {check.get('name')} — {check.get('detail')}")
        elif check.get("name") == GATE_CHECK:
            lines.append(f"  {check.get('detail')}")
    for text in manifest.get("limitations") or []:
        lines.append(f"  note: {text}")
    return lines
