"""Fine-tune a small model, convert it to run on a phone, and know what the conversion cost you.

Deliberately thin. Re-exporting the stage modules would pull `yaml`, the job
spec and the whole graph into `import litetune`, and the fast import is worth
keeping; what is here is the vocabulary those stages report in.

The file exists mainly so this is a *regular* package. Without it the
distribution installs as an implicit namespace package, and a directory named
`litetune` earlier on `sys.path` silently replaces individual submodules --
`litetune.verify` becomes someone else's code while `litetune.checks` stays
ours, with no error. For a tool whose subject is confident answers from
unverified sources, that is the wrong default to ship.
"""

from __future__ import annotations

from litetune._version import __version__
from litetune.checks import Check, CheckSet, Outcome
from litetune.metrics import Difference, Proportion, Unavailable

__all__ = [
    "Check",
    "CheckSet",
    "Difference",
    "Outcome",
    "Proportion",
    "Unavailable",
    "__version__",
]
