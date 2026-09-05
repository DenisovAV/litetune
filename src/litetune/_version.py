"""The package version, in a leaf module.

Not in `__init__.py`: `verify` and `manifest` record it in every artifact they
write, and importing it from the package root made those stages depend on the
root's contents. A future convenience re-export there would then become a
dependency of every manifest write, and a missing `__init__.py` -- which is
exactly what an incomplete commit produces -- turned a function documented as
"never raises" into one that raises at import.

`pyproject.toml` reads this file, so the number lives in one place.
"""

__version__ = "0.1.5"
