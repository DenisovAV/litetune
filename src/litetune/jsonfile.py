"""The text of a JSON file a user hands litetune, in the encodings it arrives in.

`litetune verify --json > manifest.json` is how a manifest reaches `bundle`, and
in Windows PowerShell 5.1 "Out-File and the redirection operators > and >>
create UTF-16LE", with a byte-order mark: "any Unicode encoding, except UTF7,
always creates a BOM" (about_Character_Encoding, PowerShell 5.1,
learn.microsoft.com). Read as UTF-8 that file raised `UnicodeDecodeError` -- a
`ValueError`, not the `JSONDecodeError` or `OSError` the readers caught -- so it
escaped them and ended the run without naming the file.

A byte-order mark decides the encoding where there is one, and UTF-8 is read
everywhere else. Nothing is guessed from bytes without a mark: a file that is
neither is refused, by the caller, with its name.

A leaf module, like `declarations`, so any stage can import it.
"""

from __future__ import annotations

import codecs

# Longest first: UTF-32's little-endian mark begins with UTF-16's. The `utf-16`
# and `utf-32` codecs read the mark to choose the byte order and drop it, as
# `utf-8-sig` drops UTF-8's.
_MARKS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# What each codec above is called where a refusal names it.
_NAMED = {"utf-32": "UTF-32", "utf-8-sig": "UTF-8 with a byte-order mark", "utf-16": "UTF-16"}

UNREADABLE = "is not text in UTF-8, or in UTF-16 or UTF-32 with a byte-order mark"


def marked_encoding(raw: bytes) -> str | None:
    """The encoding a byte-order mark at the start of `raw` names, or `None` where there is none.

    For the readers that take UTF-8 alone and refuse anything else by name.
    """
    for mark, codec in _MARKS:
        if raw.startswith(mark):
            return _NAMED[codec]
    return None


def decode(raw: bytes) -> str:
    """The text in `raw`. Raises `UnicodeDecodeError` for bytes that are not text in those."""
    for mark, codec in _MARKS:
        if raw.startswith(mark):
            return raw.decode(codec)
    return raw.decode("utf-8")
