"""A JSON file's text, in the encodings Windows writes it in."""

from __future__ import annotations

import codecs
import json

import pytest

from litetune import jsonfile

PAYLOAD = {"status": "passed", "note": "перевод — ✅"}
TEXT = json.dumps(PAYLOAD, ensure_ascii=False)


@pytest.mark.parametrize(
    "raw",
    [
        TEXT.encode("utf-8"),
        codecs.BOM_UTF8 + TEXT.encode("utf-8"),
        # What `>` writes in Windows PowerShell 5.1: UTF-16LE with its mark.
        codecs.BOM_UTF16_LE + TEXT.encode("utf-16-le"),
        codecs.BOM_UTF16_BE + TEXT.encode("utf-16-be"),
        # UTF-32LE's mark begins with UTF-16LE's, so the order of the checks is
        # what this one is about.
        codecs.BOM_UTF32_LE + TEXT.encode("utf-32-le"),
        codecs.BOM_UTF32_BE + TEXT.encode("utf-32-be"),
    ],
    ids=["utf-8", "utf-8-bom", "utf-16-le", "utf-16-be", "utf-32-le", "utf-32-be"],
)
def test_every_marked_encoding_reads_as_the_same_text(raw):
    assert jsonfile.decode(raw) == TEXT
    assert json.loads(jsonfile.decode(raw)) == PAYLOAD


def test_bytes_that_are_not_utf8_and_carry_no_mark_are_refused():
    """Nothing is guessed from unmarked bytes: cp1252's `—` is not UTF-8."""
    with pytest.raises(UnicodeDecodeError):
        jsonfile.decode('{"note": "—"}'.encode("cp1252"))
