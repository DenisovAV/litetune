"""The changelog and the version it describes."""

from __future__ import annotations

import re
from pathlib import Path

from litetune._version import __version__

_ROOT = Path(__file__).resolve().parent.parent
# The version at the start of a heading's text, as
# `website/lib/changelog/changelog_page.dart` matches it: an optional epoch,
# numbers and dots, then the rest of the word.
_VERSION_AT_START = re.compile(r"^((?:\d+!)?\d+(?:\.\d+)*\S*)")


def _fence_marker(stripped: str) -> re.Match[str] | None:
    """The fence a line opens or closes, if it is one.

    A backtick fence's info string may not contain a backtick, so ```` ```a`b ````
    is a paragraph, not a fence -- which is how one writes about a fence in a
    changelog. Treating it as one hid every heading after it from this file's
    reader while the page went on rendering them.
    """
    marker = re.match(r"`{3,}|~{3,}", stripped)
    if marker and marker.group(0)[0] == "`" and "`" in stripped[marker.end() :]:
        return None
    return marker


# A code span: a run of backticks closed by a run of exactly that length. The
# length matters -- ```` ```<img src=x>` ```` closes neither run, so CommonMark
# renders the tag between them, while a reader that let a shorter run close a
# longer one deleted the line before looking at it.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(?:.*?)(?<!`)\1(?!`)", re.S)


def _outside_fences(text: str) -> list[tuple[str, str]]:
    """(previous line, line) pairs outside fenced code, as CommonMark fences it.

    A fence opens with three or more backticks or tildes indented at most three
    spaces, and closes only with a run of the same character at least as long
    and nothing after it. Toggling on any fence-looking line instead let a `~~~`
    inside a backtick block end it here while the page's parser stayed inside.
    """
    pairs: list[tuple[str, str]] = []
    fence: tuple[str, int] | None = None
    previous = ""
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        marker = _fence_marker(stripped) if len(line) - len(stripped) <= 3 else None
        if fence is None:
            if marker:
                fence = (marker.group(0)[0], len(marker.group(0)))
            else:
                pairs.append((previous, line))
            previous = line
            continue
        if (
            marker
            and marker.group(0)[0] == fence[0]
            and len(marker.group(0)) >= fence[1]
            and not stripped[len(marker.group(0)) :].strip()
        ):
            fence = None
        previous = line
    return pairs


def _releases() -> list[str]:
    """The versions of the release headings, newest first.

    The page matches a heading's *text* after markdown has parsed it. This reads
    `## ` headings outside fenced code with emphasis, code spans and link syntax
    taken off, which is how the page sees every heading this file writes --
    `test_release_headings_are_written_with_hashes` keeps it to that form.
    Matching the raw line instead made `## **1.0**` a release to the page and not
    to this test.
    """
    versions: list[str] = []
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for _, line in _outside_fences(text):
        # Up to three spaces of indent still opens a heading, to markdown and so
        # to the page; trailing `#`s are a closing marker, not text.
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3 or not stripped.startswith("## "):
            continue
        heading = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped[3:])
        heading = re.sub(r"[*_`]", "", heading).strip()
        # A closing run of `#`s, which CommonMark requires a space before.
        heading = re.sub(r"\s+#+$", "", heading).strip()
        match = _VERSION_AT_START.match(heading)
        if match:
            versions.append(match.group(1))
    return versions


def test_release_headings_are_written_with_hashes():
    """No underlined headings, which the page renders and `_releases` does not read.

    Markdown also makes a heading of a line followed by `---` or `===`. The page
    would give such a release an id and this test would not see it, so the two
    would disagree about which entry is the newest. The file does not use that
    form; this keeps it so. A `---` rule after a blank line is not a heading and
    is allowed.
    """
    # Only a paragraph can be underlined into a heading: after a heading, a
    # list item, a quote or a fence, `---` is a thematic break.
    underlined = [
        previous
        for previous, line in _outside_fences((_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
        if re.fullmatch(r" {0,3}(-+|=+)\s*", line)
        and previous.strip()
        and not re.match(r" {0,3}([#>]|[-*+] |\d+[.)] |`{3,}|~{3,})", previous)
    ]
    assert not underlined, f"write these as `## ` headings: {underlined}"


def test_the_changelog_opens_with_the_version_being_shipped():
    """A version bump comes with its changelog entry, in the same change.

    The website links its version badge to `/changelog#v<version>`, and
    `website/check-build.sh` refuses a build without that entry -- but only
    where the Dart toolchain is, which is not where `pytest` runs locally or in
    `scripts/ci-local.sh`. The newest entry rather than any entry, because a
    release whose notes went under the previous heading reads as if it changed
    nothing.
    """
    releases = _releases()
    assert releases, "CHANGELOG.md has no `## <version>` headings"
    assert (
        releases[0] == __version__
    ), f"_version.py says {__version__} and the newest CHANGELOG.md entry is {releases[0]}"


def test_the_changelog_holds_no_raw_html():
    """The file stays plain markdown: no tags, no exotic link destinations.

    What makes the page safe is the check in
    `website/lib/changelog/changelog_page.dart`, which refuses to render a
    changelog whose HTML is not what a release entry is written with, links
    included. This is the tidiness rule in front of it, and it is
    deliberately not a second implementation of markdown: reviews got a
    `javascript:` link past one reader by ending a code span with a longer
    backtick run than it opened with, and past another by spelling the scheme
    `java&#x73;cript:`. A rule that has to restate CommonMark to be correct is
    the wrong shape for a test; an allowlist at the render step is not.

    So: nothing that opens a tag outside a code span, and no destination
    spelled with a scheme the filter would drop. Autolinks are markdown's own
    `<scheme:...>` form and become links.
    """
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    offenders: list[str] = []
    schemes: list[str] = []
    # Block by block: a code span never spans a blank line, and a reader that
    # let one do so deleted whole paragraphs -- with a tag in them -- between
    # two stray backticks, which is how a review got a meta refresh past this.
    # Inside one block it may span lines, so the block is searched whole.
    for block in re.split(r"\n[ \t]*\n", text):
        outside_code = _CODE_SPAN.sub("", block)
        # Autolinks become links, so they are not markup -- but only the
        # schemes the page will keep, which is the three the build's own check
        # keeps. Striking any other here, `<javascript:...>` or `<ftp:...>`,
        # would hide from the scheme check below a link the build refuses.
        outside_code = re.sub(
            r"<(?:(?:https?|mailto):[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>",
            "",
            outside_code,
            flags=re.I,
        )
        offenders += re.findall(r"</?[A-Za-z!?][^\s>]{0,20}", outside_code)
        schemes += re.findall(r"(?:\]\(|\]:\s*)\s*([A-Za-z][A-Za-z0-9+.-]*)\s*:", outside_code)
    assert not offenders, (
        f"raw HTML reaches the page as markup: {offenders} "
        "-- write it in backticks if it should read as text"
    )
    unwanted = sorted({s for s in schemes if s.lower() not in {"http", "https", "mailto"}})
    assert not unwanted, f"link destinations the page's filter would drop: {unwanted}"


def test_every_release_is_listed_once():
    """Two entries for one version leave a reader to guess which is true."""
    releases = _releases()
    duplicated = sorted({v for v in releases if releases.count(v) > 1})
    assert not duplicated, f"listed more than once: {duplicated}"
