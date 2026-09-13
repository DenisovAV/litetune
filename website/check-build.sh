#!/usr/bin/env bash
# Refuse a build that would publish an empty page over a working site.
#
# `jaspr build` exits zero on a render that produced nothing, and Firebase
# Hosting will happily serve the result. This has a known trigger: a reused
# incremental cache snapshots the page before the route table registers, which
# is why both `deploy.sh` and the CI build clear `build/jaspr` and
# `.dart_tool/build` first. This script is the check that runs after, in case
# something else produces the same shape.
#
#   ./check-build.sh [build-dir]      # default: build/jaspr
#
# One script rather than the same three lines in the workflow and in
# deploy.sh: the marker below is the kind of thing that gets updated in one
# copy, and the copy that then never fails is the one guarding the deploy.
set -euo pipefail

BUILD_DIR="${1:-$(cd "$(dirname "$0")" && pwd)/build/jaspr}"

fail() { echo "check-build: $*" >&2; exit 1; }

[ -s "$BUILD_DIR/index.html" ] || fail "index.html is missing or empty"

# A marker from the page body, not just "litetune" anywhere in the file. The
# word appears 14 times in <head> alone -- title, canonical link, JSON-LD --
# so a `grep litetune` passes on a head-only render, which is precisely the
# build this script exists to reject. `pip install litetune` is in the hero
# (lib/landing/sections/hero.dart:45) and nowhere in the head.
#
# Scoped to the body rather than grepped over the whole file, because
# "somewhere in index.html" is what made the previous check useless. A
# document consisting only of `<head>pip install litetune</head>` passes a
# plain grep for the marker while having no body at all.
#
# This is a text range and not a parse, and the difference is worth naming
# rather than glossing: a `<body` occurring inside a script string or an
# attribute in the head would open the range early, and the marker after it
# would then pass. Neither can come out of jaspr, which emits one lowercase
# `<body` on the boundary line -- the character class is only so that a
# hand-written uppercase `<BODY>` is not rejected for nothing.
#
# What it proves is that the body is not empty, not that the page is whole:
# a render that produced the hero and stopped passes. That is the failure mode
# on record -- the route table not yet registered, so nothing renders -- and a
# check that tried to assert every section would go stale on the next edit.
#
# `grep -c`, not `grep -q`, in every pipeline here: `-q` exits at the first
# match, the `sed` feeding it is then writing into a closed pipe, and under
# `pipefail` its SIGPIPE fails the pipeline -- a false "no body" on a good build.
# Reproduced with a 1.4 MB page: five runs of `sed | grep -q` all returned
# status 141, `sed | grep -c` none. `-c` reads to the end.
sed -n '/<[Bb][Oo][Dd][Yy]/,$p' "$BUILD_DIR/index.html" \
  | grep -c "pip install litetune" >/dev/null \
  || fail "index.html rendered a head with no page body"

# The other files a deploy publishes. index.html references the first two from
# its head, and an index pointing at an asset the build did not write goes out
# as a page with no icon or no client bundle.
for artefact in main.client.dart.js favicon.svg; do
  [ -s "$BUILD_DIR/$artefact" ] || fail "$artefact is missing or empty"
done

# The sitemap is the one output nothing on the page would reveal missing, and
# it is conditional: `build_command.dart:366` writes it only when
# --sitemap-domain was passed. Both callers pass it, so its absence means the
# build was made some other way and is not the one that should be deployed --
# but say that, because the cause is a missing flag and not a broken render.
[ -s "$BUILD_DIR/sitemap.xml" ] \
  || fail "sitemap.xml is missing or empty -- was this built without --sitemap-domain?"

# The second page, and the one fact both pages state. The version is read from
# the package the way `lib/project.dart` reads it -- the whole
# `__version__ = "..."` line, since the module docstring above it holds quoted
# text of its own -- and must appear in the nav bar and as a release heading in
# the changelog.
#
# The heading is the check that has a cause behind it: the nav's version links
# to `/changelog#v<version>`, so a version bumped without a changelog entry
# builds cleanly and ships a link to nothing. That is also the release that
# most needs its entry, because a release that makes an earlier result wrong
# says so there.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ "$(grep -c '^__version__ = ' "$REPO_ROOT/src/litetune/_version.py" || true)" = 1 ] \
  || fail "$REPO_ROOT/src/litetune/_version.py does not have exactly one __version__ line"
VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "$REPO_ROOT/src/litetune/_version.py")"
[ -n "$VERSION" ] || fail "cannot read __version__ from $REPO_ROOT/src/litetune/_version.py"

CHANGELOG_HTML="$BUILD_DIR/changelog/index.html"
[ -s "$CHANGELOG_HTML" ] || fail "changelog/index.html is missing or empty"
sed -n '/<[Bb][Oo][Dd][Yy]/,$p' "$CHANGELOG_HTML" | grep -c 'class="changelog-body"' >/dev/null \
  || fail "changelog/index.html rendered a head with no page body"
# The newest release heading in the rendered changelog body, id and text both:
# the page renders the id from the text, so a match on both is a heading this
# page produced, not the same characters somewhere else in the file. The newest
# rather than any, as `tests/test_changelog.py` requires -- an entry for the
# new version written under the old heading would otherwise pass.
NEWEST="$(sed -n '/class="changelog-body"/,$p' "$CHANGELOG_HTML" | grep -o '<h2 id="v[^"]*">[^<]*' | awk 'NR == 1' || true)"
case "$NEWEST" in
  "<h2 id=\"v$VERSION\">$VERSION"*) ;;
  *) fail "CHANGELOG.md does not open with an entry for $VERSION, and the nav links to one (newest: ${NEWEST:-none})" ;;
esac
sed -n '/<[Bb][Oo][Dd][Yy]/,$p' "$BUILD_DIR/index.html" | grep -cF ">v$VERSION<" >/dev/null \
  || fail "index.html does not show version $VERSION"

echo "check-build: $BUILD_DIR looks like a full render of $VERSION"
