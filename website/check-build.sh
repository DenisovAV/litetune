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
# What this proves is that the body is not empty, not that the page is whole:
# a render that produced the hero and stopped passes. That is the failure mode
# on record -- the route table not yet registered, so nothing renders -- and a
# check that tried to assert every section would go stale on the next edit.
sed -n '/<body/,$p' "$BUILD_DIR/index.html" | grep -q "pip install litetune" \
  || fail "index.html rendered a head with no page body"

# The other files a deploy publishes. index.html references the first two from
# its head, and an index pointing at an asset the build did not write goes out
# as a page with no icon or no client bundle. sitemap.xml comes from
# --sitemap-domain and is the file nothing on the page would reveal missing.
for artefact in main.client.dart.js favicon.svg sitemap.xml; do
  [ -s "$BUILD_DIR/$artefact" ] || fail "$artefact is missing or empty"
done

echo "check-build: $BUILD_DIR looks like a full render"
