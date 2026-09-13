#!/usr/bin/env bash
# Build + deploy the litetune website to Firebase Hosting.
#
# The project id is deliberately not in this repository: it is an
# infrastructure identifier, it is not otherwise public (the site is static,
# with no client Firebase SDK to carry it), and naming it here would point at
# every other resource in the same project. Pass it in the environment:
#
#   LITETUNE_FIREBASE_PROJECT=<project-id> ./deploy.sh
#
# The hosting site is named in firebase.json, so no `.firebaserc` and no
# target mapping is needed -- which is what lets this script and the CI
# workflow deploy the same place without either of them carrying the project.
set -euo pipefail

WEBSITE_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="https://litetune.dev"
PROJECT="${LITETUNE_FIREBASE_PROJECT:?set LITETUNE_FIREBASE_PROJECT to the Firebase project id}"

cd "$WEBSITE_DIR"

# Clear jaspr's own leftovers off the two ports `jaspr build` uses, and stop
# if anything else is on them.
#
# The ports are 8080 and 5567: jaspr_cli 0.23.1 `build_command.dart:226` takes
# `project.port ?? defaultServePort` and `:229` `serverProxyPort`, which
# `project.dart:296-297` define as those two. This list used to also carry
# 8181 and 5467, which `build` never opens -- 5467 is the dev command's webdev
# port and 8181 the Dart VM service. Killing on those was reaching past what
# the build needs, at a machine where 8181 is exactly where a `flutter run`
# would be.
#
# And it used to kill whatever it found, by port alone. On the machine this
# was written on, 8080 was held by an unrelated Python server, which this
# would have SIGKILLed with no prompt. jaspr's own processes all carry
# "jaspr" in their command line -- the `sh .../bin/jaspr` wrapper, the
# `jaspr.dart-*.snapshot` CLI and the `-Djaspr.dev.web=...` renderer -- so
# that is the only thing matched now. `*dart*` would have caught every Dart
# VM on the machine, including another project's `dart run build_runner
# serve`, which also defaults to 8080.
#
# Refusing to continue matters as much as not killing. Observed with lsof:
# with a foreign listener on 127.0.0.1:8080, jaspr's renderer still binds
# *:8080 (it passes `shared: true`), the build then fetches
# `http://localhost:$serverPort` (`build_command.dart:301`) -- which resolved
# to the more specific listener, not to jaspr's -- prints
# `Generating route "/"` and never returns. A 5567 collision at least fails loudly with "Address already in
# use"; 8080 is the silent one, and it cost 14 minutes before anyone looked.
blocked=0
for p in 8080 5567; do
  # Listeners only. `lsof -ti :8080` also answers for anything with an open
  # connection *to* someone's port 8080 -- a browser tab, an IDE, a lingering
  # CLOSE_WAIT -- and none of those can stop jaspr binding the port. Measured:
  # with a listener and one client both present, `lsof -ti :8080` returned
  # both pids and `-sTCP:LISTEN` returned only the listener. Without this the
  # loop refused builds nothing was blocking, and before that killed clients
  # that were merely talking to something else. `-nP` also skips the reverse
  # DNS and port-name lookups, which stall on a remote peer.
  for pid in $(lsof -nP -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null); do
    cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    case "$cmd" in
      "") ;;
      *jaspr*) kill -9 "$pid" 2>/dev/null || true ;;
      *) echo "!!  port $p is held by pid $pid, which is not jaspr:" >&2
         echo "!!    ${cmd:0:100}" >&2
         blocked=1 ;;
    esac
  done
done
if [ "$blocked" -ne 0 ]; then
  echo "!!" >&2
  echo "!!  Not killing it, and not building either: jaspr would bind the port" >&2
  echo "!!  anyway and then hang talking to the other server. Stop that process," >&2
  echo "!!  or set a free port under \`jaspr:\` in pubspec.yaml." >&2
  exit 1
fi

# The page shows the version in src/litetune/_version.py, so deploying names it
# on the live site, and it shows whatever this checkout's site code is. The
# workflow deploys only merged code, and a version only once PyPI serves it as
# a final release; a deploy from here should not be the way around either. A
# preview does not need this: `jaspr serve` shows the same page without
# publishing it. The rules are the build job's gate in site.yml, less its
# freshness test: a deploy from here is of this checkout, by choice.
[ "$(grep -c '^__version__ = ' ../src/litetune/_version.py || true)" = 1 ] \
  || { echo "!!  ../src/litetune/_version.py does not have exactly one __version__ line" >&2; exit 1; }
VERSION="$(sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' ../src/litetune/_version.py)"
[ -n "$VERSION" ] || { echo "!!  cannot read __version__ from ../src/litetune/_version.py" >&2; exit 1; }
if ! [[ "$VERSION" =~ ^([0-9]+!)?[0-9]+(\.[0-9]+)*(\.post[0-9]+)?$ ]]; then
  echo "!!  $VERSION is not a final release; pip does not install one by default." >&2
  exit 1
fi
pypi_json="$(mktemp)"
# curl's own exit as well as the status: a 200 whose body stalled past
# `--max-time` still reports 200, with the body cut short.
curl_exit=0
status="$(curl -s --connect-timeout 10 --max-time 30 -o "$pypi_json" -w '%{http_code}' "https://pypi.org/pypi/litetune/$VERSION/json")" \
  || curl_exit=$?
# Installable as exactly this version: PyPI looks versions up normalised, so it
# also answers for a spelling the site would show differently.
verdict="$( [ "$status" = 200 ] && jq -r --arg v "$VERSION" '
  if (.info.version | type) != "string" then "unreadable"
  elif .info.version != $v then "respelled"
  elif .info.yanked != false then "yanked"
  elif ((.urls // []) | length) == 0 then "no files"
  else "installable" end' "$pypi_json" 2>/dev/null || true)"
rm -f "$pypi_json"
if [ "$verdict" != installable ] || [ "$curl_exit" != 0 ]; then
  echo "!!  PyPI does not serve litetune $VERSION as installable (HTTP ${status:-none}, ${verdict:-no verdict}, curl $curl_exit)." >&2
  echo "!!  Not deploying a site that names a version pip does not install." >&2
  exit 1
fi
# What gets built is the working tree, so the commit, the release's tag and the
# files the build reads are all checked: a commit on `main` with local edits to
# the site would otherwise publish the edits. The repository is named rather
# than taken from whatever this clone calls `origin`, and the comparisons are
# against FETCH_HEAD and object ids rather than a short `origin/main`, which
# resolves a tag of that name before the remote branch -- how an unmerged
# commit got past the same test in the workflow during review. The fetch names
# `refs/heads/main` for the same reason: asked for `main`, git brings back a tag
# of that name if the remote has one.
canonical=https://github.com/DenisovAV/litetune.git
git fetch --quiet "$canonical" refs/heads/main
main_commit="$(git rev-parse FETCH_HEAD)"
if ! git merge-base --is-ancestor HEAD "$main_commit"; then
  echo "!!  HEAD is not on DenisovAV/litetune main; the live site is built only from merged code." >&2
  exit 1
fi
# The commit the release tag names -- peeled, for an annotated tag -- must be on
# `main` too, as the workflow requires: `publish.yml` has no ancestry test, and
# a release cut on an unmerged commit reaches PyPI anyway. A tag whose commit
# this clone does not have is not on `main` either, and fails the same test.
# `ls-remote` patterns match the end of a ref name -- `x/refs/tags/v0.1.6` too --
# so the exact names are picked out of its answer: the peeled `^{}` line for an
# annotated tag, else the tag itself, `v` spelling first.
remote_tags="$(git ls-remote --tags "$canonical")"
tagged_commit="$(awk -v v="refs/tags/v$VERSION" -v b="refs/tags/$VERSION" '
  $2 == v "^{}" { vp = $1 } $2 == v { vt = $1 } $2 == b "^{}" { bp = $1 } $2 == b { bt = $1 }
  END { print (vp != "" ? vp : vt != "" ? vt : bp != "" ? bp : bt) }' <<<"$remote_tags")"
if [ -z "$tagged_commit" ] || ! git merge-base --is-ancestor "$tagged_commit" "$main_commit" 2>/dev/null; then
  echo "!!  the release tag for $VERSION is missing or not on main." >&2
  exit 1
fi
# And the tagged commit's own version must be this one -- a tag on an older
# merged commit passes the test above. On `main`, so this clone has it.
tagged_source="$(git show "$tagged_commit:src/litetune/_version.py" 2>/dev/null || true)"
tagged_version="$( [ "$(grep -c '^__version__ = ' <<<"$tagged_source" || true)" = 1 ] \
  && sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' <<<"$tagged_source" || true)"
if [ "$tagged_version" != "$VERSION" ]; then
  echo "!!  the commit tagged for $VERSION says ${tagged_version:-nothing readable} in its own _version.py." >&2
  exit 1
fi
if [ -n "$(git status --porcelain -- . ../CHANGELOG.md ../src/litetune/_version.py)" ]; then
  echo "!!  uncommitted changes to the site's inputs; commit or stash them first:" >&2
  git status --short -- . ../CHANGELOG.md ../src/litetune/_version.py >&2
  exit 1
fi

echo "==> Building Jaspr site (static)…"
# A reused incremental build cache has been observed to snapshot the page
# before the route table registered, producing a near-empty index.html that
# then got deployed. A clean rebuild is cheap and removes the failure mode.
rm -rf build/jaspr .dart_tool/build
jaspr build --sitemap-domain "$DOMAIN"

# The same check CI runs, on the same file, so a local deploy cannot publish
# a page CI would have refused.
./check-build.sh

echo "==> Deploying to Firebase Hosting…"
firebase deploy --only hosting --project "$PROJECT"

echo "==> Done."
