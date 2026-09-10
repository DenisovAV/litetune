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
