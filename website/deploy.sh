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

# Free jaspr's dev ports so the build's transient server can bind.
for p in 5567 8080 8181 5467; do
  lsof -ti ":$p" 2>/dev/null | xargs kill -9 2>/dev/null || true
done

echo "==> Building Jaspr site (static)…"
# A reused incremental build cache has been observed to snapshot the page
# before the route table registered, producing a near-empty index.html that
# then got deployed. A clean rebuild is cheap and removes the failure mode.
rm -rf build/jaspr .dart_tool/build
jaspr build --sitemap-domain "$DOMAIN"

echo "==> Deploying to Firebase Hosting…"
firebase deploy --only hosting --project "$PROJECT"

echo "==> Done."
