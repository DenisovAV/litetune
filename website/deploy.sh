#!/usr/bin/env bash
# Build + deploy the litetune website to Firebase Hosting.
#
# Deploys to the `litetune` hosting site in the same Firebase project that
# serves fluttergemma.dev. The target mapping lives in .firebaserc.
set -euo pipefail

WEBSITE_DIR="$(cd "$(dirname "$0")" && pwd)"
DOMAIN="https://litetune.dev"
PROJECT="aichat-c0c27"
TARGET="litetune"

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

echo "==> Deploying to Firebase Hosting ($TARGET)…"
firebase deploy --only "hosting:$TARGET" --project "$PROJECT"

echo "==> Done."
