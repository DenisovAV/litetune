#!/usr/bin/env bash
# Run what CI runs, on the platform CI runs it on.
#
# This exists because a green local suite and a red CI run happened at the same
# time, twice, on the same commit. The cause was not a flaky runner: `os.waitid`
# is absent from CPython on macOS before 3.13 and present on Linux, so the same
# test took different branches on the two, and five tests that pass here failed
# there. A script that reran the same commands on this laptop would have said
# nothing. The platform is the point.
#
#   scripts/ci-local.sh            # the whole matrix, then the wheel job
#   scripts/ci-local.sh 3.13       # one interpreter
#   scripts/ci-local.sh wheel      # just the wheel job
#
# Needs podman (or docker: set ENGINE=docker). Nothing is written to the
# working tree -- the source is mounted read-only and copied inside.
set -euo pipefail

ENGINE="${ENGINE:-podman}"
# Kept in step with .github/workflows/ci.yml by the check at the bottom, which
# fails loudly rather than letting the two drift apart in silence.
VERSIONS=(3.10 3.12 3.13)
IMAGE_PREFIX="docker.io/library/python"

# A `credHelpers` entry in ~/.docker/config.json sends podman to
# `docker-credential-gcloud` for every registry, and it fails with "error
# getting credentials" once those tokens expire -- on `run` as well as on
# `pull`, and even with `--pull=never`. An empty config for the duration
# sidesteps it without touching the user's own.
SCRATCH="$(mktemp -d)"; echo '{}' > "${SCRATCH}/config.json"
export DOCKER_CONFIG="$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pull() {
  "$ENGINE" image exists "$1" 2>/dev/null && return 0
  "$ENGINE" pull "$1" >/dev/null
}

check_job() {
  local version="$1" image="${IMAGE_PREFIX}:$1-slim"
  echo "==> check (${version}) on linux"
  pull "$image"
  "$ENGINE" run --rm -v "${repo_root}:/src:ro" "$image" bash -euo pipefail -c '
    cp -a /src /w && cd /w
    # The tree carries a macOS virtualenv and caches; an editable install over
    # them resolves to the host interpreter and the run silently becomes a
    # different one.
    rm -rf .venv* .mypy_cache .pytest_cache .ruff_cache **/__pycache__ 2>/dev/null || true
    pip install --quiet --disable-pip-version-check -e ".[dev]"
    ruff check src tests
    ruff format --check src tests
    mypy
    pytest -q
  '
}

wheel_job() {
  local image="${IMAGE_PREFIX}:3.10-slim"
  echo "==> wheel on linux"
  pull "$image"
  "$ENGINE" run --rm -v "${repo_root}:/src:ro" "$image" bash -euo pipefail -c '
    cp -a /src /w && cd /w
    rm -rf dist .venv* 2>/dev/null || true
    pip install --quiet --disable-pip-version-check build twine
    python -m build --wheel --sdist >/dev/null
    twine check dist/*
    python -m venv /tmp/fresh
    /tmp/fresh/bin/pip install --quiet dist/*.whl
    /tmp/fresh/bin/litetune --help >/dev/null
    /tmp/fresh/bin/python -c "import litetune, pathlib; \
      assert litetune.__version__, \"no version\"; \
      assert (pathlib.Path(litetune.__file__).parent / \"py.typed\").is_file(), \"no py.typed\""
  '
}

# The matrix here and the matrix in the workflow must be the same list. They
# were written apart, and a script that quietly tests fewer versions than CI is
# worse than no script: it reports green for a matrix it did not run.
assert_matrix_matches() {
  local declared
  declared="$(grep -o 'python-version: \[.*\]' "${repo_root}/.github/workflows/ci.yml" |
    tr -d '"[]' | sed 's/python-version: //' | tr ',' ' ' | xargs)"
  if [ "$declared" != "${VERSIONS[*]}" ]; then
    echo "the workflow tests [$declared] and this script tests [${VERSIONS[*]}]" >&2
    echo "bring them back into step before trusting either" >&2
    exit 1
  fi
}

assert_matrix_matches
case "${1:-all}" in
  all)   for v in "${VERSIONS[@]}"; do check_job "$v"; done; wheel_job ;;
  wheel) wheel_job ;;
  *)     check_job "$1" ;;
esac
echo "==> everything CI runs, run here"
