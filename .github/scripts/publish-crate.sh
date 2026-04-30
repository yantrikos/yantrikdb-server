#!/usr/bin/env bash
# Publish a single workspace crate to crates.io with idempotent semantics:
#
#   - exit 0 on success
#   - exit 0 if the crate is ALREADY PUBLISHED at this version (re-run safety)
#   - exit non-zero on any OTHER failure (manifest invalid, auth, network…)
#
# Background: yantrikos/yantrikdb-server v0.7.3 silently failed to publish
# because the previous version of this workflow used a blanket
# `continue-on-error: true`, which swallowed a real "all dependencies must
# have a version requirement specified" manifest error. The tag was pushed,
# the workflow reported success, but crates.io stayed at v0.7.2.
#
# This script makes the distinction loud.
#
# Usage:
#   .github/scripts/publish-crate.sh <crate-name>
#
# Required env:
#   CARGO_REGISTRY_TOKEN

set -euo pipefail

CRATE="${1:?usage: publish-crate.sh <crate-name>}"

if [[ -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  echo "::error::CARGO_REGISTRY_TOKEN is not set"
  exit 2
fi

echo "::group::cargo publish -p ${CRATE} --no-verify"
# Capture combined output AND exit code without `set -e` aborting us.
set +e
out="$(cargo publish -p "${CRATE}" --no-verify 2>&1)"
rc=$?
set -e
echo "${out}"
echo "::endgroup::"

if [[ $rc -eq 0 ]]; then
  echo "::notice::${CRATE}: published successfully"
  exit 0
fi

# Idempotent re-run: crates.io reports "already exists on crates.io index"
# verbatim when a version is already published. That's an expected outcome
# of running this workflow against a tag that's already been published.
if echo "${out}" | grep -qE "already exists on crates\.io"; then
  echo "::notice::${CRATE}: already published at this version (idempotent re-run, exit 0)"
  exit 0
fi

# Anything else is a genuine publish failure that needs to surface in the
# workflow status. The most painful class historically:
#   "all dependencies must have a version requirement specified when publishing"
# (v0.7.3 → v0.7.4 incident).
echo "::error::${CRATE}: publish failed for a non-idempotent reason; see log group above"
exit "$rc"
