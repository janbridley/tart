#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="tart-tools/src/sbpl"
COMMIT_HASH="be6e8eac029b183056b7e4402879f15d2c85f61b" #v0.147.0
BASE_URL="https://raw.githubusercontent.com/openai/codex/${COMMIT_HASH}/codex-rs/sandboxing/src"

mkdir -p "${DEST_DIR}"

echo "Vendoring Codex SBPL files at commit ${COMMIT_HASH}..."

curl -sSLo "${DEST_DIR}/restricted_read_only_platform_defaults.sbpl" \
    "${BASE_URL}/restricted_read_only_platform_defaults.sbpl"

curl -sSLo "${DEST_DIR}/seatbelt_base_policy.sbpl" \
    "${BASE_URL}/seatbelt_base_policy.sbpl"

curl -sSLo "${DEST_DIR}/LICENSE" \
    "https://raw.githubusercontent.com/openai/codex/${COMMIT_HASH}/LICENSE"

echo "Done."
