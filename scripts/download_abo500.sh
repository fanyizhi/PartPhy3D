#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-data}"
ARCHIVE="${DESTINATION}/abo_500.zip"
URL="https://uofi.box.com/shared/static/743ydh4n1xi0dj05lcyyg4evqk2n4dko.zip"

mkdir -p "${DESTINATION}"
curl --fail --location --continue-at - "${URL}" --output "${ARCHIVE}"
unzip -n "${ARCHIVE}" -d "${DESTINATION}"

echo "ABO-500 downloaded under ${DESTINATION}. The original ABO release provides"
echo "glTF/GLB models, but they must be matched to these scenes and staged with"
echo "verified camera alignment and metric scale before running the mesh baseline."
