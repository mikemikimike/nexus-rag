#!/usr/bin/env bash
# RUN ON AN INTERNET-CONNECTED MACHINE. Fetches the Qdrant server binary into vendor/qdrant.
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH="$(uname -m)"
[ "$ARCH" = "x86_64" ] || { echo "Only x86_64 handled here; download the matching asset from https://github.com/qdrant/qdrant/releases and place the 'qdrant' binary in vendor/"; exit 1; }
mkdir -p vendor
curl -fL "https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz" -o vendor/qdrant.tgz
tar -xzf vendor/qdrant.tgz -C vendor qdrant
rm vendor/qdrant.tgz
chmod +x vendor/qdrant
vendor/qdrant --version 2>/dev/null || true
echo "vendor/qdrant ready"
