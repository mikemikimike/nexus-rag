#!/usr/bin/env bash
# RUN ON AN INTERNET-CONNECTED MACHINE with the SAME OS, CPU arch and Python minor version as the target.
# Skip entirely if your closed network already has an internal PyPI mirror.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv /tmp/rag-build
. /tmp/rag-build/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip freeze > requirements.lock
mkdir -p wheelhouse
pip download -r requirements.lock -d wheelhouse
pip download pip setuptools wheel -d wheelhouse
echo "wheelhouse: $(ls wheelhouse | wc -l) files; lock written to requirements.lock"
