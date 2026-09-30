#!/usr/bin/env bash
# Creates .venv from ./wheelhouse (no internet). With an internal PyPI mirror instead, set PIP_INDEX_URL
# and run:  .venv/bin/pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install --no-index --find-links wheelhouse -U pip setuptools wheel || true
.venv/bin/pip install --no-index --find-links wheelhouse -r requirements.lock
[ -f .env ] || cp .env.example .env
echo "venv ready. Next: edit .env, then run scripts/smoke_test.py"
