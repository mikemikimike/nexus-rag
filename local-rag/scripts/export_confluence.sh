#!/usr/bin/env bash
# Export from Confluence (confluence-cli), then index what changed. Index runs even if some pages failed.
cd "$(dirname "$0")/.."
rc=0
.venv/bin/python scripts/export_confluence.py "$@" || rc=$?
.venv/bin/python ingest.py || rc=$?
exit $rc
