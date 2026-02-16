#!/usr/bin/env bash
set -e
ROLE="${ROLE:-client}"
if [ "$ROLE" = "aggregator" ]; then
  python -u aggregator.py
else
  python -u client.py
fi
