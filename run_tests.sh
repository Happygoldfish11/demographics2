#!/usr/bin/env bash
# Run the full BISG estimator test suite.
#
# Usage:
#   ./run_tests.sh              # quiet
#   ./run_tests.sh -v           # verbose
#   ./run_tests.sh -k pattern   # filter by name (any pytest flag works)

set -euo pipefail

cd "$(dirname "$0")"

# If pytest isn't available, install requirements first.
if ! command -v pytest >/dev/null 2>&1; then
    echo "pytest not found. Installing requirements..."
    pip install -r requirements.txt
fi

exec python -m pytest tests/ "$@"
