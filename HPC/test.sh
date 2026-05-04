#!/bin/bash
# Quick test script for HPC: runs a single fit to verify the pipeline works
# before submitting the full job array.
#
# Usage: ./HPC/test.sh WSe2
#        ./HPC/test.sh WS2

MATERIAL=${1:-WSe2}
RUN_ID="test_$(date +%Y%m%d_%H%M%S)"

echo "=== HPC Pipeline Test ==="
echo "Material: $MATERIAL"
echo "Run ID:   $RUN_ID"
echo ""

# Run a single fit (index 0)
echo "--- Running single fit (index 0) ---"
python3 scripts/run_grid.py ${MATERIAL} --start 0 --end 1 --run-id ${RUN_ID}

echo ""
echo "--- Scoring results ---"
python3 scripts/run_grid.py ${MATERIAL} --score --run-id ${RUN_ID} --top 1

echo ""
echo "--- Generating plots ---"
python3 scripts/run_grid.py ${MATERIAL} --score --run-id ${RUN_ID} --top 1 --plot

echo ""
echo "=== Test complete ==="
echo "Results in: Data/${MATERIAL}_run_${RUN_ID}/"
echo "To clean up: rm -rf Data/${MATERIAL}_run_${RUN_ID}"
