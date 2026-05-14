#!/bin/bash
# Quick test script for HPC: runs a few EDC points to verify the pipeline works
# before submitting the full job array.
#
# Usage: ./HPC/edc_test.sh

RUN_ID="test_$(date +%Y%m%d_%H%M%S)"

echo "=== EDC Gamma HPC Pipeline Test ==="
echo "Run ID:   $RUN_ID"
echo ""

# Run a tiny chunk (6 points out of ~6M)
echo "--- Running chunk 0/1000000 (6 points) ---"
python3 scripts/edc_grid_gamma.py --chunk 0/1000000

echo ""
echo "--- Checking output ---"
ls -lh Data/edc_grid_gamma/

echo ""
echo "=== Test complete ==="
echo "Results in: Data/edc_grid_gamma/"
echo "To clean up: rm -rf Data/edc_grid_gamma/"
