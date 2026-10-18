# SGE job array worker script for EDC Gamma grid sweep
# Called by edc_gamma_job.sh for each task in the array
# Computes chunk boundaries and runs edc_grid_gamma.py

N_TASKS=$1
RUN_ID=$2
OFFSET=${3:-0}
GRID_CONFIG=${4:-}
INTERLAYER_PARAMS=${5:-}

# Convert 1-indexed SGE task ID to 0-indexed, then apply offset
TASK_ID=$((SGE_TASK_ID - 1 + OFFSET))

echo "Task $SGE_TASK_ID (offset $OFFSET): global chunk $TASK_ID/$N_TASKS (run: $RUN_ID)"

OUTPUT_FILE=Scratch/edc_G_${RUN_ID}_task${TASK_ID}.out
ERROR_FILE=Scratch/edc_G_${RUN_ID}_task${TASK_ID}.err

if [ -n "$GRID_CONFIG" ] && [ -n "$INTERLAYER_PARAMS" ]; then
    python3 scripts/edc_grid_gamma.py \
        --chunk ${TASK_ID}/${N_TASKS} \
        --id ${RUN_ID} \
        --config "$GRID_CONFIG" \
        --interlayer-params "$INTERLAYER_PARAMS" \
        >"${OUTPUT_FILE}" 2>"${ERROR_FILE}"
else
    python3 scripts/edc_grid_gamma.py \
        --chunk ${TASK_ID}/${N_TASKS} \
        --id ${RUN_ID} \
        >"${OUTPUT_FILE}" 2>"${ERROR_FILE}"
fi
