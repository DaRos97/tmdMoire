# SGE job array worker script for EDC Gamma grid sweep
# Called by edc_gamma_job.sh for each task in the array
# Computes chunk boundaries and runs edc_grid_gamma.py

N_TASKS=$1
RUN_ID=$2
OFFSET=${3:-0}

# Convert 1-indexed SGE task ID to 0-indexed, then apply offset
TASK_ID=$((SGE_TASK_ID - 1 + OFFSET))

echo "Task $SGE_TASK_ID (offset $OFFSET): global chunk $TASK_ID/$N_TASKS (run: $RUN_ID)"

OUTPUT_FILE=Scratch/edc_gamma_${RUN_ID}_task${TASK_ID}.out
ERROR_FILE=Scratch/edc_gamma_${RUN_ID}_task${TASK_ID}.err

python3 scripts/edc_grid_gamma.py --chunk ${TASK_ID}/${N_TASKS} --id ${RUN_ID} >${OUTPUT_FILE} 2>${ERROR_FILE}
