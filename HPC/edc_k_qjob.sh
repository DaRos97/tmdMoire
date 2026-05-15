# SGE job array worker script for EDC K grid sweep
# Called by edc_k_job.sh for each task in the array
# Computes chunk boundaries and runs edc_grid_k.py

N_TASKS=$1
RUN_ID=$2

# Convert 1-indexed SGE task ID to 0-indexed
TASK_ID=$((SGE_TASK_ID - 1))

echo "Task $SGE_TASK_ID: chunk $TASK_ID/$N_TASKS (run: $RUN_ID)"

OUTPUT_FILE=Scratch/edc_k_${RUN_ID}_task${SGE_TASK_ID}.out
ERROR_FILE=Scratch/edc_k_${RUN_ID}_task${SGE_TASK_ID}.err

python3 scripts/edc_grid_k.py --chunk ${TASK_ID}/${N_TASKS} --run-id ${RUN_ID} >${OUTPUT_FILE} 2>${ERROR_FILE}
