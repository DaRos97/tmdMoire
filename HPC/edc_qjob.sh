# SGE job array worker script for EDC Gamma grid sweep
# Called by edc_job.sh for each task in the array
# Computes chunk boundaries and runs edc_grid_gamma.py

N_TASKS=$1

# Convert 1-indexed SGE task ID to 0-indexed
TASK_ID=$((SGE_TASK_ID - 1))

echo "Task $SGE_TASK_ID: chunk $TASK_ID/$N_TASKS"

OUTPUT_FILE=Scratch/edc_gamma_task${SGE_TASK_ID}.out
ERROR_FILE=Scratch/edc_gamma_task${SGE_TASK_ID}.err

python3 scripts/edc_grid_gamma.py --chunk ${TASK_ID}/${N_TASKS} >${OUTPUT_FILE} 2>${ERROR_FILE}
