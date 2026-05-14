# SGE job array worker script
# Called by mono_job.sh for each task in the array
# Passes chunk specifier to Python; boundaries computed from fit_config.json

MATERIAL=$1
N_TASKS=$2
RUN_ID=$3

# Convert 1-indexed SGE task ID to 0-indexed
TASK_ID=$((SGE_TASK_ID - 1))

echo "Task $SGE_TASK_ID: chunk $TASK_ID/$N_TASKS (material: $MATERIAL, run: $RUN_ID)"

OUTPUT_FILE=Scratch/grid_${MATERIAL}_${RUN_ID}_task${SGE_TASK_ID}.out
ERROR_FILE=Scratch/grid_${MATERIAL}_${RUN_ID}_task${SGE_TASK_ID}.err

python3 scripts/run_monolayer_grid.py ${MATERIAL} --chunk ${TASK_ID}/${N_TASKS} --run-id ${RUN_ID} >${OUTPUT_FILE} 2>${ERROR_FILE}
