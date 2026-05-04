# SGE job array worker script
# Called by job.sh for each task in the array
# Computes chunk boundaries from SGE_TASK_ID and runs run_grid.py

MATERIAL=$1
TOTAL=$2
N_TASKS=$3
RUN_ID=$4

# Convert 1-indexed SGE task ID to 0-indexed
TASK_ID=$((SGE_TASK_ID - 1))

# Distribute TOTAL fits across N_TASKS tasks evenly
# First (TOTAL % N_TASKS) tasks get one extra fit
BASE=$((TOTAL / N_TASKS))
REMAINDER=$((TOTAL % N_TASKS))

if [ $TASK_ID -lt $REMAINDER ]; then
    CHUNK_SIZE=$((BASE + 1))
    START=$((TASK_ID * (BASE + 1)))
else
    CHUNK_SIZE=$BASE
    START=$((REMAINDER * (BASE + 1) + (TASK_ID - REMAINDER) * BASE))
fi
END=$((START + CHUNK_SIZE))

echo "Task $SGE_TASK_ID: fits [$START, $END) of $TOTAL (run: $RUN_ID)"

OUTPUT_FILE=Scratch/grid_${MATERIAL}_${RUN_ID}_task${SGE_TASK_ID}.out
ERROR_FILE=Scratch/grid_${MATERIAL}_${RUN_ID}_task${SGE_TASK_ID}.err

python3 scripts/run_grid.py ${MATERIAL} --start ${START} --end ${END} --run-id ${RUN_ID} >${OUTPUT_FILE} 2>${ERROR_FILE}
