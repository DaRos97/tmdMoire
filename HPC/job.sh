# SGE job array submission for monolayer grid search
# Submits 128 parallel tasks (one per CPU on rademaker)
# Each task runs ~28 fits from the 3600 total
#
# Usage: ./job.sh WSe2
#        ./job.sh WSe2 001      # with run ID
#        ./job.sh WS2 002

MATERIAL=$1
RUN_ID=${2:-default}
N_TASKS=128
TOTAL=1536

qsub -N grid_${MATERIAL}_${RUN_ID} \
     -o HPC/out_${MATERIAL}_${RUN_ID}.out \
     -e HPC/out_${MATERIAL}_${RUN_ID}.err \
     -t 1-${N_TASKS} \
     -q rademaker \
     HPC/qjob.sh ${MATERIAL} ${TOTAL} ${N_TASKS} ${RUN_ID}
