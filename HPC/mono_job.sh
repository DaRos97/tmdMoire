# SGE job array submission for monolayer grid search
# Submits N_TASKS parallel tasks (one per CPU on rademaker)
# Each task runs a chunk of the grid, boundaries computed by Python
#
# Usage: ./mono_job.sh WSe2
#        ./mono_job.sh WSe2 001      # with run ID
#        ./mono_job.sh WS2 002
#        ./mono_job.sh WSe2 default 256  # with custom number of tasks

MATERIAL=$1
RUN_ID=${2:-default}
N_TASKS=${3:-128}

qsub -N grid_${MATERIAL}_${RUN_ID} \
     -o HPC/out_${MATERIAL}_${RUN_ID}.out \
     -e HPC/out_${MATERIAL}_${RUN_ID}.err \
     -t 1-${N_TASKS} \
     -q rademaker \
     HPC/mono_qjob.sh ${MATERIAL} ${N_TASKS} ${RUN_ID}
