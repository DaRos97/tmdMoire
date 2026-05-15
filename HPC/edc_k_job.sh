# SGE job array submission for EDC K grid sweep
# Submits N_TASKS parallel tasks (one per CPU on rademaker)
# Each task computes a chunk of the 2D (Vk, phiK) parameter grid
#
# Usage: ./HPC/edc_k_job.sh
#        ./HPC/edc_k_job.sh 256            # with custom number of tasks
#        ./HPC/edc_k_job.sh 128 001        # with run ID

N_TASKS=${1:-128}
RUN_ID=${2:-default}

qsub -N edc_k_${RUN_ID} \
     -o HPC/out_edc_k_${RUN_ID}.out \
     -e HPC/out_edc_k_${RUN_ID}.err \
     -t 1-${N_TASKS} \
     -q rademaker \
     HPC/edc_k_qjob.sh ${N_TASKS} ${RUN_ID}
