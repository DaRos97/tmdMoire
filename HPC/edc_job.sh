# SGE job array submission for EDC Gamma grid sweep
# Submits N_TASKS parallel tasks (one per CPU on rademaker)
# Each task computes a chunk of the 6D parameter grid
#
# Usage: ./HPC/edc_job.sh
#        ./HPC/edc_job.sh 256            # with custom number of tasks
#        ./HPC/edc_job.sh 128 001        # with run ID

N_TASKS=${1:-128}
RUN_ID=${2:-default}

qsub -N edc_gamma_${RUN_ID} \
     -o HPC/out_edc_gamma_${RUN_ID}.out \
     -e HPC/out_edc_gamma_${RUN_ID}.err \
     -t 1-${N_TASKS} \
     -q rademaker \
     HPC/edc_qjob.sh ${N_TASKS} ${RUN_ID}
