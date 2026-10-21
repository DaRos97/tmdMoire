# SGE job array submission for EDC K grid sweep
# Submits N_TASKS parallel tasks (one per CPU on selected compute nodes).
# The K sweep defaults to one task for the full 2D (Vk, phiK) grid; pass a
# larger task count only when deliberately splitting the sweep.
#
# Check free CPUs on target nodes:
#   qhost | grep -E 'compute-2-11|compute-2-12|compute-2-13|compute-3-01|compute-3-02|compute-3-03|compute-3-04|compute-4-01|compute-4-02|compute-4-03|compute-4-04|compute-4-05|compute-4-06|compute-4-07|compute-4-08'
#
# Usage: ./HPC/edc_k_job.sh
#        ./HPC/edc_k_job.sh 001              # with run ID (one task)
#        ./HPC/edc_k_job.sh 001 8            # with run ID and custom task count

RUN_ID=${1:-default}
N_TASKS=${2:-1}

qsub -N edc_k_${RUN_ID} \
     -o HPC/out_edc_k_${RUN_ID}.out \
     -e HPC/out_edc_k_${RUN_ID}.err \
     -t 1-${N_TASKS} \
     -l hostname='compute-2-11|compute-2-12|compute-2-13|compute-3-01|compute-3-02|compute-3-03|compute-3-04|compute-4-01|compute-4-02|compute-4-03|compute-4-04|compute-4-05|compute-4-06|compute-4-07|compute-4-08' \
     HPC/edc_k_qjob.sh ${N_TASKS} ${RUN_ID}
