# SGE job array submission for EDC Gamma grid sweep
# Submits N_TASKS parallel tasks (one per CPU on selected compute nodes)
# Each task computes a chunk of the 4D parameter grid (Vg, phiG, w1p, w1d; w2p/w2d fixed)
#
# When N_TASKS > 128: first 128 tasks run on queue rademaker,
# remaining tasks run on the specified compute nodes.
#
# Check free CPUs on target nodes:
#   qhost | grep -E 'compute-2-11|compute-2-12|compute-2-13|compute-3-01|compute-3-02|compute-3-03|compute-3-04|compute-4-01|compute-4-02|compute-4-03|compute-4-04|compute-4-05|compute-4-06|compute-4-07|compute-4-08'
#
# Usage: ./HPC/edc_gamma_job.sh
#        ./HPC/edc_gamma_job.sh 001              # with run ID (default 128 tasks)
#        ./HPC/edc_gamma_job.sh 001 256          # with run ID and custom number of tasks

RUN_ID=${1:-default}
N_TASKS=${2:-128}

NODES='compute-2-11|compute-2-12|compute-2-13|compute-3-01|compute-3-02|compute-3-03|compute-3-04|compute-4-01|compute-4-02|compute-4-03|compute-4-04|compute-4-05|compute-4-06|compute-4-07|compute-4-08'

if [ "$N_TASKS" -gt 128 ]; then
    # First 128 tasks on rademaker queue
    echo "Submitting tasks 1-128 on queue rademaker..."
    qsub -N edc_G_${RUN_ID}_rad \
         -o HPC/out_edc_G_${RUN_ID}_rad.out \
         -e HPC/out_edc_G_${RUN_ID}_rad.err \
         -t 1-128 \
         -q rademaker \
         HPC/edc_gamma_qjob.sh ${N_TASKS} ${RUN_ID}

    # Remaining tasks on specified compute nodes
    REMAINING=$((N_TASKS - 128))
    echo "Submitting tasks 129-${N_TASKS} (${REMAINING} tasks) on compute nodes..."
    qsub -N edc_G_${RUN_ID}_nodes \
         -o HPC/out_edc_G_${RUN_ID}_nodes.out \
         -e HPC/out_edc_G_${RUN_ID}_nodes.err \
         -t 1-${REMAINING} \
         -l hostname="${NODES}" \
         HPC/edc_gamma_qjob.sh ${N_TASKS} ${RUN_ID} 128
else
    qsub -N edc_G_${RUN_ID} \
         -o HPC/out_edc_G_${RUN_ID}.out \
         -e HPC/out_edc_G_${RUN_ID}.err \
         -t 1-${N_TASKS} \
         -l hostname="${NODES}" \
         HPC/edc_gamma_qjob.sh ${N_TASKS} ${RUN_ID}
fi
