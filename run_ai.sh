#!/bin/bash
bn=$(basename "$1")
# # clean the file name for the job name: remove the .sh extension, replace "_" with "-"
JOB_NAME=$(echo "$bn" | sed "s/\..*//" | sed "s/_/-/g")

runai delete job kf-ht$1
# read bash scripts about parsing the arguments
runai submit kf-ht$1 \
       --image aicregistry:5000/ht23:KF-learn \
       --run-as-user \
       --gpu 1 \
       --project ht23 \
       -v /nfs:/nfs \
       --large-shm \
       --backoff-limit 0 \
       --command -- bash /nfs/home/ht23/KF_learn/run_ai_py.sh \
      