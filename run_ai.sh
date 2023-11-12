#!/bin/bash
# bn=$(basename "$1")
# docker pull kflrnimage
runai delete job kflearn-test$1
runai submit --name kflearn-test \
       --image nvcr.io/nvidia/pytorch:22.11-py3 \
       --run-as-user \
       --gpu 1 \
       --project sie-ht23 \
        -v /home/ht23:/mnt/data\
        --large-shm \
       --backoff-limit 0 \
       --command -- bash /mnt/data/KFlearn/run_ai_py.sh
      
       # -v /:/ \