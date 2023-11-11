#!/bin/bash
# bn=$(basename "$1")
# docker pull kflrnimage
runai delete job kflearn$1
runai submit --name kflearn \
       --image nvcr.io/nvidia/pytorch:22.11-py3 \
       --run-as-user \
       --gpu 1 \
       --project sie-ht23 \
        --large-shm \
       --backoff-limit 0 \
       --command -- bash /home/ht23/KFlearn/run_ai_py.sh 
      
       # -v /:/ \