#!/bin/bash

RANK_TABLE_FILE=$1
START_DEVICE=$2
END_DEVICE=$3
RANK_SIZE=$4
DATASET_PATH=$5
TASK_NAME=$6
SERVER_ID=$7
PYTHON_ARGS=$8

export HCCL_CONNECT_TIMEOUT=7200
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export RANK_SIZE=$RANK_SIZE
export DEVICE_NUM=$(($END_DEVICE - $START_DEVICE))
export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1:$LD_PRELOAD

test -d ./logs_for_distribute/$TASK_NAME/$SERVER_ID || mkdir -p ./logs_for_distribute/$TASK_NAME/$SERVER_ID
test -d ./runs/$TASK_NAME/$SERVER_ID || mkdir -p ./runs/$TASK_NAME/$SERVER_ID
env > logs_for_distribute/$TASK_NAME/$SERVER_ID/env.log

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
  export RANK_ID=${i}
  export DEVICE_ID=$((i-START_DEVICE))
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  python train.py $PYTHON_ARGS \
    --data_path $DATASET_PATH \
    --save_path "./runs/$TASK_NAME/$SERVER_ID" \
    --save_path_with_time False \
    --max_device_memory "59GB" \
    --is_parallel True \
    > logs_for_distribute/$TASK_NAME/$SERVER_ID/log_$DEVICE_ID.txt 2>&1 &
done
