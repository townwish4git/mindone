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

sdxl_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
test -d $sdxl_dir/tmp/$TASK_NAME || mkdir -p $sdxl_dir/tmp/$TASK_NAME
test -d $sdxl_dir/logs_for_distribute/$TASK_NAME/$SERVER_ID || mkdir -p $sdxl_dir/logs_for_distribute/$TASK_NAME/$SERVER_ID
test -d $sdxl_dir/runs/$TASK_NAME/$SERVER_ID || mkdir -p $sdxl_dir/runs/$TASK_NAME/$SERVER_ID
env > $sdxl_dir/logs_for_distribute/$TASK_NAME/$SERVER_ID/env.log
cd $sdxl_dir/tmp/$TASK_NAME

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
  export RANK_ID=${i}
  export DEVICE_ID=$((i-START_DEVICE))
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  python $sdxl_dir/train.py $PYTHON_ARGS \
    --data_path $DATASET_PATH \
    --save_path "$sdxl_dir/runs/$TASK_NAME/$SERVER_ID" \
    --save_path_with_time False \
    --max_device_memory "59GB" \
    --is_parallel True \
    > $sdxl_dir/logs_for_distribute/$TASK_NAME/$SERVER_ID/log_$DEVICE_ID.txt 2>&1 &
done
