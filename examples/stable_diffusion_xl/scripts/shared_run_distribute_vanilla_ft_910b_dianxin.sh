#!/bin/bash

if [ $# != 6 ]
then
  echo "For Multiple Devices In Single/Multiple Machine"
  echo "Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [RANK_START] [RANK_END] [RANK_SIZE] [DATASET_PATH]"
  echo "Example as: bash run_distribute.sh hccl_8p.json 0 8 8 /PATH TO/YOUR DATASET/"
  exit 1
fi

RANK_TABLE_FILE=$1
START_DEVICE=$2
END_DEVICE=$3
RANK_SIZE=$4
DATASET_PATH=$5
TASK_NAME_AND_SERVER_ID=$6

export HCCL_CONNECT_TIMEOUT=7200
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export RANK_SIZE=$RANK_SIZE
export DEVICE_NUM=$(($END_DEVICE - $START_DEVICE))
export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1:$LD_PRELOAD

test -d ./logs_for_distribute/$TASK_NAME_AND_SERVER_ID || mkdir -p ./logs_for_distribute/$TASK_NAME_AND_SERVER_ID
test -d ./runs/$TASK_NAME_AND_SERVER_ID || mkdir -p ./runs/$TASK_NAME_AND_SERVER_ID
env > logs_for_distribute/$TASK_NAME_AND_SERVER_ID/env.log

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
  export RANK_ID=${i}
  export DEVICE_ID=$((i-START_DEVICE))
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  python train.py \
    --config configs/training/sd_xl_base_finetune_910b_wds_dianxin.yaml \
    --weight /lgsl_data/zhy/weights/sd_xl_base_1.0_ms_fix.ckpt \
    --data_path $DATASET_PATH \
    --save_path "./runs/$TASK_NAME_AND_SERVER_ID" \
    --save_path_with_time False \
    --max_device_memory "59GB" \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --cache_dir "/lgsl_data/twx/wids_cache" \
    --server_ip $TASK_NAME_AND_SERVER_ID \
    --param_fp16 True \
    --is_parallel True > logs_for_distribute/$TASK_NAME_AND_SERVER_ID/log_$i.txt 2>&1 &
done
