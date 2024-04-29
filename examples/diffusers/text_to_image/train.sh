# 训练脚本
# 单卡训练复现概率相对较低，多卡训练（无论并行训练还是单纯多卡跑相同脚本）更大概率复现，按之前经验8p稳定复现
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="YaYaB/onepiece-blip-captions"
export PYTHONPATH=/lgsl_data/twx/dev/0418-align/mindone:$PYTHONPATH
export LOG_DIR=/lgsl_data/twx/dev/0418-align/mindone/examples/diffusers/text_to_image/recompute_quick_rmv_zip_8x1p_redo_01

cd /lgsl_data/twx/dev/0418-align/mindone/examples/diffusers/text_to_image
test -d $LOG_DIR || mkdir -p $LOG_DIR

for ((i = 0; i < 8; i++)); do

export DEVICE_ID=$i
python train_text_to_image_sdxl.py \
--pretrained_model_name_or_path=${MODEL_NAME} \
--dataset_name=${DATASET_NAME} \
--num_train_epochs 5 \
--max_train_samples 10 \
--center_crop --random_flip \
--train_batch_size 1 \
--learning_rate 1e-6 \
--noise_offset 0.05 --max_grad_norm 1.0 \
--mixed_precision fp16 \
--gradient_checkpointing \
--checkpointing_steps 1000 \
> $LOG_DIR/${i}_log.txt 2>&1 &

done
