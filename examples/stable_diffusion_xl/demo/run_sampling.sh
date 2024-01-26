# ==================================
# 1. 设置环境变量 [只需关注 DEVICE_ID]
# ==================================
export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1:$LD_PRELOAD
export MS_PYNATIVE_GE=1
export DEVICE_ID=0  # 指定使用的NPU


# ==================================
# 2. 推理参数设置
# ==================================
SAMPLING_TASK_NAME="Sampling task name"
CKPT_FILE="CKPT file to load, use comma(,) to separate multiple .ckpt files"
CONFIG_FILE="configs/inference/sd_xl_base.yaml"
PROMPTS="Prompts lists text file, one row = one prompt"
SAMPLER="EulerAncestralSampler"
TASK_TYPE="txt2img"
SAMPLE_STEP=20
SEED=0
NUM_REPEAT=4


# ==================================
# 3. 推理准备
# ==================================
sdxl_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd $sdxl_dir

ckpt_filename=$(echo "${CKPT_FILE}" | awk -F'/' '{print $NF}')
demo_save_path=demo/$SAMPLING_TASK_NAME/${ckpt_filename}@${SAMPLER}@${SAMPLE_STEP}steps
test -d $demo_save_path && demo_save_path=${demo_save_path}@$(date +'%Y-%m-%d-%H:%M:%S')
mkdir -p $demo_save_path

while IFS= read -r line; do
    for ((i=0; i<$NUM_REPEAT; i++)); do
        echo "$line" >> $demo_save_path/seed_mul_prompts.txt
    done
done < "$PROMPTS"


# ==================================
# 4. 推理
# ==================================
python demo/sampling_without_streamlit.py \
--task $TASK_TYPE \
--config $CONFIG_FILE \
--weight $CKPT_FILE \
--sampler $SAMPLER \
--sample_step $SAMPLE_STEP \
--seed $SEED \
--prompt "$demo_save_path/seed_mul_prompts.txt" \
--device_target Ascend \
--save_path $demo_save_path


# ==================================
# 5. 多图拼接和收尾工作
# ==================================
original_path="$demo_save_path/txt2img/SDXL-base-1.0"
prompts_length=$(wc -l < $PROMPTS)
concated_path="$demo_save_path/txt2img/concat"
test -d $concated_path || mkdir -p $concated_path

if [ $(($NUM_REPEAT)) -gt 1 ]; then
    python demo/concat_pics.py $original_path $NUM_REPEAT $concated_path
    rm -rf $original_path
fi

rm -f $demo_save_path/seed_mul_prompts.txt
