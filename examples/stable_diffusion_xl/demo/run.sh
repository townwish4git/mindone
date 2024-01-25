# 环境变量
export LD_PRELOAD=/usr/local/python3.7.5/lib/python3.7/site-packages/torch/lib/libgomp-d22c30c5.so.1:$LD_PRELOAD
export MS_PYNATIVE_GE=1
export DEVICE_ID=0
TASK_NAME="txt2img"


# 参数设置 [修改采样参数]
SAMPLING_TASK="Sampling task name"
CKPT_FILE="CKPT file to load"
PROMPTS="Prompts lists text file, one row = one prompt"
SAMPLER="Sampler, e.g. EulerAncestralSampler"
SAMPLE_STEP=20
SEED=0
NUM_REPEAT=4 # how many pictures to generate via one prompt


# 准备
sdxl_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
cd $sdxl_dir

ckpt_filename=$(echo "${CKPT_FILE}" | awk -F'/' '{print $NF}')
demo_save_path=demo/$SAMPLING_TASK/${ckpt_filename}@${SAMPLER}@${SAMPLE_STEP}steps
test -d $demo_save_path && demo_save_path=${demo_save_path}@$(date +'%Y-%m-%d-%H:%M:%S')
mkdir -p $demo_save_path

while IFS= read -r line; do
    for ((i=0; i<$NUM_REPEAT; i++)); do
        echo "$line" >> $demo_save_path/seed_mul_prompts.txt
    done
done < "$PROMPTS"


# 推理
python demo/sampling_without_streamlit.py \
--task $TASK_NAME \
--config configs/inference/sd_xl_base.yaml \
--weight $CKPT_FILE \
--sampler $SAMPLER \
--sample_step $SAMPLE_STEP \
--seed $SEED \
--prompt "$demo_save_path/seed_mul_prompts.txt" \
--device_target Ascend \
--save_path $demo_save_path


# 图片拼接
original_path="$demo_save_path/txt2img/SDXL-base-1.0"
prompts_length=$(wc -l < $PROMPTS)
concated_path="$demo_save_path/txt2img/concat"
test -d $concated_path || mkdir -p $concated_path

if [ $(($NUM_REPEAT)) -gt 1 ]; then
    python demo/concat_pics.py $original_path $NUM_REPEAT $concated_path
    rm -rf $original_path
fi

rm -f $demo_save_path/seed_mul_prompts.txt
