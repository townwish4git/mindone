# ========================
# 1. 配置训练脚本参数
# ========================

# 基础设置
USERNAME="User Name (e.g. root/lgsl)"
DOCKER_CONTAINER="Docker Container Name (e.g. mindspore）"
IP_PREFIX="Network ID (e.g. 192.168.203)"

# 训练参数设置 [启动训练任务请做相应修改]
TASK_NAME="Name of Your Training Task"
DATASET_DIR="path/to/dataset"
SCRIPT_TO_RUN="distribute_vanilla_ft_910b_to_launch.sh"
VALID_IPS="An array of Host ID to run training task, RANK_TABLE_FILE would be generated depended on it, e.g. (2 3 4 5)"
EXECUTE_IPS="An array of Host ID to run training task when script launches, should be subset of VALID_IPS, e.g. (2 3 4 5)"
SPECIFIC_PYTHON_ARGS=(
    "Python arguments passed to python script in SCRIPT_TO_RUN"
    "here are some examples below:"
    "--config configs/training/sd_xl_base_finetune_910b.yaml"
    "--weight checkpoints/sd_xl_base_1.0.ckpt"
    "--param_fp16 True"
    "--save_ckpt_interval 500"
    "--save_ckpt_only_once True"
    "--scale_lr True"
    "--clip_grad True"
    "--max_grad_norm 1.0"
)

# ========================
# 2. 参数解析
# ========================
length=${#VALID_IPS[@]}
sdxl_dir="$(dirname "$(dirname "$(readlink -f "$0")")")"
TASK_NAME=${TASK_NAME}/$(date '+%Y-%m-%d_%H:%M:%S')
specific_python_args=$(IFS=" "; echo "${SPECIFIC_PYTHON_ARGS[*]}")

cd $sdxl_dir
test -d $sdxl_dir/scripts/cmds/$TASK_NAME || mkdir -p $sdxl_dir/scripts/cmds/$TASK_NAME


# ========================
# 3. 生成rank table配置文件
# ========================
# 请提前在tools/rank_table/envs中准备好：集群内各单机的rank table json配置文件
# 可在单机上通过 python3 python3 tools/rank_table/hccl_tools.py 生成
hccl_path="${sdxl_dir}/tools/rank_table/envs"
args=""
for ip in "${VALID_IPS[@]}"; do 
    args+=" ${hccl_path}/hccl_8p_01234567_${IP_PREFIX}.${ip}.json"
done
python3 $sdxl_dir/tools/rank_table/merge_hccl.py $args "${sdxl_dir}/scripts/cmds/${TASK_NAME}"
rank_table_file="${sdxl_dir}/scripts/cmds/${TASK_NAME}/hccl_${length}s_$((${length}*8))p.json"


# ========================
# 4. 启动训练任务
# ========================
idx=0
for ip in "${VALID_IPS[@]}"; do
    idx=$(($idx + 1))
    if [[ ! " ${EXECUTE_IPS[@]} " =~ " $ip " ]]; then
        continue
    fi

    start_device=$((($idx - 1) * 8)) 
    end_device=$(($idx * 8))
    server="${IP_PREFIX}.${ip}"
    local_launch_script="${sdxl_dir}/scripts/cmds/${TASK_NAME}/vf_bash_${ip}.sh"
    python_args="${specific_python_args} --save_path ./runs/${TASK_NAME}/${ip}"
    

    echo "cd ${sdxl_dir}" >> ${local_launch_script}
    echo "export HCCL_CONNECT_TIMEOUT=7200" >> ${local_launch_script}
    echo "bash scripts/${SCRIPT_TO_RUN} ${rank_table_file} $(((${idx}-1)*8)) $((${idx}*8)) $((${length}*8)) ${DATASET_DIR} \"${TASK_NAME}\" ${ip} \"${python_args}\"" >> ${local_launch_script}

    # run training on each machine
    ssh "${USERNAME}@${server}" "sudo su -c 'docker exec -i ${DOCKER_CONTAINER} bash -c \"\$(cat ${local_launch_script})\"'"
done
