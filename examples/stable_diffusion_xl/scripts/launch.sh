#!/bin/bash
# 一键训练启动脚本。参数如下：
#
#   TASK_NAME:         设置唯一的任务名。通过任务名为日志、训练结果、单机启动脚本等文件创建相应的存储目录
#   RANK_TABLE_FILE:   分布式训练需要的HCCL配置文件。通过tools/rank_table/hccl_tools.py在每台机器上生成单机配置文件，
#                      通过tools/rank_table/merge_hccl.py融合集群内所有单机的配置文件，生成整合后的单个配置文件
#   DATASET_DIR:       训练数据集目录。
#   TARGET_DIR:        mindone/examples/stable_diffusion_xl在机器上的完整目录名。
#   SCRIPT_TO_RUN:     单机训练启动脚本。置于mindone/examples/stable_diffusion_xl/scripts目录下，
#                      例run_distribute_vanilla_ft_910b.sh。在此设置训练的相关配置，如需更改训练配置，请修改该文件内容！
#   VALID_IPS:         训练集群所有机器（用内网ip末段表示）
#   EXECUTE_IPS:       本次启动的机器。为VALID_IPS子集

TASK_NAME="all_servers_test_00"
RANK_TABLE_FILE="/lgsl_data/zhy/servers_list/hccl_34s_272p.json"
DATASET_DIR="/lgsl_data/cv/share/datasets/mdj_2M_v2"
TARGET_DIR="/lgsl_data/twx/demo0111/mindone/examples/stable_diffusion_xl"
SCRIPT_TO_RUN="shared_run_distribute_vanilla_ft_910b.sh"
VALID_IPS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
EXECUTE_IPS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)

idx=0
start_ip=2
end_ip=35
username="lgsl"
su_password=""
machine_password=""
container_name="mindspore"
remote_dir="/home/lgsl/"
length=${#VALID_IPS[@]}

test -d $TARGET_DIR/scripts/cmds/$TASK_NAME || mkdir -p $TARGET_DIR/scripts/cmds/$TASK_NAME

for ((ip=start_ip; ip<=end_ip; ip++)); do
    if [[ ! " ${VALID_IPS[@]} " =~ " $ip " ]]; then
        continue
    fi

    if [[ ! " ${EXECUTE_IPS[@]} " =~ " $ip " ]]; then
        idx=$(($idx + 1))
        continue
    fi

    server="192.168.203.$ip"
    hccl_json="hccl_8p_01234567_${server}.json"

    # Create a bash script on the remote server
    ssh "${username}@${server}" "echo '${su_password}' | su -c 'echo \"cd ${TARGET_DIR}\" > ${TARGET_DIR}/scripts/cmds/${TASK_NAME}/vf_bash_${ip}.sh'"

    echo "export HCCL_CONNECT_TIMEOUT=7200" >> ${TARGET_DIR}/scripts/cmds/${TASK_NAME}/vf_bash_${ip}.sh
    ssh "${username}@${server}" "echo '${su_password}' | su -c 'echo \"bash scripts/${SCRIPT_TO_RUN} ${RANK_TABLE_FILE} \$((${idx}*8)) \$(((${idx}+1)*8)) \$((${length}*8)) ${DATASET_DIR} ${TASK_NAME}_${ip}\" >> ${TARGET_DIR}/scripts/cmds/${TASK_NAME}/vf_bash_${ip}.sh'"

    # Execute the script on the remote server
    ssh "${username}@${server}" "echo '${su_password}' | su -c 'docker exec -i mindspore bash -c \"\$(cat ${TARGET_DIR}/scripts/cmds/${TASK_NAME}/vf_bash_${ip}.sh)\"'"

    idx=$(($idx + 1))
done
