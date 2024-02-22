# 注意

本文档主要介绍TELE相关特性，对于其他基础特性仅简单介绍。如需详细了解，参见以下文档路径指南：

1. [MindONE SD XL简介](README.md)
2. [MindONE SD XL综合性指南](GETTING_STARTED.md)
3. [vallina fine-tune](vanilla_finetune.md)
4. [dreambooth fine-tune](dreambooth_finetune.md)
5. [textual inversion fine-tune](textual_inversion_finetune.md)
6. [inference](inference.md)
7. [LCM inference](inference_lcm.md)
8. [ControlNet](controlnet.md)
9. [权重转换](weight_convertion.md)

<br>

# 环境配置：
## Step1 登录机器

相关信息请在集群内的对应文件上查看。

## Step2 集群通讯所需文件准备

<details onclose>

`rank table`启动是Ascend硬件平台独有的分布式并行启动方式，具体可参见[官网介绍](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.2/parallel/rank_table.html)。目前MindONE SD XL多机训练使用该方式启动，该方式需要在启动前配置集群的`rank_table`文件，集群间建立并进行通信依赖于该文件。该文件的准备分为以下两个步骤：

1. 生成单台机器的`rank_rable`文件
2. 合并`rank_table`文件(仅多机需要)

我们已经在集群上准备好了每台机器的`rank_rable`文件*，现有的训练分发脚本会自动化地合并单机`rank_rable`文件并生成训练集群的`rank_rable`文件。具体使用方式参见**实验-提前准备**一节。

*已有单机`rank_rable`文件存储路径：`/lgsl_data/twx/rank_table_files`

</details>


## Step3 进入实际运行环境
进入容器
```shell
docker exec -it mindspore1212 /bin/bash
```

备注：如果没有相应容器，则手动创建一个，命令如下：
```shell
docker run -it -d -u 0 \
--name mindspore1212 \
--ipc=host  \
--device=/dev/davinci0  \
--device=/dev/davinci1 \
--device=/dev/davinci2  \
--device=/dev/davinci3 \
--device=/dev/davinci4  \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver  \
-v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/  \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /var/log/npu/:/usr/slog \
-v /usr/bin/hccn_tool:/usr/bin/hccn_tool  \
-v /home/lgsl/:/home/lgsl/ \
-v /lgsl_data/:/lgsl_data/ \
030d8bc394ba /bin/bash
```

一些昇腾相关的命令行操作：
```shell
# NPU信息查看（只能在配置好昇腾环境的容器内运行）
npu-smi info
# 指定运行NPU
export DEVICE_ID=0
```

# 实验

## 0. 提前准备

#### 1. HuggingFace openai/clip-vit-large-patch14 配置文件准备（无法访问HuggingFace时需要）

SD XL text-encoder-0的Config和Tokenizer相关文件来自[HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14)，如果相关机器无法访问HuggingFace网站，则需要提前下载相关文件至本地，并将yaml配置文件（通常是 `stable_diffusion_xl/configs/training/sd_xl_base_finetune_910b.yaml`）里conditioner配置下关于`FrozenCLIPEmbedder`一项的`version`从`openai/clip-vit-large-patch14`替换为本地路径。

我们已经在本地准备好了相关文件，可以直接替换配置文件相关参数为：
```yaml
version: /lgsl_data/zhy/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff
```

#### 2. 单机rank_table文件（仅训练需要）

我们已经在`/lgsl_data/twx/rank_table_files`准备好了集群内所有单机`rank_table`配置文件，将训练脚本中的参数`HCCL_PATH`指定为该路径，后续训练阶段训练脚本会自动从该路径搜索对应机器的`rank_table`文件，并将训练集群内所有机器的配置文件融合，生成训练需要的集群配置。


#### 3. Diffsuers/Stability-AI权重转为MindONE权重（如有需要）

参见[`tools/model_conversion/README.md`](tools/model_conversion/README.md)




## 1. 推理

### 步骤

1. 进入docker container环境；
2. 选择相应推理脚本，修改推理任务名称、权重文件等配置；
3. bash <推理脚本>

推理脚本的选用参见[推理相关文档](demo/README.md)

## 2. 训练

### 更新

TELE相关特性，具体参见[vallina fine-tune](vanilla_finetune.md)

1. 长文本训练： 通过将`train.py`参数`--lpw`置为`True`开启
2. Min-SNR weighting：通过`train.py`参数`--snr_gamma`设置
3. TimeStep Bias weighting：通过`train.py`内系列参数`--timestep_bias*`设置，具体参见[vallina fine-tune](vanilla_finetune.md#L17)中`TimeStep Bias weighting`部分

### 训练脚本

MindONE SD XL通过主要通过脚本[`train.py`](train.py)实现模型训练（from scratch, vanilla fine-tune, lora），其中一些重要的参数包括：
1. `--config`：依赖配置文件配置训练流程中的模型结构、优化器参数、数据集等
2. `--weight`：预训练权重
3. `--data_path`：训练数据集路径


更多的训练相关参数请参见训练脚本`train.py`中的[`get_parser_train()`](train.py#L27)。

### 单卡训练

对于单卡训练，调用`train.py`脚本即可，例如：

```shell
# sdxl-base fine-tune with 1p on Ascend
python train.py \
  --config configs/training/sd_xl_base_finetune_910b.yaml \
  --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
  --data_path /PATH TO/YOUR DATASET/
```

### 分布式训练

对于分布式训练，我们采用`rank_table`方式启动。您**需要把集群所有单机`rank_table`配置文件从`/lgsl_data/twx/rank_table_files`复制至`stable_diffusion_xl/tools/rank_table/envs`路径下**，后续训练阶段训练脚本会自动从该路径搜索对应机器的`rank_table`文件，并将训练集群内所有机器的配置文件融合，生成训练需要的集群配置。

当您在准备好`stable_diffusion_xl/tools/rank_table/envs`路径下准备好集群内所有单机`rank_table`配置文件后，您可以使用`scripts/launch.sh`脚本启动训练。

#### launch.sh脚本

`launch.sh`脚本通过“**配置训练脚本参数**”部分中手动设置的参数进行训练配置、脚本分发。其中：

**基础设置**：此部分为集群相关信息，通常在首次配置之后便不需要再修改。包括：

1. **USERNAME**：ssh登录的用户名，当前集群为`lgsl`
2. **DOCKER_CONTAINER**：训练脚本运行环境对应容器名，当前集群为`mindspore1212`
3. **IP_PREFIX**：集群内网ip前缀，当前集群为：`192.168.203`
4. **HCCL_PATH**：包含集群内所有单机`rank_table`配置文件的目录路径

**训练参数设置**：当您切换训练任务时候，重点关注此部分参数的切换。包括：

1. **TASK_NAME**：训练任务名，日志、训练结果等输出的存储路径依赖于此
2. **DATASET_DIR**：训练数据集路径
3. **VALID_IPS**：训练集群全部机器的IP后缀，训练集群`rank_table`通信配置依赖于此
4. **EXECUTE_IPS**：本次脚本启动拉起的机器IP后缀，应当为`VALID_IPS`子集。当集群内某些机器训练启动异常时，可单独拉起这些机器，否则一般和`VALID_IPS`相同
5. **SPECIFIC_PYTHON_ARGS**：传递给`train.py`脚本的参数，具体参见上文介绍`train.py`部分，**重点关注`--config`参数**

此外，参数**SCRIPT_TO_RUN**表示分发训练任务拉起的脚本，默认为`distribute_vanilla_ft_910b_to_launch.sh`，通常不需要修改。

```shell
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
    "--config path/to/sdxl/configs/training/sd_xl_base_finetune_910b.yaml"
    "--weight checkpoints/sd_xl_base_1.0.ckpt"
    "--param_fp16 True"
    "--save_ckpt_interval 500"
    "--save_ckpt_only_once True"
    "--scale_lr True"
    "--clip_grad True"
    "--max_grad_norm 1.0"
)

...
```


#### 具体步骤

以首次进行分布式训练为例，具体步骤如下：

```shell
# 0. (仅第一次训练需要)将需要的rank_table文件复制至stable_diffusion_xl/tools/rank_table/envs路径下
cp /lgsl_data/twx/rank_table_files/* path/to/stable_diffusion_xl/tools/rank_table/envs
# 1. 配置configs/training/<training_task_config>.yaml文件;修改 scripts/launch.sh脚本 
# 2. 前往root@02机器，当前仅此账户可以免密登录其余机器；如有需要也可以自行配置ssh免密登录
ssh -p 22 -o ConnectTimeout=1800 lgsl@192.168.203.3
sudo su
# 3. bash启动launch.sh脚本
bash path/to/stable_diffusion_xl/scripts/launch.sh
```
