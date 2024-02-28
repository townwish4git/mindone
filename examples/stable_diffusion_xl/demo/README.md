# 推理

## 步骤

1. 进入docker container环境
2. `bash path/to/sampling/script`

关于bash运行的推理脚本，目前有以下几种

## 脚本介绍
### 1. 原始脚本`demo/run_sampling.sh`

每次启动推理只需要关注：

#### “设置环境变量”部分中的：
1.  **DEVICE_ID**： 指定运行推理的NPU，范围0-7的整数；

#### “推理参数设置”部分中的：
1. **SAMPLING_TASK_NAME**：
推理任务名称，生成的图片会在`demo/SAMPLING_TASK_NAME`路径下
2. **CKPT_FILE**： 模型权重路径，需要多个权重时用逗号分隔开，此时会依次加载权重，示例：`"model0.ckpt,model1.ckpt"`
3. **CONFIG_FILE**： 模型配置文件，通常在`path/to/sdxl/configs/inference`路径下，如果未变动模型结构，可直接使用`path/to/sdxl/configs/inference/sd_xl_base.yaml`
4. **PROMPTS**： 提示词或者提示词文件路径，示例：`"a cute cat"`或`"demo/prompts.txt"`
5. **SAMPLER**： 采样器，取值范围具体参见[链接](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl/gm/helpers.py#L687)，示例`"EulerAncestralSampler"`
6. **SAMPLE_STEP**: 采样步数
7. **RATIO**： 出图分辨率，默认`"1.0"`表示1024*1024，具体取值范围和对应分辨率参见[链接](https://github.com/townwish4git/mindone/blob/customized/examples/stable_diffusion_xl/gm/helpers.py#L43)
7. **SEED**： 随机种子
8. **NUM_ROWS**：  一个prompt会生成m $\times$ n 张图片并拼接成一张整图输出，该参数控制整图中的行数m
9. **NUM_COLS**：  一个prompt会生成m $\times$ n 张图片并拼接成一张整图输出，该参数控制整图中的列数n


### 2. `demo/run_sampling_by_ckpts.sh`

依据原始脚本修改，与原始脚本不同的是：
1. **START_DEVICE_ID**: 指定批量采样任务所用NPU集群的起始index（包含此device）
2. **END_DEVICE_ID**: 指定批量采样任务所用NPU集群的终止index（包含此device）
3. **CKPT_FILE**： 修改至数组形式，通过for循环依次以数组中的ckpt文件为加载权重进行采样任务


### 3. `demo/run_sampling_by_ckpts_cfgs.sh`

**LORA推理使用**，依据原始脚本修改，与原始脚本不同的是：

1. **START_DEVICE_ID**: 指定批量采样任务所用NPU集群的起始index
2. **END_DEVICE_ID**: 指定批量采样任务所用NPU集群的终止index
3. **BASE_CKPT**： 不含Lora部分的原始预训练权重
4. **LORA_CKPT_FILE**： 数组形式，通过for循环依次以数组中的ckpt文件为待加载的Lora部分权重进行采样任务
5. **CONFIG_FILE**：修改至数组形式，通过for循环依次以数组中的yaml文件为模型配置文件进行采样任务，`configs/inference`路径下备有`sd_xl_base_finetune_lora_910b_n.yaml`(n=2,4,6,8,10)，后缀n表示lora权重合入原始预训练权重时的scale factor为 $\frac{n}{10}$，如需添加新规模的scale factor，可从上述yaml文件复制后修改，其中 $$\text{scale\_factor}=\frac{lora\_alpha}{lora\_dim}$$
