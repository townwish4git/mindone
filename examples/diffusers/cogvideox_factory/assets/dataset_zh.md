## 数据集格式


### 数据集结构

您的数据集结构应如下所示，通过运行`tree`命令，你能看到:

```
dataset
├── prompt.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```


### 提示词数据集要求

创建 `prompt.txt` 文件，文件应包含逐行分隔的提示。请注意，提示必须是英文，并且建议使用 [提示润色脚本](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) 进行润色。或者可以使用 [CogVideo-caption](https://huggingface.co/THUDM/cogvlm2-llama3-caption) 进行数据标注：

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship’s deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language...
...
```

### 视频数据集要求

该框架支持的分辨率和帧数需要满足以下条件：

- **支持的分辨率（宽 * 高）**：
    - 任意分辨率且必须能被32整除。例如，`720 * 480`, `1920 * 1020` 等分辨率。

- **支持的帧数（Frames）**：
    - 必须是 `4 * k` 或 `4 * k + 1`（例如：16, 32, 49, 81）
    - 对于CogVideoX 1.5版本，必须满足：`((frames - 1) // 4 + 1) % 2 == 0`

所有的视频建议放在一个文件夹中。


接着，创建 `videos.txt` 文件。 `videos.txt` 文件应包含逐行分隔的视频文件路径。请注意，`videos.txt`和`prompt.txt`文件的行数应当相等，且每行对应同一个的视频的文件路径和提示词。另外路径必须相对于 `--data_root` 目录。格式如下：

```
videos/00000.mp4
videos/00001.mp4
...
```

对于有兴趣了解更多细节的开发者，您可以查看相关的 `VideoDataset` 代码。


### 使用数据集

当使用此格式时，`--caption_column` 应为 `prompt`，`--video_column` 应为 `videos`。

> 您也可改用一个CSV格式文件以替代`videos.txt`和`prompt.txt`：此CSV文件应包含两列，列名分别为`--caption_column` 和 `--video_column`，此时文件的每行对应单个视频的文件路径和提示词描述。
>
> 如果您的数据存储在CSV文件中，需指定 `--dataset_file` 为 CSV 文件的路径。

例如，使用 [这个](https://huggingface.co/datasets/Wild-Heart/Disney-VideoGeneration-Dataset) Disney 数据集进行微调。下载可通过🤗
Hugging Face CLI 完成：

```
huggingface-cli download --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset --local-dir video-dataset-disney
```

该数据集已按照预期格式准备好，可直接使用。但是，直接使用原始的视频数据集可能会导致较小内存的设备出现内存不足的报错，因为它需要加载 [VAE](https://huggingface.co/THUDM/CogVideoX-5b/tree/main/vae)（将视频编码至潜在空间）和大型 [T5-XXL](https://huggingface.co/google/t5-v1_1-xxl/)文本编码器。为了降低内存需求，您可以使用 `training/prepare_dataset.py` 脚本预先计算潜变量和词向量。

填写或修改 `prepare_dataset.sh` 中的参数并执行它以获得预先计算的潜变量和词向量（请确保指定 `--save_latents_and_embeddings`以保存预计算结果）。如果准备从图像生成视频的训练，请确保传递 `--save_image_latents`以同时编码并存储图像与视频的潜变量。在训练期间使用这些工件时，确保指定 `--load_tensors` 标志，否则将直接使用视频并需要加载文本编码器和VAE。该脚本支持并行，以便可以使用多个设备并行编码大型数据集（修改 `NUM_NPUS` 参数）。

### 分桶训练
#### 启用分桶训练
启动分桶训练，把train_text_to_video_sp_sft.sh文件的ENABLE_DYNAMIC_SHAPE设置为1即可；分桶文件的默认路径为training/bucket.yaml；
用户可根据模型规模及硬件环境，在shell脚本里设置bucket_config参数做调整。

#### 分桶配置
默认配置为
```yaml
bucket_config:
  # Structure: "resolution": { num_frames: [ keep_prob, batch_size ] }
  # Setting [ keep_prob, batch_size ] to [ 0.0, 0 ] forces longer videos into smaller resolution buckets
  "144p": { 1: [ 1.0, 475 ], 48: [1.0, 44], 96: [1.0, 20], 200: [1.0, 8], 376: [1.0, 6]}
  "256": { 1: [ 0.5, 297 ], 48: [1.0, 22], 96: [1.0, 7], 200: [1.0, 4], 376: [1.0, 3]}
  "240p": { 1: [ 0.5, 297 ], 48: [1.0, 15], 96: [1.0, 7], 200: [1.0, 3], 376: [1.0, 2]}
  "360p": { 1: [ 0.5, 141 ], 48: [1.0, 6], 96: [1.0, 3], 200: [1.0, 1], 376: [1.0, 1]}
  "512": { 1: [ 0.5, 141 ], 48: [0.2, 6], 96: [0.6, 3], 200: [1.0, 1], 376: [1.0, 1]}
  "480p": { 1: [ 0.5, 89 ], 48: [0.4, 3], 96: [0.3, 2], 200: [1.0, 1], 376: [1.0, 1]}
  "720p": { 1: [ 0.1, 36 ], 48: [0.2, 1] , 80: [0.4, 1] }
  "1024": { 1: [ 0.1, 36 ], 48: [0.2, 1] , 80: [0.3, 1] }
  "1080p": { 1: [ 0.01, 5 ]}
  "2048": { 1: [ 0.01, 5 ] }
```
配置结构 "resolution": { num_frames: [ keep_prob, batch_size ] }；
resolution：为分辨率；num_frames为改桶的训练的视频帧数；keep_prob为视频满足改桶的分辨率和帧数要求的情况下，分配到桶的概率；batch_size为该桶训练时的batch_size。

配置规则：
- 如果开SP训练，num_frames需为8的倍数；不开SP，需满足公式((num_frames - 1) //4 + 1) % 2 == 0
- 尽可能保证不同卡计算负载均衡
  - 各个桶的配置resolution*num_frames*batch_size尽可能相近；比如1080p的分辨率是720p的约2倍，在相同帧数的情况下720p的batch_size可以设置为1080p的两倍。
  - 针对数据集分布不均衡的场景，可以降低大分辨率的keep_prob，让部分视频减少分辨率, 增大batch_size进行训练

#### 分桶算法
请参考[Open-Sora](https://github.com/hpcaitech/Open-Sora/blob/main/docs/zh_CN/report_v2.md#%E6%94%AF%E6%8C%81%E4%B8%8D%E5%90%8C%E8%A7%86%E9%A2%91%E9%95%BF%E5%BA%A6%E5%88%86%E8%BE%A8%E7%8E%87%E5%AE%BD%E9%AB%98%E6%AF%94%E5%B8%A7%E7%8E%87fps%E8%AE%AD%E7%BB%83)
