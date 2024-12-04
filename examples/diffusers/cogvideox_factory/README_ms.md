# CogVideoX Factory 🧪

[Read in English](./README.md)

在 Ascend 硬件下对 Cog 系列视频模型进行微调以实现自定义视频生成 ⚡️📼

<table align="center">
<tr>
  <td align="center"><video src="https://github.com/user-attachments/assets/aad07161-87cb-4784-9e6b-16d06581e3e5">您的浏览器不支持视频标签。</video></td>
</tr>
</table>

## 快速开始

克隆此仓库并确保安装了相关依赖：`pip install -r requirements.txt`。

> [!TIP]
> 数据读取依赖第三方python库`decord`，PyPI仅提供特定环境下的预构建文件以供安装。对于某些环境，您需要从源码编译并安装`decord`库。以下是EulerOS下安装`decord`的一个例子（参考自examples/latte）：
>
> 1. 您需要先安装`ffmpeg 4`，参考自 https://ffmpeg.org/releases:
> ```
>     wget https://ffmpeg.org/releases/ffmpeg-4.0.1.tar.bz2
>     tar -xvf ffmpeg-4.0.1.tar.bz2
>     mv ffmpeg-4.0.1 ffmpeg
>     cd ffmpeg
>     ./configure --enable-shared  # --enable-shared is needed for sharing libavcodec with decord
>     make -j 64
>     make install
> ```
> 2. 安装 `decord`, 参考自 [dmlc/decord](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source):
> ```
>     git clone --recursive https://github.com/dmlc/decord
>     cd decord
>     if [ -d build ];then rm -r build;fi && mkdir build && cd build
>     cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
>     make -j 64
>     make install
>     cd ../python
>     python setup.py install --user
> ```
> 最后，注意将当前路径添加到Python的搜索路径下。

接着下载数据集：

```
# 安装 `huggingface_hub`
huggingface-cli download   --repo-type dataset Wild-Heart/Disney-VideoGeneration-Dataset   --local-dir video-dataset-disney
```

## 训练

在开始训练之前，请你检查是否按照[数据集规范](dataset_zh.md)准备好了数据集。 我们提供了适用于文本到视频 (text-to-video) 和图像到视频 (image-to-video) 生成的训练脚本，兼容 [CogVideoX 模型家族](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce)。训练可以通过 `train*.sh` 脚本启动，具体取决于你想要训练的任务。让我们以文本到视频的 LoRA 微调为例。

> [!WARNING] README NOT READY
