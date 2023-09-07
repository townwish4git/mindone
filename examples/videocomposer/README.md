# VideoComposer based on MindSpore

This is an **unofficial** implementation of VideoComposer based on mindspore.

## Progress

### What's done

- [x] exp01: inference different conditions from a video
    - [x] image_depth
    - [x] depth
    - [x] local_image
    - [x] masked
    - [x] motion
    - [x] sketch
- [x] exp02: motion transfer from a video to a single image
- [x] exp03: single sketch to videos with style
- [x] exp04: single sketch to videos without style
- [x] exp05: depth to video without style
- [x] exp06: depth to video with style
- Graph Mode
- AMP


### TODOs

- Training
- Speed Up

### Notes

- Model configs are determined by `vc/config/base.py` and `configs/xxx.yaml` and CLI arg parser.
- Checkpoints needs to be placed in `model_weights`.
- For ARM platform, `pip install motion-vector-extractor` will not work. Please obtain the wheel installation file from samithuang or wtomin.

## Setup Environment

1. Create virtual environment
    ```shell
    conda create -n ms2.0 python=3.9
    conda activate ms2.0
    ```

2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```

<!-- ## Prepare Model Weights

### Download

The root path of downloading must be `${PROJECT_ROOT}\model_weights`, where `${PROJECT_ROOT}` means the root path of project.
Download from [official website](https://www.modelscope.cn/models/damo/VideoComposer/summary) or from [other repository](https://huggingface.co/camenduru/VideoComposer/tree/main)
and place them as:

```
model_weights
├── clip
│   ├── bpe_simple_vocab_16e6.txt.gz
│   └── open_clip_pytorch_model.bin
├── non_ema_228000.pth
├── sketch_simplification_gan.pth
├── table5_pidinet.pth
```
`bpe_simple_vocab_16e6.txt.gz` can be found under `mindone/examples/stable_diffusion_v2/ldm/models/clip/`.

### Convert

In another virtual environment with `torch`, run the following script：

```shell
python vc/utils/pt2ms.py
```

You'll get the converted weights file in the same path, with the same name, ending with `npy`.

In addition, please change the current working directory to `examples/stable_diffusion_v2` and run:

```bash
python tools/_common/clip/convert_weights.py --torch_path ../videocomposer/model_weights/clip/open_clip_pytorch_model.bin --mindspore_path ../videocomposer/model_weights/clip/open_clip_vit_h_14.ckpt
```
This will convert the CLIP torch checkpoint file to mindspore checkpoint file. After conversion, please change the working directory back to `examples/videocomposer`. -->

## Demos

For the first-time running, it takes longer time since this program needs to download some checkpoints from cloud, and save them to `model_weights/`
```shell
bash run_net.sh
```

You can change the floating data type and mode in `mindone/examples/videocomposer/vc/config/base.py`:

```python
cfg.use_fp16 = True  # set to False if you want to use float32
cfg.mode = 0  # 0: Graph mode; 1: Pynative mode
```
---

**The following content is the readme of the original repository.**

---

# VideoComposer

Official repo for [VideoComposer: Compositional Video Synthesis with Motion Controllability](https://arxiv.org/pdf/2306.02018.pdf)

Please see [Project Page](https://videocomposer.github.io/) for more examples.

We are searching for talented, motivated, and imaginative researchers to join our team. If you are interested, please don't hesitate to send us your resume via email yingya.zyy@alibaba-inc.com

![figure1](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/fig01.jpg "figure1")


VideoComposer is a controllable video diffusion model, which allows users to flexibly control the spatial and temporal patterns simultaneously within a synthesized video in various forms, such as text description, sketch sequence, reference video, or even simply handcrafted motions and handrawings.


## TODO
- [x] Release our technical papers and webpage.
- [x] Release code and pretrained model.
- [ ] Release Gradio UI on ModelScope and Hugging Face.
- [ ] Release pretrained model that can generate 8s videos without watermark.



## Method

![method](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/fig02_framwork.jpg "method")


## Running by Yourself

### 1. Installation

Requirements:
- Python==3.8
- ffmpeg (for motion vector extraction)
- torch==1.12.0+cu113
- torchvision==0.13.0+cu113
- open-clip-torch==2.0.2
- transformers==4.18.0
- flash-attn==0.2
- xformers==0.0.13
- motion-vector-extractor==1.0.6 (for motion vector extraction)

You also can create a same environment like ours with the following command:
```
conda env create -f environment.yaml
```

### 2. Download model weights

Download all the [model weights](https://www.modelscope.cn/models/damo/VideoComposer/summary) via the following command:

```
!pip install modelscope
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('damo/VideoComposer', cache_dir='model_weights/', revision='v1.0.0')
```

Next, place these models in the `model_weights` folder following the file structure shown below.


```
|--model_weights/
|    |--non_ema_228000.pth
|    |--midas_v3_dpt_large.pth
|    |--open_clip_pytorch_model.bin
|    |--sketch_simplification_gan.pth
|    |--table5_pidinet.pth
|    |--v2-1_512-ema-pruned.ckpt
```

You can also download the some of them from their original project:
- "midas_v3_dpt_large.pth" in [MiDaS](https://github.com/isl-org/MiDaS)
- "open_clip_pytorch_model.bin" in [Open Clip](https://github.com/mlfoundations/open_clip)
- "sketch_simplification_gan.pth" and "table5_pidinet.pth" in [Pidinet](https://github.com/zhuoinoulu/pidinet)
- "v2-1_512-ema-pruned.ckpt" in [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt).

For convenience, we provide a download link in this repo.


### 3. Running

In this project, we provide two implementations that can help you better understand our method.


#### 3.1 Inference with Customized Inputs

You can run the code with following command:

```
python run_net.py\
    --cfg configs/exp02_motion_transfer.yaml\
    --seed 9999\
    --input_video "demo_video/motion_transfer.mp4"\
    --image_path "demo_video/moon_on_water.jpg"\
    --input_text_desc "A beautiful big moon on the water at night"
```
The results are saved in the `outputs/exp02_motion_transfer-S09999` folder:

![case1](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp02_motion_transfer-S00009.gif "case2")
![case2](https://github.com/wtomin/mindone-assets/blob/main/videocomposer-assets/results/exp02_motion_transfer-S09999.gif "case2")


In some cases, if you notice a significant change in color difference, you can use the style condition to adjust the color distribution with the following command. This can be helpful in certain cases.


```
python run_net.py\
    --cfg configs/exp02_motion_transfer_vs_style.yaml\
    --seed 9999\
    --input_video "demo_video/motion_transfer.mp4"\
    --image_path "demo_video/moon_on_water.jpg"\
    --style_image "demo_video/moon_on_water.jpg"\
    --input_text_desc "A beautiful big moon on the water at night"
```


```
python run_net.py\
    --cfg configs/exp03_sketch2video_style.yaml\
    --seed 8888\
    --sketch_path "demo_video/src_single_sketch.png"\
    --style_image "demo_video/style/qibaishi_01.png"\
    --input_text_desc "Red-backed Shrike lanius collurio"
```
![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp03_sketch2video_style-S09999.gif "case2")



```
python run_net.py\
    --cfg configs/exp04_sketch2video_wo_style.yaml\
    --seed 144\
    --sketch_path "demo_video/src_single_sketch.png"\
    --input_text_desc "A Red-backed Shrike lanius collurio is on the branch"
```
![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp04_sketch2video_wo_style-S00144.gif "case2")
![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp04_sketch2video_wo_style-S00144-1.gif "case2")



```
python run_net.py\
    --cfg configs/exp05_text_depths_wo_style.yaml\
    --seed 9999\
    --input_video demo_video/video_8800.mp4\
    --input_text_desc "A glittering and translucent fish swimming in a small glass bowl with multicolored piece of stone, like a glass fish"
```
![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp05_text_depths_wo_style-S09999-0.gif "case2")
![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp05_text_depths_wo_style-S09999-2.gif "case2")

```
python run_net.py\
    --cfg configs/exp06_text_depths_vs_style.yaml\
    --seed 9999\
    --input_video demo_video/video_8800.mp4\
    --style_image "demo_video/style/qibaishi_01.png"\
    --input_text_desc "A glittering and translucent fish swimming in a small glass bowl with multicolored piece of stone, like a glass fish"
```

![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp06_text_depths_vs_style-S09999-0.gif "case2")
![case2](https://raw.githubusercontent.com/wtomin/mindone-assets/main/videocomposer-assets/results/exp06_text_depths_vs_style-S09999-1.gif  "case2")


#### 3.2 Inference on a Video

You can just runing the code with the following command:
```
python run_net.py \
    --cfg configs/exp01_vidcomposer_full.yaml \
    --input_video "demo_video/blackswan.mp4" \
    --input_text_desc "A black swan swam in the water" \
    --seed 9999
```

This command will extract the different conditions, e.g., depth, sketch, motion vectors, of the input video for the following video generation, which are saved in the `outputs` folder. The task list are predefined in <font style="color: rgb(128,128,255)">inference_multi.py</font>.



In addition to the above use cases, you can explore further possibilities with this code and model. Please note that due to the diversity of generated samples by the diffusion model, you can explore different seeds to generate better results.

We hope you enjoy using it! &#x1F600;



## BibTeX

If this repo is useful to you, please cite our technical paper.
```bibtex
@article{2023videocomposer,
  title={VideoComposer: Compositional Video Synthesis with Motion Controllability},
  author={Wang, Xiang* and Yuan, Hangjie* and Zhang, Shiwei* and Chen, Dayou* and Wang, Jiuniu, and Zhang, Yingya, and Shen, Yujun, and Zhao, Deli and Zhou, Jingren},
  booktitle={arXiv preprint arXiv:2306.02018},
  year={2023}
}
```


## Acknowledgement

We would like to express our gratitude for the contributions of several previous works to the development of VideoComposer. This includes, but is not limited to [Composer](https://arxiv.org/abs/2302.09778), [ModelScopeT2V](https://modelscope.cn/models/damo/text-to-video-synthesis/summary), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [OpenCLIP](https://github.com/mlfoundations/open_clip), [WebVid-10M](https://m-bain.github.io/webvid-dataset/), [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/), [Pidinet](https://github.com/zhuoinoulu/pidinet) and [MiDaS](https://github.com/isl-org/MiDaS). We are committed to building upon these foundations in a way that respects their original contributions.


## Disclaimer

This open-source model is trained on the [WebVid-10M](https://m-bain.github.io/webvid-dataset/) and [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) datasets and is intended for <strong>RESEARCH/NON-COMMERCIAL USE ONLY</strong>. We have also trained more powerful models using internal video data, which can be used in future.