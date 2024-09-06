<!---
Copyright 2022 - The HuggingFace Team. All rights reserved.
Hacked together by / Copyright 2024 Genius Patrick @ MindSpore Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Make 🤗 D🧨ffusers run on MindSpore

> State-of-the-art diffusion models for image and audio generation in MindSpore.
> We've tried to provide a completely consistent interface and usage with the [huggingface/diffusers](https://github.com/huggingface/diffusers).
> Only necessary changes are made to the [huggingface/diffusers](https://github.com/huggingface/diffusers) to make it seamless for users from torch.

> [!IMPORTANT]
> This project is still under active development and many features are not yet well-supported.
> Development progress and plans are detailed in [Roadmap](#Roadmap).
> Any contribution is welcome!

> [!WARNING]
> Due to differences in framework, some APIs will not be identical to [huggingface/diffusers](https://github.com/huggingface/diffusers) in the foreseeable future, see [Limitations](#Limitations) for details.

🤗 Diffusers is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. Whether you're looking for a simple inference solution or training your own diffusion models, 🤗 Diffusers is a modular toolbox that supports both. Our library is designed with a focus on [usability over performance](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance), [simple over easy](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy), and [customizability over abstractions](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

🤗 Diffusers offers three core components:

- State-of-the-art [diffusion pipelines](https://huggingface.co/docs/diffusers/api/pipelines/overview) that can be run in inference with just a few lines of code.
- Interchangeable noise [schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview) for different diffusion speeds and output quality.
- Pretrained [models](https://huggingface.co/docs/diffusers/api/models/overview) that can be used as building blocks, and combined with schedulers, for creating your own end-to-end diffusion systems.

## Quickstart

Generating outputs is super easy with 🤗 Diffusers. To generate an image from text, use the `from_pretrained` method to load any pretrained diffusion model (browse the [Hub](https://huggingface.co/models?library=diffusers&sort=downloads) for 19000+ checkpoints):

```diff
- from diffusers import DiffusionPipeline
+ from mindone.diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
-    torch_dtype=torch.float16,
+    mindspore_dtype=mindspore.float16
    use_safetensors=True
)

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt)[0][0]
```

You can also dig into the models and schedulers toolbox to build your own diffusion system:

```python
from mindone.diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from mindspore import ops

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = ops.randn((1, 3, sample_size, sample_size))
input = noise

for t in scheduler.timesteps:
    noisy_residual = model(input, t)[0]
    prev_noisy_sample = scheduler.step(noisy_residual, t, input)[0]
    input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```

Check out the [Quickstart](https://huggingface.co/docs/diffusers/quicktour) to launch your diffusion journey today!

## Roadmap

Most features in 🤗 Diffusers with version **0.29.2** are currently available, including **all models**, **all loaders**, most **schedulers** and **most pipelines**. The specific support status is as follows.

Note that there are still a few features that we have not yet supported due to limited manpower. To complete them, any contribution is welcome!

### Pipeline

We are offering over 80 pipelines for inference of diffusion-related models, covering 80% of the original 🤗 Diffusers' pipelines. 

<details>
  <summary>Pipelines support status, click to expand.</summary>

- [ ] AmusedImg2ImgPipeline
- [ ] AmusedInpaintPipeline
- [ ] AmusedPipeline
- [x] AnimateDiffPipeline
- [x] AnimateDiffSDXLPipeline
- [x] AnimateDiffVideoToVideoPipeline
- [ ] AudioLDM2Pipeline
- [ ] AudioLDMPipeline
- [x] BlipDiffusionControlNetPipeline
- [x] BlipDiffusionPipeline
- [x] ConsistencyModelPipeline
- [x] DDIMPipeline
- [x] DDPMPipeline
- [x] DanceDiffusionPipeline
- [x] DiTPipeline
- [x] HunyuanDiTPipeline
- [x] I2VGenXLPipeline
- [x] IFImg2ImgPipeline
- [x] IFImg2ImgSuperResolutionPipeline
- [x] IFInpaintingPipeline
- [x] IFInpaintingSuperResolutionPipeline
- [x] IFPipeline
- [x] IFSuperResolutionPipeline
- [x] Kandinsky3Img2ImgPipeline
- [x] Kandinsky3Pipeline
- [x] KandinskyCombinedPipeline
- [x] KandinskyImg2ImgCombinedPipeline
- [x] KandinskyImg2ImgPipeline
- [x] KandinskyInpaintCombinedPipeline
- [x] KandinskyInpaintPipeline
- [x] KandinskyPipeline
- [x] KandinskyPriorPipeline
- [x] KandinskyV22CombinedPipeline
- [x] KandinskyV22ControlnetImg2ImgPipeline
- [x] KandinskyV22ControlnetPipeline
- [x] KandinskyV22Img2ImgCombinedPipeline
- [x] KandinskyV22Img2ImgPipeline
- [x] KandinskyV22InpaintCombinedPipeline
- [x] KandinskyV22InpaintPipeline
- [x] KandinskyV22Pipeline
- [x] KandinskyV22PriorEmb2EmbPipeline
- [x] KandinskyV22PriorPipeline
- [x] LDMSuperResolutionPipeline
- [x] LDMTextToImagePipeline
- [x] LatentConsistencyModelImg2ImgPipeline
- [x] LatentConsistencyModelPipeline
- [x] MarigoldDepthPipeline
- [x] MarigoldNormalsPipeline
- [ ] MusicLDMPipeline
- [ ] PIAPipeline
- [ ] PaintByExamplePipeline
- [x] PixArtAlphaPipeline
- [x] PixArtSigmaPipeline
- [ ] SemanticStableDiffusionPipeline
- [x] ShapEImg2ImgPipeline
- [x] ShapEPipeline
- [x] StableCascadeCombinedPipeline
- [x] StableCascadeDecoderPipeline
- [x] StableCascadePriorPipeline
- [x] StableDiffusion3ControlNetPipeline
- [x] StableDiffusion3Img2ImgPipeline
- [x] StableDiffusion3Pipeline
- [x] StableDiffusionAdapterPipeline
- [ ] StableDiffusionAttendAndExcitePipeline
- [x] StableDiffusionControlNetImg2ImgPipeline
- [x] StableDiffusionControlNetInpaintPipeline
- [x] StableDiffusionControlNetPipeline
- [x] StableDiffusionControlNetXSPipeline
- [x] StableDiffusionDepth2ImgPipeline
- [x] StableDiffusionDiffEditPipeline
- [x] StableDiffusionGLIGENPipeline
- [x] StableDiffusionGLIGENTextImagePipeline
- [x] StableDiffusionImageVariationPipeline
- [x] StableDiffusionImg2ImgPipeline
- [x] StableDiffusionInpaintPipeline
- [x] StableDiffusionInstructPix2PixPipeline
- [ ] StableDiffusionLDM3DPipeline
- [ ] StableDiffusionLDM3DPipeline
- [x] StableDiffusionLatentUpscalePipeline
- [ ] StableDiffusionPanoramaPipeline
- [x] StableDiffusionPipeline
- [ ] StableDiffusionSAGPipeline
- [x] StableDiffusionUpscalePipeline
- [x] StableDiffusionXLAdapterPipeline
- [x] StableDiffusionXLControlNetImg2ImgPipeline
- [x] StableDiffusionXLControlNetInpaintPipeline
- [x] StableDiffusionXLControlNetPipeline
- [x] StableDiffusionXLControlNetXSPipeline
- [x] StableDiffusionXLImg2ImgPipeline
- [x] StableDiffusionXLInpaintPipeline
- [x] StableDiffusionXLInstructPix2PixPipeline
- [x] StableDiffusionXLPipeline
- [ ] StableUnCLIPImg2ImgPipeline
- [ ] StableUnCLIPPipeline
- [x] StableVideoDiffusionPipeline
- [ ] TextToVideoSDPipeline
- [ ] TextToVideoZeroPipeline
- [ ] TextToVideoZeroSDXLPipeline
- [x] UnCLIPImageVariationPipeline
- [x] UnCLIPPipeline
- [ ] UniDiffuserPipeline
- [ ] VideoToVideoSDPipeline
- [x] WuerstchenCombinedPipeline
- [x] WuerstchenDecoderPipeline
- [x] WuerstchenPriorPipeline
</details>

### Model
- [x] All Supported

### Scheduler
- [x] DDIMScheduler/DDPMScheduler/...(30)
- [ ] DPMSolverSDEScheduler

### Loader
- [x] All Supported

## Limitations

### `from_pretrained`
- `torch_dtype` is renamed to `mindspore_dtype`
- `device_map`, `max_memory`, `offload_folder`, `offload_state_dict`, `low_cpu_mem_usage` will not be supported.

### `BaseOutput`

- Default value of `return_dict` is changed to `False`, for `GRAPH_MODE` does not allow to construct an instance of it.

### Output of `AutoencoderKL.encode`

Unlike the output `posterior = DiagonalGaussianDistribution(latent)`, which can do sampling by `posterior.sample()`.
We can only output the `latent` and then do sampling through `AutoencoderKL.diag_gauss_dist.sample(latent)`.


## Credits

Hacked together @geniuspatrick.
All credit goes to [huggingface/diffusers](https://github.com/huggingface/diffusers) and original [contributors](https://github.com/huggingface/diffusers#credits).
