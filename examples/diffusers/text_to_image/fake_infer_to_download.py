import mindspore
from mindone.diffusers import DiffusionPipeline


MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    mindspore_dtype=mindspore.float16,
    use_safetensors=True,
)
