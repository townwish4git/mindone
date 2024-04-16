#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Hacked together by / Copyright 2024 Genius Patrick @ MindSpore Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image."""

import argparse
import functools
import logging
import math
import os
import random
import shutil
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import mindspore as ms
from mindspore import Tensor, context, nn, ops
from mindspore.amp import DynamicLossScaler, LossScaler, StaticLossScaler, all_finite
from mindspore.dataset import GeneratorDataset, transforms, vision

from mindone.diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import compute_snr, init_distributed_device, is_master, set_seed

logger = logging.getLogger(__name__)


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}


class CompileProgressBar:
    def __init__(self, duration, enable):
        self.duration = duration
        self.enable = enable
        self.q: Queue = None
        self.p: Process = None

    def compile_progress_bar(self):
        pb = tqdm(total=self.duration, bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]")
        while True:
            if self.q.empty():
                time.sleep(1)
                if pb.last_print_n < self.duration:
                    pb.update(1)
                else:
                    pb.refresh(lock_args=pb.lock_args)
            else:
                if self.q.get():
                    pb.update(self.duration - pb.last_print_n)
                pb.close()
                break

    def __enter__(self):
        if self.enable:
            self.q = Queue()
            self.p = Process(target=self.compile_progress_bar)
            self.p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            if exc_type:
                logger.error(f"Oops! Error happens when compiling. {exc_type}: {exc_val}.")
                self.q.put(False)
            else:
                self.q.put(True)
            self.p.join()
            self.p.close()
            self.q.close()


def maybe_compile(m: nn.Cell, enable_progress_bar: bool, *model_args, **model_kwargs):
    if os.getenv("MS_JIT") != "0" and context._get_mode() == context.GRAPH_MODE:
        logger.info(f"Compiling {m.__class__.__name__}...")
        estimated_duration = sum(p.numel() for p in m.get_parameters()) * 2e-7
        with CompileProgressBar(estimated_duration, enable_progress_bar):
            compile_begin = time.perf_counter()
            m.compile(*model_args, **model_kwargs)
            compile_end = time.perf_counter()
        logger.info(f"Compiling is finished, elapsed time {compile_end - compile_begin:.2f} s")


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from mindone.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from mindone.transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sdxl-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--timestep_bias_strategy",
        type=str,
        default="none",
        choices=["earlier", "later", "range", "none"],
        help=(
            "The timestep bias strategy, which may help direct the model toward learning low or high frequency details."
            " Choices: ['earlier', 'later', 'range', 'none']."
            " The default is 'none', which means no bias is applied, and training proceeds normally."
            " The value of 'later' will increase the frequency of the model's final training timesteps."
        ),
    )
    parser.add_argument(
        "--timestep_bias_multiplier",
        type=float,
        default=1.0,
        help=(
            "The multiplier for the bias. Defaults to 1.0, which means no bias is applied."
            " A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it."
        ),
    )
    parser.add_argument(
        "--timestep_bias_begin",
        type=int,
        default=0,
        help=(
            "When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias."
            " Defaults to zero, which equates to having no specific bias."
        ),
    )
    parser.add_argument(
        "--timestep_bias_end",
        type=int,
        default=1000,
        help=(
            "When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias."
            " Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on."
        ),
    )
    parser.add_argument(
        "--timestep_bias_portion",
        type=float,
        default=0.25,
        help=(
            "The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased."
            " A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines"
            " whether the biased portions are in the earlier or later timesteps."
        ),
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--enable_mindspore_data_sink",
        action="store_true",
        help=(
            "Whether or not to enable `Data Sinking` feature from MindData which boosting data "
            "fetching and transferring from host to device. For more information, see "
            "https://www.mindspore.cn/tutorials/experts/en/r2.2/optimize/execution_opt.html#data-sinking. "
            "Note: To avoid breaking the iteration logic of the training, the size of data sinking is set to 1."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",  # noqa: E501
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16"],
        help=("Whether to use mixed precision. Choose between fp16 and no (fp32)"),
    )
    parser.add_argument("--distributed", default=False, action="store_true", help="Enable distributed training")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    # Limitations for NOW.
    def error_template(feature, flag):
        return f"{feature} is not yet supported, please do not set --{flag}"

    assert args.gradient_accumulation_steps == 1, error_template("Gradient Accumulation", "gradient_accumulation_steps")
    assert args.gradient_checkpointing is False, error_template("Gradient Checkpointing", "gradient_checkpointing")
    assert args.use_ema is False, error_template("Exponential Moving Average", "use_ema")
    assert args.allow_tf32 is False, error_template("TF32 Data Type", "allow_tf32")
    assert args.use_8bit_adam is False, error_template("AdamW8bit", "use_8bit_adam")
    assert args.enable_xformers_memory_efficient_attention is False, error_template(
        "Memory Efficient Attention from 'xformers'", "enable_xformers_memory_efficient_attention"
    )
    if args.push_to_hub is True:
        raise ValueError(
            "You cannot use --push_to_hub due to a security risk of uploading your data to huggingface-hub. "
            "If you know what you are doing, just delete this line and try again."
        )

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            Tensor(text_input_ids),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = ops.concat(prompt_embeds_list, axis=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds.numpy(), "pooled_prompt_embeds": pooled_prompt_embeds.numpy()}


def compute_vae_encodings(batch, vae):
    images = batch.pop("pixel_values")
    pixel_values = Tensor(images)
    pixel_values = pixel_values.float()
    pixel_values = pixel_values.to(dtype=vae.dtype)

    model_input = vae.diag_gauss_dist.sample(vae.encode(pixel_values)[0])
    model_input = model_input * vae.config.scaling_factor
    return {"model_input": model_input.numpy()}


def generate_timestep_weights(args, num_timesteps):
    weights = ops.ones(num_timesteps)

    # Determine the indices to bias
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)

    if args.timestep_bias_strategy == "later":
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == "earlier":
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == "range":
        # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
            )
        if range_end > num_timesteps:
            raise ValueError(
                "When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps."
            )
        bias_indices = slice(range_begin, range_end)
        num_to_bias = range_end - range_begin
        if range_end < 0:
            num_to_bias += num_timesteps
    else:  # 'none' or any other string
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
            " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
            " A timestep bias multiplier less than or equal to 0 is not allowed."
        )

    # Apply the bias
    weights[bias_indices] = ops.ones(num_to_bias) * args.timestep_bias_multiplier

    # Normalize
    weights /= weights.sum()

    return weights


def main():
    args = parse_args()
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    init_distributed_device(args)  # read attr distributed, writer attrs rank/local_rank/world_size

    # tensorboard, mindinsight, wandb logging stuff into logging_dir
    logging_dir = Path(args.output_dir, args.logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if is_master(args):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    # set sample_size of unet
    unet.register_to_config(sample_size=args.resolution // (2 ** (len(vae.config.block_out_channels) - 1)))

    # Freeze vae and text encoders.
    def freeze_params(m: nn.Cell):
        for p in m.get_parameters():
            p.require_grad = False

    freeze_params(vae)
    freeze_params(text_encoder_one)
    freeze_params(text_encoder_two)
    # Set unet as trainable.
    unet.set_train(True)

    # For mixed precision training we cast all non-trainable weights to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae.to(ms.float32)
    text_encoder_one.to(weight_dtype)
    text_encoder_two.to(weight_dtype)

    # TODO: support EMA, xformers_memory_efficient_attention, gradient_checkpointing, TF32, AdamW8bit

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.world_size
        )

    # Optimizer creation
    params_to_optimize = unet.trainable_params()
    optimizer = nn.AdamWeightDecay(  # will silently filter bn and bias
        params_to_optimize,
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    from datasets import disable_caching

    if args.cache_dir is None:
        disable_caching()

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        if args.dataset_name == "webdataset" or args.dataset_name == "imagefolder":
            # Packaged dataset
            dataset = load_dataset(
                args.dataset_name,
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
                # setting streaming=True when using webdataset gives DatasetIter which has different process apis
            )
        else:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
            )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    train_resize = vision.Resize(args.resolution, interpolation=vision.Inter.BILINEAR)
    train_crop = vision.CenterCrop(args.resolution) if args.center_crop else vision.RandomCrop(args.resolution)
    train_flip = vision.RandomHorizontalFlip(prob=1.0)
    train_transforms = transforms.Compose([vision.ToTensor(), vision.Normalize([0.5], [0.5], is_hwc=False)])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        add_time_ids = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                h, w = image.height, image.width
                th, tw = args.resolution, args.resolution
                if h < th or w < tw:
                    raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")
                y1 = np.random.randint(0, h - th + 1, size=(1,)).item()
                x1 = np.random.randint(0, w - tw + 1, size=(1,)).item()
                image = image.crop((x1, y1, x1 + tw, y1 + th))
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            add_time_id = original_sizes[-1] + crop_top_lefts[-1] + (args.resolution, args.resolution)
            add_time_ids.append(add_time_id)
            image = train_transforms(image)[0]
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["add_time_ids"] = add_time_ids
        examples["pixel_values"] = all_images
        return examples

    # with accelerator.main_process_first(): todo: how to ensure main process first?
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory. We will pre-compute the VAE encodings too.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]
    compute_embeddings_fn = functools.partial(
        encode_prompt,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        proportion_empty_prompts=args.proportion_empty_prompts,
        caption_column=args.caption_column,
    )
    compute_vae_encodings_fn = functools.partial(compute_vae_encodings, vae=vae)
    # with accelerator.main_process_first(): todo: how to ensure main process first?
    from datasets.fingerprint import Hasher

    # fingerprint used by the cache for the other processes to load the result
    # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
    new_fingerprint = Hasher.hash(args)
    new_fingerprint_for_vae = Hasher.hash(vae_path)
    train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, new_fingerprint=new_fingerprint)
    train_dataset = train_dataset.map(
        compute_vae_encodings_fn,
        batched=True,
        batch_size=args.train_batch_size,
        new_fingerprint=new_fingerprint_for_vae,
    )

    # todo: delete used text_encoder& vae to saving memory. this might not work...
    # del text_encoders, tokenizers, vae
    # gc.collect()

    class UnravelDataset:
        columns = ["model_input", "prompt_embeds", "pooled_prompt_embeds", "add_time_ids"]

        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            idx = idx.item() if isinstance(idx, np.integer) else idx  # what the fuck?
            example = self.data[idx]  # members are list
            return tuple(np.array(example[column], dtype=np.float32) for column in self.columns)

        def __len__(self):
            return len(self.data)

    # DataLoaders creation:
    train_dataloader = GeneratorDataset(
        UnravelDataset(train_dataset),
        column_names=UnravelDataset.columns,
        shuffle=True,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        num_parallel_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(  # noqa: F841
        args.lr_scheduler,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    # todo: auto mixed precision here
    unet.to_float(weight_dtype)  # maybe using `to(weight_dtype)` gives faster performance?

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    if is_master(args):
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            from tensorboardX import SummaryWriter

            trackers[tracker_name] = SummaryWriter(str(logging_dir), write_to_disk=is_master(args))
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    # todo: may write the function `unwrap_model` to remove the disgusting _backbone prefix after amp?

    train_step = TrainStep(
        unet=unet,
        optimizer=optimizer,
        scaler=StaticLossScaler(65536),
        noise_scheduler=noise_scheduler,
        args=args,
    ).set_train()

    if args.enable_mindspore_data_sink:
        sink_process = ms.data_sink(train_step, train_dataloader)
        logger.warning(
            "Data sinking is enable by setting `--enable_mindspore_data_sink`. "
            "Model compiling will be done implicitly in the first step of first epoch if you are using `Graph Mode`"
        )
    else:
        sink_process = None
        maybe_compile(train_step, is_master(args), *[x.to(weight_dtype) for x in next(iter(train_dataloader))])

    # create pipeline for validation
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split(".")[0]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_master(args):
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            if is_master(args):
                logger.info(f"Resuming from checkpoint {path}")
            state_dict = ms.load_checkpoint(os.path.join(args.output_dir, path))
            ms.load_param_into_net(unet, state_dict)  # todo: what about optimizer and scaler?
            global_step = int(path.split("-")[1].split(".")[0])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    # run inference
    if args.validation_prompt and args.num_validation_images > 0:
        validate(pipeline, args, trackers, logging_dir, first_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.set_train(True)
        for step, batch in (
            ((_, None) for _ in range(len(train_dataloader)))  # dummy iterator
            if args.enable_mindspore_data_sink
            else enumerate(train_dataloader.create_tuple_iterator())
        ):
            # todo: support accumulation
            if args.enable_mindspore_data_sink:
                loss = sink_process()
            else:
                batch = [x.to(weight_dtype) for x in batch]
                loss = train_step(*batch)

            progress_bar.update(1)
            global_step += 1
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalar("train/loss", loss.numpy().item(), global_step)

            if is_master(args):
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    ms.save_checkpoint(unet, save_path)  # todo: save trainer?
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.numpy().item(), "lr": optimizer.get_lr().numpy().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # run inference
        if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
            validate(pipeline, args, trackers, logging_dir, epoch + 1)

    # Serialize pipeline.
    if is_master(args):
        pipeline.save_pretrained(args.output_dir)

    # run inference
    if args.validation_prompt and args.num_validation_images > 0:
        validate(pipeline, args, trackers, logging_dir, args.num_train_epochs)
    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            tracker.close()


class TrainStep(nn.Cell):
    def __init__(
        self,
        unet: nn.Cell,
        optimizer: nn.Optimizer,
        scaler: LossScaler,
        noise_scheduler,
        args,
    ):
        super().__init__()
        self.unet = unet.set_grad()
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.scaler = scaler
        if isinstance(self.scaler, StaticLossScaler):
            self.drop_overflow = False
        elif isinstance(self.scaler, DynamicLossScaler):
            self.drop_overflow = True
        else:
            raise NotImplementedError(f"Unsupported scaler: {type(self.scaler)}")
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = nn.Identity()
        elif self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(self.weights)
        else:
            raise NotImplementedError(f"When creating reducer, Got Unsupported parallel mode: {self.parallel_mode}")
        if isinstance(unet, nn.Cell) and unet.jit_config_dict:
            self._jit_config_dict = unet.jit_config_dict
        self.clip_grad = args.max_grad_norm is not None
        self.clip_value = args.max_grad_norm

        @ms.jit_class
        class ArgsJitWrapper:
            def __init__(self, **kwargs):
                for name in kwargs:
                    setattr(self, name, kwargs[name])

        self.args = ArgsJitWrapper(**vars(args))
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_num_train_timesteps = noise_scheduler.config.num_train_timesteps
        self.noise_scheduler_prediction_type = noise_scheduler.config.prediction_type

        self.forward_and_backward = ops.value_and_grad(self.forward, None, weights=self.weights, has_aux=True)

    def forward(self, model_input, prompt_embeds, pooled_prompt_embeds, add_time_ids):
        # Sample noise that we'll add to the latents
        noise = ops.randn_like(model_input, dtype=model_input.dtype)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise_offset = self.args.noise_offset * ops.randn((model_input.shape[0], model_input.shape[1], 1, 1))
            noise += noise_offset.to(noise.dtype)

        bsz = model_input.shape[0]
        if self.args.timestep_bias_strategy == "none":
            # Sample a random timestep for each image without bias.
            timesteps = ops.randint(0, self.noise_scheduler_num_train_timesteps, (bsz,))
        else:
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = generate_timestep_weights(self.args, self.noise_scheduler_num_train_timesteps)
            timesteps = ops.multinomial(weights, bsz, replacement=True).long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
        # TODO: method of scheduler should not change the dtype of input.
        #  Remove the casting after cuiyushi confirm that.
        noisy_model_input = noisy_model_input.to(model_input.dtype)

        # Predict the noise residual
        unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds}
        model_pred = self.unet(
            noisy_model_input,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]

        if self.noise_scheduler_prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler_prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        elif self.noise_scheduler_prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = model_input
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler_prediction_type}")

        if self.args.snr_gamma is None:
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = ops.stack([snr, self.args.snr_gamma * ops.ones_like(timesteps)], axis=1).min(axis=1)[0]
            if self.noise_scheduler_prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler_prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        loss = self.scaler.scale(loss)
        return loss, model_pred

    def update(self, loss, grads):
        if self.clip_grad:
            loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.clip_value)))
        else:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def construct(self, *inputs):
        (loss, model_pred), grads = self.forward_and_backward(*inputs)
        grads = self.grad_reducer(grads)
        loss = self.scaler.unscale(loss)
        grads = self.scaler.unscale(grads)

        if self.drop_overflow:
            status = all_finite(grads)
            if status:
                loss = self.update(loss, grads)
            loss = ops.depend(loss, self.scaler.adjust(status))
        else:
            loss = self.update(loss, grads)

        return loss


def validate(pipeline, args, trackers, logging_dir, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline.unet.set_train(False)

    # run inference
    generator = np.random.Generator(np.random.PCG64(seed=args.seed)) if args.seed else None
    pipeline_args = {"prompt": args.validation_prompt}
    images = [
        pipeline(**pipeline_args, generator=generator, num_inference_steps=25)[0][0]
        for _ in range(args.num_validation_images)
    ]

    if is_master(args):
        validation_logging_dir = os.path.join(logging_dir, "validation", f"epoch{epoch}")
        os.makedirs(validation_logging_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(validation_logging_dir, f"{idx:04d}.jpg"))

    for tracker_name, tracker in trackers.items():
        if tracker_name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.add_images("validation", np_images, epoch, dataformats="NHWC")

    logger.info("Validation done.")


if __name__ == "__main__":
    main()
