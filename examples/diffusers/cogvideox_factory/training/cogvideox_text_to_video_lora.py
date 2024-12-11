# Copyright 2024 The HuggingFace Team.
# All rights reserved.
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

import gc
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import auto_mixed_precision
from mindspore.dataset import GeneratorDataset
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell

from mindone.diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
    ConfigMixin,
    SchedulerMixin,
)
from mindone.diffusers._peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from mindone.diffusers.optimization import get_scheduler
from mindone.diffusers.training_utils import (
    AttrJitWrapper,
    cast_training_params,
    init_distributed_device,
    is_master,
    set_seed,
)
from mindone.diffusers.utils import convert_unet_state_dict_to_peft, export_to_video
from mindone.diffusers.utils.logging import get_logger
from mindone.trainers.zero import TrainOneStepWrapper, prepare_train_network
from mindone.transformers import T5EncoderModel

from args import get_args  # isort:skip
from dataset import VideoDatasetWithResizing, VideoDatasetWithResizeAndRectangleCrop  # isort:skip
from utils import get_optimizer  # isort:skip


logger = get_logger(__name__)


def log_validation(
    trackers,
    pipe: CogVideoXPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    # run inference
    generator = np.random.Generator(np.random.PCG64(seed=args.seed)) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np")[0][0]
        videos.append(video)

    for tracker_name, tracker_writer in trackers.items():
        phase_name = "test" if is_final_validation else "validation"
        if tracker_name == "tensorboard":
            if is_master(args):
                video_filenames = []
                for i, video in enumerate(videos):
                    prompt = (
                        pipeline_args["prompt"][:25]
                        .replace(" ", "_")
                        .replace(" ", "_")
                        .replace("'", "_")
                        .replace('"', "_")
                        .replace("/", "_")
                    )
                    filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                    export_to_video(video, filename, fps=8)
                    video_filenames.append(filename)

            tracker_writer.add_video(phase_name, np.stack(videos), epoch, fps=8, dataformats="NTCHW")
        if tracker_name == "wandb":
            logger.warning(f"image logging not implemented for {tracker_name}")

    return videos


class CollateFunction:
    def __init__(self, weight_dtype: ms.Type, load_tensors: bool, use_rope: bool) -> None:
        self.weight_dtype = weight_dtype
        self.load_tensors = load_tensors
        self.use_rope = use_rope

    def __call__(self, examples: Dict[str, Any]) -> Dict[str, ms.Tensor]:
        text_input_ids = [x["text_input_ids"] for x in examples]
        text_input_ids = np.stack(text_input_ids)

        videos = [x["video"] for x in examples]
        videos = np.stack(videos)

        if self.use_rope:
            rotary_positional_embeddings = [x["rotary_positional_embeddings"] for x in examples]
            rotary_positional_embeddings = np.stack(rotary_positional_embeddings)
            return videos, text_input_ids, rotary_positional_embeddings
        else:
            return videos, text_input_ids


def set_params_requires_grad(m: nn.Cell, requires_grad: bool):
    for p in m.get_parameters():
        p.requires_grad = requires_grad


def main(args):
    # Init context about MindSpore
    ms.set_context(
        mode=args.mindspore_mode,
        jit_config={"jit_level": args.jit_level},
    )

    # read attr distributed, writer attrs rank/local_rank/world_size:
    #   args.local_rank = mindspore.communication.get_local_rank()
    #   args.world_size = mindspore.communication.get_group_size()
    #   args.rank = mindspore.communication.get_rank()
    init_distributed_device(args)

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
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

    # Prepare models and scheduler
    # Loading order changed for MindSpore adaptation
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = ms.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else ms.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        mindspore_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )
    set_params_requires_grad(transformer, False)

    text_encoder, vae = None, None
    if not args.load_tensors:
        # Load Text-encoder & VAE only needed. Because currently the MindSpore memory pool does
        # not have the function of releasing memory fragments. Deleting them after loading still
        # causes memory fragments which might result in OOM on devices.
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )

        vae = AutoencoderKLCogVideoX.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )

        if args.enable_slicing:
            vae.enable_slicing()
        if args.enable_tiling:
            vae.enable_tiling()

        set_params_requires_grad(text_encoder, False)
        set_params_requires_grad(vae, False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = ms.float32
    if args.mixed_precision == "fp16":
        weight_dtype = ms.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = ms.bfloat16

    transformer.to(dtype=weight_dtype)
    if not args.load_tensors:
        text_encoder.to(dtype=weight_dtype)
        vae.to(dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, output_dir):
        if is_master(args):
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, CogVideoXTransformer3DModel):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

            CogVideoXPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        # @a-r-r-o-w: This is a bit of a hack but I don't know any other solution.
        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(transformer)):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    # Define models to load or save for load_model_hook() and save_model_hook()
    models = [transformer]

    # Dataset and DataLoader
    # Moving tokenize & prepare RoPE in dataset preprocess, which need some configs
    transformer_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    vae_config = AutoencoderKLCogVideoX.from_config(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )

    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae_config.block_out_channels) - 1)

    dataset_init_kwargs = {
        "data_root": args.data_root,
        "dataset_file": args.dataset_file,
        "caption_column": args.caption_column,
        "video_column": args.video_column,
        "max_num_frames": args.max_num_frames,
        "id_token": args.id_token,
        "height_buckets": args.height_buckets,
        "width_buckets": args.width_buckets,
        "frame_buckets": args.frame_buckets,
        "load_tensors": args.load_tensors,
        "random_flip": args.random_flip,
        "tokenizer": None if args.load_tensors else tokenizer,
        "max_sequence_length": None if args.load_tensors else transformer_config.max_text_seq_length,
        "use_rotary_positional_embeddings": transformer_config.use_rotary_positional_embeddings,
        "vae_scale_factor_spatial": VAE_SCALE_FACTOR_SPATIAL,
        "patch_size": transformer_config.patch_size,
        "patch_size_t": transformer_config.patch_size_t if hasattr(transformer_config, "patch_size_t") else None,
        "attention_head_dim": transformer_config.attention_head_dim,
        "base_height": transformer_config.sample_height * VAE_SCALE_FACTOR_SPATIAL,
        "base_width": transformer_config.sample_width * VAE_SCALE_FACTOR_SPATIAL,
    }
    if args.video_reshape_mode is None:
        train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
    else:
        train_dataset = VideoDatasetWithResizeAndRectangleCrop(
            video_reshape_mode=args.video_reshape_mode, **dataset_init_kwargs
        )

    collate_fn = CollateFunction(weight_dtype, args.load_tensors, transformer_config.use_rotary_positional_embeddings)

    train_dataloader = GeneratorDataset(
        train_dataset,
        column_names=["examples"],
        shard_id=args.rank,
        num_shards=args.world_size,
        num_parallel_workers=args.dataloader_num_workers,
    ).batch(
        batch_size=args.train_batch_size,
        per_batch_map=lambda examples, batch_info: collate_fn(examples),
        input_columns=["examples"],
        output_columns=["videos", "text_input_ids", "rotary_positional_embeddings"]
        if transformer_config.use_rotary_positional_embeddings
        else ["videos", "text_input_ids"],
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * args.world_size
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.world_size,
        num_training_steps=args.max_train_steps * args.world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Make sure the trainable params are in float32.
    # Do AMP wrapper manually
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=ms.float32)
        transformer = auto_mixed_precision(transformer, amp_level=args.amp_level, dtype=ms.float16)

    # Optimization parameters
    # Do Not define grouped learning rate here since it is not used but results in Call Depth Overflow failure
    # It might be a design flaw of MindSpore optimizer which should be fixed, but now we just avoid it.
    transformer_lora_parameters = transformer.trainable_params()
    num_trainable_parameters = sum(param.numel() for param in transformer_lora_parameters)

    optimizer = get_optimizer(
        params_to_optimize=transformer_lora_parameters,
        optimizer_name=args.optimizer,
        learning_rate=lr_scheduler,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # create train_step for training
    train_step = TrainStepForCogVideo(
        vae=vae,
        vae_config=vae_config,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
        weight_dtype=weight_dtype,
        args=args,
        use_rotary_positional_embeddings=transformer_config.use_rotary_positional_embeddings,
    ).set_train(True)

    loss_scaler = DynamicLossScaleUpdateCell(loss_scale_value=65536.0, scale_factor=2, scale_window=2000)

    if args.distributed and args.zero_stage != 0:
        train_step = prepare_train_network(
            train_step,
            optimizer=optimizer,
            scale_sense=loss_scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=True,
            clip_norm=args.max_grad_norm,
            zero_stage=args.zero_stage,
        )
    else:
        train_step = TrainOneStepWrapper(
            train_step,
            optimizer=optimizer,
            scale_sense=loss_scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=True,
            clip_norm=args.max_grad_norm,
        )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if is_master(args):
        with open(logging_dir / "hparams.yml", "w") as f:
            yaml.dump(vars(args), f, indent=4)
    trackers = dict()
    for tracker_name in args.report_to.split(","):
        if tracker_name == "tensorboard":
            trackers[tracker_name] = SummaryWriter(str(logging_dir), write_to_disk=is_master(args))
        else:
            logger.warning(f"Tracker {tracker_name} is not implemented, omitting...")

    tracker_name = args.tracker_name or "cogvideox-lora"

    # Train!
    total_batch_size = args.train_batch_size * args.world_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if is_master(args):
                logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            if is_master(args):
                logger.info(f"Resuming from checkpoint {path}")
            # TODO: load optimizer & grad scaler etc. like accelerator.load_state
            load_model_hook(models, os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not is_master(args),
    )

    train_dataloader_iter = train_dataloader.create_tuple_iterator(num_epochs=args.num_train_epochs - first_epoch)

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.set_train(True)

        for step, batch in enumerate(train_dataloader_iter):
            loss, _, _ = train_step(*batch)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if train_step.accum_steps == 1 or train_step.cur_accum_step.item() == 0:
                progress_bar.update(1)
                global_step += 1

            if is_master(args):
                if global_step % args.checkpointing_steps == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # TODO: save optimizer & grad scaler etc. like accelerator.save_state
                    os.makedirs(save_path, exist_ok=True)
                    save_model_hook(models, save_path)
                    logger.info(f"Saved state to {save_path}")

            last_lr = optimizer.get_lr()
            last_lr = last_lr[0] if isinstance(last_lr, tuple) else last_lr  # grouped lr scenario
            logs = {"loss": loss.item(), "lr": last_lr.item()}
            progress_bar.set_postfix(**logs)
            for tracker_name, tracker in trackers.items():
                if tracker_name == "tensorboard":
                    tracker.add_scalars("train", logs, global_step)

            if global_step >= args.max_train_steps:
                break

        if is_master(args):
            if args.validation_prompt is not None and (epoch + 1) % args.validation_epochs == 0:
                pipe = CogVideoXPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=transformer,
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    mindspore_dtype=weight_dtype,
                )

                if args.enable_slicing:
                    pipe.vae.enable_slicing()
                if args.enable_tiling:
                    pipe.vae.enable_tiling()

                validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                for validation_prompt in validation_prompts:
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                        "max_sequence_length": transformer_config.max_text_seq_length,
                    }

                    log_validation(
                        trackers=trackers,
                        pipe=pipe,
                        args=args,
                        pipeline_args=pipeline_args,
                        epoch=epoch,
                        is_final_validation=False,
                    )

                del pipe
                gc.collect()
                ms.hal.empty_cache()

    if is_master(args):
        dtype = (
            ms.float16
            if args.mixed_precision == "fp16"
            else ms.bfloat16
            if args.mixed_precision == "bf16"
            else ms.float32
        )

        if hasattr(transformer, "_backbone"):
            transformer = transformer._backbone

        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        CogVideoXPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # Cleanup trained models to save memory
        del transformer
        gc.collect()
        ms.hal.empty_cache()

        # Final test inference
        pipe = CogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            mindspore_dtype=weight_dtype,
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        # Load LoRA weights
        lora_scaling = args.lora_alpha / args.lora_rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        # Run inference
        validation_outputs = []
        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
            for validation_prompt in validation_prompts:
                pipeline_args = {
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }

                video = log_validation(
                    trackers=trackers,
                    pipe=pipe,
                    args=args,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    is_final_validation=True,
                )
                validation_outputs.extend(video)


class TrainStepForCogVideo(nn.Cell):
    def __init__(
        self,
        vae: Optional[nn.Cell],
        vae_config: Optional[ConfigMixin],
        text_encoder: Optional[nn.Cell],
        transformer: nn.Cell,
        scheduler: SchedulerMixin,
        weight_dtype: ms.Type,
        args: AttrJitWrapper,
        use_rotary_positional_embeddings: bool,
    ):
        super().__init__()

        vae_config = vae_config or vae.config

        self.weight_dtype = weight_dtype
        self.vae = vae
        self.vae_dtype = None if vae is None else vae.dtype
        self.vae_scaling_factor = vae_config.scaling_factor
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.scheduler = scheduler
        self.scheduler_num_train_timesteps = scheduler.config.num_train_timesteps

        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.args = AttrJitWrapper(**vars(args))

    def compute_prompt_embeddings(
        self,
        text_input_ids,
        num_videos_per_prompt: int = 1,
        dtype: ms.Type = None,
    ):
        batch_size = text_input_ids.shape[0]
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile((1, num_videos_per_prompt, 1))
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def diagonal_gaussian_distribution_sample(self, latent_dist: ms.Tensor) -> ms.Tensor:
        mean, logvar = ops.chunk(latent_dist, 2, axis=1)
        logvar = ops.clamp(logvar, -30.0, 20.0)
        std = ops.exp(0.5 * logvar)

        sample = ops.randn_like(mean, dtype=mean.dtype)
        x = mean + std * sample

        return x

    def construct(self, videos, text_input_ids_or_prompt_embeds, image_rotary_emb=None):
        # Encode videos
        if not self.args.load_tensors:
            videos = videos.permute(0, 2, 1, 3, 4).to(self.vae_dtype)  # [B, C, F, H, W]
            videos = self.vae.encode(videos)[0]

        videos = self.diagonal_gaussian_distribution_sample(videos) * self.vae_scaling_factor
        videos = videos.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        videos = videos.to(dtype=self.weight_dtype)
        model_input = videos

        # Encode prompts
        if not args.load_tensors:
            prompt_embeds = self.compute_prompt_embeddings(
                text_input_ids_or_prompt_embeds,
                dtype=self.weight_dtype,
            )
        else:
            prompt_embeds = text_input_ids_or_prompt_embeds.to(dtype=self.weight_dtype)

        # Sample noise that will be added to the latents
        noise = ops.randn_like(model_input, dtype=model_input.dtype)
        batch_size, num_frames, num_channels, height, width = model_input.shape

        # Sample a random timestep for each image
        timesteps = ops.randint(
            0,
            self.scheduler_num_train_timesteps,
            (batch_size,),
            dtype=ms.int64,
        )

        # Rotary embeds is Prepared in dataset.
        if self.use_rotary_positional_embeddings:
            image_rotary_emb = image_rotary_emb[0].to(self.weight_dtype)  # [2, S, D]

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.scheduler.add_noise(model_input, noise, timesteps)

        # Predict the noise residual
        model_output = self.transformer(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]

        model_pred = self.scheduler.get_velocity(model_output, noisy_model_input, timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=ms.float32)
        weights = 1 / (1 - alphas_cumprod[timesteps])
        while len(weights.shape) < len(model_pred.shape):
            weights = weights.unsqueeze(-1)

        target = model_input

        loss = ops.mean(
            (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
            axis=1,
        )
        loss = loss.mean()

        return loss


if __name__ == "__main__":
    args = get_args()
    main(args)