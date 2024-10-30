import argparse
import ast
import os
import sys
import time
from functools import partial

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from gm.data.loader import create_loader
from gm.helpers import (
    EMA,
    SD_XL_BASE_RATIOS,
    VERSION2SPECS,
    create_model,
    get_grad_reducer,
    get_learning_rate,
    get_loss_scaler,
    get_optimizer,
    load_checkpoint,
    save_checkpoint,
    set_default,
)
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Tensor, nn

from townwish_mindone_testing.utils.amp import auto_mixed_precision


def count_params(model, verbose=False):
    total_params = sum([param.size for param in model.get_parameters()])
    trainable_params = sum([param.size for param in model.get_parameters() if param.requires_grad])

    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params, trainable_params


def get_parser_train():
    parser = argparse.ArgumentParser(description="train with sd-xl")
    parser.add_argument("--version", type=str, default="SDXL-base-1.0", choices=["SDXL-base-1.0", "SDXL-refiner-1.0"])
    parser.add_argument("--config", type=str, default="configs/training/sd_xl_base_finetune_controlnet_910b.yaml")
    parser.add_argument(
        "--task",
        type=str,
        default="txt2img",
        choices=[
            "txt2img",
        ],
    )

    parser.add_argument(
        "--group_lr_scaler", default=10.0, type=float, help="scaler for lr of a particular group of params"
    )
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=False, type=ast.literal_eval, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--use_ema", action="store_true", help="whether use ema")
    parser.add_argument("--weight", type=str, default="checkpoints/sd_xl_base_1.0_ms_controlnet_init.ckpt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sd_xl_base_ratios", type=str, default="1.0")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="./runs")
    parser.add_argument("--save_path_with_time", type=ast.literal_eval, default=True)
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    parser.add_argument("--save_ckpt_interval", type=int, default=10000, help="save ckpt interval")
    parser.add_argument(
        "--max_num_ckpt",
        type=int,
        default=None,
        help="Max number of ckpts saved. If exceeds, delete the oldest one. Set None: keep all ckpts.",
    )
    parser.add_argument("--optimizer_weight", type=str, default=None, help="load optimizer weight")
    parser.add_argument("--save_optimizer", type=ast.literal_eval, default=False, help="enable save optimizer")
    parser.add_argument("--data_sink", type=ast.literal_eval, default=False)
    parser.add_argument("--sink_size", type=int, default=1000)
    parser.add_argument(
        "--dataset_load_tokenizer", type=ast.literal_eval, default=True, help="create dataset with tokenizer"
    )
    parser.add_argument(
        "--total_step",
        type=int,
        default=None,
        help="The number of training steps. If not provided, will use the `total_step` in training yaml file.",
    )
    parser.add_argument(
        "--per_batch_size",
        type=int,
        default=None,
        help="The batch size for training. If not provided, will use `per_batch_size` in training yaml file.",
    )

    # args for infer
    parser.add_argument("--infer_during_train", type=ast.literal_eval, default=False)
    parser.add_argument("--infer_interval", type=int, default=1, help="log interval")

    # args for env
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument("--ms_amp_level", type=str, default="O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument("--param_fp16", type=ast.literal_eval, default=False)
    parser.add_argument("--overflow_still_update", type=ast.literal_eval, default=True)
    parser.add_argument("--max_device_memory", type=str, default=None)
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False)

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument(
        "--ckpt_url", type=str, default="", help="ModelArts: obs path to pretrain model checkpoint file"
    )
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to output folder")
    parser.add_argument(
        "--multi_data_url", type=str, default="", help="ModelArts: list of obs paths to multi-dataset folders"
    )
    parser.add_argument(
        "--pretrain_url", type=str, default="", help="ModelArts: list of obs paths to multi-pretrain model files"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/cache/pretrain_ckpt/",
        help="ModelArts: local device path to checkpoint folder",
    )
    return parser


def train(args):
    # 1. Init Env
    args = set_default(args)

    # 2. Create LDM Engine
    config = OmegaConf.load(args.config)
    model, _ = create_model(
        config,
        checkpoints=args.weight,
        freeze=False,
        load_filter=False,
        param_fp16=args.param_fp16,
        amp_level=args.ms_amp_level,
    )
    assert isinstance(model.model, nn.Cell)

    if config.model.params.network_config.params.sd_locked:
        model.model.set_train(False)
        model.model.diffusion_model.controlnet.set_train(True)
    else:
        model.model.set_train(True)

    # 3. Create dataloader
    assert "data" in config
    if args.total_step is not None:
        config.data["total_step"] = args.total_step
    if args.per_batch_size is not None:
        config.data["per_batch_size"] = args.per_batch_size
    dataloader = create_loader(
        data_path=args.data_path,
        rank=args.rank,
        rank_size=args.rank_size,
        tokenizer=model.conditioner.tokenize if args.dataset_load_tokenizer else None,
        token_nums=len(model.conditioner.embedders) if args.dataset_load_tokenizer else None,
        **config.data,
    )

    # 4. Create train step func
    assert "optim" in config
    lr = get_learning_rate(config.optim, config.data.total_step)
    scaler = get_loss_scaler(ms_loss_scaler="static", scale_value=1024)
    optimizer = get_optimizer(
        config.optim,
        lr,
        params=model.model.trainable_params() + model.conditioner.trainable_params(),
        group_lr_scaler=args.group_lr_scaler,
    )
    reducer = get_grad_reducer(is_parallel=args.is_parallel, parameters=optimizer.parameters)
    if args.optimizer_weight:
        print(f"Loading optimizer from {args.optimizer_weight}")
        load_checkpoint(optimizer, args.optimizer_weight, remove_prefix="ldm_with_loss_grad.optimizer.")

    if args.use_ema:
        ema = EMA(model, ema_decay=0.9999)
    else:
        ema = None

    if args.ms_mode == 1:
        # Pynative Mode
        train_step_fn = partial(
            model.train_step_pynative,
            grad_func=model.get_grad_func(
                optimizer, reducer, scaler, jit=True, overflow_still_update=args.overflow_still_update
            ),
        )
        model = auto_mixed_precision(model, args.ms_amp_level)
        jit_config = None
    elif args.ms_mode == 0:
        # Graph Mode
        from gm.models.trainer_factory import TrainOneStepCellControlNet

        train_step_fn = TrainOneStepCellControlNet(
            model,
            optimizer,
            reducer,
            scaler,
            overflow_still_update=args.overflow_still_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm,
            ema=ema,
        )
        train_step_fn = auto_mixed_precision(train_step_fn, amp_level=args.ms_amp_level)
        if model.disable_first_stage_amp:
            train_step_fn.first_stage_model.to_float(ms.float32)
        jit_config = ms.JitConfig()
    else:
        raise ValueError("args.ms_mode value must in [0, 1]")

    num_params, num_trainable_params = count_params(model)
    print(f"Total number of parameters: {num_params:,}")
    print(f"Total number of trainable parameters: {num_trainable_params:,}")

    # 5. Start Training
    if args.max_num_ckpt is not None and args.max_num_ckpt <= 0:
        raise ValueError("args.max_num_ckpt must be None or a positive integer!")
    if args.task == "txt2img":
        train_fn = train_txt2img if not args.data_sink else train_txt2img_datasink
        train_fn(
            args, train_step_fn, dataloader=dataloader, optimizer=optimizer, model=model, jit_config=jit_config, ema=ema
        )
    elif args.task == "img2img":
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown task {args.task}")


def train_txt2img(
    args, train_step_fn, dataloader, optimizer=None, model=None, ema=None, **kwargs
):  # for print  # for infer/ckpt
    dtype = ms.float32 if args.ms_amp_level not in ("O2", "O3") else ms.float16
    total_step = dataloader.get_dataset_size()
    loader = dataloader.create_tuple_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()

    ckpt_queue = []
    for i, data in enumerate(loader):
        if not args.dataset_load_tokenizer:
            # Get data, image and tokens, to tensor
            data = data[0]
            data = {k: (Tensor(v, dtype) if k != "txt" else v.tolist()) for k, v in data.items()}

            image = data[model.input_key]
            tokens, _ = model.conditioner.tokenize(data)
            tokens = [Tensor(t) for t in tokens]
        else:
            image, tokens = data[0], data[1:]
            image, tokens = Tensor(image), [Tensor(t) for t in tokens]

        # Train a step
        if i == 0:
            print(
                "The first step will be compiled for the graph, which may take a long time; "
                "You can come back later :)",
                flush=True,
            )
        loss, overflow = train_step_fn(image, *tokens)

        # Print meg
        if (i + 1) % args.log_interval == 0 and args.rank % 8 == 0:
            print(
                f"Step {i + 1}/{total_step}, size: {image.shape[:]}, loss: {loss.asnumpy():.6f}"
                f", time cost: {(time.time()-s_time) * 1000 / args.log_interval:.2f} ms",
                flush=True,
            )
            s_time = time.time()

        # Save checkpoint
        if (i + 1) % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{(i + 1)}.ckpt")
            if isinstance(model.model, nn.Cell):
                model.model.set_train(False)  # only unet
                save_checkpoint(
                    model if not ema else ema,
                    save_ckpt_dir,
                    ckpt_queue,
                    args.max_num_ckpt,
                    only_save_lora=False
                    if not hasattr(model.model.diffusion_model, "only_save_lora")
                    else model.model.diffusion_model.only_save_lora,
                )
                model.model.set_train(True)  # only unet
            else:
                model.save_checkpoint(save_ckpt_dir)
            ckpt_queue.append(save_ckpt_dir)

            if args.save_optimizer:
                save_optimizer_dir = os.path.join(args.save_path, "optimizer.ckpt")
                ms.save_checkpoint(optimizer, save_optimizer_dir)
                print(f"save optimizer weight to {save_optimizer_dir}")

        # Infer during train
        if (i + 1) % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {i + 1}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{i+1}_rank_{args.rank}"),
            )
            print(f"Step {i + 1}/{total_step}, infer done.", flush=True)


def train_txt2img_datasink(
    args, train_step_fn, dataloader, optimizer=None, model=None, jit_config=None, ema=None, **kwargs
):  # for print  # for infer/ckpt
    total_step = dataloader.get_dataset_size()
    epochs = total_step // args.sink_size
    assert args.dataset_load_tokenizer

    train_fn_sink = ms.data_sink(fn=train_step_fn, dataset=dataloader, sink_size=args.sink_size, jit_config=jit_config)

    ckpt_queue = []
    for epoch in range(epochs):
        cur_step = args.sink_size * (epoch + 1)

        if epoch == 0:
            print(
                "The first epoch will be compiled for the graph, which may take a long time; "
                "You can come back later :)",
                flush=True,
            )

        s_time = time.time()
        loss, _ = train_fn_sink()
        e_time = time.time()

        # Print meg
        if cur_step % args.log_interval == 0 and args.rank % 8 == 0:
            if optimizer.dynamic_lr:
                cur_lr = optimizer.learning_rate(Tensor((cur_step - 1), ms.int32)).asnumpy().item()
            else:
                cur_lr = optimizer.learning_rate.asnumpy().item()
            print(
                f"Step {cur_step}/{total_step}, lr: {cur_lr}, loss: {loss.asnumpy():.6f}"
                f", per step time: {(e_time - s_time) * 1000 / args.sink_size:.2f} ms",
                flush=True,
            )

        # Save checkpoint
        if cur_step % args.save_ckpt_interval == 0 and args.rank % 8 == 0:
            save_ckpt_dir = os.path.join(args.save_path, "weights", args.version + f"_{cur_step}.ckpt")
            if isinstance(model.model, nn.Cell):
                model.model.set_train(False)  # only unet
                save_checkpoint(
                    model if not ema else ema,
                    save_ckpt_dir,
                    ckpt_queue,
                    args.max_num_ckpt,
                    only_save_lora=False
                    if not hasattr(model.model.diffusion_model, "only_save_lora")
                    else model.model.diffusion_model.only_save_lora,
                )
                model.model.set_train(True)  # only unet
            else:
                model.save_checkpoint(save_ckpt_dir)
            ckpt_queue.append(save_ckpt_dir)

        # Infer during train
        if cur_step % args.infer_interval == 0 and args.infer_during_train:
            print(f"Step {cur_step}/{total_step}, infer starting...")
            infer_during_train(
                model=model,
                prompt="Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                save_path=os.path.join(args.save_path, "txt2img/", f"step_{cur_step}_rank_{args.rank}"),
            )
            print(f"Step {cur_step}/{total_step}, infer done.", flush=True)


def infer_during_train(model, prompt, save_path):
    from gm.helpers import init_sampling, perform_save_locally

    version_dict = VERSION2SPECS.get(args.version)
    W, H = SD_XL_BASE_RATIOS[args.sd_xl_base_ratios]
    C = version_dict["C"]
    F = version_dict["f"]
    is_legacy = version_dict["is_legacy"]

    value_dict = {
        "prompt": prompt,
        "negative_prompt": "",
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
        "crop_coords_top": 0,
        "crop_coords_left": 0,
        "aesthetic_score": 6.0,
        "negative_aesthetic_score": 2.5,
    }
    sampler, num_rows, num_cols = init_sampling(steps=40, num_cols=1)

    out = model.do_sample(
        sampler,
        value_dict,
        num_rows * num_cols,
        H,
        W,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=False,
        filter=None,
        amp_level="O2",
    )
    perform_save_locally(save_path, out)


if __name__ == "__main__":
    parser = get_parser_train()
    args, _ = parser.parse_known_args()
    train(args)
