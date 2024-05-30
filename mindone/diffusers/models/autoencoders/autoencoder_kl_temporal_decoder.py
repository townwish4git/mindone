# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import Dict, Tuple, Union

import mindspore as ms
from mindspore import nn, ops

from ...configuration_utils import ConfigMixin, register_to_config
from ..activations import SiLU
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from ..modeling_outputs import AutoencoderKLOutput
from ..modeling_utils import ModelMixin
from ..normalization import GroupNorm
from ..unets.unet_3d_blocks import MidBlockTemporalDecoder, UpBlockTemporalDecoder
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder


class TemporalDecoder(nn.Cell):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True
        )
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=self.layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
        )

        # up
        self.up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            up_block = UpBlockTemporalDecoder(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = nn.CellList(self.up_blocks)

        self.conv_norm_out = GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)

        self.conv_act = SiLU()
        self.conv_out = nn.Conv2d(
            in_channels=block_out_channels[0],
            out_channels=out_channels,
            kernel_size=3,
            pad_mode="pad",
            padding=1,
            has_bias=True,
        )

        conv_out_kernel_size = (3, 1, 1)
        # padding = [int(k // 2) for k in conv_out_kernel_size]
        padding = (1, 1, 0, 0, 0, 0)
        self.time_conv_out = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=conv_out_kernel_size,
            pad_mode="pad",
            padding=padding,
            has_bias=True,
        )

        self.gradient_checkpointing = False

    def construct(
        self,
        sample: ms.Tensor,
        image_only_indicator: ms.Tensor,
        num_frames: int = 1,
    ) -> ms.Tensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample, image_only_indicator=image_only_indicator)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, image_only_indicator=image_only_indicator)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        batch_frames, channels, height, width = sample.shape
        batch_size = batch_frames // num_frames
        sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        sample = self.time_conv_out(sample)

        sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)

        return sample


class AutoencoderKLTemporalDecoder(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding images into latents and decoding latent representations into images.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            Tuple of downsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(64,)`):
            Tuple of block output channels.
        layers_per_block: (`int`, *optional*, defaults to 1): Number of layers per block.
        latent_channels (`int`, *optional*, defaults to 4): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): Sample input size.
        scaling_factor (`float`, *optional*, defaults to 0.18215):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
        force_upcast (`bool`, *optional*, default to `True`):
            If enabled it will force the VAE to run in float32 for high image resolution pipelines, such as SD-XL. VAE
            can be fine-tuned / trained to a lower range without loosing too much precision in which case
            `force_upcast` can be set to `False` - see: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        latent_channels: int = 4,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = TemporalDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1, has_bias=True)
        self.diag_gauss_dist = DiagonalGaussianDistribution()

        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, TemporalDecoder)):
            module.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:  # type: ignore
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):  # type: ignore
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.name_cells():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):  # type: ignore
        r"""
        Sets the attention processor to use to compute attention.
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.name_cells():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.name_cells():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def encode(
        self, x: ms.Tensor, return_dict: bool = False
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`ms.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)

        if not return_dict:
            return (moments,)

        return AutoencoderKLOutput(latent=moments)

    def decode(
        self,
        z: ms.Tensor,
        num_frames: int,
        return_dict: bool = False,
    ) -> Union[DecoderOutput, ms.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`ms.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        batch_size = z.shape[0] // num_frames
        image_only_indicator = ops.zeros((batch_size, num_frames), dtype=z.dtype)
        decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def construct(
        self,
        sample: ms.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = False,
        num_frames: int = 1,
    ) -> Union[DecoderOutput, ms.Tensor]:
        r"""
        Args:
            sample (`ms.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        latent = self.encode(x)[0]
        if sample_posterior:
            z = self.diag_gauss_dist.sample(latent)
        else:
            z = self.diag_gauss_dist.mode(latent)

        dec = self.decode(z, num_frames=num_frames)[0]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
