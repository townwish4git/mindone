import logging
import time

import numpy as np
import mindspore as ms
from mindspore import nn, ops

from gm.helpers import get_batch, get_unique_embedder_keys_from_conditioner


logger = logging.getLogger(__name__)


class Sampler(nn.Cell):
    def __init__(
            self,
            conditioner,
            vae,
            scale_factor,
            scheduler,
            denoiser,
            num_steps=40,
            cfg=5.0,
    ):
        super().__init__()

        self.conditioner = conditioner
        self.vae = vae
        self.scheduler = scheduler
        self.denoiser = denoiser

        self.scale_factor = scale_factor
        self.num_steps = num_steps
        self.cfg = cfg

        self.vae.to_float(ms.float32)
    
    def tokenize(self, batch, **kwargs):
        tokens, lengths = self.conditioner.tokenize(batch, **kwargs)
        tokens = ops.concat([ms.Tensor(t) for t in tokens])
        return tokens
    
    def embedding(self, uc_tokens, c_tokens, force_zero_embeddings=None):
        uc_tokens = ops.chunk(uc_tokens, 5)
        c_tokens = ops.chunk(c_tokens, 5)

        uc_vector, uc_crossattn, uc_concat = self.conditioner(*uc_tokens)
        c_vector, c_crossattn, c_concat = self.conditioner(*c_tokens)

        if force_zero_embeddings is not None:
            logger.warning(
                "\'force_zero_embeddings\' is calling, and attention that:"
                "so far, parameter \'force_zero_embeddings\' only supports \'txt\'."
            )
            uc_vector[:, :1280] *= 0.0
            uc_crossattn *= 0.0
        
        vector = ops.concat([uc_vector, c_vector])
        crossattn = ops.concat([uc_crossattn, c_crossattn])
        
        return vector, crossattn

    @ms.jit
    def latents_decode(self, latents):
        latents = latents.astype(ms.float32)

        latents = latents / self.scale_factor
        imgs = self.vae.decode(latents)
        return imgs

    def do_sample(
            self,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            force_uc_zero_embeddings=None,
            amp_level="O0",
            init_latent_path=None,  # '/path/to/sdxl_init_latent.npy'
            lpw=False,
            max_embeddings_multiples=4,
            **kwargs,
    ):
        """From gm.models.diffusion.DiffusionEngine"""
        logger.info("Sampling starts.")

        dtype = ms.float32 if amp_level not in ("O2", "O3") else ms.float16

        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []

        # Aggregate sampling information
        num_samples = [num_samples]
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.conditioner), value_dict, num_samples, dtype=dtype
        )

        # Generate random noise
        shape = (np.prod(num_samples), C, H // F, W // F)
        if init_latent_path is not None:
            logger.info(f"Loading latent noise from {init_latent_path}")
            randn = ms.Tensor(np.load(init_latent_path), ms.float32)
        else:
            randn = ms.Tensor(np.random.randn(*shape), ms.float32)

        # Get text embeddings
        uc_tokens = self.tokenize(batch_uc, lpw=lpw, max_embeddings_multiples=max_embeddings_multiples)
        c_tokens = self.tokenize(batch, lpw=lpw, max_embeddings_multiples=max_embeddings_multiples)
        vector, crossattn = self.embedding(uc_tokens, c_tokens, force_zero_embeddings=force_uc_zero_embeddings)

        # Sampling
        logger.info("Sample Starting...")
        samples_x = self.sample(randn, vector, crossattn, None, force_zero_embeddings=force_uc_zero_embeddings)
        samples_x = samples_x.asnumpy()
        samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)
        logger.info("Sample Done.")

        return samples

    def sample(self, noise, vector, crossattn, concat, force_zero_embeddings=None):
        latents = self.scheduler.sample(noise, vector, crossattn, concat)
        imgs = self.latents_decode(latents)
        return imgs
