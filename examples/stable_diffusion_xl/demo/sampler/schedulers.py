from typing import Dict, Union

import tqdm
import numpy as np
import mindspore as ms
from mindspore import nn, ops

from gm.util import append_dims
from gm.modules.diffusionmodules.sampling_utils import get_ancestral_step, to_d


class DiscreteScheduler:
    def __init__(self):
        pass

    def init_sigmas(self):
        raise NotImplementedError()

    def init_sampling_sigmas(self):
        raise NotImplementedError()

    @property
    def has_sigmas(self):
        return hasattr(self, "sigmas")

    @property
    def has_sampling_sigmas(self):
        return hasattr(self, "sampling_sigmas")

    def sigma_to_idx(self, sigma):
        if not self.has_sigmas:
            self.init_sigmas()

        dists = sigma - self.sigmas[:, None]
        return ops.abs(dists).argmin(axis=0).reshape(sigma.shape)

    def idx_to_sigma(self, idx):
        if not self.has_sigmas:
            self.init_sigmas()

        return self.sigmas[idx]

    def quantize_c_noise(self, c_noise):
        return self.sigma_to_idx(c_noise)

    def prepare_inputs(self, noise):
        if not self.has_sampling_sigmas:
            self.init_sampling_sigmas()

        return noise * ops.sqrt(1.0 + self.sampling_sigmas[0] ** 2)

    def get_scaling_factors(self, sigma):
        c_skip = ops.ones_like(sigma)
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        c_noise = self.quantize_c_noise(sigma.copy())
        return c_skip, c_out, c_in, c_noise


class EulerDiscreteScheduler(nn.Cell, DiscreteScheduler):
    def __init__(
            self,
            unet,
            num_sampling_steps=20,
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.0120,
            interpolation_type="linear",
            timestep_spacing="leading",
            steps_offset=1,
            append_zero=True,
            cfg=5.0,
        ):
        super().__init__()

        self.unet = unet
        self.cfg = cfg

        self.num_sampling_steps = num_sampling_steps
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.init_sigmas()

        self.num_sampling_steps = num_sampling_steps
        self.interpolation_type = interpolation_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.append_zero = append_zero
        self.init_sampling_sigmas()


    def init_sigmas(self):
        self.betas = (np.linspace(self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps).astype(np.float32)) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5
        self.sigmas = ms.Tensor(sigmas, ms.float32)

    def init_sampling_sigmas(self):
        if not self.has_sigmas:
            self.init_sigmas()

        sigmas = self.sigmas.asnumpy()

        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(0, self.num_train_timesteps - 1, self.num_sampling_steps, dtype=np.float32)[::-1]
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_sampling_steps
            timesteps = (np.arange(0, self.num_sampling_steps) * step_ratio).round()[::-1].astype(np.float32)
            timesteps += self.steps_offset
        else:
            raise NotImplementedError(f"Unsupported type `{self.timestep_spacing}`")

        if self.num_sampling_steps < self.num_train_timesteps:
            if self.interpolation_type == "linear":
                sampling_sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas) 
            elif self.interpolation_type == "log_linear":
                sampling_sigmas = np.linspace(np.log(sigmas[-1]), np.log(sigmas[0]), self.num_sampling_steps + 1)
                sampling_sigmas = np.exp(sigmas)
            else:
                raise NotImplementedError(f"Unsupported type `{self.interpolation_type}`")
        elif self.num_sampling_steps == self.num_train_timesteps:
            sampling_sigmas = np.flip(sigmas, (0,))
        else:
            raise ValueError
        
        if self.append_zero:
            sampling_sigmas = np.concatenate([sampling_sigmas, np.zeros([1], dtype=sampling_sigmas.dtype)])

        self.sampling_sigmas = ms.tensor(sampling_sigmas, ms.float32)

    def denoise(self, latents, sigma, vector, crossattn, concat):
        latents = latents.tile((2, 1, 1, 1))
        c_skip, c_out, c_in, c_noise = self.get_scaling_factors(sigma)
        c_noise = c_noise.broadcast_to((latents.shape[0],))
        model_pred = self.unet(
            latents * c_in,
            c_noise,
            concat=concat,
            context=crossattn,
            y=vector,
        )
        model_output = latents * c_skip + model_pred * c_out
        uc_output, c_output = model_output.chunk(2)
        denoised = uc_output + self.cfg * (c_output - uc_output)
        return denoised

    def construct(self, latents, sigma, next_sigma, vector, crossattn, concat):
        denoised = self.denoise(latents, sigma, vector, crossattn, concat)
        d = to_d(latents, sigma, denoised)
        dt = next_sigma - sigma
        latents = latents + d * dt
        return latents

    def sample(self, noise, vector, crossattn, concat):
        x = self.prepare_inputs(noise)

        sampling_generator = tqdm.tqdm(
            range(self.num_sampling_steps),
            desc=f"Sampling with {self.__class__.__name__} for {self.num_sampling_steps} steps",
        )

        for idx in sampling_generator:
            x = self(x, self.sampling_sigmas[idx], self.sampling_sigmas[idx + 1], vector, crossattn, concat)
        
        return x


class EulerDiscreteAncestralScheduler(EulerDiscreteScheduler):
    def __init__(self, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_noise = s_noise

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)
        return x + dt * d

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = ops.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + ops.randn_like(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def construct(self, latents, sigma, next_sigma, vector, crossattn, concat):
        denoised = self.denoise(latents, sigma, vector, crossattn, concat)

        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma)
        latents = self.ancestral_euler_step(latents, denoised, sigma, sigma_down)
        latents = self.ancestral_step(latents, sigma, next_sigma, sigma_up)

        return latents


if __name__ == "__main__":
    ms.set_context(mode=0)

    class TestUnet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.id = nn.Identity()

        def construct(self, x, *args, **kwargs):
            return self.id(x)

    unet = TestUnet()
    scheduler = EulerDiscreteScheduler(unet, num_sampling_steps=40)

    latents = ops.randn(1, 4, 128, 128)
    vector = ops.randn(2, 2816)
    crossattn = ops.randn(2, 77, 2048)
    concat = None

    out = scheduler.sample(latents, vector, crossattn, concat)

    print("done")