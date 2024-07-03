__version__ = "0.27.1"

from typing import TYPE_CHECKING

from .utils import _LazyModule

# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "models": [
        "AsymmetricAutoencoderKL",
        "AutoencoderKL",
        "AutoencoderKLTemporalDecoder",
        "AutoencoderTiny",
        "ConsistencyDecoderVAE",
        "ControlNetModel",
        "ControlNetXSAdapter",
        "DiTTransformer2DModel",
        "HunyuanDiT2DModel",
        "I2VGenXLUNet",
        "Kandinsky3UNet",
        "ModelMixin",
        "MotionAdapter",
        "MultiAdapter",
        "PixArtTransformer2DModel",
        "PriorTransformer",
        "SD3ControlNetModel",
        "SD3MultiControlNetModel",
        "SD3Transformer2DModel",
        "T2IAdapter",
        "T5FilmDecoder",
        "Transformer2DModel",
        "StableCascadeUNet",
        "UNet1DModel",
        "UNet2DConditionModel",
        "UNet2DModel",
        "UNet3DConditionModel",
        "UNetControlNetXSModel",
        "UNetMotionModel",
        "UNetSpatioTemporalConditionModel",
        "UVit2DModel",
        "VQModel",
    ],
    "optimization": [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ],
    "pipelines": [
        "BlipDiffusionPipeline",
        "ConsistencyModelPipeline",
        "DanceDiffusionPipeline",
        "DDIMPipeline",
        "DDPMPipeline",
        "DiffusionPipeline",
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
        "KandinskyV22CombinedPipeline",
        "KandinskyV22ControlnetImg2ImgPipeline",
        "KandinskyV22ControlnetPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22Img2ImgPipeline",
        "KandinskyV22InpaintCombinedPipeline",
        "KandinskyV22InpaintPipeline",
        "KandinskyV22Pipeline",
        "KandinskyV22PriorEmb2EmbPipeline",
        "KandinskyV22PriorPipeline",
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3Pipeline",
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
        "StableCascadeCombinedPipeline",
        "StableCascadeDecoderPipeline",
        "StableCascadePriorPipeline",
        "StableDiffusion3Pipeline",
        "StableDiffusionDepth2ImgPipeline",
        "StableDiffusionImageVariationPipeline",
        "StableDiffusionImg2ImgPipeline",
        "StableDiffusionInpaintPipeline",
        "StableDiffusionInstructPix2PixPipeline",
        "StableDiffusionLatentUpscalePipeline",
        "StableDiffusionPipeline",
        "StableDiffusionUpscalePipeline",
        "StableDiffusionXLImg2ImgPipeline",
        "StableDiffusionXLInpaintPipeline",
        "StableDiffusionXLInstructPix2PixPipeline",
        "StableDiffusionXLPipeline",
        "StableVideoDiffusionPipeline",
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ],
    "schedulers": [
        "ConsistencyDecoderScheduler",
        "CMStochasticIterativeScheduler",
        "DDIMScheduler",
        "DDIMInverseScheduler",
        "DDIMParallelScheduler",
        "DDPMScheduler",
        "DDPMParallelScheduler",
        "DDPMWuerstchenScheduler",
        "DEISMultistepScheduler",
        "DPMSolverMultistepScheduler",
        "DPMSolverMultistepInverseScheduler",
        "DPMSolverSinglestepScheduler",
        "EDMEulerScheduler",
        "EulerAncestralDiscreteScheduler",
        "EulerDiscreteScheduler",
        "FlowMatchEulerDiscreteScheduler",
        "HeunDiscreteScheduler",
        "IPNDMScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "KDPM2DiscreteScheduler",
        "LCMScheduler",
        "LMSDiscreteScheduler",
        "PNDMScheduler",
        "RePaintScheduler",
        "SASolverScheduler",
        "UnCLIPScheduler",
        "UniPCMultistepScheduler",
        "VQDiffusionScheduler",
        "SchedulerMixin",
    ],
    "utils": [
        "logging",
    ],
}

if TYPE_CHECKING:
    from .configuration_utils import ConfigMixin
    from .models import (
        AsymmetricAutoencoderKL,
        AutoencoderKL,
        AutoencoderKLTemporalDecoder,
        AutoencoderTiny,
        ConsistencyDecoderVAE,
        ControlNetModel,
        ControlNetXSAdapter,
        DiTTransformer2DModel,
        HunyuanDiT2DModel,
        I2VGenXLUNet,
        Kandinsky3UNet,
        ModelMixin,
        MotionAdapter,
        MultiAdapter,
        PixArtTransformer2DModel,
        PriorTransformer,
        SD3ControlNetModel,
        SD3MultiControlNetModel,
        SD3Transformer2DModel,
        StableCascadeUNet,
        T2IAdapter,
        T5FilmDecoder,
        Transformer2DModel,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        UNet3DConditionModel,
        UNetControlNetXSModel,
        UNetMotionModel,
        UNetSpatioTemporalConditionModel,
        UVit2DModel,
        VQModel,
    )
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipelines import (
        BlipDiffusionPipeline,
        ConsistencyModelPipeline,
        DDIMPipeline,
        DDPMPipeline,
        DiffusionPipeline,
        Kandinsky3Img2ImgPipeline,
        Kandinsky3Pipeline,
        KandinskyCombinedPipeline,
        KandinskyImg2ImgCombinedPipeline,
        KandinskyImg2ImgPipeline,
        KandinskyInpaintCombinedPipeline,
        KandinskyInpaintPipeline,
        KandinskyPipeline,
        KandinskyPriorPipeline,
        KandinskyV22CombinedPipeline,
        KandinskyV22ControlnetImg2ImgPipeline,
        KandinskyV22ControlnetPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22Img2ImgPipeline,
        KandinskyV22InpaintCombinedPipeline,
        KandinskyV22InpaintPipeline,
        KandinskyV22Pipeline,
        KandinskyV22PriorEmb2EmbPipeline,
        KandinskyV22PriorPipeline,
        LatentConsistencyModelImg2ImgPipeline,
        LatentConsistencyModelPipeline,
        StableCascadeCombinedPipeline,
        StableCascadeDecoderPipeline,
        StableCascadePriorPipeline,
        StableDiffusion3Pipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionPipeline,
        StableDiffusionUpscalePipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
        StableDiffusionXLPipeline,
        StableVideoDiffusionPipeline,
        WuerstchenCombinedPipeline,
        WuerstchenDecoderPipeline,
        WuerstchenPriorPipeline,
    )
    from .schedulers import (
        CMStochasticIterativeScheduler,
        ConsistencyDecoderScheduler,
        DDIMInverseScheduler,
        DDIMParallelScheduler,
        DDIMScheduler,
        DDPMParallelScheduler,
        DDPMScheduler,
        DDPMWuerstchenScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EDMDPMSolverMultistepScheduler,
        EDMEulerScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        FlowMatchEulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LCMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SASolverScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        TCDScheduler,
        UnCLIPScheduler,
        UniPCMultistepScheduler,
        VQDiffusionScheduler,
    )
    from .utils import logging

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
