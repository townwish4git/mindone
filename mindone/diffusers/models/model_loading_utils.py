# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import os
from collections import OrderedDict
from typing import List, Optional, Union

import mindspore as ms

from ...safetensors.mindspore import load_file as safe_load_file
from ..utils import SAFETENSORS_FILE_EXTENSION, logging

logger = logging.get_logger(__name__)

_CLASS_REMAPPING_DICT = {
    "Transformer2DModel": {
        "ada_norm_zero": "DiTTransformer2DModel",
        "ada_norm_single": "PixArtTransformer2DModel",
    }
}


def _fetch_remapped_cls_from_config(config, old_class):
    previous_class_name = old_class.__name__
    remapped_class_name = _CLASS_REMAPPING_DICT.get(previous_class_name).get(config["norm_type"], None)

    # Details:
    # https://github.com/huggingface/diffusers/pull/7647#discussion_r1621344818
    if remapped_class_name:
        # load diffusers library to import compatible and original scheduler
        diffusers_library = importlib.import_module(__name__.split(".")[0] + ".diffusers")
        remapped_class = getattr(diffusers_library, remapped_class_name)
        logger.info(
            f"Changing class object to be of `{remapped_class_name}` type from `{previous_class_name}` type."
            f"This is because `{previous_class_name}` is scheduled to be deprecated in a future version. Note that this"
            " DOESN'T affect the final results."
        )
        return remapped_class
    else:
        return old_class


def load_state_dict(checkpoint_file: Union[str, os.PathLike], variant: Optional[str] = None):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        file_extension = os.path.basename(checkpoint_file).split(".")[-1]
        if file_extension == SAFETENSORS_FILE_EXTENSION:
            return safe_load_file(checkpoint_file)
        else:
            raise NotImplementedError(
                f"Only supports deserialization of weights file in safetensors format, but got {checkpoint_file}"
            )
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install "
                        "git-lfs and run `git lfs install` followed by `git lfs pull` in the folder "
                        "you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained "
                        "model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' " f"at '{checkpoint_file}'. "
            )


def _load_state_dict_into_model(model_to_load, state_dict: OrderedDict) -> List[str]:
    # TODO: error_msgs is always empty for now. Maybe we need to rewrite MindSpore's `load_param_into_net`.
    #  Error msgs should contain caught exception like size mismatch instead of missing/unexpected keys.
    # TODO: We should support loading float16 state_dict into float32 model, like PyTorch's behavior.
    error_msgs = []
    # TODO: State dict loading in mindspore does not cast dtype correctly. We do it manually. It's might unsafe.
    local_state = {k: v for k, v in model_to_load.parameters_and_names()}
    for k, v in state_dict.items():
        if k in local_state:
            v.set_dtype(local_state[k].dtype)
        else:
            pass  # unexpect key keeps origin dtype
    ms.load_param_into_net(model_to_load, state_dict, strict_load=True)
    return error_msgs
