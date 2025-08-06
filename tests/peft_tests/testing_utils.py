# Copyright 2023-present the HuggingFace Inc. team.
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
import unittest
from contextlib import contextmanager

import numpy as np

import mindspore as ms
from mindspore.device_context.ascend import device_count as ascend_device_count
from mindspore.device_context.ascend import is_available as is_ascend_available


def require_non_cpu(test_case):
    """
    Decorator marking a test that requires a hardware accelerator backend. These tests are skipped when there are no
    hardware accelerator available.
    """
    return unittest.skipUnless(is_ascend_available(), "test requires a hardware accelerator")(test_case)


def require_mindspore_npu(test_case):
    """
    Decorator marking a test that requires a NPU. Will be skipped when no NPU is available.
    """
    if not is_ascend_available():
        return unittest.skip("test requires Ascend NPU")(test_case)
    else:
        return test_case


def require_mindspore_multi_npu(test_case):
    """
    Decorator marking a test that requires multiple NPUs. Will be skipped when less than 2 NPUs are available.
    """
    if not is_ascend_available() or ascend_device_count() < 2:
        return unittest.skip("test requires multiple NPUs")(test_case)
    else:
        return test_case


def require_multi_accelerator(test_case):
    """
    Decorator marking a test that requires multiple hardware accelerators. These tests are skipped on a machine without
    multiple accelerators.
    """
    return unittest.skipUnless(
        is_ascend_available() and ascend_device_count() > 1, "test requires multiple hardware accelerators"
    )(test_case)


@contextmanager
def temp_seed(seed: int):
    """Temporarily set the random seed. This works for python numpy, pytorch."""

    np_state = np.random.get_state()
    np.random.seed(seed)

    torch_state = ms.get_rng_state()
    ms.manual_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(np_state)
        ms.set_rng_state(torch_state)


def get_state_dict(model, unwrap_compiled=True):
    """
    Get the state dict of a model. If the model is compiled, unwrap it first.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)
    return model.state_dict()
