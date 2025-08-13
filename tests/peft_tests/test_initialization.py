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


import itertools
import platform
import re
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import patch

import pytest
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from huggingface_hub.utils import reset_sessions
from scipy import stats
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.peft import LoraConfig, PeftModel, get_peft_model, inject_adapter_in_model, set_peft_model_state_dict
from mindone.peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
from mindone.peft.tuners.lora.config import CordaConfig
from mindone.peft.tuners.lora.layer import LoraLayer
from mindone.peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap
from mindone.safetensors.mindspore import load_file
from mindone.transformers import AutoModelForCausalLM


class TestLoraInitialization:
    """Test class to check the initialization of LoRA adapters."""

    def get_uniform(self, amin, amax, size=(10000,)):
        amin = amin if isinstance(amin, ms.Tensor) else ms.tensor(amin)
        amax = amax if isinstance(amax, ms.Tensor) else ms.tensor(amax)

        samples = ops.uniform(shape=size, minval=amin, maxval=amax)
        return samples

    def get_normal(self, mean, std, size=(10000,)):
        samples = ops.normal(shape=size, mean=mean, stddev=std)
        return samples

    def get_model(self):
        class MyModule(nn.Cell):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = mint.nn.Linear(1000, 1000)
                self.embed = mint.nn.Embedding(1000, 1000)
                self.conv2d = mint.nn.Conv2d(100, 100, 3)

            def construct(self, x):
                x_int = (100 * x).int()
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.embed(x_int), self.conv2d(x_4d)

        return MyModule().set_train(False)

    @pytest.fixture
    def data(self):
        return mint.rand(10, 1000)

    def test_lora_linear_init_default(self):
        # default is True
        ms.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"])
        model = get_peft_model(model, config)
        weight_A = model.linear.lora_A["default"].weight
        weight_B = model.linear.lora_B["default"].weight

        # use statistical test to check if weight A is from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.flatten().numpy(), unif.flatten().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a normal distribution
        normal = self.get_normal(weight_A.mean().item(), weight_A.std().item())
        _, p_value = stats.kstest(weight_A.flatten().numpy(), normal.flatten().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_linear_init_gaussian(self):
        # use gaussian init
        ms.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.linear.lora_A["default"].weight
        weight_B = model.linear.lora_B["default"].weight

        # use statistical test to check if weight A is from a normal distribution
        normal = self.get_normal(0.0, 1 / config.r)
        _, p_value = stats.kstest(weight_A.flatten().numpy(), normal.flatten().numpy())

        assert p_value > 0.5

        # check that weight A is *not* from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.flatten().numpy(), unif.flatten().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_linear_false(self):
        ms.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_B = model.linear.lora_B["default"].weight

        # with init_lora_weights=False, weight B should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not mint.allclose(weight_B, mint.zeros_like(weight_B))

    def test_lora_conv2d_default(self):
        # default is True
        ms.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"])
        model = get_peft_model(model, config)
        weight_A = model.conv2d.lora_A["default"].weight
        weight_B = model.conv2d.lora_B["default"].weight

        # use statistical test to check if weight A is from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.flatten().numpy(), unif.flatten().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a normal distribution
        normal = self.get_normal(weight_A.mean().item(), weight_A.std().item())
        _, p_value = stats.kstest(weight_A.flatten().numpy(), normal.flatten().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_conv2d_init_gaussian(self):
        # use gaussian init
        ms.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"], init_lora_weights="gaussian")
        model = get_peft_model(model, config)
        weight_A = model.conv2d.lora_A["default"].weight
        weight_B = model.conv2d.lora_B["default"].weight

        # use statistical test to check if weight A is from a normal distribution
        normal = self.get_normal(0.0, 1 / config.r)
        _, p_value = stats.kstest(weight_A.flatten().numpy(), normal.flatten().numpy())
        assert p_value > 0.5

        # check that weight A is *not* from a uniform distribution
        unif = self.get_uniform(weight_A.min().item(), weight_A.max().item())
        _, p_value = stats.kstest(weight_A.flatten().numpy(), unif.flatten().numpy())
        assert p_value < 0.05

        # check that weight B is zero
        assert (weight_B == 0.0).all()

    def test_lora_conv2d_false(self):
        ms.manual_seed(0)

        model = self.get_model()
        config = LoraConfig(target_modules=["conv2d"], init_lora_weights=False)
        model = get_peft_model(model, config)
        weight_B = model.conv2d.lora_B["default"].weight

        # with init_lora_weights=False, weight B should *not* be zero. We don't care so much about the actual values
        # as long as they are not zero, in order to avoid identity transformation.
        assert not mint.allclose(weight_B, mint.zeros_like(weight_B))

    def test_lora_scaling_default(self):
        # default is True
        ms.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=False
        config = LoraConfig(target_modules=["linear", "conv2d"], lora_alpha=3, r=16, use_rslora=False)
        model = get_peft_model(model, config)

        expected_scaling = config.lora_alpha / config.r

        assert model.linear.scaling["default"] == expected_scaling
        assert model.conv2d.scaling["default"] == expected_scaling

    # testcase for bugfix for issue 2194
    def test_rank_alpha_pattern_override(self):
        ms.manual_seed(0)

        layer = self.get_model()
        model = nn.SequentialCell(layer, layer)
        config = LoraConfig(
            target_modules=["linear"],
            lora_alpha=1,
            r=8,
            use_rslora=False,
            rank_pattern={"linear": 8},
            alpha_pattern={"0.linear": 2},
        )
        model = get_peft_model(model, config)
        scaling_with_rank_pattern = model.model[0].linear.scaling

        layer = self.get_model()
        model = nn.SequentialCell(layer, layer)
        config = LoraConfig(
            target_modules=["linear"], lora_alpha=1, r=8, use_rslora=False, alpha_pattern={"0.linear": 2}
        )
        model = get_peft_model(model, config)
        scaling_without_rank_pattern = model.model[0].linear.scaling

        assert scaling_with_rank_pattern == scaling_without_rank_pattern

    def test_lora_pissa_linear_init_default(self, data):
        model = self.get_model()
        output = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert mint.allclose(output, peft_model(data)[0], atol=1e-06)

        config = LoraConfig(init_lora_weights="pissa_niter_16", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert mint.allclose(output, peft_model(data)[0], atol=1e-06)

    def test_lora_olora_linear_init_default(self, data):
        model = self.get_model()
        output = model(data)[0]

        # Both OLoRA and olora should work
        config = LoraConfig(init_lora_weights="OLoRA", target_modules=["linear"])
        peft_model = get_peft_model(deepcopy(model), config)
        assert mint.allclose(output, peft_model(data)[0], atol=1e-06)

    def test_lora_pissa_conversion_same_output_after_loading(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_config_keys_before = list(peft_model.peft_config.keys())
        peft_config_dict_before = peft_model.peft_config["default"].to_dict()
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        peft_config_keys_after = list(peft_model.peft_config.keys())
        peft_config_dict_after = peft_model.peft_config["default"].to_dict()
        assert peft_config_keys_before == peft_config_keys_after
        assert peft_config_dict_before == peft_config_dict_after

        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_pissa_conversion_same_output_after_loading_with_rank_pattern(self, data, tmp_path):
        # same as above, but using rank_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use rank_pattern here; note that since there is only a single linear layer, r is completely overridden
        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8, rank_pattern={"linear": 32})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 32
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 64
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_pissa_conversion_same_output_after_loading_with_alpha_pattern(self, data, tmp_path):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use alpha_pattern here; note that since there is only a single linear layer, lora_alpha is completely
        # overridden
        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], alpha_pattern={"linear": 5})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 5 / 8
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        assert model_converted.base_model.model.linear.scaling["default"] == 10 / 16
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_lora_pissa_conversion_same_output_after_loading_with_rslora(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8, use_rslora=True)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model.peft_config["default"].init_lora_weights = "pissa"

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_pissa = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 8 / (8**0.5)
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_pissa, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # same scale as before with a little bit of floating point imprecision
        assert model_converted.base_model.model.linear.scaling["default"] == pytest.approx(8 / (8**0.5))
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_pissa_rank_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="pissa", target_modules=["linear"], r=8, rank_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "pissa-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    def test_pissa_alpha_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="pissa", target_modules=["linear"], r=8, alpha_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "pissa-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    def test_olora_conversion_same_output_after_loading(self, data, tmp_path):
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], r=8)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_config_keys_before = list(peft_model.peft_config.keys())
        peft_config_dict_before = peft_model.peft_config["default"].to_dict()
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        peft_config_keys_after = list(peft_model.peft_config.keys())
        peft_config_dict_after = peft_model.peft_config["default"].to_dict()
        assert peft_config_keys_before == peft_config_keys_after
        assert peft_config_dict_before == peft_config_dict_after

        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_conversion_same_output_after_loading_with_rank_pattern(self, data, tmp_path):
        # same as above, but using rank_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use rank_pattern here; note that since there is only a single linear layer, r is completely overridden
        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], r=8, rank_pattern={"linear": 32})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 32
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 64
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_conversion_same_output_after_loading_with_alpha_pattern(self, data, tmp_path):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        # use alpha_pattern here; note that since there is only a single linear layer, lora_alpha is completely
        # overridden
        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], alpha_pattern={"linear": 5})
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 5 / 8
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        assert model_converted.base_model.model.linear.scaling["default"] == 10 / 16
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_conversion_same_output_after_loading_with_rslora(self, data, tmp_path):
        # same as above, but using alpha_pattern
        model = self.get_model()
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="olora", target_modules=["linear"], r=8, use_rslora=True)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.save_pretrained(tmp_path / "init-model")

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight *= 2.0
        output_olora = peft_model(data)[0]

        # sanity check
        tol = 1e-06
        assert not mint.allclose(output_base, output_olora, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "olora-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model")
        output_loaded = model_loaded(data)[0]

        assert mint.allclose(output_olora, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8
        assert model_loaded.base_model.model.linear.scaling["default"] == 8 / (8**0.5)
        # sanity check: the base model weights were indeed changed
        assert not mint.allclose(
            model.linear.weight, model_loaded.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "olora-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "olora-model-converted")
        output_converted = model_converted(data)[0]

        assert mint.allclose(output_olora, output_converted, atol=tol, rtol=tol)
        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # same scale as before with a little bit of floating point imprecision
        assert model_converted.base_model.model.linear.scaling["default"] == pytest.approx(8 / (8**0.5))
        # base model weights should be the same as the initial model
        assert mint.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )

    def test_olora_rank_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="olora", target_modules=["linear"], r=8, rank_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "olora-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    def test_olora_alpha_pattern_and_rslora_raises(self, tmp_path):
        # it's not possible to determine the correct scale when using rslora with rank or alpha pattern, because the
        # scale is not stored in the state_dict
        model = self.get_model()
        config = LoraConfig(
            init_lora_weights="olora", target_modules=["linear"], r=8, alpha_pattern={"linear": 2}, use_rslora=True
        )
        peft_model = get_peft_model(model, config)
        peft_model.save_pretrained(tmp_path / "init-model")

        msg = re.escape("Passing `path_initial_model_for_weight_conversion` to `save_pretrained`")
        with pytest.raises(ValueError, match=msg):
            peft_model.save_pretrained(
                tmp_path / "olora-model", path_initial_model_for_weight_conversion=tmp_path / "init-model"
            )

    @pytest.mark.parametrize(
        "config_kwargs, should_warn",
        [
            # no warning
            ({"init_lora_weights": "pissa", "target_modules": ["linear"]}, False),
            ({"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"]}, False),
            ({"init_lora_weights": "olora", "target_modules": ["linear"]}, False),
            ({"init_lora_weights": "pissa", "target_modules": ["linear"], "use_rslora": True}, False),
            ({"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"], "use_rslora": True}, False),
            ({"init_lora_weights": "olora", "target_modules": ["linear"], "use_rslora": True}, False),
            ({"init_lora_weights": "pissa", "target_modules": ["linear"], "rank_pattern": {"linear": 8}}, False),
            (
                {"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"], "rank_pattern": {"linear": 8}},
                False,
            ),
            ({"init_lora_weights": "olora", "target_modules": ["linear"], "rank_pattern": {"linear": 8}}, False),
            ({"init_lora_weights": "pissa", "target_modules": ["linear"], "alpha_pattern": {"linear": 8}}, False),
            (
                {"init_lora_weights": "pissa_niter_3", "target_modules": ["linear"], "alpha_pattern": {"linear": 8}},
                False,
            ),
            ({"init_lora_weights": "olora", "target_modules": ["linear"], "alpha_pattern": {"linear": 8}}, False),
            # warning
            (
                {
                    "init_lora_weights": "pissa",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa_niter_3",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "olora",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa_niter_3",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "olora",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "pissa_niter_3",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
            (
                {
                    "init_lora_weights": "olora",
                    "target_modules": ["linear"],
                    "use_rslora": True,
                    "rank_pattern": {"linear": 8},
                    "alpha_pattern": {"linear": 8},
                },
                True,
            ),
        ],
    )
    def test_lora_config_pissa_olora_warns(self, config_kwargs, should_warn, recwarn):
        # Using post training conversion of modified base weights to restore their initial values (PiSSA, OLoRA) cannot
        # be correctly done when using rslora + rank_pattern/alpha_pattern. We can't really know if the user intends
        # this when they'll eventually call save_pretrained (i.e. if they'll pass
        # path_initial_model_for_weight_conversionl). Therefore, we only warn but don't raise an error here.
        msg = re.escape("Using Rank-Stabilized LoRA with rank_pattern/alpha_pattern and post-training conversion")
        if should_warn:
            LoraConfig(**config_kwargs)
            assert len(recwarn.list) == 1
            with pytest.warns(UserWarning, match=msg):
                LoraConfig(**config_kwargs)
        else:
            LoraConfig(**config_kwargs)
            assert not recwarn.list

    @pytest.mark.parametrize("init_method", ["pissa", "olora"])
    @pytest.mark.parametrize("pissa_olora_loaded_first", [False, True])
    def test_load_pissa_olora_with_other_adapter_warns(self, init_method, pissa_olora_loaded_first, recwarn, tmp_path):
        # Since PiSSA/OLoRA modifies the base weights, it should not be combined with other adapters. Check for a
        # warning. See #2184.

        # create an adapter without PiSSA/OloRA
        model_id = "townwish/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = get_peft_model(model, LoraConfig(init_lora_weights=True))
        model.save_pretrained(tmp_path / "adapter0")
        del model

        # create a model with PiSSA/OLoRA
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = get_peft_model(model, LoraConfig(init_lora_weights=init_method))
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load the model
        if pissa_olora_loaded_first:
            path0, path1 = tmp_path / "adapter1", tmp_path / "adapter0"
        else:
            path0, path1 = tmp_path / "adapter0", tmp_path / "adapter1"

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = PeftModel.from_pretrained(model, path0)
        model = model.load_adapter(path1, adapter_name="other")

        if init_method == "pissa":
            msg = "PiSSA changes the base weights of the model and should thus not be used with other adapters"
        else:
            msg = "OLoRA changes the base weights of the model and should thus not be used with other adapters"
        assert any(str(w.message).startswith(msg) for w in recwarn.list)

    def test_lora_rslora_scaling(self):
        # default is True
        ms.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear", "conv2d"], lora_alpha=3, r=16, use_rslora=True)
        model = get_peft_model(model, config)

        expected_scaling = config.lora_alpha / (config.r**0.5)

        assert model.linear.scaling["default"] == expected_scaling
        assert model.embed.scaling["default"] == expected_scaling
        assert model.conv2d.scaling["default"] == expected_scaling

    def test_lora_default_scaling_pattern(self):
        # default is True
        ms.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=False with rank and alpha pattern
        config = LoraConfig(
            target_modules=["linear", "conv2d"],
            rank_pattern={"conv2d": 16},
            alpha_pattern={"linear": 11, "conv2d": 13},
            lora_alpha=17,
            r=25,
            use_rslora=False,
        )
        model = get_peft_model(model, config)

        expected_scaling = {
            "linear": config.alpha_pattern["linear"] / config.r,
            "conv2d": config.alpha_pattern["conv2d"] / config.rank_pattern["conv2d"],
        }

        assert model.linear.scaling["default"] == expected_scaling["linear"]
        assert model.conv2d.scaling["default"] == expected_scaling["conv2d"]

    def test_lora_rslora_scaling_pattern(self):
        # default is True
        ms.manual_seed(0)

        model = self.get_model()

        # check scaling factor use_rslora=True with rank and alpha pattern
        config = LoraConfig(
            target_modules=["linear", "conv2d"],
            rank_pattern={"conv2d": 16},
            alpha_pattern={"linear": 11, "conv2d": 13},
            lora_alpha=17,
            r=25,
            use_rslora=True,
        )
        model = get_peft_model(model, config)

        expected_scaling = {
            "linear": config.alpha_pattern["linear"] / (config.r**0.5),
            "conv2d": config.alpha_pattern["conv2d"] / (config.rank_pattern["conv2d"] ** 0.5),
        }

        assert model.linear.scaling["default"] == expected_scaling["linear"]
        assert model.conv2d.scaling["default"] == expected_scaling["conv2d"]

    def test_lora_use_dora_linear(self, data):
        # check that dora is a no-op when initialized
        ms.manual_seed(0)
        model = self.get_model()
        output_base, _, _ = model(data)

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear"], use_dora=True)
        model = get_peft_model(model, config)

        with model.disable_adapter():
            output_disabled, _, _ = model(data)
        output_dora, _, _ = model(data)

        assert mint.allclose(output_base, output_disabled)
        assert mint.allclose(output_base, output_dora)

    def test_lora_use_dora_linear_init_false(self, data):
        # with init_lora_weights=False, dora should not be a no-op
        ms.manual_seed(0)
        model = self.get_model()
        output_base, _, _ = model(data)

        # check scaling factor use_rslora=True
        config = LoraConfig(target_modules=["linear"], use_dora=True, init_lora_weights=False)
        model = get_peft_model(model, config)

        with model.disable_adapter():
            output_disabled, _, _ = model(data)
        output_dora, _, _ = model(data)

        assert mint.allclose(output_base, output_disabled)
        assert not mint.allclose(output_base, output_dora)

    def test_lora_use_dora_with_megatron_core_raises(self):
        megatron_config = {"does-not": "matter-here"}
        with pytest.raises(ValueError, match="DoRA does not support megatron_core"):
            LoraConfig(target_modules=["linear"], use_dora=True, megatron_config=megatron_config)

    @pytest.fixture
    def mha_cls(self):
        class ModelMha(nn.Cell):
            def __init__(self, kdim=None, vdim=None):
                super().__init__()
                self.mha = nn.MultiheadAttention(10, 2, kdim=kdim, vdim=vdim)
                self.lin0 = mint.nn.Linear(10, 2)
                self.sm = nn.LogSoftmax(axis=-1)

            def construct(self, X):
                X = X.float()
                X, _ = self.mha(X, X, X)
                X = self.lin0(X)
                X = self.sm(X)
                return X

        return ModelMha

    def test_mha_load_init_model_first(self, mha_cls):
        # This test used to fail and require a workaround, for more context, see:
        # https://github.com/huggingface/peft/pull/1324#issuecomment-2252473980
        # The workaround was that _restore_weights had to be called manually on lora.MHA layers in order to make loading
        # the state dict work. With recent changes, this workaround is no longer required, so that test has been
        # deleted.
        inputs = mint.rand(10, 10, 10)
        model = mha_cls()
        config = LoraConfig(target_modules=["mha"], init_lora_weights=False)
        model = get_peft_model(model, config).set_train(False)
        restore_state_dict = {k: v for k, v in model.state_dict().items()}

        del model

        model = mha_cls()
        model = get_peft_model(model, config)
        # the workaround used to be:
        # for module in model.modules():
        #     if isinstance(module, peft.tuners.lora.layer.MultiheadAttention):
        #         module._restore_weights()
        model(inputs)
        model.load_state_dict(restore_state_dict)

    def test_mha_with_separate_qkv_embed_raises(self, mha_cls):
        # passing different kdim and vdim results in separate parameters for q, k, v, which is not supported (yet)
        model = mha_cls(kdim=20, vdim=30)
        config = LoraConfig(target_modules=["mha"])
        msg = "Only same embed for query/key/value is supported as of now for MultiheadAttention"
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    def test_mha_with_dora_raises(self, mha_cls):
        model = mha_cls()
        config = LoraConfig(target_modules=["mha"], use_dora=True)
        msg = re.escape("MultiheadAttention does not support DoRA (yet), please set use_dora to False")
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    def test_mha_exposes_attributes(self, mha_cls):
        # MHA requires a bunch of attributes to be exposed, try to check them exhaustively here
        model = mha_cls()
        embed_dim = model.mha.embed_dim
        kdim = model.mha.kdim
        vdim = model.mha.vdim
        qkv_same_embed_dim = model.mha._qkv_same_embed_dim
        num_heads = model.mha.num_heads
        dropout = model.mha.dropout
        batch_first = model.mha.batch_first
        head_dim = model.mha.head_dim
        in_proj_weight = model.mha.in_proj_weight
        in_proj_bias = model.mha.in_proj_bias
        out_proj = model.mha.out_proj
        bias_k = model.mha.bias_k
        bias_v = model.mha.bias_v
        add_zero_attn = model.mha.add_zero_attn

        config = LoraConfig(target_modules=["mha"])
        peft_model = get_peft_model(model, config)
        assert peft_model.base_model.mha.embed_dim == embed_dim
        assert peft_model.base_model.mha.kdim == kdim
        assert peft_model.base_model.mha.vdim == vdim
        assert peft_model.base_model.mha._qkv_same_embed_dim == qkv_same_embed_dim
        assert peft_model.base_model.mha.num_heads == num_heads
        assert peft_model.base_model.mha.dropout == dropout
        assert peft_model.base_model.mha.batch_first == batch_first
        assert peft_model.base_model.mha.head_dim == head_dim
        if in_proj_weight is not None:
            assert mint.allclose(peft_model.base_model.mha.in_proj_weight, in_proj_weight)
        else:
            assert peft_model.base_model.mha.in_proj_weight is None
        if in_proj_bias is not None:
            assert mint.allclose(peft_model.base_model.mha.in_proj_bias, in_proj_bias)
        else:
            assert peft_model.base_model.mha.in_proj_bias is None
        assert peft_model.base_model.mha.out_proj is out_proj
        if bias_k is not None:
            assert mint.allclose(peft_model.base_model.mha.bias_k, bias_k)
        else:
            assert peft_model.base_model.mha.bias_k is None
        if bias_v is not None:
            assert mint.allclose(peft_model.base_model.mha.bias_v, bias_v)
        else:
            assert peft_model.base_model.mha.bias_v is None
        assert peft_model.base_model.mha.add_zero_attn == add_zero_attn

    def test_mha_merge_masks_method(self, mha_cls):
        # MHA requires a merge_masks method to be exposed, check that it works
        model = mha_cls()
        config = LoraConfig(target_modules=["mha"])
        peft_model = get_peft_model(model, config)

        attn_mask = mint.randint(0, 2, (10, 10))
        key_padding_mask = mint.randint(0, 2, (10, 10))
        query = mint.rand(10, 10, 10)
        merged_mask0, mask_type0 = model.mha.merge_masks(attn_mask, key_padding_mask, query)
        merged_mask1, mask_type1 = peft_model.base_model.mha.merge_masks(attn_mask, key_padding_mask, query)

        assert mint.allclose(merged_mask0, merged_mask1)
        assert mask_type0 == mask_type1

    def test_lora_with_bias_extra_params(self):
        # lora with lora_bias=True
        model = self.get_model()
        config = LoraConfig(target_modules=["linear", "conv2d"], lora_bias=False)
        model_no_bias = get_peft_model(model, config)

        model = self.get_model()
        config = LoraConfig(target_modules=["linear", "conv2d"], lora_bias=True)
        model_bias = get_peft_model(model, config)

        # check that bias for LoRA B is set
        assert model_no_bias.base_model.model.linear.lora_B["default"].bias is None
        assert model_bias.base_model.model.linear.lora_B["default"].bias.shape == (1000,)
        assert model_no_bias.base_model.model.conv2d.lora_B["default"].bias is None
        assert model_bias.base_model.model.conv2d.lora_B["default"].bias.shape == (100,)

        # check that the same params are present except for the extra bias term
        params_no_bias = {name for name, _ in model_no_bias.parameters_and_names()}
        params_bias = {name for name, _ in model_bias.parameters_and_names()}
        extra_params = {
            "base_model.model.linear.lora_B.default.bias",
            "base_model.model.conv2d.lora_B.default.bias",
        }
        assert params_bias - params_no_bias == extra_params
        assert params_no_bias.issubset(params_bias)

    @pytest.mark.parametrize(
        "extra_kwargs",
        [
            {"use_dora": True},
            {"init_lora_weights": "eva"},
            {"init_lora_weights": "gaussian"},
            {"init_lora_weights": "olora"},
            {"init_lora_weights": "pissa"},
            {"init_lora_weights": "pissa_niter_3"},
        ],
    )
    def test_lora_with_bias_incompatible_arguments(self, extra_kwargs):
        # some arguments don't work in conjunction with lora_bias and should raise
        # just check the common chunk of the error message
        msg = "The argument lora_bias=True is"
        with pytest.raises(ValueError, match=msg):
            LoraConfig(target_modules=["linear"], lora_bias=True, **extra_kwargs)


class TestNoInfiniteRecursionDeepspeed:
    # see #1892 for details
    classes = [
        PeftModel,
    ]

    @pytest.fixture
    def wrap_init(self):
        # emulates the wrapper from DeepSpeed
        import functools

        def decorator(f):
            @functools.wraps(f)
            def wrapper(self, *args, **kwargs):
                hasattr(self, "abc")  # any hasattr will do
                f(self, *args, **kwargs)

            return wrapper

        return decorator

    @pytest.fixture
    def model(self):
        class MyModule(nn.Cell):
            def __init__(self):
                super().__init__()
                self.linear = mint.nn.Linear(10, 10)
                # to emulate LMs:
                self.prepare_inputs_for_generation = None
                self._prepare_encoder_decoder_kwargs_for_generation = None

        return MyModule()

    @pytest.mark.parametrize("cls", classes)
    def test_no_infinite_recursion(self, cls, model, wrap_init):
        original_init = cls.__init__
        try:
            cls.__init__ = wrap_init(cls.__init__)
            # this would trigger an infinite loop before the fix in 1892
            cls(model, LoraConfig(target_modules=["linear"]))
        finally:
            # ensure there are no side effects of this test
            cls.__init__ = original_init


class TestLoadAdapterOfflineMode:
    base_model = "townwish/tiny-random-OPTForCausalLM"
    peft_model_id = "townwish/tiny-OPTForCausalLM-lora"

    # make sure that PEFT honors offline mode
    @contextmanager
    def hub_offline_ctx(self):
        # this is required to simulate offline mode, setting the env var dynamically inside the test does not work
        # because the value is checked only once at the start of the session
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            reset_sessions()
            yield
        reset_sessions()

    def test_load_from_hub_then_offline_model(self):
        # this uses LoRA but it's the same mechanism for other methods
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model)

        # first ensure that the adapter model has been downloaded
        PeftModel.from_pretrained(base_model, self.peft_model_id)

        del base_model

        base_model = AutoModelForCausalLM.from_pretrained(self.base_model)
        with self.hub_offline_ctx():
            # does not raise
            PeftModel.from_pretrained(base_model, self.peft_model_id)

    @pytest.fixture
    def changed_default_cache_dir(self, tmp_path, monkeypatch):
        # ensure that this test does not interact with other tests that may use the HF cache
        monkeypatch.setattr("huggingface_hub.constants.HF_HOME", tmp_path)
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", tmp_path / "hub")
        monkeypatch.setattr("huggingface_hub.constants.HF_TOKEN_PATH", tmp_path / "token")

    def load_checkpoints(self, cache_dir):
        # download model and lora checkpoint to a specific cache dir
        snapshot_download(self.base_model, cache_dir=cache_dir)
        snapshot_download(self.peft_model_id, cache_dir=cache_dir)

    def test_load_checkpoint_offline_non_default_cache_dir(self, changed_default_cache_dir, tmp_path):
        # See #2373 for context
        self.load_checkpoints(tmp_path)
        with self.hub_offline_ctx():
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model, cache_dir=tmp_path)
            PeftModel.from_pretrained(base_model, self.peft_model_id, cache_dir=tmp_path)


class TestCustomModelConfigWarning:
    # Check potential warnings when the user provided base_model_name_or_path is overridden by PEFT. See #2001 for
    # context. We use LoRA for this test but the same applies to other methods
    @pytest.fixture
    def custom_module(self):
        class MyModule(nn.Cell):
            def __init__(self):
                super().__init__()
                self.lin = mint.nn.Linear(10, 10)

        return MyModule()

    def test_no_warning_by_default_transformers_model(self, recwarn):
        # first a sanity test that there is no warning by default when using a model from transformers
        model = AutoModelForCausalLM.from_pretrained("townwish/tiny-random-OPTForCausalLM")
        get_peft_model(model, LoraConfig())
        for warning in recwarn.list:
            assert "renamed" not in str(warning.message)

    def test_no_warning_by_default_custom_model(self, custom_module, recwarn):
        # same as above but with a custom model
        get_peft_model(custom_module, LoraConfig(target_modules=["lin"]))
        for warning in recwarn.list:
            assert "renamed" not in str(warning.message)

    def test_warning_name_transformers_model(self, recwarn):
        # The base_model_name_or_path provided by the user is overridden.
        model = AutoModelForCausalLM.from_pretrained("townwish/tiny-random-OPTForCausalLM")
        custom_name = "custom_name"
        get_peft_model(model, LoraConfig(base_model_name_or_path=custom_name))
        msg = f"was renamed from '{custom_name}' to 'townwish/tiny-random-OPTForCausalLM'"
        assert any(msg in str(warning.message) for warning in recwarn.list)

    def test_warning_name_custom_model(self, custom_module, recwarn):
        custom_name = "custom_name"
        get_peft_model(custom_module, LoraConfig(target_modules=["lin"], base_model_name_or_path=custom_name))
        msg = f"was renamed from '{custom_name}' to 'None'"
        assert any(msg in str(warning.message) for warning in recwarn.list)

    def test_warning_name_custom_model_with_custom_name(self, custom_module, recwarn):
        custom_name = "custom_name"
        custom_module.name_or_path = "foobar"
        get_peft_model(custom_module, LoraConfig(target_modules=["lin"], base_model_name_or_path=custom_name))
        msg = f"was renamed from '{custom_name}' to 'foobar'"
        assert any(msg in str(warning.message) for warning in recwarn.list)


def test_from_pretrained_missing_keys_warning(recwarn, tmp_path):
    # For more context, see issue 2115
    # When loading a PEFT adapter and we're missing a PEFT-specific weight, there should be a warning.
    model = AutoModelForCausalLM.from_pretrained("townwish/tiny-random-OPTForCausalLM")
    config = LoraConfig()
    model = get_peft_model(model, config)
    state_dict = model.state_dict()

    # first, sanity check that there are no warnings if no key is missing
    model.save_pretrained(tmp_path)
    del model
    model = AutoModelForCausalLM.from_pretrained("townwish/tiny-random-OPTForCausalLM")
    model = PeftModel.from_pretrained(model, tmp_path)
    msg = "Found missing adapter keys"
    assert not any(msg in str(w.message) for w in recwarn.list)

    # remove a key from the state_dict
    missing_key = "base_model.model.model.decoder.layers.0.self_attn.v_proj.lora_A.default.weight"

    def new_state_dict():
        return {k: v for k, v in state_dict.items() if k != missing_key}

    model.state_dict = new_state_dict
    model.save_pretrained(tmp_path)
    del model

    model = AutoModelForCausalLM.from_pretrained("townwish/tiny-random-OPTForCausalLM")
    model = PeftModel.from_pretrained(model, tmp_path)
    assert any(msg in str(w.message) for w in recwarn.list)
    assert any(missing_key in str(w.message) for w in recwarn.list)


class TestNamingConflictWarning:
    """
    Tests for warnings related to naming conflicts between adapter names and tuner prefixes. References: Issue 2252
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.peft_config = LoraConfig()
        self.prefix = PEFT_TYPE_TO_PREFIX_MAPPING[self.peft_config.peft_type]
        self.base_model = AutoModelForCausalLM.from_pretrained("townwish/tiny-random-OPTForCausalLM")

    def _save_and_reload_model(self, model, adapter_name, tmp_path):
        # Helper method to save and reload the PEFT model
        model.save_pretrained(tmp_path, selected_adapters=[adapter_name])
        del model
        reloaded_base_model = AutoModelForCausalLM.from_pretrained(tmp_path / adapter_name)
        return PeftModel.from_pretrained(reloaded_base_model, tmp_path / adapter_name)

    def test_no_warning_without_naming_conflict_get_peft_model(self, recwarn):
        # No warning should be raised when there is no naming conflict during get_peft_model.
        non_conflict_adapter = "adapter"
        _ = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        expected_msg = f"Adapter name {non_conflict_adapter} should not be contained in the prefix {self.prefix}."
        assert not any(expected_msg in str(w.message) for w in recwarn.list)

    def test_no_warning_without_naming_conflict_add_adapter(self, recwarn):
        # No warning should be raised when adding an adapter without naming conflict.
        non_conflict_adapter = "adapter"
        other_non_conflict_adapter = "other_adapter"
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        _ = model.add_adapter(other_non_conflict_adapter, self.peft_config)
        expected_msg = f"Adapter name {other_non_conflict_adapter} should not be contained in the prefix {self.prefix}."
        assert not any(expected_msg in str(w.message) for w in recwarn.list)

    def test_no_warning_without_naming_conflict_save_and_load(self, recwarn, tmp_path):
        # No warning should be raised when saving and loading the model without naming conflict.
        non_conflict_adapter = "adapter"
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        _ = self._save_and_reload_model(model, non_conflict_adapter, tmp_path)
        expected_msg = f"Adapter name {non_conflict_adapter} should not be contained in the prefix {self.prefix}."
        assert not any(expected_msg in str(w.message) for w in recwarn.list)

    def test_warning_naming_conflict_get_peft_model(self, recwarn):
        # Warning is raised when the adapter name conflicts with the prefix in get_peft_model.
        conflicting_adapter_name = self.prefix[:-1]
        _ = get_peft_model(self.base_model, self.peft_config, adapter_name=conflicting_adapter_name)
        expected_msg = f"Adapter name {conflicting_adapter_name} should not be contained in the prefix {self.prefix}."
        assert any(expected_msg in str(w.message) for w in recwarn.list)

    def test_warning_naming_conflict_add_adapter(self, recwarn):
        # Warning is raised when adding an adapter with a name that conflicts with the prefix.
        conflicting_adapter = self.prefix[1:]
        non_conflict_adapter = "adapter"
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=non_conflict_adapter)
        _ = model.add_adapter(conflicting_adapter, self.peft_config)
        expected_msg = f"Adapter name {conflicting_adapter} should not be contained in the prefix {self.prefix}."
        assert any(expected_msg in str(w.message) for w in recwarn.list)

    def test_warning_naming_conflict_save_and_load(self, recwarn, tmp_path):
        # Warning is raised when saving and loading the model with a naming conflict.
        conflicting_adapter = self.prefix[:-1]
        model = get_peft_model(self.base_model, self.peft_config, adapter_name=conflicting_adapter)
        _ = self._save_and_reload_model(model, conflicting_adapter, tmp_path)
        expected_msg = f"Adapter name {conflicting_adapter} should not be contained in the prefix {self.prefix}."
        assert any(expected_msg in str(w.message) for w in recwarn.list)


class TestHotSwapping:
    """Tests for the hotswapping function"""

    def get_model(self):
        class MLP(nn.Cell):
            def __init__(self, bias=True):
                super().__init__()
                self.lin0 = mint.nn.Linear(10, 20, bias=True)
                self.relu = nn.ReLU()
                self.lin1 = mint.nn.Linear(20, 5, bias=False)

            def construct(self, X):
                X = X.float()
                X = self.lin0(X)
                X = self.relu(X)
                X = self.lin1(X)
                return X

        ms.manual_seed(0)
        return MLP()

    def get_model_conv2d(self):
        class ConvModel(nn.Cell):
            def __init__(self):
                super().__init__()
                self.conv = mint.nn.Conv2d(3, 10, kernel_size=3)

            def construct(self, X):
                return self.conv(X)

        ms.manual_seed(0)
        return ConvModel()

    # this works with all adapters except prompt learning, but we don't test all
    # as it is unnecessary and would be slow
    @pytest.mark.parametrize(
        "config",
        [
            LoraConfig(init_lora_weights=0, target_modules=["lin0"]),
            LoraConfig(init_lora_weights=0, target_modules=["lin0", "lin1"]),
        ],
    )
    @pytest.mark.parametrize("do_compile", [False, True])
    def test_hotswap_works(self, config, do_compile, tmp_path):
        # Load 2 different adapters and check that we can hotswap between them, with the model optionally being
        # compiled.
        atol, rtol = 1e-4, 1e-4
        inputs = mint.rand(3, 10)

        # create adapter 0
        model = self.get_model()
        ms.manual_seed(0)
        model = get_peft_model(model, config)
        model.set_train(False)
        output0 = model(inputs) if not do_compile else model.compile_and_run(inputs)
        model.save_pretrained(tmp_path / "adapter0")

        del model

        # create adapter 1
        model = self.get_model()
        ms.manual_seed(1)
        model = get_peft_model(model, config)
        model.set_train(False)
        output1 = model(inputs) if not do_compile else model.compile_and_run(inputs)
        model.save_pretrained(tmp_path / "adapter1")

        # sanity check: they're not the same
        assert not mint.allclose(output0, output1, atol=atol, rtol=rtol)

        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")
        output_loaded0 = model(inputs) if not do_compile else model.compile_and_run(inputs)

        # sanity check: same output after loading for adapter 0
        assert mint.allclose(output0, output_loaded0, atol=atol, rtol=rtol)

        # hotswap with adapter 1
        hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")
        output_loaded1 = model(inputs)

        # real check: model now behaves like adapter 1
        assert mint.allclose(output1, output_loaded1, atol=atol, rtol=rtol)

        # hotswap back to adapter 0
        hotswap_adapter(model, tmp_path / "adapter0", adapter_name="default")
        output_loaded_back0 = model(inputs)

        # real check: model now behaves again like adapter 0
        assert mint.allclose(output0, output_loaded_back0, atol=atol, rtol=rtol)

    def test_hotswap_missing_key_works(self, tmp_path):
        # When a key is missing, it is fine, the extra weight is zeroed out
        config = LoraConfig(target_modules=["lin0", "lin1"])

        model = self.get_model()
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config)

        # remove one key from the state_dict
        key = "base_model.model.lin1.lora_A.default.weight"
        state_dict = model.state_dict()
        del state_dict[key]
        model.state_dict = lambda: state_dict
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        # sanity check: the missing weight is not already all zeros
        assert not (model.base_model.model.lin1.lora_A["default"].weight == 0).all()
        hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")
        # after hotswapping, it is zeroed out
        assert (model.base_model.model.lin1.lora_A["default"].weight == 0).all()

    def test_hotswap_extra_key_raises(self, tmp_path):
        # When there is an extra key, raise
        config = LoraConfig(target_modules=["lin0"])

        model = self.get_model()
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path / "adapter0")
        del model

        model = self.get_model()
        model = get_peft_model(model, config)

        # add an unexpected key
        state_dict = model.state_dict()
        new_key = "base_model.model.lin1.lora_A.default.weight"
        state_dict[new_key] = mint.zeros(8, 20)
        model.state_dict = lambda: state_dict
        model.save_pretrained(tmp_path / "adapter1")
        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")

        msg = f"Hot swapping the adapter did not succeed, unexpected keys found: {new_key}"
        with pytest.raises(RuntimeError, match=msg):
            hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")

    @pytest.mark.parametrize("ranks", [(7, 13), (13, 7)])
    def test_hotswap_works_different_ranks_alphas(self, ranks, tmp_path):
        # same as test_hotswap_works but different rank and alpha
        # Load 2 different adapters and check that we can hotswap between them, with the model optionally being
        # compiled.
        atol, rtol = 1e-4, 1e-4
        inputs = mint.rand(3, 10)

        # create adapter 0
        config0 = LoraConfig(target_modules=["lin0", "lin1"], r=ranks[0], lora_alpha=ranks[0], init_lora_weights=False)
        model = self.get_model()
        ms.manual_seed(0)
        model = get_peft_model(model, config0)
        model.set_train(False)
        output0 = model(inputs)
        model.save_pretrained(tmp_path / "adapter0")

        del model

        # create adapter 1
        config1 = LoraConfig(target_modules=["lin0"], r=ranks[1], lora_alpha=ranks[1], init_lora_weights=False)
        model = self.get_model()
        ms.manual_seed(1)
        model = get_peft_model(model, config1)
        model.set_train(False)
        output1 = model(inputs)
        model.save_pretrained(tmp_path / "adapter1")

        # sanity check: they're not the same
        assert not mint.allclose(output0, output1, atol=atol, rtol=rtol)

        del model

        # load adapter 0
        model = self.get_model()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")
        output_loaded0 = model(inputs)

        # sanity check: same output after loading for adapter 0
        assert mint.allclose(output0, output_loaded0, atol=atol, rtol=rtol)

        # hotswap with adapter 1
        hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")
        output_loaded1 = model(inputs)

        # real check: model now behaves like adapter 1
        assert mint.allclose(output1, output_loaded1, atol=atol, rtol=rtol)

        # hotswap back to adapter 0
        hotswap_adapter(model, tmp_path / "adapter0", adapter_name="default")
        output_loaded_back0 = model(inputs)

        # real check: model now behaves again like adapter 0
        assert mint.allclose(output0, output_loaded_back0, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("ranks", [(7, 13), (13, 7)])
    def test_hotswap_works_different_ranks_alphas_conv2d(self, ranks, tmp_path):
        # same as previous test, but for a Conv2d model
        atol, rtol = 1e-4, 1e-4
        inputs = mint.rand(3, 3, 10, 10)

        # create adapter 0
        config0 = LoraConfig(target_modules=["conv"], r=ranks[0], init_lora_weights=False)
        model = self.get_model_conv2d()
        ms.manual_seed(0)
        model = get_peft_model(model, config0)
        model.set_train(False)
        output0 = model(inputs)
        model.save_pretrained(tmp_path / "adapter0")

        del model

        # create adapter 1
        config1 = LoraConfig(target_modules=["conv"], r=ranks[1], init_lora_weights=False)
        model = self.get_model_conv2d()
        ms.manual_seed(1)
        model = get_peft_model(model, config1)
        model.set_train(False)
        output1 = model(inputs)
        model.save_pretrained(tmp_path / "adapter1")

        # sanity check: they're not the same
        assert not mint.allclose(output0, output1, atol=atol, rtol=rtol)

        del model

        # load adapter 0
        model = self.get_model_conv2d()
        model = PeftModel.from_pretrained(model, tmp_path / "adapter0")
        output_loaded0 = model(inputs)

        # sanity check: same output after loading for adapter 0
        assert mint.allclose(output0, output_loaded0, atol=atol, rtol=rtol)

        # hotswap with adapter 1
        hotswap_adapter(model, tmp_path / "adapter1", adapter_name="default")
        output_loaded1 = model(inputs)

        # real check: model now behaves like adapter 1
        assert mint.allclose(output1, output_loaded1, atol=atol, rtol=rtol)

        # hotswap back to adapter 0
        hotswap_adapter(model, tmp_path / "adapter0", adapter_name="default")
        output_loaded_back0 = model(inputs)

        # real check: model now behaves again like adapter 0
        assert mint.allclose(output0, output_loaded_back0, atol=atol, rtol=rtol)

    def test_prepare_model_for_compiled_hotswap_scalings_are_tensors(self):
        config = LoraConfig(target_modules=["lin0", "lin1"])
        model = self.get_model()
        model = get_peft_model(model, config)

        # sanity check: all scalings are floats
        scalings_before = {}
        for name, module in model.cells_and_names():
            if hasattr(module, "scaling"):
                for key, val in module.scaling.items():
                    assert isinstance(val, float)
                    scalings_before[f"{name}.{key}"] = val

        prepare_model_for_compiled_hotswap(model)

        scalings_after = {}
        for name, module in model.cells_and_names():
            if hasattr(module, "scaling"):
                for key, val in module.scaling.items():
                    assert isinstance(val, ms.Tensor)
                    scalings_after[f"{name}.{key}"] = val.item()

        assert scalings_before == scalings_after

    def test_prepare_model_for_compiled_hotswap_rank_padding_works(self):
        old_rank = 8
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank)
        model = self.get_model()
        model = get_peft_model(model, config)

        # sanity check
        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                assert param.shape[0] == old_rank
            elif "lora_B" in name:
                assert param.shape[1] == old_rank

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)

        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                assert param.shape[0] == new_rank
            elif "lora_B" in name:
                assert param.shape[1] == new_rank

    def test_prepare_model_for_compiled_hotswap_same_rank_padding_works(self):
        # same as previous test, but ensure there is no error if the rank to pad to is the same
        old_rank = 8
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank)
        model = self.get_model()
        model = get_peft_model(model, config)
        prepare_model_for_compiled_hotswap(model, target_rank=old_rank)

        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                assert param.shape[0] == old_rank
            elif "lora_B" in name:
                assert param.shape[1] == old_rank

    def test_prepare_model_for_compiled_hotswap_conv2d_rank_padding_works(self):
        # same as previous test, but for a Conv2d model
        old_rank = 8
        config = LoraConfig(target_modules=["conv"], r=old_rank)
        model = self.get_model_conv2d()
        model = get_peft_model(model, config)

        # sanity check
        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                assert param.shape[0] == old_rank
            elif "lora_B" in name:
                assert param.shape[1] == old_rank

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)

        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                assert param.shape[0] == new_rank
            elif "lora_B" in name:
                assert param.shape[1] == new_rank

    def test_prepare_model_for_compiled_hotswap_lower_rank_padding_raises(self):
        # when trying to pad to a lower rank, raise an error
        old_rank0 = 8
        old_rank1 = 10
        new_rank = 9
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank0, rank_pattern={"lin1": old_rank1})
        model = self.get_model()
        model = get_peft_model(model, config)

        msg = re.escape("Trying to pad the adapter to the target rank 9, but the original rank is larger (10)")
        with pytest.raises(ValueError, match=msg):
            prepare_model_for_compiled_hotswap(model, target_rank=new_rank)

    def test_prepare_model_for_compiled_hotswap_with_rank_pattern(self):
        old_rank0 = 8
        old_rank1 = 9
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank0, rank_pattern={"lin1": old_rank1})
        model = self.get_model()
        model = get_peft_model(model, config)

        # sanity check
        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                if "lin0" in name:
                    assert param.shape[0] == old_rank0
                else:
                    assert param.shape[0] == old_rank1
            elif "lora_B" in name:
                if "lin0" in name:
                    assert param.shape[1] == old_rank0
                else:
                    assert param.shape[1] == old_rank1

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)

        for name, param in model.parameters_and_names():
            if "lora_A" in name:
                assert param.shape[0] == new_rank
            elif "lora_B" in name:
                assert param.shape[1] == new_rank

    def test_prepare_model_for_compiled_hotswap_model_no_adapter_raises(self):
        model = self.get_model()
        msg = re.escape("No adapter layers found on the model")
        with pytest.raises(ValueError, match=msg):
            prepare_model_for_compiled_hotswap(model)

    def test_prepare_model_for_compiled_hotswap_does_not_change_output(self):
        # preparing the model for hotswapping should not change the model output
        inputs = mint.rand(3, 10)
        model = self.get_model().set_train(False)
        output_base = model(inputs)

        old_rank = 8
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank, init_lora_weights=False)
        model = get_peft_model(model, config).set_train(False)
        output_before = model(inputs)

        # sanity check: LoRA changed output
        assert not mint.allclose(output_base, output_before)

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)
        output_after = model(inputs)

        assert mint.allclose(output_before, output_after)

    def test_prepare_model_for_compiled_hotswap_does_not_change_output_conv2d(self):
        # preparing the model for hotswapping should not change the model output
        inputs = mint.rand(3, 3, 10, 10)
        model = self.get_model_conv2d().set_train(False)
        output_base = model(inputs)

        old_rank = 8
        config = LoraConfig(target_modules=["conv"], r=old_rank, init_lora_weights=False)
        model = get_peft_model(model, config).set_train(False)
        output_before = model(inputs)

        # sanity check: LoRA changed output
        assert not mint.allclose(output_base, output_before)

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)
        output_after = model(inputs)

        assert mint.allclose(output_before, output_after)

    def test_prepare_model_for_compiled_hotswap_scalings_update_config(self):
        old_rank0 = 11
        old_rank1 = 13
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank0, rank_pattern={"lin1": old_rank1})
        model = self.get_model()
        model = get_peft_model(model, config)

        new_rank = 15
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank, config=model.peft_config)

        assert model.peft_config["default"].r == new_rank
        assert model.peft_config["default"].rank_pattern == {"lin1": new_rank}

    def test_prepare_model_for_compiled_hotswap_lora_bias(self):
        # When setting lora_bias=True in the LoraConfig, the LoRA B parameter will have a bias term. Check that padding
        # still works correctly. Note that the LoRA A parameter still won't have a bias term.
        old_rank = 8
        config = LoraConfig(target_modules=["lin0", "lin1"], r=old_rank, lora_bias=True)
        model = self.get_model()
        model = get_peft_model(model, config)

        # sanity check
        for name, param in model.parameters_and_names():
            if "lora_A" in name and name.endswith(".weight"):
                assert param.shape[0] == old_rank
            elif "lora_B" in name and name.endswith(".weight"):
                assert param.shape[1] == old_rank
            elif "lora_A" in name and name.endswith(".bias"):
                assert False, "LoRA A should not have a bias term"
            elif "lora_B" in name and name.endswith(".bias"):
                assert param.shape[0] in (5, 20)  # output shapes of the 2 layers

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)

        for name, param in model.parameters_and_names():
            if "lora_A" in name and name.endswith(".weight"):
                assert param.shape[0] == new_rank
            elif "lora_B" in name and name.endswith(".weight"):
                assert param.shape[1] == new_rank
            elif "lora_A" in name and name.endswith(".bias"):
                assert False, "LoRA A should not have a bias term"
            elif "lora_B" in name and name.endswith(".bias"):
                assert param.shape[0] in (5, 20)  # output shapes of the 2 layers

    def test_prepare_model_for_compiled_hotswap_conv2d_lora_bias(self):
        # same as previous test, but for a Conv2d model
        old_rank = 8
        config = LoraConfig(target_modules=["conv"], r=old_rank, lora_bias=True)
        model = self.get_model_conv2d()
        model = get_peft_model(model, config)

        # sanity check
        for name, param in model.parameters_and_names():
            if "lora_A" in name and name.endswith(".weight"):
                assert param.shape[0] == old_rank
            elif "lora_B" in name and name.endswith(".weight"):
                assert param.shape[1] == old_rank
            elif "lora_A" in name and name.endswith(".bias"):
                assert False, "LoRA A should not have a bias term"
            elif "lora_B" in name and name.endswith(".bias"):
                assert param.shape[0] == 10  # output shape of conv layer

        new_rank = 13
        prepare_model_for_compiled_hotswap(model, target_rank=new_rank)

        for name, param in model.parameters_and_names():
            if "lora_A" in name and name.endswith(".weight"):
                assert param.shape[0] == new_rank
            elif "lora_B" in name and name.endswith(".weight"):
                assert param.shape[1] == new_rank
            elif "lora_A" in name and name.endswith(".bias"):
                assert False, "LoRA A should not have a bias term"
            elif "lora_B" in name and name.endswith(".bias"):
                assert param.shape[0] == 10  # output shape of conv layer


def test_import_peft_type_to_model_mapping_deprecation_warning(recwarn):
    # This is for backwards compatibility: In #2282, PEFT_TYPE_TO_MODEL_MAPPING was removed as it was redundant with
    # PEFT_TYPE_TO_TUNER_MAPPING. However, third party code could still use this mapping, e.g.:
    # https://github.com/AutoGPTQ/AutoGPTQ/blob/6689349625de973b9ee3016c28c11f32acf7f02c/auto_gptq/utils/peft_utils.py#L8
    # TODO: Remove after 2026-01

    # first check that there is no warning under normal circumstances
    from mindone.peft.peft_model import PeftModel  # noqa

    expected = (
        "PEFT_TYPE_TO_MODEL_MAPPING is deprecated, please use `from peft import PEFT_TYPE_TO_TUNER_MAPPING` instead"
    )
    warnings = (w.message.args[0] for w in recwarn.list)
    assert not any(w.startswith(expected) for w in warnings)

    from mindone.peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING  # noqa

    # check that there is a warning with this message after importing the variable
    warnings = (w.message.args[0] for w in recwarn.list)
    assert any(w.startswith(expected) for w in warnings)
