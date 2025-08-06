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
import copy
import json
import os
import pickle
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from unittest import mock

import pytest
import yaml
from packaging import version

import mindspore as ms
from mindspore import mint

from mindone.diffusers import StableDiffusionPipeline
from mindone.peft import (
    LoraConfig,
    PeftModel,
    PeftType,
    PromptLearningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    prepare_model_for_kbit_training,
)
from mindone.peft.tuners.lora import LoraLayer
from mindone.peft.tuners.tuners_utils import BaseTunerLayer
from mindone.peft.utils import _get_submodules
from mindone.safetensors.mindspore import save_file

from .testing_utils import get_state_dict

HUB_MODEL_ACCESSES = {}

CONFIG_TESTING_KWARGS = (
    # LoRA
    {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": None,
        "lora_dropout": 0.05,
        "bias": "none",
    },
)

CLASSES_MAPPING = {
    "lora": (LoraConfig, CONFIG_TESTING_KWARGS[0]),
}

DECODER_MODELS_EXTRA = {}


# Adapted from https://github.com/huggingface/transformers/blob/48327c57182fdade7f7797d1eaad2d166de5c55b/src/transformers/activations.py#LL166C7-L166C22
class ClassInstantier(OrderedDict):
    def __getitem__(self, key, *args, **kwargs):
        # check if any of the kwargs is inside the config class kwargs
        if any(kwarg in self[key][1] for kwarg in kwargs):
            new_config_kwargs = self[key][1].copy()
            new_config_kwargs.update(kwargs)
            return (self[key][0], new_config_kwargs)

        return super().__getitem__(key, *args, **kwargs)

    def get_grid_parameters(self, grid_parameters, filter_params_func=None):
        r"""
        Returns a list of all possible combinations of the parameters in the config classes.

        Args:
            grid_parameters (`dict`):
                A dictionary containing the parameters to be tested. There should be at least the key "model_ids" which
                contains a list of model ids to be tested. The other keys should be the name of the config class
                post-fixed with "_kwargs" and the value should be a dictionary containing the parameters to be tested
                for that config class.
            filter_params_func (`callable`, `optional`):
                A function that takes a list of tuples and returns a list of tuples. This function is used to filter
                out the tests that needs for example to be skipped.

        Returns:
            generated_tests (`list`):
                A list of tuples containing the name of the test, the model id, the config class and the config class
                kwargs.
        """
        generated_tests = []
        model_list = grid_parameters["model_ids"]
        task_type = grid_parameters["task_type"] if "task_type" in grid_parameters else None

        for model_id in model_list:
            for key, value in self.items():
                if f"{key}_kwargs" in grid_parameters:
                    peft_configs = []
                    current_peft_config = value[1].copy()
                    for current_key, current_value in grid_parameters[f"{key}_kwargs"].items():
                        for kwarg in current_value:
                            current_peft_config.update({current_key: kwarg})

                            if task_type is not None:
                                current_peft_config.update({"task_type": task_type})

                            peft_configs.append(current_peft_config.copy())
                else:
                    current_peft_config = value[1].copy()
                    if task_type is not None:
                        current_peft_config.update({"task_type": task_type})
                    peft_configs = [current_peft_config]

                for peft_config in peft_configs:
                    generated_tests.append((f"test_{model_id}_{key}", model_id, value[0], peft_config))

        if filter_params_func is not None:
            generated_tests = filter_params_func(generated_tests)

        return generated_tests


@contextmanager
def hub_online_once(model_id: str):
    """Set env[HF_HUB_OFFLINE]=1 (and patch transformers/hugging_face_hub to think that it was always that way)
    for model ids that were seen already so that the hub is not contacted twice for the same model id in said context.
    The cache (`HUB_MODEL_ACCESSES`) also tracks the number of cache hits per model id.

    The reason for doing a context manager and not patching specific methods (e.g., `from_pretrained`) is that there
    are a lot of places (`PeftConfig.from_pretrained`, `get_peft_state_dict`, `load_adapter`, ...) that possibly
    communicate with the hub to download files / check versions / etc.

    Note that using this context manager can cause problems when used in code sections that access different resources.
    Example:

    ```
    def test_something(model_id, config_kwargs):
        with hub_online_once(model_id):
            model = ...from_pretrained(model_id)
            self.do_something_specific_with_model(model)
    ```
    It is assumed that `do_something_specific_with_model` is an absract method that is implement by several tests.
    Imagine the first test simply does `model.generate([1,2,3])`. The second call from another test suite however uses
    a tokenizer (`AutoTokenizer.from_pretrained(model_id)`) - this will fail since the first pass was online but didn't
    use the tokenizer and we're now in offline mode and cannot fetch the tokenizer. The recommended workaround is to
    extend the cache key (`model_id` passed to `hub_online_once` in this case) by something in case the tokenizer is
    used, so that these tests don't share a cache pool with the tests that don't use a tokenizer.
    """
    global HUB_MODEL_ACCESSES
    override = {}

    try:
        if model_id in HUB_MODEL_ACCESSES:
            override = {"HF_HUB_OFFLINE": "1"}
            HUB_MODEL_ACCESSES[model_id] += 1
        else:
            if model_id not in HUB_MODEL_ACCESSES:
                HUB_MODEL_ACCESSES[model_id] = 0
        with (
            # strictly speaking it is not necessary to set the environment variable since most code that's out there
            # is evaluating it at import time and we'd have to reload the modules for it to take effect. It's
            # probably still a good idea to have it if there's some dynamic code that checks it.
            mock.patch.dict(os.environ, override),
            mock.patch("huggingface_hub.constants.HF_HUB_OFFLINE", override.get("HF_HUB_OFFLINE", False) == "1"),
            mock.patch("transformers.utils.hub._is_offline_mode", override.get("HF_HUB_OFFLINE", False) == "1"),
        ):
            yield
    except Exception:
        # in case of an error we have to assume that we didn't access the model properly from the hub
        # for the first time, so the next call cannot be considered cached.
        if HUB_MODEL_ACCESSES.get(model_id) == 0:
            del HUB_MODEL_ACCESSES[model_id]
        raise


PeftTestConfigManager = ClassInstantier(CLASSES_MAPPING)
PeftTestConfigManagerForDecoderModels = ClassInstantier({**CLASSES_MAPPING, **DECODER_MODELS_EXTRA})


class PeftCommonTester:
    r"""
    A large testing suite for testing common functionality of the PEFT models.

    Attributes:
        transformers_class (`transformers.PreTrainedModel`):
            The transformers class that is being tested.
    """

    transformers_class = None

    def prepare_inputs_for_common(self):
        raise NotImplementedError

    def check_modelcard(self, tmp_dirname, model):
        # check the generated README.md
        filename = os.path.join(tmp_dirname, "README.md")
        assert os.path.exists(filename)
        with open(filename, encoding="utf-8") as f:
            readme = f.read()
        metainfo = re.search(r"---\n(.*?)\n---", readme, re.DOTALL).group(1)
        dct = yaml.safe_load(metainfo)
        assert dct["library_name"] == "peft"

        if hasattr(model, "config"):
            assert dct["base_model"] == model.config.to_dict()["_name_or_path"]
        else:  # a custom model
            assert "base_model" not in dct

    def check_config_json(self, tmp_dirname, model):
        # check the generated config.json
        filename = os.path.join(tmp_dirname, "adapter_config.json")
        assert os.path.exists(filename)
        with open(filename, encoding="utf-8") as f:
            config = json.load(f)

        if hasattr(model, "config"):  # custom models don't have a config attribute
            assert config["base_model_name_or_path"] == model.config.to_dict()["_name_or_path"]

    def perturb_trainable_token_weights_if_used(self, model, config_kwargs, adapter_name="default", scale=1.0):
        """TrainableTokensLayer is initialized to be a no-op by default. Since there's currently no way to pass
        `init_weights=False` to the trainable tokens layer when used in conjunction with LoRA, we have to do it like
        this to make sure that it is *not* a no-op (essentially simulating "training" of the adapter).
        """
        raise NotImplementedError

    def _test_model_attr(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            assert hasattr(model, "save_pretrained")
            assert hasattr(model, "from_pretrained")
            assert hasattr(model, "push_to_hub")

    def _test_adapter_name(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config, adapter_name="test-adapter")
            correctly_converted = False
            for n, _ in model.parameters_and_names():
                if "test-adapter" in n:
                    correctly_converted = True
                    break

            assert correctly_converted

    def _test_prepare_for_training(self, model_id, config_cls, config_kwargs):
        raise NotImplementedError

    def _test_load_model_low_cpu_mem_usage(self, model_id, config_cls, config_kwargs):
        raise NotImplementedError

    def _test_save_pretrained(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        # ensure that the weights are randomly initialized
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_lora_weights"] = False

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                if safe_serialization:
                    model.save_pretrained(tmp_dirname)
                else:
                    model.save_pretrained(tmp_dirname, safe_serialization=False)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                with warnings.catch_warnings(record=True) as recs:
                    model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
                    # ensure that there is no warning
                    assert not any("Found missing adapter keys" in str(rec.message) for rec in recs)

                state_dict = get_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_state_dict(model_from_pretrained, unwrap_compiled=True)

                # check if tensors equal
                for key in state_dict.keys():
                    assert mint.allclose(
                        state_dict[key], state_dict_from_pretrained[key]
                    )

                target_adapter_filename = "adapter_model.safetensors" if safe_serialization else "adapter_model.ckpt"

                # check if `adapter_model.safetensors` is present
                assert os.path.exists(os.path.join(tmp_dirname, target_adapter_filename))

                # check if `adapter_config.json` is present
                assert os.path.exists(os.path.join(tmp_dirname, "adapter_config.json"))

                # check if `model.safetensors` is not present
                assert not os.path.exists(os.path.join(tmp_dirname, "model.safetensors"))

                # check if `config.json` is not present
                assert not os.path.exists(os.path.join(tmp_dirname, "config.json"))

                self.check_modelcard(tmp_dirname, model)
                self.check_config_json(tmp_dirname, model)

    def _test_save_pretrained_selected_adapters(self, model_id, config_cls, config_kwargs, safe_serialization=True):
        # ensure that the weights are randomly initialized
        if issubclass(config_cls, LoraConfig):
            config_kwargs = config_kwargs.copy()
            config_kwargs["init_lora_weights"] = False

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            new_adapter_config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model.add_adapter("new_adapter", new_adapter_config)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                if safe_serialization:
                    model.save_pretrained(tmp_dirname)
                else:
                    model.save_pretrained(tmp_dirname, safe_serialization=False)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

                new_adapter_dir = os.path.join(tmp_dirname, "new_adapter")
                model_from_pretrained.load_adapter(new_adapter_dir, "new_adapter")

                state_dict = get_state_dict(model, unwrap_compiled=True)
                state_dict_from_pretrained = get_state_dict(model_from_pretrained, unwrap_compiled=True)

                # check if same keys
                assert state_dict.keys() == state_dict_from_pretrained.keys()

                # check if tensors equal
                for key in state_dict.keys():
                    assert mint.allclose(
                        state_dict[key], state_dict_from_pretrained[key]
                    )

                target_adapter_filename = "adapter_model.safetensors" if safe_serialization else "adapter_model.ckpt"

                # check if `adapter_model.safetensors` is present
                assert os.path.exists(os.path.join(tmp_dirname, target_adapter_filename))
                assert os.path.exists(os.path.join(new_adapter_dir, target_adapter_filename))

                # check if `adapter_config.json` is present
                assert os.path.exists(os.path.join(tmp_dirname, "adapter_config.json"))
                assert os.path.exists(os.path.join(new_adapter_dir, "adapter_config.json"))

                # check if `model.safetensors` is not present
                assert not os.path.exists(os.path.join(tmp_dirname, "model.safetensors"))
                assert not os.path.exists(os.path.join(new_adapter_dir, "model.safetensors"))

                # check if `config.json` is not present
                assert not os.path.exists(os.path.join(tmp_dirname, "config.json"))
                assert not os.path.exists(os.path.join(new_adapter_dir, "config.json"))

                self.check_modelcard(tmp_dirname, model)
                self.check_config_json(tmp_dirname, model)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname, selected_adapters=["default"])

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)

                assert "default" in model_from_pretrained.peft_config.keys()
                assert "new_adapter" not in model_from_pretrained.peft_config.keys()

    def _test_from_pretrained_config_construction(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
            model = get_peft_model(model, config)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                model_from_pretrained = PeftModel.from_pretrained(
                    model_from_pretrained, tmp_dirname, is_trainable=False, config=config
                )

                assert model_from_pretrained.peft_config["default"].inference_mode
                assert model_from_pretrained.peft_config["default"] is config

    def _test_load_multiple_adapters(self, model_id, config_cls, config_kwargs):
        # just ensure that this works and raises no error
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)
                del model

                model = self.transformers_class.from_pretrained(model_id)
                model = PeftModel.from_pretrained(model, tmp_dirname)
                load_result1 = model.load_adapter(tmp_dirname, adapter_name="other")
                load_result2 = model.load_adapter(tmp_dirname, adapter_name="yet-another")

                # VBLoRA uses a shared "vblora_vector_bank" across all layers, causing it to appear
                # in the missing keys list, which leads to failed test cases. So
                # skipping the missing keys check for VBLoRA.
                if config.peft_type != "VBLORA":
                    assert load_result1.missing_keys == []
                    assert load_result2.missing_keys == []

    def _test_merge_layers_fp16(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig,):
            # Merge layers only supported for LoRA and IA³
            return pytest.skip(f"Test not applicable for {config_cls}")

        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id, mindspore_dtype=ms.float16)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)
            model = model.to(dtype=ms.float16)

            model.set_train(False)

            # This should simply work
            _ = model.merge_and_unload()

    def _test_merge_layers_nan(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig,):
            # Merge layers only supported for LoRA and IA³
            return
        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model = get_peft_model(model, config)

            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            dummy_input = self.prepare_inputs_for_testing()

            model.set_train(False)

            # This should work
            logits_unmerged = model(**dummy_input)[0]

            model = model.merge_and_unload()
            logits_merged = model(**dummy_input)[0]

            assert mint.allclose(logits_unmerged, logits_merged, atol=1e-3, rtol=1e-3)

            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            for name, module in model.parameters_and_names():
                if (
                    "lora_A" in name
                    or "ia3" in name
                    or "lora_E" in name
                    or "lora_B" in name
                    or "vera_lambda" in name
                    or "fourierft_spectrum" in name
                ):
                    module[0] = ms.tensor(float("nan"))

            with pytest.raises(
                ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
            ):
                model = model.merge_and_unload(safe_merge=True)

            for name, module in model.named_parameters():
                if (
                    "lora_A" in name
                    or "ia3" in name
                    or "lora_E" in name
                    or "lora_B" in name
                    or "vera_lambda" in name
                    or "fourierft_spectrum" in name
                ):
                    module.data[0] = ms.tensor(float("inf"))

            with pytest.raises(
                ValueError, match="NaNs detected in the merged weights. The adapter default seems to be broken"
            ):
                model = model.merge_and_unload(safe_merge=True)

    def _test_merge_layers(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        if ("gpt2" in model_id.lower()) and (config_cls != LoraConfig):
            self.skipTest("Merging GPT2 adapters not supported for IA³ (yet)")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model = get_peft_model(model, config)

            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            dummy_input = self.prepare_inputs_for_testing()
            model.set_train(False)
            logits = model(**dummy_input)[0]

            model.merge_adapter()
            logits_merged = model(**dummy_input)[0]
            model.unmerge_adapter()
            logits_unmerged = model(**dummy_input)[0]

            model = model.merge_and_unload()

            # check that PEFT layers are completely removed
            assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
            logits_merged_unloaded = model(**dummy_input)[0]

            conv_ids = ["Conv2d", "Conv3d", "Conv2d2"]
            atol, rtol = 1e-4, 1e-4
            if (config.peft_type in {"IA3", "LORA"}) and (model_id in conv_ids):
                # for some reason, the Conv introduces a larger error
                atol, rtol = 0.3, 0.01
            assert mint.allclose(logits, logits_merged, atol=atol, rtol=rtol)
            assert mint.allclose(logits, logits_unmerged, atol=atol, rtol=rtol)
            assert mint.allclose(logits, logits_merged_unloaded, atol=atol, rtol=rtol)

            # For this test to work, weights should not be initialized to identity transform (e.g.
            # init_lora_weights should be False).
            transformers_model = self.transformers_class.from_pretrained(model_id)
            logits_transformers = transformers_model(**dummy_input)[0]
            assert not mint.allclose(logits_merged, logits_transformers, atol=1e-10, rtol=1e-10)

            # test that the logits are identical after a save-load-roundtrip
            if hasattr(model, "save_pretrained"):
                # model is a transformers model
                tmp_dirname = tempfile.mkdtemp()
                # note: not using the context manager here because it fails on Windows CI for some reason
                try:
                    model.save_pretrained(tmp_dirname)
                    model_from_pretrained = self.transformers_class.from_pretrained(tmp_dirname)
                finally:
                    try:
                        shutil.rmtree(tmp_dirname)
                    except PermissionError:
                        # windows error
                        pass
            else:
                # model is not a transformers model
                model_from_pretrained = pickle.loads(pickle.dumps(model))

            logits_merged_from_pretrained = model_from_pretrained(**dummy_input)[0]
            assert mint.allclose(logits_merged, logits_merged_from_pretrained, atol=atol, rtol=rtol)

    def _test_merge_layers_multi(self, model_id, config_cls, config_kwargs):
        supported_peft_types = [
            PeftType.LORA,
            PeftType.LOHA,
            PeftType.LOKR,
            PeftType.IA3,
            PeftType.OFT,
            PeftType.BOFT,
            PeftType.HRA,
            PeftType.BONE,
        ]

        if config_kwargs.get("trainable_token_indices", None) is not None:
            self.skipTest(
                "Merging two adapters with trainable tokens is tested elsewhere since adapters with "
                "the same token indices cannot be merged."
            )

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        if config.peft_type not in supported_peft_types:
            return

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config)

            dummy_input = self.prepare_inputs_for_testing()
            model.set_train(False)

            with ms._no_grad():
                logits_adapter_1 = model(**dummy_input)[0]

            model.add_adapter("adapter-2", config)
            model.set_adapter("adapter-2")
            model.set_train(False)

            with ms._no_grad():
                logits_adapter_2 = model(**dummy_input)[0]

            assert not mint.allclose(logits_adapter_1, logits_adapter_2, atol=1e-3, rtol=1e-3)

            model.set_adapter("default")

            with ms._no_grad():
                logits_adapter_1_after_set = model(**dummy_input)[0]

            assert mint.allclose(logits_adapter_1_after_set, logits_adapter_1, atol=1e-3, rtol=1e-3)

            model_copy = copy.deepcopy(model)
            model_copy_2 = copy.deepcopy(model)
            model_merged_all = model.merge_and_unload(adapter_names=["adapter-2", "default"])

            with ms._no_grad():
                logits_merged_all = model_merged_all(**dummy_input)[0]

            assert not mint.allclose(logits_merged_all, logits_adapter_2, atol=1e-3, rtol=1e-3)
            assert not mint.allclose(logits_merged_all, logits_adapter_1, atol=1e-3, rtol=1e-3)

            model_merged_adapter_2 = model_copy.merge_and_unload(adapter_names=["adapter-2"])

            with ms._no_grad():
                logits_merged_adapter_2 = model_merged_adapter_2(**dummy_input)[0]

            assert mint.allclose(logits_merged_adapter_2, logits_adapter_2, atol=1e-3, rtol=1e-3)

            model_merged_adapter_default = model_copy_2.merge_and_unload(adapter_names=["default"])

            with ms._no_grad():
                logits_merged_adapter_default = model_merged_adapter_default(**dummy_input)[0]

            assert mint.allclose(logits_merged_adapter_default, logits_adapter_1, atol=1e-3, rtol=1e-3)

    def _test_merge_layers_is_idempotent(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)
            model.set_train(False)
            ms.manual_seed(0)
            model.merge_adapter()
            logits_0 = model(**self.prepare_inputs_for_testing())[0]

            # merging again should not change anything
            # also check warning:
            with pytest.warns(UserWarning, match="All adapters are already merged, nothing to do"):
                model.merge_adapter()
            logits_1 = model(**self.prepare_inputs_for_testing())[0]

            assert mint.allclose(logits_0, logits_1, atol=1e-6, rtol=1e-6)

    def _test_safe_merge(self, model_id, config_cls, config_kwargs):
        ms.manual_seed(0)
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = model.set_train(False)

            inputs = self.prepare_inputs_for_testing()
            logits_base = model(**inputs)[0]

            model = get_peft_model(model, config).set_train(False)
            logits_peft = model(**inputs)[0]

            atol, rtol = 1e-6, 1e-6  # default
            # Initializing with LN tuning cannot be configured to change the outputs (unlike init_lora_weights=False)
            # if not issubclass(config_cls, LNTuningConfig):
            # sanity check that the logits are different
            assert not mint.allclose(logits_base, logits_peft, atol=atol, rtol=rtol)

            model_unloaded = model.merge_and_unload(safe_merge=True)
            logits_unloaded = model_unloaded(**inputs)[0]

            conv_ids = ["Conv2d", "Conv3d", "Conv2d2"]
            if issubclass(config_cls, (LoraConfig,)) and model_id in conv_ids:  # more instability with Conv
                atol, rtol = 1e-3, 1e-3

            # check that the logits are the same after unloading
            assert mint.allclose(logits_peft, logits_unloaded, atol=atol, rtol=rtol)

            # Ensure that serializing with safetensors works, there was an error when weights were not contiguous
            with tempfile.TemporaryDirectory() as tmp_dirname:
                # serializing with torch.save works
                ms.save_checkpoint(model_unloaded.state_dict(), os.path.join(tmp_dirname, "model.ckpt"))

                # serializing with safetensors works
                save_file(model_unloaded.state_dict(), os.path.join(tmp_dirname, "model.safetensors"))

    def _test_generate(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()

            # check if `generate` works
            _ = model.generate(**inputs)

    def _test_generate_pos_args(self, model_id, config_cls, config_kwargs, raises_err: bool):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()
            if raises_err:
                with pytest.raises(TypeError):
                    # check if `generate` raises an error if positional arguments are passed
                    _ = model.generate(inputs["input_ids"])
            else:
                # check if `generate` works if positional arguments are passed
                _ = model.generate(inputs["input_ids"])

    def _test_generate_half_prec(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig,):
            return pytest.skip(f"Test not applicable for {config_cls}")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id, mindspore_dtype=ms.bfloat16)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            input_ids = ms.tensor([[1, 1, 1], [2, 1, 2]])
            attention_mask = ms.tensor([[1, 1, 1], [1, 0, 1]])

            # check if `generate` works
            _ = model.generate(input_ids=input_ids, attention_mask=attention_mask)

    def _test_training(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()

            # check if `training` works
            output = model(**inputs)[0]
            loss = output.sum()
            loss.backward()
            parameter_prefix = model.prefix
            for n, param in model.named_parameters():
                if (parameter_prefix in n) or ("modules_to_save" in n) or ("token_adapter.trainable_tokens" in n):
                    assert param.grad is not None
                else:
                    assert param.grad is None

    def _test_inference_safetensors(self, model_id, config_cls, config_kwargs):
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()

            # check if `training` works
            output = model(**inputs)[0]
            logits = output[0]

            loss = output.sum()
            loss.backward()

            # set to eval mode, since things like dropout can affect the output otherwise
            model.set_train(False)
            logits = model(**inputs)[0][0]

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname, safe_serialization=True)
                assert "adapter_model.safetensors" in os.listdir(tmp_dirname)
                assert "adapter_model.bin" not in os.listdir(tmp_dirname)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(
                    self.torch_device
                )

                logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
                assert mint.allclose(logits, logits_from_pretrained, atol=1e-4, rtol=1e-4)

    def _test_training_layer_indexing(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig,):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            layers_to_transform=[0],
            **config_kwargs,
        )
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()

            # check if `training` works
            output = model(**inputs)[0]
            logits = output[0]

            loss = output.sum()
            loss.backward()

            has_trainable_tokens = config_kwargs.get("trainable_token_indices", None) is not None
            nb_trainable = 0

            for n, param in model.named_parameters():
                if "lora" in n or (has_trainable_tokens and "trainable_tokens" in n):
                    assert param.grad is not None
                    nb_trainable += 1
                else:
                    assert param.grad is None

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname).to(
                    self.torch_device
                )

                logits_from_pretrained = model_from_pretrained(**inputs)[0][0]
                assert mint.allclose(logits, logits_from_pretrained, atol=1e-4, rtol=1e-4)

            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)
            nb_trainable_all = 0

            for n, param in model.named_parameters():
                if "lora" in n or (has_trainable_tokens and "trainable_tokens" in n):
                    nb_trainable_all += 1

            assert nb_trainable < nb_trainable_all

    def _test_training_gradient_checkpointing(self, model_id, config_cls, config_kwargs):
        if config_cls == PrefixTuningConfig:
            return pytest.skip(f"Test not applicable for {config_cls}")

        if (config_cls == AdaLoraConfig) and ("roberta" in model_id.lower()):
            # TODO: no gradients on the "dense" layer, other layers work, not sure why
            self.skipTest("AdaLora with RoBERTa does not work correctly")

        if (config_cls == OFTConfig) and ("deberta" in model_id.lower()):
            # TODO: no gradients on the "dense" layer, other layers work, not sure why
            self.skipTest("OFT with Deberta does not work correctly")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)

            if not getattr(model, "supports_gradient_checkpointing", False):
                return pytest.skip(f"Model {model_id} does not support gradient checkpointing")

            model.gradient_checkpointing_enable()

            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()

            # check if `training` works
            output = model(**inputs)[0]

            loss = output.sum()
            loss.backward()

            for n, param in model.named_parameters():
                if "prompt_encoder." in n:  # prompt tuning methods
                    if not issubclass(config_cls, CPTConfig):
                        assert param.grad is not None
                    elif (
                        "delta_embedding" in n
                    ):  # delta_embedding is the embedding that should be updated with grads in CPT
                        assert param.grad is not None
                elif hasattr(model, "prefix") and (model.prefix in n):  # non-prompt tuning methods
                    assert param.grad is not None
                elif "trainable_tokens_" in n:  # trainable tokens layer
                    assert param.grad is not None
                else:
                    assert param.grad is None

    def _test_peft_model_device_map(self, model_id, config_cls, config_kwargs):
        if config_cls not in (LoraConfig, VBLoRAConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)

            model = get_peft_model(model, config)

            with tempfile.TemporaryDirectory() as tmp_dirname:
                model.save_pretrained(tmp_dirname)

                model_from_pretrained = self.transformers_class.from_pretrained(model_id)
                _ = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname, device_map={"": "cpu"}).to(
                    self.torch_device
                )

    def _test_training_prompt_learning_tasks(self, model_id, config_cls, config_kwargs):
        if not issubclass(config_cls, PromptLearningConfig):
            return pytest.skip(f"Test not applicable for {config_cls}")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config)

            inputs = self.prepare_inputs_for_testing()

            # check if `training` works
            output = model(**inputs)[0]
            loss = output.sum()
            loss.backward()

            if issubclass(config_cls, CPTConfig):
                parameters = []
                for name, param in model.prompt_encoder.named_parameters():
                    if name != "default.embedding.weight":
                        parameters.append(param)
            else:
                parameters = model.prompt_encoder.parameters()

            # check that prompt encoder has grads
            for param in parameters:
                assert param.grad is not None

    def _test_delete_adapter(self, model_id, config_cls, config_kwargs):
        supported_peft_types = [
            PeftType.LORA,
            PeftType.LOHA,
            PeftType.LOKR,
            PeftType.IA3,
            PeftType.OFT,
            PeftType.BOFT,
            PeftType.VERA,
            PeftType.FOURIERFT,
            PeftType.HRA,
            PeftType.VBLORA,
            PeftType.BONE,
        ]
        # IA3 does not support deleting adapters yet, but it just needs to be added
        # AdaLora does not support multiple adapters
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.peft_type not in supported_peft_types:
            return pytest.skip(f"Test not applicable for {config.peft_type}")

        if hasattr(config, "trainable_token_indices"):
            return pytest.skip("This is currently not supported. See https://github.com/huggingface/peft/issues/2381")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            adapter_to_delete = "delete_me"
            model = get_peft_model(model, config)
            model.add_adapter(adapter_to_delete, config)
            model.set_adapter(adapter_to_delete)
            model.delete_adapter(adapter_to_delete)
            assert adapter_to_delete not in model.peft_config
            assert model.active_adapters == ["default"]

            key_list = [key for key, _ in model.named_modules()]
            for key in key_list:
                _, target, _ = _get_submodules(model, key)
                attributes_to_check = getattr(target, "adapter_layer_names", []) + getattr(
                    target, "other_param_names", []
                )
                for attr in attributes_to_check:
                    assert adapter_to_delete not in getattr(target, attr)

            # check that we can also delete the last remaining adapter
            model.delete_adapter("default")
            assert "default" not in model.peft_config
            assert model.active_adapters == []

            input = self.prepare_inputs_for_testing()
            # note: we cannot call model(**input) because PeftModel always expects there to be at least one adapter
            model.base_model(**input)  # should not raise an error

    def _test_delete_inactive_adapter(self, model_id, config_cls, config_kwargs):
        # same as test_delete_adapter, but this time an inactive adapter is deleted
        supported_peft_types = [
            PeftType.LORA,
            PeftType.LOHA,
            PeftType.LOKR,
            PeftType.IA3,
            PeftType.OFT,
            PeftType.BOFT,
            PeftType.FOURIERFT,
            PeftType.HRA,
            PeftType.VBLORA,
            PeftType.BONE,
        ]
        # IA3 does not support deleting adapters yet, but it just needs to be added
        # AdaLora does not support multiple adapters
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        if config.peft_type not in supported_peft_types:
            return pytest.skip(f"Test not applicable for {config.peft_type}")

        if hasattr(config, "trainable_token_indices"):
            return pytest.skip("This is currently not supported. See https://github.com/huggingface/peft/issues/2381")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            adapter_to_delete = "delete_me"
            model = get_peft_model(model, config)
            model.add_adapter(adapter_to_delete, config)
            # "delete_me" is added but not activated
                        model.delete_adapter(adapter_to_delete)
            assert adapter_to_delete not in model.peft_config
            assert model.active_adapters == ["default"]

            key_list = [key for key, _ in model.named_modules()]
            for key in key_list:
                _, target, _ = _get_submodules(model, key)
                attributes_to_check = getattr(target, "adapter_layer_names", []) + getattr(
                    target, "other_param_names", []
                )
                for attr in attributes_to_check:
                    assert adapter_to_delete not in getattr(target, attr)

            # check that we can also delete the last remaining adapter
            model.delete_adapter("default")
            assert "default" not in model.peft_config
            assert model.active_adapters == []

            input = self.prepare_inputs_for_testing()
            # note: we cannot call model(**input) because PeftModel always expects there to be at least one adapter
            model.base_model(**input)  # should not raise an error

    def _test_delete_unknown_adapter_raises(self, model_id, config_cls, config_kwargs):
        # Check that we get a nice error message when trying to delete an adapter that does not exist.
        config = config_cls(base_model_name_or_path=model_id, **config_kwargs)
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            adapter_to_delete = "delete_me"
            model = get_peft_model(model, config)

            msg = "Adapter unknown-adapter does not exist"
            with pytest.raises(ValueError, match=msg):
                model.delete_adapter("unknown-adapter")

    def _test_unload_adapter(self, model_id, config_cls, config_kwargs):
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
        num_params_base = len(model.state_dict())

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)

        if config.peft_type not in (
            "LORA",
            "ADALORA",
            "IA3",
            "BOFT",
            "OFT",
            "VERA",
            "FOURIERFT",
            "HRA",
            "VBLORA",
            "BONE",
        ):
            with pytest.raises(AttributeError):
                model = model.unload()
        else:
            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            dummy_input = self.prepare_inputs_for_testing()
            logits_with_adapter = model(**dummy_input)[0]

            with hub_online_once(model_id):
                transformers_model = self.transformers_class.from_pretrained(model_id)
                logits_transformers = transformers_model(**dummy_input)[0]

                model.set_train(False)
                model = model.unload()
                logits_unload = model(**dummy_input)[0]
                num_params_unloaded = len(model.state_dict())

                # check that PEFT layers are completely removed
                assert not any(isinstance(module, BaseTunerLayer) for module in model.modules())
                assert not mint.allclose(logits_with_adapter, logits_unload, atol=1e-10, rtol=1e-10)
                assert mint.allclose(logits_transformers, logits_unload, atol=1e-4, rtol=1e-4)
                assert num_params_base == num_params_unloaded

    def _test_weighted_combination_of_adapters_lora(self, model, config, adapter_list, weight_list):
        model.add_adapter(adapter_list[1], config)
        model.add_adapter(adapter_list[2], replace(config, r=20))

        # test re-weighting single adapter
        model.add_weighted_adapter([adapter_list[0]], [weight_list[0]], "single_adapter_reweighting")

        # test svd re-weighting with multiple adapters
        model.add_weighted_adapter(adapter_list[1:], weight_list[1:], "multi_adapter_svd_reweighting")

        # test ties_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_ties_svd_reweighting",
            combination_type="ties_svd",
            density=0.5,
        )

        # test dare_linear_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_dare_linear_svd_reweighting",
            combination_type="dare_linear_svd",
            density=0.5,
        )

        # test dare_ties_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_dare_ties_svd_reweighting",
            combination_type="dare_ties_svd",
            density=0.5,
        )

        # test magnitude_prune_svd re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:],
            weight_list[1:],
            "multi_adapter_magnitude_prune_svd_reweighting",
            combination_type="magnitude_prune_svd",
            density=0.5,
        )

        # test cat re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[1:], weight_list[1:], "multi_adapter_cat_reweighting", combination_type="cat"
        )

        # test linear re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2], weight_list[:2], "multi_adapter_linear_reweighting", combination_type="linear"
        )

        # test ties re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2], weight_list[:2], "multi_adapter_ties_reweighting", combination_type="ties", density=0.5
        )

        # test dare_linear re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2],
            weight_list[:2],
            "multi_adapter_dare_linear_reweighting",
            combination_type="dare_linear",
            density=0.5,
        )

        # test dare_ties re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2],
            weight_list[:2],
            "multi_adapter_dare_ties_reweighting",
            combination_type="dare_ties",
            density=0.5,
        )

        # test magnitude_prune re-weighting with multiple adapters
        model.add_weighted_adapter(
            adapter_list[:2],
            weight_list[:2],
            "multi_adapter_magnitude_prune_reweighting",
            combination_type="magnitude_prune",
            density=0.5,
        )

        # test linear re-weighting with multiple adapters with only first adapter having non zero weight
        model.add_weighted_adapter(
            adapter_list[:2],
            [weight_list[0], 0],
            "multi_adapter_linear_reweighting_single_enabled",
            combination_type="linear",
        )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_linear_reweighting_uneven_r",
                combination_type="linear",
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_ties_reweighting_uneven_r",
                combination_type="ties",
                density=0.5,
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_dare_linear_reweighting_uneven_r",
                combination_type="dare_linear",
                density=0.5,
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_dare_ties_reweighting_uneven_r",
                combination_type="dare_ties",
                density=0.5,
            )

        with pytest.raises(ValueError):
            model.add_weighted_adapter(
                adapter_list[1:],
                weight_list[1:],
                "multi_adapter_magnitude_prune_reweighting_uneven_r",
                combination_type="magnitude_prune",
                density=0.5,
            )

        new_adapters = [
            "single_adapter_reweighting",
            "multi_adapter_svd_reweighting",
            "multi_adapter_ties_svd_reweighting",
            "multi_adapter_dare_linear_svd_reweighting",
            "multi_adapter_dare_ties_svd_reweighting",
            "multi_adapter_magnitude_prune_svd_reweighting",
            "multi_adapter_cat_reweighting",
            "multi_adapter_linear_reweighting",
            "multi_adapter_linear_reweighting_single_enabled",
            "multi_adapter_ties_reweighting",
            "multi_adapter_dare_linear_reweighting",
            "multi_adapter_dare_ties_reweighting",
            "multi_adapter_magnitude_prune_reweighting",
        ]
        for new_adapter in new_adapters:
            assert new_adapter in model.peft_config

        key_list = [key for key, _ in model.named_modules()]
        for key in key_list:
            _, target, _ = _get_submodules(model, key)
            if isinstance(target, LoraLayer):
                for adapter_name in new_adapters:
                    if "single" in adapter_name:
                        new_delta_weight = target.get_delta_weight(adapter_name)
                        weighted_original_delta_weights = target.get_delta_weight(adapter_list[0]) * weight_list[0]
                        assert mint.allclose(new_delta_weight, weighted_original_delta_weights, atol=1e-4, rtol=1e-4)
                    elif "svd" in adapter_name:
                        assert target.r[adapter_name] == 20
                    elif "linear" in adapter_name:
                        assert target.r[adapter_name] == 8
                    elif "cat" in adapter_name:
                        assert target.r[adapter_name] == 28

        dummy_input = self.prepare_inputs_for_testing()
        model.set_train(False)
        for adapter_name in new_adapters:
            # ensuring new adapters pass the forward loop
            model.set_adapter(adapter_name)
            assert model.active_adapter == adapter_name
            assert model.active_adapters == [adapter_name]
            model(**dummy_input)[0]

    def _test_weighted_combination_of_adapters_ia3(self, model, config, adapter_list, weight_list):
        model.add_adapter(adapter_list[1], config)
        model.add_adapter(adapter_list[2], config)

        # test re-weighting single adapter
        model.add_weighted_adapter([adapter_list[0]], [weight_list[0]], "single_adapter_reweighting")

        # test re-weighting with multiple adapters
        model.add_weighted_adapter(adapter_list[1:], weight_list[1:], "multi_adapter_reweighting")

        new_adapters = [
            "single_adapter_reweighting",
            "multi_adapter_reweighting",
        ]
        for new_adapter in new_adapters:
            assert new_adapter in model.peft_config

        dummy_input = self.prepare_inputs_for_testing()
        model.set_train(False)
        for adapter_name in new_adapters:
            # ensuring new adapters pass the forward loop
            model.set_adapter(adapter_name)
            assert model.active_adapter == adapter_name
            assert model.active_adapters == [adapter_name]
            model(**dummy_input)[0]

    def _test_weighted_combination_of_adapters(self, model_id, config_cls, config_kwargs):
        if issubclass(config_cls, AdaLoraConfig):
            # AdaLora does not support adding more than 1 adapter
            return pytest.skip(f"Test not applicable for {config_cls}")
        if model_id.endswith("qwen2"):
            # Qwen2 fails with weighted adapter combinations using SVD
            return pytest.skip(f"Test does not work with model {model_id}")

        adapter_list = ["adapter1", "adapter_2", "adapter_3"]
        weight_list = [0.5, 1.5, 1.5]
        # Initialize the config
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )

        if not isinstance(config, (LoraConfig, IA3Config)):
            # This test is only applicable for Lora and IA3 configs
            return pytest.skip(f"Test not applicable for {config}")

        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config, adapter_list[0])

            if isinstance(config, LoraConfig):
                self._test_weighted_combination_of_adapters_lora(model, config, adapter_list, weight_list)
            elif isinstance(config, IA3Config):
                self._test_weighted_combination_of_adapters_ia3(model, config, adapter_list, weight_list)
            else:
                pytest.skip(f"Test not applicable for {config}")

    def _test_disable_adapter(self, model_id, config_cls, config_kwargs):
        task_type = config_kwargs.get("task_type")
        if (task_type == "SEQ_2_SEQ_LM") and (config_cls in (PromptTuningConfig, PromptEncoderConfig)):
            self.skipTest("Seq2Seq + prompt tuning/prompt encoder does not work with disabling adapters")

        def get_output(model):
            # helper function that works with different model types
            ms.manual_seed(0)

            if hasattr(model, "generate"):
                # let's check the scores, not the output ids, since the latter can easily be identical even if the
                # weights are slightly changed
                output = model.generate(**input, return_dict_in_generate=True, output_scores=True).scores[0]
                # take element 0, as output is a tuple
            else:
                output = model(**input)

            if hasattr(output, "images"):  # for SD
                import numpy as np

                img = output.images[0]
                return torch.from_numpy(np.array(img))

            return output

        # initialize model
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)

            # output from BASE MODEL
            input = self.prepare_inputs_for_testing()
            output_before = get_output(model)

            # output from PEFT MODEL
            if hasattr(self, "instantiate_sd_peft"):
                # SD models are instantiated differently
                peft_model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)
            else:
                config = config_cls(
                    base_model_name_or_path=model_id,
                    **config_kwargs,
                )
                peft_model = get_peft_model(model, config)

            # trainable_token_indices doesn't have support for `init_weights` so we have to do this manually
            self.perturb_trainable_token_weights_if_used(model, config_kwargs)

            output_peft = get_output(peft_model)

            # first check trivial case is not true that peft does not affect the output; for this to work, init_weight
            # must be False (if the config supports it)
            if isinstance(peft_model, StableDiffusionPipeline):
                # for SD, check that most pixels have different values
                assert (output_before != output_peft).float().mean() > 0.8
            else:
                assert not mint.allclose(output_before, output_peft)

            # output with DISABLED ADAPTER
            if isinstance(peft_model, StableDiffusionPipeline):
                with peft_model.unet.disable_adapter():
                    with peft_model.text_encoder.disable_adapter():
                        output_peft_disabled = get_output(peft_model)
                # for SD, very rarely, a pixel can differ
                assert (output_before != output_peft_disabled).float().mean() < 1e-4
            else:
                with peft_model.disable_adapter():
                    output_peft_disabled = get_output(peft_model)
                assert mint.allclose(output_before, output_peft_disabled, atol=1e-6, rtol=1e-6)

                # after leaving the disable_adapter context, the output should be the same as with enabled adapter again
                # see #1501
                output_peft_after_disabled = get_output(peft_model)
                assert mint.allclose(output_peft, output_peft_after_disabled, atol=1e-6, rtol=1e-6)

            # TODO: add tests to check if disabling adapters works after calling merge_adapter

    def _test_adding_multiple_adapters_with_bias_raises(self, model_id, config_cls, config_kwargs):
        # When trying to add multiple adapters with bias in Lora, AdaLora or BOFTConfig, an error should be
        # raised. Also, the peft model should not be left in a half-initialized state.
        if not issubclass(config_cls, (LoraConfig, AdaLoraConfig, BOFTConfig)):
            return pytest.skip(f"Test not applicable for {config_cls}")

        with hub_online_once(model_id):
            config_kwargs = config_kwargs.copy()
            config_kwargs["bias"] = "all"
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )

            model = self.transformers_class.from_pretrained(model_id)
            model = get_peft_model(model, config, "adapter0")

            if config_cls == LoraConfig or config_cls == AdaLoraConfig:
                with pytest.raises(ValueError):
                    model.add_adapter("adapter1", replace(config, r=20))

            if config_cls == BOFTConfig:
                with pytest.raises(ValueError):
                    model.add_adapter("adapter1", replace(config, boft_block_num=1, boft_block_size=0))

            # (superficial) test that the model is not left in a half-initialized state when adding an adapter fails
            assert "adapter1" not in model.peft_config
            assert "adapter1" not in model.base_model.peft_config

    def _test_passing_input_embeds_works(self, test_name, model_id, config_cls, config_kwargs):
        # https://github.com/huggingface/peft/issues/727
        with hub_online_once(model_id):
            model = self.transformers_class.from_pretrained(model_id)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            model = get_peft_model(model, config, adapter_name="test-adapter")
            dummy_input = self.prepare_inputs_for_testing()
            inputs_embeds = model.get_input_embeddings()(dummy_input["input_ids"])
            # just check that no error is raised
            model.forward(inputs_embeds=inputs_embeds)
