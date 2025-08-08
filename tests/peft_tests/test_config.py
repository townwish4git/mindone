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
import tempfile

import pytest

from mindone.peft import LoraConfig, PeftConfig, PeftType, TaskType

PEFT_MODELS_TO_TEST = [("lewtun/tiny-random-OPTForCausalLM-delta", "v1")]

# Config classes and their mandatory parameters
ALL_CONFIG_CLASSES = ((LoraConfig, {}),)


class TestPeftConfig:
    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_methods(self, config_class, mandatory_kwargs):
        r"""
        Test if all configs have the expected methods. Here we test
        - to_dict
        - save_pretrained
        - from_pretrained
        - from_json_file
        """
        # test if all configs have the expected methods
        config = config_class(**mandatory_kwargs)
        assert hasattr(config, "to_dict")
        assert hasattr(config, "save_pretrained")
        assert hasattr(config, "from_pretrained")
        assert hasattr(config, "from_json_file")

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    @pytest.mark.parametrize("valid_task_type", list(TaskType) + [None])
    def test_valid_task_type(self, config_class, mandatory_kwargs, valid_task_type):
        r"""
        Test if all configs work correctly for all valid task types
        """
        config_class(task_type=valid_task_type, **mandatory_kwargs)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_invalid_task_type(self, config_class, mandatory_kwargs):
        r"""
        Test if all configs correctly raise the defined error message for invalid task types.
        """
        invalid_task_type = "invalid-task-type"
        with pytest.raises(
            ValueError,
            match=f"Invalid task type: '{invalid_task_type}'. Must be one of the following task types: {', '.join(TaskType)}.",
        ):
            config_class(task_type=invalid_task_type, **mandatory_kwargs)

    def test_from_peft_type(self):
        r"""
        Test if the config is correctly loaded using:
        - from_peft_type
        """
        from mindone.peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        for peft_type in (PeftType.LORA,):  # only LoRA is supported for now.
            expected_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]
            mandatory_config_kwargs = {}

            config = PeftConfig.from_peft_type(peft_type=peft_type, **mandatory_config_kwargs)
            assert type(config) is expected_cls

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_from_pretrained(self, config_class, mandatory_kwargs):
        r"""
        Test if the config is correctly loaded using:
        - from_pretrained
        """
        for model_name, revision in PEFT_MODELS_TO_TEST:
            # Test we can load config from delta
            config_class.from_pretrained(model_name, revision=revision)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_save_pretrained(self, config_class, mandatory_kwargs):
        r"""
        Test if the config is correctly saved and loaded using
        - save_pretrained
        """
        config = config_class(**mandatory_kwargs)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            assert config.to_dict() == config_from_pretrained.to_dict()

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_from_json_file(self, config_class, mandatory_kwargs):
        config = config_class(**mandatory_kwargs)
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_path = os.path.join(tmp_dirname, "adapter_config.json")
            config_from_json = config_class.from_json_file(config_path)
            assert config.to_dict() == config_from_json

            # Also test with a runtime_config entry -- they should be ignored, even if they
            # were accidentally saved to disk
            config_from_json["runtime_config"] = {"ephemeral_gpu_offload": True}
            json.dump(config_from_json, open(config_path, "w"))

            config_from_json = config_class.from_json_file(config_path)
            assert config.to_dict() == config_from_json

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_to_dict(self, config_class, mandatory_kwargs):
        r"""
        Test if the config can be correctly converted to a dict using:
        - to_dict
        """
        config = config_class(**mandatory_kwargs)
        assert isinstance(config.to_dict(), dict)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_from_pretrained_cache_dir(self, config_class, mandatory_kwargs):
        r"""
        Test if the config is correctly loaded with extra kwargs
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            for model_name, revision in PEFT_MODELS_TO_TEST:
                # Test we can load config from delta
                config_class.from_pretrained(model_name, revision=revision, cache_dir=tmp_dirname)

    def test_from_pretrained_cache_dir_remote(self):
        r"""
        Test if the config is correctly loaded with a checkpoint from the hub
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            PeftConfig.from_pretrained("ybelkada/test-st-lora", cache_dir=tmp_dirname)
            assert "models--ybelkada--test-st-lora" in os.listdir(tmp_dirname)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_save_pretrained_with_runtime_config(self, config_class, mandatory_kwargs):
        r"""
        Test if the config correctly removes runtime config when saving
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            for model_name, revision in PEFT_MODELS_TO_TEST:
                cfg = config_class.from_pretrained(model_name, revision=revision)
                # NOTE: cfg is always a LoraConfig here, because the configuration of the loaded model was a LoRA.
                # Hence we can expect a runtime_config to exist regardless of config_class.
                cfg.runtime_config.ephemeral_gpu_offload = True
                cfg.save_pretrained(tmp_dirname)
                cfg = config_class.from_pretrained(tmp_dirname)
                assert not cfg.runtime_config.ephemeral_gpu_offload

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_set_attributes(self, config_class, mandatory_kwargs):
        # manually set attributes and check if they are correctly written
        config = config_class(peft_type="test", **mandatory_kwargs)

        # save pretrained
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            assert config.to_dict() == config_from_pretrained.to_dict()

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_config_copy(self, config_class, mandatory_kwargs):
        # see https://github.com/huggingface/peft/issues/424
        config = config_class(**mandatory_kwargs)
        copied = copy.copy(config)
        assert config.to_dict() == copied.to_dict()

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_config_deepcopy(self, config_class, mandatory_kwargs):
        # see https://github.com/huggingface/peft/issues/424
        config = config_class(**mandatory_kwargs)
        copied = copy.deepcopy(config)
        assert config.to_dict() == copied.to_dict()

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_config_pickle_roundtrip(self, config_class, mandatory_kwargs):
        # see https://github.com/huggingface/peft/issues/424
        config = config_class(**mandatory_kwargs)
        copied = pickle.loads(pickle.dumps(config))
        assert config.to_dict() == copied.to_dict()

    @pytest.mark.parametrize(
        "config_class",
        [
            LoraConfig,
        ],
    )
    def test_save_pretrained_with_target_modules(self, config_class):
        # See #1041, #1045
        config = config_class(target_modules=["a", "list"])
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            assert config.to_dict() == config_from_pretrained.to_dict()
            # explicit test that target_modules should be converted to set
            assert isinstance(config_from_pretrained.target_modules, set)

    def test_regex_with_layer_indexing_lora(self):
        # This test checks that an error is raised if `target_modules` is a regex expression and `layers_to_transform` or
        # `layers_pattern` are not None

        invalid_config1 = {"target_modules": ".*foo", "layers_to_transform": [0]}
        invalid_config2 = {"target_modules": ".*foo", "layers_pattern": ["bar"]}

        valid_config = {"target_modules": ["foo"], "layers_pattern": ["bar"], "layers_to_transform": [0]}

        with pytest.raises(ValueError, match="`layers_to_transform` cannot be used when `target_modules` is a str."):
            LoraConfig(**invalid_config1)

        with pytest.raises(ValueError, match="`layers_pattern` cannot be used when `target_modules` is a str."):
            LoraConfig(**invalid_config2)

        # should run without errors
        LoraConfig(**valid_config)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_from_pretrained_forward_compatible(self, config_class, mandatory_kwargs, tmp_path, recwarn):
        """
        Make it possible to load configs that contain unknown keys by ignoring them.

        The idea is to make PEFT configs forward-compatible with future versions of the library.
        """
        config = config_class(**mandatory_kwargs)
        config.save_pretrained(tmp_path)
        # add a spurious key to the config
        with open(tmp_path / "adapter_config.json") as f:
            config_dict = json.load(f)
        config_dict["foobar"] = "baz"
        config_dict["spam"] = 123
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(config_dict, f)

        msg = f"Unexpected keyword arguments ['foobar', 'spam'] for class {config_class.__name__}, these are ignored."
        config_from_pretrained = config_class.from_pretrained(tmp_path)

        assert len(recwarn) == 1
        assert recwarn.list[0].message.args[0].startswith(msg)
        assert "foo" not in config_from_pretrained.to_dict()
        assert "spam" not in config_from_pretrained.to_dict()
        assert config.to_dict() == config_from_pretrained.to_dict()
        assert isinstance(config_from_pretrained, config_class)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_from_pretrained_forward_compatible_load_from_peft_config(
        self, config_class, mandatory_kwargs, tmp_path, recwarn
    ):
        """Exact same test as before, but instead of using LoraConfig.from_pretrained, AdaLoraconfig.from_pretrained,
        etc. use PeftConfig.from_pretrained. This covers a previously existing bug where only the known arguments from
        PeftConfig would be used instead of the more specific config (which is known thanks to the peft_type
        attribute).

        """
        config = config_class(**mandatory_kwargs)
        config.save_pretrained(tmp_path)
        # add a spurious key to the config
        with open(tmp_path / "adapter_config.json") as f:
            config_dict = json.load(f)
        config_dict["foobar"] = "baz"
        config_dict["spam"] = 123
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(config_dict, f)

        msg = f"Unexpected keyword arguments ['foobar', 'spam'] for class {config_class.__name__}, these are ignored."
        config_from_pretrained = PeftConfig.from_pretrained(tmp_path)  # <== use PeftConfig here

        assert len(recwarn) == 1
        assert recwarn.list[0].message.args[0].startswith(msg)
        assert "foo" not in config_from_pretrained.to_dict()
        assert "spam" not in config_from_pretrained.to_dict()
        assert config.to_dict() == config_from_pretrained.to_dict()
        assert isinstance(config_from_pretrained, config_class)

    @pytest.mark.parametrize("config_class, mandatory_kwargs", ALL_CONFIG_CLASSES)
    def test_from_pretrained_sanity_check(self, config_class, mandatory_kwargs, tmp_path):
        """Following up on the previous test about forward compatibility, we *don't* want any random json to be accepted as
        a PEFT config. There should be a minimum set of required keys.
        """
        non_peft_json = {"foo": "bar", "baz": 123}
        with open(tmp_path / "adapter_config.json", "w") as f:
            json.dump(non_peft_json, f)

        msg = (
            f"The {config_class.__name__} config that is trying to be loaded is missing required keys: {{'peft_type'}}."
        )
        with pytest.raises(TypeError, match=msg):
            config_class.from_pretrained(tmp_path)

    def test_lora_config_layers_to_transform_validation(self):
        """Test that specifying layers_pattern without layers_to_transform raises an error"""
        with pytest.raises(
            ValueError, match="When `layers_pattern` is specified, `layers_to_transform` must also be specified."
        ):
            LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], layers_pattern="model.layers")

        # Test that specifying both layers_to_transform and layers_pattern works fine
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            layers_to_transform=[0, 1, 2],
            layers_pattern="model.layers",
        )
        assert config.layers_to_transform == [0, 1, 2]
        assert config.layers_pattern == "model.layers"

        # Test that not specifying either works fine
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
        )
        assert config.layers_to_transform is None
        assert config.layers_pattern is None
