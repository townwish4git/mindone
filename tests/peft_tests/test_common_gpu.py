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
import gc
import tempfile
import unittest

import pytest
from datasets import load_dataset
from parameterized import parameterized

from mindspore import nn

from mindone.peft import LoraConfig, PeftModel, TaskType, get_peft_model
from mindone.peft.tuners.lora.config import LoraRuntimeConfig
from mindone.peft.utils import infer_device
from mindone.transformers import AutoImageProcessor, AutoModelForCausalLM
from mindone.transformers.mindspore_utils import Conv1D

from .testing_utils import require_mindspore_npu, require_multi_accelerator, require_non_cpu


@require_non_cpu
class PeftGPUCommonTests(unittest.TestCase):
    r"""
    A common tester to run common operations that are performed on GPU such as generation, loading in 8bit, etc.
    """

    def setUp(self):
        self.seq2seq_model_id = "google/flan-t5-base"
        self.causal_lm_model_id = "facebook/opt-350m"
        self.audio_model_id = "openai/whisper-large"
        self.device = infer_device()

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif is_xpu_available():
            torch.xpu.empty_cache()
        gc.collect()

    @pytest.mark.multi_gpu_tests
    @require_multi_accelerator
    def test_lora_causal_lm_multi_gpu_inference(self):
        r"""
        Test if LORA can be used for inference on multiple GPUs.
        """
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, device_map="balanced")
        tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)

        assert set(model.hf_device_map.values()) == set(range(device_count))

        model = get_peft_model(model, lora_config)
        assert isinstance(model, PeftModel)

        dummy_input = "This is a dummy input:"
        input_ids = tokenizer(dummy_input, return_tensors="pt").input_ids.to(self.device)

        # this should work without any problem
        _ = model.generate(input_ids=input_ids)

    @require_non_cpu
    @pytest.mark.single_gpu_tests
    def test_serialization_shared_tensors(self):
        model_checkpoint = "roberta-base"
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1, bias="all"
        )
        model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=11).to(self.device)
        model = get_peft_model(model, peft_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)

    def test_apply_GS_hra_inference(self):
        # check for different result with and without apply_GS
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float32,
        ).eval()

        torch.manual_seed(0)
        config_hra = HRAConfig(r=8, init_weights=True, apply_GS=False)
        model = get_peft_model(model, config_hra).eval()

        random_input = torch.LongTensor([[1, 0, 1, 0, 1, 0]]).to(model.device)
        logits_hra = model(random_input).logits

        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float32,
        )
        torch.manual_seed(0)
        config_hra_GS = HRAConfig(r=8, init_weights=True, apply_GS=True)
        model = get_peft_model(model, config_hra_GS)

        logits_hra_GS = model(random_input).logits

        assert not torch.allclose(logits_hra, logits_hra_GS)

    @require_non_cpu
    @pytest.mark.single_gpu_tests
    def test_apply_GS_hra_conv2d_inference(self):
        # check for different result with and without apply_GS
        model_id = "microsoft/resnet-18"
        image_processor = AutoImageProcessor.from_pretrained(model_id)
        dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
        image = dataset["test"]["image"][0]
        data = image_processor(image, return_tensors="pt")

        model = AutoModelForImageClassification.from_pretrained(model_id).eval()
        torch.manual_seed(0)
        config_hra = HRAConfig(r=8, init_weights=True, target_modules=["convolution"], apply_GS=False)
        model = get_peft_model(model, config_hra).eval()

        logits_hra = model(**data).logits

        model = AutoModelForImageClassification.from_pretrained(model_id).eval()
        torch.manual_seed(0)
        config_hra_GS = HRAConfig(r=8, init_weights=True, target_modules=["convolution"], apply_GS=True)
        model = get_peft_model(model, config_hra_GS)

        logits_hra_GS = model(**data).logits

        assert not torch.allclose(logits_hra, logits_hra_GS)

    @require_non_cpu
    @pytest.mark.single_gpu_tests
    def test_r_odd_hra_inference(self):
        # check that an untrained HRA adapter can't be initialized as an identity tranformation
        # when r is an odd number
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float32,
        ).eval()

        random_input = torch.LongTensor([[1, 0, 1, 0, 1, 0]]).to(model.device)

        torch.manual_seed(0)
        logits = model(random_input).logits

        config_hra = HRAConfig(r=7, init_weights=True, apply_GS=False)
        model = get_peft_model(model, config_hra).eval()
        logits_hra = model(random_input).logits

        assert not torch.allclose(logits, logits_hra)
