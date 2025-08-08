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
import unittest

import pytest
from transformers import AutoTokenizer

import mindspore as ms

from mindone.peft import LoraConfig, PeftModel, get_peft_model
from mindone.transformers import AutoModelForCausalLM

from .testing_utils import require_multi_accelerator, require_non_cpu


@require_non_cpu
class PeftGPUCommonTests(unittest.TestCase):
    r"""
    A common tester to run common operations that are performed on GPU such as generation, loading in 8bit, etc.
    """

    def setUp(self):
        self.seq2seq_model_id = "google/flan-t5-base"
        self.causal_lm_model_id = "facebook/opt-350m"
        self.audio_model_id = "openai/whisper-large"

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        ms.hal.empty_cache()
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

        model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, revision="refs/pr/40")
        tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)

        model = get_peft_model(model, lora_config)
        assert isinstance(model, PeftModel)

        dummy_input = "This is a dummy input:"
        input_ids = ms.Tensor.from_numpy(tokenizer(dummy_input, return_tensors="np").input_ids)

        # this should work without any problem
        _ = model.generate(input_ids=input_ids)
