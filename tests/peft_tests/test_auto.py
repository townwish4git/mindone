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
import tempfile

from transformers import AutoTokenizer

import mindspore as ms

from mindone.peft import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    get_peft_model,
)
from mindone.transformers import AutoModelForCausalLM


class TestPeftAutoModel:
    dtype = ms.bfloat16

    def test_peft_causal_lm(self):
        model_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        load_kwargs = {"revision": "refs/pr/2"}
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        assert isinstance(model, PeftModelForCausalLM)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModelForCausalLM.from_pretrained(tmp_dirname)
            assert isinstance(model, PeftModelForCausalLM)

        # check if kwargs are passed correctly
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, mindspore_dtype=self.dtype, **load_kwargs)
        assert isinstance(model, PeftModelForCausalLM)
        assert model.base_model.lm_head.weight.dtype == self.dtype

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForCausalLM.from_pretrained(
            model_id, adapter_name, is_trainable, mindspore_dtype=self.dtype, **load_kwargs
        )

    def test_peft_causal_lm_extended_vocab(self):
        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM-extended-vocab"
        model = AutoPeftModelForCausalLM.from_pretrained(model_id)
        assert isinstance(model, PeftModelForCausalLM)

        # check if kwargs are passed correctly
        model = AutoPeftModelForCausalLM.from_pretrained(model_id, mindspore_dtype=self.dtype)
        assert isinstance(model, PeftModelForCausalLM)
        assert model.base_model.lm_head.weight.dtype == self.dtype

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModelForCausalLM.from_pretrained(model_id, adapter_name, is_trainable, mindspore_dtype=self.dtype)

    def test_peft_whisper(self):
        model_id = "peft-internal-testing/tiny_WhisperForConditionalGeneration-lora"
        model = AutoPeftModel.from_pretrained(model_id)
        assert isinstance(model, PeftModel)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)

            model = AutoPeftModel.from_pretrained(tmp_dirname)
            assert isinstance(model, PeftModel)

        # check if kwargs are passed correctly
        model = AutoPeftModel.from_pretrained(model_id, mindspore_dtype=self.dtype)
        assert isinstance(model, PeftModel)
        assert model.base_model.model.model.encoder.embed_positions.weight.dtype == self.dtype

        adapter_name = "default"
        is_trainable = False
        # This should work
        _ = AutoPeftModel.from_pretrained(model_id, adapter_name, is_trainable, mindspore_dtype=self.dtype)

    def test_embedding_size_not_reduced_if_greater_vocab_size(self, tmp_path):
        # See 2415
        # There was a bug in AutoPeftModels where the embedding was always resized to the vocab size of the tokenizer
        # when the tokenizer was found. This makes sense if the vocabulary was extended, but some models like Qwen
        # already start out with "spare" embeddings, i.e. the embedding size is larger than the vocab size. This could
        # result in the embedding being shrunk, which in turn resulted in an error when loading the weights.

        # first create a checkpoint; it is important that the tokenizer is also saved in the same location
        model_id = "Qwen/Qwen2-0.5B"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = get_peft_model(model, LoraConfig(modules_to_save=["lm_head", "embed_token"]))
        model.save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)

        # does not raise; without the fix, it raises:
        # > size mismatch for base_model.model.lm_head.modules_to_save.default.weight: copying a param with shape
        # torch.Size([151936, 896]) from checkpoint, the shape in current model is torch.Size([151646, 896]).
        AutoPeftModelForCausalLM.from_pretrained(tmp_path)
