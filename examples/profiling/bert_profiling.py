# Copyright 2024 Cisco Systems, Inc. and its affiliates
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
#
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import BertForSequenceClassification, BertTokenizer
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from profile_in_json import get_model_inference_profile


def bert_input_constructor(batch_size, seq_len, tokenizer, batch_num=1):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer(
        [fake_seq] * (batch_size * batch_num),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = torch.tensor([1] * (batch_size * batch_num))
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with get_accelerator().device(0):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    batch_size = 4
    seq_len = 128
    enable_profile = True
    dataset = bert_input_constructor(batch_size, seq_len, tokenizer, batch_num=100)
    if enable_profile:
        flops, macs, params = get_model_inference_profile(
            model,
            kwargs=dataset,
            print_profile=True,
            detailed=True,
            output_file="./bert_profile.json",
        )
    else:
        inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
        outputs = model(inputs)
