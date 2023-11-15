"""
Copyright (c) 2023 SymbioticLab, The University of Michigan

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# This file was modified from
# https://github.com/SymbioticLab/Oobleck/blob/3b7a0c2f19bff0991e623ffbeb8a5b365853bf3a/oobleck/module/model.py

import random
from typing import Any

import torch
from infscale.module.sharding import Sharder
from infscale.module.zoo import Zoo

RANDOM_SEED = 42


class ModelWrapper:
    """
    A wrapper model class of Hugging Face model downloaded from Hugging Face Hub (https://huggingface.co/models).

    It runs huggingface.utils.fx.symbolic_trace to get GraphModule
    and shard it to multiple GraphModules for pipeline execution.

    Model initialization must be done before distributed initialization.
    """

    def __init__(self, model_name: str, sample_inputs: dict[str, Any]):
        """Initialize the class."""
        # Initialize CPU seed
        random.seed(RANDOM_SEED)
        torch.default_generator.manual_seed(RANDOM_SEED)

        mmd = Zoo.get_model_metadata(model_name)

        self.sample_inputs = sample_inputs
        self.trace_input_names = list(sample_inputs.keys())

        self.layers = Sharder.shard(mmd, self.trace_input_names)
        self.model_name = model_name

        self.total_num_params = sum(
            sum(p.numel() for p in layer.parameters()) for layer in self.layers
        )

        self.model_args = mmd.config
