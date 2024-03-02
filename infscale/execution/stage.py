"""Stage class."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device
from infscale import get_logger

if TYPE_CHECKING:
    import torch.fx as fx
    from torch import Tensor


logger = get_logger()


class Stage(nn.Module):
    """Stage class."""

    def __init__(
        self,
        stage_id: str,
        layers: list[fx.GraphModule],
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize stage class instance."""
        super().__init__()

        self.id = stage_id
        self.layers = deepcopy(layers)
        self.device = device

        self._init_layers()

    def forward(self, inputs: tuple[Tensor]) -> tuple[Tensor]:
        """Run layers in the stage."""
        logger.debug(f"calling forward with inputs of type {type(inputs)}")
        for layer in self.layers:
            inputs = layer(*inputs)

        return inputs

    def _init_layers(self):
        """Initialize meta layers and move them to a device."""
        for layer in self.layers:
            self._init_tensors(layer)

    def _init_tensors(self, layer: torch.fx.GraphModule):
        """Initialize meta tensors and move them to a device."""
        # FIXME: need to update values from pretrained model
        #        currently random initialization is applied
        for param_name, param in layer.named_parameters():
            set_module_tensor_to_device(
                layer, param_name, self.device, torch.rand(param.shape)
            )

        for buffer_name, buffer in layer.named_buffers():
            set_module_tensor_to_device(
                layer, buffer_name, self.device, torch.rand(buffer.shape)
            )
