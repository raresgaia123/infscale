"""ModelMetaData."""
from abc import abstractmethod
from typing import List

from accelerate import init_empty_weights
from infscale import get_logger
from transformers import (AutoModelForCausalLM,
                          AutoModelForImageClassification,
                          AutoModelForPreTraining, PretrainedConfig,
                          PreTrainedModel)

AutoModelType = (
    AutoModelForPreTraining | AutoModelForCausalLM | AutoModelForImageClassification
)

logger = get_logger()


class BaseModelMetaData:
    """Base class for model meta data implementation."""

    def __init__(self, name: str, config: PretrainedConfig):
        """Initialize class."""
        self.name: str = name
        self.config: PretrainedConfig = config

        self.model: AutoModelType = None
        self.split_points: List[str] = None

    @abstractmethod
    def get_model(self) -> PreTrainedModel:
        """Abstract method to get model."""

    @abstractmethod
    def get_split_points(self) -> List[str]:
        """Abstract method to get split points."""


class Gpt2ModelMetaData(BaseModelMetaData):
    """Gpt2 model meta data class."""

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        with init_empty_weights():
            self.model = AutoModelForPreTraining.from_config(self.config)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"transformer.h.{i}")
        self.split_points.append("transformer.ln_f")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points


class BertModelMetaData(BaseModelMetaData):
    """Bert model meta data class."""

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"bert.encoder.layer.{i}")
        self.split_points.append("cls")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points


class T5ModelMetaData(BaseModelMetaData):
    """T5 model meta data class."""

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        with init_empty_weights():
            self.model = AutoModelForPreTraining.from_config(self.config)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_layers):
            self.split_points.append(f"encoder.block.{i}")
        for i in range(self.config.num_decoder_layers):
            self.split_points.append(f"decoder.block.{i}")
        self.split_points.append("lm_head")

        logger.debug(f"#layers = {self.config.num_layers}")
        logger.debug(f"#decoder_layers = {self.config.num_decoder_layers}")

        return self.split_points


class VitModelMetaData(BaseModelMetaData):
    """Vit model meta data class."""

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        with init_empty_weights():
            self.model = AutoModelForImageClassification.from_config(self.config)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        # Sharding for the Google's HuggingFace ViT model
        # e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i in range(self.config.num_hidden_layers):
            self.split_points.append(f"vit.encoder.layer.{i}")
        self.split_points.append("vit.layernorm")

        logger.debug(f"#hidden_layers = {self.config.num_hidden_layers}")

        return self.split_points


class ResnetModelMetaData(BaseModelMetaData):
    """Resnet model meta data class."""

    def get_model(self) -> PreTrainedModel:
        """Get model."""
        if self.model:
            return self.model

        with init_empty_weights():
            self.model = AutoModelForImageClassification.from_config(self.config)

        assert self.model, f"Given model {self.name} is not supported yet."

        return self.model

    def get_split_points(self) -> List[str]:
        """Get split points."""
        # Sharding for the Microsoft's HuggingFace ResNet model
        # e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
        if self.split_points:
            return self.split_points

        self.split_points: List[str] = []

        for i, depth in enumerate(self.config.depths):
            for j in range(depth):
                self.split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")

        self.split_points.append("resnet.pooler")

        logger.debug(f"#depths = {sum(self.config.depths)}")

        return self.split_points
