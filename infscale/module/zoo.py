"""Zoo class."""

from infscale.module.model_metadata import (BaseModelMetaData,
                                            BertModelMetaData,
                                            Gpt2ModelMetaData,
                                            ResnetModelMetaData,
                                            T5ModelMetaData, VitModelMetaData)
from transformers import AutoConfig


class Zoo:
    """Collection of models supported in InfScale."""

    model_metadata_dict = {
        "openai-gpt": Gpt2ModelMetaData,
        "gpt2": Gpt2ModelMetaData,
        "bert": BertModelMetaData,
        "t5": T5ModelMetaData,
        "vit": VitModelMetaData,
        "resnet": ResnetModelMetaData,
    }

    @classmethod
    def get_model_metadata(cls, name: str) -> BaseModelMetaData:
        """Return a meta model."""
        config = AutoConfig.from_pretrained(name)

        model_type = config.model_type
        if model_type not in cls.model_metadata_dict:
            raise KeyError(f"Model type '{model_type}' is not supported.")

        return cls.model_metadata_dict[model_type](name, config)
