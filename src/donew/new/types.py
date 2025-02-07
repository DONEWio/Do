

from typing import Union
from smolagents import TransformersModel, HfApiModel, LiteLLMModel

Model = Union[TransformersModel, HfApiModel, LiteLLMModel]

__all__ = [
    "Model",
]
