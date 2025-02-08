from typing import Union
from smolagents import TransformersModel, HfApiModel, LiteLLMModel, ChatMessage, MessageRole

Model = Union[TransformersModel, HfApiModel, LiteLLMModel]





__all__ = [
    "Model",
    "ChatMessage",
    "MessageRole",
    "LiteLLMModel",
    "TransformersModel",
    "HfApiModel",
]
