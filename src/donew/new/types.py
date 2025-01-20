from typing import Protocol


class Model(Protocol):
    """Protocol for model interfaces"""

    async def generate(self, prompt: str, **kwargs) -> str: ...


class Provision(Protocol):
    """Protocol for context interfaces"""

    async def setup(self) -> None: ...
    async def cleanup(self) -> None: ...
