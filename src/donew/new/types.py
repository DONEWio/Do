from typing import Protocol


class Model(Protocol):
    """Protocol for model interfaces"""

    async def generate(self, prompt: str, **kwargs) -> str: ...


class Provision(Protocol):
    """Protocol for context interfaces"""

    def setup(self) -> None: ...
    def cleanup(self) -> None: ...
