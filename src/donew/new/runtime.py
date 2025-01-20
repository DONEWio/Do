from typing import Optional
from dataclasses import dataclass


@dataclass
class Runtime:
    """Runtime configuration for task execution"""

    executor: str = "smolagents.local"
    workspace: Optional[str] = None
