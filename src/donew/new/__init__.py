from typing import Any, Optional

from donew.new.doers import BaseDoer
from donew.new.doers.super import SuperDoer
from donew.new.runtime import Runtime


async def New(config: dict[str, Any]) -> SuperDoer:
    """Create a new SuperDoer instance for task execution.

    Args:
        config: Configuration dictionary containing:
            - model: Model instance (required)
            - runtime: Runtime configuration (optional)
                - executor: Executor type (default: "smolagents.local")
                - workspace: Workspace path (optional)

    Returns:
        SuperDoer instance
    """
    model = config["model"]
    runtime = Runtime(**config["runtime"]) if "runtime" in config else None
    return SuperDoer(_model=model, _runtime=runtime)
