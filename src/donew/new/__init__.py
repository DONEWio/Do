from typing import Any
from donew.new.doers.super import SuperDoer


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
    name = config["name"]
    purpose = config["purpose"]
    model = config["model"]
    return SuperDoer(_model=model, _purpose=purpose, _name=name)
