from typing import List, Optional, Union, Sequence

from src.donew.see.processors import BaseTarget
from src.donew.see.processors.web import WebProcessor


async def See(
    paths, config: Optional[dict] = None
) -> Union[BaseTarget, Sequence[BaseTarget]]:
    """Static method to analyze images using global or override config

    Args:
        image_paths: Single image path or list of image paths
        config: Optional config override {"ocr_provider": OCRProvider, "device": str}

    Returns:
        Single Target or sequence of Targets depending on input
    """
    if config is None:
        config = {"headless": True}

    # Initialize processors with config

    # Handle single path
    if isinstance(paths, str):
        if paths.endswith(".pdf"):
            raise NotImplementedError("PDF processing not implemented")
        elif paths.startswith("http"):
            web_processor = WebProcessor(config["headless"])
            result = await web_processor.process(paths)
            return result[0]
        raise NotImplementedError("File type not implemented")

    # Handle list of paths
    if isinstance(paths, list):
        results: Sequence[BaseTarget] = []
        for path in paths:
            if path.endswith(".pdf"):
                raise NotImplementedError("PDF processing not implemented")
            elif path.startswith("http"):
                web_processor = WebProcessor(config["headless"])
                web_result = await web_processor.process(path)
                results.extend(web_result)
            else:
                raise NotImplementedError("File type not implemented")
        return results
    raise ValueError("image_paths must be a string or list of strings")
