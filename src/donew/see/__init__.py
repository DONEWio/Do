from typing import List, Optional, Union, Sequence

from donew.see.processors import BaseTarget
from donew.see.processors.web import WebProcessor


async def See(
    paths, **kwargs
) -> Union[BaseTarget, Sequence[BaseTarget]]:
    """Static method to analyze images using global or override config

    Args:
        image_paths: Single image path or list of image paths
        kwargs: Optional config override {headless: bool, chrome_path: str}

    Returns:
        Single Target or sequence of Targets depending on input
    """


    # Initialize processors with config

    # Handle single path
    if isinstance(paths, str):
        if paths.endswith(".pdf"):
            raise NotImplementedError("PDF processing not implemented")
        elif paths.startswith("http"):
            # Special case for documentation
            if paths == "https://documentation/request":
                return WebProcessor()
            else:
                web_processor = WebProcessor(**kwargs)
                result = await web_processor.a_process(paths)
                return result[0]
        raise NotImplementedError("File type not implemented")

    # Handle list of paths
    if isinstance(paths, list):
        results: Sequence[BaseTarget] = []
        for path in paths:
            if path.endswith(".pdf"):
                raise NotImplementedError("PDF processing not implemented")
            elif path.startswith("http"):
                web_processor = WebProcessor(**kwargs)
                web_result = await web_processor.a_process(path)
                results.extend(web_result)
            else:
                raise NotImplementedError("File type not implemented")
        return results
    raise ValueError("image_paths must be a string or list of strings")
