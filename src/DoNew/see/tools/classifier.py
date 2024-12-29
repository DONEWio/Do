import numpy as np
import onnxruntime
import os
import fitz


from typing import IO, Union

import numpy as np  # Import numpy for array manipulation


def get_page_txt_img_ratio(page: fitz.Page) -> float:
    text_blocks = page.get_text("blocks")
    text_area = sum(
        [(block[2] - block[0]) * (block[3] - block[1]) for block in text_blocks]
    )
    page_area = abs(page.rect.width * page.rect.height)
    # Calculate the percentage of the page covered by text
    text_percentage = text_area / page_area if page_area > 0 else 0
    # Extract images from the page and calculate total image area
    image_list = page.get_images(full=True)
    image_area = 0
    for img in image_list:
        xref = img[0]
        img_info = page.get_image_info(xref)
        for info in img_info:
            img_rect = info["bbox"]  # Get the bounding box of the image
            image_area += abs((img_rect[2] - img_rect[0]) * (img_rect[3] - img_rect[1]))
    # Calculate the ratio of text area to image area
    ratio = (
        text_percentage / (image_area / page_area) if image_area > 0 else float("inf")
    )
    return ratio


def analyze_pdf_page(page: fitz.Page) -> bool:
    text_blocks = page.get_text("blocks") or []
    sorted_blocks = sorted(text_blocks, key=lambda b: b[1])

    # Get page dimensions for normalization
    page_width = page.rect.width
    page_height = page.rect.height

    # Normalize measurements relative to page dimensions
    left_margins = [block[0] / page_width for block in sorted_blocks]
    vertical_gaps = [
        (sorted_blocks[i + 1][1] - sorted_blocks[i][3]) / page_height
        for i in range(len(sorted_blocks) - 1)
    ]

    # Normalize horizontal gaps
    horizontal_gaps = []
    for i in range(len(sorted_blocks)):
        same_line_blocks = [
            b for b in sorted_blocks if abs(b[1] - sorted_blocks[i][1]) < 5
        ]
        same_line_blocks.sort(key=lambda b: b[0])
        for j in range(len(same_line_blocks) - 1):
            gap = (same_line_blocks[j + 1][0] - same_line_blocks[j][2]) / page_width
            horizontal_gaps.append(gap)

    # Calculate normalized statistics
    avg_margin = sum(left_margins) / len(left_margins) if left_margins else 0
    margin_variance = (
        sum((m - avg_margin) ** 2 for m in left_margins) / len(left_margins)
        if left_margins
        else 0
    )

    avg_vgap = sum(vertical_gaps) / len(vertical_gaps) if vertical_gaps else 0
    vgap_variance = (
        sum((g - avg_vgap) ** 2 for g in vertical_gaps) / len(vertical_gaps)
        if vertical_gaps
        else 0
    )

    avg_hgap = sum(horizontal_gaps) / len(horizontal_gaps) if horizontal_gaps else 0
    hgap_variance = (
        sum((g - avg_hgap) ** 2 for g in horizontal_gaps) / len(horizontal_gaps)
        if horizontal_gaps
        else 0
    )
    gap_variance = (vgap_variance + hgap_variance) / 2

    # Initialize counters for horizontal and vertical blocks
    horizontal_count = 0
    vertical_count = 0

    # Determine orientation of each block
    for block in sorted_blocks:
        width = block[2] - block[0]
        height = block[3] - block[1]
        if (
            height > width and height / width > 1.5
        ):  # Exclude blocks that are roughly square
            vertical_count += 1
        else:
            horizontal_count += 1

    # Calculate the ratio of horizontal to vertical blocks
    if vertical_count > 0:
        orientation_ratio = horizontal_count / vertical_count
    else:
        orientation_ratio = float("inf")  # Infinite if no vertical blocks

    # Add these new metrics:

    # Text block density (number of blocks per page area)
    block_density = len(sorted_blocks) / (page_width * page_height)

    # Text block size consistency
    block_sizes = [
        (block[2] - block[0]) * (block[3] - block[1]) for block in sorted_blocks
    ]
    avg_block_size = sum(block_sizes) / len(block_sizes) if block_sizes else 0
    size_variance = (
        sum((s - avg_block_size) ** 2 for s in block_sizes) / len(block_sizes)
        if block_sizes
        else 0
    )

    # Text block aspect ratios (helps identify tables vs paragraphs)
    aspect_ratios = [
        (
            (block[2] - block[0]) / (block[3] - block[1])
            if (block[3] - block[1]) > 0
            else 0
        )
        for block in sorted_blocks
    ]
    aspect_variance = np.var(aspect_ratios) if aspect_ratios else 0

    # Column detection (measure alignment of left margins)
    left_positions = [block[0] for block in sorted_blocks]
    unique_margins = set(
        round(pos / 20) * 20 for pos in left_positions
    )  # Round to nearest 20 units
    column_count = len(unique_margins)

    return {
        "margin_variance": float(margin_variance),
        "gap_variance": float(gap_variance),
        "vgap_variance": float(vgap_variance),
        "hgap_variance": float(hgap_variance),
        "orientation_ratio": float(orientation_ratio),
        "block_density": float(block_density),
        "size_variance": float(size_variance),
        "aspect_variance": float(aspect_variance),
        "column_count": column_count,
    }


def get_page_graphics_metrics(page: fitz.Page):
    page_width = int(page.rect.width)
    page_height = int(page.rect.height)
    paths = page.get_drawings()
    image_list = page.get_images(full=True)

    # Create a binary mask for the page
    mask = np.zeros((page_height, page_width), dtype=bool)

    # Mark all areas covered by graphics
    for path in paths:
        rect = path["rect"]
        x0, y0 = max(0, int(rect[0])), max(0, int(rect[1]))
        x1, y1 = min(page_width, int(rect[2])), min(page_height, int(rect[3]))
        mask[y0:y1, x0:x1] = True

    # Calculate actual coverage (avoiding double-counting)
    graphics_density = np.sum(mask) / (page_width * page_height)

    # Count different types of shapes
    line_count = sum(1 for path in paths if path["items"][0][0] == "l")
    rect_count = sum(1 for path in paths if path["items"][0][0] == "re")
    curve_count = sum(1 for path in paths if path["items"][0][0] in ("c", "v", "qu"))
    total_shapes = len(paths)

    return {
        "graphics_density": float(graphics_density),
        "curve_to_line_ratio": (
            float(curve_count / line_count) if line_count > 0 else float("inf")
        ),
        "rect_to_total_ratio": (
            float(rect_count / total_shapes) if total_shapes > 0 else 0
        ),
        "image_count": len(image_list),
    }


def analyze_whitespace(page: fitz.Page):
    # Create a binary mask of text/image areas
    width, height = int(page.rect.width), int(page.rect.height)
    mask = np.ones((height, width), dtype=np.uint8)

    # Mark text blocks
    for block in page.get_text("blocks"):
        x0, y0, x1, y1 = map(int, block[:4])
        mask[y0:y1, x0:x1] = 0

    # Mark images
    for img in page.get_images(full=True):
        for info in page.get_image_info(img[0]):
            bbox = info["bbox"]
            x0, y0, x1, y1 = map(int, bbox)
            mask[y0:y1, x0:x1] = 0

    # Calculate whitespace metrics
    whitespace_ratio = np.sum(mask) / mask.size

    # Find largest contiguous whitespace
    from scipy.ndimage import label

    labeled_array, num_features = label(mask)
    largest_whitespace = (
        max(np.bincount(labeled_array.flat)[1:]) if num_features > 0 else 0
    )

    return {
        "whitespace_ratio": float(whitespace_ratio),
        "largest_whitespace_ratio": float(largest_whitespace / mask.size),
    }


def get_pdf_text_image_ratios(pdf_path_or_bytes: Union[str, bytes, IO[bytes]]) -> list:
    if isinstance(pdf_path_or_bytes, (bytes, IO)):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)
    ratios = []
    for page in doc:
        ratios.append(ratios.append(get_page_txt_img_ratio(page)))
    doc.close()
    return ratios


def is_text_pdf(
    pdf_path_or_bytes: Union[str, bytes, IO[bytes]],
    page_threshold=0.9,
    text_threshold=0.25,
) -> bool:
    # Determine if the PDF is primarily text based on the ratio of text to images
    try:
        page_text_ratios = get_pdf_text_image_ratios(pdf_path_or_bytes)
        text_dominant_pages = [
            ratio for ratio in page_text_ratios if ratio > text_threshold
        ]
        text_dominant_ratio = len(text_dominant_pages) / len(page_text_ratios)
        print(
            "text_dominant_pages",
            text_dominant_pages,
            "page_text_ratios",
            page_text_ratios,
            "text_dominant_ratio",
            text_dominant_ratio,
        )
        return (text_dominant_ratio >= page_threshold, text_dominant_ratio)
    except Exception as e:
        print(repr(e))
        return (False, -1)  # probably an image file anyways


def get_page_metrics(page: fitz.Page):
    page.clean_contents()
    graphics = get_page_graphics_metrics(page)
    whitespace = analyze_whitespace(page)
    layout = analyze_pdf_page(page)
    txt_img_ratio = get_page_txt_img_ratio(page)
    return {**graphics, **whitespace, **layout, **{"txt_img_ratio": txt_img_ratio}}


def predict(session, input_data):
    input_name = session.get_inputs()[0].name
    input_data = np.array(input_data, dtype=np.float32)
    predictions = session.run(None, {input_name: input_data})
    return predictions


class Classifier:
    def __init__(self, model_path: str):
        self.session = onnxruntime.InferenceSession(model_path)

    def classify(self, page: fitz.Page):
        metrics = get_page_metrics(page)
        input = [list(metrics.values())]
        prediction = predict(self.session, input)
        return prediction

    def classify_from_metrics(self, metrics):
        input = [list(metrics.values())]
        prediction = predict(self.session, input)
        return prediction
