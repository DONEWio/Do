from typing import List, Dict, Any, Literal, Tuple, Optional, Union
from pathlib import Path
import pymupdf
import tempfile
import os
import base64
from PIL import Image
from processors import manual, public, BaseProcessor, BaseTarget
from ocr_providers import OCRProvider

import concurrent.futures
from dataclasses import dataclass, field
from processors.image import ImageProcessor
from tools.classifier import Classifier
import time


@dataclass
class PDFPage(BaseTarget):
    """A class representing a PDF document with OCR capabilities.

    This class handles PDF documents and provides functionality for:
    - OCR text extraction and bounding box detection
    - Annotated and raw image generation
    - Text content processing with positional information
    - Debug information storage

    Attributes:
        _page_num (int): The current page number being processed
        _needs_ocr (bool): Whether the document requires OCR processing
        _annotated_image (Optional[str]): Base64 encoded annotated image with detected text/elements
        _raw_image (Optional[str]): Base64 encoded raw page image
        _raw_images (Optional[Tuple[Image.Image, List[Tuple[int, int, int, int]]]]):
            Extracted text content and bounding boxes
        _text_content (Optional[Tuple[str, List[Tuple[int, int, int, int]]]]):
            Extracted text content and bounding boxes
        _debug_info (Optional[Dict[str, Any]]): Debug information from processing
        _metadata (Dict[str, Any]): Additional metadata about the document
    """

    _page_num: Optional[int] = None
    _needs_ocr: Optional[bool] = None
    _annotated_image: Optional[str] = None
    _raw_image: Optional[Image.Image] = None
    _raw_images: Optional[Tuple[Image.Image, List[Tuple[int, int, int, int]]]] = None
    _text_content: Optional[Tuple[str, List[Tuple[int, int, int, int]]]] = None
    _debug_info: Optional[Dict[str, Any]] = None
    _metadata: Dict[str, Any] = field(default_factory=dict)

    def _from_bbox_to_text(
        self, bboxes: Tuple[str, List[Tuple[int, int, int, int]]]
    ) -> str:
        text, bboxes = bboxes
        # Start Generation Here
        # Ensure 'text' is a list of strings
        if isinstance(text, str):
            # Split text into segments, assuming spaces or newlines
            text_list = text.strip().split()
        else:
            text_list = text

        # Check if lengths match
        if len(text_list) != len(bboxes):
            print("Warning: number of text elements and bounding boxes do not match.")
            return " ".join(text_list) if isinstance(text_list, list) else text_list

        # Create list of (text, bbox) tuples
        text_bbox_pairs = list(zip(text_list, bboxes))

        # Define functions to compute the y-center and height of the bbox
        def y_center(bbox):
            x1, y1, x2, y2 = bbox
            return (y1 + y2) / 2

        def bbox_height(bbox):
            x1, y1, x2, y2 = bbox
            return abs(y2 - y1)

        # Sort the text_bbox_pairs by y-center (top to bottom)
        text_bbox_pairs.sort(key=lambda item: y_center(item[1]))

        # Group the text_bbox_pairs into lines based on y-coordinate proximity
        lines = []
        current_line = []
        current_y = None
        line_threshold = 0.5  # Proportion of bbox height to consider same line

        for text_elem, bbox in text_bbox_pairs:
            yc = y_center(bbox)
            h = bbox_height(bbox)
            if current_y is None:
                current_y = yc
                current_line.append((text_elem, bbox))
            else:
                if abs(yc - current_y) <= h * line_threshold:
                    current_line.append((text_elem, bbox))
                else:
                    # Sort the current line by x-coordinate
                    current_line.sort(key=lambda item: (item[1][0] + item[1][2]) / 2)
                    lines.append(current_line)
                    current_line = [(text_elem, bbox)]
                    current_y = yc

        # Add the last line
        if current_line:
            current_line.sort(key=lambda item: (item[1][0] + item[1][2]) / 2)
            lines.append(current_line)

        # Concatenate text within lines and join lines
        line_texts = [" ".join([text for text, bbox in line]) for line in lines]
        concatenated_text = "\n".join(line_texts)

        return concatenated_text

    @public(order=1)
    def get_text(
        self, with_bbox: bool = False
    ) -> Union[str, Tuple[str, List[Tuple[int, int, int, int]]]]:
        """Return the text content of the document in a human-readable format optimized for LLM processing.

        This method returns the text content that has been extracted from the document and formatted
        appropriately for consumption by language models. The text is organized spatially based on the
        original document layout, with proper line breaks and spacing preserved.

        Returns:
            str | Tuple[str, List[Tuple[int, int, int, int]]]: The formatted text content. If with_bbox=True,
            also returns the bounding box coordinates for each text element.
        """
        if with_bbox:
            return self._from_bbox_to_text(self._text_content)
        return self._text_content

    @public(order=2)
    def get_image(
        self, with_bbox: bool = False
    ) -> Union[Image.Image, Tuple[Image.Image, List[Tuple[int, int, int, int]]]]:
        """Return the image content of the document as a PIL Image object.

        This method returns the document's image content as a PIL Image object, which can be used for
        further processing or display.

        Returns:
            Image.Image | Tuple[Image.Image, List[Tuple[int, int, int, int]]]: The image content. If with_bbox=True,
            also returns the bounding box coordinates for each text element.
        """
        if with_bbox:
            return self._raw_images
        return self._raw_image


@dataclass
class PDFInteraction:
    """Record of PDF document navigation"""
    page_num: int
    interaction_type: Literal["navigate"]
    timestamp: float
    data: Optional[Dict[str, Any]] = None


@dataclass
class PageAnalysis:
    """Analysis results for a single page"""
    page_num: int
    needs_ocr: bool
    metrics: Dict[str, float]  # All the metrics from get_page_metrics
    content_type: Literal["text", "mixed", "image"]  # Determined by txt_img_ratio
    word_count: int
    estimated_reading_time: float  # in seconds


@dataclass
class PDFDocument(BaseTarget):
    """PDF-specific target implementation that handles document-level operations.

    This class manages PDF documents and their pages, providing access to text content, images, and metadata.
    It supports OCR processing when needed and maintains state about the current page being processed.

    Attributes:
        _current_page_index (int): Index of current page being processed, or None for document-level operations
        _num_pages (int): Total number of pages in the PDF document
        _doc (pymupdf.Document): The underlying PDF document object
        _pages (List[PDFDocument]): List of processed PDF pages
        _raw_image (str): Path to raw image file of current page
        _text_content (List[str]): Extracted text content from current page
        _annotated_image (str): Path to annotated/debug image
        _debug_info (Dict): Additional debug information
        needs_ocr (bool): Whether OCR processing is required for current page
        page_num (int): Current page number (0-based)
    """

    _num_pages: Optional[int] = None
    _doc: Optional[pymupdf.Document] = None
    _current_page_index: int = 0
    _pages: List[PDFPage] = field(default_factory=list)
    _annotated_image: str = ""  # base64 encoded
    _raw_image: str = ""  # base64 encoded
    _text_content: List[str] = field(default_factory=list)
    _debug_info: Dict[str, Any] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)
    _interaction_history: List[PDFInteraction] = field(default_factory=list)
    _page_analyses: Dict[int, PageAnalysis] = field(default_factory=dict)

    def _analyze_page(self, page_num: int) -> PageAnalysis:
        """Analyze a page to determine its characteristics using existing metrics"""
        page = self._doc[page_num]
        metrics = get_page_metrics(page)  # Using existing function from classifier
        
        # Get word count from text content
        text_content = self._pages[page_num].get_text()
        if isinstance(text_content, tuple):
            text = text_content[0]
            words = text.split() if isinstance(text, str) else [w for t in text for w in t.split()]
            word_count = len(words)
        else:
            word_count = 0

        # Determine content type based on txt_img_ratio
        txt_img_ratio = metrics['txt_img_ratio']
        if txt_img_ratio < 0.1:
            content_type = "image"
        elif txt_img_ratio > 2.0:  # Text is dominant
            content_type = "text"
        else:
            content_type = "mixed"

        # Estimate reading time (avg reading speed: 250 words per minute)
        estimated_reading_time = word_count / 250 * 60  # in seconds

        return PageAnalysis(
            page_num=page_num,
            needs_ocr=self._pages[page_num]._needs_ocr,
            metrics=metrics,
            content_type=content_type,
            word_count=word_count,
            estimated_reading_time=estimated_reading_time
        )

    def _record_interaction(self, interaction_type: Literal["navigate"], page_num: int, data: Optional[Dict[str, Any]] = None):
        """Record an interaction in the history"""
        self._interaction_history.append(PDFInteraction(
            page_num=page_num,
            interaction_type=interaction_type,
            timestamp=time.time(),
            data=data
        ))

    @public(order=1)
    def goto(self, page_index: int):
        """Go to a specific page"""
        if page_index < 0 or page_index >= self._num_pages:
            raise ValueError(f"Page index {page_index} out of range")
        self._current_page_index = page_index
        self._record_interaction("navigate", page_index)

    @public(order=1)
    def next(self):
        """Go to the next page"""
        if self._current_page_index < self._num_pages - 1:
            self._current_page_index += 1
            self._record_interaction("navigate", self._current_page_index)

    @public(order=1)
    def prev(self):
        """Go to the previous page"""
        if self._current_page_index > 0:
            self._current_page_index -= 1
            self._record_interaction("navigate", self._current_page_index)

    def get_page_analysis(self, page_num: int) -> PageAnalysis:
        """Get analysis for a specific page, computing it if not already cached"""
        if page_num not in self._page_analyses:
            self._page_analyses[page_num] = self._analyze_page(page_num)
        return self._page_analyses[page_num]

    def get_document_stats(self) -> Dict[str, Any]:
        """Get overall document statistics"""
        # Ensure all pages are analyzed
        for page_num in range(self._num_pages):
            if page_num not in self._page_analyses:
                self._page_analyses[page_num] = self._analyze_page(page_num)

        # Compile statistics
        ocr_pages = [p for p in self._page_analyses.values() if p.needs_ocr]
        content_types = {p.content_type: 0 for p in self._page_analyses.values()}
        for p in self._page_analyses.values():
            content_types[p.content_type] += 1

        total_words = sum(p.word_count for p in self._page_analyses.values())
        total_reading_time = sum(p.estimated_reading_time for p in self._page_analyses.values())

        # Calculate average metrics across pages
        avg_metrics = {}
        for metric in self._page_analyses[0].metrics.keys():
            values = [p.metrics[metric] for p in self._page_analyses.values()]
            avg_metrics[metric] = sum(values) / len(values)

        return {
            "total_pages": self._num_pages,
            "ocr_pages": len(ocr_pages),
            "content_types": content_types,
            "total_words": total_words,
            "estimated_total_reading_time": total_reading_time,
            "average_metrics": avg_metrics
        }

    async def _get_state_dict(self) -> Dict[str, Any]:
        """Get document state including page analysis and interactions"""
        # Get current page analysis
        current_analysis = self.get_page_analysis(self._current_page_index)
        
        # Get current page text preview (first 100 words)
        current_text = self.get_current_text()
        if isinstance(current_text, tuple):
            preview_text = current_text[0]
        else:
            preview_text = current_text
        words = preview_text.split()
        preview = " ".join(words[:100]) + ("..." if len(words) > 100 else "")

        # Format timeline
        timeline_rows = []
        for interaction in self._interaction_history:
            time_str = time.strftime("%H:%M:%S", time.localtime(interaction.timestamp))
            action = f"Navigated to page {interaction.page_num + 1}"
            timeline_rows.append([time_str, action])

        # Get document stats
        doc_stats = self.get_document_stats()

        return {
            "sections": [
                {
                    "name": "Timeline",
                    "type": "table",
                    "headers": ["Time", "Action"],
                    "rows": timeline_rows,
                },
                {
                    "name": "Document Analysis",
                    "type": "keyvalue",
                    "data": {
                        "Overview": {
                            "Total Pages": str(doc_stats["total_pages"]),
                            "OCR Required": f"{doc_stats['ocr_pages']} pages",
                            "Content Types": f"{doc_stats['content_types']['text']} text, {doc_stats['content_types']['mixed']} mixed, {doc_stats['content_types']['image']} image",
                            "Total Words": str(doc_stats["total_words"]),
                            "Total Reading Time": f"~{doc_stats['estimated_total_reading_time'] / 60:.1f} minutes"
                        },
                        "Average Metrics": {
                            k: f"{v:.2f}" for k, v in doc_stats["average_metrics"].items()
                        }
                    }
                },
                {
                    "name": f"Current Page ({self._current_page_index + 1}/{self._num_pages})",
                    "type": "keyvalue",
                    "data": {
                        "Analysis": {
                            "Content Type": current_analysis.content_type,
                            "OCR Required": str(current_analysis.needs_ocr),
                            "Text/Image Ratio": f"{current_analysis.metrics['txt_img_ratio']:.2f}",
                            "Graphics Density": f"{current_analysis.metrics['graphics_density']:.2f}",
                            "Whitespace Ratio": f"{current_analysis.metrics['whitespace_ratio']:.2f}",
                            "Word Count": str(current_analysis.word_count),
                            "Reading Time": f"~{current_analysis.estimated_reading_time / 60:.1f} minutes"
                        }
                    }
                },
                {
                    "name": "Preview",
                    "type": "keyvalue",
                    "data": {
                        "Current Page": {
                            "Text": preview
                        }
                    }
                }
            ]
        }

    @public(order=2)
    @manual(template="Current page text see -> {extendee}", extends=PDFPage.get_text)
    def get_current_text(self, with_bbox: bool = False) -> str:
        """Return human-readable text for LLM consumption"""
        return self._pages[self._current_page_index].get_text(with_bbox)

    @public(order=3)
    @manual(template="Current page image see -> {extendee}", extends=PDFPage.get_image)
    def get_current_image(self, with_bbox: bool = False) -> str:
        """Return base64 encoded image for LLM consumption"""
        if not self._raw_image:
            return ""
        try:
            with open(self._raw_image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading image file {self._raw_image}: {e}")
            return ""

    @public(order=5)
    def debug(self) -> Dict[str, Any]:
        """Return debug information"""
        debug_info = {
            "annotated_image": self._annotated_image,
            **self._debug_info,
        }

        if self.page_num is not None:
            debug_info.update(
                {
                    "page_num": self.page_num,
                    "needs_ocr": self.needs_ocr,
                }
            )

        return debug_info

    @public(order=4)
    async def close(self):
        """Close the PDF document and clean up resources.

        This method closes the underlying PDF document to free up system resources.
        """
        try:
            if hasattr(self, "_doc") and self._doc:
                self._doc.close()
        except Exception as e:
            print(f"Error closing PDF document: {e}")


class PDFProcessor(BaseProcessor[str]):
    """Processor for PDF documents"""

    def __init__(self, ocr_provider: OCRProvider, device: str = None):
        """Initialize with classifier and image processor"""
        self.classifier = Classifier(model_path="models/trained_classifier_model.onnx")
        self.image_processor = ImageProcessor(ocr_provider, device)

    async def process(self, file_path: str) -> List[PDFDocument]:
        """Process a PDF file and return list of targets (one per page)"""
        # First get page classifications
        needs_ocr_pages = self.classify_pages(file_path)

        # Create document-level target
        doc = pymupdf.open(file_path)

        doc_target = PDFDocument(
            _doc=doc,
            _annotated_image="",
            _raw_image="",
            _text_content=[],
            _debug_info={},
            _metadata=doc.metadata,  # Use whatever metadata PyMuPDF gives us
            _num_pages=len(doc),  # Set the number of pages
        )

        pages: List[PDFPage] = []
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_target = await self.process_page(
                page, page_num, needs_ocr_pages[page_num]
            )
            page_target.type = "page"
            pages.append(page_target)

        doc_target._pages = pages  # Assign the list of pages to the document
        doc.close()
        return [doc_target]

    async def process_page(
        self, page: pymupdf.Page, page_num: int, needs_ocr: bool
    ) -> PDFPage:
        """Process a single PDF page"""
        # Convert page to image
        pix = page.get_pixmap()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)

            if needs_ocr:
                # Use ImageProcessor for OCR and analysis
                image_targets = await self.image_processor.process(tmp.name)
                image_target = image_targets[0]  # We only have one image

                target = PDFPage(
                    _annotated_image=image_target._annotated_image,
                    _raw_image=tmp.name,
                    _text_content=image_target._text_content,
                    _debug_info=image_target._debug_info,
                    _metadata={},  # Additional metadata if any
                    _page_num=page_num,
                    _needs_ocr=needs_ocr,
                )
            else:
                # Extract text directly from PDF
                blocks = page.get_text("blocks")
                text = [block[4] for block in blocks]
                boxes = [
                    (int(block[0]), int(block[1]), int(block[2]), int(block[3]))
                    for block in blocks
                ]

                target = PDFPage(
                    _annotated_image="",  # No annotation for direct PDF text
                    _raw_image=tmp.name,
                    _text_content=(text, boxes),
                    _debug_info={"boxes": boxes},
                    _metadata={},  # Additional metadata if any
                    _page_num=page_num,
                    _needs_ocr=needs_ocr,
                )

            return target

    def classify_pages(self, file_path: str) -> List[bool]:
        """Classify pages to determine if they need OCR"""
        doc = pymupdf.open(file_path)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.classifier.classify, doc))
        doc.close()
        return results
