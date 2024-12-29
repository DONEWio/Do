from typing import List, Dict, Any, Tuple, Union, Optional
from . import BaseProcessor, BaseTarget
from dataclasses import dataclass, field
import torch
from PIL import Image as PILImage
import supervision as sv
import tempfile
import os
import base64
import numpy as np
from ocr_providers import OCRProvider
from io import BytesIO
from processors import manual, public
from torchvision.transforms import ToPILImage
from torchvision.ops import box_convert
import io
import time


def ocr(
    image_path: str, provider: OCRProvider
) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
    """
    Run OCR using the specified provider (defaults to Google Cloud Vision)
    Returns:
    - text: List of text contents grouped by paragraphs
    - boxes: List of bounding boxes in (x1,y1,x2,y2) format
    """
    ocr_provider = provider
    return ocr_provider.process(image_path)


def get_yolo_model(model_path):
    from ultralytics import YOLO

    # Load the model.
    model = YOLO(model_path)
    return model


def get_caption_model_processor(
    model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None
):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == "cpu":
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float32
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name_or_path, device_map=None, torch_dtype=torch.float16
            ).to(device)
    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM

        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large", trust_remote_code=True
        )

        # Modify this section to ensure consistent dtype
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, trust_remote_code=True
        ).to(device)

        # Ensure processor outputs match model dtype
        processor.image_processor.do_normalize = True
        processor.image_processor.dtype = dtype

    return {"model": model.to(device), "processor": processor}


def annotate(
    image_source: np.ndarray,
    boxes: torch.Tensor,
    logits: torch.Tensor,
    phrases: List[str],
    text_scale: float,
    text_padding=5,
    text_thickness=2,
    thickness=3,
) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    from util.box_annotator import BoxAnnotator

    box_annotator = BoxAnnotator(
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=text_thickness,
        thickness=thickness,
    )  # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels, image_size=(w, h)
    )

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates


def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    """
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    """
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem["bbox"]
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem["bbox"]
            if (
                i != j
                and IoU(box1, box2) > iou_threshold
                and box_area(box1) > box_area(box2)
            ):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox
                box_added = False
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem["bbox"]
                        if is_inside(box3, box1):  # ocr inside icon
                            box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                filtered_boxes.append(
                                    {
                                        "type": "text",
                                        "bbox": box1_elem["bbox"],
                                        "interactivity": True,
                                        "content": box3_elem["content"],
                                    }
                                )
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            break
                        elif is_inside(box1, box3):  # icon inside ocr
                            box_added = True
                            try:
                                filtered_boxes.append(
                                    {
                                        "type": "icon",
                                        "bbox": box1_elem["bbox"],
                                        "interactivity": True,
                                        "content": None,
                                    }
                                )
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            break
                        else:
                            continue
                if not box_added:
                    filtered_boxes.append(
                        {
                            "type": "icon",
                            "bbox": box1_elem["bbox"],
                            "interactivity": True,
                            "content": None,
                        }
                    )

            else:
                filtered_boxes.append(box1)
    return filtered_boxes  # torch.tensor(filtered_boxes)


@torch.inference_mode()
def get_parsed_content_icon(
    filtered_boxes,
    starting_idx,
    image_source,
    caption_model_processor,
    prompt=None,
    batch_size=32,
):
    # Number of samples per batch, --> 256 roughly takes 23 GB of GPU memory for florence model

    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0] * image_source.shape[1]), int(
            coord[2] * image_source.shape[1]
        )
        ymin, ymax = int(coord[1] * image_source.shape[0]), int(
            coord[3] * image_source.shape[0]
        )
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = (
        caption_model_processor["model"],
        caption_model_processor["processor"],
    )
    if not prompt:
        if "florence" in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    generated_texts = []
    device = model.device
    for i in range(0, len(croped_pil_image), batch_size):
        start = time.time()
        batch = croped_pil_image[i : i + batch_size]
        if model.device.type == "cuda":
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt"
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch, text=[prompt] * len(batch), return_tensors="pt"
            ).to(device=device)
        if "florence" in model.config.name_or_path:
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=100,
                num_beams=3,
                do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1,
            )  # temperature=0.01, do_sample=True,
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text = [gen.strip() for gen in generated_text]
        generated_texts.extend(generated_text)

    return generated_texts


def get_parsed_content_icon_phi3v(
    filtered_boxes, ocr_bbox, image_source, caption_model_processor
):
    to_pil = ToPILImage()
    if ocr_bbox:
        non_ocr_boxes = filtered_boxes[len(ocr_bbox) :]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        xmin, xmax = int(coord[0] * image_source.shape[1]), int(
            coord[2] * image_source.shape[1]
        )
        ymin, ymax = int(coord[1] * image_source.shape[0]), int(
            coord[3] * image_source.shape[0]
        )
        cropped_image = image_source[ymin:ymax, xmin:xmax, :]
        croped_pil_image.append(to_pil(cropped_image))

    model, processor = (
        caption_model_processor["model"],
        caption_model_processor["processor"],
    )
    device = model.device
    messages = [
        {"role": "user", "content": "<|image_1|>\ndescribe the icon in one sentence"}
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    batch_size = 5  # Number of samples per batch
    generated_texts = []

    for i in range(0, len(croped_pil_image), batch_size):
        images = croped_pil_image[i : i + batch_size]
        image_inputs = [
            processor.image_processor(x, return_tensors="pt") for x in images
        ]
        inputs = {
            "input_ids": [],
            "attention_mask": [],
            "pixel_values": [],
            "image_sizes": [],
        }
        texts = [prompt] * len(images)
        for i, txt in enumerate(texts):
            input = processor._convert_images_texts_to_inputs(
                image_inputs[i], txt, return_tensors="pt"
            )
            inputs["input_ids"].append(input["input_ids"])
            inputs["attention_mask"].append(input["attention_mask"])
            inputs["pixel_values"].append(input["pixel_values"])
            inputs["image_sizes"].append(input["image_sizes"])
        max_len = max([x.shape[1] for x in inputs["input_ids"]])
        for i, v in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = torch.cat(
                [
                    processor.tokenizer.pad_token_id
                    * torch.ones(1, max_len - v.shape[1], dtype=torch.long),
                    v,
                ],
                dim=1,
            )
            inputs["attention_mask"][i] = torch.cat(
                [
                    torch.zeros(1, max_len - v.shape[1], dtype=torch.long),
                    inputs["attention_mask"][i],
                ],
                dim=1,
            )
        inputs_cat = {
            k: (v.to(dtype=model.dtype) if isinstance(v, torch.Tensor) else v).to(
                device
            )
            for k, v in inputs.items()
        }

        generation_args = {
            "max_new_tokens": 25,
            "temperature": 0.01,
            "do_sample": False,
        }
        generate_ids = model.generate(
            **inputs_cat,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args,
        )
        # # remove input tokens
        generate_ids = generate_ids[:, inputs_cat["input_ids"].shape[1] :]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = [res.strip("\n").strip() for res in response]
        generated_texts.extend(response)

    return generated_texts


def remove_overlap(boxes, iou_threshold, ocr_bbox=None):
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.95

    boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1 in enumerate(boxes):
        # if not any(IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2) for j, box2 in enumerate(boxes) if i != j):
        is_valid_box = True
        for j, box2 in enumerate(boxes):
            # keep the smaller box
            if (
                i != j
                and IoU(box1, box2) > iou_threshold
                and box_area(box1) > box_area(box2)
            ):
                is_valid_box = False
                break
        if is_valid_box:
            # add the following 2 lines to include ocr bbox
            if ocr_bbox:
                # only add the box if it does not overlap with any ocr bbox
                if not any(
                    IoU(box1, box3) > iou_threshold and not is_inside(box1, box3)
                    for k, box3 in enumerate(ocr_bbox)
                ):
                    filtered_boxes.append(box1)
            else:
                filtered_boxes.append(box1)
    return torch.tensor(filtered_boxes)


def predict_yolo(model, image_path, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """Use huggingface model to replace the original model"""
    # model = model['model']
    if scale_img:
        result = model.predict(
            source=image_path,
            conf=box_threshold,
            imgsz=imgsz,
            iou=iou_threshold,  # default 0.7
        )
    else:
        result = model.predict(
            source=image_path,
            conf=box_threshold,
            iou=iou_threshold,  # default 0.7
        )
    boxes = result[0].boxes.xyxy  # .tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def get_som_labeled_img(
    img: PILImage.Image,
    model=None,
    BOX_TRESHOLD=0.01,
    output_coord_in_ratio=False,
    ocr_bbox=None,
    text_scale=0.4,
    text_padding=5,
    draw_bbox_config=None,
    caption_model_processor=None,
    ocr_text=[],
    use_local_semantics=True,
    iou_threshold=0.9,
    prompt=None,
    scale_img=False,
    imgsz=None,
    batch_size=None,
):
    """ocr_bbox: list of xyxy format bbox"""
    image_source = img.convert("RGB")
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    # print('image size:', w, h)
    xyxy, logits, phrases = predict_yolo(
        model=model,
        image_path=img.filename if hasattr(img, "filename") else img,
        box_threshold=BOX_TRESHOLD,
        imgsz=imgsz,
        scale_img=scale_img,
        iou_threshold=0.1,
    )
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    image_source = np.asarray(image_source)
    phrases = [str(i) for i in range(len(phrases))]

    # annotate the image with labels
    h, w, _ = image_source.shape
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox = ocr_bbox.tolist()
    else:
        print("no ocr bbox!!!")
        ocr_bbox = None
    # filtered_boxes = remove_overlap(boxes=xyxy, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox)
    # starting_idx = len(ocr_bbox)
    # print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

    ocr_bbox_elem = [
        {"type": "text", "bbox": box, "interactivity": False, "content": txt}
        for box, txt in zip(ocr_bbox, ocr_text)
    ]
    xyxy_elem = [
        {"type": "icon", "bbox": box, "interactivity": True, "content": None}
        for box in xyxy.tolist()
    ]
    filtered_boxes = remove_overlap_new(
        boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem
    )

    # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x["content"] is None)
    # get the index of the first 'content': None
    starting_idx = next(
        (i for i, box in enumerate(filtered_boxes_elem) if box["content"] is None), -1
    )
    filtered_boxes = torch.tensor([box["bbox"] for box in filtered_boxes_elem])

    # get parsed icon local semantics
    if use_local_semantics:
        caption_model = caption_model_processor["model"]
        if "phi3_v" in caption_model.config.model_type:
            parsed_content_icon = get_parsed_content_icon_phi3v(
                filtered_boxes, ocr_bbox, image_source, caption_model_processor
            )
        else:
            parsed_content_icon = get_parsed_content_icon(
                filtered_boxes,
                starting_idx,
                image_source,
                caption_model_processor,
                prompt=prompt,
                batch_size=batch_size,
            )
        # Preserve original text spacing
        ocr_text = [f"Text Box ID {i}: {txt.strip()}" for i, txt in enumerate(ocr_text)]
        icon_start = len(ocr_text)
        parsed_content_icon_ls = []
        # fill the filtered_boxes_elem None content with parsed_content_icon in order
        for i, box in enumerate(filtered_boxes_elem):
            if box["content"] is None:
                content = parsed_content_icon.pop(0)
                box["content"] = content.strip() if content else None
        for i, txt in enumerate(parsed_content_icon):
            parsed_content_icon_ls.append(
                f"Icon Box ID {str(i+icon_start)}: {txt.strip()}"
            )
        parsed_content_merged = ocr_text + parsed_content_icon_ls
    else:
        # Preserve original text spacing
        ocr_text = [f"Text Box ID {i}: {txt.strip()}" for i, txt in enumerate(ocr_text)]
        parsed_content_merged = ocr_text

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")

    phrases = [i for i in range(len(filtered_boxes))]

    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source,
            boxes=filtered_boxes,
            logits=logits,
            phrases=phrases,
            **draw_bbox_config,
        )
    else:
        annotated_frame, label_coordinates = annotate(
            image_source=image_source,
            boxes=filtered_boxes,
            logits=logits,
            phrases=phrases,
            text_scale=text_scale,
            text_padding=text_padding,
        )

    pil_img = PILImage.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("ascii")
    if output_coord_in_ratio:
        # h, w, _ = image_source.shape
        label_coordinates = {
            k: [v[0] / w, v[1] / h, v[2] / w, v[3] / h]
            for k, v in label_coordinates.items()
        }
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates, filtered_boxes_elem


@dataclass
class Image(BaseTarget):
    """Image-specific target implementation that handles image-level operations.

    This class manages images, providing access to text content, images, and metadata.
    It supports OCR processing and maintains state about the image being processed.

    Attributes:
        _annotated_image (Optional[str]): Base64 encoded annotated image with detected text/elements.
        _raw_image (Optional[Image.Image]): PIL Image object of the raw image.
        _text_content (Optional[Tuple[List[str], List[Tuple[int, int, int, int]]]]):
            Extracted text content and bounding boxes.
        _debug_info (Optional[Dict[str, Any]]): Debug information from processing.
        _metadata (Dict[str, Any]): Additional metadata about the image.
    """

    _annotated_image: Optional[str] = None
    _raw_image: Optional[PILImage.Image] = None
    _text_content: Optional[Tuple[List[str], List[Tuple[int, int, int, int]]]] = None
    _debug_info: Optional[Dict[str, Any]] = field(default_factory=dict)
    _metadata: Dict[str, Any] = field(default_factory=dict)

    def _from_bbox_to_text(
        self, text_bboxes: Tuple[List[str], List[Tuple[int, int, int, int]]]
    ) -> str:
        """Reconstruct text from OCR results using bounding boxes.

        Args:
            text_bboxes: A tuple containing list of text segments and their corresponding bounding boxes.

        Returns:
            A reconstructed string representing the text content, organized based on spatial layout.
        """
        texts, bboxes = text_bboxes

        # Check if lengths match
        if len(texts) != len(bboxes):
            print("Warning: Number of text elements and bounding boxes do not match.")
            return " ".join(texts) if isinstance(texts, list) else texts

        # Create list of (text, bbox) tuples
        text_bbox_pairs = list(zip(texts, bboxes))

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
    ) -> Union[str, Tuple[List[str], List[Tuple[int, int, int, int]]]]:
        """Return the text content of the image in a human-readable format optimized for LLM processing.

        Args:
            with_bbox (bool): If True, returns text with bounding boxes.

        Returns:
            Union[str, Tuple[List[str], List[Tuple[int, int, int, int]]]]: The extracted text content.
            If with_bbox=True, returns a tuple of texts and their bounding boxes.
        """
        if not self._text_content:
            return ""

        if with_bbox:
            return self._text_content
        else:
            return self._from_bbox_to_text(self._text_content)

    @public(order=2)
    def get_image(
        self, with_bbox: bool = False
    ) -> Union[str, Tuple[str, List[Tuple[int, int, int, int]]]]:
        """Return the image content as a base64 encoded string.

        Args:
            with_bbox (bool): If True, returns image with bounding boxes.

        Returns:
            Union[str, Tuple[str, List[Tuple[int, int, int, int]]]]: The image content.
            If with_bbox=True, returns a tuple of image and bounding boxes.
        """
        if not self._raw_image:
            return ""

        try:
            buffered = BytesIO()
            if with_bbox and self._annotated_image:
                # Return annotated image with bounding boxes
                return self._annotated_image, self._text_content[1]
            else:
                # Return raw image
                self._raw_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                return img_str
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""

    def debug(self) -> Dict[str, Any]:
        """Return debug information including annotated image.

        Returns:
            Dict[str, Any]: Debug information from processing.
        """
        return {
            "annotated_image": self._annotated_image,
            **(self._debug_info or {}),
        }

    async def close(self):
        """Clean up resources."""
        try:
            if self._raw_image and hasattr(self._raw_image, "close"):
                self._raw_image.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class ImageProcessor(BaseProcessor[Union[str, PILImage.Image]]):
    """Processor for image content with ML models for vision analysis"""

    def __init__(self, ocr_provider: OCRProvider, device: str = None):
        """Initialize vision models and processors

        Args:
            ocr_provider: OCR service provider
            device: Device to run models on ('cuda', 'mps', or 'cpu')
        """
        if not device:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(f"Loading vision models to {device}")

        # Initialize vision models
        self.device = device
        self.som_model = get_yolo_model(model_path="weights/icon_detect_v1_5/best.pt")
        self.som_model.to(device)

        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="weights/icon_caption_florence",
            device=device,
        )

        self.ocr_provider = ocr_provider

    async def process(self, source: Union[str, PILImage.Image]) -> List[Image]:
        """Process image and return analysis results.

        Args:
            source (Union[str, PILImage.Image]): Image path or PIL Image object.

        Returns:
            List[Image]: A list containing a single Image target with the processing results.
        """
        # Load and convert image if needed
        if isinstance(source, str):
            image = PILImage.open(source).convert("RGB")
        else:
            image = source.convert("RGB")

        w, h = image.size

        # Configure visualization parameters
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }

        # Run OCR analysis
        raw_text, raw_boxes = ocr(image, provider=self.ocr_provider)

        # Get semantic analysis with YOLO and Florence
        annotated_image_str, label_coordinates, parsed_content = get_som_labeled_img(
            image,
            model=self.som_model,
            BOX_TRESHOLD=0.05,
            output_coord_in_ratio=True,
            ocr_bbox=raw_boxes,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=raw_text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128,
        )

        # Create debug info
        debug_info = {
            "label_coordinates": label_coordinates,
            "parsed_content": parsed_content,
            "image_size": (w, h),
            "raw_ocr": {"text": raw_text, "boxes": raw_boxes},
        }

        # Create target with results
        target = Image(
            _annotated_image=annotated_image_str,
            _raw_image=image,
            _text_content=(raw_text, raw_boxes),
            _debug_info=debug_info,
            _metadata={},  # Add any metadata if necessary
        )

        return [target]

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        import cv2

        _, buffer = cv2.imencode(".png", image)
        return base64.b64encode(buffer).decode("ascii")
