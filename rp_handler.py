#!/usr/bin/env python3.13
import runpod
from PIL import Image
from typing import AsyncGenerator, Any
import torch
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from ultralytics import SAM
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
from pydantic import BaseModel, Field, field_validator

# Set device to CUDA explicitly
DEVICE = torch.device("cuda")
SAM_MODEL_PATH = "/models/sam/sam2.1_l.pt"
HF_MODEL_NAME = "google/owlv2-large-patch14"
print(f"Device: {DEVICE}, SAM Model Path: {SAM_MODEL_PATH}, HF Model: {HF_MODEL_NAME}")

# Global model instantiation to avoid reinitializing on each call
print("Loading SAM model from local file...")
global_sam_model = SAM(SAM_MODEL_PATH).to(DEVICE)
print("Loading OWLv2 models from local cache...")
global_processor = Owlv2Processor.from_pretrained(HF_MODEL_NAME, local_files_only=True)
global_owlv2_model = Owlv2ForObjectDetection.from_pretrained(
    HF_MODEL_NAME, local_files_only=True
).to(DEVICE)
print("All models loaded successfully!")


class SegmentationMode(str, Enum):
    """Enum defining the available segmentation modes."""

    SEMANTIC = "semantic"
    AUTOMATIC = "automatic"


class SegmentationInput(BaseModel):
    """Pydantic model for segmentation input parameters."""

    image_base64: str = Field(..., description="Base64 encoded image data")
    text_prompt: str | None = Field(
        None, description="Text prompt for semantic segmentation"
    )
    automatic: bool = Field(False, description="Whether to use automatic segmentation")

    @field_validator("image_base64")
    @classmethod
    def validate_image(cls, v: str) -> str:
        try:
            image_data = base64.b64decode(v)
            Image.open(BytesIO(image_data))
            return v
        except Exception as e:
            raise ValueError(f"Invalid image data: {str(e)}")

    def get_pil_image(self) -> Image.Image:
        """Convert base64 image to PIL Image."""
        image_data = base64.b64decode(self.image_base64)
        return Image.open(BytesIO(image_data)).convert("RGB")

    @property
    def mode(self) -> SegmentationMode:
        """Determine the segmentation mode based on input parameters."""
        return (
            SegmentationMode.SEMANTIC
            if self.text_prompt
            else SegmentationMode.AUTOMATIC
        )

    def validate_mode(self) -> None:
        """Validate that only one mode is specified."""
        modes = [bool(self.text_prompt), self.automatic]
        active_modes = sum(modes)

        if active_modes == 0:
            raise ValueError("No segmentation mode specified")
        if active_modes > 1:
            raise ValueError("Multiple segmentation modes specified")


class SegmentationOutput(BaseModel):
    """Pydantic model for segmentation output."""

    image_base64: str = Field(..., description="Base64 encoded segmented image")
    size_bytes: int = Field(..., description="Size of the image in bytes")
    width: int = Field(..., description="Width of the image in pixels")
    height: int = Field(..., description="Height of the image in pixels")
    quality: int = Field(..., description="JPEG quality used (if compressed)")

    @classmethod
    def from_pil_image(
        cls, image: Image.Image, max_size_bytes: int = 19_000_000
    ) -> "SegmentationOutput":
        """Create SegmentationOutput from PIL Image, ensuring size is under limit."""
        buffer = BytesIO()
        width, height = image.size
        quality = 95
        format = "PNG" if image.mode == "RGBA" else "JPEG"

        # First try with high quality
        image.save(buffer, format=format, quality=quality)
        size = buffer.tell()

        # If too large, compress with JPEG
        if size > max_size_bytes and format == "PNG":
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            format = "JPEG"
            buffer = BytesIO()
            image.save(buffer, format=format, quality=quality)
            size = buffer.tell()

        # If still too large, reduce quality
        while size > max_size_bytes and quality > 5:
            quality -= 5
            buffer = BytesIO()
            image.save(buffer, format=format, quality=quality)
            size = buffer.tell()

        if size > max_size_bytes:
            raise ValueError(f"Unable to compress image below {max_size_bytes} bytes")

        return cls(
            image_base64=base64.b64encode(buffer.getvalue()).decode(),
            size_bytes=size,
            width=width,
            height=height,
            quality=quality,
        )


class ImageSegmenter:
    """Class for handling image segmentation with different modes."""

    def __init__(self):
        print("Using globally instantiated SAM model")
        self.sam_model = global_sam_model

    def _get_semantic_regions(
        self, image: Image.Image, text_prompt: str
    ) -> list[list[float]]:
        """Get regions from semantic segmentation using OWLv2."""
        processor = global_processor
        model = global_owlv2_model

        texts = [[text_prompt]]
        inputs = processor(text=texts, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [image.size[::-1]]  # (height, width)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=0.3
        )

        regions = []
        if results and len(results) > 0:
            det = results[0]
            regions = [box.tolist() for box in det.get("boxes", [])]
        return regions

    @staticmethod
    def _create_transparent_mask(
        result: np.ndarray, original_image: Image.Image, bbox: list[int] | None = None
    ) -> Image.Image | None:
        """Create a transparent image containing the segmented object."""
        image_array = np.array(original_image)
        mask = result > 0.5 if result.ndim == 2 else np.max(result, axis=0) > 0.5

        # Use connected component analysis to preserve significant components
        mask_uint8 = np.uint8(mask)
        mask_contiguous = np.ascontiguousarray(mask_uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_contiguous, connectivity=8, ltype=cv2.CV_32S
        )

        if num_labels > 1:
            total_area = np.sum(stats[1:, cv2.CC_STAT_AREA])
            significant_areas = stats[1:, cv2.CC_STAT_AREA]
            area_threshold = max(int(0.05 * total_area), 1000)
            significant_labels = np.where(significant_areas >= area_threshold)[0] + 1

            if len(significant_labels) > 0:
                new_mask = np.isin(labels, significant_labels)
                mask = new_mask
            else:
                largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                if largest_area < 500:
                    return None
                largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                mask = labels == largest_label

        non_zero_ratio = np.count_nonzero(mask) / mask.size
        if non_zero_ratio < 0.01:
            return None

        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.zeros((*image_array.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = image_array[..., :3]
        rgba[..., 3] = alpha
        rgba[alpha == 0] = [0, 0, 0, 0]

        image = Image.fromarray(rgba, "RGBA")

        if bbox:
            x1, y1, x2, y2 = bbox
            image = image.crop((x1, y1, x2, y2))
        elif alpha.any():
            rows, cols = np.any(alpha, axis=1), np.any(alpha, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            padding = 5
            ymin, ymax = max(0, ymin - padding), min(alpha.shape[0], ymax + padding)
            xmin, xmax = max(0, xmin - padding), min(alpha.shape[1], xmax + padding)

            if (ymax - ymin) < 20 or (xmax - xmin) < 20:
                return None
            image = image.crop((xmin, ymin, xmax, ymax))

        w, h = image.size
        if w < 20 or h < 20 or w * h < 400:
            return None

        return image

    def _process_mask(
        self, mask: np.ndarray, image: Image.Image, bbox: list[int] | None = None
    ) -> str:
        """Process a single mask and return base64 encoded PNG."""
        try:
            transparent_image = self._create_transparent_mask(mask, image, bbox)
            if transparent_image is None:
                return ""
            buffer = BytesIO()
            transparent_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            print(f"Warning: Failed to process a segment: {str(e)}")
            return ""

    def segment_image(self, input_data: SegmentationInput) -> list[str]:
        """Segment image based on input parameters and return base64 encoded PNGs."""
        pil_image = input_data.get_pil_image()
        segmented_images = []

        if input_data.mode == SegmentationMode.SEMANTIC and input_data.text_prompt:
            regions = self._get_semantic_regions(pil_image, input_data.text_prompt)
            if regions:
                results = self.sam_model.predict(pil_image, bboxes=regions)
                masks_to_process = []
                bboxes_to_process = []

                for result, region in zip(results, regions):
                    if (
                        not hasattr(result, "masks")
                        or not result.masks
                        or not hasattr(result.masks, "data")
                    ):
                        continue
                    mask_data = (
                        result.masks.data.cpu().numpy()
                        if isinstance(result.masks.data, torch.Tensor)
                        else result.masks.data
                    )
                    mask_data = np.squeeze(mask_data)
                    if mask_data.ndim == 2:
                        masks_to_process.append(mask_data)
                        bboxes_to_process.append([int(x) for x in region])
                    elif mask_data.ndim == 3:
                        for m in mask_data:
                            masks_to_process.append(m)
                            bboxes_to_process.append([int(x) for x in region])

                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self._process_mask, mask, pil_image, bbox)
                        for mask, bbox in zip(masks_to_process, bboxes_to_process)
                    ]
                    segmented_images = [f.result() for f in futures if f.result()]
        else:
            results = self.sam_model.predict(pil_image)
            masks_to_process = []

            for result in results:
                if (
                    not hasattr(result, "masks")
                    or not result.masks
                    or not hasattr(result.masks, "data")
                ):
                    continue
                mask_data = (
                    result.masks.data.cpu().numpy()
                    if isinstance(result.masks.data, torch.Tensor)
                    else result.masks.data
                )
                mask_data = np.squeeze(mask_data)
                if mask_data.ndim == 2:
                    masks_to_process.append(mask_data)
                elif mask_data.ndim == 3:
                    masks_to_process.extend(m for m in mask_data)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_mask, mask, pil_image)
                    for mask in masks_to_process
                ]
                segmented_images = [f.result() for f in futures if f.result()]

        return segmented_images


def segment_image(
    image_base64: str, text_prompt: str | None = None, automatic: bool = False
) -> list[str]:
    """Convenience function to segment an image with the specified parameters."""
    segmenter = ImageSegmenter()
    input_data = SegmentationInput(
        image_base64=image_base64, text_prompt=text_prompt, automatic=automatic
    )
    return segmenter.segment_image(input_data)


# Initialize segmenter once for all requests
segmenter = ImageSegmenter()


def convert_to_numpy(data: torch.Tensor | NDArray | Any) -> NDArray:
    """Convert various data types to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


async def process_segments(
    input_data: SegmentationInput,
) -> AsyncGenerator[dict[str, Any], None]:
    """Process segments and yield them one by one."""
    try:
        image = input_data.get_pil_image()

        if input_data.mode == SegmentationMode.SEMANTIC and input_data.text_prompt:
            regions = segmenter._get_semantic_regions(image, input_data.text_prompt)
            if regions:
                results = segmenter.sam_model.predict(image, bboxes=regions)
                for result, region in zip(results, regions):
                    if (
                        not hasattr(result, "masks")
                        or not result.masks
                        or not hasattr(result.masks, "data")
                    ):
                        continue

                    mask_data = convert_to_numpy(result.masks.data)
                    mask_data = mask_data.squeeze()

                    if mask_data.ndim == 2:
                        masks = [mask_data]
                    else:
                        masks = [m for m in mask_data]

                    for mask in masks:
                        transparent_image = segmenter._create_transparent_mask(
                            mask, image, [int(x) for x in region]
                        )
                        if transparent_image:
                            try:
                                output = SegmentationOutput.from_pil_image(
                                    transparent_image
                                )
                                yield {
                                    "status": "processing",
                                    "output": output.model_dump(),
                                }
                            except ValueError as e:
                                print(
                                    f"Warning: Skipping segment due to size constraints: {e}"
                                )
                                continue
        else:
            results = segmenter.sam_model.predict(image)
            for result in results:
                if (
                    not hasattr(result, "masks")
                    or not result.masks
                    or not hasattr(result.masks, "data")
                ):
                    continue

                mask_data = convert_to_numpy(result.masks.data)
                mask_data = mask_data.squeeze()

                if mask_data.ndim == 2:
                    masks = [mask_data]
                else:
                    masks = [m for m in mask_data]

                for mask in masks:
                    transparent_image = segmenter._create_transparent_mask(mask, image)
                    if transparent_image:
                        try:
                            output = SegmentationOutput.from_pil_image(
                                transparent_image
                            )
                            yield {
                                "status": "processing",
                                "output": output.model_dump(),
                            }
                        except ValueError as e:
                            print(
                                f"Warning: Skipping segment due to size constraints: {e}"
                            )
                            continue

        yield {"status": "completed"}

    except Exception as e:
        yield {"status": "error", "error": str(e)}


async def handler(event) -> AsyncGenerator[dict[str, Any], None]:
    """RunPod handler function that returns an async generator of segmentation results."""
    try:
        input_data = SegmentationInput(**event["input"])
        async for result in process_segments(input_data):
            yield result
    except Exception as e:
        yield {"status": "error", "error": str(e)}


# Start the serverless function with generator support enabled
runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
