#!/usr/bin/env python3.13
import torch
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from ultralytics import SAM
from transformers import (
    Owlv2Processor,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
)
from transformers.models.owlv2 import Owlv2Processor as Owlv2ProcessorType
import cv2
from pydantic import BaseModel, Field, field_validator
from typing import cast, Any
import os
from PIL import Image

# Constants
DEVICE = torch.device("cuda")
SAM_MODEL_PATH = "/models/sam/sam2.1_l.pt"
HF_MODEL_NAME = "google/owlv2-large-patch14"
LOCAL_MODEL_PATH = "/models/huggingface/owlv2-large-patch14"

# Global model instances
global_sam_model = None
global_processor: Owlv2ProcessorType | None = None
global_image_processor: Owlv2ImageProcessor | None = None
global_owlv2_model = None


def initialize_models() -> None:
    """Initialize all models and processors."""
    global global_sam_model, global_processor, global_image_processor, global_owlv2_model

    print(f"Device: {DEVICE}, SAM Model Path: {SAM_MODEL_PATH}, HF Model: {HF_MODEL_NAME}")

    # Initialize SAM model
    print(f"Loading SAM model from {SAM_MODEL_PATH}...")
    if not os.path.exists(SAM_MODEL_PATH):
        raise RuntimeError(f"SAM model not found at {SAM_MODEL_PATH}")
    global_sam_model = SAM(SAM_MODEL_PATH)
    if torch.cuda.is_available():
        global_sam_model = global_sam_model.cuda()
        print("SAM model moved to CUDA")
    print(f"SAM model loaded successfully, size: {os.path.getsize(SAM_MODEL_PATH) / (1024*1024):.2f} MB")

    # Initialize OWLv2 models
    print(f"Loading OWLv2 models from {LOCAL_MODEL_PATH}...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise RuntimeError(f"OWLv2 model directory not found at {LOCAL_MODEL_PATH}")

    try:
        # Load directly from local directory with local_files_only=True
        processor_result = Owlv2Processor.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
        )

        if isinstance(processor_result, tuple) and len(processor_result) == 2:
            processor, image_processor = processor_result
            if isinstance(image_processor, Owlv2ImageProcessor):
                global_processor = cast(Owlv2ProcessorType, processor)
                global_image_processor = image_processor
            else:
                global_processor = cast(Owlv2ProcessorType, processor_result)
                global_image_processor = None
        else:
            global_processor = cast(Owlv2ProcessorType, processor_result)
            global_image_processor = None

        global_owlv2_model = Owlv2ForObjectDetection.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("All models loaded successfully from local directory!")
    except Exception as e:
        print(f"Error loading models from local directory: {str(e)}")
        print("Attempting to load from HuggingFace cache...")
        try:
            # Try loading from cache with offline mode
            processor_result = Owlv2Processor.from_pretrained(
                HF_MODEL_NAME,
                cache_dir="/models/huggingface",
                local_files_only=True,
                trust_remote_code=True,
            )
            if isinstance(processor_result, tuple) and len(processor_result) == 2:
                processor, image_processor = processor_result
                if isinstance(image_processor, Owlv2ImageProcessor):
                    global_processor = cast(Owlv2ProcessorType, processor)
                    global_image_processor = image_processor
                else:
                    global_processor = cast(Owlv2ProcessorType, processor_result)
                    global_image_processor = None
            else:
                global_processor = cast(Owlv2ProcessorType, processor_result)
                global_image_processor = None

            global_owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                HF_MODEL_NAME,
                cache_dir="/models/huggingface",
                local_files_only=True,
                trust_remote_code=True,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
            )
            print("Models loaded successfully from cache!")
        except Exception as e:
            print(f"Fatal error loading models: {str(e)}")
            raise


class SegmentationMode(str, Enum):
    """Enum defining the available segmentation modes."""
    SEMANTIC = "semantic"
    AUTOMATIC = "automatic"


class SegmentationInput(BaseModel):
    """Pydantic model for segmentation input parameters."""
    image_base64: str = Field(..., description="Base64 encoded image data")
    text_prompt: str | None = Field(None, description="Text prompt for semantic segmentation")
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
    def from_pil_image(cls, image: Image.Image, max_size_bytes: int = 19_000_000) -> "SegmentationOutput":
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
        if global_sam_model is None:
            initialize_models()
        self.sam_model = global_sam_model

    def _get_semantic_regions(self, image: Image.Image, text_prompt: str) -> list[list[float]]:
        """Get regions from semantic segmentation using OWLv2."""
        if global_processor is None or global_owlv2_model is None:
            initialize_models()

        texts = [[text_prompt]]

        # Process inputs using the correct processor components
        if isinstance(global_image_processor, Owlv2ImageProcessor):
            image_inputs = cast(
                dict[str, Any],
                global_image_processor(images=image, return_tensors="pt"),
            )
            text_inputs = cast(
                dict[str, Any], global_processor(text=texts, return_tensors="pt")
            )
            inputs = {**image_inputs, **text_inputs}
        else:
            inputs = cast(
                dict[str, Any],
                global_processor(text=texts, images=image, return_tensors="pt"),
            )

        # Move inputs to device if needed
        if torch.cuda.is_available():
            inputs = {
                k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        # Forward pass
        with torch.no_grad():
            outputs = global_owlv2_model(**inputs)

        target_sizes = [image.size[::-1]]  # (height, width)

        # Use the correct processor for post-processing
        if isinstance(global_image_processor, Owlv2ImageProcessor):
            results = cast(
                list[dict[str, Any]],
                global_processor.post_process_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.2
                ),
            )
        else:
            results = cast(
                list[dict[str, Any]],
                global_processor.post_process_grounded_object_detection(
                    outputs=outputs, target_sizes=target_sizes, threshold=0.2
                ),
            )

        regions = []
        if results and len(results) > 0:
            det = results[0]
            boxes = det.get("boxes", [])
            scores = det.get("scores", [])
            
            # Print scores for each detected box
            print(f"\nOWLv2 Detection Results for '{text_prompt}':")
            for idx, (box, score) in enumerate(zip(boxes, scores)):
                print(f"  Box {idx + 1}: Confidence = {score:.3f}")
                regions.append(box.tolist())
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
                for region in regions:
                    region_int = [int(x) for x in region]
                    cropped_img = pil_image.crop((region_int[0], region_int[1], region_int[2], region_int[3]))
                    results = self.sam_model.predict(cropped_img)
                    
                    # Find the largest mask from all results
                    largest_mask = None
                    largest_area = 0
                    
                    for result in results:
                        if (not hasattr(result, "masks") or not result.masks or not hasattr(result.masks, "data")):
                            continue
                            
                        mask_data = (result.masks.data.cpu().numpy() if isinstance(result.masks.data, torch.Tensor) else result.masks.data)
                        mask_data = np.squeeze(mask_data)
                        
                        # Handle both 2D and 3D masks
                        if mask_data.ndim == 2:
                            area = np.sum(mask_data > 0.5)
                            if area > largest_area:
                                largest_area = area
                                largest_mask = mask_data
                        elif mask_data.ndim == 3:
                            for m in mask_data:
                                area = np.sum(m > 0.5)
                                if area > largest_area:
                                    largest_area = area
                                    largest_mask = m
                    
                    # Process only the largest mask if found
                    if largest_mask is not None:
                        processed_mask = self._process_mask(largest_mask, cropped_img, None)
                        if processed_mask:
                            segmented_images.append(processed_mask)
                            
        else:
            results = self.sam_model.predict(pil_image)
            largest_mask = None
            largest_area = 0
            
            for result in results:
                if (not hasattr(result, "masks") or not result.masks or not hasattr(result.masks, "data")):
                    continue
                    
                mask_data = (result.masks.data.cpu().numpy() if isinstance(result.masks.data, torch.Tensor) else result.masks.data)
                mask_data = np.squeeze(mask_data)
                
                if mask_data.ndim == 2:
                    area = np.sum(mask_data > 0.5)
                    if area > largest_area:
                        largest_area = area
                        largest_mask = mask_data
                elif mask_data.ndim == 3:
                    for m in mask_data:
                        area = np.sum(m > 0.5)
                        if area > largest_area:
                            largest_area = area
                            largest_mask = m
            
            if largest_mask is not None:
                processed_mask = self._process_mask(largest_mask, pil_image)
                if processed_mask:
                    segmented_images.append(processed_mask)

        return segmented_images


def convert_to_numpy(data: torch.Tensor | NDArray | Any) -> NDArray:
    """Convert various data types to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def verify_model_paths() -> None:
    """Verify that all required model files exist."""
    # Check SAM model
    if not os.path.exists(SAM_MODEL_PATH):
        raise RuntimeError(f"SAM model not found at {SAM_MODEL_PATH}")
    
    # Check OWLv2 model files
    required_files = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
    for file in required_files:
        path = os.path.join(LOCAL_MODEL_PATH, file)
        if not os.path.exists(path):
            raise RuntimeError(f"Required OWLv2 model file not found: {path}")


def segment_image(
    image_base64: str, text_prompt: str | None = None, automatic: bool = False
) -> list[str]:
    """Convenience function to segment an image with the specified parameters."""
    segmenter = ImageSegmenter()
    input_data = SegmentationInput(
        image_base64=image_base64, text_prompt=text_prompt, automatic=automatic
    )
    return segmenter.segment_image(input_data)
