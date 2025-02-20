from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Union, Any
import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from ultralytics import SAM, YOLOWorld
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Constants
CLIP_MODEL = "openai/clip-vit-large-patch14"

# Configure device and models based on hardware
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    YOLO_MODEL = "yolov8x-worldv2.pt"
    SAM_MODEL = "sam2.1_l.pt"
else:
    DEVICE = torch.device("cpu")
    YOLO_MODEL = "yolov8s-worldv2.pt"
    SAM_MODEL = "sam2.1_t.pt"
print(DEVICE, YOLO_MODEL, SAM_MODEL)
class SegmentationMode(Enum):
    """Enum defining the available segmentation modes."""
    SEMANTIC = auto()
    AUTOMATIC = auto()

@dataclass
class SegmentationInput:
    """Data class for segmentation input parameters."""
    image: Image.Image
    text_prompt: Optional[str] = None
    automatic: bool = False

    def validate(self) -> "SegmentationInput":
        """Validate the input parameters."""
        # Count active modes
        modes = [
            bool(self.text_prompt),
            self.automatic
        ]
        active_modes = sum(modes)

        if active_modes == 0:
            raise ValueError("No segmentation mode specified")
        if active_modes > 1:
            raise ValueError("Multiple segmentation modes specified")

        return self

    @property
    def mode(self) -> SegmentationMode:
        """Determine the segmentation mode based on input parameters."""
        if self.text_prompt:
            return SegmentationMode.SEMANTIC
        return SegmentationMode.AUTOMATIC

class ImageSegmenter:
    """Class for handling image segmentation with different modes."""
    
    def __init__(self):
        print("Initializing models...")
        self.sam_model = SAM(SAM_MODEL)
        self.yolo_model: Optional[YOLOWorld] = None
        
        # Move models to device
        self.sam_model = self.sam_model.to(DEVICE)

    def _init_yolo(self, text_prompt: str) -> None:
        """Initialize YOLO model with text prompt."""
        self.yolo_model = YOLOWorld(YOLO_MODEL)
        self.yolo_model.to(DEVICE)
        self.yolo_model.set_classes([text_prompt])

    def _get_semantic_regions(self, image: Image.Image, text_prompt: str) -> List[List[float]]:
        """Get regions from semantic segmentation using OWLv2 (google/owlv2-large-patch14)."""
        # Initialize processor and model
        processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14")

        # Prepare the text query; using a single query for the provided text_prompt
        texts = [[f"a photo of {text_prompt}"]]

        # Process image and text
        inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Set target size as (height, width) for post-processing
        target_sizes = torch.tensor([image.size[::-1]])  # image.size is (width, height)

        # Post-process to get detection results
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)

        regions = []
        if results and len(results) > 0:
            det = results[0]
            boxes = det.get("boxes", [])
            for box in boxes:
                regions.append(box.tolist())
        
        return regions

    @staticmethod
    def _create_transparent_mask(
        result: np.ndarray,
        original_image: Image.Image
    ) -> Image.Image:
        """Create a transparent image containing the segmented object.
        
        Args:
            result: Binary mask from segmentation
            original_image: Original input image
            
        Returns:
            PIL Image with the segmented object and transparent background
        """
        # Convert PIL Image to numpy array
        image_array = np.array(original_image)
        
        # If there are multiple mask channels, combine them into a single 2D mask
        if result.ndim > 2:
            # Combine along the first axis
            result = np.max(result, axis=0)
        mask = result > 0.5
        
        # Create alpha channel (255 where mask is True, 0 where False)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        
        # Create RGBA image
        rgba = np.zeros((*image_array.shape[:2], 4), dtype=np.uint8)
        
        # Copy RGB channels from original image
        rgba[..., :3] = image_array[..., :3]
        
        # Set alpha channel
        rgba[..., 3] = alpha
        
        # Set fully transparent pixels' RGB to 0
        rgba[alpha == 0] = [0, 0, 0, 0]
        
        # Convert to PIL Image
        image = Image.fromarray(rgba, 'RGBA')
        
        # Crop to content bounds if there is content
        if alpha.any():
            # Find the bounds of the non-zero alpha values
            rows = np.any(alpha, axis=1)
            cols = np.any(alpha, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Add a small padding
            padding = 2
            ymin = max(0, ymin - padding)
            ymax = min(alpha.shape[0], ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(alpha.shape[1], xmax + padding)
            
            # Crop the image
            image = image.crop((xmin, ymin, xmax, ymax))
        
        return image

    def segment_image(self, input_data: SegmentationInput) -> List[str]:
        """
        Segment image based on input parameters and return base64 encoded PNGs.
        For semantic segmentation mode, YOLOWorld is used to generate bounding boxes for the given keyword, and each cropped region is sent to SAM for segmentation.

        Args:
            input_data: Validated SegmentationInput object

        Returns:
            List of base64 encoded PNG strings
        """
        # Validate input
        input_data.validate()

        segmented_images = []

        if input_data.mode == SegmentationMode.SEMANTIC and input_data.text_prompt:
            # Use YOLOWorld solely to get bounding boxes
            regions = self._get_semantic_regions(input_data.image, input_data.text_prompt)

            if regions:
                for region in regions:
                    x1, y1, x2, y2 = map(int, region)
                    # Crop region based on YOLO bounding box
                    cropped_image = input_data.image.crop((x1, y1, x2, y2))

                    # Run SAM segmentation on the cropped region; use a fresh copy
                    results = self.sam_model.predict(cropped_image.copy())  # type: ignore

                    # Process results for this cropped region
                    for result in results:
                        if not hasattr(result, 'masks') or not result.masks or not hasattr(result.masks, 'data'):
                            continue

                        mask_data = result.masks.data
                        if isinstance(mask_data, torch.Tensor):
                            mask_data = mask_data.cpu().numpy()
                        mask_data = np.squeeze(mask_data)

                        if mask_data.ndim == 2:
                            masks_to_process = [mask_data]
                        elif mask_data.ndim == 3:
                            masks_to_process = [m for m in mask_data]
                        else:
                            masks_to_process = []

                        for mask in masks_to_process:
                            try:
                                transparent_image = self._create_transparent_mask(mask, cropped_image)

                                buffer = BytesIO()
                                transparent_image.save(buffer, format="PNG")
                                base64_string = base64.b64encode(buffer.getvalue()).decode()
                                segmented_images.append(base64_string)
                            except Exception as e:
                                print(f"Warning: Failed to process a segment: {str(e)}")
        else:
            # Automatic segmentation: process the whole image
            results = self.sam_model.predict(input_data.image.copy())  # type: ignore

            for result in results:
                if not hasattr(result, 'masks') or not result.masks or not hasattr(result.masks, 'data'):
                    continue

                mask_data = result.masks.data
                if isinstance(mask_data, torch.Tensor):
                    mask_data = mask_data.cpu().numpy()
                mask_data = np.squeeze(mask_data)

                if mask_data.ndim == 2:
                    masks_to_process = [mask_data]
                elif mask_data.ndim == 3:
                    masks_to_process = [m for m in mask_data]
                else:
                    masks_to_process = []

                for mask in masks_to_process:
                    try:
                        transparent_image = self._create_transparent_mask(mask, input_data.image)

                        buffer = BytesIO()
                        transparent_image.save(buffer, format="PNG")
                        base64_string = base64.b64encode(buffer.getvalue()).decode()
                        segmented_images.append(base64_string)
                    except Exception as e:
                        print(f"Warning: Failed to process a segment: {str(e)}")

        return segmented_images

def segment_image(
    image: Image.Image,
    text_prompt: Optional[str] = None,
    automatic: bool = False
) -> List[str]:
    """
    Convenience function to segment an image with the specified parameters.
    
    Args:
        image: PIL Image to segment
        text_prompt: Text description for semantic segmentation
        automatic: Whether to use automatic segmentation
        
    Returns:
        List of base64 encoded PNG strings of segmented images
    """
    segmenter = ImageSegmenter()
    input_data = SegmentationInput(
        image=image,
        text_prompt=text_prompt,
        automatic=automatic
    )
    return segmenter.segment_image(input_data)
