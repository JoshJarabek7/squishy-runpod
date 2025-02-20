from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from ultralytics import SAM
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2

# Set device to CUDA explicitly
DEVICE = torch.device("cuda")
SAM_MODEL = "sam2.1_l.pt"
print(f"Device: {DEVICE}, SAM Model: {SAM_MODEL}")

# Global model instantiation to avoid reinitializing on each call
global_sam_model = SAM(SAM_MODEL).to(DEVICE)
processor_result = Owlv2Processor.from_pretrained("google/owlv2-large-patch14")
global_processor = processor_result[0] if isinstance(processor_result, tuple) else processor_result
global_owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14")
global_owlv2_model.to(DEVICE)  # In-place modification without reassignment

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
        modes = [bool(self.text_prompt), self.automatic]
        active_modes = sum(modes)

        if active_modes == 0:
            raise ValueError("No segmentation mode specified")
        if active_modes > 1:
            raise ValueError("Multiple segmentation modes specified")

        return self

    @property
    def mode(self) -> SegmentationMode:
        """Determine the segmentation mode based on input parameters."""
        return SegmentationMode.SEMANTIC if self.text_prompt else SegmentationMode.AUTOMATIC

class ImageSegmenter:
    """Class for handling image segmentation with different modes."""
    
    def __init__(self):
        print("Using globally instantiated SAM model")
        self.sam_model = global_sam_model

    def _get_semantic_regions(self, image: Image.Image, text_prompt: str) -> List[List[float]]:
        """Get regions from semantic segmentation using OWLv2."""
        processor = global_processor
        model = global_owlv2_model

        texts = [[text_prompt]]
        inputs = processor(text=texts, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [image.size[::-1]]  # (height, width)
        results = processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.3)

        regions = []
        if results and len(results) > 0:
            det = results[0]
            regions = [box.tolist() for box in det.get("boxes", [])]
        return regions

    @staticmethod
    def _create_transparent_mask(result: np.ndarray, original_image: Image.Image, bbox: Optional[List[int]] = None) -> Image.Image:
        """Create a transparent image containing the segmented object."""
        image_array = np.array(original_image)
        mask = result > 0.5 if result.ndim == 2 else np.max(result, axis=0) > 0.5

        # Use connected component analysis for the largest island
        mask_uint8 = np.uint8(mask)
        mask_contiguous = np.ascontiguousarray(mask_uint8)  # Ensure contiguous array
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_contiguous, connectivity=8, ltype=cv2.CV_32S)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            mask = (labels == largest_label)

        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.zeros((*image_array.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = image_array[..., :3]
        rgba[..., 3] = alpha
        rgba[alpha == 0] = [0, 0, 0, 0]

        image = Image.fromarray(rgba, 'RGBA')
        
        # Crop to bounding box or content bounds
        if bbox:
            x1, y1, x2, y2 = bbox
            image = image.crop((x1, y1, x2, y2))
        elif alpha.any():
            rows, cols = np.any(alpha, axis=1), np.any(alpha, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            padding = 2
            ymin, ymax = max(0, ymin - padding), min(alpha.shape[0], ymax + padding)
            xmin, xmax = max(0, xmin - padding), min(alpha.shape[1], xmax + padding)
            image = image.crop((xmin, ymin, xmax, ymax))

        return image

    def _process_mask(self, mask: np.ndarray, image: Image.Image, bbox: Optional[List[int]] = None) -> str:
        """Process a single mask and return base64 encoded PNG."""
        try:
            transparent_image = self._create_transparent_mask(mask, image, bbox)
            buffer = BytesIO()
            transparent_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        except Exception as e:
            print(f"Warning: Failed to process a segment: {str(e)}")
            return ""

    def segment_image(self, input_data: SegmentationInput) -> List[str]:
        """Segment image based on input parameters and return base64 encoded PNGs."""
        input_data.validate()
        segmented_images = []

        if input_data.mode == SegmentationMode.SEMANTIC and input_data.text_prompt:
            # Get all regions from OWLv2 at once
            regions = self._get_semantic_regions(input_data.image, input_data.text_prompt)
            if regions:
                # Batch process all bounding boxes with SAM
                results = self.sam_model.predict(input_data.image, bboxes=regions)
                masks_to_process = []
                bboxes_to_process = []

                for result, region in zip(results, regions):
                    if not hasattr(result, 'masks') or not result.masks or not hasattr(result.masks, 'data'):
                        continue
                    mask_data = result.masks.data.cpu().numpy() if isinstance(result.masks.data, torch.Tensor) else result.masks.data
                    mask_data = np.squeeze(mask_data)
                    if mask_data.ndim == 2:
                        masks_to_process.append(mask_data)
                        bboxes_to_process.append([int(x) for x in region])
                    elif mask_data.ndim == 3:
                        for m in mask_data:
                            masks_to_process.append(m)
                            bboxes_to_process.append([int(x) for x in region])

                # Parallel processing of masks
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self._process_mask, mask, input_data.image, bbox) 
                               for mask, bbox in zip(masks_to_process, bboxes_to_process)]
                    segmented_images = [f.result() for f in futures if f.result()]
        else:
            # Automatic segmentation on the whole image
            results = self.sam_model.predict(input_data.image)
            masks_to_process = []

            for result in results:
                if not hasattr(result, 'masks') or not result.masks or not hasattr(result.masks, 'data'):
                    continue
                mask_data = result.masks.data.cpu().numpy() if isinstance(result.masks.data, torch.Tensor) else result.masks.data
                mask_data = np.squeeze(mask_data)
                if mask_data.ndim == 2:
                    masks_to_process.append(mask_data)
                elif mask_data.ndim == 3:
                    masks_to_process.extend(m for m in mask_data)

            # Parallel processing of masks
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._process_mask, mask, input_data.image) for mask in masks_to_process]
                segmented_images = [f.result() for f in futures if f.result()]

        return segmented_images

def segment_image(
    image: Image.Image,
    text_prompt: Optional[str] = None,
    automatic: bool = False
) -> List[str]:
    """Convenience function to segment an image with the specified parameters."""
    segmenter = ImageSegmenter()
    input_data = SegmentationInput(
        image=image,
        text_prompt=text_prompt,
        automatic=automatic
    )
    return segmenter.segment_image(input_data)