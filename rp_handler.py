#!/usr/bin/env python3.13
import runpod
from typing import AsyncGenerator, Any
from utils import (
    SegmentationInput,
    SegmentationOutput,
    ImageSegmenter,
    verify_model_paths,
    initialize_models,
)
import torch
import numpy as np

# Verify model paths before initializing
verify_model_paths()
initialize_models()
segmenter = ImageSegmenter()


async def process_segments(
    input_data: SegmentationInput,
) -> AsyncGenerator[dict[str, Any], None]:
    """Process segments and yield them one by one."""
    try:
        image = input_data.get_pil_image()

        if input_data.mode == "semantic" and input_data.text_prompt:
            regions = segmenter._get_semantic_regions(image, input_data.text_prompt)
            if regions:
                for region in regions:
                    # Convert region coordinates to integers and crop the image
                    region_int = [int(x) for x in region]
                    cropped_img = image.crop((region_int[0], region_int[1], region_int[2], region_int[3]))
                    
                    # Run SAM on the cropped image
                    results = segmenter.sam_model.predict(cropped_img)
                    
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
                        transparent_image = segmenter._create_transparent_mask(largest_mask, cropped_img)
                        if transparent_image:
                            try:
                                output = SegmentationOutput.from_pil_image(transparent_image)
                                yield {
                                    "status": "processing",
                                    "output": output.model_dump(),
                                }
                            except ValueError as e:
                                print(f"Warning: Skipping segment due to size constraints: {e}")
        else:
            # Automatic mode - process the full image
            results = segmenter.sam_model.predict(image)
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
                transparent_image = segmenter._create_transparent_mask(largest_mask, image)
                if transparent_image:
                    try:
                        output = SegmentationOutput.from_pil_image(transparent_image)
                        yield {
                            "status": "processing",
                            "output": output.model_dump(),
                        }
                    except ValueError as e:
                        print(f"Warning: Skipping segment due to size constraints: {e}")

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
