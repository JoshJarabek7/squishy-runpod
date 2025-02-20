import runpod
from PIL import Image
from typing import AsyncGenerator, Dict, Any, Union
import base64
from io import BytesIO
import torch
import numpy as np
from numpy.typing import NDArray

from models import SegmentationInput, SegmentationOutput
from complete import ImageSegmenter

# Initialize segmenter once for all requests
segmenter = ImageSegmenter()

def convert_to_numpy(data: Union[torch.Tensor, NDArray, Any]) -> NDArray:
    """Convert various data types to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)

async def process_segments(input_data: SegmentationInput) -> AsyncGenerator[Dict[str, Any], None]:
    """Process segments and yield them one by one."""
    try:
        # Validate input mode
        input_data.validate_mode()
        
        # Get PIL image
        image = input_data.get_pil_image()
        
        # Get segments based on mode
        if input_data.mode == "semantic" and input_data.text_prompt:
            # Get all regions from OWLv2 at once
            regions = segmenter._get_semantic_regions(image, input_data.text_prompt)
            if regions:
                # Process each region
                results = segmenter.sam_model.predict(image, bboxes=regions)
                for result, region in zip(results, regions):
                    if not hasattr(result, 'masks') or not result.masks or not hasattr(result.masks, 'data'):
                        continue
                    
                    # Process each mask
                    mask_data = convert_to_numpy(result.masks.data)
                    mask_data = mask_data.squeeze()
                    
                    # Handle both 2D and 3D masks
                    if mask_data.ndim == 2:
                        masks = [mask_data]
                    else:
                        masks = [m for m in mask_data]
                    
                    # Process each mask
                    for mask in masks:
                        transparent_image = segmenter._create_transparent_mask(mask, image, [int(x) for x in region])
                        if transparent_image:
                            try:
                                output = SegmentationOutput.from_pil_image(transparent_image)
                                yield {"status": "processing", "output": output.dict()}
                            except ValueError as e:
                                print(f"Warning: Skipping segment due to size constraints: {e}")
                                continue
        else:
            # Automatic segmentation
            results = segmenter.sam_model.predict(image)
            for result in results:
                if not hasattr(result, 'masks') or not result.masks or not hasattr(result.masks, 'data'):
                    continue
                
                # Process each mask
                mask_data = convert_to_numpy(result.masks.data)
                mask_data = mask_data.squeeze()
                
                # Handle both 2D and 3D masks
                if mask_data.ndim == 2:
                    masks = [mask_data]
                else:
                    masks = [m for m in mask_data]
                
                # Process each mask
                for mask in masks:
                    transparent_image = segmenter._create_transparent_mask(mask, image)
                    if transparent_image:
                        try:
                            output = SegmentationOutput.from_pil_image(transparent_image)
                            yield {"status": "processing", "output": output.dict()}
                        except ValueError as e:
                            print(f"Warning: Skipping segment due to size constraints: {e}")
                            continue
        
        # Signal completion
        yield {"status": "completed"}
        
    except Exception as e:
        yield {"status": "error", "error": str(e)}

async def handler(event) -> Dict[str, Any]:
    """RunPod handler function."""
    try:
        # Parse and validate input
        input_data = SegmentationInput(**event["input"])
        
        # Return generator
        return runpod.serverless.stream(process_segments(input_data))
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler}) 