import os
from pathlib import Path
from ultralytics import SAM
import shutil
import torch

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def segment_images(input_dir='input_images', output_base_dir='output_images'):
    # Load SAM 2.1 large model
    model = SAM('sam2.1_l.pt')
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        print(f"Processing {image_file}...")
        
        # Create output directory for this image
        image_name = os.path.splitext(image_file)[0]
        output_dir = os.path.join(output_base_dir, image_name)
        ensure_dir(output_dir)
        
        # Full path to input image
        input_path = os.path.join(input_dir, image_file)
        
        try:
            # Run segmentation - this will segment everything in the image
            results = model(input_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            # Save each segmented mask as a separate image
            for idx, result in enumerate(results):
                # Save original image with mask overlay
                result.save_crop(save_dir=output_dir, file_name=Path(f'segment_{idx}.png'))
                
                # Save the mask itself
                if hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[0]
                    result.masks.save_masks(os.path.join(output_dir, f'mask_{idx}.png'))
                
            # Copy original image to output directory
            shutil.copy2(input_path, os.path.join(output_dir, image_file))
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

if __name__ == "__main__":
    segment_images()
    print("Processing complete!") 