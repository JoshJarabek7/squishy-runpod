from ultralytics import YOLOWorld
import os
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def detect_objects():
    # Initialize models
    model = YOLOWorld('yolov8x-worldv2.pt')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Move CLIP model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model.to(device)  # type: ignore
    
    # Define the image-to-keyword mapping
    image_keywords = {
        'multi_focus.jpg': ['book'],
        'single_focus.jpg': ['head']
    }
    
    # Create output directory
    output_dir = 'output_detections'
    ensure_dir(output_dir)
    
    # Process each image
    for image_file, keywords in image_keywords.items():
        input_path = os.path.join('input_images', image_file)
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue
            
        print(f"Processing {image_file} for keywords: {keywords}")
        
        # Set the classes for detection
        model.set_classes(keywords)
        
        # Run YOLO-World detection
        results = model.predict(
            source=input_path,
            save=True,  # Save the annotated image
            save_dir=output_dir,  # Directory to save results
            conf=0.25,  # Confidence threshold
            save_txt=True  # Save detection results in YOLO format
        )
        
        # Load original image for CLIP verification
        image = Image.open(input_path)
        
        # Process results
        for r in results:
            if r.boxes is not None:
                if isinstance(r.boxes.data, torch.Tensor):
                    boxes = r.boxes.data.cpu().numpy()
                else:
                    boxes = r.boxes.data
                print(f"Found {len(boxes)} potential objects in {image_file}")
                
                # Process each detection
                for idx, box in enumerate(boxes):
                    yolo_conf = float(box[4])  # Confidence is in the 5th position
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Crop the region
                    cropped_region = image.crop((x1, y1, x2, y2))
                    
                    # Prepare text inputs for CLIP
                    text_inputs = [f"a photo of a {keyword}" for keyword in keywords]
                    
                    # Process image and text with CLIP
                    inputs = clip_processor(
                        images=cropped_region,
                        text=text_inputs,
                        return_tensors="pt",
                        padding=True
                    )  # type: ignore
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    # Get CLIP similarity scores
                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        image_features = outputs.image_embeds
                        text_features = outputs.text_embeds
                        similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                        clip_conf = float(similarity.max())
                    
                    # Combined confidence score
                    combined_conf = (yolo_conf + clip_conf) / 2
                    
                    print(f"- Detection {idx + 1}:")
                    print(f"  YOLO confidence: {yolo_conf:.2f}")
                    print(f"  CLIP confidence: {clip_conf:.2f}")
                    print(f"  Combined confidence: {combined_conf:.2f}")
                    
                    # Save the cropped region
                    crop_path = os.path.join(output_dir, f"{image_file[:-4]}_crop_{idx}.jpg")
                    cropped_region.save(crop_path)

if __name__ == "__main__":
    detect_objects() 