#!/usr/bin/env python3.13
import os
from pathlib import Path
from PIL import Image
from utils import segment_image, initialize_models
import base64
from io import BytesIO


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def save_segmented_images(
    base64_strings: list[str], output_dir: str, prefix: str
) -> None:
    """Save base64 encoded images to files."""
    for idx, b64_string in enumerate(base64_strings):
        if not b64_string:  # Skip empty strings
            continue

        # Decode base64 string
        image_data = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_data))

        # Save image
        output_path = os.path.join(output_dir, f"{prefix}_{idx}.png")
        image.save(output_path)
        print(f"Saved: {output_path}")


def main():
    # Setup directories
    input_dir = "input_images"
    output_dir = "output_images"
    ensure_dir(input_dir)
    ensure_dir(output_dir)

    # Image configuration
    image_file = "goats.jpg"
    input_path = os.path.join(input_dir, image_file)

    # Check if image exists
    if not os.path.exists(input_path):
        print(f"\nError: {image_file} not found in the input directory.")
        print(f"Please add {image_file} to the 'input_images' directory before running the script.")
        return

    print(f"\nProcessing {image_file}...")

    try:
        # Initialize models
        initialize_models()
        
        # Load and process image
        image = Image.open(input_path).convert("RGB")
        image_base64 = image_to_base64(image)

        # Run semantic segmentation for "goat"
        print("Running semantic segmentation for 'goat'...")
        semantic_results = segment_image(image_base64, text_prompt="goat")
        if semantic_results:
            save_segmented_images(semantic_results, output_dir, "goats_semantic")
        else:
            print("No segments found for keyword 'goat'")

    except Exception as e:
        print(f"Error processing {image_file}: {str(e)}")


if __name__ == "__main__":
    try:
        main()
        print("\nProcessing complete!")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
