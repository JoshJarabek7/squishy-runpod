#!/usr/bin/env python3.13
import base64
from PIL import Image
import json
import os

def image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
        # Save to bytes
        from io import BytesIO
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

# Create test input
input_data = {
    "input": {
        "image_base64": image_to_base64("input_images/goats.jpg"),
        "text_prompt": "goat",
        "automatic": False
    }
}

# Save to test_input.json
with open("test_input.json", "w") as f:
    json.dump(input_data, f, indent=2)

print("Created test_input.json. Now you can run:")
print("python rp_handler.py --rp_serve_api")
print("\nOr for synchronous testing:")
print("python rp_handler.py") 