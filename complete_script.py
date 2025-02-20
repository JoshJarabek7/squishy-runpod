import os
from pathlib import Path
from PIL import Image
from rp_handler import segment_image


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_segmented_images(
    base64_strings: list[str], output_dir: str, prefix: str
) -> None:
    """Save base64 encoded images to files."""
    import base64
    from io import BytesIO

    for idx, b64_string in enumerate(base64_strings):
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

    # Image configurations
    image_configs = {
        "multi_focus.jpg": {"semantic": ["book"], "output_prefix": "multi_focus"},
        "single_focus.jpg": {
            "semantic": ["man", "hair", "sunglasses"],
            "output_prefix": "single_focus",
        },
        "single_focus_cropped.png": {
            "semantic": ["man", "hair", "sunglasses"],
            "output_prefix": "single_focus_cropped",
        },
    }

    # Check if images exist
    missing_images = []
    for image_file in image_configs:
        if not os.path.exists(os.path.join(input_dir, image_file)):
            missing_images.append(image_file)

    if missing_images:
        print("\nWarning: The following images are missing from the input directory:")
        for img in missing_images:
            print(f"- {img}")
        print(
            "\nPlease add these images to the 'input_images' directory before running the script."
        )
        return

    # Process each image
    for image_file, config in image_configs.items():
        input_path = os.path.join(input_dir, image_file)
        print(f"\nProcessing {image_file}...")

        try:
            image = Image.open(input_path).convert("RGB")

            # 1. Automatic segmentation
            print("Running automatic segmentation...")
            auto_results = segment_image(image, automatic=True)
            if auto_results:
                save_segmented_images(
                    auto_results, output_dir, f"{config['output_prefix']}_auto"
                )
            else:
                print("No segments found in automatic mode")

            # 2. Semantic segmentation for each keyword
            for keyword in config["semantic"]:
                print(f"Running semantic segmentation for '{keyword}'...")
                semantic_results = segment_image(image, text_prompt=keyword)
                if semantic_results:
                    save_segmented_images(
                        semantic_results,
                        output_dir,
                        f"{config['output_prefix']}_{keyword}",
                    )
                else:
                    print(f"No segments found for keyword '{keyword}'")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue


if __name__ == "__main__":
    try:
        main()
        print("\nProcessing complete!")
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
