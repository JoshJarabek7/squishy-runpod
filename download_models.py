#!/usr/bin/env python3.13
from huggingface_hub import snapshot_download
from transformers import (
    Owlv2Processor,
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    PreTrainedTokenizer,
)
from transformers.models.owlv2 import Owlv2Processor as Owlv2ProcessorType
from ultralytics import SAM
import os
import shutil
from typing import cast


def print_dir_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            size = os.path.getsize(os.path.join(root, f)) / (1024*1024)
            print(f"{subindent}{f} ({size:.2f} MB)")


def verify_sam_path(model_path: str) -> bool:
    """Verify SAM model file exists and has correct size."""
    try:
        print(f"\nVerifying SAM model at {model_path}...")
        if not os.path.exists(model_path):
            print("✗ SAM model file not found")
            return False
            
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if size_mb < 100:  # SAM model should be several hundred MB
            print(f"✗ SAM model file seems too small: {size_mb:.2f} MB")
            return False
            
        print(f"✓ SAM model file exists and has reasonable size: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"✗ Error checking SAM model: {str(e)}")
        return False


def verify_owlv2_path(model_dir: str) -> bool:
    """Verify OWLv2 model files exist and have correct structure."""
    try:
        print(f"\nVerifying OWLv2 model at {model_dir}...")
        
        required_files = [
            "config.json",
            "preprocessor_config.json", 
            "pytorch_model.bin",
            "tokenizer_config.json",
            "special_tokens_map.json"
        ]
        
        missing_files = []
        total_size = 0
        
        for file in required_files:
            path = os.path.join(model_dir, file)
            if not os.path.exists(path):
                missing_files.append(file)
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                total_size += size_mb
                print(f"✓ Found {file} ({size_mb:.2f} MB)")
        
        # Check for vocabulary file (either vocab.txt or vocab.json)
        vocab_found = False
        for vocab_file in ["vocab.txt", "vocab.json"]:
            vocab_path = os.path.join(model_dir, vocab_file)
            if os.path.exists(vocab_path):
                size_mb = os.path.getsize(vocab_path) / (1024 * 1024)
                total_size += size_mb
                print(f"✓ Found {vocab_file} ({size_mb:.2f} MB)")
                vocab_found = True
                break
        if not vocab_found:
            missing_files.append("vocab file (vocab.txt or vocab.json)")

        if missing_files:
            print(f"✗ Missing required files: {', '.join(missing_files)}")
            return False

        if total_size < 100:  # OWLv2 model should be several GB
            print(f"✗ Total model size seems too small: {total_size:.2f} MB")
            return False

        print(f"✓ All required OWLv2 files present, total size: {total_size:.2f} MB")
        return True
    except Exception as e:
        print(f"✗ Error checking OWLv2 model: {str(e)}")
        return False


def main():
    # Create base directories
    base_dir = os.path.expanduser("~/squishy-models")
    sam_dir = os.path.join(base_dir, "sam")
    hf_dir = os.path.join(base_dir, "huggingface")
    
    os.makedirs(sam_dir, exist_ok=True)
    os.makedirs(hf_dir, exist_ok=True)
    
    print(f"Using base directory: {base_dir}")
    
    # Download and save SAM model
    print("\nDownloading SAM model...")
    target_path = os.path.join(sam_dir, "sam2.1_l.pt")
    
    # Create SAM model instance which will trigger download to default location
    model = SAM("sam2.1_l.pt")
    
    # Determine the correct location of the downloaded SAM model
    downloaded_file = os.path.join(os.getcwd(), "sam2.1_l.pt")
    if not os.path.exists(downloaded_file):
        # Fallback to the default cache directory
        cache_dir = os.path.expanduser("~/.cache/ultralytics")
        downloaded_file = os.path.join(cache_dir, "sam2.1_l.pt")
    
    # Move the downloaded file to our target location
    if os.path.exists(downloaded_file):
        shutil.copy2(downloaded_file, target_path)  # Using copy2 instead of move for testing
        print(f"SAM model copied to {target_path}")
    else:
        raise RuntimeError("SAM model not found in expected locations")

    # Verify the download
    if os.path.exists(target_path):
        size = os.path.getsize(target_path) / (1024 * 1024)
        print(f"SAM model downloaded to {target_path} ({size:.2f} MB)")
    else:
        print(f"Error: Failed to download SAM model to {target_path}")
        return

    # Download complete OWLv2 model repository
    print("\nDownloading OWLv2 model...")
    model_id = "google/owlv2-large-patch14"
    local_dir = os.path.join(hf_dir, "owlv2-large-patch14")

    # First download using snapshot
    print(f"Downloading complete model repository to {local_dir}")
    snapshot_download(
        repo_id=model_id, 
        local_dir=local_dir
    )

    # Then save model and processor properly
    print("Loading and saving processor...")
    processor_result = Owlv2Processor.from_pretrained(
        model_id,
        local_files_only=False  # Allow download during build
    )

    # Handle processor components
    if isinstance(processor_result, tuple) and len(processor_result) == 2:
        text_processor = cast(PreTrainedTokenizer, processor_result[0])
        image_processor = cast(Owlv2ImageProcessor, processor_result[1])

        # Save components separately
        text_processor.save_pretrained(os.path.join(local_dir, "text_processor"))
        image_processor.save_pretrained(os.path.join(local_dir, "image_processor"))
    else:
        processor = cast(Owlv2ProcessorType, processor_result)
        processor.save_pretrained(local_dir)

    print("Loading and saving model...")
    model = Owlv2ForObjectDetection.from_pretrained(
        model_id,
        local_files_only=False  # Allow download during build
    )
    model.save_pretrained(local_dir)

    print("\nFinal directory structure:")
    print_dir_structure(base_dir)

    # Verify paths and files without loading models
    sam_ok = verify_sam_path(target_path)
    owlv2_ok = verify_owlv2_path(local_dir)

    if sam_ok and owlv2_ok:
        print("\n✅ All model files downloaded and verified successfully!")
        print("\nPaths that will be used in rp_handler.py:")
        print(f"SAM_MODEL_PATH = {target_path}")
        print(f"LOCAL_MODEL_PATH = {local_dir}")
        print("\nTo use these models in the container:")
        print(f"1. Copy {target_path} to /models/sam/sam2.1_l.pt")
        print(f"2. Copy {local_dir} to /models/huggingface/owlv2-large-patch14")
    else:
        print("\n❌ Some model files are missing or invalid")


if __name__ == "__main__":
    main()
