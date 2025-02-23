#!/usr/bin/env python3.13
import json
import base64
import requests
from pathlib import Path
import argparse

def save_segments(mode: str):
    """Save segments from the API response."""
    # Setup paths
    output_dir = Path("/squishy-runpod/output_images")
    output_dir.mkdir(exist_ok=True)
    
    # Clear existing files for this mode
    for f in output_dir.glob(f"{mode}_segment_*.png"):
        f.unlink()
    
    # Load the appropriate input file
    input_file = Path(f"test_input_{mode}.json")
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    # Send request
    print(f"Sending {mode} request...")
    with input_file.open() as f:
        input_data = json.load(f)
    
    response = requests.post(
        "http://localhost:8000/runsync",
        json=input_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print(f"Error: Got status code {response.status_code}")
        print("Response:", response.text)
        return
        
    try:
        # Parse the RunPod job response
        job_response = response.json()
        
        if job_response["status"] != "COMPLETED":
            print(f"Error: Job status is {job_response['status']}")
            if "error" in job_response:
                print("Error:", job_response["error"])
            return
            
        # Get the output array
        results = job_response["output"]
        
        # Process each result
        counter = 0
        for result in results:
            # Skip non-processing results (like the final "completed" status)
            if result.get("status") != "processing" or "output" not in result:
                continue
                
            # Get base64 image data
            img_data = result["output"]["image_base64"]
            
            # Save image
            output_file = output_dir / f"{mode}_segment_{counter}.png"
            with output_file.open("wb") as f:
                f.write(base64.b64decode(img_data))
            print(f"Saved segment {counter} to {output_file}")
            counter += 1
            
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        print(f"Response text: {response.text[:500]}...")  # Print first 500 chars of response
        return
    
    print(f"\nProcessed {counter} images")
    print(f"Files in {output_dir}:")
    for f in sorted(output_dir.glob(f"{mode}_segment_*.png")):
        print(f"  {f.name} ({f.stat().st_size} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Test segmentation API")
    parser.add_argument("mode", choices=["semantic", "automatic"], help="Segmentation mode to test")
    args = parser.parse_args()
    
    save_segments(args.mode)

if __name__ == "__main__":
    main() 