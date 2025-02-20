#!/bin/bash

# Exit immediately if a command exits with a non-zero status, treat unset variables as an error, and fail on pipe errors.
set -euo pipefail

# Check if the input image exists
if [ ! -f "input_images/single_focus.jpg" ]; then
  echo "Error: input_images/single_focus.jpg not found."
  exit 1
fi

# Convert the image to base64 and remove any newline characters
IMAGE_BASE64=$(base64 "input_images/single_focus.jpg" | tr -d '\n')

# Construct the JSON test input. Here we specify the base64-encoded image and a text prompt for segmentation.
TEST_INPUT='{"input": {"image_base64": "'"$IMAGE_BASE64"'", "text_prompt": "Segment this image"}}'

# Write the test input to a file
echo "$TEST_INPUT" > test_input.json

# Run the Python handler with the test input. This assumes rp_handler.py is in the current directory.
python rp_handler.py --test_input_file test_input.json