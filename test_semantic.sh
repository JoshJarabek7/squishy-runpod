#!/bin/bash

# Set absolute path for output directory
OUTPUT_DIR="/squishy-runpod/output_images"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Created output directory at ${OUTPUT_DIR}"

# First clear any existing files
rm -f "${OUTPUT_DIR}"/semantic_segment_*.png

# Send request and process response directly
echo "Sending request to server..."
counter=0

# Save response to temporary file
TEMP_RESP=$(mktemp)
curl -X POST http://localhost:8000/runsync \
     -H "Content-Type: application/json" \
     -d @test_input_semantic.json > "$TEMP_RESP"

# Process each line
while IFS= read -r line; do
    if [[ $line == *"DEBUG"* && $line == *"image_base64"* ]]; then
        # Extract everything between image_base64":"..." up to the next quote
        if [[ $line =~ image_base64\":\"([^\"]+) ]]; then
            base64_img="${BASH_REMATCH[1]}"
            output_file="${OUTPUT_DIR}/semantic_segment_${counter}.png"
            printf "%s" "$base64_img" | base64 -d > "$output_file"
            echo "Saved segment $counter to $output_file"
            counter=$((counter + 1))
        fi
    fi
done < "$TEMP_RESP"

# Cleanup
rm -f "$TEMP_RESP"

echo "Processed $counter images"
echo "Files in ${OUTPUT_DIR}:"
ls -l "${OUTPUT_DIR}"/semantic_segment_*.png 