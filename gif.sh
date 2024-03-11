#!/bin/bash
files=./output/20240311-144708/images.txt
base_dir=$(dirname "$files")

# Check if files.txt exists
if [ ! -f "$files" ]; then
    echo "Error: files.txt not found"
    exit 1
fi

# Create a directory for the annotated images
output_dir="$base_dir/annotated_images"

mkdir -p "$output_dir"

echo >"$base_dir/annotated.txt"

# Read filenames from files.txt and process each file
while IFS= read -r filename; do
    # Extract filename without extension
    file_name=$(basename "$filename" | cut -d. -f1)

    # Build the output file path
    output_file="$output_dir/${file_name}.png"
    echo "$output_file">>"$base_dir/annotated.txt"

    # Run the convert command to annotate and add a grey border
    convert "$filename" -bordercolor gray -border 30 -gravity South -pointsize 24 -fill orange \
           -pointsize 24 -fill orange -annotate +0+2 "$file_name" "$output_file"
    echo $file_name
done < "$files"

convert -delay 10 @"$base_dir/annotated.txt" "$base_dir/output.gif"
