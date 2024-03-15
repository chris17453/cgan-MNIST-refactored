#!/bin/bash

files=$1
num_images=$2
base_dir=$(dirname "$files")

if [ -z "$num_images" ]; then
    num_images=1
fi

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
total_files=$(wc -l < "$files")
step_size=$(( total_files / num_images ))
echo "STEP SIZE:" $step_size
counter=0
current_file=-1
while IFS= read -r filename && [ $counter -lt $num_images ]; do
    if [ "$current_file" -eq -1 ] || [ "$current_file" -eq "$step_size" ]; then
        echo $current_file
        # Extract filename without extension
        file_name=$(basename "$filename" | cut -d. -f1)

        # Build the output file path
        output_file="$output_dir/${file_name}.png"
        echo "$output_file">>"$base_dir/annotated.txt"

        # Run the convert command to annotate and add a grey border
        convert "$filename" -bordercolor gray -border 30 -gravity South -pointsize 24 -fill orange \
               -pointsize 24 -fill orange -annotate +0+2 "$file_name" "$output_file"
        echo $file_name

        ((counter++))
        current_file=0
    fi

    ((current_file++))
done < "$files"

convert -delay 10 @"$base_dir/annotated.txt" "$base_dir/output.gif"

