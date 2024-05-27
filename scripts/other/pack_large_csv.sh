#!/bin/bash

# Filename: pack_large_csv.sh
# Purpose: Compress each CSV file larger than 100MB into its own tar.gz file, unstage the original CSV, add the new tar.gz to staging, and update .gitignore

# Find CSV files larger than 100MB
input_files=$(find . -type f -name '*.csv' -size +100M)

# Check if any files were found
if [[ -z "$input_files" ]]; then
    echo "No large CSV files found to pack."
    exit 0
fi

# Compress each file individually
for input_file in $input_files; do
    output_file="${input_file}.tar.gz"

    # Check the modification times for debugging
    if [ -f "$output_file" ]; then
        input_file_mtime=$(stat -c %y "$input_file")
        output_file_mtime=$(stat -c %y "$output_file")
        echo "Input File: $input_file, Last Modified: $input_file_mtime"
        echo "Output File: $output_file, Last Modified: $output_file_mtime"
    fi

    # Only proceed if the output file is older than the input file
    if [ -f "$output_file" ] && [ "$output_file" -nt "$input_file" ]; then
        echo "Skipping $input_file: tar.gz file is up-to-date."
        printf "\n"
        continue
    fi

    # Get original size in MB
    original_size=$(du -sh "$input_file" | cut -f1)

    # Create a tar.gz file for each, removing the './' prefix and adding .tar.gz
    tar -czvf "$output_file" "$input_file"

    # Get compressed size in MB
    compressed_size=$(du -sh "$output_file" | cut -f1)

    # Print original and compressed file sizes
    echo "Original size of $input_file: $original_size"
    echo "Compressed size of $output_file: $compressed_size"

    # Check if the file is tracked by Git
    if git ls-files --error-unmatch "$input_file" > /dev/null 2>&1; then
        # Unstage the original file
        git rm --cached "$input_file"
    else
        echo "$input_file is not tracked by Git."
    fi

    # Add the compressed file to git
    git add "$output_file"

    # Normalize file path and add to .gitignore if not already included
    normalized_path="${input_file#./}"
    if ! grep -q "^$normalized_path$" .gitignore; then
        echo "$normalized_path" >> .gitignore
        echo "Added $normalized_path to .gitignore."
    fi

    printf "\n"
done

echo "Packed all large CSV files into individual tar.gz files and updated Git staging."
