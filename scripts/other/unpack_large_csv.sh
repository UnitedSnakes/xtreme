#!/bin/bash

# Filename: unpack_large_csv.sh
# Purpose: Unpack each tar.gz file to its original location, handle duplicates based on content, specifically for .csv.tar.gz files

# Find and process each .csv.tar.gz file
find . -type f -name "*.csv.tar.gz" | while IFS= read -r input_file; do
    # Determine the original CSV file name by removing .tar.gz
    output_file="${input_file%.tar.gz}"

    # Check the modification times for debugging
    if [ -f "$output_file" ]; then
        input_file_mtime=$(stat -c %y "$input_file")
        output_file_mtime=$(stat -c %y "$output_file")
        echo "Input File: $input_file, Last Modified: $input_file_mtime"
        echo "Output File: $output_file, Last Modified: $output_file_mtime"
    fi

    # Only proceed if the output file is older than the input file
    if [ -f "$output_file" ] && [ "$output_file" -nt "$input_file" ]; then
        echo "Skipping $input_file: csv file is up-to-date."
        printf "\n"
        continue
    fi

    # Check if the original CSV file exists
    if [ -f "$output_file" ]; then
        # Temporarily extract file for comparison
        temp_dir=$(mktemp -d)
        if tar -xzf "$input_file" -C "$temp_dir"; then
            extracted_file="$temp_dir/$output_file"

            # Compare files
            if [ -f "$extracted_file" ]; then
                if cmp -s "$extracted_file" "$output_file"; then
                    echo "Skipping '$output_file': file is identical."
                else
                    echo "File '$output_file' exists but is different."
                    echo "Differences:"
                    diff "$extracted_file" "$output_file"
                    echo -n "Do you want to replace it? [y/N] "

                    read -r answer < /dev/tty

                    if [[ "$answer" =~ ^[Yy]$ ]]; then
                        mv -f "$extracted_file" "$output_file"
                        echo "Replaced '$output_file'."
                    else
                        echo "Did not replace '$output_file'."
                    fi
                fi
            else

                echo "Failed to find the extracted file '$extracted_file'."
            fi
        else
            echo "Failed to extract '$output_file'. The file may be corrupted."
        fi
        # Clean up temporary directory
        rm -r "$temp_dir"
    else
        # If no original file exists, simply extract it
        if tar -xzvf "$input_file"; then
            echo "Extracted '$input_file' to '$output_file'."
        else
            echo "Failed to extract from '$input_file'. The file may be corrupted."
        fi
    fi

    printf "\n"

done < <(find . -type f -name "*.csv.tar.gz")
