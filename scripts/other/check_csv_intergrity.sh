#!/bin/bash

# Filename: check_gzip.sh
# Purpose: Check for errors in a large gzip file

# File to check
gzip_file="results/Varsha/GPT4/all_results_04_29.csv.tar.gz"

# Check gzip file integrity
gzip -t "$gzip_file"
if [ $? -ne 0 ]; then
  echo "gzip file is corrupted"
  exit 1
fi

# Split file and check each part
split -b 100M "$gzip_file" part_

for part in part_*; do
  gunzip -c "$part" > /dev/null
  if [ $? -ne 0 ]; then
    echo "Error in $part"
    break
  fi
done

# Check file line by line
zcat "$gzip_file" | while IFS= read -r line; do
  echo "$line" > /dev/null
  if [ $? -ne 0 ]; then
    echo "Error at line: $line"
    break
  fi
done

# Clean up
rm part_*
