#!/usr/bin/env bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file> <output_directory> <start_year> <end_year>"
    exit 1
fi

# Assign arguments to variables
input_file="$1"
output_directory="$2"
start_year="$3"
end_year="$4"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "Input file '$input_file' not found."
    exit 1
fi

# Check if the output directory exists
if [ ! -d "$output_directory" ]; then
    echo "Output directory '$output_directory' not found."
    exit 1
fi

# Check if start_year and end_year are valid
if ! [[ "$start_year" =~ ^[0-9]{4}$ && "$end_year" =~ ^[0-9]{4}$ ]]; then
    echo "Invalid start_year or end_year format. Please provide four-digit years."
    exit 1
fi

# Set the pattern for the year range
year_pattern=$(seq -s'|' $start_year $end_year)

# Filter and download the URLs
awk "/$year_pattern/" "$input_file" | wget -i - -P "$output_directory"

