#!/usr/bin/env bash

################################################################################
# Script Name: download_files.sh
# Description: This script downloads files listed in a text file using wget,
#              allowing for parallel downloads. It accepts optional command-
#              line arguments to specify the download directory and the file
#              containing the list of paths to download.
# Usage:       ./download_files.sh [-d <download_directory>] [-f <list_of_files>]
# Author:      Luke Gloege (lucas.gloege@yale.edu)
# Date:        2024-05-16
################################################################################

# Default values (uncomment to use)
#download_directory="../raw/"
#file="./links.txt"

# Number of lines to process concurrently
batch_size=100

# Define usage function
usage() {
    echo "Usage: $0 [-d <download_directory>] [-f <list_of_paths_file>]"
    exit 1
}

# Parse command-line options
while getopts ":d:f:" opt; do
    case ${opt} in
        d ) download_directory=$OPTARG ;;
        f ) file=$OPTARG ;;
        \? ) usage ;;
        : ) usage ;;
    esac
done

# Check if download directory and file are provided
if [ -z "$download_directory" ] || [ -z "$file" ]; then
    usage
fi

# Download file and send command to background
line_count=0
while IFS= read -r path; do
    # Increment line counter
    ((line_count++))
    
    wget -P "$download_directory" "$path" &

    # Check if batch size is reached
    if [ "$line_count" -eq "$batch_size" ]; then
        # Wait for batch to finish
        wait
        
        # Reset line counter
        line_count=0
    fi
done < "$file"

wait
