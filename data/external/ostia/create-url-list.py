from datetime import datetime, timedelta

# Base URL for the file path
base_url = "https://opendap.earthdata.nasa.gov/collections/C2586786218-POCLOUD/granules"

# Start and end dates
start_date = datetime(1993, 1, 1)
end_date = datetime(2022, 12, 31)

# Generate links
current_date = start_date
links = []

while current_date <= end_date:
    year = current_date.strftime("%Y")
    month = current_date.strftime("%m")
    day = current_date.strftime("%d")
    timestamp = current_date.strftime("%Y%m%d120000")
    
    # Construct the URL
    url = f"{base_url}/{timestamp}-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB_REP-v02.0-fv02.0.dap.nc4"
    links.append(url)
    
    # Increment the date by one day
    current_date += timedelta(days=1)

# Write links to a file
output_file = "./url-list.txt"
with open(output_file, "w") as f:
    f.write("\n".join(links))

print(f"Links generated and saved to {output_file}")
