from datetime import datetime, timedelta

# Base URL for the file path
base_url = "https://dap.ceda.ac.uk/neodc/eocis/data/global_and_regional/sea_surface_temperature/CDR_v3/Analysis/L4/v3.0.1"

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
    url = f"{base_url}/{year}/{month}/{day}/{timestamp}-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR3.0-v02.0-fv01.0.nc"

    # 2022 has slightly different URL
    if str(year) == str(2022):
        url = f"{base_url}/{year}/{month}/{day}/{timestamp}-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_ICDR3.0-v02.0-fv01.0.nc"
        
    links.append(url)
    
    # Increment the date by one day
    current_date += timedelta(days=1)

# Write links to a file
output_file = "./url-list.txt"
with open(output_file, "w") as f:
    f.write("\n".join(links))

print(f"Links generated and saved to {output_file}")
