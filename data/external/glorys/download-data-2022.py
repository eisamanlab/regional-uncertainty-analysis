import copernicusmarine

dataset_id = "cmems_mod_glo_phy_myint_0.083deg_P1M-m"

year = 2022

copernicusmarine.subset(
    credentials_file="/home/ljg48/.copernicusmarine/.copernicusmarine-credentials",  
    dataset_id=f"{dataset_id}",
    variables=["so"],
    minimum_longitude=-180,
    maximum_longitude=179.9166717529297,
    minimum_latitude=-80,
    maximum_latitude=90,
    start_datetime=f"{year}-01-01T00:00:00",
    end_datetime=f"{year}-12-31T00:00:00",
    minimum_depth=0.49402499198913574,
    maximum_depth=0.49402499198913574,
    file_format="netcdf",
    output_filename = f"{dataset_id}-{year}.nc",
    output_directory = "/home/ljg48/palmer_scratch/glorys/raw"
)
