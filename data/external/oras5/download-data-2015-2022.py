import cdsapi

dataset = "reanalysis-oras5"
request = {
    "product_type": ["operational"],
    "vertical_resolution": "single_level",
    "variable": ["sea_surface_salinity"],
    "year": [
        "2015", "2016", "2017",
        "2018", "2019", "2020",
        "2021", "2022"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

