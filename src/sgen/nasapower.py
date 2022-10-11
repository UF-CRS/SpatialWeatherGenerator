"""
Interface for NASA Power data.
"""
import io
import time
import json
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import datetime as dt
import requests
import tqdm

from rasterio.features import geometry_mask
from shapely.geometry import Polygon, box, Point


# Use whole year start and end dates for calculating historical statistics
NASAPOWER_START = dt.date(1984, 1, 1)
NASAPOWER_END = dt.date(dt.datetime.today().year - 1, 12, 31)

# Precipitation, T_max, T_min, Solar radiation
# Unites mm/d, celcius, celcius, MJ/m2/d
SGEN_PARAMS_IN_POWER = ["PRECTOTCORR", "T2M_MAX", "T2M_MIN", "ALLSKY_SFC_SW_DWN"]


def get_NASA_Power_data_for_region(region: Polygon) -> xr.Dataset:

    dataset = build_POWER_dataset_for_region(region)

    # Need to then run independent single point queries to get elevations of each grid
    dataset = build_elevation_grid(dataset)

    # Mask pixels that are within region bounding box but not within region polygon
    dataset = mask_to_region(dataset, region)

    return dataset


def build_elevation_grid(dataset: xr.Dataset) -> xr.DataArray:
    elevation = xr.zeros_like(dataset["PRECTOTCORR"])
    for lat in elevation.lat:
        for lon in elevation.lon:
            elev_grid = get_elevation_for_NASA_POWER_grid_square(float(lat), float(lon))
            elevation.loc[dict(lat=lat, lon=lon)] = elev_grid
            time.sleep(1)  # Max 60 requests per minute, this is so we don't get blocked
    dataset = dataset.assign(ELEV=elevation)
    return dataset


def mask_to_region(dataset: xr.Dataset, region: Polygon) -> xr.DataArray:
    region_mask = xr.zeros_like(dataset["PRECTOTCORR"])
    region_mask = region_mask.assign_attrs(
        long_name="In Region Mask", standard_name="Region Mask", units="bool"
    )
    # Do this crudely for now
    for lat in region_mask.lat:
        for lon in region_mask.lon:
            point = Point(lon, lat)
            # Use square buffer to get pixels with centroids outside region but which
            # cover an ismuth of the region
            region_mask.loc[{"lat": lat, "lon": lon}] = point.buffer(0.25, 1).intersects(region)
    dataset = dataset.assign(REGION=region_mask.astype(bool))
    return dataset


def build_POWER_dataset_for_region(region: Polygon) -> xr.Dataset:
    # NASA Power limits request to 10 deg boxes, so split region into sub-regions
    subregions = split_region_into_10_deg_boxes(region)

    print("Requesting NASA Power data.")
    subregions_sets = []
    for idx, subregion in enumerate(subregions):
        print(f"On {idx+1} / {len(subregions)}.")
        subregion_data = request_all_data_from_POWER_for_region(subregion)
        subregions_sets.append(subregion_data)

    region_dataset = xr.merge(subregions_sets)

    return region_dataset


def request_all_data_from_POWER_for_region(region: Polygon) -> xr.Dataset:

    year_sets = []
    for year in tqdm.tqdm(list(range(NASAPOWER_START.year, NASAPOWER_END.year + 1))):
        year_dataset = request_POWER_region(
            region, dt.date(year, 1, 1), dt.date(year, 12, 31), SGEN_PARAMS_IN_POWER
        )
        time.sleep(1)  # Max 60 requests per minute, this is so we don't get blocked
        year_sets.append(year_dataset)

    return xr.concat(year_sets, dim="time")


def split_region_into_10_deg_boxes(region: Polygon) -> list[Polygon]:
    min_lon, min_lat, max_lon, max_lat = region.bounds

    if (max_lon - min_lon <= 10) and (max_lat - min_lat <= 10):
        return [region]

    subregions = []
    prior_lon = min_lon
    prior_lat = min_lat
    for lon in np.arange(min_lon + 10, max_lon + 10, 10):
        if lon > max_lon:
            lon = max_lon
        for lat in np.arange(min_lat + 10, max_lat + 10, 10):
            if lat > max_lat:
                lat = max_lat
            subregions.append(box(prior_lon, prior_lat, lon, lat))
            prior_lat = lat
        prior_lon = lon
    return subregions


def request_POWER_region(
    region: Polygon, start_date: dt.date, end_date: dt.date, parameters: list[str]
) -> xr.Dataset:
    """
    Retrieve daily values for a region NASA POWER.

    Region will be requested by its bounding box.

    See: https://power.larc.nasa.gov/api/pages/?urls.primaryName=Daily

    Notes
    ---
    Only 20 parameters can be requested at a time.
    All values are taken at 2m reference point.
    """
    if len(parameters) > 20:
        raise RuntimeError("NASA POWER allows only 20 parameters at a time.")

    power_url = "https://power.larc.nasa.gov/api/temporal/daily/regional?"
    lon_min, lat_min, lon_max, lat_max = region.bounds
    if np.abs(lon_max - lon_min) < 2 or np.abs(lat_max - lat_min) < 2:
        # NASA POWER region request requires at least 2 degs in lat and lon
        buffr = max(np.abs(lon_max - lon_min), np.abs(lat_max - lat_min))
        lon_min, lat_min, lon_max, lat_max = region.buffer(buffr).bounds

    payload = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "latitude-min": lat_min,
        "latitude-max": lat_max,
        "longitude-min": lon_min,
        "longitude-max": lon_max,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "netcdf",
    }

    r = requests.get(power_url, params=payload)

    xarray_io = io.BytesIO(r.content)

    dataset = xr.open_dataset(xarray_io)

    return dataset


def get_elevation_for_NASA_POWER_grid_square(lat: float, lon: float) -> float:
    # Format a simple request and get elevation from header data
    request_url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        "start=20200101&end=20200102&"
        f"latitude={lat}&longitude={lon}&community=ag&parameters="
        "T2M_MAX&format=json&header=true"
    )
    r = requests.get(request_url)
    response = json.loads(r.content)
    elevation = response["geometry"]["coordinates"][2]

    return elevation
