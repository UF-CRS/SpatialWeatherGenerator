"""
sgen - A spatial weather generator calibrated on NASA POWER data.

sgen is based on the following papers:
    Richardson 1981: 10.1029/WR017i001p00182
    Wilks 1998: 10.1016/S0022-1694(98)00186-3
    Wilks 1999: 10.1016/S0168-1923(99)00037-4
"""
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, Polygon

from sgen import stations, calibration, nasapower


def build_spatial_weather_generator(polygon: Polygon) -> stations.StationCollection:
    # Polygon should be in WGS84 projection with (lon, lat) points.
    historical_data = nasapower.get_NASA_Power_data_for_region(polygon)
    weather_stations = build_weather_stations_from_dataset(historical_data)
    station_collection = stations.StationCollection(weather_stations)
    interstation_calibrator = calibration.InterStationCalibrator()
    interstation_calibrator.calibrate_station_collection(station_collection)
    return station_collection


def build_weather_stations_from_dataset(dataset: xr.Dataset) -> gpd.GeoDataFrame:
    """
    Build a set of calibrated WeatherStations from a Dataset of NASA Power data.
    """
    print("Building weather stations for each pixel.")
    stations = []
    for lat in dataset.lat:
        for lon in dataset.lon:
            # Check to make sure pixel is within original region
            if not dataset["REGION"].sel(lat=lat, lon=lon).all():
                continue
            pixel_weatherdata = dataset.sel(lat=lat, lon=lon)
            pixel_weatherstation = build_weather_station_for_pixel(pixel_weatherdata)
            stations.append(pixel_weatherstation)

    weather_stations = gpd.GeoDataFrame(
        stations,
        geometry=[station.location for station in stations],
        columns=["station"],
    )
    return weather_stations


def build_weather_station_for_pixel(pixel_weatherdata) -> stations.WeatherStation:
    calibrator = calibration.StationCalibrator()
    station = stations.WeatherStation(
        Point(pixel_weatherdata.lon, pixel_weatherdata.lat),
        pixel_weatherdata["ELEV"].max(),
        pixel_weatherdata,
    )
    calibrator.calibrate_station(station)
    return station
