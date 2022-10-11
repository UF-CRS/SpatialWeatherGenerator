"""
Integration tests for modules.
"""
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pytest

import sgen


@pytest.fixture
def nasapower_dataset(datadir, zona_code):
    return xr.open_dataset(datadir / f"Zona_{zona_code}_NASAPOWER.nc")


@pytest.mark.slow
def test_build_weather_stations_from_dataset(nasapower_dataset, zona_code):
    built_stations = sgen.build_weather_stations_from_dataset(nasapower_dataset)
    built_stations.to_pickle(
        f"test_calibration/Zona_{zona_code}_stations.pkl", protocol=-1
    )

    for idx, station in built_stations.iterrows():
        assert station["station"].calibrated
        assert station["station"].ppt_amount_params.shape == (12, 3)
        lon, lat = station.geometry.xy
        assert lon in nasapower_dataset.lon
        assert lat in nasapower_dataset.lat


@pytest.fixture
def zona_region(datadir, zona_code):
    zonas = gpd.read_file(datadir / "Bolsa_de_Cereales_Zonas.geojson")
    return zonas[zonas["Zona"] == zona_code].iloc[0].geometry


@pytest.mark.slow
def test_build_spatial_weather_generator(zona_region):
    station_collection = sgen.build_spatial_weather_generator(zona_region)
    station_collection.simulate_weather(dt.date(2020, 1, 1), 2000)
