"""
Tests for stations.py
"""
import pickle
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from shapely.geometry import Point

from sgen import stations
from sgen import calibration


@pytest.fixture
def nasapower_dataset(datadir, zona_code):
    return xr.open_dataset(datadir / f"Zona_{zona_code}_NASAPOWER.nc")


class TestWeatherStation:
    @pytest.fixture
    def single_pixel(self, nasapower_dataset):
        lat = nasapower_dataset.lat[0]
        lon = nasapower_dataset.lon[0]
        return nasapower_dataset.sel(lat=lat, lon=lon)

    @pytest.fixture
    def elevation(self, single_pixel):
        return single_pixel["ELEV"].mean()

    @pytest.fixture
    def station_location(self, nasapower_dataset):
        lat = nasapower_dataset.lat[0]
        lon = nasapower_dataset.lon[0]
        return Point(lon, lat)

    @pytest.fixture
    def station_calibrator(self):
        return calibration.StationCalibrator()

    @pytest.fixture
    def weather_station(
        self, station_location, single_pixel, station_calibrator, elevation
    ):
        station = stations.WeatherStation(station_location, elevation, single_pixel)
        station_calibrator.calibrate_station(station)
        return station

    @pytest.fixture
    def driving_random_variables(self):
        u = np.random.uniform(size=500)
        v = np.random.uniform(size=500)
        return [u, v]

    @pytest.fixture
    def long_driving_variables(self):
        u = np.random.uniform(size=100 * 365)
        v = np.random.uniform(size=100 * 365)
        return [u, v]

    def test_simulate_weather(self, weather_station, driving_random_variables):
        u, v = driving_random_variables
        start_date = dt.date(2020, 6, 1)
        simulated_weather = weather_station.simulate_weather(u, v, start_date)
        assert all(item in simulated_weather.columns for item in ["TMAX", "TMIN", "SRAD", "RAIN"])

    def test_simulated_rainfall_occurence_is_close_to_historical_mean(
        self, weather_station, long_driving_variables
    ):
        u, v = long_driving_variables
        start_date = dt.date(2020, 6, 1)
        end_date = start_date + pd.Timedelta(days=len(u) - 1)
        simulated_rainfall = weather_station.simulate_rainfall(u, v, start_date, None)
        simulated_rainfall = pd.Series(
            simulated_rainfall, index=pd.date_range(start_date, end_date)
        )
        simulated_rainfall[simulated_rainfall > 0] = 1
        simulated_rainfall_monthly = simulated_rainfall.resample("M").sum()
        simulated_rainfall_monthly_avg = simulated_rainfall_monthly.groupby(
            simulated_rainfall_monthly.index.month
        ).mean()
        station_historical = weather_station.historical_data
        # Set rainfall to zero or 1 to count wet and dry days
        station_historical = station_historical.where(station_historical['PRECTOTCORR'] == 0, 1)
        station_monthly = station_historical.resample(time="1MS").sum(dim="time")
        station_monthly_avg = station_monthly.groupby("time.month").mean("time")
        historical_rainfall_monthly_avg = station_monthly_avg["PRECTOTCORR"].to_dataframe()
        for month in simulated_rainfall_monthly_avg.index:
            sim_rain = simulated_rainfall_monthly_avg[month]
            historical_rain = historical_rainfall_monthly_avg.loc[month, 'PRECTOTCORR']
            assert (np.abs((historical_rain - sim_rain) / historical_rain) < 0.20)

    def test_simulated_rainfall_amount_is_close_to_historical_mean(
        self, weather_station, long_driving_variables
    ):
        u, v = long_driving_variables
        start_date = dt.date(2020, 6, 1)
        end_date = start_date + pd.Timedelta(days=len(u) - 1)
        simulated_rainfall = weather_station.simulate_rainfall(u, v, start_date, None)
        simulated_rainfall = pd.Series(
            simulated_rainfall, index=pd.date_range(start_date, end_date)
        )
        simulated_rainfall_monthly = simulated_rainfall.resample("M").sum()
        simulated_rainfall_monthly_avg = simulated_rainfall_monthly.groupby(
            simulated_rainfall_monthly.index.month
        ).mean()
        station_historical = weather_station.historical_data
        station_monthly = station_historical.resample(time="1MS").sum(dim="time")
        station_monthly_avg = station_monthly.groupby("time.month").mean("time")
        historical_rainfall_monthly_avg = station_monthly_avg["PRECTOTCORR"].to_dataframe()
        for month in simulated_rainfall_monthly_avg.index:
            print(month)
            sim_rain = simulated_rainfall_monthly_avg[month]
            historical_rain = historical_rainfall_monthly_avg.loc[month, 'PRECTOTCORR']
            # Assert difference is less than 20% or 3 mm (for very dry months)
            assert ((np.abs((historical_rain - sim_rain) / historical_rain) < 0.20) or (
                np.abs(historical_rain - sim_rain) < 2
            ))


class TestStationCollection:
    @pytest.fixture
    def calibrated_station_collection(self, datadir):
        with open(datadir / "Zona_V_calibrated_collection.pkl", "rb") as f:
            return pickle.load(f)

    def test_simulate_weather(self, calibrated_station_collection):
        start_date = dt.date(2020, 1, 1)
        weather = calibrated_station_collection.simulate_weather(start_date, 1000)
        print(weather["RAIN"].isel(time=0))
