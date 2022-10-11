"""
Test calibration of stations
"""
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import matplotlib.pyplot as plt
from shapely.geometry import Point

from sgen import calibration
from sgen import stations


@pytest.fixture
def nasapower_dataset(datadir, zona_code):
    return xr.open_dataset(datadir / f"Zona_{zona_code}_NASAPOWER.nc")


@pytest.fixture
def calibrated_stations(datadir, zona_code):
    return pd.read_pickle(datadir / f"Zona_{zona_code}_stations.pkl")


@pytest.fixture
def station_collection(calibrated_stations):
    return stations.StationCollection(calibrated_stations.loc[:])


@pytest.fixture
def station_ppt_occur_cov(calibrated_stations):
    return stations.StationCollection(calibrated_stations.loc[:3])


class TestStationCalibrator:
    @pytest.fixture
    def single_pixel(self, nasapower_dataset):
        lat = nasapower_dataset.lat[0]
        lon = nasapower_dataset.lon[0]
        return nasapower_dataset.sel(lat=lat, lon=lon)

    @pytest.fixture
    def station_calibrator(self):
        return calibration.StationCalibrator()

    @pytest.fixture
    def uncalibrated_station(self, single_pixel):
        station = stations.WeatherStation(
            Point(single_pixel.lon, single_pixel.lat),
            float(np.max(single_pixel["ELEV"])),
            single_pixel,
        )
        return station

    def test_calibrate_first_order_ppt_occurence(
        self, single_pixel, station_calibrator
    ):
        parameters = station_calibrator.calibrate_first_order_ppt_occurence(
            single_pixel["PRECTOTCORR"]
        )
        assert parameters.shape == (12, 2)
        assert (parameters > 0).all().all()

    def test_calibrate_ppt_amount_mixed_exponential(
        self, single_pixel, station_calibrator
    ):
        parameters = station_calibrator.calibrate_ppt_amount_mixed_exponential(
            single_pixel["PRECTOTCORR"]
        )
        precipitiation = single_pixel["PRECTOTCORR"]
        assert (parameters > 0).all().all()
        assert parameters.max().max() < precipitiation.max()
        assert parameters.min().min() > precipitiation.min()

    def test_calibrate_tmax_tmin_srad(self, single_pixel, station_calibrator):
        B, PHI, wet_doy_params, dry_doy_params, residuals = station_calibrator.calibrate_tmax_tmin_srad(
            single_pixel
        )
        assert np.sum(dry_doy_params["SRAD_mean"] - wet_doy_params["SRAD_mean"]) > 0
        assert B.shape == (12, 3, 3)
        assert PHI.shape == (12, 3, 3)

    def test_calibrate_station(self, station_calibrator, uncalibrated_station):
        station_calibrator.calibrate_station(uncalibrated_station)
        assert uncalibrated_station.calibrated
        assert uncalibrated_station.PHI.shape == (12, 3, 3)


class TestInterStationCalibrator:
    @pytest.fixture
    def interstation_calibrator(self, datadir):
        return calibration.InterStationCalibrator()

    @pytest.mark.slow
    def test_calibrate_station_collection(
        self, station_collection, interstation_calibrator, zona_code
    ):
        interstation_calibrator.calibrate_station_collection(station_collection)
        station_collection.B.shape == (
            12,
            len(station_collection.stations) * 3,
            len(station_collection.stations) * 3,
        )
        station_collection.PHI.shape == (
            12,
            len(station_collection.stations) * 3,
            len(station_collection.stations) * 3,
        )
        with open(
            f"test_stations/Zona_{zona_code}_calibrated_collection.pkl", "wb"
        ) as f:
            import pickle

            pickle.dump(station_collection, f)


class TestInterStationTemperatureSolarRadiationCalibrator:
    @pytest.fixture
    def tempsrad_calibrator(self):
        return calibration.InterStationTemperatureSolarRadiationCalibrator()

    def test_calibrate(self, tempsrad_calibrator, station_collection):
        B, PHI, eigvals_sum = tempsrad_calibrator.calibrate(station_collection)
        B.shape == (
            12,
            len(station_collection.stations) * 3,
            len(station_collection.stations) * 3,
        )
        PHI.shape == (
            12,
            len(station_collection.stations) * 3,
            len(station_collection.stations) * 3,
        )


class TestInterStationPPTOccurenceCalibrator:
    @pytest.fixture
    def ppt_occur_calibrator(self):
        return calibration.InterStationPPTOccurenceCalibrator()

    @pytest.fixture
    def unsmoothed_cov(self, datadir):
        return np.load(datadir / "test_ppt_occur_cov_unsmoothed.npy")

    @pytest.mark.slow
    def test_calibrate(self, ppt_occur_calibrator, station_collection):
        correlation_matrix = ppt_occur_calibrator.calibrate(station_collection)
        N = len(station_collection.stations)
        assert correlation_matrix.shape == (N, N, 12)
        for i in range(correlation_matrix.shape[2]):
            assert np.all(np.linalg.eigvals(correlation_matrix[:, :, i]) > 0)

    def test_compute_two_station_correlation(
        self, ppt_occur_calibrator, station_collection
    ):
        correlation = ppt_occur_calibrator.compute_two_station_correlation(
            0,
            1,
            station_collection.stations.loc[0, "station"],
            station_collection.stations.loc[1, "station"],
            month=1,
        )
        assert correlation < 1
        assert correlation >= 0

    def test_smooth_correlations_by_station_separation(
        self, station_collection, ppt_occur_calibrator, unsmoothed_cov, datadir
    ):
        smoothed_cov = ppt_occur_calibrator.smooth_correlations_by_station_separation(
            unsmoothed_cov, station_collection.separations
        )
        plt.imshow(smoothed_cov)
        plt.show()
        for i in range(smoothed_cov.shape[2]):
            assert np.all(np.linalg.eigvals(smoothed_cov[:, :, i]) > 0)


class TestInterStationPPTAmountCalibrator:
    @pytest.fixture
    def ppt_amount_calibrator(self):
        return calibration.InterStationPPTAmountCalibrator()

    @pytest.fixture
    def ppt_occur_calibrator(self):
        return calibration.InterStationPPTOccurenceCalibrator()

    @pytest.mark.slow
    def test_calibrate(
        self, ppt_amount_calibrator, station_collection, ppt_occur_calibrator
    ):
        station_collection.ppt_occur_corr = ppt_occur_calibrator.calibrate(
            station_collection
        )
        correlation_matrix = ppt_amount_calibrator.calibrate(station_collection)
        assert correlation_matrix.shape == (
            len(station_collection.stations),
            len(station_collection.stations),
            12,
        )
        # Check all matrices are positive definite
        for i in range(correlation_matrix.shape[2]):
            assert np.all(np.linalg.eigvals(correlation_matrix[:, :, i]) > 0)
