"""
Tests for nasapower.py
"""
import datetime as dt
import xarray as xr
import pytest
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point

from sgen import nasapower


class TestNASAPower:
    @pytest.fixture
    def simple_region(self, datadir):
        return gpd.read_file(datadir / "north_central_florida.geojson").iloc[0].geometry

    @pytest.fixture
    def loaded_dataset(self, datadir, zona_code):
        return xr.load_dataset(datadir / f"Zona_{zona_code}_NASAPOWER.nc")

    @pytest.fixture
    def large_region(self, datadir):
        return gpd.read_file(datadir / "eastern_US.geojson").iloc[0].geometry

    @pytest.fixture
    def zona_region(self, datadir, zona_code):
        zonas = gpd.read_file(datadir / "Bolsa_de_Cereales_Zonas.geojson")
        return zonas[zonas["Zona"] == zona_code].iloc[0].geometry

    @pytest.mark.slow
    def test_get_NASA_power_data_for_region_returns_full_extent_Dataset(
        self, zona_region, zona_code
    ):
        result = nasapower.get_NASA_Power_data_for_region(zona_region)
        result_bbox = box(
            result.lon.min(), result.lat.min(), result.lon.max(), result.lat.max()
        )
        assert result_bbox.buffer(0.5).contains(zona_region)
        result.to_netcdf(f"test_nasapower/Zona_{zona_code}_NASAPOWER.nc")

    def test_request_POWER_region_returns_xarray_Dataset(self, simple_region):
        start_date = dt.date(2017, 1, 1)
        end_date = dt.date(2017, 12, 31)
        result = nasapower.request_POWER_region(
            simple_region, start_date, end_date, ["T2M"]
        )
        assert isinstance(result, xr.Dataset)

    def test_split_region_into_10_deg_boxes_correctly_splits_region(self, large_region):
        subregions = nasapower.split_region_into_10_deg_boxes(large_region)

        for subregion in subregions:
            min_lon, min_lat, max_lon, max_lat = subregion.bounds
            assert max_lon - min_lon <= 10
            assert max_lat - min_lat <= 10

    def test_build_elevation_grid(self, loaded_dataset):
        dataset_w_elevation = nasapower.build_elevation_grid(loaded_dataset)
        assert "ELEV" in dataset_w_elevation.keys()

    def test_mask_to_region(self, loaded_dataset, zona_region):
        dataset_w_mask = nasapower.mask_to_region(loaded_dataset, zona_region)
        for lat in dataset_w_mask.lat:
            for lon in dataset_w_mask.lon:
                point = Point(lon, lat)
                in_region = dataset_w_mask["REGION"].sel(lat=lat, lon=lon).all()
                assert in_region == point.buffer(0.25, 1).intersects(zona_region)
