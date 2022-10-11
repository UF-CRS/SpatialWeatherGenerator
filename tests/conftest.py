import os
import geopandas as gpd
from pathlib import Path
import pytest
from distutils import dir_util


zonas = gpd.read_file("test_nasapower/Bolsa_de_Cereales_Zonas.geojson")


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    modified from: https://stackoverflow.com/questions/29627341
    """
    filepath = Path(request.module.__file__)
    test_dir = filepath.parent / filepath.stem

    if test_dir.is_dir():
        dir_util.copy_tree(test_dir, str(tmpdir))

    return Path(tmpdir)


@pytest.fixture(params=zonas["Zona"])
def zona_code(request):
    return request.param
