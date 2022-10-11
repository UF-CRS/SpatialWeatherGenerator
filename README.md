# Spatial Stochastic Weather Generator

Generate regional preciptitation, temperature, and solar radiation data that is spatially correlated.

Calibrates on [NASA POWER](https://power.larc.nasa.gov/)


## Usage
```python
import datetime as dt
import geopandas as gpd
import sgen


# Regional geometry in WGS84 projection
region = gpd.read_file('region.geojson').geometry[0]

# Builds generator at 0.5 deg resolution and calibrates on NASA POWER historical archive.
generator = sgen.build_spatial_weather_generator(region)

start_date = dt.date(2020, 1, 1)
num_days = 2000

# Outputs xarray on region in WGS84 projection with daily ppt, max and min temp, and srad
# on a 0.5 deg grid
generated_weather = generator.simulate_weather(start_date, num_days)

```


## Citations
Based on the following papers by Wilks:

Multisite generalization of a daily stochastic precipitation generation model (1998)
DOI: 10.1016/S0022-1694(98)00186-3

Simultaneous stochastic simulation of daily precipitation, temperature and solar radiation at multiple sites in complex terrain (1999)
DOI: 10.1016/S0168-1923(99)00037-4

