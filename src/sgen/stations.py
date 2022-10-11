"""
Implementation of individual weather stations driven by external random vectors.
"""
import numpy as np
import time
import xarray as xr
import pandas as pd
import datetime as dt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint
from scipy.stats import multivariate_normal, norm

from sgen import nasapower
from sgen import calibration
from sgen import exceptions


class WeatherStation:
    """
    Based on station equations outlined in 10.1016/S0022-1694(98)00186-3
    """

    def __init__(
        self, location_wgs84: Point, elevation: float, historical_data: xr.Dataset
    ):
        self.location = location_wgs84
        self.elevation = elevation  # in meters
        self.historical_data = (
            historical_data.load()
        )  # Load so pickle saves store full array
        self.calibrated = False

    def simulate_weather(
        self,
        u: np.array,  # RV that forces ppt occurence
        v: np.array,  # RV that forces ppt amount
        start_date: dt.date,
        wet_state: bool = None,
    ) -> pd.DataFrame:
        if not self.calibrated:
            return RuntimeError("Station not yet calibrated")
        if u.shape != v.shape:
            raise exceptions.RandomVariableException(
                "Driving random variables u and v must be of the same length."
            )
        rainfall = self.simulate_rainfall(u, v, start_date, wet_state)
        wet_mask = rainfall > 0
        tmax_tmin_srad = self.simulate_temperature_and_radiation(
            start_date, len(u), wet_mask
        )
        return pd.DataFrame(
            np.concatenate([rainfall[:, np.newaxis], tmax_tmin_srad], axis=1),
            index=pd.date_range(start_date, start_date + pd.Timedelta(days=len(u) - 1)),
            columns=["RAIN", "TMAX", "TMIN", "SRAD"],
        )

    def simulate_temperature_and_radiation(
        self, start_date: dt.date, num_days: int, wet_mask: np.array
    ) -> pd.DataFrame:
        z = self.simulate_TTS_residuals(start_date, num_days)
        TTS = self.convert_residuals_to_TTS(z, start_date, wet_mask)
        return TTS

    def simulate_TTS_residuals(self, start_date, num_days):
        z_t = np.random.normal(size=3)  # initialize a state
        date = start_date
        day = dt.timedelta(days=1)  # use so we dont re-instance during loop

        residuals = np.zeros((num_days, 3))

        e = np.random.normal(size=(num_days, 3))

        t = 0
        while t < num_days:
            month = date.month
            B = self.B[month - 1]
            PHI = self.PHI[month - 1]
            z_t = PHI @ z_t + B @ e[t]
            residuals[t] = z_t
            t += 1
        return residuals

    def convert_residuals_to_TTS(self, z, start_date, wet_mask):
        start_doy = int(start_date.strftime("%j"))

        TTS = np.empty(z.shape)
        mu = np.empty(z.shape)
        sigma = np.empty(z.shape)
        date = start_date
        day = dt.timedelta(days=1)  # use so we dont re-instance during loop

        for t in range(z.shape[0]):
            doy = int(date.strftime("%j"))
            if wet_mask[t]:
                tmax_mu = self.wet_doy_params.loc[doy - 1, "TMAX_mean"]
                tmin_mu = self.wet_doy_params.loc[doy - 1, "TMIN_mean"]
                srad_mu = self.wet_doy_params.loc[doy - 1, "SRAD_mean"]
                tmax_s = self.wet_doy_params.loc[doy - 1, "TMAX_std"]
                tmin_s = self.wet_doy_params.loc[doy - 1, "TMIN_std"]
                srad_s = self.wet_doy_params.loc[doy - 1, "SRAD_std"]
            else:
                tmax_mu = self.dry_doy_params.loc[doy - 1, "TMAX_mean"]
                tmin_mu = self.dry_doy_params.loc[doy - 1, "TMIN_mean"]
                srad_mu = self.dry_doy_params.loc[doy - 1, "SRAD_mean"]
                tmax_s = self.dry_doy_params.loc[doy - 1, "TMAX_std"]
                tmin_s = self.dry_doy_params.loc[doy - 1, "TMIN_std"]
                srad_s = self.dry_doy_params.loc[doy - 1, "SRAD_std"]
            mu[t] = tmax_mu, tmin_mu, srad_mu
            sigma[t] = tmax_s, tmin_s, srad_s
            date += day
        TTS = mu + z * sigma
        return TTS

    def simulate_rainfall(self, u, v, start_date, wet_state) -> pd.Series:
        if not isinstance(wet_state, bool):
            wet_state = self.warmup_rainfall(start_date.month)

        date = start_date
        day = dt.timedelta(days=1)  # use so we dont re-instance during loop
        t = 0
        limit = len(u)
        p_wet = np.empty(u.shape)
        months = np.empty(u.shape, dtype=int)
        wet_days = np.empty(u.shape, dtype=bool)

        # First, do wet or dry loop which requires feedback
        while t < limit:  # Faster loop
            u_t, v_t = u[t], v[t]
            # Get the probability of today being wet from yesterday's state
            p_wet_t = self.ppt_occurence_params[date.month - 1, wet_state]
            # Determine if today if wet
            wet_state = self.is_wet_day(u_t, p_wet_t)
            p_wet[t] = p_wet_t
            months[t] = date.month
            wet_days[t] = wet_state
            date = date + day
            t += 1

        # Calculate all amount simultaneously to save time
        rainfall = self.generate_rainfall_amount(u, v, p_wet, months)
        rainfall[~wet_days] = 0

        return rainfall

    def warmup_rainfall(self, month: int):
        state = np.random.choice([0, 1])
        for _ in range(10):
            u = np.random.uniform()
            P_wet = self.ppt_occurence_params[month - 1, state]
            state = self.is_wet_day(u, P_wet)
        return state

    def is_wet_day(self, u: float, P_wet: float) -> int:
        if u < P_wet:
            state = 1
        else:
            state = 0
        return state

    def generate_rainfall_amount(
        self, u: np.array, v: np.array, P_wet: np.array, month: np.array
    ) -> float:
        params = self.ppt_amount_params[month - 1, :]
        alpha = params[:, 0]
        beta1 = params[:, 1]
        beta2 = params[:, 2]
        # Calculate hybrid beta for suppresion of large amounts at fringes (eqn 12 in paper)
        beta = self.calculate_hybrid_beta(u, P_wet, alpha, beta1, beta2)
        # Nasa POWER min ppt is 0.01 mm
        return 0.01 - beta * np.log(v)

    def calculate_hybrid_beta(self, u, P_wet, alpha, beta1, beta2):
        hybrid_beta = beta2 + 2 * (beta1 - beta2) * (1 - (u / (alpha * P_wet)))
        hybrid_beta[(u / P_wet) > alpha] = beta2[(u / P_wet) > alpha]
        return hybrid_beta


class StationCollection:
    """
    Responsible for holding regional collections of WeatherStation instances
    and their calibration correlations.
    """

    def __init__(self, stations: pd.DataFrame):
        self.stations = stations
        self.separations = self.calculate_station_separations(stations)
        self.ppt_occur_corr = None
        self.ppt_amount_corr = None
        self.B = None
        self.PHI = None
        self.calibrated = False

    def calculate_station_separations(self, stations: pd.DataFrame) -> np.array:
        separations = np.zeros((len(stations), len(stations), 3))
        for k, k_station in stations.iterrows():
            for l, l_station in stations.loc[k:].iterrows():
                if k == l:
                    separations[k, l, :] = 0
                    continue

                separations[k, l, 0] = np.abs(
                    k_station["station"].location.x - l_station["station"].location.x
                )
                separations[k, l, 1] = np.abs(
                    k_station["station"].location.y - l_station["station"].location.y
                )
                separations[k, l, 2] = np.abs(
                    k_station["station"].elevation / 1000
                    - l_station["station"].elevation / 1000
                )
        return separations.transpose(1, 0, 2) + separations

    def simulate_weather(self, start_date: dt.date, num_days: int) -> xr.Dataset:
        """
        Simulate weather from calibrations stations and collection.
        """
        rainfall = self.simulate_stations_rainfall(start_date, num_days)
        rain_mask = rainfall > 0
        tmax, tmin, srad = self.simulate_tmax_tmin_srad(start_date, num_days, rain_mask)
        weather = self.build_weather_array(rainfall, tmax, tmin, srad)
        return weather

    def build_weather_array(self, rainfall, tmax, tmin, srad) -> xr.Dataset:
        # first get time, lat, and lon coordinates for each array
        time = rainfall.index
        lat, lon = self.get_station_coord_grid()
        placeholder = np.empty((len(time), lat.shape[0], lon.shape[0]))
        placeholder[:] = np.nan
        rainfall_ar = xr.DataArray(
            data=placeholder,
            dims=["time", "lat", "lon"],
            coords=dict(lon=lon, lat=lat, time=time),
            attrs=dict(description="Rainfall", units="mm"),
            name="RAIN",
        )
        placeholder = np.empty((len(time), lat.shape[0], lon.shape[0]))
        placeholder[:] = np.nan
        tmax_ar = xr.DataArray(
            data=placeholder,
            dims=["time", "lat", "lon"],
            coords=dict(lon=lon, lat=lat, time=time),
            attrs=dict(description="Maximum daily temperature", units="degC"),
            name="TMAX",
        )
        placeholder = np.empty((len(time), lat.shape[0], lon.shape[0]))
        placeholder[:] = np.nan
        tmin_ar = xr.DataArray(
            data=placeholder,
            dims=["time", "lat", "lon"],
            coords=dict(lon=lon, lat=lat, time=time),
            attrs=dict(description="Minimum daily temperature", units="degC"),
            name="TMIN",
        )
        placeholder = np.empty((len(time), lat.shape[0], lon.shape[0]))
        placeholder[:] = np.nan
        srad_ar = xr.DataArray(
            data=placeholder,
            dims=["time", "lat", "lon"],
            coords=dict(lon=lon, lat=lat, time=time),
            attrs=dict(description="Daily solar radiation", units="MJ/m2/day"),
            name="SRAD",
        )
        for idx, station in self.stations.iterrows():
            lon, lat = station.geometry.xy
            rainfall_ar.loc[dict(lon=lon[0], lat=lat[0])] = rainfall[idx]
            tmax_ar.loc[dict(lon=lon[0], lat=lat[0])] = tmax[idx]
            tmin_ar.loc[dict(lon=lon[0], lat=lat[0])] = tmin[idx]
            srad_ar.loc[dict(lon=lon[0], lat=lat[0])] = srad[idx]

        weather = xr.merge([rainfall_ar, tmax_ar, tmin_ar, srad_ar])
        return weather

    def get_station_coord_grid(self):
        locations = self.stations.geometry
        bbox = MultiPoint(locations).bounds
        lat = np.arange(bbox[3], bbox[1] - 0.5, -0.5)
        lon = np.arange(bbox[0], bbox[2] + 0.5, 0.5)
        return lat, lon

    def simulate_stations_rainfall(self, start_date, num_days) -> pd.DataFrame:
        sim_dates = pd.date_range(
            start_date, start_date + pd.Timedelta(days=num_days - 1)
        )
        u = self.generate_uniform_RV_from_correlated_normal(
            start_date, num_days, self.ppt_occur_corr
        )
        v = self.generate_uniform_RV_from_correlated_normal(
            start_date, num_days, self.ppt_amount_corr
        )

        station_rainfall = pd.DataFrame(columns=self.stations.index, index=sim_dates)
        for idx, station in self.stations.iterrows():
            station_rainfall[idx] = station["station"].simulate_rainfall(
                u[:, idx], v[:, idx], start_date, None
            )
        return station_rainfall

    def simulate_tmax_tmin_srad(
        self, start_date, num_days, rain_mask: np.array
    ) -> pd.DataFrame:

        station_residuals = self.simulate_TTS_residuals(start_date, num_days)
        sim_dates = pd.date_range(
            start_date, start_date + pd.Timedelta(days=num_days - 1)
        )
        tmax = pd.DataFrame(columns=self.stations.index, index=sim_dates)
        tmin = pd.DataFrame(columns=self.stations.index, index=sim_dates)
        srad = pd.DataFrame(columns=self.stations.index, index=sim_dates)
        for station_num in range(len(self.stations)):
            station = self.stations.loc[station_num, "station"]
            z = station_residuals[:, station_num * 3 : station_num * 3 + 3]
            TTS = station.convert_residuals_to_TTS(
                z, start_date, rain_mask[station_num]
            )
            tmax[station_num] = TTS[:, 0]
            tmin[station_num] = TTS[:, 1]
            srad[station_num] = TTS[:, 2]
        return tmax, tmin, srad

    def simulate_TTS_residuals(self, start_date, num_days) -> pd.DataFrame:
        """
        Simulate temperature and solar radiation residuals.
        """
        K = len(self.stations)
        z_t = np.random.normal(size=3 * K)  # initialize a state
        date = start_date
        day = dt.timedelta(days=1)  # use so we dont re-instance during loop

        residuals = np.zeros((num_days, 3 * K))

        e = np.random.normal(size=(num_days, 3 * K))

        printer = np.vectorize(lambda x: "{0:5}".format(x))
        t = 0
        while t < num_days:
            month = date.month
            B = self.B[month - 1]
            PHI = self.PHI[month - 1]
            z_t = (PHI @ z_t + B @ e[t]) / (np.sqrt(1 + self.eigval_sum[month - 1]))
            residuals[t] = z_t
            t += 1
        return residuals

    def generate_uniform_RV_from_correlated_normal(
        self, start_date, num_days, cov
    ) -> np.array:
        # Produce the monthly u driving variables
        date = start_date

        RV_norm = norm()  # used for CDF calculation of each gaussian RV

        gen_count = 0
        rv = []
        while gen_count < num_days:
            next_month = dt.date(date.year, date.month + 1, 1)
            days_in_month = (next_month - date).days
            RV_gen = multivariate_normal(cov=cov[:, :, date.month], allow_singular=True)
            w = RV_gen.rvs(days_in_month)
            rv.append(w)
            gen_count += days_in_month
        rv = np.concatenate(rv)
        rv = np.apply_along_axis(norm.cdf, axis=1, arr=rv)

        return rv[:num_days]
