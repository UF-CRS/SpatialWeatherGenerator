"""
Calibration functions for station parameterization and correlations
"""
import tqdm
import multiprocessing as mp
import time
import ctypes
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from abc import ABC, abstractmethod
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import multivariate_normal, norm
from scipy.linalg import block_diag
from sklearn.linear_model import LinearRegression
from calendar import monthrange
import matplotlib.pyplot as plt

from sgen import stations


class StationCalibrator:
    def calibrate_station(self, station):
        station.ppt_occurence_params = self.calibrate_first_order_ppt_occurence(
            station.historical_data["PRECTOTCORR"]
        ).to_numpy()
        station.ppt_amount_params = self.calibrate_ppt_amount_mixed_exponential(
            station.historical_data["PRECTOTCORR"]
        ).to_numpy()
        B, PHI, wet_doy_params, dry_doy_params, residuals = self.calibrate_tmax_tmin_srad(
            station.historical_data
        )
        station.B = B
        station.PHI = PHI
        station.wet_doy_params = wet_doy_params
        station.dry_doy_params = dry_doy_params
        station.residuals = residuals
        station.calibrated = True

    def calibrate_first_order_ppt_occurence(
        self, precipitiation: xr.Dataset
    ) -> pd.DataFrame:
        """
        Calibrate first order markov chain precipitation occurance for a time series.

        Returns
        -------
        pd.DataFrame
            columns are P01, P11, rows are months [1-12]
        """
        precipitation_occurence_parameters = pd.DataFrame(columns=["P01", "P11"])
        for month in np.unique(precipitiation["time.month"]):
            month_data = precipitiation.where(
                precipitiation["time.month"] == month
            ).dropna("time")
            # Calculate probabilties over each year's month, then combine
            dry_priors = 0
            wet_priors = 0
            wet_after_dry = 0
            wet_after_wet = 0
            for date in month_data.time:
                try:
                    yday_wet = precipitiation.sel(time=date - pd.Timedelta(days=1)) > 0
                except KeyError:  # first day of dataset
                    continue
                tday_wet = precipitiation.sel(time=date) > 0
                if yday_wet:
                    wet_priors += 1
                else:
                    dry_priors += 1
                if yday_wet and tday_wet:
                    wet_after_wet += 1
                if (not yday_wet) and tday_wet:
                    wet_after_dry += 1

            P01 = wet_after_dry / dry_priors
            P11 = wet_after_wet / wet_priors
            precipitation_occurence_parameters.loc[month] = [P01, P11]

        return precipitation_occurence_parameters

    def calibrate_ppt_amount_mixed_exponential(
        self, precipitiation: xr.Dataset
    ) -> pd.DataFrame:
        """
        Fit mixed exponential distribution parameters for eq(7) in 10.1016/S0022-1694(98)00186-3

        Returns
        -------
        pd.DataFrame
            columns are beta1, beta2, alpha, rows are months [1-12]
        """
        precipitation_amount_parameters = pd.DataFrame(
            columns=["alpha", "beta1", "beta2"]
        )

        calibrator = CalibratorMixedExponential()
        for month in np.unique(precipitiation["time.month"]):
            month_data = precipitiation.where(
                precipitiation["time.month"] == month
            ).dropna("time")
            month_rainfall = month_data.where(month_data > 0).dropna(dim="time")
            alpha, beta1, beta2 = calibrator.optimize(month_rainfall)
            precipitation_amount_parameters.loc[month] = [alpha, beta1, beta2]
        return precipitation_amount_parameters

    def calibrate_tmax_tmin_srad(self, weather_data: xr.Dataset):
        wet_doy_params = self.calculate_smoothed_series(weather_data, True)
        dry_doy_params = self.calculate_smoothed_series(weather_data, False)
        tmax_residuals, tmin_residuals, srad_residuals = self.calculate_residuals(
            weather_data, wet_doy_params, dry_doy_params
        )
        residuals = dict(TMAX=tmax_residuals, TMIN=tmin_residuals, SRAD=srad_residuals)
        B, PHI = self.compute_tmax_tmin_srad_matrices(
            tmax_residuals, tmin_residuals, srad_residuals
        )
        return B, PHI, wet_doy_params, dry_doy_params, residuals

    def compute_tmax_tmin_srad_matrices(self, tmax_r, tmin_r, srad_r):
        B = np.empty((12, 3, 3))
        PHI = np.empty((12, 3, 3))
        for month in range(1, 13):
            tmax_month = tmax_r.where(tmax_r["time.month"] == month).dropna(dim="time")
            tmin_month = tmin_r.where(tmin_r["time.month"] == month).dropna(dim="time")
            srad_month = srad_r.where(srad_r["time.month"] == month).dropna(dim="time")
            R0 = self.calculate_R0(tmax_month, tmin_month, srad_month)
            R1 = self.calculate_R1(tmax_month, tmin_month, srad_month)
            PHI[month - 1] = R1 @ np.linalg.inv(R0)
            BBT = R0 - PHI[month - 1] @ R1.T
            B[month - 1] = np.linalg.cholesky(BBT)
        return B, PHI

    def calculate_R0(self, tmax_r, tmin_r, srad_r):
        rho12 = rho21 = self.correlation(tmax_r, tmin_r)[0, 1]
        rho13 = rho31 = self.correlation(tmax_r, srad_r)[0, 1]
        rho23 = rho32 = self.correlation(tmin_r, srad_r)[0, 1]
        R0 = np.array([[1, rho12, rho13], [rho21, 1, rho23], [rho31, rho32, 1]])
        return R0

    def calculate_R1(self, tmax_r, tmin_r, srad_r):
        rholag1 = self.correlation_lag1(tmax_r, tmax_r)[0, 1]
        rholag12 = self.correlation_lag1(tmax_r, tmin_r)[0, 1]
        rholag13 = self.correlation_lag1(tmax_r, srad_r)[0, 1]
        rholag21 = self.correlation_lag1(tmin_r, tmax_r)[0, 1]
        rholag2 = self.correlation_lag1(tmin_r, tmin_r)[0, 1]
        rholag23 = self.correlation_lag1(tmin_r, srad_r)[0, 1]
        rholag31 = self.correlation_lag1(srad_r, tmax_r)[0, 1]
        rholag32 = self.correlation_lag1(srad_r, tmin_r)[0, 1]
        rholag3 = self.correlation_lag1(srad_r, srad_r)[0, 1]
        R1 = np.array(
            [
                [rholag1, rholag12, rholag13],
                [rholag21, rholag2, rholag23],
                [rholag31, rholag32, rholag3],
            ]
        )
        return R1

    def correlation(self, sig_1, sig_2):
        return np.corrcoef(sig_1, sig_2)

    def correlation_lag1(self, sig_1, sig_2):
        return self.correlation(sig_1, np.roll(sig_2, 1))

    def calculate_smoothed_series(
        self, weather_data, wet_state: bool
    ) -> dict[np.array]:
        if wet_state:
            weather_data = weather_data.where(weather_data["PRECTOTCORR"] > 0)
        else:
            weather_data = weather_data.where(weather_data["PRECTOTCORR"] == 0)
        tmax_doy_mean, tmax_doy_std = self.calculate_smoothed_doy_mean_and_std(
            weather_data["T2M_MAX"]
        )
        tmin_doy_mean, tmin_doy_std = self.calculate_smoothed_doy_mean_and_std(
            weather_data["T2M_MIN"]
        )
        srad_doy_mean, srad_doy_std = self.calculate_smoothed_doy_mean_and_std(
            weather_data["ALLSKY_SFC_SW_DWN"]
        )
        columns = [
            "TMIN_mean",
            "TMIN_std",
            "TMAX_mean",
            "TMAX_std",
            "SRAD_mean",
            "SRAD_std",
        ]
        smoothed_series = pd.DataFrame(
            np.array(
                [
                    tmin_doy_mean,
                    tmin_doy_std,
                    tmax_doy_mean,
                    tmax_doy_std,
                    srad_doy_mean,
                    srad_doy_std,
                ]
            ).T,
            columns=columns,
            index=range(366),
        )
        return smoothed_series

    def calculate_residuals(self, weather_data, wet_doy_params, dry_doy_params):
        ppt = weather_data["PRECTOTCORR"]
        tmax_residuals = self.calculate_residuals_for_variable(
            weather_data["T2M_MAX"],
            ppt,
            wet_doy_params["TMAX_mean"],
            wet_doy_params["TMAX_std"],
            dry_doy_params["TMAX_mean"],
            dry_doy_params["TMAX_std"],
        )
        tmin_residuals = self.calculate_residuals_for_variable(
            weather_data["T2M_MIN"],
            ppt,
            wet_doy_params["TMIN_mean"],
            wet_doy_params["TMIN_std"],
            dry_doy_params["TMIN_mean"],
            dry_doy_params["TMIN_std"],
        )
        srad_residuals = self.calculate_residuals_for_variable(
            weather_data["ALLSKY_SFC_SW_DWN"],
            ppt,
            wet_doy_params["SRAD_mean"],
            wet_doy_params["SRAD_std"],
            dry_doy_params["SRAD_mean"],
            dry_doy_params["SRAD_std"],
        )
        return tmax_residuals, tmin_residuals, srad_residuals

    def calculate_residuals_for_variable(
        self, variable, ppt, wet_doy_mean, wet_doy_std, dry_doy_mean, dry_doy_std
    ):
        # Do this the long way for now
        residuals = xr.DataArray(dims=variable.dims, coords=variable.coords)
        for doy in range(1, 367):
            doy_mask = variable["time.dayofyear"] == doy
            wet_mask = ppt > 0
            dry_mask = ppt == 0
            wet_doy = wet_mask * doy_mask
            dry_doy = dry_mask * doy_mask
            residuals[wet_doy] = (
                (variable[wet_doy] - wet_doy_mean[doy - 1]) / wet_doy_std[doy - 1]
            )  # zero indexed
            residuals[dry_doy] = (
                (variable[dry_doy] - dry_doy_mean[doy - 1]) / dry_doy_std[doy - 1]
            )  # zero indexed
        return residuals

    def calculate_smoothed_doy_mean_and_std(self, variable: xr.DataArray):
        # TODO: See Epstein 1991 (from Appendix A Wilks 1999) for smoothing to be done here.

        # NOTE: some very wet pixels never have a dry day for a given DOY in the entire
        # NASA POWER historical archive, so we must interpolate nans after grouping by doy
        doy_mean = (
            variable.groupby("time.dayofyear")
            .mean()
            .interpolate_na(dim="dayofyear", fill_value="extrapolate")
        )
        doy_std = (
            variable.groupby("time.dayofyear")
            .std()
            .interpolate_na(dim="dayofyear", fill_value="extrapolate")
        )
        fft_doy_mean = np.fft.rfft(doy_mean)
        fft_doy_mean[3:] = 0
        smoothed_doy_mean = np.fft.irfft(fft_doy_mean)
        fft_doy_std = np.fft.rfft(doy_std)
        fft_doy_std[3:] = 0
        smoothed_doy_std = np.fft.irfft(fft_doy_std)
        return smoothed_doy_mean, smoothed_doy_std


class CalibratorMixedExponential:
    def __init__(self, termination_step=0.005, beta1=10, beta2=1, alpha=0.5):
        self.beta = [beta1, beta2]
        self.alpha = alpha
        self.termination_step = termination_step

    def log_likelihood(self, x) -> float:
        likelihood = 0

        likelihood = self.alpha * (1 / self.beta[0]) * np.exp(-x / self.beta[0])
        likelihood += (1 - self.alpha) * (1 / self.beta[1]) * np.exp(-x / self.beta[1])

        loglikelihood = np.sum(np.log(likelihood))

        return loglikelihood

    def optimize(self, data: xr.DataArray) -> tuple[float]:
        """
        Returns alpha, beta1, beta2
        """
        x = np.array(data)
        self.expectation_maximization(x)

        return self.alpha, self.beta[0], self.beta[1]

    def expectation_maximization(self, x):
        """
        see: https://stats.stackexchange.com/questions/291642
        """
        prior_LL = 0
        while np.abs(self.log_likelihood(x) - prior_LL) > self.termination_step:
            prior_LL = self.log_likelihood(x)
            P_i_given_x = self.calculate_P_i_given_x(x)
            self.alpha = np.sum(P_i_given_x) / len(x)
            self.beta[0] = (x.T @ P_i_given_x) / np.sum(P_i_given_x)
            self.beta[1] = (x.T @ (1 - P_i_given_x)) / np.sum(1 - P_i_given_x)

    def calculate_P_i_given_x(self, x: np.array) -> np.array:
        numerator = self.alpha * (1 / self.beta[0]) * np.exp(-x / self.beta[0])
        denominator = numerator + (1 - self.alpha) * (1 / self.beta[1]) * np.exp(
            -x / self.beta[1]
        )
        return numerator / denominator


class InterStationCalibrator:
    """
    Interstation correlation calibration based on method in 10.1016/S0022-1694(98)00186-3
    """

    def __init__(self):
        self.calibrators = {}
        self.calibrators["ppt_occur"] = InterStationPPTOccurenceCalibrator()
        self.calibrators["ppt_amount"] = InterStationPPTAmountCalibrator()
        self.calibrators[
            "TTS_calib"
        ] = InterStationTemperatureSolarRadiationCalibrator()

    def calibrate_station_collection(self, station_collection):
        # First, calibrate precipitation occurence
        print("Calibrating precipitation occurence")
        station_collection.ppt_occur_corr = self.calibrators["ppt_occur"].calibrate(
            station_collection
        )
        print("Calibrating precipitation amount")
        station_collection.ppt_amount_corr = self.calibrators["ppt_amount"].calibrate(
            station_collection
        )
        print("Calibrating temperature and solar radiation")
        station_collection.B, station_collection.PHI, station_collection.eigval_sum = self.calibrators[
            "TTS_calib"
        ].calibrate(
            station_collection
        )
        station_collection.calibrated = True


class InterStationTemperatureSolarRadiationCalibrator:
    """
    Calibrator for multi-station 3K x 3K correlation B and PHI from Wilks 1999.

    Note
    ----
    Construct of intermediate matrices R0 and R1 requires first caluclating correlations,
    then fitting a regression based on spatial separation in order to keep R0 and R1 consistent
    across many locations.

    See Appendix A in Wilks 1999 for method and more information.
    """

    def calibrate(self, station_collection):
        """
        eigvals_sum variable returned is sum of eigvenvalues added to the BBT
        matrix to make it positive definite. We must use these during weather generation
        to normalize the autoregressive model. See Random Functions and Hydrology,
        Bras and Rodrigues-Iturbe p98, 99 for more information.
        """
        stations = station_collection.stations
        separations = station_collection.separations
        B_months = np.empty((12, 3 * len(stations), 3 * len(stations)))
        PHI_months = np.empty((12, 3 * len(stations), 3 * len(stations)))
        eigvals_sum_months = np.empty((12, 1))
        for month in tqdm.tqdm(list(range(1, 13))):
            R0 = self.calculate_R0(stations, month, separations)
            R1 = self.calculate_R1(stations, month, separations)
            PHI = self.build_PHI(R0, R1, stations)
            R1 = PHI @ R0
            BBT = R0 - PHI @ R1.T
            # Save sum of values used to adjust BBT as we need them in the autoregressive model
            BBT, eigvals_sum = eigvalue_adjustment(BBT)
            eigvals, eigvecs = np.linalg.eig(BBT)
            B = np.linalg.cholesky(BBT)
            B_months[month - 1] = B
            PHI_months[month - 1] = PHI
            eigvals_sum_months[month - 1] = eigvals_sum
        return B_months, PHI_months, eigvals_sum_months

    def build_PHI(self, R0, R1, stations):
        """
        We make a block diagonal PHI matrix, following assumptions in Wilks 1999.
        """
        N = len(stations)
        PHI = np.zeros((3 * N, 3 * N))
        for k, k_station in stations.iterrows():
            R0_station = R0[k * 3 : k * 3 + 3, k * 3 : k * 3 + 3]
            R1_station = R1[k * 3 : k * 3 + 3, k * 3 : k * 3 + 3]
            PHI[k * 3 : k * 3 + 3, k * 3 : k * 3 + 3] = R1_station @ np.linalg.inv(
                R0_station
            )
        # Do triangular sum to get full matrix
        PHI = np.triu(PHI) + np.triu(PHI).T
        np.fill_diagonal(
            PHI, np.diagonal(PHI) * 0.5
        )  # set diagonal elements to one after two triangle sums
        return PHI

    def calculate_R0(self, stations, month: int, separations: pd.DataFrame):
        N = len(stations)
        R0 = np.zeros((3 * N, 3 * N))
        for k, k_station in stations.iterrows():
            for l, l_station in stations.loc[k:].iterrows():
                kl_correlation = self.compute_interstation_correlation(
                    k_station["station"].residuals,
                    l_station["station"].residuals,
                    month,
                )
                R0[k * 3 : k * 3 + 3, l * 3 : l * 3 + 3] = kl_correlation
        # Do triangular sum to get full matrix
        R0 = np.triu(R0) + np.triu(R0).T
        np.fill_diagonal(R0, 1)  # set diagonal elements to one after two triangle sums

        R0_smoothed = self.smooth_R_matrix(R0, stations, separations)
        # Adjust the eignvalues to ensure positive definite matrix
        R0_adjusted, _ = eigvalue_adjustment(R0_smoothed)
        return R0_adjusted

    def calculate_R1(self, stations, month: int, separations: pd.DataFrame):
        N = len(stations)
        R1 = np.zeros((3 * N, 3 * N))
        for k, k_station in stations.iterrows():
            for l, l_station in stations.iterrows():
                kl_correlation = self.compute_interstation_lagcorrelation(
                    k_station["station"].residuals,
                    l_station["station"].residuals,
                    month,
                )
                R1[k * 3 : k * 3 + 3, l * 3 : l * 3 + 3] = kl_correlation
        R1_smoothed = self.smooth_R_matrix(
            R1, stations, separations, interceptonauto=True
        )
        return R1_smoothed

    def get_residuals_for_month(
        self, residuals: xr.DataArray, variable: str, month: int
    ):
        var_residuals = residuals[variable]
        return var_residuals.where(residuals[variable]["time.month"] == month).dropna(
            dim="time"
        )

    def compute_interstation_correlation(self, k_residuals, l_residuals, month):
        tmax_k = self.get_residuals_for_month(k_residuals, "TMAX", month)
        tmin_k = self.get_residuals_for_month(k_residuals, "TMIN", month)
        srad_k = self.get_residuals_for_month(k_residuals, "SRAD", month)
        tmax_l = self.get_residuals_for_month(l_residuals, "TMAX", month)
        tmin_l = self.get_residuals_for_month(l_residuals, "TMIN", month)
        srad_l = self.get_residuals_for_month(l_residuals, "SRAD", month)
        rho11 = np.corrcoef(tmax_k, tmax_l)[0, 1]
        rho22 = np.corrcoef(tmin_k, tmin_l)[0, 1]
        rho33 = np.corrcoef(srad_k, srad_l)[0, 1]
        rho12 = rho21 = np.corrcoef(tmax_k, tmin_l)[0, 1]
        rho13 = rho31 = np.corrcoef(tmax_k, srad_l)[0, 1]
        rho23 = rho32 = np.corrcoef(tmin_k, srad_l)[0, 1]
        R0 = np.array(
            [[rho11, rho12, rho13], [rho21, rho22, rho23], [rho31, rho32, rho33]]
        )
        return R0

    def compute_interstation_lagcorrelation(self, k_residuals, l_residuals, month):
        tmax_k = self.get_residuals_for_month(k_residuals, "TMAX", month)
        tmin_k = self.get_residuals_for_month(k_residuals, "TMIN", month)
        srad_k = self.get_residuals_for_month(k_residuals, "SRAD", month)
        tmax_l = self.get_residuals_for_month(l_residuals, "TMAX", month)
        tmin_l = self.get_residuals_for_month(l_residuals, "TMIN", month)
        srad_l = self.get_residuals_for_month(l_residuals, "SRAD", month)
        tmax_k_lag = np.roll(tmax_k, 1)
        tmin_k_lag = np.roll(tmin_k, 1)
        srad_k_lag = np.roll(srad_k, 1)
        rholag11 = np.corrcoef(tmax_k_lag, tmax_l)[0, 1]
        rholag12 = np.corrcoef(tmax_k_lag, tmin_l)[0, 1]
        rholag13 = np.corrcoef(tmax_k_lag, srad_l)[0, 1]
        rholag21 = np.corrcoef(tmin_k_lag, tmax_l)[0, 1]
        rholag22 = np.corrcoef(tmin_k_lag, tmin_l)[0, 1]
        rholag23 = np.corrcoef(tmin_k_lag, srad_l)[0, 1]
        rholag31 = np.corrcoef(srad_k_lag, tmax_l)[0, 1]
        rholag32 = np.corrcoef(srad_k_lag, tmin_l)[0, 1]
        rholag33 = np.corrcoef(srad_k_lag, srad_l)[0, 1]
        R1 = np.array(
            [
                [rholag11, rholag12, rholag13],
                [rholag21, rholag22, rholag23],
                [rholag31, rholag32, rholag33],
            ]
        )
        return R1

    def smooth_R_matrix(self, matrix, stations, separations, interceptonauto=False):
        # There are nine correlations in each k, l pairing.
        # We have to smooth these correlations separately
        N = len(stations)
        smoothed_matrix = np.zeros(matrix.shape)

        # Do for each of the correlations
        # Iterate over existing R matrix, pull out all targets, fit OLS
        for corr_idx in range(9):
            row_subidx = corr_idx % 3
            col_subidx = corr_idx // 3
            log_targets = []
            targets = []
            inputs = []
            for k in range(N):
                for l in range(N):
                    if matrix[k * 3 + row_subidx, l * 3 + col_subidx] > 0:
                        log_targets.append(
                            np.log(matrix[k * 3 + row_subidx, l * 3 + col_subidx])
                        )
                    else:
                        log_targets.append(
                            np.nan
                        )
                    targets.append(matrix[k * 3 + row_subidx, l * 3 + col_subidx])
                    inputs.append(separations[k, l])
            if row_subidx == col_subidx:
                ols = LinearRegression(fit_intercept=interceptonauto)
            else:
                ols = LinearRegression(fit_intercept=True)
            # Check is there are any negative values in the correlations, namely SRAD-TMIN corr
            if np.isfinite(log_targets).all():
                ols.fit(inputs, log_targets)
            else:
                ols.fit(inputs, targets)
            for k in range(N):
                for l in range(N):
                    if np.isfinite(log_targets).all():
                        smoothed_matrix[
                            k * 3 + row_subidx, l * 3 + col_subidx
                        ] = np.exp(ols.predict(separations[k, l].reshape(1, -1)))
                    else:
                        smoothed_matrix[
                            k * 3 + row_subidx, l * 3 + col_subidx
                        ] = ols.predict(separations[k, l].reshape(1, -1))

        return smoothed_matrix


class InterStationParamCalibrator(ABC):
    def __init__(self):
        pass

    def calibrate(self, station_collection):
        self.station_collection = station_collection
        stations = station_collection.stations
        N = len(stations)
        # Correlations are calculated for each month.
        corr_tensor = np.empty((N, N, 12))
        for month in tqdm.tqdm(list(range(1, 13))):
            corr_tensor[:, :, month - 1] = self._month_calibration(stations, month)
        # We must smooth the correlations by distance to ensure we have positive semi-definite
        # covariance matrices during generation
        corr_tensor = self.smooth_correlations_by_station_separation(
            corr_tensor, station_collection.separations
        )
        return corr_tensor

    def smooth_correlations_by_station_separation(self, corr_tensor, separations):
        """
        Here we perform ordinary least squares regression on a station separation smoothing
        function to ensure positive semi-definite correlation matrices are produced.

        Smoothing function is rho = exp(b1|x| + b2|y| + b3|z|).
        We predict log(rho) using OLS.
        """
        smooth_corr_tensor = np.empty(corr_tensor.shape)
        for month in range(corr_tensor.shape[2]):
            # Smooth across each month
            month_corr = corr_tensor[:, :, month]
            smooth_corr_tensor[:, :, month] = self.smooth_corr_matrix(
                month_corr, separations
            )
        return smooth_corr_tensor

    def smooth_corr_matrix(self, month_corr, separations) -> np.array:
        """
        Note
        ----
        OLS smoothing sometimes produces matrices with one eigenvalue of the order
        1e-5. We perform eigevalue matrix adjustment to correct for this.
        """
        smoothed_corr = np.zeros(month_corr.shape)
        targets, inputs = self.build_smoothing_targets(month_corr, separations)
        ols = LinearRegression(fit_intercept=False)  # No intercept per Wilks 1999
        ols.fit(inputs, targets)
        for k in range(smoothed_corr.shape[0]):
            for l in range(smoothed_corr.shape[1]):
                smoothed_corr[k, l] = np.exp(
                    ols.predict(separations[k, l].reshape(1, -1))
                )
        if not np.all(np.linalg.eigvals(smoothed_corr) > 0):  # PD check
            smoothed_corr, _ = eigvalue_adjustment(smoothed_corr)

        return smoothed_corr

    def build_smoothing_targets(self, month_corr, separations):
        """
        Targets are log correlations.
        """
        targets = []
        inputs = []
        for k in range(month_corr.shape[0]):
            for l in range(k, month_corr.shape[1]):
                targets.append(np.log(month_corr[k, l]))
                inputs.append(separations[k, l])
        return np.array(targets), np.array(inputs)

    def pool(self, shared_array) -> mp.Pool:
        """Build a process pool to run station pair calibrations."""
        return mp.Pool(
            mp.cpu_count(), initializer=self.mp_initializer, initargs=(shared_array,)
        )

    def build_mp_jobs(self, stations: pd.DataFrame, month: int) -> list:
        MP_args = []
        N = len(stations)
        for k, k_station in stations.iterrows():
            for l, l_station in stations.loc[k:].iterrows():
                if k == l:
                    continue
                MP_args.append(
                    [k, l, N, k_station["station"], l_station["station"], month]
                )
        return MP_args

    def _month_calibration(self, stations, month: int) -> np.array:
        N = len(stations)
        # multiprocessing this calibration using shared mememory rawarray
        corr_matrix = mp.Array(ctypes.c_double, np.zeros(N * N), lock=False)

        # Computes the above diagonal correlations
        with self.pool(corr_matrix) as p:
            p.map(self.mp_calibration_target, self.build_mp_jobs(stations, month))

        corr_matrix = np.frombuffer(corr_matrix, dtype="float64").reshape(N, N)

        corr_matrix = corr_matrix + corr_matrix.T + np.identity(n=N)

        return corr_matrix

    def mp_initializer(self, shared_array: mp.Array):
        globals()["shared_array"] = shared_array

    def mp_calibration_target(self, args):
        k, l, N, k_station, l_station, month = args
        global shared_array
        shared_array[k * N + l] = self.compute_two_station_correlation(
            k, l, k_station, l_station, month
        )

    @abstractmethod
    def compute_two_station_correlation(self):
        pass

    @abstractmethod
    def get_month_historical_correlation(self):
        pass


class InterStationPPTOccurenceCalibrator(InterStationParamCalibrator):
    def compute_two_station_correlation(self, k, l, k_station, l_station, month):

        # Get args to pass in to root finding alg
        observed_correlation = self.get_month_historical_correlation(
            k_station, l_station, month
        )
        month_length = monthrange(2020, month)[1]
        start_date = dt.date(2020, month, 1)
        result = minimize_scalar(
            MinimizationTargets.ppt_occurence_two_stations,
            args=(
                observed_correlation,
                month,
                month_length,
                start_date,
                k_station,
                l_station,
            ),
            bounds=(0, 1),
            method="Bounded",
        )
        omega = result["x"]
        return omega

    def get_month_historical_correlation(self, k_station, l_station, month):
        k_station_historical = k_station.historical_data["PRECTOTCORR"]
        k_station_historical[k_station.historical_data["PRECTOTCORR"] > 0] = 1
        l_station_historical = l_station.historical_data["PRECTOTCORR"]
        l_station_historical[l_station.historical_data["PRECTOTCORR"] > 0] = 1
        k_station_month = k_station_historical[
            k_station_historical["time.month"] == month
        ]
        l_station_month = l_station_historical[
            l_station_historical["time.month"] == month
        ]
        observed_correlation = np.corrcoef(k_station_month, l_station_month)[0, 1]
        return observed_correlation


class InterStationPPTAmountCalibrator(InterStationParamCalibrator):
    def compute_two_station_correlation(self, k, l, k_station, l_station, month):
        # Get args to pass in to root finding alg
        observed_correlation = self.get_month_historical_correlation(
            k_station, l_station, month
        )
        # Get omega, occurence correlation parameter between k and l
        omega = self.station_collection.ppt_occur_corr[k, l, month - 1]

        month_length = monthrange(2020, month)[1]
        start_date = dt.date(2020, month, 1)
        result = minimize_scalar(
            MinimizationTargets.ppt_amount_two_stations,
            args=(
                omega,
                observed_correlation,
                month,
                month_length,
                start_date,
                k_station,
                l_station,
            ),
            bounds=(0, 1),
            method="Bounded",
        )
        zeta = result["x"]
        return zeta

    def get_month_historical_correlation(self, k_station, l_station, month):
        k_station_historical = k_station.historical_data["PRECTOTCORR"]
        l_station_historical = l_station.historical_data["PRECTOTCORR"]
        k_station_month = k_station_historical[
            k_station_historical["time.month"] == month
        ]
        l_station_month = l_station_historical[
            l_station_historical["time.month"] == month
        ]
        observed_correlation = np.corrcoef(k_station_month, l_station_month)[0, 1]
        return observed_correlation


class MinimizationTargets:
    def ppt_occurence_two_stations(
        omega,
        obs_corr,
        month: int,
        month_length: int,
        start_date: dt.date,
        k_station,
        l_station,
    ):
        # Minimize difference between generated and observered binary series correlaton
        RV_len = 10000 - (10000 % month_length)
        # Define the covariance matrix for w_k, w_l and then generate the series
        cov = np.array([[1, omega], [omega, 1]])
        u_k, u_l = generate_two_uniform_series_from_correlated_normal(
            cov, length=RV_len
        )
        # Use u for u and v as we only care about occurence here
        k_occur = []
        l_occur = []

        i = 0
        # Get the stations to simulate that month only
        while i < RV_len:
            u_t_k = u_k[i : i + month_length]
            u_t_l = u_l[i : i + month_length]
            k_rainfall = k_station.simulate_rainfall(
                u_t_k, u_t_k, start_date, wet_state=None
            )
            l_rainfall = l_station.simulate_rainfall(
                u_t_l, u_t_l, start_date, wet_state=None
            )
            k_rainfall[k_rainfall > 0] = 1
            l_rainfall[l_rainfall > 0] = 1
            k_occur.append(k_rainfall)
            l_occur.append(l_rainfall)
            i = i + month_length
        k_occur = np.concatenate(k_occur, axis=None)
        l_occur = np.concatenate(l_occur, axis=None)

        simulation_correlation = np.corrcoef(k_occur, l_occur)[0, 1]

        return np.abs(obs_corr - simulation_correlation)

    def ppt_amount_two_stations(
        zeta: float,
        omega: float,
        obs_corr: float,
        month: int,
        month_length: int,
        start_date: dt.date,
        k_station,
        l_station,
    ):

        RV_len = 10000 - (10000 % month_length)
        # Define the covariance matrix for w_k, w_l and then generate the series
        occur_cov = np.array([[1, omega], [omega, 1]])
        amount_cov = np.array([[1, zeta], [zeta, 1]])
        u_k, u_l = generate_two_uniform_series_from_correlated_normal(
            occur_cov, length=RV_len
        )
        v_k, v_l = generate_two_uniform_series_from_correlated_normal(
            amount_cov, length=RV_len
        )
        # Use u for both as we only care about occurence here
        k_amount = []
        l_amount = []

        i = 0
        # Get the stations to simulate that month only
        while i < RV_len:
            u_t_k = u_k[i : i + month_length]
            u_t_l = u_l[i : i + month_length]
            v_t_k = v_k[i : i + month_length]
            v_t_l = v_l[i : i + month_length]

            k_rainfall = k_station.simulate_rainfall(
                u_t_k, v_t_k, start_date, wet_state=None
            )
            l_rainfall = l_station.simulate_rainfall(
                u_t_k, v_t_k, start_date, wet_state=None
            )
            k_amount.append(k_rainfall)
            l_amount.append(l_rainfall)
            i = i + month_length
        k_amount = np.concatenate(k_amount, axis=None)
        l_amount = np.concatenate(l_amount, axis=None)

        simulation_correlation = np.corrcoef(k_amount, l_amount)[0, 1]

        return np.abs(obs_corr - simulation_correlation)


def eigvalue_adjustment(matrix):
    """
    Adjust matrix until it is positive definite, as in Random Functions in Hydrology 1985
    page 98.
    """
    adjusted = False
    adjustment_values = []
    while not np.all(np.linalg.eigvals(matrix) > 1e-15):
        # NOTE: eigvals must be above 1e-15 otherwise np.linalg.cholesky will
        # think they are not positive definite because of rounding
        eigvals = np.linalg.eigvals(matrix)
        abs_min_val = np.abs(np.min(eigvals))
        # Sometimes 0 eigenvector in numpy eig solution for BBT, perturb a little
        # or stuck with -1e-17 eigenvector
        if abs_min_val < 1e-7:
            abs_min_val = 0.0000001
        matrix = matrix + np.identity(matrix.shape[1]) * abs_min_val
        adjustment_values.append(abs_min_val)
        adjusted = True

    if adjusted:
        # See Wilks 1999 Appendix for final covariance to correlation matrix adjustment
        S0_adjusted = matrix
        D = np.diag(1 / np.sqrt(np.diagonal(S0_adjusted)))
        matrix = D @ S0_adjusted @ D
        # After this final adjustment, the correlation matrix sometimes has an eigenvalue
        # around -1e-17, which we need to correct with a final run
        if not np.all(np.linalg.eigvals(matrix) > 1e-15):
            matrix, _ = eigvalue_adjustment(matrix)

    return matrix, np.sum(adjustment_values)


def generate_two_uniform_series_from_correlated_normal(cov: np.array, length):
    # Generate correlated w_k, w_l series
    # Seed each iteration for consistent results across optimization step
    RV_gen = multivariate_normal(cov=cov, seed=42, allow_singular=True)
    RV_norm = norm()
    w = RV_gen.rvs(length)
    u1 = norm.cdf(w[:, 0])
    u2 = norm.cdf(w[:, 1])
    return u1, u2
