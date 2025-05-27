# forecasting_models.py

import streamlit as st
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive, Naive, HistoricAverage,
    CrostonOptimized, ADIDA, IMAPA, AutoETS
)
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from window_ops.expanding import expanding_mean
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# ───────────────────────── STATSFORECAST ───────────────────────── #

@st.cache_resource
def run_statsforecast_models(Y: pd.DataFrame, horizon: int, season_length: int):
    models = [
        SeasonalNaive(season_length=season_length),
        Naive(),
        HistoricAverage(),
        CrostonOptimized(),
        ADIDA(),
        IMAPA(),
        AutoETS(season_length=season_length)
    ]
    sf = StatsForecast(models=models, freq='D', n_jobs=1)
    fcst = sf.forecast(df=Y, h=horizon)
    return sf, fcst

# ───────────────────────── MLFORECAST ───────────────────────── #

@st.cache_resource
def run_mlforecast_models(Y: pd.DataFrame, horizon: int, _h_param_deprecated: int):
    mlf = MLForecast(
        models=[
            LGBMRegressor(max_depth=10, random_state=42),
            XGBRegressor(max_depth=10, eval_metric='rmse', random_state=42),
            LinearRegression()
        ],
        freq='D',
        lags=list(range(1, 7)),
        lag_transforms={1: [expanding_mean]},
        date_features=['year', 'month', 'day', 'dayofweek',
                       'quarter', 'week', 'dayofyear'],
    )
    mlf.fit(Y, prediction_intervals=PredictionIntervals())
    fcst = mlf.predict(h=horizon, level=[90])
    return mlf, fcst

# ───────────────────────── COMBINARE PREVIZIUNI ───────────────────────── #

@st.cache_data
def combine_forecasts(sf_fcst: pd.DataFrame, mlf_fcst: pd.DataFrame) -> pd.DataFrame:
    return sf_fcst.merge(mlf_fcst, on=['unique_id', 'ds'], how='left')