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

# Helper function to prepare future exogenous features
def _prepare_future_X_df(Y_df: pd.DataFrame, horizon: int, external_feature_col: str | None) -> pd.DataFrame | None:
    if external_feature_col is None or external_feature_col not in Y_df.columns:
        # st.info(f"Coloana '{external_feature_col}' nu a fost găsită. Nu se vor genera caracteristici exogene viitoare.")
        return None

    if Y_df[external_feature_col].isnull().all():
        st.warning(f"Coloana '{external_feature_col}' conține numai valori NaN. Nu se pot genera caracteristici exogene viitoare.")
        return None

    future_dfs = []
    series_freq = 'D' # Assuming daily frequency based on prior processing

    for uid, group in Y_df.groupby('unique_id', observed=True):
        group_sorted = group.sort_values('ds')
        last_hist_date = group_sorted['ds'].iloc[-1]
        
        last_known_val_series = group_sorted[external_feature_col].dropna()
        if last_known_val_series.empty:
            st.warning(f"Toate valorile '{external_feature_col}' sunt NaN pentru ID-ul unic '{uid}'. Se va folosi 0 pentru viitor.")
            last_known_val = 0
        else:
            last_known_val = last_known_val_series.iloc[-1]
        
        future_dates = pd.date_range(start=last_hist_date + pd.Timedelta(days=1), periods=horizon, freq=series_freq)
        
        df_uid_fut = pd.DataFrame({
            'unique_id': uid,
            'ds': future_dates,
            external_feature_col: last_known_val
        })
        future_dfs.append(df_uid_fut)
    
    if not future_dfs:
        # st.info("Nu s-au putut genera caracteristici exogene viitoare.")
        return None
        
    final_future_X_df = pd.concat(future_dfs).reset_index(drop=True)
    # st.info(f"Caracteristicile exogene viitoare ('{external_feature_col}') au fost pregătite folosind ultima valoare cunoscută.")
    return final_future_X_df

# ───────────────────────── STATSFORECAST ───────────────────────── #

@st.cache_resource
def run_statsforecast_models(Y: pd.DataFrame, horizon: int, season_length: int):
    model_list = [
        SeasonalNaive(season_length=season_length),
        Naive(),
        HistoricAverage(),
        CrostonOptimized(),
        ADIDA(),
        IMAPA(),
    ]
    
    future_X_for_sf = None
    ets_model = AutoETS(season_length=season_length)

    if 'external_feature' in Y.columns:
        Y['external_feature'] = pd.to_numeric(Y['external_feature'], errors='coerce').ffill().bfill()
        if not Y['external_feature'].isnull().all(): # Proceed only if external_feature is usable
            ets_model = AutoETS(season_length=season_length) # Removed exogenous, as it's handled by StatsForecast wrapper
            future_X_for_sf = _prepare_future_X_df(Y, horizon, 'external_feature')
        else:
            st.warning("Coloana 'external_feature' conține numai NaN după procesare și nu va fi utilizată de AutoETS.")


    model_list.append(ets_model)
    
    sf = StatsForecast(models=model_list, freq='D', n_jobs=1)
    
    # Fit uses Y which includes y and potentially external_feature for AutoETS
    sf.fit(Y)
    
    # Predict needs future X_df if exogenous features are used by any model
    fcst = sf.predict(h=horizon, X_df=future_X_for_sf)
    return sf, fcst

# ───────────────────────── MLFORECAST ───────────────────────── #

@st.cache_resource
def run_mlforecast_models(Y: pd.DataFrame, horizon: int, window_size_ml: int): # Renamed _h_param_deprecated
    models_ml = [
        LGBMRegressor(max_depth=10, random_state=42),
        XGBRegressor(max_depth=10, eval_metric='rmse', random_state=42),
        LinearRegression()
    ]
    
    # Use window_size_ml for lags if provided and valid, otherwise default
    lags_list = list(range(1, max(1, window_size_ml) + 1)) if window_size_ml > 0 else list(range(1,7))

    mlf = MLForecast(
        models=models_ml,
        freq='D',
        lags=lags_list,
        lag_transforms={1: [expanding_mean]}, # Consider if lag transforms need adjustment with window_size_ml
        date_features=['year', 'month', 'day', 'dayofweek',
                       'quarter', 'week', 'dayofyear'],
    )
    
    # Prepare the main DataFrame for fitting MLForecast
    # It should contain unique_id, ds, y, and any dynamic exogenous features.
    df_for_ml_fit_cols = ['unique_id', 'ds', 'y']
    future_X_for_ml = None # For predict method

    if 'external_feature' in Y.columns:
        # Ensure external_feature is processed on the original Y DataFrame
        # as _prepare_future_X_df might also use it.
        Y['external_feature'] = pd.to_numeric(Y['external_feature'], errors='coerce').ffill().bfill()
        if not Y['external_feature'].isnull().all(): # Proceed only if external_feature is usable
            df_for_ml_fit_cols.append('external_feature')
            # future_X_for_ml will contain future values of external_feature
            future_X_for_ml = _prepare_future_X_df(Y, horizon, 'external_feature')
        else:
            st.warning("Coloana 'external_feature' conține numai NaN după procesare și nu va fi utilizată de modelele ML.")
    
    df_for_ml_fit = Y[df_for_ml_fit_cols]

    # MLForecast's fit method expects dynamic exogenous features to be part of the main DataFrame.
    # The X_df parameter in fit is for static features.
    mlf.fit(df_for_ml_fit, static_features=[], prediction_intervals=PredictionIntervals())
    fcst = mlf.predict(h=horizon, X_df=future_X_for_ml, level=[90])
    return mlf, fcst

# ───────────────────────── COMBINARE PREVIZIUNI ───────────────────────── #

@st.cache_data
def combine_forecasts(sf_fcst: pd.DataFrame, mlf_fcst: pd.DataFrame) -> pd.DataFrame:
    return sf_fcst.merge(mlf_fcst, on=['unique_id', 'ds'], how='left')