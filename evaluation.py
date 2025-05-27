# evaluation.py

import streamlit as st
import pandas as pd
from statsforecast import StatsForecast # For type hinting if needed, and cross_validation
from mlforecast import MLForecast # For type hinting if needed, and cross_validation
from utils import mse, mae, mape, smape # Assuming utils.py is in the same directory

# ───────────────────────── CROSS‑VALIDARE ───────────────────────── #

@st.cache_data
def perform_cross_validation(_sf: StatsForecast, _mlf: MLForecast,
                             Y: pd.DataFrame, horizon: int) -> pd.DataFrame:
    cv_sf  = _sf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon)
    cv_mlf = _mlf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon,
                                  level=[90], static_features=[])
    return cv_sf.merge(cv_mlf.drop(columns=['y']),
                       on=['unique_id', 'ds', 'cutoff'], how='left')

# ───────────────────────── EVALUARE ───────────────────────── #

@st.cache_data
def evaluate_cross_validation_results(cv_df: pd.DataFrame) -> pd.DataFrame:
    model_cols = [c for c in cv_df.columns if c not in
                  ['unique_id', 'y', 'ds', 'cutoff', 'lo-90', 'hi-90']]
    records = []
    for (uid, cutoff), group in cv_df.groupby(['unique_id', 'cutoff']):
        for m in model_cols:
            y_true = group['y']
            y_pred = group[m]
            records.extend([
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'mse',  'model': m, 'error': mse(y_true, y_pred)},
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'mae',  'model': m, 'error': mae(y_true, y_pred)},
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'mape', 'model': m, 'error': mape(y_true, y_pred)},
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'smape','model': m, 'error': smape(y_true, y_pred)}
            ])
    return pd.DataFrame(records)

# ─────────────────── SELECTARE MODEL BEST‑OF‑FOUR ─────────────────── #

@st.cache_data
def choose_best_forecasting_model(eval_df: pd.DataFrame):
    """Return the overall‑best model and a leaderboard DataFrame."""
    leaderboard = (
        eval_df
        .groupby(['model', 'metric'])['error']
        .mean()
        .unstack()
    )
    leaderboard['composite'] = leaderboard[['mse', 'mae', 'mape', 'smape']].mean(axis=1)
    best_model = leaderboard['composite'].idxmin()
    return best_model, leaderboard.sort_values('composite')