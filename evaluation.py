# evaluation.py

import streamlit as st
import pandas as pd
from statsforecast import StatsForecast # For type hinting if needed, and cross_validation
from mlforecast import MLForecast # For type hinting if needed, and cross_validation
from utils import mse, mae, mape, smape # Assuming utils.py is in the same directory

# ───────────────────────── CROSS‑VALIDARE ───────────────────────── #

# @st.cache_data
# def perform_cross_validation(_sf: StatsForecast, _mlf: MLForecast,
# Y: pd.DataFrame, horizon: int) -> pd.DataFrame:
#     cv_sf  = _sf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon)
#     cv_mlf = _mlf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon,
#                                   level=[90], static_features=[])
#     return cv_sf.merge(cv_mlf.drop(columns=['y']),
#                        on=['unique_id', 'ds', 'cutoff'], how='left')

@st.cache_data
def _perform_single_ml_cross_validation(
    _mlf_model: MLForecast,  # Prefixed with underscore
    Y_df: pd.DataFrame,
    horizon: int,
    suffix: str,
    level_value: int = 90 # Default level for prediction intervals
) -> pd.DataFrame:
    """Performs CV for a single MLForecast model and suffixes its columns."""
    cv_df = _mlf_model.cross_validation( # Use prefixed argument
        df=Y_df,
        h=horizon,
        n_windows=3,
        step_size=horizon,
        level=[level_value] if level_value else None, # Pass level if specified
        static_features=[] # Assuming static features are handled within model setup or not used here
    )
    
    # Suffix the model names
    # Models are keys in _mlf_model.models dict
    model_base_names = list(_mlf_model.models.keys()) # Use prefixed argument
    rename_map = {name: f"{name}{suffix}" for name in model_base_names}
    
    # Also suffix prediction interval columns if they exist
    # Common pattern is ModelName-lo-LEVEL or ModelName-hi-LEVEL
    for base_name in model_base_names:
        if level_value:
            lo_col = f"{base_name}-lo-{level_value}"
            hi_col = f"{base_name}-hi-{level_value}"
            if lo_col in cv_df.columns:
                rename_map[lo_col] = f"{base_name}{suffix}-lo-{level_value}"
            if hi_col in cv_df.columns:
                rename_map[hi_col] = f"{base_name}{suffix}-hi-{level_value}"

    cv_df = cv_df.rename(columns=rename_map)
    return cv_df

@st.cache_data
def run_all_cross_validation_and_evaluation(
    _sf_model: StatsForecast,             # Prefixed with underscore
    _mlf_model_no_exog: MLForecast,       # Prefixed with underscore
    _mlf_model_with_exog: MLForecast | None, # Prefixed with underscore
    Y_df: pd.DataFrame,
    horizon: int
):
    """
    Runs cross-validation for all models (StatsForecast, MLForecast without exog,
    and MLForecast with exog if available).
    Combines CV results, evaluates them, and chooses the best overall model.
    Ensures ML model names in CV/eval results are suffixed correctly.
    """
    all_cv_dfs = []
    
    # Y_df is assumed to have 'external_feature' processed by pages/3_...py
    # No need for Y_df_for_exog_cv and its processing here anymore.

    # 1. StatsForecast CV
    # AutoETS within _sf_model will use 'external_feature' if present in Y_df.
    st.write("Performing StatsForecast Cross-Validation...")
    cv_sf = _sf_model.cross_validation(df=Y_df, h=horizon, n_windows=3, step_size=horizon)
    all_cv_dfs.append(cv_sf)

    # 2. MLForecast without exogenous features CV
    # For this, we need to ensure 'external_feature' is NOT passed if it exists in Y_df.
    # So, we create a view of Y_df without it.
    st.write("Performing MLForecast (no exogenous) Cross-Validation...")
    cols_for_no_exog_cv = ['unique_id', 'ds', 'y']
    # Add other columns from Y_df that are not 'external_feature' if any were dynamic and not date/lag related
    # For simplicity, assuming only 'external_feature' is the dynamic one we control here.
    # If other dynamic features were part of Y_df and intended for no_exog, this needs adjustment.
    # However, mlf_model_no_exog was fit on Y[['unique_id', 'ds', 'y']], so this is consistent.
    Y_df_no_exog_view = Y_df[cols_for_no_exog_cv]
    cv_mlf_no_exog = _perform_single_ml_cross_validation(
        _mlf_model_no_exog, Y_df_no_exog_view, horizon, "_no_exog"
    )
    all_cv_dfs.append(cv_mlf_no_exog.drop(columns=['y']))

    # 3. MLForecast with exogenous features CV (if model exists)
    if _mlf_model_with_exog:
        st.write("Performing MLForecast (with exogenous) Cross-Validation...")
        # This model requires 'external_feature' to be present in Y_df for CV.
        # The Y_df passed to this function is assumed to have it processed.
        if 'external_feature' in Y_df.columns and not Y_df['external_feature'].isnull().all():
            # Pass Y_df directly, as it contains the processed external_feature
            cv_mlf_with_exog = _perform_single_ml_cross_validation(
                _mlf_model_with_exog, Y_df, horizon, "_with_exog"
            )
            all_cv_dfs.append(cv_mlf_with_exog.drop(columns=['y']))
        else:
            st.warning("Skipping MLForecast (with exogenous) CV as 'external_feature' is not usable in the input Y_df.")

    # Merge all CV results
    # Start with the first df (cv_sf which contains 'y')
    final_cv_df = all_cv_dfs[0]
    for i in range(1, len(all_cv_dfs)):
        final_cv_df = final_cv_df.merge(all_cv_dfs[i], on=['unique_id', 'ds', 'cutoff'], how='left')
    
    st.write("Evaluating combined cross-validation results...")
    eval_df = evaluate_cross_validation_results(final_cv_df)
    
    st.write("Choosing best overall model...")
    best_model, leaderboard = choose_best_forecasting_model(eval_df)
    
    return final_cv_df, eval_df, best_model, leaderboard

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