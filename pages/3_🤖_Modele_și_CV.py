import streamlit as st
from forecasting_models import run_statsforecast_models, run_mlforecast_models, combine_forecasts
from evaluation import run_all_cross_validation_and_evaluation, evaluate_cross_validation_results, choose_best_forecasting_model
from state_tools import init_state

init_state()
st.header("🤖 Antrenare Modele & Cross-Validation")

if st.session_state.get("pipeline_ran"):
    st.success("Modelele sunt deja antrenate. Treci la pagina următoare ➡️")
    st.stop()

import pandas as pd # Add import for pd

Y_df_original = st.session_state.get("Y_df")
if Y_df_original is None:
    st.info("Încă nu există date procesate.")
    st.stop()

Y_df = Y_df_original.copy() # Work with a copy

# --- Centralized Processing of external_feature ---
exog_feature_usable = False
if 'external_feature' in Y_df.columns:
    st.write("Procesare caracteristică exogenă în Y_df...")
    Y_df['external_feature'] = pd.to_numeric(Y_df['external_feature'], errors='coerce')
    # Check if to_numeric resulted in all NaNs before ffill/bfill
    if not Y_df['external_feature'].isnull().all():
        Y_df['external_feature'] = Y_df['external_feature'].ffill().bfill()
        # Final check after ffill/bfill
        if not Y_df['external_feature'].isnull().all():
            exog_feature_usable = True
            st.write("Caracteristica exogenă a fost procesată și este utilizabilă.")
        else:
            st.warning("Caracteristica exogenă ('external_feature') conține numai valori NaN după procesare completă. Nu va fi utilizată de modelele care o necesită.")
            # Optionally, drop the column if it's all NaN to prevent issues
            # Y_df = Y_df.drop(columns=['external_feature'])
    else:
        st.warning("Caracteristica exogenă ('external_feature') a devenit integral NaN după conversia la numeric. Nu va fi utilizată.")
        # Optionally, drop the column
        # Y_df = Y_df.drop(columns=['external_feature'])
else:
    st.write("Nu s-a găsit coloana 'external_feature' în Y_df.")
# --- End of Centralized Processing ---

horizon = st.session_state.horizon
season_length = st.session_state.season_length
window_size_ml = st.session_state.window_size_ml

# Train models
with st.spinner("🧠 StatsForecast…"):
    sf_model, sf_forecast = run_statsforecast_models(Y_df, horizon, season_length)

with st.spinner("🧠 MLForecast…"):
    # Unpack the four return values
    mlf_model_no_exog, mlf_model_with_exog, mlf_forecast_no_exog, mlf_forecast_with_exog = \
        run_mlforecast_models(Y_df, horizon, window_size_ml)

forecast_df = combine_forecasts(sf_forecast, mlf_forecast_no_exog, mlf_forecast_with_exog)

# Cross-validation & evaluation
# Decide which ML model object to pass for CV or perform CV for both and combine
# For now, let's assume we want to evaluate both ML model variants if the 'with_exog' exists
# This requires perform_cross_validation to be adapted or called twice.
# For simplicity in this step, we'll pass mlf_model_no_exog for now,
# and the next step will be to adapt evaluation.py
# TODO: Adapt evaluation.py to handle both mlf_model_no_exog and mlf_model_with_exog

# The 'mlf_model' variable was previously used for CV.
# We need to decide what 'best_model' refers to.
# Let's pass mlf_model_no_exog for now and adjust evaluation.py next.
# If mlf_model_with_exog exists, it should also be cross-validated.

# This part needs significant rework in the next step (evaluation.py)
# For now, to make the script runnable, we'll use the no_exog model for CV.
# This is a temporary measure.
current_ml_model_for_cv = mlf_model_no_exog
if mlf_model_with_exog:
    # Ideally, CV both and merge results. For now, prioritize with_exog if it exists.
    # This choice impacts what 'best_model' will be.
    # The visualization logic now expects eval_df to have suffixed names.
    # This means perform_cross_validation needs to handle this.
    pass # We will handle this in evaluation.py

# cv_df = perform_cross_validation(sf_model, current_ml_model_for_cv, Y_df, horizon)
# The above line will be replaced by logic in evaluation.py
# For now, we'll create an empty eval_df or a placeholder
# This is TEMPORARY until evaluation.py is updated.
st.session_state.mlf_model_no_exog = mlf_model_no_exog
st.session_state.mlf_model_with_exog = mlf_model_with_exog

# The actual CV and evaluation will be triggered from within evaluation.py adjustments
# For now, let's assume eval_df will be populated correctly later.
# eval_df = pd.DataFrame() # Placeholder
# leaderboard = pd.DataFrame() # Placeholder
# best_model = "N/A" # Placeholder

# Call a new function in evaluation.py that handles all CV and merging
from evaluation import run_all_cross_validation_and_evaluation
cv_df, eval_df, best_model, leaderboard = run_all_cross_validation_and_evaluation(
    sf_model, mlf_model_no_exog, mlf_model_with_exog, Y_df, horizon
)

# Persist in state
st.session_state.update(
    forecast_df=forecast_df,
    cv_df=cv_df,
    eval_df=eval_df,
    best_model=best_model,
    leaderboard=leaderboard,
    pipeline_ran=True,
)

st.success(f"🎉 Modele antrenate. Cel mai bun: **{best_model}**")
st.dataframe(leaderboard)
