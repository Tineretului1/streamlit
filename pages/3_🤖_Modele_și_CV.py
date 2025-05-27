import streamlit as st
from forecasting_models import run_statsforecast_models, run_mlforecast_models, combine_forecasts
from evaluation import perform_cross_validation, evaluate_cross_validation_results, choose_best_forecasting_model
from state_tools import init_state

init_state()
st.header("🤖 Antrenare Modele & Cross-Validation")

if st.session_state.get("pipeline_ran"):
    st.success("Modelele sunt deja antrenate. Treci la pagina următoare ➡️")
    st.stop()

Y_df = st.session_state.get("Y_df")
if Y_df is None:
    st.info("Încă nu există date procesate.")
    st.stop()

horizon = st.session_state.horizon
season_length = st.session_state.season_length
window_size_ml = st.session_state.window_size_ml

# Train models
with st.spinner("🧠 StatsForecast…"):
    sf_model, sf_forecast = run_statsforecast_models(Y_df, horizon, season_length)

with st.spinner("🧠 MLForecast…"):
    mlf_model, mlf_forecast_no_exog, mlf_forecast_with_exog = run_mlforecast_models(Y_df, horizon, window_size_ml)

forecast_df = combine_forecasts(sf_forecast, mlf_forecast_no_exog, mlf_forecast_with_exog)

# Cross-validation & evaluation
cv_df = perform_cross_validation(sf_model, mlf_model, Y_df, horizon)
eval_df = evaluate_cross_validation_results(cv_df)
best_model, leaderboard = choose_best_forecasting_model(eval_df)

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
