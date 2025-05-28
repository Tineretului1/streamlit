import streamlit as st
from forecasting_models import run_statsforecast_models, run_mlforecast_models, combine_forecasts
from evaluation import run_all_cross_validation_and_evaluation
from state_tools import init_state
# import pickle # Not needed for Supabase auth
# from pathlib import Path # Not needed for Supabase auth
# import streamlit_authenticator as stauth # Removed
import pandas as pd
init_state() # Keep for standalone page runs

current_theme_to_apply = st.session_state.get('themebutton', 'light') # Get theme, default to light if not set

if current_theme_to_apply == 'dark':
    st._config.set_option('theme.base', "dark")
    st._config.set_option('theme.backgroundColor', "#1c1c1e")           # dark gray (background)
    st._config.set_option('theme.secondaryBackgroundColor', "#2c2c2e")  # slightly lighter dark gray
    st._config.set_option('theme.primaryColor', "#ff79c6")              # soft pink
    st._config.set_option('theme.textColor', "#f8f8f2")                 # light neutral text
else:  # Light theme
    st._config.set_option('theme.base', "light")
    st._config.set_option('theme.backgroundColor', "#fdfdfd")           # warm white
    st._config.set_option('theme.secondaryBackgroundColor', "#e6f0ff")  # soft blue background
    st._config.set_option('theme.primaryColor', "#1e90ff")              # dodger blue
    st._config.set_option('theme.textColor', "#1a1a1a")                 # dark gray text

# ----------------------------------
#  🔐 Authentication Check (using Supabase from app.py)
# ----------------------------------
if not st.session_state.get("user_email"):
    st.warning("🔑 Please log in to access this page.")
    try:
        st.page_link("app.py", label="Go to Login Page", icon="🏠")
    except AttributeError:
        st.info("Navigate to the main page to log in.")
    st.stop()
else:
    # -------------------- Logged-in area --------------------
    st.sidebar.success(f"Logged in as **{st.session_state.get('user_email', '')}**")

     # Initialize state for authenticated users

    # --- Page specific content ---
    st.header(f"🤖 Antrenare Modele & Cross-Validation - Welcome *{st.session_state.get('user_email', '')}*")

    if st.session_state.get("pipeline_ran"):
        st.success("Modelele sunt deja antrenate. Treci la pagina următoare ➡️")
        st.stop()

    Y_df_original = st.session_state.get("Y_df")
    if Y_df_original is None:
        st.info("Încă nu există date procesate. Rulează întâi **Upload & Config**.")
        st.stop()

    Y_df = Y_df_original.copy() # Work with a copy

    # --- Centralized Processing of external_feature ---
    exog_feature_usable = False
    if 'external_feature' in Y_df.columns:
        st.write("Procesare caracteristică exogenă în Y_df...")
        Y_df['external_feature'] = pd.to_numeric(Y_df['external_feature'], errors='coerce')
        if not Y_df['external_feature'].isnull().all():
            Y_df['external_feature'] = Y_df['external_feature'].ffill().bfill()
            if not Y_df['external_feature'].isnull().all():
                exog_feature_usable = True
                st.write("Caracteristica exogenă a fost procesată și este utilizabilă.")
            else:
                st.warning("Caracteristica exogenă ('external_feature') conține numai valori NaN după procesare completă. Nu va fi utilizată de modelele care o necesită.")
        else:
            st.warning("Caracteristica exogenă ('external_feature') a devenit integral NaN după conversia la numeric. Nu va fi utilizată.")
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
        mlf_model_no_exog, mlf_model_with_exog, mlf_forecast_no_exog, mlf_forecast_with_exog = \
            run_mlforecast_models(Y_df, horizon, window_size_ml)

    forecast_df = combine_forecasts(sf_forecast, mlf_forecast_no_exog, mlf_forecast_with_exog)
    
    st.session_state.mlf_model_no_exog = mlf_model_no_exog
    st.session_state.mlf_model_with_exog = mlf_model_with_exog

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

# Removed old authenticator messages
# elif auth_status is False:
#     st.error('Username/password is incorrect')
# else: # auth_status is None
#     st.warning('Please enter your username and password')