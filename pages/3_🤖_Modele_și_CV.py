import streamlit as st
from forecasting_models import run_statsforecast_models, run_mlforecast_models, combine_forecasts
# from evaluation import run_all_cross_validation_and_evaluation, evaluate_cross_validation_results, choose_best_forecasting_model
from evaluation import run_all_cross_validation_and_evaluation # Simplified import
from state_tools import init_state
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import pandas as pd # Ensure pandas is imported

# --- Authentication (aligned with app.py) ---
# --- încărcare parole ---
file_path = Path(__file__).resolve().parent.parent / "hashed_pw.pkl" # Adjusted path
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

names      = ["Sandru Rares", "Trial Account"]
usernames  = ["rrares", "trial"]

credentials = {"usernames": {}}
for idx, un in enumerate(usernames):
    credentials["usernames"][un] = {
        "name": names[idx],
        "password": hashed_passwords[idx]
    }

authenticator = stauth.Authenticate(
    credentials=credentials, # Use keyword arguments
    cookie_name="some_cookie_name",    # Must match app.py
    key="some_signature_key",          # Must match app.py
    cookie_expiry_days=30
)

# --- Retrieve authentication status from session state ---
name_from_session = st.session_state.get("name")
authentication_status_from_session = st.session_state.get("authentication_status")
# username_from_session = st.session_state.get("username") # Uncomment if 'username' is needed

# --- Page logic based on authentication status ---
if authentication_status_from_session:
    authenticator.logout("Logout", "sidebar")
    init_state() # Initialize state for authenticated users

    # --- Page specific content ---
    st.header(f"🤖 Antrenare Modele & Cross-Validation - Welcome *{name_from_session}*")

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
    
    st.session_state.mlf_model_no_exog = mlf_model_no_exog # Ensure these are in session state before CV call
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

elif authentication_status_from_session == False:
    st.error('Username/password is incorrect')
elif authentication_status_from_session is None:
    st.warning('Please enter your username and password')
    st.info("Please log in through the main application page to access this page.")