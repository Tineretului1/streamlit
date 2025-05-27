import streamlit as st
from forecasting_models import run_statsforecast_models, run_mlforecast_models, combine_forecasts
from evaluation import run_all_cross_validation_and_evaluation
from state_tools import init_state
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import pandas as pd

# ----------------------------------
#  🔐 Authentication Configuration
# ----------------------------------
# Load the list of hashed passwords.
HASHED_PW_PATH = Path(__file__).resolve().parent.parent / "hashed_pw.pkl"
with HASHED_PW_PATH.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Define users and build the credentials dictionary.
NAMES = ["Sandru Rares", "Trial Account"]
USERNAMES = ["rrares", "trial"]

credentials = {
    "usernames": {
        un: {"name": nm, "password": pw}
        for un, nm, pw in zip(USERNAMES, NAMES, hashed_passwords)
    }
}

# Instantiate the authenticator.
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="demo_app_cookie",   # Standardized cookie name
    key="demo_app_signature",        # Standardized secret key
    cookie_expiry_days=30,
)

# Draw the login form.
# This call also sets session_state variables: 'name', 'authentication_status', 'username'
authenticator.login() # Using the standard login call

# -------------------------------------------------------------------
#  🔑 Handle the authentication state held in `st.session_state`
# -------------------------------------------------------------------
auth_status = st.session_state.get("authentication_status")
name_from_session = st.session_state.get("name") # Get name from session_state
# username_from_session = st.session_state.get("username") # Available if needed

if auth_status:
    # -------------------- Logged-in area --------------------
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Logged in as **{name_from_session}**") # Display name from session_state

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

elif auth_status is False:
    st.error('Username/password is incorrect')
else: # auth_status is None
    st.warning('Please enter your username and password')
    # Optionally, guide to login if on a sub-page and not logged in.
    # st.info("Please log in to access this page.")