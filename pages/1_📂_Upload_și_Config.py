import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

import config
from data_processing import load_and_prepare
from state_tools import init_state
import streamlit_authenticator as stauth
init_state()

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
#  🔐 Authentication Configuration
# ----------------------------------
# Load the list of hashed passwords that you generated beforehand.
HASHED_PW_PATH = Path(__file__).resolve().parent.parent / "hashed_pw.pkl"
with HASHED_PW_PATH.open("rb") as file:
    hashed_passwords = pickle.load(file)

# Define users and build the credentials dictionary required by
# **streamlit-authenticator ≥ 0.3**
NAMES = ["Sandru Rares", "Trial Account"]
USERNAMES = ["rrares", "trial"]

credentials = {
    "usernames": {
        un: {"name": nm, "password": pw}
        for un, nm, pw in zip(USERNAMES, NAMES, hashed_passwords)
    }
}

# Instantiate the authenticator
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="demo_app_cookie",   # 👈 unique cookie name
    key="demo_app_signature",        # 👈 random secret key
    cookie_expiry_days=30,
)

# Draw the login form (or nothing if already logged-in)
authenticator.login()  # form title & location in the layout

# -------------------------------------------------------------------
#  🔑 Handle the authentication state held in `st.session_state`
# -------------------------------------------------------------------
auth_status = st.session_state.get("authentication_status")

if auth_status:
    # -------------------- Logged-in area --------------------
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Logged in as **{st.session_state.get('name', '')}**")


    st.header("📂 Upload & Config")

    # ------------------------- Sidebar controls -------------------------
    uploaded_file = st.sidebar.file_uploader(
        "Încarcă fișierul train.csv", type="csv", key="train_csv"
    )
    uploaded_external_file = st.sidebar.file_uploader(
        "Încarcă fișierul de date externe (ex: preț gaz)", type="csv", key="external_csv"
    )

    horizon = st.sidebar.number_input(
        "Orizont de Prognoză (zile)", min_value=1, value=config.HORIZON_DEFAULT
    )
    season_length = st.sidebar.number_input(
        "Lungimea Sezonului (zile)", min_value=1, value=config.SEASON_LENGTH_DEFAULT
    )
    window_size_ml = st.sidebar.number_input(
        "Fereastră ML (lags)", min_value=1, value=config.WINDOW_SIZE_DEFAULT
    )
    max_rows = st.sidebar.number_input(
        "Număr Maxim de Rânduri", min_value=1000, value=config.MAX_ROWS_DEFAULT, step=10000
    )

    stores_str = st.sidebar.text_input("Magazine (virgulă separat)", "")

    run_btn = st.button("🚀 Rulează Pipeline-ul de Prognoză")

    # --------------------------- Main logic ----------------------------
    if run_btn:
        if uploaded_file is None:
            st.warning("⚠️ Încarcă un fișier CSV mai întâi.")
            st.stop()

        with st.spinner("⏳ Se încarcă și pregătesc datele…"):
            stores_list = [s.strip() for s in stores_str.split(",") if s.strip()] or None
            Y_df = load_and_prepare(
                uploaded_file,
                uploaded_external_file=uploaded_external_file,
                stores=stores_list,
                max_rows=max_rows,
            )

        if Y_df.empty:
            st.error("Datele nu au putut fi procesate.")
            st.stop()

        # Persist everything we need in the session for later pages
        st.session_state.update(
            Y_df=Y_df,
            uploaded_external_file=uploaded_external_file,
            horizon=horizon,
            season_length=season_length,
            window_size_ml=window_size_ml,
            pipeline_ran=False,  # will be flipped once models are trained
        )

        st.success("✅ Date pregătite! Continuă la paginile următoare.")
        st.dataframe(Y_df.head())

        # Offer a download of the processed data
        st.download_button(
            label="📥 Descarcă Datele Procesate (Y_df.csv)",
            data=Y_df.to_csv(index=False).encode("utf-8"),
            file_name="Y_df_processed.csv",
            mime="text/csv",
        )

elif auth_status is False:
    st.error("Username/password is incorrect")
else:
    st.warning("Please enter your username and password")
