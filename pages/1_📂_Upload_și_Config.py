import streamlit as st
import pandas as pd
import config
from data_processing import load_and_prepare
from state_tools import init_state
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

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
    st.header(f"📂 Upload & Config - Welcome *{name_from_session}*")

    # Sidebar controls
    uploaded_file = st.sidebar.file_uploader("Încarcă fișierul train.csv", type="csv")
    uploaded_external_file = st.sidebar.file_uploader("Încarcă fișierul de date externe (ex: preț gaz)", type="csv")
    horizon = st.sidebar.number_input("Orizont de Prognoză (zile)",
                                      min_value=1, value=config.HORIZON_DEFAULT)
    season_length = st.sidebar.number_input("Lungimea Sezonului (zile)",
                                            min_value=1, value=config.SEASON_LENGTH_DEFAULT)
    window_size_ml = st.sidebar.number_input("Fereastră ML (lags)",
                                             min_value=1, value=config.WINDOW_SIZE_DEFAULT)
    max_rows = st.sidebar.number_input("Număr Maxim de Rânduri",
                                       min_value=1000, value=config.MAX_ROWS_DEFAULT, step=10000)
    stores_str = st.sidebar.text_input("Magazine (virgulă)", "")

    run_btn = st.button("🚀 Rulează Pipeline-ul de Prognoză", type="primary")

    if run_btn:
        if uploaded_file is None:
            st.warning("⚠️ Încarcă un fișier CSV mai întâi.")
            st.stop()

        with st.spinner("⏳ Se încarcă și pregătesc datele…"):
            stores_list = [s.strip() for s in stores_str.split(",") if s.strip()] or None
            Y_df = load_and_prepare(uploaded_file, uploaded_external_file=uploaded_external_file, stores=stores_list, max_rows=max_rows)

        if Y_df.empty:
            st.error("Datele nu au putut fi procesate.")
            st.stop()

        # Save everything we will need later
        st.session_state.update(
            Y_df=Y_df,
            uploaded_external_file=uploaded_external_file,
            horizon=horizon,
            season_length=season_length,
            window_size_ml=window_size_ml,
            pipeline_ran=False,      # still false until models are trained
        )

        st.success("✅ Date pregătite! Continuă la paginile următoare.")
        st.dataframe(Y_df.head())

        # Add download button for the full Y_df
        csv_Y_df = Y_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descarcă Datele Procesate (Y_df.csv)",
            data=csv_Y_df,
            file_name='Y_df_processed.csv',
            mime='text/csv',
        )

elif authentication_status_from_session == False:
    st.error('Username/password is incorrect')
elif authentication_status_from_session is None:
    st.warning('Please enter your username and password')
    st.info("Please log in through the main application page to access this page.")