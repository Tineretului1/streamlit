import streamlit as st
from exploratory_analysis import perform_exploratory_analysis
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
    st.header(f"📊 Analiză Exploratorie - Welcome *{name_from_session}*")

    Y_df = st.session_state.get("Y_df")
    if Y_df is None:
        st.info("Întâi încarcă datele în pagina **Upload & Config**.")
        st.stop()

    perform_exploratory_analysis(Y_df)

elif authentication_status_from_session == False:
    st.error('Username/password is incorrect')
elif authentication_status_from_session is None:
    st.warning('Please enter your username and password')
    st.info("Please log in through the main application page to access this page.")