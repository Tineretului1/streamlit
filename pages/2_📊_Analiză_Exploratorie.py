import streamlit as st
from exploratory_analysis import perform_exploratory_analysis
from state_tools import init_state
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

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
name = st.session_state.get("name") # Get name from session_state
# username = st.session_state.get("username") # Available if needed

if auth_status:
    # -------------------- Logged-in area --------------------
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Logged in as **{name}**") # Display name from session_state

    init_state() # Initialize state for authenticated users
    st.header("📊 Analiză Exploratorie")

    Y_df = st.session_state.get("Y_df")
    if Y_df is None:
        st.info("Întâi încarcă datele în pagina **Upload & Config**.")
        st.stop()

    perform_exploratory_analysis(Y_df)

elif auth_status is False:
    st.error("Username/password is incorrect")
else: # auth_status is None
    st.warning("Please enter your username and password")
    # Optionally, guide to login if on a sub-page and not logged in.
    # st.info("Please log in to access this page.")