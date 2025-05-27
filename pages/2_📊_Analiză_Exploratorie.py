import streamlit as st
from exploratory_analysis import perform_exploratory_analysis
from state_tools import init_state
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

# --- Authentication ---
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
    credentials,
    "some_cookie_name",
    "some_signature_key",
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login("Login", 'main')

if not authentication_status:
    if authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    st.stop()

authenticator.logout("Logout", "sidebar")
# --- End Authentication ---

init_state()
st.header("📊 Analiză Exploratorie")

Y_df = st.session_state.get("Y_df")
if Y_df is None:
    st.info("Întâi încarcă datele în pagina **Upload & Config**.")
    st.stop()

perform_exploratory_analysis(Y_df)
