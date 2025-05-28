import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
from state_tools import init_state   # proprie
init_state() # Initialize session state, including 'themebutton'

st.set_page_config(page_title="🚀 Pipeline de Prognoză", layout="wide")

# Apply theme based on session state
# IMPORTANT: This must be called before other elements for the theme to apply correctly on first load/rerun
current_theme = st.session_state.get('themebutton', 'dark') # Get theme

if current_theme == 'dark':
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
# Load the list of hashed passwords.
# For app.py, the path is relative to the file's current directory.
HASHED_PW_PATH = Path(__file__).parent / "hashed_pw.pkl"
with HASHED_PW_PATH.open("rb") as file:
    hashed_passwords = pickle.load(file)     # Assuming this is a list as per 1_...py

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
# Using cookie_name and key from 1_...py for consistency.
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name="demo_app_cookie",   # Standardized cookie name
    key="demo_app_signature",        # Standardized secret key
    cookie_expiry_days=30,
)

# Draw the login form.
# The login call in app.py was specific, trying to keep its essence
# but standardizing the retrieval of auth status.
# Using the login call from the original app.py for the main page,
# as it uses specific parameters for form name and location.
authenticator.login()

# -------------------------------------------------------------------
#  🔑 Handle the authentication state held in `st.session_state`
# -------------------------------------------------------------------
# streamlit-authenticator sets these session_state variables after login() is called.
auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name") # Use name from session_state for consistency
# username = st.session_state.get("username") # Available if needed

if auth_status:
    # -------------------- Logged-in area --------------------
# Theme toggle button
    
    button_label = "☾" if current_theme == 'light' else "🌣"
    if st.sidebar.button(button_label, key="theme_toggle_button"):
        selected = st.session_state['themebutton']
        if selected=='light':
            #st._config.set_option(f'theme.backgroundColor' ,"white" )
            st._config.set_option(f'theme.base' ,"dark" )
            st._config.set_option('theme.backgroundColor', "#1c1c1e")           
            st._config.set_option('theme.secondaryBackgroundColor', "#2c2c2e")  
            st._config.set_option('theme.primaryColor', "#ff79c6")              
            st._config.set_option('theme.textColor', "#f8f8f2")                 
            st.session_state['themebutton'] = 'dark'
        else:
            st._config.set_option('theme.base', "light")
            st._config.set_option('theme.backgroundColor', "#fdfdfd")           # warm white
            st._config.set_option('theme.secondaryBackgroundColor', "#e6f0ff")  # soft blue background
            st._config.set_option('theme.primaryColor', "#1e90ff")              # dodger blue
            st._config.set_option('theme.textColor', "#1a1a1a")                 # dark gray text
            st.session_state['themebutton'] = 'light'
        st.rerun()

    authenticator.logout("Logout", "sidebar")
    # No st.sidebar.success here as per original app.py structure, main title serves as welcome.

    st.title(f"📊 Dashboard Pipeline - Welcome *{name}*") # Uses name from session_state
    st.markdown(
        """
        Folosește meniul din stânga pentru a încărca date, a antrena modele și
        a descărca previziunile. Progresul este salvat în *session_state*,
        deci poți să sari între pagini fără să pierzi nimic.
        """
    )

    if st.session_state.pipeline_ran:
        st.success("Pipeline-ul a fost deja rulat ✅")
        st.write("📈 **Previziuni disponibile:**", st.session_state.forecast_df.shape)
        st.write("🏅 **Cel mai bun model:**", st.session_state.best_model)
    else:
        st.info("Începe cu pagina **Upload & Config** ➡️")

elif auth_status is False:
    st.error('Username/password is incorrect')
elif auth_status is None: # Covers the case where login form is displayed or not yet interacted with
    st.warning('Please enter your username and password')	