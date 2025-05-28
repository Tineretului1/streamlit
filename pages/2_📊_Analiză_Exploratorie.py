import streamlit as st
from exploratory_analysis import perform_exploratory_analysis
from state_tools import init_state
# import pickle # Not needed for Supabase auth
# from pathlib import Path # Not needed for Supabase auth
# import streamlit_authenticator as stauth # Removed
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
    st.header("📊 Analiză Exploratorie")

    Y_df = st.session_state.get("Y_df")
    if Y_df is None:
        st.info("Întâi încarcă datele în pagina **Upload & Config**.")
        st.stop()

    perform_exploratory_analysis(Y_df)

# Removed old authenticator messages
# elif auth_status is False:
#     st.error("Username/password is incorrect")
# else: # auth_status is None
#     st.warning("Please enter your username and password")