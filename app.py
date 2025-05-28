import pickle
from pathlib import Path
import streamlit as st
from supabase import create_client, Client
from state_tools import init_state   # proprie

# Initialize session state, including 'themebutton'
init_state()

# Set up the page
st.set_page_config(page_title="🚀 Pipeline de Prognoză", layout="wide")

# ----------------------------------
#  🔐 Supabase Authentication Setup
# ----------------------------------
# Initialize Supabase client
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(supabase_url, supabase_key)

# Authentication functions

def sign_up(email, password):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password})
        return user
    except Exception as e:
        st.error(f"Registration failed: {e}")


def sign_in(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password})
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")


def sign_out():
    try:
        supabase.auth.sign_out()
        st.session_state.user_email = None

        # Clear data related to training and files
        keys_to_delete = [
            "Y_df", "uploaded_external_file", "horizon", "season_length",
            "window_size_ml", "pipeline_ran", "forecast_df", "cv_df",
            "eval_df", "best_model", "leaderboard", "mlf_model_no_exog",
            "mlf_model_with_exog", "train_csv", "external_csv"
        ]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
        
        st.success("Successfully logged out and cleared session data.") # Optional: provide feedback
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")

# ----------------------------------
#  🎨 Theme Configuration
# ----------------------------------
current_theme = st.session_state.get('themebutton', 'dark')
if current_theme == 'dark':
    st._config.set_option('theme.base', "dark")
    st._config.set_option('theme.backgroundColor', "#1c1c1e")
    st._config.set_option('theme.secondaryBackgroundColor', "#2c2c2e")
    st._config.set_option('theme.primaryColor', "#ff79c6")
    st._config.set_option('theme.textColor', "#f8f8f2")
else:
    st._config.set_option('theme.base', "light")
    st._config.set_option('theme.backgroundColor', "#fdfdfd")
    st._config.set_option('theme.secondaryBackgroundColor', "#e6f0ff")
    st._config.set_option('theme.primaryColor', "#1e90ff")
    st._config.set_option('theme.textColor', "#1a1a1a")

# ----------------------------------
#  🛡️ Authentication Screen
# ----------------------------------
def auth_screen():
    st.title("🔐 Streamlit & Supabase Auth App")
    action = st.selectbox("Choose an action:", ["Login", "Sign Up"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if action == "Sign Up" and st.button("Register"):
        user = sign_up(email, password)
        if user and getattr(user, 'user', None):
            st.success("Registration successful. Please log in.")

    if action == "Login" and st.button("Login"):
        user = sign_in(email, password)
        if user and getattr(user, 'user', None):
            st.session_state.user_email = user.user.email
            st.success(f"Welcome back, {email}!")
            st.rerun()

# Initialize session state for user_email
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# ----------------------------------
#  🎉 Main Application
# ----------------------------------
def main_app(user_email):
    # Theme toggle
    button_label = "☾" if current_theme == 'light' else "🌣"
    if st.sidebar.button(button_label, key="theme_toggle_button"):
        if st.session_state['themebutton'] == 'light':
            st._config.set_option('theme.base', "dark")
            st._config.set_option('theme.backgroundColor', "#1c1c1e")
            st._config.set_option('theme.secondaryBackgroundColor', "#2c2c2e")
            st._config.set_option('theme.primaryColor', "#ff79c6")
            st._config.set_option('theme.textColor', "#f8f8f2")
            st.session_state['themebutton'] = 'dark'
        else:
            st._config.set_option('theme.base', "light")
            st._config.set_option('theme.backgroundColor', "#fdfdfd")
            st._config.set_option('theme.secondaryBackgroundColor', "#e6f0ff")
            st._config.set_option('theme.primaryColor', "#1e90ff")
            st._config.set_option('theme.textColor', "#1a1a1a")
            st.session_state['themebutton'] = 'light'
        st.rerun()

    # Logout button
    if st.sidebar.button("Logout"):
        sign_out()

    # Application content
    st.title(f"📊 Dashboard Pipeline - Welcome *{user_email}*")
    st.markdown(
        """
        Folosește meniul din stânga pentru a încărca date, a antrena modele și
        a descărca previziunile. Progresul este salvat în *session_state*,
        deci poți să sari între pagini fără să pierzi nimic.
        """
    )

    if st.session_state.get('pipeline_ran', False):
        st.success("Pipeline-ul a fost deja rulat ✅")
        st.write("📈 **Previziuni disponibile:**", st.session_state.forecast_df.shape)
        st.write("🏅 **Cel mai bun model:**", st.session_state.best_model)
    else:
        st.info("Începe cu pagina **Upload & Config** ➡️")

# Determine which screen to show
if st.session_state.user_email:
    main_app(st.session_state.user_email)
else:
    auth_screen()
