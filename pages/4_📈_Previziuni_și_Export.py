import streamlit as st
from visualization import visualize_forecasting_results
from state_tools import init_state
import pickle
from pathlib import Path
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

    st.header("📈 Vizualizări & Export Previziuni")

    if not st.session_state.get("pipeline_ran"):
        st.info("Rulează pagina **Modele & CV** înainte de export.")
        st.stop()

    forecast_df = st.session_state.forecast_df
    best_model = st.session_state.best_model
    leaderboard = st.session_state.leaderboard

    # --- Vizualizări
    visualize_forecasting_results(
        st.session_state.Y_df,
        forecast_df,
        st.session_state.cv_df,
        st.session_state.eval_df,
        st.session_state.leaderboard,
        st.session_state.best_model,
        st.session_state.horizon,
    )

    # --- Export logic
    st.subheader("📁 Exportă Previziuni")
    all_forecast_columns = [
        c for c in forecast_df.columns
        if c not in ("unique_id", "ds", "y") and not c.endswith("-lo-90") and not c.endswith("-hi-90")
    ]
    options = ["🏆 Cel mai bun model"] + sorted(list(set(all_forecast_columns)))
    choice = st.selectbox("Selectează previziunea pentru export:", options)

    export_col_name = None
    if choice == "🏆 Cel mai bun model":
        potential_with_exog = f"{best_model}_with_exog"
        potential_no_exog = f"{best_model}_no_exog"
        if potential_with_exog in forecast_df.columns:
            export_col_name = potential_with_exog
        elif potential_no_exog in forecast_df.columns:
            export_col_name = potential_no_exog
        elif best_model in forecast_df.columns:
            export_col_name = best_model
        else:
            st.error(f"Nu s-a putut găsi coloana pentru cel mai bun model '{best_model}' în setul de date al previziunilor.")
            st.stop()
    else:
        export_col_name = choice

    if export_col_name and export_col_name in forecast_df.columns:
        export_df = forecast_df[["unique_id", "ds", export_col_name]].rename(columns={export_col_name: "yhat"})
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name=f"forecast_{export_col_name}.csv")
        st.dataframe(export_df.head())

elif auth_status is False:
    st.error("Username/password is incorrect")
else: # auth_status is None
    st.warning("Please enter your username and password")
    # Optionally, guide to login if on a sub-page and not logged in.
    # st.info("Please log in to access this page.")