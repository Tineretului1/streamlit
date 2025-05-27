import streamlit as st
from visualization import visualize_forecasting_results
from state_tools import init_state
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import pandas as pd # Ensure pandas is imported, might be needed by visualize_forecasting_results or for robustness

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
    st.header(f"📈 Vizualizări & Export Previziuni - Welcome *{name_from_session}*")

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
        leaderboard, 
        best_model,   
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
        potential_no_exog = f"{best_model}_no_exog" # Assuming ML models might not have suffix if exog not used
        if potential_with_exog in forecast_df.columns:
            export_col_name = potential_with_exog
        elif f"{best_model}" in forecast_df.columns and (potential_no_exog in forecast_df.columns and best_model.endswith("_no_exog")): # if best_model already has _no_exog
             export_col_name = best_model
        elif potential_no_exog in forecast_df.columns: # if best_model is base and _no_exog variant exists
             export_col_name = potential_no_exog
        elif best_model in forecast_df.columns: 
            export_col_name = best_model
        else:
            st.error(f"Nu s-a putut găsi coloana pentru cel mai bun model '{best_model}' în setul de date al previziunilor. Coloane disponibile: {forecast_df.columns.tolist()}")
            st.stop()
    else:
        export_col_name = choice

    if export_col_name and export_col_name in forecast_df.columns:
        export_df = forecast_df[["unique_id", "ds", export_col_name]].rename(columns={export_col_name: "yhat"})
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name=f"forecast_{export_col_name}.csv")
        st.dataframe(export_df.head())
    elif export_col_name: # if export_col_name was set but not found (should be caught above)
        st.error(f"Coloana selectată pentru export '{export_col_name}' nu a fost găsită.")

elif authentication_status_from_session == False:
    st.error('Username/password is incorrect')
elif authentication_status_from_session is None:
    st.warning('Please enter your username and password')
    st.info("Please log in through the main application page to access this page.")