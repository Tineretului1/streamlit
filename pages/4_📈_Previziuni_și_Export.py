import streamlit as st
from visualization import visualize_forecasting_results
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

# Removed old authenticator messages
# elif auth_status is False:
#     st.error("Username/password is incorrect")
# else: # auth_status is None
#     st.warning("Please enter your username and password")