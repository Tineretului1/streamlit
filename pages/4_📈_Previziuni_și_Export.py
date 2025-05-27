import streamlit as st
from visualization import visualize_forecasting_results
from state_tools import init_state

init_state()
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
    st.session_state.horizon,
)

# --- Export logic (unchanged from your original)
st.subheader("📁 Exportă Previziuni")
# Get all potential model forecast columns from the forecast_df
all_forecast_columns = [
    c for c in forecast_df.columns
    if c not in ("unique_id", "ds", "y") and not c.endswith("-lo-90") and not c.endswith("-hi-90")
]
options = ["🏆 Cel mai bun model"] + sorted(list(set(all_forecast_columns))) # Use set to ensure unique names
choice = st.selectbox("Selectează previziunea pentru export:", options)

export_col_name = None
if choice == "🏆 Cel mai bun model":
    # best_model is a base name like 'LGBMRegressor' or 'AutoETS'
    # We need to find the corresponding column in forecast_df
    # Prioritize '_with_exog' for ML models if it exists
    potential_with_exog = f"{best_model}_with_exog"
    potential_no_exog = f"{best_model}_no_exog"
    if potential_with_exog in forecast_df.columns:
        export_col_name = potential_with_exog
    elif potential_no_exog in forecast_df.columns:
        export_col_name = potential_no_exog
    elif best_model in forecast_df.columns: # For StatsForecast models or ML models if somehow no suffix was applied
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
