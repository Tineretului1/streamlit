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
model_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds")]
options = ["🏆 Cel mai bun model"] + model_cols
choice = st.selectbox("Selectează previziunea:", options)

export_col = best_model if choice == "🏆 Cel mai bun model" else choice
export_df = forecast_df[["unique_id", "ds", export_col]].rename(columns={export_col: "yhat"})

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", data=csv, file_name=f"forecast_{export_col}.csv")
st.dataframe(export_df.head())
