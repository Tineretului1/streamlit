# app.py

import streamlit as st
import pandas as pd

# Importuri din modulele create
import config
from data_processing import load_and_prepare
from exploratory_analysis import perform_exploratory_analysis
from forecasting_models import run_statsforecast_models, run_mlforecast_models, combine_forecasts
from evaluation import perform_cross_validation, evaluate_cross_validation_results, choose_best_forecasting_model
from visualization import visualize_forecasting_results

# ───────────────────────── STREAMLIT UI ───────────────────────── #
st.set_page_config(layout="wide")
st.title("🚀 Pipeline de Prognoză Îmbunătățit")

# --- Sidebar pentru configurații ---
st.sidebar.header("⚙️ Configurații")
uploaded_file = st.sidebar.file_uploader("Încarcă fișierul train.csv", type="csv")

horizon = st.sidebar.number_input("Orizont de Prognoză (zile)", min_value=1, value=config.HORIZON_DEFAULT, step=1)
season_length = st.sidebar.number_input("Lungimea Sezonului (zile, ex: 7 pentru săptămânal)", min_value=1, value=config.SEASON_LENGTH_DEFAULT, step=1)
window_size_ml = st.sidebar.number_input("Fereastră ML pentru Lags (zile)", min_value=1, value=config.WINDOW_SIZE_DEFAULT, step=1) # Note: this param might need review if it's for MLForecast lags or other use
max_rows_to_load = st.sidebar.number_input("Număr Maxim de Rânduri de Încărcat", min_value=1000, value=config.MAX_ROWS_DEFAULT, step=10000, help="Limitează numărul de rânduri citite din CSV pentru performanță.")

stores_input_str = st.sidebar.text_input("Magazine (separate prin virgulă, ex: 1,2). Lăsați gol pentru toate.", "", help="Specificați ID-urile magazinelor. Dacă este gol, se vor folosi toate magazinele din date.")

run_pipeline = st.sidebar.button("🚀 Rulează Pipeline-ul de Prognoză")

# --- Panoul Principal ---
if run_pipeline:
    if uploaded_file is not None:
        with st.spinner("⏳ Se încarcă și se pregătesc datele..."):
            stores_list = [s.strip() for s in stores_input_str.split(',') if s.strip()] if stores_input_str else None
            Y_df = load_and_prepare(uploaded_file, stores=stores_list, max_rows=max_rows_to_load)

        if not Y_df.empty:
            st.success(f"✅ Date încărcate și pregătite: {Y_df.shape[0]} rânduri, {Y_df['unique_id'].nunique()} serii unice.")
            st.dataframe(Y_df.head())

            with st.spinner("📊 Se efectuează analiza exploratorie..."):
                perform_exploratory_analysis(Y_df)
            st.success("✅ Analiza exploratorie finalizată.")
            st.markdown("---")

            with st.spinner("🧠 Se antrenează modelele StatsForecast și se generează previziuni..."):
                sf_model, sf_forecast = run_statsforecast_models(Y_df, horizon, season_length)
            st.success("✅ Modelele StatsForecast antrenate.")

            with st.spinner("🧠 Se antrenează modelele MLForecast și se generează previziuni..."):
                mlf_model, mlf_forecast = run_mlforecast_models(Y_df, horizon, window_size_ml) # window_size_ml is _h_param_deprecated
            st.success("✅ Modelele MLForecast antrenate.")
            st.markdown("---")

            with st.spinner("🔗 Se combină previziunile..."):
                forecast_df = combine_forecasts(sf_forecast, mlf_forecast)
            st.success("✅ Previziuni combinate.")
            st.dataframe(forecast_df.head())
            st.markdown("---")

            with st.spinner("🔄 Se efectuează validarea încrucișată..."):
                cv_df = perform_cross_validation(sf_model, mlf_model, Y_df, horizon)
            st.success("✅ Validare încrucișată finalizată.")
            st.dataframe(cv_df.head())
            st.markdown("---")

            with st.spinner("📉 Se evaluează modelele..."):
                eval_df = evaluate_cross_validation_results(cv_df)
            st.success("✅ Evaluare finalizată.")
            st.markdown("---")

            st.header("🏆 Clasament și Cel Mai Bun Model")
            with st.spinner("🏅 Se alege cel mai bun model..."):
                best_model, leaderboard = choose_best_forecasting_model(eval_df)
            st.success(f"🎉 Cel mai bun model per ansamblu (bazat pe media celor 4 metrici): **{best_model}**")

            st.subheader("Clasament General al Modelelor")
            st.dataframe(leaderboard)
            csv_leaderboard = leaderboard.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Descarcă Clasamentul (CSV)",
                data=csv_leaderboard,
                file_name='leaderboard.csv',
                mime='text/csv',
            )
            st.markdown("---")

            st.header("📊 Vizualizări Diagnostice")
            with st.spinner("🎨 Se generează vizualizările..."):
                visualize_forecasting_results(Y_df, forecast_df, cv_df, eval_df, horizon)
            st.success("✅ Vizualizări generate.")
            st.markdown("---")

            st.header("📁 Exportă Previziunile Celui Mai Bun Model")
            best_model_forecast_df = (
                forecast_df[['unique_id', 'ds', best_model]]
                .rename(columns={best_model: 'yhat'})
            )
            st.dataframe(best_model_forecast_df.head())

            csv_export = best_model_forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Descarcă Previziunile pentru {best_model} (CSV)",
                data=csv_export,
                file_name='best_model_forecast.csv',
                mime='text/csv',
            )
            st.balloons()
            st.info(f"Toate graficele sunt afișate mai sus. Rezultatele principale (clasament, previziuni) sunt disponibile pentru descărcare.")

        elif uploaded_file is not None:
             st.error("❌ Datele nu au putut fi procesate. Verificați fișierul și configurațiile.")

    elif run_pipeline and uploaded_file is None:
        st.warning("⚠️ Vă rugăm să încărcați un fișier `train.csv` pentru a începe.")

else:
    st.info("ℹ️ Configurați parametrii în bara laterală și apăsați 'Rulează Pipeline-ul de Prognoză'.")