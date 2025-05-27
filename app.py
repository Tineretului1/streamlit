# app.py
import streamlit as st
import pandas as pd

# ────────── Importurile tale ──────────
import config
from data_processing import load_and_prepare
from exploratory_analysis import perform_exploratory_analysis
from forecasting_models import (
    run_statsforecast_models,
    run_mlforecast_models,
    combine_forecasts,
)
from evaluation import (
    perform_cross_validation,
    evaluate_cross_validation_results,
    choose_best_forecasting_model,
)
from visualization import visualize_forecasting_results

# ────────── Config pagină ──────────
st.set_page_config(layout="wide")
st.title("🚀 Pipeline de Prognoză Îmbunătățit")

# ────────── Inițializare session_state ──────────
if "pipeline_ran" not in st.session_state:
    st.session_state["pipeline_ran"] = False
    st.session_state["forecast_df"] = None
    st.session_state["leaderboard"] = None
    st.session_state["best_model"] = None

# ────────── Sidebar ──────────
st.sidebar.header("⚙️ Configurații")
uploaded_file = st.sidebar.file_uploader("Încarcă fișierul train.csv", type="csv")

horizon = st.sidebar.number_input(
    "Orizont de Prognoză (zile)", min_value=1, value=config.HORIZON_DEFAULT, step=1
)
season_length = st.sidebar.number_input(
    "Lungimea Sezonului (zile, ex: 7 pentru săptămânal)",
    min_value=1,
    value=config.SEASON_LENGTH_DEFAULT,
    step=1,
)
window_size_ml = st.sidebar.number_input(
    "Fereastră ML pentru Lags (zile)",
    min_value=1,
    value=config.WINDOW_SIZE_DEFAULT,
    step=1,
)

max_rows_to_load = st.sidebar.number_input(
    "Număr Maxim de Rânduri de Încărcat",
    min_value=1000,
    value=config.MAX_ROWS_DEFAULT,
    step=10000,
    help="Limitează numărul de rânduri citite din CSV pentru performanță.",
)

stores_input_str = st.sidebar.text_input(
    "Magazine (separate prin virgulă, ex: 1,2). Lăsați gol pentru toate.",
    "",
    help="Specificați ID-urile magazinelor. Dacă este gol, se vor folosi toate magazinele din date.",
)

run_pipeline_btn = st.sidebar.button("🚀 Rulează Pipeline-ul de Prognoză", type="primary")

# ═════════════════════════════════════════════════════════════════════
# 1) Rulăm pipeline-ul O SINGURĂ DATĂ și salvăm în session_state
# ═════════════════════════════════════════════════════════════════════
if run_pipeline_btn:
    if uploaded_file is None:
        st.warning("⚠️ Vă rugăm să încărcați un fișier `train.csv` pentru a începe.")
        st.stop()

    with st.spinner("⏳ Se încarcă și se pregătesc datele..."):
        stores_list = [s.strip() for s in stores_input_str.split(",") if s.strip()] or None
        Y_df = load_and_prepare(
            uploaded_file, stores=stores_list, max_rows=max_rows_to_load
        )

    if Y_df.empty:
        st.error("❌ Datele nu au putut fi procesate. Verificați fișierul și configurațiile.")
        st.stop()

    st.success(
        f"✅ Date încărcate și pregătite: {Y_df.shape[0]} rânduri, "
        f"{Y_df['unique_id'].nunique()} serii unice."
    )
    st.dataframe(Y_df.head())
    st.markdown("---")

    # Analiză exploratorie
    with st.spinner("📊 Se efectuează analiza exploratorie..."):
        perform_exploratory_analysis(Y_df)
    st.success("✅ Analiza exploratorie finalizată.")
    st.markdown("---")

    # StatsForecast
    with st.spinner("🧠 Se antrenează modelele StatsForecast..."):
        sf_model, sf_forecast = run_statsforecast_models(Y_df, horizon, season_length)
    st.success("✅ StatsForecast gata.")

    # MLForecast
    with st.spinner("🧠 Se antrenează modelele MLForecast..."):
        mlf_model, mlf_forecast = run_mlforecast_models(
            Y_df, horizon, window_size_ml
        )
    st.success("✅ MLForecast gata.")
    st.markdown("---")

    # Combinare
    with st.spinner("🔗 Se combină previziunile..."):
        forecast_df = combine_forecasts(sf_forecast, mlf_forecast)
    st.success("✅ Previziuni combinate.")
    st.dataframe(forecast_df.head())
    st.markdown("---")

    # Cross-validation
    with st.spinner("🔄 Se efectuează validarea încrucișată..."):
        cv_df = perform_cross_validation(sf_model, mlf_model, Y_df, horizon)
    st.success("✅ Validare încrucișată finalizată.")
    st.dataframe(cv_df.head())
    st.markdown("---")

    # Evaluare
    with st.spinner("📉 Se evaluează modelele..."):
        eval_df = evaluate_cross_validation_results(cv_df)
    st.success("✅ Evaluare finalizată.")
    st.markdown("---")

    # Leaderboard + best model
    with st.spinner("🏅 Se alege cel mai bun model..."):
        best_model, leaderboard = choose_best_forecasting_model(eval_df)
    st.success(f"🎉 Cel mai bun model global: **{best_model}**")

    st.subheader("Clasament General al Modelelor")
    st.dataframe(leaderboard)
    csv_leaderboard = leaderboard.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="📥 Descarcă Clasamentul (CSV)",
        data=csv_leaderboard,
        file_name="leaderboard.csv",
        mime="text/csv",
    )
    st.markdown("---")

    # Vizualizări
    with st.spinner("🎨 Se generează vizualizările..."):
        visualize_forecasting_results(Y_df, forecast_df, cv_df, eval_df, horizon)
    st.success("✅ Vizualizări generate.")
    st.markdown("---")

    # ─── Salvăm în session_state (CHEIA pentru a nu reseta) ───
    st.session_state["pipeline_ran"] = True
    st.session_state["forecast_df"] = forecast_df
    st.session_state["leaderboard"] = leaderboard
    st.session_state["best_model"] = best_model
    st.balloons()

# ═════════════════════════════════════════════════════════════════════
# 2) Interfața de download (NU calculează nimic, doar folosește session_state)
# ═════════════════════════════════════════════════════════════════════
if st.session_state.get("pipeline_ran", False):

    forecast_df = st.session_state["forecast_df"]
    best_model = st.session_state["best_model"]

    st.header("📁 Exportă Previziuni")
    st.write(
        "Alege modelul ale cărui previziuni vrei să le descarci. "
        "Datele sunt deja calculate – poți schimba oricând selecția, "
        "fără să se refacă pipeline-ul."
    )

    # Pregătim lista de opțiuni (coloane de model)
    model_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds")]
    dropdown_options = ["🏆 Cel mai bun model"] + model_cols

    # Dropdown + Download în aceeași coloană pentru UX compact
    col1, col2 = st.columns([3, 1])
    with col1:
        model_choice = st.selectbox(
            "Selectează previziunea:",
            dropdown_options,
            key="model_choice",
        )

    # Determinăm efectiv coloana care va fi exportată
    if model_choice == "🏆 Cel mai bun model":
        export_col = best_model
    else:
        export_col = model_choice

    # Pregătim dataframe-ul final
    export_df = (
        forecast_df[["unique_id", "ds", export_col]]
        .rename(columns={export_col: "yhat"})
        .copy()
    )

    with col2:
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"forecast_{export_col}.csv",
            mime="text/csv",
            key="download_btn",
            help="Se descarcă instant fără a re-rula pipeline-ul.",
        )

    st.dataframe(export_df.head())

else:
    st.info(
        "ℹ️ Configurează parametrii în bara laterală și apasă "
        "'Rulează Pipeline-ul de Prognoză'."
    )
