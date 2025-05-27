import streamlit as st
import pandas as pd
import config
from data_processing import load_and_prepare
from state_tools import init_state

init_state()
st.header("📂 Upload & Config")

# Sidebar controls
uploaded_file = st.sidebar.file_uploader("Încarcă fișierul train.csv", type="csv")
uploaded_external_file = st.sidebar.file_uploader("Încarcă fișierul de date externe (ex: preț gaz)", type="csv")
horizon = st.sidebar.number_input("Orizont de Prognoză (zile)",
                                  min_value=1, value=config.HORIZON_DEFAULT)
season_length = st.sidebar.number_input("Lungimea Sezonului (zile)",
                                        min_value=1, value=config.SEASON_LENGTH_DEFAULT)
window_size_ml = st.sidebar.number_input("Fereastră ML (lags)",
                                         min_value=1, value=config.WINDOW_SIZE_DEFAULT)
max_rows = st.sidebar.number_input("Număr Maxim de Rânduri",
                                   min_value=1000, value=config.MAX_ROWS_DEFAULT, step=10000)
stores_str = st.sidebar.text_input("Magazine (virgulă)", "")

run_btn = st.button("🚀 Rulează Pipeline-ul de Prognoză", type="primary")

if run_btn:
    if uploaded_file is None:
        st.warning("⚠️ Încarcă un fișier CSV mai întâi.")
        st.stop()

    with st.spinner("⏳ Se încarcă și pregătesc datele…"):
        stores_list = [s.strip() for s in stores_str.split(",") if s.strip()] or None
        Y_df = load_and_prepare(uploaded_file, uploaded_external_file=uploaded_external_file, stores=stores_list, max_rows=max_rows)

    if Y_df.empty:
        st.error("Datele nu au putut fi procesate.")
        st.stop()

    # Save everything we will need later
    st.session_state.update(
        Y_df=Y_df,
        uploaded_external_file=uploaded_external_file,
        horizon=horizon,
        season_length=season_length,
        window_size_ml=window_size_ml,
        pipeline_ran=False,      # still false until models are trained
    )

    st.success("✅ Date pregătite! Continuă la paginile următoare.")
    st.dataframe(Y_df.head())
