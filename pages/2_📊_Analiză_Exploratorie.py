import streamlit as st
from exploratory_analysis import perform_exploratory_analysis
from state_tools import init_state

init_state()
st.header("📊 Analiză Exploratorie")

Y_df = st.session_state.get("Y_df")
if Y_df is None:
    st.info("Întâi încarcă datele în pagina **Upload & Config**.")
    st.stop()

perform_exploratory_analysis(Y_df)
