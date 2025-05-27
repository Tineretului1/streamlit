import streamlit as st
from state_tools import init_state

st.set_page_config(page_title="🚀 Pipeline de Prognoză", layout="wide")
init_state()

st.title("📊 Dashboard Pipeline")
st.markdown(
    """
    Folosește meniul din stânga pentru a încărca date, a antrena modele și 
    a descărca previziunile. Progresul este salvat în *session_state*, 
    deci poți să sari între pagini fără să pierzi nimic.
    """
)

if st.session_state.pipeline_ran:
    st.success("Pipeline-ul a fost deja rulat ✅")
    st.write("📈 **Previziuni disponibile:**", st.session_state.forecast_df.shape)
    st.write("🏅 **Cel mai bun model:**", st.session_state.best_model)
else:
    st.info("Începe cu pagina **Upload & Config** ➡️")
