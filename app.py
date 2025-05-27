import pickle
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
from state_tools import init_state   # proprie

st.set_page_config(page_title="🚀 Pipeline de Prognoză", layout="wide")

# --- încărcare parole ---
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)     # listă sau dict, vezi mai jos

names      = ["Sandru Rares", "Trial Account"]
usernames  = ["rrares", "trial"]

# --- construim structura cerută de v0.3+ ---
credentials = {"usernames": {}}
for idx, un in enumerate(usernames):
    credentials["usernames"][un] = {
        "name": names[idx],
        "password": hashed_passwords[idx]      # ↯ dacă ai salvat ca listă
        # "password": hashed_passwords[un]     # ↯ dacă ai salvat ca dict
    }

# --- iniţializăm autentificatorul ---
authenticator = stauth.Authenticate(
    credentials,
    "some_cookie_name",
    "some_signature_key",
    cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login("Login", 'main')

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    init_state()

    st.title(f"📊 Dashboard Pipeline - Welcome *{name}*")
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
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')
