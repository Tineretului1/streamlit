# state_tools.py
import streamlit as st

DEFAULTS = dict(
    pipeline_ran=False,
    Y_df=None,
    forecast_df=None,
    leaderboard=None,
    best_model=None,
    cv_df=None,
    eval_df=None,
)

def init_state():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
