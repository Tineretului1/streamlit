# utils.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────── HELPERS PENTRU FIGURI ───────────────────────── #
def display_current_fig(fig_title: str):
    """Display current matplotlib figure in Streamlit."""
    st.pyplot(plt.gcf())
    plt.close()

# ───────────────────────── METRICI DE EROARE ───────────────────────── #
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask): # Evitarea diviziunii cu zero dacă toate y_true[mask] sunt zero
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if not np.any(mask): # Evitarea diviziunii cu zero
        return 0.0
    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100