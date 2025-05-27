# exploratory_analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.utils.statistics import plot_acf
from utils import display_current_fig # Assuming utils.py is in the same directory

# ───────────────────────── ANALIZĂ EXPLORATORIE ───────────────────────── #

def perform_exploratory_analysis(Y: pd.DataFrame):
    """Produces and displays exploratory plots in Streamlit."""
    st.subheader("Analiză Exploratorie a Datelor")
    total = Y.groupby('ds')['y'].sum()

    plt.figure()
    total.plot(title='Total Vânzări pe Dată')
    display_current_fig('total_sales_plot')

    series = TimeSeries.from_times_and_values(total.index, total.values)

    plt.figure()
    plot_acf(series, m=7, alpha=0.05, max_lag=30)
    plt.title('ACF - Sezonalitate Săptămânală (m=7)')
    display_current_fig('acf_weekly_plot')

    plt.figure()
    plot_acf(series, m=365, alpha=0.05, max_lag=400)
    plt.title('ACF - Sezonalitate Anuală (m=365)')
    display_current_fig('acf_yearly_plot')