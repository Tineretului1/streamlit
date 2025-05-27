# visualization.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import display_current_fig # Assuming utils.py is in the same directory
from statsforecast import StatsForecast # For the plot helper

# ─────────────────── VIZUALIZARE REZULTATE ─────────────────── #

def visualize_forecasting_results(Y: pd.DataFrame, fcst: pd.DataFrame,
                                  _cv: pd.DataFrame, eval_df: pd.DataFrame, horizon: int):
    st.subheader("Vizualizarea Rezultatelor")

    num_ids_to_plot = min(3, Y['unique_id'].nunique())
    ids_to_plot = Y['unique_id'].unique()[:num_ids_to_plot]

    st.write(f"Afișarea previziunilor pentru primele {num_ids_to_plot} unique_id-uri:")
    
    for uid in ids_to_plot:
        Y_uid = Y[Y['unique_id'] == uid]
        fcst_uid = fcst[fcst['unique_id'] == uid]
        
        plt.figure(figsize=(12,6))
        
        plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black')

        model_columns = [col for col in fcst_uid.columns if col not in ['unique_id', 'ds', 'lo-90', 'hi-90']]
        for model_col in model_columns:
            if model_col in fcst_uid.columns:
                 plt.plot(fcst_uid['ds'], fcst_uid[model_col], label=model_col, linestyle='--')
        
        plt.title(f'Previziuni vs Real pentru {uid} (Ultimele {3 * horizon} zile din istoric)')
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Vânzări')
        
        historical_to_show = Y_uid.tail(3 * horizon)
        if not historical_to_show.empty and not fcst_uid.empty:
            min_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
            max_date = fcst_uid['ds'].max()
            plt.xlim([min_date, max_date])

        display_current_fig(f'forecast_vs_real_{uid}')


    st.write("Distribuția Erorilor pe Metrici (MSE, sMAPE):")
    for metric in ['mse', 'smape']:
        subset = eval_df[eval_df['metric'] == metric]
        plt.figure()
        plt.title(f'Distribuția {metric.upper()}')
        sns.violinplot(data=subset, x='error', y='model', orient='h')
        plt.xlabel(f'{metric.upper()} Error')
        plt.ylabel('Model')
        plt.tight_layout()
        display_current_fig(f'{metric}_violin_plot')

    st.write("Modele Câștigătoare per Metrică:")
    winners = (
        eval_df.loc[eval_df.groupby(['unique_id', 'metric'])['error'].idxmin()]
        .groupby(['metric', 'model'])
        .size()
        .reset_index(name='n_wins')
    )
    plt.figure()
    plt.title('Număr de "Victorii" per Model și Metrică')
    try:
        sns.barplot(data=winners, x='n_wins', y='model', hue='metric')
        plt.xlabel('Număr de "Victorii" (cea mai mică eroare)')
        plt.ylabel('Model')
        plt.tight_layout()
        display_current_fig('winners_per_metric_plot')
    except ValueError as e:
        st.warning(f"Nu s-a putut genera graficul modelelor câștigătoare: {e}")