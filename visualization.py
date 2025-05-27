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

    st.write(f"Afișarea previziunilor pentru primele {num_ids_to_plot} unique_id-uri (doar modelul câștigător):")

    # Determine a primary metric to choose the winner, e.g., 'smape' or 'mse'
    # If you want to let the user choose or have a default, you can add that logic
    primary_metric = 'smape' # Or 'mse', or make it configurable

    for uid in ids_to_plot:
        Y_uid = Y[Y['unique_id'] == uid]
        fcst_uid = fcst[fcst['unique_id'] == uid]

        # Find the winning model for the current uid based on the primary metric
        eval_df_uid = eval_df[(eval_df['unique_id'] == uid) & (eval_df['metric'] == primary_metric)]
        if not eval_df_uid.empty:
            winning_model_name = eval_df_uid.loc[eval_df_uid['error'].idxmin()]['model']
        else:
            # Fallback if no evaluation data for this uid/metric (e.g., plot all or first available)
            # For simplicity, let's try to plot the first model if no winner is found
            model_columns_available = [col for col in fcst_uid.columns if col not in ['unique_id', 'ds', 'lo-90', 'hi-90']]
            winning_model_name = model_columns_available[0] if model_columns_available else None

        plt.figure(figsize=(12,6))
        plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black')

        if winning_model_name and winning_model_name in fcst_uid.columns:
            plt.plot(fcst_uid['ds'], fcst_uid[winning_model_name], label=f'{winning_model_name} (Winner - {primary_metric.upper()})', linestyle='--')
            # Optionally, plot confidence intervals if they are tied to the specific model
            # For example, if columns are named like 'AutoARIMA-lo-90', 'AutoARIMA-hi-90'
            # You'd need to adjust how 'lo-90' and 'hi-90' are handled if they are model-specific in fcst_uid
            if f'{winning_model_name}-lo-90' in fcst_uid.columns and f'{winning_model_name}-hi-90' in fcst_uid.columns:
                 plt.fill_between(fcst_uid['ds'],
                                 fcst_uid[f'{winning_model_name}-lo-90'],
                                 fcst_uid[f'{winning_model_name}-hi-90'],
                                 alpha=0.2, label=f'{winning_model_name} 90% CI')
            elif 'lo-90' in fcst_uid.columns and 'hi-90' in fcst_uid.columns and len(fcst_uid.columns) <= 5: # crude check if CI is generic
                 plt.fill_between(fcst_uid['ds'], fcst_uid['lo-90'], fcst_uid['hi-90'], alpha=0.2, label='90% CI (if applicable)')


        plt.title(f'Previziuni (Model Câștigător) vs Real pentru {uid} (Ultimele {3 * horizon} zile din istoric)')
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Vânzări')

        historical_to_show = Y_uid.tail(3 * horizon)
        if not historical_to_show.empty and not fcst_uid.empty:
            min_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
            max_date = fcst_uid['ds'].max()
            plt.xlim([min_date, max_date])

        display_current_fig(f'forecast_vs_real_winner_{uid}')


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