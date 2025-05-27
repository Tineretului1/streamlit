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
            model_columns_available = [
                col for col in fcst_uid.columns
                if col not in ['unique_id', 'ds'] and not col.endswith('-lo-90') and not col.endswith('-hi-90')
            ]
            winning_model_name = model_columns_available[0] if model_columns_available else None

        # Determine the actual column name to plot from fcst_uid based on winning_model_name
        plot_col_name = None
        ci_lo_col = None
        ci_hi_col = None

        if winning_model_name:
            # Check for ML model suffixes first
            potential_with_exog = f"{winning_model_name}_with_exog"
            potential_no_exog = f"{winning_model_name}_no_exog"
            
            if potential_with_exog in fcst_uid.columns:
                plot_col_name = potential_with_exog
            elif potential_no_exog in fcst_uid.columns:
                plot_col_name = potential_no_exog
            elif winning_model_name in fcst_uid.columns: # For StatsForecast models or direct match
                plot_col_name = winning_model_name
            
            if plot_col_name:
                # Check for corresponding CI columns
                if f'{plot_col_name}-lo-90' in fcst_uid.columns and f'{plot_col_name}-hi-90' in fcst_uid.columns:
                    ci_lo_col = f'{plot_col_name}-lo-90'
                    ci_hi_col = f'{plot_col_name}-hi-90'
                # Fallback for older generic CI names if specific ones aren't found (less likely with new naming)
                elif 'lo-90' in fcst_uid.columns and 'hi-90' in fcst_uid.columns and winning_model_name not in ['LGBMRegressor', 'XGBRegressor', 'LinearRegression']: # only for non-ML
                     # This generic CI might not be accurate for the specific winning model if it's an ML one
                     # but can be a fallback for StatsForecast models if their CIs are generically named.
                     # However, StatsForecast models usually have model-specific CI names too.
                     pass # Prefer model-specific CIs handled above.

        plt.figure(figsize=(12,6))
        plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black')

        if plot_col_name:
            plt.plot(fcst_uid['ds'], fcst_uid[plot_col_name], label=f'{winning_model_name} (Winner - {primary_metric.upper()})', linestyle='--')
            if ci_lo_col and ci_hi_col:
                 plt.fill_between(fcst_uid['ds'],
                                 fcst_uid[ci_lo_col],
                                 fcst_uid[ci_hi_col],
                                 alpha=0.2, label=f'{winning_model_name} 90% CI')
        elif winning_model_name:
            st.caption(f"Nu s-a putut găsi coloana de previziune pentru modelul câștigător '{winning_model_name}' în setul de date pentru {uid}.")


        plt.title(f'Previziuni (Model Câștigător: {winning_model_name or "N/A"}) vs Real pentru {uid} (Ultimele {3 * horizon} zile din istoric)')
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Vânzări')

        historical_to_show = Y_uid.tail(3 * horizon)
        if not historical_to_show.empty and not fcst_uid.empty:
            min_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
            max_date = fcst_uid['ds'].max()
            plt.xlim([min_date, max_date])

        display_current_fig(f'forecast_vs_real_winner_{uid}')

    # --- New section for comparing with/without external feature ---
    st.subheader("Compararea Impactului Caracteristicii Exogene pentru Modelele ML")
    ml_base_models = ['LGBMRegressor', 'XGBRegressor', 'LinearRegression']

    for uid in ids_to_plot:
        Y_uid = Y[Y['unique_id'] == uid]
        fcst_uid = fcst[fcst['unique_id'] == uid]

        for base_model_name in ml_base_models:
            col_no_exog = f"{base_model_name}_no_exog"
            col_with_exog = f"{base_model_name}_with_exog"

            if col_no_exog in fcst_uid.columns and col_with_exog in fcst_uid.columns:
                plt.figure(figsize=(12, 7))
                plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black', alpha=0.7)
                plt.plot(fcst_uid['ds'], fcst_uid[col_no_exog], label=f'{base_model_name} (Fără Exogenă)', linestyle=':')
                plt.plot(fcst_uid['ds'], fcst_uid[col_with_exog], label=f'{base_model_name} (Cu Exogenă)', linestyle='--')
                
                # Plot confidence intervals if they exist
                ci_lo_no_exog = f"{col_no_exog}-lo-90"
                ci_hi_no_exog = f"{col_no_exog}-hi-90"
                if ci_lo_no_exog in fcst_uid.columns and ci_hi_no_exog in fcst_uid.columns:
                    plt.fill_between(fcst_uid['ds'], fcst_uid[ci_lo_no_exog], fcst_uid[ci_hi_no_exog], alpha=0.15, label=f'{base_model_name} (Fără Exogenă) 90% CI')

                ci_lo_with_exog = f"{col_with_exog}-lo-90"
                ci_hi_with_exog = f"{col_with_exog}-hi-90"
                if ci_lo_with_exog in fcst_uid.columns and ci_hi_with_exog in fcst_uid.columns:
                    plt.fill_between(fcst_uid['ds'], fcst_uid[ci_lo_with_exog], fcst_uid[ci_hi_with_exog], alpha=0.15, label=f'{base_model_name} (Cu Exogenă) 90% CI')

                plt.title(f'Impact Caracteristică Exogenă: {base_model_name} pentru {uid}')
                plt.legend()
                plt.xlabel('Data')
                plt.ylabel('Vânzări')

                historical_to_show = Y_uid.tail(3 * horizon)
                if not historical_to_show.empty and not fcst_uid.empty:
                    min_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
                    max_date = fcst_uid['ds'].max()
                    plt.xlim([min_date, max_date])
                
                display_current_fig(f'exog_impact_{base_model_name}_{uid}')
            elif col_no_exog in fcst_uid.columns and col_with_exog not in fcst_uid.columns:
                # This case might occur if external feature was all NaN and _with_exog models weren't trained/renamed
                st.caption(f"Pentru {uid}, modelul {base_model_name} nu are o variantă 'cu exogenă' disponibilă în setul de date al previziunilor (posibil caracteristica exogenă să fi fost invalidă).")


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