# visualization.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import display_current_fig # Assuming utils.py is in the same directory
from statsforecast import StatsForecast # For the plot helper

# ─────────────────── VIZUALIZARE REZULTATE ─────────────────── #

def visualize_forecasting_results(Y: pd.DataFrame, fcst: pd.DataFrame,
                                  _cv: pd.DataFrame, eval_df: pd.DataFrame,
                                  leaderboard: pd.DataFrame, best_model_overall: str,
                                  horizon: int):
    st.header("📈 Analiza Performanței Modelelor")

    # --- Section 1: Overall Best Model Performance ---
    st.subheader(f"🏆 Performanța Modelului General Câștigător: **{best_model_overall}**")
    
    st.write("Clasament General (Leaderboard) pe baza mediei erorilor de cross-validare:")
    st.dataframe(leaderboard.style.highlight_min(axis=0, subset=['mse', 'mae', 'mape', 'smape', 'composite'], color='lightgreen'))

    plt.figure(figsize=(10, 6))
    leaderboard_sorted_composite = leaderboard.sort_values('composite')
    sns.barplot(x=leaderboard_sorted_composite['composite'], y=leaderboard_sorted_composite.index, palette='viridis', orient='h')
    plt.xlabel('Eroare Compusă Medie (mai mic = mai bun)')
    plt.ylabel('Model')
    plt.title('Performanța Generală a Modelelor (Eroare Compusă)')
    plt.tight_layout()
    display_current_fig('overall_performance_composite_barchart')

    # Keep num_ids_to_plot and ids_to_plot for the exogenous impact section
    num_ids_to_plot = min(3, Y['unique_id'].nunique())
    ids_to_plot = Y['unique_id'].unique()[:num_ids_to_plot]
    
    st.divider()

    # --- Simplified Section: Impact of Exogenous Feature ---
    st.subheader("💡 Impactul Caracteristicii Exogene asupra Modelelor ML")
    st.write(f"Analizând primele {num_ids_to_plot} ID-uri unice:")
    
    ml_base_models = ['LGBMRegressor', 'XGBRegressor', 'LinearRegression']
    comparison_metric = 'smape' # Metric used for deciding if exog helped

    for uid in ids_to_plot:
        st.markdown(f"--- \n ### ID Unic: `{uid}`")
        Y_uid = Y[Y['unique_id'] == uid]
        fcst_uid = fcst[fcst['unique_id'] == uid]

        for base_model_name in ml_base_models:
            model_name_no_exog = f"{base_model_name}_no_exog"
            model_name_with_exog = f"{base_model_name}_with_exog"

            # Check if both versions are available in forecast and evaluation data
            if model_name_no_exog in fcst_uid.columns and model_name_with_exog in fcst_uid.columns and \
               model_name_no_exog in eval_df['model'].values and model_name_with_exog in eval_df['model'].values:

                eval_no_exog = eval_df[
                    (eval_df['unique_id'] == uid) &
                    (eval_df['model'] == model_name_no_exog) &
                    (eval_df['metric'] == comparison_metric)
                ]['error'].mean() # Use mean error over CV windows

                eval_with_exog = eval_df[
                    (eval_df['unique_id'] == uid) &
                    (eval_df['model'] == model_name_with_exog) &
                    (eval_df['metric'] == comparison_metric)
                ]['error'].mean()

                if pd.isna(eval_no_exog) or pd.isna(eval_with_exog):
                    st.markdown(f"**{base_model_name}:** Date de evaluare insuficiente pentru comparație.")
                    continue

                message = ""
                plot_this_col = None
                plot_label_suffix = ""

                if eval_with_exog < eval_no_exog:
                    improvement = ((eval_no_exog - eval_with_exog) / eval_no_exog) * 100 if eval_no_exog != 0 else 0
                    message = f"<h3 style='color:green;'>DA ✔️</h3> Caracteristica exogenă a **ÎMBUNĂTĂȚIT** predicția pentru **{base_model_name}** (Eroare {comparison_metric.upper()}: {eval_with_exog:.4f} vs {eval_no_exog:.4f}, îmbunătățire: {improvement:.2f}%)."
                    plot_this_col = model_name_with_exog
                    plot_label_suffix = "(Cu Exogenă - Mai Bun)"
                elif eval_with_exog > eval_no_exog:
                    worsening = ((eval_with_exog - eval_no_exog) / eval_no_exog) * 100 if eval_no_exog != 0 else 0
                    message = f"<h3 style='color:red;'>NU ❌</h3> Caracteristica exogenă **NU A AJUTAT** (sau a înrăutățit) predicția pentru **{base_model_name}** (Eroare {comparison_metric.upper()}: {eval_with_exog:.4f} vs {eval_no_exog:.4f}, înrăutățire: {worsening:.2f}%)."
                    plot_this_col = model_name_no_exog
                    plot_label_suffix = "(Fără Exogenă - Mai Bun)"
                else:
                    message = f"<h3 style='color:orange;'>NEUTRU ➖</h3> Caracteristica exogenă **NU A AVUT IMPACT** semnificativ asupra predicției pentru **{base_model_name}** (Eroare {comparison_metric.upper()}: {eval_with_exog:.4f})."
                    plot_this_col = model_name_no_exog # Plot one version
                    plot_label_suffix = "(Impact Neutru)"
                
                st.markdown(message, unsafe_allow_html=True)

                # Plot only the better performing or default version
                plt.figure(figsize=(10, 5))
                plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black', alpha=0.8)
                
                ci_col_lo = f"{plot_this_col}-lo-90"
                ci_col_hi = f"{plot_this_col}-hi-90"

                plt.plot(fcst_uid['ds'], fcst_uid[plot_this_col], label=f'{base_model_name} {plot_label_suffix}', linestyle='--')
                if ci_col_lo in fcst_uid.columns and ci_col_hi in fcst_uid.columns:
                     plt.fill_between(fcst_uid['ds'], fcst_uid[ci_col_lo], fcst_uid[ci_col_hi], alpha=0.2, label='90% CI')
                
                plt.title(f'Performanță {base_model_name} pentru {uid}')
                plt.legend()
                plt.xlabel('Data')
                plt.ylabel('Vânzări')
                historical_to_show = Y_uid.tail(3 * horizon)
                if not historical_to_show.empty and not fcst_uid.empty:
                    min_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
                    max_date = fcst_uid['ds'].max()
                    plt.xlim([min_date, max_date])
                display_current_fig(f'exog_impact_simplified_{base_model_name}_{uid}')
            
            elif model_name_no_exog in fcst_uid.columns: # Only no_exog version exists
                 st.markdown(f"**{base_model_name}:** Doar varianta fără caracteristică exogenă este disponibilă.")
                 plt.figure(figsize=(10, 5))
                 plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black', alpha=0.8)
                 plt.plot(fcst_uid['ds'], fcst_uid[model_name_no_exog], label=f'{base_model_name} (Fără Exogenă)', linestyle='--')
                 ci_col_lo = f"{model_name_no_exog}-lo-90"
                 ci_col_hi = f"{model_name_no_exog}-hi-90"
                 if ci_col_lo in fcst_uid.columns and ci_col_hi in fcst_uid.columns:
                     plt.fill_between(fcst_uid['ds'], fcst_uid[ci_col_lo], fcst_uid[ci_col_hi], alpha=0.2, label='90% CI')
                 plt.title(f'Performanță {base_model_name} (Fără Exogenă) pentru {uid}')
                 plt.legend()
                 plt.xlabel('Data')
                 plt.ylabel('Vânzări')
                 historical_to_show = Y_uid.tail(3 * horizon)
                 if not historical_to_show.empty and not fcst_uid.empty:
                    min_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
                    max_date = fcst_uid['ds'].max()
                    plt.xlim([min_date, max_date])
                 display_current_fig(f'exog_impact_simplified_{base_model_name}_{uid}_no_exog_only')
            else:
                st.markdown(f"**{base_model_name}:** Date insuficiente pentru afișare.")
    st.divider()
    # Removed other plots (error distribution, winners per metric)