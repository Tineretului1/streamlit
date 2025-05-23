# -*- coding: utf-8 -*-
"""
Enhanced forecasting pipeline - Streamlit App
=============================================
This Streamlit app allows you to run the forecasting pipeline interactively.
"""

import streamlit as st
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.backends.backend_pdf import PdfPages # Not used for direct Streamlit display
from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive, Naive, HistoricAverage,
    CrostonOptimized, ADIDA, IMAPA, AutoETS
)
from mlforecast import MLForecast
from mlforecast.utils import PredictionIntervals
from window_ops.expanding import expanding_mean
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from darts import TimeSeries
from darts.utils.statistics import plot_acf
import io

# ───────────────────────── HELPERS PENTRU FIGURI (Adaptat pentru Streamlit) ───────────────────────── #
# PLOTS_DIR can be used if we want to offer downloads of plot images later
# PLOTS_DIR = 'plots_streamlit'
# os.makedirs(PLOTS_DIR, exist_ok=True)

def display_current_fig(fig_title: str):
    """Display current matplotlib figure in Streamlit."""
    st.pyplot(plt.gcf())
    plt.close()

# ───────────────────────── METRICI DE EROARE ───────────────────────── #
# (Funcțiile mse, mae, mape, smape rămân neschimbate din scriptul original)
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

# ───────────────────────── PRELUCRARE DATE ───────────────────────── #

def _find_column(cols: pd.Index, pattern: str) -> str:
    """Return the first column whose name matches *pattern* (case‑insensitive REGEX)."""
    candidates = [c for c in cols if re.search(pattern, c, flags=re.IGNORECASE)]
    if not candidates:
        raise ValueError(f"Nicio coloană care să corespundă modelului regex '{pattern}' nu a fost găsită în setul de date.")
    return candidates[0]

@st.cache_data
def load_and_prepare(uploaded_file, stores: list[str] | None = None,
                     max_rows: int = 10_000_000) -> pd.DataFrame:
    """
    Încarcă și pregătește datele de vânzări.
    - Citește fișierul o singură dată.
    - Gestionează filtrarea pe magazine, inclusiv conversia tipului de date.
    - Agreghează vânzările și creează un index complet de date.
    """
    if uploaded_file is None:
        st.error("Vă rugăm să încărcați un fișier CSV.")
        return pd.DataFrame()

    uploaded_file.seek(0) # Esențial pentru a citi de la început la (re)execuții
    df_initial = pd.read_csv(uploaded_file, nrows=max_rows)

    if df_initial.empty:
        st.error("Fișierul CSV încărcat este gol sau nu a putut fi citit corect.")
        return pd.DataFrame()

    # 1. Găsește numele coloanelor necesare (case-insensitive)
    try:
        store_col_name = _find_column(df_initial.columns, r"store")
        item_col_name  = _find_column(df_initial.columns, r"item")
        sales_col_name = _find_column(df_initial.columns, r"sale")
        date_col_name  = _find_column(df_initial.columns, r"date")
    except ValueError as e:
        st.error(f"Eroare la identificarea coloanelor necesare: {e}")
        st.info("Asigurați-vă că fișierul CSV conține coloane pentru 'store', 'item', 'sales' și 'date' (sau variații ale acestora).")
        return pd.DataFrame()

    # 2. Selectează și redenumește coloanele la nume standard
    # Folosește .copy() pentru a evita SettingWithCopyWarning la modificările ulterioare
    df_processed = df_initial[[store_col_name, item_col_name, sales_col_name, date_col_name]].copy()
    df_processed.rename(columns={
        store_col_name: 'store',
        item_col_name:  'item',
        sales_col_name: 'sales',
        date_col_name:  'date'
    }, inplace=True)

    # 3. Converteste coloanele 'store' și 'item' la tipul string devreme
    # Acest pas este crucial pentru o filtrare consistentă dacă parametrul `stores` este o listă de string-uri.
    df_processed['store'] = df_processed['store'].astype(str)
    df_processed['item']  = df_processed['item'].astype(str)

    # 4. Filtrează după magazine, dacă este specificat
    df_to_use = df_processed # Implicit, se folosesc toate datele procesate

    if stores: # 'stores' ar trebui să fie o listă de string-uri, ex: ['1', '2']
        unique_stores_in_data_before_filter = df_processed['store'].unique()
        # st.write(f"Magazine unice în date (înainte de filtrare specifică): {unique_stores_in_data_before_filter[:10]}") # Pentru debug

        df_filtered_by_store = df_processed[df_processed['store'].isin(stores)]
        
        if df_filtered_by_store.empty:
            st.warning(
                f"Nicio dată găsită PENTRU MAGAZINELE SPECIFICATE: {stores}. "
                f"Este posibil ca aceste ID-uri de magazine să nu existe în fișierul încărcat "
                f"sau să nu aibă date asociate care trec de prelucrările inițiale. "
                f"Se vor folosi toate magazinele din fișier."
            )
            # df_to_use rămâne df_processed (adică toate magazinele)
        else:
            st.success(f"Date filtrate cu succes pentru magazinele specificate: {stores}. "
                       f"Număr de înregistrări după filtrare: {len(df_filtered_by_store)}")
            df_to_use = df_filtered_by_store
    else:
        st.info("Nu s-au specificat magazine dedicate; se vor procesa datele pentru toate magazinele.")

    # 5. Prelucrări ulterioare pe DataFrame-ul selectat (df_to_use)
    if df_to_use.empty:
        st.error("DataFrame-ul este gol înainte de agregarea finală. Verificați datele de intrare și filtrele aplicate.")
        return pd.DataFrame()

    # Asigură-te că 'date' este de tip datetime
    try:
        df_to_use['date'] = pd.to_datetime(df_to_use['date'])
    except Exception as e:
        st.error(f"Eroare la conversia coloanei 'date' în format datetime: {e}")
        return pd.DataFrame()
        
    # Creează ID-ul unic 'store_item'
    df_to_use['store_item'] = df_to_use['store'] + '_' + df_to_use['item']
    
    # Agreghează vânzările: grupează după 'date' și 'store_item' și însumează 'sales'
    # Aceasta gestionează corect duplicatele (ex: multiple vânzări pentru același item/magazin/zi)
    grouped = (
        df_to_use.groupby(['date', 'store_item'])['sales']
          .sum()
          .reset_index()
    )

    if grouped.empty:
        st.error("DataFrame-ul este gol după gruparea și însumarea vânzărilor. "
                 "Verificați dacă există date valide de vânzări pentru combinațiile de date și store_item.")
        return pd.DataFrame()

    # 6. Creează un index complet de date pentru toate unique_id-urile (store_item)
    min_date = grouped['date'].min()
    max_date = grouped['date'].max()
    
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Nu s-au putut determina limitele de date (min/max) după grupare. Verificați conținutul coloanei 'date'.")
        return pd.DataFrame()
        
    all_dates_range = pd.date_range(start=min_date, end=max_date, freq='D') # 'D' pentru frecvență zilnică
    all_unique_ids  = grouped['store_item'].unique()

    if not all_unique_ids.any(): # Verifică dacă array-ul de ID-uri unice nu este gol
        st.error("Niciun 'store_item' unic găsit după procesare. Verificați datele.")
        return pd.DataFrame()

    # Creează un MultiIndex cu toate combinațiile de date și ID-uri unice
    multi_idx = pd.MultiIndex.from_product([all_dates_range, all_unique_ids], names=['ds', 'unique_id'])
    
    # Redenumește coloanele pentru compatibilitate cu bibliotecile de prognoză
    # Setează noul index, reindexează pentru a include toate combinațiile și umple golurile
    full_df = (
        grouped.rename(columns={'date': 'ds', 'store_item': 'unique_id', 'sales': 'y'})
               .set_index(['ds', 'unique_id'])
               .reindex(multi_idx)
               .fillna({'y': 0}) # Umple valorile 'y' (vânzări) lipsă cu 0
               .reset_index()
    )
    
    # Asigură-te că 'y' este de tip float
    full_df['y'] = full_df['y'].astype(float)

    st.success(f"Pregătirea datelor finalizată. DataFrame final conține {len(full_df)} rânduri.")
    return full_df
# ───────────────────────── ANALIZĂ EXPLORATORIE ───────────────────────── #

def exploratory_analysis(Y: pd.DataFrame):
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

# ───────────────────────── STATSFORECAST ───────────────────────── #

@st.cache_resource # Cache model objects
def run_statsforecast(Y: pd.DataFrame, horizon: int, season_length: int):
    models = [
        SeasonalNaive(season_length=season_length),
        Naive(),
        HistoricAverage(),
        CrostonOptimized(),
        ADIDA(),
        IMAPA(),
        AutoETS(season_length=season_length)
    ]
    sf = StatsForecast(models=models, freq='D', n_jobs=1) # n_jobs=1 for stability in web apps
    fcst = sf.forecast(df=Y, h=horizon)
    return sf, fcst

# ───────────────────────── MLFORECAST ───────────────────────── #

@st.cache_resource # Cache model objects
def run_mlforecast(Y: pd.DataFrame, horizon: int, _h_param_deprecated: int): # _h_param_deprecated not used by mlf.predict
    mlf = MLForecast(
        models=[
            LGBMRegressor(max_depth=10, random_state=42),
            XGBRegressor(max_depth=10, eval_metric='rmse', random_state=42),
            LinearRegression()
        ],
        freq='D',
        lags=list(range(1, 7)),
        lag_transforms={1: [expanding_mean]},
        date_features=['year', 'month', 'day', 'dayofweek',
                       'quarter', 'week', 'dayofyear'],
    )
    mlf.fit(Y, prediction_intervals=PredictionIntervals())
    fcst = mlf.predict(h=horizon, level=[90])
    return mlf, fcst

# ───────────────────────── COMBINARE PREVIZIUNI ───────────────────────── #

@st.cache_data
def combine_forecasts(sf_fcst: pd.DataFrame, mlf_fcst: pd.DataFrame) -> pd.DataFrame:
    return sf_fcst.merge(mlf_fcst, on=['unique_id', 'ds'], how='left')

# ───────────────────────── CROSS‑VALIDARE ───────────────────────── #

@st.cache_data
def cross_validate(_sf: StatsForecast, _mlf: MLForecast, # Use _ to indicate cached objects
                   Y: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Re-initialize models for cross-validation if they are not serializable enough for deep copy
    # For simplicity, we assume the passed _sf and _mlf are fine,
    # but in complex scenarios, re-initialization might be safer.
    cv_sf  = _sf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon)
    cv_mlf = _mlf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon,
                                  level=[90])
    return cv_sf.merge(cv_mlf.drop(columns=['y']),
                       on=['unique_id', 'ds', 'cutoff'], how='left')

# ───────────────────────── EVALUARE ───────────────────────── #

@st.cache_data
def evaluate_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    model_cols = [c for c in cv_df.columns if c not in
                  ['unique_id', 'y', 'ds', 'cutoff', 'lo-90', 'hi-90']] # Adjusted for MLForecast output
                  # Original script had 'lo', 'hi'. MLForecast with level=[90] outputs 'lo-90', 'hi-90'
    records = []
    for (uid, cutoff), group in cv_df.groupby(['unique_id', 'cutoff']):
        for m in model_cols:
            y_true = group['y']
            y_pred = group[m]
            records.extend([
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'mse',  'model': m, 'error': mse(y_true, y_pred)},
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'mae',  'model': m, 'error': mae(y_true, y_pred)},
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'mape', 'model': m, 'error': mape(y_true, y_pred)},
                {'unique_id': uid, 'cutoff': cutoff,
                 'metric': 'smape','model': m, 'error': smape(y_true, y_pred)}
            ])
    return pd.DataFrame(records)

# ─────────────────── SELECTARE MODEL BEST‑OF‑FOUR ─────────────────── #

@st.cache_data
def choose_best_model(eval_df: pd.DataFrame):
    """Return the overall‑best model and a leaderboard DataFrame."""
    leaderboard = (
        eval_df
        .groupby(['model', 'metric'])['error']
        .mean()
        .unstack()
    )
    leaderboard['composite'] = leaderboard[['mse', 'mae', 'mape', 'smape']].mean(axis=1)
    best_model = leaderboard['composite'].idxmin()
    return best_model, leaderboard.sort_values('composite')

# ─────────────────── VIZUALIZARE REZULTATE (Adaptat pentru Streamlit) ─────────────────── #

def visualize_results(Y: pd.DataFrame, fcst: pd.DataFrame,
                      _cv: pd.DataFrame, eval_df: pd.DataFrame, horizon: int):
    st.subheader("Vizualizarea Rezultatelor")

    # Plot Forecast vs Real (adaptat)
    # Afișează pentru un număr limitat de unique_id-uri pentru performanță
    num_ids_to_plot = min(3, Y['unique_id'].nunique())
    ids_to_plot = Y['unique_id'].unique()[:num_ids_to_plot]

    st.write(f"Afișarea previziunilor pentru primele {num_ids_to_plot} unique_id-uri:")
    
    # Sf object needs to be created with models, even if empty list, to call plot method
    sf_plot_helper = StatsForecast(models=[], freq='D')

    for uid in ids_to_plot:
        Y_uid = Y[Y['unique_id'] == uid]
        fcst_uid = fcst[fcst['unique_id'] == uid]
        
        plt.figure(figsize=(12,6))
        # sf_plot_helper.plot needs a DataFrame with 'ds' and 'y' for historical data,
        # and the forecast DataFrame.
        # We plot Y_uid against all model forecasts in fcst_uid
        
        combined_plot_df = pd.concat([
            Y_uid[['ds','y']].set_index('ds'),
            fcst_uid.set_index('ds').drop('unique_id', axis=1)
        ], axis=1).reset_index()

        # Plot historical data
        plt.plot(Y_uid['ds'], Y_uid['y'], label='Actual Sales', color='black')

        # Plot forecasts for all models
        model_columns = [col for col in fcst_uid.columns if col not in ['unique_id', 'ds', 'lo-90', 'hi-90']]
        for model_col in model_columns:
            if model_col in fcst_uid.columns: # Ensure column exists
                 plt.plot(fcst_uid['ds'], fcst_uid[model_col], label=model_col, linestyle='--')
        
        plt.title(f'Previziuni vs Real pentru {uid} (Ultimele {3 * horizon} zile din istoric)')
        plt.legend()
        plt.xlabel('Data')
        plt.ylabel('Vânzări')
        
        # Limit in-sample length for clarity
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


# ───────────────────────── STREAMLIT UI ───────────────────────── #
st.set_page_config(layout="wide")
st.title("🚀 Pipeline de Prognoză Îmbunătățit")

# --- Sidebar pentru configurații ---
st.sidebar.header("⚙️ Configurații")
uploaded_file = st.sidebar.file_uploader("Încarcă fișierul train.csv", type="csv")

# Valori default din scriptul original
HORIZON_DEFAULT = 30
SEASON_LENGTH_DEFAULT = 7
WINDOW_SIZE_DEFAULT = 6 * 30 # ~6 luni

horizon = st.sidebar.number_input("Orizont de Prognoză (zile)", min_value=1, value=HORIZON_DEFAULT, step=1)
season_length = st.sidebar.number_input("Lungimea Sezonului (zile, ex: 7 pentru săptămânal)", min_value=1, value=SEASON_LENGTH_DEFAULT, step=1)
window_size_ml = st.sidebar.number_input("Fereastră ML pentru Lags (zile)", min_value=1, value=WINDOW_SIZE_DEFAULT, step=1)
max_rows_to_load = st.sidebar.number_input("Număr Maxim de Rânduri de Încărcat", min_value=1000, value=10_000_000, step=10000, help="Limitează numărul de rânduri citite din CSV pentru performanță.")

stores_input_str = st.sidebar.text_input("Magazine (separate prin virgulă, ex: 1,2). Lăsați gol pentru toate.", "", help="Specificați ID-urile magazinelor. Dacă este gol, se vor folosi toate magazinele din date.")

run_pipeline = st.sidebar.button("🚀 Rulează Pipeline-ul de Prognoză")

# --- Panoul Principal ---
if run_pipeline:
    if uploaded_file is not None:
        with st.spinner("⏳ Se încarcă și se pregătesc datele..."):
            stores_list = [s.strip() for s in stores_input_str.split(',') if s.strip()] if stores_input_str else None
            Y_df = load_and_prepare(uploaded_file, stores=stores_list, max_rows=max_rows_to_load)

        if not Y_df.empty:
            st.success(f"✅ Date încărcate și pregătite: {Y_df.shape[0]} rânduri, {Y_df['unique_id'].nunique()} serii unice.")
            st.dataframe(Y_df.head())

            with st.spinner("📊 Se efectuează analiza exploratorie..."):
                exploratory_analysis(Y_df)
            st.success("✅ Analiza exploratorie finalizată.")
            st.markdown("---")

            with st.spinner("🧠 Se antrenează modelele StatsForecast și se generează previziuni..."):
                sf_model, sf_forecast = run_statsforecast(Y_df, horizon, season_length)
            st.success("✅ Modelele StatsForecast antrenate.")

            with st.spinner("🧠 Se antrenează modelele MLForecast și se generează previziuni..."):
                # Parametrul 'h' (window_size_ml) din run_mlforecast original nu era folosit pentru predict,
                # mlf.predict folosește parametrul 'h' pentru orizont.
                # Păstrăm window_size_ml pentru coerență cu logica originală a scriptului dacă ar fi
                # folosit pentru altceva, dar pentru .predict, orizontul este 'horizon'.
                mlf_model, mlf_forecast = run_mlforecast(Y_df, horizon, window_size_ml)
            st.success("✅ Modelele MLForecast antrenate.")
            st.markdown("---")

            with st.spinner("🔗 Se combină previziunile..."):
                forecast_df = combine_forecasts(sf_forecast, mlf_forecast)
            st.success("✅ Previziuni combinate.")
            st.dataframe(forecast_df.head())
            st.markdown("---")

            with st.spinner("🔄 Se efectuează validarea încrucișată..."):
                # Transmitem obiectele model deja antrenate
                cv_df = cross_validate(sf_model, mlf_model, Y_df, horizon)
            st.success("✅ Validare încrucișată finalizată.")
            st.dataframe(cv_df.head())
            st.markdown("---")

            with st.spinner("📉 Se evaluează modelele..."):
                eval_df = evaluate_cv(cv_df)
            st.success("✅ Evaluare finalizată.")
            # st.dataframe(eval_df.head()) # Poate fi prea mare, afișăm leaderboard mai jos
            st.markdown("---")

            st.header("🏆 Clasament și Cel Mai Bun Model")
            with st.spinner("🏅 Se alege cel mai bun model..."):
                best_model, leaderboard = choose_best_model(eval_df)
            st.success(f"🎉 Cel mai bun model per ansamblu (bazat pe media celor 4 metrici): **{best_model}**")

            st.subheader("Clasament General al Modelelor")
            st.dataframe(leaderboard)
            csv_leaderboard = leaderboard.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="📥 Descarcă Clasamentul (CSV)",
                data=csv_leaderboard,
                file_name='leaderboard.csv',
                mime='text/csv',
            )
            st.markdown("---")

            st.header("📊 Vizualizări Diagnostice")
            with st.spinner("🎨 Se generează vizualizările..."):
                visualize_results(Y_df, forecast_df, cv_df, eval_df, horizon)
            st.success("✅ Vizualizări generate.")
            st.markdown("---")

            st.header("📁 Exportă Previziunile Celui Mai Bun Model")
            best_model_forecast_df = (
                forecast_df[['unique_id', 'ds', best_model]]
                .rename(columns={best_model: 'yhat'})
            )
            st.dataframe(best_model_forecast_df.head())

            csv_export = best_model_forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Descarcă Previziunile pentru {best_model} (CSV)",
                data=csv_export,
                file_name='best_model_forecast.csv',
                mime='text/csv',
            )
            st.balloons()
            st.info(f"Toate graficele sunt afișate mai sus. Rezultatele principale (clasament, previziuni) sunt disponibile pentru descărcare.")

        elif uploaded_file is not None: # Y_df este gol, dar fișierul a fost încărcat
             st.error("❌ Datele nu au putut fi procesate. Verificați fișierul și configurațiile.")

    elif run_pipeline and uploaded_file is None:
        st.warning("⚠️ Vă rugăm să încărcați un fișier `train.csv` pentru a începe.")

else:
    st.info("ℹ️ Configurați parametrii în bara laterală și apăsați 'Rulează Pipeline-ul de Prognoză'.")