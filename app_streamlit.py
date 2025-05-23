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

# ───────────────────────── FUNCȚII AJUTĂTOARE PENTRU FIGURI (Adaptat pentru Streamlit) ───────────────────────── #
def display_current_fig(fig_title: str):
    """Afișează figura matplotlib curentă în Streamlit și o închide."""
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
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    if not np.any(mask):
        return 0.0
    return np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100

# ───────────────────────── PRELUCRAREA DATELOR ───────────────────────── #

def _find_column(cols: pd.Index, pattern: str) -> str:
    """Returnează prima coloană al cărei nume corespunde modelului regex (insensibil la majuscule)."""
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

    uploaded_file.seek(0)
    df_initial = pd.read_csv(uploaded_file, nrows=max_rows)

    if df_initial.empty:
        st.error("Fișierul CSV încărcat este gol sau nu a putut fi citit corect.")
        return pd.DataFrame()

    try:
        store_col_name = _find_column(df_initial.columns, r"store")
        item_col_name  = _find_column(df_initial.columns, r"item")
        sales_col_name = _find_column(df_initial.columns, r"sale")
        date_col_name  = _find_column(df_initial.columns, r"date")
    except ValueError as e:
        st.error(f"Eroare la identificarea coloanelor necesare: {e}")
        st.info("Asigurați-vă că fișierul CSV conține coloane pentru 'store', 'item', 'sales' și 'date' (sau variații ale acestora).")
        return pd.DataFrame()

    df_processed = df_initial[[store_col_name, item_col_name, sales_col_name, date_col_name]].copy()
    df_processed.rename(columns={
        store_col_name: 'store',
        item_col_name:  'item',
        sales_col_name: 'sales',
        date_col_name:  'date'
    }, inplace=True)

    df_processed['store'] = df_processed['store'].astype(str)
    df_processed['item']  = df_processed['item'].astype(str)

    df_to_use = df_processed

    if stores:
        df_filtered_by_store = df_processed[df_processed['store'].isin(stores)]
        if df_filtered_by_store.empty:
            st.warning(
                f"Nicio dată găsită PENTRU MAGAZINELE SPECIFICATE: {stores}. "
                f"Este posibil ca aceste ID-uri de magazine să nu existe în fișierul încărcat "
                f"sau să nu aibă date asociate care trec de prelucrările inițiale. "
                f"Se vor folosi toate magazinele din fișier."
            )
        else:
            st.success(f"Date filtrate cu succes pentru magazinele specificate: {stores}. "
                       f"Număr de înregistrări după filtrare: {len(df_filtered_by_store)}")
            df_to_use = df_filtered_by_store
    else:
        st.info("Nu s-au specificat magazine dedicate; se vor procesa datele pentru toate magazinele.")

    if df_to_use.empty:
        st.error("DataFrame-ul este gol înainte de agregarea finală. Verificați datele de intrare și filtrele aplicate.")
        return pd.DataFrame()

    try:
        df_to_use['date'] = pd.to_datetime(df_to_use['date'])
    except Exception as e:
        st.error(f"Eroare la conversia coloanei 'date' în format datetime: {e}")
        return pd.DataFrame()
        
    df_to_use['store_item'] = df_to_use['store'] + '_' + df_to_use['item']
    
    grouped = (
        df_to_use.groupby(['date', 'store_item'])['sales']
          .sum()
          .reset_index()
    )

    if grouped.empty:
        st.error("DataFrame-ul este gol după gruparea și însumarea vânzărilor. "
                 "Verificați dacă există date valide de vânzări pentru combinațiile de date și store_item.")
        return pd.DataFrame()

    min_date = grouped['date'].min()
    max_date = grouped['date'].max()
    
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Nu s-au putut determina limitele de date (min/max) după grupare. Verificați conținutul coloanei 'date'.")
        return pd.DataFrame()
        
    all_dates_range = pd.date_range(start=min_date, end=max_date, freq='D')
    all_unique_ids  = grouped['store_item'].unique()

    if not all_unique_ids.any():
        st.error("Niciun 'store_item' unic găsit după procesare. Verificați datele.")
        return pd.DataFrame()

    multi_idx = pd.MultiIndex.from_product([all_dates_range, all_unique_ids], names=['ds', 'unique_id'])
    
    full_df = (
        grouped.rename(columns={'date': 'ds', 'store_item': 'unique_id', 'sales': 'y'})
               .set_index(['ds', 'unique_id'])
               .reindex(multi_idx)
               .fillna({'y': 0})
               .reset_index()
    )
    
    full_df['y'] = full_df['y'].astype(float)

    st.success(f"Pregătirea datelor finalizată. DataFrame final conține {len(full_df)} rânduri.")
    return full_df

# ───────────────────────── ANALIZĂ EXPLORATORIE A DATELOR ───────────────────────── #

def exploratory_analysis(Y: pd.DataFrame):
    """Produce și afișează grafice exploratorii în Streamlit."""
    st.subheader("Analiză Exploratorie a Datelor de Vânzări")
    total = Y.groupby('ds')['y'].sum()

    plt.figure()
    total.plot(title='Vânzări Totale Agregate pe Zi')
    display_current_fig('total_sales_plot')

    series = TimeSeries.from_times_and_values(total.index, total.values)

    plt.figure()
    plot_acf(series, m=7, alpha=0.05, max_lag=30)
    plt.title('Funcția de Autocorelație (ACF) - Sezonalitate Săptămânală (lag=7)')
    display_current_fig('acf_weekly_plot')

    plt.figure()
    plot_acf(series, m=365, alpha=0.05, max_lag=400) # max_lag ajustat pentru a vizualiza lag-ul anual
    plt.title('Funcția de Autocorelație (ACF) - Sezonalitate Anuală (lag=365)')
    display_current_fig('acf_yearly_plot')

# ───────────────────────── MODELE STATSFORECAST ───────────────────────── #

@st.cache_resource
def run_statsforecast(Y: pd.DataFrame, horizon: int, season_length: int):
    """Rulează modelele StatsForecast și returnează obiectul model și previziunile."""
    models = [
        SeasonalNaive(season_length=season_length),
        Naive(),
        HistoricAverage(),
        CrostonOptimized(),
        ADIDA(),
        IMAPA(),
        AutoETS(season_length=season_length)
    ]
    sf = StatsForecast(models=models, freq='D', n_jobs=1)
    fcst = sf.forecast(df=Y, h=horizon)
    return sf, fcst

# ───────────────────────── MODELE MLFORECAST ───────────────────────── #

@st.cache_resource
def run_mlforecast(Y: pd.DataFrame, horizon: int, _h_param_deprecated: int): # _h_param_deprecated not used by mlf.predict
    """Rulează modelele MLForecast și returnează obiectul model și previziunile."""
    mlf = MLForecast(
        models=[
            LGBMRegressor(max_depth=10, random_state=42),
            XGBRegressor(max_depth=10, eval_metric='rmse', random_state=42),
            LinearRegression()
        ],
        freq='D',
        lags=list(range(1, 7)), # Lags standard, pot fi ajustate
        lag_transforms={1: [expanding_mean]},
        date_features=['year', 'month', 'day', 'dayofweek',
                       'quarter', 'week', 'dayofyear'],
    )
    mlf.fit(Y, prediction_intervals=PredictionIntervals()) # Antrenare cu intervale de predicție
    fcst = mlf.predict(h=horizon, level=[90]) # Predicție cu interval de încredere de 90%
    return mlf, fcst

# ───────────────────────── COMBINAREA PREVIZIUNILOR ───────────────────────── #

@st.cache_data
def combine_forecasts(sf_fcst: pd.DataFrame, mlf_fcst: pd.DataFrame) -> pd.DataFrame:
    """Combină DataFrame-urile de previziuni de la StatsForecast și MLForecast."""
    return sf_fcst.merge(mlf_fcst, on=['unique_id', 'ds'], how='left')

# ───────────────────────── VALIDARE ÎNCRUCIȘATĂ ───────────────────────── #

@st.cache_data
def cross_validate(_sf: StatsForecast, _mlf: MLForecast,
                   Y: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Efectuează validarea încrucișată pentru modelele StatsForecast și MLForecast."""
    # n_windows și step_size pot fi parametrizate dacă este necesar
    cv_sf  = _sf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon)
    cv_mlf = _mlf.cross_validation(df=Y, h=horizon, n_windows=3, step_size=horizon,
                                   level=[90]) # Asigură-te că level este specificat și aici
    # Elimină coloana 'y' din cv_mlf înainte de merge pentru a evita sufixele '_x', '_y' dacă 'y' există în ambele
    return cv_sf.merge(cv_mlf.drop(columns=['y'], errors='ignore'), 
                       on=['unique_id', 'ds', 'cutoff'], how='left')

# ───────────────────────── EVALUAREA PERFORMANȚEI MODELELOR ───────────────────────── #

@st.cache_data
def evaluate_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Calculează metricile de eroare (MSE, MAE, MAPE, sMAPE) din rezultatele validării încrucișate."""
    # Identifică automat coloanele model, excluzând 'lo-90', 'hi-90' și alte coloane non-model
    # Acest regex încearcă să evite coloanele cu 'lo-' sau 'hi-' la începutul numelui
    model_cols = [c for c in cv_df.columns if c not in
                  ['unique_id', 'y', 'ds', 'cutoff'] and not c.startswith(('lo-', 'hi-'))]
    
    records = []
    for (uid, cutoff), group in cv_df.groupby(['unique_id', 'cutoff']):
        for m in model_cols:
            if m not in group.columns: # Verifică dacă coloana model există în grup
                # st.warning(f"Coloana model '{m}' nu a fost găsită pentru unique_id '{uid}' și cutoff '{cutoff}'. Se omite.")
                continue
            y_true = group['y']
            y_pred = group[m]
            
            # Verifică dacă y_pred conține NaN-uri și tratează-le dacă este necesar
            if y_pred.isnull().any():
                # st.warning(f"Predicțiile pentru modelul '{m}' (uid: {uid}, cutoff: {cutoff}) conțin NaN-uri. Acestea vor afecta metricile.")
                # Opțional: umple NaN-urile sau le exclude, deși metricile sklearn ar trebui să le gestioneze dacă y_true nu are NaN-uri corespunzătoare
                pass

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

# ─────────────────── SELECTAREA CELUI MAI BUN MODEL GENERAL ─────────────────── #

@st.cache_data
def choose_best_model(eval_df: pd.DataFrame):
    """
    Selectează cel mai bun model pe baza unei medii compozite a metricilor de eroare
    și returnează numele modelului și un clasament general.
    """
    if eval_df.empty:
        st.warning("DataFrame-ul de evaluare este gol. Nu se poate alege cel mai bun model.")
        return None, pd.DataFrame()

    leaderboard = (
        eval_df
        .groupby(['model', 'metric'])['error']
        .mean()
        .unstack()
    )
    # Asigură-te că toate metricile necesare sunt prezente înainte de a calcula 'composite'
    required_metrics = ['mse', 'mae', 'mape', 'smape']
    available_metrics = [metric for metric in required_metrics if metric in leaderboard.columns]
    
    if not available_metrics:
        st.error("Nicio metrică de evaluare disponibilă în leaderboard. Verifică procesul de evaluare.")
        return None, leaderboard

    leaderboard['composite_error_avg'] = leaderboard[available_metrics].mean(axis=1)
    
    # Gestionează cazul în care 'composite_error_avg' nu are valori (ex. toate erorile sunt NaN)
    if leaderboard['composite_error_avg'].isnull().all():
        st.warning("Nu s-a putut calcula eroarea compozită medie. Este posibil ca toate erorile să fie NaN.")
        best_model_name = "N/A"
    else:
        best_model_name = leaderboard['composite_error_avg'].idxmin()
        
    return best_model_name, leaderboard.sort_values('composite_error_avg')

# ─────────────────── VIZUALIZAREA REZULTATELOR (Adaptat pentru Streamlit) ─────────────────── #

def visualize_results(Y: pd.DataFrame, fcst: pd.DataFrame, 
                      _cv_df_for_context_only: pd.DataFrame, # Renamed, as cv_df might not be directly used here now
                      eval_df: pd.DataFrame, horizon: int, 
                      best_model_name: str | None):
    """Afișează grafice relevante pentru rezultatele prognozei."""
    st.subheader("Vizualizarea Detaliată a Rezultatelor")

    # Plot Previziuni vs Real DOAR PENTRU MODELUL CÂȘTIGĂTOR
    num_ids_to_plot = min(3, Y['unique_id'].nunique()) # Limitează numărul de ID-uri pentru afișare
    ids_to_plot = Y['unique_id'].unique()[:num_ids_to_plot]

    st.write(f"Afișarea previziunilor modelului câștigător ({best_model_name if best_model_name else 'N/A'}) pentru primele {num_ids_to_plot} ID-uri unice:")
    
    if not best_model_name or best_model_name == "N/A":
        st.info("Nu s-a putut identifica un model câștigător pentru afișarea previziunilor specifice.")
        # Afișează graficele de distribuție a erorilor chiar dacă modelul câștigător nu e clar
    else:
        if best_model_name not in fcst.columns:
            st.warning(f"Coloana pentru modelul câștigător '{best_model_name}' nu există în setul de date cu previziuni. Se omite graficul de previziuni.")
        else:
            for uid in ids_to_plot:
                Y_uid = Y[Y['unique_id'] == uid]
                fcst_uid = fcst[fcst['unique_id'] == uid]
                
                plt.figure(figsize=(12,6))
                
                # Date istorice
                plt.plot(Y_uid['ds'], Y_uid['y'], label='Vânzări Reale', color='black', linewidth=1.5)

                # Previziunea modelului câștigător
                plt.plot(fcst_uid['ds'], fcst_uid[best_model_name], label=f'Previziune ({best_model_name})', linestyle='--', color='blue')

                # Interval de încredere, dacă este disponibil pentru modelul câștigător (specific MLForecast aici)
                # Verifică dacă coloanele pentru intervalul de încredere (ex: 'LGBMRegressor-lo-90', 'LGBMRegressor-hi-90')
                # corespund modelului câștigător. Aceasta necesită o potrivire mai complexă a numelor
                # sau ca `best_model_name` să fie numele de bază (ex. 'LGBMRegressor') și apoi să construim numele coloanelor.
                # Pentru simplitate, dacă `best_model_name` este un model MLForecast și are intervale, le vom căuta.
                # Aceasta este o simplificare; o soluție robustă ar stoca explicit numele coloanelor de intervale.
                low_interval_col = f'{best_model_name}-lo-90' # Presupunând formatul MLForecast
                high_interval_col = f'{best_model_name}-hi-90'
                if low_interval_col in fcst_uid.columns and high_interval_col in fcst_uid.columns:
                    plt.fill_between(fcst_uid['ds'], 
                                     fcst_uid[low_interval_col], 
                                     fcst_uid[high_interval_col], 
                                     color='skyblue', alpha=0.3, label='Interval de Încredere 90%')
                
                plt.title(f'Model Câștigător ({best_model_name}) vs. Real pentru {uid} (Istoric recent și Orizont)')
                plt.legend()
                plt.xlabel('Data')
                plt.ylabel('Vânzări')
                
                historical_to_show = Y_uid.tail(3 * horizon) # Afișează istoric relevant
                if not historical_to_show.empty and not fcst_uid.empty:
                    min_plot_date = min(historical_to_show['ds'].min(), fcst_uid['ds'].min())
                    max_plot_date = fcst_uid['ds'].max()
                    plt.xlim([min_plot_date, max_plot_date])

                display_current_fig(f'forecast_vs_real_winner_{uid}')

    # Distribuția Erorilor pe Metrici (rămâne utilă)
    st.write("Distribuția Erorilor Agregate pe Model și Metrică (MSE, sMAPE):")
    if eval_df.empty:
        st.info("Nu există date de evaluare pentru a afișa distribuția erorilor.")
    else:
        for metric in ['mse', 'smape']:
            if metric not in eval_df['metric'].unique():
                # st.info(f"Metrica {metric.upper()} nu este disponibilă în datele de evaluare.")
                continue
            subset = eval_df[eval_df['metric'] == metric]
            if subset.empty:
                continue
            plt.figure(figsize=(10, max(5, len(subset['model'].unique()) * 0.5))) # Ajustează înălțimea dinamic
            plt.title(f'Distribuția Erorii {metric.upper()} pe Model')
            sns.violinplot(data=subset, x='error', y='model', orient='h', cut=0) # cut=0 pentru a nu extinde cozile
            plt.xlabel(f'Eroare {metric.upper()}')
            plt.ylabel('Model')
            plt.tight_layout()
            display_current_fig(f'{metric}_violin_plot')

    # Modele Câștigătoare per Metrică (număr de "victorii" pe unique_id)
    st.write("Performanța Modelelor: Numărul de ID-uri Unice unde Modelul a Obținut Cea Mai Mică Eroare (per Metrică):")
    if eval_df.empty:
        st.info("Nu există date de evaluare pentru a afișa modelele câștigătoare per metrică.")
    else:
        try:
            # Găsește modelul cu cea mai mică eroare pentru fiecare unique_id și metrică
            winners_per_id_metric = eval_df.loc[eval_df.groupby(['unique_id', 'metric'])['error'].idxmin()]
            
            # Numără de câte ori fiecare model a "câștigat" per metrică
            win_counts = (
                winners_per_id_metric.groupby(['metric', 'model'])
                .size()
                .reset_index(name='num_wins_per_id')
            )
            
            if win_counts.empty:
                st.info("Nu s-au putut calcula 'victoriile' per model și metrică.")
            else:
                plt.figure(figsize=(10, max(5, len(win_counts['model'].unique()) * 0.5))) # Ajustează înălțimea
                plt.title('Număr de "Victorii" (Cea Mai Mică Eroare) per Model și Metrică')
                sns.barplot(data=win_counts, x='num_wins_per_id', y='model', hue='metric', dodge=True)
                plt.xlabel('Număr de ID-uri Unice cu Eroare Minimă')
                plt.ylabel('Model')
                plt.legend(title='Metrică')
                plt.tight_layout()
                display_current_fig('wins_per_model_metric_plot')
        except Exception as e: # Captură mai generală pentru orice problemă neașteptată
            st.warning(f"Nu s-a putut genera graficul modelelor câștigătoare per metrică: {e}")


# ───────────────────────── INTERFAȚA UTILIZATOR STREAMLIT ───────────────────────── #
st.set_page_config(layout="wide")
st.title("Aplicație Interactivă pentru Prognoza Vânzărilor")

# --- Bara Laterală pentru Configurații ---
st.sidebar.header("Configurații pentru Pipeline")
uploaded_file = st.sidebar.file_uploader("1. Încarcă fișierul de date (CSV)", type="csv")

HORIZON_DEFAULT = 30
SEASON_LENGTH_DEFAULT = 7
WINDOW_SIZE_DEFAULT = 6 * 30 

horizon = st.sidebar.number_input("2. Orizont de Prognoză (zile)", min_value=1, value=HORIZON_DEFAULT, step=1, help="Numărul de zile pentru care se va face prognoza.")
season_length = st.sidebar.number_input("3. Lungimea Sezonului (zile)", min_value=1, value=SEASON_LENGTH_DEFAULT, step=1, help="Ex: 7 pentru sezonalitate săptămânală, 365 pentru anuală.")
window_size_ml = st.sidebar.number_input("4. Fereastră Istorică pentru Modele ML (zile)", min_value=1, value=WINDOW_SIZE_DEFAULT, step=1, help="Relevant pentru calculul anumitor feature-uri bazate pe lag-uri; orizontul de predicție ML este setat de 'Orizont de Prognoză'.")
max_rows_to_load = st.sidebar.number_input("5. Număr Maxim de Rânduri de Încărcat din CSV", min_value=1000, value=10_000_000, step=10000, help="Limitează datele citite pentru a gestiona performanța cu fișiere mari.")

stores_input_str = st.sidebar.text_input("6. Filtru Magazine (opțional)", "", help="Introduceți ID-urile magazinelor separate prin virgulă (ex: 1,2). Lăsați gol pentru a procesa toate magazinele.")

run_pipeline_button = st.sidebar.button("Rulează Pipeline-ul de Prognoză", type="primary")

# --- Panoul Principal ---
if run_pipeline_button:
    if uploaded_file is not None:
        with st.spinner("Se încarcă și se pregătesc datele..."):
            stores_list = [s.strip() for s in stores_input_str.split(',') if s.strip()] if stores_input_str else None
            Y_df = load_and_prepare(uploaded_file, stores=stores_list, max_rows=max_rows_to_load)

        if not Y_df.empty:
            st.success(f"Date încărcate și pregătite: {Y_df.shape[0]} rânduri, {Y_df['unique_id'].nunique()} serii unice (combinații magazin-produs).")
            st.subheader("Sumar Date Pregătite")
            st.dataframe(Y_df.head())
            st.markdown("---")

            with st.spinner("Se efectuează analiza exploratorie a datelor..."):
                exploratory_analysis(Y_df)
            st.success("Analiza exploratorie finalizată.")
            st.markdown("---")

            sf_model, mlf_model = None, None # Inițializare pentru a avea acces ulterior

            with st.spinner("Se antrenează modelele StatsForecast și se generează previziuni..."):
                sf_model, sf_forecast = run_statsforecast(Y_df, horizon, season_length)
            st.success("Modelele StatsForecast antrenate și previziuni generate.")

            with st.spinner("Se antrenează modelele MLForecast și se generează previziuni..."):
                mlf_model, mlf_forecast = run_mlforecast(Y_df, horizon, window_size_ml)
            st.success("Modelele MLForecast antrenate și previziuni generate.")
            st.markdown("---")

            with st.spinner("Se combină seturile de previziuni..."):
                forecast_df = combine_forecasts(sf_forecast, mlf_forecast)
            st.success("Previziuni combinate într-un singur tabel.")
            st.subheader("Sumar Previziuni Combinate")
            st.dataframe(forecast_df.head())
            st.markdown("---")

            cv_df, eval_df = pd.DataFrame(), pd.DataFrame() # Inițializare

            if sf_model and mlf_model: # Continuă doar dacă modelele au fost antrenate
                with st.spinner("Se efectuează validarea încrucișată a modelelor..."):
                    cv_df = cross_validate(sf_model, mlf_model, Y_df, horizon)
                st.success("Validarea încrucișată finalizată.")
                st.subheader("Sumar Date Validare Încrucișată")
                st.dataframe(cv_df.head())
                st.markdown("---")

                with st.spinner("Se evaluează performanța modelelor pe baza validării încrucișate..."):
                    eval_df = evaluate_cv(cv_df)
                st.success("Evaluarea modelelor finalizată.")
                # st.dataframe(eval_df.head()) # Poate fi prea detaliat; clasamentul este mai util
                st.markdown("---")
            else:
                st.warning("Antrenarea modelelor a eșuat sau a fost omisă; validarea și evaluarea nu pot continua.")
            
            best_model_name = None
            leaderboard = pd.DataFrame()

            if not eval_df.empty:
                st.header("Performanța Generală a Modelelor și Alegerea Modelului Câștigător")
                with st.spinner("Se determină cel mai bun model pe baza metricilor agregate..."):
                    best_model_name, leaderboard = choose_best_model(eval_df)
                
                if best_model_name and best_model_name != "N/A":
                    st.success(f"Modelul Câștigător General (pe baza mediei erorilor compozite): **{best_model_name}**")
                elif best_model_name == "N/A":
                    st.warning("Nu s-a putut determina un model câștigător clar pe baza mediei compozite a erorilor.")
                else:
                    st.error("Nu s-a putut identifica un model câștigător.")


                st.subheader("Clasamentul General al Modelelor (Eroare Medie pe Metrici)")
                st.dataframe(leaderboard)
                csv_leaderboard = leaderboard.to_csv(index=True).encode('utf-8')
                st.download_button(
                    label="Descarcă Clasamentul General (CSV)",
                    data=csv_leaderboard,
                    file_name='model_performance_leaderboard.csv',
                    mime='text/csv',
                )
                st.markdown("---")
            else:
                st.warning("Nu există date de evaluare pentru a genera clasamentul și a alege cel mai bun model.")


            st.header("Grafice Diagnostice și Vizualizarea Previziunilor")
            with st.spinner("Se generează vizualizările finale..."):
                # Transmite best_model_name la visualize_results
                visualize_results(Y_df, forecast_df, cv_df, eval_df, horizon, best_model_name)
            st.success("Vizualizări diagnostice și de previziuni generate.")
            st.markdown("---")

            if best_model_name and best_model_name != "N/A" and best_model_name in forecast_df.columns:
                st.header(f"Exportă Previziunile Modelului Câștigător: {best_model_name}")
                best_model_forecast_df = (
                    forecast_df[['unique_id', 'ds', best_model_name]]
                    .rename(columns={best_model_name: 'yhat_best_model'}) # Nume generic pentru coloana de predicție
                )
                st.dataframe(best_model_forecast_df.head())

                csv_export = best_model_forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Descarcă Previziunile pentru {best_model_name} (CSV)",
                    data=csv_export,
                    file_name=f'forecast_{best_model_name}.csv',
                    mime='text/csv',
                )
            elif best_model_name and best_model_name != "N/A" and best_model_name not in forecast_df.columns:
                 st.warning(f"Modelul câștigător '{best_model_name}' a fost identificat, dar coloana corespunzătoare nu se găsește în DataFrame-ul de previziuni combinat. Exportul nu este posibil.")
            else:
                st.info("Nu s-a putut exporta previziunile deoarece un model câștigător nu a fost determinat sau nu este valid.")
            
            st.success("Procesul de prognoză a fost finalizat!")
            st.balloons() # Un mic semn de finalizare

        elif uploaded_file is not None: # Y_df este gol, dar fișierul a fost încărcat
             st.error("Datele nu au putut fi procesate. Verificați structura fișierului și configurațiile pipeline-ului.")

    elif run_pipeline_button and uploaded_file is None:
        st.warning("Vă rugăm să încărcați un fișier de date (de ex., `train.csv`) pentru a începe procesul de prognoză.")

else:
    st.info("Configurați parametrii în bara laterală și apăsați 'Rulează Pipeline-ul de Prognoză' pentru a începe.")