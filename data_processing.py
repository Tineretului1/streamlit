# data_processing.py

import streamlit as st
import pandas as pd
import re

# ───────────────────────── PRELUCRARE DATE ───────────────────────── #

def _find_column(cols: pd.Index, pattern: str) -> str:
    """Return the first column whose name matches *pattern* (case‑insensitive REGEX)."""
    candidates = [c for c in cols if re.search(pattern, c, flags=re.IGNORECASE)]
    if not candidates:
        raise ValueError(f"Nicio coloană care să corespundă modelului regex '{pattern}' nu a fost găsită în setul de date.")
    return candidates[0]

@st.cache_data
def load_and_prepare(uploaded_file, uploaded_external_file=None, stores: list[str] | None = None,
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
        st.error("Fișierul CSV principal încărcat este gol sau nu a putut fi citit corect.")
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
    df_processed = df_initial[[store_col_name, item_col_name, sales_col_name, date_col_name]].copy()
    df_processed.rename(columns={
        store_col_name: 'store',
        item_col_name:  'item',
        sales_col_name: 'sales',
        date_col_name:  'date'
    }, inplace=True)

    # 3. Converteste coloanele 'store' și 'item' la tipul string devreme
    df_processed['store'] = df_processed['store'].astype(str)
    df_processed['item']  = df_processed['item'].astype(str)

    # 4. Filtrează după magazine, dacă este specificat
    df_to_use = df_processed

    if stores:
        unique_stores_in_data_before_filter = df_processed['store'].unique()
        df_filtered_by_store = df_processed[df_processed['store'].isin(stores)]
        
        if df_filtered_by_store.empty:
            st.warning(
                f"Nicio dată găsită PENTRU MAGAZINELE SPECIFICATE: {stores}. "
                f"Se vor folosi toate magazinele din fișier."
            )
        else:
            st.success(f"Date filtrate cu succes pentru magazinele specificate: {stores}. "
                       f"Număr de înregistrări după filtrare: {len(df_filtered_by_store)}")
            df_to_use = df_filtered_by_store
    else:
        st.info("Nu s-au specificat magazine dedicate; se vor procesa datele pentru toate magazinele.")

    # 5. Prelucrări ulterioare pe DataFrame-ul selectat (df_to_use)
    if df_to_use.empty:
        st.error("DataFrame-ul este gol înainte de agregarea finală.")
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
        st.error("DataFrame-ul este gol după gruparea și însumarea vânzărilor.")
        return pd.DataFrame()

    min_date = grouped['date'].min()
    max_date = grouped['date'].max()
    
    if pd.isna(min_date) or pd.isna(max_date):
        st.error("Nu s-au putut determina limitele de date (min/max) după grupare.")
        return pd.DataFrame()
        
    all_dates_range = pd.date_range(start=min_date, end=max_date, freq='D')
    all_unique_ids  = grouped['store_item'].unique()

    if not all_unique_ids.any():
        st.error("Niciun 'store_item' unic găsit după procesare.")
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

    # Procesare fișier extern, dacă este furnizat
    if uploaded_external_file is not None:
        try:
            uploaded_external_file.seek(0)
            df_ext = pd.read_csv(uploaded_external_file) # Use a temporary name

            if df_ext.empty:
                st.info("Fișierul de date externe este gol și va fi ignorat.")
                # df_ext remains empty, so no merge will happen
            elif len(df_ext.columns) != 2:
                st.warning(
                    f"Fișierul de date externe trebuie să conțină exact două coloane (o coloană de dată și o coloană de preț/valoare). "
                    f"Am găsit {len(df_ext.columns)} coloane: {', '.join(df_ext.columns)}. Fișierul va fi ignorat."
                )
                df_ext = pd.DataFrame() # Mark as empty to skip merge
            else:
                # Has exactly two columns
                col1_name, col2_name = df_ext.columns[0], df_ext.columns[1]
                date_col_original_name = None
                price_col_original_name = None
                valid_external_data = False

                # Try to identify date and price columns
                try:
                    # Test col1 as date
                    pd.to_datetime(df_ext[col1_name], errors='raise', infer_datetime_format=True)
                    date_col_original_name = col1_name
                    price_col_original_name = col2_name
                    valid_external_data = True
                except (ValueError, TypeError, pd.errors.ParserError):
                    # col1 failed, try col2 as date
                    try:
                        pd.to_datetime(df_ext[col2_name], errors='raise', infer_datetime_format=True)
                        date_col_original_name = col2_name
                        price_col_original_name = col1_name
                        valid_external_data = True
                    except (ValueError, TypeError, pd.errors.ParserError):
                        st.warning(
                            f"Nu s-a putut identifica o coloană de dată validă în fișierul extern din coloanele '{col1_name}' și '{col2_name}'. "
                            "Asigurați-vă că una dintre coloane conține date calendaristice valide (ex: YYYY-MM-DD). Fișierul va fi ignorat."
                        )
                        df_ext = pd.DataFrame() # Mark as empty

                if valid_external_data and not df_ext.empty:
                    st.info(f"În fișierul extern, coloana de dată identificată: '{date_col_original_name}', "
                              f"coloana de preț/valoare identificată: '{price_col_original_name}'.")
                    
                    # Create new columns for 'ds' and 'external_feature' to avoid SettingWithCopyWarning
                    df_temp = pd.DataFrame()
                    df_temp['ds'] = pd.to_datetime(df_ext[date_col_original_name])
                    df_temp['external_feature'] = df_ext[price_col_original_name]
                    
                    df_ext = df_temp # Replace df_ext with the new DataFrame

                    # Asigură-te că 'external_feature' este numeric
                    df_ext['external_feature'] = pd.to_numeric(df_ext['external_feature'], errors='coerce')

                    # Elimină duplicatele de date (pe baza 'ds'), păstrând prima apariție pentru a asigura granularitatea zilnică
                    df_ext = df_ext.drop_duplicates(subset=['ds'], keep='first')

                    # Verifică dacă df_ext mai are date după procesare
                    if df_ext.empty or df_ext['external_feature'].isnull().all():
                        st.warning(f"După procesare, fișierul extern nu conține date valide sau toate valorile prețului/valorii sunt nule. Va fi ignorat.")
                    else:
                        full_df = pd.merge(full_df, df_ext, on='ds', how='left')
                        # Umple valorile NaN în coloana de feature extern folosind ffill și apoi bfill
                        full_df['external_feature'] = full_df['external_feature'].ffill().bfill()

                        if 'external_feature' in full_df.columns:
                             st.info(f"Datele externe din coloana originală '{price_col_original_name}' au fost integrate ca 'external_feature'.")
                        if 'external_feature' in full_df.columns and full_df['external_feature'].isnull().any():
                             st.warning("Unele valori pentru caracteristica externă (după ffill/bfill și integrare) rămân NaN.")
                elif not valid_external_data: # This case is if parsing failed and df_ext was already set to empty
                    pass # Warning already shown, df_ext is empty.
            # If df_ext is empty at this point (either initially, or due to validation failure), no merge happens.
        except Exception as e:
            st.error(f"Eroare la procesarea fișierului de date externe: {e}. Acesta va fi ignorat.")

    st.success(f"Pregătirea datelor finalizată. DataFrame final conține {len(full_df)} rânduri.")
    return full_df