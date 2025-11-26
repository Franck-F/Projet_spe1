import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """
    Loads the electricity data, parses dates, and filters for France.
    Returns a pandas DataFrame.
    """
    # Path relative to the dashboard directory when running from there, 
    # or absolute path. Adjusting to be robust.
    # Assuming app runs from dashboard/ or root.
    # We'll try to find the data file.
    
    possible_paths = [
        "../data/raw/time_series_60min.csv",
        "data/raw/time_series_60min.csv",
        "C:/Users/conta/Downloads/Projet_spe1/data/raw/time_series_60min.csv"
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        st.error("Fichier de données introuvable. Veuillez vous assurer que 'time_series_60min.csv' est dans 'data/raw/'.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            file_path,
            parse_dates=['utc_timestamp', 'cet_cest_timestamp'],
            low_memory=False
        )
        
        # Filter for France and Denmark columns and timestamps
        # We look for columns starting with FR_ or DK_ or the specific price column
        cols_to_keep = ['utc_timestamp', 'cet_cest_timestamp', 'IT_NORD_FR_price_day_ahead'] + [col for col in df.columns if col.startswith('FR_') or col.startswith('DK_')]
        
        df_filtered = df[cols_to_keep].copy()
        
        # Rename timestamps for convenience
        df_filtered.rename(columns={
            'utc_timestamp': 'UTC',
            'cet_cest_timestamp': 'Local_Time'
        }, inplace=True)
        
        # Set Local_Time as index for easier plotting
        df_filtered['Local_Time'] = pd.to_datetime(df_filtered['Local_Time'], utc=True).dt.tz_convert('Europe/Paris')
        df_filtered.set_index('Local_Time', inplace=True)
        
        return df_filtered
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()
