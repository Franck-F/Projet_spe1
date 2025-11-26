import streamlit as st
import plotly.express as px
import pandas as pd

def render_comparison(df):
    st.header("Comparaison France vs. Danemark")
    
    # Load columns
    fr_load = 'FR_load_actual_entsoe_transparency'
    dk_load = 'DK_load_actual_entsoe_transparency'
    
    if fr_load not in df.columns or dk_load not in df.columns:
        st.warning("Données insuffisantes pour la comparaison.")
        return
        
    # --- Load Comparison ---
    st.subheader("Comparaison de la Charge")
    
    df_comp = df[[fr_load, dk_load]].resample('D').mean()
    
    fig = px.line(df_comp, title="Comparaison de la Charge Moyenne Journalière (MW)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Charge (MW)", legend_title="Pays")
    
    new_names = {fr_load: 'France', dk_load: 'Danemark'}
    fig.for_each_trace(lambda t: t.update(name = new_names.get(t.name, t.name)))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("Note : La France a une charge nettement plus élevée que le Danemark en raison de sa taille et de sa population.")
    
    # --- Price Comparison ---
    st.subheader("Comparaison des Prix")
    # Look for price columns
    fr_price = [c for c in df.columns if 'FR' in c and 'price' in c]
    dk_price = [c for c in df.columns if 'DK' in c and 'price' in c] 
    
    price_cols = fr_price + dk_price
    if price_cols:
        df_price = df[price_cols].resample('D').mean()
        fig_price = px.line(df_price, title="Comparaison des Prix Moyens Journaliers")
        fig_price.update_layout(xaxis_title="Date", yaxis_title="Prix (€/MWh)", legend_title="Indicateur")
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Données de prix non disponibles pour la comparaison.")
