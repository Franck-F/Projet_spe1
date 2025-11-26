import streamlit as st
import plotly.express as px
import pandas as pd

def render_denmark(df):
    st.header("Danemark - Aperçu Électrique")
    
    # Filter for Denmark columns
    dk_cols = [c for c in df.columns if c.startswith('DK_')]
    if not dk_cols:
        st.warning("Aucune donnée disponible pour le Danemark.")
        return
        
    df_dk = df[dk_cols]
    
    # --- KPI Cards ---
    col1, col2, col3 = st.columns(3)
    
    load_col = 'DK_load_actual_entsoe_transparency'
        
    avg_load = df_dk[load_col].mean() if load_col in df_dk else 0
    max_load = df_dk[load_col].max() if load_col in df_dk else 0
    
    # Solar
    solar_col = 'DK_solar_generation_actual'
    total_solar = df_dk[solar_col].sum() / 1e6 if solar_col in df_dk else 0
    
    with col1:
        st.metric("Charge Moyenne (MW)", f"{avg_load:,.0f}")
    with col2:
        st.metric("Charge Max (MW)", f"{max_load:,.0f}")
    with col3:
        st.metric("Solaire Total (TWh)", f"{total_solar:.2f}")

    st.markdown("---")
    
    # --- Tabs for Analysis ---
    tab1, tab2 = st.tabs(["Analyse de la Charge", "Mix Énergétique"])
    
    with tab1:
        st.subheader("Charge au fil du temps")
        if load_col in df_dk:
            df_plot = df_dk[load_col].resample('D').mean()
            fig = px.line(df_plot, title="Charge Moyenne Journalière (MW)")
            fig.update_layout(xaxis_title="Date", yaxis_title="Charge (MW)")
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.subheader("Mix Énergétique")
        gen_cols = ['DK_solar_generation_actual', 'DK_wind_generation_actual']
        avail_gen = [c for c in gen_cols if c in df_dk.columns]
        
        if avail_gen:
            df_gen = df_dk[avail_gen].resample('D').mean()
            fig_gen = px.area(df_gen, title="Mix Énergétique Moyen Journalier")
            fig_gen.update_layout(xaxis_title="Date", yaxis_title="Production (MW)", legend_title="Source")
            
            new_names = {'DK_solar_generation_actual': 'Solaire', 'DK_wind_generation_actual': 'Eolien'}
            fig_gen.for_each_trace(lambda t: t.update(name = new_names.get(t.name, t.name)))
            
            st.plotly_chart(fig_gen, use_container_width=True)
