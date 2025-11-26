import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render_overview(df):
    st.header("General Overview")
    
    if df.empty:
        st.warning("No data available to display.")
        return

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate KPIs
    avg_load = df['FR_load_actual_entsoe_transparency'].mean()
    max_load = df['FR_load_actual_entsoe_transparency'].max()
    total_solar = df['FR_solar_generation_actual'].sum() / 1e6 # TWh
    total_wind = df['FR_wind_onshore_generation_actual'].sum() / 1e6 # TWh
    
    with col1:
        st.metric("Avg Load (MW)", f"{avg_load:,.0f}")
    with col2:
        st.metric("Max Load (MW)", f"{max_load:,.0f}")
    with col3:
        st.metric("Total Solar (TWh)", f"{total_solar:.2f}")
    with col4:
        st.metric("Total Wind (TWh)", f"{total_wind:.2f}")
        
    st.markdown("---")
    
    # --- Global Time Series ---
    st.subheader("Electricity Load Over Time")
    
    # Downsample for performance if needed, or just plot
    # Plotting all points might be heavy, let's resample to daily mean for the overview if range is large
    
    time_range = df.index.max() - df.index.min()
    if time_range.days > 365:
        df_plot = df.resample('D').mean()
        title_suffix = "(Daily Average)"
    else:
        df_plot = df
        title_suffix = "(Hourly)"
        
    fig = px.line(df_plot, y='FR_load_actual_entsoe_transparency', 
                  title=f"French Electricity Load {title_suffix}",
                  labels={'FR_load_actual_entsoe_transparency': 'Load (MW)', 'Local_Time': 'Date'})
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Load (MW)")
    st.plotly_chart(fig, use_container_width=True)
