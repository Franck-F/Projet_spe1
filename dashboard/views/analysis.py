import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def render_analysis(df):
    st.header("Detailed Analysis")
    
    if df.empty:
        st.warning("No data available.")
        return

    tab1, tab2, tab3 = st.tabs(["Generation Mix", "Price Analysis", "Load vs Temperature"])
    
    with tab1:
        st.subheader("Generation Mix")
        
        # Identify generation columns
        gen_cols = [
            'FR_solar_generation_actual',
            'FR_wind_onshore_generation_actual',
            # Add others if they exist in the dataset, based on the notebook I saw:
            # The notebook showed many columns, but I only saw a subset in the `view_file` output.
            # I'll stick to what I saw or generic ones, but let's try to be safe.
            # If columns don't exist, we skip them.
        ]
        
        # Let's check which columns actually exist
        available_gen_cols = [c for c in gen_cols if c in df.columns]
        
        if available_gen_cols:
            # Stacked area chart for generation
            # Resample to daily for better visualization of long periods
            df_daily = df[available_gen_cols].resample('D').mean()
            
            fig_gen = px.area(df_daily, x=df_daily.index, y=available_gen_cols,
                              title="Daily Average Generation Mix",
                              labels={'value': 'Generation (MW)', 'variable': 'Source'})
            st.plotly_chart(fig_gen, use_container_width=True)
        else:
            st.info("Generation data not available in the selected subset.")

    with tab2:
        st.subheader("Price Analysis")
        # Check for price column. In the notebook I saw 'AT_price_day_ahead', 'DE_LU_price_day_ahead'.
        # I need to check if there is a 'FR_price_day_ahead' or similar. 
        # The notebook output didn't explicitly show FR price in the first 800 lines, but it's likely there.
        # I'll look for any column with 'price' in it.
        
        price_cols = [c for c in df.columns if 'price' in c]
        if price_cols:
            selected_price = st.selectbox("Select Price Metric", price_cols)
            
            # Histogram of prices
            fig_hist = px.histogram(df, x=selected_price, nbins=50, title=f"Distribution of {selected_price}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Price vs Load Scatter
            if 'FR_load_actual_entsoe_transparency' in df.columns:
                # Sample down for scatter plot performance
                df_sample = df.sample(n=min(5000, len(df)))
                fig_scatter = px.scatter(df_sample, x='FR_load_actual_entsoe_transparency', y=selected_price,
                                         title=f"Load vs {selected_price} (Sampled)",
                                         opacity=0.5)
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Price data not found.")

    with tab3:
        st.subheader("Seasonality & Patterns")
        # Heatmap of Load (Hour of Day vs Day of Week)
        if 'FR_load_actual_entsoe_transparency' in df.columns:
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.day_name()
            
            # Pivot table
            heatmap_data = df.groupby(['DayOfWeek', 'Hour'])['FR_load_actual_entsoe_transparency'].mean().unstack()
            
            # Reorder days
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(days_order)
            
            fig_heat = px.imshow(heatmap_data, 
                                 labels=dict(x="Hour of Day", y="Day of Week", color="Avg Load (MW)"),
                                 title="Average Load Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)
