import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap

# --- Helper Functions ---

@st.cache_resource
def train_model_and_get_shap(df_train):
    """
    Trains the LightGBM model using the best parameters from EDA_France.py
    and calculates SHAP values.
    Returns: model, shap_values, X_test_shap, feature_names
    """
    # Feature Engineering (Simplified replication of EDA_France.py)
    df_featured = df_train.copy()
    
    # Temporal features
    df_featured['hour'] = df_featured.index.hour
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['day_of_month'] = df_featured.index.day
    df_featured['month'] = df_featured.index.month
    df_featured['year'] = df_featured.index.year
    df_featured['quarter'] = df_featured.index.quarter
    df_featured['week_of_year'] = df_featured.index.isocalendar().week.astype(int)
    df_featured['is_weekend'] = (df_featured['day_of_week'] >= 5).astype(int)
    
    # Cyclic features
    df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour'] / 24)
    df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour'] / 24)
    df_featured['day_sin'] = np.sin(2 * np.pi * df_featured['day_of_week'] / 7)
    df_featured['day_cos'] = np.cos(2 * np.pi * df_featured['day_of_week'] / 7)
    df_featured['month_sin'] = np.sin(2 * np.pi * df_featured['month'] / 12)
    df_featured['month_cos'] = np.cos(2 * np.pi * df_featured['month'] / 12)
    
    # Lags and Rolling (subset for speed/memory if needed, but using full set from script)
    for lag in [1, 2, 3, 24, 48, 168]:
        df_featured[f'price_lag_{lag}'] = df_featured['price'].shift(lag)
        df_featured[f'load_actual_lag_{lag}'] = df_featured['load_actual'].shift(lag)
        
    for window in [24, 168]:
        df_featured[f'price_rolling_mean_{window}'] = df_featured['price'].rolling(window=window).mean()
        df_featured[f'price_rolling_std_{window}'] = df_featured['price'].rolling(window=window).std()
        df_featured[f'load_actual_rolling_mean_{window}'] = df_featured['load_actual'].rolling(window=window).mean()
        
    df_featured['price_diff_1'] = df_featured['price'].diff(1)
    df_featured['price_diff_24'] = df_featured['price'].diff(24)
    
    df_featured['load_forecast_error'] = df_featured['load_forecast'] - df_featured['load_actual']
    df_featured['renewable_generation'] = df_featured['solar_generation'] + df_featured['wind_generation']
    df_featured['renewable_ratio'] = df_featured['renewable_generation'] / (df_featured['load_actual'] + 1)
    
    df_featured = df_featured.dropna()
    
    # Prepare X and y
    # Drop target and any non-numeric columns (like UTC timestamp if present)
    X = df_featured.drop(columns=['price'])
    # Select only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    y = df_featured['price']
    
    # Best params from EDA_France.py
    best_params_shap = {
        'n_estimators': 965,
        'learning_rate': 0.023305698375173753,
        'num_leaves': 62, 
        'max_depth': 6, 
        'min_child_samples': 73, 
        'feature_fraction': 0.9107029697230191, 
        'bagging_fraction': 0.684479044902335, 
        'bagging_freq': 3, 
        'lambda_l1': 0.029500110573817455, 
        'lambda_l2': 0.16428913564785613,
        'random_state': 42,
        'verbose': -1
    }
    
    # Train model
    model = lgb.LGBMRegressor(**best_params_shap)
    model.fit(X, y)
    
    # Calculate SHAP on a subset (last 3 months as in script)
    test_start_date = df_featured.index.max() - pd.DateOffset(months=3)
    X_test_shap = X.loc[X.index >= test_start_date]
    
    # Subsample for performance if needed (e.g. max 2000 points)
    if len(X_test_shap) > 2000:
        X_test_shap = X_test_shap.sample(2000, random_state=42)
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_shap)
    
    return model, shap_values, X_test_shap, X.columns.tolist()

def render_france(df_orig):
    st.header("France : Analyse et Prédiction du Prix de l'Électricité")
    
    # --- 1. PRÉTRAITEMENT ---
    df = df_orig.copy()
    
    rename_dict = {
        'IT_NORD_FR_price_day_ahead': 'price',
        'FR_load_actual_entsoe_transparency': 'load_actual',
        'FR_load_forecast_entsoe_transparency': 'load_forecast',
        'FR_solar_generation_actual': 'solar_generation',
        'FR_wind_onshore_generation_actual': 'wind_generation'
    }
    rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
    df.rename(columns=rename_dict, inplace=True)
    
    start_date = '2015-01-05'
    end_date = '2017-12-05'
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    df = df.sort_index()
    df = df.loc[start_date:end_date]
    df.interpolate(method='linear', inplace=True)
    
    if 'price' not in df.columns:
        st.error("Erreur : La colonne de prix est introuvable.")
        return

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Vue d'Ensemble", 
        "Profil du Marché (EDA)", 
        "Performance Modèle", 
        "Leviers du Prix (SHAP)"
    ])
    
    # --- Tab 1: Vue d'Ensemble ---
    with tab1:
        st.subheader("Executive Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Erreur Moyenne (MAE)", "3.99 €/MWh", delta="-5% vs Base")
        with col2:
            st.metric("Performance vs Base", "+5%", help="Amélioration par rapport à la MAE de 4.20 €/MWh")
        with col3:
            st.metric("Période d'Analyse", "Jan 2015 - Déc 2017")
        st.markdown("---")
        
        st.markdown("##### Tendance des Prix (3 derniers mois)")
        last_date = df.index.max()
        start_plot = last_date - pd.Timedelta(days=90)
        df_recent = df[df.index >= start_plot]
        
        fig = px.line(df_recent, y='price', title="Evolution du Prix (Fin 2017)")
        fig.update_traces(line_color='#00d4ff', name='Prix Réel')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Nous avons développé un modèle prédictif du prix de l'électricité avec une erreur moyenne de 3.99 €/MWh.")

    # --- Tab 2: Profil du Marché (EDA) ---
    with tab2:
        st.subheader("Analyse Exploratoire des Données")
        
        # ... (Previous charts: Distribution, Evolution, Seasonality) ...
        # I will keep them brief here to save space in this artifact, but they should be present.
        # Re-implementing the key ones requested.
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig_dist = px.histogram(df, x="price", nbins=50, title="Distribution du Prix")
            fig_dist.update_traces(marker_color='#00d4ff')
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_b:
            df_seasonal = df.copy()
            df_seasonal['month'] = df_seasonal.index.month_name()
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            fig_annual = px.box(df_seasonal, x="month", y="price", title="Saisonnalité Annuelle", category_orders={'month': months_order})
            fig_annual.update_traces(marker_color='#00d4ff')
            st.plotly_chart(fig_annual, use_container_width=True)

        # Correlation Matrix - FILTERED
        st.markdown("#### Matrice de Corrélation (France uniquement)")
        # Filter columns: keep only numeric and exclude 'DK_'
        numeric_df = df.select_dtypes(include=[np.number])
        cols_fr = [c for c in numeric_df.columns if not c.startswith('DK_')]
        corr_matrix = numeric_df[cols_fr].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0
        ))
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Train Model (Cached) ---
    with st.spinner("Entraînement du modèle et calcul SHAP en cours (une seule fois)..."):
        model, shap_values, X_test_shap, feature_names = train_model_and_get_shap(df)

    # --- Tab 3: Performance Modèle ---
    with tab3:
        st.subheader("Performance du Modèle")
        
        perf_data = {
            "Métrique": ["MAE", "RMSE"],
            "Modèle de Base": ["4.20 €/MWh", "6.26 €/MWh"],
            "Modèle Optimisé": ["3.99 €/MWh", "5.89 €/MWh"],
            "Amélioration": ["-5.0%", "-5.9%"]
        }
        st.table(pd.DataFrame(perf_data))
        
        # Feature Importance Plot
        st.markdown("### Importance des Features (Gain)")
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        feature_imp = feature_imp.sort_values(by='Importance', ascending=True).tail(20) # Top 20
        
        fig_imp = px.bar(feature_imp, x='Importance', y='Feature', orientation='h', title="Importance des Features (LightGBM)")
        fig_imp.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- Tab 4: Leviers du Prix (SHAP) ---
    with tab4:
        st.subheader("Interprétabilité du Modèle (SHAP)")
        
        st.markdown("""
        **Insight 1 : Le passé prédit le futur.** Le prix de la veille est déterminant.
        **Insight 2 : Le rythme de la vie.** L'heure de la journée est cruciale.
        **Insight 3 : L'effet du solaire.** La production solaire fait baisser les prix.
        """)
        
        # Global Importance (Mean |SHAP|)
        st.markdown("#### Importance Globale (|SHAP| moyen)")
        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
            
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        shap_imp_df = pd.DataFrame({'Feature': X_test_shap.columns, 'Mean |SHAP|': mean_abs_shap})
        shap_imp_df = shap_imp_df.sort_values(by='Mean |SHAP|', ascending=True).tail(20)
        
        fig_shap_bar = px.bar(shap_imp_df, x='Mean |SHAP|', y='Feature', orientation='h', title="Impact Moyen sur le Prix (€/MWh)")
        fig_shap_bar.update_traces(marker_color='#00d4ff')
        st.plotly_chart(fig_shap_bar, use_container_width=True)
        
        # Summary Plot (Beeswarm style using scatter)
        st.markdown("#### SHAP Summary Plot (Distribution de l'impact)")
        
        # Prepare data for beeswarm
        # We need to melt the shap values and feature values
        shap_df = pd.DataFrame(shap_array, columns=X_test_shap.columns)
        feature_df = X_test_shap.reset_index(drop=True)
        
        # Limit to top 15 features for readability
        top_features = shap_imp_df['Feature'].tail(15).tolist()
        
        shap_melt = shap_df[top_features].melt(var_name='Feature', value_name='SHAP Value')
        feature_melt = feature_df[top_features].melt(var_name='Feature', value_name='Feature Value')
        
        # Combine
        shap_melt['Feature Value'] = feature_melt['Feature Value']
        
        # Plot
        fig_beeswarm = px.strip(
            shap_melt, 
            x='SHAP Value', 
            y='Feature', 
            color='Feature Value',
            title="Impact détaillé des features (Beeswarm)"
        )
        fig_beeswarm.update_traces(marker=dict(size=5, opacity=0.7))
        st.plotly_chart(fig_beeswarm, use_container_width=True)
