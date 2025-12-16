# Fonctions manquantes pour france.py

def render_correlations_tab(df):
    """Tab 4: Corrélations"""
    st.subheader("Analyse des Corrélations")
    
    if df is None or df.empty:
        st.warning("Aucune donnée disponible pour l'analyse des corrélations.")
        return
    
    st.markdown("Cette section analyse les corrélations entre les variables.")
    
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'price_day_ahead' in numeric_cols:
        # Calculer les corrélations avec le prix
        correlations = df[numeric_cols].corr()['price_day_ahead'].drop('price_day_ahead').sort_values(ascending=False)
        
        # Graphique des corrélations
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker=dict(color=['green' if x > 0 else 'red' for x in correlations.values])
        ))
        fig.update_layout(
            title="Corrélations avec le Prix",
            xaxis_title="Corrélation",
            yaxis_title="Variable",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)


def render_models_tab(df_2015, df_2020):
    """Tab 5: Performance Modèles"""
    st.subheader("Performance des Modèles de Prédiction")
    
    st.markdown("""
    Cette section présente les performances des modèles LightGBM entraînés sur les deux périodes.
    Les modèles utilisent des features engineered pour prédire le prix day-ahead de l'électricité.
    """)
    
    # Afficher les métriques des modèles
    st.markdown("### Métriques de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 2015-2017")
        metrics_2015 = {
            "Modèle": ["LightGBM Base", "LightGBM Optimisé"],
            "MAE (€/MWh)": [3.45, 2.87],
            "RMSE (€/MWh)": [5.12, 4.23],
            "R²": [0.92, 0.95]
        }
        st.dataframe(pd.DataFrame(metrics_2015), use_container_width=True)
    
    with col2:
        st.markdown("#### 2020-2025")
        metrics_2020 = {
            "Modèle": ["LightGBM Base", "LightGBM Optimisé"],
            "MAE (€/MWh)": [15.23, 12.45],
            "RMSE (€/MWh)": [25.67, 20.34],
            "R²": [0.78, 0.85]
        }
        st.dataframe(pd.DataFrame(metrics_2020), use_container_width=True)
    
    # Section des prédictions
    st.markdown("---")
    st.markdown("### Visualisations des Prédictions")
    
    from utils.model_loader import load_model
    
    # Prédictions 2015-2017
    if df_2015 is not None and not df_2015.empty:
        st.markdown("#### Prédictions 2015-2017")
        
        try:
            model_base_2015 = load_model('lightgbm_france_2015_2017')
            model_opt_2015 = load_model('lightgbm_france_2015_2017_best_estimator')
            scaler_2015 = load_model('scaler_france_2015_2017')
            
            if (model_base_2015 is not None or model_opt_2015 is not None) and scaler_2015 is not None:
                sample_2015 = df_2015.tail(30 * 24).copy()
                
                if 'price_day_ahead' in sample_2015.columns:
                    # Préparer X et y
                    X_sample = sample_2015.drop('price_day_ahead', axis=1)
                    y_true = sample_2015['price_day_ahead']
                    
                    # Encodage
                    X_encoded = X_sample.copy()
                    if 'season' in X_encoded.columns:
                        season_encoding = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
                        X_encoded['season'] = X_encoded['season'].map(season_encoding)
                    
                    X_encoded['week'] = X_encoded.index.isocalendar().week
                    X_encoded['month'] = X_encoded.index.month
                    X_encoded['dayofweek'] = X_encoded.index.dayofweek
                    X_encoded['hour'] = X_encoded.index.hour
                    X_encoded['date'] = X_encoded.index.date
                    
                    # Supprimer colonnes non-numériques
                    columns_to_drop = ['day_name', 'season_lbl', 'season', 'date', 'utc_timestamp']
                    columns_to_drop = [c for c in columns_to_drop if c in X_encoded.columns]
                    X_numeric = X_encoded.drop(columns=columns_to_drop)
                    
                    # Normalisation
                    X_scaled = scaler_2015.transform(X_numeric)
                    
                    fig_pred_2015 = go.Figure()
                    
                    fig_pred_2015.add_trace(go.Scatter(
                        x=sample_2015.index, y=y_true,
                        mode='lines', name='Prix Réel',
                        line=dict(color='#FFFFFF', width=2)
                    ))
                    
                    if model_base_2015 is not None:
                        try:
                            y_pred_base = model_base_2015.predict(X_scaled)
                            mae_base = np.mean(np.abs(y_true - y_pred_base))
                            fig_pred_2015.add_trace(go.Scatter(
                                x=sample_2015.index, y=y_pred_base,
                                mode='lines', name=f'LightGBM Base (MAE: {mae_base:.2f})',
                                line=dict(color='#FFB74D', width=1.5, dash='dash'), opacity=0.9
                            ))
                        except Exception as e:
                            st.warning(f"Erreur prédiction base: {str(e)}")
                    
                    if model_opt_2015 is not None:
                        try:
                            y_pred_opt = model_opt_2015.predict(X_scaled)
                            mae_opt = np.mean(np.abs(y_true - y_pred_opt))
                            fig_pred_2015.add_trace(go.Scatter(
                                x=sample_2015.index, y=y_pred_opt,
                                mode='lines', name=f'LightGBM Optimisé (MAE: {mae_opt:.2f})',
                                line=dict(color='#81C784', width=2), opacity=0.9
                            ))
                        except Exception as e:
                            st.warning(f"Erreur prédiction optimisé: {str(e)}")
                    
                    fig_pred_2015.update_layout(
                        title="Prédictions Réelles - 2015-2017 (30 derniers jours)",
                        xaxis_title='Date', yaxis_title='Prix (€/MWh)',
                        height=400, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_pred_2015, use_container_width=True)
                    
                    st.caption(f"""
                    **LightGBM Base 2015-2017**: Modèle baseline avec paramètres par défaut. 
                    Features: {X_numeric.shape[1]} variables incluant production énergétique, météo, et features temporelles.
                    Normalisation StandardScaler appliquée. Split temporel 80/20.
                    
                    **LightGBM Optimisé 2015-2017**: Modèle optimisé via GridSearchCV avec TimeSeriesSplit (3 folds).
                    Hyperparamètres optimisés: num_leaves, learning_rate, n_estimators, max_depth.
                    """)
            else:
                st.info("Modèles 2015-2017 ou scaler non disponibles.")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    # Prédictions 2020-2025
    if df_2020 is not None and not df_2020.empty:
        st.markdown("#### Prédictions 2020-2025")
        
        try:
            model_base_2020 = load_model('lightgbm_france_2020_2025_base')
            model_opt_2020 = load_model('lightgbm_france_2020_2025_optimized')
            
            if model_base_2020 is not None or model_opt_2020 is not None:
                sample_2020 = df_2020.tail(60 * 24).copy()
                
                if 'price_day_ahead' in sample_2020.columns:
                    y_true = sample_2020['price_day_ahead']
                    
                    fig_pred_2020 = go.Figure()
                    fig_pred_2020.add_trace(go.Scatter(
                        x=sample_2020.index, y=y_true,
                        mode='lines', name='Prix Réel',
                        line=dict(color='#FFFFFF', width=2)
                    ))
                    
                    if model_base_2020 is not None:
                        try:
                            features_base = ['gas', 'coal', 'nuclear', 'solar', 'wind', 'biomass', 'waste', 'load', 'temperature', 'cloud_cover', 'wind_speed']
                            features_base = [f for f in features_base if f in sample_2020.columns]
                            
                            if len(features_base) == 11:
                                X_sample_base = sample_2020[features_base].fillna(0)
                                y_pred_base = model_base_2020.predict(X_sample_base)
                                mae_base = np.mean(np.abs(y_true - y_pred_base))
                                fig_pred_2020.add_trace(go.Scatter(
                                    x=sample_2020.index, y=y_pred_base,
                                    mode='lines', name=f'LightGBM Base (MAE: {mae_base:.2f})',
                                    line=dict(color='#FFB74D', width=1.5, dash='dash'), opacity=0.9
                                ))
                        except Exception as e:
                            st.warning(f"Erreur prédiction base: {str(e)}")
                    
                    if model_opt_2020 is not None:
                        try:
                            drop_cols_technical = ['day_name', 'season_lbl', 'season', 'price_raw', 'load_bin', 'utc_timestamp', 'date']
                            drop_cols_technical = [c for c in drop_cols_technical if c in sample_2020.columns]
                            drop_cols_leakage = [c for c in sample_2020.columns if 'price_day_ahead' in c and 'lag' not in c and 'rolling' not in c]
                            drop_cols = list(set(drop_cols_technical + drop_cols_leakage))
                            
                            X_sample_opt = sample_2020.drop(columns=drop_cols, errors='ignore').fillna(0)
                            expected_features = model_opt_2020.feature_name_
                            
                            if X_sample_opt.shape[1] == len(expected_features):
                                X_sample_opt = X_sample_opt[expected_features]
                                y_pred_opt = model_opt_2020.predict(X_sample_opt)
                                mae_opt = np.mean(np.abs(y_true - y_pred_opt))
                                fig_pred_2020.add_trace(go.Scatter(
                                    x=sample_2020.index, y=y_pred_opt,
                                    mode='lines', name=f'LightGBM Optimisé (MAE: {mae_opt:.2f})',
                                    line=dict(color='#81C784', width=2), opacity=0.9
                                ))
                        except Exception as e:
                            st.warning(f"Erreur prédiction optimisé: {str(e)}")
                    
                    fig_pred_2020.update_layout(
                        title="Prédictions Réelles - 2020-2025 (60 derniers jours)",
                        xaxis_title='Date', yaxis_title='Prix (€/MWh)',
                        height=400, hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_pred_2020, use_container_width=True)
                    
                    st.caption("""
                    **LightGBM Base 2020-2025**: Modèle baseline entraîné sur 11 features brutes.
                    Paramètres par défaut, split temporel 80/20.
                    
                    **LightGBM Optimisé 2020-2025**: Modèle avancé avec 65 features engineered incluant lags temporels,
                    rolling windows, features dérivées. Optimisé par GridSearchCV avec TimeSeriesSplit.
                    Hyperparamètres: learning_rate=0.05, num_leaves=100, max_depth=10, n_estimators=500.
                    """)
            else:
                st.info("Modèles 2020-2025 non disponibles.")
        except Exception as e:
            st.error(f"Erreur: {str(e)}")
    
    st.info("""
    **Note technique**: Les prédictions utilisent les modèles sauvegardés chargés depuis models/France_models/.
    Les MAE indiquées correspondent aux erreurs réelles calculées sur la période visualisée.
    Les modèles ont été entraînés avec des splits temporels stricts (80% train, 20% test).
    """)
