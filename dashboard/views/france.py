import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.data_loader import load_france_data


def render_france(df_orig):
    """
    Renders the enhanced France page with visualizations from both periods.
    """
    st.header("🇫🇷 France : Analyse du Prix de l'Électricité")
    
    # --- Load Both Datasets ---
    with st.spinner("Chargement des datasets France..."):
        datasets = load_france_data()
    
    if not datasets:
        st.error("Aucun dataset France chargé. Vérifiez les fichiers CSV.")
        return
    
    df_2015 = datasets.get('2015_2017')
    df_2020 = datasets.get('2020_2025')
    
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'Ensemble",
        "📈 EDA 2015-2017",
        "📉 EDA 2020-2025",
        "🤖 Performance Modèles",
        "🔍 SHAP Interprétabilité"
    ])
    
    # ========== TAB 1: Vue d'Ensemble ==========
    with tab1:
        st.subheader("Résumé Comparatif")
        
        st.markdown("""
        Ce dashboard présente l'analyse prédictive du prix de l'électricité en France 
        sur deux périodes distinctes aux caractéristiques très différentes.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📌 Période 2015-2017")
            if df_2015 is not None:
                st.metric("Observations", f"{len(df_2015):,}")
                if 'price_day_ahead' in df_2015.columns:
                    st.metric("Prix Moyen", f"{df_2015['price_day_ahead'].mean():.2f} €/MWh")
                    st.metric("Écart-Type", f"{df_2015['price_day_ahead'].std():.2f} €")
            st.info("**Période stable** : Marché prévisible, peu de volatilité. Idéal pour l'entraînement de modèles.")
        
        with col2:
            st.markdown("### 📌 Période 2020-2025")
            if df_2020 is not None:
                st.metric("Observations", f"{len(df_2020):,}")
                if 'price_day_ahead' in df_2020.columns:
                    st.metric("Prix Moyen", f"{df_2020['price_day_ahead'].mean():.2f} €/MWh")
                    st.metric("Écart-Type", f"{df_2020['price_day_ahead'].std():.2f} €")
            st.warning("**Période volatile** : Crise COVID-19, crise énergétique 2022, prix négatifs. Données complexes.")
        
        st.markdown("---")
        st.markdown("### 🏆 Performance des Modèles (Résumé)")
        
        perf_data = {
            "Période": ["2015-2017", "2020-2025"],
            "LightGBM MAE (Optimisé)": ["0.16", "0.61"],
            "LightGBM R²": ["1.00", "0.998"],
            "ARIMAX MAE": ["-", "28.74"],
            "ARIMAX R²": ["-", "0.453"],
        }
        st.table(pd.DataFrame(perf_data))
    
    # ========== TAB 2: EDA 2015-2017 ==========
    with tab2:
        st.subheader("Analyse Exploratoire 2015-2017")
        
        if df_2015 is None:
            st.warning("Dataset 2015-2017 non disponible.")
        else:
            st.info("""
            **Description** : Cette période représente un marché électrique *stable et prévisible*.
            Les prix suivent des patterns saisonniers clairs avec peu de valeurs extrêmes.
            """)
            
            price_col = 'price_day_ahead' if 'price_day_ahead' in df_2015.columns else None
            if price_col is None:
                st.error("Colonne de prix introuvable.")
                return
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Distribution
                fig_dist = px.histogram(df_2015, x=price_col, nbins=50, 
                                        title="Distribution du Prix (2015-2017)")
                fig_dist.update_traces(marker_color='#636EFA')
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption("📝 Distribution quasi-normale, centrée autour de 35-45 €/MWh.")
            
            with col_b:
                # Saisonnalité mensuelle
                df_month = df_2015.copy()
                df_month['month'] = df_month.index.month_name()
                months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December']
                fig_box = px.box(df_month, x='month', y=price_col, 
                                title="Saisonnalité Annuelle (2015-2017)",
                                category_orders={'month': months_order})
                fig_box.update_traces(marker_color='#636EFA')
                st.plotly_chart(fig_box, use_container_width=True)
                st.caption("📝 Pics en hiver (chauffage), creux en été.")
            
            # Évolution temporelle
            st.markdown("#### Évolution du Prix dans le Temps")
            # Convert to numeric to avoid object dtype error
            price_series = pd.to_numeric(df_2015[price_col], errors='coerce')
            daily_mean = price_series.resample('D').mean()
            fig_line = px.line(daily_mean, title="Prix Journalier Moyen (2015-2017)")
            fig_line.update_traces(line_color='#636EFA')
            st.plotly_chart(fig_line, use_container_width=True)
            st.caption("📝 Tendance stable avec une légère saisonnalité. Pas de chocs majeurs.")
    
    # ========== TAB 3: EDA 2020-2025 ==========
    with tab3:
        st.subheader("Analyse Exploratoire 2020-2025")
        
        if df_2020 is None:
            st.warning("Dataset 2020-2025 non disponible.")
        else:
            st.warning("""
            **Description** : Cette période est marquée par une **extrême volatilité** :
            - 📉 **2020** : Chute des prix (COVID-19, baisse de la demande)
            - 📈 **2022** : Explosion des prix (crise gazière, tensions géopolitiques)
            - 🔄 **2023-2024** : Retour progressif à la normale
            """)
            
            price_col = 'price_day_ahead' if 'price_day_ahead' in df_2020.columns else None
            if price_col is None:
                st.error("Colonne de prix introuvable.")
                return
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Distribution avec queue épaisse
                fig_dist = px.histogram(df_2020, x=price_col, nbins=100,
                                        title="Distribution du Prix (2020-2025)")
                fig_dist.update_traces(marker_color='#EF553B')
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption("📝 Distribution asymétrique avec queue épaisse à droite (pics 2022).")
            
            with col_b:
                # Box par année
                df_year = df_2020.copy()
                df_year['year'] = df_year.index.year
                fig_box_year = px.box(df_year, x='year', y=price_col,
                                      title="Distribution par Année")
                fig_box_year.update_traces(marker_color='#EF553B')
                st.plotly_chart(fig_box_year, use_container_width=True)
                st.caption("📝 2022 = année exceptionnelle avec des prix > 500 €/MWh.")
            
            # Évolution temporelle
            st.markdown("#### Évolution du Prix dans le Temps")
            # Convert to numeric to avoid object dtype error
            price_series_2020 = pd.to_numeric(df_2020[price_col], errors='coerce')
            daily_mean_2020 = price_series_2020.resample('D').mean()
            fig_line = px.line(daily_mean_2020, title="Prix Journalier Moyen (2020-2025)")
            fig_line.update_traces(line_color='#EF553B')
            st.plotly_chart(fig_line, use_container_width=True)
            st.caption("📝 Pic de crise énergétique visible mi-2022, suivi d'une normalisation progressive.")
            
            # Mix énergétique (si colonnes disponibles)
            if 'nuclear' in df_2020.columns and 'solar' in df_2020.columns:
                st.markdown("#### Mix Énergétique (Moyennes Mensuelles)")
                energy_cols = ['nuclear', 'solar']
                if 'wind' in df_2020.columns:
                    energy_cols.append('wind')
                # Convert all to numeric
                df_energy = df_2020[energy_cols].apply(pd.to_numeric, errors='coerce')
                df_mix = df_energy.resample('M').mean()
                fig_mix = px.area(df_mix, title="Évolution du Mix Énergétique")
                st.plotly_chart(fig_mix, use_container_width=True)
                st.caption("📝 Le nucléaire reste dominant, les renouvelables progressent.")
    
    # ========== TAB 4: Performance Modèles ==========
    with tab4:
        st.subheader("Performance des Modèles Prédictifs")
        
        st.markdown("""
        Nous avons testé deux approches de modélisation :
        1. **LightGBM** : Modèle Gradient Boosting, très performant pour les relations non-linéaires.
        2. **ARIMAX** : Modèle statistique classique avec variables exogènes (order=(1,1,1), seasonal=(0,0,0,0)).
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Résultats 2015-2017")
            st.caption("*Source: France_2015_2017_ML1.ipynb & ML_Optimisé.ipynb*")
            perf_2015 = {
                "Métrique": ["MAE", "RMSE", "R²"],
                "LightGBM (Base)": ["0.41", "1.21", "0.995"],
                "LightGBM (Optimisé)": ["0.16", "0.28", "1.00"],
            }
            st.table(pd.DataFrame(perf_2015))
            st.success("✅ Optimisation : MAE réduite de 61% (0.41 → 0.16), R² parfait à 1.00")
        
        with col2:
            st.markdown("### 📊 Résultats 2020-2025")
            st.caption("*Source: France_2020_2025_Modeling.ipynb*")
            perf_2020 = {
                "Métrique": ["MAE", "RMSE", "R²"],
                "LightGBM (Base)": ["0.85", "2.21", "0.997"],
                "LightGBM (Optimisé)": ["0.61", "1.86", "0.998"],
                "ARIMAX": ["28.74", "34.91", "0.453"],
            }
            st.table(pd.DataFrame(perf_2020))
            st.success("✅ LightGBM Optimisé : MAE 0.61, R² 0.998 — Excellente performance !")
        
        st.markdown("---")
        st.info("""
        **Insight clé** : 
        - **2015-2017** : Marché prévisible → LightGBM atteint R² = 0.85 avec MAE < 4€.
        - **2020-2025** : La crise énergétique de 2022 crée un "distribution shift". 
          Même LightGBM peine (MAE ~18€) car le régime de prix a radicalement changé.
        - **ARIMAX** : Modèle linéaire inadapté aux multi-régimes (R² négatif = pire qu'une moyenne).
        """)
    
    # ========== TAB 5: SHAP ==========
    with tab5:
        st.subheader("Interprétabilité (SHAP)")
        
        st.markdown("""
        L'analyse SHAP permet de comprendre **pourquoi** le modèle fait une prédiction donnée.
        Voici les principaux leviers identifiés :
        """)
        
        st.markdown("### 🔑 Top Features (2015-2017)")
        features_2015 = {
            "Feature": ["price_lag_1h", "hour", "load_actual", "solar_generation", "price_lag_24h"],
            "Impact": ["+++", "++", "++", "+", "+"],
            "Explication": [
                "Le prix de l'heure précédente est le meilleur prédicteur.",
                "L'heure de la journée influence la demande.",
                "La charge réelle reflète la demande instantanée.",
                "Plus de solaire = prix plus bas (effet merit-order).",
                "Le prix d'il y a 24h capture les cycles journaliers."
            ]
        }
        st.table(pd.DataFrame(features_2015))
        
        st.markdown("### 🔑 Top Features (2020-2025)")
        features_2020 = {
            "Feature": ["gas", "load", "nuclear", "wind", "solar"],
            "Impact": ["+++", "++", "++", "+", "+"],
            "Explication": [
                "Le prix du gaz drive les prix électriques (centrales à gaz marginales).",
                "La demande reste un facteur clé.",
                "Le nucléaire, production de base, influence la stabilité.",
                "L'éolien contribue à la baisse des prix.",
                "Le solaire aussi, mais avec une saisonnalité forte."
            ]
        }
        st.table(pd.DataFrame(features_2020))
        
        st.info("""
        💡 **Insight** : En 2015-2017, les lags de prix dominent (marché prévisible). 
        En 2020-2025, les fondamentaux (gaz, nucléaire) prennent le dessus car le marché 
        est plus réactif aux conditions de production.
        """)
