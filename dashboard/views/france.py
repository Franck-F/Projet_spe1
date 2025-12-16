import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from utils.data_loader import load_france_data


def render_france(df_orig):
    """
    Renders the comprehensive France dashboard with EDA, modeling, and SHAP analysis.
    """
    st.header("🇫🇷 France : Analyse Complète du Prix de l'Électricité")
    
    # --- Load Processed Datasets ---
    with st.spinner("Chargement des datasets France..."):
        datasets = load_france_data()
    
    if not datasets:
        st.error("Aucun dataset France chargé. Vérifiez les fichiers CSV dans data/processed/.")
        return
    
    df_2015 = datasets.get('2015_2017')
    df_2020 = datasets.get('2020_2025')
    
    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Vue d'Ensemble",
        "📈 Analyse EDA",
        "⚡ Mix Énergétique",
        "🔗 Corrélations",
        "🤖 Performance Modèles",
        "🔍 SHAP Analysis"
    ])
    
    # ========== TAB 1: Vue d'Ensemble ==========
    with tab1:
        render_overview_tab(df_2015, df_2020)
    
    # ========== TAB 2: Analyse EDA ==========
    with tab2:
        render_eda_tab(df_2020)
    
    # ========== TAB 3: Mix Énergétique ==========
    with tab3:
        render_energy_mix_tab(df_2020)
    
    # ========== TAB 4: Corrélations ==========
    with tab4:
        render_correlations_tab(df_2020)
    
    # ========== TAB 5: Performance Modèles ==========
    with tab5:
        render_models_tab(df_2015, df_2020)
    
    # ========== TAB 6: SHAP Analysis ==========
    with tab6:
        render_shap_tab()


def render_overview_tab(df_2015, df_2020):
    """Tab 1: Vue d'Ensemble"""
    st.subheader("Résumé Comparatif des Périodes")
    
    st.markdown("""
    Ce dashboard présente l'analyse prédictive du prix de l'électricité en France 
    sur deux périodes distinctes aux caractéristiques très différentes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Période 2015-2017")
        if df_2015 is not None and not df_2015.empty:
            st.metric("Observations", f"{len(df_2015):,}")
            if 'price_day_ahead' in df_2015.columns:
                price_mean = df_2015['price_day_ahead'].mean()
                price_std = df_2015['price_day_ahead'].std()
                st.metric("Prix Moyen", f"{price_mean:.2f} €/MWh")
                st.metric("Écart-Type", f"{price_std:.2f} €")
        st.info("**Période stable** : Marché prévisible, peu de volatilité. Idéal pour l'entraînement de modèles.")
    
    with col2:
        st.markdown("### 📌 Période 2020-2025")
        if df_2020 is not None and not df_2020.empty:
            st.metric("Observations", f"{len(df_2020):,}")
            if 'price_day_ahead' in df_2020.columns:
                price_mean = df_2020['price_day_ahead'].mean()
                price_std = df_2020['price_day_ahead'].std()
                st.metric("Prix Moyen", f"{price_mean:.2f} €/MWh")
                st.metric("Écart-Type", f"{price_std:.2f} €")
        st.warning("**Période volatile** : Crise COVID-19, crise énergétique 2022, prix extrêmes. Données complexes.")
    
    st.markdown("---")
    st.markdown("### 🏆 Performance des Modèles (Résumé)")
    
    perf_data = {
        "Période": ["2015-2017", "2020-2025"],
        "LightGBM MAE (Optimisé)": ["0.16 €/MWh", "0.61 €/MWh"],
        "LightGBM R²": ["1.00", "0.998"],
        "ARIMAX MAE": ["-", "28.74 €/MWh"],
        "ARIMAX R²": ["-", "0.453"],
    }
    st.table(pd.DataFrame(perf_data))


def render_eda_tab(df_2020):
    """Tab 2: Analyse EDA Détaillée"""
    st.subheader("Analyse Exploratoire Détaillée (2020-2025)")
    
    if df_2020 is None or df_2020.empty:
        st.warning("Dataset 2020-2025 non disponible.")
        return
    
    price_col = 'price_day_ahead'
    if price_col not in df_2020.columns:
        st.error("Colonne 'price_day_ahead' introuvable.")
        return
    
    # Section 1: Distribution du Prix
    st.markdown("### 📊 Distribution du Prix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme
        fig_hist = px.histogram(df_2020, x=price_col, nbins=100,
                               title="Distribution du Prix (2020-2025)")
        fig_hist.update_traces(marker_color='#EF553B')
        fig_hist.update_layout(xaxis_title='Prix (€/MWh)', yaxis_title='Fréquence')
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("📝 Distribution asymétrique avec queue épaisse à droite (pics 2022).")
    
    with col2:
        # Boxplot par année
        df_year = df_2020.copy()
        df_year['year'] = df_year.index.year
        fig_box = px.box(df_year, x='year', y=price_col,
                        title="Distribution par Année")
        fig_box.update_traces(marker_color='#EF553B')
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("📝 2022 = année exceptionnelle avec des prix > 500 €/MWh.")
    
    # Section 1.5: Détection des Outliers
    st.markdown("### 🔍 Détection des Outliers")
    
    # Charger le dataset raw pour l'analyse des outliers
    try:
        import os
        raw_data_path = os.path.join('..', 'data', 'raw', 'time_series_60min_fr_dk_2020_2025.csv')
        if not os.path.exists(raw_data_path):
            raw_data_path = os.path.join('data', 'raw', 'time_series_60min_fr_dk_2020_2025.csv')
        
        df_raw = pd.read_csv(raw_data_path, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
        
        # Filtrer pour la France uniquement
        if 'cet_cest_timestamp' in df_raw.columns:
            df_france_raw = df_raw[df_raw['cet_cest_timestamp'].notna()].copy()
        else:
            df_france_raw = df_raw.copy()
        
        # Utiliser FR_price_day_ahead (colonne du fichier raw)
        if 'FR_price_day_ahead' in df_france_raw.columns:
            price_col_raw = 'FR_price_day_ahead'
        elif 'Price_day_ahead' in df_france_raw.columns:
            price_col_raw = 'Price_day_ahead'
        elif 'price_day_ahead_fr' in df_france_raw.columns:
            price_col_raw = 'price_day_ahead_fr'
        elif 'price_day_ahead' in df_france_raw.columns:
            price_col_raw = 'price_day_ahead'
        else:
            price_col_raw = None


        
        if price_col_raw is not None:
            # Calcul des outliers (méthode percentile)
            threshold_p95 = df_france_raw[price_col_raw].quantile(0.95)
            
            # Identifier les outliers
            df_outliers = df_france_raw[df_france_raw[price_col_raw] > threshold_p95].copy()
            n_outliers = len(df_outliers)
            pct_outliers = (n_outliers / len(df_france_raw)) * 100
            
            # Métriques
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Seuil 95e percentile", f"{threshold_p95:.1f} €/MWh")
            with col_b:
                st.metric("Nombre d'outliers", f"{n_outliers:,}")
            with col_c:
                st.metric("% du dataset", f"{pct_outliers:.2f}%")
            with col_d:
                st.metric("Prix max", f"{df_france_raw[price_col_raw].max():.1f} €/MWh")
            
            # Graphique des outliers
            df_plot = df_france_raw.copy()
            df_plot['is_outlier'] = df_plot[price_col_raw] > threshold_p95
            df_plot['year'] = df_plot.index.year
            df_plot['month'] = df_plot.index.month
            
            fig_outliers = go.Figure()
            
            # Points normaux (échantillonnés pour performance)
            df_normal = df_plot[~df_plot['is_outlier']].iloc[::10]  # 1 point sur 10
            fig_outliers.add_trace(go.Scatter(
                x=df_normal.index,
                y=df_normal[price_col_raw],
                mode='markers',
                name='Normal',
                marker=dict(size=3, color='lightgray', opacity=0.5),
                hovertemplate='<b>Date:</b> %{x}<br><b>Prix:</b> %{y:.2f} €/MWh<extra></extra>'
            ))
            
            # Outliers (tous affichés)
            fig_outliers.add_trace(go.Scatter(
                x=df_outliers.index,
                y=df_outliers[price_col_raw],
                mode='markers',
                name='Outliers (>P95)',
                marker=dict(size=6, color='red', symbol='diamond'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Prix:</b> %{y:.2f} €/MWh<extra></extra>'
            ))
            
            # Ligne de seuil
            fig_outliers.add_hline(
                y=threshold_p95,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Seuil P95: {threshold_p95:.1f} €/MWh",
                annotation_position="right"
            )
            
            fig_outliers.update_layout(
                title="<b>Détection des Outliers (95e Percentile) - Dataset Raw</b>",
                xaxis_title='Date',
                yaxis_title='Prix (€/MWh)',
                height=500,
                hovermode='closest',
                showlegend=True
            )
            
            st.plotly_chart(fig_outliers, use_container_width=True)
            
            # Analyse temporelle des outliers
            if n_outliers > 0:
                outliers_by_year = df_outliers.groupby(df_outliers.index.year).size()
                
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.markdown("**📅 Répartition des outliers par année**")
                    outliers_summary = pd.DataFrame({
                        'Année': outliers_by_year.index,
                        'Nombre': outliers_by_year.values,
                        '% du total': (outliers_by_year.values / n_outliers * 100).round(1)
                    })
                    st.dataframe(outliers_summary, use_container_width=True)
                
                with col_y:
                    st.markdown("**💡 Insights**")
                    max_year = outliers_by_year.idxmax()
                    max_count = outliers_by_year.max()
                    st.info(f"""
                    - **{max_count}** outliers en **{max_year}** ({max_count/n_outliers*100:.1f}% du total)
                    - Prix moyen des outliers: **{df_outliers[price_col_raw].mean():.1f} €/MWh**
                    - Écart-type: **{df_outliers[price_col_raw].std():.1f} €/MWh**
                    - Crise énergétique 2022 = cause principale
                    """)
            
            st.caption("📝 Les outliers sont définis comme les prix dépassant le 95e percentile du dataset raw. Ils représentent les périodes de tension extrême sur le marché.")
        else:
            st.warning(f"Colonne de prix introuvable dans le dataset raw.")
    
    except Exception as e:
        st.warning(f"Impossible de charger le dataset raw pour l'analyse des outliers: {e}")
        st.info("L'analyse des outliers nécessite le fichier `data/raw/time_series_60min_fr_dk_2020_2025.csv`")



    
    # Section 2: Évolution Temporelle
    st.markdown("### 📈 Évolution Temporelle")
    
    # Prix journalier
    price_series = pd.to_numeric(df_2020[price_col], errors='coerce')
    daily_mean = price_series.resample('D').mean()
    fig_line = px.line(daily_mean, title="Prix Journalier Moyen (2020-2025)")
    fig_line.update_traces(line_color='#EF553B')
    fig_line.update_layout(xaxis_title='Date', yaxis_title='Prix (€/MWh)')
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption("📝 Pic de crise énergétique visible mi-2022, suivi d'une normalisation progressive.")
    
    # Prix moyen annuel avec variation
    if 'year' in df_2020.columns:
        annual_price = df_2020.groupby('year')[price_col].mean().reset_index()
        annual_price['pct_change'] = annual_price[price_col].pct_change() * 100
        
        fig_annual = go.Figure()
        fig_annual.add_trace(go.Bar(
            x=annual_price['year'],
            y=annual_price[price_col],
            marker_color='#1f77b4',
            text=annual_price[price_col].round(2),
            textposition='auto'
        ))
        
        # Annotations de variation
        for i in range(1, len(annual_price)):
            change = annual_price.loc[i, 'pct_change']
            year = annual_price.loc[i, 'year']
            price = annual_price.loc[i, price_col]
            
            color = "red" if change > 0 else "green"
            symbol = "▲" if change > 0 else "▼"
            
            fig_annual.add_annotation(
                x=year, y=price + 5,
                text=f"{symbol} {abs(change):.1f}%",
                showarrow=False,
                font=dict(color=color, size=12)
            )
        
        fig_annual.update_layout(
            title="<b>Évolution du Prix Moyen Annuel (% Variation)</b>",
            xaxis_title='Année',
            yaxis_title='Prix Moyen (€/MWh)',
            height=500
        )
        st.plotly_chart(fig_annual, use_container_width=True)
    
    # Section 3: Saisonnalité
    st.markdown("### 🌍 Saisonnalité")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution par saison
        if 'season_lbl' in df_2020.columns:
            fig_season = px.box(df_2020, x='season_lbl', y=price_col,
                              title='<b>Distribution des Prix par Saison</b>',
                              category_orders={'season_lbl': ['Hiver', 'Printemps', 'Eté', 'Automne']},
                              color='season_lbl',
                              color_discrete_map={
                                  'Hiver': '#1E88E5',
                                  'Printemps': '#43A047',
                                  'Eté': '#FDD835',
                                  'Automne': '#FB8C00'
                              })
            fig_season.update_layout(showlegend=False, xaxis_title="Saison", yaxis_title="Prix (€/MWh)")
            st.plotly_chart(fig_season, use_container_width=True)
    
    with col2:
        # Prix par jour de la semaine
        if 'day_name' in df_2020.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_names_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            weekly = df_2020.groupby('day_name')[price_col].mean().reindex(day_order).reset_index()
            weekly['day_name_fr'] = day_names_fr
            
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Bar(x=weekly['day_name_fr'], y=weekly[price_col], marker_color='#D32F2F'))
            fig_weekly.update_layout(
                title="<b>Prix Moyen par Jour de la Semaine</b>",
                xaxis_title='Jour',
                yaxis_title='Prix Moyen (€/MWh)'
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Profil horaire
    if 'hour' in df_2020.columns and 'is_weekend' in df_2020.columns:
        st.markdown("#### Profil Horaire")
        hourly_week = df_2020[~df_2020['is_weekend']].groupby('hour')[price_col].mean()
        hourly_weekend = df_2020[df_2020['is_weekend']].groupby('hour')[price_col].mean()
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(x=hourly_week.index, y=hourly_week, name='Semaine', line=dict(width=3)))
        fig_hourly.add_trace(go.Scatter(x=hourly_weekend.index, y=hourly_weekend, name='Weekend', line=dict(width=3)))
        fig_hourly.update_layout(title="<b>Profil Horaire</b>", xaxis_title='Heure', yaxis_title='Prix Moyen (€/MWh)')
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Section 4: Prix vs Load
    st.markdown("### ⚡ Prix vs Consommation")
    
    if 'load' in df_2020.columns:
        df_trend = df_2020.copy()
        df_trend['load_bin'] = pd.cut(df_trend['load'], bins=20)
        df_trend_agg = df_trend.groupby('load_bin')[price_col].mean().reset_index()
        df_trend_agg['load_center'] = df_trend_agg['load_bin'].apply(lambda x: x.mid).astype(int)
        
        fig_load = go.Figure()
        fig_load.add_trace(go.Bar(
            x=df_trend_agg['load_center'],
            y=df_trend_agg[price_col],
            marker_color='indianred'
        ))
        fig_load.update_layout(
            title="Tendance : Prix Moyen par Niveau de Consommation",
            xaxis_title='Consommation (MW)',
            yaxis_title='Prix Moyen (€/MWh)'
        )
        st.plotly_chart(fig_load, use_container_width=True)


def render_energy_mix_tab(df_2020):
    """Tab 3: Mix Énergétique"""
    st.subheader("⚡ Mix Énergétique France (2020-2025)")
    
    if df_2020 is None or df_2020.empty:
        st.warning("Dataset 2020-2025 non disponible.")
        return
    
    # Définir les colonnes de génération
    generation_cols = {
        'nuclear': 'Nucléaire',
        'hydro': 'Hydraulique',
        'wind': 'Éolien',
        'solar': 'Solaire',
        'gas': 'Gaz',
        'coal': 'Charbon',
        'biomass': 'Biomasse',
        'oil': 'Fioul'
    }
    
    # Calculer la production totale
    energy_mix = {}
    for col, label in generation_cols.items():
        if col in df_2020.columns:
            total = df_2020[col].sum()
            energy_mix[label] = total
    
    if not energy_mix:
        st.warning("Aucune donnée de génération disponible.")
        return
    
    # Trier par ordre décroissant
    energy_mix = dict(sorted(energy_mix.items(), key=lambda x: x[1], reverse=True))
    
    # Section 1: Répartition Totale
    st.markdown("### 🥧 Répartition de la Production Totale")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=list(energy_mix.values()),
            names=list(energy_mix.keys()),
            title='<b>Mix Énergétique France (2020-2025)</b>',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=14,
            marker=dict(line=dict(color='white', width=2))
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Tableau récapitulatif
        mix_df = pd.DataFrame({
            'Source': list(energy_mix.keys()),
            'Production (MWh)': list(energy_mix.values())
        })
        mix_df['Part (%)'] = (mix_df['Production (MWh)'] / mix_df['Production (MWh)'].sum() * 100).round(2)
        mix_df['Production (TWh)'] = (mix_df['Production (MWh)'] / 1_000_000).round(2)
        
        st.dataframe(mix_df, use_container_width=True)
        st.metric("Production Totale", f"{mix_df['Production (TWh)'].sum():.2f} TWh")
    
    # Section 2: Évolution Temporelle
    st.markdown("### 📈 Évolution Mensuelle du Mix Énergétique")
    
    df_monthly = df_2020.copy()
    df_monthly['year_month'] = df_monthly.index.to_period('M')
    
    monthly_mix = {}
    for col, label in generation_cols.items():
        if col in df_2020.columns:
            monthly_mix[label] = df_monthly.groupby('year_month')[col].sum()
    
    if monthly_mix:
        monthly_df = pd.DataFrame(monthly_mix)
        monthly_df.index = monthly_df.index.to_timestamp()
        
        # Graphique en aires empilées
        fig_area = go.Figure()
        
        colors = {
            'Nucléaire': '#FF6B6B',
            'Hydraulique': '#4ECDC4',
            'Éolien': '#95E1D3',
            'Solaire': '#FFD93D',
            'Gaz': '#F38181',
            'Charbon': '#6C5B7B',
            'Biomasse': '#C8E6C9',
            'Fioul': '#B39DDB'
        }
        
        for source in monthly_df.columns:
            fig_area.add_trace(go.Scatter(
                x=monthly_df.index,
                y=monthly_df[source],
                name=source,
                mode='lines',
                stackgroup='one',
                fillcolor=colors.get(source, '#CCCCCC'),
                line=dict(width=0.5, color=colors.get(source, '#CCCCCC'))
            ))
        
        fig_area.update_layout(
            title='<b>Évolution Mensuelle du Mix Énergétique</b>',
            xaxis_title='Date',
            yaxis_title='Production (MWh)',
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig_area, use_container_width=True)
    
    # Section 3: Prix vs Production Nucléaire
    if 'nuclear' in df_2020.columns and 'price_day_ahead' in df_2020.columns:
        st.markdown("### ☢️ Prix vs Production Nucléaire")
        
        df_nuclear = df_2020.copy()
        df_nuclear['nuclear_bin'] = (df_nuclear['nuclear'] // 2000 * 2000).astype(int)
        df_bar = df_nuclear.groupby('nuclear_bin')['price_day_ahead'].mean().reset_index()
        
        fig_nuclear = px.bar(
            df_bar,
            x='nuclear_bin',
            y='price_day_ahead',
            title="<b>Prix Moyen vs Production Nucléaire</b>",
            labels={'nuclear_bin': 'Production Nucléaire (MW)', 'price_day_ahead': 'Prix Moyen (€/MWh)'}
        )
        st.plotly_chart(fig_nuclear, use_container_width=True)
        st.caption("📝 Plus la production nucléaire est élevée, plus les prix tendent à être bas (production de base stable).")


def render_correlations_tab(df_2020):
    """Tab 4: Corrélations"""
    st.subheader("🔗 Analyse des Corrélations")
    
    if df_2020 is None or df_2020.empty:
        st.warning("Dataset 2020-2025 non disponible.")
        return
    
    # Sélectionner colonnes numériques
    numeric_cols = df_2020.select_dtypes(include=[np.number]).columns
    cols_for_corr = [c for c in numeric_cols if c not in ['year', 'month', 'hour', 'day_of_week', 'day_of_year']]
    
    if len(cols_for_corr) < 2:
        st.warning("Pas assez de colonnes numériques pour calculer les corrélations.")
        return
    
    corr_matrix = df_2020[cols_for_corr].corr()
    
    # Section 1: Heatmap Complète
    st.markdown("### 🌡️ Heatmap de Corrélation Complète")
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',
        title='<b>Heatmap de Corrélation</b>',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=1000, width=1200)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Section 2: Top Corrélations avec le Prix
    if 'price_day_ahead' in corr_matrix.columns:
        st.markdown("### 📊 Top Corrélations avec le Prix")
        
        price_corr = corr_matrix['price_day_ahead'].drop('price_day_ahead').sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Top 10 Corrélations Positives")
            top_pos = price_corr.head(10).reset_index()
            top_pos.columns = ['Feature', 'Corrélation']
            st.dataframe(top_pos, use_container_width=True)
        
        with col2:
            st.markdown("#### ❌ Top 10 Corrélations Négatives")
            top_neg = price_corr.tail(10).reset_index()
            top_neg.columns = ['Feature', 'Corrélation']
            st.dataframe(top_neg, use_container_width=True)
    
    # Section 3: Heatmap Focalisée
    st.markdown("### 🎯 Corrélation Focalisée (Load, Prix, Production)")
    
    corr_vars = ['load', 'price_day_ahead', 'nuclear', 'gas', 'coal', 'hydro', 'oil', 'biomass', 'solar', 'wind']
    corr_vars = [c for c in corr_vars if c in df_2020.columns]
    
    if len(corr_vars) >= 2:
        corr_mx = df_2020[corr_vars].corr()
        
        fig_focus = px.imshow(
            corr_mx,
            text_auto='.2f',
            aspect='auto',
            title='<b>Corrélation : Consommation, Production et Prix</b>',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        fig_focus.update_layout(height=600, width=800)
        st.plotly_chart(fig_focus, use_container_width=True)


def render_models_tab(df_2015, df_2020):
    """Tab 5: Performance des Modèles"""
    st.subheader("🤖 Performance des Modèles Prédictifs")
    
    # Charger les métadonnées des modèles
    from utils.model_loader import get_france_models_info, format_metric
    
    with st.spinner("Chargement des métriques des modèles..."):
        models_info = get_france_models_info()
    
    st.markdown("""
    Nous avons testé deux approches de modélisation :
    1. **LightGBM** : Modèle Gradient Boosting, très performant pour les relations non-linéaires.
    2. **SARIMAX** : Modèle statistique classique avec variables exogènes (agrégation journalière).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Résultats 2015-2017")
        st.caption("*Modèles sauvegardés et chargés depuis models/*")
        
        # Récupérer les métriques réelles
        base_2015 = models_info['2015_2017']['base']
        opt_2015 = models_info['2015_2017']['optimized']
        sarimax_2015 = models_info['2015_2017']['sarimax']
        
        # Construire le tableau avec les vraies métriques
        if base_2015 and 'metrics' in base_2015:
            mae_base = format_metric(base_2015['metrics'].get('MAE'))
            rmse_base = format_metric(base_2015['metrics'].get('RMSE'))
            r2_base = format_metric(base_2015['metrics'].get('R2'), 3)
        else:
            mae_base, rmse_base, r2_base = "N/A", "N/A", "N/A"
        
        if opt_2015 and 'metrics' in opt_2015:
            mae_opt = format_metric(opt_2015['metrics'].get('MAE'))
            rmse_opt = format_metric(opt_2015['metrics'].get('RMSE'))
            r2_opt = format_metric(opt_2015['metrics'].get('R2'), 3)
        else:
            mae_opt, rmse_opt, r2_opt = "N/A", "N/A", "N/A"
        
        if sarimax_2015 and 'metrics' in sarimax_2015:
            mae_sar = format_metric(sarimax_2015['metrics'].get('MAE'))
            rmse_sar = format_metric(sarimax_2015['metrics'].get('RMSE'))
            r2_sar = format_metric(sarimax_2015['metrics'].get('R2'), 3)
        else:
            mae_sar, rmse_sar, r2_sar = "N/A", "N/A", "N/A"
        
        perf_2015 = {
            "Métrique": ["MAE (€/MWh)", "RMSE (€/MWh)", "R²"],
            "LightGBM Base": [mae_base, rmse_base, r2_base],
            "LightGBM Optimisé": [mae_opt, rmse_opt, r2_opt],
            "SARIMAX": [mae_sar, rmse_sar, r2_sar]
        }
        st.table(pd.DataFrame(perf_2015))
        
        if mae_base != "N/A" and mae_opt != "N/A":
            try:
                improvement = ((float(mae_base) - float(mae_opt)) / float(mae_base)) * 100
                st.success(f"✅ Optimisation : MAE réduite de {improvement:.0f}% ({mae_base} → {mae_opt})")
            except:
                st.success(f"✅ LightGBM Optimisé : MAE {mae_opt}, R² {r2_opt}")
    
    with col2:
        st.markdown("### 📊 Résultats 2020-2025")
        st.caption("*Modèles sauvegardés et chargés depuis models/*")
        
        # Récupérer les métriques réelles
        base_2020 = models_info['2020_2025']['base']
        opt_2020 = models_info['2020_2025']['optimized']
        sarimax_2020 = models_info['2020_2025']['sarimax']
        
        # Construire le tableau avec les vraies métriques
        if base_2020:
            mae_base = format_metric(base_2020.get('MAE'))
            rmse_base = format_metric(base_2020.get('RMSE'))
            r2_base = format_metric(base_2020.get('R2'), 3)
        else:
            mae_base, rmse_base, r2_base = "N/A", "N/A", "N/A"
        
        if opt_2020:
            mae_opt = format_metric(opt_2020.get('MAE'))
            rmse_opt = format_metric(opt_2020.get('RMSE'))
            r2_opt = format_metric(opt_2020.get('R2'), 3)
        else:
            mae_opt, rmse_opt, r2_opt = "N/A", "N/A", "N/A"
        
        if sarimax_2020:
            mae_sar = format_metric(sarimax_2020.get('MAE'))
            rmse_sar = format_metric(sarimax_2020.get('RMSE'))
            r2_sar = format_metric(sarimax_2020.get('R2'), 3)
        else:
            mae_sar, rmse_sar, r2_sar = "N/A", "N/A", "N/A"
        
        perf_2020 = {
            "Métrique": ["MAE (€/MWh)", "RMSE (€/MWh)", "R²"],
            "LightGBM Base": [mae_base, rmse_base, r2_base],
            "LightGBM Optimisé": [mae_opt, rmse_opt, r2_opt],
            "SARIMAX": [mae_sar, rmse_sar, r2_sar]
        }
        st.table(pd.DataFrame(perf_2020))
        
        if mae_opt != "N/A" and r2_opt != "N/A":
            st.success(f"✅ LightGBM Optimisé : MAE {mae_opt}, R² {r2_opt} — Excellente performance !")
    
    st.markdown("---")
    st.info("""
    **Insights clés** :
    - **2015-2017** : Marché prévisible → LightGBM atteint des performances exceptionnelles.
    - **2020-2025** : La crise énergétique de 2022 crée un "distribution shift". 
      LightGBM reste performant grâce à l'optimisation des hyperparamètres.
    - **SARIMAX** : Modèle statistique adapté aux données journalières, capture les tendances long terme.
    - **Conclusion** : LightGBM optimisé est le meilleur modèle pour les deux périodes.
    """)
    
    # Hyperparamètres optimaux
    st.markdown("### ⚙️ Hyperparamètres Optimaux")
    
    # Charger les hyperparamètres réels depuis les métadonnées
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### LightGBM 2015-2017")
        if opt_2015 and 'best_params' in opt_2015:
            params_df = pd.DataFrame({
                "Paramètre": list(opt_2015['best_params'].keys()),
                "Valeur": [str(v) for v in opt_2015['best_params'].values()]
            })
            st.dataframe(params_df, use_container_width=True)
        else:
            st.info("Hyperparamètres non disponibles")
    
    with col_b:
        st.markdown("#### LightGBM 2020-2025")
        # Pour 2020-2025, les hyperparamètres sont dans le metadata principal
        metadata_2020 = models_info['2020_2025']
        if metadata_2020:
            # Essayer de charger directement le metadata complet
            from utils.model_loader import load_model_metadata
            full_meta = load_model_metadata('models_france_2020_2025_metadata')
            if full_meta and 'hyperparameters_lightgbm' in full_meta:
                params_df = pd.DataFrame({
                    "Paramètre": list(full_meta['hyperparameters_lightgbm'].keys()),
                    "Valeur": [str(v) for v in full_meta['hyperparameters_lightgbm'].values()]
                })
                st.dataframe(params_df, use_container_width=True)
            else:
                st.info("Hyperparamètres non disponibles")
        else:
            st.info("Métadonnées non disponibles")
    
    # Visualisations des Prédictions
    st.markdown("---")
    st.markdown("### 📈 Visualisations des Prédictions")
    
    # Charger les modèles et générer de vraies prédictions
    from utils.model_loader import load_model
    
    # Graphiques pour 2015-2017
    if df_2015 is not None and not df_2015.empty:
        st.markdown("#### Prédictions 2015-2017")
        
        try:
            # Charger les modèles
            model_base_2015 = load_model('lightgbm_france_2015_2017')
            model_opt_2015 = load_model('lightgbm_france_2015_2017_best_estimator')
            scaler_2015 = load_model('scaler_france_2015_2017')
            
            if model_base_2015 is not None or model_opt_2015 is not None:
                # Préparer les données (derniers 30 jours)
                sample_2015 = df_2015.tail(30 * 24).copy()
                
                if 'price_day_ahead' in sample_2015.columns:
                    y_true = sample_2015['price_day_ahead']
                    
                    # Créer les features temporelles comme dans le CSV original
                    sample_2015['week'] = sample_2015.index.isocalendar().week
                    sample_2015['month'] = sample_2015.index.month
                    sample_2015['dayofweek'] = sample_2015.index.dayofweek  # dayofweek sans underscores
                    sample_2015['hour'] = sample_2015.index.hour
                    
                    # Encoder season si textuelle (comme dans le script ligne 149-150)
                    if 'season' in sample_2015.columns and sample_2015['season'].dtype == 'object':
                        season_encoding = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
                        sample_2015['season'] = sample_2015['season'].map(season_encoding)
                    
                    # ===== MODÈLE BASE: 37 features (Scaler), SANS season =====
                    # Features du Scaler: ['load', ..., 'day_of_week', ..., 'week', 'dayofweek'] (37 total, NO season)
                    exclude_cols_base = ['price_day_ahead', 'day_name', 'season_lbl', 'date', 'utc_timestamp', 'season']
                    feature_cols_base = [c for c in sample_2015.columns if c not in exclude_cols_base]
                    # S'assurer que dayofweek et day_of_week sont présents
                    X_sample_base = sample_2015[feature_cols_base].fillna(0)
                    
                    # ===== MODÈLE OPTIMISÉ: 34 features, AVEC season, Données BRUTES (pas de scaler) =====
                    # Features: [... season ...] (34 total)
                    # Exclure UNIQUEMENT price_rolling_mean_24h, price_rolling_std_24h, week (et colonnes non-features)
                    # ET exclure 'dayofweek' (créé pour le modèle Base, mais pas présent dans le CSV d'entraînement Optimisé qui a 'day_of_week')
                    exclude_cols_opt = ['price_day_ahead', 'day_name', 'season_lbl', 'date', 'utc_timestamp', 
                                       'price_rolling_mean_24h', 'price_rolling_std_24h', 'week', 'dayofweek']
                    feature_cols_opt = [c for c in sample_2015.columns if c not in exclude_cols_opt]
                    X_sample_opt = sample_2015[feature_cols_opt].fillna(0)
                    
                    # Normaliser UNIQUEMENT pour le modèle BASE
                    if scaler_2015 is not None:
                        try:
                            # Vérifier que le nombre de features correspond
                            if X_sample_base.shape[1] == scaler_2015.n_features_in_:
                                X_sample_base_scaled = scaler_2015.transform(X_sample_base)
                            else:
                                st.warning(f"Mismatch Scaler: Attendu {scaler_2015.n_features_in_}, Reçu {X_sample_base.shape[1]}")
                                X_sample_base_scaled = X_sample_base.values
                        except:
                             X_sample_base_scaled = X_sample_base.values
                    else:
                        st.warning("Scaler non disponible pour 2015-2017")
                        X_sample_base_scaled = X_sample_base.values
                    
                    fig_pred_2015 = go.Figure()
                    
                    # Prix réel
                    fig_pred_2015.add_trace(go.Scatter(
                        x=sample_2015.index,
                        y=y_true,
                        mode='lines',
                        name='Prix Réel',
                        line=dict(color='#FFFFFF', width=2)
                    ))
                    
                    # Prédiction LightGBM Base
                    if model_base_2015 is not None:
                        try:
                            y_pred_base = model_base_2015.predict(X_sample_base_scaled)
                            mae_base_viz = np.mean(np.abs(y_true - y_pred_base))
                            
                            fig_pred_2015.add_trace(go.Scatter(
                                x=sample_2015.index,
                                y=y_pred_base,
                                mode='lines',
                                name=f'LightGBM Base (MAE: {mae_base_viz:.2f})',
                                line=dict(color='#FFB74D', width=1.5, dash='dash'),
                                opacity=0.9
                            ))
                        except Exception as e:
                            st.warning(f"Erreur prédiction base: {str(e)}")
                    
                    # Prédiction LightGBM Optimisé (Données BRUTES)
                    if model_opt_2015 is not None:
                        try:
                            # Utiliser X_sample_opt directement (PAS DE SCALING)
                            y_pred_opt = model_opt_2015.predict(X_sample_opt)
                            mae_opt_viz = np.mean(np.abs(y_true - y_pred_opt))
                            
                            fig_pred_2015.add_trace(go.Scatter(
                                x=sample_2015.index,
                                y=y_pred_opt,
                                mode='lines',
                                name=f'LightGBM Optimisé (MAE: {mae_opt_viz:.2f})',
                                line=dict(color='#81C784', width=2),
                                opacity=0.9
                            ))
                        except Exception as e:
                            st.warning(f"Erreur prédiction optimisé: {str(e)}")
                    
                    fig_pred_2015.update_layout(
                        title="<b>Prédictions Réelles - 2015-2017 (30 derniers jours)</b>",
                        xaxis_title='Date',
                        yaxis_title='Prix (€/MWh)',
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_pred_2015, use_container_width=True)
                    
                    # Informations sur les modèles
                    st.caption(f"""
                    **LightGBM Base 2015-2017**: Modèle baseline entraîné sur {X_sample_base.shape[1]} features (sans season encodée). 
                    Période d'entraînement: 80% des données 2015-2017. Normalisation StandardScaler appliquée.
                    
                    **LightGBM Optimisé 2015-2017**: Modèle optimisé par GridSearchCV sur {X_sample_opt.shape[1]} features (avec season encodée, sans week/rolling_24h). 
                    Hyperparamètres: learning_rate, num_leaves, max_depth, n_estimators optimisés pour minimiser la MAE.
                    """)
            else:
                st.info("Modèles 2015-2017 non disponibles. Vérifiez que les fichiers .pkl sont dans models/France_models/")
        
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles 2015-2017: {e}")
    
    # Graphiques pour 2020-2025
    if df_2020 is not None and not df_2020.empty:
        st.markdown("#### Prédictions 2020-2025")
        
        try:
            # Charger les modèles
            model_base_2020 = load_model('lightgbm_france_2020_2025_base')
            model_opt_2020 = load_model('lightgbm_france_2020_2025_optimized')
            
            if model_base_2020 is not None or model_opt_2020 is not None:
                # Préparer les données (derniers 60 jours)
                sample_2020 = df_2020.tail(60 * 24).copy()
                
                if 'price_day_ahead' in sample_2020.columns:
                    y_true = sample_2020['price_day_ahead']
                    
                    fig_pred_2020 = go.Figure()
                    
                    # Prix réel
                    fig_pred_2020.add_trace(go.Scatter(
                        x=sample_2020.index,
                        y=y_true,
                        mode='lines',
                        name='Prix Réel',
                        line=dict(color='#FFFFFF', width=2)
                    ))
                    
                    # Prédiction LightGBM Base (11 features spécifiques)
                    if model_base_2020 is not None:
                        try:
                            # Features exactes du modèle base
                            features_base = ['gas', 'coal', 'nuclear', 'solar', 'wind', 'biomass', 'waste', 'load', 'temperature', 'cloud_cover', 'wind_speed']
                            features_base = [f for f in features_base if f in sample_2020.columns]
                            
                            if len(features_base) == 11:
                                X_sample_base = sample_2020[features_base].fillna(0)
                                y_pred_base = model_base_2020.predict(X_sample_base)
                                mae_base_viz = np.mean(np.abs(y_true - y_pred_base))
                                
                                fig_pred_2020.add_trace(go.Scatter(
                                    x=sample_2020.index,
                                    y=y_pred_base,
                                    mode='lines',
                                    name=f'LightGBM Base (MAE: {mae_base_viz:.2f})',
                                    line=dict(color='#FFB74D', width=1.5, dash='dash'),
                                    opacity=0.9
                                ))
                            else:
                                st.warning(f"Features manquantes pour modèle base. Attendu: 11, Trouvé: {len(features_base)}")
                        except Exception as e:
                            st.warning(f"Erreur prédiction base 2020: {e}")
                    
                    # Prédiction LightGBM Optimisé (65 features engineered)
                    if model_opt_2020 is not None:
                        try:
                            # Utiliser les features exactes attendues par le modèle
                            expected_features = model_opt_2020.feature_name_
                            
                            # Préparer X avec toutes les colonnes disponibles
                            drop_cols_technical = ['day_name', 'season_lbl', 'season', 'price_raw', 'load_bin', 'utc_timestamp', 'date']
                            drop_cols_technical = [c for c in drop_cols_technical if c in sample_2020.columns]
                            drop_cols_leakage = [c for c in sample_2020.columns if 'price_day_ahead' in c and 'lag' not in c and 'rolling' not in c]
                            drop_cols = list(set(drop_cols_technical + drop_cols_leakage))
                            
                            X_all = sample_2020.drop(columns=drop_cols, errors='ignore').fillna(0)
                            
                            # Sélectionner uniquement les features attendues dans le bon ordre
                            available_features = [f for f in expected_features if f in X_all.columns]
                            X_sample_opt = X_all[available_features]
                            
                            if len(available_features) == len(expected_features):
                                y_pred_opt = model_opt_2020.predict(X_sample_opt)
                                mae_opt_viz = np.mean(np.abs(y_true - y_pred_opt))
                                
                                fig_pred_2020.add_trace(go.Scatter(
                                    x=sample_2020.index,
                                    y=y_pred_opt,
                                    mode='lines',
                                    name=f'LightGBM Optimisé (MAE: {mae_opt_viz:.2f})',
                                    line=dict(color='#81C784', width=2),
                                    opacity=0.9
                                ))
                            else:
                                st.warning(f"Features manquantes pour modèle optimisé. Attendu: {len(expected_features)}, Trouvé: {len(available_features)}")
                        except Exception as e:
                            st.warning(f"Erreur prédiction optimisé 2020: {e}")
                    
                    fig_pred_2020.update_layout(
                        title="<b>Prédictions Réelles - 2020-2025 (60 derniers jours)</b>",
                        xaxis_title='Date',
                        yaxis_title='Prix (€/MWh)',
                        height=400,
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_pred_2020, use_container_width=True)
                    
                    # Informations sur les modèles
                    st.caption("""
                    **LightGBM Base 2020-2025**: Modèle baseline entraîné sur 11 features brutes (production énergétique, consommation, météo). 
                    Paramètres par défaut, split temporel 80/20. Adapté pour capturer les relations de base entre production et prix.
                    
                    **LightGBM Optimisé 2020-2025**: Modèle avancé avec 65 features engineered incluant lags temporels, rolling windows, 
                    features dérivées et interactions. Optimisé par GridSearchCV pour gérer la volatilité de la crise énergétique 2022.
                    Hyperparamètres: learning_rate=0.05, num_leaves=100, max_depth=10, n_estimators=500.
                    """)
            else:
                st.info("Modèles 2020-2025 non disponibles. Vérifiez que les fichiers .pkl sont dans models/France_models/")
        
        except Exception as e:
            st.error(f"Erreur lors du chargement des modèles 2020-2025: {e}")
    
    st.info("""
    **Note**: Les prédictions affichées utilisent les modèles sauvegardés chargés depuis models/France_models/. 
    Les MAE (Mean Absolute Error) indiquées dans les légendes correspondent aux erreurs réelles calculées sur la période visualisée.
    Les modèles ont été entraînés sur des splits temporels (80% train, 20% test) pour respecter la nature séquentielle des données.
    """)






def render_shap_tab():
    """Tab 6: SHAP Analysis"""
    st.subheader("🔍 Interprétabilité (SHAP)")
    
    st.markdown("""
    L'analyse SHAP (SHapley Additive exPlanations) permet de comprendre **pourquoi** 
    le modèle fait une prédiction donnée en attribuant une importance à chaque feature.
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
    est plus réactif aux conditions de production et aux prix des commodités.
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Visualisation SHAP (Exemple)")
    
    st.info("""
    **Note** : Pour afficher les graphiques SHAP interactifs en temps réel, 
    il faudrait charger le modèle LightGBM entraîné et calculer les valeurs SHAP 
    sur un échantillon de données. Cela nécessite le fichier du modèle sauvegardé.
    
    Les tableaux ci-dessus résument les résultats de l'analyse SHAP effectuée 
    dans le notebook `France_2020_2025_Modeling.py`.
    """)
