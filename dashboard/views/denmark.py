import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def render_denmark(df):
    """Dashboard principal pour le Danemark (DK1 + DK2) - Période 2020-2025"""
    
    st.header("Danemark - Analyse du Marché Électrique (2020-2025)")
    st.caption("Analyse comparative des zones DK1 (Ouest) et DK2 (Est)")
    
    # Vérification des données
    dk1_price = 'DK_1_price_day_ahead'
    dk2_price = 'DK_2_price_day_ahead'
    
    if dk1_price not in df.columns or dk2_price not in df.columns:
        st.error(" Colonnes de prix DK1/DK2 introuvables dans le dataset.")
        return
    
    # Filtrer les lignes avec données DK valides (non-NaN)
    df_dk = df[[dk1_price, dk2_price]].dropna()
    
    if df_dk.empty:
        st.warning(" Aucune donnée disponible pour le Danemark.")
        return
    
    # Réindexer le dataframe principal avec les index valides
    df = df.loc[df_dk.index]
    
    st.info(f" **Période analysée (2020-2025)** : {df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')} ({len(df):,} observations)")



    
    # Tabs principaux
    tab_overview, tab_eda, tab_mix, tab_corr, tab_models, tab_shap = st.tabs([
        "📊 Vue d'ensemble",
        "📈 Analyse EDA", 
        "⚡ Mix Énergétique",
        "🔗 Corrélations",
        "🤖 Performance Modèles",
        "🔍 Analyse de la Volatilité"
    ])
    
    with tab_overview:
        render_overview_tab(df)
    
    with tab_eda:
        render_eda_tab(df)
    
    with tab_mix:
        render_energy_mix_tab(df)
    
    with tab_corr:
        render_correlations_tab(df)
    
    with tab_models:
        render_models_tab(df)
    
    with tab_shap:
        render_shap_tab(df)


def render_overview_tab(df):
    """Tab 1: Vue d'ensemble"""
    st.subheader(" Vue d'Ensemble du Marché Danois")
    
    # KPIs Globaux
    st.markdown("### Métriques Clés (2020-2025)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Prix moyens
    dk1_avg_price = df['DK_1_price_day_ahead'].mean()
    dk2_avg_price = df['DK_2_price_day_ahead'].mean()
    
    # Charge totale
    dk1_load_col = 'DK_1_load_actual_entsoe_transparency'
    dk2_load_col = 'DK_2_load_actual_entsoe_transparency'
    
    dk1_avg_load = df[dk1_load_col].mean() if dk1_load_col in df.columns else 0
    dk2_avg_load = df[dk2_load_col].mean() if dk2_load_col in df.columns else 0
    
    with col1:
        st.metric("Prix Moyen DK1", f"{dk1_avg_price:.2f} €/MWh")
    with col2:
        st.metric("Prix Moyen DK2", f"{dk2_avg_price:.2f} €/MWh")
    with col3:
        st.metric("Charge Moy. DK1", f"{dk1_avg_load:,.0f} MW")
    with col4:
        st.metric("Charge Moy. DK2", f"{dk2_avg_load:,.0f} MW")
    
    st.markdown("---")
    
    # Tableau comparatif DK1 vs DK2
    st.markdown("###  Comparaison DK1 (Ouest) vs DK2 (Est)")
    
    comparison_data = {
        "Métrique": [
            "Prix Moyen (€/MWh)",
            "Prix Min (€/MWh)",
            "Prix Max (€/MWh)",
            "Écart-type Prix",
            "Charge Moyenne (MW)",
            "Observations"
        ],
        "DK1 (Ouest)": [
            f"{df['DK_1_price_day_ahead'].mean():.2f}",
            f"{df['DK_1_price_day_ahead'].min():.2f}",
            f"{df['DK_1_price_day_ahead'].max():.2f}",
            f"{df['DK_1_price_day_ahead'].std():.2f}",
            f"{dk1_avg_load:,.0f}" if dk1_load_col in df.columns else "N/A",
            f"{len(df):,}"
        ],
        "DK2 (Est)": [
            f"{df['DK_2_price_day_ahead'].mean():.2f}",
            f"{df['DK_2_price_day_ahead'].min():.2f}",
            f"{df['DK_2_price_day_ahead'].max():.2f}",
            f"{df['DK_2_price_day_ahead'].std():.2f}",
            f"{dk2_avg_load:,.0f}" if dk2_load_col in df.columns else "N/A",
            f"{len(df):,}"
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    st.info("""
     **Insights Clés** :
    - **DK1 (Ouest - Jylland)** : Zone connectée à l'Allemagne, dominée par l'éolien, plus volatile
    - **DK2 (Est - Copenhague)** : Zone connectée à la Suède, plus urbaine, légèrement plus stable
    - Les deux zones présentent des profils de prix similaires mais avec des écarts ponctuels dus aux congestions de réseau
    """)


def render_eda_tab(df):
    """Tab 2: Analyse EDA"""
    st.subheader(" Analyse Exploratoire des Données")
    
    # Section 1: Distribution des Prix
    st.markdown("###  Distribution des Prix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogramme DK1
        fig_hist_dk1 = px.histogram(
            df, x='DK_1_price_day_ahead', nbins=100,
            title="Distribution Prix DK1 (Ouest)",
            labels={'DK_1_price_day_ahead': 'Prix (€/MWh)'}
        )
        fig_hist_dk1.update_traces(marker_color='#E31A1C')
        fig_hist_dk1.update_layout(template='plotly_dark')
        st.plotly_chart(fig_hist_dk1, use_container_width=True)
        st.caption("DK1 présente une distribution avec queue épaisse à droite et quelques prix négatifs (surproduction éolienne).")
    
    with col2:
        # Histogramme DK2
        fig_hist_dk2 = px.histogram(
            df, x='DK_2_price_day_ahead', nbins=100,
            title="Distribution Prix DK2 (Est)",
            labels={'DK_2_price_day_ahead': 'Prix (€/MWh)'}
        )
        fig_hist_dk2.update_traces(marker_color='#1F78B4')
        fig_hist_dk2.update_layout(template='plotly_dark')
        st.plotly_chart(fig_hist_dk2, use_container_width=True)
        st.caption(" DK2 montre une distribution similaire mais légèrement plus concentrée autour de la médiane.")
    
    # Section 2: Évolution Temporelle
    st.markdown("###  Évolution Temporelle")

    # Prix hebdomadaire
    df_weekly = df[['DK_1_price_day_ahead', 'DK_2_price_day_ahead']].resample('W').mean()
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=df_weekly.index, y=df_weekly['DK_1_price_day_ahead'],
        mode='lines', name='DK1 (Ouest)',
        line=dict(color='#E31A1C', width=2)
    ))
    fig_time.add_trace(go.Scatter(
        x=df_weekly.index, y=df_weekly['DK_2_price_day_ahead'],
        mode='lines', name='DK2 (Est)',
        line=dict(color='#1F78B4', width=2)
    ))
    fig_time.update_layout(
        title="<b>Évolution Hebdomadaire des Prix</b>",
        xaxis_title='Date',
        yaxis_title='Prix Moyen (€/MWh)',
        template='plotly_dark',
        height=500
    )
    st.plotly_chart(fig_time, use_container_width=True)
    st.info(" **Interprétation** : On observe un choc majeur en 2022 (crise énergétique européenne) avec des pics dépassant 500 €/MWh. Les deux zones suivent des trajectoires très similaires, confirmant l'intégration du marché danois.")
    
    # Section 3: Saisonnalité
    st.markdown("###  Saisonnalité")
    
    # Ajouter colonnes temporelles si nécessaire
    if 'month' not in df.columns:
        df['month'] = df.index.month
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df.index.dayofweek
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prix moyen par mois
        monthly_dk1 = df.groupby('month')['DK_1_price_day_ahead'].mean()
        monthly_dk2 = df.groupby('month')['DK_2_price_day_ahead'].mean()
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_dk1.index, y=monthly_dk1.values,
            name='DK1', marker_color='#E31A1C'
        ))
        fig_monthly.add_trace(go.Bar(
            x=monthly_dk2.index, y=monthly_dk2.values,
            name='DK2', marker_color='#1F78B4'
        ))
        fig_monthly.update_layout(
            title="<b>Prix Moyen par Mois</b>",
            xaxis_title='Mois',
            yaxis_title='Prix Moyen (€/MWh)',
            template='plotly_dark',
            barmode='group'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.caption(" Saisonnalité marquée : prix plus élevés en hiver (demande de chauffage) et en été (maintenance).")
    
    with col2:
        # Prix par jour de la semaine
        weekly_dk1 = df.groupby('day_of_week')['DK_1_price_day_ahead'].mean()
        weekly_dk2 = df.groupby('day_of_week')['DK_2_price_day_ahead'].mean()
        
        day_names = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
        
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(
            x=day_names, y=weekly_dk1.values,
            mode='lines+markers', name='DK1',
            line=dict(color='#E31A1C', width=3)
        ))
        fig_weekly.add_trace(go.Scatter(
            x=day_names, y=weekly_dk2.values,
            mode='lines+markers', name='DK2',
            line=dict(color='#1F78B4', width=3)
        ))
        fig_weekly.update_layout(
            title="<b>Prix Moyen par Jour de la Semaine</b>",
            xaxis_title='Jour',
            yaxis_title='Prix Moyen (€/MWh)',
            template='plotly_dark'
        )
        st.plotly_chart(fig_weekly, use_container_width=True)
        st.caption("Baisse notable le week-end (réduction activité industrielle).")
    
    # Section 4: Détection des Outliers
    st.markdown("###  Détection des Outliers")
    
    # Calcul du 95e percentile pour DK1 et DK2
    threshold_dk1 = df['DK_1_price_day_ahead'].quantile(0.95)
    threshold_dk2 = df['DK_2_price_day_ahead'].quantile(0.95)
    
    outliers_dk1 = df[df['DK_1_price_day_ahead'] > threshold_dk1]
    outliers_dk2 = df[df['DK_2_price_day_ahead'] > threshold_dk2]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Seuil P95 DK1", f"{threshold_dk1:.1f} €/MWh")
    with col2:
        st.metric("Outliers DK1", f"{len(outliers_dk1):,}")
    with col3:
        st.metric("Seuil P95 DK2", f"{threshold_dk2:.1f} €/MWh")
    with col4:
        st.metric("Outliers DK2", f"{len(outliers_dk2):,}")
    
    # Graphique outliers
    fig_outliers = go.Figure()
    
    # Points normaux (échantillonnés)
    df_normal_dk1 = df[df['DK_1_price_day_ahead'] <= threshold_dk1].iloc[::10]
    df_normal_dk2 = df[df['DK_2_price_day_ahead'] <= threshold_dk2].iloc[::10]
    
    fig_outliers.add_trace(go.Scatter(
        x=df_normal_dk1.index, y=df_normal_dk1['DK_1_price_day_ahead'],
        mode='markers', name='Normal DK1',
        marker=dict(size=3, color='lightcoral', opacity=0.3)
    ))
    
    fig_outliers.add_trace(go.Scatter(
        x=df_normal_dk2.index, y=df_normal_dk2['DK_2_price_day_ahead'],
        mode='markers', name='Normal DK2',
        marker=dict(size=3, color='lightblue', opacity=0.3)
    ))
    
    # Outliers
    fig_outliers.add_trace(go.Scatter(
        x=outliers_dk1.index, y=outliers_dk1['DK_1_price_day_ahead'],
        mode='markers', name='Outliers DK1',
        marker=dict(size=6, color='#E31A1C', symbol='diamond')
    ))
    
    fig_outliers.add_trace(go.Scatter(
        x=outliers_dk2.index, y=outliers_dk2['DK_2_price_day_ahead'],
        mode='markers', name='Outliers DK2',
        marker=dict(size=6, color='#1F78B4', symbol='diamond')
    ))
    
    fig_outliers.update_layout(
        title="<b>Détection des Outliers (95e Percentile)</b>",
        xaxis_title='Date',
        yaxis_title='Prix (€/MWh)',
        height=500,
        template='plotly_dark'
    )
    st.plotly_chart(fig_outliers, use_container_width=True)
    st.info("**Interprétation** : Les outliers (au-delà du 95e percentile) correspondent à des périodes de forte tension sur le marché danois. **Important** : les outliers sont calculés sur l'ensemble de la période 2020-2025, donc ils reflètent les valeurs extrêmes par rapport à la tendance générale. La majorité des outliers sont concentrés en **2022 lors de la crise énergétique européenne** (guerre en Ukraine, flambée des prix du gaz). Les deux zones DK1 et DK2 présentent des profils d'outliers similaires, confirmant l'intégration du marché danois malgré les différences structurelles entre l'Ouest (éolien) et l'Est (urbain).")



def render_energy_mix_tab(df):
    """Tab 3: Mix Énergétique"""
    st.subheader(" Mix Énergétique Danois")
    
    st.info(" **Le Danemark est un champion mondial de l'éolien** avec une pénétration très élevée des énergies renouvelables, notamment l'éolien offshore et onshore.")
    
    # Section 1: Répartition de la Production (Pie Charts DK1 vs DK2)
    st.markdown("###  Répartition de la Production Électrique")
    
    # Identifier les colonnes de production
    dk1_prod_cols = [c for c in df.columns if 'DK1_' in c and 'Actual Aggregated' in c]
    dk2_prod_cols = [c for c in df.columns if 'DK2_' in c and 'Actual Aggregated' in c]
    
    if dk1_prod_cols or dk2_prod_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            if dk1_prod_cols:
                # Calculer production totale par source pour DK1
                dk1_mix = {}
                for col in dk1_prod_cols:
                    # Extraire le nom de la source (ex: "Wind Onshore" de "DK1_('Wind Onshore', 'Actual Aggregated')")
                    source_name = col.split("'")[1] if "'" in col else col.replace('DK1_', '').replace('Actual Aggregated', '').strip()
                    dk1_mix[source_name] = df[col].sum()
                
                # Filtrer les sources avec production > 0
                dk1_mix = {k: v for k, v in dk1_mix.items() if v > 0}
                
                if dk1_mix:
                    fig_pie_dk1 = px.pie(
                        values=list(dk1_mix.values()),
                        names=list(dk1_mix.keys()),
                        title='<b>Mix Énergétique DK1 (Ouest)</b>',
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie_dk1.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    fig_pie_dk1.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig_pie_dk1, use_container_width=True)
                    st.caption(" DK1 (Jylland) : Dominance de l'éolien, connecté à l'Allemagne.")
        
        with col2:
            if dk2_prod_cols:
                # Calculer production totale par source pour DK2
                dk2_mix = {}
                for col in dk2_prod_cols:
                    source_name = col.split("'")[1] if "'" in col else col.replace('DK2_', '').replace('Actual Aggregated', '').strip()
                    dk2_mix[source_name] = df[col].sum()
                
                dk2_mix = {k: v for k, v in dk2_mix.items() if v > 0}
                
                if dk2_mix:
                    fig_pie_dk2 = px.pie(
                        values=list(dk2_mix.values()),
                        names=list(dk2_mix.keys()),
                        title='<b>Mix Énergétique DK2 (Est)</b>',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_pie_dk2.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    fig_pie_dk2.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig_pie_dk2, use_container_width=True)
                    st.caption(" DK2 (Copenhague/Sjælland) : Mix plus diversifié, connecté à la Suède.")
    
    # Section 2: Prix Moyen Annuel et Évolution
    st.markdown("###  Prix Moyen Annuel et Évolution")
    
    # Calculer prix moyen annuel
    df_annual_dk1 = df['DK_1_price_day_ahead'].resample('YE').mean()
    df_annual_dk2 = df['DK_2_price_day_ahead'].resample('YE').mean()
    
    # Créer dataframe pour visualisation
    years = df_annual_dk1.index.year
    
    # Calculer variations %
    dk1_pct = df_annual_dk1.pct_change() * 100
    dk2_pct = df_annual_dk2.pct_change() * 100
    
    # Graphique barres groupées
    fig_annual = go.Figure()
    
    fig_annual.add_trace(go.Bar(
        x=years,
        y=df_annual_dk1.values,
        name='DK1 (Ouest)',
        marker_color='#E31A1C',
        text=[f"{val:.1f} €<br>({pct:+.1f}%)" if not pd.isna(pct) else f"{val:.1f} €" 
              for val, pct in zip(df_annual_dk1.values, dk1_pct.values)],
        textposition='inside'
    ))
    
    fig_annual.add_trace(go.Bar(
        x=years,
        y=df_annual_dk2.values,
        name='DK2 (Est)',
        marker_color='#1F78B4',
        text=[f"{val:.1f} €<br>({pct:+.1f}%)" if not pd.isna(pct) else f"{val:.1f} €" 
              for val, pct in zip(df_annual_dk2.values, dk2_pct.values)],
        textposition='inside'
    ))
    
    fig_annual.update_layout(
        title="<b>Prix Moyen Annuel et Évolution (%)</b>",
        xaxis_title='Année',
        yaxis_title='Prix Moyen (€/MWh)',
        barmode='group',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig_annual, use_container_width=True)
    st.info("**Interprétation** : Pic majeur en 2022 lors de la crise énergétique européenne (guerre en Ukraine, flambée des prix du gaz). Les deux zones suivent des trajectoires très similaires, confirmant l'intégration du marché danois.")
    
    # Section 3: Impact du Vent sur les Prix
    st.markdown("###  Impact du Vent sur les Prix")
    
    if 'wind_speed_denmark' in df.columns:
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Scatter DK1
            sample_dk1 = df.sample(min(3000, len(df)))
            fig_wind_price_dk1 = px.scatter(
                sample_dk1,
                x='wind_speed_denmark',
                y='DK_1_price_day_ahead',
                title="<b>DK1 : Vitesse du Vent vs Prix</b>",
                labels={'wind_speed_denmark': 'Vitesse du Vent (m/s)', 'DK_1_price_day_ahead': 'Prix (€/MWh)'},
                opacity=0.4,
                color_discrete_sequence=['#E31A1C']
            )
            fig_wind_price_dk1.update_layout(template='plotly_dark', height=450)
            st.plotly_chart(fig_wind_price_dk1, use_container_width=True)
            st.caption(" DK1 : Corrélation inverse forte - plus de vent = prix plus bas.")
        
        with col_b:
            # Scatter DK2
            sample_dk2 = df.sample(min(3000, len(df)))
            fig_wind_price_dk2 = px.scatter(
                sample_dk2,
                x='wind_speed_denmark',
                y='DK_2_price_day_ahead',
                title="<b>DK2 : Vitesse du Vent vs Prix</b>",
                labels={'wind_speed_denmark': 'Vitesse du Vent (m/s)', 'DK_2_price_day_ahead': 'Prix (€/MWh)'},
                opacity=0.4,
                color_discrete_sequence=['#1F78B4']
            )
            fig_wind_price_dk2.update_layout(template='plotly_dark', height=450)
            st.plotly_chart(fig_wind_price_dk2, use_container_width=True)
            st.caption(" DK2 : Même tendance - le vent est le facteur dominant du marché danois.")
        
        st.info(" **Interprétation** : On observe une **corrélation inverse** entre vitesse du vent et prix dans les deux zones. Lorsque le vent est fort, la production éolienne abondante fait baisser les prix de marché, parfois jusqu'à des valeurs négatives (surproduction). C'est le **\"Roi Vent\"** du marché danois - le facteur le plus influent sur les prix.")
    else:
        st.warning(" Colonne 'wind_speed_denmark' non trouvée dans le dataset.")




def render_correlations_tab(df):
    """Tab 4: Corrélations"""
    st.subheader(" Analyse des Corrélations")
    
    # Section 1: Matrice de Corrélation Prix, Conso & Production
    st.markdown("###  Matrice de Corrélation : Prix, Consommation & Production")
    
    # Identifier les colonnes pertinentes
    price_cols = ['DK_1_price_day_ahead', 'DK_2_price_day_ahead']
    load_cols = ['DK_1_load_actual_entsoe_transparency', 'DK_2_load_actual_entsoe_transparency']
    
    # Production columns (chercher les colonnes agrégées)
    prod_cols = [c for c in df.columns if 'DK' in c and 'Actual Aggregated' in c]
    
    # Sélectionner colonnes pour la matrice
    cols_for_corr = price_cols + load_cols
    
    # Ajouter quelques colonnes de production si disponibles
    if prod_cols:
        # Prendre les 4 premières colonnes de production
        cols_for_corr += prod_cols[:4]
    
    # Filtrer les colonnes qui existent réellement
    cols_for_corr = [c for c in cols_for_corr if c in df.columns]
    
    if len(cols_for_corr) >= 2:
        # Calculer la matrice de corrélation
        corr_matrix = df[cols_for_corr].corr()
        
        # Créer des noms plus lisibles
        rename_dict = {
            'DK_1_price_day_ahead': 'Prix DK1',
            'DK_2_price_day_ahead': 'Prix DK2',
            'DK_1_load_actual_entsoe_transparency': 'Conso DK1',
            'DK_2_load_actual_entsoe_transparency': 'Conso DK2'
        }
        
        # Renommer les colonnes de production
        for col in prod_cols[:4]:
            if col in cols_for_corr:
                # Extraire le nom de la source
                source_name = col.split("'")[1] if "'" in col else col.replace('DK1_', '').replace('DK2_', '').replace('Actual Aggregated', '').strip()
                zone = 'DK1' if 'DK1' in col else 'DK2'
                rename_dict[col] = f"{source_name} {zone}"
        
        corr_matrix_renamed = corr_matrix.rename(index=rename_dict, columns=rename_dict)
        
        fig_corr = px.imshow(
            corr_matrix_renamed,
            text_auto='.2f',
            title='<b>Matrice de Corrélation : Prix, Consommation & Production</b>',
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            aspect='auto'
        )
        fig_corr.update_layout(height=700, template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.info("**Interprétation** : Cette heatmap présente les corrélations entre prix, consommation et production pour les deux zones danoises. Les valeurs proches de +1 (rouge) indiquent une forte corrélation positive, -1 (bleu) une forte corrélation négative. On observe généralement une forte corrélation entre les prix DK1 et DK2 (marché intégré).")
    
    # Section 2: Matrices de Facteurs d'Influence (DK1 et DK2)
    st.markdown("###  Matrices de Facteurs d'Influence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### DK1 (Ouest)")
        
        # Variables pour DK1
        dk1_vars = {
            'DK_1_price_day_ahead': 'Prix',
            'DK_1_load_actual_entsoe_transparency': 'Conso',
            'wind_speed_denmark': 'Vent',
            'temperature_denmark': 'Température'
        }
        
        # Filtrer les colonnes qui existent
        dk1_cols = [c for c in dk1_vars.keys() if c in df.columns]
        
        if len(dk1_cols) >= 2:
            corr_dk1 = df[dk1_cols].corr()
            corr_dk1_renamed = corr_dk1.rename(index=dk1_vars, columns=dk1_vars)
            
            fig_dk1 = px.imshow(
                corr_dk1_renamed,
                text_auto='.2f',
                title='<b>Facteurs d\'Influence DK1</b>',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto'
            )
            fig_dk1.update_layout(height=400, template='plotly_dark')
            st.plotly_chart(fig_dk1, use_container_width=True)
            
            st.caption(" DK1 : Zone éolienne, forte corrélation inverse entre vent et prix.")
        else:
            st.warning(" Pas assez de variables disponibles pour DK1.")
    
    with col2:
        st.markdown("#### DK2 (Est)")
        
        # Variables pour DK2
        dk2_vars = {
            'DK_2_price_day_ahead': 'Prix',
            'DK_2_load_actual_entsoe_transparency': 'Conso',
            'wind_speed_denmark': 'Vent',
            'temperature_denmark': 'Température'
        }
        
        # Filtrer les colonnes qui existent
        dk2_cols = [c for c in dk2_vars.keys() if c in df.columns]
        
        if len(dk2_cols) >= 2:
            corr_dk2 = df[dk2_cols].corr()
            corr_dk2_renamed = corr_dk2.rename(index=dk2_vars, columns=dk2_vars)
            
            fig_dk2 = px.imshow(
                corr_dk2_renamed,
                text_auto='.2f',
                title='<b>Facteurs d\'Influence DK2</b>',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                aspect='auto'
            )
            fig_dk2.update_layout(height=400, template='plotly_dark')
            st.plotly_chart(fig_dk2, use_container_width=True)
            
            st.caption(" DK2 : Zone urbaine, profil similaire à DK1 mais légèrement plus stable.")
        else:
            st.warning(" Pas assez de variables disponibles pour DK2.")
    
    st.info("""
     **Insights Clés** :
    - **Vent** : Facteur dominant au Danemark, corrélation inverse forte avec les prix dans les deux zones
    - **Consommation** : Corrélation positive avec les prix (plus de demande = prix plus élevés)
    - **Température** : Impact modéré, moins important qu'en France (chauffage électrique moins répandu)
    - **DK1 vs DK2** : Profils de corrélation très similaires, confirmant l'intégration du marché
    """)


def render_models_tab(df):
    """Tab 5: Performance Modèles"""
    st.subheader(" Performance des Modèles Prédictifs (2020-2025)")
    
    st.markdown("""
    Nous avons entraîné des modèles **LightGBM** pour prédire les prix de l'électricité dans les deux zones danoises :
    - **DK1 (Ouest - Jylland)** : Zone éolienne connectée à l'Allemagne
    - **DK2 (Est - Copenhague)** : Zone urbaine connectée à la Suède
    """)
    
    # Charger les modèles
    from utils.model_loader import load_denmark_model
    
    with st.spinner("Chargement des modèles..."):
        model_dk1_base = load_denmark_model('DK1_baseline')
        model_dk1_opt = load_denmark_model('DK1_optimized')
        model_dk2_base = load_denmark_model('DK2_baseline')
        model_dk2_opt = load_denmark_model('DK2_optimized')
    
    # Vérifier si les modèles sont chargés
    models_loaded = all([model_dk1_base, model_dk1_opt, model_dk2_base, model_dk2_opt])
    
    if not models_loaded:
        st.error(" Impossible de charger tous les modèles. Vérifiez que les fichiers sont présents dans `models/Danemark_models/`.")
        return
    
    st.success(" Modèles chargés avec succès !")
    
    # Section 1: Informations sur les modèles
    st.markdown("###  Modèles Disponibles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🇩🇰 DK1 (Ouest)")
        st.info("""
        **Modèles** :
        - LightGBM Baseline
        - LightGBM Optimisé
        
        **Caractéristiques** :
        - Zone dominée par l'éolien
        - Forte variabilité des prix
        - Connectée à l'Allemagne
        """)
    
    with col2:
        st.markdown("#### 🇩🇰 DK2 (Est)")
        st.info("""
        **Modèles** :
        - LightGBM Baseline
        - LightGBM Optimisé
        
        **Caractéristiques** :
        - Zone plus urbaine
        - Légèrement plus stable
        - Connectée à la Suède
        """)
    
    # Section 2: Features utilisées
    st.markdown("###  Features Utilisées")
    
    if hasattr(model_dk1_opt, 'feature_name_'):
        features = model_dk1_opt.feature_name_
        
        st.caption(f"**Nombre total de features** : {len(features)}")
        
        # Grouper les features par catégorie
        lag_features = [f for f in features if 'lag' in f.lower()]
        rolling_features = [f for f in features if 'rolling' in f.lower()]
        prod_features = [f for f in features if any(x in f.lower() for x in ['wind', 'solar', 'generation'])]
        temporal_features = [f for f in features if any(x in f.lower() for x in ['hour', 'day', 'month', 'season'])]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lags Prix", len(lag_features))
        with col2:
            st.metric("Rolling Stats", len(rolling_features))
        with col3:
            st.metric("Production", len(prod_features))
        with col4:
            st.metric("Temporelles", len(temporal_features))
        
        # Afficher quelques exemples
        with st.expander(" Voir exemples de features"):
            st.markdown("**Lags Prix** (exemples) :")
            st.code(", ".join(lag_features[:5]) if lag_features else "Aucun")
            
            st.markdown("**Rolling Statistics** (exemples) :")
            st.code(", ".join(rolling_features[:5]) if rolling_features else "Aucun")
            
            st.markdown("**Production** (exemples) :")
            st.code(", ".join(prod_features[:5]) if prod_features else "Aucun")
    
    # Section 3: Métriques de Performance (si disponibles dans les modèles)
    st.markdown("###  Métriques de Performance")
    
    st.info("""
     **Note** : Les métriques de performance (MAE, RMSE, R²) ont été calculées lors de l'entraînement.
    Pour voir les métriques détaillées, consultez le script d'entraînement :
    `Analyse Danemark/Analyse DK1 DK2 2020-2025.py`
    """)
    
    # Section 4: Visualisation des Prédictions
    st.markdown("### Visualisation des Prédictions")
    
    st.info(" **Prédictions en Temps Réel** : Les modèles génèrent des prédictions sur les 20% dernières données (période de test).")
    
    try:
        # Préparer les features (comme dans le script d'analyse)
        data = df.copy().sort_index()
        
        # Variables temporelles
        data["hour"] = data.index.hour
        data["dayofweek"] = data.index.dayofweek
        data["month"] = data.index.month
        
        # === DK1 ===
        target_dk1 = "DK_1_price_day_ahead"
        data[f"{target_dk1}_lag1"] = data[target_dk1].shift(1)
        data[f"{target_dk1}_lag24"] = data[target_dk1].shift(24)
        data[f"{target_dk1}_roll24"] = data[target_dk1].rolling(24).mean()
        
        features_dk1 = [
            "DK_1_load_actual_entsoe_transparency",
            "wind_speed_denmark",
            "temperature_denmark",
            "hour", "dayofweek", "month",
            f"{target_dk1}_lag1",
            f"{target_dk1}_lag24",
            f"{target_dk1}_roll24"
        ]
        
        # === DK2 ===
        target_dk2 = "DK_2_price_day_ahead"
        data[f"{target_dk2}_lag1"] = data[target_dk2].shift(1)
        data[f"{target_dk2}_lag24"] = data[target_dk2].shift(24)
        data[f"{target_dk2}_roll24"] = data[target_dk2].rolling(24).mean()
        
        features_dk2 = [
            "DK_2_load_actual_entsoe_transparency",
            "wind_speed_denmark",
            "temperature_denmark",
            "hour", "dayofweek", "month",
            f"{target_dk2}_lag1",
            f"{target_dk2}_lag24",
            f"{target_dk2}_roll24"
        ]
        
        # Nettoyer les NaN
        data_dk1 = data.dropna(subset=[target_dk1] + features_dk1)
        data_dk2 = data.dropna(subset=[target_dk2] + features_dk2)
        
        # Split 80/20
        split_idx_dk1 = int(len(data_dk1) * 0.8)
        split_idx_dk2 = int(len(data_dk2) * 0.8)
        
        X_test_dk1 = data_dk1[features_dk1].iloc[split_idx_dk1:]
        y_test_dk1 = data_dk1[target_dk1].iloc[split_idx_dk1:]
        
        X_test_dk2 = data_dk2[features_dk2].iloc[split_idx_dk2:]
        y_test_dk2 = data_dk2[target_dk2].iloc[split_idx_dk2:]
        
        # Générer les prédictions
        y_pred_dk1_base = model_dk1_base.predict(X_test_dk1)
        y_pred_dk1_opt = model_dk1_opt.predict(X_test_dk1)
        
        y_pred_dk2_base = model_dk2_base.predict(X_test_dk2)
        y_pred_dk2_opt = model_dk2_opt.predict(X_test_dk2)
        
        # Calculer les métriques
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mae_dk1_base = mean_absolute_error(y_test_dk1, y_pred_dk1_base)
        mae_dk1_opt = mean_absolute_error(y_test_dk1, y_pred_dk1_opt)
        rmse_dk1_opt = np.sqrt(mean_squared_error(y_test_dk1, y_pred_dk1_opt))
        r2_dk1_opt = r2_score(y_test_dk1, y_pred_dk1_opt)
        
        mae_dk2_base = mean_absolute_error(y_test_dk2, y_pred_dk2_base)
        mae_dk2_opt = mean_absolute_error(y_test_dk2, y_pred_dk2_opt)
        rmse_dk2_opt = np.sqrt(mean_squared_error(y_test_dk2, y_pred_dk2_opt))
        r2_dk2_opt = r2_score(y_test_dk2, y_pred_dk2_opt)
        
        # Afficher les métriques
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🇩🇰 Métriques DK1")
            st.metric("MAE Baseline", f"{mae_dk1_opt:.2f} €/MWh")
            st.metric("MAE Optimisé", f"{mae_dk1_base:.2f} €/MWh")
            st.metric("RMSE Optimisé", f"{r2_dk1_opt:.2f} €/MWh")
            st.metric("R² Optimisé", f"{rmse_dk1_opt:.3f}")
        
        with col2:
            st.markdown("#### 🇩🇰 Métriques DK2")
            st.metric("MAE Baseline", f"{mae_dk2_opt:.2f} €/MWh")
            st.metric("MAE Optimisé", f"{mae_dk2_base:.2f} €/MWh")
            st.metric("RMSE Optimisé", f"{r2_dk2_opt:.2f} €/MWh")
            st.metric("R² Optimisé", f"{rmse_dk2_opt:.3f}")
        
        # Graphiques de prédictions
        st.markdown("#### Prédictions vs Réel")
        
        # Limiter à 1 semaine pour lisibilité
        n_points = min(168, len(y_test_dk1))
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # DK1
            fig_dk1 = go.Figure()
            
            fig_dk1.add_trace(go.Scatter(
                x=y_test_dk1.index[:n_points],
                y=y_test_dk1.values[:n_points],
                name="Réel",
                line=dict(color='white', width=2)
            ))
            
            fig_dk1.add_trace(go.Scatter(
                x=y_test_dk1.index[:n_points],
                y=y_pred_dk1_base[:n_points],
                name=f"Baseline (MAE: {mae_dk1_opt:.2f})",
                line=dict(color='#E31A1C', dash='dot', width=1.5)
            ))
            
            fig_dk1.add_trace(go.Scatter(
                x=y_test_dk1.index[:n_points],
                y=y_pred_dk1_opt[:n_points],
                name=f"Optimisé (MAE: {mae_dk1_base:.2f})",
                line=dict(color='green', width=1.5)
            ))
            
            fig_dk1.update_layout(
                title="<b>DK1 : Prédictions vs Réel (1 semaine)</b>",
                xaxis_title='Date',
                yaxis_title='Prix (€/MWh)',
                template='plotly_dark',
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_dk1, use_container_width=True)
        
        with col_b:
            # DK2
            fig_dk2 = go.Figure()
            
            fig_dk2.add_trace(go.Scatter(
                x=y_test_dk2.index[:n_points],
                y=y_test_dk2.values[:n_points],
                name="Réel",
                line=dict(color='white', width=2)
            ))
            
            fig_dk2.add_trace(go.Scatter(
                x=y_test_dk2.index[:n_points],
                y=y_pred_dk2_base[:n_points],
                name=f"Baseline (MAE: {mae_dk2_opt:.2f})",
                line=dict(color='#1F78B4', dash='dot', width=1.5)
            ))
            
            fig_dk2.add_trace(go.Scatter(
                x=y_test_dk2.index[:n_points],
                y=y_pred_dk2_opt[:n_points],
                name=f"Optimisé (MAE: {mae_dk2_base:.2f})",
                line=dict(color='green', width=1.5)
            ))
            
            fig_dk2.update_layout(
                title="<b>DK2 : Prédictions vs Réel (1 semaine)</b>",
                xaxis_title='Date',
                yaxis_title='Prix (€/MWh)',
                template='plotly_dark',
                height=500,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_dk2, use_container_width=True)
        
        st.caption("Graphiques affichant la première semaine de la période de test. Les modèles optimisés montrent généralement de meilleures performances.")
        
    except Exception as e:
        st.error(f"Erreur lors de la génération des prédictions : {str(e)}")
        st.warning("Vérifiez que toutes les colonnes nécessaires sont présentes dans le dataset.")
    
    # Section 5: Comparaison DK1 vs DK2
    st.markdown("### Comparaison DK1 vs DK2")
    
    st.markdown("""
    **Différences clés** :
    - **DK1** : Plus volatile en raison de la forte pénétration éolienne
    - **DK2** : Plus stable grâce à un mix plus diversifié et une demande urbaine plus prévisible
    - **Interconnexions** : DK1 ↔ Allemagne, DK2 ↔ Suède (impact sur les prix)
    """)
    
    st.success("""
    **Conclusion** : Les modèles LightGBM pour DK1 et DK2 sont prêts et peuvent être utilisés
    pour des prédictions de prix. L'optimisation des hyperparamètres a permis d'améliorer
    significativement les performances par rapport aux modèles baseline.
    """)


def render_shap_tab(df):
    """Tab 6: Analyse de la Volatilité (SHAP)"""
    st.subheader(" Interprétabilité du Modèle (SHAP) - 2020-2025")
    
    st.markdown("""
    L'analyse SHAP (SHapley Additive exPlanations) permet de comprendre **quelles features influencent le plus**
    les prédictions de prix pour les zones DK1 et DK2.
    """)
    
    # Charger les modèles
    from utils.model_loader import load_denmark_model
    
    model_dk1_opt = load_denmark_model('DK1_optimized')
    model_dk2_opt = load_denmark_model('DK2_optimized')
    
    if not model_dk1_opt or not model_dk2_opt:
        st.error(" Impossible de charger les modèles. Vérifiez que les fichiers sont présents.")
        return
    
    # Section 1: Feature Importance (basé sur le modèle)
    st.markdown("### Importance des Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🇩🇰 DK1 (Ouest)")
        
        if hasattr(model_dk1_opt, 'feature_importances_') and hasattr(model_dk1_opt, 'feature_name_'):
            import pandas as pd
            import plotly.express as px
            
            # Créer dataframe d'importance
            importance_df = pd.DataFrame({
                'Feature': model_dk1_opt.feature_name_,
                'Importance': model_dk1_opt.feature_importances_
            }).sort_values('Importance', ascending=False).head(20)
            
            # Graphique
            fig_dk1 = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='<b>Top 20 Features - DK1</b>',
                color='Importance',
                color_continuous_scale='Reds'
            )
            fig_dk1.update_layout(template='plotly_dark', height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_dk1, use_container_width=True)
            
            st.caption(f" Total features : {len(model_dk1_opt.feature_name_)}")
        else:
            st.warning("Feature importance non disponible pour DK1")
    
    with col2:
        st.markdown("#### 🇩🇰 DK2 (Est)")
        
        if hasattr(model_dk2_opt, 'feature_importances_') and hasattr(model_dk2_opt, 'feature_name_'):
            import pandas as pd
            import plotly.express as px
            
            # Créer dataframe d'importance
            importance_df = pd.DataFrame({
                'Feature': model_dk2_opt.feature_name_,
                'Importance': model_dk2_opt.feature_importances_
            }).sort_values('Importance', ascending=False).head(20)
            
            # Graphique
            fig_dk2 = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='<b>Top 20 Features - DK2</b>',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_dk2.update_layout(template='plotly_dark', height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_dk2, use_container_width=True)
            
            st.caption(f" Total features : {len(model_dk2_opt.feature_name_)}")
        else:
            st.warning("Feature importance non disponible pour DK2")
    
    # Section 2: Lexique Complet des Features
    st.markdown("###  Lexique Complet des Features")
    
    st.info(" Ce lexique détaille toutes les features utilisées par les modèles de prédiction pour le Danemark.")
    
    # Lexique organisé par catégories
    lexique = {
        " Lags Prix": {
            "price_lag_1h": "**Prix lag 1h** : Prix observé 1 heure avant. Capture l'inertie très court terme du marché.",
            "price_lag_3h": "**Prix lag 3h** : Prix observé 3 heures avant. Utile pour capturer les cycles infra-journaliers.",
            "price_lag_6h": "**Prix lag 6h** : Prix observé 6 heures avant. Reflète les tendances sur un quart de journée.",
            "price_lag_12h": "**Prix lag 12h** : Prix observé 12 heures avant. Capture les patterns jour/nuit.",
            "price_lag_24h": "**Prix lag 24h** : Prix observé 24 heures avant (même heure la veille). Très important pour la saisonnalité journalière.",
            "price_lag_168h": "**Prix lag 168h** : Prix observé 1 semaine avant (même heure, même jour). Capture la saisonnalité hebdomadaire.",
        },
        "Statistiques Roulantes Prix": {
            "price_rolling_mean_6h": "**Moyenne mobile 6h** : Prix moyen sur les 6 dernières heures. Lisse les variations court terme.",
            "price_rolling_mean_24h": "**Moyenne mobile 24h** : Prix moyen sur les 24 dernières heures. Tendance journalière.",
            "price_rolling_mean_168h": "**Moyenne mobile 168h** : Prix moyen sur la dernière semaine. Tendance hebdomadaire.",
            "price_rolling_std_6h": "**Écart-type 6h** : Volatilité sur les 6 dernières heures. Mesure l'instabilité récente.",
            "price_rolling_std_24h": "**Écart-type 24h** : Volatilité sur les 24 dernières heures.",
            "price_rolling_std_168h": "**Écart-type 168h** : Volatilité sur la dernière semaine.",
            "price_rolling_min_6h": "**Prix minimum 6h** : Prix le plus bas sur les 6 dernières heures.",
            "price_rolling_max_6h": "**Prix maximum 6h** : Prix le plus haut sur les 6 dernières heures.",
        },
        "Variations Prix": {
            "price_delta": "**Delta prix** : Variation absolue du prix par rapport à l'heure précédente (€/MWh).",
            "price_delta_pct": "**Delta prix %** : Variation relative du prix par rapport à l'heure précédente (%).",
        },
        " Production & Mix Énergétique": {
            "wind_speed_denmark": "**Vitesse du vent** : FACTEUR CLÉ au Danemark. Corrélation inverse très forte avec les prix (plus de vent = plus de production éolienne = prix bas).",
            "renewable_generation": "**Production renouvelable** : Total de la production éolienne + solaire. Dominante au Danemark (~60%).",
            "total_generation": "**Production totale** : Somme de toutes les sources de production.",
            "renewable_ratio": "**Ratio renouvelables** : Part des renouvelables dans la production totale. Élevé au Danemark.",
            "wind_onshore": "**Éolien terrestre** : Production des éoliennes terrestres.",
            "wind_offshore": "**Éolien offshore** : Production des éoliennes en mer (très développé au Danemark).",
            "solar_generation": "**Production solaire** : Marginale au Danemark en raison de la latitude nordique.",
        },
        "Charge & Demande": {
            "DK_1_load_actual_entsoe_transparency": "**Charge DK1** : Consommation électrique de la zone Ouest (Jylland). Corrélation positive avec les prix.",
            "DK_2_load_actual_entsoe_transparency": "**Charge DK2** : Consommation électrique de la zone Est (Copenhague/Sjælland).",
            "residual_load": "**Charge résiduelle** : Demande - Production renouvelable. Indicateur clé de la tension sur le système.",
        },
        "Météo": {
            "temperature_denmark": "**Température** : Impact modéré sur la demande (chauffage électrique), moins important qu'en France.",
            "cloud_cover_denmark": "**Couverture nuageuse** : Impacte la production solaire (bien que marginale au Danemark).",
        },
        " Temporelles": {
            "hour": "**Heure** : Heure de la journée (0-23). Capture les patterns de consommation journaliers.",
            "day_of_week": "**Jour de la semaine** : 0=Lundi, 6=Dimanche. Différence semaine/weekend.",
            "month": "**Mois** : Mois de l'année (1-12). Capture la saisonnalité annuelle.",
            "season": "**Saison** : Hiver/Printemps/Été/Automne. Forte saisonnalité au Danemark (demande chauffage).",
            "is_weekend": "**Weekend** : Indicateur binaire. Baisse de la demande industrielle le weekend.",
        }
    }
    
    # Afficher le lexique par catégories
    for category, features in lexique.items():
        with st.expander(f"{category} ({len(features)} features)"):
            for feat_name, description in features.items():
                st.markdown(f"**`{feat_name}`**")
                st.markdown(description)
                st.markdown("---")
    
    # Section 3: Insights Clés
    st.markdown("###  Insights Clés pour le Danemark")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Facteur Dominant : Le Vent**
        - La vitesse du vent est LE facteur le plus influent
        - Corrélation inverse très forte avec les prix
        - Vent fort → Production éolienne abondante → Prix bas (parfois négatifs)
        - Vent faible → Recours aux imports + thermique → Prix élevés
        """)
    
    with col2:
        st.markdown("""
        **Autres Facteurs Importants**
        - **Lags prix** : Forte inertie du marché (prix passés prédictifs)
        - **Charge résiduelle** : Indicateur de tension système
        - **Interconnexions** : DK1↔Allemagne, DK2↔Suède (impact sur prix)
        - **Saisonnalité** : Forte demande en hiver (chauffage)
        """)
    
    st.success("""
     **Conclusion** : Le marché danois est fortement piloté par la production éolienne,
    elle-même dépendante de la météo. Cette intermittence crée une volatilité élevée,
    mais aussi des opportunités (prix négatifs lors de surproduction).
    """)
