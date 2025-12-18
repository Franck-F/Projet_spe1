import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def render_comparison(df):
    """Comparaison France vs Danemark"""
    
    st.header("Comparaison France vs Danemark")
    st.caption("Analyse comparative des deux marchés électriques (2020-2025)")
    
    # Bouton pour vider le cache si les données semblent incorrectes
    if st.button("Rafraîchir les données (vider le cache)"):
        st.cache_data.clear()
        st.rerun()

    
    # Filtrer les données valides pour chaque pays
    fr_price = 'FR_price_day_ahead'
    dk1_price = 'DK_1_price_day_ahead'
    dk2_price = 'DK_2_price_day_ahead'
    
    # Vérifier disponibilité des données
    has_france = fr_price in df.columns
    has_denmark = dk1_price in df.columns and dk2_price in df.columns
    
    if not has_france or not has_denmark:
        st.error("Données insuffisantes pour la comparaison France-Danemark.")
        return
    
    # IMPORTANT: Filtrer sur la période commune 2020-2025 pour comparaison équitable
    df = df.loc['2020-01-01':'2025-12-31']
    
    # Filtrer les données valides (non-NaN) pour chaque pays
    df_france = df[[fr_price]].dropna()
    df_denmark = df[[dk1_price, dk2_price]].dropna()
    
    # Utiliser les index communs
    common_index = df_france.index.intersection(df_denmark.index)
    df = df.loc[common_index]
    
    if df.empty:
        st.warning("Pas de données communes pour la période 2020-2025.")
        return
    
    st.info(f"**Période de comparaison** : {df.index.min().strftime('%d/%m/%Y')} - {df.index.max().strftime('%d/%m/%Y')} ({len(df):,} observations)")
    
    # Créer prix moyen Danemark
    df['DK_price_avg'] = (df[dk1_price] + df[dk2_price]) / 2
    
    # Section 1: Vue d'ensemble - Métriques Clés
    st.markdown("### Métriques Clés")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fr_avg = df[fr_price].mean()
        st.metric("Prix Moyen France", f"{fr_avg:.2f} €/MWh")
    
    with col2:
        dk_avg = df['DK_price_avg'].mean()
        st.metric("Prix Moyen Danemark", f"{dk_avg:.2f} €/MWh")
    
    with col3:
        price_diff = ((dk_avg - fr_avg) / fr_avg) * 100
        st.metric("Écart Prix", f"{price_diff:+.1f}%", 
                 delta_color="inverse" if price_diff > 0 else "normal")
    
    with col4:
        fr_std = df[fr_price].std()
        dk_std = df['DK_price_avg'].std()
        st.metric("Volatilité FR", f"{fr_std:.1f} €/MWh")
        st.metric("Volatilité DK", f"{dk_std:.1f} €/MWh")
    
    # Section 2: Comparaison des Prix
    st.markdown("### Évolution des Prix")
    
    # Prix hebdomadaire
    df_weekly = df[[fr_price, 'DK_price_avg']].resample('W').mean()
    
    fig_prices = go.Figure()
    
    fig_prices.add_trace(go.Scatter(
        x=df_weekly.index,
        y=df_weekly[fr_price],
        mode='lines',
        name='France',
        line=dict(color='#0055A4', width=2)
    ))
    
    fig_prices.add_trace(go.Scatter(
        x=df_weekly.index,
        y=df_weekly['DK_price_avg'],
        mode='lines',
        name='Danemark',
        line=dict(color='#C8102E', width=2)
    ))
    
    fig_prices.update_layout(
        title="<b>Évolution Hebdomadaire des Prix</b>",
        xaxis_title='Date',
        yaxis_title='Prix Moyen (€/MWh)',
        template='plotly_dark',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_prices, use_container_width=True)
    st.info("**Interprétation** : Les deux marchés suivent des trajectoires similaires (crise 2022), mais le Danemark présente généralement une volatilité plus élevée en raison de sa forte dépendance à l'éolien.")
    
    # Section 3: Mix Énergétique
    st.markdown("### Comparaison du Mix Énergétique")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### France")
        
        # Mix France (approximatif basé sur les colonnes disponibles)
        fr_mix = {
            'Nucléaire': 70,  # Dominance nucléaire
            'Hydraulique': 12,
            'Éolien': 8,
            'Solaire': 3,
            'Gaz': 5,
            'Autres': 2
        }
        
        fig_fr = px.pie(
            values=list(fr_mix.values()),
            names=list(fr_mix.keys()),
            title='<b>Mix Énergétique France</b>',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_fr.update_traces(textposition='inside', textinfo='percent+label')
        fig_fr.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_fr, use_container_width=True)
        st.caption("Dominance du nucléaire (~70%), source pilotable et bas carbone.")
    
    with col2:
        st.markdown("#### 🇩🇰 Danemark")
        
        # Mix Danemark (approximatif)
        dk_mix = {
            'Éolien': 55,  # Champion de l'éolien
            'Solaire': 5,
            'Biomasse': 15,
            'Charbon': 10,
            'Gaz': 10,
            'Autres': 5
        }
        
        fig_dk = px.pie(
            values=list(dk_mix.values()),
            names=list(dk_mix.keys()),
            title='<b>Mix Énergétique Danemark</b>',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_dk.update_traces(textposition='inside', textinfo='percent+label')
        fig_dk.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_dk, use_container_width=True)
        st.caption("Champion mondial de l'éolien (~55%), forte variabilité.")
    
    # Section 4: Distribution des Prix
    st.markdown("### Distribution des Prix")
    
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=df[fr_price],
        name='France',
        opacity=0.6,
        marker_color='#0055A4',
        nbinsx=100
    ))
    
    fig_dist.add_trace(go.Histogram(
        x=df['DK_price_avg'],
        name='Danemark',
        opacity=0.6,
        marker_color='#C8102E',
        nbinsx=100
    ))
    
    fig_dist.update_layout(
        title="<b>Distribution des Prix (Histogramme Superposé)</b>",
        xaxis_title='Prix (€/MWh)',
        yaxis_title='Fréquence',
        barmode='overlay',
        template='plotly_dark',
        height=450
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Le Danemark présente une queue plus épaisse (prix extrêmes plus fréquents) en raison de la variabilité éolienne.")
    
    # Section 5: Volatilité Comparée
    st.markdown("### Analyse de la Volatilité")
    
    # Calculer volatilité mensuelle
    fr_vol = df[fr_price].resample('M').std()
    dk_vol = df['DK_price_avg'].resample('M').std()
    
    fig_vol = go.Figure()
    
    fig_vol.add_trace(go.Bar(
        x=fr_vol.index,
        y=fr_vol.values,
        name='France',
        marker_color='#0055A4'
    ))
    
    fig_vol.add_trace(go.Bar(
        x=dk_vol.index,
        y=dk_vol.values,
        name='Danemark',
        marker_color='#C8102E'
    ))
    
    fig_vol.update_layout(
        title="<b>Volatilité Mensuelle (Écart-type)</b>",
        xaxis_title='Mois',
        yaxis_title='Volatilité (€/MWh)',
        barmode='group',
        template='plotly_dark',
        height=450
    )
    
    st.plotly_chart(fig_vol, use_container_width=True)
    st.info("**Interprétation** : Le Danemark affiche généralement une volatilité plus élevée, particulièrement lors des périodes de faible vent (recours aux imports et moyens thermiques coûteux).")
    
    # Section 6: Tableau Comparatif
    st.markdown("### Tableau Comparatif Détaillé")
    
    comparison_data = {
        "Caractéristique": [
            "Prix Moyen (€/MWh)",
            "Prix Min (€/MWh)",
            "Prix Max (€/MWh)",
            "Volatilité (σ)",
            "Source Dominante",
            "% Renouvelables",
            "Interconnexions",
            "Prix Négatifs",
            "Facteur Clé"
        ],
        "🇫🇷 France": [
            f"{df[fr_price].mean():.2f}",
            f"{df[fr_price].min():.2f}",
            f"{df[fr_price].max():.2f}",
            f"{df[fr_price].std():.2f}",
            "Nucléaire (~70%)",
            "~25%",
            "Multiples (DE, ES, IT, UK...)",
            "Rares",
            "Production nucléaire"
        ],
        "🇩🇰 Danemark": [
            f"{df['DK_price_avg'].mean():.2f}",
            f"{df['DK_price_avg'].min():.2f}",
            f"{df['DK_price_avg'].max():.2f}",
            f"{df['DK_price_avg'].std():.2f}",
            "Éolien (~55%)",
            "~60%",
            "DE (DK1), SE (DK2)",
            "Fréquents",
            "Vitesse du vent"
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Section 7: Insights Clés
    st.markdown("### Insights Clés")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🇫🇷 France - Modèle Nucléaire**
        - ✅ **Stabilité** : Prix plus stables grâce au nucléaire pilotable
        - ✅ **Bas carbone** : Mix décarboné (~90%)
        - ✅ **Indépendance** : Forte capacité de production domestique
        - ⚠️ **Rigidité** : Moins flexible face aux pics de demande
        - ⚠️ **Risque** : Dépendance à la disponibilité du parc nucléaire
        """)
    
    with col2:
        st.markdown("""
        **🇩🇰 Danemark - Modèle Éolien**
        - ✅ **Renouvelables** : Champion mondial de l'éolien
        - ✅ **Innovation** : Leader en technologies vertes
        - ✅ **Flexibilité** : Forte interconnexion avec voisins
        - ⚠️ **Volatilité** : Prix très dépendants de la météo
        - ⚠️ **Intermittence** : Besoin d'imports lors de faible vent
        """)
    
    st.success("""
    **Conclusion** : Les deux pays illustrent des stratégies énergétiques radicalement différentes mais complémentaires.
    La France mise sur la stabilité du nucléaire, le Danemark sur l'agilité des renouvelables.
    Leur intégration au marché européen permet de mutualiser les avantages de chaque modèle.
    """)
