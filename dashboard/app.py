import streamlit as st
import pandas as pd
import base64
from pathlib import Path
from utils.data_loader import load_data
from views.france import render_france
from views.denmark import render_denmark
from views.comparison import render_comparison

# --- Page Config ---
st.set_page_config(
    page_title="Prédiction du prix de l'électricité en Europe",
    page_icon="⚡",
    layout="wide"
)

# --- Custom CSS for Fun & Attractive UI ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        transition: all 0.3s ease;
        background-color: #1f2937;
        color: #00d4ff;
        border: 1px solid #00d4ff;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        background-color: #16213e;
        color: #ffffff;
    }
    .card-container {
        background-color: #16213e;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        height: 100%;
        border: 1px solid #2d3748;
    }
    .card-title {
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
        color: #00d4ff;
    }
    .card-emoji {
        font-size: 4em;
        margin-bottom: 10px;
        display: block;
    }
    .card-img {
        margin-bottom: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .card-desc {
        color: #e0f7fa;
        margin-bottom: 15px;
        font-size: 0.9em;
    }
    h1 {
        text-align: center;
        color: #e0f7fa;
    }
    h3 {
        color: #a0aec0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Function for Images ---
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# --- Session State for Navigation ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def navigate_to(page_name):
    st.session_state.page = page_name

# --- Title ---
st.title("⚡ Prédiction du prix de l'électricité en Europe")
st.markdown("<h3 style='text-align: center;'>France - Danemark</h3>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Data ---
with st.spinner("Chargement des données..."):
    df = load_data()

if df.empty:
    st.stop()

# --- Sidebar ---
st.sidebar.title("Informations")

# Dataset Stats
st.sidebar.subheader("🗂️ Datasets")

# Période globale
min_date = df.index.min().date()
max_date = df.index.max().date()

st.sidebar.info(f"""
**Période Globale** :  
{min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}

**Total** : {len(df):,} observations
""")

# France
fr_data = df[['FR_price_day_ahead']].dropna()
st.sidebar.markdown("**🇫🇷 France**")
st.sidebar.caption(f"""
- Période : 2020-2025  
- Observations : {len(fr_data):,}  
- Prix moyen : {fr_data['FR_price_day_ahead'].mean():.2f} €/MWh
""")

# Danemark
dk_data = df[['DK_1_price_day_ahead', 'DK_2_price_day_ahead']].dropna()
if len(dk_data) > 0:
    dk_avg = (dk_data['DK_1_price_day_ahead'] + dk_data['DK_2_price_day_ahead']) / 2
    st.sidebar.markdown("**🇩🇰 Danemark**")
    st.sidebar.caption(f"""
    - Période : 2020-2025  
    - Observations : {len(dk_data):,}  
    - Prix moyen : {dk_avg.mean():.2f} €/MWh
    """)

st.sidebar.markdown("---")

# Project Info
st.sidebar.subheader("👥 Auteurs")
st.sidebar.markdown("""
*   Franck F.
*   Charlotte M.
*   Djourah O.
*   Koffi A.
*   Youssef S.
""")

st.sidebar.markdown("---")
st.sidebar.caption("Projet DataBI - 2025")

# No more specific filtering for global dataframe
# df_filtered is now just df
df_filtered = df

# --- Navigation & Content ---

if st.session_state.page == 'Home':
    st.markdown("<div style='text-align: center; margin-bottom: 30px; font-size: 1.2em; color: #e0f7fa;'>Bienvenue sur le tableau de bord interactif. Sélectionnez une zone pour commencer l'analyse.</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Load images
    current_dir = Path(__file__).parent
    fr_flag_path = current_dir / "asset" / "Flag_of_France.png"
    dk_flag_path = current_dir / "asset" / "Flag_of_Denmark.svg.png"
    
    fr_img_html = ""
    dk_img_html = ""
    
    if fr_flag_path.exists():
        fr_base64 = img_to_bytes(fr_flag_path)
        fr_img_html = f"<img src='data:image/png;base64,{fr_base64}' class='card-img' style='width: 100px; height: auto;'>"
    else:
        fr_img_html = "<span class='card-emoji'>🇫🇷</span>"
        
    if dk_flag_path.exists():
        dk_base64 = img_to_bytes(dk_flag_path)
        dk_img_html = f"<img src='data:image/png;base64,{dk_base64}' class='card-img' style='width: 100px; height: auto;'>"
    else:
        dk_img_html = "<span class='card-emoji'>🇩🇰</span>"

    with col1:
        st.markdown(f"""
        <div class="card-container">
            {fr_img_html}
            <div class="card-title">France</div>
            <div class="card-desc">Analysez la consommation, la production et les prix de l'électricité en France.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explorer la France", key="btn_fr"):
            navigate_to('France')
            st.rerun()
            
    with col2:
        st.markdown(f"""
        <div class="card-container">
            {dk_img_html}
            <div class="card-title">Danemark</div>
            <div class="card-desc">Découvrez les tendances énergétiques et l'impact de l'éolien au Danemark.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explorer le Danemark", key="btn_dk"):
            navigate_to('Denmark')
            st.rerun()
            
    with col3:
        st.markdown(f"""
        <div class="card-container">
            <div style="display: flex; justify-content: center; gap: 10px; margin-bottom: 10px;">
                {fr_img_html.replace('width: 100px', 'width: 60px')}
                {dk_img_html.replace('width: 100px', 'width: 60px')}
            </div>
            <div class="card-title">Comparaison</div>
            <div class="card-desc">Comparez les deux pays côte à côte pour comprendre les différences structurelles.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Comparer les Pays", key="btn_comp"):
            navigate_to('Comparison')
            st.rerun()

elif st.session_state.page == 'France':
    if st.button("← Retour à l'accueil"):
        navigate_to('Home')
        st.rerun()
    render_france(df_filtered)

elif st.session_state.page == 'Denmark':
    if st.button("← Retour à l'accueil"):
        navigate_to('Home')
        st.rerun()
    
    # Pas de filtre de date - utiliser toutes les données DK disponibles
    render_denmark(df_filtered)

elif st.session_state.page == 'Comparison':
    if st.button("← Retour à l'accueil"):
        navigate_to('Home')
        st.rerun()
    render_comparison(df_filtered)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; padding: 10px; font-size: 0.9em;'>
        © 2025 Projet DataBI - Tous droits réservés
    </div>
    """,
    unsafe_allow_html=True
)

