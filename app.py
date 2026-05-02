import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import base64
from urllib.parse import urlparse, parse_qs

# Streamlit Extras for premium UI
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer

# ---------------------------------------------------------
# 1. PAGE CONFIG & PERSISTENT AGENT MEMORY
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus | forensic Edition", layout="wide", initial_sidebar_state="collapsed")

if "agent_registry" not in st.session_state:
    st.session_state.agent_registry = {
        "auditor_logs": {},
        "synthesis_metrics": {"total_spend": 0, "total_users": 0},
        "anomalies_fixed": 0
    }

if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# Helper: Asset encoding
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError: return ""

# ---------------------------------------------------------
# 2. THE MULTI-AGENT ETL ENGINE (REAL DATA ONLY)
# ---------------------------------------------------------
ALL_FILES = [
    "2024-25_Campaign_Management_1769521985.csv", "2025-26_Campaign_Management_1769522231.csv",
    "GA_FY25_TimeSeries (1).csv", "GA_FY26_TimeSeries.csv",
    "GA_FY25_UTM_Totals_Jul2024-Jun2025.csv", "GA_FY26_UTM_Totals_Jul-Dec2025.csv",
    "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv", "GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv",
    "GAds_FY25_Totals_Jul2024-Jun2025.csv", "GAds_FY26_Totals_Jul-Dec2025.csv",
    "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv", "UCM Campaign Index.csv"
]

def clean_financial(series):
    """Helper for Agent 2: Cleans artifacts from real marketing data."""
    return pd.to_numeric(series.astype(str).str.replace(r'[,\%\$]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def build_master_nexus_hub():
    """Agent 2 Implementation: The Data Alchemist Synthesis"""
    # Load Index
    idx = pd.read_csv('data/UCM Campaign Index.csv')
    idx['utm_clean'] = idx['UTM campaign'].astype(str).str.lower().str.strip()
    
    # Process Google Ads
    g1 = pd.read_csv('data/GAds_FY25_Totals_Jul2024-Jun2025.csv')
    g2 = pd.read_csv('data/GAds_FY26_Totals_Jul-Dec2025.csv')
    g_all = pd.concat([g1, g2], ignore_index=True)
    g_all['utm_clean'] = g_all['Ad name'].astype(str).str.lower().str.strip()
    g_all['Cost'] = clean_financial(g_all['Cost'])
    g_agg = g_all.groupby('utm_clean').agg(GAds_Spend=('Cost', 'sum'), GAds_Clicks=('Clicks', 'sum')).reset_index()

    # Process LinkedIn
    li = pd.read_csv('data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv')
    li['utm_clean'] = li['Campaign Name'].astype(str).str.lower().str.strip()
    li['Total Spend'] = clean_financial(li['Total Spend'])
    li_agg = li.groupby('utm_clean').agg(LI_Spend=('Total Spend', 'sum'), LI_Clicks=('Clicks', 'sum')).reset_index()

    # Process GA Totals
    ga1 = pd.read_csv('data/GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', skiprows=1)
    ga2 = pd.read_csv('data/GA_FY26_UTM_Totals_Jul-Dec2025.csv', skiprows=1)
    ga_all = pd.concat([ga1, ga2], ignore_index=True)
    ga_all['utm_clean'] = ga_all['Session campaign'].astype(str).str.lower().str.strip()
    ga_all['Total users'] = pd.to_numeric(ga_all['Total users'], errors='coerce').fillna(0)
    ga_agg = ga_all.groupby('utm_clean').agg(Total_Users=('Total users', 'sum'), Engage_Rate=('Engagement rate', 'mean')).reset_index()

    # Master Synthesis
    hub = pd.merge(idx, ga_agg, on='utm_clean', how='left')
    hub = pd.merge(hub, g_agg, on='utm_clean', how='left')
    hub = pd.merge(hub, li_agg, on='utm_clean', how='left').fillna(0)
    
    hub['Total_Spend'] = hub['GAds_Spend'] + hub['LI_Spend']
    hub['CPWU'] = hub['Total_Spend'].div(hub['Total_Users'].replace(0, np.nan)).fillna(0)
    
    return hub

master_df = build_master_nexus_hub()

# ---------------------------------------------------------
# 3. GLOBAL UI: NAVIGATION & STYLING
# ---------------------------------------------------------
nav_cards_html = """
<style>
.nav-grid { display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 5px; }
.nav-card {
    background: #fff; border: 1.5px solid #eee; border-radius: 12px; padding: 10px;
    width: 120px; text-align: center; color: #333 !important; text-decoration: none;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.nav-card:hover { border-color: #C41230; transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
.nav-title { font-size: 10px; font-weight: 800; color: #C41230; margin-top: 5px; }
</style>
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card">🏠<div class="nav-title">HOME</div></a>
    <a href="?page=explorer" target="_self" class="nav-card">🕵️<div class="nav-title">AUDITOR</div></a>
    <a href="?page=cleaner" target="_self" class="nav-card">⚗️<div class="nav-title">ALCHEMIST</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card">🖥️<div class="nav-title">ARCHITECT</div></a>
    <a href="?page=graph" target="_self" class="nav-card">🕸️<div class="nav-title">CARTOGRAPHER</div></a>
</div>
"""

# ======================= HOME: 3D PARTICLE GALAXY =======================
if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #C41230; font-weight: 900; margin-top: -15px;'>CMU DATA NEXUS</h1>", unsafe_allow_html=True)
    
    scotty_b64 = get_base64_of_bin_file("scotty.png")
    
    three_js_galaxy = f"""
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>body {{ margin: 0; background: #fff; overflow: hidden; }}</style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

        const count = 35000; const pos = new Float32Array(count * 3); const colors = new Float32Array(count * 3);
        const red = new THREE.Color(0xC41230); const gray = new THREE.Color(0x6D6E71);
        for(let i=0; i<count; i++){{
            const r = 25 * Math.cbrt(Math.random()); const t = Math.random() * 2 * Math.PI; const p = Math.acos(2 * Math.random() - 1);
            pos[i*3] = r * Math.sin(p) * Math.cos(t); pos[i*3+1] = r * Math.sin(p) * Math.sin(t); pos[i*3+2] = r * Math.cos(p);
            const c = Math.random() > 0.8 ? red : gray;
            colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
        }}
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({{size: 0.05, vertexColors: true, transparent: true, opacity: 0.7}})));
        
        camera.position.z = 22;
        function animate(){{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
        animate();
    </script></body></html>
    """
    components.html(three_js_galaxy, height=800)

# ======================= AGENT 1: FORENSIC AUDITOR =======================
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header("Step 1: Forensic Auditor", "Deep relational scan across ecosystem CSVs.")
    f = st.selectbox("Select Telemetry Stream", ALL_FILES)
    raw_df = pd.read_csv(f'data/{f}', skiprows=1 if 'UTM_Totals' in f else 0)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.dataframe(dataframe_explorer(raw_df), use_container_width=True)
    with c2:
        st.metric("Total Rows Scanned", len(raw_df))
        st.metric("Anomalies Identified", raw_df.isna().sum().sum())
        st.error("Format Standard Required") if any(raw_df.dtypes == 'object') else st.success("Schema Validated")

# ======================= AGENT 4: VISUAL ARCHITECT =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header("Step 4: Visual Architect", "Executive ROI Command Center (Power BI Edition)")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Spend", f"${master_df['Total_Spend'].sum():,.2f}")
    m2.metric("Total Users", f"{master_df['Total_Users'].sum():,.0f}")
    m3.metric("Avg CPWU", f"${(master_df['Total_Spend'].sum()/master_df['Total_Users'].sum()):.2f}")
    m4.metric("Campaign Coverage", f"{len(master_df)}")
    style_metric_cards(border_left_color="#C41230")

    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(px.scatter(master_df[master_df['Total_Spend']>0], x="Total_Spend", y="Total_Users", color="Category", 
                                   size="CPWU", hover_name="utm_clean", title="The Efficiency Frontier: Spend vs Acquisition"), use_container_width=True)
    with v2:
        top_cats = master_df.groupby('Category')['Total_Spend'].sum().reset_index().sort_values('Total_Spend', ascending=False).head(5)
        st.plotly_chart(px.pie(top_cats, values="Total_Spend", names="Category", hole=0.5, 
                               title="Budget Share by Department Category", color_discrete_sequence=px.colors.sequential.Reds_r), use_container_width=True)
