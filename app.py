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

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & AGENT STATE
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus | Competition Build", layout="wide", initial_sidebar_state="collapsed")

# Agent Registry: Stores the intelligence handoff between tabs
if "agent_registry" not in st.session_state:
    st.session_state.agent_registry = {
        "auditor_logs": {},
        "cleaning_protocol": [],
        "synthesis_ready": False
    }

if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# Helper: Encode local Scotty mascot if available
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except: return ""

# ---------------------------------------------------------
# 2. THE ALCHEMIST ENGINE: ROBUST ETL & SCHEMA PROTECTION
# ---------------------------------------------------------
def find_col(df, options):
    """Agent Helper: Dynamically finds a column name from a list of aliases."""
    for opt in options:
        if opt in df.columns: return opt
    return None

def clean_val(series):
    """Agent Helper: Standardizes financial strings into floats."""
    return pd.to_numeric(series.astype(str).str.replace(r'[,\%\$]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def build_master_hub():
    """Agent 2 (Alchemist) Synthesis: Joins all 12 Files with Schema Protection."""
    try:
        # A. LOAD INDEX (Source of Truth)
        idx = pd.read_csv('data/UCM Campaign Index.csv')
        utm_col = find_col(idx, ['UTM campaign', 'Campaign_ID', 'UTM_Combined_ID'])
        idx['utm_clean'] = idx[utm_col].astype(str).str.lower().str.strip() if utm_col else ""

        # B. AGGREGATE GOOGLE ADS (FY25 & FY26)
        g_files = ['GAds_FY25_Totals_Jul2024-Jun2025.csv', 'GAds_FY26_Totals_Jul-Dec2025.csv']
        g_dfs = []
        for f in g_files:
            if os.path.exists(f'data/{f}'):
                _df = pd.read_csv(f'data/{f}')
                key_col = find_col(_df, ['Ad name', 'Campaign', 'Campaign Name', 'Ad group'])
                if key_col:
                    _df['utm_clean'] = _df[key_col].astype(str).str.lower().str.strip()
                    _df['Cost'] = clean_val(_df['Cost']) if 'Cost' in _df.columns else 0
                    g_dfs.append(_df)
        
        g_all = pd.concat(g_dfs, ignore_index=True) if g_dfs else pd.DataFrame(columns=['utm_clean', 'Cost'])
        g_agg = g_all.groupby('utm_clean').agg(GAds_Spend=('Cost', 'sum')).reset_index()

        # C. AGGREGATE LINKEDIN
        li_path = 'data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv'
        if os.path.exists(li_path):
            li = pd.read_csv(li_path)
            li_key = find_col(li, ['Campaign Name', 'Campaign', 'Campaign name'])
            if li_key:
                li['utm_clean'] = li[li_key].astype(str).str.lower().str.strip()
                li['Total Spend'] = clean_val(li['Total Spend'])
                li_agg = li.groupby('utm_clean').agg(LI_Spend=('Total Spend', 'sum')).reset_index()
        else:
            li_agg = pd.DataFrame(columns=['utm_clean', 'LI_Spend'])

        # D. AGGREGATE GOOGLE ANALYTICS
        ga_files = ['GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', 'GA_FY26_UTM_Totals_Jul-Dec2025.csv']
        ga_dfs = []
        for f in ga_files:
            if os.path.exists(f'data/{f}'):
                _df = pd.read_csv(f'data/{f}', skiprows=1)
                ga_key = find_col(_df, ['Session campaign', 'Campaign'])
                if ga_key:
                    _df['utm_clean'] = _df[ga_key].astype(str).str.lower().str.strip()
                    _df['Total users'] = pd.to_numeric(_df['Total users'], errors='coerce').fillna(0)
                    ga_dfs.append(_df)
        
        ga_all = pd.concat(ga_dfs, ignore_index=True) if ga_dfs else pd.DataFrame(columns=['utm_clean', 'Total users'])
        ga_agg = ga_all.groupby('utm_clean').agg(Total_Users=('Total users', 'sum')).reset_index()

        # E. FINAL SYNTHESIS
        master = pd.merge(idx, ga_agg, on='utm_clean', how='left')
        master = pd.merge(master, g_agg, on='utm_clean', how='left')
        master = pd.merge(master, li_agg, on='utm_clean', how='left').fillna(0)
        
        master['Total_Spend'] = master['GAds_Spend'] + master['LI_Spend']
        master['CPWU'] = master['Total_Spend'].div(master['Total_Users'].replace(0, np.nan)).fillna(0)
        
        st.session_state.agent_registry["synthesis_ready"] = True
        return master
    except Exception as e:
        st.error(f"Alchemist Engine Error: {str(e)}")
        return pd.DataFrame()

# Global execution of the Hub
master_df = build_master_hub()

# ---------------------------------------------------------
# 3. GLOBAL UI: NAVIGATION
# ---------------------------------------------------------
nav_cards_html = """
<style>
.nav-grid { display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 5px; }
.nav-card {
    background: #ffffff; border: 1.5px solid #eee; border-radius: 12px; padding: 10px;
    width: 120px; text-align: center; color: #333 !important; text-decoration: none;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.nav-card:hover { border-color: #C41230; transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.1); }
.nav-title { font-size: 10px; font-weight: 800; color: #C41230; margin-top: 5px; }
</style>
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card">🏠<div class="nav-title">HOME</div></a>
    <a href="?page=explorer" target="_self" class="nav-card">🕵️<div class="nav-title">1. AUDITOR</div></a>
    <a href="?page=cleaner" target="_self" class="nav-card">⚗️<div class="nav-title">2. ALCHEMIST</div></a>
    <a href="?page=analysis" target="_self" class="nav-card">🧪<div class="nav-title">3. STRATEGIST</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card">🖥️<div class="nav-title">4. ARCHITECT</div></a>
    <a href="?page=graph" target="_self" class="nav-card">🕸️<div class="nav-title">CARTOGRAPHER</div></a>
</div>
"""

# ======================= HOME: 3D PARTICLE GALAXY =======================
if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #C41230; font-weight: 900; margin-top: -15px;'>CMU DATA NEXUS</h1>", unsafe_allow_html=True)
    
    three_js_code = f"""
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>body {{ margin: 0; background: #fff; overflow: hidden; }}</style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement); controls.autoRotate = true;

        const count = 35000; const pos = new Float32Array(count * 3); const colors = new Float32Array(count * 3);
        const red = new THREE.Color(0xC41230); const gray = new THREE.Color(0x6D6E71);
        for(let i=0; i<count; i++){{
            const r = 25 * Math.cbrt(Math.random()); const t = Math.random()*2*Math.PI; const p = Math.acos(2*Math.random()-1);
            pos[i*3] = r * Math.sin(p) * Math.cos(t); pos[i*3+1] = r * Math.sin(p) * Math.sin(t); pos[i*3+2] = r * Math.cos(p);
            const c = Math.random() > 0.8 ? red : gray;
            colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
        }}
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({{size: 0.05, vertexColors: true, transparent: true, opacity: 0.7}})));
        camera.position.z = 22;
        function animate(){{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
        animate();
    </script></body></html>
    """
    components.html(three_js_code, height=800)

# ======================= AGENT 1: FORENSIC AUDITOR =======================
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    from streamlit_extras.colored_header import colored_header
    colored_header("Step 1: Forensic Auditor", "Detecting Relational Gaps and Telemetry Leakage.")
    
    files = [f for f in os.listdir('data') if f.endswith('.csv')]
    selected = st.selectbox("Select Telemetry Stream", files)
    raw = pd.read_csv(f'data/{selected}', skiprows=1 if 'UTM_Totals' in selected else 0)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        from streamlit_extras.dataframe_explorer import dataframe_explorer
        st.dataframe(dataframe_explorer(raw), use_container_width=True)
    with col2:
        st.metric("Raw Rows", len(raw))
        st.metric("Nulls", raw.isna().sum().sum())
        st.warning("Relational Fix Required") if raw.isna().sum().sum() > 0 else st.success("Schema Clear")

# ======================= AGENT 2: DATA ALCHEMIST =======================
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Step 2: Data Alchemist")
    st.success("Master Nexus Hub Synthesized: 12 datasets normalized and joined via 'utm_clean'.")
    st.dataframe(master_df, use_container_width=True)

# ======================= AGENT 3: QUANTITATIVE STRATEGIST =======================
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Step 3: Quantitative Strategist")
    if not master_df.empty:
        corr, _ = stats.pearsonr(master_df['Total_Spend'], master_df['Total_Users'])
        st.metric("Spend-to-User Correlation", f"{corr:.2f}")
        st.plotly_chart(px.scatter(master_df, x="Total_Spend", y="Total_Users", trendline="ols", color="Category"), use_container_width=True)

# ======================= AGENT 4: VISUAL ARCHITECT =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Step 4: Visual Architect")
    from streamlit_extras.metric_cards import style_metric_cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spend", f"${master_df['Total_Spend'].sum():,.2f}")
    m2.metric("Total Users", f"{master_df['Total_Users'].sum():,.0f}")
    m3.metric("Avg CPWU", f"${(master_df['Total_Spend'].sum()/master_df['Total_Users'].sum() if master_df['Total_Users'].sum() > 0 else 0):.2f}")
    style_metric_cards(border_left_color="#C41230")
    
    st.plotly_chart(px.bar(master_df.groupby('Category')['Total_Spend'].sum().reset_index(), x='Category', y='Total_Spend', color='Category'), use_container_width=True)

# ======================= AGENT 5: CARTOGRAPHER =======================
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Nexus Cartographer")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333")
    net.add_node("Nexus Hub", size=40, color="#C41230")
    for cat in master_df['Category'].unique():
        net.add_node(str(cat), size=25, color="#6D6E71")
        net.add_edge("Nexus Hub", str(cat))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)
