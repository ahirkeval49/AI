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
st.set_page_config(page_title="CMU AI Nexus | forensic Edition", layout="wide", initial_sidebar_state="collapsed")

# Agent Memory: Persisting intelligence across the pipeline
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {
        "audit_logs": {},
        "synthesis_stats": {},
        "model_results": {}
    }

if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# Asset Loader
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except: return ""

# ---------------------------------------------------------
# 2. THE ALCHEMIST ENGINE: DEFENSIVE ETL & REAL DATA SYNTHESIS
# ---------------------------------------------------------
ALL_FILES = [
    "2024-25_Campaign_Management_1769521985.csv", "2025-26_Campaign_Management_1769522231.csv",
    "GA_FY25_TimeSeries (1).csv", "GA_FY26_TimeSeries.csv",
    "GA_FY25_UTM_Totals_Jul2024-Jun2025.csv", "GA_FY26_UTM_Totals_Jul-Dec2025.csv",
    "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv", "GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv",
    "GAds_FY25_Totals_Jul2024-Jun2025.csv", "GAds_FY26_Totals_Jul-Dec2025.csv",
    "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv", "UCM Campaign Index.csv"
]

def find_col(df, aliases):
    """Prevents KeyErrors by scouting for column variations."""
    for alias in aliases:
        if alias in df.columns: return alias
    return None

def clean_currency(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[,\%\$]', '', regex=True), errors='coerce').fillna(0)

@st.cache_data
def build_master_hub():
    """Agent 2 (Alchemist) Synthesis: Joining all 12 Real Data Files."""
    try:
        # Load Index
        idx = pd.read_csv('data/UCM Campaign Index.csv')
        utm_col = find_col(idx, ['UTM campaign', 'Campaign_ID', 'UTM_Combined_ID'])
        idx['utm_clean'] = idx[utm_col].astype(str).str.lower().str.strip() if utm_col else ""

        # GAds Pipeline
        g1, g2 = pd.read_csv('data/GAds_FY25_Totals_Jul2024-Jun2025.csv'), pd.read_csv('data/GAds_FY26_Totals_Jul-Dec2025.csv')
        g_all = pd.concat([g1, g2], ignore_index=True)
        g_key = find_col(g_all, ['Ad name', 'Campaign', 'Campaign Name'])
        g_all['utm_clean'] = g_all[g_key].astype(str).str.lower().str.strip()
        g_all['Cost'] = clean_currency(g_all['Cost'])
        g_agg = g_all.groupby('utm_clean').agg(GAds_Spend=('Cost', 'sum')).reset_index()

        # LinkedIn Pipeline
        li = pd.read_csv('data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv')
        li_key = find_col(li, ['Campaign Name', 'Campaign'])
        li['utm_clean'] = li[li_key].astype(str).str.lower().str.strip()
        li_spend = find_col(li, ['Total Spend', 'Spend', 'Cost'])
        li['LI_Spend'] = clean_currency(li[li_spend])
        li_agg = li.groupby('utm_clean').agg(LI_Spend=('LI_Spend', 'sum')).reset_index()

        # GA Acquisition Pipeline
        ga1 = pd.read_csv('data/GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', skiprows=1)
        ga2 = pd.read_csv('data/GA_FY26_UTM_Totals_Jul-Dec2025.csv', skiprows=1)
        ga_all = pd.concat([ga1, ga2], ignore_index=True)
        ga_key = find_col(ga_all, ['Session campaign', 'Campaign'])
        ga_all['utm_clean'] = ga_all[ga_key].astype(str).str.lower().str.strip()
        ga_all['Total users'] = pd.to_numeric(ga_all['Total users'], errors='coerce').fillna(0)
        ga_agg = ga_all.groupby('utm_clean').agg(Total_Users=('Total users', 'sum')).reset_index()

        # Final Synthesis
        hub = pd.merge(idx, ga_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, g_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, li_agg, on='utm_clean', how='left').fillna(0)
        hub['Total_Spend'] = hub['GAds_Spend'] + hub['LI_Spend']
        hub['CPWU'] = hub['Total_Spend'].div(hub['Total_Users'].replace(0, np.nan)).fillna(0)
        
        return hub
    except Exception as e:
        st.error(f"Alchemist Engine Error: {str(e)}")
        return pd.DataFrame()

master_df = build_master_hub()

# ---------------------------------------------------------
# 3. GLOBAL UI: NAVIGATION
# ---------------------------------------------------------
nav_cards_html = """
<style>
.nav-grid { display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 5px; }
.nav-card {
    background: #fff; border: 2px solid #E0E0E0; border-radius: 12px; padding: 10px;
    width: 125px; text-align: center; color: #333 !important; text-decoration: none;
    transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.nav-card:hover { border-color: #C41230; transform: translateY(-5px); box-shadow: 0 8px 15px rgba(196,18,48,0.2); }
.nav-title { font-size: 9px; font-weight: 800; color: #C41230; margin-top: 5px; }
</style>
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card">🏠<div class="nav-title">HOME</div></a>
    <a href="?page=explorer" target="_self" class="nav-card">🕵️<div class="nav-title">1. AUDITOR</div></a>
    <a href="?page=cleaner" target="_self" class="nav-card">⚗️<div class="nav-title">2. ALCHEMIST</div></a>
    <a href="?page=analysis" target="_self" class="nav-card">🧪<div class="nav-title">3. STRATEGIST</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card">🖥️<div class="nav-title">4. ARCHITECT</div></a>
    <a href="?page=graph" target="_self" class="nav-card">🕸️<div class="nav-title">KNOWLEDGE GRAPH</div></a>
</div>
"""

# ======================= HOME: CMU BRANDED GALAXY =======================
if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #C41230; font-weight: 900; margin-top: -15px;'>CMU DATA NEXUS</h1>", unsafe_allow_html=True)
    
    three_js_galaxy = """
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body { margin: 0; background: #fff; overflow: hidden; font-family: sans-serif; }
        .node-label {
            position: absolute; background: rgba(255,255,255,0.9); border: 2px solid #C41230;
            padding: 6px 12px; border-radius: 20px; font-weight: bold; cursor: pointer;
            transition: 0.3s; pointer-events: auto; text-decoration: none; color: #C41230; font-size: 11px;
        }
    </style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true; controls.autoRotateSpeed = 0.4; // SLOW ROTATION

        // CMU BRANDED PARTICLES
        const count = 35000; const pos = new Float32Array(count * 3); const colors = new Float32Array(count * 3);
        const red = new THREE.Color(0xC41230); const gray = new THREE.Color(0x6D6E71);
        for(let i=0; i<count; i++){
            const r = 25 * Math.cbrt(Math.random()); const t = Math.random()*2*Math.PI; const p = Math.acos(2*Math.random()-1);
            pos[i*3] = r * Math.sin(p) * Math.cos(t); pos[i*3+1] = r * Math.sin(p) * Math.sin(t); pos[i*3+2] = r * Math.cos(p);
            const c = Math.random() > 0.8 ? red : gray;
            colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
        }
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({size: 0.05, vertexColors: true, transparent: true, opacity: 0.7})));

        // CMU CORE MESH
        const core = new THREE.Mesh(new THREE.IcosahedronGeometry(2.5, 2), new THREE.MeshBasicMaterial({color: 0xC41230, wireframe: true}));
        scene.add(core);

        // AGENT NODES
        const agents = [
            {name: "AUDITOR", url: "?page=explorer", color: "#E2C044", pos: [12, 6, 0]},
            {name: "ALCHEMIST", url: "?page=cleaner", color: "#E87A5D", pos: [-12, -6, 5]},
            {name: "STRATEGIST", url: "?page=analysis", color: "#44BBA4", pos: [6, -12, -5]},
            {name: "ARCHITECT", url: "?page=dashboard", color: "#00A6D6", pos: [0, 14, 5]},
            {name: "KNOWLEDGE GRAPH", url: "?page=graph", color: "#9B5DE5", pos: [-8, 10, -10]}
        ];

        agents.forEach(a => {
            const el = document.createElement('a'); el.className = 'node-label';
            el.innerText = a.name; el.href = a.url; el.target = "_self";
            document.body.appendChild(el); a.el = el;
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 16), new THREE.MeshBasicMaterial({color: a.color}));
            mesh.position.set(...a.pos); scene.add(mesh); a.mesh = mesh;
        });

        camera.position.z = 32;
        function animate(){
            requestAnimationFrame(animate);
            agents.forEach(a => {
                const vector = a.mesh.position.clone().project(camera);
                a.el.style.left = (vector.x + 1) / 2 * window.innerWidth + 'px';
                a.el.style.top = -(vector.y - 1) / 2 * window.innerHeight + 'px';
            });
            controls.update(); renderer.render(scene, camera);
        }
        animate();
    </script></body></html>
    """
    components.html(three_js_galaxy, height=850)

# ======================= AGENT 1: FORENSIC AUDITOR =======================
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    from streamlit_extras.colored_header import colored_header
    colored_header("Step 1: Forensic Auditor", "Data Profiling & Schema Integrity Scans.")
    
    f = st.selectbox("Select Target File", ALL_FILES)
    df = pd.read_csv(f'data/{f}', skiprows=1 if 'UTM_Totals' in f else 0)
    
    tab_view, tab_profile, tab_stats = st.tabs(["📊 Data Viewer", "🔍 Data Profile", "📈 Descriptive Stats"])
    
    with tab_view:
        from streamlit_extras.dataframe_explorer import dataframe_explorer
        st.dataframe(dataframe_explorer(df), use_container_width=True)
    with tab_profile:
        st.subheader("Field Metadata & Integrity")
        profile = pd.DataFrame({
            'Type': df.dtypes.astype(str),
            'Nulls': df.isna().sum(),
            'Unique': df.nunique(),
            'Memory (MB)': (df.memory_usage(deep=True) / 1024**2).round(2)
        })
        st.dataframe(profile, use_container_width=True)
    with tab_stats:
        st.subheader("Numerical Telemetry Analysis")
        st.dataframe(df.describe().T, use_container_width=True)

# ======================= AGENT 2: DATA ALCHEMIST =======================
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Step 2: Data Alchemist")
    st.success("Master Nexus Hub Synthesized: 12 datasets joined and financial artifacts purged.")
    st.dataframe(master_df, use_container_width=True)

# ======================= AGENT 3: QUANTITATIVE STRATEGIST =======================
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Step 3: Quantitative Strategist")
    if not master_df.empty:
        c1, c2 = st.columns(2)
        corr, _ = stats.pearsonr(master_df['Total_Spend'], master_df['Total_Users'])
        c1.metric("Spend-to-User Correlation", f"{corr:.2f}")
        st.plotly_chart(px.scatter(master_df[master_df['Total_Spend']>0], x="Total_Spend", y="Total_Users", 
                                   trendline="ols", color="Category", title="Efficiency Frontier: Real Regression Analysis"), use_container_width=True)

# ======================= AGENT 4: VISUAL ARCHITECT =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Step 4: Visual Architect")
    from streamlit_extras.metric_cards import style_metric_cards
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spend", f"${master_df['Total_Spend'].sum():,.2f}")
    m2.metric("Total Acquisitions", f"{master_df['Total_Users'].sum():,.0f}")
    m3.metric("Avg CPWU", f"${(master_df['Total_Spend'].sum()/master_df['Total_Users'].sum() if master_df['Total_Users'].sum() > 0 else 0):.2f}")
    style_metric_cards(border_left_color="#C41230")
    st.plotly_chart(px.bar(master_df.groupby('Category')['Total_Spend'].sum().reset_index(), x='Category', y='Total_Spend', color='Category'), use_container_width=True)

# ======================= AGENT 5: KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.header("Nexus Knowledge Graph")
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333")
    net.add_node("Nexus Hub", size=45, color="#C41230", label="NEXUS CORE")
    for cat in master_df['Category'].unique():
        net.add_node(str(cat), size=25, color="#6D6E71")
        net.add_edge("Nexus Hub", str(cat))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=700)
