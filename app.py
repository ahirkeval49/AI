import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
import re

# ---------------------------------------------------------
# CMU BRAND COLORS & UI CONFIG
# ---------------------------------------------------------
CMU_RED = "#C41230"
CMU_GREY = "#6D6E71"
WHITE = "#FFFFFF"
BLACK = "#0f0f0f" 
CARD_BG = "#FFFFFF" 
TEXT_DARK = "#050505"

st.set_page_config(page_title="CMU Command Center", layout="wide", initial_sidebar_state="collapsed")

# Optimized CSS for Speed and Contrast
st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; color: {WHITE}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {CMU_RED} !important; font-weight: 800; }}
    
    .console-card {{
        background-color: {CARD_BG}; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 24px;
        border-top: 3px solid {CMU_RED};
    }}
    
    /* AUDITOR: Force dark text on inputs/tables */
    [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
    .console-card p, .console-card label, .console-card div {{ color: {TEXT_DARK} !important; }}
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {{ color: {TEXT_DARK} !important; font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ color: {CMU_GREY} !important; font-weight: 700; }}
    div[data-testid="stMetric"] {{ background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid {CMU_RED}; }}

    /* Tab Speed Optimization */
    button[data-baseweb="tab"] {{ font-weight: 700 !important; text-transform: uppercase !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA PIPELINE (Cached for Speed)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_master_data():
    # Helper to clean numbers and normalize keys
    def clean(s): return pd.to_numeric(s.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)
    def norm(s): return s.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    try:
        idx = pd.read_csv('UCM Campaign Index.csv')
        idx['board_key'] = norm(idx['Monday_Board_Name'])
        idx['ga_key'] = norm(idx['UTM campaign'])

        # Spend Aggregation
        p_files = ['GAds_FY25_Totals_Jul2024-Jun2025.csv', 'GAds_FY26_Totals_Jul-Dec2025.csv', 'LinkedIn_Ad_Performance_Feb2024_Dec2025.csv']
        p_dfs = []
        for f in p_files:
            if os.path.exists(f):
                tmp = pd.read_csv(f)
                tmp = tmp[~tmp.iloc[:,1].astype(str).str.contains('Total', case=False, na=False)]
                c_name = 'Campaign' if 'Campaign' in tmp.columns else 'Campaign Name'
                res = pd.DataFrame({'board_key': norm(tmp[c_name])})
                res['Spend'] = clean(tmp['Cost']) if 'Cost' in tmp.columns else clean(tmp.get('Total Spend', pd.Series(0)))
                res['Clicks'] = clean(tmp['Clicks'])
                p_dfs.append(res)
        
        plat = pd.concat(p_dfs).groupby('board_key').sum().reset_index()

        # GA Aggregation
        ga_files = ['GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', 'GA_FY26_UTM_Totals_Jul-Dec2025.csv']
        ga_dfs = []
        for f in ga_files:
            if os.path.exists(f):
                tmp = pd.read_csv(f)
                res = pd.DataFrame({'ga_key': norm(tmp['Session campaign']), 'Users': clean(tmp['Total users']), 'Dur': clean(tmp['Average session duration'])})
                ga_dfs.append(res)
        
        ga = pd.concat(ga_dfs).groupby('ga_key').agg({'Users':'sum', 'Dur':'mean'}).reset_index()

        master = pd.merge(idx, plat, on='board_key', how='outer')
        master = pd.merge(master, ga, on='ga_key', how='outer')
        master.fillna(0, inplace=True)
        master['CPQM'] = np.where(master['Users']>0, master['Spend']/((master['Users']*master['Dur'])/60+1), 0)
        return master
    except: return pd.DataFrame()

df = get_master_data()

# ---------------------------------------------------------
# ROUTING ENGINE (Listener for 3D Nodes)
# ---------------------------------------------------------
# Check URL params to see if we clicked a 3D node
query_params = st.query_params
default_index = 0
if "page" in query_params:
    page_map = {"home": 0, "explorer": 1, "dashboard": 2, "analysis": 3, "graph": 4}
    default_index = page_map.get(query_params["page"], 0)

# ---------------------------------------------------------
# UI: NAVIGATION TABS
# ---------------------------------------------------------
tabs = st.tabs(["🌌 Nexus", "🕵️ Auditor", "🖥️ Dashboard", "🧪 Strategist", "🕸️ Knowledge Graph"])

# ======================= TAB: NEXUS =======================
with tabs[0]:
    st.markdown(f"<h1 style='text-align: center; font-size: 60px;'>CMU COMMAND CENTER</h1>", unsafe_allow_html=True)
    
    # 3D Galaxy with TOP-LEVEL navigation enabled
    galaxy_js = f"""
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>body {{ margin: 0; background: {BLACK}; overflow: hidden; }}
    .node-label {{ position: absolute; background: rgba(196,18,48,0.9); border: 2px solid {WHITE}; padding: 8px 16px; border-radius: 8px; font-weight: bold; cursor: pointer; color: {WHITE}; text-decoration: none; font-size: 12px; }}
    </style></head>
    <body><script>
        const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true}}); renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement); controls.autoRotate = true;

        // Particle background
        const pos = new Float32Array(15000 * 3); for(let i=0; i<45000; i++) pos[i] = (Math.random()-0.5) * 100;
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({{size: 0.05, color: 0x666666}})));

        // CMU Center
        const loader = new THREE.FontLoader();
        loader.load('https://unpkg.com/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json', (font) => {{
            const tGeo = new THREE.TextGeometry('CMU', {{ font: font, size: 5, height: 1 }});
            tGeo.computeBoundingBox(); tGeo.translate(-0.5*(tGeo.boundingBox.max.x-tGeo.boundingBox.min.x), -2.5, 0);
            scene.add(new THREE.Mesh(tGeo, new THREE.MeshPhongMaterial({{color: "{CMU_RED}"}})));
        }});

        const nodes = [
            {{name: "Auditor", target: "explorer", pos: [15, 5, 0]}},
            {{name: "Dashboard", target: "dashboard", pos: [-15, -5, 0]}},
            {{name: "Strategist", target: "analysis", pos: [0, 12, -5]}},
            {{name: "Knowledge Graph", target: "graph", pos: [0, -12, 5]}}
        ];

        nodes.forEach(n => {{
            const el = document.createElement('a'); el.className = 'node-label'; el.innerText = n.name;
            // CRITICAL FIX: Direct top-level window navigation
            el.href = window.location.origin + window.location.pathname + "?page=" + n.target;
            el.target = "_top";
            document.body.appendChild(el); n.el = el;
            const m = new THREE.Mesh(new THREE.SphereGeometry(1.2, 32, 32), new THREE.MeshPhongMaterial({{color: 0x666666}}));
            m.position.set(...n.pos); scene.add(m); n.mesh = m;
        }});

        camera.position.z = 35;
        function animate(){{ requestAnimationFrame(animate); 
            nodes.forEach(n => {{ const v = n.mesh.position.clone().project(camera);
                n.el.style.left = (v.x + 1) / 2 * window.innerWidth + 'px';
                n.el.style.top = -(v.y - 1) / 2 * window.innerHeight + 'px';
            }});
            controls.update(); renderer.render(scene, camera); 
        }} animate();
    </script></body></html>
    """
    components.html(galaxy_js, height=700)

# ======================= TAB: AUDITOR =======================
with tabs[1]:
    st.markdown("<div class='console-card'><h3>Planning Phase Inspector</h3>", unsafe_allow_html=True)
    all_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    f = st.selectbox("Select Raw File to Audit", all_files)
    if f:
        raw_df = pd.read_csv(f)
        st.write(f"**Filename:** {f}")
        st.dataframe(raw_df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= TAB: DASHBOARD =======================
with tabs[2]:
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spend", f"${df['Spend'].sum():,.0f}")
        c2.metric("Total Acquired Users", f"{df['Users'].sum():,.0f}")
        c3.metric("Avg CPA", f"${(df['Spend'].sum()/df['Users'].sum()):.2f}")

        st.markdown("<div class='console-card'><h3>Investment Concentration</h3>", unsafe_allow_html=True)
        fig = px.bar(df.sort_values('Spend', ascending=False).head(15), x='Monday_Board_Name', y='Spend', color_discrete_sequence=[CMU_RED])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=TEXT_DARK)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================= TAB: STRATEGIST =======================
with tabs[3]:
    st.markdown("<div class='console-card'><h3>Efficiency & AttentionROI</h3>", unsafe_allow_html=True)
    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("#### Cost Per Quality Minute")
        st.dataframe(df[df['CPQM']>0].sort_values('CPQM').head(10)[['Monday_Board_Name', 'CPQM', 'Spend']], use_container_width=True)
    with c_r:
        st.markdown("#### Diminishing Returns")
        fig_fatigue = px.scatter(df[df['Spend']>0], x='Spend', y='Users', hover_name='Monday_Board_Name', trendline="lowess", color_discrete_sequence=[CMU_RED])
        st.plotly_chart(fig_fatigue, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= TAB: GRAPH =======================
with tabs[4]:
    st.markdown("<div class='console-card'><h3>Relational Universe</h3>", unsafe_allow_html=True)
    net = Network(height="700px", width="100%", bgcolor=WHITE, font_color=TEXT_DARK)
    net.add_node("CMU", size=50, color=CMU_RED, label="CMU Hub")
    for v in df['Vendor'].unique():
        if str(v) != '0':
            net.add_node(str(v), size=30, color="#555", label=str(v))
            net.add_edge("CMU", str(v))
            subset = df[df['Vendor']==v].sort_values('Users', ascending=False).head(10)
            for _, r in subset.iterrows():
                net.add_node(str(r['Monday_Board_Name']), size=15, color=CMU_RED, label=str(r['Monday_Board_Name'])[:20])
                net.add_edge(str(v), str(r['Monday_Board_Name']))
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        st.components.v1.html(open(tmp.name, 'r').read(), height=750)
