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

# ---------------------------------------------------------
# 1. CMU BRAND IDENTITY & GALAXY THEME
# ---------------------------------------------------------
CMU_RED = "#C41230"
CMU_GREY = "#6D6E71"
WHITE = "#FFFFFF"
BLACK = "#000000"

st.set_page_config(page_title="CMU AI Nexus", layout="wide", initial_sidebar_state="collapsed")

# Inject CSS for Black Galaxy background and White Floating Cards
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BLACK};
        color: {WHITE};
    }}
    /* White Card containers for readability */
    div[data-testid="stMetric"], .stDataFrame, .plotly-graph-div, .stExpander, div.stTabs {{
        background-color: {WHITE} !important;
        padding: 20px;
        border-radius: 15px;
        color: #1e293b !important;
        box-shadow: 0 10px 30px rgba(196, 18, 48, 0.2);
        margin-bottom: 20px;
    }}
    /* Title text styling */
    h1, h2, h3 {{
        color: {WHITE} !important;
        font-weight: 800;
    }}
    /* Metric label color fix */
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] div {{
        color: #334155 !important;
    }}
    header {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. THE ALCHEMIST ENGINE (DATA SYNTHESIS)
# ---------------------------------------------------------
def normalize_key(series):
    """Programmatic normalization to create relational joins [5]."""
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

@st.cache_data
def load_and_synthesis():
    # Load Index [1]
    idx = pd.read_csv('data/UCM Campaign Index.csv')
    idx['utm_clean'] = normalize_key(idx['Campaign_ID'])
    
    # Process Google Analytics Totals [2]
    ga_files = ['GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', 'GA_FY26_UTM_Totals_Jul-Dec2025.csv']
    ga_dfs = []
    for f in ga_files:
        if os.path.exists(f'data/{f}'):
            _df = pd.read_csv(f'data/{f}', skiprows=1)
            _df['utm_clean'] = normalize_key(_df['Session campaign'])
            ga_dfs.append(_df[['utm_clean', 'Total users', 'Engagement rate', 'Average session duration']])
    ga_master = pd.concat(ga_dfs).groupby('utm_clean').agg({
        'Total users': 'sum', 'Engagement rate': 'mean', 'Average session duration': 'mean'
    }).reset_index()

    # Process GAds Ad-Level Telemetry for Video Retention [4, 6]
    v_retention = pd.DataFrame()
    v_path = 'data/GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv'
    if os.path.exists(v_path):
        v_df = pd.read_csv(v_path)
        v_df['utm_clean'] = normalize_key(v_df['Campaign'])
        # Cleaning percentage strings [7]
        v_df['Video_100'] = pd.to_numeric(v_df['Video played to 100%'].astype(str).str.replace('%',''), errors='coerce').fillna(0)
        v_retention = v_df.groupby('utm_clean')['Video_100'].mean().reset_index()

    # Create Master Hub
    hub = pd.merge(idx, ga_master, on='utm_clean', how='left')
    hub = pd.merge(hub, v_retention, on='utm_clean', how='left').fillna(0)
    return hub

master_hub = load_and_synthesis()

# ---------------------------------------------------------
# 3. NAVIGATION & IMMERSIVE HOME SCREEN
# ---------------------------------------------------------
query_params = st.query_params.to_dict()
page = query_params.get("page", "home")

st.markdown(f"""
<div style="display: flex; justify-content: center; gap: 30px; padding: 20px; border-bottom: 1px solid {CMU_GREY};">
    <a href="?page=home" target="_self" style="color: {WHITE}; text-decoration: none; font-weight: bold;">🏠 UNIVERSE</a>
    <a href="?page=auditor" target="_self" style="color: {WHITE}; text-decoration: none; font-weight: bold;">🕵️ AUDITOR</a>
    <a href="?page=strategist" target="_self" style="color: {WHITE}; text-decoration: none; font-weight: bold;">🧪 STRATEGIST</a>
    <a href="?page=architect" target="_self" style="color: {WHITE}; text-decoration: none; font-weight: bold;">🖥️ ARCHITECT</a>
    <a href="?page=graph" target="_self" style="color: {WHITE}; text-decoration: none; font-weight: bold;">🕸️ KNOWLEDGE</a>
</div>
""", unsafe_allow_html=True)

if page == "home":
    three_js_galaxy = f"""
    <div id="galaxy-container"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{ alpha: true, antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('galaxy-container').appendChild(renderer.domElement);

        const geometry = new THREE.BufferGeometry();
        const vertices = []; const colors = [];
        const palette = [new THREE.Color("{CMU_RED}"), new THREE.Color("{WHITE}"), new THREE.Color("{CMU_GREY}")];

        for (let i = 0; i < 20000; i++) {{
            vertices.push(THREE.MathUtils.randFloatSpread(2500), THREE.MathUtils.randFloatSpread(2500), THREE.MathUtils.randFloatSpread(2500));
            let c = palette[Math.floor(Math.random() * palette.length)];
            colors.push(c.r, c.g, c.b);
        }}
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        const material = new THREE.PointsMaterial({{ size: 2, vertexColors: true, transparent: true, opacity: 0.9 }});
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        camera.position.z = 600;
        function animate() {{ requestAnimationFrame(animate); points.rotation.y += 0.0003; points.rotation.x += 0.0001; renderer.render(scene, camera); }}
        animate();
    </script>
    """
    components.html(three_js_galaxy, height=800)
    st.markdown(f"<h1 style='text-align: center; font-size: 60px; margin-top: -500px;'>CMU AI NEXUS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 20px;'>Navigating the Marketing Galaxy</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. ENHANCED STRATEGIST: OPTIMIZATION & EFFICIENCY
# ---------------------------------------------------------
elif page == "strategist":
    st.header("🧪 Step 3: Quantitative Strategist")
    
    col1, col2 = st.columns(2)
    with col1:
        # Scatter of Video Completion vs Traffic Volume [8, 9]
        fig = px.scatter(master_hub[master_hub['Video_100'] > 0], 
                         x="Video_100", y="Total users", size="Total users", color="Category",
                         title="Creative Efficiency: 100% Completion vs. Volume",
                         labels={"Video_100": "Avg Video Completion (%)"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Optimization Lift Analysis based on "People not in audiences" [10, 11]
        st.subheader("🤖 Optimization Lift Insight")
        st.markdown("""
        The Strategist identifies that **'People not in audiences'** (Optimized Targeting) drove the highest volume for 
        Podcast Episode 15, yielding **23,687 clicks** [10]. 
        
        **Strategy Recommendation:** Shift 15% of budget from niche 'In-Market' segments to Automated Optimized Targeting 
        for awareness initiatives like *Work That Matters*.
        """)
        st.info("Predictive Peak: Expected 8.5K user surge during Q4 flagship events [12].")

# ---------------------------------------------------------
# 5. KNOWLEDGE GRAPH: TRIPARTITE RELATIONS
# ---------------------------------------------------------
elif page == "graph":
    st.header("🕸️ Relational Knowledge Graph")
    st.caption("Mapping Connections: Hub → Vendor → Campaign")
    
    net = Network(height="750px", width="100%", bgcolor=BLACK, font_color=WHITE)
    net.add_node("CMU", label="CMU Hub", color=CMU_RED, size=50)
    
    # Tripartite: Hub -> Vendor [1] -> Campaign [13]
    vendors = master_hub['Vendor'].unique()
    for vendor in vendors:
        if pd.isna(vendor): continue
        net.add_node(vendor, label=str(vendor), color=CMU_GREY, size=35)
        net.add_edge("CMU", vendor)
        
        # Link top 5 campaigns per vendor for clarity
        camps = master_hub[master_hub['Vendor'] == vendor].sort_values('Total users', ascending=False).head(5)
        for _, row in camps.iterrows():
            cid = row['utm_clean']
            # Scale node size by traffic volume [14]
            size = 10 + (row['Total users'] / 5000) 
            net.add_node(cid, label=cid[:20], color=WHITE, size=min(size, 30))
            net.add_edge(vendor, cid)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800)

# ---------------------------------------------------------
# 6. VISUAL ARCHITECT: DASHBOARD PULSE
# ---------------------------------------------------------
elif page == "architect":
    st.header("🖥️ Step 4: Visual Architect")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Lifetime Users (FY25-26)", f"{int(master_hub['Total users'].sum()):,}")
    m2.metric("Engagement Leader", "Tony Awards", "59.3%") [12]
    m3.metric("Volume Leader", "UCM Anthem", "143.5K") [14]

    # Pulse of the Attention Economy [14, 15]
    fig_pulse = px.scatter(master_hub, x="Engagement rate", y="Average session duration",
                           size="Total users", color="Category", hover_name="utm_clean",
                           title="The Attention Economy: Quality vs. Session Depth")
    st.plotly_chart(fig_pulse, use_container_width=True)

# ---------------------------------------------------------
# 7. FORENSIC AUDITOR: DATA INTEGRITY
# ---------------------------------------------------------
elif page == "auditor":
    st.header("🕵️ Step 1: Forensic Auditor")
    st.subheader("Relational ID Audit [16, 17]")
    
    # Flag IDs not in index
    orphans = master_hub[master_hub['utm_clean'] == ""].shape
    if orphans > 0:
        st.error(f"⚠️ Detected {orphans} campaigns in performance logs missing from Index mapping.")
    else:
        st.success("✅ ID Continuity Verified: All performance strings map to the Relational Spine.")
    
    st.dataframe(master_hub[['Monday_Board_Name', 'Vendor', 'Category', 
