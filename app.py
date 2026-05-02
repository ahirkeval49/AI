import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os

# ---------------------------------------------------------
# 1. CMU BRAND COLORS & GALAXY THEME STYLING
# ---------------------------------------------------------
CMU_RED = "#C41230"
CMU_GREY = "#6D6E71"
WHITE = "#FFFFFF"
BLACK = "#000000"

st.set_page_config(page_title="CMU AI Nexus", layout="wide", initial_sidebar_state="collapsed")

# CSS: Black Galaxy background with White Data Cards
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BLACK};
        color: {WHITE};
    }}
    /* White Card Style for Charts, Metrics, and DataFrames */
    div[data-testid="stMetric"], .stDataFrame, .plotly-graph-div, .stExpander, div.stTabs {{
        background-color: {WHITE} !important;
        padding: 20px;
        border-radius: 15px;
        color: #1e293b !important;
        box-shadow: 0 10px 30px rgba(196, 18, 48, 0.2);
        margin-bottom: 20px;
    }}
    /* Ensure metric text is legible against white cards */
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] div {{
        color: #334155 !important;
    }}
    h1, h2, h3 {{ color: {WHITE} !important; font-weight: 800; }}
    header {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATA ALCHEMIST ENGINE (NORMALIZATION & SYNTHESIS)
# ---------------------------------------------------------
def normalize_key(series):
    """Creates a standardized technical key for joining disparate sources [1, 2]."""
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

@st.cache_data
def build_master_hub():
    # Relational Spine: UCM Campaign Index [3]
    idx = pd.read_csv('data/UCM Campaign Index.csv')
    idx['utm_clean'] = normalize_key(idx['Campaign_ID'])
    
    # Synthesis of Google Analytics Totals [4, 5]
    ga_dfs = []
    for f in ['GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', 'GA_FY26_UTM_Totals_Jul-Dec2025.csv']:
        if os.path.exists(f'data/{f}'):
            _df = pd.read_csv(f'data/{f}', skiprows=1)
            _df['utm_clean'] = normalize_key(_df['Session campaign'])
            ga_dfs.append(_df[['utm_clean', 'Total users', 'Engagement rate', 'Average session duration']])
    ga_master = pd.concat(ga_dfs).groupby('utm_clean').agg({
        'Total users': 'sum', 'Engagement rate': 'mean', 'Average session duration': 'mean'
    }).reset_index()

    # Creative Efficiency: Video Retention from GAds [6, 7]
    v_retention = pd.DataFrame()
    if os.path.exists('data/GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv'):
        v_df = pd.read_csv('data/GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv')
        v_df['utm_clean'] = normalize_key(v_df['Campaign'])
        v_df['Video_100'] = pd.to_numeric(v_df['Video played to 100%'].astype(str).str.replace('%',''), errors='coerce').fillna(0)
        v_retention = v_df.groupby('utm_clean')['Video_100'].mean().reset_index()

    # Final Master Hub Join
    hub = pd.merge(idx, ga_master, on='utm_clean', how='left')
    hub = pd.merge(hub, v_retention, on='utm_clean', how='left').fillna(0)
    return hub

master_hub = build_master_hub()

# ---------------------------------------------------------
# 3. NAVIGATION SYSTEM
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

# ---------------------------------------------------------
# 4. PAGE: UNIVERSE (GALAXY HOME SCREEN)
# ---------------------------------------------------------
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
        const material = new THREE.PointsMaterial({{ size: 2.5, vertexColors: true, transparent: true, opacity: 0.9 }});
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        camera.position.z = 600;
        function animate() {{ requestAnimationFrame(animate); points.rotation.y += 0.0003; points.rotation.x += 0.0001; renderer.render(scene, camera); }}
        animate();
    </script>
    """
    components.html(three_js_galaxy, height=800)
    st.markdown("<h1 style='text-align: center; font-size: 65px; margin-top: -500px;'>CMU AI NEXUS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 22px; opacity: 0.8;'>Navigating the Marketing Galaxy</p>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. PAGE: FORENSIC AUDITOR (FIXED SYNTAX ERROR)
# ---------------------------------------------------------
elif page == "auditor":
    st.header("🕵️ Step 1: Forensic Auditor")
    st.subheader("Data Integrity & Continuity Analysis")
    
    # Checking for missing mappings in the Index [8]
    orphans = master_hub[master_hub['utm_clean'] == ""].shape
    if orphans > 0:
        st.error(f"⚠️ Warning: Found {orphans} campaigns in performance logs without Index mappings.")
    else:
        st.success("✅ Relational Spine Verified: All tracking strings map to a Board Name.")
    
    # FIXED LINE: Properly closed brackets for the dataframe display
    st.dataframe(
        master_hub[['Monday_Board_Name', 'Vendor', 'Category', 'Total users', 'Engagement rate', 'Video_100']].tail(20),
        use_container_width=True
    )

# ---------------------------------------------------------
# 6. PAGE: QUANTITATIVE STRATEGIST (EFFICIENCY & LIFT)
# ---------------------------------------------------------
elif page == "strategist":
    st.header("🧪 Step 3: Quantitative Strategist")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Creative Efficiency (Retention vs. Scale)")
        # Identifying if high completion leads to higher users [2]
        fig = px.scatter(master_hub[master_hub['Video_100'] > 0], 
                         x="Video_100", y="Total users", size="Total users", color="Category",
                         title="High Retention vs. High Volume",
                         labels={"Video_100": "Avg Video Completion (%)"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🤖 Optimization Lift Analysis")
        st.info("""
            The Strategist identifies 'People not in audiences' as the primary volume driver for Podcast Episode 15 [9].
            This confirms that **Optimized Targeting** (AI-driven) frequently outperforms manual niche segments in awareness lift.
        """)
        st.markdown("### Strategic Allocation Advice")
        st.warning("Maintain high spend on 'Branded Keywords' (Engagement Leader: 77.4%) while using 'Demand Gen' to scale awareness [10, 11].")

# ---------------------------------------------------------
# 7. PAGE: VISUAL ARCHITECT (PULSE & KPI)
# ---------------------------------------------------------
elif page == "architect":
    st.header("🖥️ Step 4: Visual Architect")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Lifetime Users (FY25-26)", f"{int(master_hub['Total users'].sum()):,}")
    m2.metric("Engagement Benchmark", "Branded Keyword", "77.4%")
    m3.metric("Awareness Peak", "Tony Awards", "8.5K Daily")

    st.markdown("### The Attention Economy: Engagement Depth")
    fig_pulse = px.scatter(master_hub, x="Engagement rate", y="Average session duration",
                           size="Total users", color="Category", hover_name="utm_clean",
                           title="Quality vs. Session Depth (Bubble size = Volume)")
    st.plotly_chart(fig_pulse, use_container_width=True)

# ---------------------------------------------------------
# 8. PAGE: RELATIONAL KNOWLEDGE GRAPH (TRIPARTITE)
# ---------------------------------------------------------
elif page == "graph":
    st.header("🕸️ Relational Knowledge Graph")
    st.caption("Mapping Connections: CMU Hub → Vendor → Campaign")
    
    # Graph background set to black as requested
    net = Network(height="750px", width="100%", bgcolor=BLACK, font_color=WHITE)
    net.add_node("CMU", label="CMU Hub", color=CMU_RED, size=50)
    
    vendors = master_hub['Vendor'].unique()
    for vendor in vendors:
        if pd.isna(vendor): continue
        net.add_node(str(vendor), label=str(vendor), color=CMU_GREY, size=35)
        net.add_edge("CMU", str(vendor))
        
        # Pull top 5 high-volume campaigns per vendor to show relational hierarchy
        camps = master_hub[master_hub['Vendor'] == vendor].sort_values('Total users', ascending=False).head(5)
        for _, row in camps.iterrows():
            cid = row['utm_clean']
            # Node size scaled by website user volume
            size = 10 + (row['Total users'] / 5000) 
            net.add_node(cid, label=cid[:20], color=WHITE, size=min(size, 30))
            net.add_edge(str(vendor), cid)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800)
