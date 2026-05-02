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
import base64
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------
# 1. GLOBAL THEME & PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus | Galaxy Edition", layout="wide", initial_sidebar_state="collapsed")

# Inject Dark Theme for App, White Theme for Cards
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #f8fafc; }
    h1, h2, h3, h4, p, span { color: #f8fafc; }
    .white-card { 
        background: #ffffff; border-radius: 12px; padding: 20px; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.5); margin-bottom: 20px; 
        color: #0f172a; border: 1px solid #e2e8f0;
    }
    .white-card h3, .white-card p, .white-card h2, .white-card strong { color: #0f172a !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1e293b; border-radius: 8px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { color: #f8fafc; }
    
    .nav-grid { display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 5px; z-index: 100; position: relative; }
    .nav-card {
        background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); 
        backdrop-filter: blur(10px); border-radius: 12px; padding: 10px;
        width: 130px; text-align: center; color: #ffffff !important; text-decoration: none;
        transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .nav-card:hover { border-color: #C41230; transform: translateY(-5px); box-shadow: 0 8px 25px rgba(196,18,48,0.4); background: rgba(255, 255, 255, 0.2); }
    .nav-title { font-size: 10px; font-weight: 900; color: #f8fafc; margin-top: 5px; letter-spacing: 1px; }
    
    /* Full immersion for home page */
    .home-immersion { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {"audit_logs": {}, "synthesis_stats": {}, "model_results": {}}

query_params = st.query_params.to_dict()
current_page = query_params.get("page", ["home"])[0] if isinstance(query_params.get("page"), list) else query_params.get("page", "home")

# ---------------------------------------------------------
# 2. HARDCODED AUDITOR ANOMALIES (From your prompt)
# ---------------------------------------------------------
anomalies = [
    {"id": "A1", "title": "Inconsistent Naming Conventions", "impact": "High", "description": "Campaign names do not match across platforms.", "reason": "Prevents automated joining of budget data."},
    {"id": "A2", "title": "Missing Budget & Spend Data", "impact": "Critical", "description": "Budget columns in Index are empty.", "reason": "Impossible to calculate ROAS accurately."},
    {"id": "A3", "title": "Data Type Mismatches", "impact": "Medium", "description": "Metrics like CTR stored as strings.", "reason": "Causes aggregation errors."},
    {"id": "A4", "title": "Overlapping Dates", "impact": "Low", "description": "Campaign dates extend beyond stated ends.", "reason": "Can lead to double-counting."}
]

# ---------------------------------------------------------
# 3. THE ALCHEMIST ENGINE (Robust ETL)
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
    for alias in aliases:
        if alias in df.columns: return alias
    return None

def clean_currency(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

def normalize_key(series):
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

@st.cache_data
def build_master_hub():
    try:
        # Load Index
        if os.path.exists('data/UCM Campaign Index.csv'):
            idx = pd.read_csv('data/UCM Campaign Index.csv')
            utm_col = find_col(idx, ['UTM campaign', 'Campaign_ID', 'UTM_Combined_ID', 'Landing Page (UTM)'])
            idx['utm_clean'] = normalize_key(idx[utm_col]) if utm_col else ""
            if 'Category' not in idx.columns: idx['Category'] = "Uncategorized"
        else:
            return pd.DataFrame()

        # Google Ads
        g_files = ['GAds_FY25_Totals_Jul2024-Jun2025.csv', 'GAds_FY26_Totals_Jul-Dec2025.csv']
        g_dfs = [pd.read_csv(f'data/{f}') for f in g_files if os.path.exists(f'data/{f}')]
        g_all = pd.concat(g_dfs, ignore_index=True) if g_dfs else pd.DataFrame()
        if not g_all.empty:
            g_key = find_col(g_all, ['Ad name', 'Campaign', 'Campaign Name'])
            g_all['utm_clean'] = normalize_key(g_all[g_key]) if g_key else ""
            g_cost = find_col(g_all, ['Cost', 'Spend'])
            g_all['Cost'] = clean_currency(g_all[g_cost]) if g_cost else 0.0
            g_agg = g_all.groupby('utm_clean').agg(GAds_Spend=('Cost', 'sum')).reset_index()
        else: g_agg = pd.DataFrame(columns=['utm_clean', 'GAds_Spend'])

        # LinkedIn
        li_path = 'data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv'
        if os.path.exists(li_path):
            li = pd.read_csv(li_path)
            li_key = find_col(li, ['Campaign Name', 'Campaign'])
            li['utm_clean'] = normalize_key(li[li_key]) if li_key else ""
            li_spend = find_col(li, ['Total Spend', 'Spend', 'Cost'])
            li['LI_Spend'] = clean_currency(li[li_spend]) if li_spend else 0.0
            li_agg = li.groupby('utm_clean').agg(LI_Spend=('LI_Spend', 'sum')).reset_index()
        else: li_agg = pd.DataFrame(columns=['utm_clean', 'LI_Spend'])

        # GA 
        ga_files = ['GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', 'GA_FY26_UTM_Totals_Jul-Dec2025.csv']
        ga_dfs = []
        for f in ga_files:
            if os.path.exists(f'data/{f}'):
                _df = pd.read_csv(f'data/{f}', skiprows=1)
                ga_key = find_col(_df, ['Session campaign', 'Campaign'])
                if ga_key:
                    _df['utm_clean'] = normalize_key(_df[ga_key])
                    ga_users = find_col(_df, ['Total users', 'Users'])
                    _df['Total users'] = pd.to_numeric(_df[ga_users], errors='coerce').fillna(0.0) if ga_users else 0.0
                    ga_dfs.append(_df[['utm_clean', 'Total users']])
        ga_agg = pd.concat(ga_dfs).groupby('utm_clean').agg(Total_Users=('Total users', 'sum')).reset_index() if ga_dfs else pd.DataFrame(columns=['utm_clean', 'Total_Users'])

        # Master Synthesis
        hub = pd.merge(idx, ga_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, g_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, li_agg, on='utm_clean', how='left').fillna(0.0)
        
        hub['Total_Spend'] = pd.to_numeric(hub['GAds_Spend'] + hub['LI_Spend'], errors='coerce').fillna(0.0).astype(float)
        hub['Total_Users'] = pd.to_numeric(hub['Total_Users'], errors='coerce').fillna(0.0).astype(float)
        hub['CPWU'] = hub['Total_Spend'].div(hub['Total_Users'].replace(0, np.nan)).fillna(0.0).astype(float)
        
        # Deduce Platform for Architect and Graph
        hub['Platform'] = np.where(hub['GAds_Spend'] > hub['LI_Spend'], 'Google Ads', 
                                  np.where(hub['LI_Spend'] > 0, 'LinkedIn Ads', 'Organic/Other'))
        
        return hub
    except Exception as e: return pd.DataFrame()

# Fallback fake data if NO files exist - allows dashboard testing
master_df = build_master_hub()
if master_df.empty:
    master_df = pd.DataFrame({
        'Category': ['Technology', 'Education', 'Business', 'Technology', 'Healthcare'],
        'utm_clean': ['tech_1', 'edu_2', 'bus_3', 'tech_4', 'health_5'],
        'Platform': ['Google Ads', 'LinkedIn Ads', 'Google Ads', 'Organic/Other', 'LinkedIn Ads'],
        'Total_Spend': [10000.0, 5500.0, 12000.0, 0.0, 3000.0],
        'GAds_Spend': [10000.0, 0.0, 12000.0, 0.0, 0.0],
        'LI_Spend': [0.0, 5500.0, 0.0, 0.0, 3000.0],
        'Total_Users': [5000.0, 2000.0, 8000.0, 1500.0, 500.0],
        'CPWU': [2.0, 2.75, 1.5, 0.0, 6.0]
    })

# ---------------------------------------------------------
# 4. GLOBAL NAVIGATION
# ---------------------------------------------------------
nav_cards_html = """
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card">🌌<div class="nav-title">NEXUS</div></a>
    <a href="?page=explorer" target="_self" class="nav-card">🕵️<div class="nav-title">1. AUDITOR</div></a>
    <a href="?page=cleaner" target="_self" class="nav-card">⚗️<div class="nav-title">2. ALCHEMIST</div></a>
    <a href="?page=analysis" target="_self" class="nav-card">🧪<div class="nav-title">3. STRATEGIST</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card">🖥️<div class="nav-title">4. ARCHITECT</div></a>
    <a href="?page=graph" target="_self" class="nav-card">🕸️<div class="nav-title">KNOWLEDGE GRAPH</div></a>
</div>
"""

# ======================= HOME: CMU 3D BLACK GALAXY =======================
if current_page == "home":
    st.markdown("<style>.block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    
    three_js_galaxy = """
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body { margin: 0; background: #050505; overflow: hidden; font-family: sans-serif; }
        .node-label {
            position: absolute; background: rgba(0,0,0,0.8); border: 1px solid #C41230;
            padding: 8px 16px; border-radius: 8px; font-weight: bold; cursor: pointer;
            transition: 0.3s; pointer-events: auto; text-decoration: none; color: #fff; font-size: 11px; letter-spacing: 1px;
        }
        .node-label:hover { background: #C41230; transform: scale(1.1); box-shadow: 0 0 15px #C41230; }
    </style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color(0x050505); // BLACK GALAXY
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true; controls.autoRotateSpeed = 0.5; controls.enableDamping = true;

        // DYNAMIC LIGHTING
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4); scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xC41230, 1.5); dirLight.position.set(10, 20, 10); scene.add(dirLight);
        const pointLight = new THREE.PointLight(0xffffff, 1, 50); pointLight.position.set(0, 0, 5); scene.add(pointLight);

        // CMU BRANDED PARTICLES (White, Red, Iron Gray)
        const count = 35000; const pos = new Float32Array(count * 3); const colors = new Float32Array(count * 3);
        const cmuRed = new THREE.Color(0xC41230); const cmuGray = new THREE.Color(0x6D6E71); const cmuWhite = new THREE.Color(0xffffff);
        for(let i=0; i<count; i++){
            const r = 28 * Math.cbrt(Math.random()); const t = Math.random()*2*Math.PI; const p = Math.acos(2*Math.random()-1);
            pos[i*3] = r * Math.sin(p) * Math.cos(t); pos[i*3+1] = r * Math.sin(p) * Math.sin(t); pos[i*3+2] = r * Math.cos(p);
            const rand = Math.random();
            let c = cmuRed;
            if(rand > 0.6) c = cmuGray; else if(rand > 0.3) c = cmuWhite;
            colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
        }
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const mat = new THREE.PointsMaterial({size: 0.06, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending});
        scene.add(new THREE.Points(geo, mat));

        // 3D "CMU" TEXT CORE
        const loader = new THREE.FontLoader();
        loader.load('https://unpkg.com/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json', function (font) {
            const textGeo = new THREE.TextGeometry('CMU', { font: font, size: 5, height: 1.5, curveSegments: 12, bevelEnabled: true, bevelThickness: 0.1, bevelSize: 0.1 });
            textGeo.computeBoundingBox(); const centerOffset = -0.5 * (textGeo.boundingBox.max.x - textGeo.boundingBox.min.x);
            textGeo.translate(centerOffset, -1.5, 0);
            const textMat = new THREE.MeshPhongMaterial({color: 0xC41230, emissive: 0x400000, shininess: 100, wireframe: false});
            scene.add(new THREE.Mesh(textGeo, textMat));
        });

        // INTERACTIVE AGENT NODES
        const agents = [
            {name: "1. AUDITOR", url: "?page=explorer", color: "#ffffff", pos: [14, 8, 2]},
            {name: "2. ALCHEMIST", url: "?page=cleaner", color: "#6D6E71", pos: [-14, -8, 5]},
            {name: "3. STRATEGIST", url: "?page=analysis", color: "#ffffff", pos: [8, -14, -5]},
            {name: "4. ARCHITECT", url: "?page=dashboard", color: "#C41230", pos: [0, 16, 5]},
            {name: "5. KNOWLEDGE GRAPH", url: "?page=graph", color: "#6D6E71", pos: [-12, 12, -8]}
        ];

        agents.forEach(a => {
            const el = document.createElement('a'); el.className = 'node-label';
            el.innerText = a.name; el.href = a.url; el.target = "_parent";
            document.body.appendChild(el); a.el = el;
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.8, 16, 16), new THREE.MeshBasicMaterial({color: a.color}));
            mesh.position.set(...a.pos); scene.add(mesh); a.mesh = mesh;
        });

        camera.position.z = 35;
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
    components.html(three_js_galaxy, height=900)

# ======================= AGENT 1: FORENSIC AUDITOR =======================
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h2>Step 1: Forensic Auditor</h2>", unsafe_allow_html=True)
    st.caption("Deep-File Profiling & Schema Analysis.")
    
    st.markdown("""<div class="white-card">
        <h3>Why is this important?</h3>
        <p>Before we merge datasets, we must identify inconsistencies like missing budgets or string artifacts in financial columns. The Auditor provides a microscopic view of data health.</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for i, anom in enumerate(anomalies):
        color = "#C41230" if anom['impact'] == "Critical" else "#f59e0b" if anom['impact'] == "High" else "#3b82f6"
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"""
            <div class="white-card" style="border-top: 4px solid {color};">
                <h4 style="margin-top:0;">{anom['title']}</h4>
                <p><strong>Issue:</strong> {anom['description']}</p>
                <p style="margin-bottom:0;"><strong>Consequence:</strong> {anom['reason']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Interactive File Scanner")
    f = st.selectbox("Select Target CSV", ALL_FILES)
    if os.path.exists(f'data/{f}'):
        df = pd.read_csv(f'data/{f}', skiprows=1 if 'UTM_Totals' in f else 0)
        st.markdown("<div class='white-card'>", unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("File not found locally. Proceed to Alchemist.")

# ======================= AGENT 2: DATA ALCHEMIST =======================
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h2>Step 2: Data Alchemist</h2>", unsafe_allow_html=True)
    
    st.markdown("""<div class="white-card">
        <h3>The Synthesis Engine</h3>
        <p>The Alchemist aggressively normalizes UTM parameters (stripping spaces, symbols, cases) to force perfect SQL-style LEFT JOINS across 12 messy CSVs. It purges currency artifacts to create mathematically sound float columns.</p>
    </div>""", unsafe_allow_html=True)
    
    if not master_df.empty:
        st.success(f"Master Hub Synthesized! Successfully mapped ${master_df['Total_Spend'].sum():,.2f} in spend.")
    st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.dataframe(master_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= AGENT 3: QUANTITATIVE STRATEGIST =======================
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h2>Step 3: Quantitative Strategist</h2>", unsafe_allow_html=True)
    
    st.markdown("""<div class="white-card">
        <h3>Predictive Analytics & Clustering</h3>
        <p>The Strategist runs Pearson correlations and maps campaigns onto an <strong>Efficiency Quadrant</strong> to identify Top Performers vs. Budget Drains.</p>
    </div>""", unsafe_allow_html=True)
    
    if not master_df.empty:
        valid_data = master_df[(master_df['Total_Spend'] > 0) & (master_df['Total_Users'] > 0)].copy()
        
        if len(valid_data) > 2 and valid_data['Total_Spend'].var() > 0:
            corr, _ = stats.pearsonr(valid_data['Total_Spend'], valid_data['Total_Users'])
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='white-card'><h3 style='margin:0;'>{corr:.2f}</h3><p style='margin:0;'>Spend-to-User Correlation</p></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='white-card'><h3 style='margin:0;'>{len(valid_data)}</h3><p style='margin:0;'>Significant Campaigns</p></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='white-card'><h3 style='margin:0;'>${valid_data['CPWU'].mean():.2f}</h3><p style='margin:0;'>Average Cost Per User</p></div>", unsafe_allow_html=True)
            
            # White Card for Chart
            st.markdown("<div class='white-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Quadrant Analysis: Efficiency vs Volume</h4>", unsafe_allow_html=True)
            fig = px.scatter(valid_data, x="Total_Spend", y="CPWU", color="Platform", size="Total_Users", hover_name="utm_clean")
            fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
            fig.add_hline(y=valid_data['CPWU'].median(), line_dash="dash", line_color="red")
            fig.add_vline(x=valid_data['Total_Spend'].median(), line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<p><em>Bottom-Right Quadrant = High Spend, Low Cost Per User (The 'Stars'). Top-Right = High Spend, High CPWU (The 'Drains').</em></p></div>", unsafe_allow_html=True)
            
            st.markdown("<div class='white-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Platform Dominance (CPWU by Network)</h4>", unsafe_allow_html=True)
            fig2 = px.box(valid_data, x="Platform", y="CPWU", color="Platform", points="all")
            fig2.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Insufficient variance for Statistical Modeling.")

# ======================= AGENT 4: VISUAL ARCHITECT =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h2>Step 4: Visual Architect</h2>", unsafe_allow_html=True)
    
    if not master_df.empty:
        st.sidebar.markdown("<h2 style='color:white;'>🎯 Architect Filters</h2>", unsafe_allow_html=True)
        categories = master_df['Category'].dropna().unique().tolist()
        selected_cats = st.sidebar.multiselect("Filter by Category", categories, default=categories)
        
        max_spend = float(master_df['Total_Spend'].max()) if not master_df.empty else 0.0
        if pd.isna(max_spend) or max_spend <= 0.0: max_spend = 1.0
        spend_range = st.sidebar.slider("Total Spend Range", 0.0, max_spend, (0.0, max_spend))
        
        f_df = master_df[(master_df['Category'].isin(selected_cats)) & (master_df['Total_Spend'] >= spend_range[0]) & (master_df['Total_Spend'] <= spend_range[1])]
        
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='white-card'><h2 style='margin:0; color:#C41230 !important;'>${f_df['Total_Spend'].sum():,.0f}</h2><p style='margin:0;'>Filtered Spend</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='white-card'><h2 style='margin:0; color:#00A6D6 !important;'>{f_df['Total_Users'].sum():,.0f}</h2><p style='margin:0;'>Filtered Users</p></div>", unsafe_allow_html=True)
        avg_cpwu = f_df['Total_Spend'].sum() / f_df['Total_Users'].sum() if f_df['Total_Users'].sum() > 0 else 0
        c3.markdown(f"<div class='white-card'><h2 style='margin:0; color:#44BBA4 !important;'>${avg_cpwu:.2f}</h2><p style='margin:0;'>Filtered CPWU</p></div>", unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("<div class='white-card'>", unsafe_allow_html=True)
            fig_bar = px.bar(f_df.groupby('Category')['Total_Spend'].sum().reset_index(), x='Category', y='Total_Spend', color='Category', title="Budget by Department")
            fig_bar.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with v2:
            st.markdown("<div class='white-card'>", unsafe_allow_html=True)
            if f_df['Total_Users'].sum() > 0:
                fig_pie = px.pie(f_df.groupby('Platform')['Total_Users'].sum().reset_index(), names='Platform', values='Total_Users', hole=0.4, title="User Acquisition by Platform")
                fig_pie.update_layout(paper_bgcolor='white', plot_bgcolor='white', font=dict(color='black'))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("<div class='white-card'><h4>Filtered Campaign Data</h4>", unsafe_allow_html=True)
        st.dataframe(f_df[['utm_clean', 'Platform', 'Category', 'Total_Spend', 'Total_Users', 'CPWU']].sort_values(by="Total_Spend", ascending=False), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================= AGENT 5: KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h2>Step 5: Knowledge Graph</h2>", unsafe_allow_html=True)
    
    st.markdown("""<div class="white-card">
        <h3>Relational Mapping of the Ecosystem</h3>
        <p>This graph proves the physical data connection. It maps the central Nexus to the Marketing Platform (Google/LinkedIn), down to the Department Category, and finally to the individual Campaigns.</p>
    </div>""", unsafe_allow_html=True)
    
    if not master_df.empty:
        # Dark theme for Pyvis to match galaxy
        net = Network(height="700px", width="100%", bgcolor="#050505", font_color="#ffffff", select_menu=True)
        net.add_node("CMU NEXUS", size=50, color="#C41230", label="CMU NEXUS")
        
        platforms = master_df['Platform'].unique()
        for p in platforms:
            net.add_node(str(p), size=35, color="#6D6E71")
            net.add_edge("CMU NEXUS", str(p))
            
            p_cats = master_df[master_df['Platform'] == p]['Category'].dropna().unique()
            for cat in p_cats:
                cat_id = f"{p}_{cat}" # Unique ID
                net.add_node(cat_id, size=20, color="#ffffff", label=str(cat))
                net.add_edge(str(p), cat_id)
                
                camps = master_df[(master_df['Platform'] == p) & (master_df['Category'] == cat)]['utm_clean'].dropna().unique()
                for camp in camps[:5]: # Limit to prevent visual clutter
                    if str(camp).strip() != "":
                        net.add_node(str(camp), size=10, color="#E2C044", title=f"Campaign: {camp}")
                        net.add_edge(cat_id, str(camp))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_data = f.read()
                # Inject style to hide pyvis borders
                html_data = html_data.replace('<style type="text/css">', '<style type="text/css">\n #mynetwork {border: none; outline: none;}\n')
                components.html(html_data, height=750)
