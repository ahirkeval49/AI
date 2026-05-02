import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
import os
import base64

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & AGENT STATE
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus | Competition Build", layout="wide", initial_sidebar_state="collapsed")

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {"audit_logs": {}, "synthesis_stats": {}, "model_results": {}}

if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except: return ""

# ---------------------------------------------------------
# 2. THE ALCHEMIST ENGINE: AGGRESSIVE ETL NORMALIZATION
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
    """Scouts for column variations."""
    for alias in aliases:
        if alias in df.columns: return alias
    return None

def clean_currency(series):
    """Removes currency artifacts ($ , %) and converts to float."""
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

def normalize_key(series):
    """THE FIX: Aggressive Alphanumeric Normalization. 
    Strips ALL spaces, underscores, and dashes to force perfect matches between files."""
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

@st.cache_data
def build_master_hub():
    """Agent 2 (Alchemist) Synthesis."""
    try:
        # Load Index (Source of Truth)
        if os.path.exists('data/UCM Campaign Index.csv'):
            idx = pd.read_csv('data/UCM Campaign Index.csv')
            utm_col = find_col(idx, ['UTM campaign', 'Campaign_ID', 'UTM_Combined_ID', 'Landing Page (UTM)'])
            idx['utm_clean'] = normalize_key(idx[utm_col]) if utm_col else ""
            if 'Category' not in idx.columns: idx['Category'] = "Uncategorized"
        else:
            return pd.DataFrame()

        # GAds Pipeline
        g_files = ['GAds_FY25_Totals_Jul2024-Jun2025.csv', 'GAds_FY26_Totals_Jul-Dec2025.csv']
        g_dfs = [pd.read_csv(f'data/{f}') for f in g_files if os.path.exists(f'data/{f}')]
        g_all = pd.concat(g_dfs, ignore_index=True) if g_dfs else pd.DataFrame()
        if not g_all.empty:
            g_key = find_col(g_all, ['Ad name', 'Campaign', 'Campaign Name', 'Campaign state'])
            g_all['utm_clean'] = normalize_key(g_all[g_key]) if g_key else ""
            g_cost_col = find_col(g_all, ['Cost', 'Spend'])
            g_all['Cost'] = clean_currency(g_all[g_cost_col]) if g_cost_col else 0.0
            g_agg = g_all.groupby('utm_clean').agg(GAds_Spend=('Cost', 'sum')).reset_index()
        else:
            g_agg = pd.DataFrame(columns=['utm_clean', 'GAds_Spend'])

        # LinkedIn Pipeline
        li_path = 'data/LinkedIn_Ad_Performance_Feb2024_Dec2025.csv'
        if os.path.exists(li_path):
            li = pd.read_csv(li_path)
            li_key = find_col(li, ['Campaign Name', 'Campaign'])
            li['utm_clean'] = normalize_key(li[li_key]) if li_key else ""
            li_spend = find_col(li, ['Total Spend', 'Spend', 'Cost'])
            li['LI_Spend'] = clean_currency(li[li_spend]) if li_spend else 0.0
            li_agg = li.groupby('utm_clean').agg(LI_Spend=('LI_Spend', 'sum')).reset_index()
        else:
            li_agg = pd.DataFrame(columns=['utm_clean', 'LI_Spend'])

        # GA Pipeline
        ga_files = ['GA_FY25_UTM_Totals_Jul2024-Jun2025.csv', 'GA_FY26_UTM_Totals_Jul-Dec2025.csv']
        ga_dfs = []
        for f in ga_files:
            if os.path.exists(f'data/{f}'):
                _df = pd.read_csv(f'data/{f}', skiprows=1)
                ga_key = find_col(_df, ['Session campaign', 'Campaign'])
                if ga_key:
                    _df['utm_clean'] = normalize_key(_df[ga_key])
                    ga_users_col = find_col(_df, ['Total users', 'Users'])
                    _df['Total users'] = pd.to_numeric(_df[ga_users_col], errors='coerce').fillna(0.0) if ga_users_col else 0.0
                    ga_dfs.append(_df[['utm_clean', 'Total users']])
        ga_agg = pd.concat(ga_dfs).groupby('utm_clean').agg(Total_Users=('Total users', 'sum')).reset_index() if ga_dfs else pd.DataFrame(columns=['utm_clean', 'Total_Users'])

        # Synthesis
        hub = pd.merge(idx, ga_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, g_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, li_agg, on='utm_clean', how='left').fillna(0.0)
        
        # STRICT FLOAT CASTING
        hub['Total_Spend'] = pd.to_numeric(hub['GAds_Spend'] + hub['LI_Spend'], errors='coerce').fillna(0.0).astype(float)
        hub['Total_Users'] = pd.to_numeric(hub['Total_Users'], errors='coerce').fillna(0.0).astype(float)
        hub['CPWU'] = hub['Total_Spend'].div(hub['Total_Users'].replace(0, np.nan)).fillna(0.0).astype(float)
        
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
    width: 130px; text-align: center; color: #333 !important; text-decoration: none;
    transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
.nav-card:hover { border-color: #C41230; transform: translateY(-5px); box-shadow: 0 8px 15px rgba(196,18,48,0.2); }
.nav-title { font-size: 10px; font-weight: 900; color: #C41230; margin-top: 5px; }
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

# ======================= HOME: CMU 3D GALAXY =======================
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
            position: absolute; background: rgba(255,255,255,0.95); border: 2px solid #C41230;
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
        controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

        // DYNAMIC LIGHTING FOR RED TEXT
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7); scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(10, 20, 10); scene.add(dirLight);

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

        // 3D "CMU" TEXT CORE - GLOWING CARDINAL RED
        const loader = new THREE.FontLoader();
        loader.load('https://unpkg.com/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json', function (font) {
            const textGeo = new THREE.TextGeometry('CMU', {
                font: font, size: 4.5, height: 1.2, curveSegments: 12,
                bevelEnabled: true, bevelThickness: 0.1, bevelSize: 0.1, bevelOffset: 0, segments: 5
            });
            textGeo.computeBoundingBox();
            const centerOffset = -0.5 * (textGeo.boundingBox.max.x - textGeo.boundingBox.min.x);
            textGeo.translate(centerOffset, -1.5, 0);
            const textMat = new THREE.MeshPhongMaterial({color: 0xC41230, emissive: 0x220000, shininess: 100});
            const textMesh = new THREE.Mesh(textGeo, textMat);
            scene.add(textMesh);
        });

        // INTERACTIVE AGENT NODES
        const agents = [
            {name: "AUDITOR", url: "?page=explorer", color: "#E2C044", pos: [12, 6, 2]},
            {name: "ALCHEMIST", url: "?page=cleaner", color: "#E87A5D", pos: [-12, -6, 5]},
            {name: "STRATEGIST", url: "?page=analysis", color: "#44BBA4", pos: [6, -12, -5]},
            {name: "ARCHITECT", url: "?page=dashboard", color: "#00A6D6", pos: [0, 14, 5]},
            {name: "KNOWLEDGE GRAPH", url: "?page=graph", color: "#9B5DE5", pos: [-10, 10, -8]}
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
    colored_header("Step 1: Forensic Auditor", "Deep-File Profiling & Schema Analysis.")
    
    f = st.selectbox("Select Target CSV", ALL_FILES)
    if os.path.exists(f'data/{f}'):
        df = pd.read_csv(f'data/{f}', skiprows=1 if 'UTM_Totals' in f else 0)
        
        t1, t2, t3 = st.tabs(["📊 Data Viewer", "🔍 Data Profile", "📈 Descriptive Stats"])
        with t1:
            from streamlit_extras.dataframe_explorer import dataframe_explorer
            st.dataframe(dataframe_explorer(df), use_container_width=True)
        with t2:
            profile = pd.DataFrame({'Type': df.dtypes.astype(str), 'Nulls': df.isna().sum(), 'Unique': df.nunique()})
            st.dataframe(profile, use_container_width=True)
        with t3:
            st.dataframe(df.describe(include='all').T, use_container_width=True)
    else:
        st.error("File not found in data directory.")

# ======================= AGENT 2: DATA ALCHEMIST =======================
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    from streamlit_extras.colored_header import colored_header
    colored_header("Step 2: Data Alchemist", "Synthesized Master Hub: Standardized financial strings and cross-platform joins.")
    
    if not master_df.empty:
        total_spend = master_df['Total_Spend'].sum()
        if total_spend == 0:
            st.warning("⚠️ ALCHEMIST ALERT: Master Hub generated, but Total Spend is $0. Aggressive alphanumeric matching failed to find overlapping keys. Verify file headers.")
        else:
            st.success(f"Join Successful! Matched ${total_spend:,.2f} in marketing spend across datasets.")
    
    st.dataframe(master_df, use_container_width=True)

# ======================= AGENT 3: QUANTITATIVE STRATEGIST =======================
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    from streamlit_extras.colored_header import colored_header
    colored_header("Step 3: Quantitative Strategist", "Inferring mathematical truth from the cleaned hub.")
    
    if not master_df.empty:
        c1, c2 = st.columns(2)
        valid_data = master_df[(master_df['Total_Spend'] > 0) & (master_df['Total_Users'] > 0)]
        
        # DEFENSIVE CHECK: Prevent SciPy KeyError / Zero Variance Crash
        if len(valid_data) > 2 and valid_data['Total_Spend'].var() > 0 and valid_data['Total_Users'].var() > 0:
            corr, _ = stats.pearsonr(valid_data['Total_Spend'], valid_data['Total_Users'])
            c1.metric("Spend-to-User Correlation (Pearson)", f"{corr:.2f}")
            c2.metric("Significant Campaigns Tracked", len(valid_data))
            st.plotly_chart(px.scatter(valid_data, x="Total_Spend", y="Total_Users", 
                                       trendline="ols", color="Category", title="Efficiency Frontier: Regression Analysis"), use_container_width=True)
        else:
            c1.metric("Spend-to-User Correlation", "N/A")
            c2.metric("Significant Campaigns Tracked", len(valid_data))
            st.warning("⚠️ Insufficient variance for Pearson Correlation. Ensure Agent 2 successfully joined Spend and User data.")

# ======================= AGENT 4: VISUAL ARCHITECT =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.metric_cards import style_metric_cards
    colored_header("Step 4: Visual Architect", "Interactive Executive ROI Command Center.")
    
    if not master_df.empty:
        st.sidebar.header("🎯 Architect Filters")
        categories = master_df['Category'].dropna().unique().tolist()
        selected_cats = st.sidebar.multiselect("Filter by Category", categories, default=categories)
        
        # DEFENSIVE SLIDER: Prevent Streamlit API Exception
        max_spend = float(master_df['Total_Spend'].max()) if not master_df.empty else 0.0
        if pd.isna(max_spend) or max_spend <= 0.0:
            max_spend = 1.0 # Minimum valid range to prevent crash
            st.sidebar.warning("No Spend Data available to filter. Displaying fallback slider.")
            
        spend_range = st.sidebar.slider("Filter by Total Spend", 0.0, max_spend, (0.0, max_spend))
        
        filtered_df = master_df[
            (master_df['Category'].isin(selected_cats)) & 
            (master_df['Total_Spend'] >= spend_range[0]) & 
            (master_df['Total_Spend'] <= spend_range[1])
        ]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Filtered Spend", f"${filtered_df['Total_Spend'].sum():,.2f}")
        m2.metric("Filtered Acquisitions", f"{filtered_df['Total_Users'].sum():,.0f}")
        avg_cpwu = filtered_df['Total_Spend'].sum() / filtered_df['Total_Users'].sum() if filtered_df['Total_Users'].sum() > 0 else 0
        m3.metric("Filtered Avg CPWU", f"${avg_cpwu:.2f}")
        style_metric_cards(border_left_color="#C41230")
        
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(px.bar(filtered_df.groupby('Category')['Total_Spend'].sum().reset_index(), 
                                   x='Category', y='Total_Spend', color='Category', title="Budget Allocation"), use_container_width=True)
        with v2:
            if filtered_df['Total_Users'].sum() > 0:
                st.plotly_chart(px.pie(filtered_df.groupby('Category')['Total_Users'].sum().reset_index(), 
                                       names='Category', values='Total_Users', hole=0.4, title="User Acquisition Share"), use_container_width=True)
            else:
                st.info("No user acquisitions in selected range.")
            
        st.dataframe(filtered_df[['utm_clean', 'Category', 'Total_Spend', 'Total_Users', 'CPWU']].sort_values(by="Total_Spend", ascending=False), use_container_width=True)
    else:
        st.warning("Master Hub is empty. Check Alchemist module.")

# ======================= AGENT 5: KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    from streamlit_extras.colored_header import colored_header
    colored_header("Step 5: Knowledge Graph", "Relational Mapping of University Data Streams.")
    
    if not master_df.empty:
        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333")
        net.add_node("Nexus Hub", size=45, color="#C41230", label="NEXUS CORE")
        for cat in master_df['Category'].unique():
            if pd.notna(cat) and str(cat).strip() != "":
                net.add_node(str(cat), size=25, color="#6D6E71")
                net.add_edge("Nexus Hub", str(cat))
                cat_campaigns = master_df[master_df['Category'] == cat]['utm_clean'].dropna().unique()
                for camp in cat_campaigns[:10]: 
                    if str(camp).strip() != "":
                        net.add_node(str(camp), size=10, color="#E2C044", title=f"Campaign: {camp}")
                        net.add_edge(str(cat), str(camp))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=700)
