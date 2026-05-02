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
from urllib.parse import urlparse, parse_qs

# Streamlit Extras for premium UI
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.dataframe_explorer import dataframe_explorer

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION & ROUTING SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus", layout="wide", initial_sidebar_state="collapsed")

if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

# ---------------------------------------------------------
# 2. CUSTOM UI: TOP-LEFT RADIAL MENU
# ---------------------------------------------------------
radial_menu_html = """
<style>
    .radial-nav { position: fixed; top: 20px; left: 20px; z-index: 999999; }
    .menu-button {
        width: 60px; height: 60px; background-color: #C41230; color: white;
        border-radius: 50%; display: flex; justify-content: center; align-items: center;
        font-size: 24px; cursor: pointer; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        transition: transform 0.3s ease; position: relative; z-index: 2;
    }
    .menu-label {
        position: absolute; top: 18px; left: 75px; width: 250px;
        color: #888; font-family: sans-serif; font-size: 14px; font-weight: bold;
        pointer-events: none; transition: opacity 0.3s;
    }
    .radial-nav:hover .menu-button { transform: rotate(90deg); }
    .radial-nav:hover .menu-label { opacity: 0; }
    .menu-item {
        position: absolute; width: 50px; height: 50px; background-color: #333; color: white;
        border-radius: 50%; display: flex; justify-content: center; align-items: center;
        font-size: 10px; text-decoration: none; font-family: sans-serif; font-weight: bold; text-align: center;
        opacity: 0; transform: scale(0) translate(0, 0); transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3); border: 1px solid #555; z-index: 1; top: 5px; left: 5px;
    }
    .radial-nav:hover .item-home { opacity: 1; transform: translate(75px, 0px) scale(1); }
    .radial-nav:hover .item-exp  { opacity: 1; transform: translate(65px, 50px) scale(1); }
    .radial-nav:hover .item-cln  { opacity: 1; transform: translate(35px, 85px) scale(1); }
    .radial-nav:hover .item-ana  { opacity: 1; transform: translate(-10px, 100px) scale(1); }
    .radial-nav:hover .item-dash { opacity: 1; transform: translate(-55px, 85px) scale(1); }
    .radial-nav:hover .item-kg   { opacity: 1; transform: translate(-85px, 50px) scale(1); }
    .menu-item:hover { background-color: #C41230; border-color: #FFF; color: #FFF;}
</style>
<div class="radial-nav">
    <div class="menu-label">Toggle to move to different tabs ⭢</div>
    <div class="menu-button">✦</div>
    <a href="?page=home" target="_self" class="menu-item item-home" title="3D Home">HOME</a>
    <a href="?page=explorer" target="_self" class="menu-item item-exp" title="Data Explorer">EXPLORE</a>
    <a href="?page=cleaner" target="_self" class="menu-item item-cln" title="Data Cleaner">CLEAN</a>
    <a href="?page=analysis" target="_self" class="menu-item item-ana" title="Data Analysis">STATS</a>
    <a href="?page=dashboard" target="_self" class="menu-item item-dash" title="Interactive Dashboard">DASH</a>
    <a href="?page=graph" target="_self" class="menu-item item-kg" title="Knowledge Graph">GRAPH</a>
</div>
"""
st.markdown(radial_menu_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. GLOBAL DATA PROCESSING LOGIC (Implementing Assessment Fixes)
# ---------------------------------------------------------
ALL_FILES = [
    "2024-25_Campaign_Management_1769521985.csv", "2025-26_Campaign_Management_1769522231.csv",
    "GA_FY25_TimeSeries (1).csv", "GA_FY26_TimeSeries.csv",
    "GA_FY25_UTM_Totals_Jul2024-Jun2025.csv", "GA_FY26_UTM_Totals_Jul-Dec2025.csv",
    "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv", "GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv",
    "GAds_FY25_Totals_Jul2024-Jun2025.csv", "GAds_FY26_Totals_Jul-Dec2025.csv",
    "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv", "UCM Campaign Index.csv"
]

@st.cache_data
def load_raw_file(filename):
    try:
        # Assessment Fix: Skip malformed headers on GA UTM files
        skip = 1 if 'UTM_Totals' in filename else 0
        df = pd.read_csv(f'data/{filename}', skiprows=skip)
        
        # Assessment Fix: Header Deduplication (Pandas adds .1 to dupes)
        duplicate_cols = [c for c in df.columns if c.endswith('.1')]
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)
            
        return df
    except Exception as e:
        return pd.DataFrame({'Error': [f"Could not load {filename}: {str(e)}"]})

def extract_utm_campaign(url):
    """Assessment Fix: Extracts utm_campaign from URLs to fill incomplete Index mappings"""
    try:
        parsed = urlparse(str(url))
        return parse_qs(parsed.query).get('utm_campaign', [np.nan])[0]
    except:
        return np.nan

@st.cache_data
def build_master_models():
    # 1. Load & Combine Multi-Year Files
    index_df = load_raw_file("UCM Campaign Index.csv")
    ga_utm = pd.concat([load_raw_file("GA_FY25_UTM_Totals_Jul2024-Jun2025.csv"), load_raw_file("GA_FY26_UTM_Totals_Jul-Dec2025.csv")], ignore_index=True)
    ga_time = pd.concat([load_raw_file("GA_FY25_TimeSeries (1).csv"), load_raw_file("GA_FY26_TimeSeries.csv")], ignore_index=True)
    gads_perf = load_raw_file("GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv")
    linkedin_perf = load_raw_file("LinkedIn_Ad_Performance_Feb2024_Dec2025.csv")

    if 'Error' in index_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 2. Assessment Fix: Missing Index Mappings (URL Parsing)
    if 'UTM campaign' in index_df.columns and 'Landing Page (UTM)' in index_df.columns:
        index_df['UTM campaign'] = index_df['UTM campaign'].fillna(index_df['Landing Page (UTM)'].apply(extract_utm_campaign))
        index_df['utm_clean'] = index_df['UTM campaign'].astype(str).str.lower().str.strip()

    # 3. Assessment Fix: GA TimeSeries Wide-to-Long Reshaping & Zero Handling
    if 'Session campaign' in ga_time.columns:
        melted_ga = pd.melt(ga_time, id_vars=['Session campaign', 'Segment'], var_name='Day', value_name='User_Count')
        melted_ga['Day_Number'] = melted_ga['Day'].str.extract(r'(\d+)').astype(float)
        melted_ga['User_Count'] = pd.to_numeric(melted_ga['User_Count'], errors='coerce').fillna(0)
        melted_ga = melted_ga[melted_ga['User_Count'] > 0] # Filter zero inflation
    else:
        melted_ga = pd.DataFrame()

    # 4. Assessment Fix: GAds Data Type Mismatches & Sentinels
    gads_perf.replace('--', np.nan, inplace=True)
    if 'Ad name' in gads_perf.columns:
        gads_perf = gads_perf[~gads_perf['Ad name'].astype(str).str.contains('Total', case=False, na=False)]
    for col in ['Clicks', 'Impr.', 'CTR', 'Cost']:
        if col in gads_perf.columns:
            gads_perf[col] = pd.to_numeric(gads_perf[col].astype(str).str.replace(r'[,\%\$]', '', regex=True), errors='coerce')

    # 5. Assessment Fix: LinkedIn High Column Cardinality
    if len(linkedin_perf) > 0 and 'Error' not in linkedin_perf.columns:
        linkedin_clean = linkedin_perf.dropna(thresh=len(linkedin_perf) * 0.2, axis=1)
    else:
        linkedin_clean = linkedin_perf

    # 6. Master Merge
    ga_agg = pd.DataFrame(columns=['utm_clean', 'Total_Website_Users', 'Average_Engagement_Rate'])
    if 'Session campaign' in ga_utm.columns:
        ga_utm['utm_clean'] = ga_utm['Session campaign'].astype(str).str.lower().str.strip()
        ga_agg = ga_utm.groupby('utm_clean').agg(Total_Website_Users=('Total users', 'sum'), Average_Engagement_Rate=('Engagement rate', 'mean')).reset_index()

    gads_agg = pd.DataFrame(columns=['utm_clean', 'Total_GAds_Spend'])
    if 'Ad name' in gads_perf.columns and 'Cost' in gads_perf.columns:
        gads_perf['utm_clean'] = gads_perf['Ad name'].astype(str).str.lower().str.strip()
        gads_agg = gads_perf.groupby('utm_clean').agg(Total_GAds_Spend=('Cost', 'sum')).reset_index()

    li_agg = pd.DataFrame(columns=['utm_clean', 'Total_LinkedIn_Spend'])
    if 'Campaign Name' in linkedin_clean.columns and 'Total Spend' in linkedin_clean.columns:
        linkedin_clean['utm_clean'] = linkedin_clean['Campaign Name'].astype(str).str.lower().str.strip()
        li_agg = linkedin_clean.groupby('utm_clean').agg(Total_LinkedIn_Spend=('Total Spend', 'sum')).reset_index()

    master_df = index_df.copy()
    if 'utm_clean' in master_df.columns:
        master_df = pd.merge(master_df, ga_agg, on='utm_clean', how='left')
        master_df = pd.merge(master_df, gads_agg, on='utm_clean', how='left')
        master_df = pd.merge(master_df, li_agg, on='utm_clean', how='left')
        master_df.fillna({'Total_GAds_Spend': 0, 'Total_LinkedIn_Spend': 0, 'Total_Website_Users': 0}, inplace=True)
        master_df['Total_Combined_Spend'] = master_df['Total_GAds_Spend'] + master_df['Total_LinkedIn_Spend']
        master_df['CPWU'] = master_df['Total_Combined_Spend'].div(master_df['Total_Website_Users'].replace(0, np.nan)).fillna(0)
    
    return index_df, ga_utm, melted_ga, gads_perf, linkedin_clean, master_df

index_df, ga_utm, melted_ga, gads_perf, linkedin_clean, master_df = build_master_models()

# ---------------------------------------------------------
# 4. VIEW RENDERING (ROUTING)
# ---------------------------------------------------------

# ======================= PAGE: 3D HOME =======================
if current_page == "home":
    st.markdown("<h1 style='text-align: center; color: #C41230; margin-top: 20px;'>CMU Data Nexus</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Drag to rotate camera. <b>Click the floating Agent Nodes</b> to explore the specific tabs.</p>", unsafe_allow_html=True)
    
    three_js_app = """
    <!DOCTYPE html>
    <html>
    <head>
        <style> body { margin: 0; overflow: hidden; background-color: #050505; border-radius: 10px; } </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <script>
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
            const renderer = new THREE.WebGLRenderer({alpha: true, antialias: true});
            renderer.setSize(window.innerWidth, 600);
            document.body.appendChild(renderer.domElement);
            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.5;

            // Background Particles
            const particlesCount = 4000;
            const posArray = new Float32Array(particlesCount * 3);
            for(let i = 0; i < particlesCount * 3; i++) { posArray[i] = (Math.random() - 0.5) * 20; }
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
            const material = new THREE.PointsMaterial({ size: 0.02, color: 0x555555 });
            const particlesMesh = new THREE.Points(geometry, material);
            scene.add(particlesMesh);

            // 5 Interactive Nodes
            const agents = [];
            const sphereGeo = new THREE.SphereGeometry(0.5, 32, 32);
            const nodesConfig = [
                { color: 0xE2C044, pos: [-3, 2, 0], url: "?page=explorer" }, // Yellow
                { color: 0xE87A5D, pos: [3, 2, 0], url: "?page=cleaner" },  // Orange
                { color: 0x44BBA4, pos: [-4, -1, 2], url: "?page=analysis" }, // Green
                { color: 0x00A6D6, pos: [4, -1, -2], url: "?page=dashboard" }, // Blue
                { color: 0x9B5DE5, pos: [0, 0, 3], url: "?page=graph" }        // Purple
            ];

            nodesConfig.forEach(config => {
                const mat = new THREE.MeshBasicMaterial({color: config.color});
                const mesh = new THREE.Mesh(sphereGeo, mat);
                mesh.position.set(...config.pos);
                mesh.userData = { url: config.url };
                scene.add(mesh); agents.push(mesh);
            });

            camera.position.z = 8;
            const raycaster = new THREE.Raycaster();
            const mouse = new THREE.Vector2();

            window.addEventListener('pointerdown', (event) => {
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / 600) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(agents);
                if(intersects.length > 0) window.parent.location.search = intersects[0].object.userData.url;
            });

            window.addEventListener('pointermove', (event) => {
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / 600) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);
                const intersects = raycaster.intersectObjects(agents);
                document.body.style.cursor = intersects.length > 0 ? 'pointer' : 'default';
            });

            function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(three_js_app, height=600)

# ======================= PAGE 1: MASTER DATA EXPLORER =======================
elif current_page == "explorer":
    colored_header(label="Tab 1: Master Data Explorer", description="View and filter all 12 raw data files.", color_name="yellow-70")
    selected_file = st.selectbox("Select a Dataset to Explore", ALL_FILES)
    df = load_raw_file(selected_file)
    
    if not df.empty and 'Error' not in df.columns:
        st.metric("Total Rows", len(df))
        filtered_df = dataframe_explorer(df, case=False)
        st.dataframe(filtered_df, use_container_width=True, height=500)
    else:
        st.error(df['Error'].iloc[0] if 'Error' in df.columns else "File is empty.")

# ======================= PAGE 2: DATA CLEANER =======================
elif current_page == "cleaner":
    colored_header(label="Tab 2: Data Cleaner & Anomaly Resolution", description="Automated pipeline fixes applied based on Data Quality Assessment.", color_name="orange-70")
    
    c1, c2, c3 = st.columns(3)
    c1.info("**Fix 1: Index Mapping Extraction**\n\nParsed 'Landing Page' URLs to extract missing `utm_campaign` tags.")
    c2.info("**Fix 2: GA Deduplication & Melting**\n\nDropped duplicate `.1` headers and melted wide TimeSeries to long format.")
    c3.info("**Fix 3: GAds & LinkedIn Formatting**\n\nRegex stripped '%/,' symbols and dropped >80% empty LinkedIn columns.")
    
    st.subheader("Final Cleaned Master Hub Preview")
    if not master_df.empty:
        display_cols = [c for c in ['Unique_Campaign_ID', 'Category', 'Total_Combined_Spend', 'Total_Website_Users', 'CPWU'] if c in master_df.columns]
        st.dataframe(master_df[display_cols].sort_values(by='CPWU', ascending=True).head(50), use_container_width=True)

# ======================= PAGE 3: DATA ANALYSIS =======================
elif current_page == "analysis":
    colored_header(label="Tab 3: Statistical Data Analysis", description="Regression models and Hypothesis testing.", color_name="green-70")
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Time-Series Regression")
        if not melted_ga.empty:
            top_camps = melted_ga.groupby('Session campaign')['User_Count'].sum().nlargest(5).index
            fig_trend = px.scatter(
                melted_ga[melted_ga['Session campaign'].isin(top_camps)], 
                x="Day_Number", y="User_Count", color="Session campaign", trendline="ols", template="plotly_dark"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
    with colB:
        st.subheader("A/B Hypothesis Testing (T-Test)")
        st.markdown("Comparing Average Engagement Rate: **Google Ads vs LinkedIn Ads**")
        if 'Session source' in ga_utm.columns and 'Engagement rate' in ga_utm.columns:
            ga_utm['Engagement rate'] = pd.to_numeric(ga_utm['Engagement rate'], errors='coerce').fillna(0)
            google_eng = ga_utm[ga_utm['Session source'].str.contains('google', na=False, case=False)]['Engagement rate']
            li_eng = ga_utm[ga_utm['Session source'].str.contains('linkedin', na=False, case=False)]['Engagement rate']
            
            if len(google_eng) > 0 and len(li_eng) > 0:
                t_stat, p_val = stats.ttest_ind(google_eng, li_eng, equal_var=False)
                style_metric_cards(background_color="#222", border_left_color="#44BBA4")
                mc1, mc2 = st.columns(2)
                mc1.metric("T-Statistic", f"{t_stat:.4f}")
                mc2.metric("P-Value", f"{p_val:.4e}")
                
                if p_val < 0.05:
                    st.success("Result: Statistically Significant (p < 0.05).")
                else:
                    st.warning("Result: Not Statistically Significant.")

# ======================= PAGE 4: INTERACTIVE DASHBOARD =======================
elif current_page == "dashboard":
    colored_header(label="Tab 4: Interactive Dashboard", description="Cross-platform financial and operational insights.", color_name="light-blue-70")
    
    if not master_df.empty:
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total Spend Logged", f"${master_df['Total_Combined_Spend'].sum():,.2f}")
        mc2.metric("Total Users Acquired", f"{master_df['Total_Website_Users'].sum():,.0f}")
        avg_cpwu = master_df['Total_Combined_Spend'].sum() / master_df['Total_Website_Users'].sum() if master_df['Total_Website_Users'].sum() > 0 else 0
        mc3.metric("Global Cost Per User", f"${avg_cpwu:.2f}")
        style_metric_cards(background_color="#1E1E1E", border_left_color="#00A6D6")
        
        st.divider()
        st.subheader("Cross-Platform Campaign ROI Matrix")
        valid_master = master_df[(master_df['Total_Website_Users'] > 0) & (master_df['Total_Combined_Spend'] > 0)]
        
        fig_roi = px.scatter(
            valid_master, x="Total_Combined_Spend", y="Total_Website_Users", 
            size="Average_Engagement_Rate" if 'Average_Engagement_Rate' in valid_master.columns else None, 
            color='Category' if 'Category' in valid_master.columns else None,
            hover_name="Unique_Campaign_ID" if 'Unique_Campaign_ID' in valid_master.columns else None,
            labels={"Total_Combined_Spend": "Ad Spend ($)", "Total_Website_Users": "Website Users"},
            template="plotly_dark", height=600
        )
        fig_roi.add_hline(y=valid_master['Total_Website_Users'].mean(), line_dash="dot", annotation_text="Avg Users")
        fig_roi.add_vline(x=valid_master['Total_Combined_Spend'].mean(), line_dash="dot", annotation_text="Avg Spend")
        st.plotly_chart(fig_roi, use_container_width=True)

# ======================= PAGE 5: KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    colored_header(label="Tab 5: Knowledge Graph", description="Network topology of Campaign Structure, Vendors, and Mediums.", color_name="violet-70")
    
    if not master_df.empty and 'Unique_Campaign_ID' in master_df.columns:
        G = nx.Graph()
        
        for _, row in master_df.dropna(subset=['Unique_Campaign_ID', 'Category', 'Vendor']).head(50).iterrows():
            campaign = str(row['Unique_Campaign_ID'])
            category = str(row['Category'])
            vendor = str(row['Vendor'])
            
            G.add_node(category, title="Category", group=1, size=30, color="#E2C044")
            G.add_node(campaign, title="Campaign", group=2, size=20, color="#C41230")
            G.add_node(vendor, title="Vendor", group=3, size=25, color="#00A6D6")
            
            G.add_edge(category, campaign)
            G.add_edge(campaign, vendor)

        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white", select_menu=True)
        net.from_nx(G)
        net.repulsion(node_distance=150, spring_length=150)
        
        path = os.path.join(tempfile.gettempdir(), 'knowledge_graph.html')
        net.save_graph(path)
        
        with open(path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=650)
    else:
        st.warning("Insufficient categorical data loaded to generate Knowledge Graph.")
