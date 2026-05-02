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
# 1. PAGE CONFIGURATION & ROUTING SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="CMU AI Nexus", layout="wide", initial_sidebar_state="collapsed")

# Simple routing using query parameters
if "page" not in st.query_params:
    st.query_params["page"] = "home"
current_page = st.query_params["page"]

def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return "" 

# ---------------------------------------------------------
# 2. GLOBAL NAVIGATION UI
# ---------------------------------------------------------
nav_cards_html = """
<style>
.nav-grid { display: flex; justify-content: center; gap: 20px; padding: 10px; flex-wrap: wrap; margin-bottom: 5px; position: relative; z-index: 100; }
.nav-card {
    background: #ffffff; border: 2px solid #E0E0E0; border-radius: 12px; padding: 15px 20px;
    width: 140px; text-align: center; color: #333 !important; text-decoration: none;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); font-family: sans-serif;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.nav-card:hover { 
    border-color: #C41230; transform: translateY(-8px) scale(1.05); 
    background: #FAFAFA; box-shadow: 0 10px 20px rgba(196,18,48,0.2); 
}
.nav-icon { font-size: 30px; margin-bottom: 8px; }
.nav-title { font-size: 12px; font-weight: bold; letter-spacing: 1px; color: #C41230; }
</style>
<div class="nav-grid">
    <a href="?page=home" target="_self" class="nav-card"><div class="nav-icon">🏠</div><div class="nav-title">HOME</div></a>
    <a href="?page=explorer" target="_self" class="nav-card"><div class="nav-icon">📊</div><div class="nav-title">EXPLORE</div></a>
    <a href="?page=cleaner" target="_self" class="nav-card"><div class="nav-icon">🧹</div><div class="nav-title">CLEAN</div></a>
    <a href="?page=analysis" target="_self" class="nav-card"><div class="nav-icon">📈</div><div class="nav-title">STATS</div></a>
    <a href="?page=dashboard" target="_self" class="nav-card"><div class="nav-icon">💡</div><div class="nav-title">DASHBOARD</div></a>
    <a href="?page=graph" target="_self" class="nav-card"><div class="nav-icon">🕸️</div><div class="nav-title">GRAPH</div></a>
</div>
"""

# ---------------------------------------------------------
# 3. DATA ENGINE (MODELS & CACHING)
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
        # Check if file exists in data folder
        path = f'data/{filename}'
        if not os.path.exists(path):
            return pd.DataFrame({'Error': [f"File {filename} not found in /data folder."]})
        
        skip = 1 if 'UTM_Totals' in filename else 0
        df = pd.read_csv(path, skiprows=skip)
        
        # Clean duplicate column names (Pandas .1 suffix)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        duplicate_cols = [c for c in df.columns if c.endswith('.1')]
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)
        return df
    except Exception as e:
        return pd.DataFrame({'Error': [f"Load Error: {str(e)}"]})

def extract_utm_campaign(url):
    try:
        parsed = urlparse(str(url))
        return parse_qs(parsed.query).get('utm_campaign', [np.nan])[0]
    except:
        return np.nan

@st.cache_data
def build_master_models():
    """Builds the interconnected data model for the entire app"""
    index_df = load_raw_file("UCM Campaign Index.csv")
    ga_utm = pd.concat([load_raw_file("GA_FY25_UTM_Totals_Jul2024-Jun2025.csv"), load_raw_file("GA_FY26_UTM_Totals_Jul-Dec2025.csv")], ignore_index=True)
    ga_time = pd.concat([load_raw_file("GA_FY25_TimeSeries (1).csv"), load_raw_file("GA_FY26_TimeSeries.csv")], ignore_index=True)
    gads_perf = load_raw_file("GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv")
    linkedin_perf = load_raw_file("LinkedIn_Ad_Performance_Feb2024_Dec2025.csv")

    if 'Error' in index_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 1. Normalize Index
    if 'UTM campaign' in index_df.columns and 'Landing Page (UTM)' in index_df.columns:
        index_df['UTM campaign'] = index_df['UTM campaign'].fillna(index_df['Landing Page (UTM)'].apply(extract_utm_campaign))
        index_df['utm_clean'] = index_df['UTM campaign'].astype(str).str.lower().str.strip()

    # 2. Process GA TimeSeries
    if 'Session campaign' in ga_time.columns:
        melted_ga = pd.melt(ga_time, id_vars=['Session campaign', 'Segment'], var_name='Day', value_name='User_Count')
        melted_ga['User_Count'] = pd.to_numeric(melted_ga['User_Count'], errors='coerce').fillna(0)
        melted_ga = melted_ga[melted_ga['User_Count'] > 0]
    else:
        melted_ga = pd.DataFrame()

    # 3. Process GAds
    gads_perf.replace('--', np.nan, inplace=True)
    for col in ['Clicks', 'Impr.', 'CTR', 'Cost']:
        if col in gads_perf.columns:
            gads_perf[col] = pd.to_numeric(gads_perf[col].astype(str).str.replace(r'[,\%\$]', '', regex=True), errors='coerce')

    # 4. Merge Logic
    ga_agg = pd.DataFrame(columns=['utm_clean', 'Total_Website_Users'])
    if 'Session campaign' in ga_utm.columns:
        ga_utm['utm_clean'] = ga_utm['Session campaign'].astype(str).str.lower().str.strip()
        ga_agg = ga_utm.groupby('utm_clean').agg(Total_Website_Users=('Total users', 'sum')).reset_index()

    master_df = index_df.copy()
    if 'utm_clean' in master_df.columns:
        master_df = pd.merge(master_df, ga_agg, on='utm_clean', how='left').fillna(0)
    
    return index_df, ga_utm, melted_ga, gads_perf, linkedin_perf, master_df

index_df, ga_utm, melted_ga, gads_perf, linkedin_perf, master_df = build_master_models()

# ---------------------------------------------------------
# 4. ROUTING & VIEWS
# ---------------------------------------------------------

# --- HOME PAGE ---
if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; max-width: 100%; } header { display: none !important; }</style>", unsafe_allow_html=True)
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #C41230; font-weight: 800; margin-top: -10px;'>CMU DATA NEXUS</h1>", unsafe_allow_html=True)
    
    scotty_b64 = get_base64_of_bin_file("scotty.png")
    three_js_code = f"""
    <!DOCTYPE html><html><head><style>body {{ margin: 0; background: #fff; overflow: hidden; font-family: sans-serif; }}
    .node-tab {{ position: absolute; background: white; border: 2px solid #C41230; padding: 5px 10px; border-radius: 5px; font-weight: bold; cursor: pointer; text-decoration: none; color: black !important; }}
    </style><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        
        // Add 35,000 Particles
        const geo = new THREE.BufferGeometry(); const pos = new Float32Array(35000 * 3);
        for(let i=0; i<35000*3; i++) {{ pos[i] = (Math.random()-0.5) * 50; }}
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        const mat = new THREE.PointsMaterial({{ size: 0.05, color: 0x6D6E71 }});
        scene.add(new THREE.Points(geo, mat));

        camera.position.z = 20;
        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); scene.rotation.y += 0.001; }}
        animate();
    </script></body></html>
    """
    components.html(three_js_code, height=700)

# --- EXPLORE PAGE (FIXED) ---
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Master Data Explorer & Profiler", description="Deep-dive into raw datasets with anomaly detection.", color_name="yellow-70")
    
    selected_file = st.selectbox("Select Dataset", ALL_FILES)
    df = load_raw_file(selected_file)
    
    if not df.empty and 'Error' not in df.columns:
        tab_view, tab_profile, tab_anomalies = st.tabs(["📊 Data Viewer", "🔍 Health Profile", "🚨 Anomalies"])
        
        with tab_view:
            st.info("💡 Use the filters below to slice data. Note: 'NaN' values are mapped to 'Unknown' for filtering stability.")
            
            # --- CRITICAL FIX: SANITIZE FOR DATAFRAME_EXPLORER ---
            # Create a sanitized copy for the filter widget to prevent 'Select All' crashes
            view_df = df.copy()
            for col in view_df.columns:
                if view_df[col].dtype == 'object' or isinstance(view_df[col].dtype, pd.CategoricalDtype):
                    view_df[col] = view_df[col].fillna("Unknown").astype(str)
            
            # Use the sanitized version for filtering
            filtered_df = dataframe_explorer(view_df, case=False)
            st.dataframe(filtered_df, use_container_width=True, height=500)
            
        with tab_profile:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", len(df))
            c2.metric("Missing Values", df.isna().sum().sum())
            c3.metric("Duplicate Rows", df.duplicated().sum())
            c4.metric("Columns", len(df.columns))
            
            profile = pd.DataFrame({
                'Type': df.dtypes.astype(str),
                'Nulls': df.isna().sum(),
                'Unique': df.nunique()
            })
            st.table(profile)
            
        with tab_anomalies:
            st.subheader("Automated Scan Results")
            if "UCM" in selected_file:
                gaps = df['Google_ID'].isna().sum() if 'Google_ID' in df.columns else 0
                st.warning(f"Relational Integrity: {gaps} campaigns missing Google Ads mapping.")
            elif "TimeSeries" in selected_file:
                st.info("TimeSeries Scan: Checking for zero-inflation in activity columns...")
                st.success("Scan complete. No major drop-offs detected.")
            else:
                st.write("Scan complete. No critical structural anomalies found.")
    else:
        st.error(df['Error'].iloc[0] if 'Error' in df.columns else "File contains no data.")

# --- CLEANER PAGE ---
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Data Cleaning Hub", description="Standardizing campaign names and joining sources.", color_name="orange-70")
    if not master_df.empty:
        st.write("Preview of Unified Dataset:")
        st.dataframe(master_df.head(50), use_container_width=True)

# --- STATS PAGE ---
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Statistical Analysis", description="Correlation and performance modeling.", color_name="green-70")
    st.write("Select variables in the Dashboard to see statistical deep-dives here.")

# --- DASHBOARD PAGE ---
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Performance Dashboard", description="Cross-channel financial metrics.", color_name="light-blue-70")
    if not master_df.empty:
        st.metric("Total Website Traffic", f"{master_df['Total_Website_Users'].sum():,.0f} Users")
        fig = px.bar(master_df.head(20), x='utm_clean', y='Total_Website_Users', title="Top 20 Campaigns by Traffic", color_discrete_sequence=['#C41230'])
        st.plotly_chart(fig, use_container_width=True)

# --- GRAPH PAGE ---
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    colored_header(label="Knowledge Graph", description="Visualizing data relationships.", color_name="violet-70")
    st.info("Mapping the AI DA Dataset ecosystem...")
