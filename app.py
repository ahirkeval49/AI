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
CARD_BG = "#FFFFFF" # White cards per request
TEXT_DARK = "#050505"

st.set_page_config(page_title="CMU Command Center", layout="wide", initial_sidebar_state="collapsed")

# CSS: White Cards, Black Text for Selectors, and Clean Theme
st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; color: {WHITE}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {CMU_RED} !important; font-weight: 800; font-family: 'Segoe UI', sans-serif; }}
    
    .console-card {{
        background-color: {CARD_BG}; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 24px;
        border-top: 3px solid {CMU_RED};
    }}
    
    /* Ensure content inside white cards is readable */
    .console-card p, .console-card div, .console-card span, .console-card li, .console-card label {{
        color: {TEXT_DARK} !important;
    }}
    
    /* AUDITOR: Make file names in selectbox black */
    div[data-baseweb="select"] * {{ color: {TEXT_DARK} !important; }}
    
    div[data-testid="stMetricValue"] {{ color: {TEXT_DARK} !important; font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ color: {CMU_GREY} !important; font-weight: 700; text-transform: uppercase; }}
    div[data-testid="stMetric"] {{ background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid {CMU_RED}; }}
    
    /* Fast Nav Buttons (Same Tab) */
    div.stButton > button {{
        background-color: #222; color: {WHITE}; border: 1px solid #444; border-radius: 8px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 1px; font-size: 13px; height: 50px;
    }}
    div.stButton > button:hover {{ background-color: {CMU_RED}; border-color: {CMU_RED}; }}
    div.stButton > button * {{ color: {WHITE} !important; }}
    
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ROUTING ENGINE (Bypasses "New Tab" issue)
# ---------------------------------------------------------
if "page" in st.query_params:
    st.session_state.page = st.query_params["page"]
elif "page" not in st.session_state:
    st.session_state.page = "home"

def navigate(page_name):
    st.session_state.page = page_name
    st.query_params["page"] = page_name
    st.rerun()

current_page = st.session_state.page

# ---------------------------------------------------------
# DATA ENGINEERING: THE SPLIT-KEY PIPELINE
# ---------------------------------------------------------
ALL_FILES = [
    "2024-25_Campaign_Management_1769521985.csv", "2025-26_Campaign_Management_1769522231.csv",
    "GA_FY25_TimeSeries (1).csv", "GA_FY26_TimeSeries.csv",
    "GA_FY25_UTM_Totals_Jul2024-Jun2025.csv", "GA_FY26_UTM_Totals_Jul-Dec2025.csv",
    "GAds_AudiencePerformance_by_Campaign_FY24-FY26.csv", "GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad.csv",
    "GAds_FY25_Totals_Jul2024-Jun2025.csv", "GAds_FY26_Totals_Jul-Dec2025.csv",
    "LinkedIn_Ad_Performance_Feb2024_Dec2025.csv", "UCM Campaign Index.csv"
]

def sanitize_filename(name):
    return re.sub(r'[^a-z0-9]', '', name.lower().replace('.csv', '').replace('.xlsx', ''))

@st.cache_data(show_spinner=False)
def smart_load(target_name, skiprows=0):
    sanitized_target = sanitize_filename(target_name)
    search_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(__file__)), os.path.join(os.getcwd(), 'data')]
    for d in search_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            for f in os.listdir(d):
                if sanitize_filename(f) == sanitized_target or sanitize_filename(f).startswith(sanitized_target):
                    path = os.path.join(d, f)
                    try:
                        return pd.read_csv(path, skiprows=skiprows, on_bad_lines='skip') if f.lower().endswith('.csv') else pd.read_excel(path, skiprows=skiprows)
                    except Exception: pass
    return None

def find_col(df, aliases):
    if df is None or df.empty: return None
    for alias in aliases:
        for col in df.columns:
            if alias.lower() in str(col).lower(): return col
    return None

def clean_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

def normalize_key(series):
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

@st.cache_data(show_spinner=False)
def build_master_pipeline():
    try:
        # 1. LOAD INDEX
        idx_raw = smart_load('ucmcampaignindex')
        if idx_raw is None or idx_raw.empty: return pd.DataFrame()
        idx = pd.DataFrame()
        idx['board_key'] = normalize_key(idx_raw.get('Monday_Board_Name', pd.Series(dtype=str)))
        idx['ga_key'] = normalize_key(idx_raw.get('UTM campaign', pd.Series(dtype=str)))
        idx['Category'] = idx_raw.get('Category', pd.Series(dtype=str)).fillna('Uncategorized')
        idx['Vendor'] = idx_raw.get('Vendor', pd.Series(dtype=str)).fillna('Unknown')
        idx['Display_Name'] = idx_raw.get('Monday_Board_Name', pd.Series(dtype=str))
        idx = idx.drop_duplicates(subset=['board_key']).dropna(subset=['board_key'])

        # 2. LOAD MONDAY.COM (BUDGETS)
        mon_raw = pd.concat([smart_load(f) for f in ['202425campaignmanagement', '202526campaignmanagement'] if smart_load(f) is not None])
        if not mon_raw.empty:
            mon = pd.DataFrame()
            mon['board_key'] = normalize_key(mon_raw[find_col(mon_raw, ['name'])])
            mon['Budget'] = clean_num(mon_raw[find_col(mon_raw, ['budget'])])
            mon = mon.groupby('board_key').first().reset_index()
            idx = pd.merge(idx, mon, on='board_key', how='left')

        # 3. LOAD PLATFORMS (DYNAMIC SPEND CALCULATION)
        plat_dfs = []
        for f in ['gadsfy25totals', 'gadsfy26totals', 'gadsfy24fy26monthlyweeklyperformance']:
            g_df = smart_load(f)
            if g_df is not None and not g_df.empty:
                camp_col = find_col(g_df, ['campaign', 'ad name'])
                if camp_col:
                    g_df = g_df[~g_df[camp_col].astype(str).str.contains('Total', case=False, na=False)]
                    ext = pd.DataFrame()
                    ext['board_key'] = normalize_key(g_df[camp_col])
                    # Fix: Calculate Spend if Cost column is missing
                    conv = clean_num(g_df.get('Conversions', pd.Series(0)))
                    cpcv = clean_num(g_df.get('Cost / conv.', pd.Series(0)))
                    ext['Spend'] = clean_num(g_df.get('Cost', g_df.get('Spend', conv * cpcv)))
                    ext['Clicks'] = clean_num(g_df.get('Clicks', pd.Series(0)))
                    plat_dfs.append(ext)
        
        li_df = smart_load('linkedinadperformance')
        if li_df is not None:
            ext = pd.DataFrame()
            ext['board_key'] = normalize_key(li_df[find_col(li_df, ['campaign'])])
            ext['Spend'] = clean_num(li_df[find_col(li_df, ['total spend', 'cost'])])
            ext['Clicks'] = clean_num(li_df.get('Clicks', pd.Series(0)))
            plat_dfs.append(ext)
            
        plat_agg = pd.concat(plat_dfs).groupby('board_key').sum().reset_index() if plat_dfs else pd.DataFrame()

        # 4. LOAD GA
        ga_raw = pd.concat([smart_load(f) for f in ['gafy25utmtotals', 'gafy26utmtotals'] if smart_load(f) is not None])
        if not ga_raw.empty:
            if 'session campaign' not in str(ga_raw.columns).lower(): ga_raw.columns = ga_raw.iloc[0]; ga_raw = ga_raw[1:]
            ga_df = pd.DataFrame()
            ga_df['ga_key'] = normalize_key(ga_raw[find_col(ga_raw, ['campaign'])])
            ga_df['Users'] = clean_num(ga_raw[find_col(ga_raw, ['total users', 'users'])])
            ga_df['Eng_Rate'] = clean_num(ga_raw[find_col(ga_raw, ['engagement rate'])])
            ga_df['Duration'] = clean_num(ga_raw[find_col(ga_raw, ['average session duration', 'duration'])])
            ga_agg = ga_df.groupby('ga_key').agg({'Users':'sum', 'Eng_Rate':'mean', 'Duration':'mean'}).reset_index()
        else: ga_agg = pd.DataFrame()

        # 5. MERGE
        master = pd.merge(idx, plat_agg, on='board_key', how='outer')
        master = pd.merge(master, ga_agg, on='ga_key', how='outer')
        master['Display_Name'] = master['Display_Name'].fillna(master['board_key']).fillna(master['ga_key'])
        master.fillna({'Spend':0, 'Clicks':0, 'Users':0, 'Duration':0, 'Budget':0, 'Category':'Uncategorized', 'Vendor':'Other'}, inplace=True)
        master['CPWU'] = np.where(master['Users'] > 0, master['Spend'] / master['Users'], 0)
        master['CPQM'] = np.where((master['Users']*master['Duration']) > 0, master['Spend'] / ((master['Users']*master['Duration'])/60), 0)
        return master[(master['Spend'] > 0) | (master['Users'] > 0)]
    except Exception: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_timeseries():
    dfs = []
    for f in ['gafy25timeseries', 'gafy26timeseries']:
        df = smart_load(f)
        if df is not None and not df.empty:
            if 'session campaign' not in str(df.columns).lower(): df.columns = df.iloc[0]; df = df[1:]
            day_cols = [c for c in df.columns if 'day' in str(c).lower() and any(char.isdigit() for char in str(c))]
            if day_cols:
                m = df.melt(id_vars=[c for c in df.columns if c not in day_cols], value_vars=day_cols, var_name='D', value_name='u')
                m['Day'] = m['D'].astype(str).str.extract(r'(\d+)').astype(float)
                dfs.append(m[['Day', 'u']])
    if dfs:
        res = pd.concat(dfs)
        res['Users'] = clean_num(res['u'])
        return res.groupby('Day')['Users'].sum().reset_index().sort_values('Day')
    return pd.DataFrame()

master_df = build_master_pipeline()
ts_data = load_timeseries()

# ---------------------------------------------------------
# UI: FAST NAVIGATION
# ---------------------------------------------------------
cols = st.columns(5)
if cols[0].button("Nexus", use_container_width=True): navigate("home")
if cols[1].button("Auditor", use_container_width=True): navigate("explorer")
if cols[2].button("Dashboard", use_container_width=True): navigate("dashboard")
if cols[3].button("Strategist", use_container_width=True): navigate("analysis")
if cols[4].button("Knowledge Graph", use_container_width=True): navigate("graph")
st.markdown("<hr style='border-color: #333; margin-top: 0px;'>", unsafe_allow_html=True)

# ======================= NEXUS (3D GALAXY) =======================
if current_page == "home":
    st.markdown("<h1 style='text-align: center; font-size: 60px; margin-top: 50px;'>CMU COMMAND CENTER</h1>", unsafe_allow_html=True)
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

        // BACKGROUND
        const pos = new Float32Array(15000 * 3); for(let i=0; i<45000; i++) pos[i] = (Math.random()-0.5) * 100;
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({{size: 0.05, color: 0x444444}})));

        // CMU TEXT LOGIC
        const loader = new THREE.FontLoader();
        loader.load('https://unpkg.com/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json', (font) => {{
            const tGeo = new THREE.TextGeometry('CMU', {{ font: font, size: 5, height: 1 }});
            tGeo.computeBoundingBox(); tGeo.translate(-0.5*(tGeo.boundingBox.max.x-tGeo.boundingBox.min.x), -2.5, 0);
            scene.add(new THREE.Mesh(tGeo, new THREE.MeshPhongMaterial({{color: "{CMU_RED}"}})));
        }});

        const agents = [
            {{name: "Auditor", target: "explorer", pos: [12, 5, 2]}},
            {{name: "Dashboard", target: "dashboard", pos: [-12, -5, 4]}},
            {{name: "Strategist", target: "analysis", pos: [2, 10, -6]}},
            {{name: "Knowledge Graph", target: "graph", pos: [6, -10, -4]}}
        ];
        agents.forEach(a => {{
            const el = document.createElement('a'); el.className = 'node-label'; el.innerText = a.name; el.href = "?page=" + a.target; el.target = "_top";
            document.body.appendChild(el); a.el = el;
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 32), new THREE.MeshPhongMaterial({{color: 0x666666}}));
            mesh.position.set(...a.pos); scene.add(mesh); a.mesh = mesh;
        }});

        camera.position.z = 35;
        function animate(){{ requestAnimationFrame(animate); 
            agents.forEach(a => {{ const vector = a.mesh.position.clone().project(camera);
                a.el.style.left = (vector.x + 1) / 2 * window.innerWidth + 'px';
                a.el.style.top = -(vector.y - 1) / 2 * window.innerHeight + 'px';
            }});
            controls.update(); renderer.render(scene, camera); 
        }} animate();
    </script></body></html>
    """
    components.html(galaxy_js, height=700)

# ======================= AUDITOR =======================
elif current_page == "explorer":
    st.markdown("<h1>Forensic Auditor</h1>")
    st.markdown("<div class='console-card'><h3>Planning Phase & Raw File Inspection</h3>", unsafe_allow_html=True)
    f = st.selectbox("Select File", ALL_FILES)
    df = smart_load(f)
    if df is not None:
        st.write(f"**Inspecting:** {f}")
        st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= DASHBOARD =======================
elif current_page == "dashboard":
    st.markdown("<h1>Pipeline Dashboard</h1>")
    v_sel = st.multiselect("Platforms", master_df['Vendor'].unique(), default=master_df['Vendor'].unique())
    f_df = master_df[master_df['Vendor'].isin(v_sel)]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spend", f"${f_df['Spend'].sum():,.0f}")
    m2.metric("Total Users", f"{f_df['Users'].sum():,.0f}")
    m3.metric("Avg CPA", f"${(f_df['Spend'].sum()/f_df['Users'].sum()):.2f}" if f_df['Users'].sum()>0 else "$0")

    st.markdown("<div class='console-card'><h3>Campaign Investment Overview</h3>", unsafe_allow_html=True)
    fig_spend = px.bar(f_df.sort_values('Spend', ascending=False).head(15), x='Display_Name', y='Spend', color_discrete_sequence=[CMU_RED])
    fig_spend.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=TEXT_DARK)
    st.plotly_chart(fig_spend, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= STRATEGIST =======================
elif current_page == "analysis":
    st.markdown("<h1>Quantitative Strategist</h1>")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='console-card'><h3>Ad Fatigue Model</h3>", unsafe_allow_html=True)
        v_df = master_df[(master_df['Spend']>0) & (master_df['Users']>0)]
        fig_p = px.scatter(v_df, x='Spend', y='Users', hover_name='Display_Name', trendline="lowess", color_discrete_sequence=[CMU_RED])
        fig_p.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=TEXT_DARK)
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='console-card'><h3>Cost Per Quality Minute</h3>", unsafe_allow_html=True)
        st.dataframe(master_df.sort_values('CPQM').head(10)[['Display_Name', 'CPQM', 'Spend', 'Users']].style.format({'CPQM':'${:.2f}'}), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================= GRAPH =======================
elif current_page == "graph":
    st.markdown("<h1>Relational Knowledge Graph</h1>")
    if not master_df.empty:
        net = Network(height="700px", width="100%", bgcolor=WHITE, font_color=TEXT_DARK)
        net.add_node("CMU", size=50, color=CMU_RED, label="CMU Hub")
        for v in master_df['Vendor'].unique():
            net.add_node(v, size=30, color="#555", label=v); net.add_edge("CMU", v)
            for _, r in master_df[master_df['Vendor']==v].sort_values('Users', ascending=False).head(10).iterrows():
                net.add_node(r['Display_Name'], size=15, color=CMU_RED, label=r['Display_Name'][:20]); net.add_edge(v, r['Display_Name'])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=".") as tmp:
            net.save_graph(tmp.name)
            components.html(open(tmp.name, 'r').read(), height=750)
            os.remove(tmp.name)
