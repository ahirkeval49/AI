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

# Clean CSS with White Cards & Custom Button Styling
st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; color: {WHITE}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {CMU_RED} !important; font-weight: 800; font-family: 'Segoe UI', sans-serif; }}
    p, span, label, li, td, th {{ font-family: 'Segoe UI', sans-serif; }}
    
    .console-card {{
        background-color: {CARD_BG}; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 24px;
        border-top: 3px solid {CMU_RED};
    }}
    
    .console-card p, .console-card div, .console-card span, .console-card li {{ color: {TEXT_DARK} !important; }}
    .console-card h3, .console-card h4 {{ color: {CMU_RED} !important; }}
    div[data-baseweb="select"] * {{ color: {TEXT_DARK} !important; }}
    
    div[data-testid="stMetricValue"] {{ color: {TEXT_DARK} !important; font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ color: {CMU_GREY} !important; font-weight: 700; text-transform: uppercase; }}
    div[data-testid="stMetric"] {{ background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid {CMU_RED}; }}
    
    div.stButton > button {{
        background-color: #222; color: {WHITE}; border: 1px solid #444; border-radius: 8px;
        font-weight: 700; text-transform: uppercase; letter-spacing: 1px; font-size: 13px; height: 50px;
        transition: 0.2s ease-in-out;
    }}
    div.stButton > button:hover {{ background-color: {CMU_RED}; border-color: {CMU_RED}; color: {WHITE} !important; transform: translateY(-2px); }}
    div.stButton > button * {{ color: {WHITE} !important; }}
    
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# ROUTING ENGINE (Listener for 3D Node Redirects)
# ---------------------------------------------------------
# Check query parameters first
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

@st.cache_data(ttl=60)
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

def normalize_key(series):
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

def clean_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

@st.cache_data
def build_master_pipeline():
    try:
        idx_raw = smart_load('ucmcampaignindex')
        if idx_raw is None or idx_raw.empty: return pd.DataFrame()
        
        idx = pd.DataFrame()
        idx['board_key'] = normalize_key(idx_raw.get('Monday_Board_Name', pd.Series(dtype=str)))
        idx['ga_key'] = normalize_key(idx_raw.get('UTM campaign', pd.Series(dtype=str)))
        idx['Category'] = idx_raw.get('Category', pd.Series(dtype=str)).fillna('Uncategorized')
        idx['Vendor'] = idx_raw.get('Vendor', pd.Series(dtype=str)).fillna('Unknown')
        idx['Display_Name'] = idx_raw.get('Monday_Board_Name', pd.Series(dtype=str))
        idx = idx.drop_duplicates(subset=['board_key']).dropna(subset=['board_key'])

        mon1 = smart_load('202425campaignmanagement')
        mon2 = smart_load('202526campaignmanagement')
        mon_raw = pd.concat([df for df in [mon1, mon2] if df is not None])
        if not mon_raw.empty:
            mon = pd.DataFrame()
            mon_name_col = find_col(mon_raw, ['name'])
            mon_bud_col = find_col(mon_raw, ['budget'])
            mon_run_col = find_col(mon_raw, ['run dates'])
            mon['board_key'] = normalize_key(mon_raw[mon_name_col]) if mon_name_col else ""
            mon['Budget'] = clean_num(mon_raw[mon_bud_col]) if mon_bud_col else 0.0
            mon['Run_Dates'] = mon_raw[mon_run_col].astype(str) if mon_run_col else ""
            mon = mon.groupby('board_key').first().reset_index()
            idx = pd.merge(idx, mon, on='board_key', how='left')

        plat_dfs = []
        for f in ['gadsfy25totals', 'gadsfy26totals', 'gadsfy24fy26monthlyweeklyperformance']:
            g_df = smart_load(f)
            if g_df is not None and not g_df.empty:
                camp_col = find_col(g_df, ['campaign', 'ad name'])
                if camp_col:
                    g_df = g_df[~g_df[camp_col].astype(str).str.contains('Total', case=False, na=False)]
                    ext = pd.DataFrame()
                    ext['board_key'] = normalize_key(g_df[camp_col])
                    cost_col = find_col(g_df, ['cost', 'spend'])
                    clk_col = find_col(g_df, ['clicks'])
                    cpc_col = find_col(g_df, ['avg. cpc', 'cpc'])
                    if cost_col: ext['Spend'] = clean_num(g_df[cost_col])
                    elif clk_col and cpc_col: ext['Spend'] = clean_num(g_df[clk_col]) * clean_num(g_df[cpc_col])
                    else: ext['Spend'] = 0.0
                    ext['Clicks'] = clean_num(g_df[clk_col]) if clk_col else 0.0
                    plat_dfs.append(ext)
        
        li_df = smart_load('linkedinadperformance')
        if li_df is not None:
            ext = pd.DataFrame()
            ext['board_key'] = normalize_key(li_df['Campaign Name'] if 'Campaign Name' in li_df.columns else li_df.iloc[:, 0])
            ext['Spend'] = clean_num(li_df['Total Spend'] if 'Total Spend' in li_df.columns else pd.Series(0))
            ext['Clicks'] = clean_num(li_df['Clicks'] if 'Clicks' in li_df.columns else pd.Series(0))
            plat_dfs.append(ext)
                
        plat_agg = pd.concat(plat_dfs).groupby('board_key').sum().reset_index() if plat_dfs else pd.DataFrame(columns=['board_key', 'Spend', 'Clicks'])

        ga_dfs = []
        for f in ['gafy25utmtotals', 'gafy26utmtotals']:
            ga_raw = smart_load(f, skiprows=0)
            if ga_raw is not None and not ga_raw.empty:
                if 'session campaign' not in str(ga_raw.columns).lower(): ga_raw.columns = ga_raw.iloc[0]; ga_raw = ga_raw[1:]
                camp_col = find_col(ga_raw, ['session campaign', 'campaign'])
                if camp_col:
                    ext = pd.DataFrame()
                    ext['ga_key'] = normalize_key(ga_raw[camp_col])
                    ext['Users'] = clean_num(ga_raw[find_col(ga_raw, ['total users', 'users'])])
                    ext['Eng_Rate'] = clean_num(ga_raw[find_col(ga_raw, ['engagement rate'])])
                    ext['Duration'] = clean_num(ga_raw[find_col(ga_raw, ['average session duration', 'duration'])])
                    ga_dfs.append(ext)
                    
        ga_agg = pd.concat(ga_dfs).groupby('ga_key').agg({'Users':'sum', 'Eng_Rate':'mean', 'Duration':'mean'}).reset_index() if ga_dfs else pd.DataFrame(columns=['ga_key', 'Users', 'Eng_Rate', 'Duration'])

        master = pd.merge(idx, plat_agg, on='board_key', how='outer')
        master = pd.merge(master, ga_agg, on='ga_key', how='outer')
        master['Display_Name'] = master['Display_Name'].fillna(master['board_key']).fillna(master['ga_key']).replace('', 'Unknown Campaign')
        master['Category'] = master['Category'].fillna('Uncategorized')
        master['Vendor'] = master['Vendor'].fillna('Platform/Organic')
        master.fillna({'Spend':0, 'Clicks':0, 'Users':0, 'Eng_Rate':0, 'Duration':0, 'Budget':0}, inplace=True)
        master['CPQM'] = np.where(master['Users'] > 0, master['Spend'] / ((master['Users'] * master['Duration']) / 60 + 0.1), 0)
        return master[(master['Spend'] > 0) | (master['Users'] > 0)]
    except Exception: return pd.DataFrame()

@st.cache_data
def load_timeseries():
    ts1 = smart_load('gafy25timeseries')
    ts2 = smart_load('gafy26timeseries')
    dfs = []
    for df in [ts1, ts2]:
        if df is not None and not df.empty:
            if 'session campaign' not in str(df.columns).lower(): df.columns = df.iloc[0]; df = df[1:]
            day_cols = [c for c in df.columns if 'day' in str(c).lower() and any(char.isdigit() for char in str(c))]
            if day_cols:
                m = df.melt(id_vars=[c for c in df.columns if c not in day_cols], value_vars=day_cols, var_name='D', value_name='u')
                m['Day'] = m['D'].astype(str).str.extract(r'(\d+)').astype(float)
                m['Users'] = clean_num(m['u'])
                dfs.append(m[['Day', 'Users']])
    return pd.concat(dfs).groupby('Day')['Users'].sum().reset_index().sort_values('Day') if dfs else pd.DataFrame()

master_df = build_master_pipeline()
ts_data = load_timeseries()

# ---------------------------------------------------------
# UI: NAVIGATION
# ---------------------------------------------------------
nav_cols = st.columns(5)
if nav_cols[0].button("Nexus", use_container_width=True): navigate("home")
if nav_cols[1].button("Auditor", use_container_width=True): navigate("explorer")
if nav_cols[2].button("Dashboard", use_container_width=True): navigate("dashboard")
if nav_cols[3].button("Strategist", use_container_width=True): navigate("analysis")
if nav_cols[4].button("Knowledge Graph", use_container_width=True): navigate("graph")

st.markdown("<hr style='border-color: #333; margin-top: 0px;'>", unsafe_allow_html=True)

# ======================= HOME: 3D CMU GALAXY =======================
if current_page == "home":
    st.markdown(f"<h1 style='text-align: center; font-size: 60px; margin-top: 50px;'>CMU COMMAND CENTER</h1>", unsafe_allow_html=True)
    
    # 3D Galaxy with Iframe Breakout Navigation
    three_js_galaxy = f"""
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; background: {BLACK}; overflow: hidden; }}
        .node-label {{
            position: absolute; background: rgba(196,18,48,0.9); border: 2px solid {WHITE};
            padding: 8px 16px; border-radius: 8px; font-weight: bold; cursor: pointer;
            transition: 0.3s; color: {WHITE}; text-decoration: none; font-size: 12px;
            text-transform: uppercase; text-align: center;
        }}
        .node-label:hover {{ transform: scale(1.1); background: {WHITE}; color: {CMU_RED}; }}
    </style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color("{BLACK}");
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true}});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true; controls.autoRotateSpeed = 0.5; controls.enableDamping = true;
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8); scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2); dirLight.position.set(10, 20, 10); scene.add(dirLight);

        // Background Particles
        const count = 15000; const pos = new Float32Array(count * 3);
        for(let i=0; i<count*3; i++) pos[i] = (Math.random() - 0.5) * 100;
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        scene.add(new THREE.Points(geo, new THREE.PointsMaterial({{size: 0.05, color: 0x444444}})));

        // CMU Centerpiece
        const loader = new THREE.FontLoader();
        loader.load('https://unpkg.com/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json', (font) => {{
            const tGeo = new THREE.TextGeometry('CMU', {{ font: font, size: 5, height: 1 }});
            tGeo.computeBoundingBox(); tGeo.translate(-0.5*(tGeo.boundingBox.max.x-tGeo.boundingBox.min.x), -2.5, 0);
            scene.add(new THREE.Mesh(tGeo, new THREE.MeshPhongMaterial({{color: "{CMU_RED}"}})));
        }});

        const agents = [
            {{name: "Auditor", target: "explorer", pos: [15, 5, 2]}},
            {{name: "Dashboard", target: "dashboard", pos: [-15, -5, 4]}},
            {{name: "Strategist", target: "analysis", pos: [2, 12, -6]}},
            {{name: "Knowledge Graph", target: "graph", pos: [6, -12, -4]}}
        ];

        agents.forEach(a => {{
            const el = document.createElement('div'); el.className = 'node-label'; el.innerText = a.name;
            // CRITICAL: window.parent.location used to bypass iframe sandbox
            el.onclick = () => {{ window.parent.location.href = window.parent.location.pathname + "?page=" + a.target; }};
            document.body.appendChild(el); a.el = el;
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(1.2, 32, 32), new THREE.MeshPhongMaterial({{color: 0x666666}}));
            mesh.position.set(...a.pos); scene.add(mesh); a.mesh = mesh;
        }});

        camera.position.z = 35;
        function animate(){{ requestAnimationFrame(animate); 
            agents.forEach(a => {{
                const vector = a.mesh.position.clone().project(camera);
                a.el.style.left = (vector.x + 1) / 2 * window.innerWidth + 'px';
                a.el.style.top = -(vector.y - 1) / 2 * window.innerHeight + 'px';
            }});
            controls.update(); renderer.render(scene, camera); 
        }} animate();
    </script></body></html>
    """
    st.markdown("<div style='margin-top: -30px;'></div>", unsafe_allow_html=True)
    components.html(three_js_galaxy, height=800)

# ======================= AGENT 1: AUDITOR =======================
elif current_page == "explorer":
    st.markdown("<h1>Forensic Auditor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='console-card'><h3>Planning Phase Inspector</h3>", unsafe_allow_html=True)
    f = st.selectbox("Select File", ALL_FILES)
    df = smart_load(f)
    if df is not None:
        st.write(f"**Inspecting:** {f}")
        st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= AGENT 4: DASHBOARD =======================
elif current_page == "dashboard":
    st.markdown("<h1>Pipeline Dashboard</h1>", unsafe_allow_html=True)
    v_sel = st.multiselect("Platforms", master_df['Vendor'].unique(), default=master_df['Vendor'].unique())
    f_df = master_df[master_df['Vendor'].isin(v_sel)]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spend", f"${f_df['Spend'].sum():,.0f}")
    m2.metric("Total Users", f"{f_df['Users'].sum():,.0f}")
    m3.metric("Avg CPA", f"${(f_df['Spend'].sum() / f_df['Users'].sum() if f_df['Users'].sum() > 0 else 0):.2f}")
    st.markdown("<div class='console-card'><h3>Campaign Spend & Investment Overview</h3>", unsafe_allow_html=True)
    fig = px.bar(f_df.sort_values('Spend', ascending=False).head(12), x='Display_Name', y='Spend', color_discrete_sequence=[CMU_RED])
    fig.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=TEXT_DARK)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= AGENT 3: STRATEGIST =======================
elif current_page == "analysis":
    st.markdown("<h1>Quantitative Strategist</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='console-card'><h3>Ad Fatigue Model</h3>", unsafe_allow_html=True)
        fig_p = px.scatter(master_df[master_df['Spend']>0], x='Spend', y='Users', hover_name='Display_Name', trendline="lowess", color_discrete_sequence=[CMU_RED])
        fig_p.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color=TEXT_DARK)
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='console-card'><h3>Efficiency: Lowest CPQM</h3>", unsafe_allow_html=True)
        st.dataframe(master_df.sort_values('CPQM').head(10)[['Display_Name', 'CPQM', 'Spend']], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================= AGENT 5: GRAPH =======================
elif current_page == "graph":
    st.markdown("<h1>Ecosystem Relational Graph</h1>", unsafe_allow_html=True)
    if not master_df.empty:
        net = Network(height="700px", width="100%", bgcolor=WHITE, font_color=TEXT_DARK)
        net.add_node("CMU", size=50, color=CMU_RED, label="CMU Hub")
        for v in master_df['Vendor'].unique():
            if str(v) != '0':
                net.add_node(v, size=30, color="#555555", label=v); net.add_edge("CMU", v)
                for _, r in master_df[master_df['Vendor']==v].sort_values('Users', ascending=False).head(10).iterrows():
                    net.add_node(str(r['Display_Name']), size=15, color=CMU_RED, label=str(r['Display_Name'])[:20]); net.add_edge(v, str(r['Display_Name']))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=".") as tmp:
            net.save_graph(tmp.name)
            components.html(open(tmp.name, 'r').read(), height=750)
            os.remove(tmp.name)
