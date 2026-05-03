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

# ---------------------------------------------------------
# CMU BRAND COLORS & CONFIGURATION
# ---------------------------------------------------------
CMU_RED = "#C41230"
CMU_GREY = "#6D6E71"
WHITE = "#FFFFFF"
BLACK = "#050505"

st.set_page_config(page_title="CMU Data Systems | Command Center", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; color: {WHITE}; }}
    .stSidebar {{ background-color: #111111; color: {WHITE}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {CMU_RED} !important; font-weight: 900; font-family: 'Segoe UI', sans-serif; letter-spacing: -0.5px; }}
    
    .console-card {{
        background-color: {WHITE};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(196, 18, 48, 0.15);
        color: #111111 !important;
        margin-bottom: 24px;
        border-top: 4px solid {CMU_RED};
    }}
    .console-card h1, .console-card h2, .console-card h3, .console-card h4 {{ color: {CMU_RED} !important; }}
    .console-card p, .console-card strong, .console-card li {{ color: #333333 !important; }}
    
    div[data-testid="stPlotlyChart"] {{ background-color: {WHITE}; border-radius: 12px; padding: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
    div[data-testid="stMetricValue"] {{ color: #111111 !important; font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ color: {CMU_GREY} !important; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; }}
    div[data-testid="stMetric"] {{ background-color: {WHITE}; padding: 15px; border-radius: 12px; border-left: 5px solid {CMU_RED}; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
    
    .nav-grid {{ display: flex; justify-content: center; gap: 15px; padding: 10px; margin-bottom: 5px; z-index: 100; position: relative; }}
    .nav-card {{
        background: rgba(255, 255, 255, 0.05); border: 1px solid {CMU_GREY}; 
        backdrop-filter: blur(10px); border-radius: 8px; padding: 10px;
        width: 140px; text-align: center; color: {WHITE} !important; text-decoration: none;
        transition: 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    .nav-card:hover {{ border-color: {CMU_RED}; transform: translateY(-5px); box-shadow: 0 8px 20px rgba(196,18,48,0.5); background: rgba(196, 18, 48, 0.15); }}
    .nav-title {{ font-size: 10px; font-weight: 900; color: {WHITE}; margin-top: 5px; letter-spacing: 1px; text-transform: uppercase; }}
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = {"audit_logs": {}, "synthesis_stats": {}, "model_results": {}}

query_params = st.query_params.to_dict()
current_page = query_params.get("page", ["home"])[0] if isinstance(query_params.get("page"), list) else query_params.get("page", "home")

# ---------------------------------------------------------
# 2. SMART LOADER ENGINE: SOLVES GITHUB PATHING & EXCEL ISSUES
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
def smart_load(filename, skiprows=0):
    """
    Searches both Root and /data folders. 
    Automatically handles .csv or .xlsx regardless of what is requested.
    """
    target_name, _ = os.path.splitext(filename.lower())
    search_dirs = [os.getcwd(), os.path.join(os.getcwd(), 'data')]
    
    for d in search_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            for f in os.listdir(d):
                f_name, f_ext = os.path.splitext(f.lower())
                if f_name == target_name and f_ext in ['.csv', '.xlsx', '.xls']:
                    path = os.path.join(d, f)
                    try:
                        if f_ext == '.csv':
                            return pd.read_csv(path, skiprows=skiprows)
                        else:
                            return pd.read_excel(path, skiprows=skiprows)
                    except Exception as e:
                        return None
    return None

def find_col(df, aliases):
    if df is None or df.empty: return None
    for alias in aliases:
        if alias in df.columns: return alias
    return None

def clean_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').fillna(0.0)

def normalize_key(series):
    return series.astype(str).str.lower().str.replace(r'[^a-z0-9]', '', regex=True).replace('nan', '')

@st.cache_data
def build_master_hub():
    try:
        # Load Index safely
        idx = smart_load('UCM Campaign Index')
        if idx is not None and not idx.empty:
            utm_col = find_col(idx, ['UTM campaign', 'Campaign_ID', 'UTM_Combined_ID', 'Landing Page (UTM)'])
            idx['utm_clean'] = normalize_key(idx[utm_col]) if utm_col else ""
            if 'Category' not in idx.columns: idx['Category'] = "Uncategorized"
        else:
            return pd.DataFrame()

        # GAds Pipeline
        g_dfs, v_dfs = [], []
        for f in ['GAds_FY25_Totals_Jul2024-Jun2025', 'GAds_FY26_Totals_Jul-Dec2025', 'GAds_FY24-FY26_Monthly_Weekly_Performance_by_Ad']:
            df = smart_load(f)
            if df is not None and not df.empty:
                g_key = find_col(df, ['Ad name', 'Campaign', 'Campaign Name'])
                if g_key:
                    df['utm_clean'] = normalize_key(df[g_key])
                    if find_col(df, ['Cost', 'Spend']):
                        df['Cost'] = clean_num(df[find_col(df, ['Cost', 'Spend'])])
                        g_dfs.append(df[['utm_clean', 'Cost']])
                    
                    v_cols = {'Video played to 25%': 'V25', 'Video played to 50%': 'V50', 'Video played to 75%': 'V75', 'Video played to 100%': 'V100'}
                    has_video = False
                    for k, v in v_cols.items():
                        if k in df.columns:
                            df[v] = clean_num(df[k])
                            has_video = True
                    if has_video:
                        v_dfs.append(df[['utm_clean'] + list(v_cols.values())])

        g_agg = pd.concat(g_dfs).groupby('utm_clean').agg(GAds_Spend=('Cost', 'sum')).reset_index() if g_dfs else pd.DataFrame(columns=['utm_clean', 'GAds_Spend'])
        v_agg = pd.concat(v_dfs).groupby('utm_clean').mean().reset_index() if v_dfs else pd.DataFrame()

        # LinkedIn Pipeline
        li_agg = pd.DataFrame(columns=['utm_clean', 'LI_Spend'])
        li = smart_load('LinkedIn_Ad_Performance_Feb2024_Dec2025')
        if li is not None and not li.empty:
            li_key = find_col(li, ['Campaign Name', 'Campaign'])
            if li_key:
                li['utm_clean'] = normalize_key(li[li_key])
                li_spend = find_col(li, ['Total Spend', 'Spend', 'Cost'])
                li['LI_Spend'] = clean_num(li[li_spend]) if li_spend else 0.0
                li_agg = li.groupby('utm_clean').agg(LI_Spend=('LI_Spend', 'sum')).reset_index()

        # GA Metrics Pipeline
        ga_dfs = []
        for f in ['GA_FY25_UTM_Totals_Jul2024-Jun2025', 'GA_FY26_UTM_Totals_Jul-Dec2025']:
            _df = smart_load(f, skiprows=0) # Smart load checks headers
            if _df is not None and not _df.empty:
                # If GA exports have an extra summary row, standard drop
                if 'Session campaign' not in _df.columns and len(_df) > 1:
                    _df.columns = _df.iloc[0]
                    _df = _df[1:]
                    
                ga_key = find_col(_df, ['Session campaign', 'Campaign'])
                if ga_key:
                    _df['utm_clean'] = normalize_key(_df[ga_key])
                    u_col = find_col(_df, ['Total users', 'Users'])
                    e_col = find_col(_df, ['Engagement rate'])
                    d_col = find_col(_df, ['Average session duration'])
                    
                    _df['Total_Users'] = clean_num(_df[u_col]) if u_col else 0.0
                    _df['Engagement_Rate'] = clean_num(_df[e_col]) if e_col else 0.0
                    _df['Session_Duration'] = clean_num(_df[d_col]) if d_col else 0.0
                    ga_dfs.append(_df[['utm_clean', 'Total_Users', 'Engagement_Rate', 'Session_Duration']])
                    
        ga_agg = pd.concat(ga_dfs).groupby('utm_clean').agg(Total_Users=('Total_Users', 'sum'), Engagement_Rate=('Engagement_Rate', 'mean'), Session_Duration=('Session_Duration', 'mean')).reset_index() if ga_dfs else pd.DataFrame(columns=['utm_clean', 'Total_Users', 'Engagement_Rate', 'Session_Duration'])

        # Master Synthesis Join
        hub = idx if not idx.empty else pd.DataFrame(columns=['utm_clean'])
        hub = pd.merge(hub, ga_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, g_agg, on='utm_clean', how='left')
        hub = pd.merge(hub, li_agg, on='utm_clean', how='left')
        if not v_agg.empty: hub = pd.merge(hub, v_agg, on='utm_clean', how='left')
            
        hub.fillna(0.0, inplace=True)
        hub['Total_Spend'] = hub['GAds_Spend'] + hub['LI_Spend']
        hub['CPWU'] = hub['Total_Spend'].div(hub['Total_Users'].replace(0, np.nan)).fillna(0.0)
        
        hub['Vendor'] = np.where(hub['GAds_Spend'] > hub['LI_Spend'], 'Google Ads', np.where(hub['LI_Spend'] > 0, 'LinkedIn', 'Organic/Other'))
        return hub
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def load_timeseries_data():
    try:
        ts_dfs = []
        for f in ['GA_FY25_TimeSeries (1)', 'GA_FY26_TimeSeries']:
            df = smart_load(f, skiprows=0)
            if df is not None and not df.empty:
                if 'Date' not in df.columns and len(df) > 1:
                    df.columns = df.iloc[0]
                    df = df[1:]
                ts_dfs.append(df)
                
        if ts_dfs:
            ts = pd.concat(ts_dfs, ignore_index=True)
            date_col = find_col(ts, ['Date', 'Day', 'Day Index'])
            s_col = find_col(ts, ['Sessions', 'sessions'])
            u_col = find_col(ts, ['Total users', 'Users'])
            if date_col and s_col and u_col:
                ts['day'] = ts[date_col].astype(str)
                ts['sessions'] = clean_num(ts[s_col])
                ts['users'] = clean_num(ts[u_col])
                return ts[['day', 'sessions', 'users']].groupby('day').sum().reset_index().sort_values('day').tail(90)
    except: pass
    return pd.DataFrame()

master_df = build_master_hub()
ts_data = load_timeseries_data()

# ---------------------------------------------------------
# 3. GLOBAL UI: NAVIGATION
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

# ======================= HOME: 3D CMU GALAXY =======================
if current_page == "home":
    st.markdown("<style>.block-container { padding: 0; } header { visibility: hidden; }</style>", unsafe_allow_html=True)
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    
    three_js_galaxy = f"""
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; background: {BLACK}; overflow: hidden; font-family: sans-serif; }}
        .node-label {{
            position: absolute; background: rgba(0,0,0,0.85); border: 2px solid {CMU_RED};
            padding: 8px 16px; border-radius: 8px; font-weight: bold; cursor: pointer;
            transition: 0.3s; pointer-events: auto; text-decoration: none; color: {WHITE}; font-size: 11px;
            box-shadow: 0 4px 15px rgba(196,18,48,0.4);
        }}
        .node-label:hover {{ background: {CMU_RED}; transform: scale(1.1); }}
    </style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color("{BLACK}");
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true, alpha: true}});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true; controls.autoRotateSpeed = 0.5; controls.enableDamping = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8); scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2); dirLight.position.set(10, 20, 10); scene.add(dirLight);

        const count = 35000; const pos = new Float32Array(count * 3); const colors = new Float32Array(count * 3);
        const cRed = new THREE.Color("{CMU_RED}"); const cGrey = new THREE.Color("{CMU_GREY}"); const cWhite = new THREE.Color("{WHITE}");
        for(let i=0; i<count; i++){{
            const r = 25 * Math.cbrt(Math.random()); const t = Math.random()*2*Math.PI; const p = Math.acos(2*Math.random()-1);
            pos[i*3] = r * Math.sin(p) * Math.cos(t); pos[i*3+1] = r * Math.sin(p) * Math.sin(t); pos[i*3+2] = r * Math.cos(p);
            const rand = Math.random();
            let c = cRed;
            if(rand > 0.6) c = cWhite; else if(rand > 0.3) c = cGrey;
            colors[i*3] = c.r; colors[i*3+1] = c.g; colors[i*3+2] = c.b;
        }}
        const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const mat = new THREE.PointsMaterial({{size: 0.06, vertexColors: true, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending }});
        scene.add(new THREE.Points(geo, mat));

        const loader = new THREE.FontLoader();
        loader.load('https://unpkg.com/three@0.128.0/examples/fonts/helvetiker_bold.typeface.json', function (font) {{
            const textGeo = new THREE.TextGeometry('CMU', {{ font: font, size: 6, height: 1.5, curveSegments: 10, bevelEnabled: true, bevelThickness: 0.1, bevelSize: 0.05 }});
            textGeo.computeBoundingBox(); const centerOffset = -0.5 * (textGeo.boundingBox.max.x - textGeo.boundingBox.min.x);
            textGeo.translate(centerOffset, -1.2, 0);
            const textMat = new THREE.MeshPhongMaterial({{color: "{CMU_RED}", emissive: 0x400000, shininess: 100}});
            scene.add(new THREE.Mesh(textGeo, textMat));
        }});

        const agents = [
            {{name: "AUDITOR", url: "?page=explorer", color: "{CMU_GREY}", pos: [12, 6, 2]}},
            {{name: "ALCHEMIST", url: "?page=cleaner", color: "{CMU_GREY}", pos: [-12, -6, 5]}},
            {{name: "STRATEGIST", url: "?page=analysis", color: "{WHITE}", pos: [6, -12, -5]}},
            {{name: "ARCHITECT", url: "?page=dashboard", color: "{CMU_RED}", pos: [0, 14, 5]}},
            {{name: "KNOWLEDGE GRAPH", url: "?page=graph", color: "{WHITE}", pos: [-10, 10, -8]}}
        ];

        agents.forEach(a => {{
            const el = document.createElement('a'); el.className = 'node-label';
            el.innerText = a.name; el.href = a.url; el.target = "_self";
            document.body.appendChild(el); a.el = el;
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(1.2, 32, 32), new THREE.MeshPhongMaterial({{color: a.color, shininess: 100}}));
            mesh.position.set(...a.pos); scene.add(mesh); a.mesh = mesh;
        }});

        camera.position.z = 35;
        function animate(){{
            requestAnimationFrame(animate);
            agents.forEach(a => {{
                const vector = a.mesh.position.clone().project(camera);
                a.el.style.left = (vector.x + 1) / 2 * window.innerWidth + 'px';
                a.el.style.top = -(vector.y - 1) / 2 * window.innerHeight + 'px';
            }});
            controls.update(); renderer.render(scene, camera);
        }}
        animate();
    </script></body></html>
    """
    st.markdown(f"<h1 style='text-align: center; color: {WHITE}; font-weight: 900; margin-top: -15px; text-transform: uppercase;'>CMU DATA SYSTEMS</h1>", unsafe_allow_html=True)
    components.html(three_js_galaxy, height=850)

# ======================= AGENT 1: FORENSIC AUDITOR =======================
elif current_page == "explorer":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1>🕵️ Step 1: Forensic Auditor</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="console-card">
        <h3 style="margin-top: 0;">Integrity Rules Engine</h3>
        <p style="margin-bottom: 0;">The Auditor scans raw files directly from the repository root to identify <strong>Orphan IDs</strong> and anomalies before they corrupt downstream models.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='color: white;'>Interactive Schema Explorer</h3>", unsafe_allow_html=True)
    f = st.selectbox("Select Target File", ALL_FILES)
    df = smart_load(f)
    
    if df is not None and not df.empty:
        t1, t2, t3, t4 = st.tabs(["📊 Data Viewer", "🔍 Data Profile", "📈 Descriptive Stats", "🚨 Orphan ID Scan"])
        with t1:
            st.markdown("<div class='console-card'>", unsafe_allow_html=True)
            st.dataframe(df.head(100), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with t2:
            st.markdown("<div class='console-card'>", unsafe_allow_html=True)
            profile = pd.DataFrame({'Data Type': df.dtypes.astype(str), 'Null Count': df.isna().sum(), 'Unique Values': df.nunique()})
            st.dataframe(profile, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with t3:
            st.markdown("<div class='console-card'>", unsafe_allow_html=True)
            st.dataframe(df.describe(include='all').T, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with t4:
            st.markdown("<div class='console-card'>", unsafe_allow_html=True)
            st.markdown("<h4>Orphan ID Detection</h4>", unsafe_allow_html=True)
            if 'UCM Campaign Index' in f:
                st.info("Viewing the Master Index. No cross-reference possible.")
            else:
                idx_df = smart_load('UCM Campaign Index')
                if idx_df is not None and not idx_df.empty:
                    idx_keys = set(normalize_key(idx_df[find_col(idx_df, ['UTM campaign', 'Campaign_ID'])]).tolist())
                    t_key = find_col(df, ['Ad name', 'Campaign', 'Session campaign'])
                    if t_key:
                        orphans = list(set(normalize_key(df[t_key]).tolist()) - idx_keys)
                        orphans = [o for o in orphans if o and o != 'nan']
                        if orphans:
                            st.error(f"⚠️ {len(orphans)} Orphan IDs operating without budget/project mapping.")
                            st.dataframe(pd.DataFrame({"Orphan_IDs": orphans}), use_container_width=True)
                        else:
                            st.success("✅ Clean: All parsed IDs map correctly to the Master Index.")
                    else: st.warning("No campaign key column found to cross-reference.")
                else: st.warning("UCM Campaign Index not found to cross-reference.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='console-card'><strong style='color:{CMU_RED};'>⚠️ File not found.</strong> We searched your root deployment directory but could not locate the file. Check GitHub casing and extensions.</div>", unsafe_allow_html=True)

# ======================= AGENT 2: DATA ALCHEMIST =======================
elif current_page == "cleaner":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1>⚗️ Step 2: Data Alchemist</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="console-card">
        <h3 style="margin-top: 0;">The Synthesis Engine</h3>
        <p style="margin-bottom: 0;">
        <strong>1. Standardization:</strong> Aggressively cleans financial fields, removing currency symbols, and enforcing strict numeric arrays.<br>
        <strong>2. Normalization:</strong> Strips special characters and unifies casing to create a flawless <code>utm_clean</code> primary key for SQL-style LEFT JOINS.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='console-card'>", unsafe_allow_html=True)
    if not master_df.empty:
        st.success(f"✅ Master Hub Synthesized. {len(master_df)} campaigns merged dynamically from root files.")
        st.dataframe(master_df, use_container_width=True)
    else:
        st.error("Alchemist failed to synthesize. Ensure the Index file is present in the repository.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= AGENT 3: QUANTITATIVE STRATEGIST =======================
elif current_page == "analysis":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1>🧪 Step 3: Quantitative Strategist</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="console-card">
        <h3 style="margin-top: 0;">Data-Driven Predictive Modeling</h3>
        <p style="margin-bottom: 0;">
        The Strategist models physical relationships (like Spend vs Acquisition and Video Resonance) to prove mathematical correlations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not master_df.empty:
        c1, c2, c3 = st.columns(3)
        valid_spend = master_df[(master_df['Total_Spend'] > 0) & (master_df['Total_Users'] > 0)]
        
        # Spend Correlation
        if len(valid_spend) > 2 and valid_spend['Total_Spend'].var() > 0:
            corr, _ = stats.pearsonr(valid_spend['Total_Spend'], valid_spend['Total_Users'])
            c1.metric("Spend-to-User Correlation", f"{corr:.2f}")
            c2.metric("Significant Campaigns", len(valid_spend))
            c3.metric("Cost per Acquired User (Avg)", f"${valid_spend['CPWU'].mean():.2f}")
            
            st.markdown("<div class='console-card'><h3>Spend vs Users (Efficiency Frontier)</h3>", unsafe_allow_html=True)
            fig_s = px.scatter(valid_spend, x="Total_Spend", y="Total_Users", color="Category", trendline="ols")
            fig_s.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color="#111111")
            st.plotly_chart(fig_s, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            c1.metric("Spend Correlation", "N/A")
            c2.metric("Significant Campaigns", len(valid_spend))
            c3.metric("Cost per Acquired User", "N/A")
            st.markdown(f"<div class='console-card'><strong style='color:{CMU_RED};'>⚠️ Insufficient mapped spend variance for mathematical regression.</strong></div>", unsafe_allow_html=True)

        # Video Resonance Correlation
        if 'V100' in master_df.columns and 'Engagement_Rate' in master_df.columns:
            valid_vid = master_df[(master_df['V100'] > 0) & (master_df['Engagement_Rate'] > 0)]
            if len(valid_vid) > 2 and valid_vid['V100'].var() > 0:
                vr, _ = stats.pearsonr(valid_vid['V100'], valid_vid['Engagement_Rate'])
                st.markdown(f"<div class='console-card'><h3>📊 Resonance Modeling: Video Completion vs Engagement</h3><p><strong>Pearson r = {vr:.2f}</strong></p>", unsafe_allow_html=True)
                fig_v = px.scatter(valid_vid, x="V100", y="Engagement_Rate", size="Total_Users", hover_name="utm_clean", color_discrete_sequence=[CMU_RED], trendline="ols")
                fig_v.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color="#111111", xaxis_title="Video Played 100% (%)", yaxis_title="Website Engagement Rate (%)")
                st.plotly_chart(fig_v, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Master Hub is empty. Strategist offline.")

# ======================= AGENT 4: VISUAL ARCHITECT (DASHBOARD) =======================
elif current_page == "dashboard":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(f"<h2 style='color: {WHITE};'>🎯 Oracle Filters</h2>", unsafe_allow_html=True)
        if not master_df.empty:
            vendors = master_df['Vendor'].dropna().unique().tolist()
            sel_vendor = st.multiselect("Platform Vendor", vendors, default=vendors)
            categories = master_df['Category'].dropna().unique().tolist()
            sel_cat = st.multiselect("Department Category", categories, default=categories)
        else:
            sel_vendor, sel_cat = [], []

    st.markdown("<h1>🖥️ Step 4: Visual Architect</h1>", unsafe_allow_html=True)
    
    if not master_df.empty:
        f_df = master_df[(master_df['Vendor'].isin(sel_vendor)) & (master_df['Category'].isin(sel_cat))]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Filtered Acquired Users", f"{f_df.get('Total_Users', pd.Series([0])).sum():,.0f}")
        m2.metric("Avg Website Engagement", f"{f_df.get('Engagement_Rate', pd.Series([0])).mean():.1f}%")
        m3.metric("Filtered Spend Tracked", f"${f_df.get('Total_Spend', pd.Series([0])).sum():,.2f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("<div class='console-card'><h3>🔵 The Heartbeat: Temporal Velocity</h3>", unsafe_allow_html=True)
            if not ts_data.empty:
                fig_ts = px.line(ts_data, x='day', y=['sessions', 'users'], color_discrete_map={"sessions": CMU_GREY, "users": CMU_RED})
                fig_ts.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color="#111111", legend_title_text='')
                st.plotly_chart(fig_ts, use_container_width=True)
            else: st.info("TimeSeries data unavailable.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_right:
            st.markdown("<div class='console-card'><h3>🟣 Department Allocation</h3>", unsafe_allow_html=True)
            fig_bar = px.bar(f_df.groupby('Category')['Total_Spend'].sum().reset_index(), x='Category', y='Total_Spend', color_discrete_sequence=[CMU_RED])
            fig_bar.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color="#111111")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Real Retention Heatmap 
        st.markdown("<div class='console-card'><h3>🔥 Video Retention Heatmap (Real GAds Data)</h3>", unsafe_allow_html=True)
        v_cols = ['V25', 'V50', 'V75', 'V100']
        if all(c in f_df.columns for c in v_cols):
            v_df = f_df[f_df['V25'] > 0][['utm_clean'] + v_cols].head(10)
            if not v_df.empty:
                v_melt = v_df.melt(id_vars=["utm_clean"], var_name="Completion", value_name="Retention %")
                fig_heat = px.density_heatmap(v_melt, x="Completion", y="utm_clean", z="Retention %", color_continuous_scale="Reds", text_auto=True)
                fig_heat.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color="#111111")
                st.plotly_chart(fig_heat, use_container_width=True)
            else: st.info("No video retention data available for selected filters.")
        else: st.info("Video completion columns missing from raw data sources.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='console-card'><h3>🌍 The Attention Economy: Engagement vs Duration</h3>", unsafe_allow_html=True)
        if 'Session_Duration' in f_df.columns and f_df['Session_Duration'].sum() > 0:
            fig_att = px.scatter(f_df, x="Engagement_Rate", y="Session_Duration", size="Total_Users", hover_name="utm_clean", color="Vendor", color_discrete_sequence=[CMU_RED, CMU_GREY, "#000000"])
            fig_att.update_layout(paper_bgcolor=WHITE, plot_bgcolor=WHITE, font_color="#111111", xaxis_title="Engagement Rate (%)", yaxis_title="Avg Session Duration (s)")
            st.plotly_chart(fig_att, use_container_width=True)
        else: st.info("Session Duration data unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No data available. Dashboard offline.")

# ======================= AGENT 5: KNOWLEDGE GRAPH =======================
elif current_page == "graph":
    st.markdown(nav_cards_html, unsafe_allow_html=True)
    st.markdown("<h1>🕸️ Step 5: Knowledge Graph</h1>", unsafe_allow_html=True)
    
    st.markdown("""<div class="console-card">
        <h3>Tripartite Relational Universe</h3>
        <p>This graph maps the physical data flow generated by the Alchemist: <strong>CMU Data Systems → Platform Vendor → Targeted Campaign</strong>. Node sizes are scaled by actual Total User volume.</p>
    </div>""", unsafe_allow_html=True)
    
    if not master_df.empty:
        net = Network(height="750px", width="100%", bgcolor=BLACK, font_color=WHITE, select_menu=True)
        net.add_node("CMU", size=60, color=CMU_RED, label="CMU Hub")
        
        for v in master_df['Vendor'].unique():
            if pd.isna(v) or v == "": continue
            net.add_node(str(v), size=35, color=CMU_GREY, label=str(v))
            net.add_edge("CMU", str(v))
            
            camps = master_df[master_df['Vendor'] == v].sort_values('Total_Users', ascending=False).head(15)
            for _, row in camps.iterrows():
                camp = str(row['utm_clean'])
                users = max(row.get('Total_Users', 0), 1)
                n_size = min(max(np.sqrt(users) / 2, 10), 45)
                
                if camp and camp != "nan":
                    net.add_node(camp, size=n_size, color=WHITE, title=f"{camp} | Users: {users:,.0f}")
                    net.add_edge(str(v), camp)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_data = f.read().replace('<style type="text/css">', '<style type="text/css">\n #mynetwork {border: none; outline: none;}\n')
                components.html(html_data, height=800)
