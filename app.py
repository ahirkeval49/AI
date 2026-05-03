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
CARD_BG = "#1a1a1a"

st.set_page_config(page_title="CMU Command Center", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Zero-Latency React Tabs & Dark Mode
st.markdown(f"""
<style>
    .stApp {{ background-color: {BLACK}; color: {WHITE}; }}
    h1, h2, h3, h4, h5, h6 {{ color: {CMU_RED} !important; font-weight: 800; font-family: 'Segoe UI', sans-serif; }}
    p, span, div, label, li, td, th {{ color: {WHITE}; font-family: 'Segoe UI', sans-serif; }}
    
    .console-card {{
        background-color: {CARD_BG}; border-radius: 12px; padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); margin-bottom: 24px;
        border-top: 3px solid {CMU_RED}; border-left: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;
    }}
    
    div[data-testid="stMetricValue"] {{ color: {WHITE} !important; font-weight: 900; }}
    div[data-testid="stMetricLabel"] {{ color: {CMU_GREY} !important; font-weight: 700; text-transform: uppercase; }}
    div[data-testid="stMetric"] {{ background-color: #222; padding: 15px; border-radius: 10px; border-left: 4px solid {CMU_RED}; }}
    
    /* ZERO-LATENCY NATIVE TAB OVERRIDE */
    div[data-baseweb="tab-list"] {{
        display: flex; justify-content: center; gap: 15px; margin-bottom: 20px;
        background-color: transparent; border: none;
    }}
    div[data-baseweb="tab"] {{
        background-color: #222 !important; color: {WHITE} !important; border: 1px solid #444 !important; 
        border-radius: 8px !important; padding: 10px 20px !important; height: auto !important;
        font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 1px !important;
        transition: 0.2s ease-in-out; margin: 0;
    }}
    div[data-baseweb="tab"]:hover {{ background-color: #333 !important; transform: translateY(-2px); }}
    div[data-baseweb="tab"][aria-selected="true"] {{
        background-color: {CMU_RED} !important; border-color: {CMU_RED} !important; color: {WHITE} !important;
    }}
    div[data-baseweb="tab-highlight"] {{ display: none; }}
    
    header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

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

@st.cache_data
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
        # --- 1. LOAD INDEX ---
        idx_raw = smart_load('ucmcampaignindex')
        if idx_raw is None or idx_raw.empty: return pd.DataFrame()
        
        idx = pd.DataFrame()
        idx['board_key'] = normalize_key(idx_raw.get('Monday_Board_Name', pd.Series(dtype=str)))
        idx['ga_key'] = normalize_key(idx_raw.get('UTM campaign', pd.Series(dtype=str)))
        idx['Category'] = idx_raw.get('Category', pd.Series(dtype=str)).fillna('Uncategorized')
        idx['Vendor'] = idx_raw.get('Vendor', pd.Series(dtype=str)).fillna('Unknown')
        idx['Display_Name'] = idx_raw.get('Monday_Board_Name', pd.Series(dtype=str))
        idx = idx.drop_duplicates(subset=['board_key']).dropna(subset=['board_key'])

        # --- 2. LOAD MONDAY.COM (BUDGETS) ---
        mon1 = smart_load('202425campaignmanagement')
        mon2 = smart_load('202526campaignmanagement')
        mon_raw = pd.concat([df for df in [mon1, mon2] if df is not None])
        if not mon_raw.empty:
            mon = pd.DataFrame()
            mon_name = find_col(mon_raw, ['name'])
            mon_bud = find_col(mon_raw, ['budget'])
            mon_run = find_col(mon_raw, ['run dates'])
            mon['board_key'] = normalize_key(mon_raw[mon_name]) if mon_name else ""
            mon['Budget'] = clean_num(mon_raw[mon_bud]) if mon_bud else 0.0
            mon['Run_Dates'] = mon_raw[mon_run].astype(str) if mon_run else ""
            mon = mon.groupby('board_key').first().reset_index()
            idx = pd.merge(idx, mon, on='board_key', how='left')

        # --- 3. LOAD PLATFORMS ---
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
        if li_df is not None and not li_df.empty:
            camp_col = find_col(li_df, ['campaign name', 'campaign'])
            if camp_col:
                ext = pd.DataFrame()
                ext['board_key'] = normalize_key(li_df[camp_col])
                ext['Spend'] = clean_num(li_df[find_col(li_df, ['total spend', 'spend', 'cost'])])
                ext['Clicks'] = clean_num(li_df[find_col(li_df, ['clicks'])])
                plat_dfs.append(ext)
                
        plat_agg = pd.concat(plat_dfs).groupby('board_key').sum().reset_index() if plat_dfs else pd.DataFrame(columns=['board_key', 'Spend', 'Clicks'])

        # --- 4. LOAD IMPACT (GA) ---
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

        # --- 5. THE MASTER MERGE ---
        master = pd.merge(idx, plat_agg, on='board_key', how='outer')
        master = pd.merge(master, ga_agg, on='ga_key', how='outer')
        
        master['Display_Name'] = master['Display_Name'].fillna(master['board_key']).fillna(master['ga_key']).replace('', 'Unknown Campaign')
        master['Category'] = master['Category'].fillna('Uncategorized')
        master['Vendor'] = master['Vendor'].fillna('Platform/Organic')
        master.fillna({'Spend':0, 'Clicks':0, 'Users':0, 'Eng_Rate':0, 'Duration':0, 'Budget':0}, inplace=True)
        
        # Calculate God-Tier KPIs
        master['Dropoff_Rate'] = np.where(master['Clicks'] > 10, ((master['Clicks'] - master['Users']) / master['Clicks']).clip(0, 1), 0)
        master['CPWU'] = np.where(master['Users'] > 0, master['Spend'] / master['Users'], 0)
        master['Engaged_Mins'] = (master['Users'] * master['Duration']) / 60
        master['CPQM'] = np.where(master['Engaged_Mins'] > 0.5, master['Spend'] / master['Engaged_Mins'], 0)
        
        master = master[(master['Spend'] > 0) | (master['Users'] > 0)]
        return master
    except Exception as e: 
        return pd.DataFrame()

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
    if dfs:
        return pd.concat(dfs).groupby('Day')['Users'].sum().reset_index().sort_values('Day')
    return pd.DataFrame()

master_df = build_master_pipeline()
ts_data = load_timeseries()

# ---------------------------------------------------------
# UI: ZERO-LATENCY TABS
# ---------------------------------------------------------
tab_nexus, tab_auditor, tab_dash, tab_strat, tab_graph = st.tabs([
    "🌌 Nexus", "🕵️ Auditor", "🖥️ Dashboard", "🧪 Strategist", "🕸️ Knowledge Graph"
])

# ======================= TAB 1: NEXUS (3D GALAXY) =======================
with tab_nexus:
    st.markdown(f"<h1 style='text-align: center; font-size: 50px; margin-top: 20px;'>CMU COMMAND CENTER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; letter-spacing: 2px;'>DETERMINISTIC SPLIT-KEY ENGINE</p>", unsafe_allow_html=True)
    
    # 3D Galaxy - Font Loader Removed, Replaced with Fail-Proof Orbiting Nodes
    three_js_galaxy = f"""
    <!DOCTYPE html><html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>body {{ margin: 0; background: {BLACK}; overflow: hidden; font-family: sans-serif; }}</style></head>
    <body><script>
        const scene = new THREE.Scene(); scene.background = new THREE.Color("{BLACK}");
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({{antialias: true, alpha: true}});
        renderer.setSize(window.innerWidth, window.innerHeight); document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.autoRotate = true; controls.autoRotateSpeed = 1.0; controls.enableDamping = true;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.8); scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1.2); dirLight.position.set(10, 20, 10); scene.add(dirLight);

        // Background Stars
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

        // The Central CMU Hub
        const coreGeo = new THREE.SphereGeometry(2.5, 32, 32);
        const coreMat = new THREE.MeshPhongMaterial({{color: "{CMU_RED}", emissive: 0x440000, shininess: 100}});
        const core = new THREE.Mesh(coreGeo, coreMat);
        scene.add(core);

        // Orbiting System Nodes
        const nodes = [
            {{color: 0xffffff, pos: [8, 4, 0]}},
            {{color: 0x666666, pos: [-7, -5, 4]}},
            {{color: 0xffffff, pos: [0, 8, -6]}}
        ];
        nodes.forEach(n => {{
            const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.8, 32, 32), new THREE.MeshPhongMaterial({{color: n.color, shininess: 80}}));
            mesh.position.set(...n.pos); scene.add(mesh);
        }});

        camera.position.z = 25;
        function animate(){{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
        animate();
    </script></body></html>
    """
    components.html(three_js_galaxy, height=750)

# ======================= TAB 2: AUDITOR =======================
with tab_auditor:
    st.markdown("<h1>🕵️ Forensic Auditor & Planner</h1>", unsafe_allow_html=True)
    st.markdown("<div class='console-card'><h3>📋 Planning Phase & Raw Files</h3>", unsafe_allow_html=True)
    st.markdown("<p>Select a raw operational file below to review the explicit planning intent and budgets before they hit the execution pipeline.</p>", unsafe_allow_html=True)
    
    f = st.selectbox("Select Target CSV", ALL_FILES, key="auditor_select")
    df = smart_load(f)
    
    if df is not None and not df.empty:
        v1, v2, v3, v4 = st.tabs(["📊 Raw Data View", "🔍 Column Profile", "📈 Data Stats", "⚠️ Structural Anomalies"])
        with v1: st.dataframe(df.head(100), use_container_width=True)
        with v2:
            profile = pd.DataFrame({'Data Type': df.dtypes.astype(str), 'Null Count': df.isna().sum(), 'Unique Values': df.nunique()})
            st.dataframe(profile, use_container_width=True)
        with v3: st.dataframe(df.describe(include='all').T, use_container_width=True)
        with v4:
            anomalies_found = 0
            day_cols = [c for c in df.columns if 'day' in str(c).lower() and any(char.isdigit() for char in str(c))]
            if len(day_cols) > 5:
                st.error("🚨 **Wide-Format TimeSeries Detected:** The timeline is stretched horizontally across columns. *Alchemist Fix: Automatically melted.*")
                anomalies_found += 1
            if any(df.astype(str).apply(lambda x: x.str.contains('Total: Campaigns', case=False, na=False).any())):
                st.error("🚨 **Google Ads 'Total' Row Detected:** Contains string-based aggregation rows. *Alchemist Fix: Automatically dropped.*")
                anomalies_found += 1
            if any(df.astype(str).apply(lambda x: x.str.contains(' --', case=False, na=False).any())):
                st.warning("⚠️ **String Nulls Detected:** Platform exported '--' instead of empty cells. *Alchemist Fix: RegEx mapped to 0.0.*")
                anomalies_found += 1
            if anomalies_found == 0:
                st.success("✅ No severe structural anomalies detected in this file.")
    st.markdown("</div>", unsafe_allow_html=True)

# ======================= TAB 3: DASHBOARD =======================
with tab_dash:
    st.markdown("<h1>🖥️ Pipeline Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("<div class='console-card'><h3>🎯 Quick Filters</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if not master_df.empty:
        vendors = master_df['Vendor'].unique().tolist()
        categories = master_df['Category'].unique().tolist()
        with c1: sel_vend = st.multiselect("Filter by Platform/Vendor", vendors, default=vendors)
        with c2: sel_cat = st.multiselect("Filter by Department/Category", categories, default=categories)
        f_df = master_df[(master_df['Vendor'].isin(sel_vend)) & (master_df['Category'].isin(sel_cat))]
    else: f_df = pd.DataFrame()
    st.markdown("</div>", unsafe_allow_html=True)

    if not f_df.empty:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Ecosystem Spend", f"${f_df['Spend'].sum():,.0f}")
        m2.metric("Total Acquired Users", f"{f_df['Users'].sum():,.0f}")
        avg_cpa = f_df['Spend'].sum() / f_df['Users'].sum() if f_df['Users'].sum() > 0 else 0
        m3.metric("System Average CPA", f"${avg_cpa:.2f}")
        avg_drop = (f_df['Clicks'].sum() - f_df['Users'].sum()) / f_df['Clicks'].sum() if f_df['Clicks'].sum() > 0 else 0
        m4.metric("Avg Pipeline Drop-off", f"{avg_drop:.1%}")

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("<div class='console-card'><h3>⏳ Flight Risk (Pacing vs Budget)</h3>", unsafe_allow_html=True)
            current_date = pd.to_datetime('2026-05-03') 
            pacing = []
            for _, r in f_df.iterrows():
                if r['Budget'] > 0 and pd.notna(r['Run_Dates']) and '-' in str(r['Run_Dates']):
                    try:
                        d_start, d_end = pd.to_datetime(str(r['Run_Dates']).split('-')[0].strip()), pd.to_datetime(str(r['Run_Dates']).split('-')[1].strip())
                        if d_start <= d_end:
                            pct_time = min(max((current_date - d_start).days / ((d_end - d_start).days + 1), 0), 1)
                            pct_spend = r['Spend'] / r['Budget']
                            pacing.append({'Campaign': r['Display_Name'][:35] + '...', 'Time Elapsed': pct_time, 'Budget Spent': pct_spend, 'Pacing Delta': pct_spend - pct_time})
                    except: pass
            if pacing:
                st.dataframe(pd.DataFrame(pacing).sort_values('Pacing Delta').style.format({'Time Elapsed': '{:.1%}', 'Budget Spent': '{:.1%}', 'Pacing Delta': '{:+.1%}'}), use_container_width=True)
            else: st.info("No active budgeted campaigns found.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_right:
            st.markdown("<div class='console-card'><h3>📈 Temporal Pulse (Users)</h3>", unsafe_allow_html=True)
            if not ts_data.empty:
                fig_ts = px.line(ts_data, x='Day', y='Users', color_discrete_sequence=[CMU_RED])
                fig_ts.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
                st.plotly_chart(fig_ts, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='console-card'><h3>🌊 Traffic Attribution Waterfall</h3>", unsafe_allow_html=True)
        top = f_df[f_df['Users'] > 0].sort_values('Users', ascending=False).head(15)
        if not top.empty:
            nodes = list(top['Vendor'].unique()) + list(top['Category'].unique()) + list(top['Display_Name'].unique())
            n_map = {n: i for i, n in enumerate(nodes)}
            links = []
            for _, r in top.iterrows():
                links.append({'source': n_map[r['Vendor']], 'target': n_map[r['Category']], 'value': r['Users']})
                links.append({'source': n_map[r['Category']], 'target': n_map[r['Display_Name']], 'value': r['Users']})
            l_df = pd.DataFrame(links).groupby(['source','target']).sum().reset_index()
            fig_s = go.Figure(go.Sankey(node=dict(label=nodes, color=CMU_RED, pad=15, thickness=20), link=dict(source=l_df['source'], target=l_df['target'], value=l_df['value'], color="rgba(196,18,48,0.4)")))
            fig_s.update_layout(paper_bgcolor=CARD_BG, font_color=WHITE, height=450)
            st.plotly_chart(fig_s, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("<div class='console-card'><h3>🌍 Attention Economy Grid</h3>", unsafe_allow_html=True)
            fig_at = px.scatter(f_df[f_df['Users']>0], x="Eng_Rate", y="Duration", size="Users", hover_name="Display_Name", color="Vendor", color_discrete_sequence=[CMU_RED, "#555", "#999"])
            fig_at.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_at, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col_y:
            st.markdown("<div class='console-card'><h3>🟣 Department Allocation</h3>", unsafe_allow_html=True)
            fig_bar = px.bar(f_df.groupby('Category')['Spend'].sum().reset_index(), x='Category', y='Spend', color_discrete_sequence=[CMU_RED])
            fig_bar.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else: st.error("Dashboard offline. Waiting for valid data files.")

# ======================= TAB 4: STRATEGIST =======================
with tab_strat:
    st.markdown("<h1>🧪 Quantitative Strategist</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='console-card'><h3>📉 Ad Fatigue (Diminishing Returns)</h3>", unsafe_allow_html=True)
        reg_df = master_df[(master_df['Spend'] > 0) & (master_df['Users'] > 0)]
        if len(reg_df) > 5:
            p = np.polyfit(reg_df['Spend'], reg_df['Users'], 2)
            f = np.poly1d(p)
            x_ax = np.linspace(reg_df['Spend'].min(), reg_df['Spend'].max(), 100)
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=reg_df['Spend'], y=reg_df['Users'], mode='markers', name='Campaigns', marker=dict(color=CMU_RED, size=8), text=reg_df['Display_Name']))
            fig_p.add_trace(go.Scatter(x=x_ax, y=f(x_ax), mode='lines', name='Fatigue Curve', line=dict(color=CMU_GREY, dash='dash', width=3)))
            fig_p.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_p, use_container_width=True)
        else: st.info("Not enough spend variance to plot regression curve.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c2:
        st.markdown("<div class='console-card'><h3>🏆 True Quality: Lowest CPQM</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:12px; color:#aaa;'>Cost Per Quality Minute (Spend / Engaged Minutes)</p>", unsafe_allow_html=True)
        cpqm_df = master_df[master_df['CPQM'] > 0].sort_values('CPQM').head(8)
        if not cpqm_df.empty:
            st.dataframe(cpqm_df[['Display_Name', 'Vendor', 'CPQM', 'Spend', 'Users']].style.format({'CPQM': '${:.2f}', 'Spend': '${:,.0f}'}), use_container_width=True)
        else: st.info("Not enough duration data to model CPQM.")
        st.markdown("</div>", unsafe_allow_html=True)

    col_x, col_y = st.columns(2)
    with col_x:
        st.markdown("<div class='console-card'><h3>🧠 Lexical Resonance (Copywriting)</h3>", unsafe_allow_html=True)
        ads_df = smart_load('gadsfy24fy26monthlyweeklyperformance')
        if ads_df is not None and 'Headline 1' in ads_df.columns:
            ads_df['Clicks'] = clean_num(ads_df.get('Clicks', pd.Series(0)))
            ads_df['Impr.'] = clean_num(ads_df.get('Impr.', pd.Series(0)))
            words_data = []
            stopwords = {"the", "and", "to", "of", "a", "in", "for", "is", "on", "with", "at", "as", "by", "--", "cmu"}
            for _, row in ads_df.dropna(subset=['Headline 1']).iterrows():
                text = str(row['Headline 1']).lower()
                words = set(re.findall(r'\b[a-z]{3,}\b', text)) - stopwords
                for w in words: words_data.append({'Keyword': w.capitalize(), 'Clicks': row['Clicks'], 'Impr': row['Impr.']})
            if words_data:
                wd_df = pd.DataFrame(words_data).groupby('Keyword').sum().reset_index()
                wd_df['CTR'] = wd_df['Clicks'] / wd_df['Impr']
                wd_df = wd_df[wd_df['Impr'] > 5000].sort_values('CTR', ascending=False).head(8)
                st.dataframe(wd_df[['Keyword', 'CTR', 'Clicks', 'Impr']].style.format({'CTR': '{:.2%}', 'Clicks': '{:,.0f}', 'Impr': '{:,.0f}'}), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_y:
        st.markdown("<div class='console-card'><h3>📊 Ad Format A/B Testing</h3>", unsafe_allow_html=True)
        if ads_df is not None and 'Ad type' in ads_df.columns:
            format_df = ads_df.groupby('Ad type').agg({'Clicks': 'sum', 'Impr.': 'sum'}).reset_index()
            format_df = format_df[format_df['Impr.'] > 0]
            format_df['CTR'] = format_df['Clicks'] / format_df['Impr.']
            fig_fmt = px.bar(format_df.sort_values('CTR', ascending=False), x='Ad type', y='CTR', color_discrete_sequence=[CMU_RED])
            fig_fmt.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=WHITE, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333', tickformat='.1%'))
            st.plotly_chart(fig_fmt, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ======================= TAB 5: KNOWLEDGE GRAPH =======================
with tab_graph:
    st.markdown("<h1>🕸️ Ecosystem Relational Graph</h1>", unsafe_allow_html=True)
    
    st.markdown("""<div class="console-card">
        <p>This physics-based network maps the successful relational joins generated by our Split-Key pipeline: <strong>Vendor → Campaign Category → Campaign Name</strong>. Nodes are draggable.</p>
    </div>""", unsafe_allow_html=True)
    
    if not master_df.empty:
        net = Network(height="700px", width="100%", bgcolor=CARD_BG, font_color=WHITE)
        net.add_node("CMU", size=50, color=CMU_RED, label="CMU Hub")
        
        for vend in master_df['Vendor'].unique():
            if str(vend) not in ['0', 'nan', 'Unknown', 'Platform/Organic']:
                net.add_node(vend, size=35, color="#666", label=vend)
                net.add_edge("CMU", vend)
                
                vend_df = master_df[master_df['Vendor'] == vend]
                for cat in vend_df['Category'].unique():
                    cat_node = f"{vend}_{cat}"
                    net.add_node(cat_node, size=25, color="#888", label=cat)
                    net.add_edge(vend, cat_node)
                    
                    camps = vend_df[vend_df['Category'] == cat].sort_values('Users', ascending=False).head(8)
                    for _, r in camps.iterrows():
                        camp_name = str(r['Display_Name'])[:30] + "..." if len(str(r['Display_Name'])) > 30 else str(r['Display_Name'])
                        if camp_name and camp_name != "nan":
                            net.add_node(camp_name, size=15, color=CMU_RED, label=camp_name)
                            net.add_edge(cat_node, camp_name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=".") as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html_code = f.read().replace('<style type="text/css">', '<style type="text/css">\n #mynetwork {border: none; outline: none;}\n')
                components.html(html_code, height=750)
            os.remove(tmp.name)
    else:
        st.error("No joined data available to render graph.")
